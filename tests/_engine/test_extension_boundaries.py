"""Public-run regressions, isolated so a blocked extension cannot hang pytest.

Controller ruling N-I1 explicitly replaces the former total-runtime < 7 s
contract with two real-monotonic bounds. From calling Swarm.run to trigger
readiness, allow < 20 s when waiting for a lease renewal, otherwise < 7 s.
From the first stop request, round-clock advance, or judge cancellation to
Swarm.run exit, require < 7 s. Journal inspection and observer cleanup are
outside that second interval. The runner-only clock never measures latency.
Partial-output readiness means delivery before the blocked submit is interrupted;
its fallback serializer is observed after that first termination trigger.
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic import model_serializer

from swarm_arena import (
    AgentSpec,
    Goal,
    Limits,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    Score,
    ScriptedManager,
    ScriptTool,
    Swarm,
    TeamSpec,
)
from swarm_arena._model.results import Candidate
from swarm_arena.executor import Job, LocalExecutor
from swarm_arena.manager import Manager, RoundReport
from swarm_arena.scoring import Check, CheckResult, Judge, JudgeContext, Lexicographic
from swarm_arena.tool import RunStatus, ToolOutput

_SIGNAL_TIMEOUT_S = 30
_STOP_TIMEOUT_S = 7

_HOOK_ENTERED = threading.Event()
_HOOK_RENEWED = threading.Event()
_CHECK_ENTERED = threading.Event()
_JUDGE_ENTERED = threading.Event()
_JUDGE_RENEWED = threading.Event()
_REPORT_KIND = "object"
_SERIALIZER_ENTERED = threading.Event()
_PARTIAL_READY = threading.Event()


def _block_serializer() -> dict[str, Any]:
    print(f"SERIALIZER_THREAD {threading.current_thread().name}", flush=True)
    _SERIALIZER_ENTERED.set()
    threading.Event().wait()
    return {}


class BlockingCheckResult(CheckResult):
    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return _block_serializer()


class SerializingCheck(Check):
    name = "serialized"

    async def check(self, candidate: Any, ctx: Any) -> CheckResult:
        return BlockingCheckResult(check=self.name, passed=True)


class BlockingToolOutput(ToolOutput):
    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return _block_serializer()


class SerializingExecutor(LocalExecutor):
    async def submit(self, job: Job) -> ToolOutput:
        output = await super().submit(job)
        return BlockingToolOutput.model_validate(output.model_dump(mode="python"))


class PartialSerializingExecutor(LocalExecutor):
    async def submit(self, job: Job) -> ToolOutput:
        await super().submit(job)
        assert job.on_partial is not None
        await job.on_partial(BlockingToolOutput(status=RunStatus.CANCELLED, partial=True))
        # Exercise the partial-output fallback once the round interrupts submit.
        print("PARTIAL_READY", flush=True)
        _PARTIAL_READY.set()
        await anyio.sleep_forever()
        raise AssertionError("unreachable")


class BlockingHookTool(ScriptTool):
    def is_deterministic(self, request: Any) -> bool:
        print(f"HOOK_THREAD {threading.current_thread().name}", flush=True)
        _HOOK_ENTERED.set()
        threading.Event().wait()
        return True


class FailingCollectTool(ScriptTool):
    def collect(self, request: Any, workspace: Path, run: Any) -> Any:
        error = ValueError("COLLECT_SECRET_29")
        error.add_note("NOTE_SECRET_29")
        raise error from RuntimeError("CAUSE_SECRET_29")


class BlockingCheck(Check):
    name = "hang"

    async def check(self, candidate: Any, ctx: Any) -> Any:
        print("CHECK_ENTERED", flush=True)
        _CHECK_ENTERED.set()
        await anyio.sleep_forever()


class MalformedManager(Manager):
    async def run_round(self, ctx: Any) -> RoundReport:
        if _REPORT_KIND == "type":
            return object()  # ty: ignore[invalid-return-type]
        summary = {"object": {"bad": object()}, "json": {"bad": 1}, "utf8": "\ud800"}[_REPORT_KIND]
        return RoundReport.model_construct(summary=summary)


class PlainJudge(Judge):
    name = "quality"

    async def judge(self, goal: Goal, candidates: Sequence[Any], ctx: JudgeContext) -> list[Score]:
        self.scoring = True
        return [Score(candidate_id=c.id, key=(1.0,)) for c in candidates]


class BlockingRubricJudge(PlainJudge):
    def rubric_hash(self) -> str:
        if getattr(self, "scoring", False):
            print(f"JUDGE_THREAD {threading.current_thread().name}", flush=True)
            _JUDGE_ENTERED.set()
            threading.Event().wait()
        return "fixed"


class BlockingPanelJudge(PlainJudge):
    def panel_hash(self) -> str:
        if getattr(self, "scoring", False):
            print(f"JUDGE_THREAD {threading.current_thread().name}", flush=True)
            _JUDGE_ENTERED.set()
            threading.Event().wait()
        return "fixed"


class MalformedScoreJudge(PlainJudge):
    async def judge(self, goal: Goal, candidates: Sequence[Any], ctx: JudgeContext) -> list[Score]:
        return [
            score.model_copy(update={"llm_call_ids": [object()]})
            for score in await super().judge(goal, candidates, ctx)
        ]


def score(goal: Goal, candidate: Any) -> Score:
    return Score.maximize(1.0)


async def _wait_for_signal(event: threading.Event, label: str) -> None:
    assert await anyio.to_thread.run_sync(event.wait, _SIGNAL_TIMEOUT_S, abandon_on_cancel=True), (
        f"timed out waiting for {label}"
    )


class _RoundClock:
    """Control only the round runner's time; executor and lease clocks stay real."""

    def __init__(self, timeout_s: float) -> None:
        self.now = anyio.current_time()
        self.deadline = self.now + timeout_s
        self.deadline_observed = False
        self.changed = anyio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(anyio, name)

    def current_time(self) -> float:
        self.deadline_observed |= self.now > self.deadline
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.changed.set()
        self.changed = anyio.Event()

    async def sleep_until(self, deadline: float) -> None:
        while self.now < deadline:
            await self.changed.wait()


async def _worker(mode: str, root: Path) -> None:
    with pytest.MonkeyPatch.context() as patch:
        await _scoped_worker(mode, root, patch)


async def _scoped_worker(mode: str, root: Path, patch: pytest.MonkeyPatch) -> None:
    # Observe real extension/lease/journal signals before stopping or expiring work.
    from swarm_arena._journal.lease import Lease
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._store.local import LocalStore

    global _REPORT_KIND
    mode, _, variant = mode.partition(":")
    if mode == "manager":
        _REPORT_KIND = variant or "object"
    original_renew = Lease.renew

    def renew(self: Lease) -> None:
        original_renew(self)
        if _HOOK_ENTERED.is_set():
            print("RENEWED_AFTER_HOOK", flush=True)
            _HOOK_RENEWED.set()
        if _JUDGE_ENTERED.is_set():
            _JUDGE_RENEWED.set()

    patch.setattr(Lease, "renew", renew)
    module = "test_extension_boundaries"
    tool_name = {"hook": "BlockingHookTool", "collect": "FailingCollectTool"}.get(mode)
    command = [
        sys.executable,
        "-c",
        "import json,os; json.dump({'data': {'value': 1}}, "
        "open(os.environ['SWARM_ARENA_RESULT'], 'w'))",
    ]
    normal_tool = Ref.of(ScriptTool(command=command, grace_s=0.1))
    manager = Ref.of(ScriptedManager([{"inputs": {}}]))
    agents = [
        AgentSpec(
            name="a",
            rationale="exercise extension",
            tool=Ref(path=f"{module}:{tool_name}", config={"command": command, "grace_s": 0.1})
            if tool_name
            else normal_tool,
            manager=Ref(path=f"{module}:MalformedManager") if mode == "manager" else manager,
        )
    ]
    if mode in {"hook", "manager"}:
        agents.append(
            AgentSpec(name="peer", rationale="healthy", tool=normal_tool, manager=manager)
        )
    plan = Plan(
        goal=Goal(id="g", namespace="boundaries"),
        teams=[TeamSpec(name="t", rationale="test", agents=agents)],
        judges=[Ref.of(ProgramJudge(f"{module}:score"))],
        rounds=[RoundSpec()],
        checks=[Ref(path=f"{module}:BlockingCheck")]
        if mode == "check"
        else [Ref(path=f"{module}:SerializingCheck")]
        if mode == "check-serializer"
        else [],
    )
    plans = [plan]
    if mode in {"rubric", "panel", "lex-rubric", "lex-panel"}:
        judge_type = "BlockingRubricJudge" if mode.endswith("rubric") else "BlockingPanelJudge"
        judge_ref = Ref(path=f"{module}:{judge_type}")
        if mode.startswith("lex-"):
            judge_ref = Ref.of(Lexicographic([judge_ref]))
        plans = [plan.model_copy(update={"judges": [judge_ref]})]
    elif mode == "score":
        healthy = Ref.of(ProgramJudge(f"{module}:score")).model_copy(update={"name": "healthy"})
        plans = [
            plan.model_copy(
                update={
                    "judges": [Ref(path=f"{module}:MalformedScoreJudge"), healthy],
                    "ranking": "healthy",
                }
            ),
            plan.model_copy(update={"goal": Goal(id="peer", namespace="boundaries")}),
        ]
    config = RunConfig(
        run_root=str(root / "runs"),
        artifact_root=str(root / "artifacts"),
        arena_root=str(root / "arena"),
        executor=Ref.of(
            (
                SerializingExecutor
                if mode == "output-serializer"
                else PartialSerializingExecutor
                if mode == "partial-serializer"
                else LocalExecutor
            )(workspace_root=str(root / "ws"))
        ),
        # These are watchdogs, not the mechanism that makes a probe succeed.
        # Entry signals control termination; these watchdogs leave time for setup.
        limits=Limits(round_timeout_s=120, run_timeout_s=180, tool_timeout_s=60),
        lease_ttl_s=3.0,
        settle_timeout_s=0.2,
    )
    deadline_clock: _RoundClock | None = None
    if mode.endswith("serializer") and variant != "stop":
        from swarm_arena._engine import round as round_module

        deadline_clock = _RoundClock(config.limits.effective_round_timeout_s())
        patch.setattr(round_module, "anyio", deadline_clock)
    swarm = Swarm(config)
    live_store = LocalStore(root / "runs" / "boundary-run")
    stop_started: float | None = None
    deadline_started: float | None = None
    run_finished = anyio.Event()
    run_started_signal = anyio.Event()
    trigger_observed = anyio.Event()
    run_started: float | None = None
    run_ended: float | None = None
    elapsed_to_trigger: float | None = None
    trigger_limit_s = 20 if mode in {"hook", "rubric", "panel", "lex-rubric", "lex-panel"} else 7

    async def run() -> dict[str, list[Candidate]]:
        nonlocal run_started, run_ended
        run_started = time.monotonic()
        run_started_signal.set()
        try:
            return await swarm.run(plans, run_id="boundary-run")
        finally:
            run_ended = time.monotonic()
            run_finished.set()

    def record_trigger() -> None:
        nonlocal elapsed_to_trigger
        if elapsed_to_trigger is None:
            assert run_started is not None
            elapsed = time.monotonic() - run_started
            assert elapsed < trigger_limit_s, (
                f"{mode}: pre-trigger readiness took {elapsed:.3f}s; must be < {trigger_limit_s}s"
            )
            elapsed_to_trigger = elapsed
            trigger_observed.set()

    async def watch_trigger() -> None:
        await run_started_signal.wait()
        assert run_started is not None
        remaining = trigger_limit_s - (time.monotonic() - run_started)
        with anyio.move_on_after(max(0, remaining)):
            await trigger_observed.wait()
        assert elapsed_to_trigger is not None, (
            f"{mode}: pre-trigger readiness timed out after "
            f"{time.monotonic() - run_started:.3f}s; must be < {trigger_limit_s}s"
        )

    async def wait_for_event(kind: str, *, agent: str | None = None) -> None:
        with anyio.fail_after(_SIGNAL_TIMEOUT_S):
            while True:
                live = await anyio.to_thread.run_sync(read_committed, live_store)
                if live is not None and any(
                    e.type == kind
                    and (agent is None or e.agent == agent)
                    and (kind != "stop_requested" or e.payload["now"])
                    for e in live[1]
                ):
                    return
                await anyio.sleep(0.02)

    async def stop() -> None:
        nonlocal stop_started
        record_trigger()
        stop_started = time.monotonic()
        await swarm.stop(now=True)

    def expire_round(label: str, seconds: float) -> None:
        nonlocal deadline_started
        assert deadline_clock is not None
        if deadline_started is None:
            record_trigger()
            deadline_started = time.monotonic()
        deadline_clock.advance(seconds)
        print(label, flush=True)

    async def end_on_signal() -> None:
        if mode == "hook":
            await _wait_for_signal(_HOOK_ENTERED, "hook entry")
            await _wait_for_signal(_HOOK_RENEWED, "lease renewal after hook entry")
            # A renewal alone does not prove the healthy peer was allowed to finish.
            await wait_for_event("tool_end", agent="peer")
        elif mode == "check":
            await _wait_for_signal(_CHECK_ENTERED, "check entry")
        elif mode == "partial-serializer":
            # The fallback serializer is entered only after submit is interrupted.
            await _wait_for_signal(_PARTIAL_READY, "partial output delivery")
            if deadline_clock is None:
                await stop()
            else:
                # The real round deadline must interrupt submit before the
                # engine can enter its partial-output fallback serializer.
                expire_round(
                    "ROUND_DEADLINE_AFTER_PARTIAL_READY",
                    config.limits.effective_round_timeout_s() + 1,
                )
        else:
            await _wait_for_signal(_SERIALIZER_ENTERED, "serializer entry")
        if mode.endswith("serializer"):
            await _wait_for_signal(_SERIALIZER_ENTERED, "serializer entry")
            # Reaching a loop checkpoint after entry proves the blocked serializer
            # does not own the loop; no heartbeat sleep races the run deadline.
            await anyio.lowlevel.checkpoint()
            print("LOOP_RESPONSIVE", flush=True)
        if deadline_clock is not None:
            # For partial output the deadline already interrupted submit; wake
            # the real watcher again only after fallback serializer entry.
            expire_round(
                "ROUND_DEADLINE_AFTER_SERIALIZER_ENTRY",
                1
                if deadline_started is not None
                else config.limits.effective_round_timeout_s() + 1,
            )
        elif stop_started is None:
            await stop()
        with anyio.fail_after(_STOP_TIMEOUT_S):
            await run_finished.wait()

    if mode in {"rubric", "panel", "lex-rubric", "lex-panel"}:
        # Lifecycle termination belongs to the separate barrier-supervision fix.
        # This probe owns only thread isolation and live loop/lease/stop progress.
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            tasks.start_soon(watch_trigger)
            await _wait_for_signal(_JUDGE_ENTERED, "judge metadata entry")
            await _wait_for_signal(_JUDGE_RENEWED, "lease renewal after judge entry")
            stop_recorded = False
            if variant == "stop":
                await stop()
                with anyio.fail_after(_STOP_TIMEOUT_S):
                    await wait_for_event("stop_requested")
                stop_recorded = True
            else:
                record_trigger()
                stop_started = time.monotonic()
            tasks.cancel_scope.cancel()
        assert stop_started is not None
        assert run_ended is not None
        print(
            json.dumps(
                {
                    "renewed": True,
                    "stop_recorded": stop_recorded,
                    "elapsed_after_stop": run_ended - stop_started,
                    "elapsed_to_trigger": elapsed_to_trigger,
                }
            ),
            flush=True,
        )
        return
    if mode in {"hook", "check", "check-serializer", "output-serializer", "partial-serializer"}:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(end_on_signal)
            tasks.start_soon(watch_trigger)
            result = await run()
            # Let the observer finish rather than cancelling an unobserved signal.
    else:
        result = await run()
    assert run_ended is not None
    elapsed_after_stop = None if stop_started is None else run_ended - stop_started
    elapsed_after_deadline = None if deadline_started is None else run_ended - deadline_started
    store = LocalStore(root / "runs" / "boundary-run")
    committed = read_committed(store)
    assert committed is not None
    events = [event.model_dump(mode="json") for event in committed[1]]
    print(
        json.dumps(
            {
                "elapsed_to_trigger": elapsed_to_trigger,
                "elapsed_after_stop": elapsed_after_stop,
                "elapsed_after_deadline": elapsed_after_deadline,
                "round_deadline_observed": deadline_clock is not None
                and deadline_clock.deadline_observed,
                "result": list(result),
                "events": events,
            }
        ),
        flush=True,
    )


def _probe(mode: str, root: Path) -> tuple[dict[str, Any], str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), env.get("PYTHONPATH", "")])
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import anyio, sys; from pathlib import Path; "
            "from test_extension_boundaries import _worker; "
            "anyio.run(_worker, sys.argv[1], Path(sys.argv[2]))",
            mode,
            str(root),
        ],
        env=env,
        capture_output=True,
        text=True,
        # Several bounded signal waits may precede the independently bounded stop.
        timeout=4 * _SIGNAL_TIMEOUT_S + 2 * _STOP_TIMEOUT_S,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(process.stdout.splitlines()[-1]), process.stdout


def test_blocked_request_hook_abandoned_with_peer_and_renewal(tmp_path: Path) -> None:
    data, stdout = _probe("hook", tmp_path)
    assert data["elapsed_to_trigger"] < 20
    assert "HOOK_THREAD swarm-arena-daemon-" in stdout
    assert "HOOK_THREAD MainThread" not in stdout
    assert "RENEWED_AFTER_HOOK" in stdout
    assert data["elapsed_after_stop"] < 7
    assert any(e["type"] == "stop_requested" and e["payload"]["now"] for e in data["events"])
    assert any(e["type"] == "tool_end" and e["agent"] == "peer" for e in data["events"])
    assert any(e["type"] == "goal_finished" for e in data["events"])


def test_blocked_check_keeps_output_and_durable_settlement(tmp_path: Path) -> None:
    data, stdout = _probe("check", tmp_path)
    assert data["elapsed_to_trigger"] < 7
    assert "CHECK_ENTERED" in stdout
    assert data["elapsed_after_stop"] < 7
    assert any(e["type"] == "stop_requested" and e["payload"]["now"] for e in data["events"])
    ends = [e for e in data["events"] if e["type"] == "tool_end"]
    assert len(ends) == 1
    candidate = ends[0]["payload"]["candidate"]
    assert candidate["output"]["status"] == "ok"
    assert candidate["output"]["data"] == {"value": 1}
    assert candidate["checks"] == [
        {"check": "hang", "passed": False, "reasons": ["check interrupted"], "details": {}}
    ]
    assert any(e["type"] == "settle" and e["seq"] > ends[0]["seq"] for e in data["events"])
    assert any(e["type"] == "goal_finished" for e in data["events"])


@pytest.mark.parametrize("mode", ["check-serializer", "output-serializer", "partial-serializer"])
@pytest.mark.parametrize("stop", [False, True], ids=["deadline", "stop-now"])
def test_blocked_result_serializer_is_supervised(tmp_path: Path, mode: str, stop: bool) -> None:
    data, stdout = _probe(mode + (":stop" if stop else ""), tmp_path)
    assert data["elapsed_to_trigger"] < 7
    assert "SERIALIZER_THREAD swarm-arena-daemon-" in stdout
    assert "SERIALIZER_THREAD MainThread" not in stdout
    assert "LOOP_RESPONSIVE" in stdout
    events = data["events"]
    if stop:
        assert data["elapsed_after_stop"] < 7
        assert any(e["type"] == "stop_requested" and e["payload"]["now"] for e in events)
    else:
        assert not any(e["type"] == "stop_requested" for e in events)
        assert data["elapsed_after_stop"] is None
        assert data["round_deadline_observed"]
        assert stdout.index("SERIALIZER_THREAD") < stdout.index(
            "ROUND_DEADLINE_AFTER_SERIALIZER_ENTRY"
        )
        if mode == "partial-serializer":
            assert stdout.index("PARTIAL_READY") < stdout.index(
                "ROUND_DEADLINE_AFTER_PARTIAL_READY"
            )
            assert stdout.index("ROUND_DEADLINE_AFTER_PARTIAL_READY") < stdout.index(
                "SERIALIZER_THREAD"
            )
        assert data["elapsed_after_deadline"] < 7
    ends = [event for event in events if event["type"] == "tool_end"]
    assert len(ends) == 1
    candidate = ends[0]["payload"]["candidate"]
    if mode == "check-serializer":
        assert candidate["output"]["status"] == "ok"
        assert candidate["output"]["data"] == {"value": 1}
        assert candidate["checks"] == [
            {
                "check": "serialized",
                "passed": False,
                "reasons": ["check interrupted"],
                "details": {},
            }
        ]
    else:
        assert candidate["output"]["status"] == "error"
        assert candidate["output"]["data"] == {}
        assert candidate["warnings"] == ["tool error: CancelledError"]
    assert any(e["type"] == "settle" and e["seq"] > ends[0]["seq"] for e in events)
    assert any(e["type"] == "goal_finished" for e in events)


def test_collect_diagnostic_never_publishes_exception_values(tmp_path: Path) -> None:
    data, _ = _probe("collect", tmp_path)
    published = list((tmp_path / "runs").glob("**/logs/*.err"))
    assert len(published) == 1
    log = published[0].read_bytes()
    assert b"collect failed: ValueError" in log
    # The uploaded blob must exist independently of the published run log.
    assert any(p.read_bytes() == log for p in (tmp_path / "artifacts").rglob("*") if p.is_file())
    for sentinel in ("COLLECT_SECRET_29", "NOTE_SECRET_29", "CAUSE_SECRET_29"):
        assert sentinel.encode() not in log
        assert sentinel not in json.dumps(data["events"])
        assert all(
            sentinel.encode() not in p.read_bytes()
            for p in (tmp_path / "artifacts").rglob("*")
            if p.is_file()
        )


@pytest.mark.parametrize("kind", ["object", "json", "type", "utf8"])
def test_malformed_manager_is_contained_beside_healthy_peer(tmp_path: Path, kind: str) -> None:
    data, _ = _probe(f"manager:{kind}", tmp_path)
    assert data["result"] == ["g"]
    assert any(e["type"] == "tool_end" and e["agent"] == "peer" for e in data["events"])
    assert any(e["type"] == "goal_finished" for e in data["events"])
    statuses = [e for e in data["events"] if e["type"] == "agent_status" and e["agent"] == "a"]
    # Complete report validation now rejects malformed summaries before loop access.
    expected = "ValidationError" if kind in {"json", "utf8"} else "TypeError"
    assert any(e["payload"]["reason"] == f"manager error: {expected}" for e in statuses)


@pytest.mark.parametrize("mode", ["rubric", "panel", "lex-rubric", "lex-panel"])
@pytest.mark.parametrize("stop", [False, True], ids=["renewal", "stop-now"])
def test_blocked_judge_metadata_keeps_loop_responsive(
    tmp_path: Path, mode: str, stop: bool
) -> None:
    data, stdout = _probe(mode + (":stop" if stop else ""), tmp_path)
    assert data["elapsed_to_trigger"] < 20
    assert "JUDGE_THREAD swarm-arena-daemon-" in stdout
    assert "JUDGE_THREAD MainThread" not in stdout
    assert data["elapsed_after_stop"] < 7
    assert data["renewed"]
    assert data["stop_recorded"] is stop


def test_malformed_score_does_not_abort_later_judge_or_goal(tmp_path: Path) -> None:
    data, _ = _probe("score", tmp_path)
    assert set(data["result"]) == {"g", "peer"}
    assert {e["goal_id"] for e in data["events"] if e["type"] == "goal_finished"} == {"g", "peer"}
    assert any(e["type"] == "judge_error" and e["goal_id"] == "g" for e in data["events"])
    assert any(e["type"] == "score" and e["goal_id"] == "g" for e in data["events"])
    assert any(e["type"] == "score" and e["goal_id"] == "peer" for e in data["events"])
