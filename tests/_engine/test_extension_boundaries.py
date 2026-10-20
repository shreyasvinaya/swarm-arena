"""Public-run regressions, isolated so a blocked extension cannot hang pytest."""

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
from swarm_arena.executor import Job, LocalExecutor
from swarm_arena.manager import Manager, RoundReport
from swarm_arena.scoring import Check, CheckResult, Judge, JudgeContext, Lexicographic
from swarm_arena.tool import RunStatus, ToolOutput

_HOOK_ENTERED = threading.Event()
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


async def _worker(mode: str, root: Path) -> None:
    # Journal/lease internals are used only for evidence, never to drive the run.
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
        if _JUDGE_ENTERED.is_set():
            _JUDGE_RENEWED.set()

    Lease.renew = renew
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
        limits=Limits(
            round_timeout_s=20 if variant == "stop" and mode.endswith("serializer") else 1.5,
            # Stop probes must outlive the stop-file poll and lease-renewal wait.
            run_timeout_s=30
            if variant == "stop"
            else 3
            if mode in {"rubric", "panel", "lex-rubric", "lex-panel"}
            else 8,
            tool_timeout_s=1,
        ),
        lease_ttl_s=1.5,
        settle_timeout_s=0.2,
    )
    started = time.monotonic()
    swarm = Swarm(config)

    if mode in {"check-serializer", "output-serializer", "partial-serializer"}:

        async def stop_on_serializer() -> None:
            if mode == "partial-serializer" and variant == "stop":
                assert await anyio.to_thread.run_sync(_PARTIAL_READY.wait, 5)
                await swarm.stop(now=True)
            assert await anyio.to_thread.run_sync(_SERIALIZER_ENTERED.wait, 5)
            # This heartbeat cannot advance while a serializer owns the event loop.
            await anyio.sleep(0.02)
            print("LOOP_RESPONSIVE", flush=True)
            if variant == "stop" and mode != "partial-serializer":
                await swarm.stop(now=True)

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(stop_on_serializer)
            result = await swarm.run(plans, run_id="boundary-run")
            tasks.cancel_scope.cancel()
    elif mode in {"rubric", "panel", "lex-rubric", "lex-panel"}:
        # Lifecycle termination belongs to the separate barrier-supervision fix.
        # This probe owns only thread isolation and live loop/lease/stop progress.
        async def run() -> None:
            await swarm.run(plans, run_id="boundary-run")

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            assert await anyio.to_thread.run_sync(_JUDGE_ENTERED.wait, 5)
            assert await anyio.to_thread.run_sync(_JUDGE_RENEWED.wait, 5)
            stop_recorded = False
            if variant == "stop":
                await swarm.stop(now=True)
                live_store = LocalStore(root / "runs" / "boundary-run")
                with anyio.fail_after(5):
                    while not stop_recorded:
                        live = await anyio.to_thread.run_sync(read_committed, live_store)
                        stop_recorded = live is not None and any(
                            e.type == "stop_requested" and e.payload["now"] for e in live[1]
                        )
                        if not stop_recorded:
                            await anyio.sleep(0.02)
            print(json.dumps({"renewed": True, "stop_recorded": stop_recorded}), flush=True)
            tasks.cancel_scope.cancel()
        return
    else:
        result = await swarm.run(plans, run_id="boundary-run")
    store = LocalStore(root / "runs" / "boundary-run")
    committed = read_committed(store)
    assert committed is not None
    events = [event.model_dump(mode="json") for event in committed[1]]
    print(
        json.dumps(
            {"elapsed": time.monotonic() - started, "result": list(result), "events": events}
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
        timeout=12,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout.splitlines()[-1]), process.stdout


def test_blocked_request_hook_abandoned_with_peer_and_renewal(tmp_path: Path) -> None:
    data, stdout = _probe("hook", tmp_path)
    assert "HOOK_THREAD swarm-arena-daemon-" in stdout
    assert "HOOK_THREAD MainThread" not in stdout
    assert "RENEWED_AFTER_HOOK" in stdout
    assert data["elapsed"] < 7
    assert any(e["type"] == "tool_end" and e["agent"] == "peer" for e in data["events"])
    assert any(e["type"] == "goal_finished" for e in data["events"])


def test_blocked_check_keeps_output_and_durable_settlement(tmp_path: Path) -> None:
    data, stdout = _probe("check", tmp_path)
    assert "CHECK_ENTERED" in stdout
    assert data["elapsed"] < 7
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
    assert "SERIALIZER_THREAD swarm-arena-daemon-" in stdout
    assert "SERIALIZER_THREAD MainThread" not in stdout
    assert "LOOP_RESPONSIVE" in stdout
    assert data["elapsed"] < 7
    events = data["events"]
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
    if stop:
        assert any(e["type"] == "stop_requested" and e["payload"]["now"] for e in events)


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
    expected = "ValueError" if kind == "utf8" else "TypeError"
    assert any(e["payload"]["reason"] == f"manager error: {expected}" for e in statuses)


@pytest.mark.parametrize("mode", ["rubric", "panel", "lex-rubric", "lex-panel"])
@pytest.mark.parametrize("stop", [False, True], ids=["renewal", "stop-now"])
def test_blocked_judge_metadata_keeps_loop_responsive(
    tmp_path: Path, mode: str, stop: bool
) -> None:
    data, stdout = _probe(mode + (":stop" if stop else ""), tmp_path)
    assert "JUDGE_THREAD swarm-arena-daemon-" in stdout
    assert "JUDGE_THREAD MainThread" not in stdout
    assert data["renewed"]
    assert data["stop_recorded"] is stop


def test_malformed_score_does_not_abort_later_judge_or_goal(tmp_path: Path) -> None:
    data, _ = _probe("score", tmp_path)
    assert set(data["result"]) == {"g", "peer"}
    assert {e["goal_id"] for e in data["events"] if e["type"] == "goal_finished"} == {"g", "peer"}
    assert any(e["type"] == "judge_error" and e["goal_id"] == "g" for e in data["events"])
    assert any(e["type"] == "score" and e["goal_id"] == "g" for e in data["events"])
    assert any(e["type"] == "score" and e["goal_id"] == "peer" for e in data["events"])
