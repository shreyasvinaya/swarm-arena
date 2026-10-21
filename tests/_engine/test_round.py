import dataclasses
import threading
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import pytest
from _fakes.engine import EngineEnv, take_over_retrying
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import PyTool, write_then_sleep

from swarm_arena import (
    AgentSpec,
    FunctionTool,
    Goal,
    PartSpec,
    Plan,
    Ref,
    RoundSpec,
    ScriptedManager,
    TeamSpec,
)
from swarm_arena._agent.context import AgentContext
from swarm_arena._agent.manager import Manager
from swarm_arena._agent.tool_run import Inflight
from swarm_arena._agent.watchdog import Watchdog
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import LLMScope, current_llm_scope
from swarm_arena._engine import round as round_module
from swarm_arena._engine.goal import r0_revision_draft
from swarm_arena._engine.round import AgentSlot, RoundRunner
from swarm_arena._errors import Fenced
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import Resources
from swarm_arena._model.planning_types import RoundReport
from swarm_arena._model.results import AgentStatus
from swarm_arena._model.tooling import Estimate, RunStatus, ToolOutput
from swarm_arena._scoring.check import Check
from swarm_arena._tooling.tool import Tool
from swarm_arena.executor import Executor, Job

PLAN = Plan(
    goal=Goal(id="g", namespace="n"),
    teams=[],
    judges=[Ref(path="swarm_arena:ProgramJudge", config={"function": "_fakes.tools:score_min_y"})],
)


def _with_manager(name: str, manager: Manager) -> AgentSlot:
    ref = Ref.of(FunctionTool("_fakes.tools:square"))
    spec = AgentSpec(name=name, rationale="Explore", tool=ref, manager=Ref.of(ScriptedManager([])))
    team = TeamSpec(name="t", rationale="Explore", agents=[spec])
    return AgentSlot(
        address=f"g/t/{name}",
        team=team,
        spec=spec,
        manager=manager,
        tool_ref=ref,
        goal=PLAN.goal,
        goal_key=PLAN.goal.key(),
        part=None,
        notes="",
        watchdog=Watchdog(Limits()),
        status=AgentStatus.ACTIVE,
    )


def _scripted(name: str, requests: Sequence[dict[str, Any]]) -> AgentSlot:
    return _with_manager(name, ScriptedManager(requests))


def _runner(name: str) -> AgentSlot:
    return _scripted(name, [{"inputs": {"x": 1.0}}])


class _BarrierManager(Manager):
    def __init__(
        self, starts: list[float], spans: list[tuple[float, float]], both_entered: anyio.Event
    ) -> None:
        super().__init__()
        self.starts, self.spans, self.both_entered = starts, spans, both_entered

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        start = anyio.current_time()
        self.starts.append(start)
        if len(self.starts) == 2:
            self.both_entered.set()
        await self.both_entered.wait()
        self.spans.append((start, anyio.current_time()))
        return RoundReport(summary="")


class _RoundTime:
    """Advance only the runner's clock, leaving executor and journal timing real."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.offset = 0.0
        self.changed = anyio.Event()
        self.ticked = anyio.Event()
        self.last_tick = float("-inf")
        monkeypatch.setattr(round_module, "anyio", self)

    def __getattr__(self, name: str) -> Any:
        return getattr(anyio, name)

    def current_time(self) -> float:
        return anyio.current_time() + self.offset

    def advance(self, seconds: float) -> None:
        self.offset += seconds
        self.changed.set()
        self.changed = anyio.Event()

    async def sleep_until(self, deadline: float) -> None:
        self.last_tick = self.current_time()
        self.ticked.set()
        self.ticked = anyio.Event()
        with anyio.move_on_after(max(0, deadline - self.current_time())):
            await self.changed.wait()

    async def wait_past(self, deadline: float) -> None:
        while self.last_tick < deadline:
            await self.ticked.wait()


class _Raiser(Manager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        raise RuntimeError("boom")


def _raiser(name: str) -> AgentSlot:
    return _with_manager(name, _Raiser())


def _slow_script(name: str, marker: Path) -> AgentSlot:
    slot = _scripted(name, [{"inputs": {}}])
    slot.tool_ref = Ref.of(PyTool({"code": write_then_sleep(marker)}))
    return slot


def _failing(name: str, runs: int) -> AgentSlot:
    slot = _scripted(name, [{"inputs": {}} for _ in range(runs)])
    slot.tool_ref = Ref.of(FunctionTool("_fakes.tools:boom", deterministic=False))
    slot.spec = slot.spec.model_copy(update={"replicates": runs})
    return slot


class _ScopeProbe(Manager):
    def __init__(self, seen: list[LLMScope]) -> None:
        super().__init__()
        self.seen = seen

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        scope = current_llm_scope()
        assert scope is not None
        self.seen.append(scope)
        call = await scope.tranche.reserve_call(Amount(usd=0.01))
        outcome = await scope.tranche.settle_call(call, Amount(usd=0.01))
        assert not outcome.overflow
        return RoundReport(summary="")


class _GateExecutor(Executor):
    def __init__(self, inner: Executor) -> None:
        super().__init__()
        self.inner = inner
        self.entered = anyio.Event()
        self.release = anyio.Event()

    async def submit(self, job: Job) -> ToolOutput:
        self.entered.set()
        await self.release.wait()
        return await self.inner.submit(job)

    async def capacity(self) -> Resources | None:
        return await self.inner.capacity()

    async def aclose(self) -> None:
        await self.inner.aclose()


def _round(env: EngineEnv, agents: Sequence[AgentSlot], **kw: Any) -> RoundRunner:
    return RoundRunner(
        env.deps, goal_id="g", plan=PLAN, agents=agents, round_spec=RoundSpec(), rounds_left=1, **kw
    )


def _construction_round(
    env: EngineEnv, component: type[Tool] | type[Check], agents: Sequence[AgentSlot]
) -> RoundRunner:
    slot = agents[0]
    plan = PLAN
    if component is Tool:
        slot.tool_ref = slot.tool_ref.model_copy(update={"name": "faulty"})
    else:
        part = PartSpec(
            name="p",
            goal=Goal(id="sub", namespace="n"),
            checks=[Ref(path="_fakes.tools:AlwaysPass", name="faulty")],
        )
        plan = PLAN.model_copy(update={"parts": [part]})
        slot.team = slot.team.model_copy(update={"part": "p"})
        slot.part, slot.goal, slot.goal_key = "p", part.goal, part.goal.key()
    return RoundRunner(
        env.deps, goal_id="g", plan=plan, agents=agents, round_spec=RoundSpec(), rounds_left=1
    )


@pytest.mark.parametrize("component", [Tool, Check], ids=["tool", "check"])
@pytest.mark.parametrize("stop", ["deadline", "stop_now", "fenced"])
async def test_runtime_constructor_is_supervised(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, component: type[Tool] | type[Check], stop: str
) -> None:
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    peer_ran = anyio.Event()
    manager_calls: list[str] = []
    original = Ref.build

    def blocked(self: Ref, base: type[Any]) -> Any:
        if self.name == "faulty" and base is component:
            entered.set()
            # The safety bound also releases the old synchronous check build on RED.
            release.wait(3)
            returned.set()
        return original(self, base)

    class Probe(Manager):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        async def run_round(self, ctx: AgentContext) -> RoundReport:
            manager_calls.append(self.name)
            peer_ran.set()
            return RoundReport(summary="peer ran")

    e = env.with_limits(round_timeout_s=0.5 if stop == "deadline" else 60.0)
    runner = _construction_round(
        e, component, [_with_manager("a", Probe("a")), _with_manager("b", Probe("b"))]
    )
    monkeypatch.setattr(Ref, "build", blocked)

    async def trigger() -> None:
        assert await anyio.to_thread.run_sync(entered.wait, 1)
        await peer_ran.wait()
        if stop == "stop_now":
            e.deps.stops.stop_now = True
        elif stop == "fenced":
            e.deps.journal.fenced.set()

    try:
        with anyio.fail_after(2):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(trigger)
                if stop == "fenced":
                    with pytest.raises(Fenced):
                        await runner.run(1)
                else:
                    out = await runner.run(1)
                    assert out.statuses == {
                        "t/a": AgentStatus.FINISHED,
                        "t/b": AgentStatus.FINISHED,
                    }
                    assert out.reasons["t/a"] == "runtime preparation cancelled"
                    assert len(e.events("agent_status")) == 2
        assert manager_calls == ["b"]
        assert not returned.is_set()
        before = e.event_types()
        release.set()
        assert await anyio.to_thread.run_sync(returned.wait, 1)
        await anyio.lowlevel.checkpoint()
        assert manager_calls == ["b"]
        assert e.event_types() == before
    finally:
        release.set()


@pytest.mark.parametrize("component", [Tool, Check], ids=["tool", "check"])
async def test_runtime_constructor_error_is_private_to_slot(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, component: type[Tool] | type[Check]
) -> None:
    secret = str(uuid4())
    original = Ref.build

    def raising(self: Ref, base: type[Any]) -> Any:
        if self.name == "faulty" and base is component:
            error = RuntimeError(secret)
            error.add_note(secret)
            raise error from ValueError(secret)
        return original(self, base)

    runner = _construction_round(env, component, [_runner("a"), _runner("b")])
    monkeypatch.setattr(Ref, "build", raising)
    try:
        out = await runner.run(1)
    except Exception as error:
        assert_secret_absent(error, secret, sanitized=True)
        raise
    assert out.statuses == {"t/a": AgentStatus.FINISHED, "t/b": AgentStatus.FINISHED}
    assert out.reasons == {"t/a": "agent error: RuntimeError", "t/b": ""}
    assert [candidate.agent for candidate in out.candidates] == ["b"]
    assert [event.agent for event in env.events("tool_start")] == ["b"]
    assert_secret_absent(RuntimeError(repr(out)), secret)
    assert_secret_absent(
        RuntimeError("".join(event.model_dump_json() for event in env.events("agent_status"))),
        secret,
    )


@pytest.mark.parametrize("component", [Tool, Check], ids=["tool", "check"])
@pytest.mark.parametrize("nested", [False, True])
async def test_runtime_constructor_fenced_propagates_bare(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    component: type[Tool] | type[Check],
    nested: bool,
) -> None:
    original = Ref.build

    def fenced(self: Ref, base: type[Any]) -> Any:
        if self.name == "faulty" and base is component:
            error = Fenced("run ownership lost")
            if nested:
                raise ExceptionGroup("construction failed", [error])
            raise error
        return original(self, base)

    runner = _construction_round(env, component, [_runner("a")])
    monkeypatch.setattr(Ref, "build", fenced)
    with pytest.raises(Fenced):
        await runner.run(1)
    assert not env.events("agent_status")


async def test_frontier_committed_before_any_tool_start(env: EngineEnv) -> None:
    await _round(env, [_runner("a"), _runner("b")]).run(1)
    types = env.event_types()
    assert types.index("frontier") < types.index("tool_start")
    assert env.is_durable("frontier")


async def test_round_started_commits_with_its_frontier(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = env.deps.board.frontier

    def committing_frontier() -> Frontier:  # forces a commit while the frontier is built
        anyio.from_thread.run(env.deps.journal.commit)
        return real()

    monkeypatch.setattr(env.deps.board, "frontier", committing_frontier)
    await _round(env, [_runner("a")]).run(1)
    rs, fr = env.events("round_started")[0], env.events("frontier")[0]
    assert fr.seq == rs.seq + 1
    assert env.segment_index(rs.seq) == env.segment_index(fr.seq)


async def test_managers_run_in_parallel(env: EngineEnv) -> None:
    starts: list[float] = []
    spans: list[tuple[float, float]] = []
    both_entered = anyio.Event()
    with anyio.fail_after(5):  # default round_timeout_s 14,400: the watchers must end
        await _round(
            env,
            [
                _with_manager(name, _BarrierManager(starts, spans, both_entered))
                for name in ("a", "b")
            ],
        ).run(1)
    assert max(s for s, _ in spans) < min(e for _, e in spans)  # the two sleeps overlap


async def test_manager_exception_ends_only_that_agent(env: EngineEnv) -> None:
    out = await _round(env, [_raiser("a"), _runner("b")]).run(1)
    assert out.statuses == {"t/a": AgentStatus.FINISHED, "t/b": AgentStatus.FINISHED}
    assert out.reasons["t/a"] == "manager error: RuntimeError"
    assert len(out.candidates) == 1


async def test_fenced_propagates_from_manager(env: EngineEnv) -> None:
    env.on_commit(
        "frontier", lambda ev: (take_over_retrying(env.store), None)[1]
    )  # fenced before any reserve commits
    with pytest.raises(Fenced):
        await _round(env, [_runner("a"), _runner("b")]).run(1)
    assert not any(
        "manager error" in (e.payload.get("reason") or "") for e in env.events("agent_status")
    )


async def test_same_round_identical_requests_both_run(env: EngineEnv) -> None:
    same = [{"inputs": {"x": 1.0}}]
    out = await _round(env, [_scripted("a", same), _scripted("b", same)]).run(1)
    assert len({c.id for c in out.candidates}) == 2
    assert env.event_types().count("tool_start") == 2
    assert {c.agent for c in out.candidates} == {"a", "b"}  # neither result is linked to the other


async def test_earlier_round_identical_request_is_linked(env: EngineEnv) -> None:
    first = await _round(env, [_scripted("a", [{"inputs": {"x": 1.0}}])]).run(1)
    out = await _round(
        env, [_scripted("b", [{"inputs": {"x": 1.0}}])], repeat_entries=first.repeat_entries
    ).run(2)
    assert out.candidates == []
    assert env.event_types().count("tool_start") == 1


async def test_restarted_round_hands_on_its_resumed_entries_at_its_end(env: EngineEnv) -> None:
    x = [{"inputs": {"x": 1.0}}]
    before = await _round(env, [_scripted("a", x), _scripted("b", [])]).run(
        1
    )  # a's committed x=1.0, then the "crash"
    ((key, cx),) = before.repeat_entries
    frontier = Frontier.model_validate(env.events("frontier")[0].payload["frontier"])
    out = await _round(
        env, [_scripted("a", x), _scripted("b", x)], frontier=frontier, resumed={"t/a": [(key, cx)]}
    ).run(1)
    assert [e.agent for e in env.events("tool_start")] == [
        "a",
        "b",
    ]  # restart: a gets return_own, b runs its own
    cb = next(c for c in out.candidates if c.agent == "b")
    assert out.repeat_entries == [(key, cx), (key, cb)]  # a's private pair joins at the round's end


async def test_round_timeout_cancels_with_partial(
    env: EngineEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _RoundTime(monkeypatch)
    marker = tmp_path / "m"
    e = env.with_limits(round_timeout_s=5.0)

    async def expire_after_start() -> None:
        await e.wait_for(marker)
        clock.advance(5.0)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(expire_after_start)
            out = await _round(e, [_slow_script("a", marker)]).run(1)
    assert out.candidates[0].output.status == RunStatus.CANCELLED
    assert out.candidates[0].output.partial
    assert out.candidates[0].round == 1


async def test_run_budget_drains_then_cancels_inflight(env: EngineEnv, tmp_path: Path) -> None:
    marker = tmp_path / "m"
    slot = _slow_script("a", marker)
    slot.tool_ref = Ref.of(PyTool({"code": write_then_sleep(marker), "grace_s": 0.1}))

    async def budget_when_marker() -> None:
        await env.wait_for(marker)
        env.deps.stops.run_budget = True

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(budget_when_marker)
            out = await _round(env, [slot]).run(1)
    c = out.candidates[0]
    assert c.output.status == RunStatus.CANCELLED
    assert c.output.partial is True
    assert env.event_types().count("tool_start") == 1


async def test_run_budget_flag_is_never_cleared(env: EngineEnv) -> None:
    env.deps.stops.run_budget = True  # the ledger is not exhausted
    out = await _round(env, [_runner("a"), _scripted("b", [])]).run(1)
    assert out.candidates == []
    assert env.deps.stops.run_budget is True


@pytest.mark.parametrize("flag", ["stop", "run_budget", "goal_budget"])
async def test_queued_job_never_starts_after_admission_closes(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    gate = _GateExecutor(
        env.deps.executor
    )  # queue-entry barrier: `entered`; release barrier: `release`
    e = dataclasses.replace(env, deps=dataclasses.replace(env.deps, executor=gate))
    closed, real_close = anyio.Event(), Inflight.close

    def signalling_close(self: Inflight) -> None:
        real_close(self)
        closed.set()

    monkeypatch.setattr(Inflight, "close", signalling_close)

    async def close_while_queued() -> None:
        await gate.entered.wait()  # b is reserved and waits inside submit
        e.deps.stops.goal_budget.add("g") if flag == "goal_budget" else setattr(
            e.deps.stops, flag, True
        )
        await closed.wait()  # the stop watcher has closed admission
        gate.release.set()  # only now can b reach on_started

    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(close_while_queued)
            out = await _round(e, [_runner("b")]).run(1)
    assert "tool_start" not in e.event_types()
    assert out.candidates == []
    (res_b,) = [
        ev.payload["reservation"]["id"]
        for ev in e.events("reserve")
        if ev.payload["reservation"]["kind"] == "tool"
    ]
    settle_b = next(ev for ev in e.events("settle") if ev.payload["id"] == res_b)
    assert Amount.model_validate(settle_b.payload["actual"]) == Amount()


async def test_starved_agent(
    env: EngineEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _RoundTime(monkeypatch)
    e = env.with_executor(max_concurrent=1).with_limits(round_timeout_s=5.0)
    started = anyio.Event()
    both_submitted = anyio.Event()
    submissions = 0
    original = e.deps.executor.submit

    async def submit(job: Job) -> ToolOutput:
        nonlocal submissions
        submissions += 1
        if submissions == 2:
            both_submitted.set()
        return await original(job)

    monkeypatch.setattr(e.deps.executor, "submit", submit)

    async def observe(marker: Path) -> None:
        await e.wait_for(marker)
        started.set()

    async def expire_after_start() -> None:
        await started.wait()
        await both_submitted.wait()
        clock.advance(5.0)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(observe, tmp_path / "ma")
            tasks.start_soon(observe, tmp_path / "mb")
            tasks.start_soon(expire_after_start)
            out = await _round(
                e, [_slow_script("a", tmp_path / "ma"), _slow_script("b", tmp_path / "mb")]
            ).run(1)
            tasks.cancel_scope.cancel()
    assert sorted(out.statuses.values()) == [AgentStatus.FINISHED, AgentStatus.STARVED]
    assert sum(p.exists() for p in (tmp_path / "ma", tmp_path / "mb")) == 1


async def test_watchdog_stops_agent(env: EngineEnv) -> None:
    out = await _round(env, [_failing("a", runs=3)]).run(1)
    assert out.statuses["t/a"] == AgentStatus.WATCHDOG_STOPPED
    assert "watchdog" in env.event_types()


async def test_stop_blocks_new_runs(env: EngineEnv) -> None:
    env.deps.stops.stop = True
    out = await _round(env, [_runner("a")]).run(1)
    assert out.candidates == []
    assert "tool_start" not in env.event_types()
    assert out.statuses == {"t/a": AgentStatus.FINISHED}


async def test_stop_now_cancels_inflight_with_partial(env: EngineEnv, tmp_path: Path) -> None:
    marker = tmp_path / "m"

    async def stop_when_marker() -> None:  # not on the tool_start commit, which comes before "go"
        await env.wait_for(marker)
        env.deps.stops.stop_now = True

    with anyio.fail_after(20):  # grace_s 10; the tool would sleep 600 s
        async with anyio.create_task_group() as tg:
            tg.start_soon(stop_when_marker)
            out = await _round(env, [_slow_script("a", marker)]).run(1)
    c = out.candidates[0]
    assert (c.output.status, c.output.partial, c.output.data) == (
        RunStatus.CANCELLED,
        True,
        {"step": 1},
    )
    assert env.is_durable("tool_end")


async def test_goal_budget_exhausted_blocks_new_runs_and_drains(env: EngineEnv) -> None:
    from swarm_arena._agent.manager import LimitReached
    from swarm_arena._model.tooling import Usage

    errors: list[LimitReached] = []
    second_started = anyio.Event()
    third_refused = anyio.Event()

    class PaidExecutor(_GateExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            assert job.on_started is not None
            await job.on_started(Job.Started(kind="process", job_id=job.id))
            assert job.request is not None
            if job.request.inputs["x"] == 2:
                second_started.set()
                await third_refused.wait()
            else:
                await second_started.wait()
            return ToolOutput(status=RunStatus.OK, usage=Usage(usd=0.85))

    class BudgetManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            async def first_then_third() -> None:
                await ctx.run_tool({"x": 1})
                try:
                    await ctx.run_tool({"x": 3})
                except LimitReached as error:
                    errors.append(error)
                    third_refused.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(first_then_third)
                tasks.start_soon(ctx.run_tool, {"x": 2})
            return RoundReport(summary="")

    e = env.with_limits(max_usd=2.0, default_run_reservation=Estimate(tokens=0, usd=0.5))
    e = dataclasses.replace(
        e, deps=dataclasses.replace(e.deps, executor=PaidExecutor(e.deps.executor))
    )
    out = await _round(e, [_with_manager("a", BudgetManager())]).run(1)
    assert len(errors) == 1
    assert errors[0].limit == "budget"
    assert len(out.candidates) == 2
    assert all(c.output.status == RunStatus.OK for c in out.candidates)
    assert "g" in e.deps.stops.goal_budget
    assert e.event_types().count("tool_start") == 2


async def test_agent_round_task_journaled(env: EngineEnv) -> None:
    await _round(env, [_runner("a"), _runner("b")]).run(1)
    assert sorted(e.payload["a2a"]["id"] for e in env.events("agent_status")) == [
        "g/t/a#r1",
        "g/t/b#r1",
    ]


async def test_agent_status_has_round_task_id(env: EngineEnv) -> None:
    await _round(env, [_runner("a"), _raiser("b")]).run(2)
    events = env.events("agent_status")
    assert len(events) == 2
    for event in events:
        assert event.context_id == env.deps.run_id
        assert event.task_id
        assert event.task_id == event.payload["a2a"]["id"] == f"g/t/{event.agent}#r2"


async def test_manager_round_runs_inside_its_llm_scope(env: EngineEnv) -> None:
    seen: list[LLMScope] = []
    await _round(
        env, [_with_manager("a", _ScopeProbe(seen)), _with_manager("b", _ScopeProbe(seen))]
    ).run(1)
    assert sorted(
        (s.role, s.run_id, s.goal_id, s.team, s.agent, s.round, s.candidate_id) for s in seen
    ) == [
        ("manager", env.deps.run_id, "g", "t", "a", 1, None),
        ("manager", env.deps.run_id, "g", "t", "b", 1, None),
    ]
    assert all(
        s.journal is env.deps.journal
        and s.models is env.deps.config.models
        and s.cache is env.deps.config.cache
        for s in seen
    )
    tranche_holders = sorted(
        e.payload["reservation"]["holder"]
        for e in env.events("reserve")
        if e.payload["reservation"]["kind"] == "tranche"
    )
    assert tranche_holders == ["g/t/a", "g/t/b"]  # scope.tranche is each agent's own tranche
    assert current_llm_scope() is None  # no scope leaks out of the round


async def test_manager_error_never_exposes_exception_values(env: EngineEnv) -> None:
    SECRET = "secret-manager-exception-sentinel"

    class SecretManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            error = RuntimeError(SECRET)
            error.add_note(SECRET)
            raise error from ValueError(SECRET)

    out = await _round(env, [_with_manager("a", SecretManager()), _runner("b")]).run(1)
    assert out.reasons["t/a"] == "manager error: RuntimeError"
    assert SECRET not in repr(out)
    assert SECRET not in "".join(e.model_dump_json() for e in env.events("agent_status"))
    assert len(out.candidates) == 1


async def test_fenced_manager_raises_bare_fenced(env: EngineEnv) -> None:
    class FencedManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            raise Fenced("run ownership lost")

    with pytest.raises(Fenced):
        await _round(env, [_with_manager("a", FencedManager()), _scripted("b", [])]).run(1)
    assert env.events("agent_status") == []


async def test_notes_and_learnings_survive_round(env: EngineEnv) -> None:
    class NoteManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            await ctx.save_notes("next round")
            candidate = await ctx.run_tool({"x": 1})
            await ctx.post_learning(
                title="Observation", text="Try another input", evidence=[candidate.id]
            )
            return RoundReport(summary="saved")

    slot = _with_manager("a", NoteManager())
    out = await _round(env, [slot]).run(1)
    assert slot.notes == "next round"
    assert len(out.learnings) == 1
    assert out.learnings[0].goal_id == "g"
    assert env.events("agent_status")[0].payload["a2a"]["artifacts"]


async def test_resume_does_not_open_or_journal_round_again(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await _round(env, [_runner("a")]).run(1)
    frontier = Frontier.model_validate(env.events("frontier")[0].payload["frontier"])

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("restarted round must reuse shares")

    monkeypatch.setattr(env.deps.ledger, "open_round", refuse)
    await _round(env, [_runner("a")], frontier=frontier, resumed={"t/a": first.repeat_entries}).run(
        1
    )
    assert len(env.events("round_started")) == len(env.events("frontier")) == 1
    assert env.events("frontier")[0].payload["round"] == 1


async def test_restored_nonstarted_candidate_does_not_count_as_watchdog_progress(
    env: EngineEnv,
) -> None:
    from _fakes.tools import CommandRaisesTool

    slot = _scripted("a", [{"inputs": {}}])
    slot.tool_ref = Ref.of(CommandRaisesTool({"code": "pass"}))
    first = await _round(env, [slot]).run(1)
    assert not env.events("tool_start")
    assert len(first.candidates) == 1
    frontier = Frontier.model_validate(env.events("frontier")[0].payload["frontier"])
    slot.manager = ScriptedManager([])
    slot.watchdog = Watchdog(Limits(watchdog_idle_rounds=1))
    out = await _round(env, [slot], frontier=frontier, resumed={"t/a": first.repeat_entries}).run(1)
    assert out.statuses["t/a"] == AgentStatus.FINISHED
    assert slot.watchdog.reason is None


async def test_restored_lost_start_counts_for_watchdog(env: EngineEnv) -> None:
    slot = _scripted("a", [])
    slot.watchdog = Watchdog(Limits(watchdog_idle_rounds=1))
    out = await _round(env, [slot], started_runs={"t/a": 1}).run(1)
    assert out.statuses["t/a"] == AgentStatus.WATCHDOG_STOPPED


@pytest.mark.parametrize("flag", ["stop", "run_budget", "goal_budget", "run_timeout"])
async def test_converge_uses_final_holder_and_ignores_ordinary_stops(
    env: EngineEnv, flag: str
) -> None:
    env.deps.journal.append(r0_revision_draft(PLAN, PLAN.base_dir))
    if flag == "goal_budget":
        env.deps.stops.goal_budget.add("g")
    else:
        setattr(env.deps.stops, flag, True)
    seen: list[LLMScope] = []
    out = await _round(
        env, [_with_manager("a", _ScopeProbe(seen)), _runner("b")], converge=True
    ).run(2)
    assert len(out.candidates) == 1
    rs = env.events("round_started")[0]
    assert rs.payload["converge"] is True
    assert rs.payload["round"] == 2
    assert rs.payload["shares"] == {}
    assert env.segment_index(rs.seq) == env.segment_index(env.events("frontier")[0].seq)
    assert env.events("round_closed") == []
    assert all(
        event.payload["reservation"]["holder"] == "g/:final" for event in env.events("reserve")
    )
    from swarm_arena._journal.replay import RunState

    state = RunState.from_events(env._events())
    assert state.goals["g"].converge_round == 2
    assert state.goals["g"].round == 0
    assert state.goals["g"].rounds_closed == 0


async def test_converge_stop_now_blocks_tools(env: EngineEnv) -> None:
    env.deps.stops.stop_now = True
    out = await _round(env, [_runner("a")], converge=True).run(1)
    assert out.candidates == []
    assert env.events("tool_start") == []


async def test_converge_inherits_assigned_part_parents(env: EngineEnv) -> None:
    from swarm_arena import PartSpec
    from swarm_arena._errors import SoftError
    from swarm_arena._model.results import Score

    part = PartSpec(name="p", goal=Goal(id="sub", namespace="n"))
    plan = PLAN.model_copy(update={"parts": [part]})
    env.deps.journal.append(r0_revision_draft(plan, plan.base_dir))
    slot = _runner("a")
    slot.team = slot.team.model_copy(update={"part": "p"})
    slot.part, slot.goal, slot.goal_key = "p", part.goal, part.goal.key()
    first = await RoundRunner(
        env.deps, goal_id="g", plan=plan, agents=[slot], round_spec=RoundSpec(), rounds_left=1
    ).run(1)
    parent = first.candidates[0]
    assert parent.goal_id == "g"
    assert parent.part == "p"
    end = env.events("tool_end")[0]
    env.deps.board.put_score(
        Score(id="score", candidate_id=parent.id, judge="program", key=(1.0,)),
        epoch=end.epoch,
        seq=end.seq,
    )

    class JoinManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            assert ctx.seed_parents == (parent.id,)
            assert ctx.get_candidate(parent.id).id == parent.id
            with pytest.raises(SoftError, match="own part"):
                ctx.best(1, part="p")
            await ctx.run_tool({"x": 2}, parents=[parent.id])
            return RoundReport(summary="joined")

    out = await RoundRunner(
        env.deps,
        goal_id="g",
        plan=plan,
        agents=[_with_manager("join", JoinManager())],
        round_spec=RoundSpec(),
        rounds_left=1,
        converge=True,
        seed_parents=[parent.id, parent.id],
    ).run(2)
    assert len(out.candidates) == 1
    assert out.candidates[0].parent_ids == [parent.id]
    assert [p.candidate_id for p in out.candidates[0].request.parents] == [parent.id]
    assert out.candidates[0].read_ids == [parent.id]


async def test_part_unnamed_ranking_and_digest_round(env: EngineEnv) -> None:
    from swarm_arena import PartSpec
    from swarm_arena._model.results import Score

    part = PartSpec(name="p", goal=Goal(id="sub", namespace="n"))
    plan = PLAN.model_copy(update={"parts": [part]})
    slot = _runner("a")
    slot.team = slot.team.model_copy(update={"part": "p"})
    slot.part, slot.goal, slot.goal_key = "p", part.goal, part.goal.key()
    first = await RoundRunner(
        env.deps, goal_id="g", plan=plan, agents=[slot], round_spec=RoundSpec(), rounds_left=1
    ).run(1)
    c = first.candidates[0]
    end = env.events("tool_end")[0]
    env.deps.board.put_score(
        Score(id="score", candidate_id=c.id, judge="program", key=(1.0,)),
        epoch=end.epoch,
        seq=end.seq,
    )

    class RankingProbe(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            assert ctx.best(1)[0].candidate_id == c.id
            assert "1 new since the last round" in ctx.digest
            return RoundReport(summary="ranked")

    class LearnManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            candidate = await ctx.run_tool({"x": 3})
            await ctx.post_learning(
                title="Seen", text="Use the first result", evidence=[candidate.id]
            )
            return RoundReport(summary="")

    # Publish a namespace learning in the prior round before the next frontier.
    learner = _with_manager("l", LearnManager())
    await _round(env, [learner]).run(1)
    await env.deps.journal.commit()
    slot.manager = RankingProbe()
    out = await RoundRunner(
        env.deps, goal_id="g", plan=plan, agents=[slot], round_spec=RoundSpec(), rounds_left=1
    ).run(2)
    assert out.reasons["t/a"] == ""


@pytest.mark.parametrize("flag", ["run_budget", "goal_budget"])
async def test_budget_drain_uses_only_inflight_job_grace(
    env: EngineEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    marker = tmp_path / "drain"
    slot = _slow_script("a", marker)
    slot.tool_ref = Ref.of(PyTool({"code": write_then_sleep(marker), "grace_s": 0.2}))
    idle = _scripted("b", [])
    idle.tool_ref = Ref.of(FunctionTool("_fakes.tools:square", grace_s=600.0))
    times: dict[str, float] = {}
    real = Inflight.cancel_all

    def record(self: Inflight) -> None:
        times["cancel"] = anyio.current_time()
        real(self)

    monkeypatch.setattr(Inflight, "cancel_all", record)
    real_close = Inflight.close

    def close(self: Inflight) -> None:
        times.setdefault("closed", anyio.current_time())
        real_close(self)

    monkeypatch.setattr(Inflight, "close", close)

    async def latch() -> None:
        await env.wait_for(marker)
        times["flag"] = anyio.current_time()
        if flag == "goal_budget":
            env.deps.stops.goal_budget.add("g")
        else:
            env.deps.stops.run_budget = True

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(latch)
            out = await _round(env, [slot, idle]).run(1)
    assert 0.2 <= times["cancel"] - times["flag"] < 2.0
    assert times["cancel"] - times["closed"] < 0.4
    assert out.candidates[0].output.partial
    assert out.candidates[0].output.status == RunStatus.CANCELLED


async def test_cancellation_allows_executor_collection_after_grace(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _RoundTime(monkeypatch)
    entered = anyio.Event()
    delivered = anyio.Event()

    class CollectingExecutor(_GateExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            assert job.on_started is not None
            assert job.on_partial is not None
            await job.on_started(Job.Started(kind="process", job_id=job.id))
            entered.set()
            try:
                await anyio.sleep_forever()
            except anyio.get_cancelled_exc_class():
                with anyio.CancelScope(shield=True):
                    after_grace = clock.current_time() + 5.5
                    clock.advance(5.5)
                    await clock.wait_past(after_grace)
                    await job.on_partial(
                        ToolOutput(
                            status=RunStatus.CANCELLED, partial=True, data={"collected": True}
                        )
                    )
                    delivered.set()
                raise
            raise AssertionError("unreachable")

    e = dataclasses.replace(
        env, deps=dataclasses.replace(env.deps, executor=CollectingExecutor(env.deps.executor))
    )
    slot = _runner("a")
    slot.tool_ref = Ref.of(FunctionTool("_fakes.tools:square", grace_s=0.0))

    async def stop() -> None:
        await entered.wait()
        e.deps.stops.stop_now = True

    with anyio.fail_after(15):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(stop)
            out = await _round(e, [slot]).run(1)
    assert delivered.is_set()
    assert out.candidates[0].output.data == {"collected": True}
    assert e.is_durable("tool_end")


async def test_prestart_cancellation_settles_zero_and_retains_preparation(
    env: EngineEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from swarm_arena._model.tooling import Command, ToolRequest

    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    workspaces: list[Path] = []
    marker = tmp_path / "must-not-launch"
    slot = _scripted("a", [{"inputs": {}}])
    slot.tool_ref = Ref.of(PyTool({"code": f"open({str(marker)!r}, 'w').close()", "grace_s": 0.1}))
    original = PyTool.command

    def command(self: PyTool, request: ToolRequest, workspace: Path) -> Command:
        workspaces.append(workspace)
        entered.set()
        assert release.wait(10)
        (workspace / "late_write").write_text("retained")
        done.set()
        return original(self, request, workspace)

    monkeypatch.setattr(PyTool, "command", command)

    async def stop() -> None:
        assert await anyio.to_thread.run_sync(entered.wait, 5)
        env.deps.stops.stop_now = True

    try:
        with anyio.fail_after(15):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(stop)
                out = await _round(env, [slot]).run(1)
        assert out.statuses["t/a"] == AgentStatus.STARVED
        assert not env.events("tool_start")
        assert not env.events("command_start")
        assert Amount.model_validate(env.events("settle")[0].payload["actual"]) == Amount()
        before = env.event_types()
        release.set()
        assert await anyio.to_thread.run_sync(done.wait, 5)
        await anyio.sleep(0.05)
        assert env.event_types() == before
        assert not marker.exists()
        assert (workspaces[0] / "late_write").exists()
    finally:
        release.set()


async def test_effective_tool_uses_base_catalog_identity(env: EngineEnv) -> None:
    from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog
    from swarm_arena._tooling.staging import tool_identity

    slot = _runner("a")
    entry = CatalogEntry(
        name="square",
        kind="tool",
        ref=slot.tool_ref.path,
        config=dict(slot.tool_ref.config),
        description="Square a number",
    )
    config = env.deps.config.model_copy(update={"planner_catalog": PlannerCatalog(entries=[entry])})
    seen: list[Job] = []

    class RecordingExecutor(_GateExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            seen.append(job)
            return await self.inner.submit(job)

    deps = dataclasses.replace(
        env.deps, config=config, executor=RecordingExecutor(env.deps.executor)
    )
    override = {"timeout_s": 12.0}
    out = await RoundRunner(
        deps,
        goal_id="g",
        plan=PLAN,
        agents=[slot],
        round_spec=RoundSpec(tool_overrides=override),
        rounds_left=1,
    ).run(1)
    assert seen[0].tool is not None
    assert seen[0].tool.config["timeout_s"] == 12.0
    assert out.candidates[0].request.round_overrides == override
    assert out.candidates[0].tool_identity == tool_identity(slot.tool_ref, entry)
    assert slot.tool_ref.config.get("timeout_s") != 12.0


async def test_tranche_pending_is_conservatively_closed(env: EngineEnv) -> None:
    class PendingManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            scope = current_llm_scope()
            assert scope is not None
            await scope.tranche.reserve_call(Amount(usd=0.01))
            return RoundReport(summary="pending")

    config = env.deps.config.model_copy(update={"settle_timeout_s": 0.05})
    e = dataclasses.replace(env, deps=dataclasses.replace(env.deps, config=config))
    with anyio.fail_after(5):
        await _round(e, [_with_manager("a", PendingManager())]).run(1)
    assert Amount.model_validate(env.events("settle")[0].payload["actual"]) == Amount(usd=0.01)


async def test_expired_run_clock_latches_timeout(env: EngineEnv) -> None:
    from swarm_arena._engine.stops import RunClock

    e = dataclasses.replace(env, deps=dataclasses.replace(env.deps, clock=RunClock(100_000.0)))
    out = await _round(e, [_runner("a")]).run(1)
    assert out.candidates == []
    assert e.deps.stops.run_timeout


async def test_nested_fenced_manager_raises_bare_fenced(env: EngineEnv) -> None:
    class NestedFencedManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            async def fenced() -> None:
                raise Fenced("run ownership lost")

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(fenced)
            return RoundReport(summary="")

    with pytest.raises(Fenced):
        await _round(env, [_with_manager("a", NestedFencedManager())]).run(1)
    assert env.events("agent_status") == []


async def test_idle_manager_cancelled_without_collection_allowance(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _RoundTime(monkeypatch)
    entered, cancelled = anyio.Event(), anyio.Event()
    original = Inflight.cancel_all

    def signal_cancel(self: Inflight) -> None:
        original(self)
        cancelled.set()

    monkeypatch.setattr(Inflight, "cancel_all", signal_cancel)

    class IdleManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            entered.set()
            await anyio.sleep_forever()
            return RoundReport(summary="")

    async def expire() -> None:
        await entered.wait()
        clock.advance(1.0)
        await cancelled.wait()
        clock.advance(5.0)

    e = env.with_limits(round_timeout_s=1.0)
    with anyio.fail_after(3):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(expire)
            out = await _round(e, [_with_manager("a", IdleManager())]).run(1)
    assert out.statuses == {"t/a": AgentStatus.FINISHED}


async def test_restarted_owner_can_read_and_parent_its_recovered_candidate(env: EngineEnv) -> None:
    from swarm_arena._errors import SoftError

    first = await _round(env, [_runner("a"), _scripted("b", [])]).run(1)
    parent = first.candidates[0]
    frontier = Frontier.model_validate(env.events("frontier")[0].payload["frontier"])

    class Owner(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            assert ctx.get_candidate(parent.id).id == parent.id
            await ctx.run_tool({"x": 2}, parents=[parent.id])
            return RoundReport(summary="")

    class Peer(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            with pytest.raises(SoftError, match="unknown candidate"):
                ctx.get_candidate(parent.id)
            await ctx.run_tool({"x": 3})
            return RoundReport(summary="")

    out = await _round(
        env,
        [_with_manager("a", Owner()), _with_manager("b", Peer())],
        frontier=frontier,
        resumed={"t/a": first.repeat_entries},
    ).run(1)
    assert len(out.candidates) == 2
    assert parent.id not in {c.id for c in out.candidates}
    child = next(c for c in out.candidates if c.agent == "a")
    assert child.parent_ids == [parent.id]
    assert child.read_ids == [parent.id]
    assert out.repeat_entries[0] == first.repeat_entries[0]


@pytest.mark.parametrize(
    "hook", ["get_description", "get_input_model", "editable", "extra_actions"]
)
@pytest.mark.parametrize("failure", ["raise", "hang"])
async def test_metadata_preparation_is_supervised_per_agent(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, hook: str, failure: str
) -> None:
    secret = str(uuid4())
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    owner = threading.get_ident()
    hook_threads: list[int] = []
    ticks: list[float] = []
    managers: list[str] = []
    original_build = Ref.build

    def build(self: Ref, base: type[Any]) -> Any:
        instance = original_build(self, base)
        if self.name == "faulty" and base is Tool:
            original_hook = getattr(instance, hook)

            def metadata() -> Any:
                hook_threads.append(threading.get_ident())
                entered.set()
                if failure == "raise":
                    error = ValueError(secret)
                    error.add_note(secret)
                    raise error from RuntimeError(secret)
                release.wait(1)
                returned.set()
                return original_hook()

            setattr(instance, hook, metadata)
        return instance

    class Probe(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            assert threading.get_ident() == owner
            managers.append(ctx.spec.name)
            ctx.search_learnings("heartbeat peer")
            return RoundReport(summary="healthy")

    async def heartbeat() -> None:
        while True:
            ticks.append(anyio.current_time())
            await anyio.sleep(0.005)

    e = env.with_limits(round_timeout_s=0.15)
    runner = _construction_round(
        e, Tool, [_with_manager("a", Probe()), _with_manager("b", Probe())]
    )
    monkeypatch.setattr(Ref, "build", build)
    try:
        start = anyio.current_time()
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(heartbeat)
            out = await runner.run(1)
            tasks.cancel_scope.cancel()
        elapsed = anyio.current_time() - start
        assert entered.is_set()
        assert managers == ["b"]
        assert hook_threads
        assert all(thread != owner for thread in hook_threads)
        assert out.reasons["t/a"]
        assert out.reasons["t/b"] == ""
        assert len(e.events("agent_status")) == 2
        assert_secret_absent(RuntimeError(repr(out)), secret)
        assert_secret_absent(RuntimeError(repr(e.events("agent_status"))), secret)
        if failure == "hang":
            assert not returned.is_set()
            assert elapsed < 0.7
            assert len(ticks) >= 5
            assert max(b - a for a, b in pairwise(ticks)) < 0.1
        before = e.event_types()
    finally:
        release.set()
    if failure == "hang":
        assert await anyio.to_thread.run_sync(returned.wait, 1)
        await anyio.lowlevel.checkpoint()
        assert e.event_types() == before
        assert managers == ["b"]


@pytest.mark.parametrize(
    ("part_policy", "main_policy"), [("pass_fail", "reasons"), ("reasons", "pass_fail")]
)
@pytest.mark.parametrize("reloaded", [False, True])
async def test_seed_feedback_follows_producer_group_and_revision(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    part_policy: str,
    main_policy: str,
    reloaded: bool,
) -> None:
    from swarm_arena._model.results import Candidate

    secret = str(uuid4())

    def check(policy: str) -> Ref:
        return Ref(
            path="_fakes.tools:FeedbackCheck",
            name="shared",
            config={"feedback": policy, "reason": secret},
        )

    part = PartSpec(name="p", goal=Goal(id="sub", namespace="n"), checks=[check(part_policy)])
    plan = PLAN.model_copy(update={"parts": [part], "checks": [check(main_policy)]})
    env.deps.journal.append(r0_revision_draft(plan, plan.base_dir))
    producer = _runner("a")
    producer.team = producer.team.model_copy(update={"part": "p"})
    producer.part, producer.goal, producer.goal_key = "p", part.goal, part.goal.key()
    out = await RoundRunner(
        env.deps,
        goal_id="g",
        plan=plan,
        agents=[producer, _runner("b")],
        round_spec=RoundSpec(),
        rounds_left=1,
    ).run(1)
    parents = out.candidates
    assert len(parents) == 2
    assert all(c.checks[0].reasons == [secret] for c in parents)
    from swarm_arena._journal.event import EventDraft

    # A durable result without a producer-status policy (for example a crash)
    # must stay private even when its check name matches a known group.
    unrecorded = parents[0].model_copy(update={"id": "c-unrecorded"})
    event = await env.deps.journal.append_durable(
        EventDraft(
            type="tool_end", goal_id="g", payload={"candidate": unrecorded.model_dump(mode="json")}
        )
    )
    env.deps.board.put_candidate(unrecorded, epoch=event.epoch, seq=event.seq)
    parents.append(unrecorded)
    if reloaded:
        from _fakes.tools import FeedbackCheck

        # A later connector load must not redefine a producing policy.
        original = FeedbackCheck.__init__

        def reload_check(self: FeedbackCheck, config: Any = None) -> None:
            original(self, config)
            self.feedback = "reasons"

        monkeypatch.setattr(FeedbackCheck, "__init__", reload_check)
    seen: list[Candidate] = []

    class Reader(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            seen.extend(ctx.get_candidate(id) for id in ctx.seed_parents)
            return RoundReport(summary="read parents")

    # Swap both policies in the receiving revision. The producing policy wins.
    revised = plan.model_copy(
        update={
            "revision": 1,
            "checks": [check(part_policy)],
            "parts": [part.model_copy(update={"checks": [check(main_policy)]})],
        }
    )
    env.deps.journal.append(
        EventDraft(
            type="plan_revision",
            goal_id="g",
            payload={
                "plan": revised.model_dump(mode="json"),
                "origin": "human",
                "first_round": 2,
            },
        )
    )
    reader = _with_manager("join", Reader())
    second = await RoundRunner(
        env.deps,
        goal_id="g",
        plan=revised,
        agents=[reader],
        round_spec=RoundSpec(),
        rounds_left=1,
        converge=True,
        seed_parents=[c.id for c in parents],
    ).run(2)
    assert second.reasons["t/join"] == ""
    assert len(seen) == 3
    for candidate in seen:
        policy = part_policy if candidate.part == "p" else main_policy
        if candidate.id == unrecorded.id:
            policy = "pass_fail"
        assert candidate.checks[0].reasons == ([secret] if policy == "reasons" else [])
        assert candidate.checks[0].details == ({"private": secret} if policy == "reasons" else {})
    assert all(
        e.payload["candidate"]["checks"][0]["reasons"] == [secret] for e in env.events("tool_end")
    )
