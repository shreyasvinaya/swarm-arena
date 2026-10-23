import asyncio
import dataclasses
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
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
    RunConfig,
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
from swarm_arena._engine.round import AgentSlot, EngineDeps, RoundOutcome, RoundRunner
from swarm_arena._errors import Fenced, _core_error
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._journal.event import EventDraft
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import Resources
from swarm_arena._model.planning_types import RoundReport
from swarm_arena._model.results import AgentStatus
from swarm_arena._model.tooling import Estimate, RunStatus, ToolOutput
from swarm_arena._scoring.check import Check
from swarm_arena._store.artifacts import ArtifactStore
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
        with anyio.fail_after(10):
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
        with anyio.fail_after(10):
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


class _PreparedRound(RoundRunner):
    async def run(self, round: int) -> RoundOutcome:
        await self.deps.journal.commit()
        return await super().run(round)


def _prepared_round(deps: EngineDeps, *, plan: Plan, **kwargs: Any) -> RoundRunner:
    """Supply the initial committed plan normally owned by RunRunner."""
    from swarm_arena._journal.reader import read_committed

    committed = read_committed(deps.store)
    events = [*(committed[1] if committed else []), *deps.journal._queue]
    if not any(event.type == "plan_revision" for event in events):
        deps.journal.append(r0_revision_draft(plan, plan.base_dir))
    return _PreparedRound(deps, plan=plan, **kwargs)


def _round(env: EngineEnv, agents: Sequence[AgentSlot], **kw: Any) -> RoundRunner:
    return _prepared_round(
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
    return _prepared_round(
        env.deps, goal_id="g", plan=plan, agents=agents, round_spec=RoundSpec(), rounds_left=1
    )


@pytest.mark.parametrize("component", [Tool, Check], ids=["tool", "check"])
@pytest.mark.parametrize("stop", ["deadline", "stop_now", "fenced"])
async def test_runtime_constructor_is_supervised(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, component: type[Tool] | type[Check], stop: str
) -> None:
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    peer_ran = anyio.Event()
    clock = _RoundTime(monkeypatch)
    workers: list[threading.Thread] = []
    manager_calls: list[str] = []
    original = Ref.build

    def blocked(self: Ref, base: type[Any]) -> Any:
        if self.name == "faulty" and base is component:
            workers.append(threading.current_thread())
            entered.set()
            # The safety bound also releases the old synchronous check build on RED.
            assert release.wait(20), "constructor release deadline"
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

    e = env.with_limits(round_timeout_s=60.0)
    runner = _construction_round(
        e, component, [_with_manager("a", Probe("a")), _with_manager("b", Probe("b"))]
    )
    monkeypatch.setattr(Ref, "build", blocked)

    async def trigger() -> None:
        assert await anyio.to_thread.run_sync(entered.wait, 10)
        await peer_ran.wait()
        if stop == "deadline":
            clock.advance(61)
        elif stop == "stop_now":
            e.deps.stops.stop_now = True
        elif stop == "fenced":
            e.deps.journal.fenced.set()

    try:
        with anyio.fail_after(10):
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
        assert await anyio.to_thread.run_sync(returned.wait, 5)
        await anyio.to_thread.run_sync(workers[0].join, 5)
        assert not workers[0].is_alive()
        assert manager_calls == ["b"]
        assert e.event_types() == before
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()


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
            error = _core_error(Fenced, "run ownership lost")
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
    loop = asyncio.get_running_loop()

    def committing_frontier() -> Frontier:  # forces a commit while the frontier is built
        asyncio.run_coroutine_threadsafe(env.deps.journal.commit(), loop).result(timeout=5)
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
async def test_queued_job_obeys_admission_closure_kind(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    gate = _GateExecutor(
        env.deps.executor
    )  # queue-entry barrier: `entered`; release barrier: `release`
    e = dataclasses.replace(env, deps=dataclasses.replace(env.deps, executor=gate))
    closed, real_close = anyio.Event(), Inflight.close

    def signalling_close(self: Inflight, *, budget: bool = False) -> None:
        real_close(self, budget=budget)
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
    if flag == "stop":
        assert "tool_start" not in e.event_types()
        assert out.candidates == []
    else:
        assert e.event_types().count("tool_start") == 1
        assert len(out.candidates) == 1
        assert out.candidates[0].output.status == RunStatus.OK
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
    with anyio.fail_after(10):
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
            raise _core_error(Fenced, "run ownership lost")

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
    from swarm_arena._model.results import Score

    part = PartSpec(name="p", goal=Goal(id="sub", namespace="n"))
    plan = PLAN.model_copy(update={"parts": [part]})
    env.deps.journal.append(r0_revision_draft(plan, plan.base_dir))
    slot = _runner("a")
    slot.team = slot.team.model_copy(update={"part": "p"})
    slot.part, slot.goal, slot.goal_key = "p", part.goal, part.goal.key()
    first = await _prepared_round(
        env.deps, goal_id="g", plan=plan, agents=[slot], round_spec=RoundSpec(), rounds_left=1
    ).run(1)
    parent = first.candidates[0]
    assert parent.goal_id == "g"
    assert parent.part == "p"
    await env.deps.journal.append_durable(
        EventDraft(
            type="score",
            goal_id="g",
            payload={
                "score": Score(
                    id="score", candidate_id=parent.id, judge="program", key=(1.0,)
                ).model_dump(mode="json")
            },
        )
    )

    class JoinManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            assert ctx.seed_parents == (parent.id,)
            assert ctx.get_candidate(parent.id).id == parent.id
            assert ctx.best(1, part="p")[0].candidate_id == parent.id
            await ctx.run_tool({"x": 2}, parents=[parent.id])
            return RoundReport(summary="joined")

    out = await _prepared_round(
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
    first = await _prepared_round(
        env.deps, goal_id="g", plan=plan, agents=[slot], round_spec=RoundSpec(), rounds_left=1
    ).run(1)
    c = first.candidates[0]
    await env.deps.journal.append_durable(
        EventDraft(
            type="score",
            goal_id="g",
            payload={
                "score": Score(
                    id="score", candidate_id=c.id, judge="program", key=(1.0,)
                ).model_dump(mode="json")
            },
        )
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
    out = await _prepared_round(
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

    def close(self: Inflight, *, budget: bool = False) -> None:
        times.setdefault("closed", anyio.current_time())
        real_close(self, budget=budget)

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
    assert 5.0 <= times["cancel"] - times["flag"] < 7.0
    assert 5.0 <= times["cancel"] - times["closed"] < 7.0
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
    workers: list[threading.Thread] = []
    marker = tmp_path / "must-not-launch"
    slot = _scripted("a", [{"inputs": {}}])
    slot.tool_ref = Ref.of(PyTool({"code": f"open({str(marker)!r}, 'w').close()", "grace_s": 0.1}))
    original = PyTool.command

    def command(self: PyTool, request: ToolRequest, workspace: Path) -> Command:
        workers.append(threading.current_thread())
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
        with anyio.fail_after(5):
            await anyio.to_thread.run_sync(workers[0].join, 5)
        assert not workers[0].is_alive()
        assert env.event_types() == before
        assert not marker.exists()
        assert (workspaces[0] / "late_write").exists()
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()


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
    out = await _prepared_round(
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

    config = env.deps.config.model_copy(update={"settle_timeout_s": 2.0})
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
                raise _core_error(Fenced, "run ownership lost")

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
    workers: list[threading.Thread] = []
    beat_target: list[int] = []
    beats_reached = anyio.Event()
    peer_ran = anyio.Event()
    clock = _RoundTime(monkeypatch)
    managers: list[str] = []
    original_build = Ref.build

    def build(self: Ref, base: type[Any]) -> Any:
        instance = original_build(self, base)
        if self.name == "faulty" and base is Tool:
            original_hook = getattr(instance, hook)

            def metadata() -> Any:
                workers.append(threading.current_thread())
                hook_threads.append(threading.get_ident())
                entered.set()
                if failure == "raise":
                    error = ValueError(secret)
                    error.add_note(secret)
                    raise error from RuntimeError(secret)
                assert release.wait(20), "metadata release deadline"
                returned.set()
                return original_hook()

            setattr(instance, hook, metadata)
        return instance

    class Probe(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            assert threading.get_ident() == owner
            managers.append(ctx.spec.name)
            ctx.search_learnings("heartbeat peer")
            peer_ran.set()
            return RoundReport(summary="healthy")

    async def heartbeat() -> None:
        while True:
            ticks.append(anyio.current_time())
            if beat_target and len(ticks) >= beat_target[0]:
                beats_reached.set()
            await anyio.sleep(0.01)

    async def trigger_deadline() -> None:
        assert await anyio.to_thread.run_sync(entered.wait, 10)
        await peer_ran.wait()
        beat_target.append(len(ticks) + 5)
        with anyio.fail_after(2):
            await beats_reached.wait()
        assert not returned.is_set()
        clock.advance(61)

    e = env.with_limits(round_timeout_s=60)
    runner = _construction_round(
        e, Tool, [_with_manager("a", Probe()), _with_manager("b", Probe())]
    )
    monkeypatch.setattr(Ref, "build", build)
    try:
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tasks:
                if failure == "hang":
                    tasks.start_soon(heartbeat)
                    tasks.start_soon(trigger_deadline)
                out = await runner.run(1)
                tasks.cancel_scope.cancel()
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
            assert beats_reached.is_set()
        before = e.event_types()
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()
    if failure == "hang":
        assert returned.is_set()
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
    out = await _prepared_round(
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
    second = await _prepared_round(
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


@pytest.mark.parametrize("newest", [False, True])
async def test_round_gives_past_run_hint(env: EngineEnv, tmp_path: Path, newest: bool) -> None:
    from _fakes.journals import FakeRun, add_pointer

    from swarm_arena._agent.repeats import repeat_key
    from swarm_arena._board.leaderboard import format_value
    from swarm_arena.tool import ToolRequest

    slot = _runner("a")
    ranking = PLAN.ranking_name()
    past = FakeRun(str(tmp_path / "past-runs"), "r-past", goal=slot.goal)
    key = repeat_key(
        goal_key=slot.goal_key,
        tool_ref=slot.tool_ref,
        request=ToolRequest(inputs={"x": 1.0}),
        deterministic=True,
    )
    past.commit(past.plan(ranking=ranking), past.candidate("c-past", repeat_key=key))
    if newest:
        past.commit(
            past.candidate("c-new", repeat_key=key),
            past.candidate("c-partial", repeat_key=key, partial=True),
            past.candidate("c-late", repeat_key=key, status=RunStatus.TIMEOUT),
        )
    else:
        past.commit(past.score("s-past", "c-past", 0.25, judge=ranking))
    add_pointer(env.deps.config.arena_root, past)
    out = await _round(env, [slot]).run(1)
    (candidate,) = out.candidates
    assert candidate.run_id == env.deps.run_id
    expected = "c-new: unscored" if newest else f"c-past: {ranking}={format_value(0.25)}"
    assert f"past-run match {expected}" in candidate.warnings
    assert out.repeat_entries == [(key, candidate)]


async def test_round_gives_past_run_hint_for_part_agent(env: EngineEnv, tmp_path: Path) -> None:
    from _fakes.journals import FakeRun, add_pointer

    from swarm_arena._agent.repeats import repeat_key
    from swarm_arena.tool import ToolRequest

    slot = _runner("a")
    past = FakeRun(str(tmp_path / "past-runs"), "r-past", goal=PLAN.goal)
    draft = past.plan(parts=["a"], ranking=PLAN.ranking_name())
    recorded = Plan.model_validate(draft.payload["plan"])
    part = recorded.parts[0]
    slot.part, slot.goal, slot.goal_key = part.name, part.goal, part.goal.key()
    slot.team = slot.team.model_copy(update={"part": part.name})
    current = PLAN.model_copy(update={"parts": [part]})
    await env.deps.journal.append_durable(r0_revision_draft(current, current.base_dir))
    key = repeat_key(
        goal_key=slot.goal_key,
        tool_ref=slot.tool_ref,
        request=ToolRequest(inputs={"x": 1.0}),
        deterministic=True,
    )
    past.commit(draft, past.candidate("c-past", part="a", repeat_key=key))
    add_pointer(env.deps.config.arena_root, past)
    out = await _prepared_round(
        env.deps, goal_id="g", plan=current, agents=[slot], round_spec=RoundSpec(), rounds_left=1
    ).run(1)
    (candidate,) = out.candidates
    assert "past-run match c-past: unscored" in candidate.warnings
    assert out.repeat_entries == [(key, candidate)]


async def test_own_match_wins_over_past_run(env: EngineEnv, tmp_path: Path) -> None:
    from _fakes.journals import FakeRun, add_pointer

    from swarm_arena._agent.repeats import repeat_key
    from swarm_arena.tool import ToolRequest

    slot = _runner("a")
    past = FakeRun(str(tmp_path / "past-runs"), "r-past", goal=slot.goal)
    key = repeat_key(
        goal_key=slot.goal_key,
        tool_ref=slot.tool_ref,
        request=ToolRequest(inputs={"x": 1.0}),
        deterministic=True,
    )
    past.commit(past.plan(), past.candidate("c-past", repeat_key=key))
    add_pointer(env.deps.config.arena_root, past)
    first = await _round(env, [slot]).run(1)
    out = await _round(env, [slot], repeat_entries=first.repeat_entries).run(2)
    assert out.candidates == []
    assert any(e.payload.get("refused") == first.candidates[0].id for e in env.events("watchdog"))


def test_seed_view_preserves_part_and_count_capability() -> None:
    from unittest.mock import Mock

    from swarm_arena._board.base import BoardView
    from swarm_arena._engine.round import _SeedView

    view = Mock(spec=BoardView, goal=PLAN.goal.key(), frontier=Frontier(), part="p")
    view.count_learnings.return_value = 7
    wrapped = _SeedView(view, {})
    assert wrapped.part == "p"
    assert wrapped.count_learnings("n", goal=view.goal) == 7
    view.count_learnings.assert_called_once_with("n", goal=view.goal)


async def test_digest_ranks_parts_by_their_judges(env: EngineEnv) -> None:
    from swarm_arena import Score

    part = PartSpec(
        name="a",
        goal=Goal(id="sub", namespace="n"),
        judges=[
            Ref(
                path="swarm_arena:ProgramJudge",
                name="a",
                config={"function": "_fakes.tools:score_min_y"},
            )
        ],
    )
    plan = PLAN.model_copy(update={"parts": [part]})
    slot = _runner("part")
    slot.team = slot.team.model_copy(update={"part": "a"})
    slot.part, slot.goal, slot.goal_key = "a", part.goal, part.goal.key()
    first = await _prepared_round(
        env.deps, goal_id="g", plan=plan, agents=[slot], round_spec=RoundSpec(), rounds_left=2
    ).run(1)
    candidate = first.candidates[0]
    await env.deps.journal.append_durable(
        EventDraft(
            type="score",
            goal_id="g",
            payload={
                "score": Score(
                    id="a-score",
                    candidate_id=candidate.id,
                    judge="a",
                    key=(1.0,),
                    value=1,
                ).model_dump(mode="json")
            },
        )
    )
    seen: list[str] = []

    class DigestProbe(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            seen.append(ctx.digest)
            return RoundReport(summary="read digest")

    out = await _prepared_round(
        env.deps,
        goal_id="g",
        plan=plan,
        agents=[_with_manager("main", DigestProbe())],
        round_spec=RoundSpec(),
        rounds_left=1,
    ).run(2)
    assert out.reasons["t/main"] == ""
    assert len(seen) == 1
    assert any(line.startswith("Part a (a, ") for line in seen[0].splitlines())
    assert candidate.id in seen[0]


@pytest.fixture
async def provider_env(
    tmp_path: Path, run_config: RunConfig, artifacts: ArtifactStore, local_executor: LocalExecutor
) -> AsyncIterator[EngineEnv]:
    from _engine.conftest import env as engine_env
    from _fakes.engine import with_limits

    config = with_limits(run_config, max_usd=10.0, default_run_reservation=Estimate(usd=0.1))
    async with asynccontextmanager(cast(Any, engine_env).__wrapped__)(
        tmp_path, config, artifacts, local_executor
    ) as result:
        yield result


class _ProviderManager(Manager):
    def __init__(self, *, fail_first: bool = True) -> None:
        super().__init__()
        self.calls = 0
        self.fail_first = fail_first

    def models(self) -> tuple[str, ...]:
        return ("provider-model",)

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        from swarm_arena._errors import ProviderFailed

        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise _core_error(ProviderFailed, "anthropic")
        await ctx.run_tool({"x": self.calls})
        return RoundReport(summary="ok")


@pytest.mark.parametrize("healthy", [True, False, None])
async def test_provider_probe_recovery_before_frontier_and_shares(
    provider_env: EngineEnv, monkeypatch: pytest.MonkeyPatch, healthy: bool | None
) -> None:
    env = provider_env
    from swarm_arena._engine import stops as stops_module
    from swarm_arena.config import ModelCatalog, ModelSpec

    manager = _ProviderManager()
    slot = _with_manager("a", manager)
    peer = _runner("b")
    env.deps.config = env.deps.config.model_copy(
        update={
            "models": ModelCatalog(
                {"provider-model": ModelSpec(model="anthropic/test", roles=["manager"])}
            )
        }
    )
    order: list[str] = []

    async def probe(provider: str, scope: LLMScope) -> bool:
        assert provider == "anthropic"
        assert (scope.role, scope.goal_id, scope.team, scope.agent, scope.round) == (
            "manager",
            "g",
            "",
            "",
            2,
        )
        assert scope.tranche._holder == "g/:judges"
        order.append("probe")
        return bool(healthy)

    first = await _prepared_round(
        env.deps, goal_id="g", plan=PLAN, agents=[slot, peer], round_spec=RoundSpec(), rounds_left=2
    ).run(1)
    assert first.statuses["t/a"] == AgentStatus.STOPPED, first.reasons
    assert env.deps.provider_stops.stopped()[slot.address] == {"anthropic"}
    monkeypatch.setattr(stops_module, "_probe", probe if healthy is not None else None)
    frontier = env.deps.board.frontier

    def read_frontier() -> Frontier:
        order.append("frontier")
        return frontier()

    monkeypatch.setattr(env.deps.board, "frontier", read_frontier)
    second = await _prepared_round(
        env.deps, goal_id="g", plan=PLAN, agents=[slot, peer], round_spec=RoundSpec(), rounds_left=1
    ).run(2)
    assert order == (["frontier"] if healthy is None else ["probe", "frontier"])
    assert manager.calls == (1 if healthy is False else 2)
    if healthy is False:
        assert slot.status == AgentStatus.STOPPED
    else:
        assert second.statuses["t/a"] == AgentStatus.FINISHED
        assert any(c.agent == "a" and c.output.status == RunStatus.OK for c in second.candidates)
        rs = next(e for e in env.events("round_started") if e.round == 2)
        fr = next(e for e in env.events("frontier") if e.round == 2)
        assert fr.seq == rs.seq + 1
        assert rs.payload["shares"]["g/t/a"] == rs.payload["shares"]["g/t/b"]
        assert Amount.model_validate(rs.payload["shares"]["g/t/a"]).usd > 0
        assert any(
            e.payload["reservation"]["holder"] == "g/t/a" and e.seq > rs.seq
            for e in env.events("reserve")
        )


async def test_restarted_round_does_not_probe_or_clear_provider_causes(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine import stops as stops_module

    slot = _runner("a")
    stops = env.deps.provider_stops
    stops.register(slot.address, frozenset({"anthropic"}), lambda: None)
    await stops.fail("anthropic", source="manager")
    slot.status = AgentStatus.STOPPED

    async def forbidden(provider: str, scope: LLMScope) -> bool:
        pytest.fail("restarted round probed")

    monkeypatch.setattr(stops_module, "_probe", forbidden)
    await _prepared_round(
        env.deps,
        goal_id="g",
        plan=PLAN,
        agents=[slot],
        round_spec=RoundSpec(),
        rounds_left=1,
        frontier=env.deps.board.frontier(),
    ).run(1)
    assert slot.status == AgentStatus.STOPPED
    assert stops.stopped()[slot.address] == {"anthropic"}


@pytest.mark.parametrize("converge", [False, True])
@pytest.mark.parametrize("disposition", ["newer_failure", "pruned", "watchdog"])
async def test_provider_probe_preserves_newer_failure_and_permanent_stops(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, converge: bool, disposition: str
) -> None:
    from swarm_arena._engine import stops as stops_module

    slot = _runner("a")
    service = env.deps.provider_stops
    service.register(slot.address, frozenset({"anthropic"}), lambda: None)
    await service.fail("anthropic", source="manager")
    slot.status = AgentStatus.STOPPED
    order: list[str] = []

    async def probe(provider: str, scope: LLMScope) -> bool:
        order.append("probe")
        if disposition == "newer_failure":
            await service.fail(provider, source="judge")
        else:
            slot.status = (
                AgentStatus.PRUNED if disposition == "pruned" else AgentStatus.WATCHDOG_STOPPED
            )
            service.release(slot.address)
        return True

    frontier = env.deps.board.frontier

    def read_frontier() -> Frontier:
        order.append("frontier")
        return frontier()

    monkeypatch.setattr(stops_module, "_probe", probe)
    monkeypatch.setattr(env.deps.board, "frontier", read_frontier)
    outcome = await _prepared_round(
        env.deps,
        goal_id="g",
        plan=PLAN,
        agents=[slot],
        round_spec=RoundSpec(),
        rounds_left=1,
        converge=converge,
    ).run(1)
    assert order == ["probe", "frontier"]
    assert outcome.candidates == []
    assert (
        slot.status
        == {
            "newer_failure": AgentStatus.STOPPED,
            "pruned": AgentStatus.PRUNED,
            "watchdog": AgentStatus.WATCHDOG_STOPPED,
        }[disposition]
    )
    if disposition == "newer_failure":
        assert service.generations() == {"anthropic": 2}
        assert service.stopped()[slot.address] == {"anthropic"}
    else:
        assert slot.address not in service.stopped()


@pytest.mark.parametrize("control", ["fenced", "terminal", "caller"])
async def test_provider_probe_preserves_owner_controls(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    from swarm_arena._engine import stops as stops_module
    from swarm_arena._errors import _SignalCategory

    slot = _runner("a")
    stops = env.deps.provider_stops
    stops.register(slot.address, frozenset({"anthropic"}), lambda: None)
    await stops.fail("anthropic", source="manager")
    slot.status = AgentStatus.STOPPED
    entered = anyio.Event()

    async def probe(provider: str, scope: LLMScope) -> bool:
        entered.set()
        if control == "caller":
            await anyio.sleep_forever()
        if control == "fenced":
            raise _core_error(Fenced, "run ownership lost")
        raise _core_error(RuntimeError, "terminal probe", _category=_SignalCategory.TERMINAL)

    monkeypatch.setattr(stops_module, "_probe", probe)
    runner = _round(env, [slot])
    if control == "caller":
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tasks:

                async def cancel() -> None:
                    await entered.wait()
                    scope.cancel()

                tasks.start_soon(cancel)
                await runner._probe_stopped(2)
                pytest.fail("probe swallowed caller cancellation")
        assert scope.cancelled_caught
    else:
        with pytest.raises(Fenced if control == "fenced" else RuntimeError):
            await runner._probe_stopped(2)
    assert stops.stopped()[slot.address] == {"anthropic"}


@pytest.mark.parametrize("component", [Tool, Manager], ids=["tool", "manager"])
async def test_provider_resolution_is_concurrent_and_bounded_per_slot(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, component: type[Tool] | type[Manager]
) -> None:
    from collections.abc import Generator
    from contextlib import contextmanager
    from contextvars import ContextVar

    from swarm_arena._engine import stops as stops_module

    release = threading.Event()
    entered = {name: anyio.Event() for name in ("stuck-a", "stuck-b")}
    resolved = {name: anyio.Event() for name in entered}
    scopes: dict[str, anyio.CancelScope] = {}
    workers: list[threading.Thread] = []
    current: ContextVar[anyio.CancelScope] = ContextVar("resolution_scope")
    loop = asyncio.get_running_loop()
    original = Ref.build
    original_daemon = round_module.run_sync_daemon
    peer_calls: list[str] = []
    probes: list[str] = []

    class ResolutionClock:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        @contextmanager
        def move_on_at(self, deadline: float) -> Generator[anyio.CancelScope]:
            with anyio.move_on_at(deadline) as scope:
                token = current.set(scope)
                try:
                    yield scope
                finally:
                    current.reset(token)

    def started(name: str, scope: anyio.CancelScope, worker: threading.Thread) -> None:
        scopes[name] = scope
        workers.append(worker)
        entered[name].set()

    def blocked(self: Ref, base: type[Any]) -> Any:
        if self.name in entered and base is component:
            loop.call_soon_threadsafe(started, self.name, current.get(), threading.current_thread())
            assert release.wait(30), "constructor release deadline"
        return original(self, base)

    async def observe_resolution(fn: Any, *args: Any) -> Any:
        ref = getattr(fn, "__self__", None)
        watched = isinstance(ref, Ref) and ref.name in entered and args == (component,)
        try:
            return await original_daemon(fn, *args)
        finally:
            if watched:
                resolved[ref.name].set()

    class Peer(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            peer_calls.append("ran")
            return RoundReport(summary="peer ran")

    slots = [_runner("a"), _runner("b"), _with_manager("peer", Peer())]
    for slot in slots[:2]:
        if component is Tool:
            slot.tool_ref = slot.tool_ref.model_copy(update={"name": "stuck-" + slot.spec.name})
        else:
            slot.manager = None
            slot.spec = slot.spec.model_copy(
                update={
                    "manager": slot.spec.manager.model_copy(
                        update={"name": "stuck-" + slot.spec.name}
                    )
                }
            )
    e = env.with_limits(round_timeout_s=60.0, run_timeout_s=120.0)
    service = e.deps.provider_stops
    service.register(slots[2].address, frozenset({"fake"}), lambda: None)
    await service.fail("fake", source="manager")

    async def probe(provider: str, scope: LLMScope) -> bool:
        probes.append(provider)
        return True

    monkeypatch.setattr(Ref, "build", blocked)
    monkeypatch.setattr(stops_module, "_probe", probe)
    monkeypatch.setattr(round_module, "anyio", ResolutionClock())
    monkeypatch.setattr(round_module, "run_sync_daemon", observe_resolution)
    task = asyncio.create_task(_round(e, slots).run(1))
    try:
        with anyio.fail_after(15):
            # Neither constructor is released: serial resolution cannot reach both gates.
            await entered["stuck-a"].wait()
            await entered["stuck-b"].wait()
            assert not any(event.is_set() for event in resolved.values())
            assert scopes["stuck-a"] is not scopes["stuck-b"]
            scopes["stuck-a"].deadline = anyio.current_time()
            await resolved["stuck-a"].wait()
            assert not resolved["stuck-b"].is_set(), "slot expiry cancelled its peer"
            assert not task.done()
            scopes["stuck-b"].deadline = anyio.current_time()
            out = await task
        assert resolved["stuck-b"].is_set()
        assert not release.is_set()
        assert all(worker.is_alive() for worker in workers)
        assert probes == ["fake"]
        assert peer_calls == ["ran"]
        assert out.reasons["t/a"] == out.reasons["t/b"] == "runtime preparation cancelled"
        assert out.statuses["t/peer"] == AgentStatus.FINISHED
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()


@pytest.mark.parametrize("flag", ["run_budget", "goal_budget"])
@pytest.mark.parametrize("grace", [0.0, 5.0])
async def test_budget_close_during_reserve_durability(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, flag: str, grace: float
) -> None:
    closed = anyio.Event()
    real_close = Inflight.close
    observed_grace: list[float] = []

    def close(self: Inflight, *, budget: bool = False) -> None:
        if budget:
            observed_grace.append(self.max_grace_s())
        real_close(self, budget=budget)
        closed.set()

    monkeypatch.setattr(Inflight, "close", close)
    journal = env.deps.journal
    append = journal.append_durable

    async def gated(draft: EventDraft, **kwargs: Any) -> Any:
        result = await append(draft, **kwargs)
        if draft.type == "reserve" and draft.payload["reservation"]["kind"] == "tool":
            if flag == "goal_budget":
                env.deps.stops.goal_budget.add("g")
            else:
                env.deps.stops.run_budget = True
            await closed.wait()
        return result

    monkeypatch.setattr(journal, "append_durable", gated)
    slot = _runner("b")
    slot.tool_ref = Ref.of(FunctionTool("_fakes.tools:square", grace_s=grace))
    with anyio.fail_after(30):
        out = await _round(env, [slot]).run(1)
    assert len(env.events("tool_start")) == 1
    assert len(out.candidates) == 1
    assert observed_grace == [grace]
    reserves = [
        e.payload["reservation"]
        for e in env.events("reserve")
        if e.payload["reservation"]["kind"] == "tool"
    ]
    assert len(reserves) == 1
    settles = [e for e in env.events("settle") if e.payload["id"] == reserves[0]["id"]]
    assert len(settles) == 1
    assert Amount.model_validate(settles[0].payload["actual"]) == Amount()
