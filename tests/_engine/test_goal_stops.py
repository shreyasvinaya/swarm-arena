"""Late-stop publication and bounded planner/final-judge lifetimes."""

import json
import threading
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import anyio
import pytest
from _engine.test_goal import _parts_plan, _plan, _raise_flag
from _fakes.engine import EngineEnv
from _fakes.planners import SequencePlanner
from _fakes.tools import ProviderJudge

from swarm_arena import Goal, Limits, Plan, ProgramJudge, Ref, RunConfig
from swarm_arena._agent.manager import LimitReached
from swarm_arena._budget.amount import USD_LIMIT, Amount
from swarm_arena._budget.ledger import Ledger, Reservation
from swarm_arena._budget.llm_scope import LLMScope, current_llm_scope
from swarm_arena._engine import goal as goal_module
from swarm_arena._engine.goal import GoalOutcome, GoalRunner
from swarm_arena._engine.stops import RunClock
from swarm_arena._errors import (
    Fenced,
    _core_error,
    _is_core_signal,
    _SignalCategory,
    _stop_cleanup_error,
)
from swarm_arena._exec.executor import Job
from swarm_arena._journal.event import Event, EventDraft
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._model.planning_types import History
from swarm_arena._model.results import Candidate, Score
from swarm_arena._model.tooling import ToolOutput
from swarm_arena._scoring.judge import JudgeContext
from swarm_arena._scoring.runner import JudgeError, JudgeOutcome

_REASONS = {
    "stop": "stop",
    "stop_now": "stop --now",
    "deadline": "run_timeout_s",
    "run_budget": "run budget",
    "goal_budget": "goal budget",
}


@pytest.fixture(autouse=True)
def fast_commits(env: EngineEnv) -> None:
    env.deps.journal._group_window_s = 0.001


class _GoalTime:
    """Control grace expiry without changing journal or AnyIO timeout clocks."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 0.0
        self.sampled = anyio.Event()
        monkeypatch.setattr(goal_module, "anyio", self)

    def __getattr__(self, name: str) -> Any:
        return getattr(anyio, name)

    def current_time(self) -> float:
        self.sampled.set()
        return self.now


def _signal(env: EngineEnv, flag: str) -> None:
    if flag == "deadline":
        env.deps.clock = RunClock(env.deps.config.limits.run_timeout_s + 1)
    else:
        _raise_flag(env.deps.stops, flag, goal_id="g")


def _assert_terminal(env: EngineEnv, out: GoalOutcome, reason: str) -> None:
    assert (out.status, out.reason) == ("stopped", reason)
    assert env.events("goal_finished")[-1].payload["reason"] == reason
    view = json.loads(env.store.read("goals/g/solution.json").data)
    assert view["reason"] == reason
    assert view["source_head"]["seq"] >= env.events("goal_finished")[-1].seq


@pytest.mark.parametrize("phase", ["converge", "final"])
@pytest.mark.parametrize("flag", list(_REASONS))
async def test_late_signal_is_in_terminal_journal_and_view(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, phase: str, flag: str
) -> None:
    reached: list[str] = []

    def signal() -> None:
        reached.append(phase)
        _signal(env, flag)

    if phase == "converge":
        env.on_commit("tool_end", lambda ev: signal() if ev.team == "conv" else None)
    else:
        real = ProgramJudge.judge

        async def judge(
            self: ProgramJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
        ) -> list[Score]:
            if ctx.final:
                signal()
            return await real(self, goal, candidates, ctx)

        monkeypatch.setattr(ProgramJudge, "judge", judge)
    runner = GoalRunner(
        env.deps, goal_id="g", plan=_plan(teams=1, rounds=1, converge_team=True), replanner=None
    )
    out = await runner.run()
    assert reached
    _assert_terminal(env, out, _REASONS[flag])
    assert out.finalists


@pytest.mark.parametrize("flag", ["stop_now", "deadline"])
async def test_interrupted_part_join_keeps_unjoined_best_parts(env: EngineEnv, flag: str) -> None:
    env.on_commit("frontier", lambda ev: _signal(env, flag) if ev.round == 2 else None)
    runner = GoalRunner(env.deps, goal_id="g", plan=_parts_plan(), replanner=None)
    out = await runner.run()
    _assert_terminal(env, out, _REASONS[flag])
    assert not env.events("join_failed")
    assert not env.events("no_result")
    expected = {c.part: c.id for c in env.candidates() if c.part is not None}
    assert env.events("final_order")[-1].payload["parts"] == expected
    view = json.loads(env.store.read("goals/g/solution.json").data)
    assert view["status"] == "unjoined"
    assert {item["candidate"]["id"] for item in view["parts"]} == set(expected.values())


async def test_converge_usage_overflow_with_free_final_is_stopped(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = env.deps.executor.submit

    async def submit(job: Job) -> ToolOutput:
        output = await real(job)
        if job.kind == "tool" and job.env.get("SWARM_ARENA_TEAM") == "conv":
            return output.model_copy(
                update={"usage": output.usage.model_copy(update={"usd": USD_LIMIT + 1})}
            )
        return output

    monkeypatch.setattr(env.deps.executor, "submit", submit)
    runner = GoalRunner(
        env.deps,
        goal_id="g",
        plan=_plan(teams=1, rounds=1, converge_team=True, converge_outputs=2),
        replanner=None,
    )
    out = await runner.run()
    assert len([e for e in env.events("settle") if e.payload.get("usage_overflow")]) == 1
    with pytest.raises(LimitReached, match="usage"):
        await env.deps.ledger.reserve("g/:final", Amount(), kind="final", goal_id="g")
    _assert_terminal(env, out, "run budget")
    assert out.finalists


@pytest.mark.parametrize("flag", ["stop", "stop_now", "deadline"])
@pytest.mark.parametrize("synchronous", [False, True])
async def test_pending_replan_is_abandoned(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, flag: str, synchronous: bool
) -> None:
    entered = threading.Event()
    release = threading.Event()
    completed = anyio.Event()
    outcomes: list[GoalOutcome] = []
    threads: list[threading.Thread] = []
    plan = _plan(teams=1, rounds=2)
    planner = SequencePlanner([plan])

    async def hanging_async(goal: Goal, config: RunConfig, history: History | None) -> Plan:
        entered.set()
        scope = current_llm_scope()
        assert scope is not None
        await scope.tranche.reserve_call(Amount(usd=0.01))
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    def hanging_sync(goal: Goal, config: RunConfig, history: History | None) -> Plan:
        threads.append(threading.current_thread())
        entered.set()
        # Keep the pre-fix repro bounded even if the hook runs on the event loop.
        if threading.current_thread() is not threading.main_thread():
            release.wait()
        return plan

    monkeypatch.setattr(planner, "plan", hanging_sync if synchronous else hanging_async)
    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 2.0})
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=planner)

    async def drive() -> None:
        outcomes.append(await runner.run())
        completed.set()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(drive)
            with anyio.fail_after(10):
                while not entered.is_set():  # noqa: ASYNC110 - daemon thread signal
                    await anyio.sleep(0.01)
            _signal(env, flag)
            with anyio.move_on_after(7):
                await completed.wait()
            group.cancel_scope.cancel()
        assert completed.is_set(), "stop must abandon a suspended replanner"
        if synchronous:
            assert threads[0].daemon
            assert threads[0] is not threading.main_thread()
        _assert_terminal(env, outcomes[0], _REASONS[flag])
        assert len(env.events("plan_revision")) == 1
        assert env.events("plan_rejected")[-1].payload == {
            "origin": "autonomous",
            "errors": ["planner stopped"],
            "first_round": 2,
        }
        assert [e.payload["round"] for e in env.events("round_closed")] == [1]
        assert not runner.admission._peaks
        assert all(r.state != "open" for r in runner.recorded.reservations.values())
    finally:
        release.set()


@pytest.mark.parametrize("flag", ["stop_now", "deadline"])
@pytest.mark.parametrize("already_stopped", [False, True])
async def test_hanging_final_judge_is_bounded_and_keeps_journaled_scores(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, flag: str, already_stopped: bool
) -> None:
    clock = _GoalTime(monkeypatch)
    entered = anyio.Event()
    completed = anyio.Event()
    outcomes: list[GoalOutcome] = []
    scopes = []
    real = ProviderJudge.judge

    async def judge(
        self: ProviderJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if ctx.final:
            scope = current_llm_scope()
            assert scope is not None
            scopes.append(scope)
            await scope.tranche.reserve_call(Amount(usd=0.01))
            entered.set()
            await anyio.sleep_forever()
        return await real(self, goal, candidates, ctx)

    monkeypatch.setattr(ProviderJudge, "judge", judge)
    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 2.0})
    if already_stopped:
        env.on_commit("round_closed", lambda ev: _signal(env, flag))
    runner = GoalRunner(
        env.deps,
        goal_id="g",
        plan=_plan(teams=1, rounds=1, judges=[Ref(path="_fakes.tools:ProviderJudge", name="y")]),
        replanner=None,
    )

    async def drive() -> None:
        outcomes.append(await runner.run())
        completed.set()

    async with anyio.create_task_group() as group:
        group.start_soon(drive)
        with anyio.fail_after(10):
            await entered.wait()
        if not already_stopped:
            _signal(env, flag)
        # Grace includes construction and durable reservation. Expire it only
        # after the judge is suspended and the supervisor has latched the stop.
        with anyio.fail_after(10):
            await clock.sampled.wait()
        clock.now += env.deps.config.settle_timeout_s + 1
        with anyio.move_on_after(7):
            await completed.wait()
        group.cancel_scope.cancel()
    assert completed.is_set(), "final judging must finish within the post-stop grace"
    _assert_terminal(env, outcomes[0], _REASONS[flag])
    assert [c.id for c in outcomes[0].finalists] == [env.candidates()[0].id]
    assert not [e for e in env.events("score") if e.round == 0]
    assert {e.payload["reason"] for e in env.events("judge_error") if e.round == 0} == {
        "judge error: TimeoutError"
    }
    final_reserves = [
        e.payload["reservation"]
        for e in env.events("reserve")
        if e.payload["reservation"]["kind"] == "final"
    ]
    assert len(final_reserves) == 1
    settlements = [e for e in env.events("settle") if e.payload["id"] == final_reserves[0]["id"]]
    assert len(settlements) == 1
    assert settlements[0].payload["estimated"] is True
    assert settlements[0].payload["actual"] == final_reserves[0]["amount"]
    assert scopes[0].tranche.pending() == 0
    assert all(r.state != "open" for r in runner.recorded.reservations.values())
    assert not runner.admission._peaks


@pytest.mark.parametrize("flag", ["stop_now", "deadline"])
async def test_final_grace_can_expire_before_judge_dispatch(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    clock = _GoalTime(monkeypatch)
    reserving = anyio.Event()
    dispatched = anyio.Event()
    completed = anyio.Event()
    outcomes: list[GoalOutcome] = []
    append = env.deps.journal.append_durable
    judge = ProviderJudge.judge

    async def blocked_reserve(draft: EventDraft) -> Event:
        event = await append(draft)
        if draft.type == "reserve" and draft.payload["reservation"]["kind"] == "final":
            reserving.set()
            await anyio.sleep_forever()
        return event

    async def observe_judge(
        self: ProviderJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if ctx.final:
            dispatched.set()
        return await judge(self, goal, candidates, ctx)

    monkeypatch.setattr(env.deps.journal, "append_durable", blocked_reserve)
    monkeypatch.setattr(ProviderJudge, "judge", observe_judge)
    env.on_commit("round_closed", lambda ev: _signal(env, flag))
    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 2.0})
    runner = GoalRunner(
        env.deps,
        goal_id="g",
        plan=_plan(teams=1, rounds=1, judges=[Ref(path="_fakes.tools:ProviderJudge", name="y")]),
        replanner=None,
    )

    async def drive() -> None:
        outcomes.append(await runner.run())
        completed.set()

    async with anyio.create_task_group() as group:
        group.start_soon(drive)
        with anyio.fail_after(10):
            await reserving.wait()
            await clock.sampled.wait()
        assert not completed.is_set()
        clock.now += env.deps.config.settle_timeout_s + 1
        with anyio.fail_after(7):
            await completed.wait()
    _assert_terminal(env, outcomes[0], _REASONS[flag])
    assert not dispatched.is_set()
    assert [c.id for c in outcomes[0].finalists] == [env.candidates()[0].id]
    assert not [e for e in env.events("score") if e.round == 0]
    assert {e.payload["reason"] for e in env.events("judge_error") if e.round == 0} == {
        "judge error: TimeoutError"
    }
    assert all(r.state != "open" for r in runner.recorded.reservations.values())
    assert not runner.admission._peaks


@pytest.mark.parametrize("cancelled", [False, True])
async def test_replanner_fencing_propagates_and_user_cancellation_is_rejected(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    from swarm_arena._errors import Fenced

    error = anyio.get_cancelled_exc_class() if cancelled else Fenced
    plan = _plan(teams=1)
    planner = SequencePlanner([plan])

    async def fail(goal: Goal, config: RunConfig, history: History | None) -> Plan:
        raise error("hook failure") if cancelled else _core_error(Fenced, "run ownership lost")

    monkeypatch.setattr(planner, "plan", fail)
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=planner)
    if cancelled:
        assert not await runner._replan()
        assert env.events("plan_rejected")[-1].payload["errors"] == [
            "planner error: CancelledError"
        ]
    else:
        with pytest.raises(error):
            await runner._replan()
        assert not env.events("plan_rejected")
    assert not env.events("plan_revision")


async def test_final_timeout_without_journaled_scores_still_has_no_result(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def judge(
        self: ProgramJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if ctx.final:
            _signal(env, "stop_now")
            await anyio.sleep_forever()
        return []

    monkeypatch.setattr(ProgramJudge, "judge", judge)
    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 2.0})
    runner = GoalRunner(
        env.deps, goal_id="g", plan=_plan(teams=1, rounds=1, converge_team=True), replanner=None
    )
    with anyio.fail_after(20):
        out = await runner.run()
    assert (out.status, out.reason, out.finalists) == ("no_result", "no passing candidate", [])
    assert not env.events("score")
    assert env.events("goal_finished")[-1].payload["reason"] == "no passing candidate"
    view = json.loads(env.store.read("goals/g/solution.json").data)
    assert (view["status"], view["reason"]) == ("no_result", "no passing candidate")


@pytest.mark.parametrize("flag", ["stop_now", "deadline"])
async def test_interrupted_join_replan_keeps_unjoined_parts(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    plan = _parts_plan(fail_join=True)
    planner = SequencePlanner([plan])

    async def hang(goal: Goal, config: RunConfig, history: History | None) -> Plan:
        assert history is not None
        assert history.join_failures
        _signal(env, flag)
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    monkeypatch.setattr(planner, "plan", hang)
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=planner)
    with anyio.fail_after(20):
        out = await runner.run()
    _assert_terminal(env, out, _REASONS[flag])
    assert len(env.events("join_failed")) == 1
    assert not env.events("no_result")
    view = json.loads(env.store.read("goals/g/solution.json").data)
    assert view["status"] == "unjoined"
    assert len(view["parts"]) == 2


class TestFinalReservationOwnership:
    @pytest.fixture
    def run_config(self, run_config: RunConfig) -> RunConfig:
        return run_config.model_copy(
            update={
                "limits": run_config.limits.model_copy(
                    update={"max_tokens": 1_000_000, "max_usd": 100.0}
                ),
                "settle_timeout_s": 2.0,
            }
        )

    @pytest.mark.parametrize("phase", ["acknowledgement", "dispatched"])
    async def test_stop_settles_final_reservation_before_completion(
        self, env: EngineEnv, monkeypatch: pytest.MonkeyPatch, phase: str
    ) -> None:
        clock = _GoalTime(monkeypatch)
        suspended = anyio.Event()
        admitted: list[Reservation] = []
        dispatched: list[str] = []
        cancelled_ack: list[str] = []
        balances: list[tuple[int, int]] = []
        journal = env.deps.journal
        ledger = env.deps.ledger
        real_append = journal.append
        real_durable = journal.append_durable
        real_judge = ProviderJudge.judge

        def append(draft: EventDraft) -> Event:
            if draft.type == "reserve" and draft.payload["reservation"]["kind"] == "final":
                admitted.append(Reservation.model_validate(draft.payload["reservation"]))
                balances.append((ledger._total.tokens, ledger._total._micro_usd))
                if phase == "acknowledgement":
                    _signal(env, "stop_now")
            return real_append(draft)

        async def durable(draft: EventDraft) -> Event:
            try:
                event = await real_durable(draft)
                if (
                    phase == "acknowledgement"
                    and draft.type == "reserve"
                    and draft.payload["reservation"]["kind"] == "final"
                ):
                    suspended.set()
                    await anyio.sleep_forever()
                return event
            except anyio.get_cancelled_exc_class():
                if draft.type == "reserve" and draft.payload["reservation"]["kind"] == "final":
                    cancelled_ack.append(draft.payload["reservation"]["id"])
                raise

        async def judge(
            self: ProviderJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
        ) -> list[Score]:
            if ctx.final:
                dispatched.append(ctx.batch_id)
                _signal(env, "stop_now")
                suspended.set()
                await anyio.sleep_forever()
            return await real_judge(self, goal, candidates, ctx)

        monkeypatch.setattr(journal, "append", append)
        monkeypatch.setattr(journal, "append_durable", durable)
        monkeypatch.setattr(ProviderJudge, "judge", judge)
        runner = GoalRunner(
            env.deps,
            goal_id="g",
            plan=_plan(
                teams=1, rounds=1, judges=[Ref(path="_fakes.tools:ProviderJudge", name="y")]
            ),
            replanner=None,
        )
        outcomes: list[GoalOutcome] = []

        async def drive() -> None:
            outcomes.append(await runner.run())

        with anyio.fail_after(20):
            async with anyio.create_task_group() as group:
                group.start_soon(drive)
                await suspended.wait()
                await clock.sampled.wait()
                clock.now += env.deps.config.settle_timeout_s + 1
        out = outcomes[0]
        _assert_terminal(env, out, "stop --now")
        assert out.finalists
        assert len(admitted) == 1
        reservation = admitted[0]
        was_dispatched = phase == "dispatched"
        assert bool(dispatched) is was_dispatched
        assert cancelled_ack == ([] if was_dispatched else [reservation.id])
        committed = read_committed(env.store)
        assert committed is not None
        state = RunState.from_events(committed[1])
        assert all(r.state != "open" for r in state.reservations.values()), (
            "bounded final judging stranded its direct final reservation"
        )
        settlements = [
            e for e in committed[1] if e.type == "settle" and e.payload["id"] == reservation.id
        ]
        assert len(settlements) == 1
        settlement = settlements[0]
        charge = reservation.amount if was_dispatched else Amount()
        assert settlement.payload["actual"] == charge.model_dump()
        assert settlement.payload["estimated"] is was_dispatched
        assert settlement.seq < next(e.seq for e in committed[1] if e.type == "goal_finished")
        assert ledger._charges[reservation.id] == charge
        refund = reservation.amount - charge
        assert (ledger._total.tokens, ledger._total._micro_usd) == (
            balances[0][0] + refund.tokens,
            balances[0][1] + refund._micro_usd,
        )
        replay = Ledger.replay(state, journal=journal, ids=env.deps.ids)
        assert ledger._charges == replay._charges
        assert ledger._total == replay._total
        assert ledger._work == replay._work
        assert ledger._reserve_pool.left == replay._reserve_pool.left
        assert {g: (p.left, p.ended) for g, p in ledger._goals.items()} == {
            g: (p.left, p.ended) for g, p in replay._goals.items()
        }
        assert {g: p.left for g, p in ledger._slices.items()} == {
            g: p.left for g, p in replay._slices.items()
        }
        assert ledger.free() == replay.free()
        assert ledger.remaining("g/:final") == replay.remaining("g/:final")

    async def test_failed_final_reserve_ack_retains_charge_without_completion(
        self, env: EngineEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        journal = env.deps.journal
        original = journal._publish
        admitted: list[Reservation] = []
        dispatched: list[str] = []
        real_judge = ProviderJudge.judge

        async def publish() -> None:
            finals = [
                Reservation.model_validate(e.payload["reservation"])
                for e in journal._queue
                if e.type == "reserve" and e.payload["reservation"]["kind"] == "final"
            ]
            await original()
            if finals:
                admitted.extend(finals)
                raise RuntimeError("acknowledgement unavailable")

        async def judge(
            self: ProviderJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
        ) -> list[Score]:
            if ctx.final:
                dispatched.append(ctx.batch_id)
            return await real_judge(self, goal, candidates, ctx)

        monkeypatch.setattr(journal, "_publish", publish)
        monkeypatch.setattr(ProviderJudge, "judge", judge)
        runner = GoalRunner(
            env.deps,
            goal_id="g",
            plan=_plan(
                teams=1, rounds=1, judges=[Ref(path="_fakes.tools:ProviderJudge", name="y")]
            ),
            replanner=None,
        )
        with anyio.fail_after(20), pytest.raises(Fenced):
            await runner.run()
        assert len(admitted) == 1
        reservation = admitted[0]
        assert not dispatched
        assert not env.events("goal_finished")
        assert not env.store.exists("goals/g/solution.json")
        assert not [e for e in env.events("settle") if e.payload["id"] == reservation.id]
        committed = read_committed(env.store)
        assert committed is not None
        state = RunState.from_events(committed[1])
        assert state.reservations[reservation.id].state == "open"
        ledger = env.deps.ledger
        assert ledger._charges[reservation.id] == reservation.amount
        replay = Ledger.replay(state, journal=journal, ids=env.deps.ids)
        assert ledger._charges == replay._charges
        assert ledger._total == replay._total
        assert ledger.free() == replay.free()
        assert ledger.remaining("g/:final") == replay.remaining("g/:final")


@pytest.mark.parametrize("flag", ["stop_now", "deadline"])
async def test_final_refresh_timeout_is_core_and_recovery_can_finish(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    clock = _GoalTime(monkeypatch)
    judging, reading = anyio.Event(), anyio.Event()
    errors: list[Exception] = []
    completed = anyio.Event()
    real_judge = ProviderJudge.judge
    real_read = goal_module._read_chain
    config = env.deps.config.model_copy(update={"settle_timeout_s": 2.0})
    env.deps.config = config
    plan = _plan(teams=1, rounds=1, judges=[Ref(path="_fakes.tools:ProviderJudge", name="y")])

    async def judge(
        self: ProviderJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if ctx.final:
            judging.set()
            await anyio.sleep_forever()
        return await real_judge(self, goal, candidates, ctx)

    async def read(*args: Any, **kwargs: Any) -> Any:
        if judging.is_set() and not reading.is_set():
            reading.set()
            await anyio.sleep_forever()
        return await real_read(*args, **kwargs)

    monkeypatch.setattr(ProviderJudge, "judge", judge)
    monkeypatch.setattr(goal_module, "_read_chain", read)

    async def drive() -> None:
        try:
            await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
        except Exception as error:
            errors.append(error)
        finally:
            completed.set()

    with anyio.fail_after(20):
        async with anyio.create_task_group() as group:
            group.start_soon(drive)
            await judging.wait()
            _signal(env, flag)
            await clock.sampled.wait()
            clock.now += config.settle_timeout_s + 1
            await reading.wait()
            clock.now += config.settle_timeout_s + 1
            await completed.wait()
    await env.deps.journal.commit()
    assert len(errors) == 1, "a storage timeout must escape final-judge classification"
    assert isinstance(errors[0], RuntimeError)
    assert _is_core_signal(errors[0])
    assert str(errors[0]) == "goal storage settlement timed out"
    committed = read_committed(env.store)
    assert committed is not None
    events = committed[1]
    assert not [event for event in events if "final_batch" in event.payload]
    assert not env.events("no_result")
    assert not env.events("goal_finished")
    assert not [event for event in env.events("judge_error") if event.round == 0]
    assert not env.store.exists("goals/g/solution.json")
    state = RunState.from_events(events)
    assert not state.goals["g"].finished
    monkeypatch.setattr(ProviderJudge, "judge", real_judge)
    monkeypatch.setattr(goal_module, "_read_chain", real_read)
    with anyio.fail_after(20):
        recovered = await GoalRunner(
            env.deps,
            goal_id="g",
            plan=plan,
            replanner=None,
            state=state.goals["g"],
            journal_r0=False,
        ).run()
    _assert_terminal(env, recovered, _REASONS[flag])
    assert recovered.finalists
    assert len(env.events("goal_finished")) == 1
    assert not env.events("no_result")


async def test_final_writer_timeout_propagates_without_judge_errors(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = env.deps.journal
    journal._group_window_s = 0
    journal._settle_timeout_s = 0.2
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    errors: list[Exception] = []
    judge_errors: list[JudgeError] = []
    completed = anyio.Event()
    create = env.store.create
    runner = GoalRunner(
        env.deps,
        goal_id="g",
        plan=_plan(teams=1, rounds=1, judges=[Ref(path="_fakes.tools:ProviderJudge", name="y")]),
        replanner=None,
    )
    publish_scores = runner._publish_scores

    def blocked_create(key: str, data: bytes) -> str:
        if key.startswith("journal/") and b'"kind":"final"' in data:
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(30)
        return create(key, data)

    def observe_scores(outcome: JudgeOutcome, scope: LLMScope, round: int, **kwargs: Any) -> None:
        if round == 0:
            judge_errors.extend(outcome.errors)
        publish_scores(outcome, scope, round, **kwargs)

    async def drive() -> None:
        try:
            await runner.run()
        except Exception as error:
            errors.append(error)
        finally:
            completed.set()

    monkeypatch.setattr(env.store, "create", blocked_create)
    monkeypatch.setattr(runner, "_publish_scores", observe_scores)
    try:
        with anyio.fail_after(20):
            async with anyio.create_task_group() as group:
                group.start_soon(drive)
                assert await anyio.to_thread.run_sync(entered.wait, 10)
                journal._stop_storage()
                await completed.wait()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert _is_core_signal(errors[0])
        assert str(errors[0]) == "stop cleanup timed out"
        assert not judge_errors
        assert (runner.status, runner.reason) != ("no_result", "provider_failed")
        assert not [event for event in env.events("judge_error") if event.round == 0]
        assert not env.events("no_result")
        assert not env.events("goal_finished")
        assert not env.store.exists("goals/g/solution.json")
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()


@pytest.mark.parametrize("site", ["replan", "policy"])
async def test_writer_timeout_bypasses_component_error_handlers(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    plan = _plan(teams=1, rounds=1)
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([plan]))
    error = _stop_cleanup_error()

    async def fail(*args: Any, **kwargs: Any) -> Any:
        env.deps.journal._settlement_expired = True
        env.deps.journal._fail(error)
        raise error

    monkeypatch.setattr(runner, "_supervised_step", fail)
    operation = runner._replan() if site == "replan" else runner._round_decision(1)
    with pytest.raises(RuntimeError, match=r"^stop cleanup timed out$") as caught:
        await operation
    assert caught.value is not error
    assert _is_core_signal(caught.value)
    assert not env.events("plan_rejected")
    assert not env.events("decision")


@pytest.mark.parametrize("site", ["final", "replan", "policy"])
@pytest.mark.parametrize(
    "kind",
    ["cleanup", "overflow", "fence", "limit", "misleading_limit", "validation", "cancel", "group"],
)
@pytest.mark.parametrize("nested", [False, True])
async def test_goal_handlers_use_registered_categories(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    kind: str,
    nested: bool,
) -> None:
    import asyncio

    from _fakes.board import candidate
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._outcomes import _classify_outcome

    secret = "private-goal-outcome"
    errors: dict[str, BaseException] = {
        "cleanup": _stop_cleanup_error(),
        "overflow": _core_error(
            LimitReached,
            "usage_overflow",
            "budget exhausted by usage overflow",
            _category=_SignalCategory.TERMINAL,
        ),
        "fence": _core_error(Fenced, "run ownership lost"),
        "limit": _core_error(LimitReached, "budget", "budget used up"),
        "misleading_limit": _core_error(LimitReached, "usage_overflow", "local refusal"),
        "validation": _core_error(ValueError, "invalid field"),
        "cancel": _classify_outcome(asyncio.CancelledError(secret)),
        "group": _classify_outcome(BaseExceptionGroup(secret, [asyncio.CancelledError(secret)])),
    }
    error = errors[kind]
    error.args = (secret,)
    error.add_note(secret)
    error.__dict__.update(category="control", limit=secret)
    if nested:
        error = BaseExceptionGroup(secret, [asyncio.CancelledError(secret), error])
    plan = _plan(teams=1, rounds=1)
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([plan]))
    invocations: list[str] = []

    async def fail(*args: Any, **kwargs: Any) -> Any:
        invocations.append(site)
        raise error

    monkeypatch.setattr(runner, "_supervised_step", fail)
    operation = (
        runner._final([candidate("c", goal_id="g")])
        if site == "final"
        else runner._replan()
        if site == "replan"
        else runner._round_decision(1)
    )
    if kind in {"cleanup", "overflow", "fence"}:
        with pytest.raises(Exception, match=r"timed out|overflow|ownership lost") as caught:
            await operation
        assert_secret_absent(caught.value, secret)
        if kind == "overflow":
            assert isinstance(caught.value, LimitReached)
            assert caught.value.limit == "usage_overflow"
        elif kind == "fence":
            assert isinstance(caught.value, Fenced)
        else:
            assert str(caught.value) == "stop cleanup timed out"
        assert not env.events("plan_rejected")
        assert not env.events("decision")
        assert not env.events("judge_error")
    elif kind in {"limit", "misleading_limit"} and site == "final":
        assert await operation == []
        assert (runner.status, runner.reason) == ("no_result", "final slice used up")
        assert not env.events("judge_error")
    else:
        await operation
        await env.deps.journal.commit()
        event_kind = {"final": "judge_error", "replan": "plan_rejected", "policy": "decision"}[site]
        assert env.events(event_kind)
        if kind in {"limit", "misleading_limit"}:
            payload = env.events(event_kind)[-1].payload
            if site == "replan":
                assert payload["errors"] == ["planner error: LimitReached"]
            else:
                assert payload["decision"]["policy"] == "policy error: LimitReached"
        assert secret not in str([event.payload for event in env.events(event_kind)])
    assert invocations == [site]


@pytest.mark.parametrize("site", ["final", "replan", "policy"])
async def test_goal_handler_caller_cancellation_wins_terminal_outcome(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
) -> None:
    import asyncio

    from _fakes.board import candidate

    plan = _plan(teams=1, rounds=1)
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([plan]))
    task = asyncio.current_task()
    assert task is not None

    async def fail(*args: Any, **kwargs: Any) -> Any:
        task.cancel()
        raise ExceptionGroup(
            "private", [_stop_cleanup_error(), _core_error(Fenced, "run ownership lost")]
        )

    monkeypatch.setattr(runner, "_supervised_step", fail)
    operation = (
        runner._final([candidate("c", goal_id="g")])
        if site == "final"
        else runner._replan()
        if site == "replan"
        else runner._round_decision(1)
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await operation
    finally:
        task.uncancel()
        # Cleanup may or may not have consumed the native cancellation delivery.
        with suppress(asyncio.CancelledError):
            await anyio.lowlevel.checkpoint()
    assert not env.events("plan_rejected")
    assert not env.events("decision")
    assert not env.events("judge_error")


class TestPlannerBudgetRefusal:
    @pytest.fixture
    def run_config(self, tmp_path: Path, artifact_root: str) -> RunConfig:
        return RunConfig(
            run_root=str(tmp_path / "runs"),
            arena_root=str(tmp_path / "arena"),
            artifact_root=artifact_root,
            lease_ttl_s=60,
            limits=Limits(max_usd=1, max_tokens=1_000_000),
        )

    async def test_refused_planner_still_finishes_goal(self, env: EngineEnv) -> None:
        class BudgetPlanner(SequencePlanner):
            async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
                scope = current_llm_scope()
                assert scope is not None
                await scope.tranche.reserve_call(Amount(usd=50, tokens=10))
                raise AssertionError("reservation unexpectedly succeeded")

        plan = _plan(teams=1, rounds=2)
        runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=BudgetPlanner([plan]))
        with anyio.fail_after(30):
            outcome = await runner.run()
        assert outcome.status == "no_result"
        assert len(env.events("goal_finished")) == 1
        assert env.events("plan_rejected")[-1].payload["errors"] == ["planner error: LimitReached"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    ("site", "origin"),
    [
        ("final", "close"),
        ("replan", "close"),
        ("final", "supervisor"),
        ("replan", "supervisor"),
        ("policy", "supervisor"),
    ],
)
async def test_core_interrupt_bypasses_goal_handler_classification(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: type[BaseException],
    site: str,
    origin: str,
) -> None:
    from _fakes.board import candidate

    error = interrupt()
    plan = _plan(teams=1, rounds=1)
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([plan]))

    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(runner, "_close" if origin == "close" else "_supervised_step", fail)
    operation = (
        runner._final([candidate("c", goal_id="g")])
        if site == "final"
        else runner._replan()
        if site == "replan"
        else runner._round_decision(1)
    )
    with pytest.raises(interrupt) as caught:
        await operation
    assert caught.value is error
    assert not env.events("plan_rejected")
    assert not env.events("decision")


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("site", ["final", "replan", "policy"])
async def test_user_hook_interrupt_stays_contained_at_goal_handlers(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: type[BaseException],
    site: str,
) -> None:
    from _fakes.board import candidate

    secret = "private-hook-interrupt"
    plan = _plan(teams=1, rounds=1)
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([plan]))
    supervised = runner._supervised_step

    async def hook() -> Any:
        raise interrupt(secret)

    async def invoke(*args: Any, **kwargs: Any) -> Any:
        return await supervised(hook, **kwargs)

    monkeypatch.setattr(runner, "_supervised_step", invoke)
    operation = (
        runner._final([candidate("c", goal_id="g")])
        if site == "final"
        else runner._replan()
        if site == "replan"
        else runner._round_decision(1)
    )
    await operation
    await env.deps.journal.commit()
    event_kind = {"final": "judge_error", "replan": "plan_rejected", "policy": "decision"}[site]
    assert env.events(event_kind)
    assert secret not in str([event.payload for event in env.events(event_kind)])
