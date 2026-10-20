"""Late-stop publication and bounded planner/final-judge lifetimes."""

import json
import threading
from collections.abc import Sequence

import anyio
import pytest
from _engine.test_goal import _parts_plan, _plan, _raise_flag
from _fakes.engine import EngineEnv
from _fakes.planners import SequencePlanner
from _fakes.tools import ProviderJudge

from swarm_arena import Goal, Plan, ProgramJudge, Ref, RunConfig
from swarm_arena._agent.manager import LimitReached
from swarm_arena._budget.amount import USD_LIMIT, Amount
from swarm_arena._budget.ledger import Ledger, Reservation
from swarm_arena._budget.llm_scope import current_llm_scope
from swarm_arena._engine.goal import GoalOutcome, GoalRunner
from swarm_arena._engine.stops import RunClock
from swarm_arena._errors import Fenced
from swarm_arena._exec.executor import Job
from swarm_arena._journal.event import Event, EventDraft
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._model.planning_types import History
from swarm_arena._model.results import Candidate, Score
from swarm_arena._model.tooling import ToolOutput
from swarm_arena._scoring.judge import JudgeContext

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
    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 0.02})
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
            with anyio.move_on_after(3):
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
            if not already_stopped and flag == "deadline":
                env.deps.config = env.deps.config.model_copy(
                    update={
                        "limits": env.deps.config.limits.model_copy(update={"run_timeout_s": 0.03})
                    }
                )
                env.deps.clock = RunClock(0)
            entered.set()
            await anyio.sleep_forever()
        return await real(self, goal, candidates, ctx)

    monkeypatch.setattr(ProviderJudge, "judge", judge)
    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 0.1})
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
        if not already_stopped and flag == "stop_now":
            _signal(env, flag)
        with anyio.move_on_after(3):
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


@pytest.mark.parametrize("cancelled", [False, True])
async def test_replanner_fencing_and_cancellation_propagate(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    from swarm_arena._errors import Fenced

    error = anyio.get_cancelled_exc_class() if cancelled else Fenced
    plan = _plan(teams=1)
    planner = SequencePlanner([plan])

    async def fail(goal: Goal, config: RunConfig, history: History | None) -> Plan:
        raise error("hook failure")

    monkeypatch.setattr(planner, "plan", fail)
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=planner)
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
    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 0.03})
    runner = GoalRunner(
        env.deps, goal_id="g", plan=_plan(teams=1, rounds=1, converge_team=True), replanner=None
    )
    with anyio.fail_after(3):
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
    with anyio.fail_after(3):
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
                "settle_timeout_s": 0.02,
            }
        )

    @pytest.mark.parametrize("phase", ["acknowledgement", "dispatched"])
    async def test_stop_settles_final_reservation_before_completion(
        self, env: EngineEnv, monkeypatch: pytest.MonkeyPatch, phase: str
    ) -> None:
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
                    journal._group_window_s = 1.0
            return real_append(draft)

        async def durable(draft: EventDraft) -> Event:
            try:
                return await real_durable(draft)
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
        with anyio.fail_after(5):
            out = await runner.run()
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
        with anyio.fail_after(5), pytest.raises(Fenced):
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
