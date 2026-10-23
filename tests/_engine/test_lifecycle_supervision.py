"""Public-run regressions for user hooks suspended outside manager rounds."""

import threading
from collections.abc import Sequence
from functools import partial
from typing import Any

import anyio
import pytest
from _engine.test_extension_boundaries import _RoundClock
from _engine.test_goal import _plan
from _fakes.engine import EngineEnv
from _fakes.run import events

from swarm_arena import Goal, ProgramJudge, RunConfig, ScriptedManager, Swarm
from swarm_arena._engine import round as round_module
from swarm_arena._engine import run as run_module
from swarm_arena._engine.round import EngineDeps
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.planning_types import RoundState
from swarm_arena._model.results import Candidate, Score
from swarm_arena._planning.policies import SuccessiveHalving
from swarm_arena._scoring.judge import JudgeContext


@pytest.mark.parametrize(
    "phase",
    [
        "barrier",
        "converge_barrier",
        "policy",
        "policy_sync",
        "manager",
        "converge_manager",
        "judge_constructor",
    ],
)
@pytest.mark.parametrize("signal", ["stop_now", "deadline"])
async def test_public_run_supervises_lifecycle_hooks(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, phase: str, signal: str
) -> None:
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    done = anyio.Event()
    # Healthy journal settlement must not race CI scheduling. Hook entry, not
    # elapsed setup time, triggers the deadline below.
    cfg = run_config.model_copy(update={"settle_timeout_s": 2.0})
    cfg = cfg.model_copy(update={"limits": cfg.limits.model_copy(update={"run_timeout_s": 180.0})})
    swarm = Swarm(cfg)
    real_judge = ProgramJudge.judge
    round_clock = _RoundClock(cfg.limits.run_timeout_s)
    if signal == "deadline":
        monkeypatch.setattr(round_module, "anyio", round_clock)

    def block() -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(180), "hook safety release expired"

    real_watch_clock = run_module.RunRunner._watch_clock

    async def watch_clock(self: run_module.RunRunner, deps: EngineDeps) -> None:
        if signal == "deadline":
            assert await anyio.to_thread.run_sync(entered.wait, 60, abandon_on_cancel=True)
            deps.clock._rebase(deps.config.limits.run_timeout_s)
            # Converge rounds cache their monotonic deadline before hook entry.
            # Advance that runner-only clock with the active-time clock.
            round_clock.advance(deps.config.limits.run_timeout_s + 1)
        await real_watch_clock(self, deps)

    monkeypatch.setattr(run_module.RunRunner, "_watch_clock", watch_clock)

    async def judge(
        self: ProgramJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if not ctx.final and ctx.round == (2 if phase == "converge_barrier" else 1):
            entered.set()
            await anyio.sleep_forever()
        return await real_judge(self, goal, candidates, ctx)

    async def policy(self: SuccessiveHalving, state: RoundState) -> Any:
        entered.set()
        await anyio.sleep_forever()

    def policy_sync(self: SuccessiveHalving, state: RoundState) -> Any:
        block()
        return None

    real_init = ScriptedManager.__init__
    real_judge_init = ProgramJudge.__init__

    def construct_judge(self: ProgramJudge, *args: Any, **kwargs: Any) -> None:
        block()
        real_judge_init(self, *args, **kwargs)

    def construct(self: ScriptedManager, *args: Any, **kwargs: Any) -> None:
        if phase != "converge_manager" or args[0].requests[0].inputs["x"] >= 50:
            block()
        real_init(self, *args, **kwargs)

    real_publish = JournalWriter._publish

    async def publish(self: JournalWriter) -> None:
        start = any(e.type == "goal_started" for e in self._queue)
        await real_publish(self)
        if start:
            if phase in {"barrier", "converge_barrier"}:
                monkeypatch.setattr(ProgramJudge, "judge", judge)
            elif phase in {"policy", "policy_sync"}:
                monkeypatch.setattr(
                    SuccessiveHalving, "after_round", policy if phase == "policy" else policy_sync
                )
            elif phase == "judge_constructor":
                monkeypatch.setattr(ProgramJudge, "__init__", construct_judge)
            else:
                monkeypatch.setattr(ScriptedManager, "__init__", construct)

    monkeypatch.setattr(JournalWriter, "_publish", publish)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )

    async def drive() -> None:
        await swarm.run(_plan(teams=1, rounds=1, converge_team=phase.startswith("converge_")))
        done.set()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(drive)
            assert await anyio.to_thread.run_sync(entered.wait, 60, abandon_on_cancel=True)
            if signal == "stop_now":
                await swarm.stop(now=True)
            # Observe natural completion; cancellation only cleans up a failing regression.
            with anyio.move_on_after(30):
                await done.wait()
            returned = done.is_set()
            group.cancel_scope.cancel()
        assert returned, "the run must return under its own stop/deadline supervisor"
        assert not release.is_set(), "completion must not wait for the blocked hook"
        assert events(cfg, "goal_finished")
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()


@pytest.mark.parametrize("phase", ["acknowledgement", "dispatched"])
async def test_interrupted_barrier_settles_every_admitted_reservation(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from _fakes.tools import ProviderJudge

    from swarm_arena import Ref
    from swarm_arena._budget.amount import Amount
    from swarm_arena._engine.goal import GoalRunner
    from swarm_arena._journal.event import Event, EventDraft
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.replay import RunState

    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 2.0})
    journal = env.deps.journal
    durable = journal.append_durable
    real_judge = ProviderJudge.judge

    async def acknowledge(draft: EventDraft) -> Event:
        event = await durable(draft)
        if (
            phase == "acknowledgement"
            and draft.type == "reserve"
            and draft.payload["reservation"]["kind"] == "judge"
        ):
            env.deps.stops.stop_now = True
            await anyio.sleep_forever()
        return event

    async def judge(
        self: ProviderJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if phase == "dispatched" and not ctx.final:
            env.deps.stops.stop_now = True
            await anyio.sleep_forever()
        return await real_judge(self, goal, candidates, ctx)

    monkeypatch.setattr(journal, "append_durable", acknowledge)
    monkeypatch.setattr(ProviderJudge, "judge", judge)
    p = _plan(teams=1, rounds=1, judges=[Ref(path="_fakes.tools:ProviderJudge", name="y")])
    await GoalRunner(env.deps, goal_id="g", plan=p, replanner=None).run()
    committed = read_committed(env.store)
    assert committed is not None
    state = RunState.from_events(committed[1])
    reservations = [r for r in state.reservations.values() if r.reservation["kind"] == "judge"]
    assert len(reservations) == 1
    reservation = reservations[0]
    assert reservation.state != "open"
    assert reservation.actual == (
        reservation.reservation["amount"] if phase == "dispatched" else Amount().model_dump()
    )
    settlements = [
        e for e in env.events("settle") if e.payload["id"] == reservation.reservation["id"]
    ]
    assert len(settlements) == 1
    assert settlements[0].seq < env.events("goal_finished")[-1].seq
