"""Concurrent goal completion must apply returns in committed journal order."""

from pathlib import Path

import anyio
import pytest

from swarm_arena import Goal, Limits, Plan, RoundSpec, RunConfig
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger
from swarm_arena._engine.goal import GoalRunner, r0_revision_draft
from swarm_arena._engine.round import EngineDeps
from swarm_arena._engine.stops import RunClock, StopState
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._journal.event import EventDraft
from swarm_arena._journal.lease import Head, Lease
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.views import ViewWriter
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.ids import IdGen
from swarm_arena._model.tooling import Estimate
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("delayed_step", ["refresh", "view"])
async def test_concurrent_goal_returns_match_committed_order(
    tmp_path: Path,
    run_config: RunConfig,
    local_executor: LocalExecutor,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    delayed_step: str,
) -> None:
    config = run_config.model_copy(
        update={"limits": Limits(max_tokens=7, max_usd=0.000007), "reserve_fraction": 0.0}
    )
    store = LocalStore(Path(config.run_root) / "order")
    lease = Lease.acquire_new(store, owner_id="engine", host="local", ttl_s=60)
    ids = IdGen("order", lease.doc.epoch)
    async with JournalWriter(
        store,
        lease,
        run_id="order",
        next_seq=1,
        local_path=tmp_path / "order.jsonl",
        sync_interval_s=60,
        group_window_s=0,
    ) as journal:
        ledger = Ledger(
            config.limits, reserve_fraction=0, goal_weights=None, journal=journal, ids=ids
        )
        await journal.append_durable(
            EventDraft(type="run_config", payload={"config": config.model_dump(mode="json")})
        )
        slices = {g: Amount() for g in ("g1", "g2", "g3")}
        await ledger.hold_final_slices(slices)
        await journal.append_durable(
            EventDraft(
                type="run_started",
                payload={"final_slices": {g: a.model_dump() for g, a in slices.items()}},
            )
        )
        for g in slices:
            ledger.open_goal(g)
            await journal.append_durable(EventDraft(type="goal_started", goal_id=g))
        shares = ledger.open_round("g2", 1, ["g2/t/a"], 1)
        await journal.append_durable(
            EventDraft(
                type="round_started",
                goal_id="g2",
                payload={"round": 1, "shares": {h: a.model_dump() for h, a in shares.items()}},
            )
        )
        reservation = await ledger.reserve(
            "g2/t/a", Amount(tokens=1, usd=0.000001), kind="tool", goal_id="g2"
        )
        await ledger.settle(reservation.id, Amount(tokens=1, usd=0.000001))
        assert ledger.free() == Amount(tokens=1, usd=0.000001)
        assert ledger.goal_remaining("g1") == Estimate(tokens=2, usd=0.000002)
        assert ledger.goal_remaining("g2") == Estimate(tokens=1, usd=0.000001)
        board = ArenaBoard(arena_root=config.arena_root, cache_dir=str(tmp_path / "cache"))
        board.open("order", config.run_root, ["n"])
        views = ViewWriter(store, journal)
        deps = EngineDeps(
            run_id="order",
            config=config,
            store=store,
            journal=journal,
            ledger=ledger,
            board=board,
            executor=local_executor,
            artifacts=artifacts,
            ids=ids,
            views=views,
            stops=StopState(),
            clock=RunClock(0),
        )
        plans = {
            g: Plan(goal=Goal(id=g, namespace="n"), teams=[], judges=[], rounds=[RoundSpec()])
            for g in slices
        }
        for plan in plans.values():
            await journal.append_durable(r0_revision_draft(plan, plan.base_dir))
        first = GoalRunner(deps, goal_id="g1", plan=plans["g1"], replanner=None, journal_r0=False)
        second = GoalRunner(deps, goal_id="g2", plan=plans["g2"], replanner=None, journal_r0=False)
        arrived, finished = anyio.Event(), anyio.Event()
        real_write = views.write
        real_refresh = first._refresh

        async def slow_write(key: str, content: bytes, *, source: Head) -> None:
            if key == "goals/g1/solution.json" and delayed_step == "view":
                arrived.set()
                await finished.wait()
            await real_write(key, content, source=source)

        async def slow_refresh() -> None:
            await real_refresh()
            if delayed_step == "refresh" and first.recorded.goals["g1"].end_reason is not None:
                arrived.set()
                await finished.wait()

        monkeypatch.setattr(views, "write", slow_write)
        monkeypatch.setattr(first, "_refresh", slow_refresh)

        async def finish_second() -> None:
            await arrived.wait()
            await second.run()
            finished.set()

        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                group.start_soon(first.run)
                group.start_soon(finish_second)
        committed = read_committed(store)
        assert committed is not None
        _, events = committed
        assert [e.goal_id for e in events if e.type == "goal_finished"] == ["g1", "g2"]
        replay = Ledger.replay(RunState.from_events(events), journal=journal, ids=ids)
        assert ledger.goal_remaining("g3") == replay.goal_remaining("g3")
        assert ledger.goal_remaining("g3") == Estimate(tokens=5, usd=0.000005)
        assert ledger.free() == replay.free() == Amount(tokens=1, usd=0.000001)
        for g in slices:
            assert ledger.goal_remaining(g) == replay.goal_remaining(g)
            assert ledger.remaining(f"{g}/:final") == replay.remaining(f"{g}/:final")
        assert ledger._total == replay._total
        assert ledger._work == replay._work
        assert ledger._reserve_pool.left == replay._reserve_pool.left
        assert ledger._charges == replay._charges
        assert (
            ledger.open_round("g3", 1, ["g3/t/a"], 1)
            == replay.open_round("g3", 1, ["g3/t/a"], 1)
            == {"g3/t/a": Amount(tokens=5, usd=0.000005)}
        )
