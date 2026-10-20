from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Goal, Limits, Plan, RunConfig
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger, Reservation
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import GoalState, RunState, continues_on_resume
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.ids import IdGen
from swarm_arena._store.local import LocalStore
from swarm_arena.manager import LimitReached

HOLDERS = ["g1/t/a", "g1/t/b", "g1/:judges", "g1/:final", "g2/:final", ":run"]
SLICES = {"g1": Amount(usd=5), "g2": Amount(usd=5)}


@dataclass
class LedgerEnv:
    ledger: Ledger
    journal: JournalWriter
    ids: IdGen
    limits: Limits
    store: LocalStore

    async def committed(self) -> list[Event]:
        committed = await anyio.to_thread.run_sync(read_committed, self.store)
        assert committed is not None
        return committed[1]

    async def count(self, type: str) -> int:
        return sum(e.type == type for e in await self.committed())

    async def charged(self, holder: str) -> Amount:
        state = RunState.from_events(await self.committed())
        total = Amount()
        for value in state.reservations.values():
            reservation = Reservation.model_validate(value.reservation)
            if reservation.holder == holder:
                total += (
                    Amount.model_validate(value.actual)
                    if value.state == "settled" and not reservation.keep_charged
                    else reservation.amount
                )
        return total

    async def last_reservation(self) -> Reservation:
        event = next(e for e in reversed(await self.committed()) if e.type == "reserve")
        return Reservation.model_validate(event.payload["reservation"])

    async def emit(self, type: str, payload: dict[str, Any], goal_id: str = "") -> None:
        await self.journal.append_durable(
            EventDraft(type=EventType(type), goal_id=goal_id, payload=payload)
        )

    async def start(self, slices: Mapping[str, Amount] = SLICES) -> None:
        await self.ledger.hold_final_slices(slices)
        await self.emit(
            "run_started", {"final_slices": {g: a.model_dump() for g, a in slices.items()}}
        )

    async def goal(self, goal_id: str, pool: Amount | None = None) -> None:
        self.ledger.open_goal(goal_id, pool=pool)
        await self.emit(
            "goal_started", {} if pool is None else {"pool": pool.model_dump()}, goal_id
        )

    async def round(
        self, goal_id: str, round: int, holders: Sequence[str], rounds_left: int
    ) -> None:
        self.ledger.open_round(goal_id, round, holders, rounds_left=rounds_left)
        await self.emit(
            "round_started",
            {"round": round, "shares": {h: self.ledger.remaining(h).model_dump() for h in holders}},
            goal_id,
        )

    async def end(self, goal_id: str, reason: str | None = None) -> None:
        await self.emit("goal_finished", {"reason": reason}, goal_id)
        await self.ledger.end_goal(goal_id)

    async def spend(self, holder: str, amount: Amount, actual: Amount | None = None) -> None:
        reservation = await self.ledger.reserve(
            holder, amount, kind="tool", goal_id=holder.split("/")[0]
        )
        await self.ledger.settle(reservation.id, amount if actual is None else actual)


@pytest.fixture
async def env(store: LocalStore, tmp_path: Path, run_config: RunConfig) -> AsyncIterator[LedgerEnv]:
    limits = Limits(max_usd=100.0)
    config = run_config.model_copy(update={"limits": limits})
    lease = Lease.acquire_new(store, owner_id="ledger", host="local", ttl_s=60)
    ids = IdGen("ledger-run", 1)
    async with JournalWriter(
        store,
        lease,
        run_id="ledger-run",
        next_seq=1,
        local_path=tmp_path / "journal.jsonl",
        sync_interval_s=60,
        group_window_s=0,
    ) as journal:
        ledger = Ledger(limits, reserve_fraction=0.15, goal_weights=None, journal=journal, ids=ids)
        result = LedgerEnv(ledger, journal, ids, limits, store)
        await result.emit("run_config", {"config": config.model_dump(mode="json")})
        yield result


@pytest.fixture
async def ledger(env: LedgerEnv) -> Ledger:
    return env.ledger


async def _setup(env: LedgerEnv) -> None:
    await env.start()
    await env.goal("g1")
    await env.round("g1", 1, ["g1/t/a", "g1/t/b"], rounds_left=2)


def _balances(ledger: Ledger) -> dict[str, Amount]:
    return {
        **{h: ledger.remaining(h) for h in HOLDERS},
        "free": ledger.free(),
        "charged": sum(ledger._charges.values(), Amount()),
    }


async def test_final_slices_must_fit_reserve(ledger: Ledger) -> None:
    with pytest.raises(ValueError, match="reserve"):
        await ledger.hold_final_slices({"g1": Amount(usd=10), "g2": Amount(usd=10)})
    await ledger.hold_final_slices(SLICES)


async def test_round_and_agent_shares(ledger: Ledger) -> None:
    await ledger.hold_final_slices(SLICES)
    ledger.open_goal("g1")
    ledger.open_round("g1", 1, ["g1/t/a", "g1/t/b"], rounds_left=2)
    assert ledger.remaining("g1/t/a") == Amount(usd=10.625)


async def test_reserve_is_durable_and_refuses_overdraw(
    ledger: Ledger, env: LedgerEnv, store: LocalStore
) -> None:
    await _setup(env)
    res = await ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head is not None
    seq = next(
        e.seq for e in await env.committed() if e.payload.get("reservation", {}).get("id") == res.id
    )
    assert owner[0].head.seq >= seq
    assert await env.last_reservation() == res
    with pytest.raises(LimitReached) as err:
        await ledger.reserve("g1/t/a", Amount(usd=1), kind="tool", goal_id="g1")
    assert err.value.limit == "budget"


async def test_settle_is_idempotent_and_returns_unused_to_holder(
    ledger: Ledger, env: LedgerEnv
) -> None:
    await _setup(env)
    res = await ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    await ledger.settle(res.id, Amount(usd=4))
    await ledger.settle(res.id, Amount(usd=4))
    assert ledger.remaining("g1/t/a") == Amount(usd=6.625)
    assert await env.count("settle") == 1


async def test_estimated_settle_keeps_full_charge(ledger: Ledger, env: LedgerEnv) -> None:
    await _setup(env)
    res = await ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    await ledger.settle(res.id, Amount(), estimated=True)
    assert ledger.remaining("g1/t/a") == Amount(usd=0.625)


async def test_unused_agent_share_stays_in_goal_pool(ledger: Ledger, env: LedgerEnv) -> None:
    await _setup(env)
    await env.spend("g1/t/a", Amount(usd=10), Amount(usd=4))
    ledger.open_round("g1", 2, ["g1/t/a"], rounds_left=1)
    assert ledger.remaining("g1/t/a") == Amount(usd=38.5)


async def test_end_goal_returns_unused_to_running_and_queued_goals(
    ledger: Ledger, env: LedgerEnv
) -> None:
    await _setup(env)
    await env.spend("g1/t/a", Amount(usd=10), Amount(usd=4))
    await ledger.end_goal("g1")
    ledger.open_goal("g2")
    ledger.open_round("g2", 1, ["g2/t/a"], rounds_left=1)
    assert ledger.remaining("g2/t/a") == Amount(usd=42.5 + 38.5)


async def test_final_uses_slice_then_reserve_then_goal_pool(ledger: Ledger, env: LedgerEnv) -> None:
    await _setup(env)
    await ledger.reserve("g1/:judges", Amount(usd=5), kind="judge", goal_id="g1")
    await ledger.reserve("g1/:judges", Amount(usd=1), kind="judge", goal_id="g1")
    assert ledger.remaining("g1/:judges") == Amount(usd=41.5)
    assert ledger.remaining("g1/:final") == Amount(usd=46.5)
    await ledger.reserve("g1/:final", Amount(usd=5), kind="final", goal_id="g1")
    await ledger.reserve("g1/:final", Amount(usd=1), kind="final", goal_id="g1")
    assert ledger.remaining("g1/:judges") == Amount(usd=40.5)
    await ledger.reserve("g1/:final", Amount(usd=40.5), kind="final", goal_id="g1")
    with pytest.raises(LimitReached):
        await ledger.reserve("g1/:final", Amount(usd=0.01), kind="final", goal_id="g1")


async def test_overspend_is_recorded(ledger: Ledger, env: LedgerEnv) -> None:
    await _setup(env)
    res = await ledger.reserve("g1/t/a", Amount(usd=1), kind="tool", goal_id="g1")
    await ledger.settle(res.id, Amount(usd=1.5))
    event = next(e for e in reversed(await env.committed()) if e.type == "settle")
    assert event.payload["overspend"] == {"tokens": 0, "usd": 0.5}


async def test_parallel_reserve_never_overdraws(ledger: Ledger) -> None:
    await ledger.hold_final_slices(SLICES)
    ledger.open_goal("g1")
    ledger.open_round("g1", 1, ["g1/t/a", "g1/t/b"], rounds_left=2)
    results: list[str] = []

    async def one() -> None:
        try:
            await ledger.reserve("g1/t/a", Amount(usd=1.0), kind="tool", goal_id="g1")
            results.append("ok")
        except LimitReached as err:
            results.append(err.limit)

    async with anyio.create_task_group() as tg:
        for _ in range(20):
            tg.start_soon(one)
    assert results.count("ok") == 10
    assert results.count("budget") == 10
    assert ledger.remaining("g1/t/a") == Amount(usd=0.625)


async def test_unlimited_when_caps_unset(env: LedgerEnv) -> None:
    ledger = Ledger(
        Limits(), reserve_fraction=0.15, goal_weights=None, journal=env.journal, ids=env.ids
    )
    await ledger.hold_final_slices({"g1": Amount()})
    ledger.open_goal("g1")
    ledger.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
    await ledger.reserve("g1/t/a", Amount(usd=1e9), kind="tool", goal_id="g1")
    assert not ledger.goal_exhausted("g1")
    assert not ledger.run_exhausted()


async def test_replay_charges_unsettled_in_full(env: LedgerEnv) -> None:
    await _setup(env)
    await env.round("g1", 2, ["g1/t/a"], rounds_left=1)
    await env.ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    await env.spend("g1/t/a", Amount(usd=10), Amount(usd=4))
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert replayed.remaining("g1/t/a") == Amount(usd=28.5)
    assert await env.charged("g1/t/a") == Amount(usd=14)


async def test_replay_matches_live_ledger(ledger: Ledger, env: LedgerEnv) -> None:
    await _setup(env)
    await env.goal("g2")
    await env.spend("g1/t/a", Amount(usd=10.625))
    await ledger.reserve("g1/:judges", Amount(usd=6), kind="judge", goal_id="g1")
    await env.end("g2")
    live = {h: ledger.remaining(h) for h in HOLDERS}
    before = await env.committed()
    replayed = Ledger.replay(RunState.from_events(before), journal=env.journal, ids=env.ids)
    assert {h: replayed.remaining(h) for h in HOLDERS} == live
    assert replayed.remaining("g1/t/a") == Amount()
    assert replayed.remaining("g1/t/b") == Amount(usd=10.625)
    assert await env.committed() == before


async def test_replay_restores_final_slices(env: LedgerEnv) -> None:
    await _setup(env)
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    before = replayed.remaining("g1/:judges")
    await replayed.reserve("g1/:final", Amount(usd=5), kind="final", goal_id="g1")
    assert replayed.remaining("g1/:judges") == before
    await replayed.reserve("g1/:final", Amount(usd=0.01), kind="final", goal_id="g1")
    assert replayed.remaining("g1/:judges") == before - Amount(usd=0.01)


async def test_limits_changed_replayed_at_its_position(env: LedgerEnv) -> None:
    await _setup(env)
    await env.goal("g2")
    await env.end("g2")
    new = Limits(max_usd=200.0)
    env.ledger.change_limits(new)
    await env.emit("limits_changed", {"old": env.limits.model_dump(), "new": new.model_dump()})
    state = RunState.from_events(await env.committed())
    for _ in range(2):
        replayed = Ledger.replay(state, journal=env.journal, ids=env.ids)
        replayed.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
        assert replayed.remaining("g1/t/a") == Amount(usd=170.0)
        assert replayed.limits == new


async def _lifecycles(env: LedgerEnv) -> None:
    await _setup(env)
    await env.goal("g2")
    await env.spend("g1/t/a", Amount(usd=4))
    await env.end("g1", "stop")
    await env.round("g2", 1, ["g2/t/a"], rounds_left=1)
    await env.spend("g2/t/a", Amount(usd=50))
    await env.end("g2", "stop")
    assert env.ledger.free() == Amount(usd=31.0)
    await env.goal("g1", Amount(usd=15.5))
    with pytest.raises(ValueError, match="free pool"):
        env.ledger.open_goal("g2", pool=Amount(usd=16.0))
    await env.goal("g2", Amount(usd=15.5))
    assert env.ledger.free() == Amount()


async def test_continued_goal_reopens_from_free_pool(env: LedgerEnv) -> None:
    await _lifecycles(env)
    env.ledger.open_round("g1", 2, ["g1/t/a"], rounds_left=2)
    assert env.ledger.remaining("g1/t/a") == Amount(usd=7.75)


async def test_replay_is_history_only_across_lifecycles(env: LedgerEnv) -> None:
    await _lifecycles(env)
    a, prefix = _balances(env.ledger), await env.committed()
    await env.round("g1", 2, ["g1/t/a"], rounds_left=2)
    await env.spend("g1/t/a", Amount(usd=5))
    await env.end("g1")
    b, full = _balances(env.ledger), await env.committed()
    assert (
        _balances(Ledger.replay(RunState.from_events(prefix), journal=env.journal, ids=env.ids))
        == a
    )
    assert (
        _balances(Ledger.replay(RunState.from_events(full), journal=env.journal, ids=env.ids)) == b
    )
    assert (a["charged"], b["charged"]) == (Amount(usd=54.0), Amount(usd=59.0))


async def test_exhaustion_flags(ledger: Ledger, env: LedgerEnv) -> None:
    await _setup(env)
    await env.goal("g2")
    await env.spend("g1/t/a", Amount(usd=10))
    await ledger.reserve("g1/:judges", Amount(usd=37.5), kind="judge", goal_id="g1")
    assert ledger.goal_exhausted("g1")
    assert not ledger.run_exhausted()
    await env.round("g2", 1, ["g2/t/a"], rounds_left=1)
    await env.spend("g2/t/a", Amount(usd=1), Amount(usd=43))
    assert ledger.run_exhausted()
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert replayed.run_exhausted()


@pytest.mark.parametrize("reason", ["stop", "stop --now", "run budget"])
def test_continues_on_resume(reason: str) -> None:
    limits = Limits(max_usd=100)
    state = RunState(
        config=RunConfig(limits=limits),
        goals={"g1": GoalState(status="stopped", rounds_closed=1, end_reason=reason)},
        plans={"g1": [(Plan(goal=Goal(id="g1", namespace="n"), teams=[], judges=[]), "human")]},
    )
    assert continues_on_resume(state, "g1", limits) == (reason != "run budget")
    assert continues_on_resume(state, "g1", Limits(max_usd=200))
    state.goals["g1"] = GoalState(status="stopped", rounds_closed=3, end_reason=reason)
    assert not continues_on_resume(state, "g1", Limits(max_usd=200))
    state.goals["g1"] = GoalState(status="finished", end_reason=reason)
    assert not continues_on_resume(state, "g1", Limits(max_usd=200))


async def test_replay_invalid_payload_does_not_expose_values(env: LedgerEnv) -> None:
    await _setup(env)
    state = RunState.from_events(await env.committed())
    sentinel = "secret-budget-value"
    state.ledger_events[1] = state.ledger_events[1].model_copy(
        update={"payload": {"final_slices": {"g1": {"usd": sentinel}}}}
    )
    with pytest.raises(ValueError, match="replay payload") as exc:
        Ledger.replay(state, journal=env.journal, ids=env.ids)
    assert_secret_absent(exc.value, sentinel)


async def test_mixed_units_fund_and_refund_in_source_order(env: LedgerEnv) -> None:
    ledger = Ledger(
        Limits(max_tokens=100, max_usd=100),
        reserve_fraction=0.15,
        goal_weights=None,
        journal=env.journal,
        ids=env.ids,
    )
    await ledger.hold_final_slices({"g1": Amount(tokens=10, usd=5), "g2": Amount(usd=5)})
    ledger.open_goal("g1")
    res = await ledger.reserve("g1/:final", Amount(tokens=20, usd=12), kind="final", goal_id="g1")
    assert ledger.remaining(":run") == Amount()
    assert ledger.remaining("g1/:judges") == Amount(tokens=37, usd=40.5)
    await ledger.settle(res.id, Amount(tokens=12, usd=6))
    assert ledger.remaining(":run") == Amount(tokens=3, usd=4)
    assert ledger.remaining("g1/:judges") == Amount(tokens=45, usd=46.5)
    assert ledger.free() == Amount(tokens=1)


@pytest.mark.parametrize("holder", ["g1/:judges", "g1/:planner", "g1/:curator"])
async def test_services_share_reserve_but_never_final_slices(env: LedgerEnv, holder: str) -> None:
    await _setup(env)
    res = await env.ledger.reserve(holder, Amount(usd=6), kind="planner", goal_id="g1")
    assert env.ledger.remaining(":run") == Amount()
    assert env.ledger.remaining("g1/:final") == Amount(usd=46.5)
    await env.ledger.settle(res.id, Amount(usd=4))
    assert env.ledger.remaining(":run") == Amount(usd=1)
    assert env.ledger.remaining("g1/:judges") == Amount(usd=43.5)
    assert _balances(
        Ledger.replay(RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids)
    ) == _balances(env.ledger)


async def test_keep_charged_settlement_survives_replay(env: LedgerEnv) -> None:
    await _setup(env)
    res = await env.ledger.reserve(
        "g1/t/a", Amount(usd=10), kind="tool", goal_id="g1", keep_charged=True
    )
    await env.ledger.settle(res.id, Amount(usd=4))
    assert env.ledger.remaining("g1/t/a") == Amount(usd=0.625)
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert _balances(replayed) == _balances(env.ledger)
    assert await env.charged("g1/t/a") == Amount(usd=10)


async def test_failed_reserve_keeps_debit(env: LedgerEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    await _setup(env)

    async def fail(draft: EventDraft) -> Event:
        raise RuntimeError("journal unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(env.journal, "append_durable", fail)
        with pytest.raises(RuntimeError, match="journal unavailable"):
            await env.ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    assert env.ledger.remaining("g1/t/a") == Amount(usd=0.625)
    assert await env.count("reserve") == 0


async def test_failed_settle_does_not_release_budget(
    env: LedgerEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _setup(env)
    res = await env.ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")

    async def fail(draft: EventDraft) -> Event:
        raise RuntimeError("journal unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(env.journal, "append_durable", fail)
        with pytest.raises(RuntimeError, match="journal unavailable"):
            await env.ledger.settle(res.id, Amount(usd=4))
    assert env.ledger.remaining("g1/t/a") == Amount(usd=0.625)
    assert await env.count("settle") == 0
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert replayed.remaining("g1/t/a") == Amount(usd=0.625)
    await env.ledger.settle(res.id, Amount(usd=4))
    assert env.ledger.remaining("g1/t/a") == Amount(usd=6.625)


async def test_parallel_settles_wait_for_durable_acknowledgement(
    env: LedgerEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _setup(env)
    res = await env.ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    entered, release = anyio.Event(), anyio.Event()
    original = env.journal._wait_for
    completed: list[bool] = []

    async def delayed(seq: int) -> None:
        entered.set()
        await release.wait()
        await original(seq)

    async def one() -> None:
        await env.ledger.settle(res.id, Amount(usd=4))
        completed.append(True)

    with monkeypatch.context() as patch:
        patch.setattr(env.journal, "_wait_for", delayed)
        async with anyio.create_task_group() as tg:
            tg.start_soon(one)
            tg.start_soon(one)
            await entered.wait()
            try:
                assert completed == []
            finally:
                release.set()
    assert completed == [True, True]
    assert await env.count("settle") == 1
    assert env.ledger.remaining("g1/t/a") == Amount(usd=6.625)


async def test_late_settle_does_not_enlarge_new_round_share(env: LedgerEnv) -> None:
    await _setup(env)
    res = await env.ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    await env.round("g1", 2, ["g1/t/a"], rounds_left=1)
    assert env.ledger.remaining("g1/t/a") == Amount(usd=32.5)
    await env.ledger.settle(res.id, Amount(usd=4))
    assert env.ledger.remaining("g1/t/a") == Amount(usd=32.5)
    assert env.ledger.remaining("g1/:judges") == Amount(usd=43.5)
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert _balances(replayed) == _balances(env.ledger)


async def test_late_refund_to_ended_goal_returns_to_free_pool(env: LedgerEnv) -> None:
    await _setup(env)
    res = await env.ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    await env.end("g1")
    await env.end("g2")
    assert env.ledger.free() == Amount(usd=75)
    await env.ledger.settle(res.id, Amount(usd=4))
    assert env.ledger.free() == Amount(usd=81)
    assert env.ledger.remaining("g1/t/a") == Amount()
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert _balances(replayed) == _balances(env.ledger)


async def test_round_started_restores_exact_shares_and_skips_converge(env: LedgerEnv) -> None:
    await _setup(env)
    await env.emit(
        "round_started", {"round": 2, "shares": {"g1/t/a": Amount(usd=7).model_dump()}}, "g1"
    )
    await env.emit("round_started", {"round": 3, "shares": {}, "converge": True}, "g1")
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert replayed.remaining("g1/t/a") == Amount(usd=7)
    assert replayed.remaining("g1/t/b") == Amount()


async def test_raised_caps_after_all_goals_end_go_to_free_pool(env: LedgerEnv) -> None:
    await _setup(env)
    await env.end("g1")
    await env.end("g2")
    new = Limits(max_usd=200)
    env.ledger.change_limits(new)
    await env.emit("limits_changed", {"old": env.limits.model_dump(), "new": new.model_dump()})
    assert env.ledger.free() == Amount(usd=170)
    assert env.ledger.remaining(":run") == Amount(usd=20)
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert _balances(replayed) == _balances(env.ledger)
    with pytest.raises(AttributeError):
        setattr(env.ledger, "limits", Limits())  # noqa: B010 - probe the runtime setter


async def test_weighted_pools_keep_exact_remainders(env: LedgerEnv) -> None:
    ledger = Ledger(
        Limits(max_tokens=101, max_usd=0.000101),
        reserve_fraction=0.15,
        goal_weights={"g1": 1, "g2": 2},
        journal=env.journal,
        ids=env.ids,
    )
    await ledger.hold_final_slices({"g1": Amount(), "g2": Amount()})
    ledger.open_goal("g1")
    ledger.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
    ledger.open_goal("g2")
    ledger.open_round("g2", 1, ["g2/t/a"], rounds_left=1)
    assert ledger.remaining("g1/t/a") == Amount(tokens=28, usd=0.000028)
    assert ledger.remaining("g2/t/a") == Amount(tokens=57, usd=0.000057)
    assert ledger.free() == Amount(tokens=1, usd=0.000001)
    await ledger.end_goal("g1")
    await ledger.end_goal("g2")
    assert ledger.free() == Amount(tokens=86, usd=0.000086)


@pytest.mark.parametrize("amount", [Amount(tokens=-1), Amount(usd=-0.01)])
async def test_negative_costs_refused_without_pool_mutation(env: LedgerEnv, amount: Amount) -> None:
    await _setup(env)
    before = _balances(env.ledger)
    with pytest.raises(ValueError, match="nonnegative"):
        await env.ledger.reserve("g1/t/a", amount, kind="tool", goal_id="g1")
    assert _balances(env.ledger) == before


async def test_decimal_usd_sum_is_exact(env: LedgerEnv) -> None:
    await _setup(env)
    for _ in range(100):
        await env.ledger.reserve("g1/t/a", Amount(usd=0.1), kind="tool", goal_id="g1")
    assert env.ledger.remaining("g1/t/a") == Amount(usd=0.625)
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert replayed.remaining("g1/t/a") == Amount(usd=0.625)


async def test_replay_negative_pool_is_exhausted(env: LedgerEnv) -> None:
    await _setup(env)
    await env.emit(
        "reserve",
        {
            "reservation": Reservation(
                id="replayed-overdraw",
                holder="g1/t/a",
                kind="tool",
                goal_id="g1",
                amount=Amount(usd=90),
            ).model_dump()
        },
    )
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert replayed.goal_exhausted("g1")
    assert replayed.run_exhausted()


async def test_limits_removing_cap_allows_unlimited_reservations(env: LedgerEnv) -> None:
    await _setup(env)
    env.ledger.change_limits(Limits())
    await env.ledger.reserve("g1/t/a", Amount(usd=1e9), kind="tool", goal_id="g1")
    assert not env.ledger.run_exhausted()


async def test_failed_overspend_settle_retry_only_charges_once(
    env: LedgerEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _setup(env)
    res = await env.ledger.reserve("g1/t/a", Amount(usd=1), kind="tool", goal_id="g1")

    async def fail(draft: EventDraft) -> Event:
        raise RuntimeError("journal unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(env.journal, "append_durable", fail)
        with pytest.raises(RuntimeError, match="journal unavailable"):
            await env.ledger.settle(res.id, Amount(usd=2))
    assert env.ledger.remaining("g1/t/a") == Amount(usd=8.625)
    await env.ledger.settle(res.id, Amount(usd=2))
    assert env.ledger.remaining("g1/t/a") == Amount(usd=8.625)
    assert await env.count("settle") == 1


async def test_final_overspend_uses_available_sources_in_order(env: LedgerEnv) -> None:
    await _setup(env)
    res = await env.ledger.reserve("g1/:final", Amount(usd=1), kind="final", goal_id="g1")
    await env.ledger.settle(res.id, Amount(usd=7))
    assert env.ledger.remaining(":run") == Amount(usd=3)
    assert env.ledger.remaining("g1/:judges") == Amount(usd=45.5)
    assert env.ledger.remaining("g1/:final") == Amount(usd=45.5)
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert _balances(replayed) == _balances(env.ledger)


async def test_run_cap_prevents_reserving_against_another_goals_overspend(env: LedgerEnv) -> None:
    await _setup(env)
    await env.spend("g1/t/a", Amount(usd=1), Amount(usd=95))
    await env.goal("g2")
    await env.round("g2", 1, ["g2/t/a"], rounds_left=1)
    assert env.ledger.remaining("g2/t/a") == Amount(usd=5)
    with pytest.raises(LimitReached):
        await env.ledger.reserve("g2/t/a", Amount(usd=6), kind="tool", goal_id="g2")


async def test_reserve_during_settle_acknowledgement_matches_replay(
    env: LedgerEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _setup(env)
    res = await env.ledger.reserve("g1/:judges", Amount(usd=5), kind="judge", goal_id="g1")
    committed, acknowledge = anyio.Event(), anyio.Event()
    original = env.journal.append_durable

    async def delayed_ack(draft: EventDraft) -> Event:
        event = await original(draft)
        if draft.type == EventType.SETTLE:
            committed.set()
            await acknowledge.wait()
        return event

    with monkeypatch.context() as patch:
        patch.setattr(env.journal, "append_durable", delayed_ack)
        async with anyio.create_task_group() as tg:
            tg.start_soon(env.ledger.settle, res.id, Amount())
            await committed.wait()
            try:
                await env.ledger.reserve("g1/:judges", Amount(usd=1), kind="judge", goal_id="g1")
            finally:
                acknowledge.set()
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert _balances(replayed) == _balances(env.ledger)


async def test_cancellation_after_settle_admission_preserves_live_replay_parity(
    env: LedgerEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _setup(env)
    res = await env.ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    admitted, release, completed = anyio.Event(), anyio.Event(), anyio.Event()
    original = env.journal._wait_for
    scopes: list[anyio.CancelScope] = []

    async def delayed(seq: int) -> None:
        admitted.set()
        await release.wait()
        await original(seq)

    async def settling() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await env.ledger.settle(res.id, Amount(usd=4))
            completed.set()

    with monkeypatch.context() as patch:
        patch.setattr(env.journal, "_wait_for", delayed)
        async with anyio.create_task_group() as tg:
            tg.start_soon(settling)
            await admitted.wait()
            scopes[0].cancel()
            release.set()
    assert completed.is_set()
    await env.ledger.settle(res.id, Amount(usd=4))
    assert await env.count("settle") == 1
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    assert _balances(replayed) == _balances(env.ledger)


async def test_aggregate_overspend_beyond_amount_range_stays_exact_and_exhausted(
    env: LedgerEnv,
) -> None:
    await _setup(env)
    a = await env.ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    b = await env.ledger.reserve("g1/t/b", Amount(usd=10), kind="tool", goal_id="g1")
    await env.ledger.settle(a.id, Amount(usd=1e9))
    await env.ledger.settle(b.id, Amount(usd=1e9))
    replayed = Ledger.replay(
        RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids
    )
    for ledger in (env.ledger, replayed):
        assert ledger.run_exhausted()
        assert ledger.goal_exhausted("g1")
        assert ledger.remaining("g1/t/a") == Amount()
        assert sum(amount._micro_usd for amount in ledger._charges.values()) == 2 * 10**15
        assert ledger._work._micro_usd == 85 * 10**6 - 2 * 10**15
