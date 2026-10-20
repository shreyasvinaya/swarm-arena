from dataclasses import dataclass, field
from typing import cast

import anyio
import anyio.lowlevel
import pytest

from swarm_arena import Limits
from swarm_arena._budget.amount import USD_LIMIT, Amount
from swarm_arena._budget.ledger import Ledger, Reservation
from swarm_arena._budget.tranche import Tranche
from swarm_arena._journal.event import EventDraft
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.ids import IdGen
from swarm_arena.manager import LimitReached


@dataclass
class _Journal:
    events: list[EventDraft] = field(default_factory=list)

    async def append_durable(self, event: EventDraft) -> None:
        self.events.append(event)
        await anyio.lowlevel.checkpoint()


@dataclass
class LedgerEnv:
    ledger: Ledger
    journal: _Journal

    def count(self, kind: str) -> int:
        return sum(e.type == kind for e in self.journal.events)

    def last_reservation(self) -> Reservation:
        event = next(e for e in reversed(self.journal.events) if e.type == "reserve")
        return Reservation.model_validate(event.payload["reservation"])

    def charged(self, holder: str) -> Amount:
        charges: dict[str, Amount] = {}
        for e in self.journal.events:
            if e.type == "reserve":
                r = Reservation.model_validate(e.payload["reservation"])
                if r.holder == holder:
                    charges[r.id] = r.amount
            elif e.type == "settle" and e.payload["id"] in charges:
                if not e.payload["estimated"]:
                    charges[e.payload["id"]] = Amount.model_validate(e.payload["actual"])
        return sum(charges.values(), Amount())


async def _env(limits: Limits) -> LedgerEnv:
    journal = _Journal()
    ledger = Ledger(
        limits,
        reserve_fraction=0.15,
        goal_weights=None,
        journal=cast(JournalWriter, journal),
        ids=IdGen("tranche", 1),
    )
    await ledger.hold_final_slices({"g1": Amount()})
    ledger.open_goal("g1")
    ledger.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
    return LedgerEnv(ledger, journal)


@pytest.fixture
async def env() -> LedgerEnv:
    return await _env(Limits(max_usd=100))


async def test_calls_spend_locally_and_new_tranche_when_used_up(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(tokens=25_000, usd=0.25))
    ids = [await t.reserve_call(Amount(usd=0.1)) for _ in range(3)]
    assert env.count("reserve") == 2
    assert len(set(ids)) == 3
    assert t.pending() == 3


async def test_call_bigger_than_tranche_gets_own_reservation(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    cid = await t.reserve_call(Amount(usd=1.0))
    assert env.last_reservation().kind == "call"
    assert env.last_reservation().id == cid
    await t.settle_call(cid, Amount(usd=0.2))
    assert env.charged("g1/t/a") == Amount(usd=0.2)
    assert t.pending() == 0


async def test_close_returns_unused_and_charges_unsettled_calls(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    a = await t.reserve_call(Amount(usd=0.05))
    await t.reserve_call(Amount(usd=0.05))
    await t.settle_call(a, Amount(usd=0.01))
    await t.close()
    assert env.charged("g1/t/a") == Amount(usd=0.06)
    assert t.pending() == 0
    settles = env.count("settle")
    await t.close()
    assert env.count("settle") == settles
    with pytest.raises(ValueError, match="closed"):
        await t.reserve_call(Amount())


async def test_parallel_calls_open_one_new_tranche(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    ids = [await t.reserve_call(Amount(usd=0.2))]

    async def one() -> None:
        ids.append(await t.reserve_call(Amount(usd=0.1)))

    async with anyio.create_task_group() as tg:
        tg.start_soon(one)
        tg.start_soon(one)
    assert env.count("reserve") == 2
    assert len(set(ids)) == 3


async def test_settle_after_close_is_noop(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    a = await t.reserve_call(Amount(usd=0.05))
    await t.close()
    settles = env.count("settle")
    await t.settle_call(a, Amount(usd=0.01))
    assert env.charged("g1/t/a") == Amount(usd=0.05)
    assert env.count("settle") == settles


async def test_new_tranche_shrinks_to_what_is_left() -> None:
    env = await _env(Limits(max_tokens=20_000))
    await env.ledger.reserve("g1/t/a", Amount(tokens=7_000), kind="tool", goal_id="g1")
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(tokens=25_000))
    await t.reserve_call(Amount(tokens=5_000))
    assert env.last_reservation().amount == Amount(tokens=10_000)
    with pytest.raises(LimitReached) as exc:
        await t.reserve_call(Amount(tokens=12_000))
    assert exc.value.limit == "budget"


@pytest.mark.parametrize("limits", [Limits(), Limits(max_tokens=100_000), Limits(max_usd=1)])
async def test_only_capped_units_shrink(limits: Limits) -> None:
    env = await _env(limits)
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(tokens=25_000, usd=0.25))
    await t.reserve_call(Amount(tokens=1_000, usd=0.1))
    assert env.last_reservation().amount == Amount(tokens=25_000, usd=0.25)


async def test_refunds_reused_and_settlement_is_idempotent(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.3))
    a = await t.reserve_call(Amount(usd=0.2))
    await t.settle_call(a, Amount(usd=0.1))
    await t.settle_call(a, Amount(usd=0))
    await t.reserve_call(Amount(usd=0.2))
    assert env.count("reserve") == 1
    await t.close()
    assert env.charged("g1/t/a") == Amount(usd=0.3)


async def test_overspend_and_direct_unsettled_call(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    a = await t.reserve_call(Amount(usd=0.1))
    await t.settle_call(a, Amount(usd=0.3))
    await t.reserve_call(Amount(usd=0.5))
    await t.close()
    assert env.charged("g1/t/a") == Amount(usd=0.8)


async def test_exact_micro_dollars_at_boundary() -> None:
    env = await _env(Limits())
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=USD_LIMIT))
    a = await t.reserve_call(Amount(usd=USD_LIMIT - 0.000001))
    await t.reserve_call(Amount(usd=0.000001))
    assert env.count("reserve") == 1
    await t.settle_call(a, Amount(usd=USD_LIMIT - 0.000002))
    await t.close()
    assert env.charged("g1/t/a") == Amount(usd=USD_LIMIT - 0.000001)
