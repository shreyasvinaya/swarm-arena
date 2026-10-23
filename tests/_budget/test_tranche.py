import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import anyio
import anyio.lowlevel
import pytest

from swarm_arena import Limits, RunConfig
from swarm_arena._budget.amount import USD_LIMIT, Amount
from swarm_arena._budget.ledger import Ledger, Reservation
from swarm_arena._budget.tranche import Tranche
from swarm_arena._journal.event import Event, EventDraft
from swarm_arena._journal.lease import Lease
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.ids import IdGen
from swarm_arena._store.local import LocalStore
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
    await journal.append_durable(
        EventDraft(type="run_config", payload={"config": RunConfig(limits=limits).model_dump()})
    )
    await ledger.hold_final_slices({"g1": Amount()})
    await journal.append_durable(
        EventDraft(type="run_started", payload={"final_slices": {"g1": Amount().model_dump()}})
    )
    ledger.open_goal("g1")
    await journal.append_durable(EventDraft(type="goal_started", goal_id="g1"))
    shares = ledger.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
    await journal.append_durable(
        EventDraft(
            type="round_started",
            goal_id="g1",
            payload={"round": 1, "shares": {h: a.model_dump() for h, a in shares.items()}},
        )
    )
    return LedgerEnv(ledger, journal)


def _replay(env: LedgerEnv) -> Ledger:
    events = [
        Event.model_validate(
            dict(
                draft.model_dump(),
                epoch=1,
                seq=seq,
                ts=utc_now(),
                run_id="tranche",
                context_id="tranche",
            )
        )
        for seq, draft in enumerate(env.journal.events, 1)
    ]
    return Ledger.replay(
        RunState.from_events(events),
        journal=cast(JournalWriter, env.journal),
        ids=IdGen("tranche", 2),
    )


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


@pytest.mark.parametrize("direct", [False, True], ids=["batched", "direct"])
@pytest.mark.parametrize("after_commit", [False, True], ids=["group-wait", "durable-ack"])
@pytest.mark.parametrize("native", [False, True], ids=["anyio", "native"])
async def test_cancelled_reserve_returns_unused_funding_on_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direct: bool, after_commit: bool, native: bool
) -> None:
    store = LocalStore(tmp_path / "store")
    lease = Lease.acquire_new(store, owner_id="cancel", host="local", ttl_s=60)
    ids = IdGen("cancel", 1)
    config = RunConfig(limits=Limits(max_usd=1), reserve_fraction=0)

    def committed_events() -> list[Event]:
        snapshot = read_committed(store)
        assert snapshot is not None
        return snapshot[1]

    async with JournalWriter(
        store,
        lease,
        run_id="cancel",
        next_seq=1,
        local_path=tmp_path / "journal.jsonl",
        sync_interval_s=60,
        group_window_s=0.05,
    ) as journal:
        ledger = Ledger(
            config.limits, reserve_fraction=0, goal_weights=None, journal=journal, ids=ids
        )
        await journal.append_durable(
            EventDraft(type="run_config", payload={"config": config.model_dump(mode="json")})
        )
        await ledger.hold_final_slices({"g": Amount()})
        await journal.append_durable(
            EventDraft(type="run_started", payload={"final_slices": {"g": Amount().model_dump()}})
        )
        ledger.open_goal("g")
        await journal.append_durable(EventDraft(type="goal_started", goal_id="g"))
        shares = ledger.open_round("g", 1, ["g/t/a"], 1)
        await journal.append_durable(
            EventDraft(
                type="round_started",
                goal_id="g",
                payload={"round": 1, "shares": {h: a.model_dump() for h, a in shares.items()}},
            )
        )
        tranche = Tranche(ledger, "g/t/a", "g", size=Amount(usd=0.25))
        scope = anyio.CancelScope()
        committed, release_ack = anyio.Event(), anyio.Event()
        original = journal.append_durable

        async def delayed_ack(draft: EventDraft) -> Event:
            event = await original(draft)
            if draft.type == "reserve":
                committed.set()
                await release_ack.wait()
            return event

        if after_commit:
            monkeypatch.setattr(journal, "append_durable", delayed_ack)
        dispatched: list[str] = []
        callers: list[asyncio.Task[object]] = []
        cancelled: list[bool] = []

        async def request() -> None:
            caller = asyncio.current_task()
            assert caller is not None
            callers.append(caller)
            try:
                with scope:
                    dispatched.append(
                        await tranche.reserve_call(Amount(usd=0.5 if direct else 0.1))
                    )
            except asyncio.CancelledError:
                cancelled.append(True)

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(request)
                if after_commit:
                    await committed.wait()
                    assert any(e.type == "reserve" for e in committed_events())
                else:
                    while not ledger._entries:
                        await anyio.lowlevel.checkpoint()
                    assert not any(e.type == "reserve" for e in committed_events())
                if native:
                    callers[0].cancel()
                else:
                    scope.cancel()
                # Let a shielded acquisition finish, while cancellation remains pending.
                release_ack.set()
        assert scope.cancelled_caught is not native
        assert cancelled == ([True] if native else [])
        assert dispatched == []
        await journal.commit()
        outcome = await tranche.close()
        assert await tranche.close() is outcome
        assert not outcome.overflow
        assert tranche.pending() == 0
        events = committed_events()
        reserves = [e for e in events if e.type == "reserve"]
        settles = [e for e in events if e.type == "settle"]
        assert len(reserves) == len(settles) == 1
        reservation = Reservation.model_validate(reserves[0].payload["reservation"])
        assert reservation.kind == ("call" if direct else "tranche")
        assert settles[0].payload["id"] == reservation.id
        assert Amount.model_validate(settles[0].payload["actual"]) == Amount()
        assert settles[0].payload["estimated"] is False
        replay = Ledger.replay(
            RunState.from_events(events), journal=journal, ids=IdGen("cancel", 2)
        )
        for state in (ledger, replay):
            assert state.remaining("g/t/a") == Amount(usd=1)
            assert state._charges == {reservation.id: Amount()}
            assert all(entry.settled for entry in state._entries.values())
        assert replay.free() == ledger.free()
        assert replay.goal_remaining("g") == ledger.goal_remaining("g")
        # A fresh paid call proves the writer and returned funding remain usable.
        followup = Tranche(ledger, "g/t/a", "g", size=Amount(usd=0.25))
        cid = await followup.reserve_call(Amount(usd=1))
        await followup.settle_call(cid, Amount(usd=0.1))
        await followup.close()
        replay = Ledger.replay(
            RunState.from_events(committed_events()), journal=journal, ids=IdGen("cancel", 2)
        )
        assert replay.remaining("g/t/a") == ledger.remaining("g/t/a") == Amount(usd=0.9)
        assert all(entry.settled for entry in replay._entries.values())


async def test_cancelled_batch_cannot_fund_calls_before_durability_is_confirmed(
    env: LedgerEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    tranche = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    append = env.journal.append_durable
    attempts = 0

    async def fail_acknowledgement(draft: EventDraft) -> None:
        nonlocal attempts
        await append(draft)
        if draft.type == "reserve":
            attempts += 1
            if attempts == 1:
                raise asyncio.CancelledError
            raise RuntimeError("durability not confirmed")

    monkeypatch.setattr(env.journal, "append_durable", fail_acknowledgement)
    with pytest.raises(asyncio.CancelledError):
        await tranche.reserve_call(Amount(usd=0.05))
    with pytest.raises(RuntimeError, match="durability not confirmed"):
        await tranche.reserve_call(Amount(usd=0.05))
    assert attempts == 2
    await tranche.close()
    assert env.count("settle") == 2
    assert env.charged("g1/t/a") == Amount()


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


@pytest.mark.parametrize("limits", [Limits(max_usd=1), Limits(max_tokens=1000), Limits()])
async def test_close_accepts_individually_valid_usage(
    limits: Limits, caplog: pytest.LogCaptureFixture
) -> None:
    env = await _env(limits)
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    first = await t.reserve_call(Amount(usd=0.1))
    second = await t.reserve_call(Amount(usd=0.1))
    await t.settle_call(first, Amount(usd=USD_LIMIT))
    await t.settle_call(second, Amount(usd=0.000001))

    outcome = await t.close()
    assert env.charged("g1/t/a") == Amount(usd=USD_LIMIT)
    assert outcome.overflow
    assert t.pending() == 0
    settles = env.count("settle")
    assert await t.close() is outcome
    assert await t.settle_call(second, Amount()) is outcome
    assert env.count("settle") == settles == 1
    assert [r.getMessage() for r in caplog.records] == [
        "reported usage exceeds the supported range; charged as the range limit"
    ]
    assert env.ledger.run_exhausted()
    assert env.ledger.goal_exhausted("g1")
    with pytest.raises(LimitReached) as exc:
        await env.ledger.reserve("g1/t/a", Amount(tokens=1), kind="call", goal_id="g1")
    assert exc.value.limit == "usage_overflow"
    event = next(e for e in env.journal.events if e.type == "settle")
    assert event.payload["usage_overflow"] is True
    replayed = _replay(env)
    assert replayed.run_exhausted()
    assert replayed.goal_exhausted("g1")
    assert replayed.remaining("g1/t/a") == env.ledger.remaining("g1/t/a") == Amount()
    assert replayed._charges == env.ledger._charges
    with pytest.raises(LimitReached) as exc:
        await replayed.reserve(":run", Amount(), kind="planner", goal_id="")
    assert exc.value.limit == "usage_overflow"


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("usd", [USD_LIMIT + 1, USD_LIMIT + 0.0000001])
async def test_single_over_range_usage_is_explicit_and_retry_safe(
    direct: bool, usd: float, caplog: pytest.LogCaptureFixture
) -> None:
    env = await _env(Limits(max_tokens=1000))
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    cid = await t.reserve_call(Amount(usd=0.5 if direct else 0.1))
    # Normal Amount construction forbids this; probe the reported-usage seam defensively.
    actual = Amount.model_construct(tokens=7, usd=usd)
    settlement = await t.settle_call(cid, actual)
    assert settlement.overflow
    assert env.ledger.run_exhausted()
    outcome = await t.close()
    assert outcome is settlement
    assert await t.close() is outcome
    assert env.charged("g1/t/a") == Amount(tokens=7, usd=USD_LIMIT)
    assert env.ledger.run_exhausted()
    assert env.count("settle") == 1
    assert len(caplog.records) == 1


@pytest.mark.parametrize("direct", [False, True])
async def test_two_overflows_and_late_settlements_are_noops(direct: bool) -> None:
    env = await _env(Limits())
    first = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    second = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    estimate = Amount(usd=0.5 if direct else 0.1)
    a = await first.reserve_call(estimate)
    b = await second.reserve_call(estimate)
    late = await env.ledger.reserve("g1/t/a", estimate, kind="tool", goal_id="g1")
    actual = Amount.model_construct(tokens=11, usd=USD_LIMIT + 1)
    assert (await first.settle_call(a, actual)).overflow
    charges = dict(env.ledger._charges)
    count = env.count("settle")
    assert (await second.settle_call(b, actual)).overflow
    assert (await first.close()).overflow
    await env.ledger.settle(late.id, Amount(usd=USD_LIMIT))
    await env.ledger.settle(late.id, Amount())
    assert env.ledger._charges == charges
    assert env.count("settle") == count == 1
    # Local room left in an existing batch must not admit more paid work.
    with pytest.raises(LimitReached) as exc:
        await second.reserve_call(Amount())
    assert exc.value.limit == "usage_overflow"
    assert (await second.close()).overflow
    replayed = _replay(env)
    assert replayed._charges == charges
    await replayed.settle(late.id, Amount(usd=USD_LIMIT))
    assert replayed._charges == charges


async def test_pending_estimate_can_overflow_batch_at_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = await _env(Limits(max_usd=1))
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(tokens=10, usd=0.25))
    first = await t.reserve_call(Amount(usd=0.1))
    await t.reserve_call(Amount(tokens=2, usd=0.1))
    await t.settle_call(first, Amount(tokens=1, usd=USD_LIMIT))
    outcome = await t.close()
    assert outcome.overflow
    assert env.charged("g1/t/a") == Amount(tokens=3, usd=USD_LIMIT)
    assert await t.close() is outcome
    assert len(caplog.records) == 1


async def test_ordinary_close_has_stable_non_overflow_outcome(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    cid = await t.reserve_call(Amount(usd=0.1))
    settlement = await t.settle_call(cid, Amount(usd=0.01))
    assert not settlement.overflow
    assert await t.close() is settlement
    assert await t.close() is settlement


async def test_overflow_close_retries_failed_journal_once(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = await _env(Limits(max_usd=1))
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    first = await t.reserve_call(Amount(usd=0.1))
    # Leave a pending estimate so overflow is first discovered during close.
    await t.reserve_call(Amount(usd=0.1))
    await t.settle_call(first, Amount(usd=USD_LIMIT))
    append = env.journal.append_durable
    failed = False

    async def fail_once(event: EventDraft) -> None:
        nonlocal failed
        if event.type == "settle" and not failed:
            failed = True
            raise RuntimeError("journal unavailable")
        await append(event)

    monkeypatch.setattr(env.journal, "append_durable", fail_once)
    with pytest.raises(RuntimeError, match="journal unavailable"):
        await t.close()
    assert t.pending() == 0
    with pytest.raises(ValueError, match="closed"):
        await t.reserve_call(Amount())
    outcome = await t.close()
    assert outcome.overflow
    assert await t.close() is outcome
    assert env.charged("g1/t/a") == Amount(usd=USD_LIMIT)
    assert env.count("settle") == 1
    assert len(caplog.records) == 1


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("retry", ["close", "report"])
async def test_failed_overflow_report_can_be_committed_by_close(
    direct: bool,
    retry: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _env(Limits())
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    cid = await t.reserve_call(Amount(usd=0.5 if direct else 0.1))
    append = env.journal.append_durable
    failed = False

    async def fail_once(event: EventDraft) -> None:
        nonlocal failed
        if event.type == "settle" and not failed:
            failed = True
            raise RuntimeError("journal unavailable")
        await append(event)

    monkeypatch.setattr(env.journal, "append_durable", fail_once)
    with pytest.raises(RuntimeError, match="journal unavailable"):
        await t.settle_call(cid, Amount.model_construct(tokens=7, usd=USD_LIMIT + 1))
    assert env.ledger.run_exhausted()
    with pytest.raises(LimitReached) as exc:
        await env.ledger.reserve(":run", Amount(), kind="call", goal_id="")
    assert exc.value.limit == "usage_overflow"
    if retry == "report":
        assert (
            await t.settle_call(cid, Amount.model_construct(tokens=7, usd=USD_LIMIT + 1))
        ).overflow
    assert (await t.close()).overflow
    assert env.count("settle") == 1
    assert env.charged("g1/t/a") == Amount(tokens=7, usd=USD_LIMIT)
    assert _replay(env).run_exhausted()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("retry", ["close", "report"])
async def test_normalized_overflow_retries_without_invalid_amount(
    direct: bool, retry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = await _env(Limits())
    tranche = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    rid = await tranche.reserve_call(Amount(usd=0.5 if direct else 0.1))
    actual = Amount(tokens=7, usd=USD_LIMIT)
    append = env.journal.append_durable
    failed = False

    async def fail_once(event: EventDraft) -> None:
        nonlocal failed
        if event.type == "settle" and not failed:
            failed = True
            raise RuntimeError("journal unavailable")
        await append(event)

    monkeypatch.setattr(env.journal, "append_durable", fail_once)
    with pytest.raises(RuntimeError, match="journal unavailable"):
        await tranche._settle_call(rid, actual, overflow=True)
    assert env.ledger.run_exhausted()
    if retry == "report":
        assert (await tranche._settle_call(rid, actual, overflow=True)).overflow
    assert (await tranche.close()).overflow
    assert (await tranche.close()).overflow
    assert env.charged("g1/t/a") == actual
    assert env.count("settle") == 1
    assert _replay(env).run_exhausted()


async def test_overflow_marker_requires_normalized_limit(env: LedgerEnv) -> None:
    tranche = Tranche(env.ledger, "g1/t/a", "g1", size=Amount())
    rid = await tranche.reserve_call(Amount(usd=0.1))
    with pytest.raises(ValueError, match="supported USD limit"):
        await tranche._settle_call(rid, Amount(usd=0.01), overflow=True)
    assert tranche.pending() == 1
    assert env.count("settle") == 0
    assert not (await tranche.settle_call(rid, Amount(usd=0.01))).overflow
    assert env.charged("g1/t/a") == Amount(usd=0.01)


@pytest.mark.parametrize(
    "case", ["free credit fits", "tranche used up, ledger fits one tranche", "neither"]
)
async def test_can_cover_matches_reserve_call(case: str, tmp_path: Path) -> None:
    env = await _env(Limits(max_usd=1 if case != "neither" else 0.3))
    tranche = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(tokens=25_000, usd=0.25))
    await tranche.reserve_call(Amount(usd=0.1 if case == "free credit fits" else 0.25))
    est = Amount(usd=0.1)
    before = list(env.journal.events)
    assert tranche.can_cover(est) is (case != "neither")
    assert env.journal.events == before
    if case == "neither":
        with pytest.raises(LimitReached):
            await tranche.reserve_call(est)
    else:
        await tranche.reserve_call(est)
    await tranche.close()


@pytest.mark.parametrize(("tokens", "usd"), [(None, None), (100, None), (None, 0.1), (100, 0.1)])
@pytest.mark.parametrize("size", [Amount(), Amount(tokens=25_000, usd=0.25)])
async def test_can_cover_shrinks_batches_and_respects_uncapped_units(
    tokens: int | None,
    usd: float | None,
    size: Amount,
) -> None:
    env = await _env(Limits(max_tokens=tokens, max_usd=usd))
    tranche = Tranche(env.ledger, "g1/t/a", "g1", size=size)
    estimate = Amount(tokens=10, usd=0.01)
    assert tranche.can_cover(estimate)
    await tranche.reserve_call(estimate)
    await tranche.close()
    assert not tranche.can_cover(Amount())


async def test_can_cover_rejects_overflow_even_with_free_credit(env: LedgerEnv) -> None:
    tranche = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    await tranche.reserve_call(Amount(usd=0.01))
    other = Tranche(env.ledger, "g1/t/a", "g1", size=Amount())
    rid = await other.reserve_call(Amount(usd=0.01))
    await other._settle_call(rid, Amount(usd=USD_LIMIT), overflow=True)
    assert not tranche.can_cover(Amount())
    with pytest.raises(LimitReached, match="overflow"):
        await tranche.reserve_call(Amount())


async def test_cancelled_overflow_lock_wait_keeps_one_admission_latch(env: LedgerEnv) -> None:
    from swarm_arena._errors import _core_category, _SignalCategory

    tranche = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    rid = await tranche.reserve_call(Amount(usd=0.1))
    batch = tranche._calls[rid].batch
    assert batch is not None
    entry = env.ledger._entries[batch.id]
    await entry.lock.acquire()
    task = asyncio.create_task(tranche._settle_call(rid, Amount(usd=USD_LIMIT), overflow=True))
    try:
        await anyio.wait_all_tasks_blocked()
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        before = list(env.journal.events)
        assert not tranche.can_cover(Amount())
        with pytest.raises(LimitReached) as caught:
            await tranche.reserve_call(Amount())
        assert _core_category(caught.value) is _SignalCategory.TERMINAL
        assert env.journal.events == before
    finally:
        entry.lock.release()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert (await tranche.close()).overflow
    assert env.count("settle") == 1
    assert _replay(env).run_exhausted()
