from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

import anyio
import anyio.lowlevel
import pytest

from swarm_arena import Limits, RunConfig
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger
from swarm_arena._journal.event import Event, EventDraft
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.ids import IdGen
from swarm_arena.manager import LimitReached


class MemoryJournal:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def append_durable(self, draft: EventDraft) -> Event:
        event = Event(
            **draft.model_dump(),
            epoch=1,
            seq=len(self.events) + 1,
            ts=utc_now(),
            run_id="oracle",
            context_id="oracle",
        )
        # Exercise the serialized Event boundary; this journal has no I/O.
        self.events.append(Event.model_validate_json(event.model_dump_json()))
        await anyio.lowlevel.checkpoint()
        return event


@dataclass
class Entry:
    sources: list[str]
    funding: list[int]
    share: str | None
    original: int
    charge: int
    keep: bool
    settled: bool = False


class Oracle:
    # This oracle stores integers only, separately for each capped unit.
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.reserve = cap * 15 // 100
        self.workcap = cap - self.reserve
        self.pool: dict[str, int] = {"reserve": self.reserve}
        self.workids: set[str] = set()
        self.goals: dict[str, str] = {}
        self.ended: set[str] = set()
        self.shares: dict[str, str] = {}
        self.shareleft: dict[str, int] = {}
        self.entries: dict[str, Entry] = {}
        self.weights = {"g1": 1, "g2": 2, "g3": 3}
        self.free = self.workcap
        self.counter = 0

    def start(self) -> None:
        for goal in self.weights:
            name = f"{goal}@0"
            self.goals[goal] = name
            self.pool[name] = 0
            self.workids.add(name)
            self.pool[goal + "/slice"] = 5
            self.pool["reserve"] -= 5
        amount, self.free = self.free, 0
        self.distribute(amount)

    def distribute(self, amount: int) -> None:
        active = [g for g in self.goals if g not in self.ended]
        denominator = sum(self.weights[g] for g in active)
        allocations = {g: amount * self.weights[g] // denominator for g in active}
        for g, value in allocations.items():
            self.pool[self.goals[g]] += value
        self.free += amount - sum(allocations.values())

    def sources(self, holder: str) -> tuple[list[str], str | None]:
        if holder == ":run":
            return ["reserve"], None
        goal, suffix = holder.split("/", 1)
        if suffix == ":final":
            return [goal + "/slice", "reserve", self.goals[goal]], None
        if suffix.startswith(":"):
            return ["reserve", self.goals[goal]], None
        return [self.goals[goal]], self.shares.get(holder)

    def remaining(self, holder: str) -> int:
        if holder.count("/") == 2 and holder not in self.shares:
            return 0
        sources, share = self.sources(holder)
        available = sum(max(0, self.pool[s]) for s in sources)
        if share is not None:
            available = min(available, max(0, self.shareleft[share]))
        return min(available, max(0, self.cap - sum(e.charge for e in self.entries.values())))

    def newround(self, goal: str, holders: list[str], rounds: int) -> None:
        self.shares = {h: s for h, s in self.shares.items() if not h.startswith(goal + "/")}
        share = max(0, self.pool[self.goals[goal]]) // rounds // len(holders)
        for holder in holders:
            self.counter += 1
            name = f"share-{self.counter}"
            self.shares[holder] = name
            self.shareleft[name] = share

    def debit(self, source: str, amount: int) -> None:
        self.pool[source] -= amount
        if source in self.workids and source not in [
            self.goals[g] for g in self.goals if g not in self.ended
        ]:
            refund = max(0, self.pool[source])
            self.pool[source] -= refund
            self.distribute(refund)

    def reserve_entry(self, ident: str, holder: str, value: int, keep: bool) -> None:
        sources, share = self.sources(holder)
        remaining = value
        funding: list[int] = []
        for index, source in enumerate(sources):
            part = (
                remaining
                if index == len(sources) - 1
                else min(remaining, max(0, self.pool[source]))
            )
            funding.append(part)
            self.debit(source, part)
            remaining -= part
        if share is not None:
            self.shareleft[share] -= value
        self.entries[ident] = Entry(sources, funding, share, value, value, keep)

    def settle(self, ident: str, actual: int, estimated: bool) -> None:
        entry = self.entries[ident]
        if entry.settled:
            return
        target = entry.original if entry.keep or estimated else actual
        newfunding = entry.funding.copy()
        difference = target - entry.charge
        if difference >= 0:
            unassigned = difference
            for index, source in enumerate(entry.sources):
                part = (
                    unassigned
                    if index == len(entry.sources) - 1
                    else min(unassigned, max(0, self.pool[source]))
                )
                newfunding[index] += part
                unassigned -= part
        else:
            refund = -difference
            for index in reversed(range(len(entry.sources))):
                part = min(refund, newfunding[index])
                newfunding[index] -= part
                refund -= part
        for source, old, new in zip(entry.sources, entry.funding, newfunding, strict=True):
            self.debit(source, new - old)
        if entry.share is not None:
            self.shareleft[entry.share] -= difference
        entry.funding = newfunding
        entry.charge = target
        entry.settled = True

    def end(self, goal: str) -> None:
        source = self.goals[goal]
        refund = max(0, self.pool[source])
        self.pool[source] -= refund
        self.ended.add(goal)
        self.shares = {h: s for h, s in self.shares.items() if not h.startswith(goal + "/")}
        self.distribute(refund)

    def reopen(self, goal: str, amount: int) -> None:
        self.counter += 1
        source = f"{goal}@{self.counter}"
        self.goals[goal] = source
        self.pool[source] = amount
        self.workids.add(source)
        self.free -= amount
        self.ended.remove(goal)

    def raise_cap(self, new: int) -> None:
        delta = new - self.cap
        reserved = delta * 15 // 100
        self.pool["reserve"] += reserved
        self.workcap += delta - reserved
        self.distribute(delta - reserved)
        self.cap = new

    def check(self) -> None:
        assert self.cap == self.free + sum(self.pool.values()) + sum(
            e.charge for e in self.entries.values()
        )


HOLDERS = [":run"] + [
    f"{g}/{suffix}" for g in ("g1", "g2", "g3") for suffix in (":final", ":judges", "t/a", "t/b")
]


def amount(a: int, b: int) -> Amount:
    return Amount(tokens=a, usd=b / 1_000_000)


def units(value: Amount) -> tuple[int, int]:
    return value.tokens, value._micro_usd


def check_live(ledger: Ledger, oracles: tuple[Oracle, Oracle]) -> None:
    for oracle in oracles:
        oracle.check()
    for holder in HOLDERS:
        assert units(ledger.remaining(holder)) == tuple(o.remaining(holder) for o in oracles), (
            holder
        )
    assert units(ledger.free()) == tuple(o.free for o in oracles)
    assert (ledger._reserve_pool.left.tokens, ledger._reserve_pool.left._micro_usd) == tuple(
        o.pool["reserve"] for o in oracles
    )
    assert (ledger._work.tokens, ledger._work._micro_usd) == tuple(
        o.free + sum(o.pool[s] for s in o.workids) for o in oracles
    )
    assert set(ledger._shares) == set(oracles[0].shares)
    for holder, share in ledger._shares.items():
        assert (share.left.tokens, share.left._micro_usd) == tuple(
            o.shareleft[o.shares[holder]] for o in oracles
        ), holder
    for ident, charge in ledger._charges.items():
        assert units(charge) == tuple(o.entries[ident].charge for o in oracles)


@pytest.mark.parametrize("seed", range(20))
async def test_randomized_integer_oracle_replay(seed: int) -> None:
    rng = random.Random(seed)
    journal = MemoryJournal()
    ids = IdGen("oracle", 1)
    oracles = (Oracle(1013), Oracle(1001))
    limits = Limits(max_tokens=1013, max_usd=0.001001)
    config = RunConfig(limits=limits, goal_weights={"g1": 1, "g2": 2, "g3": 3})
    ledger = Ledger(
        limits,
        reserve_fraction=0.15,
        goal_weights=config.goal_weights,
        journal=cast(JournalWriter, journal),
        ids=ids,
    )

    async def emit(kind: str, payload: dict[str, Any], goal: str = "") -> None:
        await journal.append_durable(EventDraft(type=kind, payload=payload, goal_id=goal))

    await emit("run_config", {"config": config.model_dump(mode="json")})
    await ledger.hold_final_slices({g: amount(5, 5) for g in ("g1", "g2", "g3")})
    await emit(
        "run_started", {"final_slices": {g: amount(5, 5).model_dump() for g in ("g1", "g2", "g3")}}
    )
    for oracle in oracles:
        oracle.start()
    for goal in ("g1", "g2", "g3"):
        ledger.open_goal(goal)
        await emit("goal_started", {}, goal)
    replay_rng = random.Random(seed + 1000)
    checkpoints = 0
    rounds = dict.fromkeys(("g1", "g2", "g3"), 0)
    for step in range(180):
        operation = rng.choices(
            ["reserve", "settle", "round", "end", "reopen", "raise"], [36, 26, 15, 6, 7, 10]
        )[0]
        active = [g for g in rounds if g not in oracles[0].ended]
        if operation == "reserve":
            holder = rng.choice(HOLDERS)
            a, b = (o.remaining(holder) for o in oracles)
            want = amount(rng.randint(0, a + 20), rng.randint(0, b + 20))
            admitted = want.tokens <= a and want._micro_usd <= b
            # Ended agents and absent grants are invalid even for zero amounts.
            if holder.count("/") == 2 and holder not in oracles[0].shares:
                continue
            keep = rng.random() < 0.15
            try:
                reservation = await ledger.reserve(
                    holder,
                    want,
                    kind="tool",
                    goal_id="" if holder == ":run" else holder.split("/")[0],
                    keep_charged=keep,
                )
            except LimitReached:
                assert not admitted
            else:
                assert admitted, (seed, step, "overdraw")
                for oracle, value in zip(oracles, units(want), strict=True):
                    oracle.reserve_entry(reservation.id, holder, value, keep)
        elif operation == "settle" and oracles[0].entries:
            ident = rng.choice(list(oracles[0].entries))
            original = tuple(o.entries[ident].original for o in oracles)
            values = tuple(rng.randint(0, v * 2 + 80) for v in original)
            estimated = rng.random() < 0.15
            await ledger.settle(ident, amount(*values), estimated=estimated)
            for oracle, value in zip(oracles, values, strict=True):
                oracle.settle(ident, value, estimated)
        elif operation == "round" and active:
            goal = rng.choice(active)
            holders = [goal + "/t/a", goal + "/t/b"][: rng.randint(1, 2)]
            remaining_rounds = rng.randint(1, 3)
            rounds[goal] += 1
            grants = ledger.open_round(goal, rounds[goal], holders, remaining_rounds)
            for oracle in oracles:
                oracle.newround(goal, holders, remaining_rounds)
            await emit(
                "round_started",
                {
                    "round": rounds[goal],
                    "shares": {h: grant.model_dump() for h, grant in grants.items()},
                },
                goal,
            )
        elif operation == "end" and active:
            goal = rng.choice(active)
            await emit("goal_finished", {"reason": "stop"}, goal)
            await ledger.end_goal(goal)
            for oracle in oracles:
                oracle.end(goal)
        elif operation == "reopen" and oracles[0].ended:
            goal = rng.choice(sorted(oracles[0].ended))
            values = tuple(o.free // 2 for o in oracles)
            ledger.open_goal(goal, pool=amount(*values))
            for oracle, value in zip(oracles, values, strict=True):
                oracle.reopen(goal, value)
            await emit("goal_started", {"pool": amount(*values).model_dump()}, goal)
        elif operation == "raise":
            values = tuple(o.cap + rng.randint(1, 90) for o in oracles)
            old = ledger.limits
            new = Limits(max_tokens=values[0], max_usd=values[1] / 1_000_000)
            ledger.change_limits(new)
            for oracle, value in zip(oracles, values, strict=True):
                oracle.raise_cap(value)
            await emit("limits_changed", {"old": old.model_dump(), "new": new.model_dump()})
        check_live(ledger, oracles)
        if replay_rng.random() < 0.35 or step == 179:
            replay = Ledger.replay(
                RunState.from_events(journal.events),
                journal=cast(JournalWriter, journal),
                ids=ids,
            )
            check_live(replay, oracles)
            assert set(replay._shares) == set(ledger._shares)
            for holder in HOLDERS:
                assert replay.remaining(holder) == ledger.remaining(holder), (seed, step, holder)
                if holder in ledger._shares:
                    assert replay._shares[holder].left == ledger._shares[holder].left
            checkpoints += 1
    assert checkpoints > 0
