"""Account for durable reservations, ordered funding, and history-only replay."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import anyio

from swarm_arena._agent.manager import LimitReached
from swarm_arena._budget.amount import USD_LIMIT, Amount, _usd_from_micro_units
from swarm_arena._budget.split import equal_shares, round_pool, weighted_shares
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.ids import IdGen
from swarm_arena._model.limits import Limits
from swarm_arena._model.run_config import RunConfig

ReservationKind = Literal["tool", "tranche", "call", "judge", "planner", "curator", "final"]


class Reservation(FrozenModel):
    """
    Identify a durable charge held before work is dispatched.

    Parameters
    ----------
    id : str
        Unique reservation identifier.
    holder : str
        Agent address, goal service address, or ``:run``.
    kind : str
        Tool, tranche, call, judge, planner, curator, or final reservation.
    amount : Amount
        Nonnegative requested charge.
    goal_id : str, default=""
        Associated goal, empty for run-level work.
    keep_charged : bool, default=False
        Retain the entire reservation even after reported settlement.

    Examples
    --------
    >>> Reservation(id="r1", holder=":run", kind="planner", amount=Amount()).id
    'r1'
    """

    id: str
    holder: str
    kind: ReservationKind
    amount: Amount
    goal_id: str = ""
    keep_charged: bool = False


@dataclass(frozen=True)
class _Balance:
    """Keep aggregate debt exact even when multiple bounded amounts exceed a cap."""

    tokens: int
    _micro_usd: int

    def __add__(self, other: "Amount | _Balance") -> "_Balance":
        """Add exact units without converting aggregate debt to float dollars."""
        return _Balance(self.tokens + other.tokens, self._micro_usd + other._micro_usd)

    def __sub__(self, other: "Amount | _Balance") -> "_Balance":
        """Subtract exact units without imposing an individual Amount's range."""
        return _Balance(self.tokens - other.tokens, self._micro_usd - other._micro_usd)


class _Pool:
    """Keep a funding source alive across round changes and continued goals."""

    def __init__(self, left: Amount | None = None, *, work: bool = False) -> None:
        """Initialize a bounded allocation with an unbounded integer debt balance."""
        left = Amount() if left is None else left
        self.left = _Balance(left.tokens, left._micro_usd)
        self.work = work
        self.ended = False


@dataclass
class _Entry:
    """Retain funding provenance so refunds return to the original sources."""

    reservation: Reservation
    funding: list[tuple[_Pool, Amount]]
    share: _Pool | None
    settled: bool = False
    lock: anyio.Lock = field(default_factory=anyio.Lock)


def _positive(amount: Amount | _Balance) -> Amount:
    """Clamp each exact unit independently to zero."""
    return Amount(
        tokens=max(0, amount.tokens), usd=_usd_from_micro_units(max(0, amount._micro_usd))
    )


def _minimum(a: Amount, b: Amount) -> Amount:
    """Take the minimum in integer token and micro-dollar units."""
    return Amount(
        tokens=min(a.tokens, b.tokens), usd=_usd_from_micro_units(min(a._micro_usd, b._micro_usd))
    )


def _nonnegative(amount: Amount) -> Amount:
    """Validate accounting inputs without exposing rejected values."""
    amount = Amount.model_validate(amount.model_dump())
    if amount.tokens < 0 or amount._micro_usd < 0:
        raise ValueError("budget amount must be nonnegative")
    return amount


class Ledger:
    """
    Allocate work and reserve pools and commit every spending reservation.

    Bookkeeping and journal admission share one synchronous ordering. A later
    reservation cannot return before preceding settlements are durable.
    Internal pool references preserve refund destinations across round and
    goal lifecycle changes.
    Uncapped units have zero available balances and are ignored at admission.
    Journaled usage overflow permanently exhausts every goal and the run,
    including unlimited budgets, and freezes later settlement charges.

    Parameters
    ----------
    limits : Limits
        Run caps; unset units are unlimited.
    reserve_fraction : float
        Fraction of each cap reserved for goal services and final slices.
    goal_weights : Mapping[str, float] or None
        Relative goal allocations; missing weights default to one.
    journal : JournalWriter
        Active durable journal writer.
    ids : IdGen
        Reservation identifier generator.

    Examples
    --------
    The inert journal below makes accidental journal access fail. Construction
    and pool allocation need no storage.

    >>> from typing import cast
    >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
    ...                 goal_weights=None, journal=cast(JournalWriter, object()),
    ...                 ids=IdGen("example", 1))
    """

    def __init__(
        self,
        limits: Limits,
        *,
        reserve_fraction: float,
        goal_weights: Mapping[str, float] | None,
        journal: JournalWriter,
        ids: IdGen,
    ) -> None:
        config = RunConfig(
            limits=limits, reserve_fraction=reserve_fraction, goal_weights=goal_weights
        )
        self._limits = config.limits
        self._fraction = Fraction(str(config.reserve_fraction))
        self._weights = dict(config.goal_weights or {})
        self._journal = journal
        self._ids = ids
        cap = Amount(tokens=limits.max_tokens or 0, usd=limits.max_usd or 0)
        self._total = _Balance(cap.tokens, cap._micro_usd)
        reserve = self._reserve_part(cap)
        self._reserve_pool = _Pool(reserve)
        self._work = self._total - reserve
        self._free = cap - reserve
        self._goals: dict[str, _Pool] = {}
        self._slices: dict[str, _Pool] = {}
        self._shares: dict[str, _Pool] = {}
        self._holder_goals: dict[str, str] = {}
        self._opened: set[str] = set()
        self._held = False
        self._entries: dict[str, _Entry] = {}
        self._charges: dict[str, Amount] = {}
        self._overflow_reservation: str | None = None

    @property
    def limits(self) -> Limits:
        """
        Return the current human-controlled limits.

        Returns
        -------
        Limits
            Current immutable limits, including any replayed cap increases.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> ledger.limits.max_usd
        100.0
        """
        return self._limits

    @classmethod
    def replay(cls, state: RunState, *, journal: JournalWriter, ids: IdGen) -> "Ledger":
        """
        Re-execute committed history without journaling or consulting later state.

        Parameters
        ----------
        state : RunState
            State containing the original run_config and ordered ledger events.
        journal : JournalWriter
            Writer attached for future live actions.
        ids : IdGen
            Generator for future reservations; replay never advances it.

        Returns
        -------
        Ledger
            Reconstructed balances with unsettled reservations charged in full.

        Raises
        ------
        ValueError
            If the history lacks configuration or contains malformed payloads.

        Examples
        --------
        >>> from typing import cast
        >>> from swarm_arena._model.canonical import utc_now
        >>> config = RunConfig(limits=Limits(max_usd=100))
        >>> event = Event(type="run_config", payload={"config": config.model_dump(mode="json")},
        ...               epoch=1, seq=1, ts=utc_now(), run_id="example", context_id="example")
        >>> state = RunState.from_events([event])
        >>> ledger = Ledger.replay(state, journal=cast(JournalWriter, object()),
        ...                        ids=IdGen("example", 1))
        >>> ledger.limits.max_usd
        100.0
        """
        events = sorted(state.ledger_events, key=lambda event: event.seq)
        try:
            if not events or events[0].type != EventType.RUN_CONFIG:
                raise ValueError("missing original run_config")
            config = RunConfig.model_validate(events[0].payload["config"])
            ledger = cls(
                config.limits,
                reserve_fraction=config.reserve_fraction,
                goal_weights=config.goal_weights,
                journal=journal,
                ids=ids,
            )
            for event in events[1:]:
                ledger._replay_event(event)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ValueError("invalid ledger replay payload") from None
        return ledger

    def _replay_event(self, event: Event) -> None:
        """Use the live synchronous bookkeeping without invoking journal methods."""
        payload = event.payload
        match event.type:
            case EventType.RUN_STARTED:
                self._hold(
                    {g: Amount.model_validate(a) for g, a in payload["final_slices"].items()}
                )
            case EventType.GOAL_STARTED:
                pool = payload.get("pool")
                self.open_goal(
                    event.goal_id, pool=None if pool is None else Amount.model_validate(pool)
                )
            case EventType.ROUND_STARTED:
                if not payload.get("converge", False):
                    self._restore_shares(
                        event.goal_id,
                        {h: Amount.model_validate(a) for h, a in payload["shares"].items()},
                    )
            case EventType.RESERVE:
                self._book(Reservation.model_validate(payload["reservation"]))
            case EventType.SETTLE:
                entry = self._entries[payload["id"]]
                if not entry.settled and self._overflow_reservation is None:
                    actual = _nonnegative(Amount.model_validate(payload["actual"]))
                    overflow = payload.get("usage_overflow", False)
                    target = (
                        actual
                        if overflow
                        else self._charge_for(entry, actual, payload["estimated"])
                    )
                    self._reconcile(entry, target)
                    if overflow:
                        self._overflow_reservation = entry.reservation.id
                    entry.settled = True
            case EventType.GOAL_FINISHED:
                self._end(event.goal_id)
            case EventType.LIMITS_CHANGED:
                self.change_limits(Limits.model_validate(payload["new"]))

    async def hold_final_slices(self, slices: Mapping[str, Amount]) -> None:
        """
        Hold final slices and allocate work over every running or queued goal.

        Parameters
        ----------
        slices : Mapping[str, Amount]
            Every goal identifier and its final reservation estimate.

        Raises
        ------
        ValueError
            If slices exceed the reserve or have already been held.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> import anyio
        >>> anyio.run(ledger.hold_final_slices, {"g1": Amount(usd=5), "g2": Amount(usd=5)})
        >>> ledger.remaining("g1/:final")
        Amount(tokens=0, usd=52.5)
        """
        self._hold(slices)

    def _hold(self, slices: Mapping[str, Amount]) -> None:
        """Initialize pools without an await, including during replay."""
        if self._held:
            raise ValueError("final reserve slices already held")
        validated = {g: self._capped(_nonnegative(a)) for g, a in slices.items()}
        # Compare incrementally so an excessive sum cannot exceed Amount's range.
        left = self._reserve_pool.left
        for amount in validated.values():
            if not self._fits(amount, _positive(left)):
                raise ValueError("final slices exceed the reserve")
            left -= amount
        self._goals = {g: _Pool(work=True) for g in validated}
        self._slices = {g: _Pool(a) for g, a in validated.items()}
        self._reserve_pool.left = left
        work, self._free = self._free, Amount()
        self._distribute(work)
        self._held = True

    def open_goal(self, goal_id: str, *, pool: Amount | None = None) -> None:
        """
        Start a goal or reopen an ended goal from the free work pool.

        Parameters
        ----------
        goal_id : str
            Registered goal identifier.
        pool : Amount or None, default=None
            Exact pool for a continued goal; None starts its initial lifecycle.

        Raises
        ------
        ValueError
            If the lifecycle is invalid or the requested pool does not fit.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> import anyio
        >>> anyio.run(ledger.hold_final_slices, {"g1": Amount(usd=5), "g2": Amount(usd=5)})
        >>> ledger.open_goal("g1")
        """
        current = self._goal(goal_id)
        if pool is None:
            if current.ended:
                raise ValueError(f"continued goal requires a pool: {goal_id}")
        else:
            amount = self._capped(_nonnegative(pool))
            if not current.ended or not self._fits(amount, self._free):
                raise ValueError(f"goal cannot reopen from free pool: {goal_id}")
            self._free -= amount
            self._goals[goal_id] = _Pool(amount, work=True)
        self._opened.add(goal_id)

    def free(self) -> Amount:
        """
        Return work budget held by no goal, including split remainders.

        Returns
        -------
        Amount
            Unallocated work pool available to continued goals.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> import anyio
        >>> anyio.run(ledger.hold_final_slices, {"g1": Amount(usd=5), "g2": Amount(usd=5)})
        >>> ledger.free()
        Amount(tokens=0, usd=0.0)
        """
        return self._free

    def change_limits(self, limits: Limits) -> None:
        """
        Apply raised caps and distribute added work over goals not ended.

        Parameters
        ----------
        limits : Limits
            New limits; capped units may increase or become unlimited.

        Raises
        ------
        ValueError
            If a cap decreases or an unlimited unit becomes capped.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> ledger.change_limits(Limits(max_usd=200))
        >>> ledger.limits.max_usd
        200.0
        """
        limits = Limits.model_validate(limits.model_dump())
        for name in ("max_tokens", "max_usd"):
            old, new = getattr(self._limits, name), getattr(limits, name)
            if new is not None and (old is None or new < old):
                raise ValueError(f"budget cap may only increase: {name}")
        added = Amount(
            tokens=0
            if limits.max_tokens is None
            else limits.max_tokens - (self._limits.max_tokens or 0),
            usd=0
            if limits.max_usd is None
            else _usd_from_micro_units(
                Amount(usd=limits.max_usd)._micro_usd
                - Amount(usd=self._limits.max_usd or 0)._micro_usd
            ),
        )
        reserve = self._reserve_part(added)
        work = added - reserve
        self._total += added
        self._work += work
        self._reserve_pool.left += reserve
        self._distribute(work)
        self._limits = limits

    def open_round(
        self, goal_id: str, round: int, holders: Sequence[str], rounds_left: int
    ) -> dict[str, Amount]:
        """
        Allocate equal agent shares of this round's available work pool.

        Parameters
        ----------
        goal_id : str
            Active goal identifier.
        round : int
            Ordinary round number.
        holders : Sequence[str]
            Distinct agent addresses belonging to this goal.
        rounds_left : int
            Remaining rounds including this one.

        Returns
        -------
        dict[str, Amount]
            Exact retained grants to journal as ``round_started.shares``.
            These are not clipped by the current global run balance.
            ``remaining(holder)`` computes the currently spendable balance.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> import anyio
        >>> anyio.run(ledger.hold_final_slices, {"g1": Amount(usd=5), "g2": Amount(usd=5)})
        >>> ledger.open_goal("g1")
        >>> ledger.open_round("g1", 1, ["g1/t/a"], rounds_left=2)
        {'g1/t/a': Amount(tokens=0, usd=21.25)}
        """
        goal = self._goal(goal_id)
        if goal_id not in self._opened or goal.ended or round < 1:
            raise ValueError(f"goal has no active round: {goal_id}")
        shares = equal_shares(round_pool(_positive(goal.left), rounds_left), holders)
        self._restore_shares(goal_id, shares)
        return {holder: self._capped(amount) for holder, amount in shares.items()}

    def _restore_shares(self, goal_id: str, shares: Mapping[str, Amount]) -> None:
        """Replace round grants exactly, without recomputing journaled shares."""
        self._goal(goal_id)
        validated: dict[str, _Pool] = {}
        for holder, amount in shares.items():
            if not holder.startswith(f"{goal_id}/") or len(holder.split("/")) != 3:
                raise ValueError(f"invalid agent holder: {holder}")
            validated[holder] = _Pool(self._capped(_nonnegative(amount)))
        self._shares = {h: p for h, p in self._shares.items() if self._holder_goals[h] != goal_id}
        self._shares.update(validated)
        self._holder_goals.update(dict.fromkeys(validated, goal_id))

    def remaining(self, holder: str) -> Amount:
        """
        Return available capped units under a holder's ordered funding rule.

        Parameters
        ----------
        holder : str
            Agent address, goal service address, or ``:run``.

        Returns
        -------
        Amount
            Spendable balance; uncapped units are zero and ignored at admission.
            An agent without a current share has no capped budget.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> import anyio
        >>> anyio.run(ledger.hold_final_slices, {"g1": Amount(usd=5), "g2": Amount(usd=5)})
        >>> ledger.open_goal("g1")
        >>> grants = ledger.open_round("g1", 1, ["g1/t/a", "g1/t/b"], rounds_left=2)
        >>> ledger.remaining("g1/t/a")
        Amount(tokens=0, usd=10.625)
        """
        if self._overflow_reservation is not None:
            return Amount()
        if holder not in self._shares and len(holder.split("/")) == 3:
            return Amount()
        pools, share = self._sources(holder)
        left = sum((_positive(p.left) for p in pools), Amount())
        if share is not None:
            left = _minimum(left, _positive(share.left))
        return self._capped(_minimum(left, _positive(self._total)))

    async def reserve(
        self,
        holder: str,
        amount: Amount,
        *,
        kind: ReservationKind,
        goal_id: str,
        keep_charged: bool = False,
    ) -> Reservation:
        """
        Debit synchronously, then commit before allowing paid work to begin.

        Parameters
        ----------
        holder : str
            Funding holder.
        amount : Amount
            Nonnegative requested reservation.
        kind : str
            Kind of paid work.
        goal_id : str
            Associated goal, empty for run-level work.
        keep_charged : bool, default=False
            Keep the full reservation charged on settlement.

        Returns
        -------
        Reservation
            Durably committed reservation.

        Raises
        ------
        LimitReached
            If any capped unit exceeds the holder or run budget, or usage
            overflow has permanently exhausted the run.
        ValueError
            If the amount or holder is invalid.

        Notes
        -----
        A failed durable append retains the debit conservatively.

        Examples
        --------
        >>> res = await ledger.reserve("g1/t/a", Amount(usd=1),
        ...                            kind="tool", goal_id="g1")  # doctest: +SKIP
        """
        self._check_usage_overflow()
        amount = _nonnegative(amount)
        if holder != ":run" and holder.split("/", 1)[0] != goal_id:
            raise ValueError(f"holder does not belong to goal: {holder}")
        if not self._fits(amount, self.remaining(holder)):
            raise LimitReached("budget", f"budget used up for {holder}")
        reservation = Reservation(
            id=self._ids.next("r"),
            holder=holder,
            kind=kind,
            amount=amount,
            goal_id=goal_id,
            keep_charged=keep_charged,
        )
        self._book(reservation)
        await self._journal.append_durable(
            EventDraft(
                type=EventType.RESERVE,
                goal_id=goal_id,
                payload={"reservation": reservation.model_dump()},
            )
        )
        return reservation

    def _book(self, reservation: Reservation) -> None:
        """Debit recorded funding without live admission checks during replay."""
        if reservation.id in self._entries:
            raise ValueError("reservation identifier already used")
        amount = self._capped(_nonnegative(reservation.amount))
        pools, share = self._sources(reservation.holder)
        funding: list[tuple[_Pool, Amount]] = []
        left = amount
        for i, pool in enumerate(pools):
            take = left if i == len(pools) - 1 else _minimum(left, _positive(pool.left))
            self._debit(pool, take)
            funding.append((pool, take))
            left -= take
        if share is not None:
            share.left -= amount
        self._total -= amount
        self._entries[reservation.id] = _Entry(reservation, funding, share)
        self._charges[reservation.id] = reservation.amount

    async def settle(self, reservation_id: str, actual: Amount, *, estimated: bool = False) -> None:
        """
        Durably settle once, returning unused funding to its original sources.

        Parameters
        ----------
        reservation_id : str
            Existing reservation identifier.
        actual : Amount
            Nonnegative reported usage.
        estimated : bool, default=False
            Keep the original reservation charged when actual usage is unknown.

        Notes
        -----
        Bookkeeping precedes the writer's synchronous queue admission, so live
        funding follows journal order even during concurrent calls. Any later
        reservation waits for a head covering both events before returning.
        Settlement is shielded after admission; a failed append restores any
        released charge conservatively. Concurrent settlements serialize by
        reservation identifier.
        After usage overflow, later settlements are no-ops.

        Examples
        --------
        >>> await ledger.settle(res.id, Amount(usd=0.4))  # doctest: +SKIP
        """
        await self._settle(reservation_id, actual, estimated=estimated)

    async def settle_overflow(self, reservation_id: str, actual: Amount) -> None:
        """
        Journal a range-limit usage charge and permanently exhaust the run.

        Parameters
        ----------
        reservation_id : str
            Reservation whose usage exceeded the supported USD range.
        actual : Amount
            Reported tokens and exactly ``USD_LIMIT`` dollars.

        Notes
        -----
        The settlement carries ``usage_overflow=True`` for history-only replay.
        All later reservations raise ``LimitReached("usage_overflow")``, even
        with unlimited caps. Later settlements are no-ops, retaining earlier
        charges. A failed durable append retains exhaustion conservatively;
        retrying this same overflow settlement can finish the durable write.

        Examples
        --------
        >>> from unittest.mock import AsyncMock
        >>> journal = AsyncMock(spec=JournalWriter)
        >>> ledger = Ledger(Limits(), reserve_fraction=0.15, goal_weights=None,
        ...                 journal=journal, ids=IdGen("example", 1))
        >>> async def example() -> bool:
        ...     res = await ledger.reserve(":run", Amount(usd=1), kind="call", goal_id="")
        ...     await ledger.settle_overflow(res.id, Amount(tokens=7, usd=USD_LIMIT))
        ...     return ledger.run_exhausted()
        >>> anyio.run(example)
        True
        """
        await self._settle(reservation_id, actual, usage_overflow=True)

    def _check_usage_overflow(self) -> None:
        """Reject local and durable admission after untrustworthy usage."""
        if self._overflow_reservation is not None:
            raise LimitReached("usage_overflow", "budget exhausted by usage overflow")

    async def _settle(
        self,
        reservation_id: str,
        actual: Amount,
        *,
        estimated: bool = False,
        usage_overflow: bool = False,
    ) -> None:
        """Apply ordinary or terminal settlement with identical durable ordering."""
        entry = self._entries[reservation_id]
        async with entry.lock:
            if entry.settled:
                return
            if self._overflow_reservation is not None and (
                self._overflow_reservation != reservation_id or not usage_overflow
            ):
                return
            actual = _nonnegative(actual)
            if usage_overflow and actual.usd != USD_LIMIT:
                raise ValueError("overflow settlement requires the supported USD limit")
            target = actual if usage_overflow else self._charge_for(entry, actual, estimated)
            current = self._charges[reservation_id]
            overspend = _positive(actual - entry.reservation.amount)
            draft = EventDraft(
                type=EventType.SETTLE,
                goal_id=entry.reservation.goal_id,
                payload={
                    "id": reservation_id,
                    "actual": actual.model_dump(),
                    "estimated": estimated,
                    "overspend": overspend.model_dump() if overspend != Amount() else None,
                },
            )
            if usage_overflow:
                draft.payload["usage_overflow"] = True
            self._reconcile(entry, target)
            if usage_overflow:
                self._overflow_reservation = reservation_id
            try:
                with anyio.CancelScope(shield=True):
                    await self._journal.append_durable(draft)
            except BaseException:
                if self._overflow_reservation in (None, reservation_id):
                    self._reconcile(entry, current + _positive(target - current))
                raise
            entry.settled = True

    def _charge_for(self, entry: _Entry, actual: Amount, estimated: bool) -> Amount:
        """Choose the authoritative charged amount independently of reporting."""
        return entry.reservation.amount if estimated or entry.reservation.keep_charged else actual

    def _reconcile(self, entry: _Entry, target: Amount) -> None:
        """Retain earlier sources first and refund later funding sources first."""
        left = self._capped(target)
        extra_left = _positive(left - self._capped(self._charges[entry.reservation.id]))
        updated: list[tuple[_Pool, Amount]] = []
        for i, (pool, charged) in enumerate(entry.funding):
            if i == len(entry.funding) - 1:
                take = left
            else:
                extra = _minimum(extra_left, _positive(pool.left))
                take = _minimum(left, charged + extra)
                extra_left -= extra
            self._debit(pool, take - charged)
            updated.append((pool, take))
            left -= take
        delta = self._capped(target - self._charges[entry.reservation.id])
        if entry.share is not None:
            entry.share.left -= delta
        self._total -= delta
        self._charges[entry.reservation.id] = target
        entry.funding = updated

    async def end_goal(self, goal_id: str) -> None:
        """
        Return unused work to running and queued goals, or to the free pool.

        Parameters
        ----------
        goal_id : str
            Goal whose engine-written goal_finished event has committed.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> import anyio
        >>> anyio.run(ledger.hold_final_slices, {"g1": Amount(usd=5), "g2": Amount(usd=5)})
        >>> ledger.open_goal("g1")
        >>> anyio.run(ledger.end_goal, "g1")
        >>> anyio.run(ledger.end_goal, "g2")
        >>> ledger.free()
        Amount(tokens=0, usd=85.0)
        """
        self._end(goal_id)

    def _end(self, goal_id: str) -> None:
        """End a pool once while retaining references held by open reservations."""
        pool = self._goal(goal_id)
        if pool.ended:
            return
        returned = _positive(pool.left)
        pool.left -= returned
        pool.ended = True
        self._restore_shares(goal_id, {})
        self._distribute(returned)

    def run_exhausted(self) -> bool:
        """
        Check capped work-pool exhaustion or terminal usage overflow.

        Returns
        -------
        bool
            True after usage overflow or a nonpositive capped work-pool unit.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> ledger.run_exhausted()
        False
        """
        return self._overflow_reservation is not None or self._exhausted(self._work)

    def goal_exhausted(self, goal_id: str) -> bool:
        """
        Check a goal's work pool including agent charges and open reservations.

        Parameters
        ----------
        goal_id : str
            Registered goal identifier.

        Returns
        -------
        bool
            True after run-wide usage overflow or a nonpositive capped unit.

        Examples
        --------
        >>> from typing import cast
        >>> ledger = Ledger(Limits(max_usd=100), reserve_fraction=0.15,
        ...                 goal_weights=None, journal=cast(JournalWriter, object()),
        ...                 ids=IdGen("example", 1))
        >>> import anyio
        >>> anyio.run(ledger.hold_final_slices, {"g1": Amount(usd=5), "g2": Amount(usd=5)})
        >>> ledger.goal_exhausted("g1")
        False
        """
        goal = self._goal(goal_id)
        return self._overflow_reservation is not None or self._exhausted(goal.left)

    def _exhausted(self, amount: _Balance) -> bool:
        """Inspect exact units, ignoring unlimited caps."""
        return (self._limits.max_tokens is not None and amount.tokens <= 0) or (
            self._limits.max_usd is not None and amount._micro_usd <= 0
        )

    def _capped(self, amount: Amount) -> Amount:
        """Omit unlimited units from finite allocation pools."""
        return Amount(
            tokens=amount.tokens if self._limits.max_tokens is not None else 0,
            usd=amount.usd if self._limits.max_usd is not None else 0,
        )

    def _fits(self, amount: Amount, left: Amount) -> bool:
        """Compare only capped units using Amount's exact comparisons."""
        return amount.fits_in(
            left,
            cap_tokens=self._limits.max_tokens is not None,
            cap_usd=self._limits.max_usd is not None,
        )

    def _reserve_part(self, amount: Amount) -> Amount:
        """Floor the reserve fraction in integer units."""
        return Amount(
            tokens=amount.tokens * self._fraction.numerator // self._fraction.denominator,
            usd=_usd_from_micro_units(
                amount._micro_usd * self._fraction.numerator // self._fraction.denominator
            ),
        )

    def _goal(self, goal_id: str) -> _Pool:
        """Look up a registered work pool with an identifier-only failure."""
        if goal_id not in self._goals:
            raise ValueError(f"unknown goal: {goal_id}")
        return self._goals[goal_id]

    def _sources(self, holder: str) -> tuple[list[_Pool], _Pool | None]:
        """Resolve ordered funding sources and an optional agent round ceiling."""
        if holder == ":run":
            return [self._reserve_pool], None
        goal_id, _, suffix = holder.partition("/")
        goal = self._goal(goal_id)
        if suffix == ":final":
            return [self._slices[goal_id], self._reserve_pool, goal], None
        if suffix in {":judges", ":planner", ":curator"}:
            return [self._reserve_pool, goal], None
        if holder in self._shares and not goal.ended:
            return [goal], self._shares[holder]
        raise ValueError(f"unknown budget holder: {holder}")

    def _debit(self, pool: _Pool, amount: Amount) -> None:
        """Apply a signed charge and redistribute any late refund to ended work."""
        pool.left -= amount
        if pool.work:
            self._work -= amount
            if pool.ended:
                returned = _positive(pool.left)
                pool.left -= returned
                self._distribute(returned)

    def _distribute(self, amount: Amount) -> None:
        """Split available work while keeping every rounding remainder free."""
        weights = {
            g: self._weights.get(g, 1.0) for g, pool in self._goals.items() if not pool.ended
        }
        shares = weighted_shares(amount, weights)
        for goal_id, share in shares.items():
            self._goals[goal_id].left += share
        self._free += amount - sum(shares.values(), Amount())
