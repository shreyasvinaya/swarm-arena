"""Reserve durable batches and account for individual calls locally."""

import logging
from dataclasses import dataclass, field

import anyio
import anyio.lowlevel

from swarm_arena._agent.manager import LimitReached
from swarm_arena._budget.amount import USD_LIMIT, Amount, _usd_from_micro_units
from swarm_arena._budget.ledger import Ledger, _nonnegative

_logger = logging.getLogger(__name__)
_USD_LIMIT_MICROS = Amount(usd=USD_LIMIT)._micro_usd


@dataclass(frozen=True, kw_only=True)
class TrancheOutcome:
    """
    Flag conservative range-limit charging for downstream journal producers.

    Parameters
    ----------
    overflow : bool, default=False
        Whether any reported call or batch usage exceeded the supported range.

    Examples
    --------
    >>> TrancheOutcome().overflow
    False
    """

    overflow: bool = False


@dataclass
class _Batch:
    """Retain a committed tranche and exact locally available units."""

    id: str
    tokens: int
    micros: int
    calls: list[str] = field(default_factory=list)
    settled: bool = False
    overflow: bool = False


@dataclass
class _Call:
    """Track either a local call or an independent durable reservation."""

    estimate: Amount
    batch: _Batch | None
    actual: Amount | None = None
    overflow: bool = False
    settled: bool = False


class Tranche:
    """
    Serialize call admission against committed batches for one budget holder.

    Parameters
    ----------
    ledger : Ledger
        Run's durable reservation ledger.
    holder : str
        Agent or goal service funding address.
    goal_id : str
        Associated goal identifier.
    size : Amount
        Nonnegative target batch size. Only capped units shrink to the ledger's
        remaining balance; an oversized call reserves independently.

    Notes
    -----
    Calls use unique local identifiers within a committed tranche. Their
    settlements update local room; closing settles every batch using reported
    actuals and the full estimates of calls still pending. Late settlements
    after closing are ignored, retaining conservative charges.
    New reservations are shielded through durability and local registration.
    Cancellation before a call identifier is returned leaves its unused
    funding tracked for zero-usage settlement at close.

    Examples
    --------
    >>> from unittest.mock import AsyncMock
    >>> from swarm_arena._journal.writer import JournalWriter
    >>> from swarm_arena._model.ids import IdGen
    >>> from swarm_arena._model.limits import Limits
    >>> journal = AsyncMock(spec=JournalWriter)
    >>> ledger = Ledger(Limits(), reserve_fraction=0.15, goal_weights=None,
    ...                 journal=journal, ids=IdGen("example", 1))
    >>> async def example() -> tuple[int, Amount]:
    ...     await ledger.hold_final_slices({"g1": Amount()})
    ...     ledger.open_goal("g1")
    ...     ledger.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
    ...     tranche = Tranche(ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    ...     cid = await tranche.reserve_call(Amount(usd=0.1))
    ...     assert tranche.pending() == 1
    ...     outcome = await tranche.settle_call(cid, Amount(usd=0.01))
    ...     assert not outcome.overflow
    ...     outcome = await tranche.close()
    ...     event = journal.append_durable.call_args.args[0]
    ...     return tranche.pending(), Amount.model_validate(event.payload["actual"])
    >>> anyio.run(example)
    (0, Amount(tokens=0, usd=0.01))
    """

    def __init__(self, ledger: Ledger, holder: str, goal_id: str, *, size: Amount) -> None:
        """Initialize independent call bookkeeping and its admission lock."""
        self._ledger = ledger
        self._holder = holder
        self._goal_id = goal_id
        self._size = _nonnegative(size)
        self._lock = anyio.Lock()
        self._batches: list[_Batch] = []
        self._calls: dict[str, _Call] = {}
        self._closed = False
        self._outcome = TrancheOutcome()

    def _reported_amount(self, tokens: int, micros: int, *, overflow: bool = False) -> Amount:
        """Contain reported usage in exact units, explicitly flagging overflow."""
        if overflow or micros > _USD_LIMIT_MICROS:
            if not self._outcome.overflow:
                self._outcome = TrancheOutcome(overflow=True)
                _logger.warning(
                    "reported usage exceeds the supported range; charged as the range limit"
                )
            micros = _USD_LIMIT_MICROS
        return _nonnegative(Amount(tokens=tokens, usd=_usd_from_micro_units(micros)))

    async def reserve_call(self, estimate: Amount) -> str:
        """
        Admit a call locally or commit a sufficient new reservation first.

        Parameters
        ----------
        estimate : Amount
            Nonnegative maximum call usage.

        Returns
        -------
        str
            Unique identifier to pass to ``settle_call``.

        Raises
        ------
        LimitReached
            If the capped remaining budget cannot cover the call.
        ValueError
            If the tranche is closed or the amount is invalid.

        Examples
        --------
        >>> from unittest.mock import AsyncMock
        >>> from swarm_arena._journal.writer import JournalWriter
        >>> from swarm_arena._model.ids import IdGen
        >>> from swarm_arena._model.limits import Limits
        >>> journal = AsyncMock(spec=JournalWriter)
        >>> ledger = Ledger(Limits(), reserve_fraction=0.15, goal_weights=None,
        ...                 journal=journal, ids=IdGen("example", 1))
        >>> async def example() -> tuple[int, Amount]:
        ...     await ledger.hold_final_slices({"g1": Amount()})
        ...     ledger.open_goal("g1")
        ...     ledger.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
        ...     tranche = Tranche(ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
        ...     cid = await tranche.reserve_call(Amount(usd=0.05))
        ...     assert tranche.pending() == 1
        ...     outcome = await tranche.close()
        ...     event = journal.append_durable.call_args.args[0]
        ...     return tranche.pending(), Amount.model_validate(event.payload["actual"])
        >>> anyio.run(example)
        (0, Amount(tokens=0, usd=0.05))
        """
        async with self._lock:
            if self._closed:
                raise ValueError("tranche is closed")
            self._ledger._check_usage_overflow()
            estimate = _nonnegative(estimate)
            if not estimate.fits_in(self._size, cap_tokens=True, cap_usd=True):
                with anyio.CancelScope(shield=True):
                    res = await self._ledger.reserve(
                        self._holder, estimate, kind="call", goal_id=self._goal_id
                    )
                    call = self._calls[res.id] = _Call(estimate, None, actual=Amount())
                await anyio.lowlevel.checkpoint_if_cancelled()
                call.actual = None
                return res.id
            batch = next(
                (
                    b
                    for b in reversed(self._batches)
                    if b.tokens >= estimate.tokens and b.micros >= estimate._micro_usd
                ),
                None,
            )
            if batch is None:
                remaining = self._ledger.remaining(self._holder)
                limits = self._ledger.limits
                size = Amount(
                    tokens=min(self._size.tokens, remaining.tokens)
                    if limits.max_tokens is not None
                    else self._size.tokens,
                    usd=_usd_from_micro_units(min(self._size._micro_usd, remaining._micro_usd))
                    if limits.max_usd is not None
                    else self._size.usd,
                )
                if not estimate.fits_in(size, cap_tokens=True, cap_usd=True):
                    raise LimitReached("budget", f"budget used up for {self._holder}")
                with anyio.CancelScope(shield=True):
                    res = await self._ledger.reserve(
                        self._holder, size, kind="tranche", goal_id=self._goal_id
                    )
                    batch = _Batch(res.id, size.tokens, size._micro_usd)
                    self._batches.append(batch)
                await anyio.lowlevel.checkpoint_if_cancelled()
            assert batch is not None
            cid = f"{batch.id}/call-{len(batch.calls) + 1}"
            batch.tokens -= estimate.tokens
            batch.micros -= estimate._micro_usd
            batch.calls.append(cid)
            self._calls[cid] = _Call(estimate, batch)
            return cid

    async def settle_call(self, reservation_id: str, actual: Amount) -> TrancheOutcome:
        """
        Settle an admitted call once, ignoring arrivals after close.

        Parameters
        ----------
        reservation_id : str
            Identifier returned by ``reserve_call``.
        actual : Amount
            Nonnegative reported usage, including any overspend.

        Returns
        -------
        TrancheOutcome
            Cumulative range-overflow flag, available for caller journaling.

        Notes
        -----
        An over-range report is explicitly charged at ``USD_LIMIT`` with its
        tokens preserved. One value-free warning is emitted per tranche.
        Normal Amount construction still rejects over-range estimates.

        Examples
        --------
        >>> from unittest.mock import AsyncMock
        >>> from swarm_arena._journal.writer import JournalWriter
        >>> from swarm_arena._model.ids import IdGen
        >>> from swarm_arena._model.limits import Limits
        >>> journal = AsyncMock(spec=JournalWriter)
        >>> ledger = Ledger(Limits(), reserve_fraction=0.15, goal_weights=None,
        ...                 journal=journal, ids=IdGen("example", 1))
        >>> async def example() -> tuple[int, Amount]:
        ...     await ledger.hold_final_slices({"g1": Amount()})
        ...     ledger.open_goal("g1")
        ...     ledger.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
        ...     tranche = Tranche(ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
        ...     cid = await tranche.reserve_call(Amount(usd=0.05))
        ...     outcome = await tranche.settle_call(cid, Amount(usd=0.01))
        ...     assert tranche.pending() == 0 and not outcome.overflow
        ...     outcome = await tranche.close()
        ...     event = journal.append_durable.call_args.args[0]
        ...     return tranche.pending(), Amount.model_validate(event.payload["actual"])
        >>> anyio.run(example)
        (0, Amount(tokens=0, usd=0.01))
        """
        async with self._lock:
            if self._closed:
                return self._outcome
            call = self._calls[reservation_id]
            if call.actual is not None:
                if call.batch is None:
                    await self._settle_direct(reservation_id, call)
                elif call.batch.overflow and not call.batch.settled:
                    await self._settle_batch(call.batch)
                return self._outcome
            overflow = actual.usd > USD_LIMIT
            actual = self._reported_amount(actual.tokens, actual._micro_usd, overflow=overflow)
            call.actual = actual
            call.overflow = overflow
            if call.batch is None:
                await self._settle_direct(reservation_id, call)
            else:
                call.batch.tokens += call.estimate.tokens - actual.tokens
                call.batch.micros += call.estimate._micro_usd - actual._micro_usd
            if call.batch is not None:
                batch = call.batch
                reported_micros = sum(
                    c.actual._micro_usd
                    for cid in batch.calls
                    if (c := self._calls[cid]).actual is not None
                )
                if overflow or reported_micros > _USD_LIMIT_MICROS:
                    batch.overflow = True
                    await self._settle_batch(batch)
            return self._outcome

    async def _settle_direct(self, reservation_id: str, call: _Call) -> None:
        """Retry durable direct-call settlement without forgetting reported usage."""
        if call.settled:
            return
        actual = call.actual or call.estimate
        if call.overflow:
            await self._ledger.settle_overflow(reservation_id, actual)
        else:
            await self._ledger.settle(reservation_id, actual, estimated=call.actual is None)
        call.actual = actual
        call.settled = True

    async def _settle_batch(self, batch: _Batch) -> None:
        """Commit batch usage once, including a terminal overflow mark."""
        if batch.settled:
            return
        amounts = [self._calls[c].actual or self._calls[c].estimate for c in batch.calls]
        micros = sum(a._micro_usd for a in amounts)
        batch.overflow = batch.overflow or micros > _USD_LIMIT_MICROS
        actual = self._reported_amount(
            sum(a.tokens for a in amounts), micros, overflow=batch.overflow
        )
        if batch.overflow:
            await self._ledger.settle_overflow(batch.id, actual)
        else:
            await self._ledger.settle(batch.id, actual)
        batch.settled = True

    def pending(self) -> int:
        """
        Count admitted calls whose usage has not arrived before close.

        Returns
        -------
        int
            Number of unsettled calls; zero after closing.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> Tranche(Mock(spec=Ledger), "g1/t/a", "g1", size=Amount()).pending()
        0
        """
        return 0 if self._closed else sum(c.actual is None for c in self._calls.values())

    async def close(self) -> TrancheOutcome:
        """
        Return unused batch funding and charge pending calls at their estimates.

        Returns
        -------
        TrancheOutcome
            Stable terminal outcome, including any conservative range charge.

        Notes
        -----
        Repeated closes are idempotent. If a durable settlement fails, another
        close retries the remaining settlements without reopening admission.
        Reservations acquired by cancelled calls whose identifiers were never
        returned have zero usage; their unused funding is returned here.
        Batch usage is summed in integer micro-dollars. A total above
        ``USD_LIMIT`` charges exactly that limit, preserving reported tokens
        (which have no upper bound), and flags overflow for caller journaling.
        The journaled overflow settlement permanently exhausts the entire run,
        including unlimited budgets. Later ledger settlements retain their
        existing charges without further arithmetic.

        Examples
        --------
        >>> from unittest.mock import AsyncMock
        >>> from swarm_arena._journal.writer import JournalWriter
        >>> from swarm_arena._model.ids import IdGen
        >>> from swarm_arena._model.limits import Limits
        >>> journal = AsyncMock(spec=JournalWriter)
        >>> ledger = Ledger(Limits(), reserve_fraction=0.15, goal_weights=None,
        ...                 journal=journal, ids=IdGen("example", 1))
        >>> async def example() -> tuple[int, Amount]:
        ...     await ledger.hold_final_slices({"g1": Amount()})
        ...     ledger.open_goal("g1")
        ...     ledger.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
        ...     tranche = Tranche(ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
        ...     cid = await tranche.reserve_call(Amount(usd=0.05))
        ...     pending = await tranche.reserve_call(Amount(usd=0.05))
        ...     outcome = await tranche.settle_call(cid, Amount(usd=0.01))
        ...     assert tranche.pending() == 1
        ...     outcome = await tranche.close()
        ...     assert await tranche.close() is outcome
        ...     event = journal.append_durable.call_args.args[0]
        ...     return tranche.pending(), Amount.model_validate(event.payload["actual"])
        >>> anyio.run(example)
        (0, Amount(tokens=0, usd=0.06))
        """
        async with self._lock:
            self._closed = True
            for batch in self._batches:
                await self._settle_batch(batch)
            for cid, call in self._calls.items():
                if call.batch is None:
                    await self._settle_direct(cid, call)
            return self._outcome
