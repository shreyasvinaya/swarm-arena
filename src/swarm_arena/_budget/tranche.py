"""Reserve durable batches and account for individual calls locally."""

from dataclasses import dataclass, field

import anyio

from swarm_arena._agent.manager import LimitReached
from swarm_arena._budget.amount import Amount, _usd_from_micro_units
from swarm_arena._budget.ledger import Ledger, _nonnegative


@dataclass
class _Batch:
    """Retain a committed tranche and exact locally available units."""

    id: str
    tokens: int
    micros: int
    calls: list[str] = field(default_factory=list)
    settled: bool = False


@dataclass
class _Call:
    """Track either a local call or an independent durable reservation."""

    estimate: Amount
    batch: _Batch | None
    actual: Amount | None = None


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

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> tranche = Tranche(Mock(spec=Ledger), "g1/t/a", "g1",
    ...                   size=Amount(tokens=25_000, usd=0.25))
    >>> tranche.pending()
    0
    >>> call_id = await tranche.reserve_call(Amount(usd=0.1))  # doctest: +SKIP
    >>> await tranche.settle_call(call_id, Amount(usd=0.01))  # doctest: +SKIP
    >>> await tranche.close()  # doctest: +SKIP
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
        >>> cid = await tranche.reserve_call(Amount(usd=0.05))  # doctest: +SKIP
        """
        async with self._lock:
            if self._closed:
                raise ValueError("tranche is closed")
            estimate = _nonnegative(estimate)
            if not estimate.fits_in(self._size, cap_tokens=True, cap_usd=True):
                res = await self._ledger.reserve(
                    self._holder, estimate, kind="call", goal_id=self._goal_id
                )
                self._calls[res.id] = _Call(estimate, None)
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
                res = await self._ledger.reserve(
                    self._holder, size, kind="tranche", goal_id=self._goal_id
                )
                batch = _Batch(res.id, size.tokens, size._micro_usd)
                self._batches.append(batch)
            cid = f"{batch.id}/call-{len(batch.calls) + 1}"
            batch.tokens -= estimate.tokens
            batch.micros -= estimate._micro_usd
            batch.calls.append(cid)
            self._calls[cid] = _Call(estimate, batch)
            return cid

    async def settle_call(self, reservation_id: str, actual: Amount) -> None:
        """
        Settle an admitted call once, ignoring arrivals after close.

        Parameters
        ----------
        reservation_id : str
            Identifier returned by ``reserve_call``.
        actual : Amount
            Nonnegative reported usage, including any overspend.

        Examples
        --------
        >>> await tranche.settle_call(cid, Amount(usd=0.01))  # doctest: +SKIP
        """
        async with self._lock:
            if self._closed:
                return
            call = self._calls[reservation_id]
            if call.actual is not None:
                return
            actual = _nonnegative(actual)
            if call.batch is None:
                await self._ledger.settle(reservation_id, actual)
            else:
                call.batch.tokens += call.estimate.tokens - actual.tokens
                call.batch.micros += call.estimate._micro_usd - actual._micro_usd
            call.actual = actual

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

    async def close(self) -> None:
        """
        Return unused batch funding and charge pending calls at their estimates.

        Notes
        -----
        Repeated closes are idempotent. If a durable settlement fails, another
        close retries the remaining settlements without reopening admission.

        Examples
        --------
        >>> await tranche.close()  # doctest: +SKIP
        """
        async with self._lock:
            self._closed = True
            for batch in self._batches:
                if batch.settled:
                    continue
                amounts = [self._calls[c].actual or self._calls[c].estimate for c in batch.calls]
                actual = Amount(
                    tokens=sum(a.tokens for a in amounts),
                    usd=_usd_from_micro_units(sum(a._micro_usd for a in amounts)),
                )
                await self._ledger.settle(batch.id, actual)
                batch.settled = True
            for cid, call in self._calls.items():
                if call.batch is None and call.actual is None:
                    await self._ledger.settle(cid, call.estimate, estimated=True)
                    call.actual = call.estimate
