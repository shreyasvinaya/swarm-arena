"""Lease helpers for tests that take over a run while its owner is still renewing."""

from typing import Any

from swarm_arena._errors import Fenced
from swarm_arena._journal.lease import Lease


def force_take_over(store: Any, *, owner_id: str, attempts: int = 50, **kwargs: Any) -> Lease:
    """
    Force a takeover, retrying when the live owner's renewal wins the race.

    A forced takeover reads the owner document and then writes it conditionally. A
    concurrent renewal between those steps makes the write fail with ``Fenced``; the
    engine's own takeover rereads and retries the same way.

    Parameters
    ----------
    store : Any
        Run store that holds the owner document.
    owner_id : str
        Identity of the successor.
    attempts : int, default=50
        Maximum takeover attempts.
    **kwargs : Any
        Other ``Lease.take_over`` arguments, for example ``host`` and ``ttl_s``.

    Returns
    -------
    Lease
        The successor's lease.

    Raises
    ------
    Fenced
        If every attempt loses to a concurrent renewal.

    Examples
    --------
    >>> callable(force_take_over)
    True
    """
    for attempt in range(attempts):
        try:
            return Lease.take_over(store, owner_id=owner_id, force=True, **kwargs)
        except Fenced:
            if attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")
