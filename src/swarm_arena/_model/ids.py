"""Generate run-scoped ids and random human-learning and run ids."""

import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal

from swarm_arena._model.canonical import sha256_hex


class IdGen:
    """
    Generate ids with independent counters per prefix.

    Parameters
    ----------
    run_id : str
        Run identity to hash into a 64-bit tag.
    epoch : int
        Ownership epoch to include in every generated id.
    counters : Mapping[str, int] or None, optional
        Last issued counter per prefix, copied to continue across epochs.

    Examples
    --------
    >>> ids = IdGen("local", 0)
    >>> ids.next("c") == f"c-{run_tag('local')}-e0-0001"
    True
    """

    def __init__(self, run_id: str, epoch: int, counters: Mapping[str, int] | None = None) -> None:
        """Copy the counters and retain the run tag and epoch."""
        self._tag = run_tag(run_id)
        self._epoch = epoch
        self._counters = dict(counters) if counters is not None else {}

    def next(self, prefix: Literal["c", "L", "s", "r", "llm", "x", "k"]) -> str:
        """
        Issue the next id for a supported prefix.

        Parameters
        ----------
        prefix : {"c", "L", "s", "r", "llm", "x", "k"}
            Kind of item whose counter should advance.

        Returns
        -------
        str
            Id with a run tag, epoch, and at least four counter digits.

        Raises
        ------
        ValueError
            If the prefix is unsupported.

        Examples
        --------
        >>> ids = IdGen("local", 2, {"c": 192})
        >>> ids.next("c").endswith("-e2-0193")
        True
        """
        if prefix not in ("c", "L", "s", "r", "llm", "x", "k"):
            raise ValueError("unsupported id prefix")
        counter = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = counter
        return f"{prefix}-{self._tag}-e{self._epoch}-{counter:04d}"

    def counters(self) -> dict[str, int]:
        """
        Return a copy of the last issued counter per prefix.

        Returns
        -------
        dict[str, int]
            Counter snapshot suitable for resuming in another epoch.

        Examples
        --------
        >>> IdGen("local", 0, {"c": 5}).counters()
        {'c': 5}
        """
        return self._counters.copy()


def run_tag(run_id: str) -> str:
    """
    Hash a run id into its 16-character lowercase hexadecimal tag.

    Parameters
    ----------
    run_id : str
        Run identity encoded as UTF-8 before hashing.

    Returns
    -------
    str
        First 16 characters of the SHA-256 digest.

    Examples
    --------
    >>> len(run_tag("local"))
    16
    """
    return sha256_hex(run_id.encode())[:16]


def human_learning_id() -> str:
    """
    Generate a human-learning id independent of a run or epoch.

    Returns
    -------
    str
        ``L-h-`` followed by 16 random hexadecimal characters.

    Examples
    --------
    >>> human_learning_id().startswith("L-h-")
    True
    """
    return "L-h-" + secrets.token_hex(8)


def new_run_id(now: datetime) -> str:
    """
    Generate a run id from an aware time and a random suffix.

    Parameters
    ----------
    now : datetime
        Aware time to normalize to UTC, at second precision.

    Returns
    -------
    str
        UTC timestamp followed by an underscore and six random hex digits.

    Raises
    ------
    ValueError
        If the supplied datetime is naive.

    Examples
    --------
    >>> new_run_id(datetime(2026, 10, 18, 18, 30, tzinfo=UTC)).startswith(
    ...     "2026-10-18T18-30-00Z_"
    ... )
    True
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ") + "_" + secrets.token_hex(3)
