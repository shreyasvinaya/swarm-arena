"""Stop an agent after repeated refusals, failures, or unproductive rounds."""

from swarm_arena._model.limits import Limits
from swarm_arena._model.tooling import RunStatus


class Watchdog:
    """
    Retain progress across rounds and latch the first stopping reason.

    Parameters
    ----------
    limits : Limits
        Refusal, consecutive-failure, and idle-round thresholds.

    Attributes
    ----------
    reason : str or None
        Identifier/count-only explanation, absent until a threshold trips.
    limit : str or None
        Limits field corresponding to the first stopping reason.

    Examples
    --------
    >>> w = Watchdog(Limits(max_consecutive_failures=1))
    >>> w.record_run(RunStatus.ERROR, passed=False)
    >>> w.limit
    'max_consecutive_failures'
    """

    def __init__(self, limits: Limits) -> None:
        """Initialize an agent's progress history."""
        self._limits = limits
        self._refusals = 0
        self._failures = 0
        self._idle_rounds = 0
        self._passed = False
        self.reason: str | None = None
        self.limit: str | None = None

    def record_run(self, status: RunStatus, *, passed: bool) -> None:
        """
        Record a tool outcome and whether it supplied a new passing candidate.

        Parameters
        ----------
        status : RunStatus
            OK resets failures; ERROR, OOM, and TIMEOUT increment them.
            Other outcomes neither increment nor reset failures.
        passed : bool
            Whether this run supplied a new passing candidate.

        Examples
        --------
        >>> w = Watchdog(Limits())
        >>> w.record_run(RunStatus.OK, passed=True)
        >>> w.end_round(ran=True) is None
        True
        """
        self._passed |= passed
        if status == RunStatus.OK:
            self._failures = 0
        elif status in {RunStatus.ERROR, RunStatus.OOM, RunStatus.TIMEOUT}:
            self._failures += 1
            if self._failures >= self._limits.max_consecutive_failures:
                self._stop("max_consecutive_failures", f"{self._failures} consecutive failures")

    def record_refusal(self) -> None:
        """
        Count a refused request and stop only after the allowed count.

        Examples
        --------
        >>> w = Watchdog(Limits(max_refused_repeats=1))
        >>> w.record_refusal()
        >>> w.record_refusal()
        >>> w.reason
        'refused a repeated request more than 1 times'
        """
        self._refusals += 1
        if self._refusals > self._limits.max_refused_repeats:
            self._stop(
                "max_refused_repeats",
                f"refused a repeated request more than {self._limits.max_refused_repeats} times",
            )

    def end_round(self, *, ran: bool) -> str | None:
        """
        Count unproductive active rounds and reset the per-round progress flag.

        Parameters
        ----------
        ran : bool
            Whether this agent ran any tool this round; starved rounds do not count.

        Returns
        -------
        str or None
            Latched stopping reason, if any.

        Examples
        --------
        >>> w = Watchdog(Limits(watchdog_idle_rounds=1))
        >>> w.end_round(ran=False) is None
        True
        >>> w.end_round(ran=True)
        'no new passing candidate in 1 rounds'
        """
        if ran:
            self._idle_rounds = 0 if self._passed else self._idle_rounds + 1
            if self._idle_rounds >= self._limits.watchdog_idle_rounds:
                self._stop(
                    "watchdog_idle_rounds",
                    f"no new passing candidate in {self._idle_rounds} rounds",
                )
        self._passed = False
        return self.reason

    def _stop(self, limit: str, reason: str) -> None:
        """Latch the first stopping threshold without replacing its explanation."""
        if self.reason is None:
            self.limit, self.reason = limit, reason
