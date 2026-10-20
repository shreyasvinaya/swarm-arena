"""Serialize goal capacity reservations and revision publication."""

from collections.abc import Awaitable, Callable

import anyio

from swarm_arena._errors import PlanError


class RunAdmission:
    """
    Reserve each running goal's peak agent count under one run-wide lock.

    Parameters
    ----------
    max_agents_total : int
        Maximum sum of reserved peaks, including future converge teams.

    Examples
    --------
    >>> admission = RunAdmission(4)
    >>> admission.max_agents_total
    4
    """

    def __init__(self, max_agents_total: int) -> None:
        """Initialize one lock and independent reservations for this run."""
        self.max_agents_total = max_agents_total
        self._lock = anyio.Lock()
        self._peaks: dict[str, int] = {}

    async def update(
        self,
        goal_id: str,
        peak: int,
        *,
        publish: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """
        Check capacity and atomically publish an already prepared revision.

        Snapshot and planner work must finish before entering this operation.
        A failed publication retains the previous reservation. Publication is
        shielded so cancellation cannot split its commit and capacity update.

        Parameters
        ----------
        goal_id : str
            Running goal whose reservation is replaced.
        peak : int
            Maximum concurrent ordinary or converge agents.
        publish : Callable or None, default=None
            Durable publication callback, called only after capacity fits.

        Raises
        ------
        PlanError
            The updated sum would exceed the run-wide maximum.
        ValueError
            The proposed count is negative.

        Examples
        --------
        >>> import anyio
        >>> admission = RunAdmission(2)
        >>> anyio.run(admission.update, "g", 2)
        """
        if peak < 0:
            raise ValueError("agent peak must be nonnegative")
        async with self._lock:
            others = sum(value for key, value in self._peaks.items() if key != goal_id)
            if others + peak > self.max_agents_total:
                raise PlanError([f"max_agents_total: goal {goal_id} exceeds available capacity"])
            with anyio.CancelScope(shield=True):
                if publish is not None:
                    await publish()
                self._peaks[goal_id] = peak

    async def release(self, goal_id: str) -> None:
        """
        Release a finished goal's reservation.

        Parameters
        ----------
        goal_id : str
            Goal leaving the scheduler.

        Examples
        --------
        >>> import anyio
        >>> anyio.run(RunAdmission(2).release, "g")
        """
        async with self._lock:
            self._peaks.pop(goal_id, None)
