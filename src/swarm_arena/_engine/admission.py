"""Serialize goal capacity reservations and revision publication."""

from collections.abc import Awaitable, Callable, Sequence

import anyio

from swarm_arena._errors import PlanError, _core_error
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.plan import Plan


class RunAdmission:
    """
    Serialize goal capacity and board identities under one run-wide lock.

    Parameters
    ----------
    max_agents_total : int
        Maximum sum of reserved peaks, including future converge teams.
    plans : Sequence[Plan], optional
        Admitted plans, including queued and finished goals. Their latest
        identities remain reserved after capacity is released.

    Examples
    --------
    >>> admission = RunAdmission(4)
    >>> admission.max_agents_total
    4
    """

    def __init__(self, max_agents_total: int, plans: Sequence[Plan] = ()) -> None:
        """Initialize one lock and independent reservations for this run."""
        self.max_agents_total = max_agents_total
        self._lock = anyio.Lock()
        self._peaks: dict[str, int] = {}
        self._identities = {plan.goal.id: self._keys(plan) for plan in plans}

    @staticmethod
    def _keys(plan: Plan) -> frozenset[GoalKey]:
        return frozenset(goal.key() for goal in [plan.goal, *(p.goal for p in plan.parts)])

    async def update(
        self,
        goal_id: str,
        peak: int,
        *,
        publish: Callable[[], Awaitable[None]] | None = None,
        plan: Plan | None = None,
    ) -> None:
        """
        Check capacity and identities before atomically publishing a revision.

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
            Durable publication callback, called only after all checks pass.
        plan : Plan or None, default=None
            Proposed admitted revision. Its GoalKeys must be disjoint from
            every other admitted goal, regardless of lifecycle or origin.

        Raises
        ------
        PlanError
            Capacity would exceed the maximum or a GoalKey collides.
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
        keys = self._keys(plan) if plan is not None else None
        async with self._lock:
            if keys is not None and any(
                keys & other for goal, other in self._identities.items() if goal != goal_id
            ):
                raise _core_error(PlanError, ["revision identity collision: GoalKey"])
            others = sum(value for key, value in self._peaks.items() if key != goal_id)
            if others + peak > self.max_agents_total:
                raise _core_error(
                    PlanError, [f"max_agents_total: goal {goal_id} exceeds available capacity"]
                )
            with anyio.CancelScope(shield=True):
                if publish is not None:
                    await publish()
                self._peaks[goal_id] = peak
                if keys is not None:
                    self._identities[goal_id] = keys

    async def release(self, goal_id: str) -> None:
        """
        Release capacity while retaining a finished goal's board identities.

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
