"""Define the planner interface independently of configurable components."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from swarm_arena._model.base import Origin
from swarm_arena._model.goal import Goal
from swarm_arena._model.plan import Plan
from swarm_arena._model.planning_types import History
from swarm_arena._model.run_config import RunConfig


class Planner(ABC):
    """
    Propose plans or revisions whose provenance is assigned by the core.

    Attributes
    ----------
    origin : str
        Planner origin, defaulting to ``llm``. Subclasses cannot claim ``human``.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan, Ref, RunConfig
    >>> from swarm_arena.planning import History, Planner
    >>> class FixedPlanner(Planner):
    ...     async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
    ...         return Plan(goal=goal, teams=[], judges=[Ref(path="demo:Judge")])
    >>> FixedPlanner().models()
    ()
    """

    origin: ClassVar[Origin] = "llm"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse planner subclasses claiming human-written provenance."""
        super().__init_subclass__(**kwargs)
        if cls.origin == "human":
            raise TypeError("Planner subclasses cannot claim human origin")

    def models(self) -> tuple[str, ...]:
        """
        Return the model aliases this planner may use.

        Returns
        -------
        tuple[str, ...]
            Model aliases used for capability and budget checks; empty by default.

        Examples
        --------
        >>> from swarm_arena.planning import Planner
        >>> Planner.models(None)
        ()
        """
        return ()

    @abstractmethod
    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        """
        Propose an initial plan or revise the latest plan using available evidence.

        Parameters
        ----------
        goal : Goal
            Goal to solve.
        config : RunConfig
            Human-controlled run settings and component catalog.
        history : History or None
            Prior revisions and results, or None for an initial proposal.

        Returns
        -------
        Plan
            Proposed plan. The core replaces its provenance fields on admission.

        Examples
        --------
        >>> import asyncio
        >>> from swarm_arena import Goal, Plan, Ref, RunConfig
        >>> from swarm_arena.planning import History, Planner
        >>> class FixedPlanner(Planner):
        ...     async def plan(self, goal: Goal, config: RunConfig,
        ...                    history: History | None) -> Plan:
        ...         return Plan(goal=goal, teams=[], judges=[Ref(path="demo:Judge")])
        >>> goal = Goal(id="g", namespace="example")
        >>> proposal = asyncio.run(FixedPlanner().plan(goal, RunConfig(), history=None))
        >>> proposal.goal.id, len(proposal.teams)
        ('g', 0)
        """
        raise NotImplementedError
