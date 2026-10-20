"""Define configurable managers and the action-limit exception."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from swarm_arena._model.planning_types import RoundReport
from swarm_arena._model.ref import Configurable

if TYPE_CHECKING:
    from swarm_arena._agent.context import AgentContext


class LimitReached(RuntimeError):
    """
    Signal that an action cannot proceed within a configured limit.

    Parameters
    ----------
    limit : str
        Limits field name, ``budget``, or ``stop``.
    message : str
        Explanation containing identifiers, never sensitive input values.

    Attributes
    ----------
    limit : str
        Name of the limit that refused the action.

    Examples
    --------
    >>> from swarm_arena.manager import LimitReached
    >>> err = LimitReached("budget", "goal budget used up")
    >>> err.limit, str(err)
    ('budget', 'goal budget used up')
    """

    def __init__(self, limit: str, message: str) -> None:
        """Retain the limit identifier and initialize the runtime error message."""
        super().__init__(message)
        self.limit = limit


class Manager(Configurable, ABC):
    """
    Run an agent round through the core's budget-checked context.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None, optional
        Settings validated by the manager's nested Config model.

    Examples
    --------
    >>> from swarm_arena.manager import Manager, RoundReport
    >>> class QuietManager(Manager):
    ...     async def run_round(self, ctx: object) -> RoundReport:
    ...         return RoundReport(summary="done")
    >>> QuietManager().models()
    ()
    """

    def preflight(self) -> None:
        """
        Check connector prerequisites before starting a run.

        Examples
        --------
        >>> from swarm_arena.manager import Manager
        >>> Manager.preflight(None) is None
        True
        """
        return None

    def models(self) -> tuple[str, ...]:
        """
        Return catalog aliases used by this manager.

        Returns
        -------
        tuple[str, ...]
            Model aliases for capability, fingerprint, and budget checks.

        Examples
        --------
        >>> from swarm_arena.manager import Manager
        >>> Manager.models(None)
        ()
        """
        return ()

    @abstractmethod
    async def run_round(self, ctx: "AgentContext") -> RoundReport:
        """
        Execute one agent round through the supplied context.

        Parameters
        ----------
        ctx : AgentContext
            Core-created context exposing this agent's permitted actions.

        Returns
        -------
        RoundReport
            Summary of the completed round.

        Examples
        --------
        >>> report = await manager.run_round(ctx)  # doctest: +SKIP
        """
        raise NotImplementedError
