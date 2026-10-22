"""Human decisions at the collaborative planning boundary."""

from dataclasses import dataclass
from typing import Protocol

from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.plan import Plan
from swarm_arena._planning.provenance import _detach_plan


@dataclass(frozen=True)
class Approve:
    """Accept the draft without edits.

    Examples
    --------
    >>> Approve() == Approve()
    True
    """


@dataclass(frozen=True)
class Feedback:
    """Request another draft with human feedback.

    Parameters
    ----------
    text : str
        Feedback included in every subsequent draft request.

    Examples
    --------
    >>> Feedback("Add a team").text
    'Add a team'
    """

    text: str


@dataclass(frozen=True)
class Edited:
    """Propose an edited plan for catalog and goal admission.

    Parameters
    ----------
    plan : Plan
        Edited proposal, retaining its source directory.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan
    >>> Edited(Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])).plan.origin
    'human'
    """

    plan: Plan


def _detach_decision(decision: object) -> Approve | Feedback | Edited:
    """Normalize user-owned decision fields inside the preparation worker."""
    if isinstance(decision, Approve):
        return Approve()
    if isinstance(decision, Feedback):
        return Feedback(_context_string(decision.text, "feedback text"))
    if isinstance(decision, Edited):
        return Edited(_detach_plan(decision.plan))
    raise TypeError("invalid reviewer decision")


class Reviewer(Protocol):
    """Review drafts on the planning loop with cooperative cancellation.

    Parameters
    ----------
    *args : object
        Unused protocol construction arguments; implement a concrete reviewer.
    **kwargs : object
        Unused protocol construction keywords; implement a concrete reviewer.

    Examples
    --------
    >>> from swarm_arena import Plan
    >>> class Human:
    ...     async def review(self, plan: Plan) -> Approve:
    ...         return Approve()
    >>> reviewer: Reviewer = Human()
    """

    async def review(self, plan: Plan) -> Approve | Feedback | Edited:
        """Return a human decision about a detached draft.

        Parameters
        ----------
        plan : Plan
            Draft offered for review.

        Returns
        -------
        Approve or Feedback or Edited
            Acceptance, another draft request, or an edited proposal.

        Examples
        --------
        >>> from swarm_arena import Plan
        >>> class Human:
        ...     async def review(self, plan: Plan) -> Approve:
        ...         return Approve()
        >>> reviewer: Reviewer = Human()
        """
        ...
