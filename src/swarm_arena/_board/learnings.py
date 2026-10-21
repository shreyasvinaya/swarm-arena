"""Vet evidence-backed learnings against an agent's fixed arena view."""

import re
from collections.abc import Sequence
from typing import Final

from pydantic import ValidationError

from swarm_arena._board.base import BoardView
from swarm_arena._errors import SoftError, _core_signal
from swarm_arena._model.diagnostics import validation_summary
from swarm_arena._model.results import Candidate, Learning
from swarm_arena._model.tooling import RunStatus

DUPLICATE_JACCARD: Final = 0.6
LEARNING_GUIDE: Final = "Write for a colleague: what you tried, what happened, and the number."


def title_tokens(title: str) -> frozenset[str]:
    """
    Split a title into distinct lowercase word tokens.

    Parameters
    ----------
    title : str
        Learning title.

    Returns
    -------
    frozenset[str]
        Tokens matched by the Unicode word pattern.

    Examples
    --------
    >>> title_tokens("Fit log(nu), NU") == frozenset({"fit", "log", "nu"})
    True
    """
    return frozenset(re.findall(r"\w+", title.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """
    Measure the intersection over union of two title token sets.

    Parameters
    ----------
    a : frozenset[str]
        First set.
    b : frozenset[str]
        Second set.

    Returns
    -------
    float
        Similarity, or zero when both sets are empty.

    Examples
    --------
    >>> jaccard(frozenset("abc"), frozenset("abcde"))
    0.6
    >>> jaccard(frozenset(), frozenset())
    0.0
    """
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def find_duplicates(
    view: BoardView, own: Sequence[Learning], *, namespace: str, title: str
) -> list[Learning]:
    """
    Find near-duplicates among the top search hit and own visible learnings.

    Parameters
    ----------
    view : BoardView
        Fixed arena snapshot.
    own : Sequence[Learning]
        Agent's own learnings, including unpublished round items.
    namespace : str
        Namespace of the proposed learning.
    title : str
        Proposed title.

    Returns
    -------
    list[Learning]
        Distinct visible matches with title Jaccard similarity at least 0.6.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> view = Mock(spec=BoardView)
    >>> view.search_learnings.return_value = []
    >>> find_duplicates(view, [], namespace="toy", title="Converged in 8 steps")
    []
    """
    tokens = title_tokens(title)
    matches: dict[str, Learning] = {}
    for item in [*view.search_learnings(namespace, title, limit=1), *own]:
        if (
            item.namespace == namespace
            and not item.hidden
            and jaccard(tokens, title_tokens(item.title)) >= DUPLICATE_JACCARD
        ):
            matches.setdefault(item.id, item)
    return list(matches.values())


def is_verified(evidence: Sequence[Candidate]) -> bool:
    """
    Check whether any evidence candidate completed successfully and passed gates.

    Parameters
    ----------
    evidence : Sequence[Candidate]
        Supporting candidates resolved from the view or own items.

    Returns
    -------
    bool
        Whether at least one complete, successful, gated-in candidate exists.

    Examples
    --------
    >>> is_verified([])
    False
    """
    return any(
        not item.output.partial and item.output.status == RunStatus.OK and not item.gated_out
        for item in evidence
    )


def vet_learning(
    view: BoardView,
    own: Sequence[Candidate | Learning],
    *,
    namespace: str,
    author: str,
    title: str,
    text: str,
    evidence: Sequence[str],
    refines: Sequence[str],
    duplicate_reason: str | None,
) -> bool:
    """
    Validate a proposed learning before allocating its identity or publishing it.

    Parameters
    ----------
    view : BoardView
        Fixed arena snapshot authorizing evidence and refinement reads.
    own : Sequence[Candidate or Learning]
        Agent's own round items, visible in addition to the snapshot.
    namespace : str
        Posting namespace.
    author : str
        Team/agent address or a human-prefixed author.
    title : str
        Single-line title, trimmed by the production Learning validator.
    text : str
        Body of at most 1,500 characters.
    evidence : Sequence[str]
        Candidate identifiers, required for non-human authors.
    refines : Sequence[str]
        Visible learning identifiers refined by this post.
    duplicate_reason : str or None
        Nonblank explanation permitting a near-duplicate without refinement.

    Returns
    -------
    bool
        Whether the resolved evidence verifies this learning.

    Raises
    ------
    SoftError
        If content, evidence, refinements or duplicate justification is invalid.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> view = Mock(spec=BoardView)
    >>> view.search_learnings.return_value = []
    >>> vet_learning(view, [], namespace="toy", author="human:alice",
    ...     title="Baseline", text="Converged in 8 steps.", evidence=[],
    ...     refines=[], duplicate_reason=None)
    False
    """
    try:
        proposed = Learning(
            id="pending",
            namespace=namespace,
            run_id="pending",
            goal_id="pending",
            round=0,
            author=author,
            title=title,
            text=text,
            evidence=list(evidence),
            refines=list(refines),
        )
    except ValidationError as error:
        raise _core_signal(
            SoftError(f"invalid learning: {validation_summary(error)}. {LEARNING_GUIDE}")
        ) from None

    candidates = {item.id: item for item in own if isinstance(item, Candidate)}
    learnings = {
        item.id: item
        for item in own
        if isinstance(item, Learning) and item.namespace == namespace and not item.hidden
    }
    resolved: list[Candidate] = []
    for id in proposed.evidence:
        candidate = candidates.get(id)
        if candidate is None:
            try:
                candidate = view.get_candidate(id)
            except (KeyError, ValueError):
                raise _core_signal(
                    SoftError(f"unknown candidate, not in your view: {id}")
                ) from None
        resolved.append(candidate)
    for id in proposed.refines:
        learning = learnings.get(id)
        if learning is None:
            try:
                learning = view.get_learning(id)
            except (KeyError, ValueError):
                raise _core_signal(SoftError(f"unknown learning, not in your view: {id}")) from None
        if learning.namespace != namespace or learning.hidden:
            raise _core_signal(SoftError(f"unknown learning, not in your view: {id}"))

    duplicates = find_duplicates(
        view, list(learnings.values()), namespace=namespace, title=proposed.title
    )
    if (
        duplicates
        and not any(item.id in proposed.refines for item in duplicates)
        and not (duplicate_reason and duplicate_reason.strip())
    ):
        id = duplicates[0].id
        raise _core_signal(
            SoftError(
                f'near-duplicate of {id}. Cite it with refines=["{id}"] or pass duplicate_reason.'
            )
        )
    return is_verified(resolved)
