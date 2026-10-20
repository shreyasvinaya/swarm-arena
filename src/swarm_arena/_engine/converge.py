"""Select finalists from an already ordered candidate ranking."""

from collections.abc import Callable, Sequence

from swarm_arena._model.results import Candidate, Score


def select_finalists(
    ranked: Sequence[tuple[Candidate, Score]],
    *,
    top_n: int,
    distinct: bool,
    fingerprint: Callable[[Candidate], str],
) -> list[Candidate]:
    """
    Retain leading candidates, optionally skipping duplicate fingerprints.

    Parameters
    ----------
    ranked : Sequence[tuple[Candidate, Score]]
        Eligible candidate-score pairs in the authoritative ranking order.
    top_n : int
        Maximum number retained; nonpositive values select none.
    distinct : bool
        Whether repeated fingerprints are skipped.
    fingerprint : Callable[[Candidate], str]
        Result fingerprint supplied by the caller's tool adapter. Called only
        when distinct selection is requested, until enough finalists are found.

    Returns
    -------
    list[Candidate]
        Candidates in their original ranking order.

    Examples
    --------
    >>> select_finalists([], top_n=2, distinct=True, fingerprint=lambda c: c.id)
    []
    """
    if top_n <= 0:
        return []
    finalists: list[Candidate] = []
    seen: set[str] = set()
    for candidate, _ in ranked:
        if distinct:
            key = fingerprint(candidate)
            if key in seen:
                continue
            seen.add(key)
        finalists.append(candidate)
        if len(finalists) == top_n:
            break
    return finalists
