"""Own group membership, fidelity selection, and deterministic score ranking."""

from collections.abc import Iterable, Mapping, Sequence

from swarm_arena._model.plan import Plan
from swarm_arena._model.results import Candidate, Score


def groups(plan: Plan) -> dict[str | None, list[str]]:
    """
    List ordinary teams by main goal and then declared part order.

    Parameters
    ----------
    plan : Plan
        Current plan revision; the converge team is excluded.

    Returns
    -------
    dict[str or None, list[str]]
        Main-goal group first, followed by every part, including empty groups.
        Team order within each group follows the plan.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan
    >>> groups(Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[]))
    {None: []}
    """
    result: dict[str | None, list[str]] = {None: []}
    result.update((part.name, []) for part in plan.parts)
    for team in plan.teams:
        result[team.part].append(team.name)
    return result


def group_of(plan: Plan, team: str) -> str | None:
    """
    Find the ranking group of an ordinary team.

    Parameters
    ----------
    plan : Plan
        Current plan revision.
    team : str
        Ordinary team name.

    Returns
    -------
    str or None
        Part name, or None for the main goal.

    Raises
    ------
    ValueError
        If the team is absent from the plan's ordinary teams.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan, TeamSpec
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), judges=[],
    ...     teams=[TeamSpec(name="t", rationale="Explore", agents=[])])
    >>> group_of(plan, "t") is None
    True
    """
    for item in plan.teams:
        if item.name == team:
            return item.part
    raise ValueError("unknown ordinary team")


def newest_fidelity(candidates: Iterable[Candidate]) -> str | None:
    """
    Return the fidelity of the highest candidate round.

    Parameters
    ----------
    candidates : Iterable[Candidate]
        Candidates from one ranking group, in recorded order.

    Returns
    -------
    str or None
        Fidelity of the last candidate at the highest round, or None when empty.
        Eligibility and score availability do not change the selected fidelity.

    Examples
    --------
    >>> newest_fidelity([]) is None
    True
    """
    newest: Candidate | None = None
    for candidate in candidates:
        if newest is None or candidate.round >= newest.round:
            newest = candidate
    return newest.fidelity if newest is not None else None


def _id_counter(candidate: Candidate) -> int:
    """Read the numeric trailing counter, using zero for legacy nonnumeric IDs."""
    suffix = candidate.id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isascii() and suffix.isdecimal() else 0


def ranked(
    candidates: Iterable[Candidate],
    scores: Iterable[Score],
    *,
    identity: tuple[str, str, str],
    fidelity: str | None,
    allow_partial: bool,
) -> list[tuple[Candidate, Score]]:
    """
    Rank eligible candidates under one judge identity and fidelity.

    Parameters
    ----------
    candidates : Iterable[Candidate]
        Candidates from one group.
    scores : Iterable[Score]
        Recorded scores; the last matching score for each candidate is used.
    identity : tuple[str, str, str]
        Exact judge, rubric, and panel identity to compare.
    fidelity : str or None
        Explicit fidelity, or None to select the group's highest-round fidelity.
    allow_partial : bool
        Whether partial outputs may rank. Failed checks always exclude a result.

    Returns
    -------
    list[tuple[Candidate, Score]]
        Pairs sorted by key descending, round ascending, numeric trailing ID
        counter ascending, then full ID ascending. Unscored candidates are omitted.

    Examples
    --------
    >>> ranked([], [], identity=("program", "rh", ""), fidelity=None, allow_partial=False)
    []
    """
    pool = list(candidates)
    if fidelity is None:
        fidelity = newest_fidelity(pool)
    latest = {score.candidate_id: score for score in scores if score.identity == identity}
    result = [
        (candidate, latest[candidate.id])
        for candidate in pool
        if candidate.fidelity == fidelity
        and not candidate.gated_out
        and (allow_partial or not candidate.output.partial)
        and candidate.id in latest
    ]
    result.sort(key=lambda pair: (pair[0].round, _id_counter(pair[0]), pair[0].id))
    result.sort(key=lambda pair: pair[1].key, reverse=True)
    return result


def ranked_groups(
    candidates: Sequence[Candidate],
    scores: Iterable[Score],
    *,
    identity: Mapping[str | None, tuple[str, str, str]],
    fidelity: str | None,
    every_fidelity: bool = False,
) -> dict[str | None, list[tuple[Candidate, Score]]]:
    """
    Rank selected candidates per part, breaking equal keys by input order.

    Parameters
    ----------
    candidates : Sequence[Candidate]
        Candidates already selected by current or historical producing GoalKey.
        Their order is the tie order, normally committed Position order.
    scores : Iterable[Score]
        Recorded scores in replacement order; the last matching score wins.
    identity : Mapping[str or None, tuple[str, str, str]]
        One exact judge identity per group; absent groups are omitted.
    fidelity : str or None
        Requested fidelity, or the group's highest-round fidelity when None.
    every_fidelity : bool, default=False
        Rank every fidelity together, overriding ``fidelity`` for human reviews.

    Returns
    -------
    dict[str or None, list[tuple[Candidate, Score]]]
        Nonempty groups with eligible pairs ordered by descending key and then
        input order. The engine's ``ranked`` round/ID tie rule is unchanged.

    Examples
    --------
    >>> ranked_groups([], [], identity={}, fidelity=None)
    {}
    """
    by_group: dict[str | None, list[Candidate]] = {}
    for candidate in candidates:
        by_group.setdefault(candidate.part, []).append(candidate)
    recorded = list(scores)
    result: dict[str | None, list[tuple[Candidate, Score]]] = {}
    for part, pool in by_group.items():
        if part not in identity:
            continue
        fidelities = list(dict.fromkeys(c.fidelity for c in pool)) if every_fidelity else [fidelity]
        pairs = [
            pair
            for selected in fidelities
            for pair in ranked(
                pool, recorded, identity=identity[part], fidelity=selected, allow_partial=False
            )
        ]
        order = {candidate.id: index for index, candidate in enumerate(pool)}
        pairs.sort(key=lambda pair: order[pair[0].id])
        pairs.sort(key=lambda pair: pair[1].key, reverse=True)
        if pairs:
            result[part] = pairs
    return result
