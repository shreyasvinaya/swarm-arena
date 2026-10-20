"""Select goals, scoring settings, and retained candidates for goal parts."""

from collections.abc import Sequence

from swarm_arena._model.goal import Goal
from swarm_arena._model.plan import PartSpec, Plan, TeamSpec
from swarm_arena._model.planning_types import RoundState
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate


def _part(plan: Plan, name: str) -> PartSpec:
    """Find a declared part without resolving any component references."""
    for part in plan.parts:
        if part.name == name:
            return part
    raise ValueError(f"unknown part {name}")


def team_goal(plan: Plan, team: TeamSpec) -> Goal:
    """
    Select the main goal or the team's declared part goal.

    Parameters
    ----------
    plan : Plan
        Current plan revision.
    team : TeamSpec
        Team whose part assignment selects its goal.

    Returns
    -------
    Goal
        Main goal for an unassigned team, otherwise the part goal.

    Raises
    ------
    ValueError
        If the team names an undeclared part.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan, TeamSpec
    >>> team = TeamSpec(name="search", rationale="Explore", agents=[])
    >>> team_goal(Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[]), team).id
    'g'
    """
    return plan.goal if team.part is None else _part(plan, team.part).goal


def part_scoring(plan: Plan, part: str | None) -> tuple[list[Ref], list[Ref], str]:
    """
    Select scoring references and an explicit ranking name without imports.

    Parameters
    ----------
    plan : Plan
        Current plan revision supplying inherited settings.
    part : str or None
        Part name, or None for main-goal settings.

    Returns
    -------
    tuple[list[Ref], list[Ref], str]
        Checks, judges, and ranking name. Each absent part setting inherits the
        corresponding plan setting; empty lists remain empty. A sole judge's
        explicit name supplies an absent ranking. An empty ranking name means
        callers must use the recorded Score.judge instead of importing a class.

    Raises
    ------
    ValueError
        If the part is undeclared.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan, Ref
    >>> plan = Plan(goal=Goal(id="g", namespace="n"), teams=[],
    ...             judges=[Ref(path="example:Judge", name="quality")])
    >>> part_scoring(plan, None)[2]
    'quality'
    """
    checks, judges, ranking = plan.checks, plan.judges, plan.ranking
    if part is not None:
        spec = _part(plan, part)
        checks = checks if spec.checks is None else spec.checks
        judges = judges if spec.judges is None else spec.judges
        ranking = ranking if spec.ranking is None else spec.ranking
    if not ranking and len(judges) == 1:
        ranking = judges[0].name
    return list(checks), list(judges), ranking or ""


def kept_best(state: RoundState, part: str) -> Candidate | None:
    """
    Retain a part's best eligible candidate across all completed rounds.

    Parameters
    ----------
    state : RoundState
        Accumulated candidates, scores, and current per-part identities.
    part : str
        Part to rank, including when all its agents have stopped.

    Returns
    -------
    Candidate or None
        Leading candidate on the part's current identity and newest fidelity.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan
    >>> from swarm_arena.planning import RoundState
    >>> state = RoundState(goal_id="g", round=1, rounds_total=1,
    ...     plan=Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[]),
    ...     candidates=[], scores=[], status={}, ranking_identity={})
    >>> kept_best(state, "a") is None
    True
    """
    pairs = state.ranked(part)
    return pairs[0][0] if pairs else None


def join_parents(state: RoundState, *, parents_per_part: int) -> list[Candidate]:
    """
    Select leading parents independently for each part in plan order.

    Parameters
    ----------
    state : RoundState
        Accumulated candidates and per-part ranking identities.
    parents_per_part : int
        Maximum number of eligible parents retained from each part.

    Returns
    -------
    list[Candidate]
        Parents ordered by part declaration and then by that part's ranking.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan
    >>> from swarm_arena.planning import RoundState
    >>> state = RoundState(goal_id="g", round=1, rounds_total=1,
    ...     plan=Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[]),
    ...     candidates=[], scores=[], status={}, ranking_identity={})
    >>> join_parents(state, parents_per_part=1)
    []
    """
    return [
        candidate
        for part in state.plan.parts
        for candidate, _ in state.ranked(part.name)[: max(0, parents_per_part)]
    ]


def join_failure_reasons(candidates: Sequence[Candidate]) -> list[str]:
    """
    Collect failed join checks' recorded reasons in candidate and check order.

    Parameters
    ----------
    candidates : Sequence[Candidate]
        Join attempts with their check outcomes.

    Returns
    -------
    list[str]
        Reasons from failed checks, preserving their recorded order.

    Examples
    --------
    >>> join_failure_reasons([])
    []
    """
    return [
        reason
        for candidate in candidates
        for check in candidate.checks
        if not check.passed
        for reason in check.reasons
    ]
