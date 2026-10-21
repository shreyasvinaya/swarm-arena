"""Display the committed, non-LLM history behind a candidate."""

from collections.abc import Sequence

from swarm_arena._board.arena import ArenaBoard, _namespace_frontier
from swarm_arena._board.cache import ArenaCache
from swarm_arena._board.leaderboard import format_value
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.plan import Plan
from swarm_arena._model.results import Candidate


def _historical_plan(
    cache: ArenaCache,
    frontier: Frontier,
    candidate: Candidate,
    seq: int,
    revisions: Sequence[Plan],
) -> tuple[Plan, str]:
    """Select within the candidate prefix using published plan models."""
    projection = cache._plan_projection(
        frontier,
        candidate.run_id,
        candidate.goal_id,
        through_seq=seq,
        revisions=revisions,
    )
    plan = projection.plan_for_round(candidate.goal_id, candidate.round)
    origin = next(
        origin
        for recorded, origin in projection.plans[candidate.goal_id]
        if recorded.revision == plan.revision
    )
    return plan, origin


def trace(board: ArenaBoard, candidate_id: str) -> list[str]:
    """
    Describe a candidate's committed plan, run, checks, scores and citations.

    Parameters
    ----------
    board : ArenaBoard
        Open arena reader. Read its published generation; call ``frontier()``
        explicitly to publish initial or refreshed committed state.
    candidate_id : str
        Candidate identifier in a committed source run.

    Returns
    -------
    list[str]
        Plan, run, check, score and learning lines in that order. The producing
        plan is round-applicable within the candidate's committed prefix.
        Journal and human scores retain their recorded judge names and values;
        visible citing learnings come from the same fixed namespace frontier.

    Raises
    ------
    KeyError
        If the candidate is absent from the published frontier, including before
        the first publication. Process-local hints and orphaned segments do
        not count.

    Notes
    -----
    Reads never import producer plugins or re-admit historical plans. Scores
    are displayed in recorded position order without ranking or identity
    replacement. Planner, manager and tool-internal LLM calls are not included.
    Retained plan models come from one captured generation; exact committed
    mark and candidate source prefix restrict their round-applicable selection.
    Reads do not replay history, refresh storage or initialize cache readers.
    One outer frontier scope covers the composite read; only visible citing
    learnings are selected and hydrated, with no citation limit.

    Examples
    --------
    >>> lines = trace(board, "c-1")  # doctest: +SKIP
    >>> lines[0].startswith("plan     ")  # doctest: +SKIP
    True
    """
    state = board._read_state()
    for namespace in state.namespaces:
        cache = state.readers.get(namespace)
        if cache is None:
            continue
        scoped = _namespace_frontier(state.frontier, namespace)
        with cache._lock, cache._frontier(scoped):
            candidates = cache.candidates(scoped, ids=[candidate_id])
            if candidates:
                candidate, position = candidates[0]
                revisions = state.plans[candidate.run_id].revisions[candidate.goal_id]
                return _lines(cache, scoped, candidate, position[2], revisions)
    raise KeyError(candidate_id)


def _lines(
    cache: ArenaCache,
    frontier: Frontier,
    candidate: Candidate,
    seq: int,
    revisions: Sequence[Plan],
) -> list[str]:
    """Render one committed candidate without consulting process-local hints."""
    plan, origin = _historical_plan(cache, frontier, candidate, seq, revisions)
    rationale = next(
        (
            agent.rationale
            for team in plan.teams
            if team.name == candidate.team
            for agent in team.agents
            if agent.name == candidate.agent
        ),
        "-",
    )
    usage = candidate.output.usage
    parents = ",".join(candidate.parent_ids) or "-"
    lines = [
        f"plan     {candidate.run_id} {candidate.goal_id} r{plan.revision} origin={origin} "
        f"team={candidate.team} agent={candidate.agent} rationale={rationale}",
        f"run      {candidate.id} status={candidate.output.status.value} "
        f"partial={candidate.output.partial} round={candidate.round} fidelity={candidate.fidelity} "
        f"parents={parents} usage=in:{usage.input_tokens} out:{usage.output_tokens} "
        f"usd:{format_value(usage.usd)}",
    ]
    for check in candidate.checks:
        outcome = "passed" if check.passed else "failed"
        reasons = "; ".join(check.reasons)
        lines.append(f"check    {check.check} {outcome}" + (f" {reasons}" if reasons else ""))
    for score, _ in cache.scores(frontier, candidate_ids=[candidate.id]):
        lines.append(
            f"score    {score.judge} {format_value(score.value)} key={score.key} "
            f"rubric={score.rubric_hash[:8]} panel={score.panel_hash[:8] or '-'}"
        )
    ids = cache._citing_learning_ids(frontier, candidate.id)
    lines.extend(
        f"learning {learning.id} {learning.title}"
        for learning in cache.learnings_by_id(frontier, ids)
    )
    return lines
