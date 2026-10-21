"""Display the committed, non-LLM history behind a candidate."""

from swarm_arena._board.arena import ArenaBoard, _namespace_frontier
from swarm_arena._board.cache import ArenaCache, LearningFilter
from swarm_arena._board.leaderboard import format_value
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.results import Candidate


def trace(board: ArenaBoard, candidate_id: str) -> list[str]:
    """
    Describe a candidate's committed plan, run, checks, scores and citations.

    Parameters
    ----------
    board : ArenaBoard
        Open arena reader. Capture its latest frontier once if needed; an
        already captured frontier stays fixed until explicitly refreshed.
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
        If the candidate is absent from the committed frontier. Process-local
        hints and orphaned journal segments do not count.

    Notes
    -----
    Reads never import producer plugins or re-admit historical plans. Scores
    are displayed in recorded position order without ranking or identity
    replacement. Planner, manager and tool-internal LLM calls are not included.

    Examples
    --------
    >>> lines = trace(board, "c-1")  # doctest: +SKIP
    >>> lines[0].startswith("plan     ")  # doctest: +SKIP
    True
    """
    with board._lock:
        frontier = board._latest_frontier()
        for namespace in board._namespaces:
            cache = board._cache(namespace)
            scoped = _namespace_frontier(frontier, namespace)
            candidates = cache.candidates(scoped, ids=[candidate_id])
            if candidates:
                candidate, position = candidates[0]
                return _lines(cache, scoped, candidate, position[2])
    raise KeyError(candidate_id)


def _lines(cache: ArenaCache, frontier: Frontier, candidate: Candidate, seq: int) -> list[str]:
    """Render one committed candidate without consulting process-local hints."""
    plan = cache._plan_for_round(
        candidate.run_id, candidate.goal_id, candidate.round, through_seq=seq
    )
    origin = next(
        origin
        for recorded, origin, plan_seq in cache.plans(frontier, candidate.run_id, candidate.goal_id)
        if recorded.revision == plan.revision and plan_seq < seq
    )
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
    ids = cache.find(frontier, query=None, filters=LearningFilter(), relevance_goal=None, limit=-1)
    lines.extend(
        f"learning {learning.id} {learning.title}"
        for learning in cache.learnings_by_id(frontier, ids)
        if candidate.id in learning.evidence
    )
    return lines
