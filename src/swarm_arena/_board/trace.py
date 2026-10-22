"""Display committed candidate provenance and LLM cache accounting."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.cache import ArenaCache
from swarm_arena._board.leaderboard import format_value
from swarm_arena._journal.event import EventType
from swarm_arena._journal.lease import Head, read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.plan import Plan
from swarm_arena._model.results import Candidate, LLMCall
from swarm_arena._store.base import Store


class CacheStats(FrozenModel):
    """
    Count calls, prompt-cache tokens and response-cache hits for one address.

    Parameters
    ----------
    calls : int
        Number of recorded calls, including failures and unbound probes.
    input_tokens : int
        Ordinary input tokens, excluding prompt-cache reads and writes.
    cache_read_tokens : int
        Tokens read from the provider prompt cache.
    cache_write_tokens : int
        Tokens written to the provider prompt cache.
    response_hits : int
        Calls supplied by the response cache.

    Examples
    --------
    >>> CacheStats(calls=1, input_tokens=25, cache_read_tokens=50,
    ...     cache_write_tokens=25, response_hits=0).hit_rate
    0.5
    """

    calls: int
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    response_hits: int

    @property
    def hit_rate(self) -> float:
        """
        Return the prompt-cache read share, or zero for no input tokens.

        Returns
        -------
        float
            Cache reads divided by ordinary input plus cache reads and writes.

        Examples
        --------
        >>> CacheStats(calls=0, input_tokens=0, cache_read_tokens=0,
        ...     cache_write_tokens=0, response_hits=0).hit_rate
        0.0
        """
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0


def cache_stats(calls: Iterable[LLMCall]) -> dict[str, CacheStats]:
    """
    Aggregate recorded calls by team/agent, or role when no agent is recorded.

    Parameters
    ----------
    calls : Iterable[LLMCall]
        Source-deduplicated calls from a captured generation.

    Returns
    -------
    dict[str, CacheStats]
        Agent addresses and colon-prefixed non-agent roles.

    Examples
    --------
    >>> cache_stats([])
    {}
    """
    totals: dict[str, list[int]] = {}
    for call in calls:
        key = f"{call.team}/{call.agent}" if call.agent else f":{call.role}"
        row = totals.setdefault(key, [0, 0, 0, 0, 0])
        for i, value in enumerate(
            (
                1,
                call.usage.input_tokens,
                call.usage.cache_read_tokens,
                call.usage.cache_write_tokens,
                int(call.cache_hit),
            )
        ):
            row[i] += value
    return {
        key: CacheStats(
            calls=row[0],
            input_tokens=row[1],
            cache_read_tokens=row[2],
            cache_write_tokens=row[3],
            response_hits=row[4],
        )
        for key, row in totals.items()
    }


def format_llm_step(call: LLMCall) -> str:
    """
    Render the recorded model, token count, cost and proven optional links.

    Parameters
    ----------
    call : LLMCall
        Committed core call. Cache origins are never inferred from requests.

    Returns
    -------
    str
        One shared CLI/viewer trace line.

    Examples
    --------
    >>> from swarm_arena.board import LLMCall
    >>> format_llm_step(LLMCall(id="c", run_id="r", goal_id="g", role="judge",
    ...     model="local/model", provider="local"))
    'llm c judge local/model tokens=0 cost=$0.000000'
    """
    line = (
        f"llm {call.id} {call.role} {call.model} "
        f"tokens={call.usage.total_tokens} cost=${call.cost_usd:.6f}"
    )
    if call.cache_hit and call.cached_from_call_id:
        line += f" cache-hit-of={call.cached_from_call_id}"
    if call.trace_url:
        line += f" {call.trace_url}"
    return line


def _calls_at(run_store: Store, head: Head) -> tuple[dict[str, LLMCall], str | None]:
    """Read one fixed chain without resuming or acquiring its run lease."""
    events = [event for _, batch in read_chain(run_store, head) for event in batch]
    # Reuse core duplicate-call and routing validation without interpreting a
    # standalone journal as a resumable engine run.
    state = RunState.from_events(event for event in events if event.type == EventType.LLM_CALL)
    reason = next(
        (
            str(event.payload.get("reason", event.payload.get("status", "")))
            for event in reversed(events)
            if event.type == EventType.GOAL_FINISHED
        ),
        None,
    )
    return state.llm_calls, reason


def llm_calls_of_run(run_store: Store) -> dict[str, LLMCall]:
    """
    Read calls from one authority snapshot and its committed journal chain.

    Parameters
    ----------
    run_store : Store
        Run-scoped store. This standalone read owns no lease or run deadline.

    Returns
    -------
    dict[str, LLMCall]
        Calls in committed order; empty when no authority/head exists.

    Examples
    --------
    >>> calls = llm_calls_of_run(run_store)  # doctest: +SKIP
    """
    owner = read_owner(run_store)
    return {} if owner is None or owner[0].head is None else _calls_at(run_store, owner[0].head)[0]


@dataclass(frozen=True)
class _SessionCalls:
    """Retain one external planning head's calls and completion reason."""

    calls: Mapping[str, LLMCall]
    reason: str | None


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
    replacement. Recorded LLM links are rendered once per producing run and call.
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
    with board._read_namespace() as reader:
        return reader.trace(reader.namespaces(), candidate_id)


def _lines(
    cache: ArenaCache,
    frontier: Frontier,
    candidate: Candidate,
    seq: int,
    revisions: Sequence[Plan],
    calls: Mapping[str, Mapping[str, LLMCall]],
    sessions: Mapping[str, _SessionCalls | None],
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
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []

    def emit(run_id: str, identifiers: Iterable[str], source: Mapping[str, LLMCall]) -> None:
        for identifier in identifiers:
            if identifier in source and (run_id, identifier) not in seen:
                seen.add((run_id, identifier))
                lines.append(format_llm_step(source[identifier]))

    run_calls = calls[candidate.run_id]
    lines.extend(
        [
            f"plan     {candidate.run_id} {candidate.goal_id} r{plan.revision} origin={origin} "
            f"team={candidate.team} agent={candidate.agent} rationale={rationale}",
        ]
    )
    emit(candidate.run_id, plan.planner_call_ids, run_calls)
    if plan.planning_session:
        session = sessions.get(plan.planning_session)
        label = f"planning session {plan.planning_session}"
        if session is None:
            lines.append(f"{label}: not found")
        else:
            if session.reason:
                lines.append(f"{label}: {session.reason}")
            emit(plan.planning_session, session.calls, session.calls)
    if candidate.requested_by_call:
        emit(candidate.run_id, [candidate.requested_by_call], run_calls)
    lines.append(
        f"run      {candidate.id} status={candidate.output.status.value} "
        f"partial={candidate.output.partial} round={candidate.round} fidelity={candidate.fidelity} "
        f"parents={parents} usage=in:{usage.input_tokens} out:{usage.output_tokens} "
        f"usd:{format_value(usage.usd)}",
    )
    emit(
        candidate.run_id,
        (call.id for call in run_calls.values() if call.candidate_id == candidate.id),
        run_calls,
    )
    for check in candidate.checks:
        outcome = "passed" if check.passed else "failed"
        reasons = "; ".join(check.reasons)
        lines.append(f"check    {check.check} {outcome}" + (f" {reasons}" if reasons else ""))
    for score, _ in cache.scores(frontier, candidate_ids=[candidate.id]):
        emit(candidate.run_id, score.llm_call_ids, run_calls)
        lines.append(
            f"score    {score.judge} {format_value(score.value)} key={score.key} "
            f"rubric={score.rubric_hash[:8]} panel={score.panel_hash[:8] or '-'}"
        )
    ids = cache._citing_learning_ids(frontier, candidate.id)
    for learning in cache.learnings_by_id(frontier, ids):
        if learning.posted_by_call:
            emit(learning.run_id, [learning.posted_by_call], calls.get(learning.run_id, {}))
        lines.append(f"learning {learning.id} {learning.title}")
    return lines
