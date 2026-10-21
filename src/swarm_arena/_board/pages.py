"""Render small, immutable round pages from their complete committed prefixes."""

from collections.abc import Iterable
from typing import TypeVar

from swarm_arena._board.cache import Position
from swarm_arena._board.leaderboard import flatten, format_value, group_judges, group_rows
from swarm_arena._journal.lease import Head
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.views import embed_source_head
from swarm_arena._model.results import Candidate, Score

_T = TypeVar("_T", Candidate, Score)


def page_key(goal_id: str, round: int) -> str:
    """
    Return the run-relative Markdown key for an ordinary round.

    Parameters
    ----------
    goal_id : str
        Main goal identifier.
    round : int
        One-based round number, padded to at least two digits.

    Returns
    -------
    str
        Round page storage key.

    Examples
    --------
    >>> page_key("viscosity", 2)
    'goals/viscosity/rounds/round-02.md'
    """
    return f"goals/{goal_id}/rounds/round-{round:02d}.md"


def state_positions(
    state: RunState, items: Iterable[_T], *, until_seq: int | None = None
) -> list[tuple[_T, Position]]:
    """
    Attach single-run journal positions, excluding items after an optional cutoff.

    Parameters
    ----------
    state : RunState
        Replayed source run and item positions.
    items : Iterable[Candidate or Score]
        Recorded items whose IDs are present in the state.
    until_seq : int or None, default=None
        Inclusive committed event sequence cutoff.

    Returns
    -------
    list[tuple[Candidate or Score, Position]]
        Items in input order with positions sortable by commit sequence.

    Examples
    --------
    >>> score = Score(id="s", key=(1,))
    >>> state = RunState(run_id="r", positions={"s": (2, 7)})
    >>> state_positions(state, [score])[0][1]
    ('', 'r', 7)
    >>> state_positions(state, [score], until_seq=6)
    []
    """
    return [
        (item, ("", state.run_id, state.positions[item.id][1]))
        for item in items
        if until_seq is None or state.positions[item.id][1] <= until_seq
    ]


def _closed_prefix(state: RunState, goal_id: str, round: int) -> RunState:
    """Reuse core replay for every displayed field at this round's closure."""
    seq = state.goals[goal_id].closed_seq[round]
    return RunState.from_events(event for event in state._events if event.seq <= seq)


def _cell(text: str) -> str:
    """Keep a Markdown table cell on one line and escape its separators."""
    return " ".join(text.splitlines()).replace("|", r"\|")


def round_page(
    state: RunState, goal_id: str, round: int, *, namespace_learnings: int, source: Head
) -> bytes:
    """
    Render the top ten rows, new learnings and prune reasons at a round close.

    Parameters
    ----------
    state : RunState
        State reconstructed with ``RunState.from_events`` from committed events.
        Later events are excluded from every displayed field.
    goal_id : str
        Main goal identifier with a recorded closure.
    round : int
        Closed ordinary round number.
    namespace_learnings : int
        Count from the recorded round frontier plus its new committed learnings.
    source : Head
        Committed source head embedded as the first Markdown line.

    Returns
    -------
    bytes
        UTF-8 Markdown with exactly one leading source-head comment.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> from swarm_arena import Goal, Plan
    >>> from swarm_arena._journal.event import Event
    >>> plan = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])
    >>> base = dict(run_id="r", context_id="r", epoch=1, goal_id="g",
    ...             ts=datetime(2026, 10, 18, tzinfo=timezone.utc))
    >>> events = [Event(**base, seq=1, type="plan_revision",
    ...                 payload={"plan": plan.model_dump(mode="json"), "origin": "human"}),
    ...           Event(**base, seq=2, type="round_closed", payload={"round": 1})]
    >>> head = Head(epoch=1, seq=2, segment="journal/e1-2.jsonl")
    >>> page = round_page(RunState.from_events(events), "g", 1,
    ...                   namespace_learnings=0, source=head)
    >>> b"Pruned: nothing." in page
    True
    """
    state = _closed_prefix(state, goal_id, round)
    goal = state.goals[goal_id]
    plan = state.plan_for_round(goal_id, round)
    part_keys = {part.name: part.goal.key() for part in plan.parts}
    candidates = [
        candidate
        for candidate in state.candidates.values()
        if candidate.goal_id == goal_id
        and (
            candidate.part is None
            or (
                candidate.part in part_keys
                and state.candidate_goal_keys[candidate.id] == part_keys[candidate.part]
            )
        )
    ]
    ids = {candidate.id for candidate in candidates}
    scores = [score for score in state.scores.values() if score.candidate_id in ids]
    judges = group_judges(plan, candidates=candidates, scores=scores)
    rows = flatten(
        group_rows(
            state_positions(state, candidates, until_seq=goal.closed_seq[round]),
            state_positions(state, scores, until_seq=goal.closed_seq[round]),
            judge=judges,
            run_id=state.run_id,
            identity_run=state.run_id,
            min_improvement=plan.converge.min_improvement,
        )
    )
    lower = bool(rows and rows[0].value is not None and rows[0].key[0] == -rows[0].value)
    lines = [
        f"# {plan.goal.namespace} / {goal_id} · run {state.run_id} · "
        f"round {round} of {plan.rounds_total(state.config.limits)}",
        f"Leaderboard ({judges.get(None, plan.ranking)}, "
        f"{'lower' if lower else 'higher'} is better)",
        "| # | Candidate | Team / agent | Score | Built on |",
        "|---|-----------|--------------|-------|----------|",
    ]
    for row in rows[:10]:
        cells = [
            str(row.rank),
            row.candidate_id,
            f"{row.team} / {row.agent}",
            format_value(row.value),
            ", ".join(row.built_on) or "—",
        ]
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    learnings = [
        item
        for item in state.learnings.values()
        if item.goal_id == goal_id and item.run_id == state.run_id and item.round == round
    ]
    lines.extend(["", f"New learnings ({len(learnings)})"])
    values = {row.candidate_id: row.value for row in rows}
    for item in learnings:
        status = (
            "from a human"
            if item.author.startswith("human:")
            else "verified"
            if item.verified
            else "unverified"
        )
        lines.append(f"- {item.id} **{item.title}** {item.author} · {status}")
        evidence = []
        for cid in item.evidence:
            candidate = state.candidates.get(cid)
            value = (
                format_value(values[cid])
                if cid in values
                else str(candidate.output.status)
                if candidate is not None
                else "unavailable"
            )
            evidence.append(f"{cid} ({value})")
        text = item.text.splitlines() or [""]
        text[-1] += " Evidence: " + " · ".join(evidence)
        lines.extend(f"  {line}" for line in text)
    pruned = [
        f"{'agent' if '/' in target else 'team'} `{target}` ({reason})"
        for number, decision in goal.decisions
        if number == round
        for target, reason in decision.prune.items()
    ]
    lines.extend(
        [
            "",
            "Pruned: " + ("; ".join(pruned) if pruned else "nothing") + ".",
            f"Full board: `swarm-arena board serve {state.run_id}` · {namespace_learnings} "
            f"learnings in namespace `{plan.goal.namespace}`.",
        ]
    )
    return embed_source_head("md", ("\n".join(lines) + "\n").encode(), source)
