"""Build a stable board digest with a strict UTF-8 byte budget."""

import sys
from collections.abc import Mapping
from typing import Final, Protocol, runtime_checkable

from swarm_arena._board.arena import ArenaView
from swarm_arena._board.base import BoardView
from swarm_arena._board.leaderboard import format_value
from swarm_arena._model.board_types import Row
from swarm_arena._model.goal import GoalKey

UNTRUSTED_OPEN: Final = "<<from other agents, unverified>>"
UNTRUSTED_CLOSE: Final = "<<end of content from other agents>>"


def estimate_tokens(text: str) -> int:
    """
    Count strict UTF-8 bytes as an upper bound on tokens for byte-level tokenizers.

    This is an upper bound, not an estimate: each byte-level BPE or byte-fallback
    token covers at least one byte. Valid Unicode is preserved without repair.

    Parameters
    ----------
    text : str
        Text to encode using strict UTF-8.

    Returns
    -------
    int
        UTF-8 byte length.

    Raises
    ------
    UnicodeEncodeError
        If the text contains an unpaired surrogate.

    Examples
    --------
    >>> [estimate_tokens(t) for t in ("", "abcde", "é", "日本", "🙂")]
    [0, 5, 2, 6, 4]
    """
    return len(text.encode("utf-8"))


def _clip(text: str, cap: int) -> str:
    """Cut strict UTF-8 at a complete character boundary."""
    return text.encode("utf-8")[:cap].decode("utf-8", errors="ignore")


@runtime_checkable
class _DigestGroups(Protocol):
    """Preserve fixed group capabilities through identifier-read wrappers."""

    def _digest_group_views(self) -> dict[str | None, BoardView] | None:
        """Return the wrapped view's fixed groups, if provided."""
        ...


def _digest_group_views(view: BoardView) -> dict[str | None, BoardView] | None:
    """Select declared sibling capabilities without changing recorded labels."""
    if isinstance(view, _DigestGroups):
        return view._digest_group_views()
    if not isinstance(view, ArenaView):
        return None
    if view.part is not None:
        return {view.part: view}
    return {
        None: view,
        **{part: view._sibling_view(part) for part in view._ranking_judges() if part is not None},
    }


def build_digest(
    view: BoardView,
    *,
    goal: GoalKey,
    run_id: str,
    round: int,
    ranking: str | Mapping[str | None, str],
    cap_tokens: int,
    k: int = 5,
) -> str:
    """
    Describe leaderboard rows and learning titles within a fixed byte budget.

    Board rows and learning titles sit inside untrusted-content markers. Top
    learning titles (at most ten) are dropped from the end before pinned titles,
    then part rows, then main rows. Counts cover the visible namespace and this
    goal, including omitted titles. The recent count covers the owning run's
    visible learnings in the previous round, across namespace goals; round one
    omits it. At tiny caps retain only markers and counts when possible, or
    return just the counts line clipped at a complete UTF-8 character boundary.

    Parameters
    ----------
    view : BoardView
        Fixed goal snapshot with namespace-wide learning reads. Arena groups
        use fixed sibling capabilities; recorded part labels remain provenance.
    goal : GoalKey
        Snapshot goal identity.
    run_id : str
        Owning run, independent of frontier ordering or membership.
    round : int
        Round being started, numbered from one. Count this run's visible
        learnings from round minus one, across all goals in the namespace.
    ranking : str or Mapping[str or None, str]
        Judge for every group, or per-group judges (None is the main group).
        Groups missing from a mapping are omitted.
    cap_tokens : int
        Nonnegative strict UTF-8 byte cap, an upper bound on model tokens.
    k : int, default=5
        Maximum leaderboard rows per group.

    Returns
    -------
    str
        Bounded digest including markers, rows, titles, and the count line.

    Raises
    ------
    ValueError
        If the byte cap is negative or the goal differs from the view's scope.

    Examples
    --------
    Reuse the public in-memory board example without storage:

    >>> from doctest import script_from_examples
    >>> from swarm_arena.board import BoardView
    >>> exec(script_from_examples(BoardView.__doc__))
    >>> digest = build_digest(view, goal=view.goal, run_id="r", round=2,
    ...                       ranking="program", cap_tokens=1500)
    >>> len(digest.encode("utf-8")) <= 1500
    True
    """
    if cap_tokens < 0:
        raise ValueError("digest byte cap must be nonnegative")
    if goal != view.goal:
        raise ValueError("goal outside board snapshot")
    part = getattr(view, "part", None)
    group_views = _digest_group_views(view)
    by_judge: dict[str, list[Row]] = {}

    def rows_for(group: str | None) -> list[Row]:
        """Select rows by the group's ranked candidate IDs, never recorded labels."""
        judge = ranking if isinstance(ranking, str) else ranking.get(group)
        scoped = view if group_views is None else group_views.get(group)
        if judge is None or scoped is None:
            return []
        selected = {
            candidate.id
            for candidate in scoped.best(
                scoped.goal,
                max(0, k),
                judge=judge,
                part=group if group_views is None else None,
            )
        }
        if scoped is view:
            if judge not in by_judge:
                by_judge[judge] = scoped.leaderboard(scoped.goal, judge=judge)
            rows = by_judge[judge]
        else:
            rows = scoped.leaderboard(scoped.goal, judge=judge)
        return [row for row in rows if row.candidate_id in selected]

    main_rows = rows_for(part)
    if part is not None:
        parts: dict[str, list[Row]] = {}
    else:
        if group_views is not None:
            names = {name for name in group_views if name is not None}
        elif isinstance(ranking, str):
            # Custom connectors discover labels here, but best() selects membership.
            names = {row.part for row in by_judge.get(ranking, []) if row.part is not None}
        else:
            names = {name for name in ranking if name is not None}
        parts = {name: rows_for(name) for name in sorted(names)}

    def header(label: str, rows: list[Row], group: str | None) -> str:
        """Use the selected rows' recorded identity, without importing plugins."""
        judge = ranking if isinstance(ranking, str) else ranking.get(group)
        rubric, panel = (rows[0].rubric_hash[:8], rows[0].panel_hash[:8]) if rows else ("-", "-")
        return f"{label} ({judge}, rubric {rubric}, panel {panel or '-'}):"

    def row_line(row: Row) -> str:
        """Show the latest human verdict present in this fixed view."""
        verdicts = [
            score.verdict
            for score in view.scores(row.candidate_id)
            if score.judge.startswith("human:") and score.verdict in {"accept", "reject"}
        ]
        human = f" · human: {verdicts[-1]}" if verdicts else ""
        return (
            f"{row.rank}. {row.candidate_id} {row.team}/{row.agent} "
            f"{format_value(row.value)} built on {', '.join(row.built_on) or '—'}{human}"
        )

    main_judge = ranking if isinstance(ranking, str) else ranking.get(part)
    main_header = (
        [header(f"Leaderboard {goal.id}", main_rows, part)] if main_judge is not None else []
    )
    leaders = [row_line(row) for row in main_rows]
    part_blocks = [
        (header(f"Part {name}", rows, name), [row_line(row) for row in rows])
        for name, rows in parts.items()
        if rows
    ]
    learnings = view.learnings(goal.namespace, limit=sys.maxsize)
    total = view.count_learnings(goal.namespace)
    goal_total = view.count_learnings(goal.namespace, goal=goal)
    round_count = sum(item.run_id == run_id and item.round == round - 1 for item in learnings)
    pinned = [f"- {item.id} {item.title}" for item in learnings if item.pinned]
    top = [
        f"- {item.id} {item.title}" + (" · verified" if item.verified else "")
        for item in learnings
        if not item.pinned
    ][:10]

    def count_line() -> str:
        """Count every visible learning omitted by selection or byte pruning."""
        omitted = total - len(pinned) - len(top)
        recent = f", {round_count} new since the last round" if round > 1 else ""
        return (
            f"{omitted} more learnings in namespace {goal.namespace} "
            f"({goal_total} for this goal){recent}. Use search_learnings."
        )

    fixed = [UNTRUSTED_OPEN, *main_header, "Pinned learnings:", "Top learnings:", UNTRUSTED_CLOSE]
    # Each body line has one following newline before the final counts line.
    body_bytes = sum(estimate_tokens(line) + 1 for line in [*fixed, *leaders, *pinned, *top])
    body_bytes += sum(
        estimate_tokens(title) + 1 + sum(estimate_tokens(line) + 1 for line in lines)
        for title, lines in part_blocks
    )
    for lines in (top, pinned):
        while lines and body_bytes + estimate_tokens(count_line()) > cap_tokens:
            body_bytes -= estimate_tokens(lines.pop()) + 1
    for title, lines in reversed(part_blocks):
        while lines and body_bytes + estimate_tokens(count_line()) > cap_tokens:
            body_bytes -= estimate_tokens(lines.pop()) + 1
            if not lines:
                body_bytes -= estimate_tokens(title) + 1
    while leaders and body_bytes + estimate_tokens(count_line()) > cap_tokens:
        body_bytes -= estimate_tokens(leaders.pop()) + 1
    if body_bytes + estimate_tokens(count_line()) > cap_tokens:
        minimum = "\n".join([UNTRUSTED_OPEN, UNTRUSTED_CLOSE, count_line()])
        return (
            minimum if estimate_tokens(minimum) <= cap_tokens else _clip(count_line(), cap_tokens)
        )
    return "\n".join(
        [
            UNTRUSTED_OPEN,
            *main_header,
            *leaders,
            *(line for title, lines in part_blocks if lines for line in (title, *lines)),
            "Pinned learnings:",
            *pinned,
            "Top learnings:",
            *top,
            UNTRUSTED_CLOSE,
            count_line(),
        ]
    )
