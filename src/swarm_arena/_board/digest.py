"""Build a stable board digest with a strict UTF-8 byte budget."""

import sys

from swarm_arena._board.base import BoardView
from swarm_arena._model.goal import GoalKey


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


def build_digest(
    view: BoardView,
    *,
    goal: GoalKey,
    run_id: str,
    round: int,
    ranking: str,
    cap_tokens: int,
    k: int = 5,
) -> str:
    """
    Describe leaderboard rows and learning titles within a fixed byte budget.

    Newest titles are dropped from the end before pinned titles, then all but
    the first leaderboard row. The omitted-learning count is updated on every
    drop. If the remaining layout cannot fit, only the count line is returned,
    cut at a complete UTF-8 character boundary. The recent count covers visible
    learnings from the explicit owning run, created
    in the round that just ended. It is omitted when starting round one.

    Parameters
    ----------
    view : BoardView
        Fixed goal snapshot with namespace-wide learning reads.
    goal : GoalKey
        Snapshot goal identity.
    run_id : str
        Owning run, independent of frontier ordering or membership.
    round : int
        Round being started, numbered from one. Count this run's visible
        learnings from round minus one, across all goals in the namespace.
    ranking : str
        Ranking judge name.
    cap_tokens : int
        Nonnegative strict UTF-8 byte cap, an upper bound on model tokens.
    k : int, default=5
        Maximum leaderboard rows.

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
    rows = view.leaderboard(goal, judge=ranking)[: max(0, k)]
    learnings = view.learnings(goal.namespace, limit=sys.maxsize)
    if not rows and not learnings:
        return _clip("No results yet. 0 learnings.", cap_tokens)
    round_count = (
        sum(
            item.run_id == run_id and item.round == round - 1
            for item in view.learnings(goal.namespace, limit=sys.maxsize)
        )
        if round > 1
        else 0
    )
    leaders = [
        f"{row.rank}. {row.candidate_id} {row.team}/{row.agent} "
        f"score={row.value if row.value is not None else row.key} "
        f"built on {', '.join(row.built_on) or '—'}"
        for row in rows
    ]
    pinned = [f"- {item.id} {item.title}" for item in learnings if item.pinned]
    newest = [f"- {item.id} {item.title}" for item in learnings if not item.pinned]
    headers = [f"Leaderboard ({ranking}, top {k}):", "Pinned:", "Newest:"]
    # Track byte totals while dropping lines; rendering each trial would be quadratic.
    body_bytes = sum(estimate_tokens(line) + 1 for line in [*headers, *leaders, *pinned, *newest])

    def count_line(omitted: int | None = None) -> str:
        """Count visible learnings omitted from the current title lists."""
        if omitted is None:
            omitted = len(learnings) - len(pinned) - len(newest)
        recent = f", {round_count} new since the last round" if round > 1 else ""
        return f"{omitted} more learnings{recent}. Use search_learnings."

    for lines, keep in ((newest, 0), (pinned, 0), (leaders, 1)):
        while len(lines) > keep and body_bytes + estimate_tokens(count_line()) > cap_tokens:
            body_bytes -= estimate_tokens(lines.pop()) + 1
    if body_bytes + estimate_tokens(count_line()) > cap_tokens:
        return _clip(count_line(len(learnings)), cap_tokens)
    return "\n".join(
        [
            headers[0],
            *leaders,
            headers[1],
            *pinned,
            headers[2],
            *newest,
            count_line(),
        ]
    )
