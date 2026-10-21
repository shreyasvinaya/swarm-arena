"""Adapt legacy fixture inputs to the production cache and rebuild paths."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.cache import ArenaCache, LearningFilter, goal_key_digest
from swarm_arena._board.reindex import _reindex_with_diagnostics
from swarm_arena._errors import ChainError, _core_error
from swarm_arena._journal.event import Event, EventType
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.plan import Plan
from swarm_arena._model.run_config import RunConfig


def run_root(cache: ArenaCache, run_id: str) -> str | None:
    """Adapt fixture inputs to cache reads."""
    with cache._lock:
        row = cache._conn.execute("SELECT run_root FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return row[0] if row else None


def plans(
    cache: ArenaCache, frontier: Frontier, run_id: str, goal_id: str
) -> list[tuple[Plan, str, int]]:
    """Adapt fixture inputs to cache reads."""
    with cache._lock, cache._frontier(frontier):
        rows = cache._conn.execute(
            "SELECT plan_json, origin, p.seq FROM plans p "
            "JOIN _read_sets m ON m.set_name='marks' AND m.id=p.run_id AND p.seq<=m.seq "
            "WHERE p.run_id=? AND goal_id=? ORDER BY revision",
            (run_id, goal_id),
        ).fetchall()
        return [(Plan.model_validate_json(data), origin, seq) for data, origin, seq in rows]


def search(
    cache: ArenaCache,
    namespace: str,
    query: str,
    *,
    frontier: Frontier,
    goal: GoalKey | None,
    tags: Sequence[str],
    limit: int,
) -> list[str]:
    """Adapt fixture inputs to cache reads."""
    if namespace != cache.namespace:
        raise _core_error(ValueError, "search namespace does not match cache")
    digest = goal_key_digest(goal) if goal is not None else None
    return cache.find(
        frontier,
        query=query,
        filters=LearningFilter(goal_key=digest, goal_id=goal.id if goal else None, tags=list(tags)),
        relevance_goal=(digest, goal.id) if digest is not None and goal is not None else None,
        limit=limit,
    )


def _plan_for_round(
    cache: ArenaCache, run_id: str, goal_id: str, round: int, *, through_seq: int
) -> Plan:
    """Adapt fixture inputs to cache reads."""
    with cache._lock:
        try:
            return cache._replay(run_id, through_seq=through_seq).plan_for_round(goal_id, round)
        except (KeyError, ValueError):
            raise _core_error(ChainError, "no committed plan for goal") from None


def reindex(
    config: RunConfig, namespaces: Sequence[str] = (), *, cache_dir: Path | None = None
) -> Frontier:
    """Inspect fixture state through the production board."""
    return _reindex_with_diagnostics(config, namespaces, cache_dir=cache_dir).frontier


def recorded_frontiers(events: Iterable[Event]) -> list[tuple[str, int, Frontier]]:
    """Inspect fixture state through the production board."""
    result = []
    for event in events:
        if event.type == EventType.FRONTIER:
            round_ = event.payload["round"]
            if type(round_) is not int:
                raise _core_error(ValueError, "frontier round must be an integer")
            result.append(
                (event.goal_id, round_, Frontier.model_validate(event.payload["frontier"]))
            )
    return result


def snapshot_membership(board: ArenaBoard, goal: GoalKey, frontier: Frontier) -> bytes:
    """Inspect fixture state through the production board."""
    view = board.snapshot(goal, frontier)
    with view._read():
        return view._cache.membership(view._frontier)
