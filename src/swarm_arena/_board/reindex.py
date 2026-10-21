"""Rebuild disposable arena caches and inspect exact recorded memberships."""

import fcntl
import os
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.pointers import _validate_namespace
from swarm_arena._errors import _core_error
from swarm_arena._journal.event import Event, EventType
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.run_config import RunConfig


@dataclass(frozen=True)
class _ReindexResult:
    """Return CLI diagnostics from the same rebuild, after closing its board."""

    frontier: Frontier
    run_counts: dict[str, int]
    warnings: tuple[str, ...]


def _reindex_with_diagnostics(
    config: RunConfig, namespaces: Sequence[str] = (), *, cache_dir: Path | None = None
) -> _ReindexResult:
    """Rebuild once and retain per-cache committed-run counts and warnings."""
    names = tuple(sorted({_validate_namespace(ns) for ns in namespaces}))
    board = ArenaBoard(
        arena_root=config.arena_root, cache_dir=None if cache_dir is None else str(cache_dir)
    )
    try:
        directory = board._cache_dir()
        if not names:
            names = tuple(sorted({_validate_namespace(p.stem) for p in directory.glob("*.sqlite")}))
        paths = [directory / f"{ns}.sqlite" for ns in names]
        # Validate the complete selection before creating locks or deleting data.
        for path in paths:
            for suffix in ("", "-wal", "-shm", ".lock"):
                selected = Path(str(path) + suffix)
                if selected.is_symlink() or (selected.exists() and not selected.is_file()):
                    raise _core_error(ValueError, "cache paths must be regular files")
        with ExitStack() as locks:
            if paths:
                directory.mkdir(parents=True, exist_ok=True)
            for path in paths:
                fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
                locks.callback(os.close, fd)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise _core_error(
                        RuntimeError,
                        "arena cache is in use by another swarm-arena process; stop it first",
                    ) from None
            for path in paths:
                for suffix in ("", "-wal", "-shm"):
                    Path(str(path) + suffix).unlink(missing_ok=True)
        # ArenaCache takes shared lifetime locks, so release exclusives first.
        board.open("", "", names)
        frontier = board.frontier()
        counts = {}
        warnings = []
        for namespace in names:
            cache = board._caches[namespace]
            with cache._lock:
                counts[namespace] = cache._conn.execute("SELECT count(*) FROM runs").fetchone()[0]
                warnings.extend(cache.warnings)
        return _ReindexResult(frontier, counts, tuple(warnings))
    finally:
        board.close()


def reindex(
    config: RunConfig, namespaces: Sequence[str] = (), *, cache_dir: Path | None = None
) -> Frontier:
    """
    Rebuild local caches from committed journals and immutable human objects.

    Parameters
    ----------
    config : RunConfig
        Configuration supplying the arena root.
    namespaces : Sequence[str], default=()
        Namespace slugs to rebuild. An empty sequence selects every
        ``<namespace>.sqlite`` file in the resolved cache directory.
    cache_dir : Path or None, default=None
        Explicit local cache directory, or ArenaBoard's XDG-aware default.

    Returns
    -------
    Frontier
        The single rebuilt frontier, including exact human-object names.

    Raises
    ------
    RuntimeError
        If any selected cache has an active shared or exclusive lifetime lock.
        No selected database or sidecar is deleted in that case.
    ValueError
        If a namespace is invalid or a selected path is not a regular file.

    Notes
    -----
    All selected exclusive locks are acquired before deleting databases and
    their WAL/SHM sidecars. Persistent lockfiles are retained. Locks and board
    connections are closed on every exit. This function prints nothing; the
    CLI consumes the private diagnostic result from the same rebuild.
    Rebuild equivalence compares complete logical rows, including producing
    identities and provenance. Database-local surrogate rowids (such as
    ``learnings.rowid`` used by FTS) are excluded: their allocation order may
    differ when journals and human objects are ingested in a different order.
    Like Board.frontier, this synchronous I/O belongs on a worker. Callers
    holding a run lease must use their owner's supervised abandonable boundary.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as root:
    ...     config = RunConfig(arena_root=root + "/arena")
    ...     rebuilt = reindex(config, ["toy"], cache_dir=Path(root) / "cache")
    ...     assert rebuilt == Frontier()
    """
    return _reindex_with_diagnostics(config, namespaces, cache_dir=cache_dir).frontier


def recorded_frontiers(events: Iterable[Event]) -> list[tuple[str, int, Frontier]]:
    """
    Decode frontier events in the journal order supplied by the caller.

    Parameters
    ----------
    events : Iterable[Event]
        Committed events in journal order; other event types are ignored.

    Returns
    -------
    list[tuple[str, int, Frontier]]
        Goal identifier, payload round, and validated frontier, retaining exact
        recorded roots, marks, review names, and curation names.

    Raises
    ------
    ValueError
        If a recorded frontier or its round is invalid.
    KeyError
        If a required frontier payload field is absent.

    Examples
    --------
    >>> from swarm_arena._model.canonical import utc_now
    >>> event = Event(epoch=1, seq=1, ts=utc_now(), run_id="r", context_id="r",
    ...               goal_id="g", type=EventType.FRONTIER,
    ...               payload={"round": 2, "frontier": Frontier().model_dump(mode="json")})
    >>> [(goal, round_) for goal, round_, _ in recorded_frontiers([event])]
    [('g', 2)]
    """
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
    """
    Reconstruct a snapshot and encode its namespace cache membership.

    Parameters
    ----------
    board : ArenaBoard
        Caller-owned board; it remains open for subsequent snapshots.
    goal : GoalKey
        Goal whose namespace cache supplies membership.
    frontier : Frontier
        Exact historical marks and human-object names, independent of current
        pointers and heads. Recorded run roots must still be reachable.

    Returns
    -------
    bytes
        ArenaCache's canonical sorted candidate IDs, score IDs, and learning
        IDs with pinned/hidden flags. This describes the namespace cache;
        current and historical part selection remain the view's responsibility.

    Notes
    -----
    Snapshot preparation performs I/O and follows the same worker and lease
    supervision contract as ``reindex``. The caller owns the board lifetime.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> from swarm_arena import Goal
    >>> with TemporaryDirectory() as root:
    ...     board = ArenaBoard(arena_root=root, cache_dir=root + "/cache")
    ...     try:
    ...         members = snapshot_membership(board, Goal(id="g", namespace="toy").key(),
    ...                                       Frontier())
    ...         assert members == b'{"candidates":[],"learnings":[],"scores":[]}'
    ...     finally:
    ...         board.close()
    """
    view = board.snapshot(goal, frontier)
    with view._read():
        return view._cache.membership(view._frontier)
