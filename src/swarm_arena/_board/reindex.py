"""Rebuild disposable arena caches and inspect exact recorded memberships."""

import fcntl
import os
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.pointers import _validate_namespace
from swarm_arena._errors import _core_error
from swarm_arena._model.board_types import Frontier
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
