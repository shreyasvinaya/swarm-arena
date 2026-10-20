"""Own the process-local artifact store and shared identifier source."""

import atexit
import shutil
import tempfile
import threading
from typing import Final

from swarm_arena._model.ids import IdGen
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store

_IDS: Final = IdGen("local", 0)
_ROOT_LOCK: Final = threading.Lock()
_root: str | None = None


def _artifact_root() -> str:
    """Return this process's artifact root: made on first use, removed at exit."""
    global _root
    with _ROOT_LOCK:
        if _root is None:
            _root = tempfile.mkdtemp(prefix="swarm-arena-testing-")
            atexit.register(shutil.rmtree, _root, ignore_errors=True)
        return _root


def _ids() -> IdGen:
    """Return this process's id source for candidates, jobs, and scores."""
    return _IDS


def _artifacts() -> ArtifactStore:
    """Open the artifact store under this process's artifact root."""
    return ArtifactStore(open_store(_artifact_root()))
