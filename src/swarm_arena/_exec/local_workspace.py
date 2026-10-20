"""Create private workspaces and remove tool-written directory trees safely."""

import contextlib
import logging
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, TypeVar

from swarm_arena._errors import _core_error
from swarm_arena._exec.executor import Job
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store

_LOG: Final = logging.getLogger(__name__)
_T = TypeVar("_T")
_PREFIX_UNSAFE: Final = re.compile(r"[^A-Za-z0-9._-]")
_PREFIX_MAX: Final = 64
_DIR_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_PROC_FD: Final = "/proc/self/fd"
_UNSAFE_ROOT: Final = (
    "workspace root is not safe: it and every parent directory must be owned by this user "
    "or root, and not writable by group or others unless sticky (like /tmp)"
)


def _default_root() -> Path:
    """Return this user's default workspace root, so users of one host never share it."""
    return Path(tempfile.gettempdir()) / f"swarm-arena-{os.getuid()}"


def _only_owner_can_change(info: os.stat_result) -> bool:
    """Return whether only this user or root can add, rename, or remove a directory's entries."""
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (os.getuid(), 0):
        return False
    shared = info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    return not shared or bool(info.st_mode & stat.S_ISVTX)


def _workspace_root(configured: str | None) -> Path:
    """Create the workspace root if needed; refuse one that another user could change."""
    root = Path(configured).expanduser() if configured is not None else _default_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = root.resolve(strict=True)
        safe = all(_only_owner_can_change(os.stat(path)) for path in (root, *root.parents))
    except OSError:
        raise _core_error(RuntimeError, "cannot create the workspace root") from None
    if not safe:
        raise _core_error(RuntimeError, _UNSAFE_ROOT)
    return root


def _open_workspace(configured: str | None, job: Job) -> tuple[Path, ArtifactStore]:
    """Open the job's artifact store and create its private workspace directory."""
    artifacts = ArtifactStore(open_store(job.artifact_root))
    root = _workspace_root(configured)
    prefix = _PREFIX_UNSAFE.sub("_", job.id)[:_PREFIX_MAX] + "-"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root)), artifacts


def _unlock_directory(parent: int, name: str) -> bool:
    """
    Give a locked subdirectory back to its owner without ever following a symlink.

    macOS and the BSDs have ``fchmodat(AT_SYMLINK_NOFOLLOW)``. Linux has no
    such call for directories, so the entry is opened with ``O_PATH |
    O_NOFOLLOW | O_DIRECTORY`` (which refuses a symlink and needs no read
    permission) and changed through its ``/proc/self/fd`` link. Where neither
    exists, or the entry is not a real directory any more, nothing is changed
    and a warning is logged.
    """
    with contextlib.suppress(OSError):
        if os.chmod in os.supports_follow_symlinks and os.chmod in os.supports_dir_fd:
            os.chmod(name, 0o700, dir_fd=parent, follow_symlinks=False)
            return True
        o_path = getattr(os, "O_PATH", None)
        if o_path is not None and os.path.isdir(_PROC_FD):
            fd = os.open(name, o_path | os.O_NOFOLLOW | os.O_DIRECTORY, dir_fd=parent)
            try:
                os.chmod(f"{_PROC_FD}/{fd}", 0o700)
            finally:
                os.close(fd)
            return True
    _LOG.warning("cannot unlock a locked directory without following symlinks: %s", name)
    return False


def _open_subdirectory(parent: int, name: str) -> int | None:
    """Open a real subdirectory, unlocking it first if needed; None leaves it in place."""
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=parent)
    except PermissionError:
        if not _unlock_directory(parent, name):
            return None
    except OSError:  # no longer a real directory (for example a symlink swapped in)
        return None
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=parent)
    except OSError:
        return None


def _unlocked_retry(directory: int, operation: Callable[[], _T]) -> _T:
    """
    Run an operation on a directory's entries; if it is refused, unlock the directory and retry.

    Only a refusal (EACCES or EPERM) unlocks the directory: deep in a tree a
    chmod is slow (about 8 ms per level on APFS past PATH_MAX), even when it
    changes nothing.
    """
    try:
        return operation()
    except PermissionError:
        os.fchmod(directory, 0o700)
        return operation()


def _clear_files(directory: int) -> list[str]:
    """List a directory once: unlink every entry that is not a directory, return the rest."""
    subdirectories: list[str] = []
    for name in os.listdir(directory):
        info = _unlocked_retry(
            directory, partial(os.stat, name, dir_fd=directory, follow_symlinks=False)
        )
        if stat.S_ISDIR(info.st_mode):
            subdirectories.append(name)
        else:  # a symlink is removed, never followed
            _unlocked_retry(directory, partial(os.unlink, name, dir_fd=directory))
    return subdirectories


def _open_top(top: Path) -> int:
    """Open the tree's top directory, unlocking it first when a tool locked it."""
    try:
        return os.open(top, _DIR_FLAGS)
    except PermissionError:
        parent = os.open(top.parent, _DIR_FLAGS)
        try:
            _unlock_directory(parent, top.name)
        finally:
            os.close(parent)
        return os.open(top, _DIR_FLAGS)


@dataclass
class _Frame:
    """One open level of the removal walk: its name, its inode, and the names left to do."""

    name: str
    inode: tuple[int, int]
    pending: list[str]


def _remove_tree(top: Path) -> None:
    """
    Remove a tree in linear time without recursion, holding at most two descriptors.

    Each directory is listed once: its files are unlinked and the names of its
    subdirectories are kept in its stack frame. Every child is opened from its
    parent's descriptor, so neither the recursion limit nor PATH_MAX applies,
    and a symlink is unlinked, never followed. Going back up opens ``..`` and
    checks that it is the directory the walk came from. A directory is
    unlocked only when opening it, or a stat, unlink, or rmdir in it, is
    refused; that includes the top itself, so a tool that locks its own
    workspace root does not keep it. A locked directory that cannot be
    unlocked safely is left in place.
    """
    if not stat.S_ISDIR(os.lstat(top).st_mode):
        os.unlink(top)
        return
    fd = _open_top(top)
    try:
        info = os.fstat(fd)
        stack = [_Frame("", (info.st_dev, info.st_ino), _clear_files(fd))]
        while True:
            frame = stack[-1]
            if frame.pending:
                child = frame.pending.pop()
                opened = _open_subdirectory(fd, child)
                if opened is None:
                    continue  # left in place, so this directory's rmdir fails later
                os.close(fd)
                fd = opened
                info = os.fstat(fd)
                stack.append(_Frame(child, (info.st_dev, info.st_ino), _clear_files(fd)))
                continue
            if len(stack) == 1:
                break
            stack.pop()
            parent = os.open("..", _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = parent
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != stack[-1].inode:
                raise OSError("workspace tree changed during removal")
            with contextlib.suppress(OSError):  # not empty: something in it was left
                _unlocked_retry(fd, partial(os.rmdir, frame.name, dir_fd=fd))
    finally:
        os.close(fd)
    os.rmdir(top)


_REMOVERS: Final[tuple[Callable[[Path], None], ...]] = (shutil.rmtree, _remove_tree)


def _remove_workspace(workspace: Path) -> None:
    """Remove a workspace without ever raising; log a warning for one that is left."""
    for remove in _REMOVERS:
        # RecursionError included: on Python 3.11 rmtree recurses once per directory level.
        with contextlib.suppress(Exception):
            remove(workspace)
        if not os.path.lexists(workspace):
            return
    _LOG.warning("could not remove workspace %s", workspace)
