"""Materialize validated stages and collect regular output files safely."""

import contextlib
import json
import os
import secrets
import stat
from collections.abc import Generator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from swarm_arena._exec.executor import Job
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.tooling import FileRef, check_parent_id
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.files import _listed_path
from swarm_arena._tooling.paths import (
    PARENT_DATA,
    check_key,
    fold_conflicts,
    fold_key,
    output_paths,
    parent_tree,
)
from swarm_arena._tooling.result_file import read_regular_file, symlinked_component


def _json_bytes(value: dict[str, Any], rel: str, *, compact: bool = False) -> bytes:
    """Serialize metadata without disclosing arbitrary values in an exception."""
    try:
        if compact:
            return canonical_json(value)
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    except Exception:
        raise ValueError(f"cannot serialize workspace path: {rel}") from None


@dataclass(frozen=True)
class _Layout:
    """Every destination of one stage, planned once for materialize and collection."""

    files: dict[str, tuple[str, bool]]
    """Artifact digest and read-only flag per workspace path."""
    metadata: dict[str, tuple[dict[str, Any], bool]]
    """Generated JSON value and compact-encoding flag per workspace path."""
    directories: frozenset[str]
    """Generated directories, created even when empty."""


def _stage_layout(stage: Job.Stage) -> _Layout:
    """Validate every input path, then plan all destinations without any I/O."""
    for path in stage.root:
        check_key(path)
    ids: set[str] = set()
    for parent in stage.parents:
        check_parent_id(parent.candidate_id)
        key = fold_key(parent.candidate_id)
        if key in ids:
            raise ValueError(f"duplicate parent id: {parent.candidate_id}")
        ids.add(key)
        for path in parent.overlay:
            check_key(path)
        for file in parent.files:
            check_key(file.path)
    if stage.candidate is not None:
        for path in stage.candidate.files:
            check_key(path)
    files = {path: (digest, False) for path, digest in stage.root.items()}
    metadata: dict[str, tuple[dict[str, Any], bool]] = {}
    directories = {"swarm"}
    for parent in stage.parents:
        prefix = f"parents/{parent.candidate_id}/"
        outputs = {file.path: file.sha256 for file in parent.files}
        tree = parent_tree(parent.overlay, outputs, prefix=prefix)
        files.update({prefix + path: (digest, True) for path, digest in tree.items()})
        metadata[prefix + PARENT_DATA] = (parent.data, True)
    if stage.candidate is not None:
        candidate = stage.candidate
        directories.add("candidate/files")
        files.update(
            {f"candidate/files/{path}": (digest, True) for path, digest in candidate.files.items()}
        )
        metadata["candidate/request.json"] = (candidate.request, False)
        metadata["candidate/output.json"] = (candidate.output, False)
    if stage.request_json is not None:
        metadata["swarm/request.json"] = (stage.request_json, False)
    return _Layout(files=files, metadata=metadata, directories=frozenset(directories))


def _check_layout(workspace: Path, files: AbstractSet[str], directories: AbstractSet[str]) -> None:
    """Reject conflicting paths and existing symlinks before creating anything."""
    conflicts = fold_conflicts(sorted(files), directories=sorted(directories))
    if conflicts:
        first, second = conflicts[0]
        raise ValueError(f"conflicting workspace path: {first} and {second}")
    if workspace.is_symlink():
        raise ValueError("symlink not allowed: workspace path")
    for rel in sorted(files | directories):
        try:
            if symlinked_component(workspace, rel) is not None:
                raise ValueError(f"symlink not allowed: {rel}")
            path = workspace / rel
            if path.exists():
                mode = path.lstat().st_mode
                expected = stat.S_ISREG(mode) if rel in files else stat.S_ISDIR(mode)
                if not expected:
                    raise ValueError(f"invalid workspace path: {rel}")
        except OSError:
            raise ValueError(f"cannot inspect workspace path: {rel}") from None


@contextlib.contextmanager
def _directory(root: int, parts: tuple[str, ...]) -> Generator[int, None, None]:
    """Create and pin descendant directories without following symlinks."""
    with contextlib.ExitStack() as stack:
        fd = root
        for part in parts:
            with contextlib.suppress(FileExistsError):
                os.mkdir(part, dir_fd=fd)
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            stack.callback(os.close, fd)
        yield fd


def _write(root: int, rel: str, data: bytes, *, read_only: bool) -> None:
    """Publish through pinned directories without following or truncating links."""
    parts = PurePosixPath(rel).parts
    try:
        with _directory(root, parts[:-1]) as directory:
            try:
                mode = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False).st_mode
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(mode):
                    raise ValueError(f"not a regular workspace path: {rel}")
            temp = f".stage-{secrets.token_hex(16)}"
            fd = os.open(
                temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
            )
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fchmod(stream.fileno(), 0o444 if read_only else 0o644)
                os.replace(temp, parts[-1], src_dir_fd=directory, dst_dir_fd=directory)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temp, dir_fd=directory)
    except OSError:
        raise ValueError(f"cannot write workspace path: {rel}") from None


def materialize(stage: Job.Stage, workspace: Path, artifacts: ArtifactStore) -> None:
    """Validate a stage again and write its workspace with explicit permissions.

    All input paths and parent ids are checked before any writes, including
    objects created through validator-bypassing model methods. Reject case and
    Unicode destination collisions, including ancestors, before any writes.
    Read verified artifact bytes and publish through pinned directory
    descriptors, so existing or swapped symlinks cannot redirect writes.
    Workspace ancestors are trusted executor-owned directories.

    Parameters
    ----------
    stage : Job.Stage
        Root, parent, candidate, and request contents to materialize.
    workspace : Path
        Executor-owned workspace root; the root itself cannot be a symlink.
    artifacts : ArtifactStore
        Store providing verified artifact bytes.

    Raises
    ------
    ValueError
        If paths, parent ids, metadata, or destination file types are invalid.
        Merge and layout errors name full workspace paths, such as
        ``parents/c1/p/a.txt``, so they show which parent or tree failed.

    Examples
    --------
    >>> import tempfile
    >>> from swarm_arena._store.artifacts import ArtifactStore
    >>> from swarm_arena._store.base import open_store
    >>> from swarm_arena.executor import Job
    >>> from swarm_arena.tool import ParentRef
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     artifacts = ArtifactStore(open_store(f"{tmp}/artifacts"))
    ...     parent = ParentRef(candidate_id="c1", run_id="r", tool_identity="T",
    ...                        data={"loss": 0.5}, overlay={"p/a.txt": artifacts.put_bytes(b"A")})
    ...     materialize(Job.Stage(parents=[parent]), Path(tmp) / "ws", artifacts)
    ...     sorted(p.name for p in (Path(tmp) / "ws/parents/c1").iterdir())
    ['data.json', 'p']
    """
    layout = _stage_layout(stage)
    files, directories = layout.files, layout.directories
    metadata = {
        rel: _json_bytes(value, rel, compact=compact)
        for rel, (value, compact) in layout.metadata.items()
    }
    _check_layout(workspace, files.keys() | metadata.keys(), directories)
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        root = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        raise ValueError("cannot open workspace path") from None
    with contextlib.ExitStack() as stack:
        stack.callback(os.close, root)
        for rel in sorted(directories):
            try:
                with _directory(root, PurePosixPath(rel).parts):
                    pass
            except OSError:
                raise ValueError(f"cannot create workspace path: {rel}") from None
        for rel, (digest, read_only) in files.items():
            data = artifacts.get_bytes(digest)
            _write(root, rel, data, read_only=read_only)
        for rel, data in metadata.items():
            _write(root, rel, data, read_only=rel.startswith(("parents/", "candidate/")))


def collect_output_files(
    workspace: Path,
    declared: Sequence[str],
    artifacts: ArtifactStore,
    *,
    stage: Job.Stage,
    max_bytes: int,
) -> list[FileRef]:
    """Upload declared regular outputs after checking paths and symlinks.

    Declarations are checked, before any read, against the same planned layout
    that ``materialize`` wrote, and against the later parent tree they join.
    Files are read and uploaded one at a time. Each file's size is checked
    before it is read, against what is left of ``max_bytes``, so a file over
    the cap (a sparse one too) is never read. A declaration must use the
    spelling of the directory listing, as snapshots do: on a case- or
    normalization-insensitive disk, ``results/out.txt`` also opens
    ``Results/OUT.txt``, but a case-sensitive disk would not find it.

    Parameters
    ----------
    workspace : Path
        Trusted execution workspace root.
    declared : Sequence[str]
        Relative output paths; aliases normalize before collection.
    artifacts : ArtifactStore
        Store receiving output contents.
    stage : Job.Stage
        Original materialized stage, including the effective root, parents,
        candidate files, and generated metadata. Required even for an empty stage.
    max_bytes : int
        Most bytes of all declared files together, such as
        ``LocalExecutor.Config.max_output_bytes``.

    Returns
    -------
    list[FileRef]
        References with normalized paths, basenames, hashes, and byte counts.

    Raises
    ------
    ValueError
        If a path is invalid, duplicated after normalization and folding, conflicts
        with the staged layout or with a parent tree's ``data.json``, missing,
        spelled another way on disk, symlinked, nonregular, crosses
        ``max_bytes``, or cannot be collected.

    Examples
    --------
    >>> import tempfile
    >>> from swarm_arena._store.artifacts import ArtifactStore
    >>> from swarm_arena._store.base import open_store
    >>> stage = Job.Stage(root={"p/run.py": "1" * 64})
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     artifacts = ArtifactStore(open_store(f"{tmp}/artifacts"))
    ...     size = (Path(tmp) / "out.txt").write_text("x")
    ...     [ref] = collect_output_files(
    ...         Path(tmp), ["./out.txt"], artifacts, stage=stage, max_bytes=2**20)
    ...     (ref.path, ref.size)
    ('out.txt', 1)
    >>> collect_output_files(Path("ws"), ["P/out.txt"], artifacts, stage=stage, max_bytes=1)
    Traceback (most recent call last):
    ...
    ValueError: conflicting output path with staged layout: p/run.py and P/out.txt
    """
    layout = _stage_layout(stage)
    paths = output_paths(
        declared,
        effective_files=[*layout.files, *layout.metadata],
        directories=sorted(layout.directories),
    )
    output: list[FileRef] = []
    left = max_bytes
    listings: dict[Path, dict[str, list[str]]] = {}
    for rel in paths:
        try:
            if symlinked_component(workspace, rel) is not None:
                raise ValueError(f"symlink not allowed: {rel}")
            (workspace / rel).lstat()
            listed = _listed_path(workspace, rel, listings)
        except FileNotFoundError:
            raise ValueError(f"{rel} is missing") from None
        except OSError:
            raise ValueError(f"cannot inspect output path: {rel}") from None
        if listed != rel:  # a case-sensitive disk would not find it (G6, as for snapshots)
            raise ValueError(f"declared output {rel} is spelled {listed} on disk")
        data = read_regular_file(workspace, rel, max_bytes=left)
        left -= len(data)
        try:
            digest = artifacts.put_bytes(data)
        except Exception:
            raise ValueError(f"cannot store output path: {rel}") from None
        output.append(
            FileRef(name=PurePosixPath(rel).name, path=rel, sha256=digest, size=len(data))
        )
    return output
