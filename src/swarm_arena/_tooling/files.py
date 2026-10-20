"""Snapshot goal and component files into immutable, content-addressed manifests."""

import os
import stat
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from pydantic import AliasChoices, AliasPath, BaseModel
from pydantic_core import PydanticUndefined

from swarm_arena._model.compat import _alias_paths
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Plan, TeamSpec
from swarm_arena._model.ref import Configurable, Ref
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.paths import check_relpath, fold_conflicts, fold_key, matches
from swarm_arena._tooling.result_file import (
    _directory_fd,
    _inspect_path,
    read_regular_file,
)

MAX_SNAPSHOT_FILE_BYTES: Final = 2**30
"""Most bytes of one snapshotted goal, tool, check, or judge file (1 GiB)."""


def _listed_path(base: Path, relative: str, listings: dict[Path, dict[str, list[str]]]) -> str:
    """Recover each component's directory-entry spelling using file identity.

    Each directory is listed once per snapshot: ``listings`` caches its entry
    names by folded key, so a snapshot stays linear in the number of entries.

    Parameters
    ----------
    base : Path
        Trusted source root.
    relative : str
        Glob result which may retain a literal pattern's alias spelling.
    listings : dict[Path, dict[str, list[str]]]
        Cache shared by one snapshot: listed directory to its entry names,
        grouped by ``fold_key``.

    Returns
    -------
    str
        Workspace-relative path using on-disk names for every component.

    Raises
    ------
    ValueError
        If no listed entry is the same file as an alias spelling.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     (Path(tmp) / "problem").mkdir()
    ...     (Path(tmp) / "problem/run.py").touch()
    ...     _listed_path(Path(tmp), "problem/run.py", {})
    'problem/run.py'
    """
    base = base.resolve()
    current = base
    for part in Path(relative).parts:
        if (names := listings.get(current)) is None:
            names = listings[current] = _listing(current)
        spellings = names.get(fold_key(part), [])
        if part in spellings:
            # An exact directory entry is authoritative on a case-sensitive volume.
            current = current / part
            continue
        target = _identity(current / part)
        listed = next((name for name in spellings if _identity(current / name) == target), None)
        if listed is None:
            raise ValueError(f"cannot find directory entry for path: {relative}")
        current = current / listed
    return current.relative_to(base).as_posix()


def _listing(directory: Path) -> dict[str, list[str]]:
    """List a directory's entry names, grouped by folded key."""
    names: dict[str, list[str]] = {}
    absolute = directory.absolute()
    with _directory_fd(Path(absolute.anchor), absolute.parts[1:]) as fd:
        for name in os.listdir(fd):
            names.setdefault(fold_key(name), []).append(name)
    return names


def _identity(path: Path) -> tuple[int, int]:
    """Return the device and inode of a path without following a final symlink."""
    absolute = path.absolute()
    _, info = _inspect_path(Path(absolute.anchor), absolute.relative_to(absolute.anchor).as_posix())
    if info is None:
        raise FileNotFoundError
    return info.st_dev, info.st_ino


def _glob_files(
    base: Path, pattern: str, listings: dict[Path, dict[str, list[str]]]
) -> Iterator[str]:
    """Expand glob segments using the shared descriptor inspection/listing helpers."""
    parts = PurePosixPath(pattern).parts
    if not parts:
        return
    base = base.resolve()
    pending = [("", 0)]
    seen: set[tuple[str, int]] = set()
    while pending:
        parent, index = pending.pop()
        if (parent, index) in seen:
            continue
        seen.add((parent, index))
        segment = parts[index]
        recursive = segment == "**"
        if recursive and index + 1 < len(parts):
            pending.append((parent, index + 1))
        if any(char in segment for char in "*?["):
            directory = base / parent
            if (names := listings.get(directory)) is None:
                names = listings[directory] = _listing(directory)
            children = [
                name
                for spellings in names.values()
                for name in spellings
                if recursive or matches(name, [segment])
            ]
        else:
            # A literal can resolve through a filesystem lookup alias. Wildcards
            # use actual listed spellings and stay case sensitive.
            children = [segment]
        for name in sorted(children):
            relative = f"{parent}/{name}" if parent else name
            link, info = _inspect_path(base, relative)
            if link is not None:
                raise ValueError(f"symlink not allowed: {link}")
            if info is None:
                continue
            listed = _listed_path(base, relative, listings)
            if stat.S_ISDIR(info.st_mode):
                if recursive:
                    pending.append((listed, index))
                elif index + 1 < len(parts):
                    pending.append((listed, index + 1))
            elif (
                stat.S_ISREG(info.st_mode) and index == len(parts) - 1 and not pattern.endswith("/")
            ):
                yield check_relpath(listed)


def snapshot_fileset(fs: FileSet, base_dir: Path, artifacts: ArtifactStore) -> FileSet:
    """Store matching regular files and return a sorted path-to-hash manifest.

    Already snapshotted sets and sets with no globs are returned unchanged,
    without accessing source files. Each glob must match at least one regular
    file. Symlinks below the base directory and reserved top-level paths fail.
    A glob spelling that only a case-insensitive lookup resolves, such as
    ``problem/RUN.py`` for a listed ``run.py``, is recorded once with the
    listed spelling. Names that both appear in a directory listing are two
    entries, even when they are hard links to one file, so a folded pair of
    them is refused. Each directory is listed once per snapshot. Each file
    may be at most ``MAX_SNAPSHOT_FILE_BYTES`` (1 GiB); a larger one is
    refused by its size, unread, with an error that names its path.

    Parameters
    ----------
    fs : FileSet
        Source globs or an existing snapshot.
    base_dir : Path
        Trusted root against which relative globs are expanded.
    artifacts : ArtifactStore
        Content-addressed store receiving the selected bytes.

    Returns
    -------
    FileSet
        Original globs with a sorted manifest of SHA-256 digests.

    Raises
    ------
    ValueError
        If a glob matches no regular file, a selected path is reserved, not
        valid UTF-8, or symlinked, two selected names conflict after folding,
        a file is larger than ``MAX_SNAPSHOT_FILE_BYTES``
        (``<path> crosses the size cap``), or the source cannot be read.

    Examples
    --------
    >>> import tempfile
    >>> from swarm_arena._store.artifacts import ArtifactStore
    >>> from swarm_arena._store.base import open_store
    >>> from swarm_arena.tool import FileSet
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     artifacts = ArtifactStore(open_store(f"{tmp}/artifacts"))
    ...     (Path(tmp) / "problem").mkdir()
    ...     size = (Path(tmp) / "problem/run.py").write_text("print(1)")
    ...     files = snapshot_fileset(FileSet(globs=["problem/*"]), Path(tmp), artifacts)
    >>> list(files.manifest), files.snapshotted
    (['problem/run.py'], True)
    """
    if fs.manifest or not fs.globs:
        return fs
    listings: dict[Path, dict[str, list[str]]] = {}
    selected: dict[str, None] = {}
    for glob in fs.globs:
        found = False
        try:
            for relative in _glob_files(base_dir, glob, listings):
                # Actual listed names stay distinct, even as hard links, while
                # lookup aliases dedupe through the shared spelling helper.
                selected[relative] = None
                found = True
        except OSError:
            raise ValueError(f"cannot snapshot glob {glob!r}") from None
        if not found:
            raise ValueError(f"glob {glob!r} matched no file under {base_dir}")
    paths = sorted(selected)
    conflicts = fold_conflicts(paths)
    if conflicts:
        first, second = conflicts[0]
        raise ValueError(f"conflicting snapshot paths: {first} and {second}")
    manifest: dict[str, str] = {}
    for rel in paths:  # one file at a time, each refused by its size when over the cap
        data = read_regular_file(base_dir, rel, max_bytes=MAX_SNAPSHOT_FILE_BYTES)
        manifest[rel] = artifacts.put_bytes(data)
    return fs.model_copy(update={"manifest": manifest})


def _snapshot_value(value: Any, raw: Any, base_dir: Path, artifacts: ArtifactStore) -> Any:
    """Freeze typed files and refs inside containers without coercing raw scalars."""
    if isinstance(value, FileSet):
        return snapshot_fileset(value, base_dir, artifacts).model_dump(mode="json")
    if isinstance(value, Ref):
        return snapshot_ref(value, base_dir, artifacts).model_dump(mode="json", exclude_none=True)
    if isinstance(value, BaseModel) and isinstance(raw, dict):
        return _snapshot_config(value, raw, base_dir, artifacts)
    if isinstance(value, Mapping) and isinstance(raw, dict):
        return {
            key: _snapshot_value(value[key], child, base_dir, artifacts) if key in value else child
            for key, child in raw.items()
        }
    if isinstance(value, (list, tuple)) and isinstance(raw, list):
        return [
            _snapshot_value(item, child, base_dir, artifacts)
            for item, child in zip(value, raw, strict=True)
        ]
    return raw


def snapshot_ref(ref: Ref, base_dir: Path, artifacts: ArtifactStore) -> Ref:
    """Snapshot explicit file sets and nested component references in a config.

    Validate against the referenced component's Config without constructing the
    component. Only file-set and reference fields present in the raw config are
    written back; nonempty FileSet defaults are frozen under their canonical
    field names. Empty FileSet defaults and other defaults stay omitted.
    Typed reference mappings retain their keys and unrelated values stay raw.
    Validation aliases retain their original input paths and precedence.
    Nested references omit absent model fields, and the returned reference is
    validated so its configuration remains reloadable from JSON and TOML.

    Parameters
    ----------
    ref : Ref
        Reference to a Configurable subclass.
    base_dir : Path
        Trusted root for the component's relative file globs.
    artifacts : ArtifactStore
        Store receiving component files, including nested references' files.

    Returns
    -------
    Ref
        Copy preserving path, name, and all unrelated raw settings.

    Raises
    ------
    TypeError
        If the referenced object is not a Configurable subclass.
    ValueError
        If importing, validating configuration, or snapshotting files fails.
        ConfigError diagnostics omit values, validator messages, and data keys.
        Only built-in Pydantic error types are shown; custom types and other
        validation exceptions become ``invalid`` with their chains suppressed.

    Examples
    --------
    >>> import tempfile
    >>> from swarm_arena import Ref
    >>> from swarm_arena._store.base import open_store
    >>> ref = Ref(path="swarm_arena.scoring:Lexicographic", config={"judges": [
    ...     {"path": "swarm_arena:ScriptJudge", "config": {"command": ["score"]}}
    ... ]})
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     artifacts = ArtifactStore(open_store(f"{tmp}/artifacts"))
    ...     frozen = snapshot_ref(ref, Path(tmp), artifacts)
    >>> Ref.model_validate_json(frozen.model_dump_json()) == frozen
    True
    """
    _, config = ref._resolve_config(Configurable)
    updated = _snapshot_config(config, ref.config, base_dir, artifacts)
    return Ref(path=ref.path, config=updated, name=ref.name)


def _snapshot_config(
    config: BaseModel, raw: dict[str, Any], base_dir: Path, artifacts: ArtifactStore
) -> dict[str, Any]:
    """Write frozen typed values back along their supplying raw input paths."""
    updated = deepcopy(raw)
    settings = type(config).model_config
    for key, field in type(config).model_fields.items():
        alias = field.validation_alias
        paths = (
            _alias_paths(
                alias.convert_to_aliases()
                if isinstance(alias, (AliasPath, AliasChoices))
                else alias
            )
            if settings.get("validate_by_alias", True)
            else []
        )
        if alias is None or settings.get(
            "validate_by_name",
            settings.get("populate_by_name", False) or not settings.get("validate_by_alias", True),
        ):
            paths.append((key,))
        path = next(
            (
                path
                for path in paths
                if AliasPath(cast(str, path[0]), *path[1:]).search_dict_for_path(raw)
                is not PydanticUndefined
            ),
            None,
        )
        value = getattr(config, key)
        if path is None:
            if not isinstance(value, FileSet) or not value.globs:
                continue
            path = (key,)
        target: Any = updated
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = _snapshot_value(
            value,
            target.get(path[-1]) if isinstance(target, dict) else target[path[-1]],
            base_dir,
            artifacts,
        )
    return updated


def _snapshot_team(team: TeamSpec, base_dir: Path, artifacts: ArtifactStore) -> TeamSpec:
    """Copy a team's agents with both tool and manager files snapshotted."""
    return team.model_copy(
        update={
            "agents": [
                agent.model_copy(
                    update={
                        "tool": snapshot_ref(agent.tool, base_dir, artifacts),
                        "manager": snapshot_ref(agent.manager, base_dir, artifacts),
                    }
                )
                for agent in team.agents
            ]
        }
    )


def snapshot_plan(plan: Plan, artifacts: ArtifactStore) -> Plan:
    """Snapshot goal, part, and component files throughout a plan.

    Resolve all source globs against the plan's base directory, including the
    convergence team's tools and managers. Preserve that private base directory
    and distinguish inherited part checks and judges from explicit empty lists.

    Parameters
    ----------
    plan : Plan
        Plan to copy with reproducible file identities.
    artifacts : ArtifactStore
        Store receiving all files selected by the plan.

    Returns
    -------
    Plan
        Copied plan with snapshotted goals and component references.

    Raises
    ------
    TypeError
        If a reference does not identify a Configurable subclass.
    ValueError
        If a reference or any file selection cannot be snapshotted.

    Examples
    --------
    >>> from swarm_arena import Plan
    >>> plan = Plan.load("plan.toml")  # doctest: +SKIP
    >>> frozen = snapshot_plan(plan, artifacts)  # doctest: +SKIP
    >>> frozen.base_dir == plan.base_dir  # doctest: +SKIP
    True
    """
    base_dir = plan.base_dir
    parts = [
        part.model_copy(
            update={
                "goal": part.goal.model_copy(
                    update={"files": snapshot_fileset(part.goal.files, base_dir, artifacts)}
                ),
                "checks": None
                if part.checks is None
                else [snapshot_ref(ref, base_dir, artifacts) for ref in part.checks],
                "judges": None
                if part.judges is None
                else [snapshot_ref(ref, base_dir, artifacts) for ref in part.judges],
            }
        )
        for part in plan.parts
    ]
    converge = plan.converge
    if converge.team is not None:
        converge = converge.model_copy(
            update={"team": _snapshot_team(converge.team, base_dir, artifacts)}
        )
    return plan.model_copy(
        update={
            "goal": plan.goal.model_copy(
                update={"files": snapshot_fileset(plan.goal.files, base_dir, artifacts)}
            ),
            "parts": parts,
            "teams": [_snapshot_team(team, base_dir, artifacts) for team in plan.teams],
            "checks": [snapshot_ref(ref, base_dir, artifacts) for ref in plan.checks],
            "judges": [snapshot_ref(ref, base_dir, artifacts) for ref in plan.judges],
            "policy": snapshot_ref(plan.policy, base_dir, artifacts),
            "converge": converge,
        }
    )
