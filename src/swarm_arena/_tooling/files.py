"""Snapshot goal and component files into immutable, content-addressed manifests."""

from pathlib import Path
from typing import Final

from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Plan, TeamSpec
from swarm_arena._model.ref import Configurable, Ref, _validate_config
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.paths import check_relpath, fold_conflicts, fold_key, matches
from swarm_arena._tooling.result_file import read_regular_file, symlinked_component

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
    for entry in directory.iterdir():
        names.setdefault(fold_key(entry.name), []).append(entry.name)
    return names


def _identity(path: Path) -> tuple[int, int]:
    """Return the device and inode of a path without following a final symlink."""
    info = path.stat(follow_symlinks=False)
    return info.st_dev, info.st_ino


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
            for path in sorted(base_dir.glob(glob)):
                relative = path.relative_to(base_dir).as_posix()
                if not matches(relative, [glob]):
                    continue
                if (link := symlinked_component(base_dir, relative)) is not None:
                    raise ValueError(f"symlink not allowed: {link}")
                if not path.is_file():
                    continue
                check_relpath(relative)
                # A lookup alias resolves to its listed name, so it dedupes with
                # it. Two listed names stay two entries, even as hard links, and
                # the shared layout checker below refuses a folded pair.
                selected[check_relpath(_listed_path(base_dir, relative, listings))] = None
                found = True
        except OSError:
            raise ValueError(f"cannot snapshot glob {glob!r}") from None
        if not found:
            raise ValueError(f"glob {glob!r} matched no file")
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


def snapshot_ref(ref: Ref, base_dir: Path, artifacts: ArtifactStore) -> Ref:
    """Snapshot explicit file sets and nested component references in a config.

    Validate against the referenced component's Config without constructing the
    component. Only file-set and reference fields present in the raw config are
    written back; omitted defaults stay omitted and other values stay raw.

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
    >>> from swarm_arena import Ref
    >>> ref = Ref(path="my_package.tools:MyTool", config={"files": ["tool/*"]})
    >>> frozen = snapshot_ref(ref, Path("."), artifacts)  # doctest: +SKIP
    """
    try:
        cls = ref.load()
    except TypeError:
        raise TypeError(f"{ref.path} is not a Configurable class") from None
    except Exception:
        # Import hooks can include arbitrary module or config values in errors.
        raise ValueError(f"cannot import {ref.path}") from None
    if not issubclass(cls, Configurable):
        raise TypeError(f"{ref.path} is not a Configurable subclass")
    config = _validate_config(cls.Config, ref.config, where=ref.path)
    updated = dict(ref.config)
    for key in type(config).model_fields:
        if key not in ref.config:
            continue
        value = getattr(config, key)
        if isinstance(value, FileSet):
            updated[key] = snapshot_fileset(value, base_dir, artifacts).model_dump(mode="json")
        elif isinstance(value, Ref):
            updated[key] = snapshot_ref(value, base_dir, artifacts).model_dump(mode="json")
        elif isinstance(value, list) and value and all(isinstance(item, Ref) for item in value):
            updated[key] = [
                snapshot_ref(item, base_dir, artifacts).model_dump(mode="json") for item in value
            ]
    return ref.model_copy(update={"config": updated})


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
