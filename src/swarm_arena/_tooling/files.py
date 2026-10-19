"""Snapshot goal and component files into immutable, content-addressed manifests."""

from pathlib import Path

from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Plan, TeamSpec
from swarm_arena._model.ref import Configurable, Ref, _validate_config
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.paths import check_relpath, fold_conflicts, fold_key, matches
from swarm_arena._tooling.result_file import read_regular_file, symlinked_component


def _listed_path(base: Path, relative: str) -> str:
    """Recover each component's directory-entry spelling using file identity.

    Parameters
    ----------
    base : Path
        Trusted source root.
    relative : str
        Glob result which may retain a literal pattern's alias spelling.

    Returns
    -------
    str
        Workspace-relative path using on-disk names for every component.

    Examples
    --------
    >>> _listed_path(Path("source"), "problem/RUN.py")  # doctest: +SKIP
    'problem/run.py'
    """
    current = base
    for part in Path(relative).parts:
        target = (current / part).stat(follow_symlinks=False)
        entries = list(current.iterdir())
        # An exact directory entry is authoritative on a case-sensitive volume.
        exact = next((entry for entry in entries if entry.name == part), None)
        if exact is not None:
            current = exact
            continue
        for entry in entries:
            if fold_key(entry.name) == fold_key(part):
                info = entry.stat(follow_symlinks=False)
                if (info.st_dev, info.st_ino) == (target.st_dev, target.st_ino):
                    current = entry
                    break
        else:
            raise ValueError(f"cannot find directory entry for path: {relative}")
    return current.relative_to(base).as_posix()


def snapshot_fileset(fs: FileSet, base_dir: Path, artifacts: ArtifactStore) -> FileSet:
    """Store matching regular files and return a sorted path-to-hash manifest.

    Already snapshotted sets and sets with no globs are returned unchanged,
    without accessing source files. Each glob must match at least one regular
    file. Symlinks below the base directory and reserved top-level paths fail.

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
        If a glob matches no regular file, a selected path is reserved or
        symlinked, or the source cannot be read.

    Examples
    --------
    >>> from swarm_arena.tool import FileSet
    >>> files = snapshot_fileset(FileSet(), Path("missing"), artifacts)  # doctest: +SKIP
    >>> files.snapshotted  # doctest: +SKIP
    True
    """
    if fs.manifest or not fs.globs:
        return fs
    selected: dict[str, tuple[int, int]] = {}
    paths: list[str] = []
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
                rel = check_relpath(_listed_path(base_dir, relative))
                found = True
                info = path.stat(follow_symlinks=False)
                identity = (info.st_dev, info.st_ino)
                key = fold_key(rel)
                previous = selected.get(key)
                if previous is None or previous != identity:
                    # Only aliases of the same file are deduplicated. Keep
                    # distinct identities for the shared layout checker below.
                    paths.append(rel)
                selected.setdefault(key, identity)
        except OSError:
            raise ValueError(f"cannot snapshot glob {glob!r}") from None
        if not found:
            raise ValueError(f"glob {glob!r} matched no file")
    paths.sort()
    conflicts = fold_conflicts(paths)
    if conflicts:
        first, second = conflicts[0]
        raise ValueError(f"conflicting snapshot paths: {first} and {second}")
    manifest = {rel: artifacts.put_bytes(read_regular_file(base_dir, rel)) for rel in paths}
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
