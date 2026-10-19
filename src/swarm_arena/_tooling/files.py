"""Snapshot goal and component files into immutable, content-addressed manifests."""

from pathlib import Path

from pydantic import ValidationError

from swarm_arena._errors import ConfigError
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Plan, TeamSpec
from swarm_arena._model.ref import Configurable, Ref, _safe_config_location
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.paths import check_relpath, matches
from swarm_arena._tooling.result_file import read_regular_file, symlinked_component


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
    manifest: dict[str, str] = {}
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
                rel = check_relpath(relative)
                found = True
                if rel not in manifest:
                    manifest[rel] = artifacts.put_bytes(read_regular_file(base_dir, rel))
        except OSError:
            raise ValueError(f"cannot snapshot glob {glob!r}") from None
        if not found:
            raise ValueError(f"glob {glob!r} matched no file")
    return fs.model_copy(update={"manifest": dict(sorted(manifest.items()))})


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
    try:
        config = cls.Config.model_validate(ref.config)
    except ValidationError as err:
        errors = err.errors(include_input=False, include_url=False, include_context=False)
        summary = "; ".join(
            f"{_safe_config_location(cls.Config, error['loc'])}: {error['type']}"
            for error in errors
        )
        raise ConfigError(f"{ref.path}: {summary}") from None
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
