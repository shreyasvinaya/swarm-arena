"""Plan workspace contents and preserve the receiving tool's edit boundary."""

from collections.abc import Collection, Mapping, Sequence
from collections.abc import Set as AbstractSet
from pathlib import PurePosixPath
from typing import Any

from swarm_arena._errors import ConfigError, SoftError, _core_signal
from swarm_arena._exec.executor import Job
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.catalog import CatalogEntry
from swarm_arena._model.goal import Goal
from swarm_arena._model.migrate import CURRENT
from swarm_arena._model.ref import Configurable, Ref
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import ParentRef, check_parent_id, is_stored_edit_ref
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.paths import (
    PARENT_DATA,
    _is_utf8,
    check_key,
    check_relpath,
    fold_conflicts,
    fold_key,
    fold_merge,
    matches,
    parent_data_conflicts,
)
from swarm_arena._tooling.result_file import request_json as _request_json


def fixed_config(ref: Ref, entry: CatalogEntry | None) -> dict[str, Any]:
    """Validate tool settings, expand defaults, and remove settable fields.

    Parameters
    ----------
    ref : Ref
        Component reference carrying fixed settings, without round overrides.
    entry : CatalogEntry or None
        Catalog entry identifying fields excluded from the fixed identity.

    Returns
    -------
    dict[str, Any]
        JSON-compatible validated configuration without settable fields.

    Examples
    --------
    >>> from swarm_arena import Ref
    >>> fixed_config(Ref(path="swarm_arena._model.ref:Configurable"), None)
    {}
    """
    _, config = ref._resolve_config(Configurable)
    try:
        result = config.model_dump(mode="json")
    except Exception:
        raise ValueError(f"{ref.path}: cannot serialize config") from None
    return {
        key: value for key, value in result.items() if entry is None or key not in entry.settable
    }


def tool_identity(ref: Ref, entry: CatalogEntry | None) -> str:
    """Hash the reference path and canonical fixed configuration.

    Parameters
    ----------
    ref : Ref
        Tool reference whose defaults are expanded before hashing.
    entry : CatalogEntry or None
        Optional catalog entry specifying settable fields to exclude.

    Returns
    -------
    str
        SHA-256 digest of the reference path and fixed configuration.

    Examples
    --------
    >>> from swarm_arena import Ref
    >>> len(tool_identity(Ref(path="swarm_arena._model.ref:Configurable"), None))
    64
    """
    config = fixed_config(ref, entry)
    try:
        return sha256_hex(canonical_json({"path": ref.path, "fixed_config": config}))
    except Exception:
        raise ValueError(f"{ref.path}: cannot encode tool identity") from None


def _shown(path: str) -> str:
    """Return a path for a message: as written, or its repr when it is not valid UTF-8."""
    return path if _is_utf8(path) else repr(path)


def _layout_collisions(paths: Collection[str], root: AbstractSet[str]) -> dict[str, str]:
    """Report new paths the sink would refuse, in this root or a later parent tree."""
    errors: dict[str, str] = {}
    for first, second in fold_conflicts(dict.fromkeys([*sorted(root), *paths])):
        for path, other in ((first, second), (second, first)):
            if path in paths:
                errors.setdefault(path, f"folded path collision with {_shown(other)}")
    for path in parent_data_conflicts(paths):
        errors.setdefault(path, f"folded path collision with parent {PARENT_DATA}")
    return errors


def validate_edits(
    edits: Mapping[str, str],
    *,
    editable: Sequence[str],
    protected: AbstractSet[str],
    max_edit_bytes: int,
    root: AbstractSet[str] = frozenset(),
) -> None:
    """Report the first refusal reason for every invalid live edit.

    Parameters
    ----------
    edits : Mapping[str, str]
        Normal-form workspace paths mapped to new text contents.
    editable : Sequence[str]
        Receiving tool's editable globs.
    protected : AbstractSet[str]
        Tool scripts, interpreters, and private paths, normalized and folded
        for case- and Unicode-insensitive protection checks.
    max_edit_bytes : int
        Maximum UTF-8 byte count per edit.
    root : AbstractSet[str], optional
        Existing workspace paths in their stored spelling. Supply the effective
        root to refuse differently spelled aliases before planning a stage.

    Raises
    ------
    SoftError
        If any path, protection, collision, glob, size, or stored-reference rule fails.
        Collisions include folded aliases of each other or of ``root``, and a
        path below a top-level directory that folds to ``data.json``, which the
        edit's later ``parents/<id>/`` tree reserves for the parent's data. A
        path that is not valid UTF-8 is shown by its ``repr``, so the message
        always encodes.
    ValueError
        If a protected path is invalid.

    Examples
    --------
    >>> validate_edits({"p/a.txt": "hi"}, editable=["p/*"], protected=set(), max_edit_bytes=10)
    >>> validate_edits({"Data.json/x": "hi"}, editable=["**"], protected=set(), max_edit_bytes=10)
    Traceback (most recent call last):
    ...
    swarm_arena._errors.SoftError: Data.json/x: folded path collision with parent data.json
    """
    protected = {fold_key(check_relpath(path)) for path in protected}
    collisions = _layout_collisions(edits, root)
    errors: list[str] = []
    for path, content in edits.items():
        reason = ""
        try:
            check_key(path)
        except ValueError as exc:
            reason = str(exc).removeprefix(f"path {path!r}: ")
        if not reason:
            if fold_key(path) in protected:
                reason = "protected"
            elif path in collisions:
                reason = collisions[path]
            elif not matches(path, editable):
                reason = "not editable"
            elif (size := len(content.encode())) > max_edit_bytes:
                reason = f"{size} bytes > max_edit_bytes {max_edit_bytes}"
            elif is_stored_edit_ref(content):
                reason = "content is a stored-edit reference"
        if reason:
            errors.append(f"{_shown(path)}: {reason}")
    if errors:
        raise _core_signal(SoftError("\n".join(errors)))


def _check_live_edits(edits: Mapping[str, str]) -> None:
    """Refuse alias keys and stored references before hashing or storing anything."""
    for path, content in edits.items():
        check_key(path)
        if is_stored_edit_ref(content):
            raise ValueError(f"{path}: content is a stored-edit reference")


def edit_hashes(edits: Mapping[str, str]) -> dict[str, str]:
    """Hash live edit text, refusing reserved stored-reference contents.

    Parameters
    ----------
    edits : Mapping[str, str]
        Normal-form paths mapped to live text, never stored references.

    Returns
    -------
    dict[str, str]
        Paths mapped to SHA-256 digests of UTF-8 contents.

    Examples
    --------
    >>> edit_hashes({"a.txt": "hello"}) == {"a.txt": sha256_hex(b"hello")}
    True
    """
    _check_live_edits(edits)
    return {path: sha256_hex(content.encode()) for path, content in edits.items()}


def put_edits(edits: Mapping[str, str], artifacts: ArtifactStore) -> None:
    """Validate all live edits before storing their UTF-8 contents.

    Parameters
    ----------
    edits : Mapping[str, str]
        Normal-form paths mapped to live text, never stored references.
    artifacts : ArtifactStore
        Store receiving each edit's contents.

    Examples
    --------
    >>> put_edits({"a.txt": "hello"}, artifacts)  # doctest: +SKIP
    """
    _check_live_edits(edits)
    for content in edits.values():
        artifacts.put_bytes(content.encode())


def parent_ref(candidate: Candidate) -> ParentRef:
    """Extract a parent's identity, output data, overlay, and output files.

    Parameters
    ----------
    candidate : Candidate
        Candidate to expose read-only to a subsequent run.

    Returns
    -------
    ParentRef
        Parent staging reference.

    Examples
    --------
    >>> parent = parent_ref(candidate)  # doctest: +SKIP
    """
    return ParentRef(
        candidate_id=candidate.id,
        run_id=candidate.run_id,
        tool_identity=candidate.tool_identity,
        data=candidate.output.data,
        overlay=candidate.overlay,
        files=candidate.output.files,
    )


def effective_files(goal: Goal, candidate: Candidate) -> dict[str, str]:
    """Combine a goal's snapshot with the candidate's full effective overlay.

    Parameters
    ----------
    goal : Goal
        Goal holding the original file manifest.
    candidate : Candidate
        Candidate holding its cumulative overlay.

    Returns
    -------
    dict[str, str]
        Goal files, replaced at exact paths by the candidate overlay.

    Raises
    ------
    ValueError
        If the paths conflict, including a respelling; the error names both
        paths under ``candidate/files/``.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> from swarm_arena._model.results import Candidate
    >>> from swarm_arena.tool import FileSet, RunStatus, ToolOutput, ToolRequest
    >>> goal = Goal(id="g", namespace="n", files=FileSet(globs=["p/*"],
    ...             manifest={"p/a.txt": "1" * 64, "p/b.txt": "1" * 64}))
    >>> candidate = Candidate(id="c1", run_id="r", goal_id="g", team="t", agent="a", round=1,
    ...                       request=ToolRequest(), output=ToolOutput(status=RunStatus.OK),
    ...                       overlay={"p/b.txt": "2" * 64})
    >>> effective_files(goal, candidate) == {"p/a.txt": "1" * 64, "p/b.txt": "2" * 64}
    True
    """
    _check_keys(goal.files.manifest)
    _check_keys(candidate.overlay)
    return fold_merge(goal.files.manifest, candidate.overlay, prefix="candidate/files/")


def _check_keys(files: Mapping[str, str]) -> None:
    """Validate all dictionary keys before any map merge can hide an alias."""
    for path in files:
        check_key(path)


def _respelled(first: str, second: str) -> bool:
    """Return whether two paths spell one folded path, or a shared ancestor, two ways."""
    for one, other in zip(PurePosixPath(first).parts, PurePosixPath(second).parts, strict=False):
        if fold_key(one) != fold_key(other):
            return False
        if one != other:
            return True
    return False


def tool_root(goal_files: Mapping[str, str], tool_files: Mapping[str, str]) -> dict[str, str]:
    """Build a tool's workspace root: the goal files, then the tool's own files.

    A tool file replaces a goal file only at the exact same spelling. A
    respelling (the same ``fold_key``, another spelling) or any other folded
    conflict is a configuration error, raised here before any run, so one
    workspace works on case-sensitive and case-insensitive filesystems. The
    goal files and the tool files are each checked alone first, so the error
    names the layer that conflicts; the respell hint is given only when a tool
    path respells a goal path.

    Parameters
    ----------
    goal_files : Mapping[str, str]
        Goal paths and artifact digests.
    tool_files : Mapping[str, str]
        Tool-owned paths and artifact digests.

    Returns
    -------
    dict[str, str]
        Root paths and artifact digests.

    Raises
    ------
    ConfigError
        ``goal files conflict: <a> and <b>`` or ``tool files conflict: <a> and
        <b>`` if one layer conflicts by itself; otherwise ``tool files do not
        fit the goal files: ...`` naming both paths.
    ValueError
        If a path is not in normal form.

    Examples
    --------
    >>> tool_root({"p/run.py": "1" * 64}, {"p/run.py": "2" * 64}) == {"p/run.py": "2" * 64}
    True
    >>> tool_root({"p/run.py": "1" * 64}, {"P/RUN.py": "2" * 64})
    Traceback (most recent call last):
    ...
    swarm_arena._errors.ConfigError: tool files do not fit the goal files: ...P/RUN.py; ...
    >>> tool_root({"p/a.txt": "1" * 64, "P/b.txt": "1" * 64}, {})
    Traceback (most recent call last):
    ...
    swarm_arena._errors.ConfigError: goal files conflict: p/a.txt and P/b.txt
    """
    _check_keys(goal_files)
    _check_keys(tool_files)
    for layer, files in (("goal", goal_files), ("tool", tool_files)):
        if conflicts := fold_conflicts(files):
            first, second = conflicts[0]
            raise ConfigError(f"{layer} files conflict: {first} and {second}")
    root = {**goal_files, **tool_files}
    if conflicts := fold_conflicts(root):
        first, second = conflicts[0]
        hint = "; spell a replaced goal file exactly as the goal does"
        raise ConfigError(
            f"tool files do not fit the goal files: conflicting workspace path: {first} and "
            f"{second}{hint if _respelled(first, second) else ''}"
        )
    return root


def plan_tool_stage(
    *,
    goal_files: Mapping[str, str],
    tool_files: Mapping[str, str],
    editable: Sequence[str],
    protected: AbstractSet[str],
    private_paths: AbstractSet[str],
    parents: Sequence[ParentRef],
    receiving_identity: str,
    edits: Mapping[str, str],
    request_json: dict[str, Any] | None,
) -> tuple[Job.Stage, dict[str, str], list[str]]:
    """Stage goal and tool files, an eligible base overlay, then live edits.

    Callers first run ``validate_edits`` with root paths and the same protection set. Recheck
    edit permissions here as an internal invariant. Only the first parent's
    overlay can enter the root, and only as a whole with a nonempty matching
    tool identity, when every path is editable here and fits the root layout
    as an exact-spelling replacement or a new path. Otherwise the overlay is
    refused with a soft warning. Every parent also remains available in its
    read-only tree. The root starts from ``tool_root``: a tool file replaces a
    goal file only at the exact same spelling.

    Parameters
    ----------
    goal_files : Mapping[str, str]
        Original goal paths and artifact digests.
    tool_files : Mapping[str, str]
        Tool-owned paths and artifact digests.
    editable : Sequence[str]
        Receiving tool's editable globs.
    protected : AbstractSet[str]
        Protected script, interpreter, and private paths.
    private_paths : AbstractSet[str]
        Check and judge paths forbidden in the tool root.
    parents : Sequence[ParentRef]
        Ordered parents, first being the proposed base.
    receiving_identity : str
        Receiving tool's fixed identity.
    edits : Mapping[str, str]
        Validated new text contents.
    request_json : dict[str, Any] or None
        Request document for the tool workspace.

    Returns
    -------
    tuple[Job.Stage, dict[str, str], list[str]]
        Stage, cumulative effective overlay, and soft warnings.

    Raises
    ------
    RuntimeError
        If private files enter the root or edits violate receiving permissions.
    ConfigError
        If the tool files do not fit the goal files, as in ``tool_root``.
    ValueError
        If a path is invalid, noncanonical, or an edit is a stored reference.
    SoftError
        If edits collide by folded key with each other or the effective root,
        or lie below a top-level directory that folds to ``data.json``.

    Examples
    --------
    >>> from swarm_arena.tool import ParentRef
    >>> base = ParentRef(candidate_id="c1", run_id="r", tool_identity="T",
    ...                  overlay={"P/notes.txt": "2" * 64})
    >>> stage, overlay, warnings = plan_tool_stage(
    ...     goal_files={"p/run.py": "1" * 64}, tool_files={"p/run.py": "3" * 64},
    ...     editable=["**/*"], protected={"p/run.py"}, private_paths=set(), parents=[base],
    ...     receiving_identity="T", edits={}, request_json=None)
    >>> stage.root == {"p/run.py": "3" * 64}, overlay
    (True, {})
    >>> warnings
    ['base parent c1 not overlaid: paths conflict with the workspace layout; paths: P/notes.txt']
    """
    _check_keys(edits)
    protected = {fold_key(check_relpath(path)) for path in protected}
    private_paths = {fold_key(check_relpath(path)) for path in private_paths}

    def editable_here(path: str) -> bool:
        """Apply the same receiving permissions to edits and base overlays."""
        key = fold_key(path)
        return key not in protected and key not in private_paths and matches(path, editable)

    for parent in parents:
        check_parent_id(parent.candidate_id)
        _check_keys(parent.overlay)
        for file in parent.files:
            check_key(file.path)
    root = tool_root(goal_files, tool_files)
    for path in root:
        if fold_key(path) in private_paths:
            raise RuntimeError(f"private path in tool root: {path}")
    for path in edits:
        if not editable_here(path):
            raise RuntimeError(f"edit path not editable here: {path}")
    overlay: dict[str, str] = {}
    warnings: list[str] = []
    if parents and parents[0].overlay:
        base = parents[0]
        reason = ""
        refused = sorted(base.overlay)
        if not base.tool_identity:
            reason = "has no tool identity"
        elif not receiving_identity or base.tool_identity != receiving_identity:
            reason = "made by another tool"
        else:
            refused = sorted(path for path in base.overlay if not editable_here(path))
            if refused:
                reason = "paths not editable here"
            elif refused := sorted(_layout_collisions(base.overlay, root.keys())):
                reason = "paths conflict with the workspace layout"
        if reason:
            warnings.append(
                f"base parent {base.candidate_id} not overlaid: {reason}; "
                f"paths: {', '.join(refused)}"
            )
        else:
            overlay.update(base.overlay)
    root = fold_merge(root, overlay)
    collisions = _layout_collisions(edits, root.keys())
    if collisions:
        raise _core_signal(
            SoftError("\n".join(f"{path}: {reason}" for path, reason in collisions.items()))
        )
    overlay.update(edit_hashes(edits))
    root = fold_merge(root, overlay)
    return Job.Stage(root=root, parents=list(parents), request_json=request_json), overlay, warnings


def plan_script_stage(
    *, own_files: Mapping[str, str], candidate: Candidate, effective_files: Mapping[str, str]
) -> Job.Stage:
    """Stage a check or judge's own files alongside read-only candidate files.

    Parameters
    ----------
    own_files : Mapping[str, str]
        Check or judge file manifest staged at the root.
    candidate : Candidate
        Candidate whose request, output, and files are evaluated.
    effective_files : Mapping[str, str]
        Candidate's goal files with its cumulative overlay applied.

    Returns
    -------
    Job.Stage
        Script workspace plan; an output file replaces an effective file only
        at the exact same path.

    Raises
    ------
    ValueError
        If a path is not in normal form, or the candidate files conflict,
        including an output that respells an effective file; the error names
        both paths under ``candidate/files/``.

    Examples
    --------
    >>> from swarm_arena._model.results import Candidate
    >>> from swarm_arena.tool import FileRef, RunStatus, ToolOutput, ToolRequest
    >>> out = FileRef(name="a.txt", path="p/a.txt", sha256="2" * 64, size=1)
    >>> candidate = Candidate(id="c1", run_id="r", goal_id="g", team="t", agent="a", round=1,
    ...                       request=ToolRequest(),
    ...                       output=ToolOutput(status=RunStatus.OK, files=[out]))
    >>> stage = plan_script_stage(
    ...     own_files={"judge.py": "3" * 64}, candidate=candidate,
    ...     effective_files={"p/a.txt": "1" * 64})
    >>> stage.root == {"judge.py": "3" * 64}, stage.candidate.files == {"p/a.txt": "2" * 64}
    (True, True)
    """
    _check_keys(own_files)
    _check_keys(effective_files)
    for file in candidate.output.files:
        check_key(file.path)
    output = candidate.output
    return Job.Stage(
        root=dict(own_files),
        candidate=Job.CandidateFiles(
            request={
                **_request_json(candidate.request),
                "schema_version": CURRENT["candidate_request"],
                "parents": candidate.parent_ids,
            },
            output={
                "schema_version": CURRENT["candidate_output"],
                "status": output.status.value,
                "partial": output.partial,
                "data": output.data,
                "usage": output.usage.model_dump(mode="json"),
                "files": [file.path for file in output.files],
            },
            files=fold_merge(
                effective_files,
                {file.path: file.sha256 for file in output.files},
                prefix="candidate/files/",
            ),
        ),
    )
