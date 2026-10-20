"""Make candidates for conformance tests without running a swarm."""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from swarm_arena._model.goal import Goal
from swarm_arena._model.limits import Limits
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import ToolOutput, ToolRequest
from swarm_arena._tooling.files import snapshot_fileset, snapshot_ref
from swarm_arena._tooling.staging import (
    parent_ref,
    plan_tool_stage,
    put_edits,
    tool_identity,
    tool_root,
    validate_edits,
)
from swarm_arena._tooling.tool import Tool
from swarm_arena.testing._components import _effective_tool_ref
from swarm_arena.testing._state import _artifacts, _ids


def make_candidate(
    output: ToolOutput,
    *,
    request: ToolRequest | None = None,
    tool: Tool | Ref | None = None,
    parents: Sequence[Candidate] = (),
    goal: Goal | None = None,
    **fields: Any,
) -> Candidate:
    """
    Make a candidate from a tool output, with the defaults of a local check.

    The id comes from this process's ``IdGen(run_id="local", epoch=0)``, so
    ids never repeat in one process. The overlay combines the accepted base
    parent's effective overlay with this request's edits, using the same
    identity, editable-path and workspace-layout rules as ``check_tool``.
    Only the first parent contributes an overlay; later parents remain
    read-only references. A refused overlay gives a ``UserWarning``.
    With ``tool``, the candidate gets that tool's identity. Tool and goal
    files are snapshotted relative to the current working directory. Edit
    text is stored in the kit's artifact store, so a child can itself be a
    base parent and its checks and judges see the complete edit chain.
    Without ``tool``, inherited overlays are refused because their identity
    cannot be verified. Explicit ``overlay`` fields bypass this computation.

    Parameters
    ----------
    output : ToolOutput
        Output of the tool run that made the candidate.
    request : ToolRequest or None, default=None
        Request of that run; None selects ``ToolRequest()``. Its round
        overrides are merged into the tool config as in ``check_tool``.
        Its round becomes the candidate's default round.
    tool : Tool or Ref or None, default=None
        Tool that made the candidate; None leaves ``tool_identity`` empty.
    parents : Sequence[Candidate], default=()
        Ordered parents, first being the proposed base. If supplied, these
        fill ``request.parents`` and must agree with any existing references.
        Otherwise the references already in ``request.parents`` are used.
        The candidate's ``parent_ids`` are derived from those references.
    goal : Goal or None, default=None
        Goal used for the run. Its files participate in overlay eligibility,
        and its id becomes the default ``goal_id``. Pass the same goal as to
        ``check_tool``; None means no goal files and ``goal_id="check"``.
    **fields : Any
        ``Candidate`` fields that replace the defaults: ``id`` (a new id),
        ``run_id="local"``, ``goal_id="check"``, ``team="team"``,
        ``agent="agent"``, ``round=request.round``, ``overlay``, and ``tool_identity``.

    Returns
    -------
    Candidate
        A validated candidate.

    Raises
    ------
    ValueError
        If parents disagree with request references, an edit violates the
        tool's edit rules, a file glob matches no file, or files conflict.
        Stored-edit references (``sha256:`` and 64 hex digits) are refused.

    Examples
    --------
    >>> from swarm_arena.testing import make_candidate
    >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
    >>> candidate = make_candidate(ToolOutput(status=RunStatus.OK, data={"f": 1.5}))
    >>> (candidate.run_id, candidate.goal_id, candidate.team, candidate.round)
    ('local', 'check', 'team', 0)
    >>> edited = make_candidate(ToolOutput(status=RunStatus.OK),
    ...                         request=ToolRequest(edits={"problem/a.txt": "2"}), team="solvers")
    >>> (list(edited.overlay), edited.team)
    (['problem/a.txt'], 'solvers')
    >>> from swarm_arena import ScriptTool
    >>> tool = ScriptTool(command=["python", "run.py"], editable=["problem/*.txt"])
    >>> base = make_candidate(edited.output, request=edited.request, tool=tool)
    >>> child = make_candidate(base.output, tool=tool, parents=[base],
    ...                        request=ToolRequest(edits={"problem/b.txt": "3"}))
    >>> sorted(child.overlay), child.parent_ids == [base.id]
    (['problem/a.txt', 'problem/b.txt'], True)
    """
    request = ToolRequest() if request is None else request
    if parents:
        refs = [parent_ref(parent) for parent in parents]
        if request.parents and request.parents != refs:
            raise ValueError("parents must match request.parents")
        request = request.model_copy(update={"parents": refs})
    values: dict[str, Any] = {
        "run_id": "local",
        "goal_id": "check" if goal is None else goal.id,
        "parent_ids": [parent.candidate_id for parent in request.parents],
        "team": "team",
        "agent": "agent",
        "round": request.round,
        "request": request,
        "output": output,
    }
    built: Tool | None = None
    if tool is not None:
        ref = _effective_tool_ref(tool, request.round_overrides)
        ref = snapshot_ref(ref, Path.cwd(), _artifacts())
        built = ref.build(Tool)
        values["tool_identity"] = tool_identity(ref, None)
    if "overlay" not in fields:
        goal_files = (
            {} if goal is None else snapshot_fileset(goal.files, Path.cwd(), _artifacts()).manifest
        )
        tool_files = {} if built is None else built._own_files()
        root = tool_root(goal_files, tool_files)
        protected = frozenset() if built is None else built._protected(root.keys())
        editable = ["**/*"] if built is None else built.editable()
        validate_edits(
            request.edits,
            editable=editable,
            protected=protected,
            max_edit_bytes=Limits().max_edit_bytes,
            root=root.keys(),
        )
        _, overlay, notes = plan_tool_stage(
            goal_files=goal_files,
            tool_files=tool_files,
            editable=editable,
            protected=protected,
            private_paths=frozenset(),
            parents=request.parents,
            receiving_identity=values.get("tool_identity", ""),
            edits=request.edits,
            request_json=None,
        )
        values["overlay"] = overlay
        put_edits(request.edits, _artifacts())
        for note in notes:
            warnings.warn(note, UserWarning, stacklevel=2)
    if "id" not in fields:
        values["id"] = _ids().next("c")
    return Candidate(**{**values, **fields})
