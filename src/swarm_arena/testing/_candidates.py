"""Make candidates for conformance tests without running a swarm."""

from pathlib import Path
from typing import Any

from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import ToolOutput, ToolRequest
from swarm_arena._tooling.files import snapshot_ref
from swarm_arena._tooling.staging import edit_hashes, put_edits, tool_identity
from swarm_arena._tooling.tool import Tool
from swarm_arena.testing._conformance import _artifacts, _ids


def make_candidate(
    output: ToolOutput,
    *,
    request: ToolRequest | None = None,
    tool: Tool | Ref | None = None,
    **fields: Any,
) -> Candidate:
    """
    Make a candidate from a tool output, with the defaults of a local check.

    The id comes from this process's ``IdGen(run_id="local", epoch=0)``, so
    ids never repeat in one process. The overlay holds the hashes of the
    request's edits, and the edit text is stored where ``check_tool`` stages
    from, so the candidate can be a base parent. With ``tool``, the candidate
    gets that tool's identity (its files snapshotted relative to the current
    working directory, as ``check_tool`` does), so ``check_tool`` applies its
    overlay.

    Parameters
    ----------
    output : ToolOutput
        Output of the tool run that made the candidate.
    request : ToolRequest or None, default=None
        Request of that run; None selects ``ToolRequest()``.
    tool : Tool or Ref or None, default=None
        Tool that made the candidate; None leaves ``tool_identity`` empty.
    **fields : Any
        ``Candidate`` fields that replace the defaults: ``id`` (a new id),
        ``run_id="local"``, ``goal_id="check"``, ``team="team"``,
        ``agent="agent"``, ``round=0``, ``overlay``, and ``tool_identity``.

    Returns
    -------
    Candidate
        A validated candidate.

    Raises
    ------
    ValueError
        If an edit path is not in normal form or an edit is a stored-edit
        reference (``sha256:`` and 64 hex digits).

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
    """
    request = ToolRequest() if request is None else request
    values: dict[str, Any] = {
        "run_id": "local",
        "goal_id": "check",
        "team": "team",
        "agent": "agent",
        "round": 0,
        "request": request,
        "output": output,
    }
    if "overlay" not in fields:
        values["overlay"] = edit_hashes(request.edits)
        put_edits(request.edits, _artifacts())
    if tool is not None and "tool_identity" not in fields:
        ref = tool if isinstance(tool, Ref) else Ref.of(tool)
        values["tool_identity"] = tool_identity(snapshot_ref(ref, Path.cwd(), _artifacts()), None)
    if "id" not in fields:
        values["id"] = _ids().next("c")
    return Candidate(**{**values, **fields})
