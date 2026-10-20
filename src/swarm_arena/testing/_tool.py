"""Prepare tool conformance jobs and probe their collection contracts."""

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from swarm_arena._exec.executor import Job
from swarm_arena._exec.local import _NOT_AN_OUTPUT
from swarm_arena._model.base import _sanitize_validation_error
from swarm_arena._model.goal import Goal
from swarm_arena._model.limits import Limits
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import (
    CompletedRun,
    Estimate,
    RunStatus,
    ToolOutput,
    ToolRequest,
)
from swarm_arena._store.base import open_store
from swarm_arena._tooling.result_file import NOTE_PREFIX, request_json
from swarm_arena._tooling.staging import (
    parent_ref,
    plan_tool_stage,
    put_edits,
    tool_identity,
    tool_root,
    validate_edits,
)
from swarm_arena._tooling.tool import Action, Tool
from swarm_arena.testing._components import _build, _effective_tool_ref
from swarm_arena.testing._contracts import _fail, _is_sequence_of, _is_set_of_str
from swarm_arena.testing._execution import _stderr_tail
from swarm_arena.testing._state import _artifact_root, _artifacts

_INVALID_COLLECT_NOTE: Final = NOTE_PREFIX + _NOT_AN_OUTPUT
# LocalExecutor gives an invalid collect result one of these statuses: the run's stop
# status after a timeout or a cancel, else error.
_REPLACED_COLLECT: Final = (RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.CANCELLED)


def _has_invalid_collect_note(tail: str) -> bool:
    """
    Return whether a stderr tail holds LocalExecutor's invalid-collect note.

    The note is a whole line. The executor writes it after the tool's last
    output, so when that output has no final newline, the note ends the tool's
    final line instead: then only the executor's own notes come after it.
    """
    lines = tail.splitlines()
    if _INVALID_COLLECT_NOTE in lines:
        return True
    for line in reversed(lines):
        if line.endswith(_INVALID_COLLECT_NOTE):
            return True
        if not line.startswith(NOTE_PREFIX):
            return False  # the tool's final line, and it does not end with the note
    return False


def _check_tool_methods(tool: Tool) -> None:
    """Assert the types that the optional Tool methods return before a run."""
    model = tool.get_input_model()
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise _fail("Tool.get_input_model() must return a pydantic BaseModel subclass")
    if not isinstance(tool.get_description(), str):
        raise _fail("Tool.get_description() must return a str")
    if not _is_sequence_of(tool.editable(), str):
        raise _fail("Tool.editable() must return a sequence of str globs")
    if not _is_set_of_str(tool.providers()):
        raise _fail("Tool.providers() must return a frozenset of str")
    actions = tool.extra_actions()
    if not _is_sequence_of(actions, Action) or len({a.name for a in actions}) != len(actions):
        raise _fail("Tool.extra_actions() must return a sequence of Action with unique names")


@dataclass(frozen=True)
class _ToolPlan:
    """A built tool and its planned workspace, ready to become a job."""

    ref: Ref
    tool: Tool
    goal: Goal
    request: ToolRequest
    stage: Job.Stage
    warnings: list[str]


def _plan_tool(
    tool: Tool | Ref,
    inputs: Mapping[str, Any],
    *,
    goal: Goal | None,
    parents: Sequence[Candidate],
    edits: Mapping[str, str] | None,
    round: int,
    round_overrides: Mapping[str, Any] | None,
    replicate: int,
) -> _ToolPlan:
    """Build the tool and plan its stage with the core's edit and overlay rules."""
    context = ToolRequest(
        round=round, round_overrides=dict(round_overrides or {}), replicate=replicate
    )
    ref, built, goal = _build(_effective_tool_ref(tool, context.round_overrides), Tool, goal)
    _check_tool_methods(built)
    model = built.get_input_model()
    try:
        params = model.model_validate(inputs).model_dump(mode="json")
    except Exception as error:  # a connector's own model can show input values
        raise _sanitize_validation_error(model, error) from None
    request = ToolRequest(
        inputs=params,
        parents=[parent_ref(parent) for parent in parents],
        edits=dict(edits or {}),
        round=context.round,
        round_overrides=context.round_overrides,
        replicate=context.replicate,
    )
    goal_files, tool_files = goal.files.manifest, built._own_files()
    root = tool_root(goal_files, tool_files)  # the core's rule: a respelled file is refused
    protected = built._protected(root.keys())
    validate_edits(
        request.edits,
        editable=built.editable(),
        protected=protected,
        max_edit_bytes=Limits().max_edit_bytes,
        root=root.keys(),
    )
    put_edits(request.edits, _artifacts())
    stage, _, notes = plan_tool_stage(
        goal_files=goal_files,
        tool_files=tool_files,
        editable=built.editable(),
        protected=protected,
        private_paths=frozenset(),
        parents=request.parents,
        receiving_identity=tool_identity(ref, None),
        edits=request.edits,
        request_json=request_json(request),
    )
    return _ToolPlan(ref, built, goal, request, stage, notes)


def _probe_collect(tool: Tool, request: ToolRequest) -> None:
    """Assert that collect() keeps a failure: run it on an empty workspace after exit 1."""
    with tempfile.TemporaryDirectory(prefix="swarm-arena-probe-") as directory:
        workspace = Path(directory)
        (workspace / "swarm").mkdir()
        stdout, stderr = workspace / "swarm/stdout.log", workspace / "swarm/stderr.log"
        stdout.touch()
        stderr.touch()
        now = datetime.now(UTC)
        run = CompletedRun(
            returncode=1,
            status=RunStatus.ERROR,
            stdout_path=stdout,
            stderr_path=stderr,
            started_at=now,
            ended_at=now,
        )
        try:
            out = tool.collect(request, workspace, run)
        except (Exception, SystemExit):  # a raise is a valid failure; interrupts propagate
            return
    if not isinstance(out, ToolOutput):
        raise _fail("Tool.collect() must return a ToolOutput")
    if getattr(out, "status", None) == RunStatus.OK:
        raise _fail("Tool.collect() must not report ok for a failed run")


def _check_tool_output(tool: Tool, request: ToolRequest, out: object) -> ToolOutput:
    """Assert the executor's output and the Tool methods that read an output."""
    if not isinstance(out, ToolOutput):
        raise _fail("Executor.submit() must return a ToolOutput")
    message = "Tool.collect() must return a valid ToolOutput (data must be finite JSON)"
    # LocalExecutor replaces an invalid collect result with an empty output and this note,
    # on every path; a tool's own text on an ok run is never taken for the note.
    if getattr(out, "status", None) in _REPLACED_COLLECT and _has_invalid_collect_note(
        _stderr_tail(out)
    ):
        raise _fail(message)
    try:  # lax, as the core re-validates it: another executor may pass it on unvalidated
        valid = ToolOutput.model_validate(out.model_dump(warnings=False))
    except Exception:
        raise _fail(message) from None
    # The checks read the validated copy, as the core would; the caller gets ``out`` itself.
    store = open_store(_artifact_root())
    for file in valid.files:
        if not store.exists(file.sha256):
            raise _fail(f"output file {file.path} is not in the artifact store")
    fingerprint = tool.fingerprint(valid)
    if not (isinstance(fingerprint, str) and fingerprint):
        raise _fail("Tool.fingerprint() must return a nonempty str")
    if tool.fingerprint(valid) != fingerprint:
        raise _fail("Tool.fingerprint() must return the same str for the same output")
    rerun = valid.model_copy(update={"logs_ref": None, "started_at": None, "ended_at": None})
    if tool.fingerprint(rerun) != fingerprint:
        raise _fail("Tool.fingerprint() must not depend on logs_ref or the run times")
    if not isinstance(tool.is_deterministic(request), bool):
        raise _fail("Tool.is_deterministic() must return a bool")
    estimate = tool.estimate(request)
    if not (estimate is None or isinstance(estimate, Estimate)):
        raise _fail("Tool.estimate() must return None or an Estimate")
    return out
