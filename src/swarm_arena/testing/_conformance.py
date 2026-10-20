"""Run tools, checks, and judges offline as the core does, and assert their contracts."""

import atexit
import contextlib
import secrets
import shutil
import tempfile
import threading
import warnings
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Final, TypeVar

import anyio
from pydantic import BaseModel

from swarm_arena._exec.env import tool_env_vars
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.local import _NOT_AN_OUTPUT, LocalExecutor
from swarm_arena._model.base import _sanitize_validation_error
from swarm_arena._model.goal import Goal
from swarm_arena._model.ids import IdGen
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._model.tooling import CompletedRun, Estimate, RunStatus, ToolOutput, ToolRequest
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._scoring.judge import Judge, JudgeContext, is_finite_score, stamp_scores
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._tooling.files import snapshot_fileset, snapshot_ref
from swarm_arena._tooling.result_file import request_json
from swarm_arena._tooling.staging import (
    parent_ref,
    plan_tool_stage,
    put_edits,
    tool_identity,
    tool_root,
    validate_edits,
)
from swarm_arena._tooling.tool import Action, Tool, effective_tool_timeout, tool_grace

_DEFAULT_GOAL: Final = Goal(id="check", namespace="testing")
_IDS: Final = IdGen("local", 0)
_ROOT_LOCK: Final = threading.Lock()
_LOG_TAIL_BYTES: Final = 2048
_NOTE_PREFIX: Final = "swarm-arena: "  # as note_in_log writes each note line
_INVALID_COLLECT_NOTE: Final = _NOTE_PREFIX + _NOT_AN_OUTPUT
# LocalExecutor gives an invalid collect result one of these statuses: the run's stop
# status after a timeout or a cancel, else error.
_REPLACED_COLLECT: Final = (RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.CANCELLED)
_NOT_OK: Final = tuple(status for status in RunStatus if status is not RunStatus.OK)
_root: str | None = None

_C = TypeVar("_C", Tool, Check, Judge)
_M = TypeVar("_M", CheckResult, Score)


def _artifact_root() -> str:
    """Return this process's artifact root: made on first use, removed at exit."""
    global _root
    with _ROOT_LOCK:
        if _root is None:
            _root = tempfile.mkdtemp(prefix="swarm-arena-testing-")
            atexit.register(shutil.rmtree, _root, ignore_errors=True)
        return _root


def _ids() -> IdGen:
    """Return this process's id source for candidates, jobs, and scores."""
    return _IDS


def _artifacts() -> ArtifactStore:
    """Open the artifact store under this process's artifact root."""
    return ArtifactStore(open_store(_artifact_root()))


def _fail(message: str) -> AssertionError:
    """Build the error for a broken contract; it survives ``python -O``, unlike ``assert``."""
    return AssertionError(message)


def _build(component: _C | Ref, base: type[_C], goal: Goal | None) -> tuple[Ref, _C, Goal]:
    """Snapshot files relative to the working directory, then build and preflight."""
    artifacts, base_dir = _artifacts(), Path.cwd()
    ref = component if isinstance(component, Ref) else Ref.of(component)
    ref = snapshot_ref(ref, base_dir, artifacts)
    built = ref.build(base)
    built.preflight()
    goal = _DEFAULT_GOAL if goal is None else goal
    files = snapshot_fileset(goal.files, base_dir, artifacts)
    return ref, built, goal.model_copy(update={"files": files})


class _Watched(Executor):
    """Pass jobs to another executor and keep every output, to report failed runs."""

    def __init__(self, inner: Executor) -> None:
        super().__init__()
        self._inner = inner
        self.outputs: list[tuple[str, object]] = []

    async def submit(self, job: Job) -> ToolOutput:
        """Submit the job to the wrapped executor and keep its output."""
        out = await self._inner.submit(job)
        self.outputs.append((job.id, out))
        return out

    async def capacity(self) -> Resources | None:
        """Return the wrapped executor's capacity."""
        return await self._inner.capacity()

    async def aclose(self) -> None:
        """Leave closing to the owner of the wrapped executor."""

    async def confirm_dead(self, started: Job.Started) -> bool | None:
        """Ask the wrapped executor whether a recorded job has stopped."""
        return await self._inner.confirm_dead(started)

    async def kill_orphan(self, started: Job.Started) -> bool:
        """Ask the wrapped executor to stop an orphan."""
        return await self._inner.kill_orphan(started)


@contextlib.asynccontextmanager
async def _executor(executor: Executor | None) -> AsyncGenerator[_Watched, None]:
    """Wrap the caller's executor, or a new LocalExecutor that is closed afterwards."""
    if executor is not None:
        yield _Watched(executor)
        return
    owned = LocalExecutor()
    try:
        yield _Watched(owned)
    finally:
        await owned.aclose()


def _stderr_tail(out: ToolOutput) -> str:
    """Read the last bytes of a run's stderr log from this process's artifact store."""
    logs_ref = getattr(out, "logs_ref", None)  # model_construct can leave it unset or wrong
    parts = logs_ref.split(":") if isinstance(logs_ref, str) else []
    if len(parts) != 3 or parts[0] != "sha256":
        return "(no stderr log)"
    try:
        data = _artifacts().get_bytes(parts[2])
    except Exception:
        return "(the stderr log cannot be read)"
    return data[-_LOG_TAIL_BYTES:].decode(errors="replace")


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
        if not line.startswith(_NOTE_PREFIX):
            return False  # the tool's final line, and it does not end with the note
    return False


def _failed_runs(
    outputs: Sequence[tuple[str, object]], statuses: Sequence[RunStatus] = (RunStatus.ERROR,)
) -> list[str]:
    """Describe each run with one of ``statuses`` and the end of its log (the tool's own text)."""
    notes = []
    for job_id, out in outputs:
        if not isinstance(out, ToolOutput):
            continue
        status = getattr(out, "status", None)  # model_construct can leave any object here
        failed = next((member for member in statuses if status == member), None)
        if failed is not None:
            tail = _stderr_tail(out)
            notes.append(
                f"run {job_id} ended with status {failed}; the end of its stderr log:\n{tail}"
            )
    return notes


@contextlib.asynccontextmanager
async def _failed_runs_on(runner: _Watched) -> AsyncGenerator[None, None]:
    """Add a note for each run that is not ok to any error from the block, then re-raise it."""
    try:
        yield
    except Exception as error:  # not BaseException: a cancellation stays as it is
        with anyio.CancelScope(shield=True):  # bounded: it reads the stored logs' tails
            notes = await anyio.to_thread.run_sync(_failed_runs, runner.outputs, _NOT_OK)
        for note in notes:
            error.add_note(note)
        raise


def _strict_copy(model: type[_M], value: _M) -> _M | None:
    """Validate a returned model again, strictly; None if it is invalid or needs a repair."""
    try:  # model_copy and model_construct skip validation
        dump = value.model_dump(warnings=False)
        valid = model.model_validate(dump, strict=True)
        return valid if valid.model_dump(warnings=False) == dump else None
    except Exception:
        return None


def _is_sequence_of(value: object, kind: type) -> bool:
    """Return whether a value is a non-str sequence whose items are all ``kind``."""
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(isinstance(item, kind) for item in value)
    )


def _is_set_of_str(value: object) -> bool:
    """Return whether a value is a frozenset of str."""
    return isinstance(value, frozenset) and all(isinstance(item, str) for item in value)


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
) -> _ToolPlan:
    """Build the tool and plan its stage with the core's edit and overlay rules."""
    ref, built, goal = _build(tool, Tool, goal)
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
        except Exception:  # a raise is a valid failure: the executor gives status error
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
    _probe_collect(tool, request)
    return out


def _check_result(result: object) -> CheckResult:
    """Assert that a check returned a CheckResult that is valid as it is."""
    if not isinstance(result, CheckResult):
        raise _fail("Check.check() must return a CheckResult")
    valid = _strict_copy(CheckResult, result)
    if valid is None:  # lax validation would turn a truthy passed="no" into False
        raise _fail("Check.check() must return a valid CheckResult")
    return valid


def _check_scores(
    scores: object, candidates: Sequence[Candidate], ctx: JudgeContext
) -> tuple[list[Score], set[str]]:
    """Assert a judge's scores; record a non-finite key as a judge error (spec 4.4)."""
    if not (isinstance(scores, list) and all(isinstance(score, Score) for score in scores)):
        raise _fail("Judge.judge() must return a list of Score")
    batch = {candidate.id for candidate in candidates}
    scored: set[str] = set()
    valid: list[Score] = []
    for score in scores:
        candidate_id = score.candidate_id
        if not isinstance(candidate_id, str) or candidate_id not in batch:
            raise _fail(f"Judge.judge() returned a score for unknown candidate {candidate_id!r}")
        if candidate_id in scored:
            raise _fail(f"Judge.judge() returned more than one score for candidate {candidate_id}")
        scored.add(candidate_id)
        if not is_finite_score(score):
            ctx.judge_error(candidate_id, "score key must be nonempty and finite")
            continue
        copy = _strict_copy(Score, score)
        if copy is None:  # lax validation would turn value="1.0" into 1.0
            raise _fail("Judge.judge() must return valid Scores")
        valid.append(copy)
    return valid, scored


def _check_judge_methods(ref: Ref, judge: Judge) -> None:
    """Assert a stable judge identity and the types of the optional Judge methods."""
    rubric = judge.rubric_hash()
    same = rubric == judge.rubric_hash() == ref.build(Judge).rubric_hash()
    if not (isinstance(rubric, str) and rubric and same):
        raise _fail("Judge.rubric_hash() must return the same nonempty str")
    panel = judge.panel_hash()
    if not (isinstance(panel, str) and judge.panel_hash() == panel):
        raise _fail("Judge.panel_hash() must return the same str")
    anchors = judge.anchor_count()
    if not (isinstance(anchors, int) and not isinstance(anchors, bool) and anchors >= 0):
        raise _fail("Judge.anchor_count() must return an int >= 0")
    if not _is_set_of_str(judge.providers()):
        raise _fail("Judge.providers() must return a frozenset of str")
    models = judge.models()
    if not (isinstance(models, tuple) and all(isinstance(model, str) for model in models)):
        raise _fail("Judge.models() must return a tuple of str")


async def check_tool(
    tool: Tool | Ref,
    inputs: Mapping[str, Any],
    *,
    goal: Goal | None = None,
    parents: Sequence[Candidate] = (),
    edits: Mapping[str, str] | None = None,
    executor: Executor | None = None,
    timeout_s: float = 60.0,
) -> ToolOutput:
    """
    Stage and run one tool job as the core does, and assert the Tool contract.

    Files that are not yet snapshotted (the tool's own files and the goal's
    files) are snapshotted relative to the current working directory. The
    inputs are validated with ``get_input_model()``, the edits follow the core
    rules (``limits.max_edit_bytes``, editable globs, protected paths), and the
    first parent's overlay is applied only when that parent has the same tool
    identity and every overlay path is editable. A refused overlay gives a
    ``UserWarning``, as the core gives a soft warning. The job runs on
    ``executor`` with the run identity ``local`` and the goal's id in its
    environment. A run that ends with status ``error`` gives a ``UserWarning``
    with the last 2 KB of its stderr log. If the call raises an error instead
    (a failed contract assert, or an error from the tool's own methods), a run
    with any status but ``ok`` (``error``, ``timeout``, ``cancelled``, ...)
    adds that text as a note to the error, which is otherwise passed on
    unchanged.

    After the run, the output must be a valid ``ToolOutput`` (finite JSON data;
    it is validated again in lax mode, as the core does, and the checks below
    read that validated copy), and its files must be in the artifact store. An
    invalid ``collect()`` result is a contract failure on every path:
    ``LocalExecutor`` replaces it with an empty output (status ``error``, or
    ``timeout`` or ``cancelled`` for a stopped run) and a note in the stderr
    log, and the kit raises the ``AssertionError`` for that note.
    ``fingerprint()`` must give the same nonempty str twice and must not
    depend on ``logs_ref`` or the run times. ``collect()`` is also called once
    on an empty workspace with a failed run: it may raise, but it must not
    report ``ok``.

    Parameters
    ----------
    tool : Tool or Ref
        Tool instance or reference to check. An instance is converted with
        ``Ref.of`` and built again, as the core builds it from a plan.
    inputs : Mapping[str, Any]
        Tool inputs, validated with the tool's input model.
    goal : Goal or None, default=None
        Goal whose files are staged; None selects
        ``Goal(id="check", namespace="testing")``.
    parents : Sequence[Candidate], default=()
        Parent candidates, first being the base. Make them with
        ``make_candidate(..., tool=tool)`` to test overlay inheritance.
    edits : Mapping[str, str] or None, default=None
        Workspace paths mapped to new UTF-8 text, as a manager supplies them.
    executor : Executor or None, default=None
        Executor that runs the job; None makes a ``LocalExecutor`` and closes
        it afterwards.
    timeout_s : float, default=60.0
        Hard cap on the tool's run time in seconds.

    Returns
    -------
    ToolOutput
        The executor's own output object, also when its status is not ``ok``.

    Raises
    ------
    AssertionError
        If the tool or executor breaks its contract; the message names the method.
    pydantic.ValidationError
        If ``inputs`` do not match the tool's input model. Only field names
        and error types are kept, never the input values.
    ValueError
        If an edit is refused (a ``SoftError``, one line per path with its
        reason), a file glob matches no file, a tool file is a goal file with
        another spelling (a ``ConfigError``, as in the core), or the tool
        cannot be built.

    Examples
    --------
    >>> import anyio
    >>> from swarm_arena import FunctionTool
    >>> from swarm_arena.testing import check_tool
    >>> out = anyio.run(check_tool, FunctionTool("my_package.tools:solve"),  # doctest: +SKIP
    ...                 {"lr": 0.01})
    >>> out.status  # doctest: +SKIP
    <RunStatus.OK: 'ok'>
    """
    plan = await anyio.to_thread.run_sync(
        partial(_plan_tool, tool, inputs, goal=goal, parents=parents, edits=edits)
    )
    for note in plan.warnings:
        warnings.warn(note, UserWarning, stacklevel=2)
    job_id = _ids().next("c")
    job = Job(
        id=job_id,
        kind="tool",
        tool=plan.ref,
        request=plan.request,
        stage=plan.stage,
        timeout_s=effective_tool_timeout(plan.tool, round_timeout_s=None, limit_s=timeout_s),
        grace_s=tool_grace(plan.tool),
        resources=Resources(),
        env=tool_env_vars(
            run_id="local", goal_id=plan.goal.id, candidate_id=job_id, team="", agent="", round=0
        ),
        secret_env=(),
        secrets_file=None,
        artifact_root=_artifact_root(),
    )
    async with _executor(executor) as runner, _failed_runs_on(runner):  # an error gets the tails
        out = await runner.submit(job)
        checked = await anyio.to_thread.run_sync(_check_tool_output, plan.tool, plan.request, out)
    for note in await anyio.to_thread.run_sync(_failed_runs, runner.outputs):
        warnings.warn(note, UserWarning, stacklevel=2)
    return checked


async def check_check(
    check: Check | Ref,
    candidate: Candidate,
    *,
    goal: Goal | None = None,
    executor: Executor | None = None,
) -> CheckResult:
    """
    Run one check on a candidate, as the core does after a tool run.

    The check's files and the goal's files are snapshotted relative to the
    current working directory, the check is built and preflighted, and it runs
    with a ``CheckContext`` whose commands go to ``executor``. The result must
    be valid in strict mode as it is: a value that a lax validation would
    repair, such as ``passed="no"``, is a contract failure. A command that ends
    with status ``error`` gives a ``UserWarning`` with the last 2 KB of its
    stderr log. If the call raises an error instead (a failed contract assert,
    or an error from ``check()`` itself), each command with any status but
    ``ok`` adds that text as a note to the error, which is otherwise passed on
    unchanged.

    Parameters
    ----------
    check : Check or Ref
        Check instance or reference to run.
    candidate : Candidate
        Candidate to check, for example from ``make_candidate``.
    goal : Goal or None, default=None
        Goal whose files the check sees under ``candidate/files/``; None selects
        ``Goal(id="check", namespace="testing")``.
    executor : Executor or None, default=None
        Executor for ``ctx.execute``; None makes a ``LocalExecutor`` and closes
        it afterwards.

    Returns
    -------
    CheckResult
        The check's result, with ``check`` set to the component name (the
        reference's ``name`` or the class's ``name``).

    Raises
    ------
    AssertionError
        If ``feedback`` is not ``"pass_fail"`` or ``"reasons"``, or ``check()``
        does not return a valid ``CheckResult``.
    ValueError
        If a file glob matches no file or the check cannot be built.

    Examples
    --------
    >>> import anyio
    >>> from swarm_arena import ScriptCheck
    >>> from swarm_arena.testing import check_check, make_candidate
    >>> from swarm_arena.tool import RunStatus, ToolOutput
    >>> check = ScriptCheck(command=["python", "checks/check.py"], files=["checks/check.py"])
    >>> candidate = make_candidate(ToolOutput(status=RunStatus.OK, data={"f": 0.5}))
    >>> result = anyio.run(check_check, check, candidate)  # doctest: +SKIP
    >>> (result.check, result.passed)  # doctest: +SKIP
    ('script-check', True)
    """
    ref, built, goal = await anyio.to_thread.run_sync(_build, check, Check, goal)
    if built.feedback not in ("pass_fail", "reasons"):
        raise _fail("Check.feedback must be 'pass_fail' or 'reasons'")
    async with _executor(executor) as runner, _failed_runs_on(runner):  # an error gets the tails
        ctx = CheckContext(goal=goal, executor=runner, artifact_root=_artifact_root(), ids=_ids())
        result = _check_result(await built.check(candidate, ctx))
    for note in await anyio.to_thread.run_sync(_failed_runs, runner.outputs):
        warnings.warn(note, UserWarning, stacklevel=2)
    return result.model_copy(update={"check": ref.component_name()})


async def check_judge(
    judge: Judge | Ref,
    goal: Goal,
    candidates: Sequence[Candidate],
    *,
    executor: Executor | None = None,
) -> list[Score]:
    """
    Judge one batch of candidates and stamp the scores, as the core does.

    The judge's files and the goal's files are snapshotted relative to the
    current working directory, and the judge is built and preflighted. Its
    identity must be stable: ``rubric_hash()`` gives the same nonempty str on
    two calls and after a second build, and ``panel_hash()`` the same str.
    Then it runs with a ``JudgeContext`` that has a new batch id. Each
    candidate error that the judge records with ``ctx.judge_error`` gives a
    ``UserWarning`` ``"judge error for <id>: <reason>"``, and so does a
    candidate that got neither a score nor a judge error. A score with a
    non-finite key is dropped as a judge error, as the core rejects it. Every
    other score must be valid in strict mode as it is. A command that ends with
    status ``error`` gives a ``UserWarning`` with the last 2 KB of its stderr
    log. If the call raises an error instead (a failed contract assert, or an
    error from ``judge()`` itself), each command with any status but ``ok``
    adds that text as a note to the error, which is otherwise passed on
    unchanged.

    Parameters
    ----------
    judge : Judge or Ref
        Judge instance or reference to run.
    goal : Goal
        Goal passed to the judge.
    candidates : Sequence[Candidate]
        Batch to judge, for example from ``make_candidate``.
    executor : Executor or None, default=None
        Executor for ``ctx.execute``; None makes a ``LocalExecutor`` and closes
        it afterwards.

    Returns
    -------
    list[Score]
        Scores stamped with ids, the judge name, the rubric hash, and the
        batch id, in the judge's order.

    Raises
    ------
    AssertionError
        If the judge identity is not stable, an optional Judge method returns
        the wrong type, or ``judge()`` does not return a list of valid
        ``Score`` objects, returns a score for an unknown candidate, or two
        scores for one candidate.
    ValueError
        If a file glob matches no file or the judge cannot be built.

    Examples
    --------
    >>> import anyio
    >>> from swarm_arena import Goal, ProgramJudge
    >>> from swarm_arena.testing import check_judge
    >>> anyio.run(check_judge, ProgramJudge("builtins:max"), Goal(id="g", namespace="demo"), [])
    []
    """
    ref, built, goal = await anyio.to_thread.run_sync(_build, judge, Judge, goal)
    await anyio.to_thread.run_sync(_check_judge_methods, ref, built)
    async with _executor(executor) as runner, _failed_runs_on(runner):  # an error gets the tails
        ctx = JudgeContext(
            goal=goal,
            executor=runner,
            artifact_root=_artifact_root(),
            ids=_ids(),
            batch_id="b-" + secrets.token_hex(4),
        )
        valid, scored = _check_scores(await built.judge(goal, candidates, ctx), candidates, ctx)
        stamped = stamp_scores(
            valid, judge=built, judge_name=ref.component_name(), ids=_ids(), batch_id=ctx.batch_id
        )
    for note in await anyio.to_thread.run_sync(_failed_runs, runner.outputs):
        warnings.warn(note, UserWarning, stacklevel=2)
    for candidate_id, reason in ctx.errors.items():
        warnings.warn(f"judge error for {candidate_id}: {reason}", UserWarning, stacklevel=2)
    for candidate in candidates:
        if candidate.id not in scored and candidate.id not in ctx.errors:
            message = f"judge gave no score and no judge error for {candidate.id}"
            warnings.warn(message, UserWarning, stacklevel=2)
    return stamped
