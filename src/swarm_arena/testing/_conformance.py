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
from functools import partial
from pathlib import Path
from typing import Any, Final, TypeVar

import anyio
from pydantic import BaseModel

from swarm_arena._exec.env import tool_env_vars
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._model.goal import Goal
from swarm_arena._model.ids import IdGen
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._model.tooling import Estimate, ToolOutput, ToolRequest
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._scoring.judge import Judge, JudgeContext, is_finite_score, stamp_scores
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._tooling.files import snapshot_fileset, snapshot_ref
from swarm_arena._tooling.paths import fold_merge
from swarm_arena._tooling.result_file import request_json
from swarm_arena._tooling.staging import (
    parent_ref,
    plan_tool_stage,
    put_edits,
    tool_identity,
    validate_edits,
)
from swarm_arena._tooling.tool import Tool, effective_tool_timeout, tool_grace

_DEFAULT_GOAL: Final = Goal(id="check", namespace="testing")
_IDS: Final = IdGen("local", 0)
_ROOT_LOCK: Final = threading.Lock()
_root: str | None = None

_C = TypeVar("_C", Tool, Check, Judge)


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


@contextlib.asynccontextmanager
async def _executor(executor: Executor | None) -> AsyncGenerator[Executor, None]:
    """Yield the caller's executor, or a new LocalExecutor that is closed afterwards."""
    if executor is not None:
        yield executor
        return
    owned = LocalExecutor()
    try:
        yield owned
    finally:
        await owned.aclose()


def _check_tool_methods(tool: Tool) -> None:
    """Assert the types that the optional Tool methods return before a run."""
    model = tool.get_input_model()
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise _fail("Tool.get_input_model() must return a pydantic BaseModel subclass")
    if not isinstance(tool.get_description(), str):
        raise _fail("Tool.get_description() must return a str")
    globs = tool.editable()
    if isinstance(globs, str) or not all(isinstance(glob, str) for glob in globs):
        raise _fail("Tool.editable() must return a sequence of str globs")
    providers = tool.providers()
    if not (isinstance(providers, frozenset) and all(isinstance(p, str) for p in providers)):
        raise _fail("Tool.providers() must return a frozenset of str")


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
    request = ToolRequest(
        inputs=built.get_input_model().model_validate(inputs).model_dump(mode="json"),
        parents=[parent_ref(parent) for parent in parents],
        edits=dict(edits or {}),
    )
    goal_files, tool_files = goal.files.manifest, built._own_files()
    root = fold_merge(goal_files, tool_files)
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


def _check_tool_output(tool: Tool, request: ToolRequest, out: object) -> ToolOutput:
    """Assert the executor's output and the Tool methods that read an output."""
    if not isinstance(out, ToolOutput):
        raise _fail("Executor.submit() must return a ToolOutput")
    store = open_store(_artifact_root())
    for file in out.files:
        if not store.exists(file.sha256):
            raise _fail(f"output file {file.path} is not in the artifact store")
    fingerprint = tool.fingerprint(out)
    if not (isinstance(fingerprint, str) and fingerprint):
        raise _fail("Tool.fingerprint() must return a nonempty str")
    if tool.fingerprint(out) != fingerprint:
        raise _fail("Tool.fingerprint() must return the same str for the same output")
    if not isinstance(tool.is_deterministic(request), bool):
        raise _fail("Tool.is_deterministic() must return a bool")
    estimate = tool.estimate(request)
    if not (estimate is None or isinstance(estimate, Estimate)):
        raise _fail("Tool.estimate() must return None or an Estimate")
    return out


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
    environment.

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
        The executor's output, also when its status is not ``ok``.

    Raises
    ------
    AssertionError
        If the tool or executor breaks its contract; the message names the method.
    pydantic.ValidationError
        If ``inputs`` do not match the tool's input model.
    ValueError
        If an edit is refused (a ``SoftError``, one line per path with its
        reason), a file glob matches no file, or the tool cannot be built.

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
    async with _executor(executor) as runner:
        out = await runner.submit(job)
    return await anyio.to_thread.run_sync(_check_tool_output, plan.tool, plan.request, out)


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
    with a ``CheckContext`` whose commands go to ``executor``.

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
        does not return a ``CheckResult``.
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
    async with _executor(executor) as runner:
        ctx = CheckContext(goal=goal, executor=runner, artifact_root=_artifact_root(), ids=_ids())
        result = await built.check(candidate, ctx)
    if not isinstance(result, CheckResult):
        raise _fail("Check.check() must return a CheckResult")
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
    current working directory, the judge is built and preflighted, and it runs
    with a ``JudgeContext`` that has a new batch id. Each candidate error the
    judge records with ``ctx.judge_error`` gives a ``UserWarning``
    ``"judge error for <id>: <reason>"``. A score with a non-finite key is
    dropped as a judge error, as the core rejects it.

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
        If ``judge()`` does not return a list of ``Score``, or returns a score
        for an unknown candidate or two scores for one candidate.
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
    async with _executor(executor) as runner:
        ctx = JudgeContext(
            goal=goal,
            executor=runner,
            artifact_root=_artifact_root(),
            ids=_ids(),
            batch_id="b-" + secrets.token_hex(4),
        )
        scores = await built.judge(goal, candidates, ctx)
    if not (isinstance(scores, list) and all(isinstance(score, Score) for score in scores)):
        raise _fail("Judge.judge() must return a list of Score")
    batch = {candidate.id for candidate in candidates}
    seen: set[str] = set()
    finite: list[Score] = []
    for score in scores:
        candidate_id = score.candidate_id
        if candidate_id not in batch:
            raise _fail(f"Judge.judge() returned a score for unknown candidate {candidate_id!r}")
        if candidate_id in seen:
            raise _fail(f"Judge.judge() returned more than one score for candidate {candidate_id}")
        seen.add(candidate_id)
        if is_finite_score(score):
            finite.append(score)
        else:
            ctx.judge_error(candidate_id, "score key must be nonempty and finite")
    for candidate_id, reason in ctx.errors.items():
        warnings.warn(f"judge error for {candidate_id}: {reason}", UserWarning, stacklevel=2)
    return stamp_scores(
        finite, judge=built, judge_name=ref.component_name(), ids=_ids(), batch_id=ctx.batch_id
    )
