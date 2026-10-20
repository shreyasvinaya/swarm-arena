"""Run tools, checks, and judges offline as the core does, and assert their contracts."""

import secrets
import warnings
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import anyio

from swarm_arena._exec.env import tool_env_vars
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.supervisor import BEFORE_KILL_TIMEOUT_S
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.goal import Goal
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._model.tooling import ToolOutput
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._scoring.judge import Judge, JudgeContext, stamp_scores
from swarm_arena._tooling.tool import Tool, effective_tool_timeout, tool_grace
from swarm_arena.testing._components import _DEFAULT_GOAL, _build, _check_goal
from swarm_arena.testing._contracts import _check_judge_methods, _check_result, _check_scores, _fail
from swarm_arena.testing._execution import _executor, _failed_runs, _failed_runs_on
from swarm_arena.testing._state import _artifact_root, _ids
from swarm_arena.testing._tool import _check_tool_output, _plan_tool, _probe_collect


async def check_tool(
    tool: Tool | Ref,
    inputs: Mapping[str, Any],
    *,
    goal: Goal | None = None,
    parents: Sequence[Candidate] = (),
    edits: Mapping[str, str] | None = None,
    round: int = 0,
    round_overrides: Mapping[str, Any] | None = None,
    replicate: int = 0,
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
    environment. A run with any status other than ``ok`` gives a ``UserWarning``
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
    report ``ok``. This probe runs in a cancellable daemon thread with a
    30-second deadline; exceeding it is a contract failure. If abandoned,
    its workspace is removed when the thread eventually returns.

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
    round : int, default=0
        Round index passed to command and collect, staged in
        ``swarm/request.json``, and exported as ``SWARM_ARENA_ROUND``.
    round_overrides : Mapping[str, Any] or None, default=None
        Top-level settings merged into the tool config before config and input
        validation, snapshotting, preflight, staging, timeout selection, and
        overlay identity comparison. Nested values replace rather than merge.
        Human-only fields (including their validation aliases) cannot be
        overridden. The supplied overrides are also recorded in the request.
    replicate : int, default=0
        Replicate index passed to command and collect and staged in the request.
    executor : Executor or None, default=None
        Executor that runs the job; None makes a ``LocalExecutor`` and closes
        it afterwards.
    timeout_s : float, default=60.0
        Hard cap on the tool's run time in seconds. A smaller configured
        tool ``timeout_s`` is preserved; an unset tool timeout uses this cap.
        The failed-run collection probe has a separate 30-second deadline.

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
        cannot be built, an override fails config validation, or a human-only
        field is overridden. Override errors omit configuration values.

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
        partial(
            _plan_tool,
            tool,
            inputs,
            goal=goal,
            parents=parents,
            edits=edits,
            round=round,
            round_overrides=round_overrides,
            replicate=replicate,
        )
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
            run_id="local",
            goal_id=plan.goal.id,
            candidate_id=job_id,
            team="",
            agent="",
            round=plan.request.round,
        ),
        secret_env=(),
        secrets_file=None,
        artifact_root=_artifact_root(),
    )
    async with _executor(executor) as runner, _failed_runs_on(runner):  # an error gets the tails
        out = await runner.submit(job)
        checked = await anyio.to_thread.run_sync(_check_tool_output, plan.tool, plan.request, out)
        with anyio.move_on_after(BEFORE_KILL_TIMEOUT_S) as probe_scope:
            await run_sync_daemon(_probe_collect, plan.tool, plan.request)
        if probe_scope.cancel_called:
            raise _fail("Tool.collect() did not return before the failed-run probe deadline")
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
    with any status other than ``ok`` gives a ``UserWarning`` with the last 2 KB of its
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
        Goal whose files the check sees under ``candidate/files/``. Its id
        must match ``candidate.goal_id``. None selects
        ``Goal(id="check", namespace="testing")`` and is refused for a
        candidate belonging to any other goal.
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
        If the candidate goal_id does not match the goal id, a file glob
        matches no file, or the check cannot be built.

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
    _check_goal(_DEFAULT_GOAL if goal is None else goal, [candidate])
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
    any status other than ``ok`` gives a ``UserWarning`` with the last 2 KB of its stderr
    log. If the call raises an error instead (a failed contract assert, or an
    error from ``judge()`` itself), each command with any status but ``ok``
    adds that text as a note to the error, which is otherwise passed on
    unchanged.

    Parameters
    ----------
    judge : Judge or Ref
        Judge instance or reference to run.
    goal : Goal
        Goal passed to the judge; its id must match every candidate goal_id.
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
        If any candidate goal_id does not match the goal id, a file glob
        matches no file, or the judge cannot be built.

    Examples
    --------
    >>> import anyio, sys, types, uuid
    >>> from swarm_arena import Goal, ProgramJudge, Score
    >>> from swarm_arena.board import Candidate
    >>> from swarm_arena.testing import check_judge, make_candidate
    >>> from swarm_arena.tool import RunStatus, ToolOutput
    >>> def quality(goal: Goal, candidate: Candidate) -> Score:
    ...     return Score.minimize(candidate.output.data["loss"])
    >>> module = types.ModuleType("kit_judge_example_" + uuid.uuid4().hex)
    >>> module.quality = quality
    >>> sys.modules[module.__name__] = module
    >>> goal = Goal(id="g", namespace="demo")
    >>> candidate = make_candidate(
    ...     ToolOutput(status=RunStatus.OK, data={"loss": 2.5}), goal=goal)
    >>> try:
    ...     scores = anyio.run(check_judge, ProgramJudge(module.__name__ + ":quality"),
    ...                        goal, [candidate])
    ...     assert len(scores) == 1
    ...     assert scores[0].candidate_id == candidate.id
    ...     assert scores[0].key == (-2.5,)
    ... finally:
    ...     del sys.modules[module.__name__]
    """
    _check_goal(goal, candidates)
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
