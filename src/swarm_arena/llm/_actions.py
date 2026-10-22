"""Adapt authorized manager actions to bounded, value-safe SDK function tools."""

import json
import math
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import anyio
from agents import FunctionTool
from agents.tool import default_tool_timeout_error_message, set_function_tool_failure_error_function
from agents.tool_context import ToolContext
from pydantic import ConfigDict, ValidationError, create_model

from swarm_arena._agent.context import (
    AgentContext,
    _ActionInflight,
    _defer_timer_error,
    _local_timer_scope,
    _LocalTimer,
)
from swarm_arena._agent.manager import LimitReached
from swarm_arena._agent.tool_run import Inflight, _inflight, track_inflight
from swarm_arena._board.digest import UNTRUSTED_CLOSE, UNTRUSTED_OPEN, untrusted
from swarm_arena._budget.llm_scope import requesting_call
from swarm_arena._errors import (
    Fenced,
    SoftError,
    _core_error,
    _exception_type_name,
    _rebuild_core_signal,
)
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.data import FiniteJsonObject, finite_json
from swarm_arena._model.diagnostics import validation_summary
from swarm_arena._model.results import Candidate, Score
from swarm_arena._outcomes import _Outcome
from swarm_arena.llm._hooks import _prepare, _sdk_control, _unwrap_control
from swarm_arena.llm._linking import ManagerDeps

MAX_RESULT_CHARS: Final = 4000
_TRUNCATED: Final = "…(truncated)"


class _Arguments(FrozenModel):
    model_config = ConfigDict(strict=True)


class _RunArguments(_Arguments):
    inputs: FiniteJsonObject
    parents: list[str] = []
    replicate: int | None = None


class _EditableRunArguments(_RunArguments):
    edits: dict[str, str] = {}


class _SearchArguments(_Arguments):
    query: str
    tags: list[str] = []
    scope: Literal["goal", "namespace"] = "goal"
    limit: int = 10


class _IdArguments(_Arguments):
    id: str


class _BestArguments(_Arguments):
    k: int
    part: str | None = None


class _LearningArguments(_Arguments):
    title: str
    text: str
    evidence: list[str]
    tags: list[str] = []
    refines: list[str] = []
    duplicate_reason: str | None = None


class _NotesArguments(_Arguments):
    text: str


class _FinishArguments(_Arguments):
    summary: str


def _score_view(scores: Sequence[Score]) -> list[dict[str, Any]]:
    return [{"judge": score.judge, "value": score.value} for score in scores]


def candidate_view(candidate: Candidate, scores: Sequence[Score]) -> dict[str, Any]:
    """
    Project a manager-visible candidate without judge-private details.

    Parameters
    ----------
    candidate : Candidate
        Candidate already feedback-redacted by AgentContext. Its producing
        policy decides whether check reasons are visible.
    scores : Sequence[Score]
        Scores obtained through AgentContext.scores_of for this candidate.

    Returns
    -------
    dict[str, Any]
        Status, request inputs, parents, sorted overlay names, data, checks,
        score values and warnings. Check details and score notes are omitted.
        Serialization and the result cap are applied by the tool adapter.

    Examples
    --------
    >>> from swarm_arena.board import Candidate
    >>> from swarm_arena.tool import ToolOutput, ToolRequest
    >>> candidate = Candidate(id="c", run_id="r", goal_id="g", team="t", agent="a",
    ...     round=1, request=ToolRequest(inputs={"x": 2}), output=ToolOutput(status="ok"))
    >>> candidate_view(candidate, [])["inputs"]
    {'x': 2}
    """
    return {
        "id": candidate.id,
        "status": candidate.output.status.value,
        "partial": candidate.output.partial,
        "inputs": candidate.request.inputs,
        "parents": candidate.parent_ids,
        "files": sorted(candidate.overlay),
        "data": candidate.output.data,
        "checks": [
            {"check": check.check, "passed": check.passed, "reasons": check.reasons}
            for check in candidate.checks
        ],
        "scores": _score_view(scores),
        "warnings": candidate.warnings,
    }


def _capped_json(text: str, cap: int) -> str:
    """Keep ordinary JSON intact; oversized output becomes a valid JSON string."""
    if len(text) <= cap:
        return text
    low, high = 0, min(len(text), cap)
    while low < high:
        middle = (low + high + 1) // 2
        if len(canonical_json(text[:middle] + _TRUNCATED).decode("utf-8")) <= cap:
            low = middle
        else:
            high = middle - 1
    return canonical_json(text[:low] + _TRUNCATED).decode("utf-8")


def _render(value: Any, *, board: bool = False) -> str:
    text = canonical_json(finite_json(value)).decode("utf-8")
    if not board:
        return _capped_json(text, MAX_RESULT_CHARS)
    # Escape before truncation, then retain the complete trust boundary even
    # when the JSON payload has to become a truncated string.
    wrapped = untrusted(text)
    safe = wrapped.removeprefix(UNTRUSTED_OPEN + "\n").removesuffix("\n" + UNTRUSTED_CLOSE)
    overhead = len(UNTRUSTED_OPEN) + len(UNTRUSTED_CLOSE) + 2
    return untrusted(_capped_json(safe, MAX_RESULT_CHARS - overhead))


def _parse(raw: str, model: type[_Arguments] | None) -> dict[str, Any]:
    try:
        value = json.loads(_context_string(raw, "arguments"))
    except json.JSONDecodeError:
        raise _core_error(SoftError, "invalid JSON action arguments") from None
    try:
        finite_json(value)
    except ValueError:
        raise _core_error(SoftError, "action arguments must be finite JSON") from None
    if not isinstance(value, dict):
        raise _core_error(SoftError, "action arguments must be an object")
    if model is None:
        return value
    try:
        return model.model_validate(value).model_dump(mode="json")
    except ValidationError as error:
        raise _core_error(SoftError, validation_summary(error, root="arguments")) from None


def _candidate_result(ctx: AgentContext, candidate: Candidate) -> dict[str, Any]:
    return candidate_view(candidate, ctx.scores_of(candidate.id))


def _read_result(ctx: AgentContext, name: str, args: dict[str, Any]) -> Any:
    """Marshal real published reads through AgentContext's captured loop token."""
    if name == "my_candidates":
        return [_candidate_result(ctx, candidate) for candidate in ctx.my_candidates()]
    if name == "get_candidate":
        return _candidate_result(ctx, ctx.get_candidate(args["id"]))
    if name == "search_learnings":
        return [
            {
                "id": learning.id,
                "title": learning.title,
                "author": learning.author,
                "verified": learning.verified,
                "pinned": learning.pinned,
            }
            for learning in ctx.search_learnings(**args)
        ]
    if name == "best":
        return [row.model_dump(mode="json") for row in ctx.best(**args)]
    learning = ctx.read_learning(args["id"])
    result = learning.model_dump(mode="json")
    evidence = []
    for candidate_id in learning.evidence:
        candidate = ctx.get_candidate(candidate_id)
        scores = ctx.scores_of(candidate_id)
        # ToolOutput has no error field. Core execution diagnostics are carried
        # by Candidate.warnings; overlay and past-run hints are not errors.
        error = (
            "; ".join(note for note in candidate.warnings if note.startswith("tool error:")) or None
            if not scores and candidate.output.status != "ok"
            else None
        )
        evidence.append({"id": candidate_id, "scores": _score_view(scores), "error": error})
    result["evidence"] = evidence
    return result


_READS: Final = {"my_candidates", "search_learnings", "read_learning", "get_candidate", "best"}


async def _act(
    ctx: AgentContext,
    deps: ManagerDeps,
    name: str,
    args: dict[str, Any],
    tool_call_id: str,
) -> str:
    if name in _READS:
        return await _prepare(
            lambda: _render(_read_result(ctx, name, args), board=name != "my_candidates")
        )
    if name == "run_tool":
        with requesting_call(deps.links.get(tool_call_id)):
            candidate = await ctx.run_tool(**args)
        return await _prepare(lambda: _render(_candidate_result(ctx, candidate)))
    if name == "post_learning":
        with requesting_call(deps.links.get(tool_call_id)):
            learning = await ctx.post_learning(**args)
        return await _prepare(lambda: _render(learning.model_dump(mode="json")))
    if name == "save_notes":
        await ctx.save_notes(args["text"])
        return _render({"ok": True})
    if name == "finish_round":
        if deps.summary is not None:
            raise _core_error(SoftError, "round already finished")
        deps.summary = args["summary"]
        return _render({"ok": True})
    result = await ctx.call_action(name, args)
    return await _prepare(lambda: _render(result, board=True))


@dataclass(eq=False)
class _OwnedCall:
    """Deliver an owned action result without giving the SDK task cancellation authority."""

    scope: anyio.CancelScope = field(default_factory=lambda: anyio.CancelScope(shield=True))
    finished: anyio.Event = field(default_factory=anyio.Event)
    inflight: _ActionInflight = field(
        default_factory=lambda: _ActionInflight(_inflight.get() or Inflight())
    )
    result: str = ""
    error: BaseException | None = None


class _OwnedActions:
    """Own all action bodies until core work has completed and controls are collected."""

    def __init__(self, group: anyio.abc.TaskGroup, timer: _LocalTimer) -> None:
        self.group = group
        self.timer = timer
        self.calls: set[_OwnedCall] = set()
        self.accepting = True

    async def call(self, operation: Callable[[], Awaitable[str]]) -> str:
        if not self.accepting:
            raise _core_error(LimitReached, "stop", "the round is stopping")
        call = _OwnedCall()
        self.calls.add(call)
        self.group.start_soon(self._run, operation, call)
        await call.finished.wait()
        if call.error is not None:
            raise call.error
        return call.result

    async def _run(self, operation: Callable[[], Awaitable[str]], call: _OwnedCall) -> None:
        try:
            with call.scope, _local_timer_scope(call.scope) as local:
                try:
                    with track_inflight(call.inflight):
                        call.result = await operation()
                except BaseException as error:
                    call.error = error
                    self.timer.record(error)
            if local.failure is not None:
                self.timer.record(local.failure)
        finally:
            self.calls.discard(call)
            call.finished.set()

    def stop(self, *, caller: bool) -> None:
        self.accepting = False
        for call in self.calls:
            call.inflight.stop(call.scope, cancel=caller)

    async def wait(self) -> None:
        while self.calls:
            await next(iter(self.calls)).finished.wait()


_owned_actions: ContextVar[_OwnedActions | None] = ContextVar("manager_owned_actions", default=None)


@contextmanager
def _own_actions(
    group: anyio.abc.TaskGroup, timer: _LocalTimer
) -> Generator[_OwnedActions, None, None]:
    """Bind one manager-owned action group through SDK-created invocation tasks."""
    owner = _OwnedActions(group, timer)
    token = _owned_actions.set(owner)
    try:
        yield owner
    finally:
        owner.accepting = False
        _owned_actions.reset(token)


def _tool(
    ctx: AgentContext,
    deps: ManagerDeps,
    *,
    name: str,
    description: str,
    schema: dict[str, Any],
    arguments: type[_Arguments] | None,
    timeout: float | None,
) -> FunctionTool:
    finish_lock = anyio.Lock() if name == "finish_round" else None

    async def execute(tool_context: ToolContext[Any], raw: str) -> str:
        outcome = _Outcome()
        error: BaseException | None = None
        result = ""
        parent = _inflight.get()
        inflight = (
            parent if isinstance(parent, _ActionInflight) else _ActionInflight(parent or Inflight())
        )
        finished = anyio.Event()
        scope = anyio.CancelScope()

        async def expire() -> None:
            assert timeout is not None
            with anyio.move_on_after(timeout) as clock:
                await finished.wait()
            if clock.cancel_called:
                inflight.stop(scope, cancel=True)

        with scope, _local_timer_scope(scope), track_inflight(inflight):
            async with anyio.create_task_group() as timers:
                if timeout is not None:
                    timers.start_soon(expire)
                try:
                    call_id = _context_string(tool_context.tool_call_id, "tool_call_id")
                    args = await _prepare(lambda: _parse(raw, arguments))
                    result = await _act(ctx, deps, name, args, call_id)
                except BaseException as caught:
                    error = caught
                finally:
                    finished.set()
        if error is not None:
            # An enclosing round timer may cancel the SDK invocation task. Its
            # owner retains the control even if the SDK discards this failure.
            _defer_timer_error(error)
        # Exiting our scope removes only its own cancellation requests. The
        # captured caller outcome now gives real owner cancellation precedence.
        # Check normal returns too: a handler may have swallowed cancellation.
        try:
            classified = outcome.classify(
                _unwrap_control(error if error is not None else TimeoutError())
            )
        except Fenced as fenced:
            classified = fenced
        if error is None and not scope.cancel_called:
            return result
        carrier = _sdk_control(classified)
        if carrier is not None:
            raise carrier from None
        if scope.cancel_called:
            assert timeout is not None
            return default_tool_timeout_error_message(tool_name=name, timeout_seconds=timeout)
        safe = _rebuild_core_signal(classified)
        message = (
            str(safe)
            if isinstance(safe, SoftError)
            else f"internal error in {name}: {_exception_type_name(classified)}"
        )
        return _render({"error": message})

    async def ordered(tool_context: ToolContext[Any], raw: str) -> str:
        if finish_lock is not None:
            async with finish_lock:
                return await execute(tool_context, raw)
        return await execute(tool_context, raw)

    async def invoke(tool_context: ToolContext[Any], raw: str) -> str:
        owner = _owned_actions.get()
        if owner is None:
            return await ordered(tool_context, raw)
        return await owner.call(lambda: ordered(tool_context, raw))

    return set_function_tool_failure_error_function(
        FunctionTool(
            name=name,
            description=description,
            params_json_schema=schema,
            on_invoke_tool=invoke,
            strict_json_schema=False,
            timeout_seconds=None,
            timeout_behavior="error_as_result",
        ),
        None,
    )


def build_tools(
    ctx: AgentContext, deps: ManagerDeps, *, action_timeout_s: float
) -> list[FunctionTool]:
    """
    Build SDK function tools backed exclusively by the authorized context.

    Schema hooks run synchronously. Async consumers must build these tools in
    their supervised, abandonable daemon preparation phase, before Runner.run.
    Invocations prepare arguments and returned values on that boundary; real
    context reads marshal back to its captured owner loop. Registered terminal
    outcomes use the shared SDK carrier and never become tool-error responses.

    Parameters
    ----------
    ctx : AgentContext
        Authorized actions and fixed, feedback-redacted published reads.
    deps : ManagerDeps
        Invocation-local LLM call links and final summary.
    action_timeout_s : float
        Adapter-owned timeout for every action except run_tool, whose clocks
        are core-owned. Ordinary timeouts return the SDK's timeout message;
        terminal or fenced unwind causes escape as controls. Caller cancellation
        takes precedence over both. The SDK's own tool timer is disabled.

    Returns
    -------
    list[FunctionTool]
        Nine standard actions followed by the context's declared extra actions.
        Every result is capped at 4000 characters, including trust delimiters.

    Raises
    ------
    ValueError
        If action names clash, schemas fail, or timeout configuration is invalid.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> from pydantic import BaseModel
    >>> from swarm_arena.llm._linking import CallLinks
    >>> class Inputs(BaseModel):
    ...     x: float
    >>> ctx = Mock(input_model=Inputs, editable=(), actions=(), tool_description="Trial")
    >>> deps = ManagerDeps(ctx=ctx, links=CallLinks())
    >>> tools = build_tools(ctx, deps, action_timeout_s=60.0)
    >>> tools[0].name, tools[0].timeout_seconds, tools[-1].name
    ('run_tool', None, 'finish_round')
    """
    outcome = _Outcome(capture_caller=False)
    try:
        if (
            type(action_timeout_s) not in (int, float)
            or not math.isfinite(action_timeout_s)
            or action_timeout_s <= 0
        ):
            raise _core_error(ValueError, "action timeout must be a positive finite number")
        run_arguments = _EditableRunArguments if ctx.editable else _RunArguments
        definitions: dict[str, tuple[type[_Arguments], str]] = {
            "run_tool": (run_arguments, _context_string(ctx.tool_description, "description")),
            "my_candidates": (_Arguments, "Read your candidates with checks and scores."),
            "search_learnings": (
                _SearchArguments,
                "Search visible learnings in goal or namespace scope.",
            ),
            "read_learning": (_IdArguments, "Read a learning and its evidence scores or errors."),
            "get_candidate": (
                _IdArguments,
                "Read a visible candidate summary with checks and scores.",
            ),
            "best": (
                _BestArguments,
                "Read the fixed leaderboard for the current or a declared part.",
            ),
            "post_learning": (_LearningArguments, "Post an evidence-backed learning."),
            "save_notes": (_NotesArguments, "Save private notes for the next round."),
            "finish_round": (_FinishArguments, "Finish the round with a summary."),
        }
        tools = []
        for name, (arguments, description) in definitions.items():
            schema_model = (
                create_model("_ToolArguments", __base__=arguments, inputs=(ctx.input_model, ...))
                if name == "run_tool"
                else arguments
            )
            schema = json.loads(canonical_json(finite_json(schema_model.model_json_schema())))
            tools.append(
                _tool(
                    ctx,
                    deps,
                    name=name,
                    description=description,
                    schema=schema,
                    arguments=arguments,
                    timeout=None if name == "run_tool" else action_timeout_s,
                )
            )
        names = set(definitions)
        for action in ctx.actions:
            name = _context_string(action.name, "name")
            if name in names:
                raise _core_error(ValueError, f"duplicate action name: {name}")
            names.add(name)
            schema = json.loads(canonical_json(finite_json(action.input_model.model_json_schema())))
            tools.append(
                _tool(
                    ctx,
                    deps,
                    name=name,
                    description=_context_string(action.description, "description"),
                    schema=schema,
                    arguments=None,
                    timeout=action_timeout_s,
                )
            )
        return tools
    except BaseException as error:
        outcome.contain(error, ValueError, "manager tool preparation")
