"""Manager function-tool contracts at the SDK and real context boundaries."""

import asyncio
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event, get_ident
from typing import Any, cast
from unittest.mock import Mock

import anyio
import pytest

pytest.importorskip("agents")

from _agent.conftest import ctx_env as ctx_env
from _agent.conftest import rt_env as rt_env
from _fakes.agent_ctx import FakeAgentContext
from _fakes.board import score
from _fakes.engine import CtxEnv
from _fakes.journals import FakeRun
from _fakes.pr1_r5_ids import hostile
from _fakes.secrecy import assert_secret_absent
from agents import Agent, FunctionTool, ModelResponse, Runner
from agents import RunConfig as SDKRunConfig
from agents.exceptions import AgentsException
from agents.tool import resolve_function_tool_failure_error_function
from agents.tool_context import ToolContext
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall
from pydantic import Field

from swarm_arena import AgentSpec, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
from swarm_arena._agent.context import _prepare_metadata
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.digest import UNTRUSTED_CLOSE, UNTRUSTED_OPEN, untrusted
from swarm_arena._board.pointers import RunPointer, write_pointer
from swarm_arena._errors import (
    Fenced,
    SoftError,
    _core_error,
    _is_core_control,
    _stop_cleanup_error,
)
from swarm_arena._model.base import FrozenModel, JsonValue
from swarm_arena._model.canonical import canonical_json
from swarm_arena._store.base import open_store
from swarm_arena.board import Row
from swarm_arena.llm._actions import MAX_RESULT_CHARS, build_tools, candidate_view
from swarm_arena.llm._hooks import _prepare, _unwrap_control
from swarm_arena.llm._linking import CallLinks, ManagerDeps
from swarm_arena.manager import AgentContext, LimitReached
from swarm_arena.scoring import CheckResult
from swarm_arena.testing import FakeModel, FakeTurn
from swarm_arena.tool import Action, ToolOutput, ToolRequest


class Inputs(FrozenModel):
    x: float = 0


class Empty(FrozenModel):
    pass


def fake_context(**kwargs: Any) -> FakeAgentContext:
    return FakeAgentContext(input_model=Inputs, **kwargs)


def tool_set(ctx: AgentContext | FakeAgentContext) -> tuple[list[FunctionTool], ManagerDeps]:
    context = cast(AgentContext, ctx)
    deps = ManagerDeps(ctx=context, links=CallLinks())
    return build_tools(context, deps, action_timeout_s=60.0), deps


def named(tools: list[FunctionTool], name: str) -> FunctionTool:
    return next(tool for tool in tools if tool.name == name)


async def invoke(
    tools: list[FunctionTool],
    deps: ManagerDeps,
    name: str,
    args: dict[str, Any] | str,
    *,
    call_id: str = "tool-1",
) -> str:
    raw = args if isinstance(args, str) else json.dumps(args)
    context = ToolContext(context=deps, tool_name=name, tool_call_id=call_id, tool_arguments=raw)
    return await named(tools, name).on_invoke_tool(context, raw)


def decoded(text: str) -> Any:
    if text.startswith(UNTRUSTED_OPEN):
        text = text.removeprefix(UNTRUSTED_OPEN + "\n").removesuffix("\n" + UNTRUSTED_CLOSE)
    return json.loads(text)


def link(deps: ManagerDeps, call_id: str, tool_id: str, name: str = "run_tool") -> None:
    deps.links.add(
        call_id,
        ModelResponse(
            output=[
                ResponseFunctionToolCall(
                    type="function_call", name=name, call_id=tool_id, arguments="{}"
                )
            ],
            usage=Usage(),
            response_id=None,
        ),
    )


def test_run_tool_has_no_sdk_timeout() -> None:
    tools, _ = tool_set(fake_context())
    assert named(tools, "run_tool").timeout_seconds is None
    assert [tool.name for tool in tools] == [
        "run_tool",
        "my_candidates",
        "search_learnings",
        "read_learning",
        "get_candidate",
        "best",
        "post_learning",
        "save_notes",
        "finish_round",
    ]
    assert all(not tool.strict_json_schema for tool in tools)
    assert all(resolve_function_tool_failure_error_function(tool) is None for tool in tools)


def test_other_actions_time_out_as_results() -> None:
    tools, _ = tool_set(fake_context(actions=tuple(ScriptTool(command=["echo"]).extra_actions())))
    assert {(t.timeout_seconds, t.timeout_behavior) for t in tools if t.name != "run_tool"} == {
        (None, "error_as_result")
    }


@pytest.mark.parametrize("editable", [(), ("*.txt",)])
def test_edits_property_only_when_editable(editable: tuple[str, ...]) -> None:
    tools, _ = tool_set(fake_context(editable=editable))
    schema = named(tools, "run_tool").params_json_schema
    assert ("edits" in schema["properties"]) == bool(editable)
    if editable:
        assert schema["properties"]["edits"]["type"] == "object"
        assert schema["properties"]["edits"]["additionalProperties"] == {"type": "string"}
    assert schema["properties"]["parents"]["items"]["type"] == "string"
    assert schema["properties"]["replicate"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert schema["required"] == ["inputs"]


def test_input_schema_keeps_nested_definitions_and_constraints() -> None:
    class Nested(FrozenModel):
        count: int = Field(ge=2)

    class Model(FrozenModel):
        nested: Nested

    tools, _ = tool_set(FakeAgentContext(input_model=Model))
    schema = named(tools, "run_tool").params_json_schema
    assert schema["$defs"]["Nested"]["properties"]["count"]["minimum"] == 2
    assert schema["$defs"]["Model"]["properties"]["nested"]["$ref"] == "#/$defs/Nested"


def test_extra_actions_exposed_with_their_schema() -> None:
    actions = tuple(ScriptTool(command=["echo"]).extra_actions())
    tools, _ = tool_set(fake_context(actions=actions))
    assert [tool.name for tool in tools][-2:] == ["list_files", "read_file"]
    for action in actions:
        assert (
            named(tools, action.name).params_json_schema == action.input_model.model_json_schema()
        )


@pytest.mark.parametrize("name", ["run_tool", "same"])
def test_extra_action_name_clashes_are_rejected(name: str) -> None:
    async def handler(ctx: AgentContext, args: Empty) -> JsonValue:
        return None

    action = Action(name=name, description="Test", input_model=Empty, handler=handler)
    with pytest.raises(ValueError, match="duplicate action name"):
        tool_set(fake_context(actions=(action,) if name == "run_tool" else (action, action)))


async def test_soft_error_comes_back_as_tool_result() -> None:
    fake = fake_context(raise_on={"post_learning": _core_error(SoftError, "duplicate learning")})
    tools, deps = tool_set(fake)
    result = await invoke(
        tools, deps, "post_learning", {"title": "Hint", "text": "Try", "evidence": ["c-1"]}
    )
    assert json.loads(result)["error"] == "duplicate learning"


@pytest.mark.parametrize("error", [ValueError, SoftError, RuntimeError])
async def test_unregistered_error_messages_are_never_model_visible(error: type[Exception]) -> None:
    secret = "secret-action-input"
    fake = fake_context(raise_on={"save_notes": error(secret)})
    tools, deps = tool_set(fake)
    result = await invoke(tools, deps, "save_notes", {"text": secret})
    assert secret not in result
    assert decoded(result)["error"] == f"internal error in save_notes: {error.__name__}"


@pytest.mark.parametrize(
    "raw",
    [
        '{"inputs": {"x": "secret-json"}',
        '["secret-json"]',
        '{"inputs":{},"replicate":true}',
        '{"inputs":{},"parents":"secret-json"}',
        '{"inputs":{},"edits":{"secret-json":"secret-json"}}',
        '{"inputs":{"x":NaN}}',
        '{"inputs":{},"extra":"secret-json"}',
    ],
)
async def test_invalid_arguments_are_safe_and_do_not_invoke_actions(raw: str) -> None:
    fake = fake_context()
    tools, deps = tool_set(fake)
    result = await invoke(tools, deps, "run_tool", raw)
    assert "error" in decoded(result)
    assert "secret-json" not in result
    assert fake.candidates == []


@pytest.mark.parametrize("kind", ["limit", "overflow", "fence", "cleanup"])
@pytest.mark.parametrize("nested", [False, True])
async def test_limit_reached_in_action_ends_the_run(kind: str, nested: bool) -> None:
    error = {
        "limit": _core_error(LimitReached, "max_tool_runs_per_round", "3 runs"),
        "overflow": _core_error(LimitReached, "usage_overflow", "usage overflow"),
        "fence": _core_error(Fenced, "run ownership lost"),
        "cleanup": _stop_cleanup_error(),
    }[kind]
    fake = fake_context(
        raise_on={
            "run_tool": ExceptionGroup("private", [ValueError("private"), error])
            if nested
            else error
        }
    )
    tools, deps = tool_set(fake)
    model = FakeModel(
        [FakeTurn(tool_calls=(("run_tool", {"inputs": {"x": 1}}),)), FakeTurn(text="must not run")]
    )
    with pytest.raises(AgentsException) as caught:
        await Runner.run(
            Agent(name="test", tools=[*tools], model=model),
            "go",
            context=deps,
            run_config=SDKRunConfig(tracing_disabled=True),
        )
    assert len(model.calls) == 1
    control = _unwrap_control(caught.value)
    assert _is_core_control(control)
    if kind == "limit":
        assert isinstance(control, LimitReached)
        assert control.limit == "max_tool_runs_per_round"
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("nested", [False, True])
async def test_real_action_terminal_cleanup_reaches_sdk_without_another_turn(
    ctx_env: CtxEnv, nested: bool
) -> None:
    ctx = ctx_env.ctx()

    async def handler(context: AgentContext, args: Empty) -> JsonValue:
        error = _stop_cleanup_error()
        if nested:
            raise ExceptionGroup("secret-action", [RuntimeError("secret-action"), error])
        raise error

    ctx.actions = (
        Action(name="cleanup", description="Test cleanup", input_model=Empty, handler=handler),
    )
    tools, deps = await _prepare(partial(tool_set, ctx))
    model = FakeModel([FakeTurn(tool_calls=(("cleanup", {}),)), FakeTurn(text="must not run")])
    with pytest.raises(AgentsException) as caught:
        await Runner.run(
            Agent(name="test", tools=[*tools], model=model),
            "go",
            context=deps,
            run_config=SDKRunConfig(tracing_disabled=True),
        )
    control = _unwrap_control(caught.value)
    assert _is_core_control(control)
    assert str(control) == "stop cleanup timed out"
    assert len(model.calls) == 1
    assert_secret_absent(caught.value, "secret-action")


@pytest.mark.parametrize("kind", ["cleanup", "fence"])
@pytest.mark.parametrize("timer_first", [False, True])
@pytest.mark.parametrize("nested", [False, True])
async def test_real_action_control_beats_owned_timer(
    ctx_env: CtxEnv, kind: str, timer_first: bool, nested: bool
) -> None:
    entered = anyio.Event()
    raised: list[str] = []

    async def handler(context: AgentContext, args: Empty) -> JsonValue:
        entered.set()
        try:
            if timer_first:
                await anyio.sleep_forever()
        finally:
            raised.append(kind)
            cause = (
                _stop_cleanup_error()
                if kind == "cleanup"
                else _core_error(Fenced, "run ownership lost")
            )
            if nested:
                raise BaseExceptionGroup("secret-unwind", [asyncio.CancelledError(), cause])
            raise cause

    ctx = ctx_env.ctx()
    ctx.actions = (Action(name="held", description="Held", input_model=Empty, handler=handler),)
    deps = ManagerDeps(ctx=ctx, links=CallLinks())
    tools = await _prepare(lambda: build_tools(ctx, deps, action_timeout_s=1.0))
    model = FakeModel([FakeTurn(tool_calls=(("held", {}),)), FakeTurn(text="must not run")])
    with anyio.fail_after(10), pytest.raises(AgentsException) as caught:
        await Runner.run(
            Agent(name="test", tools=[*tools], model=model),
            "go",
            context=deps,
            run_config=SDKRunConfig(tracing_disabled=True),
        )
    assert entered.is_set()
    assert raised == [kind]
    control = _unwrap_control(caught.value)
    assert _is_core_control(control)
    assert str(control) == ("stop cleanup timed out" if kind == "cleanup" else "run ownership lost")
    assert len(model.calls) == 1
    assert_secret_absent(caught.value, "secret-unwind")


@pytest.mark.parametrize("kind", ["cleanup", "fence"])
@pytest.mark.parametrize("timer_first", [False, True])
@pytest.mark.parametrize("owner", ["task", "scope"])
async def test_real_owner_cancellation_beats_timer_and_unwind_control(
    ctx_env: CtxEnv, kind: str, timer_first: bool, owner: str
) -> None:
    entered, unwinding, release = anyio.Event(), anyio.Event(), anyio.Event()
    raised: list[str] = []

    async def handler(context: AgentContext, args: Empty) -> JsonValue:
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            unwinding.set()
            with anyio.CancelScope(shield=True):
                try:
                    await release.wait()
                finally:
                    raised.append(kind)
                    raise (
                        _stop_cleanup_error()
                        if kind == "cleanup"
                        else _core_error(Fenced, "run ownership lost")
                    )

    ctx = ctx_env.ctx()
    ctx.actions = (Action(name="held", description="Held", input_model=Empty, handler=handler),)
    deps = ManagerDeps(ctx=ctx, links=CallLinks())
    tools = await _prepare(
        lambda: build_tools(ctx, deps, action_timeout_s=1.0 if timer_first else 60.0)
    )
    model = FakeModel([FakeTurn(tool_calls=(("held", {}),)), FakeTurn(text="must not run")])
    owner_scope = anyio.CancelScope()

    async def run() -> None:
        with owner_scope:
            await Runner.run(
                Agent(name="test", tools=[*tools], model=model),
                "go",
                context=deps,
                run_config=SDKRunConfig(tracing_disabled=True),
            )
            pytest.fail("owner cancellation became a tool response")

    task = asyncio.create_task(run())
    try:
        with anyio.fail_after(10):
            await entered.wait()
            if timer_first:
                await unwinding.wait()
            if owner == "task":
                task.cancel()
            else:
                owner_scope.cancel()
            await unwinding.wait()
            release.set()
            if owner == "task":
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
                assert owner_scope.cancelled_caught
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert raised == [kind]
    assert len(model.calls) == 1


async def test_requested_by_call_comes_from_the_tool_call_id() -> None:
    fake = fake_context()
    tools, deps = tool_set(fake)
    link(deps, "llm-1", "tool-1")
    await invoke(tools, deps, "run_tool", {"inputs": {"x": 1}})
    link(deps, "llm-2", "tool-2", "post_learning")
    await invoke(
        tools,
        deps,
        "post_learning",
        {"title": "Hint", "text": "Try", "evidence": ["c-1"]},
        call_id="tool-2",
    )
    assert fake.requesting_calls == [("run_tool", "llm-1"), ("post_learning", "llm-2")]


async def test_parallel_run_tool_calls_keep_their_own_call_ids() -> None:
    entered = 0
    ready = anyio.Event()

    class Concurrent(FakeAgentContext):
        async def run_tool(self, inputs: Any, **kwargs: Any) -> Any:
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            return await super().run_tool(inputs, **kwargs)

    fake = Concurrent(input_model=Inputs)
    tools, deps = tool_set(fake)
    for index in (1, 2):
        link(deps, f"llm-{index}", f"tool-{index}")
    with anyio.fail_after(5):
        await asyncio.gather(
            *(
                invoke(tools, deps, "run_tool", {"inputs": {"x": index}}, call_id=f"tool-{index}")
                for index in (1, 2)
            )
        )
    assert sorted(fake.requesting_calls) == [("run_tool", "llm-1"), ("run_tool", "llm-2")]


def test_candidate_view_never_holds_details() -> None:
    candidate = (
        fake_context()
        .add_candidate("c-1")
        .model_copy(
            update={
                "checks": [
                    CheckResult(check="hidden", passed=False, reasons=[], details={"case": 3})
                ]
            }
        )
    )
    view = candidate_view(candidate, [])
    assert "details" not in json.dumps(view)
    assert view["checks"] == [{"check": "hidden", "passed": False, "reasons": []}]


def test_candidate_view_has_scores_parents_inputs_and_files() -> None:
    candidate = (
        fake_context()
        .add_candidate("c-2")
        .model_copy(
            update={
                "request": ToolRequest(inputs={"x": 3}),
                "parent_ids": ["c-1"],
                "overlay": {"z.txt": "b" * 64, "a.txt": "a" * 64},
            }
        )
    )
    view = candidate_view(candidate, [score("s-1", "c-2", 1.5, judge="f", value=1.5)])
    assert (view["scores"], view["parents"], view["files"], view["inputs"]) == (
        [{"judge": "f", "value": 1.5}],
        ["c-1"],
        ["a.txt", "z.txt"],
        candidate.request.inputs,
    )


async def test_my_candidates_and_get_candidate_show_scores() -> None:
    fake = fake_context()
    fake.add_candidate("c-1")
    fake.scores.append(score("s-1", "c-1", 1.5, judge="f", value=1.5))
    tools, deps = tool_set(fake)
    own = decoded(await invoke(tools, deps, "my_candidates", {}))[0]
    other = decoded(await invoke(tools, deps, "get_candidate", {"id": "c-1"}))
    assert own["scores"] == other["scores"] == [{"judge": "f", "value": 1.5}]


async def test_read_learning_gives_each_evidence_score_or_error() -> None:
    fake = fake_context()
    fake.add_candidate("c-1")
    candidate = fake.add_candidate("c-2")
    fake.candidates[-1] = candidate.model_copy(
        update={
            "output": ToolOutput(status="error"),
            "warnings": ["parent c-0 not overlaid", "tool error: RuntimeError"],
        }
    )
    fake.add_candidate("c-3")
    fake.scores.append(score("s-1", "c-1", 2.0, judge="f", value=2.0))
    learning = await fake.post_learning("Hint", "Evidence", ["c-1", "c-2", "c-3"])
    tools, deps = tool_set(fake)
    result = decoded(await invoke(tools, deps, "read_learning", {"id": learning.id}))
    assert result["evidence"] == [
        {"id": "c-1", "scores": [{"judge": "f", "value": 2.0}], "error": None},
        {"id": "c-2", "scores": [], "error": "tool error: RuntimeError"},
        {"id": "c-3", "scores": [], "error": None},
    ]


@pytest.mark.parametrize("max_reads", [1, 2, 10])
@pytest.mark.parametrize("evidence_count", [1, 60])
async def test_read_learning_keeps_cross_goal_text_in_a_real_fixed_view(
    tmp_path: Path, max_reads: int, evidence_count: int
) -> None:
    current = FakeRun(str(tmp_path / "runs"), "current", goal_id="new-goal")
    current.commit(current.plan())
    prior = FakeRun(current.run_root, "prior", goal_id="old-goal")
    text = "Use slow cooling for lower viscosity."
    evidence_ids = (
        ["old-candidate"]
        if evidence_count == 1
        else [f"old-candidate-{i:02}" for i in range(evidence_count)]
    )
    old = [prior.candidate(id) for id in evidence_ids]
    old[0].payload["candidate"]["output"]["data"] = {"private": "out-of-view-candidate-secret"}
    prior.commit(
        prior.plan(),
        *old,
        prior.score("old-score", evidence_ids[0], 1),
        prior.learning("L-shared", "viscosity", text=text, evidence=evidence_ids),
    )
    arena_root = str(tmp_path / "arena")
    board = ArenaBoard(arena_root=arena_root, cache_dir=str(tmp_path / "cache"))
    try:
        write_pointer(
            open_store(arena_root),
            "toy",
            RunPointer(run_id=prior.run_id, run_root=prior.run_root),
        )
        board.open(current.run_id, current.run_root, ["toy"])
        view = board.snapshot(current.goal.key(), board.frontier())
        assert view.search_learnings("toy", "viscosity", goal=current.goal.key()) == []
        assert view.get_learning("L-shared").text == text
        with pytest.raises(KeyError):
            view.get_candidate(evidence_ids[0])

        # Refreshing the board cannot widen the capability already given to the agent.
        prior.commit(prior.learning("L-later", "viscosity", text="later-learning-text"))
        board.frontier()
        tool = ScriptTool(command=["echo"])
        spec = AgentSpec(
            name="agent",
            rationale="Explore",
            tool=Ref.of(tool),
            manager=Ref(path="example:Manager"),
        )
        runtime = Mock(
            goal=current.goal,
            goal_id=current.goal.id,
            goal_key=current.goal.key(),
            address="new-goal/team/agent",
            team=TeamSpec(name="team", rationale="Explore", agents=[spec]),
            round=1,
            round_spec=RoundSpec(),
            digest="",
            notes="",
            resumed=(),
            checks=[],
            tool=tool,
            own=[],
            limits=Limits(max_board_reads_per_round=max_reads),
            watchdog=Mock(limit=None),
            counters=Mock(turns=0, board_reads=0, learnings=0),
            part=None,
            rankings={"": "j"},
            sibling_views={},
            view=view,
            repeats=Mock(run_id=current.run_id),
        )
        runtime.spec = spec
        tools, deps = await _prepare(partial(tool_set, AgentContext(runtime)))
        found = decoded(
            await invoke(
                tools, deps, "search_learnings", {"query": "viscosity", "scope": "namespace"}
            )
        )
        assert [item["id"] for item in found] == ["L-shared"]
        raw = await invoke(tools, deps, "read_learning", {"id": "L-shared"})
        result = decoded(raw)
        if max_reads == 1:
            assert result == {"error": "max_board_reads_per_round reached"}
            return
        assert isinstance(result, dict)
        assert result["text"] == text
        retained = len(result["evidence"])
        assert result["evidence"] == [
            {"id": id, "scores": [], "error": "not in this view"} for id in evidence_ids[:retained]
        ]
        assert len(raw) <= MAX_RESULT_CHARS
        assert raw == untrusted(canonical_json(result).decode("utf-8"))
        if evidence_count == 60:
            omitted = result["evidence_omitted"]
            assert 0 < retained < evidence_count
            assert omitted == evidence_count - retained
            expanded: dict[str, Any] = {
                **result,
                "evidence": [
                    *result["evidence"],
                    {"id": evidence_ids[retained], "scores": [], "error": "not in this view"},
                ],
                "evidence_omitted": omitted - 1,
            }
            assert len(untrusted(canonical_json(expanded).decode("utf-8"))) > MAX_RESULT_CHARS
        else:
            assert retained == evidence_count
            assert "evidence_omitted" not in result
        assert raw.startswith(UNTRUSTED_OPEN)
        assert raw.endswith(UNTRUSTED_CLOSE)
        assert "out-of-view-candidate-secret" not in raw
        assert (runtime.counters.turns, runtime.counters.board_reads) == (2, 2)
        if max_reads == 2:
            return
        assert decoded(await invoke(tools, deps, "read_learning", {"id": "L-later"})) == {
            "error": "unknown learning: L-later"
        }
        assert decoded(await invoke(tools, deps, "get_candidate", {"id": evidence_ids[0]})) == {
            "error": f"unknown candidate: {evidence_ids[0]}"
        }
    finally:
        board.close()


@pytest.mark.parametrize("case", ["escaped_text", "oversized_evidence_id"])
async def test_learning_text_takes_priority_over_evidence_at_result_cap(case: str) -> None:
    fake = fake_context()
    ids = (
        [f"c-{i}" for i in range(60)] if case == "escaped_text" else ["c-" + "x" * MAX_RESULT_CHARS]
    )
    for id in ids:
        fake.add_candidate(id)
    text = '"\\\n🙂' * 300 if case == "escaped_text" else "Keep the learning text."
    learning = await fake.post_learning("Hint", text, ids)
    tools, deps = tool_set(fake)
    raw = await invoke(tools, deps, "read_learning", {"id": learning.id})
    result = decoded(raw)
    assert isinstance(result, dict)
    assert result["text"] == text
    retained = len(result["evidence"])
    assert result["evidence_omitted"] == len(ids) - retained
    assert result["evidence_omitted"] > 0
    assert [item["id"] for item in result["evidence"]] == ids[:retained]
    assert len(raw) <= MAX_RESULT_CHARS
    assert raw == untrusted(canonical_json(result).decode("utf-8"))
    if case == "oversized_evidence_id":
        assert result["evidence"] == []


@pytest.mark.parametrize("visibility", ["visible", "foreign"])
@pytest.mark.parametrize("search_first", [False, True])
async def test_long_learning_evidence_preserves_text_at_default_read_limit(
    ctx_env: CtxEnv, visibility: str, search_first: bool
) -> None:
    fake = fake_context()
    batch = [fake.add_candidate(f"c-{i}") for i in range(9)]
    learning = await fake.post_learning("Hint", "Keep this authorized text.", [c.id for c in batch])
    ctx = ctx_env.ctx(own=[learning, *(batch if visibility == "visible" else [])], limits=Limits())
    tools, deps = await _prepare(partial(tool_set, ctx))
    if search_first:
        found = decoded(await invoke(tools, deps, "search_learnings", {"query": "Hint"}))
        assert [item["id"] for item in found] == [learning.id]
    result = decoded(await invoke(tools, deps, "read_learning", {"id": learning.id}))
    assert result["text"] == learning.text
    if visibility == "visible":
        assert result["evidence"] == [
            {"id": c.id, "scores": [], "error": None if i < 4 else "read limit reached"}
            for i, c in enumerate(batch)
        ]
        assert ctx_env.counters.board_reads == Limits().max_board_reads_per_round == 10
        assert ctx_env.counters.turns == 11
    else:
        assert result["evidence"] == [
            {"id": c.id, "scores": [], "error": "not in this view"} for c in batch
        ]
        assert ctx_env.counters.board_reads == ctx_env.counters.turns == 1 + search_first


async def test_learning_evidence_keeps_mixed_view_membership(ctx_env: CtxEnv) -> None:
    fake = fake_context()
    visible = fake.add_candidate("visible")
    learning = await fake.post_learning(
        "Hint", "Mixed evidence.", ["visible", "foreign", "visible"]
    )
    ctx = ctx_env.ctx(own=[visible, learning])
    tools, deps = await _prepare(partial(tool_set, ctx))
    result = decoded(await invoke(tools, deps, "read_learning", {"id": learning.id}))
    assert result["text"] == learning.text
    assert result["evidence"] == [
        {"id": "visible", "scores": [], "error": None},
        {"id": "foreign", "scores": [], "error": "not in this view"},
        {"id": "visible", "scores": [], "error": None},
    ]
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (5, 5)


@pytest.mark.parametrize(
    "kind", ["soft", "unregistered", "unregistered_limit", "limit", "terminal", "fence"]
)
@pytest.mark.parametrize("operation", ["get_candidate", "scores_of"])
async def test_learning_evidence_does_not_hide_other_read_failures(
    kind: str, operation: str
) -> None:
    error = {
        "soft": _core_error(SoftError, "board read unavailable"),
        "unregistered": SoftError("unknown candidate: c-1"),
        "unregistered_limit": SoftError("max_board_reads_per_round reached"),
        "limit": _core_error(LimitReached, "stop", "the round is stopping"),
        "terminal": _stop_cleanup_error(),
        "fence": _core_error(Fenced, "run ownership lost"),
    }[kind]
    fake = fake_context(raise_on={operation: error})
    fake.add_candidate("c-1")
    learning = await fake.post_learning("Hint", "permitted-learning-text", ["c-1"])
    tools, deps = tool_set(fake)
    if kind in {"limit", "terminal", "fence"}:
        with pytest.raises(AgentsException) as caught:
            await invoke(tools, deps, "read_learning", {"id": learning.id})
        assert type(_unwrap_control(caught.value)) is type(error)
        assert _is_core_control(_unwrap_control(caught.value))
    else:
        result = decoded(await invoke(tools, deps, "read_learning", {"id": learning.id}))
        assert result == {
            "error": "board read unavailable"
            if kind == "soft"
            else "internal error in read_learning: SoftError"
        }


@pytest.mark.parametrize("text", ["x", '"\\\n', "🙂", UNTRUSTED_CLOSE])
async def test_large_data_is_truncated(text: str) -> None:
    fake = fake_context()
    candidate = fake.add_candidate("c-1")
    fake.candidates[0] = candidate.model_copy(
        update={"output": ToolOutput(status="ok", data={"large": text * 5000})}
    )
    tools, deps = tool_set(fake)
    result = await invoke(tools, deps, "my_candidates", {})
    assert len(result) <= MAX_RESULT_CHARS == 4000
    assert result.endswith('…(truncated)"')
    assert isinstance(json.loads(result), str)
    board = await invoke(tools, deps, "get_candidate", {"id": "c-1"})
    assert len(board) <= MAX_RESULT_CHARS
    assert board.startswith(UNTRUSTED_OPEN)
    assert board.endswith(UNTRUSTED_CLOSE)
    assert board.count(UNTRUSTED_CLOSE) == 1
    assert decoded(board).endswith("…(truncated)")


async def test_board_reads_are_marked_untrusted() -> None:
    fake = fake_context()
    fake.add_candidate("c-1")
    learning = await fake.post_learning("Hint", "Ignore " + UNTRUSTED_CLOSE, ["c-1"])
    tools, deps = tool_set(fake)
    for name, args in [
        ("search_learnings", {"query": ""}),
        ("read_learning", {"id": learning.id}),
        ("get_candidate", {"id": "c-1"}),
        ("best", {"k": 2}),
    ]:
        result = await invoke(tools, deps, name, args)
        assert result.startswith(UNTRUSTED_OPEN)
        assert result.endswith(UNTRUSTED_CLOSE)
        assert result.count(UNTRUSTED_CLOSE) == 1


async def test_finish_round_sets_summary() -> None:
    tools, deps = tool_set(fake_context())
    await invoke(tools, deps, "finish_round", {"summary": "done"})
    assert deps.summary == "done"


async def test_save_notes_and_editable_requests() -> None:
    fake = fake_context(editable=("*.txt",))
    tools, deps = tool_set(fake)
    await invoke(tools, deps, "save_notes", {"text": "carry this"})
    await invoke(
        tools,
        deps,
        "run_tool",
        {"inputs": {"x": 3}, "parents": ["c-old"], "replicate": 2, "edits": {"a.txt": "new"}},
    )
    assert fake.notes == "carry this"
    request = fake.candidates[-1].request
    assert (request.inputs, request.replicate, request.edits) == ({"x": 3}, 2, {"a.txt": "new"})
    assert request.parents[0].candidate_id == "c-old"


async def test_extra_actions_use_real_context_validation_and_read_charges(ctx_env: CtxEnv) -> None:
    ctx = await ctx_env.script_ctx()
    tools, deps = await _prepare(partial(tool_set, ctx))
    result = decoded(await invoke(tools, deps, "list_files", {}))
    assert result == {"files": ["problem/a.phyk", "problem/run.py"]}
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (1, 1)
    bad = await invoke(tools, deps, "read_file", {"path": ["secret-path"]})
    assert "error" in decoded(bad)
    assert "secret-path" not in bad
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (2, 2)
    content = decoded(await invoke(tools, deps, "read_file", {"path": "problem/a.phyk"}))
    assert content == {"path": "problem/a.phyk", "content": "a = 1\n"}
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (3, 3)


async def test_real_reads_keep_feedback_redaction_and_charges(ctx_env: CtxEnv) -> None:
    fake = fake_context()
    candidate = fake.add_candidate("c-1").model_copy(
        update={
            "checks": [
                CheckResult(
                    check="hidden",
                    passed=False,
                    reasons=["private-reason"],
                    details={"secret": True},
                )
            ]
        }
    )
    ctx = ctx_env.ctx(own=[candidate], limits=Limits(max_turns_per_round=20))
    tools, deps = await _prepare(partial(tool_set, ctx))
    result = await invoke(tools, deps, "get_candidate", {"id": "c-1"})
    assert decoded(result)["checks"] == [{"check": "hidden", "passed": False, "reasons": []}]
    assert "private-reason" not in result
    assert "secret" not in result
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (2, 2)


async def test_schema_hooks_run_in_abandonable_preparation() -> None:
    entered, release, finished = Event(), Event(), Event()
    loop_thread = get_ident()

    class Blocking(Inputs):
        @classmethod
        def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> dict[str, Any]:
            assert get_ident() != loop_thread
            entered.set()
            try:
                assert release.wait(10)
                return handler(schema)
            finally:
                finished.set()

    task = asyncio.create_task(_prepare(partial(tool_set, FakeAgentContext(input_model=Blocking))))
    try:
        with anyio.fail_after(5):
            await anyio.to_thread.run_sync(entered.wait, 4)
            assert entered.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not finished.is_set()
    finally:
        release.set()
        await anyio.to_thread.run_sync(finished.wait, 5)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert finished.is_set()


@pytest.mark.parametrize("swallow", [False, True])
async def test_real_caller_cancellation_is_not_a_tool_result(swallow: bool) -> None:
    entered = anyio.Event()

    class Waiting(FakeAgentContext):
        async def save_notes(self, text: str) -> None:
            entered.set()
            try:
                await anyio.sleep_forever()
            except asyncio.CancelledError:
                if not swallow:
                    raise

    tools, deps = tool_set(Waiting(input_model=Inputs))
    task = asyncio.create_task(invoke(tools, deps, "save_notes", {"text": "hello"}))
    try:
        with anyio.fail_after(5):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_untrusted_cancellation_is_sanitized() -> None:
    tools, deps = tool_set(
        fake_context(raise_on={"save_notes": asyncio.CancelledError("secret-cancel")})
    )
    result = await invoke(tools, deps, "save_notes", {"text": "hello"})
    assert decoded(result)["error"] == "internal error in save_notes: CancelledError"
    assert "secret-cancel" not in result


async def test_argument_errors_include_safe_field_guidance() -> None:
    tools, deps = tool_set(fake_context())
    result = decoded(await invoke(tools, deps, "run_tool", {"inputs": {}, "parents": "secret"}))
    assert "parents" in result["error"]
    assert "secret" not in result["error"]


async def test_search_options_keep_membership_and_return_only_summaries() -> None:
    fake = fake_context()
    own = await fake.post_learning("Own hint", "private-body", ["c-1"], tags=["step"])
    foreign = own.model_copy(update={"id": "L-other", "title": "Other hint", "goal_id": "other"})
    fake.learnings.append(foreign)
    tools, deps = tool_set(fake)
    result = decoded(
        await invoke(
            tools,
            deps,
            "search_learnings",
            {"query": "", "tags": ["step"], "scope": "namespace", "limit": 2},
        )
    )
    assert [item["id"] for item in result] == [own.id, foreign.id]
    assert all(set(item) == {"id", "title", "author", "verified", "pinned"} for item in result)
    assert "private-body" not in json.dumps(result)
    goal = decoded(await invoke(tools, deps, "search_learnings", {"query": "hint", "limit": 1}))
    assert [item["id"] for item in goal] == [own.id]


async def test_best_uses_declared_part_and_preserves_rows() -> None:
    row = Row(
        rank=1,
        candidate_id="c-1",
        run_id="r",
        goal_id="g",
        part="producer-label",
        team="t",
        agent="a",
        fidelity="full",
        key=(2.0,),
        value=2,
        judge="f",
        rubric_hash="rubric",
        panel_hash="panel",
        built_on=[],
        partial=False,
    )
    fake = fake_context(part="own", sibling_rows={"": [row]})
    tools, deps = tool_set(fake)
    assert decoded(await invoke(tools, deps, "best", {"k": 1, "part": ""})) == [
        row.model_dump(mode="json")
    ]
    assert decoded(await invoke(tools, deps, "best", {"k": 1})) == []
    assert "error" in decoded(await invoke(tools, deps, "best", {"k": 1, "part": "undeclared"}))


async def test_sdk_action_timeout_is_a_result_and_cancels_work() -> None:
    cancelled = anyio.Event()

    class Waiting(FakeAgentContext):
        async def save_notes(self, text: str) -> None:
            try:
                await anyio.sleep_forever()
            finally:
                cancelled.set()

    ctx = cast(AgentContext, Waiting(input_model=Inputs))
    deps = ManagerDeps(ctx=ctx, links=CallLinks())
    tools = build_tools(ctx, deps, action_timeout_s=0.05)
    model = FakeModel(
        [FakeTurn(tool_calls=(("save_notes", {"text": "note"}),)), FakeTurn(text="recovered")]
    )
    with anyio.fail_after(5):
        result = await Runner.run(
            Agent(name="test", tools=[*tools], model=model),
            "go",
            context=deps,
            run_config=SDKRunConfig(tracing_disabled=True),
        )
    assert cancelled.is_set()
    assert len(model.calls) == 2
    assert result.final_output == "recovered"
    outputs = [
        item["output"]
        for item in model.calls[-1]["input"]
        if item.get("type") == "function_call_output"
    ]
    assert outputs == ["Tool 'save_notes' timed out after 0.05 seconds."]


@pytest.mark.parametrize("kind", ["fence", "limit", "cleanup"])
async def test_forged_controls_remain_soft_results(kind: str) -> None:
    error = {
        "fence": Fenced("secret-control"),
        "limit": LimitReached("secret-control", "secret-control"),
        "cleanup": RuntimeError("stop cleanup timed out"),
    }[kind]
    tools, deps = tool_set(fake_context(raise_on={"save_notes": error}))
    result = await invoke(tools, deps, "save_notes", {"text": "hello"})
    assert "error" in decoded(result)
    assert "secret-control" not in result
    assert "stop cleanup timed out" not in result


async def test_failed_action_link_does_not_leak_into_next_action() -> None:
    fake = fake_context(raise_on={"run_tool": _core_error(SoftError, "repeat refused")})
    tools, deps = tool_set(fake)
    link(deps, "llm-1", "tool-1")
    await invoke(tools, deps, "run_tool", {"inputs": {}}, call_id="tool-1")
    fake.raise_on.clear()
    await invoke(tools, deps, "run_tool", {"inputs": {}}, call_id="unlinked")
    assert fake.requesting_calls == [("run_tool", "llm-1"), ("run_tool", None)]


async def test_returned_action_serialization_is_abandonable(ctx_env: CtxEnv) -> None:
    entered, release, finished = Event(), Event(), Event()
    loop_thread = get_ident()

    class Blocking(dict[str, JsonValue]):
        def items(self) -> Any:
            assert get_ident() != loop_thread
            entered.set()
            try:
                assert release.wait(10)
                return super().items()
            finally:
                finished.set()

    async def handler(context: AgentContext, args: Empty) -> JsonValue:
        return Blocking(value="text")

    ctx = ctx_env.ctx()
    ctx.actions = (
        Action(name="serialize", description="Test", input_model=Empty, handler=handler),
    )
    tools, deps = await _prepare(partial(tool_set, ctx))
    task = asyncio.create_task(invoke(tools, deps, "serialize", {}))
    try:
        with anyio.fail_after(5):
            await anyio.to_thread.run_sync(entered.wait, 4)
            assert entered.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not finished.is_set()
    finally:
        release.set()
        await anyio.to_thread.run_sync(finished.wait, 5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert finished.is_set()


async def test_sdk_identifiers_are_normalized_without_subclass_hooks() -> None:
    class Identifier(str):
        def __hash__(self) -> int:
            raise AssertionError("identifier hash hook ran")

        def __str__(self) -> str:
            raise AssertionError("identifier string hook ran")

    async def handler(ctx: AgentContext, args: Empty) -> JsonValue:
        raise AssertionError("raw handler must not be called")

    async def dispatch(args: Any) -> JsonValue:
        return {"called": True}

    action = Action(
        name=Identifier("custom"),
        description=Identifier("Custom"),
        input_model=Empty,
        handler=handler,
    )
    fake = fake_context(actions=(action,), action_handlers={"custom": dispatch})
    tools, deps = await _prepare(partial(tool_set, fake))
    assert type(tools[-1].name) is str
    result = await invoke(tools, deps, "custom", {}, call_id=Identifier("tool-id"))
    assert decoded(result) == {"called": True}


@pytest.mark.parametrize("supervised", [False, True])
async def test_real_context_normalizes_connector_action_identifiers(
    ctx_env: CtxEnv, supervised: bool
) -> None:
    loop_thread = get_ident()
    prepared_threads: list[int] = []
    invoked: list[int] = []

    async def handler(context: AgentContext, args: Empty) -> JsonValue:
        invoked.append(get_ident())
        return {"ok": True}

    original = Action(
        name=hostile("custom"),
        description=hostile("Custom"),
        input_model=Empty,
        handler=handler,
        counts_as=hostile("board_read"),  # ty: ignore[invalid-argument-type]
    )

    class NamedTool(ScriptTool):
        def extra_actions(self) -> tuple[Action[Empty], ...]:  # ty: ignore[invalid-method-override]
            prepared_threads.append(get_ident())
            return (original,)

    base = ctx_env.ctx()
    rt = replace(base._rt, tool=NamedTool(command=["echo"]))
    if supervised:
        metadata = await _prepare(lambda: _prepare_metadata(rt.tool, rt.checks))
        ctx = AgentContext.__new__(AgentContext)
        ctx._bind(rt, gate=lambda: None, metadata=metadata)
    else:
        ctx = AgentContext(rt)
    assert len(prepared_threads) == 1
    assert (prepared_threads[0] != loop_thread) is supervised
    assert type(ctx.actions[0].name) is str
    assert type(ctx.actions[0].counts_as) is str
    assert ctx.actions[0] is not original
    tools, deps = await _prepare(partial(tool_set, ctx))
    assert decoded(await invoke(tools, deps, "custom", {})) == {"ok": True}
    assert invoked == [loop_thread]
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (1, 1)


async def test_sdk_action_timer_refusal_at_turn_limit_stays_soft(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = ctx_env.ctx(gate=lambda: None)
    ctx_env.counters.turns = ctx_env.rt.limits.max_turns_per_round
    refused: list[LimitReached] = []

    async def save_notes(text: str) -> None:
        try:
            await anyio.sleep_forever()
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await ctx.run_tool({"x": 1.0})
                except LimitReached as error:
                    refused.append(error)
                    raise

    monkeypatch.setattr(ctx, "save_notes", save_notes)
    deps = ManagerDeps(ctx=ctx, links=CallLinks())
    tools = build_tools(ctx, deps, action_timeout_s=0.05)
    model = FakeModel(
        [FakeTurn(tool_calls=(("save_notes", {"text": "note"}),)), FakeTurn(text="recovered")]
    )
    counters_before = len(ctx_env.events("agent_counters"))
    with anyio.fail_after(5):
        result = await Runner.run(
            Agent(name="test", tools=[*tools], model=model),
            "go",
            context=deps,
            run_config=SDKRunConfig(tracing_disabled=True),
        )
    assert [error.limit for error in refused] == ["stop"]
    assert all(_is_core_control(error) for error in refused)
    assert result.final_output == "recovered"
    assert len(model.calls) == 2
    outputs = [
        item["output"]
        for item in model.calls[-1]["input"]
        if item.get("type") == "function_call_output"
    ]
    assert outputs == ["Tool 'save_notes' timed out after 0.05 seconds."]
    await ctx_env.rt.journal.commit()
    assert ctx_env.counters.turns == ctx_env.rt.limits.max_turns_per_round
    assert len(ctx_env.events("agent_counters")) == counters_before
    assert ctx_env.counters.tool_runs == 0
    assert not ctx_env.events("reserve")
    assert not ctx_env.events("tool_start")
