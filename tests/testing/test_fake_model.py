"""Offline SDK responses, deterministic routing and real recorder accounting."""

import inspect
import json
import traceback
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, NoReturn

import anyio
import pytest

pytest.importorskip("agents")

from _fakes.fake_models import registered_model
from _fakes.llm_run import HeldOperation, LLMRun, open_llm_run
from agents import Agent, Model, ModelResponse, ModelSettings, Runner
from agents import RunConfig as SDKRunConfig
from agents.models.interface import ModelTracing
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage
from pydantic import ValidationError

from swarm_arena import RunConfig
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import LLMScope, llm_scope
from swarm_arena._errors import _is_core_signal
from swarm_arena._journal.llm_record import drain, open_call
from swarm_arena.config import ModelCatalog, ModelSpec
from swarm_arena.llm._catalog import build_model
from swarm_arena.testing import FakeModel, FakeTurn
from swarm_arena.testing import _fake_model as fake_model_module
from swarm_arena.tool import Usage

pytestmark = pytest.mark.anyio
REVIEW_MARKER = "REVIEW_SECRET"


def assert_value_free_field_error(error: ValueError, field: str) -> None:
    rendered = [
        str(error),
        repr(error),
        repr(error.args),
        "".join(traceback.format_exception(error)),
    ]
    if isinstance(error, ValidationError):
        rendered.extend([str(error.errors()), error.json()])
    assert all(REVIEW_MARKER not in value for value in rendered)
    assert error.args == (f"invalid {field}",)
    assert error.__cause__ is None
    assert not getattr(error, "__notes__", [])
    assert _is_core_signal(error)


@pytest.fixture(autouse=True)
def isolated_registrations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the catalog after each test, without sharing consumed models."""
    monkeypatch.setattr("swarm_arena.llm._catalog._FAKE_MODELS", {})


def make_model(turns: Sequence[FakeTurn] | None = None, *, provider: str = "fake") -> FakeModel:
    return FakeModel([FakeTurn(text="ok")] if turns is None else turns, provider=provider)


async def respond(model: FakeModel, **overrides: Any) -> ModelResponse:
    kwargs: dict[str, Any] = {
        "system_instructions": "system",
        "input": "hello",
        "model_settings": ModelSettings(),
        "tools": [],
        "output_schema": None,
        "handoffs": [],
        "tracing": ModelTracing.DISABLED,
        "previous_response_id": None,
        "conversation_id": None,
        "prompt": None,
    }
    return await model.get_response(**(kwargs | overrides))


def texts(response: ModelResponse) -> list[str]:
    return [
        content.text
        for item in response.output
        if isinstance(item, ResponseOutputMessage)
        for content in item.content
        if content.type == "output_text"
    ]


async def tracked_settings(run: LLMRun, scope: LLMScope) -> ModelSettings:
    estimate = Amount(tokens=1_000, usd=0.01)
    reservation = await scope.tranche.reserve_call(estimate)
    call_id = open_call(
        scope,
        reservation_id=reservation,
        estimate=estimate,
        model_alias="m",
        model="fake/m",
        provider="fake",
    )
    metadata = run.metadata(scope, reservation_id=reservation, call_id=call_id)
    metadata["model_alias"] = "m"
    return ModelSettings(metadata=metadata)


def config() -> RunConfig:
    return RunConfig(
        models=ModelCatalog(
            {
                "m": ModelSpec(
                    model="fake/m", roles=["manager"], prices=ModelSpec.Prices(input=1, output=2)
                )
            }
        ),
    )


async def test_turns_replayed_in_order() -> None:
    model = make_model([FakeTurn(text=text) for text in ("a", "b", "c")])
    assert [texts(await respond(model))[0] for _ in range(3)] == ["a", "b", "c"]


async def test_tool_calls_become_function_calls_with_unique_ids() -> None:
    model = make_model([FakeTurn(text="working", tool_calls=(("run_tool", {"x": 1}),))] * 2)
    responses = [await respond(model) for _ in range(2)]
    function_calls = [
        item
        for response in responses
        for item in response.output
        if isinstance(item, ResponseFunctionToolCall)
    ]
    assert [i.call_id for i in function_calls] == ["fake-run_tool-1", "fake-run_tool-2"]
    assert all(json.loads(item.arguments) == {"x": 1} for item in function_calls)
    assert all(texts(response) == ["working"] for response in responses)


async def test_records_through_record_call_with_catalog_price(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        settings = await tracked_settings(run, scope)
        model = make_model()
        model.register("m")
        await respond(model, model_settings=settings)
        await drain(scope, wait=False)
        await scope.tranche.close()
        await run.journal.commit()
        assert run.calls()[0].cost_usd == pytest.approx(0.00014)
        assert run.calls()[0].provider == "fake"
        assert run.calls()[0].model == "fake/m"
        assert run.calls()[0].model_alias == "m"
        assert run.charged(scope)._micro_usd == 140
        assert Amount(usd=run.calls()[0].cost_usd)._micro_usd == 140
        assert scope.call_ids == [run.calls()[0].id]


async def test_probe_prices_without_role_check_and_records_tool_ids(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("tool")
        settings = await tracked_settings(run, scope)
        usage = Usage(input_tokens=3, output_tokens=1, cache_read_tokens=2, cache_write_tokens=1)
        model = make_model(
            [FakeTurn(tool_calls=(("run_tool", {}),), usage=usage)], provider="offline"
        )
        model.register("m")
        await respond(model, model_settings=settings)
        await drain(scope, wait=False)
        await scope.tranche.close()
        call = run.calls()[0]
        assert call.provider == "offline"
        assert call.tool_calls == [{"id": "fake-run_tool-1", "name": "run_tool"}]
        assert call.usage == usage
        assert run.charged(scope)._micro_usd == Amount(usd=call.cost_usd)._micro_usd == 9


async def test_no_turns_left_raises() -> None:
    with pytest.raises(RuntimeError, match="no turns left"):
        await respond(make_model([]), input="sentinel-secret")


def test_register_makes_fake_spec_resolve() -> None:
    m = FakeModel([FakeTurn(text="ok")])
    m.register("m")
    assert build_model(ModelSpec(model="fake/m", roles=["manager"])) is m


async def test_calls_records_inputs() -> None:
    model = make_model()
    input = [{"role": "user", "content": "hello"}]
    settings = ModelSettings(temperature=0)
    await respond(model, input=input, model_settings=settings)
    assert model.calls == [
        {
            "system": "system",
            "input": input,
            "model_settings": settings,
            "tools": [],
            "output_schema": None,
            "tracing": ModelTracing.DISABLED,
        }
    ]


def test_public_module_path() -> None:
    assert (FakeModel.__module__, FakeTurn.__module__) == (
        "swarm_arena.testing",
        "swarm_arena.testing",
    )
    assert FakeModel.register.__module__ == "swarm_arena.testing"


def test_sdk_signatures_match() -> None:
    for name in ("get_response", "stream_response"):
        assert (
            inspect.signature(getattr(FakeModel, name)).parameters.keys()
            == inspect.signature(getattr(Model, name)).parameters.keys()
        )


def test_stream_response_raises() -> None:
    with pytest.raises(NotImplementedError, match="FakeModel does not stream"):
        make_model().stream_response(
            None,
            "hello",
            ModelSettings(),
            [],
            None,
            [],
            ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )


def test_turn_is_frozen_and_keyword_only() -> None:
    turn = FakeTurn(text="ok")
    with pytest.raises(FrozenInstanceError):
        turn.text = "changed"
    assert inspect.signature(FakeTurn).parameters["text"].kind == inspect.Parameter.KEYWORD_ONLY
    assert turn.usage == Usage(input_tokens=100, output_tokens=20)


async def test_sdk_usage_includes_cached_input() -> None:
    response = await respond(
        make_model(
            [
                FakeTurn(
                    usage=Usage(
                        input_tokens=10,
                        output_tokens=2,
                        cache_read_tokens=3,
                        cache_write_tokens=4,
                    )
                )
            ]
        )
    )
    assert response.output == []
    assert response.usage.requests == 1
    assert response.usage.input_tokens == 17
    assert response.usage.output_tokens == 2
    assert response.usage.total_tokens == 19
    assert response.usage.input_tokens_details.cached_tokens == 3
    assert response.usage.input_tokens_details.cache_write_tokens == 4


async def test_real_sdk_runner_uses_the_fake() -> None:
    result = await Runner.run(
        Agent(name="offline", model=make_model()),
        "hello",
        run_config=SDKRunConfig(tracing_disabled=True),
    )
    assert result.final_output == "ok"


async def test_conversations_replay_independently_under_interleaving() -> None:
    model = make_model([FakeTurn(text="a"), FakeTurn(text="b")])
    first = anyio.Event()
    second = anyio.Event()
    outputs: dict[str, list[str]] = {"one": [], "two": []}

    async def one() -> None:
        outputs["one"] += texts(await respond(model, conversation_id="one"))
        first.set()
        await second.wait()
        outputs["one"] += texts(await respond(model, conversation_id="one"))

    async def two() -> None:
        await first.wait()
        outputs["two"] += texts(await respond(model, conversation_id="two"))
        outputs["two"] += texts(await respond(model, conversation_id="two"))
        second.set()

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(one)
            tasks.start_soon(two)
    assert outputs == {"one": ["a", "b"], "two": ["a", "b"]}


async def test_scope_routes_agents_and_resets_between_runs(tmp_path: Path) -> None:
    model = make_model([FakeTurn(text="a"), FakeTurn(text="b")])
    for _ in range(2):
        async with open_llm_run(tmp_path, config=config()) as run:
            scopes = [run.scope("manager", agent=name) for name in ("one", "two")]
            for scope in (scopes[1], scopes[0], scopes[0], scopes[1]):
                with llm_scope(scope):
                    response = await respond(
                        model, model_settings=await tracked_settings(run, scope)
                    )
                assert texts(response) == (["a"] if len(scope.call_ids) == 1 else ["b"])
            for scope in scopes:
                await drain(scope, wait=False)
                await scope.tranche.close()
            assert len(run.calls()) == 4


async def test_new_factory_replaces_consumed_registration() -> None:
    first = make_model()
    first.register("m")
    await respond(first)
    second = make_model()
    second.register("m")
    assert build_model(ModelSpec(model="fake/m", roles=["manager"])) is second
    assert second.calls == []
    assert texts(await respond(second)) == ["ok"]


async def test_registered_factory_restores_registration_and_queue() -> None:
    original = make_model()
    original.register("m")
    spec = ModelSpec(model="fake/m", roles=["manager"])
    for _ in range(2):
        with registered_model([FakeTurn(text="fresh")]) as model:
            assert build_model(spec) is model
            assert model.calls == []
            assert texts(await respond(model)) == ["fresh"]
        assert build_model(spec) is original
    with (
        pytest.raises(LookupError, match="fixture failure"),
        registered_model([], name="temporary"),
    ):
        raise LookupError("fixture failure")
    with pytest.raises(ValueError, match="fake model is not registered"):
        build_model(ModelSpec(model="fake/temporary", roles=["manager"]))


async def test_script_arguments_are_detached_between_models_and_conversations() -> None:
    args = {"nested": {"value": 1}}
    turns = [FakeTurn(tool_calls=(("run_tool", args),))]
    model = make_model(turns)
    args["nested"]["value"] = 99
    response = await respond(model, conversation_id="one")
    call = response.output[0]
    assert isinstance(call, ResponseFunctionToolCall)
    assert json.loads(call.arguments) == {"nested": {"value": 1}}
    assert texts(await respond(make_model())) == ["ok"]


async def test_nonfinite_tool_arguments_have_value_free_errors() -> None:
    model = make_model([FakeTurn(tool_calls=(("run_tool", {"sentinel-secret": float("nan")}),))])
    with pytest.raises(ValueError, match="invalid arguments") as caught:
        await respond(model)
    assert "sentinel-secret" not in "".join(traceback.format_exception(caught.value))
    assert "sentinel-secret" not in repr(caught.value.args)
    assert "sentinel-secret" not in str(caught.value)
    assert "sentinel-secret" not in repr(caught.value)
    assert not getattr(caught.value, "__notes__", [])
    assert caught.value.__cause__ is None
    assert_value_free_field_error(caught.value, "arguments")


@pytest.mark.parametrize("text", [{"payload": REVIEW_MARKER}, [REVIEW_MARKER], 42, True])
async def test_malformed_text_is_rejected_without_exposing_values(text: Any) -> None:
    model = make_model([FakeTurn(text=text)])
    with pytest.raises(ValueError, match="text") as caught:
        await respond(model)
    assert_value_free_field_error(caught.value, "text")


@pytest.mark.parametrize("arguments", [[REVIEW_MARKER], None, 42])
async def test_nonobject_tool_arguments_are_rejected_without_exposing_values(
    arguments: Any,
) -> None:
    model = make_model([FakeTurn(text="valid", tool_calls=(("run_tool", arguments),))])
    with pytest.raises(ValueError, match="arguments") as caught:
        await respond(model)
    assert_value_free_field_error(caught.value, "arguments")


async def test_arguments_are_validated_before_any_sdk_model_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sdk_model(**kwargs: Any) -> NoReturn:
        raise AssertionError("SDK model constructed before argument validation")

    for name in (
        "ResponseOutputMessage",
        "ResponseOutputText",
        "ResponseFunctionToolCall",
        "SDKUsage",
        "InputTokensDetails",
        "ModelResponse",
    ):
        monkeypatch.setattr(fake_model_module, name, sdk_model)
    model = make_model(
        [
            FakeTurn(
                text="valid",
                tool_calls=(("first", {}), ("second", [REVIEW_MARKER])),
            )
        ]
    )
    with pytest.raises(ValueError, match="arguments") as caught:
        await respond(model)
    assert_value_free_field_error(caught.value, "arguments")


async def test_abandoned_fake_callback_starts_no_second_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        settings = await tracked_settings(run, scope)
        assert settings.metadata is not None
        model = make_model()
        held = HeldOperation()
        uploads: list[bytes] = []
        original = run.artifacts.put_bytes
        completed = anyio.Event()

        def upload(data: bytes) -> str:
            uploads.append(data)
            held.wait()
            return original(data)

        async def callback() -> None:
            await respond(model, model_settings=settings)
            completed.set()

        monkeypatch.setattr(run.artifacts, "put_bytes", upload)
        try:
            with anyio.fail_after(5):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(callback)
                    await held.ready()
                    assert await drain(scope, wait=False) == [settings.metadata["call_id"]]
                    await completed.wait()
            assert len(run.calls()) == 1
            assert run.calls()[0].status == "error"
            assert run.calls()[0].request_ref is None
            assert run.calls()[0].response_ref is None
            assert run.charged(scope)._micro_usd == 10_000
            await scope.tranche.close()
        finally:
            await held.finish()
        assert len(uploads) == 1


async def test_authoritative_usage_overflow_reaches_the_recorder(tmp_path: Path) -> None:
    from swarm_arena._budget.amount import USD_LIMIT
    from swarm_arena._errors import _core_category, _SignalCategory
    from swarm_arena.manager import LimitReached

    usage = Usage(input_tokens=2 * 10**15)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        settings = await tracked_settings(run, scope)
        with pytest.raises(LimitReached) as caught:
            await respond(
                make_model([FakeTurn(text="overflow", usage=usage)]), model_settings=settings
            )
        assert _core_category(caught.value) is _SignalCategory.TERMINAL
        await drain(scope, wait=False)
        await scope.tranche.close()
        await run.journal.commit()
        [call] = run.calls()
        assert call.cost_usd == USD_LIMIT
        assert call.usage == usage
        assert call.status == "error"
        assert scope.call_ids == [call.id]
        assert run.charged(scope).usd == USD_LIMIT
        assert run.ledger.run_exhausted()
        with pytest.raises(LimitReached) as refused:
            await run.scope("manager", agent="later").tranche.reserve_call(Amount(usd=0.001))
        assert _core_category(refused.value) is _SignalCategory.TERMINAL
