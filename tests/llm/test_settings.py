"""Per-call settings and authoritative adapter merges."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anyio
import httpx2 as httpx
import litellm
import pytest
from _fakes.llm_run import open_llm_run
from agents import ModelSettings, ModelTracing
from agents.extensions.models.litellm_model import LitellmModel
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from swarm_arena._budget.llm_scope import LLMScope
from swarm_arena._model.base import LLMRole
from swarm_arena.config import CachePolicy, ModelSpec
from swarm_arena.llm._cache import bypass_response_cache, cache_switches, response_cache
from swarm_arena.llm._settings import injection_points, merge_model_settings, model_settings
from tests.llm.test_upstream_facts import THINKING_PARAMETER_CASES, _CaptureLogger, _invoke_adapter

pytestmark = pytest.mark.filterwarnings(
    "ignore:Items? .*TypedDict class .*ReadOnly.*:UserWarning:pydantic._internal._generate_schema"
)


OPUS = ModelSpec(model="anthropic/claude-opus-5-5", roles=["manager"], max_tokens=8000)


def settings(scope: LLMScope, **overrides: Any) -> ModelSettings:
    return model_settings(
        OPUS,
        alias="opus",
        scope=scope,
        reservation_id="r-1",
        call_id="llm-1",
        namespace="physika",
        **{"read": False, "write": False, **overrides},
    )


async def test_metadata_carries_tracking_ids(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        s = settings(run.scope("manager"))
        assert s.metadata == {
            "run_id": run.run_id,
            "goal_id": "g",
            "team": "t",
            "agent": "a",
            "round": "1",
            "role": "manager",
            "reservation_id": "r-1",
            "call_id": "llm-1",
            "model_alias": "opus",
        }
        assert all(type(v) is str for v in s.metadata.values())
        assert s.include_usage is True
        assert s.max_tokens == 8000


async def test_max_tokens_override_capped_by_catalog(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("manager")
        assert settings(scope, max_tokens=100).max_tokens == 100
        assert settings(scope, max_tokens=10_000).max_tokens == 8000


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-opus-5-5", "bedrock/anthropic.claude-opus-5-5", "vertex_ai/claude-opus-5-5"],
)
def test_claude_injection_points_default_ttl(model: str) -> None:
    assert injection_points(ModelSpec(model=model, roles=["manager"])) == [
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1},
    ]


def test_claude_injection_points_1h_ttl() -> None:
    pts = injection_points(
        OPUS.model_copy(update={"prompt_cache": ModelSpec.PromptCache(ttl="1h")})
    )
    assert pts is not None
    assert all(p["control"] == {"type": "ephemeral", "ttl": "1h"} for p in pts)


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-6", "gemini/gemini-3-flash", "deepseek/deepseek-chat", "bedrock/meta.llama4-70b"],
)
def test_no_injection_points_for_other_models(model: str) -> None:
    assert injection_points(ModelSpec(model=model, roles=["manager"])) is None


def test_prompt_cache_disabled_means_no_points() -> None:
    assert (
        injection_points(
            OPUS.model_copy(update={"prompt_cache": ModelSpec.PromptCache(enabled=False)})
        )
        is None
    )


async def test_cache_dict_from_switches(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("manager")
        assert (settings(scope, read=True, write=True).extra_args or {})["cache"] == {
            "use-cache": True,
            "no-cache": False,
            "no-store": False,
            "namespace": "physika",
        }
        assert (settings(scope, write=True).extra_args or {})["cache"]["no-cache"] is True
        assert (settings(scope, read=True).extra_args or {})["cache"]["no-store"] is True
        assert "cache" not in (settings(scope).extra_args or {})


async def test_temperature_comes_with_drop_params(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("manager")
        assert (settings(scope, temperature=0).extra_args or {})["drop_params"] is True
        assert "drop_params" not in (settings(scope, temperature=None).extra_args or {})


async def test_merge_protects_tracking_and_cache_controls(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("manager")
        caller = ModelSettings(
            metadata={"reservation_id": "wrong", "tag": "kept"},
            extra_args={
                "metadata": {"call_id": "wrong", "model_alias": "wrong"},
                "litellm_metadata": {"reservation_id": "wrong"},
                "cache": {"namespace": "wrong"},
                "caching": True,
                "preset_cache_key": "wrong",
                "cache_key": "wrong",
                "cache_control_injection_points": [],
                "drop_params": False,
                "seed": 17,
            },
        )
        per_call = settings(scope, read=False, write=True, temperature=0)
        merged = merge_model_settings(caller, per_call)
        assert merged.metadata == {**(per_call.metadata or {}), "tag": "kept"}
        extra = merged.extra_args or {}
        assert extra["cache"] == (per_call.extra_args or {})["cache"]
        assert extra["seed"] == 17
        assert extra["drop_params"] is True
        assert extra["cache_control_injection_points"] == injection_points(OPUS)
        assert not (
            {"preset_cache_key", "cache_key", "metadata", "litellm_metadata", "caching"}
            & extra.keys()
        )
        disabled = merge_model_settings(caller, settings(scope))
        assert "cache" not in (disabled.extra_args or {})
        assert caller.extra_args is not None
        assert caller.extra_args["caching"] is True


@pytest.mark.parametrize(
    "caller",
    [
        ModelSettings(reasoning=Reasoning(effort="high")),
        ModelSettings(extra_args={"thinking": {"type": "enabled", "budget_tokens": 1024}}),
        ModelSettings(extra_args={"reasoning_effort": "high"}),
        ModelSettings(extra_body={"reasoning_effort": "high"}),
        ModelSettings(extra_body={"thinking": {"type": "adaptive"}}),
        ModelSettings(extra_args={"extra_body": {"thinking": {"type": "enabled"}}}),
        ModelSettings(extra_args={"stream": True}),
    ],
)
async def test_merge_detects_streamed_and_thinking_channels(
    tmp_path: Path, caller: ModelSettings
) -> None:
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(caller, settings(run.scope("manager"), read=True, write=True))
        assert "cache" not in (merged.extra_args or {})


@pytest.mark.parametrize("role", ["judge", "manager"])
async def test_probe_bypass_is_complete_and_restored(tmp_path: Path, role: LLMRole) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope(role, force_no_cache_read=role == "judge")
        policy = CachePolicy(
            backend="disk", roles={role: CachePolicy.RoleSwitch(read=True, write=True)}
        )
        assert cache_switches(
            policy, role, force_no_read=False, streamed=False, thinking=False
        ) == (True, True)
        assert (settings(scope, read=True, write=True).extra_args or {})["cache"]["no-cache"] is (
            role == "judge"
        )
        with (  # noqa: PT012 - verify restoration on failure
            pytest.raises(RuntimeError, match="probe"),
            bypass_response_cache(),
        ):
            assert "cache" not in (settings(scope, read=True, write=True).extra_args or {})
            assert cache_switches(
                policy, role, force_no_read=False, streamed=False, thinking=False
            ) == (False, False)
            raise RuntimeError("probe")
        assert "cache" in (settings(scope, read=True, write=True).extra_args or {})
        per_call = settings(scope, read=True, write=True)
        with bypass_response_cache():
            assert "cache" not in (merge_model_settings(ModelSettings(), per_call).extra_args or {})
        for flags in ({"streamed": True}, {"thinking": True}):
            assert "cache" not in (
                merge_model_settings(ModelSettings(), per_call, **flags).extra_args or {}
            )


async def test_final_settings_follow_real_adapter_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return litellm.ModelResponse()

    monkeypatch.setattr(litellm, "acompletion", completion)
    async with open_llm_run(tmp_path) as run:
        per_call = settings(run.scope("manager"), read=False, write=True, temperature=0)
        caller = ModelSettings(extra_args={"metadata": {"call_id": "wrong"}, "caching": True})
        await _invoke_adapter(merge_model_settings(caller, per_call))
        assert captured["metadata"] == per_call.metadata
        assert captured["cache"] == (per_call.extra_args or {})["cache"]
        assert captured["cache_control_injection_points"] == injection_points(OPUS)
        assert captured["drop_params"] is True
        assert captured["temperature"] == 0
        assert "caching" not in captured


async def test_merged_caller_temperature_also_drops_unsupported_params(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(ModelSettings(temperature=0), settings(run.scope("manager")))
        assert merged.temperature == 0
        assert (merged.extra_args or {})["drop_params"] is True


@pytest.fixture
async def isolated_settings_upstream(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    for name in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "input_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "_async_input_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "cache", None)
    try:
        yield
    finally:
        with anyio.fail_after(60):
            await GLOBAL_LOGGING_WORKER.flush()
            await GLOBAL_LOGGING_WORKER.stop()
            await close_litellm_async_clients()


async def _invoke_model(model: str, merged: ModelSettings) -> Any:
    return await LitellmModel(model).get_response(
        system_instructions=None,
        input="hi",
        model_settings=merged,
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    )


@pytest.mark.parametrize("parameter", THINKING_PARAMETER_CASES)
@pytest.mark.parametrize("channel", ["extra_args", "extra_body", "nested_body"])
async def test_all_pinned_thinking_controls_disable_cache(
    tmp_path: Path, parameter: str, channel: str
) -> None:
    _, request, _, _ = THINKING_PARAMETER_CASES[parameter]
    if channel == "extra_args":
        caller = ModelSettings(extra_args=request)
    elif channel == "extra_body":
        caller = ModelSettings(extra_body=request)
    else:
        caller = ModelSettings(extra_args={"extra_body": request})
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(caller, settings(run.scope("manager"), read=True, write=True))
        assert "cache" not in (merged.extra_args or {})


@pytest.mark.parametrize(
    ("provider_model", "native"),
    [
        ("gemini/gemini-2.5-flash", {"thinkingConfig": {"thinkingBudget": 1024}}),
        ("ollama_chat/qwen3", {"think": True}),
        ("openai/gpt-4o-mini", {"unlisted_native_mode": True}),
    ],
)
@pytest.mark.usefixtures("isolated_settings_upstream")
async def test_native_controls_bypass_response_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider_model: str, native: dict[str, Any]
) -> None:
    pytest.importorskip("diskcache")
    pytest.importorskip("fastapi")
    cache = response_cache(CachePolicy(backend="disk", disk_dir=str(tmp_path / "cache")))
    logger = _CaptureLogger()
    monkeypatch.setattr(litellm, "cache", cache)
    monkeypatch.setattr(litellm, "callbacks", [logger])
    backend = cast(Any, cache.cache)
    read, write = backend.async_get_cache, backend.async_set_cache
    accesses = {"read": 0, "write": 0}
    written = anyio.Event()

    async def capture_read(key: str, **kwargs: Any) -> Any:
        accesses["read"] += 1
        return await read(key, **kwargs)

    async def capture_write(key: str, value: Any, **kwargs: Any) -> None:
        accesses["write"] += 1
        await write(key, value, **kwargs)
        written.set()

    monkeypatch.setattr(backend, "async_get_cache", capture_read)
    monkeypatch.setattr(backend, "async_set_cache", capture_write)
    try:
        async with open_llm_run(tmp_path / "run") as run:
            spec = ModelSpec(model=provider_model, roles=["manager"], max_tokens=8000)
            ours = model_settings(
                spec,
                alias="gemini",
                scope=run.scope("manager"),
                reservation_id="r-1",
                call_id="llm-1",
                namespace="A",
                read=True,
                write=True,
            )
            control = merge_model_settings(
                ModelSettings(extra_args={"mock_response": "answer", "seed": 17}), ours
            )
            await _invoke_model(spec.model, control)
            await logger.wait()
            with anyio.fail_after(60):
                await written.wait()
            assert accesses["read"] > 0
            assert accesses["write"] == 1
            accesses.update(read=0, write=0)
            merged = merge_model_settings(
                ModelSettings(extra_args={**native, "mock_response": "answer"}),
                ours,
            )
            for key, value in native.items():
                assert (merged.extra_args or {})[key] == value
            for _ in range(2):
                await _invoke_model(spec.model, merged)
                await logger.wait()
                assert not logger.success[-1][0]["cache_hit"]
                assert accesses == {"read": 0, "write": 0}
    finally:
        with anyio.fail_after(60):
            await GLOBAL_LOGGING_WORKER.flush()
        backend.disk_cache.close()


OUTPUT_LIMIT_ALIASES = (
    "max_tokens",
    "max_completion_tokens",
    "maxCompletionTokens",
    "max_response_output_tokens",
    "max_output_tokens",
    "maxOutputTokens",
    "maxTokens",
    "maxTokenCount",
    "max_tokens_to_sample",
    "max_new_tokens",
    "max_gen_len",
    "num_predict",
    "max_length",
)


@pytest.mark.parametrize("alias", OUTPUT_LIMIT_ALIASES)
@pytest.mark.parametrize("channel", ["extra_args", "extra_body", "nested_body"])
@pytest.mark.usefixtures("isolated_settings_upstream")
async def test_output_alias_cannot_exceed_catalog_cap(
    tmp_path: Path, alias: str, channel: str
) -> None:
    captured: list[dict[str, Any]] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http_client,
        AsyncOpenAI(api_key="offline-test", http_client=http_client) as client,
    ):
        if channel == "extra_args":
            caller = ModelSettings(extra_args={alias: 12000, "client": client})
        elif channel == "extra_body":
            caller = ModelSettings(extra_args={"client": client}, extra_body={alias: 12000})
        else:
            caller = ModelSettings(extra_args={"client": client, "extra_body": {alias: 12000}})
        async with open_llm_run(tmp_path) as run:
            ours = settings(run.scope("manager"))
            merged = merge_model_settings(caller, ours)
            await _invoke_model("openai/gpt-5", merged)
    assert merged.max_tokens == 8000
    assert captured[0]["max_completion_tokens"] <= 8000
    assert all(captured[0].get(key, 0) <= 8000 for key in OUTPUT_LIMIT_ALIASES)


@pytest.mark.parametrize("envelope", ["generationConfig", "generation_config"])
@pytest.mark.parametrize("with_alias", [True, False])
async def test_native_generation_config_cannot_override_catalog_cap(
    tmp_path: Path, envelope: str, with_alias: bool
) -> None:
    from litellm.llms.vertex_ai.gemini.transformation import transform_request_body
    from litellm.utils import get_optional_params

    payload = {envelope: {"temperature": 0.5, **({"maxOutputTokens": 12000} if with_alias else {})}}
    caller = ModelSettings(extra_body=payload)
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(caller, settings(run.scope("manager")))
    optional = get_optional_params(
        model="gemini-2.5-flash",
        custom_llm_provider="gemini",
        max_tokens=merged.max_tokens,
        extra_body=merged.extra_body,
    )
    body = transform_request_body(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-2.5-flash",
        optional_params=optional,
        custom_llm_provider="gemini",
        litellm_params={},
        cached_content=None,
    )
    serialized = json.loads(json.dumps(body))
    for key in ("generationConfig", "generation_config"):
        if key in serialized:
            config = serialized[key]
            assert all(config.get(alias, 0) <= 8000 for alias in OUTPUT_LIMIT_ALIASES)
            assert any(config.get(key) == 8000 for key in OUTPUT_LIMIT_ALIASES)
    if with_alias:
        assert payload[envelope]["maxOutputTokens"] == 12000


def test_output_alias_inventory_and_input_schema_are_preserved() -> None:
    from swarm_arena.llm._settings import _OUTPUT_LIMIT_ALIASES

    assert set(OUTPUT_LIMIT_ALIASES) == _OUTPUT_LIMIT_ALIASES
    schema = {"type": "object", "properties": {"max_tokens": {"type": "integer"}}}
    caller = ModelSettings(extra_body={"tools": [{"parameters": schema}]})
    assert (
        merge_model_settings(caller, ModelSettings(max_tokens=100)).extra_body == caller.extra_body
    )


@dataclass
class _FutureModelSettings(ModelSettings):
    future_control: bool = True


@pytest.mark.parametrize("key", ["unlisted_native_mode", "think"])
@pytest.mark.parametrize("channel", ["extra_args", "extra_body", "extra_query", "extra_headers"])
@pytest.mark.parametrize("value", [True, False, None])
async def test_unknown_parameter_disables_both_cache_directions(
    tmp_path: Path, key: str, channel: str, value: Any
) -> None:
    payload = {key: value}
    if channel == "extra_args":
        caller = ModelSettings(extra_args=payload)
    elif channel == "extra_body":
        caller = ModelSettings(extra_body=payload)
    elif channel == "extra_query":
        caller = ModelSettings(extra_query=payload)
    else:
        caller = ModelSettings(extra_headers={key: str(value)})
    original = dict(getattr(caller, channel))
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(caller, settings(run.scope("manager"), read=True, write=True))
    assert "cache" not in (merged.extra_args or {})
    assert getattr(caller, channel) == original


@pytest.mark.parametrize(
    "envelope",
    [
        "extra_body",
        "generationConfig",
        "generation_config",
        "inferenceConfig",
        "inference_config",
        "textGenerationConfig",
        "config",
        "parameters",
        "options",
        "additionalModelRequestFields",
        "additional_model_request_fields",
        "chat_template_kwargs",
    ],
)
async def test_unknown_nested_provider_key_disables_cache(tmp_path: Path, envelope: str) -> None:
    caller = ModelSettings(extra_args={"extra_body": {envelope: {"unlisted_native_mode": True}}})
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(caller, settings(run.scope("manager"), read=True, write=True))
    assert "cache" not in (merged.extra_args or {})


async def test_unlisted_sdk_field_disables_cache(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(
            _FutureModelSettings(), settings(run.scope("manager"), read=True, write=True)
        )
    assert "cache" not in (merged.extra_args or {})


async def test_schema_payload_is_not_a_provider_control_envelope(tmp_path: Path) -> None:
    caller = ModelSettings(
        extra_args={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {"type": "object", "properties": {"think": {"type": "boolean"}}},
                },
            }
        }
    )
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(caller, settings(run.scope("manager"), read=True, write=True))
    assert (merged.extra_args or {})["cache"] == {
        "use-cache": True,
        "no-cache": False,
        "no-store": False,
        "namespace": "physika",
    }


async def test_unknown_response_format_envelope_key_disables_cache(tmp_path: Path) -> None:
    caller = ModelSettings(
        extra_args={
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {}, "unlisted_native_mode": True},
            }
        }
    )
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(caller, settings(run.scope("manager"), read=True, write=True))
    assert "cache" not in (merged.extra_args or {})


async def test_pinned_retry_controls_remain_cacheable(tmp_path: Path) -> None:
    caller = ModelSettings(extra_args={"num_retries": 0, "max_retries": 0})
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(caller, settings(run.scope("manager"), read=True, write=True))
    assert (merged.extra_args or {})["cache"] == {
        "use-cache": True,
        "no-cache": False,
        "no-store": False,
        "namespace": "physika",
    }


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("stop", "END"),
        ("stop", ["END", "DONE"]),
        ("stop", []),
        ("logit_bias", {"42": -1, 43: 0.5}),
        ("logit_bias", {}),
    ],
)
@pytest.mark.parametrize("channel", ["extra_args", "extra_body", "nested"])
def test_valid_stop_and_logit_bias_preserve_cache(parameter: str, value: Any, channel: str) -> None:
    payload = {parameter: value}
    caller = (
        ModelSettings(extra_body=payload)
        if channel == "extra_body"
        else ModelSettings(
            extra_args={"extra_body": {"generationConfig": payload}}
            if channel == "nested"
            else payload
        )
    )
    cache = {"use-cache": True, "no-cache": False, "no-store": False}
    merged = merge_model_settings(caller, ModelSettings(extra_args={"cache": cache}))
    assert (merged.extra_args or {})["cache"] == cache


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("stop", None),
        ("stop", 1),
        ("stop", True),
        ("stop", ("END",)),
        ("stop", ["PRIVATE_STOP", None]),
        ("stop", {"think": "PRIVATE_STOP"}),
        ("logit_bias", None),
        ("logit_bias", "PRIVATE_BIAS"),
        ("logit_bias", [("42", -1)]),
        ("logit_bias", {"42": "PRIVATE_BIAS"}),
        ("logit_bias", {"42": None}),
        ("logit_bias", {"42": True}),
        ("logit_bias", {"42": {"think": True}}),
        ("logit_bias", {None: -1}),
        ("logit_bias", {True: -1}),
        ("logit_bias", {"42": float("nan")}),
        ("logit_bias", {"42": float("inf")}),
    ],
)
@pytest.mark.parametrize("channel", ["extra_args", "extra_body", "nested"])
def test_invalid_stop_and_logit_bias_disable_cache_without_errors(
    parameter: str, value: Any, channel: str, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {parameter: value}
    caller = (
        ModelSettings(extra_body=payload)
        if channel == "extra_body"
        else ModelSettings(extra_args={"extra_body": payload} if channel == "nested" else payload)
    )
    merged = merge_model_settings(caller, ModelSettings(extra_args={"cache": {"use-cache": True}}))
    assert "cache" not in (merged.extra_args or {})
    preserved = merged.extra_body if channel == "extra_body" else merged.extra_args
    assert isinstance(preserved, dict)
    if channel == "nested":
        preserved = preserved["extra_body"]
    assert isinstance(preserved, dict)
    assert preserved[parameter] is value
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
