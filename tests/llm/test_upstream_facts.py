import asyncio
import dataclasses
import importlib.metadata
import inspect
import typing
from collections.abc import AsyncIterator
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import anyio
import pydantic
import pytest

# The package must pin the map before either SDK imports LiteLLM.
import swarm_arena.llm  # noqa: F401

# isort: split
import agents
import litellm
from agents import (
    Agent,
    AgentOutputSchemaBase,
    FunctionTool,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    RunConfig,
    RunHooks,
    Runner,
    Usage,
)
from agents.agent import StopAtTools
from agents.exceptions import AgentsException, UserError
from agents.extensions.models.litellm_model import LitellmModel
from agents.models._retry_runtime import provider_managed_retries_disabled
from agents.models.multi_provider import MultiProvider
from agents.tool_context import ToolContext
from litellm.caching.caching import Cache, CacheMode
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
from litellm.types.caching import LiteLLMCacheType
from litellm.types.utils import PromptTokensDetailsWrapper
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

# LiteLLM creates Pydantic schemas from upstream ReadOnly TypedDicts at first call.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Items? .*TypedDict class .*ReadOnly.*:UserWarning:pydantic._internal._generate_schema"
)


@pytest.fixture(autouse=True)
async def _isolated_upstream(
    monkeypatch: pytest.MonkeyPatch,
    anyio_backend: str,
) -> AsyncIterator[None]:
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
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    try:
        yield
    finally:
        with anyio.fail_after(10):
            await GLOBAL_LOGGING_WORKER.flush()
            await GLOBAL_LOGGING_WORKER.stop()
            await close_litellm_async_clients()


class _CaptureLogger(CustomLogger):
    def __init__(self) -> None:
        self.success: list[tuple[dict[str, Any], Any]] = []
        self.failure: list[tuple[dict[str, Any], Any]] = []
        self.ready = asyncio.Queue[None]()

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.success.append((kwargs, response_obj))
        self.ready.put_nowait(None)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.failure.append((kwargs, response_obj))
        self.ready.put_nowait(None)

    async def wait(self) -> None:
        with anyio.fail_after(10):
            await self.ready.get()


class _DictSchema(AgentOutputSchemaBase):
    def __init__(self, field: str = "answer") -> None:
        self.field = field

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return "Answer"

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {self.field: {"type": "string"}},
            "required": [self.field],
            "additionalProperties": False,
        }

    def is_strict_json_schema(self) -> bool:
        return True

    def validate_json(self, json_str: str) -> Any:
        return pydantic.TypeAdapter(dict[str, str]).validate_json(json_str)


async def _invoke_adapter(
    settings: ModelSettings,
    *,
    schema: AgentOutputSchemaBase | None = None,
    tools: list[Any] | None = None,
) -> ModelResponse:
    return await LitellmModel("anthropic/claude-sonnet-4-6").get_response(
        system_instructions="system",
        input="hi",
        model_settings=settings,
        tools=tools or [],
        output_schema=schema,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    )


async def _tool(context: Any, arguments: str) -> str:
    return "ok"


def _function_tool() -> FunctionTool:
    return FunctionTool(
        name="finish_round",
        description="Finish",
        params_json_schema={"type": "object"},
        on_invoke_tool=_tool,
        strict_json_schema=False,
        timeout_seconds=1.0,
        timeout_behavior="error_as_result",
    )


def test_custom_logger_hook_signatures() -> None:
    for name in ("async_log_success_event", "async_log_failure_event"):
        params = list(inspect.signature(getattr(CustomLogger, name)).parameters)
        assert params == ["self", "kwargs", "response_obj", "start_time", "end_time"]


async def test_callback_kwargs_carry_metadata_cost_cache_and_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _CaptureLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="ok",
        metadata={"run_id": "r", "reservation_id": "res-1"},
        tools=[
            {"type": "function", "function": {"name": "finish", "parameters": {"type": "object"}}}
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    await logger.wait()
    seen, logged_response = logger.success[0]
    assert logged_response is response
    assert seen["litellm_params"]["metadata"]["reservation_id"] == "res-1"
    assert isinstance(seen["litellm_call_id"], str)
    assert {"response_cost", "cache_hit", "model", "custom_llm_provider"} <= (
        set(seen) | set(seen["standard_logging_object"])
    )
    assert {"tools", "temperature", "response_format"} <= set(seen["optional_params"])


def test_capability_helpers() -> None:
    assert not hasattr(litellm, "supports_prompt_caching")
    assert callable(litellm.utils.supports_prompt_caching)
    assert litellm.supports_function_calling("deepseek/deepseek-reasoner") is False
    assert callable(litellm.supports_response_schema)


def test_model_settings_fields() -> None:
    assert {
        "metadata",
        "extra_args",
        "include_usage",
        "max_tokens",
        "parallel_tool_calls",
        "temperature",
    } <= {f.name for f in dataclasses.fields(ModelSettings)}


def test_model_settings_metadata_is_str_to_str() -> None:
    assert typing.get_type_hints(ModelSettings)["metadata"] == dict[str, str] | None
    with pytest.raises(pydantic.ValidationError):
        ModelSettings(metadata={"round": 1})  # ty: ignore[invalid-argument-type]


def test_model_get_response_signature() -> None:
    params = inspect.signature(agents.Model.get_response).parameters
    assert all(
        params[n].kind is inspect.Parameter.KEYWORD_ONLY
        for n in ("previous_response_id", "conversation_id", "prompt")
    )


def test_model_settings_reasoning_and_resolve_merge_extra_args() -> None:
    assert "reasoning" in {f.name for f in dataclasses.fields(ModelSettings)}
    assert ModelSettings(extra_args={"a": 1}).resolve(
        ModelSettings(extra_args={"b": 2})
    ).extra_args == {"a": 1, "b": 2}


def test_agent_output_schema_base_abstract_methods() -> None:
    assert AgentOutputSchemaBase.__abstractmethods__ == {
        "is_plain_text",
        "name",
        "json_schema",
        "is_strict_json_schema",
        "validate_json",
    }


def test_drop_params_removes_unsupported_temperature() -> None:
    params = litellm.get_optional_params(
        model="o3", custom_llm_provider="openai", temperature=0, drop_params=True
    )
    assert "temperature" not in params


async def test_litellm_model_sends_extra_args_top_level_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ok"}}])

    monkeypatch.setattr(litellm, "acompletion", completion)
    injection = [{"location": "message", "role": "system"}, {"location": "message", "index": -1}]
    cache = {"use-cache": True, "no-cache": True, "no-store": False, "namespace": "ns"}
    await _invoke_adapter(
        ModelSettings(
            metadata={"reservation_id": "res-1"},
            extra_args={"cache_control_injection_points": injection, "cache": cache},
        )
    )
    assert captured["cache_control_injection_points"] == injection
    assert captured["cache"] == cache
    assert captured["metadata"]["reservation_id"] == "res-1"


async def test_real_adapter_dict_schema_and_effective_cache_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    async def completion(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return litellm.ModelResponse()

    monkeypatch.setattr(litellm, "acompletion", completion)
    cache = Cache(type=LiteLLMCacheType.LOCAL, mode=CacheMode.default_off)
    base = ModelSettings(temperature=0, extra_args={"seed": 1, "cache": {"namespace": "A"}})
    await _invoke_adapter(base, schema=_DictSchema(), tools=[_function_tool()])
    original = captured[-1]
    assert isinstance(original["response_format"], dict)
    assert original["response_format"]["json_schema"]["schema"] == _DictSchema().json_schema()
    key = cache.get_cache_key(**original)
    variants = [
        (base, _DictSchema("changed"), [_function_tool()]),
        (base.resolve(ModelSettings(temperature=0.5)), _DictSchema(), [_function_tool()]),
        (base.resolve(ModelSettings(extra_args={"seed": 2})), _DictSchema(), [_function_tool()]),
        (
            base.resolve(ModelSettings(extra_args={"cache": {"namespace": "B"}})),
            _DictSchema(),
            [_function_tool()],
        ),
        (base, _DictSchema(), []),
    ]
    for settings, schema, tools in variants:
        await _invoke_adapter(settings, schema=schema, tools=tools)
        assert cache.get_cache_key(**captured[-1]) != key
    await _invoke_adapter(
        base.resolve(ModelSettings(metadata={"reservation_id": "other"})),
        schema=_DictSchema(),
        tools=[_function_tool()],
    )
    assert cache.get_cache_key(**captured[-1]) == key
    for name, value in (("top_p", 0.2), ("reasoning_effort", "high")):
        assert cache.get_cache_key(**dict(original, **{name: value})) != key


async def test_adapter_retry_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    async def completion(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return litellm.ModelResponse()

    monkeypatch.setattr(litellm, "acompletion", completion)
    await _invoke_adapter(ModelSettings())
    assert "num_retries" not in captured[-1]
    assert "max_retries" not in captured[-1]
    await _invoke_adapter(ModelSettings(extra_args={"num_retries": 0, "max_retries": 0}))
    assert captured[-1]["num_retries"] == captured[-1]["max_retries"] == 0
    with provider_managed_retries_disabled(True):
        await _invoke_adapter(ModelSettings(extra_args={"num_retries": 4, "max_retries": 4}))
    assert captured[-1]["num_retries"] == captured[-1]["max_retries"] == 0


class _CapturingModel(Model):
    def __init__(self, *, error: Exception | None = None, tool: bool = False) -> None:
        self.settings: list[ModelSettings] = []
        self.error = error
        self.tool = tool

    async def get_response(
        self,
        system_instructions: Any,
        input: Any,
        model_settings: ModelSettings,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        *,
        previous_response_id: Any,
        conversation_id: Any,
        prompt: Any,
    ) -> ModelResponse:
        self.settings.append(model_settings)
        if self.error is not None:
            raise self.error
        output: list[Any] = (
            [
                ResponseFunctionToolCall(
                    id="fc",
                    call_id="tc",
                    name="finish_round",
                    arguments="{}",
                    type="function_call",
                )
            ]
            if self.tool
            else [
                ResponseOutputMessage(
                    id="msg",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[ResponseOutputText(text="done", annotations=[], type="output_text")],
                )
            ]
        )
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    async def stream_response(
        self,
        system_instructions: Any,
        input: Any,
        model_settings: ModelSettings,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        *,
        previous_response_id: Any,
        conversation_id: Any,
        prompt: Any,
    ) -> AsyncIterator[Any]:
        raise NotImplementedError
        yield  # pragma: no cover


async def test_model_settings_resolved_before_on_llm_start() -> None:
    class Hooks(RunHooks):
        async def on_llm_start(
            self, context: Any, agent: Any, system_prompt: Any, input_items: Any
        ) -> None:
            agent.model_settings = ModelSettings(metadata={"x": "1"})

    model = _CapturingModel()
    agent = Agent(name="test", model=model)
    await Runner.run(agent, "hi", hooks=Hooks(), run_config=RunConfig(tracing_disabled=True))
    assert model.settings[0].metadata is None
    assert agent.model_settings.metadata == {"x": "1"}


async def _run_tool_error(error: Exception) -> None:
    async def invoke(context: Any, arguments: str) -> str:
        raise error

    tool = _function_tool()
    tool.on_invoke_tool = invoke
    await Runner.run(
        Agent(name="test", model=_CapturingModel(tool=True), tools=[tool]),
        "hi",
        run_config=RunConfig(tracing_disabled=True),
    )


async def test_agents_exception_from_function_tool_leaves_runner_unchanged() -> None:
    class Control(AgentsException):
        pass

    error = Control("control")
    with pytest.raises(Control) as caught:
        await _run_tool_error(error)
    assert caught.value is error


async def test_plain_exception_from_function_tool_becomes_user_error() -> None:
    error = RuntimeError("tool")
    with pytest.raises(UserError) as caught:
        await _run_tool_error(error)
    assert caught.value.__cause__ is error


async def test_model_exception_leaves_runner_unchanged() -> None:
    error = RuntimeError("model")
    model = _CapturingModel(error=error)
    with pytest.raises(RuntimeError) as caught:
        await Runner.run(
            Agent(name="test", model=model), "hi", run_config=RunConfig(tracing_disabled=True)
        )
    assert caught.value is error
    assert len(model.settings) == 1


def test_function_tool_direct_construction_fields() -> None:
    tool = _function_tool()
    assert tool.timeout_behavior == "error_as_result"
    assert tool.timeout_seconds == 1.0
    assert tool.strict_json_schema is False


def test_runner_run_and_tool_context() -> None:
    assert {"context", "max_turns", "hooks", "run_config"} <= set(
        inspect.signature(Runner.run).parameters
    )
    assert "tool_call_id" in {f.name for f in dataclasses.fields(ToolContext)}
    assert StopAtTools(stop_at_tool_names=["finish_round"]) == {
        "stop_at_tool_names": ["finish_round"]
    }


def test_cache_constructor_parameters() -> None:
    assert {
        "type",
        "mode",
        "namespace",
        "ttl",
        "s3_bucket_name",
        "s3_path",
        "disk_cache_dir",
    } <= set(inspect.signature(Cache).parameters)


def test_redis_cache_gets_url_through_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Redis:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("litellm.caching.caching.RedisCache", Redis)
    Cache(
        type=LiteLLMCacheType.REDIS,
        mode=CacheMode.default_off,
        url="rediss://user:pw@redis.example:6380/2",
    )
    assert any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(Cache).parameters.values()
    )
    assert captured["url"] == "rediss://user:pw@redis.example:6380/2"


async def test_disk_cache_hit_flag_zero_callback_cost_and_same_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("diskcache")
    pytest.importorskip("fastapi")
    cache = Cache(
        type=LiteLLMCacheType.DISK, mode=CacheMode.default_off, disk_cache_dir=str(tmp_path)
    )
    logger = _CaptureLogger()
    monkeypatch.setattr(litellm, "cache", cache)
    monkeypatch.setattr(litellm, "callbacks", [logger])
    request: dict[str, Any] = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "mock_response": "first",
        "cache": {"use-cache": True, "namespace": "A"},
    }
    try:
        first = await litellm.acompletion(**request)
        await logger.wait()
        hit = await litellm.acompletion(**request)
        await logger.wait()
        seen = [entry[0] for entry in logger.success]
        assert [k["cache_hit"] for k in seen] == [None, True]
        assert seen[1]["response_cost"] == 0.0
        assert isinstance(first, litellm.ModelResponse)
        assert isinstance(hit, litellm.ModelResponse)
        assert hit.id == first.id
        assert logger.success[1][1].id == first.id
        assert typing.cast(Any, hit).usage == typing.cast(Any, first).usage
        assert hit._hidden_params["cache_hit"] is True
        assert hit._hidden_params["cache_key"] == cache.get_cache_key(**request)
        assert seen[1]["litellm_params"]["preset_cache_key"] == hit._hidden_params["cache_key"]
        assert hit._hidden_params["cache_key"].startswith("A:")
        assert "cache" not in seen[1]  # Namespace is exposed through the effective key.
        refreshed = await litellm.acompletion(
            **dict(
                request,
                mock_response="replacement",
                cache={"use-cache": True, "no-cache": True, "namespace": "A"},
            )
        )
        await logger.wait()
        replaced_hit = await litellm.acompletion(**request)
        await logger.wait()
        assert isinstance(refreshed, litellm.ModelResponse)
        assert isinstance(replaced_hit, litellm.ModelResponse)
        assert isinstance(replaced_hit.choices[0], litellm.Choices)
        assert refreshed.id != first.id
        assert replaced_hit.id == refreshed.id
        assert logger.success[3][1].id == refreshed.id
        assert replaced_hit.choices[0].message.content == "replacement"
        other = await litellm.acompletion(
            **dict(request, cache={"use-cache": True, "namespace": "B"})
        )
        await logger.wait()
        assert isinstance(other, litellm.ModelResponse)
        assert other.id != refreshed.id
        assert logger.success[-1][0]["cache_hit"] is None
    finally:
        typing.cast(Any, cache.cache).disk_cache.close()


def test_cost_map_reload_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    entry = litellm.get_model_cost_map(url=litellm.model_cost_map_url)["claude-sonnet-4-6"]
    assert entry["input_cost_per_token"] == 3e-06
    assert entry["cache_read_input_token_cost"] == 3e-07
    assert entry["cache_creation_input_token_cost"] == 3.75e-06
    assert entry["cache_creation_input_token_cost_above_1hr"] == 6e-06


def test_usage_detail_field_names() -> None:
    usage = AnthropicConfig().calculate_usage(
        usage_object={
            "input_tokens": 50,
            "output_tokens": 10,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 1000,
            "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 900},
        },
        reasoning_content=None,
    )
    assert usage.prompt_tokens == 1250  # Includes fresh input, reads and both write TTLs.
    assert usage.completion_tokens == 10
    details = usage.prompt_tokens_details
    assert isinstance(details, PromptTokensDetailsWrapper)
    assert details.cached_tokens == 200
    assert details.cache_creation_tokens == details.cache_write_tokens == 1000
    assert details.cache_creation_token_details is not None
    assert details.cache_creation_token_details.ephemeral_5m_input_tokens == 100
    assert details.cache_creation_token_details.ephemeral_1h_input_tokens == 900


def test_provider_exception_classes_exist() -> None:
    for name in (
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
        "APIConnectionError",
        "Timeout",
        "BadRequestError",
        "NotFoundError",
    ):
        assert issubclass(getattr(litellm, name), Exception)


async def test_failure_classes_do_not_prove_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    def mapped(text: str, model: str, provider: str) -> tuple[str, int]:
        try:
            exception_type(
                model=model,
                custom_llm_provider=provider,
                original_exception=Exception(text),
                completion_kwargs={},
                extra_kwargs={},
            )
        except Exception as error:
            return type(error).__name__, typing.cast(Any, error).status_code
        pytest.fail("mapping must raise")

    sm = ("sagemaker/e", "sagemaker")
    assert mapped("Please use an instance type with more CPU capacity or memory", *sm) == (
        "ContextWindowExceededError",
        400,
    )
    assert mapped("`inputs` tokens + `max_new_tokens` must be <= 4096", *sm) == (
        "ContextWindowExceededError",
        400,
    )
    assert mapped("The response was blocked.", "gemini-2.5-pro", "vertex_ai") == (
        "ContentPolicyViolationError",
        400,
    )
    logger = _CaptureLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    error = litellm.ContentPolicyViolationError(
        message="blocked", model="gpt-4o-mini", llm_provider="openai"
    )
    with pytest.raises(litellm.ContentPolicyViolationError):
        await litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=error,
            num_retries=0,
            max_retries=0,
        )
    await logger.wait()
    kw, response_obj = logger.failure[0]
    assert (type(kw["exception"]), response_obj) == (litellm.ContentPolicyViolationError, None)
    assert (
        kw["standard_logging_object"]["prompt_tokens"],
        kw["standard_logging_object"]["completion_tokens"],
    ) == (0, 0)
    from litellm.exceptions import ModifyResponseException

    response = litellm.ModelResponse(
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    )
    assert (
        ModifyResponseException(
            message="b", model="m", request_data={}, original_response=response
        ).original_response
        is response
    )


def test_cache_extra_provides_hit_and_disk_modules() -> None:
    pytest.importorskip("diskcache")
    assert all(find_spec(module) for module in ("fastapi", "orjson", "diskcache", "boto3", "redis"))


def test_langfuse_v5_api() -> None:
    langfuse = pytest.importorskip("langfuse")
    assert importlib.metadata.version("langfuse").startswith("5.")
    assert callable(langfuse.get_client)
    assert {"session_id", "metadata", "tags"} <= set(
        inspect.signature(langfuse.propagate_attributes).parameters
    )
    assert callable(langfuse.Langfuse.get_current_trace_id)
    assert callable(langfuse.Langfuse.get_trace_url)


def test_instrumentor_uses_exclusive_processors(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langfuse")
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    replaced: list[Any] = []
    added: list[Any] = []
    monkeypatch.setattr(agents, "set_trace_processors", replaced.append)
    monkeypatch.setattr(agents, "add_trace_processor", added.append)
    instrumentor = OpenAIAgentsInstrumentor()
    provider = TracerProvider()
    try:
        instrumentor.instrument(tracer_provider=provider)
        assert len(replaced) == 1
        assert len(replaced[0]) == 1
        assert type(replaced[0][0]).__name__ == "OpenInferenceTracingProcessor"
        assert added == []
    finally:
        instrumentor.uninstrument()
        provider.shutdown()


async def test_multi_provider_closes_each_owned_provider_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Provider(agents.ModelProvider):
        def __init__(self) -> None:
            self.closes = 0

        def get_model(self, model_name: str | None) -> Model:
            return _CapturingModel()

        async def aclose(self) -> None:
            self.closes += 1

    provider = Provider()
    multi = MultiProvider()
    monkeypatch.setattr(multi, "openai_provider", provider)
    monkeypatch.setattr(multi, "_fallback_providers", {"duplicate": provider})
    await multi.aclose()
    assert provider.closes == 1


# Pinned provider thinking controls, including fields inside native request envelopes.
# The settings detector's inventory is checked against these real transformations.
THINKING_PARAMETER_CASES: dict[str, tuple[str, dict[str, Any], tuple[str, ...], Any]] = {
    "think": ("ollama_chat", {"think": True}, ("think",), True),
    "thinking": (
        "gemini",
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        ("generationConfig", "thinkingConfig", "thinkingBudget"),
        1024,
    ),
    "reasoning": ("openrouter", {"reasoning": {"effort": "high"}}, ("reasoning", "effort"), "high"),
    "reasoning_effort": (
        "gemini",
        {"reasoning_effort": "low"},
        ("generationConfig", "thinkingConfig", "thinkingBudget"),
        1024,
    ),
    "thinkingConfig": (
        "gemini",
        {"thinkingConfig": {"thinkingBudget": 1024}},
        ("generationConfig", "thinkingConfig", "thinkingBudget"),
        1024,
    ),
    "thinking_config": (
        "gemini",
        {"thinking_config": {"thinking_budget": 1024}},
        ("generationConfig", "thinking_config", "thinking_budget"),
        1024,
    ),
    "budget_tokens": (
        "gemini",
        {"thinking": {"type": "enabled", "budget_tokens": 2048}},
        ("generationConfig", "thinkingConfig", "thinkingBudget"),
        2048,
    ),
    "thinkingBudget": (
        "gemini",
        {"thinkingConfig": {"thinkingBudget": 2048}},
        ("generationConfig", "thinkingConfig", "thinkingBudget"),
        2048,
    ),
    "thinking_budget": (
        "gemini",
        {"thinking_config": {"thinking_budget": 2048}},
        ("generationConfig", "thinking_config", "thinking_budget"),
        2048,
    ),
    "thinkingLevel": (
        "gemini",
        {"thinkingConfig": {"thinkingLevel": "high"}},
        ("generationConfig", "thinkingConfig", "thinkingLevel"),
        "high",
    ),
    "thinking_level": (
        "gemini",
        {"thinking_config": {"thinking_level": "high"}},
        ("generationConfig", "thinking_config", "thinking_level"),
        "high",
    ),
    "enable_thinking": (
        "fireworks_ai",
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ("reasoning_effort",),
        "none",
    ),
    "reasoning_budget": (
        "fireworks_ai",
        {"extra_body": {"chat_template_kwargs": {"reasoning_budget": 1024}}},
        ("reasoning_effort",),
        1024,
    ),
    "low_effort": (
        "fireworks_ai",
        {"extra_body": {"chat_template_kwargs": {"low_effort": True}}},
        ("reasoning_effort",),
        "low",
    ),
    "reasoningConfig": (
        "bedrock",
        {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}},
        ("additionalModelRequestFields", "reasoningConfig", "maxReasoningEffort"),
        "high",
    ),
    "maxReasoningEffort": (
        "bedrock",
        {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}},
        ("additionalModelRequestFields", "reasoningConfig", "maxReasoningEffort"),
        "high",
    ),
    "effort": (
        "anthropic",
        {"output_config": {"effort": "high"}},
        ("output_config", "effort"),
        "high",
    ),
}


@pytest.mark.parametrize("parameter", THINKING_PARAMETER_CASES)
def test_provider_thinking_parameter_maps_to_thinking_field(parameter: str) -> None:
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig
    from litellm.llms.ollama.chat.transformation import OllamaChatConfig
    from litellm.llms.vertex_ai.gemini.transformation import transform_request_body
    from litellm.utils import get_optional_params

    provider, request, path, expected = THINKING_PARAMETER_CASES[parameter]
    models = {
        "gemini": "gemini-2.5-flash",
        "ollama_chat": "qwen3",
        "openrouter": "openai/gpt-5",
        "bedrock": "amazon.nova-2-lite-v1:0",
        "anthropic": "claude-opus-4-6",
    }
    if provider == "fireworks_ai":
        body = FireworksAIConfig().map_extra_body_params(
            request, model="accounts/fireworks/models/deepseek-v3p1"
        )
    else:
        model = models[provider]
        optional = get_optional_params(
            model=model, custom_llm_provider=provider, max_tokens=8000, **request
        )
        if provider == "gemini":
            body = transform_request_body(
                messages=[{"role": "user", "content": "hi"}],
                model=model,
                optional_params=optional,
                custom_llm_provider=provider,
                litellm_params={},
                cached_content=None,
            )
        elif provider in {"anthropic", "bedrock", "ollama_chat"}:
            config = {
                "anthropic": AnthropicConfig,
                "bedrock": AmazonConverseConfig,
                "ollama_chat": OllamaChatConfig,
            }[provider]()
            body = config.transform_request(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                optional_params=optional,
                litellm_params={},
                headers={},
            )
        else:
            body = optional
    value: Any = body
    for field in path:
        value = value[field]
    assert value == expected


def test_thinking_detector_matches_pinned_parameter_inventory() -> None:
    from swarm_arena.llm._settings import _THINKING_CONTROLS

    assert THINKING_PARAMETER_CASES.keys() == _THINKING_CONTROLS


@pytest.mark.parametrize("provider", ["ollama_chat", "ollama"])
def test_ollama_think_maps_on_chat_and_completion_routes(provider: str) -> None:
    from litellm.llms.ollama.chat.transformation import OllamaChatConfig
    from litellm.llms.ollama.completion.transformation import OllamaConfig
    from litellm.utils import get_optional_params

    optional = get_optional_params(model="qwen3", custom_llm_provider=provider, think=True)
    config = OllamaChatConfig() if provider == "ollama_chat" else OllamaConfig()
    body = config.transform_request(
        model="qwen3",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional,
        litellm_params={},
        headers={},
    )
    assert body["think"] is True


@pytest.mark.parametrize("channel", ["extra_args", "extra_body"])
def test_unknown_provider_parameter_is_preserved_but_disables_response_cache(channel: str) -> None:
    from litellm.utils import get_optional_params

    from swarm_arena.llm._settings import merge_model_settings

    native: dict[str, Any] = {"unlisted_native_mode": True}
    # The pinned provider mapper accepts native controls beyond swarm-arena's inventory.
    optional = get_optional_params(model="gpt-4o-mini", custom_llm_provider="openai", **native)
    assert optional["extra_body"]["unlisted_native_mode"] is True
    caller = (
        ModelSettings(extra_args=native)
        if channel == "extra_args"
        else ModelSettings(extra_body=native)
    )
    merged = merge_model_settings(
        caller, ModelSettings(extra_args={"cache": {"use-cache": True, "namespace": "A"}})
    )
    assert getattr(merged, channel)["unlisted_native_mode"] is True
    assert "cache" not in (merged.extra_args or {})


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("stop", "END"), ("stop", ["END", "DONE"]), ("logit_bias", {"42": -1, "43": 0.5})],
)
@pytest.mark.parametrize("channel", ["extra_args", "extra_body"])
def test_openai_stop_and_logit_bias_remain_cacheable(
    parameter: str, value: Any, channel: str
) -> None:
    from litellm.utils import get_optional_params

    from swarm_arena.llm._settings import merge_model_settings

    native: dict[str, Any] = {parameter: value}
    optional = get_optional_params(model="gpt-4o-mini", custom_llm_provider="openai", **native)
    assert optional[parameter] == value
    caller = (
        ModelSettings(extra_args=native)
        if channel == "extra_args"
        else ModelSettings(extra_body=native)
    )
    cache = {"use-cache": True, "no-cache": False, "no-store": False, "namespace": "A"}
    merged = merge_model_settings(caller, ModelSettings(extra_args={"cache": cache}))
    assert getattr(merged, channel)[parameter] == value
    assert (merged.extra_args or {})["cache"] == cache
