"""Pinned callback adaptation, accounting, cache provenance and owner lifetime."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

pytest.importorskip("agents")

# isort: split
import httpx
import litellm
from _fakes.llm_run import HeldOperation, LLMRun, open_llm_run
from _fakes.secrecy import assert_secret_absent
from litellm.exceptions import ModifyResponseException
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients

from swarm_arena import Limits, RunConfig
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import LLMScope
from swarm_arena._engine.run import _setup_run
from swarm_arena._errors import ConfigError
from swarm_arena._journal.llm_record import drain, open_call, unbind_run
from swarm_arena._model.canonical import canonical_json
from swarm_arena.config import CachePolicy, ModelCatalog, ModelSpec, TracingConfig
from swarm_arena.llm import _logger
from swarm_arena.llm._logger import CallLogger, failure_usage, install, request_payload, usage_of
from swarm_arena.tool import Usage

pytestmark = pytest.mark.filterwarnings(
    "ignore:Items? .*TypedDict class .*ReadOnly.*:UserWarning:pydantic._internal._generate_schema"
)

_START = datetime(2026, 1, 1, tzinfo=UTC)
_END = _START + timedelta(milliseconds=125)
_ESTIMATE = Amount(tokens=9_000, usd=0.01)


@pytest.fixture(autouse=True)
async def isolated_logger(
    monkeypatch: pytest.MonkeyPatch, anyio_backend: str
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
    monkeypatch.setattr(_logger, "_installed_policy", None)
    try:
        yield
    finally:
        with anyio.fail_after(60):
            await GLOBAL_LOGGING_WORKER.flush()
            await GLOBAL_LOGGING_WORKER.stop()
            await close_litellm_async_clients()
        cache = litellm.cache
        if cache is not None and cache.type == "disk":
            cast(Any, cache.cache).disk_cache.close()


def _config(multiplier: int = 1, **kwargs: Any) -> RunConfig:
    return RunConfig(
        models=ModelCatalog(
            root={
                "m": ModelSpec(
                    model="openai/gpt-4o-mini",
                    roles=["judge"],
                    prices=ModelSpec.Prices(input=multiplier, output=2 * multiplier),
                )
            }
        ),
        limits=Limits(max_tokens=None, max_usd=None),
        **kwargs,
    )


async def _open(run: LLMRun, scope: LLMScope) -> dict[str, Any]:
    rid = await scope.tranche.reserve_call(_ESTIMATE)
    cid = open_call(
        scope,
        reservation_id=rid,
        estimate=_ESTIMATE,
        model_alias="m",
        model="openai/gpt-4o-mini",
        provider="openai",
    )
    metadata = run.metadata(scope, reservation_id=rid, call_id=cid)
    metadata["model_alias"] = "m"
    return {
        "litellm_params": {"metadata": metadata, "custom_llm_provider": "openai"},
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_call_id": "upstream-" + cid,
        "cache_hit": None,
        "response_cost": 9.9,
    }


def _response(**usage: Any) -> Any:
    return litellm.ModelResponse(
        id="response-original",
        usage=litellm.Usage(
            **(
                usage
                or {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                }
            )
        ),
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "finish", "arguments": "{}"},
                        },
                    ],
                }
            }
        ],
    )


async def test_success_event_maps_usage_cost_and_cache_fields(tmp_path: Path) -> None:
    response = _response(
        prompt_tokens=1200,
        completion_tokens=30,
        prompt_tokens_details={
            "cached_tokens": 800,
            "cache_creation_tokens": 300,
            "cache_creation_token_details": {
                "ephemeral_5m_input_tokens": 100,
                "ephemeral_1h_input_tokens": 200,
            },
        },
    )
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        await CallLogger().async_log_success_event(kw, response, _START, _END)
        assert await drain(scope, wait=True) == []
        call = run.calls()[0]
        assert call.usage == Usage(
            input_tokens=100,
            output_tokens=30,
            cache_read_tokens=800,
            cache_write_tokens=300,
            cache_write_5m_tokens=100,
            cache_write_1h_tokens=200,
        )
        assert call.usage.total_tokens == 1230
        assert (call.model, call.provider, call.latency_ms) == ("gpt-4o-mini", "openai", 125)
        assert [(tool["id"], tool["name"]) for tool in call.tool_calls] == [("tool-1", "finish")]
        assert call.litellm_call_id == kw["litellm_call_id"]
        assert run.charged(scope)._micro_usd == Amount(usd=call.cost_usd)._micro_usd


async def test_catalog_prices_override_response_cost(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        await CallLogger().async_log_success_event(
            await _open(run, scope), _response(), _START, _END
        )
        assert await drain(scope, wait=True) == []
        assert run.calls()[0].cost_usd == pytest.approx(0.0012)
        assert run.charged(scope) == Amount(tokens=1100, usd=0.0012)


async def test_two_bound_runs_price_with_their_own_catalogs(tmp_path: Path) -> None:
    async with (
        open_llm_run(tmp_path, config=_config()) as a,
        open_llm_run(tmp_path, config=_config(10)) as b,
    ):
        install(a.config)
        install(b.config)
        for run in (a, b):
            scope = run.scope("judge")
            await CallLogger().async_log_success_event(
                await _open(run, scope), _response(), _START, _END
            )
            await drain(scope, wait=True)
        a_call, b_call = a.calls()[0], b.calls()[0]
        assert (a_call.cost_usd, b_call.cost_usd) == (pytest.approx(0.0012), pytest.approx(0.012))


async def test_cache_hit_is_recorded_free(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        kw["cache_hit"] = True
        await CallLogger().async_log_success_event(kw, _response(), _START, _END)
        await drain(scope, wait=True)
        call = run.calls()[0]
        assert call.cache_hit
        assert call.cost_usd == 0
        assert call.usage == Usage(input_tokens=1000, output_tokens=100)
        assert call.cached_from_call_id is None
        assert run.charged(scope) == Amount()


async def test_failure_event_records_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        kw["exception"] = litellm.BadRequestError("PROVIDER_SECRET", "openai", "m")
        await CallLogger().async_log_failure_event(kw, None, _START, _END)
        await drain(scope, wait=True)
        call = run.calls()[0]
        assert call.status == "error"
        assert call.error is not None
        assert call.error.startswith("estimated:")
        assert (
            "PROVIDER_SECRET"
            not in canonical_json(
                [event.model_dump(mode="json") for event in run.events("llm_call")]
            ).decode()
        )
        assert "PROVIDER_SECRET" not in caplog.text


@pytest.mark.parametrize(
    "name",
    [
        "Timeout",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "RuntimeError",
        "ContentPolicyViolationError",
        "BadRequestError",
        "RateLimitError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "ContextWindowExceededError",
        "UnsupportedParamsError",
    ],
)
async def test_every_failure_without_usage_charges_the_reservation(
    name: str, tmp_path: Path
) -> None:
    exc = (
        RuntimeError("boom")
        if name == "RuntimeError"
        else getattr(litellm, name)(
            message="boom",
            model="m",
            llm_provider="openai",
            **(
                {
                    "response": httpx.Response(
                        403, request=httpx.Request("POST", "https://example.invalid")
                    )
                }
                if name == "PermissionDeniedError"
                else {}
            ),
        )
    )
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        kw["exception"] = exc
        await CallLogger().async_log_failure_event(kw, None, _START, _END)
        await drain(scope, wait=True)
        call = run.calls()[0]
        assert call.status == "error"
        assert call.cost_usd == _ESTIMATE.usd
        assert call.error is not None
        assert call.error.startswith("estimated:")
        assert run.charged(scope) == _ESTIMATE


async def test_provider_mapped_failures_stay_charged(tmp_path: Path) -> None:
    mapped: list[Exception] = []
    for message, model, provider in [
        ("The response was blocked.", "gemini-2.5-pro", "vertex_ai"),
        (
            "Please use an instance type with more CPU capacity or memory",
            "sagemaker/e",
            "sagemaker",
        ),
    ]:
        try:
            exception_type(
                model=model,
                custom_llm_provider=provider,
                original_exception=Exception(message),
                completion_kwargs={},
                extra_kwargs={},
            )
        except Exception as exc:
            mapped.append(exc)
    blocked, capacity = mapped
    assert (type(blocked), type(capacity)) == (
        litellm.ContentPolicyViolationError,
        litellm.ContextWindowExceededError,
    )
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        for exc in mapped:
            kw = await _open(run, scope)
            kw["exception"] = exc
            await CallLogger().async_log_failure_event(kw, None, _START, _END)
        await drain(scope, wait=True)
        for call in run.calls():
            assert (call.usage, call.cost_usd) == (Usage(input_tokens=9_000), 0.01)
            assert call.error is not None
            assert call.error.startswith("estimated:")
        assert run.charged(scope) == Amount(tokens=18_000, usd=0.02)


async def test_failure_with_authoritative_usage_settles_that_usage(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        kw["exception"] = ModifyResponseException(
            message="blocked SECRET", model="m", request_data={}, original_response=_response()
        )
        await CallLogger().async_log_failure_event(kw, None, _START, _END)
        await drain(scope, wait=True)
        call = run.calls()[0]
        assert (call.status, call.usage) == ("error", Usage(input_tokens=1000, output_tokens=100))
        assert call.cost_usd == pytest.approx(0.0012)
        assert call.error is not None
        assert not call.error.startswith("estimated:")
        assert run.charged(scope).tokens == 1100
        assert run.charged(scope).usd == pytest.approx(0.0012)


@pytest.mark.parametrize("failure", [False, True])
async def test_ttl_usage_uses_exact_provider_tariffs(tmp_path: Path, failure: bool) -> None:
    cfg = _config().model_copy(
        update={
            "models": ModelCatalog(
                root={
                    "m": ModelSpec(
                        model="anthropic/claude-sonnet-4-6",
                        roles=["judge"],
                    )
                }
            )
        }
    )
    response = _response(
        prompt_tokens=1000,
        completion_tokens=0,
        prompt_tokens_details={
            "cache_creation_tokens": 1000,
            "cache_creation_token_details": {
                "ephemeral_5m_input_tokens": 100,
                "ephemeral_1h_input_tokens": 900,
            },
        },
    )
    async with open_llm_run(tmp_path, config=cfg) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        kw["response_cost"] = None
        if failure:
            kw["exception"] = ModifyResponseException(
                message="blocked", model="m", request_data={}, original_response=response
            )
            await CallLogger().async_log_failure_event(kw, None, _START, _END)
        else:
            await CallLogger().async_log_success_event(kw, response, _START, _END)
        await drain(scope, wait=True)
        assert Amount(usd=run.calls()[0].cost_usd)._micro_usd == 5775
        assert run.charged(scope)._micro_usd == 5775


def test_request_payload_reads_optional_params() -> None:
    tools = [{"type": "function", "function": {"name": "finish"}}]
    kw = {
        "model": "m",
        "messages": [],
        "tools": [],
        "temperature": 1,
        "optional_params": {
            "tools": tools,
            "temperature": 0,
            "tool_choice": "auto",
            "max_tokens": 5,
            "response_format": {"type": "json_object"},
        },
    }
    payload = request_payload(kw)
    assert payload["tools"] == tools
    assert payload["temperature"] == 0
    assert set(payload) == {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "max_tokens",
        "response_format",
    }
    assert payload["tool_choice"] == "auto"
    assert payload["max_tokens"] == 5


def test_request_payload_never_holds_secrets() -> None:
    kwargs = {
        "model": "openai/gpt-6",
        "messages": [],
        "api_key": "sk-secret",
        "extra_headers": {"Authorization": "Bearer sk-secret"},
        "optional_params": {"api_key": "sk-secret"},
    }
    assert "sk-secret" not in canonical_json(request_payload(kwargs)).decode()


def test_usage_and_failure_usage_mapping() -> None:
    assert usage_of(
        {
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 4},
            }
        }
    ) == Usage(input_tokens=8, output_tokens=3, cache_read_tokens=4)
    assert failure_usage(RuntimeError("secret")) is None
    assert failure_usage(
        ModifyResponseException(
            message="blocked", model="m", request_data={}, original_response=_response()
        )
    ) == Usage(input_tokens=1000, output_tokens=100)


async def test_calls_without_swarm_metadata_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse(**kwargs: Any) -> None:
        pytest.fail("record_call must not be reached")

    monkeypatch.setattr(_logger, "record_call", refuse)
    for kw in ({}, {"litellm_params": {"metadata": {"other": "value"}}}):
        await CallLogger().async_log_success_event(kw, object(), _START, _END)
        await CallLogger().async_log_failure_event(kw, object(), _START, _END)


@pytest.mark.parametrize("defer_callback", [False, True])
async def test_acompletion_mock_response_is_recorded_and_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defer_callback: bool
) -> None:
    entered, release, recorded = anyio.Event(), anyio.Event(), anyio.Event()
    callback = CallLogger.async_log_success_event
    flush = GLOBAL_LOGGING_WORKER.flush

    async def observed(
        self: CallLogger, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        entered.set()
        if defer_callback:
            await release.wait()
        await callback(self, kwargs, response_obj, start_time, end_time)
        recorded.set()

    async def flush_releasing_callback() -> None:
        release.set()
        await flush()

    monkeypatch.setattr(CallLogger, "async_log_success_event", observed)
    monkeypatch.setattr(GLOBAL_LOGGING_WORKER, "flush", flush_releasing_callback)
    async with open_llm_run(tmp_path, config=_config()) as run:
        install(run.config)
        scope = run.scope("judge")
        kw = await _open(run, scope)
        try:
            await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=kw["messages"],
                mock_response="ok",
                metadata=kw["litellm_params"]["metadata"],
            )
            with anyio.fail_after(60):
                await entered.wait()
                if defer_callback:
                    assert not recorded.is_set()
                    assert run.calls() == []
                # Completion returns before the queued callback finishes accounting.
                await GLOBAL_LOGGING_WORKER.flush()
                assert await drain(scope, wait=True) == []
            assert recorded.is_set()
            assert len(run.calls()) == 1
            assert run.calls()[0].status == "ok"
            assert run.charged(scope) == Amount(
                tokens=run.calls()[0].usage.total_tokens, usd=run.calls()[0].cost_usd
            )
        finally:
            release.set()
            with anyio.fail_after(60):
                await flush()


async def test_response_cache_hit_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("diskcache")
    pytest.importorskip("fastapi")
    cfg = _config(cache=CachePolicy(backend="disk", disk_dir=str(tmp_path / "c")))
    async with open_llm_run(tmp_path, config=cfg) as run:
        install(cfg)
        for _ in range(2):
            scope = run.scope("judge")
            kw = await _open(run, scope)
            await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=kw["messages"],
                mock_response="ok",
                metadata=kw["litellm_params"]["metadata"],
                cache={"use-cache": True, "namespace": "judge"},
            )
            with anyio.fail_after(60):
                await GLOBAL_LOGGING_WORKER.flush()
                assert await drain(scope, wait=True) == []
        first, second = run.calls()
        assert (second.cache_hit, second.cost_usd, second.cached_from_call_id) == (
            True,
            0.0,
            first.id,
        )
        assert len(run.state().llm_cache_identity) == 2


def test_install_is_idempotent() -> None:
    cfg = _config()
    sentinel = CustomLogger()
    litellm.callbacks.append(sentinel)
    install(cfg)
    install(cfg)
    assert sum(isinstance(c, CallLogger) for c in litellm.callbacks) == 1
    assert sentinel in litellm.callbacks


@pytest.mark.parametrize("field", ["roles", "namespace"])
async def test_setup_accepts_scope_cache_policy_changes(tmp_path: Path, field: str) -> None:
    pytest.importorskip("diskcache")
    policy = CachePolicy(backend="disk", disk_dir=str(tmp_path / "cache"))
    await _setup_run((), _config(cache=policy))
    cache = litellm.cache
    await _setup_run(
        (),
        _config(cache=policy.model_copy(update={field: {} if field == "roles" else "another"})),
    )
    assert litellm.cache is cache


@pytest.mark.parametrize("field", ["backend", "disk_dir", "ttl_s"])
async def test_setup_refuses_a_different_cache_backend(tmp_path: Path, field: str) -> None:
    pytest.importorskip("diskcache")
    pytest.importorskip("fastapi")
    cfg = _config(
        cache=CachePolicy(
            backend="disk", disk_dir=str(tmp_path / "first"), redis_url_env="CACHE_URL"
        )
    )
    install(cfg)
    cache = litellm.cache
    install(cfg)
    assert litellm.cache is cache
    replacement: dict[str, Any] = {
        "backend": "redis",
        "disk_dir": "cache-secret-sentinel",
        "ttl_s": 1,
    }
    assert cfg.cache is not None
    changed = cfg.cache.model_copy(update={field: replacement[field]})
    with pytest.raises(ConfigError, match=r"response-cache backend.*per process") as caught:
        await _setup_run((), _config(cache=changed))
    assert_secret_absent(caught.value, replacement["disk_dir"], sanitized=True)
    assert litellm.cache is cache


def test_install_requires_cache_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ImportError, match="cache"):
        install(_config(cache=CachePolicy(backend="disk", disk_dir=str(tmp_path / "cache"))))
    assert litellm.cache is None
    assert _logger._installed_policy is None
    assert not litellm.callbacks


async def test_duplicate_and_late_callbacks_do_not_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        await CallLogger().async_log_success_event(kw, _response(), _START, _END)
        await drain(scope, wait=True)

        def refuse(data: bytes) -> str:
            pytest.fail("late callback uploaded a payload")

        monkeypatch.setattr(run.artifacts, "put_bytes", refuse)
        await CallLogger().async_log_success_event(kw, _response(), _START, _END)
        kw["exception"] = RuntimeError("secret")
        await CallLogger().async_log_failure_event(kw, None, _START, _END)
        unbind_run(run.run_id)
        await CallLogger().async_log_success_event(kw, _response(), _START, _END)
        assert len(run.calls()) == 1


async def test_callback_from_another_event_loop(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)

        def worker() -> None:
            asyncio.run(CallLogger().async_log_success_event(kw, _response(), _START, _END))

        await anyio.to_thread.run_sync(worker)
        assert await drain(scope, wait=True) == []
        assert run.charged(scope) == Amount(tokens=1100, usd=0.0012)


async def test_drain_abandons_blocked_adapter_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = HeldOperation()

    class Response:
        @property
        def usage(self) -> Any:
            held.wait()
            return {"prompt_tokens": 1000, "completion_tokens": 100}

    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        task = asyncio.create_task(
            CallLogger().async_log_success_event(kw, Response(), _START, _END)
        )
        try:
            await held.ready()
            with anyio.fail_after(60):
                assert await drain(scope, wait=False) == [
                    kw["litellm_params"]["metadata"]["call_id"]
                ]
                await task

            def refuse(data: bytes) -> str:
                pytest.fail("abandoned adapter uploaded")

            monkeypatch.setattr(run.artifacts, "put_bytes", refuse)
        finally:
            await held.finish()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert len(run.calls()) == 1
        assert run.charged(scope) == _ESTIMATE


async def test_callback_validation_logs_only_type_and_recovers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class Response:
        @property
        def usage(self) -> Any:
            raise ValueError("ADAPTER_SECRET")

    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        await CallLogger().async_log_success_event(kw, Response(), _START, _END)
        assert "ADAPTER_SECRET" not in caplog.text
        assert "ValueError" in caplog.text
        assert all(
            record.exc_info is None for record in caplog.records if record.name == "swarm_arena.llm"
        )
        await drain(scope, wait=False)
        assert run.charged(scope) == _ESTIMATE


@pytest.mark.parametrize("store_payloads", [False, True])
async def test_real_cache_provenance_namespaces_refresh_and_key_parameters(
    tmp_path: Path,
    store_payloads: bool,
) -> None:
    from agents import ModelSettings
    from agents.extensions.models.litellm_model import LitellmModel
    from agents.models.interface import ModelTracing

    pytest.importorskip("diskcache")
    pytest.importorskip("fastapi")
    cfg = _config(
        cache=CachePolicy(backend="disk", disk_dir=str(tmp_path / "cache")),
        tracing=TracingConfig(store_payloads=store_payloads),
    )
    async with open_llm_run(tmp_path, config=cfg) as run:
        for namespace, no_read, seed, hit in [
            ("A", False, 1, False),
            ("B", False, 1, False),
            ("B", False, 1, True),
            ("B", True, 1, False),
            ("B", False, 1, True),
            ("B", False, 2, False),
        ]:
            scope = run.scope("judge")
            kw = await _open(run, scope)
            await LitellmModel("openai/gpt-4o-mini").get_response(
                system_instructions="system",
                input="hi",
                tools=[],
                output_schema=None,
                handoffs=[],
                tracing=ModelTracing.DISABLED,
                model_settings=ModelSettings(
                    metadata=kw["litellm_params"]["metadata"],
                    extra_args={
                        "mock_response": "answer",
                        "seed": seed,
                        "cache": {"use-cache": True, "no-cache": no_read, "namespace": namespace},
                    },
                ),
            )
            assert await drain(scope, wait=True) == []
            assert run.calls()[-1].cache_hit is hit
        a, b, b_hit, refreshed, refreshed_hit, different_seed = run.calls()
        assert b_hit.cached_from_call_id == b.id
        assert refreshed_hit.cached_from_call_id == refreshed.id
        assert refreshed.id != b.id
        assert different_seed.cached_from_call_id is None
        assert all(call.request_ref is None for call in run.calls()) is (not store_payloads)
        state = run.state()
        assert state.llm_cache_identity[a.id] != state.llm_cache_identity[b.id]
        assert state.llm_cache_identity[b.id] == state.llm_cache_identity[b_hit.id]
        assert state.llm_cache_identity[refreshed.id] == state.llm_cache_identity[refreshed_hit.id]
        assert state.llm_cache_identity[b.id] != state.llm_cache_identity[refreshed.id]
        assert state.llm_cache_identity[refreshed.id] != state.llm_cache_identity[different_seed.id]


async def test_real_failure_callback_keeps_reservation(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        with pytest.raises(litellm.BadRequestError):
            await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=kw["messages"],
                metadata=kw["litellm_params"]["metadata"],
                mock_response=litellm.BadRequestError("secret", "openai", "m"),
                num_retries=0,
                max_retries=0,
            )
        assert await drain(scope, wait=True) == []
        assert run.charged(scope) == _ESTIMATE
        assert run.calls()[0].status == "error"


@pytest.mark.parametrize("invalid", [-1, True, 1.2, "USAGE_SECRET"])
def test_usage_rejects_malformed_counters(invalid: Any) -> None:
    from _fakes.secrecy import assert_secret_absent

    with pytest.raises(ValueError, match="prompt_tokens") as caught:
        usage_of({"usage": {"prompt_tokens": invalid}})
    assert_secret_absent(caught.value, "USAGE_SECRET")


def test_usage_rejects_inconsistent_ttl_subsets() -> None:
    with pytest.raises(ValueError, match="cache_creation_token_details"):
        usage_of(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "prompt_tokens_details": {
                        "cache_creation_tokens": 10,
                        "cache_creation_token_details": {
                            "ephemeral_5m_input_tokens": 6,
                            "ephemeral_1h_input_tokens": 5,
                        },
                    },
                }
            }
        )


@pytest.mark.parametrize("failure", [False, True])
async def test_malformed_callback_recovers_without_claim(tmp_path: Path, failure: bool) -> None:
    response = {"usage": {"prompt_tokens": "USAGE_SECRET"}}
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        if failure:
            error = RuntimeError("secret")
            cast(Any, error).original_response = response
            kw["exception"] = error
            await CallLogger().async_log_failure_event(kw, None, _START, _END)
        else:
            await CallLogger().async_log_success_event(kw, response, _START, _END)
        assert not run.calls()
        assert await drain(scope, wait=False) == [kw["litellm_params"]["metadata"]["call_id"]]
        assert run.charged(scope) == _ESTIMATE


async def test_terminal_cleanup_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from swarm_arena._errors import Fenced

    held = HeldOperation()
    cfg = _config().model_copy(update={"settle_timeout_s": 0.15})
    try:
        with pytest.raises(RuntimeError, match="stop cleanup timed out"):  # noqa: PT012 - whole owned lifetime
            async with open_llm_run(tmp_path, config=cfg) as run:
                scope = run.scope("judge")
                kw = await _open(run, scope)
                create = run.store.create

                def block(key: str, data: bytes) -> str:
                    if key.startswith("journal/"):
                        held.wait()
                    return create(key, data)

                monkeypatch.setattr(run.store, "create", block)
                task = asyncio.create_task(
                    CallLogger().async_log_success_event(kw, _response(), _START, _END)
                )
                try:
                    await held.ready()
                    run.stops.stop_now = True
                    with (
                        anyio.fail_after(30),
                        pytest.raises(RuntimeError, match="stop cleanup timed out") as caught,
                    ):
                        await task
                    assert not isinstance(caught.value, Fenced)
                    assert "LLM callback failed" not in caplog.text
                    assert not run.calls()
                    assert run.charged(scope) == _ESTIMATE
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
    finally:
        await held.finish()


async def test_overflow_remains_terminal(tmp_path: Path) -> None:
    from swarm_arena._budget.amount import USD_LIMIT
    from swarm_arena.manager import LimitReached

    cfg = _config().model_copy(
        update={
            "models": ModelCatalog(
                root={
                    "m": ModelSpec(
                        model="openai/gpt-4o-mini",
                        roles=["judge"],
                    )
                }
            )
        }
    )
    async with open_llm_run(tmp_path, config=cfg) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        kw["response_cost"] = USD_LIMIT + 1
        with pytest.raises(LimitReached) as caught:
            await CallLogger().async_log_success_event(kw, _response(), _START, _END)
        assert caught.value.limit == "usage_overflow"
        assert run.calls()[0].cost_usd == USD_LIMIT
        with pytest.raises(LimitReached):
            await scope.tranche.reserve_call(Amount(usd=0.001))
        await drain(scope, wait=False)


async def test_cancellation_abandons_adapter(tmp_path: Path) -> None:
    held = HeldOperation()

    class Response:
        @property
        def usage(self) -> Any:
            held.wait()
            return {"prompt_tokens": 1000}

    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        task = asyncio.create_task(
            CallLogger().async_log_success_event(kw, Response(), _START, _END)
        )
        try:
            await held.ready()
            task.cancel()
            with anyio.fail_after(30), pytest.raises(asyncio.CancelledError):
                await task
            await drain(scope, wait=False)
        finally:
            await held.finish()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert run.charged(scope) == _ESTIMATE


async def test_success_without_usage_keeps_reservation(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        await CallLogger().async_log_success_event(kw, {"id": "missing-usage"}, _START, _END)
        assert await drain(scope, wait=True) == []
        call = run.calls()[0]
        assert call.status == "error"
        assert call.error is not None
        assert call.error.startswith("estimated:")
        assert run.charged(scope) == _ESTIMATE


async def test_cache_hit_without_usage_is_still_free(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        kw["cache_hit"] = True
        await CallLogger().async_log_success_event(kw, {"id": "missing-usage"}, _START, _END)
        assert await drain(scope, wait=True) == []
        assert run.calls()[0].cache_hit is True
        assert run.charged(scope) == Amount()


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("pricing", ["catalog", "provider", "provider_cache"])
async def test_computed_usage_overflow_stops_later_reservations(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failure: bool,
    pricing: str,
) -> None:
    from swarm_arena._budget.amount import USD_LIMIT
    from swarm_arena._errors import _core_category, _is_core_signal, _SignalCategory
    from swarm_arena.manager import LimitReached

    cfg = _config()
    if pricing != "catalog":
        cfg = cfg.model_copy(
            update={
                "models": ModelCatalog(
                    root={
                        "m": ModelSpec(
                            model="anthropic/claude-sonnet-4-6",
                            roles=["judge"],
                        )
                    }
                )
            }
        )
    assert cfg.limits.max_tokens is None
    assert cfg.limits.max_usd is None
    response = _response(
        prompt_tokens=10**16,
        completion_tokens=0,
        **(
            {
                "prompt_tokens_details": {
                    "cache_creation_tokens": 10**16,
                    "cache_creation_token_details": {
                        "ephemeral_5m_input_tokens": 10**15,
                        "ephemeral_1h_input_tokens": 9 * 10**15,
                    },
                },
            }
            if pricing == "provider_cache"
            else {}
        ),
    )
    async with open_llm_run(tmp_path, config=cfg) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        if pricing != "catalog":
            kw["response_cost"] = None
        if failure:
            kw["exception"] = ModifyResponseException(
                message="blocked",
                model="m",
                request_data={},
                original_response=response,
            )
        logger = CallLogger()
        callback = logger.async_log_failure_event if failure else logger.async_log_success_event
        with pytest.raises(LimitReached) as caught:
            await callback(kw, None if failure else response, None, None)
        assert caught.value.limit == "usage_overflow"
        assert _core_category(caught.value) is _SignalCategory.TERMINAL
        assert _is_core_signal(caught.value)
        (call,) = run.calls()
        assert call.cost_usd == USD_LIMIT
        assert call.usage.total_tokens == 10**16
        assert call.error == "usage_overflow: charged at the supported USD limit"
        assert run.charged(scope) == Amount(tokens=10**16, usd=USD_LIMIT)
        assert run.events("settle")[0].payload["usage_overflow"] is True
        followup = run.scope("judge")
        for tranche in (scope.tranche, followup.tranche):
            with pytest.raises(LimitReached) as later:
                await tranche.reserve_call(Amount(tokens=1, usd=0.001))
            assert later.value.limit == "usage_overflow"
        assert await drain(scope, wait=False) == []
        assert await drain(followup, wait=False) == []
        assert len(run.calls()) == 1
        assert "LLM callback failed" not in caplog.text


@pytest.mark.parametrize("input_price", [1.0, 1.0000000000000004])
async def test_computed_charge_rounding_to_ceiling_is_not_overflow(
    tmp_path: Path,
    input_price: float,
) -> None:
    from swarm_arena._budget.amount import USD_LIMIT

    cfg = _config().model_copy(
        update={
            "models": ModelCatalog(
                root={
                    "m": ModelSpec(
                        model="openai/gpt-4o-mini",
                        roles=["judge"],
                        prices=ModelSpec.Prices(input=input_price, output=2),
                    )
                }
            )
        }
    )
    async with open_llm_run(tmp_path, config=cfg) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        await CallLogger().async_log_success_event(
            kw,
            _response(prompt_tokens=10**15, completion_tokens=0),
            None,
            None,
        )
        (call,) = run.calls()
        assert call.cost_usd == USD_LIMIT
        assert call.status == "ok"
        assert call.error is None
        followup = run.scope("judge")
        rid = await followup.tranche.reserve_call(Amount(tokens=1))
        await followup.tranche.settle_call(rid, Amount())
        assert await drain(scope, wait=False) == []
        assert await drain(followup, wait=False) == []


@pytest.mark.parametrize("terminal", [False, True])
async def test_callback_propagates_category_not_limit_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    terminal: bool,
) -> None:
    from swarm_arena._errors import (
        _core_category,
        _core_error,
        _SignalCategory,
        _stop_cleanup_error,
    )
    from swarm_arena.manager import LimitReached

    error = (
        _stop_cleanup_error()
        if terminal
        else _core_error(LimitReached, "usage_overflow", "ordinary local refusal")
    )

    async def refuse(**kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(_logger, "record_call", refuse)
    async with open_llm_run(tmp_path, config=_config()) as run:
        scope = run.scope("judge")
        kw = await _open(run, scope)
        callback = CallLogger().async_log_success_event(kw, _response(), _START, _END)
        if terminal:
            with pytest.raises(RuntimeError, match="stop cleanup timed out") as caught:
                await callback
            assert _core_category(caught.value) is _SignalCategory.TERMINAL
            assert "LLM callback failed" not in caplog.text
        else:
            await callback
            assert "LLM callback failed (LimitReached)" in caplog.text
        await drain(scope, wait=False)
