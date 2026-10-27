"""Response-cache policy, backend construction and proven origins."""

import importlib.util
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

pytest.importorskip("agents")

# isort: split
import litellm
from _fakes.llm_run import open_llm_run, record_kwargs
from _fakes.secrecy import assert_secret_absent
from litellm.caching.caching import CacheMode
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients

from swarm_arena import RunConfig
from swarm_arena._budget.amount import Amount
from swarm_arena._journal.llm_record import bind_run, drain, open_call, record_call, unbind_run
from swarm_arena.config import CachePolicy, TracingConfig
from swarm_arena.llm._cache import (
    cache_identity,
    cache_namespace,
    cache_switches,
    require_cache_extra,
    response_cache,
)
from tests.llm.test_upstream_facts import _CaptureLogger

pytestmark = pytest.mark.filterwarnings(
    "ignore:Items? .*TypedDict class .*ReadOnly.*:UserWarning:pydantic._internal._generate_schema"
)


def test_role_switches_match_spec_defaults() -> None:
    p = CachePolicy(backend="disk")
    assert [
        cache_switches(p, r, force_no_read=False, streamed=False, thinking=False)
        for r in ("judge", "planner", "manager", "curator")
    ] == [(True, True), (False, True), (False, False), (True, True)]


def test_anchors_and_final_never_read_but_still_write() -> None:
    assert cache_switches(
        CachePolicy(backend="disk"), "judge", force_no_read=True, streamed=False, thinking=False
    ) == (False, True)


@pytest.mark.parametrize(("streamed", "thinking"), [(True, False), (False, True), (True, True)])
def test_streamed_or_thinking_never_cached(streamed: bool, thinking: bool) -> None:
    assert cache_switches(
        CachePolicy(backend="disk"),
        "judge",
        force_no_read=False,
        streamed=streamed,
        thinking=thinking,
    ) == (False, False)


def test_no_policy_no_cache() -> None:
    assert cache_switches(None, "judge", force_no_read=False, streamed=False, thinking=False) == (
        False,
        False,
    )
    assert cache_switches(
        CachePolicy(backend="disk", roles={}),
        "judge",
        force_no_read=False,
        streamed=False,
        thinking=False,
    ) == (False, False)


def test_cache_namespace_override() -> None:
    assert cache_namespace(CachePolicy(backend="disk", namespace="shared"), "physika") == "shared"
    assert cache_namespace(CachePolicy(backend="disk"), "physika") == "physika"
    assert cache_namespace(None, "physika") == "physika"


def test_response_cache_disk(tmp_path: Path) -> None:
    pytest.importorskip("diskcache")
    pytest.importorskip("fastapi")
    cache = response_cache(CachePolicy(backend="disk", disk_dir=str(tmp_path)))
    try:
        assert cache.type == "disk"
        assert cache.mode == CacheMode.default_off
        assert cache.ttl == 2592000
        assert Path(cast(Any, cache.cache).disk_cache.directory) == tmp_path
    finally:
        cast(Any, cache.cache).disk_cache.close()


def test_response_cache_s3_path_split(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("boto3")
    pytest.importorskip("fastapi")
    kw: dict[str, Any] = {}

    def capture(**kwargs: Any) -> Any:
        kw.update(kwargs)
        return object()

    monkeypatch.setattr("swarm_arena.llm._cache.Cache", capture)
    response_cache(CachePolicy(backend="s3", s3_path="s3://bucket/swarm-arena/llm-cache"))
    assert (kw["s3_bucket_name"], kw["s3_path"], kw["mode"]) == (
        "bucket",
        "swarm-arena/llm-cache",
        CacheMode.default_off,
    )


def test_response_cache_redis_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("redis")
    pytest.importorskip("fastapi")
    kw: dict[str, Any] = {}

    def capture(**kwargs: Any) -> Any:
        kw.update(kwargs)
        return object()

    monkeypatch.setattr("swarm_arena.llm._cache.Cache", capture)
    monkeypatch.setenv("REDIS_URL", "rediss://user:pw@redis.example:6380/2")
    policy = CachePolicy(backend="redis", redis_url_env="REDIS_URL")
    response_cache(policy)
    assert (kw["type"], kw["url"]) == (
        "redis",
        "rediss://user:pw@redis.example:6380/2",
    )
    assert "host" not in kw
    monkeypatch.delenv("REDIS_URL")
    with pytest.raises(ValueError, match="REDIS_URL"):
        response_cache(policy)


@pytest.mark.parametrize("module", ["fastapi", "orjson", "diskcache", "boto3", "redis"])
def test_require_cache_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == module else object()
    )
    backend = {"boto3": "s3", "redis": "redis"}.get(module, "disk")
    policy = CachePolicy.model_validate(
        {"backend": backend, "s3_path": "s3://b/p", "redis_url_env": "REDIS_URL"}
    )
    with pytest.raises(ImportError, match=r"swarm-arena\[llm,cache\]") as caught:
        require_cache_extra(policy)
    assert str(caught.value) == (
        f"missing {module}; "
        'uv add "swarm-arena[llm,cache]"; see https://github.com/shreyasvinaya/swarm-arena#install'
    )


@pytest.mark.parametrize(
    "path", ["https://S3_SECRET/b", "s3:///S3_SECRET", "s3://S3_SECRET?query=value"]
)
def test_invalid_s3_paths_are_value_free(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("swarm_arena.llm._cache.require_cache_extra", lambda policy: None)
    with pytest.raises(ValueError, match="s3_path") as caught:
        response_cache(CachePolicy(backend="s3", s3_path=path))
    assert_secret_absent(caught.value, "S3_SECRET")


def test_cache_identity_requires_key_and_generating_response() -> None:
    kw = {"litellm_params": {"preset_cache_key": "A:key"}}
    first = cache_identity(kw, {"id": "response-1"})
    assert first is not None
    assert len(first) == 64
    assert first != cache_identity(
        {"litellm_params": {"preset_cache_key": "B:key"}}, {"id": "response-1"}
    )
    assert first != cache_identity(kw, {"id": "response-2"})
    assert cache_identity({}, {"id": "response-1"}) is None
    assert cache_identity(kw, {}) is None
    assert cache_identity({}, {"id": "response-1"}, effective_key="A:key") == first
    assert cache_identity(kw, {"id": "response-1"}, effective_key="B:key") is None
    assert (
        cache_identity(kw, {"id": "response-1", "_hidden_params": {"cache_key": "B:key"}}) is None
    )


async def test_cache_identity_feeds_recorder_and_replay(tmp_path: Path) -> None:
    config = RunConfig(tracing=TracingConfig(store_payloads=False))
    async with open_llm_run(tmp_path, config=config) as run:
        scope = run.scope("judge")
        originals: dict[str, str] = {}
        for namespace, response_id, hit in [
            ("A", "first", False),
            ("B", "first", False),
            ("B", "replacement", False),
            ("B", "replacement", True),
        ]:
            rid = await scope.tranche.reserve_call(Amount(tokens=100, usd=0.01))
            cid = open_call(
                scope,
                reservation_id=rid,
                estimate=Amount(tokens=100, usd=0.01),
                model_alias="opus",
                model="anthropic/claude",
                provider="anthropic",
            )
            identity = cache_identity(
                {"litellm_params": {"preset_cache_key": f"{namespace}:key"}}, {"id": response_id}
            )
            call = await record_call(
                **record_kwargs(
                    run.metadata(scope, reservation_id=rid, call_id=cid),
                    cache_identity=identity,
                    cache_hit=hit,
                )
            )
            assert call is not None
            if hit:
                assert call.cached_from_call_id == originals["B-replacement"]
            else:
                originals[f"{namespace}-{response_id}"] = cid
        await drain(scope, wait=True)
        await run.journal.commit()
        state = run.state()
        assert len(state.llm_cache_identity) == 4
        assert state.llm_request_sha256
        unbind_run(run.run_id)
        bind_run(
            run.run_id,
            journal=run.journal,
            ids=run.ids,
            artifacts=run.artifacts,
            tracing=config.tracing,
            committed=list(state.llm_calls.values()),
            request_sha256=state.llm_request_sha256,
            cache_identity=state.llm_cache_identity,
            stops=run.stops,
            clock=run.clock,
            limits=config.limits,
            settle_timeout_s=config.settle_timeout_s,
        )
        resumed = run.scope("judge")
        rid = await resumed.tranche.reserve_call(Amount(tokens=100, usd=0.01))
        cid = open_call(
            resumed,
            reservation_id=rid,
            estimate=Amount(tokens=100, usd=0.01),
            model_alias="opus",
            model="anthropic/claude",
            provider="anthropic",
        )
        call = await record_call(
            **record_kwargs(
                run.metadata(resumed, reservation_id=rid, call_id=cid),
                cache_hit=True,
                cache_identity=state.llm_cache_identity[originals["B-replacement"]],
            )
        )
        assert call is not None
        assert call.cached_from_call_id == originals["B-replacement"]
        await drain(resumed, wait=True)


async def test_real_cache_identity_tracks_namespaces_keys_and_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("diskcache")
    pytest.importorskip("fastapi")
    logger = _CaptureLogger()
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
    cache = response_cache(CachePolicy(backend="disk", disk_dir=str(tmp_path)))
    monkeypatch.setattr(litellm, "cache", cache)
    monkeypatch.setattr(litellm, "callbacks", [logger])
    identities: list[str | None] = []
    written = anyio.Event()
    written_keys: list[str] = []
    backend = cast(Any, cache.cache)
    write = backend.async_set_cache

    async def capture_write(key: str, value: Any, **kwargs: Any) -> None:
        await write(key, value, **kwargs)
        written_keys.append(key)
        written.set()

    monkeypatch.setattr(backend, "async_set_cache", capture_write)
    try:
        for namespace, no_read, seed, expected_hit in [
            ("A", False, 1, False),
            ("B", False, 1, False),
            ("B", False, 1, True),
            ("B", True, 1, False),
            ("B", False, 1, True),
            ("B", False, 2, False),
        ]:
            written = anyio.Event()
            await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="answer",
                seed=seed,
                cache={"use-cache": True, "no-cache": no_read, "namespace": namespace},
            )
            await logger.wait()
            kwargs, response = logger.success[-1]
            assert bool(kwargs["cache_hit"]) is expected_hit
            if expected_hit:
                identity = cache_identity(kwargs, response)
            else:
                with anyio.fail_after(60):
                    await written.wait()
                identity = cache_identity(kwargs, response, effective_key=written_keys[-1])
            identities.append(identity)
        assert all(identity is not None for identity in identities)
        assert identities[0] != identities[1]
        assert identities[1] == identities[2]
        assert identities[2] != identities[3]
        assert identities[3] == identities[4]
        assert identities[4] != identities[5]
    finally:
        with anyio.fail_after(60):
            await GLOBAL_LOGGING_WORKER.flush()
            await GLOBAL_LOGGING_WORKER.stop()
            await close_litellm_async_clients()
        backend.disk_cache.close()


def test_backend_construction_error_is_value_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**kwargs: Any) -> Any:
        raise ValueError("CACHE_BACKEND_SECRET")

    monkeypatch.setattr("swarm_arena.llm._cache.Cache", fail)
    monkeypatch.setattr("swarm_arena.llm._cache.require_cache_extra", lambda policy: None)
    with pytest.raises(RuntimeError, match=r"response cache construction.*ValueError") as caught:
        response_cache(CachePolicy(backend="disk", disk_dir=str(tmp_path / "cache")))
    assert_secret_absent(caught.value, "CACHE_BACKEND_SECRET")
