"""Offline wire-contract, accounting and abandonment tests for the proxy."""

import asyncio
import json
import os
import socket
import threading
import traceback
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import anyio
import pytest
from _fakes.gateway import ABSENT_FAILURES, FakeGateway

from swarm_arena._gateway.adapter import GatewayError, VirtualKey
from swarm_arena._gateway.litellm_proxy import LiteLLMProxy, adapter_for
from swarm_arena._model.settings import GatewayConfig

MASTER = "sk-master-sentinel"
START = datetime(2000, 1, 1, tzinfo=UTC)
END = datetime(2100, 1, 1, tzinfo=UTC)
CREATE: dict[str, Any] = {
    "alias": "cand-c-3f9a1c2b5d6e7f80-e1-0193",
    "max_budget_usd": 1.0,
    "duration_s": 86400.0,
    "metadata": {"candidate_id": "c-3f9a1c2b5d6e7f80-e1-0193"},
    "default_estimated_output_tokens": 4096,
}


def _proxy_at(url: str, http_timeout_s: float = 5.0) -> LiteLLMProxy:
    return LiteLLMProxy(
        url=url, master_key=MASTER, request_timeout_s=600, http_timeout_s=http_timeout_s
    )


def _proxy(gw: FakeGateway, http_timeout_s: float = 5.0) -> LiteLLMProxy:
    return _proxy_at(gw.url, http_timeout_s)


async def _proxy_and_key(
    gw: FakeGateway, http_timeout_s: float = 5.0
) -> tuple[LiteLLMProxy, VirtualKey]:
    proxy = _proxy(gw, http_timeout_s)
    key, _ = await proxy.create_key(**CREATE)
    return proxy, key


def _last(gw: FakeGateway, path: str) -> dict[str, Any]:
    return [body for _, route, body in gw.requests if route == path][-1]


def _secret(gw: FakeGateway) -> str:
    return next(iter(gw.keys.values()))["secret"]


async def _rows(proxy: LiteLLMProxy, key: VirtualKey) -> list[Any]:
    return await proxy.spend_rows(key, start=START, end=END)


def _chat(gw: FakeGateway, secret: str, model: str = "openai/gpt-6") -> tuple[Any, dict[str, Any]]:
    req = Request(
        gw.url + "/v1/chat/completions",
        data=json.dumps({"model": model}).encode(),
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
    )
    with urlopen(req, timeout=5) as response:
        return response.headers, json.loads(response.read())


async def _wait(event: threading.Event) -> None:
    with anyio.fail_after(5):
        assert await anyio.to_thread.run_sync(event.wait, 4)


async def test_preflight_strict_verified(fake_gateway: FakeGateway) -> None:
    report = await _proxy(fake_gateway).preflight()
    assert (report.verified, report.failures) == (
        False,
        (*ABSENT_FAILURES, "gateway settings introspection unavailable"),
    )
    assert report.request_timeout_s is report.batch_write_interval_s is None
    assert [p for _, p, _ in fake_gateway.requests] == ["/health/liveliness", "/model/info"]


async def test_preflight_not_strict_lists_failures() -> None:
    with FakeGateway(strict=False) as gw:
        assert (await _proxy(gw).preflight()).failures


async def test_create_key_sends_alias_budget_duration_metadata(fake_gateway: FakeGateway) -> None:
    key, secret = await _proxy(fake_gateway).create_key(**CREATE)
    body = _last(fake_gateway, "/key/generate")
    assert body == {
        "key_alias": CREATE["alias"],
        "max_budget": 1.0,
        "duration": "86400s",
        "metadata": CREATE["metadata"],
        "default_estimated_output_tokens": 4096,
    }
    assert key.alias == CREATE["alias"]
    assert secret.startswith("sk-")
    assert key.key_id != secret
    assert secret not in key.model_dump_json()
    assert fake_gateway.keys[key.key_id]["metadata"]["default_estimated_output_tokens"] == 4096


async def test_block_then_closed_at(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    assert await proxy.closed_at(key) is None
    await proxy.block_key(key)
    assert (await proxy.closed_at(key)) is not None
    assert _last(fake_gateway, "/key/block") == {"key": key.key_id}


async def test_closed_at_deleted_key(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    await proxy.delete_key(key)
    assert (await proxy.closed_at(key)) is not None
    assert _last(fake_gateway, "/key/delete") == {"keys": [key.key_id]}
    fake_gateway.keys.clear()
    assert (await proxy.closed_at(key)) is not None


async def test_spend_rows_all_pages(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    for i in range(130):
        fake_gateway.add_spend(
            key.key_id, model="openai/gpt-6", cost_usd=0.01, tokens=(10, 2), call_id=f"id-{i}"
        )
    rows = await _rows(proxy, key)
    assert len(rows) == 130
    assert rows[0].usage.input_tokens == 10
    assert rows[0].cost_usd == 0.01
    assert len({row.request_id for row in rows}) == 130
    queries = [b for _, p, b in fake_gateway.requests if p == "/spend/logs/v2"]
    assert [q["page"] for q in queries] == ["1", "2", "3"]
    assert all(
        q["api_key"] == key.key_id
        and q["start_date"] == "2000-01-01 00:00:00"
        and q["end_date"] == "2100-01-01 00:00:00"
        for q in queries
    )
    await proxy.delete_key(key)
    assert len(await _rows(proxy, key)) == 130


async def test_spend_dates_round_outward_and_include_boundaries(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    instant = datetime(2026, 10, 23, 12, 0, 0, 123456, tzinfo=UTC)
    for i, delta in enumerate((-1, 0, 1, 2)):
        fake_gateway.add_spend(
            key.key_id,
            model="m",
            cost_usd=0.01,
            tokens=(1, 2),
            call_id=str(i),
            ts=instant.replace(microsecond=0) + timedelta(seconds=delta),
        )
    rows = await proxy.spend_rows(key, start=instant, end=instant)
    assert [r.litellm_call_id for r in rows] == ["1", "2"]
    assert _last(fake_gateway, "/spend/logs/v2")["end_date"] == "2026-10-23 12:00:01"


async def test_cache_and_nullable_identity(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    for call_id in (None, "shared", "shared"):
        fake_gateway.add_spend(
            key.key_id,
            model="m",
            cost_usd=0.1,
            tokens=(10, 2),
            call_id=call_id,
            metadata={
                "additional_usage_values": {
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 2,
                }
            },
        )
    rows = await _rows(proxy, key)
    assert [r.litellm_call_id for r in rows] == [None, "shared", "shared"]
    assert len({r.request_id for r in rows}) == 3
    assert rows[0].usage.input_tokens == 5
    assert rows[0].usage.cache_read_tokens == 3
    assert rows[0].usage.cache_write_tokens == 2
    assert rows[0].usage.total_tokens == 12


async def test_errors_never_contain_secrets(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.fail["/spend/logs/v2"] = 500
    with pytest.raises(GatewayError, match="GET /spend/logs/v2: HTTP 500") as err:
        await _rows(proxy, key)
    rendered = "".join(traceback.format_exception(err.value)) + repr(err.value.args)
    assert MASTER not in rendered
    assert _secret(fake_gateway) not in rendered


PATHS = {
    "create_key": "/key/generate",
    "block_key": "/key/block",
    "spend_rows": "/spend/logs/v2",
    "delete_key": "/key/delete",
    "closed_at": "/key/info",
    "preflight": "/health/liveliness",
}


async def _call(proxy: LiteLLMProxy, call: str, key: VirtualKey) -> Any:
    if call == "create_key":
        return await proxy.create_key(**CREATE)
    if call == "spend_rows":
        return await _rows(proxy, key)
    if call == "preflight":
        return await proxy.preflight()
    return await getattr(proxy, call)(key)


@pytest.mark.parametrize("call", list(PATHS))
@pytest.mark.parametrize("fault", ["refused", "slow", "malformed"])
async def test_transport_faults_raise_gateway_error(
    fake_gateway: FakeGateway, fault: str, call: str
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    path = PATHS[call]
    if fault == "refused":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        proxy = _proxy_at(f"http://127.0.0.1:{port}", http_timeout_s=0.2)
    elif fault == "slow":
        fake_gateway.hold[path] = threading.Event()
        proxy = _proxy(fake_gateway, http_timeout_s=0.2)
    else:
        fake_gateway.malformed.add(path)
        if call in {"block_key", "delete_key"}:
            # A refused malformed reply has no side effect to confirm via info.
            fake_gateway.responses[path] = {}
    try:
        with pytest.raises(GatewayError) as err:
            await _call(proxy, call, key)
        rendered = "".join(traceback.format_exception(err.value))
        expected_paths = (path, "/key/info") if call in {"block_key", "delete_key"} else (path,)
        assert any(route in str(err.value) for route in expected_paths)
        assert MASTER not in rendered
        assert _secret(fake_gateway) not in rendered
    finally:
        if path in fake_gateway.hold:
            fake_gateway.hold[path].set()


async def test_fake_chat_returns_spend_call_id(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    headers, body = await anyio.to_thread.run_sync(_chat, fake_gateway, _secret(fake_gateway))
    rows = await _rows(proxy, key)
    assert headers["x-litellm-call-id"] == body["id"] == rows[-1].litellm_call_id
    await proxy.block_key(key)
    with pytest.raises(HTTPError) as err:
        await anyio.to_thread.run_sync(_chat, fake_gateway, _secret(fake_gateway))
    assert err.value.code == 401


def test_adapter_for_missing_master_key_env() -> None:
    with pytest.raises(ValueError, match="set LITELLM_MASTER_KEY for the LLM gateway"):
        adapter_for(GatewayConfig(url="http://gw:4000"), {})


async def test_adapter_for_config(fake_gateway: FakeGateway) -> None:
    proxy = adapter_for(GatewayConfig(url=fake_gateway.url), {"LITELLM_MASTER_KEY": MASTER})
    assert proxy.terminal_accounting is False
    assert not (await proxy.preflight()).verified


@pytest.mark.parametrize(
    "changes",
    [
        {"total_is_capped": True},
        {"total": 2},
        {"page": 2},
        {"page_size": 1},
        {"total_pages": 2},
        {"data": []},
        {"total": -1},
        {"total_is_capped": "false"},
    ],
)
async def test_incomplete_pages_rejected(
    fake_gateway: FakeGateway, changes: dict[str, Any]
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(key.key_id, model="m", cost_usd=1, tokens=(1, 2), call_id="id")
    fake_gateway.page_overrides[1] = changes
    with pytest.raises(GatewayError):
        await _rows(proxy, key)


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": ""},
        {"request_id": None},
        {"spend": -1},
        {"spend": float("nan")},
        {"prompt_tokens": -1},
        {"completion_tokens": True},
        {"startTime": "sentinel-secret"},
        {"startTime": "2026-10-23T00:00:00"},
    ],
)
async def test_malformed_spend_is_safe(fake_gateway: FakeGateway, changes: dict[str, Any]) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(key.key_id, model="m", cost_usd=1, tokens=(1, 2), call_id="id")
    fake_gateway.row_overrides = changes
    with pytest.raises(GatewayError) as err:
        await _rows(proxy, key)
    assert "sentinel-secret" not in "".join(traceback.format_exception(err.value))


async def test_cancelled_create_blocks_late_key(fake_gateway: FakeGateway) -> None:
    entered, release, blocked = threading.Event(), threading.Event(), threading.Event()
    fake_gateway.entered["/key/generate"] = entered
    fake_gateway.hold["/key/generate"] = release
    fake_gateway.entered["/key/block"] = blocked
    task = asyncio.create_task(_proxy(fake_gateway).create_key(**CREATE))
    try:
        await _wait(entered)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await _wait(blocked)
        assert len(fake_gateway.keys) == 1
        assert len([p for _, p, _ in fake_gateway.requests if p == "/key/generate"]) == 1
    finally:
        release.set()
        if not task.done():
            task.cancel()


async def test_cancelled_read_starts_no_later_page(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    import swarm_arena._gateway.litellm_proxy as module

    done = threading.Event()
    original = module.run_sync_daemon

    async def tracked(work: Any, operation: Any) -> Any:
        def wrapped() -> Any:
            try:
                return work(operation)
            finally:
                done.set()

        return await original(wrapped)

    proxy, key = await _proxy_and_key(fake_gateway)
    for i in range(51):
        fake_gateway.add_spend(key.key_id, model="m", cost_usd=1, tokens=(1, 2), call_id=str(i))
    entered, release = threading.Event(), threading.Event()
    fake_gateway.entered["/spend/logs/v2"] = entered
    fake_gateway.hold["/spend/logs/v2"] = release
    monkeypatch.setattr(module, "run_sync_daemon", tracked)
    task = asyncio.create_task(_rows(proxy, key))
    try:
        await _wait(entered)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    await _wait(done)
    assert len([p for _, p, _ in fake_gateway.requests if p == "/spend/logs/v2"]) == 1


@pytest.mark.gateway_live
async def test_live_gateway_profile_and_key_cycle() -> None:
    proxy = adapter_for(GatewayConfig(url=os.environ["SWARM_ARENA_LIVE_GATEWAY_URL"]), os.environ)
    assert isinstance((await proxy.preflight()).verified, bool)
    key, _ = await proxy.create_key(
        alias=f"cand-live-{uuid4().hex[:8]}",
        max_budget_usd=0.01,
        duration_s=600.0,
        metadata={},
        default_estimated_output_tokens=256,
    )
    try:
        await proxy.block_key(key)
        assert await proxy.closed_at(key) is not None
    finally:
        await proxy.delete_key(key)


@pytest.mark.parametrize(
    "payload", [None, [], {}, {"info": {}}, {"info": {"blocked": "false", "status": "active"}}]
)
async def test_info_malformed_is_not_closure(fake_gateway: FakeGateway, payload: Any) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.responses["/key/info"] = payload
    with pytest.raises(GatewayError):
        await proxy.closed_at(key)


@pytest.mark.parametrize(
    "changes",
    [
        {"duration_s": 0},
        {"duration_s": -1},
        {"duration_s": float("inf")},
        {"max_budget_usd": -1},
        {"max_budget_usd": True},
        {"default_estimated_output_tokens": 0},
        {"default_estimated_output_tokens": True},
        {"metadata": {"secret-sentinel": float("nan")}},
        {"alias": ""},
    ],
)
async def test_bad_create_input_does_not_dispatch(
    fake_gateway: FakeGateway, changes: dict[str, Any]
) -> None:
    with pytest.raises(GatewayError):
        await _proxy(fake_gateway).create_key(**{**CREATE, **changes})
    assert not fake_gateway.requests


async def test_body_and_operation_limits(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    import swarm_arena._gateway.litellm_proxy as module

    proxy, key = await _proxy_and_key(fake_gateway)
    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 8)
    with pytest.raises(GatewayError):
        await proxy.closed_at(key)
    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 4 * 1024 * 1024)
    monkeypatch.setattr(module, "_MAX_OPERATION_BYTES", 20)
    with pytest.raises(GatewayError):
        await proxy.preflight()


@pytest.mark.parametrize("limit", ["_MAX_ROWS", "_MAX_PAGES"])
async def test_pagination_resource_limits(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    import swarm_arena._gateway.litellm_proxy as module

    proxy, key = await _proxy_and_key(fake_gateway)
    for i in range(51):
        fake_gateway.add_spend(key.key_id, model="m", cost_usd=1, tokens=(1, 2), call_id=str(i))
    monkeypatch.setattr(module, limit, 1)
    with pytest.raises(GatewayError):
        await _rows(proxy, key)
    assert len([p for _, p, _ in fake_gateway.requests if p == "/spend/logs/v2"]) == 1


async def test_pagination_one_deadline(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    import swarm_arena._gateway.litellm_proxy as module

    proxy, key = await _proxy_and_key(fake_gateway)
    for i in range(51):
        fake_gateway.add_spend(key.key_id, model="m", cost_usd=1, tokens=(1, 2), call_id=str(i))
    original = proxy._request
    elapsed = 0.0
    real_time = module.time.monotonic

    def request(*args: Any, **kwargs: Any) -> Any:
        nonlocal elapsed
        result = original(*args, **kwargs)
        elapsed += 6
        return result

    # Patch only this module's clock reference, never time.monotonic process-wide.
    from types import SimpleNamespace

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: real_time() + elapsed))
    monkeypatch.setattr(proxy, "_request", request)
    with pytest.raises(GatewayError, match="TimeoutError"):
        await _rows(proxy, key)
    assert len([p for _, p, _ in fake_gateway.requests if p == "/spend/logs/v2"]) == 1


async def test_delayed_visibility_without_sleep(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.chat_visible_after_s = 3600
    await anyio.to_thread.run_sync(_chat, fake_gateway, _secret(fake_gateway))
    assert await _rows(proxy, key) == []
    fake_gateway.reveal_spend()
    assert len(await _rows(proxy, key)) == 1


@pytest.mark.parametrize("order", ["before_publish", "after_publish"])
@pytest.mark.parametrize("termination", ["cancel", "timeout"])
async def test_create_delivery_race(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch, order: str, termination: str
) -> None:
    import swarm_arena._gateway.litellm_proxy as module

    entered, release, returned, blocked = (threading.Event() for _ in range(4))
    original = module._CreateOwner.publish

    def publish(owner: Any, key: VirtualKey) -> None:
        if order == "after_publish":
            original(owner, key)
        entered.set()
        assert release.wait(5)
        if order == "before_publish":
            original(owner, key)
        returned.set()

    monkeypatch.setattr(module._CreateOwner, "publish", publish)
    fake_gateway.entered["/key/block"] = blocked
    task = asyncio.create_task(_proxy(fake_gateway, http_timeout_s=2).create_key(**CREATE))
    try:
        await _wait(entered)
        if termination == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(GatewayError, match="TimeoutError"):
                await task
        release.set()
        await _wait(returned)
        await _wait(blocked)
        assert len([p for _, p, _ in fake_gateway.requests if p == "/key/block"]) == 1
        assert all(row["blocked"] for row in fake_gateway.keys.values())
    finally:
        release.set()
        if not task.done():
            task.cancel()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("signal", ["fenced", "terminal", "lookalike"])
async def test_control_authority_preserved(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch, signal: str, nested: bool
) -> None:
    from swarm_arena._errors import Fenced, _core_error, _is_core_control, _SignalCategory

    if signal == "fenced":
        error = _core_error(Fenced, "safe")
    elif signal == "terminal":
        error = _core_error(RuntimeError, "safe", _category=_SignalCategory.TERMINAL)
    else:
        error = Fenced("sentinel-secret")
    raised = BaseExceptionGroup("sentinel-secret", [error]) if nested else error
    proxy = _proxy(fake_gateway)

    def request(*args: Any, **kwargs: Any) -> Any:
        raise raised

    monkeypatch.setattr(proxy, "_request", request)
    with pytest.raises(Exception, match=r"safe|ownership|GET") as caught:
        await proxy.preflight()
    if signal == "lookalike":
        assert type(caught.value) is GatewayError
    else:
        assert _is_core_control(caught.value)
        assert not isinstance(caught.value, GatewayError)
    assert "sentinel-secret" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_timeout_validation(bad: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        _proxy_at("http://127.0.0.1:4000", http_timeout_s=bad)


async def test_no_fictional_config_endpoint(fake_gateway: FakeGateway) -> None:
    req = Request(fake_gateway.url + "/config/yaml", headers={"Authorization": f"Bearer {MASTER}"})
    with pytest.raises(HTTPError) as caught:
        await anyio.to_thread.run_sync(urlopen, req)
    assert caught.value.code == 404


async def test_redirect_never_receives_master(fake_gateway: FakeGateway) -> None:
    with FakeGateway() as target:
        fake_gateway.fail["/health/liveliness"] = 302
        fake_gateway.response_headers["/health/liveliness"] = {
            "Location": target.url + "/health/liveliness"
        }
        with pytest.raises(GatewayError, match="HTTP 302"):
            await _proxy(fake_gateway).preflight()
        assert not target.requests


@pytest.mark.parametrize("error", [OSError, ConnectionError, TimeoutError, ValueError])
async def test_safe_transport_errors(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    from _fakes.secrecy import assert_secret_absent

    proxy = _proxy(fake_gateway)

    def request(*args: Any, **kwargs: Any) -> Any:
        raise error(MASTER)

    monkeypatch.setattr(proxy, "_request", request)
    with pytest.raises(GatewayError, match=f"GET /health/liveliness: {error.__name__}") as caught:
        await proxy.preflight()
    assert_secret_absent(caught.value, MASTER)


async def test_http_exception_safe(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    from http.client import HTTPException

    from _fakes.secrecy import assert_secret_absent

    proxy = _proxy(fake_gateway)

    def request(*args: Any, **kwargs: Any) -> Any:
        raise HTTPException(MASTER)

    monkeypatch.setattr(proxy, "_request", request)
    with pytest.raises(GatewayError, match="HTTPException") as caught:
        await proxy.preflight()
    assert_secret_absent(caught.value, MASTER)


async def test_management_bearer_and_http_failure(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    await proxy.preflight()
    await proxy.closed_at(key)
    await _rows(proxy, key)
    await proxy.block_key(key)
    await proxy.delete_key(key)
    assert set(fake_gateway.authorizations) == {f"Bearer {MASTER}"}
    fake_gateway.fail["/key/info"] = 403
    with pytest.raises(GatewayError, match="HTTP 403"):
        await proxy.closed_at(key)


async def test_anyio_cancellation_propagates(fake_gateway: FakeGateway) -> None:
    entered, release = threading.Event(), threading.Event()
    fake_gateway.entered["/model/info"] = entered
    fake_gateway.hold["/model/info"] = release
    propagated = False

    async def invoke() -> None:
        nonlocal propagated
        try:
            await _proxy(fake_gateway).preflight()
        except anyio.get_cancelled_exc_class():
            propagated = True
            raise

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(invoke)
            await _wait(entered)
            group.cancel_scope.cancel()
    finally:
        release.set()
    assert propagated


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (START.replace(tzinfo=None), END),
        (START, END.replace(tzinfo=None)),
        (END, START),
    ],
)
async def test_invalid_spend_window_is_safe(
    fake_gateway: FakeGateway, start: datetime, end: datetime
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    with pytest.raises(GatewayError, match="GET /spend/logs/v2"):
        await proxy.spend_rows(key, start=start, end=end)
    assert not [p for _, p, _ in fake_gateway.requests if p == "/spend/logs/v2"]


async def test_second_page_changed_counts_rejected(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    for i in range(51):
        fake_gateway.add_spend(key.key_id, model="m", cost_usd=1, tokens=(1, 2), call_id=str(i))
    fake_gateway.page_overrides[2] = {"total": 52}
    with pytest.raises(GatewayError):
        await _rows(proxy, key)


async def test_abandoned_block_owner_is_bounded(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    proxy = _proxy(fake_gateway, http_timeout_s=0.2)
    entered, release, owner_done, worker_done = (threading.Event() for _ in range(4))

    def request(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        try:
            assert release.wait(5)
            return {"token": key.key_id, "blocked": True}
        finally:
            worker_done.set()

    def cleanup() -> None:
        try:
            proxy._block_abandoned(key)
        finally:
            owner_done.set()

    monkeypatch.setattr(proxy, "_request", request)
    thread = threading.Thread(target=cleanup, daemon=True)
    thread.start()
    try:
        await _wait(entered)
        await _wait(owner_done)
        assert not release.is_set()
    finally:
        release.set()
        await _wait(worker_done)
        thread.join(timeout=5)
        assert not thread.is_alive()


async def test_spend_retains_own_hash_and_one_cost(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(key.key_id, model="m", cost_usd=0.125, tokens=(10, 2), call_id=None)
    (row,) = await _rows(proxy, key)
    assert row.key_id == key.key_id
    assert row.cost_usd == 0.125
    assert row.usage.usd is None


@pytest.mark.parametrize("hash_value", [None, "", " ", 123, "foreign-key-hash"])
async def test_spend_hash_mismatch_is_safe(fake_gateway: FakeGateway, hash_value: Any) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(key.key_id, model="m", cost_usd=0.1, tokens=(1, 2), call_id=None)
    fake_gateway.row_overrides = {"api_key": hash_value}
    with pytest.raises(GatewayError, match="GET /spend/logs/v2"):
        await _rows(proxy, key)


@pytest.mark.parametrize("timestamp", [1_729_000_000, 1_729_000_000.0, "1729000000", True])
async def test_spend_rejects_epoch_timestamps(fake_gateway: FakeGateway, timestamp: Any) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(key.key_id, model="m", cost_usd=0.1, tokens=(1, 2), call_id=None)
    fake_gateway.row_overrides = {"startTime": timestamp}
    with pytest.raises(GatewayError):
        await _rows(proxy, key)


@pytest.mark.parametrize("alias", [" ", "x" * 257])
async def test_create_identity_validated_before_dispatch(
    fake_gateway: FakeGateway, alias: str
) -> None:
    arguments: dict[str, Any] = {**CREATE, "alias": alias}
    with pytest.raises(GatewayError):
        await _proxy(fake_gateway).create_key(**arguments)
    assert not fake_gateway.requests


async def test_create_accepts_contract_hash_without_hex_grammar(fake_gateway: FakeGateway) -> None:
    fake_gateway.responses["/key/generate"] = {"token_id": "durable-hash", "key": "sk-ephemeral"}
    key, secret = await _proxy(fake_gateway).create_key(**CREATE)
    assert key.key_id == "durable-hash"
    assert secret == "sk-ephemeral"


async def test_block_delete_idempotent_outcomes(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    assert await proxy.block_key(key) is None
    assert await proxy.block_key(key) is None
    assert await proxy.delete_key(key) is None
    assert fake_gateway.keys[key.key_id]["deleted"]
    assert await proxy.delete_key(key) is None
    assert await proxy.block_key(key) is None
    fake_gateway.keys.clear()
    assert await proxy.block_key(key) is None
    assert await proxy.delete_key(key) is None
    closed = await proxy.closed_at(key)
    assert closed is not None
    assert closed.tzinfo is UTC


@pytest.mark.parametrize("operation", ["block_key", "delete_key"])
async def test_mutation_404_requires_confirmed_outcome(
    fake_gateway: FakeGateway, operation: str
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.fail[PATHS[operation]] = 404
    with pytest.raises(GatewayError, match=f"POST {PATHS[operation]}: HTTP 404"):
        await _call(proxy, operation, key)
    fake_gateway.keys[key.key_id]["deleted"] = True
    assert await _call(proxy, operation, key) is None


async def test_delete_404_does_not_treat_blocked_as_deleted(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    await proxy.block_key(key)
    fake_gateway.fail["/key/delete"] = 404
    with pytest.raises(GatewayError, match="POST /key/delete: HTTP 404"):
        await proxy.delete_key(key)


async def test_idempotency_inspection_failure_is_not_success(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.fail["/key/block"] = 404
    fake_gateway.fail["/key/info"] = 500
    with pytest.raises(GatewayError, match="GET /key/info: HTTP 500"):
        await proxy.block_key(key)


@pytest.mark.parametrize("operation", ["block_key", "delete_key"])
@pytest.mark.parametrize("status", [403, 404, 500])
async def test_failed_mutation_checks_info(
    fake_gateway: FakeGateway, operation: str, status: int
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.fail[PATHS[operation]] = status
    with pytest.raises(GatewayError):
        await _call(proxy, operation, key)
    assert _last(fake_gateway, "/key/info") == {"key": key.key_id}
    fake_gateway.keys[key.key_id]["deleted"] = True
    assert await _call(proxy, operation, key) is None


@pytest.mark.parametrize("operation", ["block_key", "delete_key"])
async def test_wrong_mutation_path_404_is_not_key_absence(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    original = proxy._request

    def request(op: Any, method: str, path: str, **kwargs: Any) -> Any:
        if path == PATHS[operation]:
            path = "/wrong" + path
        return original(op, method, path, **kwargs)

    monkeypatch.setattr(proxy, "_request", request)
    with pytest.raises(GatewayError, match="HTTP 404"):
        await _call(proxy, operation, key)
    assert fake_gateway.requests[-2][1] == "/wrong" + PATHS[operation]
    assert fake_gateway.requests[-1][1] == "/key/info"
    assert not fake_gateway.keys[key.key_id]["deleted"]
    assert not fake_gateway.keys[key.key_id]["blocked"]


@pytest.mark.parametrize("operation", ["block_key", "delete_key"])
async def test_applied_mutation_with_bad_reply_uses_info(
    fake_gateway: FakeGateway, operation: str
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.malformed.add(PATHS[operation])
    assert await _call(proxy, operation, key) is None
    assert _last(fake_gateway, "/key/info") == {"key": key.key_id}


async def test_idempotency_inspection_uses_remaining_deadline(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import swarm_arena._gateway.litellm_proxy as module

    proxy, key = await _proxy_and_key(fake_gateway)
    original = proxy._request
    real_time = module.time.monotonic
    elapsed = 0.0
    fake_gateway.fail["/key/block"] = 404

    def request(*args: Any, **kwargs: Any) -> Any:
        nonlocal elapsed
        try:
            return original(*args, **kwargs)
        finally:
            elapsed += 6

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: real_time() + elapsed))
    monkeypatch.setattr(proxy, "_request", request)
    with pytest.raises(GatewayError, match="TimeoutError"):
        await proxy.block_key(key)
    assert not [p for _, p, _ in fake_gateway.requests if p == "/key/info"]


async def test_spend_missing_hash_rejected(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.page_overrides[1] = {
        "total": 1,
        "total_pages": 1,
        "data": [
            {
                "request_id": "r",
                "litellm_call_id": None,
                "model": "m",
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "spend": 0.1,
                "startTime": "2026-10-23T00:00:00Z",
                "metadata": {},
            }
        ],
    }
    with pytest.raises(GatewayError):
        await _rows(proxy, key)


async def test_spend_iso_offset_normalized(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(key.key_id, model="m", cost_usd=1, tokens=(1, 2), call_id=None)
    fake_gateway.row_overrides = {"startTime": "2026-10-23T12:00:00+05:30"}
    (row,) = await _rows(proxy, key)
    assert row.ts == datetime(2026, 10, 23, 6, 30, tzinfo=UTC)
    assert row.ts.tzinfo is UTC


@pytest.mark.parametrize("nested", [False, True])
async def test_mutation_control_never_becomes_idempotent_success(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    from swarm_arena._errors import _core_error, _is_core_control, _SignalCategory

    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.keys[key.key_id]["blocked"] = True
    original = proxy._request
    terminal = _core_error(ValueError, "safe", _category=_SignalCategory.TERMINAL)
    error = ExceptionGroup("sentinel-secret", [terminal]) if nested else terminal

    def request(op: Any, method: str, path: str, **kwargs: Any) -> Any:
        if path == "/key/block":
            raise error
        return original(op, method, path, **kwargs)

    monkeypatch.setattr(proxy, "_request", request)
    with pytest.raises(ValueError, match="safe") as caught:
        await proxy.block_key(key)
    assert _is_core_control(caught.value)
    assert not [p for _, p, _ in fake_gateway.requests if p == "/key/info"]


@pytest.mark.parametrize("operation", ["block_key", "delete_key", "closed_at"])
async def test_all_404_wrong_prefix_is_unknown(fake_gateway: FakeGateway, operation: str) -> None:
    _, key = await _proxy_and_key(fake_gateway)
    proxy = _proxy_at(fake_gateway.url + "/wrong-prefix")
    with pytest.raises(GatewayError):
        await _call(proxy, operation, key)
    assert not fake_gateway.keys[key.key_id]["blocked"]
    assert not fake_gateway.keys[key.key_id]["deleted"]


@pytest.mark.parametrize("operation", ["block_key", "delete_key", "closed_at"])
@pytest.mark.parametrize(
    "raw",
    [
        b'{"detail":"Not Found"}',
        b"<html>default backend - 404</html>",
        b"",
        b"{not json",
        b'{"error":{"type":"not_found_error","param":"other"}}',
        b'{"error":{"type":"other","param":"key"}}',
    ],
)
async def test_ingress_404_is_unknown(
    fake_gateway: FakeGateway, operation: str, raw: bytes
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.fail["/key/info"] = 404
    fake_gateway.raw_responses["/key/info"] = raw
    if operation != "closed_at":
        fake_gateway.fail[PATHS[operation]] = 404
    with pytest.raises(GatewayError) as caught:
        await _call(proxy, operation, key)
    assert MASTER not in str(caught.value)
    assert key.key_id not in str(caught.value)


async def test_missing_key_body_obeys_byte_limit(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    import swarm_arena._gateway.litellm_proxy as module

    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.keys.clear()
    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 8)
    with pytest.raises(GatewayError):
        await proxy.closed_at(key)


async def test_environment_proxy_receives_no_master(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    with FakeGateway() as trap:
        for name in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            monkeypatch.setenv(name, trap.url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(name, "")
        assert await proxy.closed_at(key) is None
        assert not trap.requests
        assert not trap.authorizations
        assert _last(fake_gateway, "/key/info") == {"key": key.key_id}


@pytest.mark.parametrize("status", ["expired", "revoked"])
async def test_expired_and_revoked_keys_are_closed(fake_gateway: FakeGateway, status: str) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.keys[key.key_id]["status"] = status
    assert await proxy.closed_at(key) is not None
    with pytest.raises(HTTPError) as caught:
        await anyio.to_thread.run_sync(_chat, fake_gateway, _secret(fake_gateway))
    assert caught.value.code == 401
    # A closure observation does not turn failed block/delete into success.
    for operation in ("block_key", "delete_key"):
        fake_gateway.fail[PATHS[operation]] = 500
        with pytest.raises(GatewayError):
            await _call(proxy, operation, key)


@pytest.mark.parametrize("operation", ["block_key", "delete_key", "closed_at"])
@pytest.mark.parametrize("echo", [None, "another-key"])
async def test_key_info_identity_mismatch_is_unknown(
    fake_gateway: FakeGateway, operation: str, echo: str | None
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    payload: dict[str, Any] = {"info": {"status": "deleted", "blocked": True}}
    if echo is not None:
        payload["key"] = echo
    fake_gateway.responses["/key/info"] = payload
    if operation != "closed_at":
        fake_gateway.fail[PATHS[operation]] = 500
    with pytest.raises(GatewayError):
        await _call(proxy, operation, key)


@pytest.mark.parametrize("token_id", ["sk-secret-sentinel", "prefix-sk-secret-sentinel-suffix"])
async def test_create_bearer_cannot_be_durable_identity(
    fake_gateway: FakeGateway, token_id: str
) -> None:
    fake_gateway.responses["/key/generate"] = {"key": "sk-secret-sentinel", "token_id": token_id}
    with pytest.raises(GatewayError) as caught:
        await _proxy(fake_gateway).create_key(**CREATE)
    assert "sk-secret-sentinel" not in "".join(traceback.format_exception(caught.value))
    assert [p for _, p, _ in fake_gateway.requests] == ["/key/generate"]


async def test_block_reply_requires_matching_identity(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.responses["/key/block"] = {"blocked": True, "token": "another-key"}
    with pytest.raises(GatewayError):
        await proxy.block_key(key)


async def test_spend_metadata_allowlist(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(
        key.key_id,
        model="m",
        cost_usd=0.1,
        tokens=(10, 2),
        call_id=None,
        metadata={
            "spend_logs_metadata": {"candidate_id": "candidate", "secret": MASTER},
            "additional_usage_values": {
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "secret": MASTER,
            },
            "proxy_server_request": {"headers": {"Authorization": MASTER}, "messages": [MASTER]},
            "requester_ip_address": "192.0.2.1",
            "user_agent": MASTER,
        },
    )
    (row,) = await _rows(proxy, key)
    assert row.metadata == {
        "spend_logs_metadata": {"candidate_id": "candidate"},
        "additional_usage_values": {"cache_read_input_tokens": 3, "cache_creation_input_tokens": 2},
    }
    assert MASTER not in row.model_dump_json()
    assert row.usage.input_tokens == 5


@pytest.mark.parametrize("candidate", [7, ["candidate"], "c" * 257])
async def test_invalid_spend_attribution_keeps_row(
    fake_gateway: FakeGateway, candidate: Any
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(
        key.key_id,
        model="m",
        cost_usd=0.1,
        tokens=(1, 2),
        call_id=None,
        metadata={"spend_logs_metadata": {"candidate_id": candidate}},
    )
    (row,) = await _rows(proxy, key)
    assert row.key_id == key.key_id
    assert row.cost_usd == 0.1
    assert row.metadata == {}


@pytest.mark.parametrize("across_pages", [False, True])
@pytest.mark.parametrize("conflict", [False, True])
async def test_duplicate_request_ids_within_read(
    fake_gateway: FakeGateway, across_pages: bool, conflict: bool
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    count = 51 if across_pages else 2
    for i in range(count):
        fake_gateway.add_spend(key.key_id, model="m", cost_usd=0.1, tokens=(1, 2), call_id=str(i))
    # Repeat the first exact wire row at the end, including across a page boundary.
    first = dict(fake_gateway._spend[0][1])
    if conflict:
        first["spend"] = 5.0
    fake_gateway._spend[-1] = (0.0, first)
    if conflict or across_pages:
        with pytest.raises(GatewayError) as caught:
            await _rows(proxy, key)
        assert first["request_id"] not in str(caught.value)
    else:
        rows = await _rows(proxy, key)
        assert len(rows) == count - 1
        assert len({r.request_id for r in rows}) == count - 1
        # Dedup state must not survive the read (Task 16 owns that state).
        assert await _rows(proxy, key) == rows


async def test_header_drip_worker_exits_at_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import swarm_arena._gateway.litellm_proxy as module

    entered, release, finished = (threading.Event() for _ in range(3))
    sockets: list[socket.socket] = []
    original_track = module._Transport.track

    def track(transport: Any, sock: socket.socket) -> socket.socket:
        sockets.append(sock)
        return original_track(transport, sock)

    monkeypatch.setattr(module._Transport, "track", track)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            try:
                self.wfile.write(b"HTTP/1.1 200 OK\r\n")
                entered.set()
                while not release.wait(0.05):
                    self.wfile.write(b"X-Drip: still-here\r\n")
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    server_thread.start()
    proxy = _proxy_at(f"http://127.0.0.1:{server.server_port}", http_timeout_s=0.5)
    original = proxy._request

    def request(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(proxy, "_request", request)
    task = asyncio.create_task(proxy.closed_at(VirtualKey(key_id="hash", alias="alias")))
    try:
        await _wait(entered)
        with anyio.fail_after(1.5):
            with pytest.raises(GatewayError):
                await task
        # The caller is already bounded on the old code. Assert worker exit too,
        # before the fixture releases the endless header stream.
        with anyio.fail_after(1.5):
            assert await anyio.to_thread.run_sync(finished.wait, 1.0)
        assert sockets
        assert all(sock.fileno() == -1 for sock in sockets)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(5)
        await _wait(finished)
        assert not server_thread.is_alive()


@pytest.mark.parametrize("status", [200, 404])
async def test_body_drip_worker_exits_at_deadline(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    entered, release, finished = (threading.Event() for _ in range(3))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Length", "1000000")
                self.end_headers()
                entered.set()
                while not release.wait(0.05):
                    self.wfile.write(b" ")
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    proxy = _proxy_at(f"http://127.0.0.1:{server.server_port}", http_timeout_s=0.5)
    original = proxy._request

    def request(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(proxy, "_request", request)
    task = asyncio.create_task(proxy.closed_at(VirtualKey(key_id="hash", alias="alias")))
    try:
        await _wait(entered)
        with pytest.raises(GatewayError):
            await task
        with anyio.fail_after(1.5):
            assert await anyio.to_thread.run_sync(finished.wait, 1.0)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)
        await _wait(finished)
        assert not thread.is_alive()


@pytest.mark.parametrize("termination", ["cancel", "timeout"])
async def test_tls_handshake_worker_exits(
    monkeypatch: pytest.MonkeyPatch, termination: str
) -> None:
    from socketserver import BaseRequestHandler, ThreadingTCPServer

    entered, release, finished = (threading.Event() for _ in range(3))

    class Handler(BaseRequestHandler):
        def handle(self) -> None:
            self.request.settimeout(180)
            self.request.recv(4096)
            entered.set()
            release.wait(180)

    server = ThreadingTCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    # The worker must exit at its own operation deadline, long before the server
    # releases the handshake. The cancel case gets a deadline that the cancel
    # always beats under load; the timeout case keeps the short deadline under test.
    http_timeout_s = 10 if termination == "cancel" else 0.5
    proxy = _proxy_at(
        f"https://127.0.0.1:{server.server_address[1]}", http_timeout_s=http_timeout_s
    )
    original = proxy._request

    def request(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(proxy, "_request", request)
    task = asyncio.create_task(proxy.closed_at(VirtualKey(key_id="hash", alias="alias")))
    try:
        await _wait(entered)
        if termination == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(GatewayError):
                await task
        with anyio.fail_after(90):
            assert await anyio.to_thread.run_sync(finished.wait, 60)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(60)
        await _wait(finished)
        assert not thread.is_alive()


async def test_missing_key_confirmation_shares_operation_byte_budget(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    import swarm_arena._gateway.litellm_proxy as module

    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.keys.clear()
    # A malformed success acknowledgement consumes this operation's body budget.
    fake_gateway.responses["/key/block"] = {"padding": "x" * 200}
    monkeypatch.setattr(module, "_MAX_OPERATION_BYTES", 250)
    with pytest.raises(GatewayError):
        await proxy.block_key(key)
    assert fake_gateway.requests[-1][1] == "/key/info"


async def test_delete_foreign_echo_is_unknown(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.responses["/key/delete"] = {"deleted_keys": [key.key_id, "foreign"]}
    with pytest.raises(GatewayError):
        await proxy.delete_key(key)


async def test_discarded_metadata_cannot_conflict_with_same_page_row(
    fake_gateway: FakeGateway,
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(key.key_id, model="m", cost_usd=1, tokens=(1, 2), call_id=None)
    original = dict(fake_gateway._spend[0][1])
    fake_gateway._spend.append((0.0, {**original, "metadata": {"proxy_server_request": MASTER}}))
    (row,) = await _rows(proxy, key)
    assert row.cost_usd == 1
    assert row.metadata == {}
    assert MASTER not in row.model_dump_json()


async def test_hostname_resolves_in_daemon_http_worker(
    fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, key = await _proxy_and_key(fake_gateway)
    resolver_threads: list[threading.Thread] = []
    original = socket.getaddrinfo

    def resolve(*args: Any, **kwargs: Any) -> Any:
        resolver_threads.append(threading.current_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    proxy = _proxy_at(fake_gateway.url.replace("127.0.0.1", "localhost"))
    assert await proxy.closed_at(key) is None
    assert len(resolver_threads) == 1
    assert resolver_threads[0].daemon
    assert resolver_threads[0].name == "swarm-arena-daemon-work"


@pytest.mark.parametrize("operation", ["closed_at", "create_key"])
async def test_abandoned_dns_never_opens_socket(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    import swarm_arena._gateway.litellm_proxy as module

    entered, release, finished = (threading.Event() for _ in range(3))
    sockets: list[Any] = []
    original_socket = socket.socket
    proxy = _proxy_at("http://localhost:4000", http_timeout_s=0.5)
    original_request = proxy._request

    def resolve(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(5)
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 4000))]

    def make_socket(*args: Any, **kwargs: Any) -> Any:
        sock = original_socket(*args, **kwargs)
        sockets.append(sock)
        return sock

    def request(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_request(*args, **kwargs)
        finally:
            finished.set()

    # Scope socket instrumentation to this module; AnyIO owns unrelated sockets.
    from types import SimpleNamespace

    monkeypatch.setattr(
        module,
        "socket",
        SimpleNamespace(
            getaddrinfo=resolve,
            socket=make_socket,
            SOCK_STREAM=socket.SOCK_STREAM,
            AF_INET=socket.AF_INET,
            AF_INET6=socket.AF_INET6,
            SHUT_RDWR=socket.SHUT_RDWR,
        ),
    )
    monkeypatch.setattr(proxy, "_request", request)
    task = asyncio.create_task(_call(proxy, operation, VirtualKey(key_id="hash", alias="alias")))
    try:
        await _wait(entered)
        with anyio.fail_after(1.5):
            with pytest.raises(GatewayError):
                await task
        assert not finished.is_set()
        assert not sockets
        release.set()
        await _wait(finished)
        assert not sockets
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await _wait(finished)
        for sock in sockets:
            sock.close()


async def test_51_row_shifted_listing_is_incomplete(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    for i in range(1, 52):
        fake_gateway.add_spend(key.key_id, model="m", cost_usd=0.1, tokens=(1, 2), call_id=f"r{i}")
        fake_gateway._spend[-1][1]["request_id"] = f"r{i}"
    # Total stays 51, but page 2 repeats r50 instead of returning r51.
    # Even an identical repeat means the offset listing lost a row.
    fake_gateway.page_overrides[2] = {"data": [dict(fake_gateway._spend[49][1])]}
    with pytest.raises(GatewayError) as caught:
        await _rows(proxy, key)
    assert str(caught.value) == "GET /spend/logs/v2: ValueError"
    assert [q["page"] for _, p, q in fake_gateway.requests if p == "/spend/logs/v2"] == ["1", "2"]
    fake_gateway.page_overrides.clear()
    rows = await _rows(proxy, key)
    assert len(rows) == 51
    assert rows[-1].request_id == "r51"


@pytest.mark.parametrize(
    "bad", [None, 7, "bad", [], {"candidate_id": "c" * 300}, {"candidate_id": "\ud800"}]
)
async def test_invalid_attribution_section_preserves_other_metadata(
    fake_gateway: FakeGateway, bad: Any
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(
        key.key_id,
        model="m",
        cost_usd=0.1,
        tokens=(10, 2),
        call_id=None,
        metadata={
            "spend_logs_metadata": bad,
            "additional_usage_values": {"cache_read_input_tokens": 3},
        },
    )
    (row,) = await _rows(proxy, key)
    assert row.cost_usd == 0.1
    assert row.key_id == key.key_id
    assert row.metadata == {"additional_usage_values": {"cache_read_input_tokens": 3}}
    assert row.usage.input_tokens == 7


@pytest.mark.parametrize("field", ["cache_read_input_tokens", "cache_creation_input_tokens"])
@pytest.mark.parametrize("bad", [-1, True, "3", 99, float("nan"), [3]])
async def test_invalid_cache_field_drops_only_that_field(
    fake_gateway: FakeGateway, field: str, bad: Any
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    other = (
        "cache_creation_input_tokens"
        if field == "cache_read_input_tokens"
        else "cache_read_input_tokens"
    )
    fake_gateway.add_spend(
        key.key_id,
        model="m",
        cost_usd=0.1,
        tokens=(10, 2),
        call_id=None,
        metadata={
            "spend_logs_metadata": {"candidate_id": "untrusted-candidate"},
            "additional_usage_values": {field: bad, other: 2},
        },
    )
    (row,) = await _rows(proxy, key)
    assert row.metadata == {
        "spend_logs_metadata": {"candidate_id": "untrusted-candidate"},
        "additional_usage_values": {other: 2},
    }
    assert row.key_id == key.key_id
    assert row.cost_usd == 0.1
    assert row.usage.input_tokens == 8


@pytest.mark.parametrize("bad", [None, 7, "bad", [], {"secret": float("inf")}])
async def test_invalid_metadata_container_keeps_spend(fake_gateway: FakeGateway, bad: Any) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(key.key_id, model="m", cost_usd=0.1, tokens=(10, 2), call_id=None)
    fake_gateway.row_overrides = {"metadata": bad}
    (row,) = await _rows(proxy, key)
    assert row.metadata == {}
    assert row.cost_usd == 0.1
    assert row.usage.input_tokens == 10


async def test_cache_sum_drops_field_exceeding_remaining_prompt(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.add_spend(
        key.key_id,
        model="m",
        cost_usd=0.1,
        tokens=(10, 2),
        call_id=None,
        metadata={
            "additional_usage_values": {
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 8,
            }
        },
    )
    (row,) = await _rows(proxy, key)
    assert row.metadata == {"additional_usage_values": {"cache_read_input_tokens": 3}}
    assert row.usage.input_tokens == 7
    assert row.usage.cache_write_tokens == 0


@pytest.mark.parametrize("bad", [None, 7, "bad", [], {"cache_read_input_tokens": "x" * 300}])
async def test_invalid_cache_section_preserves_label_and_other_rows(
    fake_gateway: FakeGateway, bad: Any
) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    for metadata in (
        {
            "spend_logs_metadata": {"candidate_id": "untrusted-label"},
            "additional_usage_values": bad,
        },
        {"additional_usage_values": {"cache_read_input_tokens": 2}},
    ):
        fake_gateway.add_spend(
            key.key_id, model="m", cost_usd=0.1, tokens=(10, 2), call_id=None, metadata=metadata
        )
    first, second = await _rows(proxy, key)
    assert first.metadata == {"spend_logs_metadata": {"candidate_id": "untrusted-label"}}
    assert first.usage.input_tokens == 10
    assert second.metadata == {"additional_usage_values": {"cache_read_input_tokens": 2}}
    assert second.usage.input_tokens == 8
    assert first.key_id == second.key_id == key.key_id
    assert first.cost_usd == second.cost_usd == 0.1
