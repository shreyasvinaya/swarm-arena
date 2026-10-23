import threading
from typing import Any

import anyio
import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import (
    AlwaysPass,
    CountingTool,
    FixedJudge,
    NoopManager,
    RecordingExecutor,
)

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    Goal,
    PartSpec,
    Plan,
    Ref,
    RoundSpec,
    RunConfig,
    TeamSpec,
)
from swarm_arena._agent.manager import LimitReached
from swarm_arena._engine import preflight
from swarm_arena._engine.preflight import check_storage, run_preflight
from swarm_arena._errors import ConditionalWritesUnsupported, Fenced, _core_error
from swarm_arena._exec.executor import Executor


def _plan(
    *,
    agents: int = 1,
    same_config: bool = False,
    config: dict[str, Any] | None = None,
    tool: str = "_fakes.tools:CountingTool",
) -> Plan:
    return Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[
            TeamSpec(
                name="t",
                rationale="Explore",
                agents=[
                    AgentSpec(
                        name=f"a{i}",
                        rationale="Explore",
                        tool=Ref(path=tool, config=config or {}),
                        manager=Ref(path="_fakes.tools:NoopManager"),
                    )
                    for i in range(agents)
                ],
            )
        ],
        judges=[Ref(path="_fakes.tools:FixedJudge", config={"value": 1})],
    )


def _executor() -> Executor:
    return RecordingExecutor()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("site", ["component", "executor"])
async def test_preflight_exit_terminal(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, site: str, nested: bool
) -> None:
    from unittest.mock import Mock

    from swarm_arena._errors import _core_category, _SignalCategory, _stop_cleanup_error

    reached: list[str] = []

    def fail(*args: Any) -> None:
        reached.append(site)
        error = _stop_cleanup_error()
        if nested:
            raise BaseExceptionGroup("preflight", [BaseExceptionGroup("hook", [error])])
        raise error

    executor = Mock(spec=Executor, preflight=fail)
    plans = [] if site == "executor" else [_plan()]
    if site == "component":
        monkeypatch.setattr(preflight, "_run_hook", fail)
    with pytest.raises(RuntimeError) as caught:
        await run_preflight(plans, run_config, executor)
    assert _core_category(caught.value) is _SignalCategory.TERMINAL
    assert reached == [site]


async def test_preflight_once_per_distinct_config() -> None:
    CountingTool.calls = 0
    await run_preflight(
        [_plan(agents=3, same_config=True), _plan(config={"k": 2})], RunConfig(), _executor()
    )
    assert CountingTool.calls == 2


async def test_preflight_failures_collected() -> None:
    plan = _plan(tool="_fakes.tools:FailingPreflightTool")
    plan.checks.append(Ref(path="_fakes.tools:PreflightFailsCheck"))
    with pytest.raises(
        RuntimeError, match=r"preflight failed:\n- .*PreflightFailsCheck: RuntimeError"
    ) as caught:
        await run_preflight([plan], RunConfig(), _executor())
    assert "- _fakes.tools:FailingPreflightTool: RuntimeError" in str(caught.value)
    assert "no license" not in str(caught.value)


async def test_normalized_defaults_and_ref_names_do_not_repeat_hooks() -> None:
    CountingTool.calls = 0
    first, second = _plan(), _plan(config={"k": 1, "code": "", "grace_s": 10.0})
    agent = second.teams[0].agents[0]
    second.teams[0].agents[0] = agent.model_copy(
        update={"tool": agent.tool.model_copy(update={"name": "other-name"})}
    )
    await run_preflight([first, second], RunConfig(), _executor())
    assert CountingTool.calls == 1


async def test_effective_rounds_converge_and_later_revisions() -> None:
    CountingTool.calls = 0
    plan = _plan()
    plan.rounds.extend([RoundSpec(tool_overrides={"k": 2}), RoundSpec(tool_overrides={"k": 3})])
    plan = plan.model_copy(
        update={
            "converge": ConvergeSpec(
                mode="team",
                team=plan.teams[0].model_copy(update={"name": "join"}),
                round=RoundSpec(tool_overrides={"k": 4}),
            )
        }
    )
    await run_preflight([plan], RunConfig(), _executor())
    assert CountingTool.calls == 3
    plan = plan.model_copy(
        update={
            "revision": 1,
            "rounds": [RoundSpec(tool_overrides={"k": 5})],
            "converge": plan.converge.model_copy(update={"round": None}),
        }
    )
    await run_preflight([plan], RunConfig(), _executor())
    assert CountingTool.calls == 4


async def test_all_hook_interfaces_and_no_policy_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    for cls in (AlwaysPass, FixedJudge, NoopManager, RecordingExecutor):

        def hook(self: object, label: str = cls.__name__) -> None:
            calls.append(label)
            assert threading.current_thread().daemon

        monkeypatch.setattr(cls, "preflight", hook)
    plan = _plan(agents=3)
    plan.checks.append(Ref(path="_fakes.tools:AlwaysPass"))
    plan.parts.extend(
        [
            PartSpec(
                name="p", goal=Goal(id="p", namespace="n"), checks=plan.checks, judges=plan.judges
            )
        ]
    )
    plan = plan.model_copy(update={"policy": Ref(path="never_import_this:Policy")})
    await run_preflight([plan, plan], RunConfig(), _executor())
    assert sorted(calls) == ["AlwaysPass", "FixedJudge", "NoopManager", "RecordingExecutor"]


async def test_hook_and_constructor_errors_are_type_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: object) -> None:
        raise RuntimeError("PREFLIGHT_SECRET")

    monkeypatch.setattr(CountingTool, "preflight", fail)
    with pytest.raises(RuntimeError, match="CountingTool: RuntimeError") as caught:
        await run_preflight([_plan()], RunConfig(), _executor())
    assert_secret_absent(caught.value, "PREFLIGHT_SECRET", sanitized=True)

    def init(self: object, config: object) -> None:
        raise ValueError("PREFLIGHT_SECRET")

    monkeypatch.setattr(CountingTool, "__init__", init)
    with pytest.raises(RuntimeError, match="CountingTool: ValueError") as caught:
        await run_preflight([_plan()], RunConfig(), _executor())
    assert_secret_absent(caught.value, "PREFLIGHT_SECRET", sanitized=True)


async def test_hung_hook_is_abandonable_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    completed = anyio.Event()
    scope = anyio.CancelScope()

    def block(self: object) -> None:
        workers.append(threading.current_thread())
        assert workers[-1].daemon
        entered.set()
        assert release.wait(15), "preflight release deadline"

    async def preflight() -> None:
        with scope:
            await run_preflight([_plan()], RunConfig(), _executor())
        completed.set()

    monkeypatch.setattr(CountingTool, "preflight", block)
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(preflight)
            with anyio.fail_after(10):
                assert await anyio.to_thread.run_sync(entered.wait, 10)
            scope.cancel()
            with anyio.fail_after(2):
                await completed.wait()
            assert workers[0].is_alive()
            assert not release.is_set()
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()


async def test_cancelled_hook_does_not_continue_to_later_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    completed = anyio.Event()
    scope = anyio.CancelScope()
    called: list[str] = []

    def block(self: object) -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(15), "preflight release deadline"

    def later(self: object) -> None:
        called.append("executor")

    async def preflight() -> None:
        with scope:
            await run_preflight([_plan()], RunConfig(), _executor())
        completed.set()

    monkeypatch.setattr(CountingTool, "preflight", block)
    monkeypatch.setattr(RecordingExecutor, "preflight", later)
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(preflight)
            with anyio.fail_after(10):
                assert await anyio.to_thread.run_sync(entered.wait, 10)
            scope.cancel()
            with anyio.fail_after(2):
                await completed.wait()
    finally:
        release.set()
        # Join the abandoned worker before checking that it cannot dispatch the next hook.
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()
    assert called == []


async def test_part_overrides_and_executor_failures_are_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(self: object) -> None:
        raise LookupError("EXECUTOR_SECRET")

    monkeypatch.setattr(RecordingExecutor, "preflight", fail)
    plan = _plan()
    plan.parts.append(
        PartSpec(
            name="p",
            goal=Goal(id="p", namespace="n"),
            checks=[Ref(path="_fakes.tools:PreflightFailsCheck")],
        )
    )
    with pytest.raises(RuntimeError, match="PreflightFailsCheck: RuntimeError") as caught:
        await run_preflight([plan], RunConfig(), _executor())
    assert "swarm_arena.executor:LocalExecutor: LookupError" in str(caught.value)
    assert_secret_absent(caught.value, "EXECUTOR_SECRET", sanitized=True)


async def test_shared_preflight_has_its_own_hook_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []

    def block(self: object) -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(10), "preflight release deadline"

    monkeypatch.setattr(CountingTool, "preflight", block)
    try:
        with anyio.fail_after(3):
            with pytest.raises(RuntimeError, match="CountingTool: TimeoutError"):
                await run_preflight([_plan()], RunConfig(settle_timeout_s=1), _executor())
        assert entered.is_set()
        assert workers[0].is_alive()
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()


class RecordingStore:
    def __init__(self, url: str, probed: list[str], *, fail: bool = False) -> None:
        self.url, self.probed, self.fail = url, probed, fail

    def probe_conditional_writes(self) -> None:
        self.probed.append(self.url)
        if self.fail:
            raise ConditionalWritesUnsupported("create did not behave conditionally")


def test_local_run_root_with_s3_arena_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_open(url: str) -> None:
        pytest.fail("invalid root combination reached storage")

    monkeypatch.setattr(preflight, "open_store", no_open)
    with pytest.raises(ValueError, match="run_root"):
        check_storage(RunConfig(run_root="./runs", arena_root="s3://b/arena"))


def test_probe_runs_once_per_distinct_s3_root(monkeypatch: pytest.MonkeyPatch) -> None:
    probed: list[str] = []
    monkeypatch.setattr(preflight, "open_store", lambda url: RecordingStore(url, probed))
    check_storage(RunConfig(run_root="s3://b/x", arena_root="s3://b/x", artifact_root="s3://b/x"))
    assert probed == ["s3://b/x"]
    probed.clear()
    check_storage(RunConfig())
    assert probed == []
    check_storage(RunConfig(run_root="s3://b/r", arena_root="s3://b/a", artifact_root="s3://b/f"))
    assert probed == ["s3://b/r", "s3://b/a", "s3://b/f"]


def test_probe_failure_stops_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    probed: list[str] = []
    monkeypatch.setattr(preflight, "open_store", lambda url: RecordingStore(url, probed, fail=True))
    with pytest.raises(ConditionalWritesUnsupported):
        check_storage(RunConfig(run_root="s3://b/r", arena_root="s3://b/a"))
    assert probed == ["s3://b/r"]


async def test_storage_checked_first_in_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread = threading.get_ident()
    CountingTool.calls = 0

    def fail(config: RunConfig) -> None:
        assert threading.get_ident() != loop_thread
        raise _core_error(ConditionalWritesUnsupported, "storage unavailable")

    monkeypatch.setattr(preflight, "check_storage", fail)
    with pytest.raises(ConditionalWritesUnsupported):
        await run_preflight([_plan()], RunConfig(), _executor())
    assert CountingTool.calls == 0


def _gateway_config(
    gateway: Any, *, max_usd: float | None = None, require: bool | None = None
) -> RunConfig:
    from swarm_arena import Limits
    from swarm_arena.config import GatewayConfig

    return RunConfig(
        limits=Limits(max_usd=max_usd),
        llm_gateway=GatewayConfig(url=gateway.url, require_hard_cap=require),
    )


async def test_gateway_preflight_hard(fake_gateway: Any) -> None:
    from _fakes.gateway import ABSENT_FAILURES

    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.litellm_proxy import adapter_for

    config = _gateway_config(fake_gateway)
    assert config.llm_gateway is not None
    adapter = adapter_for(config.llm_gateway, {"LITELLM_MASTER_KEY": fake_gateway.master_key})
    cap, report = await gateway_preflight(config, adapter)
    assert cap == "soft"
    assert report.verified is False
    assert report.failures == (*ABSENT_FAILURES, "gateway settings introspection unavailable")
    assert not fake_gateway.keys


async def test_gateway_preflight_fails_when_hard_cap_required(fake_gateway: Any) -> None:
    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.litellm_proxy import adapter_for

    config = _gateway_config(fake_gateway, max_usd=5)
    assert config.llm_gateway is not None
    adapter = adapter_for(config.llm_gateway, {"LITELLM_MASTER_KEY": fake_gateway.master_key})
    report = await adapter.preflight()
    with pytest.raises(ValueError, match="strict profile not verified") as caught:
        await gateway_preflight(config, adapter)
    assert str(caught.value) == (
        f"LLM gateway strict profile not verified: {'; '.join(report.failures)}. "
        "Fix the gateway, or set llm_gateway.require_hard_cap = false to run with a soft cap."
    )
    assert_secret_absent(caught.value, fake_gateway.master_key)
    assert not fake_gateway.keys


async def test_gateway_preflight_soft_when_not_required(fake_gateway: Any) -> None:
    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.litellm_proxy import adapter_for

    config = _gateway_config(fake_gateway, max_usd=5, require=False)
    assert config.llm_gateway is not None
    adapter = adapter_for(config.llm_gateway, {"LITELLM_MASTER_KEY": fake_gateway.master_key})
    assert (await gateway_preflight(config, adapter))[0] == "soft"


async def test_gateway_preflight_verified_capability_stub_is_hard() -> None:
    from unittest.mock import AsyncMock, Mock

    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.adapter import GatewayAdapter, ProfileReport
    from swarm_arena.config import GatewayConfig

    report = ProfileReport(
        verified=True, failures=(), batch_write_interval_s=1, request_timeout_s=2
    )
    adapter = Mock(spec=GatewayAdapter, preflight=AsyncMock(return_value=report))
    config = RunConfig(llm_gateway=GatewayConfig(url="http://example.test", require_hard_cap=True))
    cap, actual = await gateway_preflight(config, adapter)
    assert cap == "hard"
    assert actual == report
    adapter.create_key.assert_not_called()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("fenced", [False, True])
async def test_gateway_preflight_preserves_terminal(nested: bool, fenced: bool) -> None:
    from unittest.mock import AsyncMock, Mock

    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._errors import _core_category, _SignalCategory, _stop_cleanup_error
    from swarm_arena._gateway.adapter import GatewayAdapter
    from swarm_arena.config import GatewayConfig

    error: Exception = _core_error(Fenced) if fenced else _stop_cleanup_error()
    if nested:
        error = ExceptionGroup("GATEWAY_SENTINEL", [error])
    adapter = Mock(spec=GatewayAdapter, preflight=AsyncMock(side_effect=error))
    config = RunConfig(llm_gateway=GatewayConfig(url="http://example.test"))
    with pytest.raises(RuntimeError) as caught:
        await gateway_preflight(config, adapter)
    expected = _SignalCategory.FENCED if fenced else _SignalCategory.TERMINAL
    assert _core_category(caught.value) is expected
    assert_secret_absent(caught.value, "GATEWAY_SENTINEL")


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("GATEWAY_SENTINEL"),
        Fenced("GATEWAY_SENTINEL"),
        LimitReached("stop", "GATEWAY_SENTINEL"),
        ExceptionGroup("GATEWAY_SENTINEL", [ValueError("GATEWAY_SENTINEL")]),
    ],
)
async def test_gateway_preflight_contains_untrusted_errors(error: Exception) -> None:
    from unittest.mock import AsyncMock, Mock

    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.adapter import GatewayAdapter
    from swarm_arena.config import GatewayConfig

    adapter = Mock(spec=GatewayAdapter, preflight=AsyncMock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        await gateway_preflight(
            RunConfig(llm_gateway=GatewayConfig(url="http://example.test")), adapter
        )
    assert str(caught.value) == f"gateway preflight failed: {type(error).__name__}"
    assert_secret_absent(caught.value, "GATEWAY_SENTINEL")


async def test_gateway_preflight_caller_cancel_propagates() -> None:
    from unittest.mock import Mock

    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.adapter import GatewayAdapter, ProfileReport
    from swarm_arena.config import GatewayConfig

    entered = anyio.Event()
    scope = anyio.CancelScope()

    async def held() -> ProfileReport:
        entered.set()
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    async def invoke() -> None:
        with scope:
            await gateway_preflight(
                RunConfig(llm_gateway=GatewayConfig(url="http://example.test")),
                Mock(spec=GatewayAdapter, preflight=held),
            )

    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(invoke)
            await entered.wait()
            scope.cancel()
    assert scope.cancelled_caught


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit, GeneratorExit])
async def test_gateway_preflight_passes_process_control_unchanged(
    kind: type[BaseException],
) -> None:
    from unittest.mock import AsyncMock, Mock

    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.adapter import GatewayAdapter
    from swarm_arena.config import GatewayConfig

    error = kind()
    adapter = Mock(spec=GatewayAdapter, preflight=AsyncMock(side_effect=error))
    with pytest.raises(kind) as caught:
        await gateway_preflight(
            RunConfig(llm_gateway=GatewayConfig(url="http://example.test")), adapter
        )
    assert caught.value is error


async def test_gateway_preflight_requires_gateway_config() -> None:
    from unittest.mock import Mock

    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.adapter import GatewayAdapter

    adapter = Mock(spec=GatewayAdapter)
    with pytest.raises(ValueError, match="llm_gateway is required"):
        await gateway_preflight(RunConfig(), adapter)
    adapter.preflight.assert_not_called()


@pytest.mark.parametrize("status", [401, 404, 503])
async def test_gateway_preflight_retains_core_transport_detail(
    fake_gateway: Any, status: int
) -> None:
    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.litellm_proxy import adapter_for

    config = _gateway_config(fake_gateway)
    assert config.llm_gateway is not None
    fake_gateway.fail["/health/liveliness"] = status
    adapter = adapter_for(config.llm_gateway, {"LITELLM_MASTER_KEY": fake_gateway.master_key})
    with pytest.raises(RuntimeError) as caught:
        await gateway_preflight(config, adapter)
    assert str(caught.value) == f"gateway preflight failed: GET /health/liveliness: HTTP {status}"
    assert_secret_absent(caught.value, fake_gateway.master_key)


async def test_gateway_preflight_foreign_gateway_error_is_type_only() -> None:
    from unittest.mock import AsyncMock, Mock

    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.adapter import GatewayAdapter, GatewayError
    from swarm_arena.config import GatewayConfig

    adapter = Mock(spec=GatewayAdapter, preflight=AsyncMock(side_effect=GatewayError("SECRET")))
    with pytest.raises(RuntimeError) as caught:
        await gateway_preflight(
            RunConfig(llm_gateway=GatewayConfig(url="http://example.test")), adapter
        )
    assert str(caught.value) == "gateway preflight failed: GatewayError"
    assert_secret_absent(caught.value, "SECRET")


async def test_gateway_preflight_bounds_hostile_model_diagnostics(fake_gateway: Any) -> None:
    from swarm_arena._engine.manifest import _GatewayManifest
    from swarm_arena._engine.preflight import gateway_preflight
    from swarm_arena._gateway.litellm_proxy import adapter_for

    hostile = "x\n\x1b[2Jfake: all checks passed" + "A" * 5000
    fake_gateway.responses["/model/info"] = {
        "data": [
            {"model_name": hostile if index == 0 else f"model-{index}", "model_info": {}}
            for index in range(12)
        ]
    }
    config = _gateway_config(fake_gateway, max_usd=5)
    assert config.llm_gateway is not None
    adapter = adapter_for(config.llm_gateway, {"LITELLM_MASTER_KEY": fake_gateway.master_key})
    with pytest.raises(ValueError, match="strict profile not verified") as caught:
        await gateway_preflight(config, adapter)
    detail = str(caught.value)
    assert "model <unknown>: no price" in detail
    assert "and 18 more" in detail
    assert "model-10" not in detail
    assert "model-11" not in detail
    assert detail.isprintable()
    assert len(detail) < 1500
    cap, report = await gateway_preflight(_gateway_config(fake_gateway, require=False), adapter)
    block = _GatewayManifest(
        url=fake_gateway.url,
        cap=cap,
        verified=report.verified,
        failures=report.failures,
        terminal_accounting=False,
    )
    assert len(block.failures) == 11
    assert block.failures[-1] == "and 18 more"
    assert "model <unknown>: no price" in block.failures[0]
    assert all(value.isprintable() and len(value) <= 512 for value in block.failures)
    assert _GatewayManifest.model_validate_json(block.model_dump_json()) == block
