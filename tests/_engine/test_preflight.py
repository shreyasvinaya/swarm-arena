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
from swarm_arena._engine import preflight
from swarm_arena._engine.preflight import check_storage, run_preflight
from swarm_arena._errors import ConditionalWritesUnsupported, _core_error
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
