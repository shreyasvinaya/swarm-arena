import asyncio
import gc
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from weakref import ref

import anyio
import pytest
from _fakes.engine import EngineEnv
from _fakes.secrecy import assert_secret_absent

from swarm_arena._engine import stops as stops_module
from swarm_arena._engine.stops import RunClock, StopState, round_deadline, watch_stop_files
from swarm_arena._errors import Fenced, StorageError, _core_error, _is_core_signal
from swarm_arena._store.local import LocalStore


async def test_poll_reads_only_own_epoch(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_poll, second_poll, next_poll = anyio.Event(), anyio.Event(), anyio.Event()
    polls = 0

    async def after_poll(interval_s: float) -> None:
        nonlocal polls
        assert interval_s == 0.05
        polls += 1
        if polls == 1:
            first_poll.set()
            await next_poll.wait()
        else:
            second_poll.set()
            await anyio.sleep_forever()

    monkeypatch.setattr(
        stops_module, "anyio", SimpleNamespace(lowlevel=anyio.lowlevel, sleep=after_poll)
    )
    store.create("control/stop-e1", b"")
    store.create("control/stop-now-e1", b"")
    st = StopState()
    with anyio.fail_after(60):
        async with anyio.create_task_group() as tasks:

            async def poll() -> None:
                await watch_stop_files(store, 2, st, interval_s=0.05)

            tasks.start_soon(poll)
            await first_poll.wait()
            assert not st.stop
            assert not st.stop_now
            store.create("control/stop-now-e2", b"")
            next_poll.set()
            await second_poll.wait()
            tasks.cancel_scope.cancel()
    assert st.stop_now
    assert st.stop


def test_gate_reasons() -> None:
    st = StopState(goal_budget={"g"})
    err = st.gate_for("g")()
    assert err is not None
    assert (err.limit, str(err)) == ("budget", "goal budget used up")
    assert st.gate_for("h")() is None
    st.stop_now = True
    assert (err := st.gate_for("h")()) is not None
    assert err.limit == "stop"


def test_run_clock_counts_only_running_time() -> None:
    assert RunClock(elapsed_before=10.0).elapsed() >= 10.0
    assert RunClock(elapsed_before=86_399.0).left(86_400.0) <= 1.0


def test_round_deadline_single_time_basis() -> None:
    assert round_deadline(1000.0, 5.0, 2.0) == 1002.0
    assert round_deadline(1000.0, 5.0, 86_400.0) == 1005.0


@pytest.mark.parametrize("flag", ["stop", "stop_now", "run_timeout", "run_budget"])
async def test_run_stop_latch_wakes_scheduler_once(flag: str) -> None:
    changed = anyio.Event()
    notifications: list[str] = []

    def notify() -> None:
        notifications.append(flag)
        changed.set()

    state = StopState(on_change=notify)
    setattr(state, flag, True)
    with anyio.fail_after(60):
        await changed.wait()
    setattr(state, flag, True)
    assert notifications == [flag]


@pytest.mark.parametrize("key", ["control/stop-now-e2", "control/stop-e2"])
async def test_cancelled_poll_discards_late_true(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    entered, release, exited = (threading.Event() for _ in range(3))
    calls: list[str] = []
    workers: list[threading.Thread] = []

    def exists(current: str) -> bool:
        calls.append(current)
        if current != key:
            return False
        workers.append(threading.current_thread())
        entered.set()
        try:
            assert release.wait(180)
            return True
        finally:
            exited.set()

    monkeypatch.setattr(store, "exists", exists)
    state = StopState()
    task = asyncio.create_task(watch_stop_files(store, 2, state))
    try:
        with anyio.fail_after(60):
            assert await anyio.to_thread.run_sync(entered.wait, 60)
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=30)
        assert done, "polling cancellation waited for storage"
        assert task.cancelled()
        assert not state.stop
        assert not state.stop_now
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(60):
            await asyncio.gather(task, return_exceptions=True)
            assert await anyio.to_thread.run_sync(exited.wait, 60)
            for worker in workers:
                if worker.name.startswith("swarm-arena-daemon-"):
                    await anyio.to_thread.run_sync(worker.join, 60)
                    assert not worker.is_alive()
    assert not state.stop
    assert not state.stop_now
    assert calls == ([key] if "now" in key else ["control/stop-now-e2", key])


@pytest.mark.parametrize("key", ["control/stop-now-e2", "control/stop-e2"])
@pytest.mark.parametrize("kind", ["fence", "group_fence", "storage"])
async def test_poll_rebuilds_registered_outcomes(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch, key: str, kind: str
) -> None:
    secret = "private-mutated-stop-outcome"
    error = _core_error(StorageError if kind == "storage" else Fenced, "safe stored diagnostic")
    error.args = (secret,)
    error.add_note(secret)
    error.__cause__ = OSError(secret)
    raised: BaseException = error
    if kind == "group_fence":
        raised = BaseExceptionGroup(secret, [asyncio.CancelledError(secret), error])

    def exists(current: str) -> bool:
        if current == key:
            raise raised
        return False

    monkeypatch.setattr(store, "exists", exists)
    with pytest.raises(StorageError if kind == "storage" else Fenced) as caught:
        await watch_stop_files(store, 2, StopState())
    assert caught.value is not error
    assert _is_core_signal(caught.value)
    assert_secret_absent(caught.value, secret)
    assert str(caught.value) == (
        "run ownership lost" if kind != "storage" else "safe stored diagnostic"
    )


async def test_provider_failed_stops_matching_agents_in_every_goal(env: EngineEnv) -> None:
    from swarm_arena._engine.stops import ProviderStops
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.replay import RunState

    stops = ProviderStops(env.deps.journal)
    cancelled: list[str] = []
    for address, providers in [
        ("g/t/A", {"anthropic"}),
        ("h/t/B", {"anthropic", "openai"}),
        ("h/t/C", {"openai"}),
    ]:
        stops.register(address, frozenset(providers), lambda a=address: cancelled.append(a))
    assert await stops.fail("anthropic", source="manager") == ["g/t/A", "h/t/B"]
    assert cancelled == ["g/t/A", "h/t/B"]
    assert stops.stopped() == {"g/t/A": frozenset({"anthropic"}), "h/t/B": frozenset({"anthropic"})}
    event = env.events("provider_failed")[-1]
    assert event.payload == {
        "variant": "transition",
        "transition": "failed",
        "provider": "anthropic",
        "generation": 1,
        "source": "manager",
        "stopped": ["g/t/A", "h/t/B"],
    }
    committed = read_committed(env.deps.store)
    assert committed is not None
    state = RunState.from_events(committed[1])
    rebuilt = ProviderStops(env.deps.journal, state=state)
    assert rebuilt.stopped() == stops.stopped()
    assert rebuilt.generations() == {"anthropic": 1}


async def test_failure_before_registration_and_newer_failure_during_probe(env: EngineEnv) -> None:
    from swarm_arena._engine.stops import ProviderStops

    stops = ProviderStops(env.deps.journal)
    assert await stops.fail("anthropic", source="judge") == []
    cancelled: list[str] = []
    stops.register("g/t/A", frozenset({"anthropic"}), lambda: cancelled.append("A"))
    assert cancelled == ["A"]
    generation = stops.generations()["anthropic"]
    await stops.fail("anthropic", source="tool")
    assert await stops.recover("anthropic", generation=generation) == []
    assert stops.stopped()["g/t/A"] == {"anthropic"}
    assert await stops.recover("anthropic", generation=2) == ["g/t/A"]
    assert stops.stopped()["g/t/A"] == frozenset()
    stops.activate("g/t/A")
    assert stops.stopped() == {}


async def test_provider_recovery_keeps_other_causes_and_permanent_release(env: EngineEnv) -> None:
    from swarm_arena._engine.stops import ProviderStops

    stops = ProviderStops(env.deps.journal)
    stops.register("g/t/A", frozenset({"anthropic", "openai"}), lambda: None)
    await stops.fail("anthropic", source="manager")
    await stops.fail("openai", source="tool")
    await stops.recover("anthropic", generation=1)
    assert stops.stopped() == {"g/t/A": frozenset({"openai"})}
    stops.release("g/t/A")
    await stops.recover("openai", generation=1)
    assert stops.stopped() == {}


def test_provider_signal_snapshot_and_forgery() -> None:
    from swarm_arena._errors import (
        ProviderFailed,
        _core_category,
        _rebuild_core_signal,
        _SignalCategory,
    )
    from swarm_arena._outcomes import _classify_outcome

    error = _core_error(ProviderFailed, "anthropic")
    error.provider = "private-value"
    error.args = ("private-value",)
    rebuilt = _rebuild_core_signal(error)
    assert isinstance(rebuilt, ProviderFailed)
    assert rebuilt.provider == "anthropic"
    assert _core_category(rebuilt) is _SignalCategory.PROVIDER
    assert_secret_absent(rebuilt, "private-value")
    forged = ProviderFailed("private-value")
    forged.args = ("private-value",)
    safe = _classify_outcome(forged)
    assert _core_category(safe) is not _SignalCategory.PROVIDER
    assert_secret_absent(safe, "private-value")
    with pytest.raises(asyncio.CancelledError):
        _classify_outcome(error, cancelled=True)


def test_tool_failed_provider_or_all_tool_providers() -> None:
    from unittest.mock import Mock

    from swarm_arena._engine.stops import failed_providers
    from swarm_arena._model.tooling import ToolOutput

    tool = Mock()
    tool.providers.return_value = frozenset({"openai", "gemini"})
    assert failed_providers(
        ToolOutput(status="provider_failed", failed_provider="gemini"), tool
    ) == {"gemini"}
    assert failed_providers(ToolOutput(status="provider_failed"), tool) == {"openai", "gemini"}


async def test_provider_receipts_do_not_replay_as_latch_transitions(env: EngineEnv) -> None:
    from swarm_arena._journal.event import EventDraft
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.replay import RunState

    await env.deps.journal.append_durable(
        EventDraft(
            type="provider_failed",
            payload={
                "variant": "receipt",
                "provider": "openai",
                "generation": 1,
                "source": "judge",
                "completed": True,
            },
        )
    )
    committed = read_committed(env.deps.store)
    assert committed is not None
    state = RunState.from_events(committed[1])
    assert state.provider_generations == state.provider_failed == state.provider_causes == {}


@pytest.mark.parametrize("standalone", [False, True])
async def test_provider_reporter_child_scope_and_safe_variant(
    env: EngineEnv, standalone: bool
) -> None:
    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.llm_scope import LLMScope
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._engine.stops import ProviderStops, report_provider_failure
    from swarm_arena._errors import ProviderFailed

    deps = env.deps
    service = ProviderStops(deps.journal, standalone=standalone)
    if not standalone:
        service.register("g/t/a", frozenset({"openai"}), lambda: None)
    scope = LLMScope(
        role="planner",
        run_id=deps.run_id,
        goal_id="g",
        team="",
        agent="",
        round=0,
        candidate_id=None,
        tranche=Tranche(deps.ledger, "g/:judges", "g", size=Amount()),
        journal=deps.journal,
        artifacts=deps.artifacts,
        models=deps.config.models,
        tracing=deps.config.tracing,
        cache=deps.config.cache,
        limits=deps.config.limits,
        settle_timeout_s=deps.config.settle_timeout_s,
    )
    child = scope.child().child()
    error = _core_error(ProviderFailed, "openai")
    error.provider = "private-value"
    assert await report_provider_failure(child, error, source="planner") == (
        [] if standalone else ["g/t/a"]
    )
    event = env.events("provider_failed")[-1]
    assert event.payload["variant"] == ("standalone" if standalone else "transition")
    assert event.payload["provider"] == "openai"
    assert "private-value" not in str(event.payload)
    assert service.generations() == {"openai": 1}
    with pytest.raises(RuntimeError):
        await report_provider_failure(child, ProviderFailed("openai"), source="planner")
    assert service.generations() == {"openai": 1}


def test_register_probe_example_restores_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import doctest

    from swarm_arena._budget.llm_scope import LLMScope
    from swarm_arena._engine import stops as module

    async def probe(provider: str, scope: LLMScope) -> bool:
        return True

    monkeypatch.setattr(module, "_probe", probe)
    test = doctest.DocTestParser().get_doctest(
        module.register_probe.__doc__ or "", vars(module).copy(), "register_probe", None, 0
    )
    assert doctest.DocTestRunner().run(test).failed == 0
    assert module._probe is probe


def test_agent_providers_resolves_manager_catalog_and_tool_union() -> None:
    from unittest.mock import Mock

    from swarm_arena._agent.manager import Manager
    from swarm_arena._engine.stops import agent_providers
    from swarm_arena._tooling.tool import Tool
    from swarm_arena.config import ModelCatalog, ModelSpec

    manager, tool = Mock(spec=Manager), Mock(spec=Tool)
    manager.models.return_value = ("a", "b")
    tool.providers.return_value = frozenset({"openai", "gemini"})
    catalog = ModelCatalog(
        {
            "a": ModelSpec(model="anthropic/a", roles=["manager"]),
            "b": ModelSpec(model="openai/b", roles=["manager"]),
        }
    )
    assert agent_providers(manager, tool, catalog) == {"anthropic", "openai", "gemini"}


@pytest.mark.parametrize("hashable", [False, True])
def test_provider_services_use_writer_identity(hashable: bool) -> None:
    @dataclass(unsafe_hash=hashable)
    class Writer:
        value: int = 1

    first: Any = Writer()
    second: Any = Writer()
    assert first == second
    service = stops_module.provider_stops(first)
    assert stops_module.provider_stops(first) is service
    assert stops_module.provider_stops(second) is not service


def test_provider_service_registry_does_not_retain_writer() -> None:
    @dataclass
    class Writer:
        value: int = 1

    writer: Any = Writer()
    identity = id(writer)
    weak_writer = ref(writer)
    service = stops_module.provider_stops(writer)
    del writer
    gc.collect()
    assert weak_writer() is None
    assert identity not in stops_module._services
    with pytest.raises(RuntimeError, match="provider owner is closed"):
        service._writer()
