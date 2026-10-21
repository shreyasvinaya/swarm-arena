"""Actual backend failures and stalls at retained run I/O boundaries."""

import asyncio
import threading
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.run import plan
from _fakes.secrecy import assert_secret_absent

from swarm_arena import RunConfig, Swarm
from swarm_arena._errors import StorageError, _is_core_signal
from swarm_arena._journal.lease import read_owner
from swarm_arena._model.canonical import utc_now
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("site", ["run.toml", "manifest.json", "plan.r0.toml"])
@pytest.mark.parametrize("kind", ["ordinary", "cancel", "group"])
async def test_initialization_writes_contain_backend_outcomes(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, site: str, kind: str
) -> None:
    marker = "private-storage-value"
    original = LocalStore.create
    failures: list[str] = []

    def create(store: LocalStore, key: str, data: bytes) -> str:
        if key.endswith(site):
            failures.append(key)
            if kind == "cancel":
                raise asyncio.CancelledError(marker)
            if kind == "group":
                raise BaseExceptionGroup(marker, [OSError(marker), asyncio.CancelledError(marker)])
            raise OSError(marker)
        return original(store, key, data)

    monkeypatch.setattr(LocalStore, "create", create)
    with pytest.raises(StorageError) as caught:
        await Swarm(run_config).run(plan(rounds=1), run_id="r")
    assert failures
    assert _is_core_signal(caught.value)
    assert_secret_absent(caught.value, marker)
    owner = read_owner(LocalStore(Path(run_config.run_root) / "r"))
    assert owner is not None
    assert owner[0].expires_at <= utc_now()


async def test_completion_export_observes_remaining_run_time(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine import run as run_module

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = LocalStore.create
    deadline = anyio.Event()

    class Backend:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        async def sleep(self, seconds: float) -> None:
            await deadline.wait()

    def create(store: LocalStore, key: str, data: bytes) -> str:
        if key.startswith("review/"):
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(30)
        return original(store, key, data)

    config = run_config.model_copy(update={"settle_timeout_s": 2.0})
    monkeypatch.setattr(LocalStore, "create", create)
    monkeypatch.setattr(run_module, "anyio", Backend())
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    try:
        with anyio.fail_after(15):
            assert await anyio.to_thread.run_sync(entered.wait, 12)
        deadline.set()
        done, _ = await asyncio.wait([task], timeout=3)
        assert done, "completion export ignored the run deadline"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
    assert len(workers) == 1, "abandoned export started another file"


@pytest.mark.parametrize("signal", ["stop", "deadline"])
async def test_driver_signal_abandons_journal_publication(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, signal: str
) -> None:
    from functools import partial

    from swarm_arena._engine import run as run_module
    from swarm_arena._engine.control import request_stop
    from swarm_arena._engine.round import EngineDeps

    entered, release = threading.Event(), threading.Event()
    deadline = anyio.Event()
    workers: list[threading.Thread] = []
    original = LocalStore.create
    config = run_config.model_copy(update={"lease_ttl_s": 60, "settle_timeout_s": 0.5})

    def create(store: LocalStore, key: str, data: bytes) -> str:
        if key.startswith("journal/") and b'"type":"goal_started"' in data:
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(60)
        return original(store, key, data)

    async def expire(runner: run_module.RunRunner, deps: EngineDeps) -> None:
        await deadline.wait()
        deps.stops.run_timeout = True

    monkeypatch.setattr(LocalStore, "create", create)
    monkeypatch.setattr(run_module.RunRunner, "_watch_clock", expire)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    store = LocalStore(Path(config.run_root) / "r")
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 15)
        before = read_owner(store)
        assert before is not None
        if signal == "stop":
            assert await request_stop(config.run_root, "r", now=False) == 1
        else:
            deadline.set()
        done, _ = await asyncio.wait([task], timeout=3)
        assert done, "journal publication ignored its owner's stop/deadline"
        error = task.exception()
        assert isinstance(error, RuntimeError)
        assert _is_core_signal(error)
        assert str(error) == "stop cleanup timed out"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(20):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head == before[0].head


async def test_graceful_stop_abandons_goal_solution_storage(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from functools import partial

    from swarm_arena._engine import run as run_module
    from swarm_arena._engine.control import request_stop

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = LocalStore.create
    config = run_config.model_copy(update={"lease_ttl_s": 60, "settle_timeout_s": 0.5})

    def create(store: LocalStore, key: str, data: bytes) -> str:
        if key == "goals/g/solution.json":
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(60)
        return original(store, key, data)

    monkeypatch.setattr(LocalStore, "create", create)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 20)
        await request_stop(config.run_root, "r", now=False)
        done, _ = await asyncio.wait([task], timeout=3)
        assert done, "graceful stop did not bound retained goal storage"
        error = task.exception()
        while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
            error = error.exceptions[0]
        assert isinstance(error, RuntimeError)
        assert _is_core_signal(error)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(20):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
    assert not LocalStore(Path(config.run_root) / "r").exists("review")
