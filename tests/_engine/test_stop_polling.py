"""Stop polling shares its owner's bounded lifetime in startup and driving."""

import asyncio
import threading
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.run import plan

from swarm_arena import RunConfig, Swarm
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._engine import run as run_module
from swarm_arena._engine.round import EngineDeps
from swarm_arena._engine.run import RunRunner
from swarm_arena._errors import Fenced, StorageError, _is_core_signal
from swarm_arena._journal.lease import read_owner
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.results import Candidate
from swarm_arena._outcomes import _has_core_fenced
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("phase", ["startup", "driver"])
@pytest.mark.parametrize("signal", ["cancel", "deadline"])
async def test_blocked_stop_poll_releases_owner(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, phase: str, signal: str
) -> None:
    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 2.0})
    poll_entered, work_entered, release, driving = (threading.Event() for _ in range(4))
    workers: list[threading.Thread] = []
    deadline = anyio.Event()
    original_drive = RunRunner._drive
    original_create, original_exists = LocalStore.create, LocalStore.exists
    original_frontier = ArenaBoard.frontier

    class ClockBackend:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        async def sleep(self, seconds: float) -> None:
            await deadline.wait()

    async def drive(runner: RunRunner, deps: EngineDeps, epoch: int) -> dict[str, list[Candidate]]:
        driving.set()
        return await original_drive(runner, deps, epoch)

    def hold(entered: threading.Event) -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(300), "controlled worker release deadline"

    def exists(store: LocalStore, key: str) -> bool:
        if key == "control/stop-now-e1" and (phase == "startup" or driving.is_set()):
            hold(poll_entered)
        return original_exists(store, key)

    def create(store: LocalStore, key: str, data: bytes) -> str:
        if phase == "startup" and key == "manifest.json":
            hold(work_entered)
        return original_create(store, key, data)

    def frontier(*args: Any, **kwargs: Any) -> Any:
        if phase == "driver" and driving.is_set():
            hold(work_entered)
        return original_frontier(*args, **kwargs)

    monkeypatch.setattr(run_module, "anyio", ClockBackend())
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    monkeypatch.setattr(RunRunner, "_drive", drive)
    monkeypatch.setattr(LocalStore, "exists", exists)
    monkeypatch.setattr(LocalStore, "create", create)
    monkeypatch.setattr(ArenaBoard, "frontier", frontier)
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    try:
        with anyio.fail_after(60):
            assert await anyio.to_thread.run_sync(work_entered.wait, 60)
            assert await anyio.to_thread.run_sync(poll_entered.wait, 60)
        if signal == "cancel":
            task.cancel()
        else:
            deadline.set()
        done, _ = await asyncio.wait([task], timeout=30)
        assert done, "owner waited for its blocked stop lookup"
        assert not release.is_set()
        if signal == "cancel":
            assert task.cancelled()
        elif phase == "startup":
            error = task.exception()
            assert isinstance(error, RuntimeError)
            assert str(error) == "run initialization stopped before run_started"
        else:
            assert task.result() == {"g": []}
        owner = read_owner(LocalStore(Path(config.run_root) / "r"))
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        _, pending_cleanup = await asyncio.wait([task], timeout=60)
        if task.done() and not task.cancelled():
            task.exception()
        for worker in workers:
            if worker.name.startswith("swarm-arena-daemon-"):
                await anyio.to_thread.run_sync(worker.join, 60)
    assert not pending_cleanup, "stop owner did not finish cleanup"
    assert all(
        not worker.is_alive() for worker in workers if worker.name.startswith("swarm-arena-daemon-")
    )


@pytest.mark.parametrize("phase", ["startup", "driver"])
@pytest.mark.parametrize("key", ["control/stop-now-e1", "control/stop-e1"])
@pytest.mark.parametrize("kind", ["ordinary", "group", "cancel", "fenced", "truth", "integer"])
async def test_stop_poll_failure_reaches_owner(
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    key: str,
    kind: str,
) -> None:
    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 2.0})
    work_entered, release, driving = (threading.Event() for _ in range(3))
    workers: list[threading.Thread] = []
    calls: list[str] = []
    truth_calls: list[int] = []
    marker = "private-stop-poll-backend-value"
    original_drive = RunRunner._drive
    original_create, original_exists = LocalStore.create, LocalStore.exists
    original_frontier = ArenaBoard.frontier

    class Truth:
        def __bool__(self) -> bool:
            truth_calls.append(threading.get_ident())
            raise ValueError(marker)

    async def drive(runner: RunRunner, deps: EngineDeps, epoch: int) -> dict[str, list[Candidate]]:
        driving.set()
        return await original_drive(runner, deps, epoch)

    def hold() -> None:
        workers.append(threading.current_thread())
        work_entered.set()
        assert release.wait(300), "controlled worker release deadline"

    def create(store: LocalStore, current: str, data: bytes) -> str:
        if phase == "startup" and current == "manifest.json":
            hold()
        return original_create(store, current, data)

    def frontier(*args: Any, **kwargs: Any) -> Any:
        if phase == "driver" and driving.is_set():
            hold()
        return original_frontier(*args, **kwargs)

    def exists(store: LocalStore, current: str) -> Any:
        if current != key or (phase == "driver" and not driving.is_set()):
            return original_exists(store, current)
        assert work_entered.wait(60)
        calls.append(current)
        if kind == "truth":
            return Truth()
        if kind == "integer":
            return 1
        error: BaseException
        if kind == "ordinary":
            error = OSError(marker)
        elif kind == "group":
            error = BaseExceptionGroup(marker, [asyncio.CancelledError(marker), OSError(marker)])
        elif kind == "cancel":
            error = asyncio.CancelledError(marker)
        else:
            error = Fenced(marker)
        error.add_note(marker)
        raise error

    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    monkeypatch.setattr(RunRunner, "_drive", drive)
    monkeypatch.setattr(LocalStore, "exists", exists)
    monkeypatch.setattr(LocalStore, "create", create)
    monkeypatch.setattr(ArenaBoard, "frontier", frontier)
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    try:
        with anyio.fail_after(60):
            assert await anyio.to_thread.run_sync(work_entered.wait, 60)
        done, _ = await asyncio.wait([task], timeout=30)
        assert done, "polling failure silently removed the watcher"
        assert not task.cancelled(), "backend cancellation impersonated caller cancellation"
        error = task.exception()
        assert error is not None
        assert marker not in "".join(traceback.format_exception(error))
        assert not _has_core_fenced(error)
        pending = [error]
        while pending:
            current = pending.pop()
            if isinstance(current, BaseExceptionGroup):
                pending.extend(current.exceptions)
            else:
                assert isinstance(current, StorageError)
                assert _is_core_signal(current)
        assert calls == [key]
        assert not truth_calls
        owner = read_owner(LocalStore(Path(config.run_root) / "r"))
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        _, pending_cleanup = await asyncio.wait([task], timeout=60)
        if task.done() and not task.cancelled():
            task.exception()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
    assert not pending_cleanup, "stop owner did not finish cleanup"
    assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("mode", ["run", "resume", "resume_retry"])
async def test_one_stop_watcher_spans_acquisition_setup_and_driver(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = run_config
    swarm = Swarm(config)
    if mode != "run":
        await swarm.run(plan(rounds=1), run_id="r")
    from swarm_arena._errors import _core_error
    from swarm_arena._journal.lease import Lease

    attempts: list[bool] = []
    original_take = Lease.take_over

    def take(*args: Any, **kwargs: Any) -> Lease:
        attempts.append(True)
        if len(attempts) == 1:
            raise _core_error(Fenced)
        return original_take(*args, **kwargs)

    if mode == "resume_retry":
        monkeypatch.setattr(Lease, "take_over", take)
    original = run_module.watch_stop_files
    started: list[object] = []
    live: set[asyncio.Task[Any]] = set()
    setup_watchers: list[int] = []

    async def watch(store: Any, epoch: int, stops: Any) -> None:
        task = asyncio.current_task()
        assert task is not None
        started.append(stops)
        live.add(task)
        try:
            await original(store, epoch, stops, interval_s=0.01)
        finally:
            live.remove(task)

    monkeypatch.setattr(run_module, "watch_stop_files", watch)
    monkeypatch.setattr(
        run_module, "_run_setups", [lambda config: setup_watchers.append(len(live))]
    )
    if mode != "run":
        await Swarm(config).resume("r", force=True)
    else:
        await swarm.run(plan(rounds=1), run_id="r")
    assert len(started) == 1
    assert setup_watchers == [1]
    assert not live
    if mode == "resume_retry":
        assert len(attempts) == 2


async def test_stop_at_initialization_entry_has_stopped_outcome(
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _fakes.lease_acquisition import stop_at_phase_entry

    trigger, latched = stop_at_phase_entry(monkeypatch)
    enter = run_module._RunJournal.__aenter__

    async def entered(journal: run_module._RunJournal) -> run_module._RunJournal:
        result = await enter(journal)
        trigger.set()
        return result

    monkeypatch.setattr(run_module._RunJournal, "__aenter__", entered)
    with (
        anyio.fail_after(60),
        pytest.raises(
            RuntimeError,
            match=r"^run initialization stopped before run_started$",
        ),
    ):
        await Swarm(run_config).run(plan(rounds=1), run_id="r")
    assert latched.is_set()
    owner = read_owner(LocalStore(Path(run_config.run_root) / "r"))
    assert owner is not None
    assert owner[0].expires_at <= utc_now()
    assert owner[0].head is None
