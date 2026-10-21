"""Lease-held startup stops before a blocked worker can publish later stages."""

import asyncio
import threading
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.run import all_events, plan

from swarm_arena import RunConfig, Swarm
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._cli.main import _error_message
from swarm_arena._cli.run_cmds import _summaries
from swarm_arena._engine import run as run_module
from swarm_arena._engine.control import request_stop
from swarm_arena._errors import _is_core_signal
from swarm_arena._journal.lease import read_owner
from swarm_arena._model.canonical import utc_now
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("site", ["run.toml", "manifest.json", "board_open"])
@pytest.mark.parametrize("signal", ["stop", "stop_now", "deadline"])
async def test_initialization_stops_before_worker_release(
    tmp_path: Path,
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    signal: str,
) -> None:
    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 0.3})
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    deadline = anyio.Event()

    class ClockBackend:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        async def sleep(self, seconds: float) -> None:
            assert 0 < seconds <= config.limits.run_timeout_s
            await deadline.wait()

    monkeypatch.setattr(run_module, "anyio", ClockBackend())
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = ArenaBoard.open if site == "board_open" else LocalStore.create

    def block(*args: Any, **kwargs: Any) -> Any:
        if site == "board_open" or args[1] == site:
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(20)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ArenaBoard if site == "board_open" else LocalStore,
        "open" if site == "board_open" else "create",
        block,
    )
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    store = LocalStore(Path(config.run_root) / "r")
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(entered.wait, 10)
        if signal == "deadline":
            deadline.set()
        else:
            assert await request_stop(config.run_root, "r", now=signal == "stop_now") == 1
        done, _ = await asyncio.wait([task], timeout=2)
        assert done, "initialization ignored its stop/deadline while the worker was held"
        if site == "board_open":
            assert task.result() == {"g": []}
            events = all_events(config)
            assert any(e.type == "stop_requested" for e in events)
            for event_type in ("no_result", "goal_finished"):
                matching = [e for e in events if e.type == event_type]
                assert len(matching) == 1
                assert matching[0].goal_id == "g"
                assert matching[0].payload["reason"] == "stopped"
            assert _summaries(config.run_root, "r", ["g"]) == ["no result: g stopped"]
            assert store.exists("review")
        else:
            error = task.exception()
            assert isinstance(error, RuntimeError)
            assert _is_core_signal(error)
            assert _error_message(error) == "run initialization stopped before run_started"
            assert not store.exists("review")
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()
    assert not store.exists("goals/g/rounds/round-01.md")
    if site == "run.toml":
        assert not store.exists("manifest.json")
    if site != "board_open":
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].head is None
    else:
        assert await Swarm(config).resume("r", force=True) == {"g": []}
        assert not any(e.type == "round_started" for e in all_events(config))


async def test_initialization_stop_cleanup_is_bounded(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 0.3})
    board_entered, write_entered, release = (threading.Event() for _ in range(3))
    workers: list[threading.Thread] = []
    original_open, original_create = ArenaBoard.open, LocalStore.create

    def hold(entered: threading.Event) -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(20)

    def open_board(*args: Any, **kwargs: Any) -> Any:
        hold(board_entered)
        return original_open(*args, **kwargs)

    def create(store: LocalStore, key: str, data: bytes) -> str:
        if key == "goals/g/solution.json":
            hold(write_entered)
        return original_create(store, key, data)

    monkeypatch.setattr(ArenaBoard, "open", open_board)
    monkeypatch.setattr(LocalStore, "create", create)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    store = LocalStore(Path(config.run_root) / "r")
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(board_entered.wait, 8)
            await request_stop(config.run_root, "r", now=True)
            assert await anyio.to_thread.run_sync(write_entered.wait, 8)
        done, _ = await asyncio.wait([task], timeout=2)
        assert done, "stop cleanup waited for a blocked solution write"
        error = task.exception()
        assert isinstance(error, RuntimeError)
        assert _error_message(error) == "initialization stop cleanup timed out"
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
    assert not store.exists("review")
