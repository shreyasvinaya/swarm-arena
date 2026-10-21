"""Lease-held startup stops before a blocked worker can publish later stages."""

import asyncio
import threading
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.run import plan

from swarm_arena import RunConfig, Swarm
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._engine import run as run_module
from swarm_arena._engine.control import request_stop
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
        assert task.result() == {}
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
        assert not store.exists("review")
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
