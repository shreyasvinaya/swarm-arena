"""Exercise actual page storage from public run and resume boundaries."""

import asyncio
import threading
from pathlib import Path
from typing import Any

import anyio
import pytest
from _engine.test_resume import _finished_without_round_page

from swarm_arena import RunConfig, Swarm
from swarm_arena._journal.lease import read_owner
from swarm_arena._model.canonical import utc_now
from swarm_arena._store.local import LocalStore

PAGE = "goals/g/rounds/round-01.md"


@pytest.mark.parametrize("site", ["exists", "create"])
async def test_resume_page_storage_abandons_cancelled_owner(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    config, store = await _finished_without_round_page(tmp_path, run_config)
    config = config.model_copy(update={"settle_timeout_s": 0.2})
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = getattr(LocalStore, site)

    def blocked(self: LocalStore, key: str, *args: Any) -> Any:
        if key != PAGE:
            return original(self, key, *args)
        workers.append(threading.current_thread())
        entered.set()
        try:
            assert release.wait(20)
            return original(self, key, *args)
        finally:
            exited.set()

    monkeypatch.setattr(LocalStore, site, blocked)
    task = asyncio.create_task(Swarm(config).resume("repair", force=True))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 10)
        services = [t for t in asyncio.all_tasks() if t.get_name() == "resume-repair"]
        assert services
        task.cancel()
        done, _ = await asyncio.wait([task, *services], timeout=3)
        assert len(done) == len(services) + 1
        assert not exited.is_set()
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        assert await anyio.to_thread.run_sync(exited.wait, 5)
        for worker in workers:
            if worker.name.startswith("swarm-arena-daemon-"):
                await anyio.to_thread.run_sync(worker.join, 5)
    assert read_owner(store) == owner
    if site == "exists":
        assert not (Path(store.url) / PAGE).exists()
