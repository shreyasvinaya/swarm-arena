"""Hold real owner writes and observe their eventual release across public owners."""

import asyncio
from dataclasses import dataclass, field
from functools import partial
from threading import Event
from typing import Any

import anyio
import pytest
from _fakes.llm_run import HeldOperation

from swarm_arena._engine import run as run_module
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._model.canonical import utc_now
from swarm_arena._store.local import LocalStore


@dataclass
class HeldOwnerResponse:
    """Hold one committed owner response until explicitly released and observed."""

    held: HeldOperation = field(default_factory=HeldOperation)
    released: Event = field(default_factory=Event)
    stores: list[LocalStore] = field(default_factory=list)

    def install(self, monkeypatch: pytest.MonkeyPatch, *, takeover: bool = False) -> None:
        """Patch acquisition only, leaving release and later publication unblocked."""
        operation = LocalStore.replace if takeover else LocalStore.create
        release = Lease.release

        def hold(store: LocalStore, key: str, data: bytes, **kwargs: Any) -> str:
            result = operation(store, key, data, **kwargs)
            if key == "control/owner.json" and not self.held.entered.is_set():
                self.stores.append(store)
                self.held.wait()
            return result

        def release_owner(lease: Lease) -> None:
            release(lease)
            self.released.set()

        monkeypatch.setattr(LocalStore, "replace" if takeover else "create", hold)
        monkeypatch.setattr(Lease, "release", release_owner)
        monkeypatch.setattr(
            run_module,
            "watch_stop_files",
            partial(run_module.watch_stop_files, interval_s=0.01),
        )

    async def ready(self) -> None:
        """Wait for the successful write whose response is held."""
        await self.held.ready()

    async def finish(self, task: asyncio.Task[Any]) -> None:
        """Release the worker and join the owner, including failure cleanup."""
        if not task.done():
            task.cancel()
        await self.held.finish()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)

    async def assert_released(self) -> None:
        """Observe late cleanup and an expired durable owner document."""
        assert await anyio.to_thread.run_sync(self.released.wait, 5)
        owner = read_owner(self.stores[0])
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
