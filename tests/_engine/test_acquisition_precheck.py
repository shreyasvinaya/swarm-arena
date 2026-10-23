"""An owner with a previously latched stop never writes an owner document."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from swarm_arena import RunConfig
from swarm_arena._engine import run as run_module
from swarm_arena._engine.resume import _supervised_acquire
from swarm_arena._engine.stops import StopState
from swarm_arena._journal.lease import Lease
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("flag", ["stop", "stop_now", "run_timeout"])
async def test_latched_stop_refuses_before_owner_write(
    tmp_path: Path, flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalStore(tmp_path / "run")
    stops = StopState()
    setattr(stops, flag, True)
    writes = []
    supervisors = []
    original = run_module._supervise_initialization

    @asynccontextmanager
    async def supervise(*args: Any, **kwargs: Any) -> AsyncGenerator[None]:
        supervisors.append(True)
        async with original(*args, **kwargs):
            yield

    monkeypatch.setattr(run_module, "_supervise_initialization", supervise)

    def acquire() -> Lease:
        writes.append(True)
        return Lease.acquire_new(store, owner_id="test", host="test", ttl_s=60)

    with pytest.raises(RuntimeError, match="owner already stopped"):
        await _supervised_acquire(
            store,
            1,
            stops,
            None,
            RunConfig(),
            acquire,
            stopped="owner already stopped",
            retain=False,
        )
    assert supervisors == []
    assert writes == []
    assert not (tmp_path / "run" / "owner.json").exists()
