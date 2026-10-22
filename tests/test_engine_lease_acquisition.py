"""Public owners supervise storage responses before lease acquisition returns."""

import asyncio
from pathlib import Path
from threading import Event
from typing import Any

import anyio
import pytest
from _fakes.finished import FinishedRun
from _fakes.llm_run import HeldOperation
from _fakes.run import plan

from swarm_arena import RunConfig, Swarm
from swarm_arena._engine import run as run_module
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._model.canonical import utc_now
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("component", ["run", "resume"])
@pytest.mark.parametrize("stop", ["deadline", "stop", "cancel"])
async def test_engine_acquisition_supervision(
    monkeypatch: pytest.MonkeyPatch,
    run_config: RunConfig,
    finished_run: FinishedRun,
    component: str,
    stop: str,
) -> None:
    held, released = HeldOperation(), Event()
    create, replace, release = LocalStore.create, LocalStore.replace, Lease.release
    stores: list[LocalStore] = []
    setups: list[bool] = []

    def hold_owner(store: LocalStore, key: str, data: bytes) -> Any:
        result = create(store, key, data)
        if key == "control/owner.json":
            stores.append(store)
            held.wait()
        return result

    def hold_takeover(store: LocalStore, key: str, data: bytes, **kwargs: Any) -> Any:
        result = replace(store, key, data, **kwargs)
        if key == "control/owner.json":
            stores.append(store)
            held.wait()
        return result

    def release_owner(lease: Lease) -> None:
        release(lease)
        released.set()

    monkeypatch.setattr(
        LocalStore,
        "create" if component == "run" else "replace",
        hold_owner if component == "run" else hold_takeover,
    )
    monkeypatch.setattr(Lease, "release", release_owner)
    monkeypatch.setattr(run_module, "_run_setups", [lambda config: setups.append(True)])
    config = run_config if component == "run" else finished_run.config
    elapsed = 0.0
    if component == "resume":
        committed = read_committed(LocalStore(Path(config.run_root) / finished_run.run_id))
        assert committed is not None
        elapsed = RunState.from_events(committed[1]).run_elapsed_s
    config = config.model_copy(
        update={
            "limits": config.limits.model_copy(
                update={"run_timeout_s": elapsed + 1 if stop == "deadline" else 30}
            ),
            "settle_timeout_s": 2,
        }
    )
    swarm = Swarm(config)
    operation = (
        swarm.run(plan()) if component == "run" else swarm.resume(finished_run.run_id, force=True)
    )
    task = asyncio.create_task(operation)
    try:
        await held.ready()
        if stop == "stop":
            create(stores[0], "control/stop-e1" if component == "run" else "control/stop-e2", b"")
        elif stop == "cancel":
            task.cancel()
        with (
            anyio.fail_after(4),
            pytest.raises(
                asyncio.CancelledError if stop == "cancel" else RuntimeError,
                match=None if stop == "cancel" else "initialization stopped",
            ),
        ):
            await asyncio.shield(task)
        assert setups == []
    finally:
        if not task.done():
            task.cancel()
        await held.finish()
        await asyncio.gather(task, return_exceptions=True)
    assert await anyio.to_thread.run_sync(released.wait, 5)
    owner = read_owner(stores[0])
    assert owner is not None
    assert owner[0].expires_at <= utc_now()
    assert setups == []
