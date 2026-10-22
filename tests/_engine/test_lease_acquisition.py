"""Run and resume supervise committed owner responses before setup."""

import asyncio
from pathlib import Path

import anyio
import pytest
from _fakes.finished import FinishedRun
from _fakes.lease_acquisition import HeldOwnerResponse
from _fakes.run import plan

from swarm_arena import RunConfig, Swarm
from swarm_arena._engine import run as run_module
from swarm_arena._journal.lease import read_owner
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
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
    held = HeldOwnerResponse()
    held.install(monkeypatch, takeover=component == "resume")
    setups: list[bool] = []
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
            LocalStore.create(
                held.stores[0], "control/stop-e1" if component == "run" else "control/stop-e2", b""
            )
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
        await held.finish(task)
    await held.assert_released()
    if component == "run":
        owner = read_owner(held.stores[0])
        assert owner is not None
        assert owner[0].head is None
    assert setups == []
