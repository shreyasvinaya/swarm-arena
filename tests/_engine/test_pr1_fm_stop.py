"""Immediate stop and round deadline abandon unpaid request estimates."""

import asyncio
from pathlib import Path

import anyio
import pytest
from _fakes import pr1_fm as hooks
from _fakes.cleanup import settle_tasks
from _fakes.pr1_fm_stop import WORKERS

from swarm_arena import Swarm
from swarm_arena._engine import round as round_module
from tests._engine.test_extension_boundaries import _RoundClock
from tests._engine.test_pr1_fm_outcomes import config, events, plan


@pytest.mark.parametrize("hook", ["BlockingEstimate", "BlockingProviders"])
@pytest.mark.parametrize("stop", ["now", "deadline"])
async def test_unpaid_request_hook_stops_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook: str, stop: str
) -> None:
    WORKERS.clear()
    hooks.ENTERED.clear()
    hooks.RELEASE.clear()
    hooks.EXITED.clear()
    hooks.PEER_READY = None
    p = plan("manager", "group")
    peer = p.teams[0].agents[1]
    peer = peer.model_copy(
        update={"tool": peer.tool.model_copy(update={"path": f"_fakes.pr1_fm_stop:{hook}"})}
    )
    p = p.model_copy(update={"teams": [p.teams[0].model_copy(update={"agents": [peer]})]})
    cfg = config(tmp_path).model_copy(update={"settle_timeout_s": 60.0})
    clock = _RoundClock(cfg.limits.effective_round_timeout_s())
    if stop == "deadline":
        monkeypatch.setattr(round_module, "anyio", clock)
    swarm = Swarm(cfg)
    task = asyncio.create_task(swarm.run(p, run_id="r"))
    try:
        with anyio.fail_after(60):
            assert await anyio.to_thread.run_sync(hooks.ENTERED.wait, 60)
        if stop == "now":
            await swarm.stop(now=True)
        else:
            clock.advance(cfg.limits.effective_round_timeout_s() + 1)
        with anyio.fail_after(30):
            await asyncio.shield(task)
        assert not hooks.RELEASE.is_set()
    finally:
        hooks.RELEASE.set()
        if not task.done():
            task.cancel()
        try:
            await settle_tasks(task)
        finally:
            for worker in WORKERS:
                await anyio.to_thread.run_sync(worker.join, 60)
                assert not worker.is_alive()
    journal = events(tmp_path)
    assert not [e for e in journal if e["type"] in {"reserve", "tool_start"}]
    assert any(e["type"] == "goal_finished" for e in journal)
