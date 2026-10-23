"""Offline conformance owns acquisition interruption and late release."""

import asyncio
import threading
from typing import Any

import anyio
import pytest
from _fakes.lease_acquisition import HeldOwnerResponse
from _fakes.tools import GOAL, NoopManager

from swarm_arena import FunctionTool, Limits, ProgramJudge, RunConfig
from swarm_arena._engine import run as run_module
from swarm_arena._engine import stops as stops_module
from swarm_arena._store.local import LocalStore
from swarm_arena.testing import check_judge, check_manager, make_candidate
from swarm_arena.tool import ToolOutput


@pytest.mark.parametrize("component", ["judge", "manager"])
@pytest.mark.parametrize("stop", ["deadline", "stop", "cancel"])
async def test_conformance_acquisition_supervision(
    monkeypatch: pytest.MonkeyPatch, component: str, stop: str
) -> None:
    workers: list[threading.Thread] = []
    held = HeldOwnerResponse()

    def wait_for_release() -> None:
        workers.append(threading.current_thread())
        held.held.entered.set()
        try:
            assert held.held.release.wait(300), "test did not release owner response"
        finally:
            held.held.exited.set()

    monkeypatch.setattr(held.held, "wait", wait_for_release)
    held.install(monkeypatch)
    setups: list[bool] = []
    calls: list[bool] = []

    async def manager_call(self: NoopManager, ctx: Any) -> Any:
        calls.append(True)
        raise AssertionError("manager must not run")

    async def judge_call(self: ProgramJudge, *args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        raise AssertionError("judge must not run")

    monkeypatch.setattr(NoopManager, "run_round", manager_call)
    monkeypatch.setattr(ProgramJudge, "judge", judge_call)
    monkeypatch.setattr(run_module, "_run_setups", [lambda config: setups.append(True)])
    config = RunConfig(
        limits=Limits(run_timeout_s=1 if stop == "deadline" else 300),
        settle_timeout_s=2,
    )
    deadline = anyio.Event()

    class ClockBackend:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        async def sleep(self, seconds: float) -> None:
            assert seconds == config.limits.run_timeout_s
            await deadline.wait()

    if stop == "deadline":
        monkeypatch.setattr(stops_module, "monotonic", lambda: 100.0)
        monkeypatch.setattr(run_module, "anyio", ClockBackend())

    async def invoke() -> None:
        if component == "judge":
            await check_judge(
                ProgramJudge("_fakes.tools:score_min_y"),
                GOAL,
                [make_candidate(ToolOutput(status="ok"), goal=GOAL)],
                config=config,
            )
        else:
            await check_manager(
                NoopManager(),
                goal=GOAL,
                tool=FunctionTool("_fakes.tools:square"),
                config=config,
            )

    task = asyncio.create_task(invoke())
    try:
        with anyio.fail_after(60):
            assert await anyio.to_thread.run_sync(held.held.entered.wait, 60)
        if stop == "deadline":
            deadline.set()
        elif stop == "stop":
            LocalStore.create(held.stores[0], "control/stop-e1", b"")
        elif stop == "cancel":
            task.cancel()
        with anyio.fail_after(30):
            await asyncio.wait({task})
        with pytest.raises(
            asyncio.CancelledError if stop == "cancel" else RuntimeError,
            match=None if stop == "cancel" else "^offline harness stopped$",
        ):
            task.result()
        assert not held.held.release.is_set()
        assert not held.held.exited.is_set()
        assert setups == calls == []
    finally:
        held.held.release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(60):
            await asyncio.gather(task, return_exceptions=True)
            assert await anyio.to_thread.run_sync(held.held.exited.wait, 60)
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()
    await held.assert_released()
    assert setups == calls == []
