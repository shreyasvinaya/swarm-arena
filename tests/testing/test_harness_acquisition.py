"""Offline conformance owns acquisition interruption and late release."""

import asyncio
from typing import Any

import anyio
import pytest
from _fakes.lease_acquisition import HeldOwnerResponse
from _fakes.tools import GOAL, NoopManager

from swarm_arena import FunctionTool, Limits, ProgramJudge, RunConfig
from swarm_arena._engine import run as run_module
from swarm_arena._store.local import LocalStore
from swarm_arena.testing import check_judge, check_manager, make_candidate
from swarm_arena.tool import ToolOutput


@pytest.mark.parametrize("component", ["judge", "manager"])
@pytest.mark.parametrize("stop", ["deadline", "stop", "cancel"])
async def test_conformance_acquisition_supervision(
    monkeypatch: pytest.MonkeyPatch, component: str, stop: str
) -> None:
    held = HeldOwnerResponse()
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
        limits=Limits(run_timeout_s=1 if stop == "deadline" else 30),
        settle_timeout_s=2,
    )

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
        await held.ready()
        if stop == "stop":
            LocalStore.create(held.stores[0], "control/stop-e1", b"")
        elif stop == "cancel":
            task.cancel()
        with (
            anyio.fail_after(4),
            pytest.raises(
                asyncio.CancelledError if stop == "cancel" else RuntimeError,
                match=None if stop == "cancel" else "^offline harness stopped$",
            ),
        ):
            await asyncio.shield(task)
        assert setups == calls == []
    finally:
        await held.finish(task)
    await held.assert_released()
    assert setups == calls == []
