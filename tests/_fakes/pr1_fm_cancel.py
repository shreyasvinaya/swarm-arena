"""Readiness-gated native cancellation of a public tool action."""

import asyncio
from contextlib import suppress
from pathlib import Path

import anyio
from _fakes.pr1_fm import BUDGETS, ENTERED, EXITED, RELEASE, PaidTool

from swarm_arena.manager import AgentContext, Manager, RoundReport
from swarm_arena.tool import Command, ToolRequest


class PreparingTool(PaidTool):
    def command(self, request: ToolRequest, workspace: Path) -> Command:
        if request.inputs["n"] == 3:
            ENTERED.set()
            try:
                RELEASE.wait()
            finally:
                EXITED.set()
        return super().command(request, workspace)


class CancelPreparation(Manager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        BUDGETS.append(ctx.budget.usd)
        task = asyncio.create_task(ctx.run_tool({"n": 3}))
        try:
            with anyio.fail_after(10):
                assert await anyio.to_thread.run_sync(ENTERED.wait, 10)
            task.cancel("caller cancellation")
            try:
                await task
            except asyncio.CancelledError as error:
                assert str(error) == "caller cancellation"  # noqa: PT017
            else:
                raise AssertionError("native cancellation was swallowed")
        finally:
            RELEASE.set()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            assert await anyio.to_thread.run_sync(EXITED.wait, 10)
        BUDGETS.append(ctx.budget.usd)
        await ctx.run_tool({"n": 4})
        return RoundReport()
