"""Readiness-gated native cancellation of a public tool action."""

import asyncio
from contextlib import suppress
from pathlib import Path
from threading import Thread, current_thread

import anyio
from _fakes.pr1_fm import BUDGETS, ENTERED, EXITED, RELEASE, PaidTool

from swarm_arena.manager import AgentContext, Manager, RoundReport
from swarm_arena.tool import Command, ToolRequest

WORKERS: list[Thread] = []


class PreparingTool(PaidTool):
    def command(self, request: ToolRequest, workspace: Path) -> Command:
        if request.inputs["n"] == 3:
            WORKERS.append(current_thread())
            ENTERED.set()
            try:
                assert RELEASE.wait(300), "preparation release deadline"
            finally:
                EXITED.set()
        return super().command(request, workspace)


class CancelPreparation(Manager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        BUDGETS.append(ctx.budget.usd)
        task = asyncio.create_task(ctx.run_tool({"n": 3}))
        try:
            with anyio.fail_after(60):
                assert await anyio.to_thread.run_sync(ENTERED.wait, 60)
            task.cancel("caller cancellation")
            try:
                with anyio.fail_after(30):
                    await task
            except asyncio.CancelledError as error:
                assert error.args in ((), ("caller cancellation",))  # noqa: PT017
            else:
                raise AssertionError("native cancellation was swallowed")
        finally:
            RELEASE.set()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            assert await anyio.to_thread.run_sync(EXITED.wait, 60)
            for worker in WORKERS:
                await anyio.to_thread.run_sync(worker.join, 60)
                assert not worker.is_alive()
            WORKERS.clear()
        BUDGETS.append(ctx.budget.usd)
        await ctx.run_tool({"n": 4})
        return RoundReport()
