"""Public extension fixtures for PR1 outcome and reservation regressions."""

import asyncio
import threading
from pathlib import Path
from typing import Any

import anyio
from _fakes.cleanup import settle_tasks
from pydantic import BaseModel

from swarm_arena import Goal, Score, ScriptTool
from swarm_arena._errors import Fenced, _core_error
from swarm_arena.board import Candidate
from swarm_arena.manager import AgentContext, Manager, RoundReport
from swarm_arena.scoring import Check, CheckContext, CheckResult
from swarm_arena.tool import Action, Command, CompletedRun, Estimate, ToolOutput, ToolRequest, Usage

MARK = "fm-secret-c7e51980"
PEER_READY: anyio.Event | None = None
NATIVE_READY: anyio.Event | None = None
ENTERED = threading.Event()
RELEASE = threading.Event()
EXITED = threading.Event()
BUDGETS: list[float | None] = []


def failure(kind: str) -> BaseException:
    error = ValueError(MARK)
    error.add_note(MARK)
    error.__cause__ = RuntimeError(MARK)
    if kind == "cancel":
        return asyncio.CancelledError(MARK)
    if kind == "fenced_group":
        return BaseExceptionGroup(
            MARK,
            [
                asyncio.CancelledError(MARK),
                ExceptionGroup(MARK, [_core_error(Fenced, "run ownership lost"), error]),
            ],
        )
    group = BaseExceptionGroup(MARK, [asyncio.CancelledError(MARK), error])
    group.add_note(MARK)
    group.__cause__ = RuntimeError(MARK)
    return group


def score(goal: Goal, candidate: Candidate) -> Score:
    return Score.maximize(1.0)


class Peer(Manager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        global PEER_READY
        if PEER_READY is None:
            PEER_READY = anyio.Event()
        PEER_READY.set()
        await ctx.run_tool({"n": 4})
        return RoundReport(summary="peer completed")


class Broken(Manager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        global PEER_READY
        if PEER_READY is None:
            PEER_READY = anyio.Event()
        await PEER_READY.wait()
        kind = str(ctx.goal.inputs["kind"])
        if ctx.goal.inputs["where"] == "manager":
            raise failure(kind)
        if ctx.goal.inputs["where"] == "action":
            if kind == "replace_cancel":
                global NATIVE_READY
                NATIVE_READY = anyio.Event()
                task = asyncio.create_task(ctx.call_action("broken", {}))
                try:
                    await NATIVE_READY.wait()
                    task.cancel("caller cancellation")
                    try:
                        await task
                    except asyncio.CancelledError as error:
                        import traceback

                        rendered = "".join(traceback.format_exception(error))
                        assert MARK not in rendered
                    else:
                        raise AssertionError("caller cancellation was swallowed")
                    await ctx.save_notes("action contained")
                    return RoundReport()
                finally:
                    if not task.done():
                        task.cancel()

                        await settle_tasks(task)
            try:
                await ctx.call_action("broken", {})
            except Fenced:
                raise
            except Exception as error:
                import traceback

                assert MARK not in "".join(traceback.format_exception(error))  # noqa: PT017
                await ctx.save_notes("action contained")
                return RoundReport()
        await ctx.run_tool({"n": 3, "kind": kind, "where": ctx.goal.inputs["where"]})
        return RoundReport()


class Inputs(BaseModel):
    n: int = 0
    kind: str = ""
    where: str = ""


class PaidTool(ScriptTool):
    def get_input_model(self) -> type[BaseModel]:
        return Inputs

    def estimate(self, request: ToolRequest) -> Estimate:
        if request.inputs.get("where") == "estimate":
            raise failure(str(request.inputs["kind"]))
        return Estimate(tokens=7, usd=0.000007)

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        if request.inputs.get("where") == "command":
            raise failure(str(request.inputs["kind"]))
        return super().command(request, workspace)

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        if request.inputs.get("where") == "collect":
            raise failure(str(request.inputs["kind"]))
        output = super().collect(request, workspace, run)
        return output.model_copy(update={"usage": Usage(input_tokens=4, usd=0.000004)})

    def extra_actions(self) -> tuple[Action[Any], Action[Any]]:
        async def broken(ctx: AgentContext, args: BaseModel) -> Any:
            if ctx.goal.inputs["kind"] == "replace_cancel":
                assert NATIVE_READY is not None
                NATIVE_READY.set()
                try:
                    await anyio.sleep_forever()
                except asyncio.CancelledError:
                    raise failure("cancel") from ValueError(MARK)
            raise failure(str(ctx.goal.inputs["kind"]))

        return (
            Action(name="broken", description="failure probe", input_model=Inputs, handler=broken),
            super().extra_actions()[0],
        )


class BrokenCheck(Check):
    name = "broken"

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        if candidate.agent == "bad":
            raise failure(str(ctx.goal.inputs["kind"]))
        return CheckResult(check=self.name, passed=True)
