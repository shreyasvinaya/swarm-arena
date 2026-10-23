"""Request-time synchronous extension hooks remain outside the event loop."""

import asyncio
import threading
from typing import Any

import anyio
import pytest
from _fakes.cleanup import observe_task, settle_tasks
from _fakes.engine import CtxEnv, RtEnv
from pydantic import (
    BaseModel,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_serializer,
)

from swarm_arena._agent.tool_run import command_journal, execute_tool_run
from swarm_arena._model.base import FrozenModel
from swarm_arena.executor import Job
from swarm_arena.tool import Action, RunStatus, ToolOutput


@pytest.mark.parametrize("kind", ["tool", "command"])
async def test_executor_started_serializer_runs_in_daemon(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    observed: list[threading.Thread] = []

    class Started(Job.Started):
        @model_serializer(mode="wrap")
        def serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
            observed.append(threading.current_thread())
            return handler(self)

    if kind == "tool":

        async def submit(job: Job) -> ToolOutput:
            assert job.on_started is not None
            await job.on_started(Started(kind="dispatch", job_id=job.id))
            return ToolOutput(status=RunStatus.OK)

        monkeypatch.setattr(rt_env.rt.executor, "submit", submit)
        await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    else:
        _, on_started = command_journal(
            rt_env.rt.journal, rt_env.rt.executor, candidate_id="c1", component="check"
        )
        await on_started(Started(kind="dispatch", job_id="cmd1"))
    assert observed
    assert all(thread is not threading.main_thread() and thread.daemon for thread in observed)


async def test_graceful_stop_allows_started_callback_for_running_check(rt_env: RtEnv) -> None:
    _, on_started = command_journal(
        rt_env.rt.journal, rt_env.rt.executor, candidate_id="c1", component="check"
    )
    rt_env.inflight.close()
    await on_started(Job.Started(kind="dispatch", job_id="cmd1"))
    assert rt_env.events("command_start")[-1].payload["job_id"] == "cmd1"


async def test_immediate_stop_rejects_started_callback_for_running_check(rt_env: RtEnv) -> None:
    from swarm_arena.manager import LimitReached

    _, on_started = command_journal(
        rt_env.rt.journal, rt_env.rt.executor, candidate_id="c1", component="check"
    )
    rt_env.inflight.close()
    rt_env.inflight.cancel_all()
    with pytest.raises(LimitReached):
        await on_started(Job.Started(kind="dispatch", job_id="cmd1"))
    assert not rt_env.events("command_start")


@pytest.mark.parametrize(
    "hook", ["get_input_model", "_own_files", "_protected", "editable", "is_deterministic"]
)
async def test_request_hooks_run_in_daemon(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, hook: str
) -> None:
    observed: list[threading.Thread] = []
    original = getattr(rt_env.rt.tool, hook)

    def wrapped(*args: Any) -> Any:
        observed.append(threading.current_thread())
        return original(*args)

    monkeypatch.setattr(rt_env.rt.tool, hook, wrapped)
    await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert observed
    assert all(thread is not threading.main_thread() and thread.daemon for thread in observed)


@pytest.mark.parametrize("phase", ["validate", "serialize"])
async def test_tool_input_model_runs_in_daemon(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    observed: list[threading.Thread] = []

    class Inputs(FrozenModel):
        x: float

        @field_validator("x")
        @classmethod
        def validate_x(cls, value: float) -> float:
            if phase == "validate":
                observed.append(threading.current_thread())
            return value

        @field_serializer("x")
        def serialize_x(self, value: float) -> float:
            if phase == "serialize":
                observed.append(threading.current_thread())
            return value

    monkeypatch.setattr(rt_env.rt.tool, "get_input_model", lambda: Inputs)
    await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert observed
    assert all(thread is not threading.main_thread() and thread.daemon for thread in observed)


async def test_action_input_validator_runs_in_daemon(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[threading.Thread] = []

    class Inputs(FrozenModel):
        x: int

        @field_validator("x")
        @classmethod
        def validate_x(cls, value: int) -> int:
            observed.append(threading.current_thread())
            return value

    async def handler(ctx: Any, args: BaseModel) -> str:
        return "ok"

    monkeypatch.setattr(
        ctx_env.rt.tool,
        "extra_actions",
        lambda: [
            Action(name="validate", description="Validate", input_model=Inputs, handler=handler)
        ],
    )
    assert await ctx_env.ctx().call_action("validate", {"x": 1}) == "ok"
    assert observed
    assert all(thread is not threading.main_thread() and thread.daemon for thread in observed)


async def test_graceful_stop_keeps_checks_for_completed_tool(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.tools import AlwaysPass

    from swarm_arena.executor import Job
    from swarm_arena.tool import ToolOutput

    original = rt_env.rt.executor.submit

    async def submit(job: Job) -> ToolOutput:
        output = await original(job)
        rt_env.inflight.close()
        return output

    monkeypatch.setattr(rt_env.rt.executor, "submit", submit)
    runtime = rt_env.rt_with_checks([("pass", AlwaysPass())])
    candidate = await execute_tool_run(runtime, {"x": 1}, parents=(), replicate=None, edits=None)
    assert len(candidate.checks) == 1
    assert candidate.checks[0].passed


async def test_old_inflight_cancellation_does_not_skip_new_request_checks(rt_env: RtEnv) -> None:
    from _fakes.tools import AlwaysPass

    rt_env.inflight.cancel_all()
    runtime = rt_env.rt_with_checks([("pass", AlwaysPass())])
    candidate = await execute_tool_run(runtime, {"x": 1}, parents=(), replicate=None, edits=None)
    assert candidate.checks[0].passed


async def test_action_validator_abandoned_without_late_handler(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena.manager import LimitReached

    entered, done = anyio.Event(), anyio.Event()
    release = threading.Event()
    workers: list[threading.Thread] = []
    token = anyio.lowlevel.current_token()
    handled: list[bool] = []
    errors: list[LimitReached] = []

    class Inputs(FrozenModel):
        x: int

        @field_validator("x")
        @classmethod
        def validate_x(cls, value: int) -> int:
            workers.append(threading.current_thread())
            anyio.from_thread.run_sync(entered.set, token=token)
            assert release.wait(300), "test did not release the validator"
            return value

    async def handler(ctx: Any, args: BaseModel) -> str:
        handled.append(True)
        return "ok"

    monkeypatch.setattr(
        ctx_env.rt.tool,
        "extra_actions",
        lambda: [
            Action(name="validate", description="Validate", input_model=Inputs, handler=handler)
        ],
    )
    ctx = ctx_env.ctx()

    async def invoke() -> None:
        try:
            await ctx.call_action("validate", {"x": 1})
        except LimitReached as error:
            errors.append(error)
        finally:
            done.set()

    task = asyncio.create_task(invoke())
    try:
        with anyio.fail_after(60):
            await entered.wait()
            ctx_env.env.inflight.cancel_all()
            await done.wait()
        await observe_task(task)
        assert [error.limit for error in errors] == ["stop"]
        assert not handled
    finally:
        release.set()
        try:
            await settle_tasks(task)
        finally:
            with anyio.CancelScope(shield=True):
                for worker in workers:
                    await anyio.to_thread.run_sync(worker.join, 60)
                assert all(not worker.is_alive() for worker in workers)
