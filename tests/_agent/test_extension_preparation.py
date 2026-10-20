"""Request-time synchronous extension hooks remain outside the event loop."""

import threading
from typing import Any

import pytest
from _fakes.engine import CtxEnv, RtEnv
from pydantic import BaseModel, field_serializer, field_validator

from swarm_arena._agent.tool_run import execute_tool_run
from swarm_arena._model.base import FrozenModel
from swarm_arena.tool import Action


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
    import anyio

    from swarm_arena.manager import LimitReached

    entered, done = anyio.Event(), anyio.Event()
    release = threading.Event()
    token = anyio.lowlevel.current_token()
    handled: list[bool] = []
    errors: list[LimitReached] = []

    class Inputs(FrozenModel):
        x: int

        @field_validator("x")
        @classmethod
        def validate_x(cls, value: int) -> int:
            anyio.from_thread.run_sync(entered.set, token=token)
            release.wait()
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

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(invoke)
        try:
            with anyio.fail_after(3):
                await entered.wait()
                ctx_env.env.inflight.cancel_all()
                await done.wait()
            assert [error.limit for error in errors] == ["stop"]
            assert not handled
        finally:
            release.set()
