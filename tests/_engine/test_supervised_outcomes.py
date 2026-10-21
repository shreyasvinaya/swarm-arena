"""Contain untrusted cancellation-shaped hook outcomes through public runs."""

import asyncio
import traceback
from typing import Literal
from uuid import uuid4

import anyio
import pytest
from _fakes.engine import EngineEnv
from _fakes.run import all_events, events, plan

from swarm_arena import Goal, Plan, RunConfig, Swarm
from swarm_arena._engine.goal import GoalRunner
from swarm_arena._errors import Fenced, _core_error, _rebuild_core_signal
from swarm_arena.manager import LimitReached
from swarm_arena.planning import History, Planner


@pytest.mark.parametrize("kind", ["ordinary", "mixed_group", "cancelled"])
async def test_public_replanner_contains_untrusted_outcomes(
    run_config: RunConfig, kind: Literal["ordinary", "mixed_group", "cancelled"]
) -> None:
    secret = str(uuid4())

    class FailingPlanner(Planner):
        async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
            error: BaseException
            if kind == "mixed_group":
                error = BaseExceptionGroup(
                    secret, [asyncio.CancelledError(secret), RuntimeError(secret)]
                )
            elif kind == "cancelled":
                error = asyncio.CancelledError(secret)
            else:
                error = RuntimeError(secret)
            error.add_note(secret)
            raise error from ValueError(secret)

    caught: BaseException | None = None
    result = None
    try:
        result = await Swarm(run_config, replanner=FailingPlanner()).run(plan())
    except BaseException as error:
        caught = error
    if caught is not None:
        assert secret not in str(caught)
        assert secret not in repr(caught)
        assert secret not in repr(caught.args)
        assert secret not in repr(getattr(caught, "__notes__", ()))
        assert secret not in repr(caught.__cause__)
        assert secret not in "".join(traceback.format_exception(caught))
    assert caught is None, "planner failures must be rejected without aborting the run"
    assert result is not None
    assert result["g"]
    assert secret not in repr(result)
    assert secret not in repr([event.model_dump() for event in all_events(run_config)])
    rejected = events(run_config, "plan_rejected")
    assert len(rejected) == 1
    assert rejected[0].payload["errors"][0].startswith("planner error: ")
    assert events(run_config, "goal_finished")[-1].payload["reason"] is None
    assert len(events(run_config, "round_closed")) == 2


async def test_public_replanner_preserves_real_caller_cancellation(run_config: RunConfig) -> None:
    entered = anyio.Event()

    class WaitingPlanner(Planner):
        async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
            entered.set()
            await anyio.sleep_forever()
            raise AssertionError("waiting planner resumed")

    task = asyncio.create_task(Swarm(run_config, replanner=WaitingPlanner()).run(plan()))
    try:
        with anyio.fail_after(20):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert task.cancelled()
        assert not events(run_config, "plan_rejected")
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.parametrize("kind", ["base", "ordinary_group", "mixed_group", "cancelled"])
async def test_supervised_outcome_is_fresh_value_free_core_error(env: EngineEnv, kind: str) -> None:
    secret = str(uuid4())
    error: BaseException
    if kind == "base":
        error = BaseException(secret)
    elif kind == "ordinary_group":
        error = ExceptionGroup(secret, [RuntimeError(secret)])
    elif kind == "mixed_group":
        error = BaseExceptionGroup(secret, [asyncio.CancelledError(secret), RuntimeError(secret)])
    else:
        error = asyncio.CancelledError(secret)
    error.add_note(secret)

    async def invoke() -> None:
        raise error from ValueError(secret)

    runner = GoalRunner(env.deps, goal_id="g", plan=plan(), replanner=None)
    with pytest.raises(BaseException, match=r".+") as caught:
        await runner._supervised_step(invoke, final=False)
    safe = caught.value
    assert isinstance(safe, Exception)
    assert safe is not error
    assert _rebuild_core_signal(safe) is not None
    assert secret not in repr(safe.args)
    assert not getattr(safe, "__notes__", ())
    assert safe.__cause__ is None
    assert secret not in "".join(traceback.format_exception(safe))


@pytest.mark.parametrize("kind", ["fenced", "limit"])
async def test_supervised_outcome_preserves_core_controls(env: EngineEnv, kind: str) -> None:
    error = Fenced() if kind == "fenced" else _core_error(LimitReached, "budget", "budget used up")
    if kind == "limit":
        error.args = (str(uuid4()),)
        error.add_note(str(uuid4()))

    async def invoke() -> None:
        raise error

    runner = GoalRunner(env.deps, goal_id="g", plan=plan(), replanner=None)
    with pytest.raises(type(error)) as caught:
        await runner._supervised_step(invoke, final=False)
    if kind == "fenced":
        assert caught.value is error
    else:
        assert caught.value is not error
        assert caught.value.args == ("budget used up",)
        assert not getattr(caught.value, "__notes__", ())
