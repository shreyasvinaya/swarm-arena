"""Contain untrusted cancellation-shaped hook outcomes through public runs."""

import asyncio
import traceback
from contextlib import suppress
from typing import Literal
from uuid import uuid4

import anyio
import pytest
from _fakes.cleanup import observe_task, settle_tasks
from _fakes.engine import EngineEnv
from _fakes.planners import SequencePlanner
from _fakes.run import all_events, events, plan

from swarm_arena import Goal, Plan, RunConfig, Swarm
from swarm_arena._engine.goal import GoalRunner
from swarm_arena._errors import Fenced, _core_error, _rebuild_core_signal
from swarm_arena._outcomes import _classify_outcome
from swarm_arena.manager import LimitReached
from swarm_arena.planning import History, Planner


async def test_supervised_nested_fencing_is_bare_and_value_free(env: EngineEnv) -> None:
    secret = str(uuid4())

    async def invoke() -> None:
        raise BaseExceptionGroup(
            secret,
            [
                asyncio.CancelledError(secret),
                ExceptionGroup(secret, [_core_error(Fenced, "run ownership lost")]),
            ],
        ) from ValueError(secret)

    runner = GoalRunner(env.deps, goal_id="g", plan=plan(), replanner=None)
    with pytest.raises(Fenced) as caught:
        await runner._supervised_step(invoke, final=False)
    assert caught.value.args == ("run ownership lost",)
    assert caught.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(caught.value))


def test_pending_caller_cancellation_precedes_nested_fencing() -> None:
    secret = str(uuid4())
    with pytest.raises(asyncio.CancelledError) as caught:
        _classify_outcome(
            ExceptionGroup(secret, [_core_error(Fenced, "run ownership lost")]), cancelled=True
        )
    assert caught.value.args == ()
    assert caught.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(caught.value))


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
        with anyio.fail_after(60):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observe_task(task)
        assert task.cancelled()
        assert not events(run_config, "plan_rejected")
    finally:
        if not task.done():
            task.cancel()
            await settle_tasks(task)
            assert task.cancelled()


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
    error = (
        _core_error(Fenced, "run ownership lost")
        if kind == "fenced"
        else _core_error(LimitReached, "budget", "budget used up")
    )
    if kind == "limit":
        error.args = (str(uuid4()),)
        error.add_note(str(uuid4()))

    async def invoke() -> None:
        raise error

    runner = GoalRunner(env.deps, goal_id="g", plan=plan(), replanner=None)
    with pytest.raises(type(error)) as caught:
        await runner._supervised_step(invoke, final=False)
    if kind == "fenced":
        assert caught.value is not error
        assert caught.value.args == ("run ownership lost",)
    else:
        assert caught.value is not error
        assert caught.value.args == ("budget used up",)
        assert not getattr(caught.value, "__notes__", ())


@pytest.mark.parametrize("final", [False, True], ids=["replan", "final-judge"])
async def test_external_task_cancel_race_wins_over_user_cancellation(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, final: bool
) -> None:
    secret = str(uuid4())
    p = plan()
    planner = SequencePlanner([p])
    runner = GoalRunner(env.deps, goal_id="g", plan=p, replanner=planner)
    hook_failed = asyncio.Event()
    cancellations: list[bool] = []

    async def fail(*args: object) -> Plan:
        hook_failed.set()
        error = asyncio.CancelledError(secret)
        error.add_note(secret)
        raise error from ValueError(secret)

    monkeypatch.setattr(planner, "plan", fail)
    if final:
        from swarm_arena import ProgramJudge
        from swarm_arena.testing import make_candidate
        from swarm_arena.tool import RunStatus, ToolOutput

        monkeypatch.setattr(ProgramJudge, "judge", fail)
        finalist = make_candidate(ToolOutput(status=RunStatus.OK), goal=p.goal)
        parent = asyncio.create_task(runner._final([finalist]))
    else:
        parent = asyncio.create_task(runner._replan())

    async def cancel_from_caller() -> None:
        await hook_failed.wait()
        cancellations.append(parent.cancel(secret))

    caller = asyncio.create_task(cancel_from_caller())
    caught: BaseException | None = None
    with anyio.fail_after(60):
        try:
            await parent
        except BaseException as error:
            caught = error
        await caller
    assert cancellations == [True]
    assert isinstance(caught, asyncio.CancelledError)
    assert parent.cancelled()
    assert parent.cancelling() == 1
    assert caught.args == ()
    assert not getattr(caught, "__notes__", ())
    assert caught.__cause__ is None
    assert secret not in "".join(traceback.format_exception(caught))
    await env.deps.journal.commit()
    assert not env.events("plan_rejected")
    assert not env.events("judge_error")


@pytest.mark.parametrize(
    "pending", [False, True], ids=["consumed-before-entry", "pending-at-entry"]
)
async def test_caller_cancellation_entry_baseline(env: EngineEnv, pending: bool) -> None:
    secret = str(uuid4())

    class FailingPlanner(Planner):
        async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
            raise asyncio.CancelledError(secret)

    runner = GoalRunner(env.deps, goal_id="g", plan=plan(), replanner=FailingPlanner())

    async def run() -> bool:
        caller = asyncio.current_task()
        assert caller is not None
        caller.cancel(secret)
        if not pending:
            with suppress(asyncio.CancelledError):
                await asyncio.get_running_loop().create_future()
        assert caller.cancelling() == 1
        return await runner._replan()

    parent = asyncio.create_task(run())
    if pending:
        with pytest.raises(asyncio.CancelledError) as caught:
            await parent
        assert caught.value.args == ()
        assert caught.value.__cause__ is None
        assert secret not in "".join(traceback.format_exception(caught.value))
        assert parent.cancelled()
        assert not env.events("plan_rejected")
    else:
        assert await parent is False
        assert not parent.cancelled()
        assert parent.cancelling() == 1
        await env.deps.journal.commit()
        assert env.events("plan_rejected")[-1].payload["errors"] == [
            "planner error: CancelledError"
        ]


async def test_anyio_caller_scope_cancellation_race(env: EngineEnv) -> None:
    hook_failed = asyncio.Event()
    scopes: list[anyio.CancelScope] = []
    cancellations: list[asyncio.CancelledError] = []

    async def fail() -> None:
        hook_failed.set()
        raise asyncio.CancelledError(str(uuid4()))

    runner = GoalRunner(env.deps, goal_id="g", plan=plan(), replanner=None)

    async def run() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            try:
                await runner._supervised_step(fail, final=False)
            except asyncio.CancelledError as error:
                cancellations.append(error)
                raise
        assert scope.cancelled_caught

    parent = asyncio.create_task(run())
    with anyio.fail_after(60):
        await hook_failed.wait()
        scopes[0].cancel()
        await parent
    assert len(cancellations) == 1
    assert cancellations[0].args == ()
    assert cancellations[0].__cause__ is None
    await env.deps.journal.commit()
    assert not env.events("plan_rejected")
