"""Contain runtime manager construction through public run and resume APIs."""

import asyncio
import traceback
from typing import ClassVar

import anyio
import pytest
from _fakes.run import all_events, events, plan
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Plan, Ref, RunConfig, ScriptedManager, Swarm
from swarm_arena._errors import Fenced, _core_error
from swarm_arena.manager import AgentContext, RoundReport

SECRET = "09e80b84-eae1-44bd-bfaf-4db52c50628b"


class FailingRuntimeManager(ScriptedManager):
    failure_type: ClassVar[type[Exception]] = ValueError
    registered_fence: ClassVar[bool] = False
    armed: ClassVar[bool] = False
    arm_after_preflight: ClassVar[bool] = False
    entered: ClassVar[anyio.Event | None] = None

    def __init__(self, config: ScriptedManager.Config) -> None:
        if self.armed:
            if self.failure_type is Fenced and self.registered_fence:
                raise _core_error(Fenced, "run ownership lost")
            raise self.failure_type(SECRET)
        super().__init__(config)

    def preflight(self) -> None:
        if self.arm_after_preflight:
            type(self).armed = True

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        if ctx.round == 2 and self.entered is not None:
            self.entered.set()
            await anyio.sleep_forever()
        return await super().run_round(ctx)


class HealthyManager(ScriptedManager):
    calls: ClassVar[list[int]] = []

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        self.calls.append(ctx.round)
        return await super().run_round(ctx)


def _plan() -> Plan:
    current = plan(agents=2, rounds=2)
    team = current.teams[0]
    agents = [
        agent.model_copy(
            update={
                "manager": Ref(path=f"{__name__}:{manager.__name__}", config=agent.manager.config)
            }
        )
        for agent, manager in zip(team.agents, (FailingRuntimeManager, HealthyManager), strict=True)
    ]
    return current.model_copy(
        update={
            "teams": [team.model_copy(update={"agents": agents})],
            "policy": Ref(path="swarm_arena:SuccessiveHalving", config={"keep": 1.0}),
        }
    )


@pytest.mark.parametrize("resume", [False, True])
async def test_runtime_manager_failure_is_type_only_and_peer_finishes(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    monkeypatch.setattr(FailingRuntimeManager, "armed", False)
    monkeypatch.setattr(FailingRuntimeManager, "arm_after_preflight", not resume)
    monkeypatch.setattr(FailingRuntimeManager, "entered", anyio.Event() if resume else None)
    HealthyManager.calls.clear()
    swarm = Swarm(run_config)
    if resume:
        task = asyncio.create_task(swarm.run(_plan()))
        try:
            assert FailingRuntimeManager.entered is not None
            with anyio.fail_after(10):
                await FailingRuntimeManager.entered.wait()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not events(run_config, "goal_finished")
        monkeypatch.setattr(FailingRuntimeManager, "armed", True)
        HealthyManager.calls.clear()
    try:
        if resume:
            assert swarm.run_id is not None
            result = await swarm.resume(swarm.run_id, force=True)
        else:
            result = await swarm.run(_plan())
    except BaseException as error:
        assert_secret_absent(error, SECRET, sanitized=True)
        assert SECRET not in "".join(traceback.format_exception(error))  # noqa: PT017
        raise
    assert SECRET not in repr(result)
    assert HealthyManager.calls
    assert result["g"]
    history = all_events(run_config)
    assert SECRET not in repr([event.model_dump() for event in history])
    failures = [
        event
        for event in history
        if event.type == "agent_status"
        and event.agent == "a0"
        and event.payload["reason"] == "agent error: ValueError"
    ]
    assert failures
    assert events(run_config, "goal_finished")
    assert any(event.agent == "a1" for event in events(run_config, "tool_start"))


@pytest.mark.parametrize("registered", [False, True])
async def test_runtime_manager_constructor_uses_fencing_authority(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    monkeypatch.setattr(FailingRuntimeManager, "armed", False)
    monkeypatch.setattr(FailingRuntimeManager, "arm_after_preflight", True)
    monkeypatch.setattr(FailingRuntimeManager, "failure_type", Fenced)
    monkeypatch.setattr(FailingRuntimeManager, "registered_fence", registered)
    if registered:
        with pytest.raises(Fenced) as caught:
            await Swarm(run_config).run(_plan())
        assert SECRET not in "".join(traceback.format_exception(caught.value))
    else:
        result = await Swarm(run_config).run(_plan())
        assert result["g"]
        assert events(run_config, "goal_finished")
        assert any(event.agent == "a1" for event in events(run_config, "tool_start"))
        assert any(
            event.payload.get("reason") == "agent error: Fenced"
            for event in events(run_config, "agent_status")
        )
    assert SECRET not in repr([event.model_dump() for event in all_events(run_config)])
