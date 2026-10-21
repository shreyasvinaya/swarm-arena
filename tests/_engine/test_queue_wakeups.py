"""Wake public goal scheduling on completion, released capacity and stop."""

from functools import partial
from typing import Any, ClassVar

import anyio
import pytest
from _fakes.run import all_events, events, plan

from swarm_arena import Plan, Ref, RunConfig, ScriptedManager, Swarm
from swarm_arena._engine import run as run_module
from swarm_arena._journal.writer import JournalWriter
from swarm_arena.manager import AgentContext, RoundReport
from swarm_arena.planning import Decision, Policy, RoundState


class PruneOne(Policy):
    def after_round(self, state: RoundState) -> Decision:
        return Decision(prune={"t0/a0": "release capacity"}) if state.round == 1 else Decision()


class HeldManager(ScriptedManager):
    mode: ClassVar[str]
    held: ClassVar[anyio.Event]
    release: ClassVar[anyio.Event]
    peer: ClassVar[anyio.Event]

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        if ctx.goal.id == "b":
            self.peer.set()
        elif ctx.round == (2 if self.mode == "capacity" else 1):
            self.held.set()
            await self.release.wait()
        return await super().run_round(ctx)


class NoQueuePolling:
    def __getattr__(self, name: str) -> Any:
        return getattr(anyio, name)

    async def sleep(self, delay: float) -> None:
        assert delay != 0.02, "goal scheduling must wait for a change event"
        await anyio.sleep(delay)


def _held_plan(goal_id: str, *, agents: int = 1, rounds: int = 1) -> Plan:
    current = plan(goal_id=goal_id, agents=agents, rounds=rounds)
    team = current.teams[0]
    return current.model_copy(
        update={
            "teams": [
                team.model_copy(
                    update={
                        "agents": [
                            agent.model_copy(
                                update={
                                    "manager": Ref(
                                        path=f"{__name__}:HeldManager", config=agent.manager.config
                                    )
                                }
                            )
                            for agent in team.agents
                        ]
                    }
                )
            ]
        }
    )


@pytest.mark.parametrize("cause", ["completion", "capacity", "stop"])
async def test_goal_queue_wakes_without_polling(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    HeldManager.mode = cause
    HeldManager.held, HeldManager.release, HeldManager.peer = (
        anyio.Event(),
        anyio.Event(),
        anyio.Event(),
    )
    peer_finished = anyio.Event()
    publish = JournalWriter._publish

    async def observe(self: JournalWriter) -> None:
        finished = any(e.type == "goal_finished" and e.goal_id == "b" for e in self._queue)
        await publish(self)
        if finished:
            peer_finished.set()

    monkeypatch.setattr(JournalWriter, "_publish", observe)
    monkeypatch.setattr(run_module, "anyio", NoQueuePolling())
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    cfg = run_config.model_copy(
        update={
            "limits": run_config.limits.model_copy(
                update={
                    "max_concurrent_goals": 2 if cause == "capacity" else 1,
                    "max_agents_total": 2,
                }
            )
        }
    )
    first = _held_plan(
        "a", agents=2 if cause == "capacity" else 1, rounds=2 if cause == "capacity" else 1
    )
    if cause == "capacity":
        first = first.model_copy(update={"policy": Ref(path=f"{__name__}:PruneOne")})
    swarm = Swarm(cfg)
    with anyio.fail_after(20):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(swarm.run, [first, _held_plan("b")])
            try:
                await HeldManager.held.wait()
                if cause == "completion":
                    HeldManager.release.set()
                elif cause == "stop":
                    await swarm.stop()
                await peer_finished.wait()
                assert HeldManager.peer.is_set() == (cause != "stop")
                if cause != "completion":
                    assert not [e for e in events(cfg, "goal_finished") if e.goal_id == "a"]
            finally:
                HeldManager.release.set()
    history = all_events(cfg)
    assert {e.goal_id for e in history if e.type == "goal_finished"} == {"a", "b"}
    if cause == "stop":
        assert not [e for e in history if e.type == "goal_started" and e.goal_id == "b"]
    elif cause == "capacity":
        peer_start = next(e.seq for e in history if e.type == "goal_started" and e.goal_id == "b")
        first_finish = next(
            e.seq for e in history if e.type == "goal_finished" and e.goal_id == "a"
        )
        assert peer_start < first_finish
