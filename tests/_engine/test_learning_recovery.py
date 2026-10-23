"""Recover private own-round learnings without advancing the shared frontier."""

import asyncio
from typing import ClassVar

import anyio
import pytest
from _engine.test_goal import _plan
from _fakes.engine import EngineEnv

from swarm_arena import Ref
from swarm_arena._engine.goal import GoalRunner
from swarm_arena._errors import SoftError
from swarm_arena._journal.replay import RunState
from swarm_arena.manager import AgentContext, Manager, RoundReport


class RecoverLearning(Manager):
    posted: ClassVar[dict[str, str]] = {}
    checked: ClassVar[list[str]] = []
    ready: ClassVar[anyio.Event]

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        key = ctx.spec.name
        if not ctx.notes:
            candidate = await ctx.run_tool({"x": 1.0})
            item = await ctx.post_learning(f"Observation {key}", "Current evidence", [candidate.id])
            await ctx.save_notes(item.id)
            # The next paid action commits learning and notes in the original frontier.
            await ctx.run_tool({"x": 2.0})
            self.posted[key] = item.id
            if len(self.posted) == 2:
                self.ready.set()
            await anyio.sleep_forever()
        own = ctx.read_learning(ctx.notes)
        assert own.id == self.posted[key]
        for scope in ("goal", "namespace"):
            assert own.id in {item.id for item in ctx.search_learnings("Observation", scope=scope)}
        for peer, id in self.posted.items():
            if peer != key:
                with pytest.raises(SoftError, match="unknown learning"):
                    ctx.read_learning(id)
        with pytest.raises(SoftError, match="duplicate"):
            await ctx.post_learning(own.title, own.text, own.evidence)
        refined = await ctx.post_learning(
            own.title, "Refined evidence", own.evidence, refines=[own.id]
        )
        assert refined.refines == [own.id]
        self.checked.append(key)
        return RoundReport()


async def test_resumed_round_restores_only_own_learnings(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RecoverLearning, "posted", {})
    monkeypatch.setattr(RecoverLearning, "checked", [])
    monkeypatch.setattr(RecoverLearning, "ready", anyio.Event(), raising=False)
    plan = _plan(teams=1, agents_per_team=2, rounds=1)
    team = plan.teams[0]
    plan = plan.model_copy(
        update={
            "teams": [
                team.model_copy(
                    update={
                        "agents": [
                            agent.model_copy(
                                update={"manager": Ref(path=f"{__name__}:RecoverLearning")}
                            )
                            for agent in team.agents
                        ]
                    }
                )
            ]
        }
    )
    task = asyncio.create_task(GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run())
    try:
        with anyio.fail_after(60):
            await RecoverLearning.ready.wait()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    await env.deps.journal.commit()
    state = RunState.from_events(env._events())
    assert len(state.learnings) == 2
    frontier = state.goals["g"].frontiers[1]
    outcome = await GoalRunner(
        env.deps,
        goal_id="g",
        plan=plan,
        replanner=None,
        state=state.goals["g"],
        journal_r0=False,
    ).run()
    assert outcome.finalists
    assert sorted(RecoverLearning.checked) == ["a0", "a1"]
    final = RunState.from_events(env._events())
    assert final.goals["g"].frontiers[1] == frontier
    assert len(final.learnings) == 4
    assert all(item.refines for id, item in final.learnings.items() if id not in state.learnings)
