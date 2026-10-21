"""Preserve independent board groups across public revision publication."""

from pathlib import Path

import anyio
import pytest
from _fakes.run import all_events, events, plan

from swarm_arena import Goal, PartSpec, Plan, Ref, RunConfig, Swarm
from swarm_arena._board.run_board import RunBoard
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._store.local import LocalStore
from swarm_arena.config import CatalogEntry, PlannerCatalog
from swarm_arena.planning import History, Planner


class CollidingPlanner(Planner):
    def __init__(self, *, simultaneous: bool = False, join: bool = False) -> None:
        super().__init__()
        self.simultaneous, self.join = simultaneous, join
        self.arrived: set[str] = set()
        self.ready = anyio.Event()

    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        assert history is not None
        current = history.revisions[-1]
        if self.simultaneous:
            self.arrived.add(goal.id)
            if len(self.arrived) == 2:
                self.ready.set()
            await self.ready.wait()
        elif goal.id != "g2":
            return current
        shared = Goal(id="shared" if self.simultaneous else "g1", namespace="n")
        part = current.parts[0].model_copy(update={"goal": shared})
        return current.model_copy(
            update={
                "parts": [part],
                "rounds": [*current.rounds, current.rounds[-1]] if self.join else current.rounds,
            }
        )


def _partitioned(goal_id: str, *, rounds: int = 2, fail_join: bool = False) -> Plan:
    current = plan(goal_id=goal_id, rounds=rounds, converge_team=True)
    assert current.converge.team is not None
    converge = current.converge
    if fail_join:
        assert converge.team is not None
        agent = converge.team.agents[0].model_copy(
            update={
                "manager": Ref(
                    path="swarm_arena:ScriptedManager", config={"requests": [{"inputs": {}}]}
                )
            }
        )
        converge = converge.model_copy(
            update={
                "team": converge.team.model_copy(update={"agents": [agent]}),
                "max_join_attempts": 1,
            }
        )
    return current.model_copy(
        update={
            "parts": [PartSpec(name="p", goal=Goal(id=f"{goal_id}-part", namespace="n"))],
            "teams": [current.teams[0].model_copy(update={"part": "p"})],
            "converge": converge,
        }
    )


def _assert_membership(config: RunConfig, swarm: Swarm) -> RunState:
    assert swarm.run_id is not None
    committed = read_committed(LocalStore(Path(config.run_root) / swarm.run_id))
    assert committed is not None
    owner, history = committed
    state = RunState.from_events(history)
    board = RunBoard.from_state(state, run_root=config.run_root, head=lambda: owner.head)
    try:
        for goal_id, revisions in state.plans.items():
            current = revisions[-1][0]
            for goal in [current.goal, *(part.goal for part in current.parts)]:
                rows = board.best(goal.key(), 20, judge="y")
                if not rows:
                    assert state.goals[goal_id].status == "no_result"
                    continue
                assert {candidate.goal_id for candidate in rows} == {goal_id}
                view = board.snapshot(goal.key(), board.frontier())
                for candidate in rows:
                    assert view.get_candidate(candidate.id).goal_id == goal_id
    finally:
        board.close()
    return state


@pytest.mark.parametrize("other_status", ["finished", "queued", "join"])
async def test_revision_collision_is_rejected_before_publication_and_dispatch(
    run_config: RunConfig, other_status: str
) -> None:
    cfg = run_config.model_copy(
        update={"limits": run_config.limits.model_copy(update={"max_concurrent_goals": 1})}
    )
    first = plan(goal_id="g1", rounds=1)
    second = _partitioned(
        "g2", rounds=1 if other_status == "join" else 2, fail_join=other_status == "join"
    )
    if other_status == "join":
        assert second.converge.team is not None
        cfg = cfg.model_copy(
            update={
                "planner_catalog": PlannerCatalog(
                    entries=[
                        CatalogEntry(
                            name=f"tool-{i}",
                            kind="tool",
                            ref=agent.tool.path,
                            config=agent.tool.config,
                            description="Approved tool",
                        )
                        for i, agent in enumerate(
                            [second.teams[0].agents[0], second.converge.team.agents[0]]
                        )
                    ]
                )
            }
        )
    plans = [second, first] if other_status == "queued" else [first, second]
    swarm = Swarm(cfg, replanner=CollidingPlanner(join=other_status == "join"))
    await swarm.run(plans)
    rejected = [e for e in events(cfg, "plan_rejected") if e.goal_id == "g2"]
    assert len(rejected) == 1
    assert rejected[0].payload["errors"] == ["revision identity collision: GoalKey"]
    if other_status == "join":
        assert rejected[0].payload["join_retry"] == 1
    history = all_events(cfg)
    other_finish = next(e for e in history if e.type == "goal_finished" and e.goal_id == "g1")
    other_start = next(e for e in history if e.type == "goal_started" and e.goal_id == "g1")
    if other_status == "queued":
        assert rejected[0].seq < other_start.seq
    else:
        assert other_finish.seq < rejected[0].seq
    state = _assert_membership(cfg, swarm)
    assert len(state.plans["g2"]) == 1
    assert not [
        e
        for e in history
        if e.type == "agent_status"
        and e.goal_id == "g2"
        and e.payload.get("feedback_policy", {}).get("revision") == 1
    ]
    if other_status == "join":
        assert max(e.round for e in events(cfg, "round_started") if e.goal_id == "g2") == 2
        assert all(e.round == 1 for e in events(cfg, "tool_start") if e.goal_id == "g2")


async def test_simultaneous_revisions_serialize_against_current_goal_keys(
    run_config: RunConfig,
) -> None:
    cfg = run_config.model_copy(
        update={"limits": run_config.limits.model_copy(update={"max_concurrent_goals": 2})}
    )
    planner = CollidingPlanner(simultaneous=True)
    swarm = Swarm(cfg, replanner=planner)
    with anyio.fail_after(20):
        await swarm.run([_partitioned("g1"), _partitioned("g2")])
    assert planner.arrived == {"g1", "g2"}
    rejected = events(cfg, "plan_rejected")
    assert len(rejected) == 1
    assert rejected[0].payload["errors"] == ["revision identity collision: GoalKey"]
    state = _assert_membership(cfg, swarm)
    assert sorted(len(revisions) for revisions in state.plans.values()) == [1, 2]
    assert rejected[0].seq < min(e.seq for e in events(cfg, "goal_finished"))
    refused = rejected[0].goal_id
    assert len(state.plans[refused]) == 1
    assert all(
        e.payload.get("feedback_policy", {}).get("revision") == 0
        for e in events(cfg, "agent_status")
        if e.goal_id == refused
    )
