"""Check the Planner contract and deterministic engine test planners."""

from typing import ClassVar

import pytest
from _fakes.planners import AddAgentPlanner, RaisingPlanner, SequencePlanner

from swarm_arena import AgentSpec, Goal, Plan, Ref, RunConfig, TeamSpec
from swarm_arena._model.base import Origin
from swarm_arena._model.ref import Configurable
from swarm_arena.config import CatalogEntry, PlannerCatalog
from swarm_arena.planning import History, Planner
from swarm_arena.tool import Estimate


def _plan(goal: Goal) -> Plan:
    return Plan(
        goal=goal,
        teams=[
            TeamSpec(
                name="t",
                rationale="Explore",
                agents=[
                    AgentSpec(
                        name="a",
                        rationale="Search",
                        tool=Ref(path="demo:Tool"),
                        manager=Ref(path="demo:Manager"),
                    ),
                ],
            )
        ],
        judges=[Ref(path="demo:Judge")],
    )


def _history(plan: Plan) -> History:
    return History(
        goal_id=plan.goal.id,
        revisions=[plan],
        scores=[],
        learnings=[],
        decisions=[],
        budget_left=Estimate(),
        rounds_done=1,
    )


def test_planner_subclass_cannot_claim_human() -> None:
    with pytest.raises(TypeError, match="human"):

        class Bad(Planner):
            origin = "human"

            async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
                raise NotImplementedError


def test_planner_is_abstract_and_not_configurable() -> None:
    assert not issubclass(Planner, Configurable)
    assert Planner.origin == "llm"
    with pytest.raises(TypeError, match="abstract"):
        Planner()  # ty: ignore[call-non-callable]


@pytest.mark.parametrize("origin", ["llm", "together", "autonomous"])
def test_planner_allows_nonhuman_origins(origin: Origin) -> None:
    origin_value = origin

    class Valid(Planner):
        async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
            return _plan(goal)

    class Derived(Valid):
        origin: ClassVar[Origin] = origin_value

    assert Derived.origin == origin
    assert Derived().models() == ()


async def test_sequence_planner_repeats_last_and_copies_sequence(run_config: RunConfig) -> None:
    goal = Goal(id="g", namespace="demo")
    first = _plan(goal)
    last = first.model_copy(update={"rationale": "second"})
    plans = [first, last]
    planner = SequencePlanner(plans)
    plans.clear()
    assert await planner.plan(goal, run_config, None) is first
    assert await planner.plan(goal, run_config, None) is last
    assert await planner.plan(goal, run_config, _history(last)) is last
    assert planner.models() == ()


def test_sequence_planner_requires_a_plan() -> None:
    with pytest.raises(ValueError, match="plan"):
        SequencePlanner([])


async def test_add_agent_planner_uses_entry_and_preserves_parent(run_config: RunConfig) -> None:
    goal = Goal(id="g", namespace="demo")
    parent = _plan(goal)
    entry = CatalogEntry(
        name="catalog-agent",
        kind="tool",
        ref="demo:NewTool",
        description="New approach",
        config={"x": 2},
    )
    config = run_config.model_copy(update={"planner_catalog": PlannerCatalog(entries=[entry])})
    planner = AddAgentPlanner("catalog-agent")
    child = await planner.plan(goal, config, _history(parent))
    assert len(parent.teams[0].agents) == 1
    assert len(child.teams[0].agents) == 2
    added = child.teams[0].agents[-1]
    assert added.tool.path == entry.ref
    assert added.tool.config == entry.config
    assert added.manager == parent.teams[0].agents[0].manager
    assert added.rationale
    assert added.name != "a"
    grandchild = await planner.plan(goal, config, _history(child))
    assert len({a.name for a in grandchild.teams[0].agents}) == 3
    assert grandchild.goal == goal


async def test_add_agent_planner_requires_history(run_config: RunConfig) -> None:
    with pytest.raises(ValueError, match="history"):
        await AddAgentPlanner("catalog-agent").plan(
            Goal(id="g", namespace="demo"), run_config, None
        )


async def test_raising_planner_always_raises(run_config: RunConfig) -> None:
    with pytest.raises(RuntimeError, match="planner"):
        await RaisingPlanner().plan(Goal(id="g", namespace="demo"), run_config, None)
