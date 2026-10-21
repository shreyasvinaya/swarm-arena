"""Reject ambiguous board groups before a public run can spend or join."""

import traceback

import anyio
import pytest
from _fakes.admission import ReadOwnPartManager
from _fakes.secrecy import assert_secret_absent

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    FunctionTool,
    Goal,
    PartSpec,
    Plan,
    Ref,
    RoundSpec,
    RunConfig,
    Swarm,
    TeamSpec,
    validate_plan,
)
from swarm_arena._errors import PlanError

SECRET = "private-goal-input-72ac860e"


@pytest.fixture
def run_config(run_config: RunConfig) -> RunConfig:
    return run_config.model_copy(
        update={"limits": run_config.limits.model_copy(update={"max_agents_per_fingerprint": 4})}
    )


def _plan(goal_id: str = "g") -> Plan:
    def team(name: str, part: str | None = None) -> TeamSpec:
        return TeamSpec(
            name=name,
            part=part,
            rationale="Solve",
            agents=[
                AgentSpec(
                    name=name,
                    rationale="Solve",
                    tool=Ref.of(FunctionTool("_fakes.tools:line_value")),
                    manager=Ref(path="_fakes.admission:ReadOwnPartManager"),
                )
            ],
        )

    return Plan(
        goal=Goal(id=goal_id, namespace="demo"),
        parts=[
            PartSpec(name=p, goal=Goal(id=f"{goal_id}-{p}", namespace="demo")) for p in ("a", "b")
        ],
        teams=[team(p, p) for p in ("a", "b")],
        judges=[Ref(path="_fakes.tools:FixedJudge", config={"value": 1.0})],
        rounds=[RoundSpec(), RoundSpec()],
        converge=ConvergeSpec(mode="team", team=team("join")),
    )


def _collide(plan: Plan, goal: Goal) -> Plan:
    return plan.model_copy(
        update={"parts": [plan.parts[0], plan.parts[1].model_copy(update={"goal": goal})]}
    )


@pytest.mark.parametrize("shape", ["part-part", "main-part"])
def test_validate_plan_rejects_shared_goal_keys(shape: str, run_config: RunConfig) -> None:
    plan = _plan()
    goal = plan.parts[0].goal if shape == "part-part" else plan.goal
    plan = _collide(plan, goal)
    with pytest.raises(PlanError, match="GoalKey") as caught:
        validate_plan(plan, run_config)
    assert "b" in str(caught.value)
    assert ("a" if shape == "part-part" else "g") in str(caught.value)


@pytest.mark.parametrize("shape", ["part-part", "main-part", "cross-part", "cross-main"])
async def test_run_refuses_shared_goal_keys_before_work(shape: str, run_config: RunConfig) -> None:
    plan = _plan()
    # Metadata is deliberately different; it does not distinguish GoalKeys.
    shared = plan.parts[0].goal.model_copy(
        update={"inputs": {"secret": SECRET}, "metadata": {"secret": SECRET}}
    )
    plan = plan.model_copy(
        update={"parts": [plan.parts[0].model_copy(update={"goal": shared}), plan.parts[1]]}
    )
    plans = [plan]
    if shape == "part-part":
        plans = [_collide(plan, shared.model_copy(update={"metadata": {}}))]
    elif shape == "main-part":
        plans = [_collide(plan, plan.goal)]
    elif shape == "cross-part":
        plans.append(_collide(_plan("other"), shared))
    else:
        plans.append(_plan("other").model_copy(update={"goal": shared}))
    swarm = Swarm(run_config)
    with pytest.raises(PlanError, match="GoalKey") as caught:
        await swarm.run(plans)
    assert_secret_absent(caught.value, SECRET)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert swarm.run_id is None
    assert not await anyio.Path(run_config.run_root).exists()
    assert not await anyio.Path(run_config.artifact_root).exists()


async def test_distinct_goal_keys_keep_own_part_reads_and_join(run_config: RunConfig) -> None:
    ReadOwnPartManager.reads.clear()
    out = await Swarm(run_config).run(_plan())
    assert len(out["g"]) == 1
    assert out["g"][0].team == "join"
    assert sorted(ReadOwnPartManager.reads) == [("a", "a"), ("b", "b")]
