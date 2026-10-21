from typing import Any

import pytest

from swarm_arena import Goal, PartSpec, Plan, Ref, TeamSpec
from swarm_arena._engine.parts import (
    join_failure_reasons,
    join_parents,
    kept_best,
    part_scoring,
    team_goal,
)
from swarm_arena._model.planning_types import RoundState
from swarm_arena._model.results import AgentStatus, Candidate, CheckResult, Score
from swarm_arena._model.tooling import ToolOutput, ToolRequest


def _parts_plan(*, part_a_judges: list[Ref] | None = None) -> Plan:
    return Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[TeamSpec(name=p, part=p, rationale="Explore", agents=[]) for p in ("a", "b")],
        checks=[Ref(path="unavailable:Check", name="valid")],
        judges=[Ref(path="unavailable:Judge", name="program")],
        parts=[
            PartSpec(name="a", goal=Goal(id="a", namespace="n"), judges=part_a_judges),
            PartSpec(name="b", goal=Goal(id="b", namespace="n")),
        ],
    )


def _state_two_parts(
    *,
    b_agents_stopped: bool = False,
    identities: dict[str | None, tuple[str, str, str]] | None = None,
) -> RoundState:
    candidates = [
        Candidate(
            id=id,
            run_id="r",
            goal_id="g",
            part=part,
            team=part,
            agent="agent",
            round=1,
            request=ToolRequest(),
            output=ToolOutput(status="ok"),
        )
        for id, part in [("c-a-top-by-a", "a"), ("c-b-best", "b"), ("c-b-top-by-b", "b")]
    ]
    scores = [
        Score(id=f"s-{i}-{j}", candidate_id=c.id, judge=j, key=(value,))
        for i, c in enumerate(candidates)
        for j, value in [("a", 9.0 if i < 2 else 1.0), ("b", 1.0 if i < 2 else 9.0)]
    ]
    return RoundState(
        goal_id="g",
        round=2,
        rounds_total=3,
        plan=_parts_plan(),
        candidates=candidates,
        scores=scores,
        status={"b/agent": AgentStatus.STOPPED} if b_agents_stopped else {},
        ranking_identity=identities or {"a": ("a", "", ""), "b": ("a", "", "")},
    )


def test_part_scoring_falls_back_to_plan_level() -> None:
    checks, judges, ranking = part_scoring(_parts_plan(part_a_judges=None), "a")
    assert checks == _parts_plan().checks
    assert judges == _parts_plan().judges
    assert ranking == "program"


def test_join_parents_one_per_part_in_plan_order() -> None:
    assert [c.part for c in join_parents(_state_two_parts(), parents_per_part=1)] == ["a", "b"]


def test_kept_best_returns_earlier_candidate() -> None:
    best = kept_best(_state_two_parts(b_agents_stopped=True), "b")
    assert best is not None
    assert best.id == "c-b-best"


def test_join_parents_rank_each_part_by_its_own_judge() -> None:
    st = _state_two_parts(identities={"a": ("a", "", ""), "b": ("b", "", "")})
    assert [c.id for c in join_parents(st, parents_per_part=1)] == [
        "c-a-top-by-a",
        "c-b-top-by-b",
    ]


def test_team_goal_and_scoring_overrides_are_import_free() -> None:
    plan = _parts_plan()
    assert team_goal(plan, plan.teams[0]) == plan.parts[0].goal
    main = TeamSpec(name="main", rationale="Explore", agents=[])
    assert team_goal(plan, main) == plan.goal
    override = plan.parts[0].model_copy(update={"checks": [], "judges": [], "ranking": "other"})
    plan = plan.model_copy(update={"parts": [override, plan.parts[1]]})
    assert part_scoring(plan, "a") == ([], [], "other")
    assert part_scoring(plan, None) == (plan.checks, plan.judges, "program")
    assert kept_best(_state_two_parts().model_copy(update={"ranking_identity": {}}), "a") is None


@pytest.mark.parametrize("helper", [team_goal, part_scoring])
def test_unknown_part_is_refused(helper: Any) -> None:
    plan = _parts_plan()
    arg = TeamSpec(name="team", part="unknown", rationale="Explore", agents=[])
    with pytest.raises(ValueError, match="part"):
        helper(plan, arg if helper is team_goal else "unknown")


def test_join_failure_reasons_collect_only_failed_checks() -> None:
    candidates = _state_two_parts().candidates
    failed = candidates[0].model_copy(
        update={
            "checks": [
                CheckResult(
                    check="gate", passed=False, reasons=["join incomplete", "missing fragment"]
                ),
                CheckResult(check="ok", passed=True, reasons=["passed feedback"]),
            ]
        }
    )
    assert join_failure_reasons([failed, *candidates[1:]]) == [
        "join incomplete",
        "missing fragment",
    ]


def test_part_scoring_unnamed_judge_defers_to_recorded_scores() -> None:
    plan = _parts_plan().model_copy(update={"judges": [Ref(path="missing:Judge")]})
    assert part_scoring(plan, None)[2] == ""
    assert part_scoring(plan.model_copy(update={"ranking": "explicit"}), "a")[2] == "explicit"


def test_join_parents_keep_top_two_and_handle_empty_parts() -> None:
    state = _state_two_parts()
    assert [c.id for c in join_parents(state, parents_per_part=2)] == [
        "c-a-top-by-a",
        "c-b-best",
        "c-b-top-by-b",
    ]
    assert join_parents(state, parents_per_part=0) == []
    state = state.model_copy(update={"ranking_identity": {"a": ("a", "", "")}})
    assert [c.part for c in join_parents(state, parents_per_part=1)] == ["a"]
