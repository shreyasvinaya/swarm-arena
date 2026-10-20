import pytest

from swarm_arena import AgentSpec, Goal, PartSpec, Plan, Ref, Score, TeamSpec
from swarm_arena._planning.policy import Policy, guard
from swarm_arena.board import Candidate
from swarm_arena.manager import AgentStatus
from swarm_arena.planning import Decision, RoundState
from swarm_arena.tool import ToolOutput, ToolRequest

IDENTITY = ("quality", "rubric", "")


def _state(
    teams: dict[str, dict[str, float | None]],
    *,
    parts: dict[str, dict[str, dict[str, float | None]]] | None = None,
    status: dict[str, AgentStatus] | None = None,
    round: int = 1,
    rounds_total: int = 5,
) -> RoundState:
    groups = {None: teams, **(parts or {})}
    specs: list[TeamSpec] = []
    candidates: list[Candidate] = []
    scores: list[Score] = []
    statuses = dict(status or {})
    for part, group in groups.items():
        for team, values in group.items():
            agents = dict(values)
            for address in statuses:
                owner, agent = address.split("/")
                if owner == team:
                    agents.setdefault(agent, None)
            specs.append(
                TeamSpec(
                    name=team,
                    part=part,
                    rationale="Explore",
                    agents=[
                        AgentSpec(
                            name=agent,
                            rationale="Explore",
                            tool=Ref(path="unused:Tool"),
                            manager=Ref(path="unused:Manager"),
                        )
                        for agent in agents
                    ],
                )
            )
            for agent, value in agents.items():
                statuses.setdefault(f"{team}/{agent}", AgentStatus.ACTIVE)
                if value is None:
                    continue
                candidate = Candidate(
                    id=f"c-{len(candidates) + 1}",
                    run_id="r",
                    goal_id="g",
                    team=team,
                    agent=agent,
                    part=part,
                    round=round,
                    fidelity="f1",
                    request=ToolRequest(),
                    output=ToolOutput(status="ok"),
                )
                candidates.append(candidate)
                scores.append(
                    Score.maximize(value).model_copy(
                        update={
                            "candidate_id": candidate.id,
                            "judge": IDENTITY[0],
                            "rubric_hash": IDENTITY[1],
                        }
                    )
                )
    return RoundState(
        goal_id="g",
        round=round,
        rounds_total=rounds_total,
        plan=Plan(
            goal=Goal(id="g", namespace="demo"),
            judges=[],
            teams=specs,
            parts=[
                PartSpec(name=part, goal=Goal(id=part, namespace="demo")) for part in parts or {}
            ],
        ),
        candidates=candidates,
        scores=scores,
        status=statuses,
        ranking_identity=dict.fromkeys(groups, IDENTITY),
    )


def test_guard_keeps_a_parts_last_team_and_starved_agents() -> None:
    st = _state(
        {},
        parts={"p": {"p1": {"a": 1}, "p2": {"a": 2}}},
        status={"p2/b": AgentStatus.STARVED},
    )
    d = guard(Decision(prune={"p1": "x", "p2": "y", "p2/b": "z"}), st)
    assert d.prune == {"p1": "x"}


def test_guard_sets_no_result_on_last_round_with_nothing_ranked() -> None:
    assert (
        guard(Decision(), _state({"t": {"a": None}}, round=3, rounds_total=3)).no_result
        == "no passing candidate"
    )


def test_guard_accounts_for_agent_prunes_and_inactive_teams() -> None:
    st = _state({"old": {"a": 10}, "t": {"a": 1, "b": 2}}, status={"old/a": AgentStatus.PRUNED})
    original = Decision(prune={"old": "old", "t/a": "x", "t/b": "y"}, policy="custom")
    assert guard(original, st) == Decision(prune={"t/a": "x"}, policy="custom")
    assert len(original.prune) == 3


def test_guard_protects_starved_team_even_when_it_has_a_score() -> None:
    st = _state({"t": {"a": 1}, "u": {"b": 2}}, status={"t/a": AgentStatus.STARVED})
    assert guard(Decision(prune={"t": "x", "t/a": "y"}), st).prune == {}


def test_guard_no_result_uses_all_groups_and_only_final_round() -> None:
    st = _state({"t": {"a": None}}, parts={"p": {"u": {"b": 1}}}, round=3, rounds_total=3)
    assert guard(Decision(), st).no_result is None
    assert guard(Decision(), _state({"t": {"a": None}})).no_result is None
    assert guard(Decision(no_result="existing"), st).no_result == "existing"


def test_policy_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Policy()  # ty: ignore[call-non-callable]
