import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena import Goal, PartSpec, Plan, TeamSpec
from swarm_arena._model.base import StoredModel
from swarm_arena._model.planning_types import Decision, History, RoundState
from swarm_arena._model.results import Candidate, Score
from swarm_arena._model.tooling import Estimate, ToolOutput, ToolRequest

ID = ("program", "rh", "")


def _state(rounds: list[dict[str, float | None]]) -> RoundState:
    teams = list(dict.fromkeys(team for values in rounds for team in values))
    candidates: list[Candidate] = []
    scores: list[Score] = []
    for round, values in enumerate(rounds, 1):
        for team, value in values.items():
            if value is None:
                continue
            candidate = Candidate(
                id=f"c-{len(candidates) + 1}",
                run_id="r",
                goal_id="g",
                team=team,
                agent="agent",
                round=round,
                fidelity="f1",
                request=ToolRequest(),
                output=ToolOutput(status="ok"),
            )
            candidates.append(candidate)
            scores.append(
                Score(
                    candidate_id=candidate.id,
                    key=(value,),
                    value=value,
                    judge=ID[0],
                    rubric_hash=ID[1],
                )
            )
    return RoundState(
        goal_id="g",
        round=len(rounds),
        rounds_total=len(rounds) + 2,
        plan=Plan(
            goal=Goal(id="g", namespace="n"),
            judges=[],
            teams=[TeamSpec(name=t, rationale="Explore", agents=[]) for t in teams],
        ),
        candidates=candidates,
        scores=scores,
        status={},
        ranking_identity={None: ID},
    )


def test_decision_merge_unions_and_keeps_first_reason() -> None:
    left = Decision(prune={"t1": "r1"}, policy="A")
    right = Decision(prune={"t1": "r2", "t2/a": "r3"}, no_result="none", policy="B")
    assert left.merge(right) == Decision(
        prune={"t1": "r1", "t2/a": "r3"}, no_result="none", policy="A+B"
    )
    assert left.prune == {"t1": "r1"}
    assert right.prune["t1"] == "r2"
    assert Decision(no_result="first").merge(right).no_result == "first"


@pytest.mark.parametrize(
    ("first", "second", "expected"), [("", "", ""), ("", "B", "B"), ("A", "", "A")]
)
def test_decision_merge_joins_only_nonempty_names(first: str, second: str, expected: str) -> None:
    assert Decision(policy=first).merge(Decision(policy=second)).policy == expected


def test_round_state_team_best_by_round() -> None:
    st = _state(rounds=[{"t1": 3.0}, {"t1": None}, {"t1": 5.0}])
    assert [s.value if s else None for s in st.team_best_by_round("t1")] == [3.0, 3.0, 5.0]
    best = st.team_best("t1")
    assert best is not None
    assert best.value == 5.0
    assert st.groups() == {None: ["t1"]}
    assert not isinstance(st, StoredModel)
    assert st.model_dump()["ranking_identity"] == {None: ID}


def test_round_state_empty_history_and_absent_identity() -> None:
    st = _state(rounds=[{"t1": None}, {"t1": 2.0}])
    assert st.team_best_by_round("t1")[0] is None
    empty = st.model_copy(update={"ranking_identity": {}})
    assert empty.ranked(None) == []
    assert empty.team_best("t1") is None
    assert empty.team_best_by_round("t1") == [None, None]
    assert st.model_copy(update={"round": 0}).team_best_by_round("t1") == []


def test_round_state_uses_current_group_fidelity_for_all_history() -> None:
    st = _state(rounds=[{"t1": 10.0, "t2": 1.0}, {"t1": None, "t2": 2.0}, {"t1": 3.0}])
    candidates = [
        c.model_copy(update={"fidelity": "f2"}) if c.round > 1 else c for c in st.candidates
    ]
    st = st.model_copy(update={"candidates": candidates})
    assert [s.value if s else None for s in st.team_best_by_round("t1")] == [None, None, 3.0]
    before = st.model_copy(update={"round": 2, "candidates": candidates[:-1]})
    assert before.team_best("t1") is None
    assert before.team_best_by_round("t1") == [None, None]


def test_round_state_ranks_each_group_by_its_own_identity() -> None:
    st = _state(rounds=[{"a1": 1.0, "b1": 2.0}])
    plan = st.plan.model_copy(
        update={
            "parts": [PartSpec(name=p, goal=Goal(id=p, namespace="n")) for p in ("a", "b")],
            "teams": [t.model_copy(update={"part": t.name[0]}) for t in st.plan.teams],
        }
    )
    candidates = [c.model_copy(update={"part": c.team[0]}) for c in st.candidates]
    scores = [
        Score(
            candidate_id=c.id,
            key=(value,),
            value=value,
            judge=j,
            rubric_hash=f"h{j}",
            panel_hash=panel,
        )
        for c in candidates
        for j, panel, value in [("a", "", 1.0), ("b", "", 2.0), ("b", "pb2", 3.0)]
    ]
    st = st.model_copy(
        update={
            "plan": plan,
            "candidates": candidates,
            "scores": scores,
            "ranking_identity": {"a": ("a", "ha", ""), "b": ("b", "hb", "")},
        }
    )
    assert {s.judge for _, s in st.ranked("a")} == {"a"}
    assert {s.judge for _, s in st.ranked("b")} == {"b"}
    assert {c.part for c, _ in st.ranked("a")} == {"a"}
    before = st.ranked("a")
    st2 = st.model_copy(
        update={"ranking_identity": {**st.ranking_identity, "b": ("b", "hb", "pb2")}}
    )
    assert st2.ranked("b") != []
    assert all(s.panel_hash == "pb2" for _, s in st2.ranked("b"))
    assert st2.ranked("a") == before
    assert st2.ranked("c") == []
    best = st2.team_best("b1")
    assert best is not None
    assert best.panel_hash == "pb2"
    assert st2.team_best_by_round("b1") == [best]


def test_history_fields_and_independent_defaults() -> None:
    history = History(
        goal_id="g",
        revisions=[],
        scores=[],
        learnings=[],
        decisions=[(1, Decision())],
        budget_left=Estimate(tokens=100),
        rounds_done=1,
    )
    assert history.join_failures == []
    assert history.decisions == [(1, Decision())]
    first, second = Decision(), Decision()
    first.prune["t"] = "reason"
    assert second.prune == {}


def test_planning_validation_does_not_expose_mapping_values_or_keys() -> None:
    secret = "SENTINEL-private-planning-value"
    for model, raw in [
        (Decision, {"prune": {secret: [secret]}}),
        (RoundState, {**_state(rounds=[]).model_dump(), "ranking_identity": {secret: [secret]}}),
    ]:
        with pytest.raises(ValidationError) as exc:
            model.model_validate(raw)
        assert_secret_absent(exc.value, secret, sanitized=True)
