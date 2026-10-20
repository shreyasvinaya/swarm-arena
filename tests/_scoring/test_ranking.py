from collections.abc import Iterable

import pytest

from swarm_arena import ConvergeSpec, Goal, PartSpec, Plan, TeamSpec
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._model.tooling import ToolOutput, ToolRequest
from swarm_arena._scoring.ranking import group_of, groups, newest_fidelity, ranked, ranked_groups

ID = ("program", "rh", "")


def _cand(
    id: str,
    *,
    round: int = 1,
    fid: str = "f1",
    failed_check: bool = False,
    partial: bool = False,
) -> Candidate:
    return Candidate(
        id=id,
        run_id="r",
        goal_id="g",
        team="t",
        agent="a",
        round=round,
        fidelity=fid,
        request=ToolRequest(),
        output=ToolOutput(status="ok", partial=partial),
        checks=[CheckResult(check="gate", passed=not failed_check)],
    )


def _score(
    candidate: Candidate,
    value: float,
    *,
    judge: str = "program",
    rubric: str = "rh",
    panel: str = "",
) -> Score:
    return Score(
        candidate_id=candidate.id,
        key=(value,),
        value=value,
        judge=judge,
        rubric_hash=rubric,
        panel_hash=panel,
    )


def _inputs() -> tuple[list[Candidate], list[Score]]:
    a = _cand("c-1", round=1, fid="f1")
    b = _cand("c-2", round=2, fid="f2")
    c = _cand("c-3", round=2, fid="f2")
    g = _cand("c-4", round=2, fid="f2", failed_check=True)
    p = _cand("c-5", round=2, fid="f2", partial=True)
    scores = [_score(x, 1.0) for x in (a, g, p)] + [
        _score(b, 2.0),
        _score(c, 2.0),
        _score(c, 9.0, judge="other"),
    ]
    return [a, b, c, g, p], scores


def _rank(
    candidates: Iterable[Candidate], scores: Iterable[Score]
) -> list[tuple[Candidate, Score]]:
    return ranked(candidates, scores, identity=ID, fidelity=None, allow_partial=False)


def test_ranked_filters_and_orders() -> None:
    candidates, scores = _inputs()
    assert [x.id for x, _ in _rank(iter(candidates), iter(scores))] == ["c-2", "c-3"]


def test_allow_partial_keeps_partial() -> None:
    candidates, scores = _inputs()
    out = ranked(candidates, scores, identity=ID, fidelity=None, allow_partial=True)
    assert [x.id for x, _ in out] == ["c-2", "c-3", "c-5"]


def test_explicit_fidelity_and_empty_inputs() -> None:
    candidates, scores = _inputs()
    out = ranked(candidates, scores, identity=ID, fidelity="f1", allow_partial=False)
    assert [x.id for x, _ in out] == ["c-1"]
    assert _rank([], scores) == []
    assert _rank(candidates, []) == []
    assert newest_fidelity([]) is None
    assert newest_fidelity(iter(reversed(candidates))) == "f2"


def test_ranked_uses_complete_identity_and_latest_matching_score() -> None:
    a, b = _cand("c-1"), _cand("c-2")
    old = _score(a, 9.0)
    latest = _score(a, 1.0)
    score_b = _score(b, 2.0)
    scores = [
        old,
        score_b,
        latest,
        _score(a, 20.0, rubric="different"),
        _score(b, 30.0, panel="different"),
    ]
    assert _rank([a, b], scores) == [(b, score_b), (a, latest)]


def test_newest_fidelity_does_not_fall_back_when_ineligible() -> None:
    a = _cand("c-1")
    for b in (
        _cand("c-2", round=2, fid="f2", partial=True),
        _cand("c-2", round=2, fid="f2", failed_check=True),
        _cand("c-2", round=2, fid="f2"),
    ):
        assert _rank([a, b], [_score(a, 1.0)]) == []


def test_ranked_lexicographic_keys_rounds_and_numeric_ids() -> None:
    ids = ["c-z-e2-10000", "c-z-e2-9999", "c-z-e1-9999", "c-z-e1-10001", "c-z-e1-10002"]
    candidates = [_cand(id) for id in ids]
    candidates[3] = _cand(ids[3], round=0)
    scores = [_score(c, 2.0) for c in candidates]
    scores[-1] = scores[-1].model_copy(update={"key": (2.0, -5.0)})
    assert [c.id for c, _ in _rank(candidates, scores)] == [ids[4], ids[3], ids[2], ids[1], ids[0]]


def _plan_with_parts() -> Plan:
    return Plan(
        goal=Goal(id="g", namespace="n"),
        judges=[],
        parts=[PartSpec(name=p, goal=Goal(id=p, namespace="n")) for p in ("a", "b", "empty")],
        teams=[
            TeamSpec(name=name, part=part, rationale="Explore", agents=[])
            for name, part in [("b1", "b"), ("a1", "a"), ("main-t", None), ("a2", "a")]
        ],
        converge=ConvergeSpec(mode="team", team=TeamSpec(name="join", rationale="Join", agents=[])),
    )


def test_groups_main_first_then_parts() -> None:
    plan = _plan_with_parts()
    assert groups(plan) == {None: ["main-t"], "a": ["a1", "a2"], "b": ["b1"], "empty": []}
    assert list(groups(plan)) == [None, "a", "b", "empty"]
    assert group_of(plan, "a2") == "a"
    assert group_of(plan, "main-t") is None
    for name in ("join", "missing"):
        with pytest.raises(ValueError, match="team"):
            group_of(plan, name)
    assert groups(Plan(goal=plan.goal, teams=[], judges=[])) == {None: []}


def test_ranked_groups_by_part() -> None:
    cs = [
        _cand("main"),
        _cand("a").model_copy(update={"part": "a"}),
        _cand("b").model_copy(update={"part": "b"}),
    ]
    scores = [_score(c, i, judge=j) for i, c in enumerate(cs) for j in ("program", "other")]
    result = ranked_groups(
        cs, iter(scores), identity={None: ID, "a": ("other", "rh", "")}, fidelity=None
    )
    assert set(result) == {None, "a"}
    assert result[None][0][1].judge == "program"
    assert result["a"][0][1].judge == "other"
    assert ranked_groups(cs, [], identity={None: ID}, fidelity=None) == {}


def test_ranked_groups_ties_follow_input_order() -> None:
    cs = [_cand("c-99", round=3), _cand("c-1", round=1)]
    scores = [_score(c, 1) for c in cs]
    assert [
        c.id for c, _ in ranked_groups(cs, scores, identity={None: ID}, fidelity=None)[None]
    ] == ["c-99", "c-1"]
    assert [c.id for c, _ in _rank(cs, scores)] == ["c-1", "c-99"]


def test_ranked_groups_every_fidelity() -> None:
    cs = [_cand("c-3", fid="a"), _cand("c-2", fid="b"), _cand("c-1", fid="a")]
    scores = [_score(c, 1) for c in cs]
    result = ranked_groups(
        cs, iter(scores), identity={None: ID}, fidelity=None, every_fidelity=True
    )
    assert [c.id for c, _ in result[None]] == ["c-3", "c-2", "c-1"]


def test_newest_fidelity_round_tie_goes_to_last() -> None:
    a = _cand("c-a", round=2, fid="a")
    b = _cand("c-b", round=2, fid="b").model_copy(update={"run_id": "other"})
    assert newest_fidelity(iter([a, _cand("low", round=1), b])) == "b"
    assert newest_fidelity([b, a]) == "a"
