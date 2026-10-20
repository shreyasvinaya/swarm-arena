import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from unittest.mock import ANY

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Goal, PartSpec, Plan, Ref
from swarm_arena._engine.solution import SolutionDocument, solution_view
from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.lease import Head
from swarm_arena._journal.replay import GoalState, RunState
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.results import Candidate, Score
from swarm_arena._model.tooling import ToolOutput, ToolRequest

H = Head(epoch=1, seq=99, segment="journal/e1-99.jsonl")


def _c(id: str, *, replicate: int = 0, part: str | None = None) -> Candidate:
    return Candidate(
        id=id,
        run_id="r",
        goal_id="g",
        part=part,
        team="t",
        agent="a",
        round=1,
        tool_identity="ti",
        request=ToolRequest(inputs={"x": 1, "optional": None}, replicate=replicate),
        output=ToolOutput(status="ok", data={"value": None}),
        parent_ids=["c-0"] if id == "c-2" else [],
    )


def _state_finished() -> RunState:
    plan = Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[],
        judges=[Ref(path="missing_component:Judge", name="program")],
    )
    candidates = [_c("c-0"), _c("c-1", replicate=1), _c("c-2", replicate=2)]
    return RunState.from_events(
        [
            _ev(1, "plan_revision", {"plan": plan.model_dump(mode="json"), "origin": "human"}),
            _ev(2, "run_started", {"manifest": {"run_id": "r", "git_commit": None}}),
            *[
                _ev(3 + i, "tool_end", {"candidate": c.model_dump(mode="json")})
                for i, c in enumerate(candidates)
            ],
            *[
                _ev(
                    6 + i,
                    "score",
                    {
                        "score": Score(
                            id=f"s-{i}",
                            candidate_id=c.id,
                            judge="program",
                            rubric_hash="stamped-rubric",
                            key=(float(i + 1),),
                            value=float(i + 1),
                        ).model_dump(mode="json")
                    },
                )
                for i, c in enumerate(candidates)
            ],
            _ev(9, "final_order", {"candidate_ids": ["c-2", "c-1"], "parts": None}),
            _ev(10, "goal_finished", {"reason": None}),
        ]
    )


def _ev(seq: int, type: str, payload: dict[str, Any]) -> Event:
    return Event(
        seq=seq,
        epoch=1,
        ts=datetime(2026, 10, 20, tzinfo=UTC),
        run_id="r",
        context_id="r",
        goal_id="g",
        type=EventType(type),
        payload=payload,
    )


def _state_no_result(reason: str) -> RunState:
    state = _state_finished()
    state.goals["g"] = state.goals["g"].model_copy(
        update={
            "status": "no_result",
            "no_result_reason": reason,
            "end_reason": "stop",
        }
    )
    return state


def _state_unjoined() -> RunState:
    state = _state_finished()
    plan, origin = state.plans["g"][0]
    state.plans["g"] = [
        (
            plan.model_copy(
                update={
                    "parts": [PartSpec(name=p, goal=Goal(id=p, namespace="n")) for p in ("a", "b")]
                }
            ),
            origin,
        )
    ]
    candidate = _c("c-a2", part="a")
    state.candidates[candidate.id] = candidate
    score = Score(id="s-a", candidate_id=candidate.id, judge="program", key=(1.0,), value=1.0)
    state.scores[score.id] = score
    state.goals["g"] = state.goals["g"].model_copy(
        update={
            "final_order": [],
            "final_parts": {"b": None, "a": "c-a2"},
            "end_reason": "stop --now",
        }
    )
    return state


def test_ok_solution_in_final_order_with_lineage_and_stats() -> None:
    data = json.loads(solution_view(_state_finished(), "g", H))
    assert data["status"] == "ok"
    assert [f["candidate"]["id"] for f in data["finalists"]] == ["c-2", "c-1"]
    assert [f["rank"] for f in data["finalists"]] == [1, 2]
    assert data["finalists"][0]["lineage"] == ["c-0"]
    assert data["source_head"]["seq"] == H.seq
    assert data["replicate_stats"] == [
        {
            "part": None,
            "judge": "program",
            "rubric_hash": "stamped-rubric",
            "panel_hash": "",
            "fidelity": "",
            "tool_identity": "ti",
            "inputs_sha256": ANY,
            "n": 3,
            "mean": 2.0,
            "min": 1.0,
            "max": 3.0,
        }
    ]
    assert data["replicate_stats"][0]["inputs_sha256"] == sha256_hex(
        canonical_json({"optional": None, "x": 1})
    )
    assert data["best_of_n"] == {"candidate_id": "c-2", "value": 3.0}
    assert data["finalists"][0]["plan_revision"] == 0
    assert data["manifest"] == {"run_id": "r", "git_commit": None}
    assert data["plan_revisions"][0]["goal"]["id"] == "g"
    assert data["parts"] == []
    assert SolutionDocument.model_validate_json(solution_view(_state_finished(), "g", H))


def test_no_result_solution() -> None:
    data = json.loads(solution_view(_state_no_result("no passing candidate"), "g", H))
    assert (data["status"], data["reason"], data["finalists"]) == (
        "no_result",
        "no passing candidate",
        [],
    )
    assert data["parts"] == []
    assert data["best_of_n"] is None


def test_unjoined_parts_solution_lists_best_of_each_part() -> None:
    data = json.loads(solution_view(_state_unjoined(), "g", H))
    assert (data["status"], data["reason"], data["finalists"]) == ("unjoined", "stop --now", [])
    assert [
        (p["part"], p["candidate"]["id"] if p["candidate"] else None) for p in data["parts"]
    ] == [("a", "c-a2"), ("b", None)]
    assert data["parts"][0]["scores"][0]["candidate_id"] == "c-a2"
    assert data["parts"][1] == {"part": "b", "candidate": None, "scores": [], "lineage": []}
    assert data["best_of_n"] is None


def test_view_is_pure() -> None:
    state = _state_finished()
    before = deepcopy(state)
    first = solution_view(state, "g", H)
    assert first == solution_view(state, "g", H) == canonical_json(json.loads(first))
    assert state == before


def test_replicate_stats_use_latest_matching_identity_and_fidelity() -> None:
    state = _state_finished()
    for id, judge, rubric, value in [
        ("s-aux", "auxiliary", "", 900),
        ("s-old", "program", "old", 800),
        ("s-final", "program", "stamped-rubric", 3),
    ]:
        state.scores[id] = Score(
            id=id,
            candidate_id="c-2",
            judge=judge,
            rubric_hash=rubric,
            key=(float(value),),
            value=float(value),
        )
    old = _c("c-old", replicate=3).model_copy(update={"round": 0, "fidelity": "old"})
    other = _c("c-other", replicate=4).model_copy(update={"goal_id": "other"})
    for candidate in (old, other):
        state.candidates[candidate.id] = candidate
        state.scores[candidate.id] = Score(
            candidate_id=candidate.id,
            judge="program",
            rubric_hash="stamped-rubric",
            key=(1000.0,),
            value=1000.0,
        )
    data = json.loads(solution_view(state, "g", H))
    assert data["replicate_stats"][0]["n"] == 3
    assert data["replicate_stats"][0]["mean"] == 2.0
    assert data["best_of_n"]["value"] == 3.0


@pytest.mark.parametrize(
    ("parts", "judges", "fidelities"),
    [
        (("a", "b"), ("a", "b"), ("a", "b")),
        (("a", "b"), ("main", "main"), ("same", "same")),
        ((None, "a"), ("main", "main"), ("same", "same")),
    ],
    ids=["review-repro", "parts-same-identity-fidelity", "main-and-part"],
)
def test_replicate_stats_keep_comparison_domains_separate(
    parts: tuple[str | None, str | None],
    judges: tuple[str, str],
    fidelities: tuple[str, str],
) -> None:
    plan = Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[],
        judges=[Ref(path="missing:Judge", name="main")],
        parts=[
            PartSpec(
                name=part,
                goal=Goal(id=part, namespace="n"),
                judges=[Ref(path="missing:Judge", name=judge)],
            )
            for part, judge in zip(parts, judges, strict=True)
            if part is not None
        ],
    )
    candidates = [
        _c(f"c-{i}", part=part, replicate=1).model_copy(
            update={"fidelity": fidelity, "request": ToolRequest(inputs={"x": 1}, replicate=1)}
        )
        for i, (part, fidelity) in enumerate(zip(parts, fidelities, strict=True))
    ]
    scores = [
        Score(
            id=f"s-{i}",
            candidate_id=candidate.id,
            judge=judge,
            rubric_hash=f"rubric-{judge}",
            panel_hash=f"panel-{judge}",
            key=(value,),
            value=value,
        )
        for i, (candidate, judge, value) in enumerate(
            zip(candidates, judges, (1.0, 1000.0), strict=True)
        )
    ]
    state = RunState(
        run_id="r",
        plans={"g": [(plan, "human")]},
        goals={
            "g": GoalState(
                status="stopped",
                final_parts={
                    candidate.part: candidate.id for candidate in candidates if candidate.part
                },
                end_reason="stop --now",
            )
        },
        candidates={candidate.id: candidate for candidate in candidates},
        scores={score.id: score for score in scores},
    )
    stats = json.loads(solution_view(state, "g", H))["replicate_stats"]
    assert len(stats) == 2
    assert stats == [
        {
            "part": candidate.part,
            "judge": score.judge,
            "rubric_hash": score.rubric_hash,
            "panel_hash": score.panel_hash,
            "fidelity": candidate.fidelity,
            "tool_identity": "ti",
            "inputs_sha256": sha256_hex(canonical_json({"x": 1})),
            "n": 1,
            "mean": score.value,
            "min": score.value,
            "max": score.value,
        }
        for candidate, score in zip(candidates, scores, strict=True)
    ]


def test_deterministic_groups_and_missing_values_have_no_stats() -> None:
    state = _state_finished()
    state.candidates = {
        id: c.model_copy(update={"request": c.request.model_copy(update={"replicate": 0})})
        for id, c in state.candidates.items()
    }
    assert json.loads(solution_view(state, "g", H))["replicate_stats"] == []
    state = _state_finished()
    state.scores = {id: s.model_copy(update={"value": None}) for id, s in state.scores.items()}
    data = json.loads(solution_view(state, "g", H))
    assert data["replicate_stats"] == []
    assert data["best_of_n"] is None


def test_unnamed_judge_uses_stamped_score_without_imports() -> None:
    state = _state_finished()
    plan, origin = state.plans["g"][0]
    state.plans["g"] = [
        (
            plan.model_copy(
                update={
                    "judges": [Ref(path="missing_component:Judge", config={"secret": "hidden"})]
                }
            ),
            origin,
        )
    ]
    assert json.loads(solution_view(state, "g", H))["replicate_stats"][0]["mean"] == 2.0


@pytest.mark.parametrize(
    "field", ["manifest", "best_of_n", "finalists", "parts", "replicate_stats", "plan_revisions"]
)
@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), {1: "value"}, {"x"}, b"bytes", "\ud800"]
)
def test_solution_document_rejects_non_json_subtrees(field: str, bad: Any) -> None:
    raw = json.loads(solution_view(_state_finished(), "g", H))
    value = {"SOLUTION_SECRET": [bad]}
    raw[field] = [value] if isinstance(raw[field], list) else value
    with pytest.raises(ValueError, match=r"JSON|UTF-8|validation") as caught:
        SolutionDocument.model_validate(raw)
    assert_secret_absent(caught.value, "SOLUTION_SECRET")


def test_solution_refuses_newer_version() -> None:
    raw = json.loads(solution_view(_state_finished(), "g", H))
    raw["schema_version"] = 2
    with pytest.raises(ValueError, match="newer"):
        SolutionDocument.model_validate_json(json.dumps(raw))


def test_no_result_takes_priority_over_unjoined_parts_even_with_null_reason() -> None:
    state = _state_unjoined()
    state.goals["g"] = state.goals["g"].model_copy(
        update={
            "status": "no_result",
            "no_result_reason": None,
        }
    )
    data = json.loads(solution_view(state, "g", H))
    assert (data["status"], data["reason"], data["parts"], data["finalists"]) == (
        "no_result",
        None,
        [],
        [],
    )


def test_final_order_remains_authoritative_over_score_sorting() -> None:
    state = _state_finished()
    state.goals["g"] = state.goals["g"].model_copy(update={"final_order": ["c-1", "c-2"]})
    data = json.loads(solution_view(state, "g", H))
    assert [f["candidate"]["id"] for f in data["finalists"]] == ["c-1", "c-2"]
    assert data["best_of_n"] == {"candidate_id": "c-1", "value": 2.0}


def test_replicate_groups_use_tool_identity_and_canonical_inputs() -> None:
    state = _state_finished()
    original = state.candidates["c-2"]
    state.candidates["c-2"] = original.model_copy(
        update={
            "request": original.request.model_copy(update={"inputs": {"optional": None, "x": 1}}),
        }
    )
    for id, tool, inputs in [
        ("c-new-tool", "other", {"x": 1, "optional": None}),
        ("c-new-input", "ti", {"x": 2}),
    ]:
        candidate = _c(id, replicate=1).model_copy(
            update={
                "tool_identity": tool,
                "request": ToolRequest(inputs=inputs, replicate=1),
            }
        )
        state.candidates[id] = candidate
        state.scores[id] = Score(
            id=id,
            candidate_id=id,
            judge="program",
            rubric_hash="stamped-rubric",
            key=(4.0,),
            value=4.0,
        )
    stats = json.loads(solution_view(state, "g", H))["replicate_stats"]
    assert sorted(s["n"] for s in stats) == [1, 1, 3]
    assert len({(s["tool_identity"], s["inputs_sha256"]) for s in stats}) == 3


def test_solution_rejects_invalid_json_without_exposing_input() -> None:
    malformed = '{"SOLUTION_SECRET":'
    with pytest.raises(ValueError, match="json_invalid") as caught:
        SolutionDocument.model_validate_json(malformed)
    assert_secret_absent(caught.value, "SOLUTION_SECRET")
