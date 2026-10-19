import threading
import traceback
from collections.abc import Callable
from typing import Any

import pytest
from _fakes.tools import GOAL, cand, rosen_score
from pydantic import ValidationError

from swarm_arena._model.goal import Goal
from swarm_arena._model.results import Candidate, Score
from swarm_arena._scoring import program_judge
from swarm_arena._scoring.judge import JudgeContext
from swarm_arena._scoring.program_judge import ProgramJudge

SECRET = "SENTINEL_TASK21_SECRET"


async def test_scores_each_candidate(judge_ctx: JudgeContext) -> None:
    c1, c2 = cand(id="c1", data={"f": 2.0}), cand(id="c2", data={"f": 0.5})
    scores = await ProgramJudge(rosen_score).judge(GOAL, [c1, c2], judge_ctx)
    assert [(s.candidate_id, s.key, s.value) for s in scores] == [
        ("c1", (-2.0,), 2.0),
        ("c2", (-0.5,), 0.5),
    ]
    assert judge_ctx.errors == {}


@pytest.mark.parametrize(
    ("fn", "reason"),
    [
        ("_fakes.tools:raising_score", "RuntimeError"),
        ("_fakes.tools:nan_score", "finite"),
        ("_fakes.tools:list_score", "returned list, expected Score"),
    ],
)
async def test_failures_are_judge_errors(judge_ctx: JudgeContext, fn: str, reason: str) -> None:
    assert await ProgramJudge(fn).judge(GOAL, [cand(id="c1", data={"f": 1.0})], judge_ctx) == []
    assert reason in judge_ctx.errors["c1"]


def test_forms_and_preflight() -> None:
    assert ProgramJudge(rosen_score).config == ProgramJudge("_fakes.tools:rosen_score").config
    assert ProgramJudge(ProgramJudge.Config(function="_fakes.tools:rosen_score")).config == (
        ProgramJudge(rosen_score).config
    )
    with pytest.raises(TypeError, match="lambda"):
        ProgramJudge(lambda g, c: Score.minimize(0))
    with pytest.raises(TypeError, match="sync"):
        ProgramJudge("_fakes.tools:async_score").preflight()
    with pytest.raises(TypeError, match="function"):
        ProgramJudge("_fakes.tools:GOAL").preflight()


@pytest.mark.parametrize("key", [(), (float("nan"),), (float("inf"),), None, (SECRET,), 7])
async def test_malformed_keys_are_candidate_errors(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch, key: Any
) -> None:
    def score(goal: Goal, candidate: Candidate) -> Score:
        return (
            Score.maximize(2).model_copy(update={"key": key})
            if candidate.id == "bad"
            else (Score.maximize(3))
        )

    monkeypatch.setattr(program_judge, "load_object", lambda path: score)
    scores = await ProgramJudge("example:score").judge(
        GOAL, [cand(id="bad"), cand(id="good")], judge_ctx
    )
    assert [(s.candidate_id, s.key) for s in scores] == [("good", (3.0,))]
    assert "bad" in judge_ctx.errors
    assert SECRET not in repr(judge_ctx.errors)


async def test_exception_isolated_sanitized_and_off_event_loop(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_thread = threading.get_ident()
    seen: list[int] = []

    def score(goal: Goal, candidate: Candidate) -> Score:
        seen.append(threading.get_ident())
        if candidate.id == "bad":
            raise RuntimeError(SECRET)
        return Score.maximize(4).model_copy(update={"candidate_id": "wrong"})

    monkeypatch.setattr(program_judge, "load_object", lambda path: score)
    scores = await ProgramJudge("example:score").judge(
        GOAL, [cand(id="bad"), cand(id="good")], judge_ctx
    )
    assert [s.candidate_id for s in scores] == ["good"]
    assert len(seen) == 2
    assert all(t != main_thread for t in seen)
    assert judge_ctx.errors["bad"].startswith("RuntimeError")
    assert SECRET not in repr(judge_ctx.errors)


def test_lazy_import_and_sanitized_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: str) -> object:
        raise RuntimeError(SECRET)

    monkeypatch.setattr(program_judge, "load_object", fail)
    judge = ProgramJudge("example:score")
    with pytest.raises(ValueError, match="cannot import example:score") as caught:
        judge.preflight()
    assert_safe_error(caught.value)


def assert_safe_error(error: Exception) -> None:
    surfaces = [
        str(error),
        repr(error),
        repr(error.args),
        repr(error.__notes__) if hasattr(error, "__notes__") else "",
        repr(error.__cause__),
        "".join(traceback.format_exception(error)),
    ]
    if isinstance(error, ValidationError):
        surfaces += [repr(error.errors()), error.json()]
    assert all(SECRET not in surface for surface in surfaces)


@pytest.mark.parametrize(
    "make",
    [
        lambda: ProgramJudge.Config.model_validate({"function": {"private": SECRET}}),
        lambda: ProgramJudge.Config.model_validate_json('{"function": ' + SECRET),
        lambda: ProgramJudge.Config.model_validate({"unknown": SECRET}),
    ],
)
def test_config_errors_are_sanitized(make: Callable[[], object]) -> None:
    with pytest.raises(ValidationError) as caught:
        make()
    assert_safe_error(caught.value)
