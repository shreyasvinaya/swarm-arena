"""Validate connector result contracts and shared return-type predicates."""

from collections.abc import Sequence
from typing import TypeVar

from swarm_arena._model.planning_types import RoundReport
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._outcomes import _propagate_control
from swarm_arena._scoring.judge import Judge, JudgeContext, is_finite_score

_M = TypeVar("_M", CheckResult, Score, RoundReport)


def _fail(message: str) -> AssertionError:
    """Build the error for a broken contract; it survives ``python -O``, unlike ``assert``."""
    return AssertionError(message)


def _strict_copy(model: type[_M], value: _M) -> _M | None:
    """Validate a returned model again, strictly; None if it is invalid or needs a repair."""
    try:  # model_copy and model_construct skip validation
        dump = value.model_dump(warnings=False)
        valid = model.model_validate(dump, strict=True)
        return valid if valid.model_dump(warnings=False) == dump else None
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        _propagate_control(error)
        return None


def _is_sequence_of(value: object, kind: type) -> bool:
    """Return whether a value is a non-str sequence whose items are all ``kind``."""
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(isinstance(item, kind) for item in value)
    )


def _is_set_of_str(value: object) -> bool:
    """Return whether a value is a frozenset of str."""
    return isinstance(value, frozenset) and all(isinstance(item, str) for item in value)


def _check_result(result: object) -> CheckResult:
    """Assert that a check returned a CheckResult that is valid as it is."""
    if not isinstance(result, CheckResult):
        raise _fail("Check.check() must return a CheckResult")
    valid = _strict_copy(CheckResult, result)
    if valid is None:  # lax validation would turn a truthy passed="no" into False
        raise _fail("Check.check() must return a valid CheckResult")
    return valid


def _check_scores(
    scores: object, candidates: Sequence[Candidate], ctx: JudgeContext
) -> tuple[list[Score], set[str]]:
    """Assert a judge's scores; record a non-finite key as a judge error (spec 4.4)."""
    if not (isinstance(scores, list) and all(isinstance(score, Score) for score in scores)):
        raise _fail("Judge.judge() must return a list of Score")
    batch = {candidate.id for candidate in candidates}
    scored: set[str] = set()
    valid: list[Score] = []
    for score in scores:
        candidate_id = score.candidate_id
        if not isinstance(candidate_id, str) or candidate_id not in batch:
            raise _fail(f"Judge.judge() returned a score for unknown candidate {candidate_id!r}")
        if candidate_id in scored:
            raise _fail(f"Judge.judge() returned more than one score for candidate {candidate_id}")
        scored.add(candidate_id)
        if not is_finite_score(score):
            ctx.judge_error(candidate_id, "score key must be nonempty and finite")
            continue
        copy = _strict_copy(Score, score)
        if copy is None:  # lax validation would turn value="1.0" into 1.0
            raise _fail("Judge.judge() must return valid Scores")
        valid.append(copy)
    return valid, scored


def _check_judge_methods(ref: Ref, judge: Judge) -> None:
    """Assert a stable judge identity and the types of the optional Judge methods."""
    rubric = judge.rubric_hash()
    same = rubric == judge.rubric_hash() == ref.build(Judge).rubric_hash()
    if not (isinstance(rubric, str) and rubric and same):
        raise _fail("Judge.rubric_hash() must return the same nonempty str")
    panel = judge.panel_hash()
    if not (isinstance(panel, str) and judge.panel_hash() == panel):
        raise _fail("Judge.panel_hash() must return the same str")
    anchors = judge.anchor_count()
    if not (isinstance(anchors, int) and not isinstance(anchors, bool) and anchors >= 0):
        raise _fail("Judge.anchor_count() must return an int >= 0")
    if not _is_set_of_str(judge.providers()):
        raise _fail("Judge.providers() must return a frozenset of str")
    models = judge.models()
    if not (isinstance(models, tuple) and all(isinstance(model, str) for model in models)):
        raise _fail("Judge.models() must return a tuple of str")
