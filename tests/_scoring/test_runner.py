import asyncio
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import GOAL, AlwaysPass, RecordingExecutor, cand

from swarm_arena._errors import Fenced, ProviderFailed, _core_error, _SignalCategory
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._model.goal import Goal
from swarm_arena._model.ids import IdGen
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._model.tooling import Command
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._scoring.judge import Judge, JudgeContext
from swarm_arena._scoring.program_judge import ProgramJudge
from swarm_arena._scoring.runner import (
    JudgeError,
    JudgeOutcome,
    final_judge,
    for_manager,
    judge_round,
    run_checks,
)
from swarm_arena._scoring.script_check import ScriptCheck
from swarm_arena._scoring.script_judge import ScriptJudge

SECRET = "runner-private-7e5cd74b"


@pytest.mark.parametrize("nested", [False, True])
async def test_run_checks_preserves_terminal(nested: bool) -> None:
    from unittest.mock import AsyncMock, Mock

    from swarm_arena._errors import _core_category, _stop_cleanup_error

    error: BaseException = _stop_cleanup_error()
    if nested:
        error = BaseExceptionGroup("check", [BaseExceptionGroup("hook", [error])])
    check = Mock(spec=Check, check=AsyncMock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        await run_checks(_cand(), [("check", check)], _check_ctx())
    assert _core_category(caught.value) is _SignalCategory.TERMINAL


def _cand(candidate_id: str = "c-1", *, failed: bool = False, **fields: Any) -> Candidate:
    checks = [CheckResult(check="gate", passed=False)] if failed else []
    defaults: dict[str, Any] = {"id": candidate_id, "checks": checks}
    return cand(**{**defaults, **fields})


def _judge_ctx(**fields: Any) -> JudgeContext:
    defaults: dict[str, Any] = {
        "goal": GOAL,
        "executor": RecordingExecutor(),
        "artifact_root": "/a",
    }
    return JudgeContext(**{**defaults, **fields})


def _check_ctx(**fields: Any) -> CheckContext:
    defaults: dict[str, Any] = {
        "goal": GOAL,
        "executor": RecordingExecutor(),
        "artifact_root": "/a",
    }
    return CheckContext(**{**defaults, **fields})


def _script_check() -> ScriptCheck:
    return ScriptCheck(command=[sys.executable, "-c", "pass"])


def _script_judge() -> ScriptJudge:
    return ScriptJudge(command=[sys.executable, "-c", "pass"])


class _PassCheck(AlwaysPass):
    pass


class _RaisingCheck(Check):
    name = "raising"

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        raise RuntimeError(SECRET)


class _RecordingJudge(Judge):
    name = "recording"

    def __init__(
        self, *, values: Mapping[str, float | None] | None = None, anchor_count: int = 0
    ) -> None:
        super().__init__()
        self.values = values or {}
        self.count = anchor_count
        self.seen: list[list[str]] = []
        self.last_ctx: JudgeContext | None = None

    def anchor_count(self) -> int:
        return self.count

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        self.seen.append([candidate.id for candidate in candidates])
        self.last_ctx = ctx
        return [
            Score.maximize(value).model_copy(update={"candidate_id": candidate.id})
            for candidate in candidates
            if (value := self.values.get(candidate.id, 1.0)) is not None
        ]


class _NanJudge(Judge):
    name = "nan"

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            Score.model_construct(candidate_id=candidate.id, key=(float("nan"),))
            for candidate in candidates
        ]


class _BoomJudge(Judge):
    name = "boom"

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        raise RuntimeError(SECRET)


class _ErrorRaisingJudge(_BoomJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        ctx.judge_error("c-2", "candidate failed")
        ctx.judge_error("c-2", "later failure")
        return await super().judge(goal, candidates, ctx)


class _DuplicatingJudge(_RecordingJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            Score(key=(9.0,), candidate_id="foreign"),
            Score(key=(1.0,), candidate_id="c-1"),
            Score(key=(99.0,), candidate_id="c-1"),
            Score.model_construct(key=(float("nan"),), candidate_id="c-2"),
            Score(key=(5.0,), candidate_id="c-2"),
        ]


class _ErrorReturningJudge(_RecordingJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        ctx.judge_error("c-1", "candidate failed")
        return await super().judge(goal, candidates, ctx)


class _KeyJudge(_RecordingJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        await super().judge(goal, candidates, ctx)
        keys = {"c-z": (2.0, 0.0), "c-a": (1.0, 9.0), "c-b": (2.0, 0.0)}
        return [Score(key=keys[c.id], candidate_id=c.id, value=100.0) for c in candidates]


async def test_check_exception_is_a_failed_check() -> None:
    res = await run_checks(_cand(), [("ok", _PassCheck()), ("boom", _RaisingCheck())], _check_ctx())
    assert [r.passed for r in res] == [True, False]
    assert [r.check for r in res] == ["ok", "boom"]
    assert res[1].reasons == ["check error: RuntimeError"]
    assert SECRET not in repr(res) + str([r.model_dump() for r in res])


async def test_checks_continue_in_plan_order_with_independent_contexts() -> None:
    contexts: list[CheckContext] = []
    ids = IdGen("run-1", 1)

    class RecordingCheck(Check):
        name = "recording"

        async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
            assert ctx.round == 3
            assert ctx.ids is ids
            contexts.append(ctx)
            ctx.round = 99
            return CheckResult(check=self.name, passed=True)

    ctx = _check_ctx(round=3, ids=ids)
    results = await run_checks(
        _cand(),
        [("first", RecordingCheck()), ("boom", _RaisingCheck()), ("last", RecordingCheck())],
        ctx,
    )
    assert [result.check for result in results] == ["first", "boom", "last"]
    assert contexts[0] is not contexts[1]
    assert all(context is not ctx for context in contexts)
    assert ctx.round == 3


@pytest.mark.parametrize(
    "result",
    [
        {"check": "wrong", "passed": True},
        None,
        CheckResult.model_construct(check="wrong", passed=SECRET),
        CheckResult.model_construct(check="wrong", passed=True, details={SECRET: float("nan")}),
        CheckResult.model_construct(check="wrong"),
    ],
)
async def test_invalid_check_results_fail_closed_without_values(result: object) -> None:
    class BrokenCheck(Check):
        name = "broken"

        async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
            return cast(CheckResult, result)

    [out] = await run_checks(_cand(), [("planned", BrokenCheck())], _check_ctx())
    assert out == CheckResult(
        check="planned", passed=False, reasons=["check error: invalid result"]
    )
    assert SECRET not in out.model_dump_json()


async def test_check_results_use_lax_revalidation_before_name_stamping() -> None:
    raw = CheckResult.model_construct(check="built-in", passed="no", reasons=("why",))

    class LaxCheck(Check):
        name = "lax"

        async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
            return raw

    [result] = await run_checks(_cand(), [("planned", LaxCheck())], _check_ctx())
    assert result == CheckResult(check="planned", passed=False, reasons=["why"])
    assert raw.check == "built-in"
    assert raw.passed == "no"


async def test_judge_round_skips_gated_out_and_adds_anchors() -> None:
    judge = _RecordingJudge(anchor_count=1)
    out = await judge_round(
        "g",
        [(Ref.of(judge), judge)],
        [_cand("c-1"), _cand("c-2", failed=True)],
        leaders=[_cand("c-0"), _cand("c-9")],
        ctx_for=lambda j: _judge_ctx(),
    )
    assert judge.seen == [["c-1", "c-0"]]
    assert {s.candidate_id for s in out.scores} == {"c-1", "c-0"}
    assert out.errors == []
    assert judge.last_ctx is not None
    assert judge.last_ctx.batch_id == "g-recording-c-1"
    assert judge.last_ctx.anchors == frozenset({"c-0"})
    assert {s.batch_id for s in out.scores} == {"g-recording-c-1"}


async def test_anchors_are_deduplicated_without_extending_the_prefix() -> None:
    judge = _RecordingJudge(anchor_count=2)
    ctx = _judge_ctx()
    await judge_round(
        "g",
        [(Ref.of(judge).model_copy(update={"name": "quality"}), judge)],
        [_cand("c-1")],
        [_cand("c-1"), _cand("c-0"), _cand("c-9")],
        ctx_for=lambda j: ctx,
    )
    assert judge.seen == [["c-1", "c-0"]]
    assert ctx.anchors == frozenset({"c-1", "c-0"})
    assert ctx.batch_id == "g-quality-c-1"


@pytest.mark.parametrize("candidates", [[], [_cand(failed=True)]])
async def test_empty_gated_in_batch_skips_judges_and_anchors(
    candidates: list[Candidate],
) -> None:
    judge = _RecordingJudge(anchor_count=1)

    def no_context(judge: Judge) -> JudgeContext:
        pytest.fail("an empty batch must not request a context")

    out = await judge_round("g", [(Ref.of(judge), judge)], candidates, [_cand("c-0")], no_context)
    assert out == JudgeOutcome(scores=[], errors=[])
    assert judge.seen == []


async def test_scores_are_stamped_with_ctx_ids() -> None:
    judge = _RecordingJudge()
    out = await judge_round(
        "g",
        [(Ref.of(judge).model_copy(update={"name": "quality"}), judge)],
        [_cand("c-1")],
        [],
        ctx_for=lambda j: _judge_ctx(ids=IdGen("run-1", 1)),
    )
    assert out.scores[0].id == IdGen("run-1", 1).next("s")
    assert out.scores[0].judge == "quality"
    assert out.scores[0].rubric_hash == judge.rubric_hash()


async def test_ctx_errors_become_judge_errors() -> None:
    judge = ProgramJudge("_fakes.tools:raising_score")
    out = await judge_round(
        "g",
        [(Ref.of(judge), judge)],
        [_cand("c-1"), _cand("c-2")],
        [],
        ctx_for=lambda j: _judge_ctx(),
    )
    assert out.scores == []
    assert [e.candidate_id for e in out.errors] == ["c-1", "c-2"]
    assert all(e.reason == "RuntimeError: scoring function raised" for e in out.errors)


async def test_missing_score_is_a_judge_error() -> None:
    judge = _RecordingJudge(values={"c-1": 1.0, "c-2": None})
    out = await judge_round(
        "g",
        [(Ref.of(judge), judge)],
        [_cand("c-1"), _cand("c-2")],
        [],
        ctx_for=lambda j: _judge_ctx(),
    )
    assert [s.candidate_id for s in out.scores] == ["c-1"]
    assert [(e.candidate_id, e.reason) for e in out.errors] == [("c-2", "no score")]


async def test_non_finite_and_raising_judges_become_errors() -> None:
    out = await judge_round(
        "g",
        [(Ref.of(_NanJudge()), _NanJudge()), (Ref.of(_BoomJudge()), _BoomJudge())],
        [_cand("c-1")],
        [],
        ctx_for=lambda j: _judge_ctx(),
    )
    assert out.scores == []
    assert [(e.candidate_id, e.judge, e.reason) for e in out.errors] == [
        ("c-1", "nan", "non-finite score"),
        ("c-1", "boom", "RuntimeError: judge raised"),
    ]
    assert SECRET not in out.model_dump_json()


async def test_context_error_precedes_raising_judge_and_contexts_are_per_judge() -> None:

    judges: list[Judge] = [_ErrorRaisingJudge(), _RecordingJudge()]
    contexts: list[JudgeContext] = []

    def context_for(judge: Judge) -> JudgeContext:
        ctx = _judge_ctx()
        contexts.append(ctx)
        return ctx

    out = await judge_round(
        "g",
        [(Ref.of(judge), judge) for judge in judges],
        [_cand("c-1"), _cand("c-2")],
        [],
        context_for,
    )
    assert [(e.candidate_id, e.reason) for e in out.errors] == [
        ("c-1", "RuntimeError: judge raised"),
        ("c-2", "candidate failed"),
    ]
    assert len(out.scores) == 2
    assert len(contexts) == 2
    assert contexts[0] is not contexts[1]
    assert contexts[1].errors == {}


async def test_foreign_and_duplicate_scores_are_discarded_before_stamping() -> None:

    judge = _DuplicatingJudge()
    ids = IdGen("run-1", 1)
    out = await judge_round(
        "g",
        [(Ref.of(judge), judge)],
        [_cand("c-1"), _cand("c-2")],
        [],
        lambda j: _judge_ctx(ids=ids),
    )
    assert [(s.candidate_id, s.key) for s in out.scores] == [("c-1", (1.0,))]
    assert [(e.candidate_id, e.reason) for e in out.errors] == [("c-2", "non-finite score")]
    assert ids.counters() == {"s": 1}


async def test_ctx_error_precedes_even_a_returned_score() -> None:

    judge = _ErrorReturningJudge()
    out = await judge_round("g", [(Ref.of(judge), judge)], [_cand()], [], lambda j: _judge_ctx())
    assert out.scores == []
    assert [e.reason for e in out.errors] == ["candidate failed"]


async def test_final_judge_one_batch_no_cache_read_orders_by_key() -> None:
    judge = _RecordingJudge(values={"c-1": 1.0, "c-2": 3.0, "c-3": None})
    ctx = _judge_ctx(ids=IdGen("run-1", 1), anchors=frozenset({"old"}))
    order, out = await final_judge(
        "g", (Ref.of(judge), judge), [_cand("c-1"), _cand("c-2"), _cand("c-3")], ctx
    )
    assert [c.id for c in order] == ["c-2", "c-1"]
    assert len(judge.seen) == 1
    assert judge.last_ctx is ctx
    assert ctx.final is True
    assert ctx.read_cache is False
    assert ctx.anchors == frozenset()
    assert ctx.batch_id == "g-recording-c-1"
    assert {s.candidate_id for s in out.scores} == {"c-1", "c-2"}
    expected_ids = IdGen("run-1", 1)
    assert [s.id for s in out.scores] == [expected_ids.next("s"), expected_ids.next("s")]
    assert [(e.candidate_id, e.reason) for e in out.errors] == [("c-3", "no score")]


async def test_final_judge_raising_judge_gives_no_order() -> None:
    order, out = await final_judge(
        "g",
        (Ref.of(_BoomJudge()), _BoomJudge()),
        [_cand("c-1"), _cand("c-2")],
        _judge_ctx(final=True),
    )
    assert order == []
    assert out.scores == []
    assert [e.candidate_id for e in out.errors] == ["c-1", "c-2"]
    assert all(e.reason == "RuntimeError: judge raised" for e in out.errors)
    assert SECRET not in out.model_dump_json()


async def test_final_judge_uses_lexicographic_keys_then_ids_across_fidelities() -> None:

    judge = _KeyJudge()
    order, out = await final_judge(
        "g",
        (Ref.of(judge), judge),
        [
            _cand("c-z", round=1, fidelity="low"),
            _cand("c-a", round=2, fidelity="high"),
            _cand("c-b", round=3, fidelity="high"),
        ],
        _judge_ctx(),
    )
    assert [candidate.id for candidate in order] == ["c-b", "c-z", "c-a"]
    assert len(out.scores) == 3
    assert judge.seen == [["c-z", "c-a", "c-b"]]


async def test_final_judge_empty_finalists_do_not_call_judge() -> None:
    judge = _RecordingJudge()
    order, out = await final_judge("g", (Ref.of(judge), judge), [], _judge_ctx())
    assert order == []
    assert out == JudgeOutcome(scores=[], errors=[])
    assert judge.seen == []


async def test_final_judge_reports_non_finite_scores() -> None:
    order, out = await final_judge("g", (Ref.of(_NanJudge()), _NanJudge()), [_cand()], _judge_ctx())
    assert order == []
    assert out.scores == []
    assert [e.reason for e in out.errors] == ["non-finite score"]


def test_for_manager_hides_reasons_of_pass_fail_checks() -> None:
    original = _cand(
        checks=[
            CheckResult(check="hidden", passed=False, reasons=[SECRET], details={"x": SECRET}),
            CheckResult(check="shown", passed=True, reasons=["why"], details={"n": 1}),
        ]
    )
    c = for_manager(original, {"hidden": "pass_fail", "shown": "reasons"})
    assert c.checks[0] == CheckResult(check="hidden", passed=False)
    assert c.checks[1] == original.checks[1]
    assert original.checks[0].reasons == [SECRET]
    assert original.checks[0].details == {"x": SECRET}
    assert c.id == original.id
    assert c.output == original.output
    assert all(
        not check.reasons and not check.details for check in for_manager(original, {}).checks
    )


def test_judge_outcome_and_error_are_frozen_models() -> None:
    error = JudgeError(candidate_id="c-1", judge="quality", reason="no score")
    outcome = JudgeOutcome(scores=[], errors=[error])
    assert outcome.model_dump() == {
        "scores": [],
        "provider_failed": {},
        "panel_drops": [],
        "errors": [{"candidate_id": "c-1", "judge": "quality", "reason": "no score"}],
    }
    with pytest.raises(ValueError, match="frozen"):
        error.reason = "changed"  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="frozen"):
        outcome.scores = []  # ty: ignore[invalid-assignment]


async def test_fenced_from_command_start_propagates(
    local_executor: LocalExecutor, artifact_root: str
) -> None:
    async def fenced(_: Job.Started) -> None:
        from swarm_arena._errors import _core_error

        raise _core_error(Fenced, "run ownership lost")

    with pytest.raises(Fenced, match="run ownership lost"):
        await run_checks(
            _cand(),
            [("s", _script_check())],
            _check_ctx(executor=local_executor, artifact_root=artifact_root, on_started=fenced),
        )
    sj = _script_judge()
    with pytest.raises(Fenced, match="run ownership lost"):
        await judge_round(
            "g",
            [(Ref.of(sj), sj)],
            [_cand("c-1")],
            [],
            ctx_for=lambda j: _judge_ctx(
                executor=local_executor, artifact_root=artifact_root, on_started=fenced
            ),
        )
    with pytest.raises(Fenced, match="run ownership lost"):
        await final_judge(
            "g",
            (Ref.of(sj), sj),
            [_cand("c-1")],
            _judge_ctx(executor=local_executor, artifact_root=artifact_root, on_started=fenced),
        )


class _CommandJudge(_RecordingJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        await ctx.execute(Command(argv=[sys.executable, "-c", "pass"]))
        return await super().judge(goal, candidates, ctx)


@pytest.mark.parametrize("final", [False, True])
async def test_command_jobs_and_scores_share_context_ids(final: bool) -> None:
    judge = _CommandJudge()
    ids = IdGen("run-1", 3)
    executor = RecordingExecutor()
    ctx = _judge_ctx(ids=ids, executor=executor)
    if final:
        _, outcome = await final_judge("g", (Ref.of(judge), judge), [_cand()], ctx)
    else:
        outcome = await judge_round("g", [(Ref.of(judge), judge)], [_cand()], [], lambda j: ctx)
    expected = IdGen("run-1", 3)
    assert executor.jobs[0].id == expected.next("x")
    assert outcome.scores[0].id == expected.next("s")
    assert ids.counters() == {"x": 1, "s": 1}


class _UnstampableBoomJudge(_BoomJudge):
    def rubric_hash(self) -> str:
        raise ValueError(SECRET)


@pytest.mark.parametrize("final", [False, True])
async def test_raising_judge_needs_no_score_stamping(final: bool) -> None:
    judge = _UnstampableBoomJudge()
    if final:
        order, outcome = await final_judge("g", (Ref.of(judge), judge), [_cand()], _judge_ctx())
        assert order == []
    else:
        outcome = await judge_round(
            "g", [(Ref.of(judge), judge)], [_cand()], [], lambda j: _judge_ctx()
        )
    assert outcome.scores == []
    assert [error.reason for error in outcome.errors] == ["RuntimeError: judge raised"]
    assert SECRET not in outcome.model_dump_json()


BOUNDARY_SECRET = "6ec2800b-c2de-4674-b121-cb946c0ef834"


def _private_hook_failure() -> ValueError:
    error = ValueError(BOUNDARY_SECRET)
    error.add_note(BOUNDARY_SECRET)
    error.__cause__ = RuntimeError(BOUNDARY_SECRET)
    return error


async def _boundary_outcome(judge: Judge, *, final: bool, ctx: JudgeContext) -> JudgeOutcome:
    candidates = [_cand("c-1"), _cand("c-2")]
    ref = Ref.of(_RecordingJudge()).model_copy(update={"name": "broken"})
    later = _RecordingJudge()
    if final:
        order, outcome = await final_judge("g", (ref, judge), candidates, ctx)
        assert order == []
        later_order, later_outcome = await final_judge(
            "g", (Ref.of(later), later), candidates, _judge_ctx()
        )
        assert [candidate.id for candidate in later_order] == ["c-1", "c-2"]
    else:
        outcome = await judge_round(
            "g",
            [(ref, judge), (Ref.of(later), later)],
            candidates,
            [_cand("c-0")],
            lambda current: ctx if current is judge else _judge_ctx(),
        )
        later_outcome = outcome
    assert later.seen == [["c-1", "c-2"]]
    assert [(score.candidate_id, score.judge) for score in later_outcome.scores] == [
        ("c-1", "recording"),
        ("c-2", "recording"),
    ]
    assert all(score.judge != "broken" for score in outcome.scores)
    for result in (outcome, later_outcome):
        assert BOUNDARY_SECRET not in str(result) + repr(result) + str(result.model_dump())
        assert_secret_absent(RuntimeError(result.model_dump_json()), BOUNDARY_SECRET)
        for event_type, entries in (
            (EventType.SCORE, result.scores),
            (EventType.JUDGE_ERROR, result.errors),
        ):
            for entry in entries:
                event = EventDraft(type=event_type, payload=entry.model_dump(mode="json"))
                assert EventDraft.model_validate_json(event.model_dump_json()) == event
                assert_secret_absent(RuntimeError(event.model_dump_json()), BOUNDARY_SECRET)
    return outcome


@pytest.mark.parametrize("hook", ["rubric_hash", "panel_hash", "anchor_count"])
@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("context_error", [False, True])
async def test_judge_hook_failures_are_contained(
    hook: str, final: bool, context_error: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = _RecordingJudge(anchor_count=1)
    failure = _private_hook_failure()

    def fail() -> Any:
        raise failure

    monkeypatch.setattr(judge, hook, fail)
    ctx = _judge_ctx(ids=IdGen("run-1", 1))
    if context_error:
        ctx.judge_error("c-2", "candidate failed")
    if final and hook == "anchor_count":
        # Final judging never consults the anchor hook.
        order, outcome = await final_judge("g", (Ref.of(judge), judge), [_cand()], ctx)
        assert [candidate.id for candidate in order] == ["c-1"]
        assert outcome.errors == []
        return
    try:
        outcome = await _boundary_outcome(judge, final=final, ctx=ctx)
    except Exception as error:
        assert_secret_absent(error, BOUNDARY_SECRET, sanitized=True)
        raise
    expected = [("c-1", "ValueError: judge raised")]
    expected.append(("c-2", "candidate failed" if context_error else "ValueError: judge raised"))
    if not final and hook != "anchor_count":
        expected.append(("c-0", "ValueError: judge raised"))
    assert [(error.candidate_id, error.reason) for error in outcome.errors] == expected
    assert ctx.ids.counters() == {}


class _MalformedReturnJudge(_RecordingJudge):
    def __init__(self, result: object) -> None:
        super().__init__()
        self.result = result

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return cast(list[Score], self.result)

    def rubric_hash(self) -> str:
        pytest.fail("malformed output must never reach score stamping")


@pytest.mark.parametrize(
    "result",
    [
        None,
        [Score(key=(1.0,), candidate_id="c-1"), BOUNDARY_SECRET],
        {"private": BOUNDARY_SECRET},
        "",
        b"",
    ],
)
@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("context_error", [False, True])
async def test_malformed_judge_output_fails_the_whole_invocation(
    result: object, final: bool, context_error: bool
) -> None:
    ctx = _judge_ctx(ids=IdGen("run-1", 1))
    if context_error:
        ctx.judge_error("c-2", "candidate failed")
    outcome = await _boundary_outcome(_MalformedReturnJudge(result), final=final, ctx=ctx)
    assert [(error.candidate_id, error.reason) for error in outcome.errors] == [
        ("c-1", "TypeError: judge raised"),
        ("c-2", "candidate failed" if context_error else "TypeError: judge raised"),
    ]
    assert ctx.ids.counters() == {}


@pytest.mark.parametrize(
    ("hook", "final"),
    [
        ("rubric_hash", False),
        ("rubric_hash", True),
        ("panel_hash", False),
        ("panel_hash", True),
        ("anchor_count", False),
    ],
)
@pytest.mark.parametrize("failure_type", [Fenced, asyncio.CancelledError])
async def test_judge_hooks_preserve_fencing_and_cancellation(
    hook: str,
    final: bool,
    failure_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = _RecordingJudge()
    later = _RecordingJudge()
    failure = (
        _core_error(Fenced, "run ownership lost")
        if failure_type is Fenced
        else failure_type("ownership or cancellation")
    )

    def fail() -> Any:
        raise failure

    monkeypatch.setattr(judge, hook, fail)
    call = (
        final_judge("g", (Ref.of(judge), judge), [_cand()], _judge_ctx())
        if final
        else judge_round(
            "g",
            [(Ref.of(judge), judge), (Ref.of(later), later)],
            [_cand()],
            [],
            lambda current: _judge_ctx(),
        )
    )
    if failure_type is Fenced:
        with pytest.raises(Fenced) as caught:
            await call
        assert caught.value is not failure
        assert caught.value.args == ("run ownership lost",)
        assert later.seen == []
    else:
        # These synchronous hooks raise user cancellation, without a pending
        # caller cancellation. PR1-7 requires a contained, type-only failure.
        result = await call
        outcome = result[1] if isinstance(result, tuple) else result
        assert len(outcome.errors) == 1
        assert outcome.errors[0].reason == "CancelledError: judge raised"
        assert "ownership or cancellation" not in outcome.model_dump_json()
        assert later.seen == ([] if final else [["c-1"]])


async def test_prepared_batch_funded_size_equals_submitted_batch() -> None:
    from swarm_arena._scoring.runner import prepare_judge_batch

    class Once(Judge):
        name = "once"

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.seen: list[str] = []

        def anchor_count(self) -> int:
            self.calls += 1
            assert self.calls == 1
            return 2

        async def judge(
            self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
        ) -> list[Score]:
            self.seen = [c.id for c in candidates]
            assert ctx.anchors == frozenset({"c-1", "c-2"})
            return [Score.maximize(1).model_copy(update={"candidate_id": c.id}) for c in candidates]

    judge = Once()
    candidates = [_cand("c-1")]
    prepared = prepare_judge_batch(judge, candidates, [*candidates, _cand("c-2")])
    assert len(prepared[0]) == 2
    outcome = await judge_round(
        "g",
        [(Ref(path="_fakes.tools:FixedJudge", name="once"), judge)],
        candidates,
        [],
        lambda judge: _judge_ctx(),
        prepared=prepared,
    )
    assert judge.seen == ["c-1", "c-2"]
    assert len(outcome.scores) == 2


@pytest.mark.parametrize("policy", [{}, {"gate": "unknown"}])
def test_for_manager_unknown_policy_hides_feedback(policy: dict[str, str]) -> None:
    original = _cand(
        checks=[
            CheckResult(check="gate", passed=True, reasons=[SECRET], details={"private": SECRET})
        ]
    )
    filtered = for_manager(original, policy)
    assert filtered.checks[0].reasons == []
    assert filtered.checks[0].details == {}
    assert original.checks[0].reasons == [SECRET]
    assert original.checks[0].details == {"private": SECRET}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_call_ids", [object()]),
        ("notes", {"private": BOUNDARY_SECRET}),
        ("notes", "\ud800"),
        ("value", float("nan")),
        ("value", float("inf")),
        ("panel_hash", {"private": BOUNDARY_SECRET}),
    ],
)
async def test_complete_score_validation_contains_invalid_metadata(field: str, value: Any) -> None:
    class Malformed(_RecordingJudge):
        async def judge(
            self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
        ) -> list[Score]:
            return [
                score.model_copy(update={field: value})
                for score in await super().judge(goal, candidates, ctx)
            ]

    bad, good = Malformed(), _RecordingJudge()
    refs = [Ref.of(_RecordingJudge()).model_copy(update={"name": "bad"}), Ref.of(good)]
    out = await judge_round(
        "g", list(zip(refs, [bad, good], strict=True)), [_cand()], [], lambda _: _judge_ctx()
    )
    assert len(out.scores) == len(out.errors) == 1
    assert out.scores[0].judge == "recording"
    assert out.errors[0].judge == "bad"
    expected = "TypeError" if field == "llm_call_ids" else "ValidationError"
    assert out.errors[0].reason == f"{expected}: judge raised"
    assert BOUNDARY_SECRET not in out.model_dump_json()


async def test_score_serializer_failure_is_contained() -> None:
    from pydantic import model_serializer

    class BadDump(Score):
        @model_serializer
        def serialize(self) -> dict[str, Any]:
            error = RuntimeError(BOUNDARY_SECRET)
            error.add_note(BOUNDARY_SECRET)
            raise error from ValueError(BOUNDARY_SECRET)

    class Malformed(_RecordingJudge):
        async def judge(
            self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
        ) -> list[Score]:
            return [BadDump(candidate_id=c.id, key=(1.0,)) for c in candidates]

    judge = Malformed()
    out = await judge_round(
        "g", [(Ref.of(_RecordingJudge()), judge)], [_cand()], [], lambda _: _judge_ctx()
    )
    assert out.scores == []
    assert [e.reason for e in out.errors] == ["PydanticSerializationError: judge raised"]
    assert BOUNDARY_SECRET not in out.model_dump_json()


async def test_metadata_runs_off_loop_but_score_ids_stay_on_loop() -> None:
    import threading

    observed: list[threading.Thread] = []

    class Metadata(_RecordingJudge):
        def rubric_hash(self) -> str:
            observed.append(threading.current_thread())
            return "rubric"

        def panel_hash(self) -> str:
            observed.append(threading.current_thread())
            return "panel"

    class LoopIds(IdGen):
        def next(self, prefix: Literal["c", "L", "s", "r", "llm", "x", "k"]) -> str:
            assert threading.current_thread() is threading.main_thread()
            return super().next(prefix)

    judge = Metadata()
    ctx = _judge_ctx(ids=LoopIds("r", 1))
    out = await judge_round("g", [(Ref.of(_RecordingJudge()), judge)], [_cand()], [], lambda _: ctx)
    assert len(out.scores) == 1
    assert len(observed) == 2
    assert all(t is not threading.main_thread() and t.daemon for t in observed)


async def test_omitted_candidate_is_a_judge_error() -> None:
    judge = _RecordingJudge(values={"c0": 1.0, "c1": None})
    outcome = await judge_round(
        "g", [(Ref.of(judge), judge)], [_cand("c0"), _cand("c1")], [], lambda _: _judge_ctx()
    )
    assert [s.candidate_id for s in outcome.scores] == ["c0"]
    assert [(e.candidate_id, e.reason) for e in outcome.errors] == [("c1", "no score")]


class _ProviderJudge(_RecordingJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        raise _core_error(ProviderFailed, "anthropic")


class _ProviderMetadataJudge(_RecordingJudge):
    def panel_hash(self) -> str:
        raise _core_error(ProviderFailed, "anthropic")


class _TerminalMetadataJudge(_RecordingJudge):
    def panel_hash(self) -> str:
        raise _core_error(
            RuntimeError, "stop cleanup timed out", _category=_SignalCategory.TERMINAL
        )


@pytest.mark.parametrize(
    ("member", "error_type"),
    [(_ProviderMetadataJudge, ProviderFailed), (_TerminalMetadataJudge, RuntimeError)],
)
async def test_recursive_metadata_preserves_registered_signal(
    member: type[Judge], error_type: type[Exception]
) -> None:
    from swarm_arena.scoring import Lexicographic

    judge = Lexicographic([Lexicographic([member()])])
    # Preserve the actual instances; this test is about combination, not factories.
    inner = Lexicographic([member()])
    inner._members = [member()]
    judge._members = [inner]
    with pytest.raises(error_type):
        await judge.judge(GOAL, [_cand()], _judge_ctx())


async def test_judge_round_reports_provider_failed_without_judge_errors() -> None:
    failed, healthy = _ProviderJudge(), _RecordingJudge()
    outcome = await judge_round(
        "g",
        [
            (Ref.of(failed).model_copy(update={"name": "A"}), failed),
            (Ref.of(healthy).model_copy(update={"name": "B"}), healthy),
        ],
        [_cand("c-1"), _cand("c-2")],
        [],
        lambda _: _judge_ctx(),
    )
    assert outcome.provider_failed == {"A": "anthropic"}
    assert outcome.errors == []
    assert {s.judge for s in outcome.scores} == {"B"}


async def test_final_judge_reraises_provider_failed() -> None:
    judge = _ProviderJudge()
    with pytest.raises(ProviderFailed):
        await final_judge("g", (Ref.of(judge), judge), [_cand()], _judge_ctx())


@pytest.mark.parametrize("terminal", [False, True])
async def test_judge_round_reraises_registered_limits(terminal: bool) -> None:
    from swarm_arena.manager import LimitReached

    class Limited(_RecordingJudge):
        async def judge(
            self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
        ) -> list[Score]:
            raise _core_error(
                LimitReached,
                "budget",
                "judge budget",
                _category=(_SignalCategory.TERMINAL if terminal else _SignalCategory.LIMIT),
            )

    judge = Limited()
    with pytest.raises(LimitReached):
        await judge_round(
            "g", [(Ref.of(_RecordingJudge()), judge)], [_cand()], [], lambda _: _judge_ctx()
        )
