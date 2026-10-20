import sys
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest
from _fakes.tools import GOAL, AlwaysPass, RecordingExecutor, cand

from swarm_arena._errors import Fenced
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local import LocalExecutor
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
    assert for_manager(original, {}).checks == original.checks


def test_judge_outcome_and_error_are_frozen_models() -> None:
    error = JudgeError(candidate_id="c-1", judge="quality", reason="no score")
    outcome = JudgeOutcome(scores=[], errors=[error])
    assert outcome.model_dump() == {
        "scores": [],
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
        raise Fenced("another owner")

    with pytest.raises(Fenced, match="another owner"):
        await run_checks(
            _cand(),
            [("s", _script_check())],
            _check_ctx(executor=local_executor, artifact_root=artifact_root, on_started=fenced),
        )
    sj = _script_judge()
    with pytest.raises(Fenced, match="another owner"):
        await judge_round(
            "g",
            [(Ref.of(sj), sj)],
            [_cand("c-1")],
            [],
            ctx_for=lambda j: _judge_ctx(
                executor=local_executor, artifact_root=artifact_root, on_started=fenced
            ),
        )
    with pytest.raises(Fenced, match="another owner"):
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
