import contextvars
import inspect
import threading
from collections.abc import Callable
from functools import partial
from typing import Any

import anyio
import anyio.lowlevel
import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import GOAL, cand, rosen_score
from pydantic import ValidationError

from swarm_arena._model.goal import Goal
from swarm_arena._model.results import Candidate, Score
from swarm_arena._scoring import program_judge
from swarm_arena._scoring.judge import JudgeContext
from swarm_arena._scoring.program_judge import ProgramJudge

SECRET = "6dca447c8d41f358f1c5dcc3d2e651cb"
_REQUEST: contextvars.ContextVar[str] = contextvars.ContextVar("request", default="unset")


@pytest.mark.parametrize("nested", [False, True])
async def test_program_judge_exit_terminal(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    from swarm_arena._errors import _core_category, _SignalCategory, _stop_cleanup_error

    reached: list[bool] = []

    def fail(*args: Any) -> None:
        reached.append(True)
        error = _stop_cleanup_error()
        if nested:
            raise BaseExceptionGroup("judge", [BaseExceptionGroup("score", [error])])
        raise error

    monkeypatch.setattr(program_judge, "_score_copy", fail)
    with pytest.raises(RuntimeError) as caught:
        await ProgramJudge(rosen_score).judge(GOAL, [cand(id="c1")], judge_ctx)
    assert reached == [True]
    assert _core_category(caught.value) is _SignalCategory.TERMINAL
    assert not judge_ctx.errors


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
        ("_fakes.tools:raising_score", "RuntimeError: scoring function raised"),
        ("_fakes.tools:nan_score", "ValidationError: scoring function raised"),
        ("_fakes.tools:list_score", "returned list, expected Score"),
    ],
)
async def test_failures_are_judge_errors(judge_ctx: JudgeContext, fn: str, reason: str) -> None:
    assert await ProgramJudge(fn).judge(GOAL, [cand(id="c1", data={"f": 1.0})], judge_ctx) == []
    assert reason == judge_ctx.errors["c1"]


def test_forms_and_preflight() -> None:
    assert ProgramJudge.name == "program"
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

    monkeypatch.setattr("swarm_arena._model.ref.importlib.import_module", fail)
    judge = ProgramJudge("example:score")
    with pytest.raises(ValueError, match="cannot import example:score") as caught:
        judge.preflight()
    assert_secret_absent(caught.value, SECRET)


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
    assert_secret_absent(caught.value, SECRET)


class AsyncScorer:
    async def __call__(self, goal: Goal, candidate: Candidate) -> Score:
        return Score.maximize(1)

    async def method(self, goal: Goal, candidate: Candidate) -> Score:
        return Score.maximize(1)


@pytest.mark.parametrize(
    "fn",
    [AsyncScorer(), AsyncScorer().method, partial(AsyncScorer().method), partial(AsyncScorer())],
)
@pytest.mark.parametrize("direct", [False, True])
def test_async_callable_forms_are_rejected(
    monkeypatch: pytest.MonkeyPatch, fn: Any, direct: bool
) -> None:
    monkeypatch.setattr(program_judge, "load_object", lambda path: fn)
    with pytest.raises(TypeError, match="sync"):
        ProgramJudge(fn if direct else "example:score").preflight()


async def test_returned_coroutine_is_closed(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = AsyncScorer()(GOAL, cand())
    monkeypatch.setattr(program_judge, "load_object", lambda path: lambda g, c: result)
    try:
        assert await ProgramJudge("example:score").judge(GOAL, [cand(id="bad")], judge_ctx) == []
        assert inspect.getcoroutinestate(result) == inspect.CORO_CLOSED
        assert judge_ctx.errors == {"bad": "ProgramJudge needs a sync function"}
    finally:
        result.close()


async def test_returned_custom_awaitable_is_closed(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AwaitableScore:
        closed = False

        def __await__(self) -> Any:
            yield

        def close(self) -> None:
            self.closed = True

    result = AwaitableScore()
    monkeypatch.setattr(program_judge, "load_object", lambda path: lambda g, c: result)
    assert await ProgramJudge("example:score").judge(GOAL, [cand(id="bad")], judge_ctx) == []
    assert result.closed
    assert judge_ctx.errors == {"bad": "ProgramJudge needs a sync function"}


@pytest.mark.parametrize(
    "update", [{"notes": object()}, {"value": SECRET}, {"llm_call_ids": SECRET}]
)
async def test_invalid_non_key_fields_are_candidate_errors(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch, update: dict[str, Any]
) -> None:
    def score(goal: Goal, candidate: Candidate) -> Score:
        return Score.maximize(1).model_copy(update=update if candidate.id == "bad" else {})

    monkeypatch.setattr(program_judge, "load_object", lambda path: score)
    scores = await ProgramJudge("example:score").judge(
        GOAL, [cand(id="bad"), cand(id="good")], judge_ctx
    )
    assert [s.candidate_id for s in scores] == ["good"]
    assert judge_ctx.errors == {"bad": "invalid Score returned"}  # the function did not raise
    assert SECRET not in repr(judge_ctx.errors)


async def test_score_subclass_with_an_extra_field_is_an_invalid_score(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # Score forbids extra fields, so re-validation refuses it
    class TaggedScore(Score):
        tag: str = SECRET

    tagged = TaggedScore.maximize(1)
    monkeypatch.setattr(program_judge, "load_object", lambda path: lambda g, c: tagged)
    assert await ProgramJudge("example:score").judge(GOAL, [cand(id="c")], judge_ctx) == []
    assert judge_ctx.errors == {"c": "invalid Score returned"}
    assert SECRET not in repr(judge_ctx.errors)


async def test_candidate_copy_is_made_in_the_daemon_thread(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # a deep copy of a large output must not stall the event loop
    copied_on: list[threading.Thread] = []
    real_copy = Candidate.model_copy

    def model_copy(self: Candidate, *args: Any, **kwargs: Any) -> Candidate:
        copied_on.append(threading.current_thread())
        return real_copy(self, *args, **kwargs)

    monkeypatch.setattr(Candidate, "model_copy", model_copy)
    loop_thread = threading.current_thread()
    scores = await ProgramJudge(rosen_score).judge(
        GOAL, [cand(id="c1", data={"f": 1.0}), cand(id="c2", data={"f": 2.0})], judge_ctx
    )
    assert [s.candidate_id for s in scores] == ["c1", "c2"]
    assert len(copied_on) == 2
    assert all(t is not loop_thread and t.daemon for t in copied_on)


async def test_scoring_function_sees_the_callers_context_variables(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # later plans read driver state through ContextVars
    seen: list[str] = []

    def score(goal: Goal, candidate: Candidate) -> Score:
        seen.append(_REQUEST.get())
        return Score.maximize(1)

    monkeypatch.setattr(program_judge, "load_object", lambda path: score)
    token = _REQUEST.set("driver-value")
    try:
        await ProgramJudge("example:score").judge(GOAL, [cand(id="c")], judge_ctx)
    finally:
        _REQUEST.reset(token)
    assert seen == ["driver-value"]


async def test_scoring_can_be_cancelled_while_daemon_is_blocked(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    workers: list[threading.Thread] = []
    daemon: list[bool] = []
    completed: list[bool] = []

    def score(goal: Goal, candidate: Candidate) -> Score:
        workers.append(threading.current_thread())
        daemon.append(threading.current_thread().daemon)
        started.set()
        try:
            release.wait(180)
            return Score.maximize(1)
        finally:
            finished.set()

    async def run() -> None:
        await ProgramJudge("example:score").judge(GOAL, [cand()], judge_ctx)
        completed.append(True)

    monkeypatch.setattr(program_judge, "load_object", lambda path: score)
    try:
        with anyio.fail_after(60):
            async with anyio.create_task_group() as group:
                group.start_soon(run)
                while not started.is_set():
                    await anyio.lowlevel.checkpoint()
                group.cancel_scope.cancel()
        assert not release.is_set()
        assert not finished.is_set()
        assert daemon == [True]
        assert completed == []
        assert judge_ctx.errors == {}
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()


async def test_candidate_mutation_does_not_leak_to_later_members(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._scoring.lexicographic import Lexicographic

    original = cand(id="c", data={"nested": {"f": 2}})

    def mutate(goal: Goal, candidate: Candidate) -> Score:
        candidate.output.data["nested"]["f"] = 99
        return Score.maximize(1)

    def read(goal: Goal, candidate: Candidate) -> Score:
        return Score.maximize(candidate.output.data["nested"]["f"])

    monkeypatch.setattr(
        program_judge, "load_object", lambda path: mutate if path == "example:mutate" else read
    )
    lex = Lexicographic([ProgramJudge("example:mutate"), ProgramJudge("example:read")])
    [score] = await lex.judge(GOAL, [original], judge_ctx)
    assert score.key == (1, 2)
    assert original.output.data == {"nested": {"f": 2}}


async def test_returned_future_is_cancelled(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    result = asyncio.get_running_loop().create_future()
    monkeypatch.setattr(program_judge, "load_object", lambda path: lambda g, c: result)
    assert await ProgramJudge("example:score").judge(GOAL, [cand(id="bad")], judge_ctx) == []
    assert result.cancelled()
    assert judge_ctx.errors == {"bad": "ProgramJudge needs a sync function"}


async def test_returned_awaitable_iterator_is_closed(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = AsyncScorer()(GOAL, cand())

    class WrappedAwaitable:
        def __await__(self) -> Any:
            return result.__await__()

    monkeypatch.setattr(program_judge, "load_object", lambda path: lambda g, c: WrappedAwaitable())
    try:
        assert await ProgramJudge("example:score").judge(GOAL, [cand(id="bad")], judge_ctx) == []
        assert inspect.getcoroutinestate(result) == inspect.CORO_CLOSED
        assert judge_ctx.errors == {"bad": "ProgramJudge needs a sync function"}
    finally:
        result.close()


async def test_score_subclass_is_revalidated_as_plain_score(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CustomScore(Score):
        pass

    result = CustomScore.maximize(2)
    monkeypatch.setattr(program_judge, "load_object", lambda path: lambda g, c: result)
    [score] = await ProgramJudge("example:score").judge(GOAL, [cand(id="c")], judge_ctx)
    assert type(score) is Score
    assert score.candidate_id == "c"
    assert result.candidate_id == ""


def test_bound_instance_method_is_rejected_without_losing_binding() -> None:
    class Scorer:
        def score(self, goal: Goal, candidate: Candidate) -> Score:
            return Score.maximize(1)

    with pytest.raises(TypeError, match="bound instance method"):
        ProgramJudge(Scorer().score)


async def test_system_exit_is_local_to_candidate(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def score(goal: Goal, candidate: Candidate) -> Score:
        if candidate.id == "bad":
            raise SystemExit(SECRET)
        return Score.maximize(3)

    monkeypatch.setattr(program_judge, "load_object", lambda path: score)
    try:
        scores = await ProgramJudge("example:score").judge(
            GOAL, [cand(id="bad"), cand(id="good")], judge_ctx
        )
    except SystemExit:
        pytest.fail("user SystemExit escaped judging")
    assert [s.candidate_id for s in scores] == ["good"]
    assert judge_ctx.errors == {"bad": "SystemExit: scoring function raised"}
    assert SECRET not in repr(judge_ctx.errors)


async def test_keyboard_interrupt_from_scoring_still_propagates(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def score(goal: Goal, candidate: Candidate) -> Score:
        raise KeyboardInterrupt

    monkeypatch.setattr(program_judge, "load_object", lambda path: score)
    with pytest.raises(KeyboardInterrupt):
        await ProgramJudge("example:score").judge(GOAL, [cand()], judge_ctx)
