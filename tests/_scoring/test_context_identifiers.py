from collections.abc import Sequence
from typing import Any

import pytest
from _fakes.pr1_r5_ids import HostileString, NotString, hostile
from _fakes.tools import GOAL, FixedJudge, RecordingExecutor, cand

from swarm_arena import Goal, Ref, Score
from swarm_arena._scoring.runner import _judge_batch
from swarm_arena.board import Candidate
from swarm_arena.scoring import CheckContext, JudgeContext
from swarm_arena.testing import check_judge
from swarm_arena.tool import Command


@pytest.mark.parametrize("path", ["engine", "conformance"])
@pytest.mark.parametrize("arm_at_entry", [False, True])
async def test_judge_error_strings_never_dispatch_subclass_hooks(
    monkeypatch: pytest.MonkeyPatch, path: str, arm_at_entry: bool
) -> None:
    contexts: list[JudgeContext] = []

    async def judge(
        self: FixedJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        contexts.append(ctx)
        for candidate in candidates:
            key, reason = HostileString(candidate.id), HostileString("bad")
            key.armed = reason.armed = arm_at_entry
            ctx.judge_error(key, reason)
            ctx.judge_error(key, "later")
            key.armed = reason.armed = True
        return []

    monkeypatch.setattr(FixedJudge, "judge", judge)
    built = FixedJudge(FixedJudge.Config(value=1))
    candidate = cand()
    executor = RecordingExecutor()
    if path == "engine":
        ctx = JudgeContext(goal=GOAL, executor=executor, artifact_root="/unused")
        result = await _judge_batch(Ref.of(built), built, [candidate], ctx)
        assert result.scores == []
        assert [(error.candidate_id, error.reason) for error in result.errors] == [
            (candidate.id, "bad")
        ]
    else:
        with pytest.warns(UserWarning, match=f"^judge error for {candidate.id}: bad$") as warnings:
            assert await check_judge(built, GOAL, [candidate], executor=executor) == []
        assert len(warnings) == 1
    assert len(contexts) == 1
    assert contexts[0].errors == {candidate.id: "bad"}
    assert all(type(key) is str and type(value) is str for key, value in contexts[0].errors.items())


@pytest.mark.parametrize("field", ["candidate_id", "reason"])
def test_judge_error_rejects_non_string_without_inspection(field: str) -> None:
    ctx = JudgeContext(goal=GOAL, executor=RecordingExecutor(), artifact_root="/unused")
    values: dict[str, Any] = {"candidate_id": "c", "reason": "bad", field: NotString()}
    with pytest.raises(TypeError, match=f"^{field} must be str$"):
        ctx.judge_error(**values)
    assert ctx.errors == {}


@pytest.mark.parametrize("context_type", [CheckContext, JudgeContext])
@pytest.mark.parametrize("field", ["run_id", "part", "artifact_root", "secrets_file"])
def test_context_constructor_string_inputs(context_type: type[CheckContext], field: str) -> None:
    kwargs: dict[str, Any] = dict(goal=GOAL, executor=RecordingExecutor(), artifact_root="/unused")
    kwargs[field] = hostile("value")
    ctx = context_type(**kwargs)
    assert type(getattr(ctx, field)) is str
    assert getattr(ctx, field) == "value"
    kwargs[field] = NotString()
    with pytest.raises(TypeError, match=f"^{field} must be str$"):
        context_type(**kwargs)


@pytest.mark.parametrize("field", ["fidelity", "batch_id", "anchors"])
def test_judge_constructor_identifiers(field: str) -> None:
    value = HostileString("value")
    kwargs: dict[str, Any] = dict(goal=GOAL, executor=RecordingExecutor(), artifact_root="/unused")
    kwargs[field] = frozenset([value]) if field == "anchors" else value
    value.armed = True
    ctx = JudgeContext(**kwargs)
    result = next(iter(ctx.anchors)) if field == "anchors" else getattr(ctx, field)
    assert type(result) is str
    assert result == "value"
    kwargs[field] = [NotString()] if field == "anchors" else NotString()
    with pytest.raises(TypeError, match=f"^{field} must be str$"):
        JudgeContext(**kwargs)


async def test_execute_normalizes_file_names_and_hashes() -> None:
    key, value = HostileString("input.txt"), HostileString("a" * 64)
    files: dict[str, str] = {key: value}
    key.armed = value.armed = True
    executor = RecordingExecutor()
    ctx = CheckContext(goal=GOAL, executor=executor, artifact_root="/unused")
    await ctx.execute(Command(argv=["echo"]), files=files)
    assert executor.jobs[0].stage.root == {"input.txt": "a" * 64}
    assert all(type(k) is str and type(v) is str for k, v in executor.jobs[0].stage.root.items())


@pytest.mark.parametrize("field", ["files path", "files hash"])
async def test_execute_rejects_invalid_file_identifiers(field: str) -> None:
    executor = RecordingExecutor()
    ctx = CheckContext(goal=GOAL, executor=executor, artifact_root="/unused")
    files: Any = {NotString(): "a" * 64} if field == "files path" else {"input": NotString()}
    with pytest.raises(TypeError, match=f"^{field} must be str$"):
        await ctx.execute(Command(argv=["echo"]), files=files)
    assert executor.jobs == []
