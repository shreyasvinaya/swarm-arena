"""Final replay cannot borrow completion from a different batch or lifecycle."""

from collections.abc import Sequence

import pytest
from _fakes.engine import EngineEnv

from swarm_arena import Ref
from swarm_arena._engine import goal as goal_module
from swarm_arena._engine.goal import GoalRunner
from swarm_arena._journal.event import EventDraft
from swarm_arena._model.results import Candidate
from swarm_arena._scoring.runner import JudgeError, JudgeOutcome
from swarm_arena.scoring import Judge, JudgeContext
from tests._engine.test_goal import _plan


@pytest.mark.parametrize(
    "change", ["same", "revision", "members", "lifecycle", "judge", "candidates"]
)
@pytest.mark.parametrize("failed", [False, True])
async def test_final_completion_has_its_own_identity(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, change: str, failed: bool
) -> None:
    original = goal_module.final_judge
    batches: list[list[str]] = []

    async def final(
        goal_id: str,
        ranking: tuple[Ref, Judge],
        finalists: Sequence[Candidate],
        ctx: JudgeContext,
    ) -> tuple[list[Candidate], JudgeOutcome]:
        batches.append([c.id for c in finalists])
        if failed:
            return [], JudgeOutcome(
                scores=[],
                errors=[
                    JudgeError(candidate_id=c.id, judge="y", reason="judge error: ValueError")
                    for c in finalists
                ],
            )
        return await original(goal_id, ranking, finalists, ctx)

    monkeypatch.setattr(goal_module, "final_judge", final)
    runner = GoalRunner(
        env.deps, goal_id="g", plan=_plan(teams=1, agents_per_team=2, rounds=1), replanner=None
    )
    result = await runner.run()
    assert len(batches) == 1
    pool = [runner.recorded.candidates[id] for id in batches[0]]
    expected = [] if failed else [c.id for c in result.finalists]
    assert [c.id for c in runner._recover_final() or []] == expected
    if change == "candidates":
        extra = pool[0].model_copy(update={"id": "new-input"})
        runner.recorded.candidates[extra.id] = extra
    elif change == "revision":
        runner.plan = runner.plan.model_copy(update={"revision": runner.plan.revision + 1})
    elif change == "members":
        pool = pool[:1]
    elif change == "lifecycle":
        await env.deps.journal.append_durable(EventDraft(type="goal_started", goal_id="g"))
        await runner._refresh()
    elif change == "judge":
        runner.plan = runner.plan.model_copy(
            update={
                "judges": [runner.plan.judges[0].model_copy(update={"name": "other"})],
                "ranking": "other",
            }
        )
    if change not in {"same", "members"}:
        assert runner._recover_final() is None
    runner.status, runner.reason = "finished", None
    selected = await runner._final(pool)
    assert len(batches) == (1 if change == "same" else 2)
    if change == "same":
        assert [c.id for c in selected] == expected
        assert runner.reason == ("provider_failed" if failed else None)
