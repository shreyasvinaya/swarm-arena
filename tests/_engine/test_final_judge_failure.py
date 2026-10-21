"""Retain the reason when final judging fails after candidates pass."""

from collections.abc import Sequence

from _fakes.run import all_events, events, plan

from swarm_arena import Goal, ProgramJudge, Ref, RunConfig, Swarm
from swarm_arena._journal.replay import RunState
from swarm_arena._model.results import Candidate, Score
from swarm_arena.scoring import JudgeContext

SECRET = "edc92e65-cfba-406b-9998-f1ddc48b3296"


class FinalFailureJudge(ProgramJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if ctx.final:
            raise ValueError(SECRET)
        return await super().judge(goal, candidates, ctx)


async def test_final_judge_failure_has_its_own_type_only_terminal_reason(
    run_config: RunConfig,
) -> None:
    current = plan(rounds=1)
    judge = Ref(path=f"{__name__}:FinalFailureJudge", name="y", config=current.judges[0].config)
    result = await Swarm(run_config).run(current.model_copy(update={"judges": [judge]}))
    assert result == {"g": []}
    assert events(run_config, "score")
    assert events(run_config, "judge_error")
    state = RunState.from_events(all_events(run_config))
    assert state.goals["g"].end_reason == "provider_failed"
    assert events(run_config, "no_result")[-1].payload["reason"] == "provider_failed"
    assert events(run_config, "goal_finished")[-1].payload["reason"] == "provider_failed"
    assert SECRET not in repr([e.model_dump() for e in all_events(run_config)])
