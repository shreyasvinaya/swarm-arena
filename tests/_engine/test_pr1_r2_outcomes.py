"""Public runs exercise preparation before scoring and nested callback boundaries."""

import traceback

import pytest
from _fakes.pr1_r2 import SECRET, PreparationJudge
from _fakes.run import all_events, events, plan

from swarm_arena import ProgramJudge, Ref, RunConfig, Swarm
from swarm_arena._errors import Fenced


@pytest.mark.parametrize("kind", ["fenced", "cancel", "group", "forged", "forged_group"])
@pytest.mark.parametrize("stage", ["constructor", "anchor_count", "models", "providers", "score"])
async def test_judge_boundary_outcomes_through_public_run(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, stage: str, kind: str
) -> None:
    monkeypatch.setattr(PreparationJudge, "armed", False)
    monkeypatch.setattr(PreparationJudge, "stage", stage)
    monkeypatch.setattr(PreparationJudge, "kind", kind)
    current = plan(rounds=2)
    bad = (
        Ref.of(ProgramJudge("_fakes.pr1_r2:score"))
        if stage == "score"
        else Ref(path="_fakes.pr1_r2:PreparationJudge")
    ).model_copy(update={"name": "bad"})
    current = current.model_copy(update={"judges": [bad, *current.judges]})
    if kind == "fenced":
        with pytest.raises(Fenced) as caught:
            await Swarm(run_config).run(current)
        assert SECRET not in "".join(traceback.format_exception(caught.value))
        assert {event.round for event in events(run_config, "tool_start")} == {1}
    else:
        result = await Swarm(run_config).run(current)
        assert result["g"]
        assert events(run_config, "goal_finished")
        assert {event.round for event in events(run_config, "tool_start")} == {1, 2}
        assert events(run_config, "judge_error")
    assert SECRET not in repr([event.model_dump() for event in all_events(run_config)])
