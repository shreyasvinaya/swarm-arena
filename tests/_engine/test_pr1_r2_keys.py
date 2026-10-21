"""Retain historical GoalKeys after revision replacement and resume."""

from pathlib import Path

import pytest

from swarm_arena import Goal, PartSpec, Plan, RunConfig
from swarm_arena._engine.admission import RunAdmission
from swarm_arena._engine.goal import r0_revision_draft
from swarm_arena._engine.resume import resume_run
from swarm_arena._errors import PlanError
from swarm_arena._journal.replay import RunState
from tests._engine.test_resume import _append, _crash, _events, _plan, _Recorder
from tests._engine.test_resume import recorders as recorders


@pytest.mark.parametrize("seed", [False, True])
@pytest.mark.parametrize("change", ["removed", "changed"])
async def test_historical_keys_remain_reserved(seed: bool, change: str) -> None:
    old = Plan(
        goal=Goal(id="owner", namespace="n"),
        parts=[PartSpec(name="part", goal=Goal(id="old-part", namespace="n"))],
        teams=[],
        judges=[],
    )
    new = old.model_copy(
        update={
            "revision": 1,
            "parts": []
            if change == "removed"
            else [PartSpec(name="part", goal=Goal(id="new-part", namespace="n"))],
        }
    )
    admission = RunAdmission(10, [old, new] if seed else [old])
    if not seed:
        await admission.update("owner", 1, plan=new)
    await admission.release("owner")
    proposed = Plan(goal=Goal(id="other", namespace="n"), parts=old.parts, teams=[], judges=[])
    published = False

    async def publish() -> None:
        nonlocal published
        published = True

    with pytest.raises(PlanError, match="GoalKey"):
        await admission.update("other", 1, plan=proposed, publish=publish)
    assert not published
    await admission.update("owner", 1, plan=old)


async def test_resume_reserves_all_historical_keys(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    initial = _plan(converge=True)
    initial = initial.model_copy(
        update={
            "parts": [
                PartSpec(name="old", goal=Goal(id="old-part", namespace=initial.goal.namespace))
            ],
            "teams": [initial.teams[0].model_copy(update={"part": "old"})],
        }
    )
    crash = await _crash(tmp_path, run_config, recorders, plans=[initial], lost=())
    original = RunState.from_events(_events(crash)).plans["g"][0][0]
    revised = original.model_copy(
        update={
            "revision": 1,
            "parts": [
                original.parts[0].model_copy(
                    update={
                        "goal": original.parts[0].goal.model_copy(
                            update={"inputs": {"version": "new"}}
                        )
                    }
                )
            ],
        }
    )
    from swarm_arena._planning.validate import admit_revision
    from swarm_arena._store.artifacts import ArtifactStore
    from swarm_arena._store.local import LocalStore

    revised, _ = admit_revision(
        revised,
        crash.config,
        ArtifactStore(LocalStore(Path(crash.config.artifact_root))),
        history=[original],
    )
    await _append(crash, r0_revision_draft(revised, tmp_path))
    runner = await resume_run(crash.config, crash.run_id, force=True, replanner=None)
    try:
        assert runner.resumed is not None
        deps = runner.resumed._services.deps
        assert deps is not None
        assert deps.admission is not None
        historical = next(
            e.payload["candidate"]["id"] for e in _events(crash) if e.type == "tool_end"
        )
        view = deps.board.snapshot(original.parts[0].goal.key(), deps.board.frontier())
        assert view.get_candidate(historical).goal_id == "g"
        proposed = Plan(
            goal=Goal(id="other", namespace="n"),
            parts=original.parts,
            teams=[],
            judges=[],
        )
        with pytest.raises(PlanError, match="GoalKey"):
            await deps.admission.update("other", 1, plan=proposed)
        assert (
            deps.board.snapshot(original.parts[0].goal.key(), deps.board.frontier())
            .get_candidate(historical)
            .goal_id
            == "g"
        )
    finally:
        await runner.aclose()
