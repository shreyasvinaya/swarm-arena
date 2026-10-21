"""Score provenance identifies the durable barrier and final recovery batches."""

from _engine.test_goal import _plan
from _fakes.engine import EngineEnv

from swarm_arena._engine.goal import GoalRunner


async def test_score_batch_ids_match_phase_recovery_records(env: EngineEnv) -> None:
    await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1, rounds=2), replanner=None).run()
    ids: set[str] = set()
    for phase in ("barrier", "final"):
        events = [event for event in env.events("score") if f"{phase}_batch" in event.payload]
        assert len(events) == (2 if phase == "barrier" else 1)
        for event in events:
            expected = f"{phase}-{event.payload[f'{phase}_batch']['key']}"
            assert event.payload["score"]["batch_id"] == expected
            assert expected not in ids
            ids.add(expected)
