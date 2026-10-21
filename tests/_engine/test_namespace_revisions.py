"""An admitted namespace is discoverable before takeover or restart."""

from pathlib import Path

import pytest
from _engine.test_part_identity import ReplacePart, part_driver
from _fakes.run import all_events, plan

from swarm_arena import Goal, Plan, RunConfig, Swarm
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena.planning import History


async def test_replanned_namespace_is_live_and_matches_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def change_namespace(
        self: ReplacePart, goal: Goal, config: RunConfig, history: History | None
    ) -> Plan:
        assert history is not None
        current = history.revisions[-1]
        part = current.parts[0]
        return current.model_copy(
            update={
                "parts": [
                    part.model_copy(
                        update={"goal": part.goal.model_copy(update={"namespace": "newparts"})}
                    )
                ]
            }
        )

    monkeypatch.setattr(ReplacePart, "plan", change_namespace)
    original_publish = JournalWriter._publish
    await part_driver(tmp_path, "inputs", False)
    assert JournalWriter._publish is original_publish
    config = RunConfig.load(tmp_path / "runs" / "part" / "run.toml")
    state = RunState.from_events(all_events(config))
    current = state.plans["g"][-1][0]
    assert current.parts[0].goal.namespace == "newparts"
    expected = {c.id for c in state.candidates.values() if c.part == "a" and c.round == 2}
    assert expected
    pointer = tmp_path / "arena" / "newparts" / "runs" / "part.json"
    assert pointer.exists(), "live replan did not publish its namespace pointer"
    peer = ArenaBoard(arena_root=config.arena_root, cache_dir=str(tmp_path / "peer-cache"))
    peer.open("", "", ["newparts"])
    try:
        frontier = peer.frontier()
        assert {mark.run_id for mark in frontier.marks} == {"part"}
        live = {c.id for c in peer.best(current.parts[0].goal.key(), 20, judge="y")}
        assert expected <= live
        await Swarm(config).resume("part", force=True)
        peer.frontier()
        assert {c.id for c in peer.best(current.parts[0].goal.key(), 20, judge="y")} == live
    finally:
        peer.close()
    assert (tmp_path / "arena" / "n" / "runs" / "part.json").exists()


async def test_attaching_namespace_preserves_published_board_reads(
    tmp_path: Path, run_config: RunConfig
) -> None:
    current = plan(rounds=1)
    await Swarm(run_config).run(current, run_id="attach")
    board = ArenaBoard(arena_root=run_config.arena_root, cache_dir=str(tmp_path / "reader"))
    board.open("attach", run_config.run_root, ["n"])
    try:
        frontier = board.frontier()
        before = board.best(current.goal.key(), 20, judge="y")
        assert before
        view = board.snapshot(current.goal.key(), frontier)
        board.open("attach", run_config.run_root, ["n", "newparts"])
        assert board.best(current.goal.key(), 20, judge="y") == before
        assert view.get_candidate(before[0].id) == before[0]
    finally:
        board.close()
