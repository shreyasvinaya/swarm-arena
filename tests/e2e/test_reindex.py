"""Recorded frontiers survive cache loss and pointer removal after human edits."""

import csv
import io
import json
import os
from functools import partial
from pathlib import Path

import anyio
from _fakes.finished import FinishedRun, run_again

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.human_objects import HumanObject, write_human_objects
from swarm_arena._board.reindex import recorded_frontiers, reindex, snapshot_membership
from swarm_arena._journal.event import Event
from swarm_arena._journal.lease import read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.results import Learning
from swarm_arena._scoring.human_review import CSV_HEADER, HumanReview
from swarm_arena._store.base import open_store


def _events(finished: FinishedRun, run_id: str) -> list[Event]:
    store = open_store(finished.config.run_root).child(run_id)
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head is not None
    return [event for _, events in read_chain(store, owner[0].head) for event in events]


async def test_reindex_rebuilds_identical_membership(
    finished_run: FinishedRun, tmp_path: Path
) -> None:
    config, ns = finished_run.config, finished_run.namespace
    goal_key = finished_run.plan.goal.key()
    board = ArenaBoard(arena_root=config.arena_root, cache_dir=str(tmp_path / "c1"))
    rebuilt = ArenaBoard(arena_root=config.arena_root, cache_dir=str(tmp_path / "c2"))
    fresh = ArenaBoard(arena_root=config.arena_root, cache_dir=str(tmp_path / "c3"))
    try:
        await anyio.to_thread.run_sync(board.open, "", "", [ns])
        await anyio.to_thread.run_sync(board.frontier)
        learnings = board.learnings(ns)
        assert len(learnings) >= 2
        pinned, hidden = learnings[:2]
        added = Learning(
            id="L-h-0123456789abcdef",
            namespace=ns,
            run_id="",
            goal_id=goal_key.id,
            round=0,
            author="human:alice",
            title="Human insight",
            text="Try carefully.",
        )
        arena = open_store(config.arena_root)
        names = []
        for obj in (
            HumanObject(kind="pin", learning_id=pinned.id, author="human:alice"),
            HumanObject(kind="hide", learning_id=hidden.id, author="human:alice"),
            HumanObject(kind="add", learning=added, author="human:alice"),
        ):
            names.append(
                await anyio.to_thread.run_sync(
                    partial(write_human_objects, arena, ns, "curation", [obj], now=utc_now())
                )
            )
        review = HumanReview("alice")
        out = await review.export(finished_run.run_id, config=config, out_dir=tmp_path / "review")
        csv_path = out / "review.csv"
        contents = await anyio.to_thread.run_sync(csv_path.read_text)
        rows = list(csv.DictReader(io.StringIO(contents)))[:1]
        assert rows
        rows[0].update(verdict="accept", score="7")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
        await anyio.to_thread.run_sync(csv_path.write_text, stream.getvalue())
        scores = await review.import_csv(finished_run.run_id, csv_path, config=config)
        run2 = await run_again(finished_run)
        events1 = await anyio.to_thread.run_sync(_events, finished_run, finished_run.run_id)
        events2 = await anyio.to_thread.run_sync(_events, finished_run, run2)
        frontiers1, frontiers2 = recorded_frontiers(events1), recorded_frontiers(events2)
        assert len(frontiers1) == len(frontiers2) == 2
        frontiers = frontiers1 + frontiers2
        assert all(not f.curation and not f.reviews for _, _, f in frontiers1)
        assert all(set(names) <= set(f.curation) and f.reviews for _, _, f in frontiers2)
        before = [
            await anyio.to_thread.run_sync(snapshot_membership, board, goal_key, f)
            for _, _, f in frontiers
        ]
        members = json.loads(before[-1])
        flags = {id_: (pin, hide) for id_, pin, hide in members["learnings"]}
        assert flags[pinned.id] == (True, False)
        assert flags[hidden.id] == (False, True)
        assert flags[added.id] == (False, False)
        assert members["scores"].count(scores[0].id) == 1
        assert scores[0].id not in json.loads(before[0])["scores"]
        await anyio.to_thread.run_sync(partial(reindex, config, [ns], cache_dir=tmp_path / "c2"))
        assert [
            await anyio.to_thread.run_sync(snapshot_membership, rebuilt, goal_key, f)
            for _, _, f in frontiers
        ] == before
        await anyio.to_thread.run_sync(
            os.remove, Path(config.arena_root) / ns / "runs" / f"{finished_run.run_id}.json"
        )
        assert [
            await anyio.to_thread.run_sync(snapshot_membership, fresh, goal_key, f)
            for _, _, f in frontiers
        ] == before
        await anyio.to_thread.run_sync(fresh.open, "", "", [ns])
        latest = await anyio.to_thread.run_sync(fresh.frontier)
        assert finished_run.run_id not in [m.run_id for m in latest.marks]
        assert run2 in [m.run_id for m in latest.marks]
    finally:
        for opened in (board, rebuilt, fresh):
            await anyio.to_thread.run_sync(opened.close)
