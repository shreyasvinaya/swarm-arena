"""Fixed views remain usable while another goal refreshes shared journal state."""

from pathlib import Path
from threading import Event as ThreadEvent
from threading import Timer
from time import perf_counter
from typing import Any

import anyio
import pytest
from _fakes.journals import FakeRun

from swarm_arena import Goal
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.cache import ArenaCache
from swarm_arena._board.frontier import build_frontier
from swarm_arena._journal.event import Event
from swarm_arena._journal.replay import RunState
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("operation", ["frontier", "snapshot"])
@pytest.mark.parametrize("phase", ["journal", "ingest"])
async def test_fixed_view_reads_do_not_wait_for_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, phase: str
) -> None:
    fast = FakeRun(str(tmp_path / "runs"), "fast", goal=Goal(id="fast", namespace="toy"))
    slow = FakeRun(fast.run_root, "slow", goal=Goal(id="slow", namespace="toy"))
    fast.commit(
        fast.plan(),
        fast.candidate("fast-c"),
        fast.score("fast-s", "fast-c", 1),
        fast.learning("fast-learning", "viscosity", evidence=["fast-c"]),
    )
    slow.commit(slow.plan(), slow.candidate("slow-c"))
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open(fast.run_id, fast.run_root, ["toy"])
    from _fakes.journals import add_pointer

    add_pointer(board.config.arena_root, slow)
    view = board.snapshot(fast.goal.key(), board.frontier())
    head = slow.commit(slow.candidate("later"))
    next_frontier = build_frontier(
        [(run.run_id, run.run_root, run.heads[-1]) for run in (fast, slow)], [], []
    )
    entered, release = ThreadEvent(), ThreadEvent()
    completed = anyio.Event()
    ticks: list[float] = []
    original_read = LocalStore.read
    original_index = ArenaCache._index_event

    def pause() -> None:
        if not entered.is_set():
            entered.set()
            assert release.wait(5), "test failed to release storage"

    def read(self: LocalStore, key: str) -> Any:
        if phase == "journal" and self.url.endswith("/slow") and key == head.segment:
            pause()
        return original_read(self, key)

    def index(self: ArenaCache, event: Event, state: RunState, pending: list[str]) -> None:
        if phase == "ingest" and event.payload.get("candidate", {}).get("id") == "later":
            pause()
        original_index(self, event, state, pending)

    monkeypatch.setattr(LocalStore, "read", read)
    monkeypatch.setattr(ArenaCache, "_index_event", index)

    async def refresh() -> None:
        if operation == "frontier":
            await anyio.to_thread.run_sync(board.frontier)
        else:
            await anyio.to_thread.run_sync(board.snapshot, slow.goal.key(), next_frontier)
        completed.set()

    async def heartbeat() -> None:
        while True:
            ticks.append(perf_counter())
            await anyio.sleep(0.01)

    timer = Timer(1, release.set)
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(heartbeat)
            group.start_soon(refresh)
            assert await anyio.to_thread.run_sync(entered.wait, 5)
            timer.start()  # Escape even if the regression blocks the loop itself.
            try:
                start = perf_counter()
                assert view.best(view.goal, 1, judge="j")[0].id == "fast-c"
                assert view.scores("fast-c")[0].value == 1
                assert view.lineage("fast-c").root == "fast-c"
                assert view.leaderboard(view.goal, judge="j")[0].candidate_id == "fast-c"
                assert view.get_candidate("fast-c").id == "fast-c"
                assert view.get_learning("fast-learning").id == "fast-learning"
                assert view.learnings("toy")[0].id == "fast-learning"
                assert view.search_learnings("toy", "viscosity")[0].id == "fast-learning"
                assert view.count_learnings("toy") == 1
                assert perf_counter() - start < 0.2
                before = len(ticks)
                await anyio.sleep(0.06)
                assert len(ticks) >= before + 3
                assert not completed.is_set()
            finally:
                release.set()
                await completed.wait()
                group.cancel_scope.cancel()
    finally:
        release.set()
        timer.cancel()
        board.close()
