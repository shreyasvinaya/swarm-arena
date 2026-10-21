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
from swarm_arena._model.results import Candidate, Learning, Score
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

    beat_target: list[int] = []
    beats_reached = anyio.Event()

    async def heartbeat() -> None:
        while True:
            ticks.append(perf_counter())
            if beat_target and len(ticks) >= beat_target[0]:
                beats_reached.set()
            await anyio.sleep(0.01)

    timer = Timer(5, release.set)
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
                assert perf_counter() - start < 1.0
                before = len(ticks)
                # The loop keeps running while the refresh is held: wait for heartbeats, bounded.
                beat_target.append(before + 3)
                with anyio.fail_after(2):
                    await beats_reached.wait()
                assert not completed.is_set()
            finally:
                release.set()
                await completed.wait()
                group.cancel_scope.cancel()
    finally:
        release.set()
        timer.cancel()
        board.close()


@pytest.mark.parametrize("phase", ["pointers", "journal", "ingest"])
@pytest.mark.parametrize(
    "method",
    [
        "best",
        "learnings",
        "search_learnings",
        "get_candidate",
        "get_learning",
        "scores",
        "lineage",
        "leaderboard",
    ],
)
async def test_public_board_reads_use_published_state_during_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, method: str
) -> None:
    from swarm_arena._board import arena

    run = FakeRun(str(tmp_path / "runs"), "own")
    run.commit(
        run.plan(),
        run.candidate("old"),
        run.score("score", "old", 1),
        run.learning("learning", "viscosity", evidence=["old"]),
    )
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open(run.run_id, run.run_root, ["toy"])
    board.frontier()
    key = run.goal.key()
    board.put_candidate(
        Candidate.model_validate(run.candidate("hint-c").payload["candidate"]), epoch=1, seq=100
    )
    board.put_learning(
        Learning.model_validate(run.learning("hint-l", "hint").payload["learning"]),
        epoch=1,
        seq=101,
    )
    board.put_score(
        Score.model_validate(run.score("hint-s", "hint-c", 2).payload["score"]), epoch=1, seq=102
    )

    def read() -> Any:
        if method == "best":
            return board.best(key, 5, judge="j")
        if method == "learnings":
            return board.learnings("toy"), board.learnings("toy", goal=key)
        if method == "search_learnings":
            return (
                board.search_learnings("toy", "viscosity"),
                board.search_learnings("toy", "viscosity", goal=key),
            )
        if method == "get_candidate":
            return board.get_candidate("old"), board.get_candidate("hint-c")
        if method == "get_learning":
            return board.get_learning("learning"), board.get_learning("hint-l")
        if method == "scores":
            return board.scores("old"), board.scores("hint-c")
        if method == "lineage":
            return board.lineage("old")
        return board.leaderboard(key, judge="j")

    expected = read()
    head = run.commit(run.candidate("new"), run.score("new-score", "new", 0))
    entered, release = ThreadEvent(), ThreadEvent()
    finished = anyio.Event()
    ticks: list[float] = []
    original_pointers, original_read = arena.list_pointers, LocalStore.read
    original_index = ArenaCache._index_event

    def pause() -> None:
        entered.set()
        assert release.wait(5), "test failed to release refresh"

    def pointers(*args: Any, **kwargs: Any) -> Any:
        if phase == "pointers":
            pause()
        return original_pointers(*args, **kwargs)

    def storage_read(self: LocalStore, name: str) -> Any:
        if phase == "journal" and self.url.endswith("/own") and name == head.segment:
            pause()
        return original_read(self, name)

    def index(self: ArenaCache, event: Event, state: RunState, pending: list[str]) -> None:
        if phase == "ingest" and event.payload.get("candidate", {}).get("id") == "new":
            pause()
        original_index(self, event, state, pending)

    monkeypatch.setattr(arena, "list_pointers", pointers)
    monkeypatch.setattr(LocalStore, "read", storage_read)
    monkeypatch.setattr(ArenaCache, "_index_event", index)

    async def refresh() -> None:
        try:
            await anyio.to_thread.run_sync(board.frontier)
        finally:
            finished.set()

    beat_target: list[int] = []
    beats_reached = anyio.Event()

    async def heartbeat() -> None:
        while True:
            ticks.append(perf_counter())
            if beat_target and len(ticks) >= beat_target[0]:
                beats_reached.set()
            await anyio.sleep(0.01)

    timer = Timer(5, release.set)
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(refresh)
            group.start_soon(heartbeat)
            assert await anyio.to_thread.run_sync(entered.wait, 5)
            timer.start()
            try:
                started = perf_counter()
                result = read()
                assert perf_counter() - started < 1.0
                assert result == expected
                before = len(ticks)
                # The loop keeps running while the refresh is held: wait for heartbeats, bounded.
                beat_target.append(before + 3)
                with anyio.fail_after(2):
                    await beats_reached.wait()
                assert not finished.is_set()
            finally:
                release.set()
                await finished.wait()
                group.cancel_scope.cancel()
        assert board.best(key, 1, judge="j")[0].id == "new"
    finally:
        release.set()
        timer.cancel()
        board.close()


@pytest.mark.parametrize("published", [False, True])
def test_unpublished_and_unknown_namespace_reads_never_initialize_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published: bool,
) -> None:
    from swarm_arena._board import arena

    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    if published:
        board.frontier()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("read attempted storage or cache initialization")

    monkeypatch.setattr(arena, "open_store", forbidden)
    monkeypatch.setattr(board, "_cache", forbidden)
    monkeypatch.setattr(board, "_read_cache", forbidden)
    monkeypatch.setattr(board, "frontier", forbidden)
    try:
        for namespace in ("toy", "unattached"):
            key = Goal(id="g", namespace=namespace).key()
            assert board.best(key, 1, judge="j") == []
            assert board.leaderboard(key, judge="j") == []
            assert board.learnings(namespace) == []
            assert board.learnings(namespace, goal=key) == []
            assert board.search_learnings(namespace, "text") == []
            assert board.search_learnings(namespace, "text", goal=key) == []
        for read in (board.get_candidate, board.get_learning, board.scores, board.lineage):
            with pytest.raises(KeyError):
                read("missing")
    finally:
        board.close()


def test_failed_reader_preparation_retains_published_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = FakeRun(str(tmp_path / "runs"), "own")
    run.commit(run.plan(), run.candidate("old"), run.score("os", "old", 1))
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open(run.run_id, run.run_root, ["toy"])
    old_frontier = board.frontier()
    view = board.snapshot(run.goal.key(), old_frontier)
    run.commit(run.candidate("new"), run.score("ns", "new", 0))

    def fail(namespace: str) -> ArenaCache:
        raise RuntimeError("reader preparation failed")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(board, "_read_cache", fail)
            with pytest.raises(RuntimeError, match="reader preparation failed"):
                board.frontier()
        assert board._read_state().frontier == old_frontier
        assert [c.id for c in board.best(run.goal.key(), 5, judge="j")] == ["old"]
        with pytest.raises(KeyError):
            board.get_candidate("new")
        board.frontier()
        assert [c.id for c in board.best(run.goal.key(), 5, judge="j")] == ["new", "old"]
        assert [c.id for c in view.best(view.goal, 5, judge="j")] == ["old"]
    finally:
        board.close()
