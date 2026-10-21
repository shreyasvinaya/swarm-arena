"""Public reads retain their captured readers through concurrent retirement."""

from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest
from _fakes.journals import FakeRun

from swarm_arena._board.arena import ArenaBoard, ArenaView


@pytest.fixture
def published(tmp_path: Path) -> Iterator[tuple[ArenaBoard, ArenaView, FakeRun]]:
    run = FakeRun(str(tmp_path / "runs"), "own")
    run.commit(
        run.plan(),
        run.candidate("old"),
        run.score("os", "old", 1),
        run.learning("learning", "viscosity", evidence=["old"]),
    )
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open(run.run_id, run.run_root, ["toy"])
    view = board.snapshot(run.goal.key(), board.frontier())
    yield board, view, run
    board.close()


def _observe_drain(
    board: ArenaBoard, monkeypatch: pytest.MonkeyPatch, checked: Event
) -> list[bool]:
    """Observe the first real reader-drain predicate before allowing the read to finish."""
    drained: list[bool] = []
    original = board._lifetime._condition.wait_for

    def wait_for(predicate: Callable[[], bool], timeout: float | None = None) -> bool:
        def observed() -> bool:
            result = predicate()
            if not checked.is_set():
                drained.append(result)
                checked.set()
            return result

        return original(observed, timeout)

    monkeypatch.setattr(board._lifetime._condition, "wait_for", wait_for)
    return drained


@pytest.mark.parametrize("kind", ["board", "view"])
@pytest.mark.parametrize("method", ["get_candidate", "scores"])
def test_read_captured_before_close_keeps_reader_alive(
    published: tuple[ArenaBoard, ArenaView, FakeRun],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    method: str,
) -> None:
    board, view, _ = published
    reader = board if kind == "board" else view
    cache = view._cache
    entered, draining, closed = Event(), Event(), Event()
    failures: list[BaseException] = []
    drained = _observe_drain(board, monkeypatch, draining)
    operation = "candidates" if method == "get_candidate" else "scores"
    original = getattr(cache, operation)

    def paused(*args: Any, **kwargs: Any) -> Any:
        # scores pauses after candidate validation, between two cache operations.
        entered.set()
        assert draining.wait(5)
        assert drained == [False]
        assert not closed.is_set()
        return original(*args, **kwargs)

    def shutdown() -> None:
        try:
            assert entered.wait(5)
            board.close()
        except BaseException as error:
            failures.append(error)
        finally:
            closed.set()

    monkeypatch.setattr(cache, operation, paused)
    worker = Thread(target=shutdown)
    worker.start()
    try:
        result = getattr(reader, method)("old")
        assert ([s.id for s in result] if method == "scores" else result.id) == (
            ["os"] if method == "scores" else "old"
        )
    finally:
        entered.set()
        worker.join(5)
        assert not worker.is_alive()
        assert not failures
    assert cache._closed


def test_swapped_generation_stays_pinned_until_compound_read_finishes(
    published: tuple[ArenaBoard, ArenaView, FakeRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board, view, run = published
    old = board._read_state()
    cache = view._cache
    entered, swapped, closed, draining = Event(), Event(), Event(), Event()
    drained = _observe_drain(board, monkeypatch, draining)
    failures: list[BaseException] = []
    original = cache.scores

    def paused(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert swapped.wait(5)
        assert draining.wait(5)
        assert drained == [False]
        assert not closed.is_set()
        return original(*args, **kwargs)

    def refresh_and_close() -> None:
        try:
            assert entered.wait(5)
            run.commit(run.score("new-score", "old", 2))
            board.frontier()
            assert board._read_state() is not old
            swapped.set()
            board.close()
        except BaseException as error:
            failures.append(error)
            swapped.set()
        finally:
            closed.set()

    monkeypatch.setattr(cache, "scores", paused)
    worker = Thread(target=refresh_and_close)
    worker.start()
    try:
        assert [s.id for s in board.scores("old")] == ["os"]
    finally:
        entered.set()
        worker.join(5)
        assert not worker.is_alive()
        assert not failures
    assert cache._closed


def reads(reader: ArenaBoard | ArenaView, run: FakeRun) -> dict[str, Callable[[], Any]]:
    result: dict[str, Callable[[], Any]] = {
        "best": lambda: reader.best(run.goal.key(), 1, judge="j"),
        "learnings": lambda: reader.learnings("toy"),
        "search_learnings": lambda: reader.search_learnings("toy", "viscosity"),
        "get_candidate": lambda: reader.get_candidate("old"),
        "get_learning": lambda: reader.get_learning("learning"),
        "scores": lambda: reader.scores("old"),
        "lineage": lambda: reader.lineage("old"),
        "leaderboard": lambda: reader.leaderboard(run.goal.key(), judge="j"),
    }
    if isinstance(reader, ArenaView):
        result["count_learnings"] = lambda: reader.count_learnings("toy")
    return result


@pytest.mark.parametrize("kind", ["board", "view"])
def test_reads_after_close_raise_board_closed_error(
    published: tuple[ArenaBoard, ArenaView, FakeRun],
    kind: str,
) -> None:
    board, view, run = published
    operations = reads(board if kind == "board" else view, run)
    board.close()
    for operation in operations.values():
        with pytest.raises(RuntimeError, match="arena board is closed"):
            operation()


@pytest.mark.parametrize("kind", ["board", "view"])
def test_failed_read_releases_its_pin(
    published: tuple[ArenaBoard, ArenaView, FakeRun],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    board, view, _ = published
    reader = board if kind == "board" else view

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("read failed")

    monkeypatch.setattr(view._cache, "scores", fail)
    with pytest.raises(ValueError, match="read failed"):
        reader.scores("old")
    worker = Thread(target=board.close, daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert view._cache._closed


def test_nested_view_read_continues_after_close_stops_new_reads(
    published: tuple[ArenaBoard, ArenaView, FakeRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board, view, run = published
    entered, stopped = Event(), Event()
    failures: list[BaseException] = []
    original_view = board._published_view
    original_stop = board._lifetime.stop

    def stop() -> None:
        original_stop()
        stopped.set()

    def paused(*args: Any, **kwargs: Any) -> ArenaView | None:
        result = original_view(*args, **kwargs)
        entered.set()
        assert stopped.wait(5)
        # The outer Board read already owns a pin; its nested view may finish.
        return result

    def shutdown() -> None:
        try:
            assert entered.wait(5)
            board.close()
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(board._lifetime, "stop", stop)
    monkeypatch.setattr(board, "_published_view", paused)
    worker = Thread(target=shutdown, daemon=True)
    worker.start()
    try:
        assert [c.id for c in board.best(run.goal.key(), 1, judge="j")] == ["old"]
    finally:
        entered.set()
        worker.join(5)
        assert not worker.is_alive()
        assert not failures
    assert view._cache._closed


def test_reopen_while_old_read_drains_uses_a_new_lifetime(
    published: tuple[ArenaBoard, ArenaView, FakeRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board, view, run = published
    entered, stopped = Event(), Event()
    failures: list[BaseException] = []
    original = view._cache.candidates
    original_stop = board._lifetime.stop

    def stop() -> None:
        original_stop()
        stopped.set()

    def paused(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert stopped.wait(5)
        # Acquiring both locks from the reader proves close's drain holds neither.
        assert board._refresh_lock.acquire(timeout=5)
        assert board._lock.acquire(timeout=5)
        board._lock.release()
        board._refresh_lock.release()
        board.open(run.run_id, run.run_root, ["toy"])
        board.frontier()
        assert board.get_candidate("old").id == "old"
        assert board._read_state().readers["toy"] is not view._cache
        return original(*args, **kwargs)

    def shutdown() -> None:
        try:
            assert entered.wait(5)
            board.close()
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(board._lifetime, "stop", stop)
    monkeypatch.setattr(view._cache, "candidates", paused)
    worker = Thread(target=shutdown, daemon=True)
    worker.start()
    try:
        assert view.get_candidate("old").id == "old"
    finally:
        entered.set()
        worker.join(5)
        assert not worker.is_alive()
        assert not failures
    assert view._cache._closed
    with pytest.raises(RuntimeError, match="arena board is closed"):
        view.get_candidate("old")
    assert board.get_candidate("old").id == "old"
