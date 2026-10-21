"""Rebuild caches safely and preserve recorded snapshot identities."""

import fcntl
import importlib
import os
from pathlib import Path
from typing import Any

import pytest
from _fakes.journals import START, FakeRun, add_pointer
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Plan, RunConfig
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.cache import ArenaCache
from swarm_arena._board.reindex import (
    _reindex_with_diagnostics,
    recorded_frontiers,
    reindex,
    snapshot_membership,
)
from swarm_arena._errors import ChainError
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._model.board_types import Frontier


def _seed(config: RunConfig, namespace: str = "toy", run_id: str = "r1") -> FakeRun:
    run = FakeRun(config.run_root, run_id, namespace=namespace)
    run.commit(run.plan(), run.candidate("shared"), run.score(f"s-{run_id}", "shared", 1))
    add_pointer(config.arena_root, run)
    run._lease.release()
    return run


def _unlocked(path: Path) -> None:
    fd = os.open(str(path) + ".lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_reindex_without_namespaces_uses_cache_files(run_config: RunConfig, tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    for ns in ("toy", "phys"):
        _seed(run_config, ns, f"run-{ns}")
        (directory / f"{ns}.sqlite").write_bytes(b"stale database")
    (directory / "ignored.sqlite.lock").touch()
    frontier = reindex(run_config, cache_dir=directory)
    assert {m.run_id for m in frontier.marks} == {"run-toy", "run-phys"}
    for ns in ("toy", "phys"):
        cache = ArenaCache(directory / f"{ns}.sqlite")
        try:
            assert [c.id for c, _ in cache.candidates(frontier)] == ["shared"]
        finally:
            cache.close()
    assert not (directory / "ignored.sqlite").exists()


def test_reindex_removes_wal_and_shm(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    files = [directory / ("toy.sqlite" + suffix) for suffix in ("", "-wal", "-shm")]
    for path in files:
        path.write_bytes(b"stale")
    lock = directory / "toy.sqlite.lock"
    lock.touch()
    inode = lock.stat().st_ino
    original = ArenaBoard.open

    def checked_open(self: ArenaBoard, *args: Any, **kwargs: Any) -> None:
        assert all(not path.exists() for path in files)
        _unlocked(files[0])
        original(self, *args, **kwargs)

    monkeypatch.setattr(ArenaBoard, "open", checked_open)
    reindex(run_config, ["toy", "toy"], cache_dir=directory)
    assert files[0].read_bytes().startswith(b"SQLite format 3")
    assert lock.stat().st_ino == inode
    _unlocked(files[0])


def test_reindex_refuses_while_cache_open(run_config: RunConfig, tmp_path: Path) -> None:
    path = tmp_path / "cache" / "toy.sqlite"
    held = ArenaCache(path)
    wal = Path(str(path) + "-wal")
    before = (path.read_bytes(), wal.read_bytes())
    try:
        with pytest.raises(RuntimeError, match="in use") as exc:
            reindex(run_config, ["toy"], cache_dir=path.parent)
        assert str(exc.value) == f"{path} is in use by another swarm-arena process; stop it first"
        assert (path.read_bytes(), wal.read_bytes()) == before
    finally:
        held.close()
    reindex(run_config, ["toy"], cache_dir=path.parent)
    reindex(run_config, ["toy"], cache_dir=path.parent)
    _unlocked(path)


def test_reindex_acquires_all_locks_before_deleting(run_config: RunConfig, tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    first = directory / "a.sqlite"
    first.write_bytes(b"keep first")
    held = ArenaCache(directory / "z.sqlite")
    before = {p: p.read_bytes() for p in directory.iterdir()}
    try:
        with pytest.raises(RuntimeError, match="in use"):
            reindex(run_config, ["a", "z"], cache_dir=directory)
        assert all(p.read_bytes() == data for p, data in before.items())
        _unlocked(first)
    finally:
        held.close()


@pytest.mark.parametrize("namespace", ["../SECRET_REINDEX", "/SECRET_REINDEX", "SECRET_REINDEX"])
def test_reindex_validates_all_names_before_io(
    run_config: RunConfig, tmp_path: Path, namespace: str
) -> None:
    directory = tmp_path / "not-created"
    with pytest.raises(ValueError, match="namespace") as exc:
        reindex(run_config, ["toy", namespace], cache_dir=directory)
    assert_secret_absent(exc.value, "SECRET_REINDEX")
    assert not directory.exists()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", ".lock"])
def test_reindex_rejects_symlink_paths_before_deletion(
    run_config: RunConfig, tmp_path: Path, suffix: str
) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    first = directory / "a.sqlite"
    first.write_bytes(b"keep")
    outside = tmp_path / "outside"
    outside.write_bytes(b"private")
    (directory / ("z.sqlite" + suffix)).symlink_to(outside)
    with pytest.raises(ValueError, match="cache"):
        reindex(run_config, ["a", "z"], cache_dir=directory)
    assert first.read_bytes() == b"keep"
    assert outside.read_bytes() == b"private"
    assert not (directory / "a.sqlite.lock").exists()


def test_reindex_default_directory_matches_board(run_config: RunConfig) -> None:
    board = ArenaBoard(arena_root=run_config.arena_root)
    directory = board._cache_dir()
    try:
        directory.mkdir(parents=True)
        (directory / "toy.sqlite").write_bytes(b"old")
        assert reindex(run_config) == Frontier()
        _unlocked(directory / "toy.sqlite")
    finally:
        board.close()


def test_reindex_empty_cache(run_config: RunConfig, tmp_path: Path) -> None:
    directory = tmp_path / "absent"
    assert reindex(run_config, cache_dir=directory) == Frontier()
    assert not directory.exists()


def test_reindex_diagnostics_rebuild_once_and_close(
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed(run_config, run_id="r1")
    _seed(run_config, run_id="r2")
    calls = 0
    original = ArenaBoard.frontier

    def counted(self: ArenaBoard) -> Frontier:
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(ArenaBoard, "frontier", counted)
    directory = tmp_path / "cache"
    result = _reindex_with_diagnostics(run_config, ["toy"], cache_dir=directory)
    assert isinstance(result.frontier, Frontier)
    assert result.run_counts == {"toy": 2}
    assert result.warnings
    assert any("shared" in warning for warning in result.warnings)
    assert calls == 1
    assert capsys.readouterr().out == ""
    _unlocked(directory / "toy.sqlite")


@pytest.mark.parametrize("stage", ["open", "frontier", "unlink"])
def test_reindex_failure_releases_resources(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    path = directory / "toy.sqlite"
    path.write_bytes(b"old")

    def fail_board(self: ArenaBoard, *args: Any, **kwargs: Any) -> None:
        self._cache("toy")
        raise RuntimeError("injected failure")

    original_unlink = Path.unlink

    def fail_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == path:
            raise RuntimeError("injected failure")
        original_unlink(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        if stage == "unlink":
            patch.setattr(Path, "unlink", fail_unlink)
        else:
            patch.setattr(ArenaBoard, stage, fail_board)
        with pytest.raises(RuntimeError, match="injected failure"):
            reindex(run_config, ["toy"], cache_dir=directory)
    _unlocked(path)
    reindex(run_config, ["toy"], cache_dir=directory)


def test_recorded_frontiers_in_order(tmp_path: Path) -> None:
    first = Frontier(reviews=["toy/reviews/a/20261018T000000.000Z-00000001.jsonl"])
    second = Frontier(curation=["toy/curation/20261018T000000.000Z-00000002.jsonl"])
    events = [
        Event(
            epoch=1,
            seq=i,
            ts=START,
            run_id="r",
            context_id="r",
            goal_id=goal,
            type=kind,
            round=99,
            payload={"round": round_, "frontier": f.model_dump(mode="json")},
        )
        for i, (goal, round_, kind, f) in enumerate(
            [
                ("b", 3, EventType.FRONTIER, first),
                ("a", 1, EventType.ROUND_STARTED, second),
                ("a", 2, EventType.FRONTIER, second),
            ],
            1,
        )
    ]
    result = recorded_frontiers(iter(events))
    assert result == [("b", 3, first), ("a", 2, second)]
    result[0][2].reviews.clear()
    assert events[0].payload["frontier"]["reviews"] == first.reviews


def test_recorded_frontiers_rejects_invalid_round() -> None:
    event = Event(
        epoch=1,
        seq=1,
        ts=START,
        run_id="r",
        context_id="r",
        type=EventType.FRONTIER,
        payload={"round": "SECRET_REINDEX", "frontier": {}},
    )
    with pytest.raises(ValueError, match="round") as exc:
        recorded_frontiers([event])
    assert_secret_absent(exc.value, "SECRET_REINDEX")


@pytest.mark.parametrize("change", ["inputs", "manifest"])
def test_reindex_preserves_revision_activation_and_rankings(
    run_config: RunConfig, tmp_path: Path, change: str
) -> None:
    run = FakeRun(run_config.run_root, "r1")
    initial = run.plan(parts=["a", "b"])
    old_key = Plan.model_validate(initial.payload["plan"]).parts[0].goal.key()

    def round_event(number: int, closed: bool = False) -> EventDraft:
        return EventDraft(
            type=EventType.ROUND_CLOSED if closed else EventType.ROUND_STARTED,
            goal_id=run.goal.id,
            round=number,
            payload={"round": number},
        )

    run.commit(
        initial, round_event(1), run.candidate("old", part="a"), run.score("old-score", "old", 100)
    )
    add_pointer(run_config.arena_root, run)
    directory = tmp_path / "cache"
    board = ArenaBoard(arena_root=run_config.arena_root, cache_dir=str(directory))
    board.open("", "", ["toy"])
    try:
        first = board.frontier()
        revision = run.plan(revision=1, parts=["a"], first_round=3)
        subgoal = revision.payload["plan"]["parts"][0]["goal"]
        subgoal[change if change == "inputs" else "files"] = (
            {"version": 2}
            if change == "inputs"
            else {"globs": [], "manifest": {"input.txt": "a" * 64}}
        )
        new_key = Plan.model_validate(revision.payload["plan"]).parts[0].goal.key()
        run.commit(round_event(1, True), revision)
        pending = board.frontier()
        run.commit(
            round_event(3),
            run.candidate("new", part="a", round=3),
            run.score("new-score", "new", 1),
        )
        current = board.frontier()
        frontiers = [first, pending, current]
        before = [snapshot_membership(board, run.goal.key(), f) for f in frontiers]
        ranked = [
            board.snapshot(run.goal.key(), f).leaderboard(run.goal.key(), judge="j")
            for f in frontiers
        ]
        assert [[r.candidate_id for r in rows] for rows in ranked] == [["old"], ["old"], ["new"]]
    finally:
        board.close()
        run._lease.release()
    assert reindex(run_config, ["toy"], cache_dir=directory) == current
    fresh = ArenaBoard(arena_root=run_config.arena_root, cache_dir=str(directory))
    try:
        assert [snapshot_membership(fresh, run.goal.key(), f) for f in frontiers] == before
        assert [
            fresh.snapshot(run.goal.key(), f).leaderboard(run.goal.key(), judge="j")
            for f in frontiers
        ] == ranked
        old_view = fresh.snapshot(old_key, current)
        assert old_view._visible_candidate_ids() == frozenset({"old"})
        view = fresh.snapshot(run.goal.key(), current)
        assert view._visible_candidate_ids() == frozenset({"new"})
        assert view._sibling_view("a").goal == new_key
        with pytest.raises(ValueError, match="part"):
            view._sibling_view("b")
        assert fresh.snapshot(run.goal.key(), pending)._sibling_view("a").goal == old_key
        assert fresh.snapshot(run.goal.key(), pending)._sibling_view("b").part == "b"
    finally:
        fresh.close()


def test_snapshot_membership_rejects_forged_cached_marks(
    run_config: RunConfig, tmp_path: Path
) -> None:
    run = _seed(run_config)
    board = ArenaBoard(arena_root=run_config.arena_root, cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        frontier = board.frontier()
        assert snapshot_membership(board, run.goal.key(), frontier)
        bad = frontier.model_copy(deep=True)
        bad.marks[0] = bad.marks[0].model_copy(update={"seq": bad.marks[0].seq + 1})
        with pytest.raises(ChainError):
            snapshot_membership(board, run.goal.key(), bad)
        conflict = frontier.model_copy(update={"marks": [*frontier.marks, *bad.marks]})
        with pytest.raises(ChainError):
            snapshot_membership(board, run.goal.key(), conflict)
    finally:
        board.close()


def test_docstring_examples() -> None:
    import doctest

    module = importlib.import_module("swarm_arena._board.reindex")
    result = doctest.testmod(module, raise_on_error=True)
    assert result.failed == 0
    assert result.attempted > 0
