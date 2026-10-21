"""Rebuild caches safely and preserve recorded snapshot identities."""

import fcntl
import importlib
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from _fakes.journals import START, FakeRun, add_pointer
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Plan, RunConfig
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.cache import ArenaCache
from swarm_arena._board.human_objects import HumanObject, write_human_objects
from swarm_arena._board.reindex import (
    _reindex_with_diagnostics,
    recorded_frontiers,
    reindex,
    snapshot_membership,
)
from swarm_arena._errors import ChainError
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.reader import read_chain
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.results import Learning
from swarm_arena._store.base import open_store


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
        assert (
            str(exc.value) == "arena cache is in use by another swarm-arena process; stop it first"
        )
        assert_secret_absent(exc.value, str(path))
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


def _logical_rows(directory: Path) -> dict[str, list[tuple[Any, ...]]]:
    """Compare every persisted column except database-local surrogate rowids."""
    connection = sqlite3.connect(directory / "toy.sqlite")
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        result = {}
        for (table,) in tables:
            columns = [
                row[1]
                for row in connection.execute(f'PRAGMA table_info("{table}")')
                if row[1] != "rowid"
            ]
            projection = ", ".join(f'"{column}"' for column in columns)
            result[table] = sorted(
                connection.execute(f'SELECT {projection} FROM "{table}"').fetchall(), key=repr
            )
        return result
    finally:
        connection.close()


def _snapshot_rows(board: ArenaBoard, goal: GoalKey, frontier: Frontier) -> dict[str, Any]:
    """Retain full model contents and provenance, not just ID membership."""
    view = board.snapshot(goal, frontier)
    with view._read():
        cache = view._cache
        return {
            "membership": snapshot_membership(board, goal, frontier),
            "candidates": [(c.model_dump(), p) for c, p in cache.candidates(frontier)],
            "scores": [(s.model_dump(), p) for s, p in cache.scores(frontier)],
            "learnings": [learning.model_dump() for learning in cache._learnings(frontier)],
            "shared": view.get_candidate("shared").model_dump(),
        }


@pytest.mark.parametrize("collide_all", [False, True])
def test_reindex_collisions_preserve_rows_and_every_recorded_frontier(
    tmp_path: Path, collide_all: bool
) -> None:
    config = RunConfig(run_root=str(tmp_path / "runs"), arena_root=str(tmp_path / "arena"))
    incremental = tmp_path / "incremental"
    rebuilt = tmp_path / "rebuilt"
    board = ArenaBoard(arena_root=config.arena_root, cache_dir=str(incremental))
    fresh = ArenaBoard(arena_root=config.arena_root, cache_dir=str(rebuilt))
    runs = [FakeRun(config.run_root, rid) for rid in ("z-run", "a-run")]
    snapshots = []

    def record(number: int) -> None:
        frontier = board.frontier()
        snapshots.append(_snapshot_rows(board, runs[0].goal.key(), frontier))
        runs[0].commit(
            EventDraft(
                type=EventType.FRONTIER,
                goal_id=runs[0].goal.id,
                payload={"round": number, "frontier": frontier.model_dump(mode="json")},
            )
        )

    def items(run: FakeRun, value: int) -> list[EventDraft]:
        suffix = "shared" if collide_all else run.run_id
        return [
            run.candidate("shared", team=run.run_id),
            run.score(f"score-{suffix}", "shared", value),
            run.learning(
                f"learning-{suffix}",
                "Hint",
                evidence=["shared"],
                text=f"SECRET_COLLISION_CONTENT-{run.run_id}",
                tags=[run.run_id],
            ),
        ]

    try:
        board.open("", "", ["toy"])
        runs[0].commit(runs[0].plan(), *items(runs[0], 1))
        add_pointer(config.arena_root, runs[0])
        record(1)
        # The smaller run is marked before its colliding row is committed.
        runs[1].commit(runs[1].plan())
        add_pointer(config.arena_root, runs[1])
        record(2)
        runs[1].commit(*items(runs[1], 2))
        record(3)
        final = board.frontier()
        events = [e for _, batch in read_chain(runs[0].store, runs[0].heads[-1]) for e in batch]
        frontiers = recorded_frontiers(events)
        assert len(frontiers) == 3
        before = _logical_rows(incremental)
        assert reindex(config, ["toy"], cache_dir=rebuilt) == final
        after = _logical_rows(rebuilt)
        assert before == after
        for index, (_, _, frontier) in enumerate(frontiers):
            actual = _snapshot_rows(board, runs[0].goal.key(), frontier)
            assert actual == snapshots[index]
            assert _snapshot_rows(fresh, runs[0].goal.key(), frontier) == actual
            assert actual["shared"]["run_id"] == ("a-run" if index == 2 else "z-run")
        assert "SECRET_COLLISION_CONTENT" not in repr(board._caches["toy"].warnings)
    finally:
        board.close()
        fresh.close()
        for run in runs:
            run._lease.release()


def test_reindex_logical_rows_exclude_only_local_surrogate_rowids(tmp_path: Path) -> None:
    config = RunConfig(run_root=str(tmp_path / "runs"), arena_root=str(tmp_path / "arena"))
    incremental = tmp_path / "incremental"
    rebuilt = tmp_path / "rebuilt"
    run = FakeRun(config.run_root, "r1")
    board = ArenaBoard(arena_root=config.arena_root, cache_dir=str(incremental))
    fresh = ArenaBoard(arena_root=config.arena_root, cache_dir=str(rebuilt))

    def rowids(directory: Path) -> dict[str, int]:
        connection = sqlite3.connect(directory / "toy.sqlite")
        try:
            return dict(connection.execute("SELECT id, rowid FROM learnings"))
        finally:
            connection.close()

    try:
        run.commit(
            run.plan(),
            run.candidate("shared"),
            run.score("s1", "shared", 1),
            run.learning("L1", "First", evidence=["shared"]),
        )
        add_pointer(config.arena_root, run)
        board.open("", "", ["toy"])
        frontiers = [board.frontier()]
        learning = Learning(
            id="L-h-0123456789abcdef",
            namespace="toy",
            run_id="",
            goal_id=run.goal.id,
            round=0,
            author="human:alice",
            title="Human",
            text="Human insight",
            evidence=["shared"],
            tags=["human"],
            refines=["L1"],
        )
        write_human_objects(
            open_store(config.arena_root),
            "toy",
            "curation",
            [HumanObject(kind="add", learning=learning, author="human:alice")],
            now=START,
        )
        frontiers.append(board.frontier())
        run.commit(run.learning("L2", "Last", evidence=["shared"], supersedes=["L1"]))
        frontiers.append(board.frontier())
        before = [_snapshot_rows(board, run.goal.key(), f) for f in frontiers]
        assert reindex(config, ["toy"], cache_dir=rebuilt) == frontiers[-1]
        # Prove this exercises different allocation orders rather than merely
        # omitting a column whose values happened to be identical.
        assert rowids(incremental) == {"L1": 1, learning.id: 2, "L2": 3}
        assert rowids(rebuilt) == {"L1": 1, "L2": 2, learning.id: 3}
        assert _logical_rows(incremental) == _logical_rows(rebuilt)
        assert [_snapshot_rows(fresh, run.goal.key(), f) for f in frontiers] == before
    finally:
        board.close()
        fresh.close()
        run._lease.release()
