import shutil
import sqlite3
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier

import pytest
from _fakes.journals import START, FakeRun
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Goal, Plan
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.cache import goal_key_digest
from swarm_arena._board.human_objects import HumanObject, write_human_objects
from swarm_arena._board.pointers import RunPointer, pointer_key, write_pointer
from swarm_arena._errors import ChainError, StorageError, _rebuild_core_signal
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.results import Candidate, Learning, Score
from swarm_arena._store.base import open_store


@pytest.fixture
def board(tmp_path: Path) -> Iterator[ArenaBoard]:
    result = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    result.open("", "", ["toy"])
    yield result
    result.close()


def _publish(tmp_path: Path, run: FakeRun) -> None:
    write_pointer(
        open_store(str(tmp_path / "arena")),
        run.goal.namespace,
        RunPointer(run_id=run.run_id, run_root=run.run_root),
    )


def _seed(tmp_path: Path, run_id: str = "r1") -> FakeRun:
    run = FakeRun(str(tmp_path / "runs"), run_id)
    run.commit(run.plan(), run.candidate(f"c-{run_id}"), run.score(f"s-{run_id}", f"c-{run_id}", 1))
    _publish(tmp_path, run)
    return run


def _learning(run: FakeRun, id: str = "L-1") -> Learning:
    draft = run.learning(id, "viscosity", author="human:a")
    run.commit(draft)
    return Learning.model_validate(draft.payload["learning"])


def _curate(tmp_path: Path, kind: str, learning_id: str) -> str:
    return write_human_objects(
        open_store(str(tmp_path / "arena")),
        "toy",
        "curation",
        [HumanObject.model_validate(dict(kind=kind, learning_id=learning_id, author="human:a"))],
        now=START,
    )


def _round(run: FakeRun, number: int, *, closed: bool = False) -> EventDraft:
    return EventDraft(
        type=EventType.ROUND_CLOSED if closed else EventType.ROUND_STARTED,
        goal_id=run.goal.id,
        round=number,
        payload={"round": number},
    )


def test_init_does_no_io(tmp_path: Path) -> None:
    ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    assert not (tmp_path / "arena").exists()
    assert not (tmp_path / "cache").exists()


def test_default_cache_dir_uses_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert ArenaBoard(arena_root="/x/arena")._cache_dir() == (
        tmp_path / "swarm-arena" / sha256_hex(b"/x/arena")[:12]
    )
    monkeypatch.chdir(tmp_path)
    assert (
        ArenaBoard(arena_root="arena")._cache_dir()
        == ArenaBoard(arena_root=str(tmp_path / "arena"))._cache_dir()
    )


def test_open_writes_pointer_per_namespace(tmp_path: Path, board: ArenaBoard) -> None:
    board.open("r1", str(tmp_path / "runs"), ["toy", "phys"])
    arena = open_store(str(tmp_path / "arena"))
    for ns in ("toy", "phys"):
        pointer = RunPointer.model_validate_json(arena.read(pointer_key(ns, "r1")).data)
        assert pointer.run_root == str((tmp_path / "runs").resolve())
    assert not (tmp_path / "cache").exists()


def test_reader_open_writes_nothing(tmp_path: Path, board: ArenaBoard) -> None:
    assert not (tmp_path / "arena").exists()
    assert not (tmp_path / "cache").exists()


def test_open_resume_with_other_run_root_warns(
    tmp_path: Path,
    board: ArenaBoard,
    caplog: pytest.LogCaptureFixture,
) -> None:
    board.open("r1", str(tmp_path / "a"), ["toy", "phys"])
    arena = open_store(str(tmp_path / "arena"))
    original = {ns: arena.read(pointer_key(ns, "r1")).data for ns in ("toy", "phys")}
    board.open("r1", str(tmp_path / "b"), ["toy", "phys"])
    assert len([r for r in caplog.records if r.name == "swarm_arena.board"]) == 1
    assert str(tmp_path / "a") in caplog.text
    assert str(tmp_path / "b") in caplog.text
    assert original == {ns: arena.read(pointer_key(ns, "r1")).data for ns in original}


def test_own_run_read_through_open_root_after_move(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    old, new = tmp_path / "a", tmp_path / "b"
    run = FakeRun(str(old), "r1")
    head = run.commit(run.plan(), run.candidate("c-1"))
    config = dict(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    original = ArenaBoard(**config)
    original.open("r1", str(old), ["toy"])
    original.close()
    shutil.move(old, new)
    moved = ArenaBoard(**config)
    reader = ArenaBoard(arena_root=config["arena_root"], cache_dir=str(tmp_path / "reader"))
    try:
        moved.open("r1", str(new), ["toy"])
        frontier = moved.frontier()
        assert len(frontier.marks) == 1
        mark = frontier.marks[0]
        assert (mark.run_root, mark.epoch, mark.seq) == (str(new.resolve()), head.epoch, head.seq)
        assert moved.snapshot(run.goal.key(), frontier).get_candidate("c-1").id == "c-1"
        caplog.clear()
        reader.open("", "", ["toy"])
        assert reader.frontier().marks == []
        assert len(caplog.records) == 1
    finally:
        moved.close()
        reader.close()


def test_frontier_marks_every_run_with_a_head(tmp_path: Path, board: ArenaBoard) -> None:
    runs = [_seed(tmp_path, id) for id in ("r2", "r1")]
    frontier = board.frontier()
    assert [m.run_id for m in frontier.marks] == ["r1", "r2"]
    by_id = {run.run_id: run for run in runs}
    for mark in frontier.marks:
        head = by_id[mark.run_id].heads[-1]
        assert (mark.epoch, mark.seq, mark.segment) == (head.epoch, head.seq, head.segment)


def test_frontier_skips_broken_runs(
    tmp_path: Path,
    board: ArenaBoard,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(tmp_path)
    arena = open_store(str(tmp_path / "arena"))
    bad = tmp_path / "not-a-dir"
    bad.write_text("not a directory")
    for id, root in (("r2", str(bad)), ("r3", str(tmp_path / "missing")), ("r7", "s3://b/runs")):
        write_pointer(arena, "toy", RunPointer(run_id=id, run_root=root))
    run = FakeRun(str(tmp_path / "runs"), "r4")
    _publish(tmp_path, run)
    arena.create("toy/runs/r5.json", b"not json")
    arena.create("toy/runs/r6.json", b'{"schema_version":99,"run_id":"r6","run_root":"/runs"}')
    monkeypatch.setitem(sys.modules, "s3fs", None)
    assert [m.run_id for m in board.frontier().marks] == ["r1"]
    assert len([r for r in caplog.records if r.name == "swarm_arena.board"]) == 6


def test_frontier_raises_for_own_run(tmp_path: Path, board: ArenaBoard) -> None:
    bad = tmp_path / "not-a-dir"
    bad.write_text("file")
    board.open("r9", str(bad), ["toy"])
    with pytest.raises((OSError, StorageError)):
        board.frontier()


def test_own_run_without_head_gets_no_mark(
    tmp_path: Path,
    board: ArenaBoard,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r9")
    board.open("r9", run.run_root, ["toy"])
    assert board.frontier().marks == []
    assert not caplog.records


def test_snapshot_is_exactly_the_frontier(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    first = _learning(run)
    frontier = board.frontier()
    _learning(run, "L-late")
    _curate(tmp_path, "pin", first.id)
    view = board.snapshot(run.goal.key(), frontier)
    assert [x.id for x in view.learnings("toy")] == [first.id]
    assert not view.get_learning(first.id).pinned
    assert view.frontier == frontier
    newer = board.snapshot(run.goal.key(), board.frontier())
    assert {x.id for x in newer.learnings("toy")} == {first.id, "L-late"}
    assert newer.get_learning(first.id).pinned
    assert [x.id for x in view.learnings("toy")] == [first.id]
    frontier.marks.clear()
    assert view.get_candidate("c-r1").id == "c-r1"


def test_snapshot_same_on_fresh_cache(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    _learning(run)
    _curate(tmp_path, "pin", "L-1")
    frontier = board.frontier()
    frozen = board.snapshot(run.goal.key(), frontier)
    (tmp_path / "arena" / "toy" / "runs" / "r1.json").unlink()
    fresh = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "fresh"))
    try:
        fresh.open("", "", ["toy"])
        view = fresh.snapshot(run.goal.key(), frontier)
        assert view.leaderboard(view.goal, judge="j") == frozen.leaderboard(frozen.goal, judge="j")
        assert view.learnings("toy") == frozen.learnings("toy")
        assert view._visible_candidate_ids() == frozen._visible_candidate_ids()
        assert fresh.frontier().marks == []
    finally:
        fresh.close()


def test_view_rejects_other_goal_or_namespace(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    view = board.snapshot(run.goal.key(), board.frontier())
    other = Goal(id="other", namespace="toy").key()
    with pytest.raises(ValueError, match="goal") as caught:
        view.best(other, 1, judge="j")
    assert _rebuild_core_signal(caught.value) is not None
    for method in (view.learnings, view.count_learnings):
        with pytest.raises(ValueError, match="namespace"):
            method("other")
        with pytest.raises(ValueError, match="goal"):
            method("toy", goal=other)
    with pytest.raises(ValueError, match="goal"):
        view.search_learnings("toy", "secret", goal=other)


def test_hints_never_in_snapshot(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    candidate = Candidate.model_validate(run.candidate("hint").payload["candidate"])
    learning = Learning.model_validate(run.learning("L-hint", "hint").payload["learning"])
    score = Score.model_validate(run.score("s-hint", "hint", 1).payload["score"])
    board.put_candidate(candidate, epoch=1, seq=99)
    board.put_learning(learning, epoch=1, seq=100)
    board.put_score(score, epoch=1, seq=101)
    assert board.get_candidate(candidate.id) == candidate
    assert board.get_learning(learning.id) == learning
    assert board.scores(candidate.id) == [score]
    view = board.snapshot(run.goal.key(), board.frontier())
    for method, id in (
        (view.get_candidate, "hint"),
        (view.get_learning, "L-hint"),
        (view.scores, "hint"),
    ):
        with pytest.raises(KeyError):
            method(id)
    assert "hint" not in view._visible_candidate_ids()
    assert board.learnings("toy") == []


def test_gated_out_kept_but_not_ranked(tmp_path: Path, board: ArenaBoard) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan(), run.candidate("gated", passed=False), run.score("s", "gated", 1))
    _publish(tmp_path, run)
    view = board.snapshot(run.goal.key(), board.frontier())
    assert view.get_candidate("gated").gated_out
    assert view.best(view.goal, 10, judge="j") == []


def test_hidden_never_returned(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    learning = _learning(run)
    _curate(tmp_path, "hide", learning.id)
    board.put_learning(learning, epoch=1, seq=99)
    view = board.snapshot(run.goal.key(), board.frontier())
    for reader in (board, view):
        assert reader.learnings("toy") == []
        assert reader.search_learnings("toy", "viscosity") == []
        with pytest.raises(KeyError):
            reader.get_learning(learning.id)
    assert view.count_learnings("toy") == 0


def test_best_ranks_namespace_under_own_identity(tmp_path: Path, board: ArenaBoard) -> None:
    first = _seed(tmp_path, "r1")
    first.commit(first.score("old", "c-r1", 0, rubric_hash="old"))
    first.commit(first.score("new-1", "c-r1", 1, rubric_hash="new"))
    second = _seed(tmp_path, "r2")
    second.commit(second.score("new-2", "c-r2", 2, rubric_hash="new"))
    first.commit(first.score("latest", "c-r1", 0, rubric_hash="r1-latest"))
    board.open("r2", second.run_root, ["toy"])
    view = board.snapshot(second.goal.key(), board.frontier())
    assert sorted(c.id for c in view.best(view.goal, 5, judge="j")) == ["c-r1", "c-r2"]
    assert {r.rubric_hash for r in view.leaderboard(view.goal, judge="j")} == {"new"}
    assert [c.id for c in view.best(view.goal, 5, judge="j", run_id="r1")] == ["c-r1"]


def test_view_for_part_key(tmp_path: Path, board: ArenaBoard) -> None:
    for id in ("r1", "r2"):
        run = FakeRun(str(tmp_path / "runs"), id)
        draft = run.plan(parts=["a", "b"])
        run.commit(draft, run.candidate(f"c-{id}", part="a"), run.score(f"s-{id}", f"c-{id}", 1))
        _learning(run, f"L-{id}")
        _publish(tmp_path, run)
    main = run.goal.key()
    part = Plan.model_validate(draft.payload["plan"]).parts[0].goal.key()
    view = board.snapshot(part, board.frontier())
    assert view.part == "a"
    assert sorted(c.id for c in view.best(part, 5, judge="j")) == ["c-r1", "c-r2"]
    assert view.best(part, 5, judge="j", part="a") == view.best(part, 5, judge="j")
    with pytest.raises(ValueError, match="part"):
        view.best(part, 5, judge="j", part="b")
    assert view.learning_goal == (goal_key_digest(main), main.id)
    assert {x.id for x in view.learnings("toy", goal=part)} == {"L-r1", "L-r2"}
    assert view.count_learnings("toy", goal=part) == 2


def test_namespace_search_puts_own_goal_first(tmp_path: Path, board: ArenaBoard) -> None:
    mine = _seed(tmp_path)
    _learning(mine, "L-mine")
    other = FakeRun(str(tmp_path / "runs"), "r2", goal_id="g2", start=START + timedelta(hours=1))
    other.commit(other.plan())
    _learning(other, "L-other")
    _publish(tmp_path, other)
    view = board.snapshot(mine.goal.key(), board.frontier())
    assert [x.id for x in view.search_learnings("toy", "viscosity")] == ["L-mine", "L-other"]
    assert [x.id for x in view.learnings("toy")] == ["L-mine", "L-other"]


def test_lineage_follows_parents_across_runs(tmp_path: Path, board: ArenaBoard) -> None:
    _seed(tmp_path, "r1")
    run = FakeRun(str(tmp_path / "runs"), "r2")
    child = run.candidate("c-r2")
    child.payload["candidate"]["parent_ids"] = ["c-r1", "missing"]
    run.commit(run.plan(), child)
    _publish(tmp_path, run)
    view = board.snapshot(run.goal.key(), board.frontier())
    assert view.lineage("c-r2").edges == [("c-r2", "c-r1")]
    assert set(view.lineage("c-r2").nodes) == {"c-r2", "c-r1"}


def test_count_learnings_matches_learnings(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    _learning(run)
    view = board.snapshot(run.goal.key(), board.frontier())
    assert view.count_learnings("toy") == len(view.learnings("toy")) == 1
    assert view.count_learnings("toy", goal=view.goal) == 1


@pytest.mark.parametrize("change", ["inputs", "manifest"])
def test_current_keys_and_siblings_follow_executed_revision(
    tmp_path: Path,
    board: ArenaBoard,
    change: str,
) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    initial = run.plan(parts=["a", "b"])
    old_key = Plan.model_validate(initial.payload["plan"]).parts[0].goal.key()
    run.commit(
        initial, _round(run, 1), run.candidate("old", part="a"), run.score("old-s", "old", 100)
    )
    _publish(tmp_path, run)
    board.open("r1", run.run_root, ["toy"])
    frozen = board.snapshot(run.goal.key(), board.frontier())
    revision = run.plan(revision=1, parts=["a"], first_round=3)
    subgoal = revision.payload["plan"]["parts"][0]["goal"]
    if change == "inputs":
        subgoal["inputs"] = {"version": 2}
    else:
        subgoal["files"] = {"globs": [], "manifest": {"input.txt": "a" * 64}}
    new_key = Plan.model_validate(revision.payload["plan"]).parts[0].goal.key()
    run.commit(_round(run, 1, closed=True), revision)
    pending = board.snapshot(run.goal.key(), board.frontier())
    assert [c.id for c in pending.best(pending.goal, 5, judge="j", part="a")] == ["old"]
    assert pending._sibling_view("b").part == "b"
    run.commit(
        _round(run, 3), run.candidate("new", part="a", round=3), run.score("new-s", "new", 1)
    )
    current = board.snapshot(run.goal.key(), board.frontier())
    assert [c.id for c in current.best(current.goal, 5, judge="j", part="a")] == ["new"]
    assert current._visible_candidate_ids() == frozenset({"new"})
    sibling = current._sibling_view("a")
    assert sibling.goal == new_key
    assert sibling._visible_candidate_ids() == frozenset({"new"})
    with pytest.raises(KeyError):
        sibling.get_candidate("old")
    with pytest.raises(ValueError, match="part"):
        current._sibling_view("b")
    assert frozen._sibling_view("a").goal == old_key
    assert frozen._sibling_view("b").part == "b"
    history = board.snapshot(old_key, board.frontier())
    assert [c.id for c in history.best(old_key, 5, judge="j")] == ["old"]
    assert {r.candidate_id for r in current.leaderboard(current.goal, judge="j")} == {"new"}


def test_empty_named_objects_remain_in_frontier(tmp_path: Path, board: ArenaBoard) -> None:
    arena = open_store(str(tmp_path / "arena"))
    review = write_human_objects(arena, "toy", "reviews", [], now=START, reviewer="alice")
    curation = write_human_objects(arena, "toy", "curation", [], now=START)
    frontier = board.frontier()
    assert frontier.reviews == [review]
    assert frontier.curation == [curation]
    view = board.snapshot(Goal(id="g", namespace="toy").key(), frontier)
    assert view.count_learnings("toy") == 0
    assert view.learnings("toy") == []
    assert view.search_learnings("toy", "anything") == []


def test_missing_named_object_raises(tmp_path: Path, board: ArenaBoard) -> None:
    name = "toy/curation/20261018T000000.000Z-00000001.jsonl"
    with pytest.raises(StorageError):
        board.snapshot(Goal(id="g", namespace="toy").key(), Frontier(curation=[name]))


def test_unsafe_cache_namespace_is_rejected_without_io(tmp_path: Path, board: ArenaBoard) -> None:
    secret = "SENTINEL_SECRET"
    with pytest.raises(ValueError, match="namespace") as caught:
        board._cache("../" + secret)
    assert_secret_absent(caught.value, secret)
    assert _rebuild_core_signal(caught.value) is not None
    assert not (tmp_path / "cache").exists()


def test_lazy_cache_and_hints_are_synchronized(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    candidate = Candidate.model_validate(run.candidate("hint").payload["candidate"])
    barrier = Barrier(8)

    def access(index: int) -> int:
        barrier.wait(timeout=5)
        board.put_candidate(candidate, epoch=1, seq=index)
        cache = board._cache("toy")
        assert board.get_candidate("hint") == candidate
        assert board.snapshot(run.goal.key(), board.frontier()).get_candidate("c-r1").id == "c-r1"
        return id(cache)

    with ThreadPoolExecutor(8) as pool:
        assert len(set(pool.map(access, range(8)))) == 1
    cache = board._cache("toy")
    board.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        cache._conn.execute("SELECT 1")


def test_snapshot_rejects_conflicting_marks(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    frontier = board.frontier()
    conflicting = frontier.model_copy(
        update={"marks": [*frontier.marks, frontier.marks[0].model_copy(update={"seq": 999})]}
    )
    with pytest.raises(ChainError):
        board.snapshot(run.goal.key(), conflicting)


@pytest.mark.s3
def test_frontier_and_snapshot_on_s3(s3_url: str, tmp_path: Path) -> None:
    run = FakeRun(s3_url + "/runs", "r1")
    run.commit(run.plan(), run.candidate("c-1"))
    board = ArenaBoard(arena_root=s3_url + "/arena", cache_dir=str(tmp_path / "cache"))
    try:
        board.open("r1", run.run_root, ["toy"])
        assert board.snapshot(run.goal.key(), board.frontier()).get_candidate("c-1").id == "c-1"
    finally:
        board.close()


def test_part_view_ranks_equal_keys_as_one_group_across_different_names(
    tmp_path: Path,
    board: ArenaBoard,
) -> None:
    key = Goal(id="shared", namespace="toy").key()
    for index, part in enumerate(("a", "b"), 1):
        run = FakeRun(str(tmp_path / "runs"), f"r{index}")
        draft = run.plan(parts=[part])
        draft.payload["plan"]["parts"][0]["goal"]["id"] = "shared"
        run.commit(
            draft,
            run.candidate(f"c-{index}", part=part),
            run.score(f"s-{index}", f"c-{index}", index),
        )
        _publish(tmp_path, run)
    view = board.snapshot(key, board.frontier())
    assert view._visible_candidate_ids() == frozenset({"c-1", "c-2"})
    assert [c.id for c in view.best(key, 5, judge="j")] == ["c-1", "c-2"]
    rows = view.leaderboard(key, judge="j")
    assert [row.rank for row in rows] == [1, 2]
    assert [row.part for row in rows] == ["a", "b"]


def test_own_corrupt_pointer_raises_instead_of_warning(
    tmp_path: Path,
    board: ArenaBoard,
) -> None:
    run = _seed(tmp_path)
    board.open(run.run_id, run.run_root, ["toy"])
    (tmp_path / "arena" / "toy" / "runs" / "r1.json").write_bytes(b"broken")
    with pytest.raises(StorageError, match="pointer"):
        board.frontier()


def test_frontier_extends_all_open_caches(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    board.snapshot(Goal(id="other", namespace="phys").key(), board.frontier())
    run.commit(run.candidate("late"))
    frontier = board.frontier()
    assert board._cache("phys").newest_segment("r1") == frontier.marks[0].segment


def test_large_visible_membership_uses_shared_named_sets(
    tmp_path: Path,
    board: ArenaBoard,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena._board import arena as arena_module

    run = FakeRun(str(tmp_path / "runs"), "r1")
    events = [run.plan()]
    for number in range(512):
        events.extend(
            [run.candidate(f"c-{number}"), run.score(f"s-{number}", f"c-{number}", number)]
        )
    run.commit(*events)
    _publish(tmp_path, run)
    view = board.snapshot(run.goal.key(), board.frontier())
    cache = view._cache
    cache._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 64)
    rows = view.leaderboard(view.goal, judge="j")
    assert len(rows) == 512
    assert [c.id for c in view.best(view.goal, 3, judge="j")] == ["c-0", "c-1", "c-2"]
    assert not cache._conn.in_transaction

    def no_io(*args: object, **kwargs: object) -> None:
        pytest.fail("fixed membership must not read storage or the cache")

    monkeypatch.setattr(cache, "candidates", no_io)
    monkeypatch.setattr(arena_module, "run_store", no_io)
    assert view._visible_candidate_ids() == frozenset(f"c-{n}" for n in range(512))


def test_namespace_objects_are_scoped_without_changing_recorded_frontier(
    tmp_path: Path,
    board: ArenaBoard,
) -> None:
    board.open("", "", ["toy", "phys"])
    arena = open_store(str(tmp_path / "arena"))
    for namespace in ("toy", "phys"):
        write_human_objects(arena, namespace, "curation", [], now=START)
    frontier = board.frontier()
    assert len(frontier.curation) == 2
    for namespace in ("toy", "phys"):
        view = board.snapshot(Goal(id="g", namespace=namespace).key(), frontier)
        assert view.frontier == frontier
        assert view.learnings(namespace) == []
        assert view.search_learnings(namespace, "text") == []
        assert view.count_learnings(namespace) == 0


def test_sibling_rankings_are_recorded_and_do_not_import_plugins(
    tmp_path: Path,
    board: ArenaBoard,
) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    draft = run.plan(parts=["a", "b"])
    for part in draft.payload["plan"]["parts"]:
        name = f"rank-{part['name']}"
        part["ranking"] = name
        part["judges"] = [{"path": "missing_plugin:Judge", "name": name}]
    plan = Plan.model_validate(draft.payload["plan"])
    run.commit(
        draft,
        run.candidate("a", part="a"),
        run.candidate("b", part="b"),
        run.score("s-a", "a", 1, judge="rank-a"),
        run.score("s-b", "b", 1, judge="rank-b"),
    )
    _publish(tmp_path, run)
    view = board.snapshot(plan.parts[0].goal.key(), board.frontier())
    sibling = view._sibling_view("b")
    assert sibling._ranking_judges() == {"b": "rank-b"}
    assert [c.id for c in sibling.best(sibling.goal, 2, judge="rank-b")] == ["b"]
    assert sibling._visible_candidate_ids() == frozenset({"b"})
    assert view._visible_candidate_ids() == frozenset({"a"})
    with pytest.raises(KeyError):
        view.get_candidate("b")
    assert view._sibling_view(None).part is None
    with pytest.raises(ValueError, match="part"):
        view._sibling_view("")


def test_reader_and_unscored_owner_use_newest_identity(
    tmp_path: Path,
    board: ArenaBoard,
) -> None:
    run = _seed(tmp_path)
    run.commit(run.score("latest", "c-r1", 1, rubric_hash="newest"))
    reader = board.snapshot(run.goal.key(), board.frontier())
    assert {r.rubric_hash for r in reader.leaderboard(reader.goal, judge="j")} == {"newest"}
    own = FakeRun(str(tmp_path / "runs"), "unscored")
    own.commit(own.plan())
    board.open(own.run_id, own.run_root, ["toy"])
    assert [c.id for c in board.best(own.goal.key(), 2, judge="j")] == ["c-r1"]


def test_board_reads_reuse_latest_frontier_until_refresh(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    assert [c.id for c in board.best(run.goal.key(), 5, judge="j")] == ["c-r1"]
    run.commit(run.candidate("late"), run.score("late-s", "late", 0))
    assert [c.id for c in board.best(run.goal.key(), 5, judge="j")] == ["c-r1"]
    board.frontier()
    assert [c.id for c in board.best(run.goal.key(), 5, judge="j")] == ["late", "c-r1"]


def test_one_authority_read_per_run_per_frontier(
    tmp_path: Path,
    board: ArenaBoard,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena._board import arena as arena_module
    from swarm_arena._journal.lease import OwnerDoc
    from swarm_arena._store.base import Store

    run = _seed(tmp_path)
    write_pointer(
        open_store(str(tmp_path / "arena")),
        "phys",
        RunPointer(run_id=run.run_id, run_root=run.run_root),
    )
    board.open(run.run_id, run.run_root, ["toy", "phys"])
    original = arena_module.read_owner
    reads: list[str] = []

    def read(store: Store) -> tuple[OwnerDoc, str] | None:
        reads.append(store.url)
        return original(store)

    monkeypatch.setattr(arena_module, "read_owner", read)
    frontier = board.frontier()
    board.snapshot(run.goal.key(), frontier)
    board.snapshot(Goal(id="g", namespace="phys").key(), frontier)
    assert reads == [run.store.url]


def test_hidden_hint_cannot_bypass_named_curation(tmp_path: Path, board: ArenaBoard) -> None:
    run = _seed(tmp_path)
    learning = Learning.model_validate(run.learning("hint", "hint").payload["learning"])
    board.put_learning(learning, epoch=1, seq=99)
    _curate(tmp_path, "hide", learning.id)
    with pytest.raises(KeyError):
        board.get_learning(learning.id)


def test_no_candidate_view_preserves_learning_relevance(tmp_path: Path, board: ArenaBoard) -> None:
    mine = FakeRun(str(tmp_path / "runs"), "r1")
    mine.commit(mine.plan())
    _learning(mine, "mine")
    _publish(tmp_path, mine)
    other = FakeRun(str(tmp_path / "runs"), "r2", goal_id="other", start=START + timedelta(hours=1))
    other.commit(other.plan())
    _learning(other, "other")
    _publish(tmp_path, other)
    view = board.snapshot(mine.goal.key(), board.frontier())
    assert view._visible_candidate_ids() == frozenset()
    assert [item.id for item in view.learnings("toy")] == ["mine", "other"]
    assert [item.id for item in view.search_learnings("toy", "viscosity")] == ["mine", "other"]


def test_lineage_cycle_and_unknown_ids(tmp_path: Path, board: ArenaBoard) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan(), run.candidate("a", parents=["b"]), run.candidate("b", parents=["a"]))
    _publish(tmp_path, run)
    view = board.snapshot(run.goal.key(), board.frontier())
    assert view.lineage("a").edges == [("a", "b"), ("b", "a")]
    assert board.lineage("a") == view.lineage("a")
    for reader in (view, board):
        with pytest.raises(KeyError):
            reader.scores("missing")
        with pytest.raises(KeyError):
            reader.lineage("missing")


def test_view_error_registration_rebuilds_mutated_diagnostics(
    tmp_path: Path,
    board: ArenaBoard,
) -> None:
    key = Goal(id="g", namespace="toy").key()
    view = board.snapshot(key, Frontier())
    secret = "SENTINEL_SECRET"
    with pytest.raises(ValueError, match="namespace") as caught:
        view.learnings(secret)
    assert_secret_absent(caught.value, secret)
    caught.value.args = (secret,)
    caught.value.add_note(secret)
    caught.value.__cause__ = ValueError(secret)
    rebuilt = _rebuild_core_signal(caught.value)
    assert isinstance(rebuilt, ValueError)
    assert_secret_absent(rebuilt, secret)
    assert "namespace" in str(rebuilt)


def test_main_view_maps_equal_producing_keys_to_current_part_names(
    tmp_path: Path,
    board: ArenaBoard,
) -> None:
    first = FakeRun(str(tmp_path / "runs"), "r1")
    draft = first.plan(parts=["old"])
    draft.payload["plan"]["parts"][0]["goal"]["id"] = "shared"
    first.commit(draft, first.candidate("old-name", part="old"), first.score("s1", "old-name", 1))
    _publish(tmp_path, first)
    own = FakeRun(str(tmp_path / "runs"), "r2")
    draft = own.plan(parts=["current"])
    draft.payload["plan"]["parts"][0]["goal"]["id"] = "shared"
    own.commit(
        draft, own.candidate("current-name", part="current"), own.score("s2", "current-name", 2)
    )
    board.open(own.run_id, own.run_root, ["toy"])
    view = board.snapshot(own.goal.key(), board.frontier())
    assert view._visible_candidate_ids() == frozenset({"old-name", "current-name"})
    assert [c.id for c in view.best(view.goal, 5, judge="j", part="current")] == [
        "old-name",
        "current-name",
    ]
    assert [r.rank for r in view.leaderboard(view.goal, judge="j")] == [1, 2]
    sibling = view._sibling_view("current")
    direct = board.snapshot(sibling.goal, view.frontier)
    assert (
        direct._visible_candidate_ids()
        == sibling._visible_candidate_ids()
        == frozenset({"old-name", "current-name"})
    )


def test_part_view_uses_owner_name_and_historical_recorded_judge(
    tmp_path: Path,
    board: ArenaBoard,
) -> None:
    other = FakeRun(str(tmp_path / "runs"), "r1")
    earlier = other.plan(parts=["other"])
    earlier.payload["plan"]["parts"][0]["goal"]["id"] = "shared"
    other.commit(earlier)
    _publish(tmp_path, other)
    own = FakeRun(str(tmp_path / "runs"), "r2")
    draft = own.plan(parts=["own"])
    draft.payload["plan"]["parts"][0]["goal"]["id"] = "shared"
    key = Plan.model_validate(draft.payload["plan"]).parts[0].goal.key()
    own.commit(draft, _round(own, 1), own.candidate("c", part="own"), own.score("s", "c", 1))
    board.open(own.run_id, own.run_root, ["toy"])
    view = board.snapshot(key, board.frontier())
    assert view.part == "own"
    assert view._ranking_judges() == {"own": "j"}
    assert [c.id for c in view.best(key, 5, judge="j", part="own")] == ["c"]
    own.commit(_round(own, 1, closed=True), own.plan(revision=1, parts=[], first_round=2))
    historical = board.snapshot(key, board.frontier())
    assert historical._ranking_judges() == {"own": "j"}
    assert historical._visible_candidate_ids() == frozenset({"c"})
    with pytest.raises(ValueError, match="part"):
        historical._sibling_view("own")


def test_part_learning_resolution_is_stable_and_main_sibling_has_main_scope(
    tmp_path: Path,
    board: ArenaBoard,
) -> None:
    keys = {}
    for index in (1, 2):
        run = FakeRun(
            str(tmp_path / "runs"),
            f"r{index}",
            goal_id=f"g{index}",
            start=START + timedelta(hours=index),
        )
        draft = run.plan(parts=["a"])
        draft.payload["plan"]["parts"][0]["goal"]["id"] = "shared"
        run.commit(draft)
        _learning(run, f"L-{index}")
        _publish(tmp_path, run)
        keys[index] = run.goal.key()
    board.open("r2", run.run_root, ["toy"])
    part = Goal(id="shared", namespace="toy").key()
    view = board.snapshot(part, board.frontier())
    assert view.learning_goal == (goal_key_digest(keys[2]), "g2")
    assert [item.id for item in view.learnings("toy", goal=part)] == ["L-2"]
    assert view.count_learnings("toy", goal=part) == 1
    assert [item.id for item in view.search_learnings("toy", "viscosity")] == ["L-2", "L-1"]
    main = view._sibling_view(None)
    assert main.goal == keys[2]
    assert main.learning_goal == (goal_key_digest(keys[2]), "g2")
    assert [item.id for item in main.learnings("toy", goal=main.goal)] == ["L-2"]
    board.open("", "", ["toy"])
    fallback = board.snapshot(part, view.frontier)
    assert fallback.learning_goal == (goal_key_digest(keys[1]), "g1")
    assert [item.id for item in fallback.learnings("toy", goal=part)] == ["L-1"]


@pytest.mark.parametrize("reopen", [False, True])
def test_repeated_reads_reuse_committed_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reopen: bool
) -> None:
    from collections import Counter

    from swarm_arena._journal.event import Event
    from swarm_arena._journal.replay import RunState

    runs = [FakeRun(str(tmp_path / "runs"), f"r{i}") for i in range(2)]
    for run in runs:
        run.commit(
            run.plan(parts=["a"]),
            run.candidate(f"c-{run.run_id}", part="a"),
            run.score(f"s-{run.run_id}", f"c-{run.run_id}", 1),
            *(EventDraft(type=EventType.WATCHDOG, payload={}) for _ in range(100)),
        )
        _publish(tmp_path, run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("r0", runs[0].run_root, ["toy"])
    frontier = board.frontier()
    if reopen:
        board.close()
        board.open("r0", runs[0].run_root, ["toy"])
        board.frontier()
    folds: Counter[str] = Counter()
    original = RunState._apply

    def counted(state: RunState, event: Event) -> None:
        folds[event.run_id] += 1
        original(state, event)

    monkeypatch.setattr(RunState, "_apply", counted)
    try:
        frozen = board.snapshot(runs[0].goal.key(), frontier)
        expected = {"r0": 103, "r1": 103} if reopen else {}
        assert dict(folds) == expected
        for _ in range(100):
            assert len(board.best(runs[0].goal.key(), 2, judge="j", part="a")) == 2
            assert frozen._sibling_view("a")._visible_candidate_ids() == frozenset({"c-r0", "c-r1"})
        assert dict(folds) == expected
        runs[0].commit(_round(runs[0], 1, closed=True), runs[0].plan(revision=1, parts=["b"]))
        folds.clear()
        current = board.snapshot(runs[0].goal.key(), board.frontier())
        assert dict(folds) == {"r0": 2}
        assert current._sibling_view("b").part == "b"
        assert frozen._sibling_view("a")._visible_candidate_ids() == frozenset({"c-r0", "c-r1"})
        with pytest.raises(ValueError, match="part outside board snapshot"):
            current._sibling_view("a")
        for _ in range(100):
            assert board.best(runs[0].goal.key(), 2, judge="j", part="b") == []
        assert dict(folds) == {"r0": 2}
        historical = board.snapshot(runs[0].goal.key(), frontier)
        assert historical._visible_candidate_ids() == frozen._visible_candidate_ids()
        assert historical._sibling_view("a").part == "a"
        after_historical = folds.copy()
        for _ in range(100):
            assert board.snapshot(runs[0].goal.key(), frontier)._visible_candidate_ids() == (
                frozen._visible_candidate_ids()
            )
        assert folds == after_historical
    finally:
        board.close()


@pytest.mark.parametrize("reader_kind", ["view", "board"])
@pytest.mark.parametrize("shape", ["chain", "wide"])
def test_lineage_fills_frontier_once_and_batches_ids(
    tmp_path: Path,
    board: ArenaBoard,
    monkeypatch: pytest.MonkeyPatch,
    reader_kind: str,
    shape: str,
) -> None:
    from collections import Counter
    from collections.abc import Iterable, Sequence

    size = 1000
    run = FakeRun(str(tmp_path / "runs"), "r0000")
    run.commit(
        run.plan(),
        *(
            run.candidate(f"c-{i}", parents=[f"c-{i - 1}"] if i and shape == "chain" else [])
            for i in range(size - 1)
        ),
        run.candidate(
            f"c-{size - 1}",
            parents=(
                [f"c-{size - 2}"] if shape == "chain" else [f"c-{i}" for i in range(size - 1)]
            ),
        ),
    )
    _publish(tmp_path, run)
    for i in range(1, size):
        other = FakeRun(run.run_root, f"r{i:04}")
        other.commit(other.plan())
        _publish(tmp_path, other)
    view = board.snapshot(run.goal.key(), board.frontier())
    cache = view._cache if reader_kind == "view" else board._cache("toy")
    counts: Counter[str] = Counter()
    original = cache._replace_sets

    def counted(sets: Sequence[tuple[str, Iterable[tuple[str, int | None]]]]) -> None:
        materialized = [(name, list(rows)) for name, rows in sets]
        for name, rows in materialized:
            counts[name + "_rows"] += len(rows)
            counts[name + "_fills"] += 1
        original(materialized)

    monkeypatch.setattr(cache, "_replace_sets", counted)
    cache._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 64)
    reader = view if reader_kind == "view" else board
    graph = reader.lineage(f"c-{size - 1}")
    assert len(graph.nodes) == size
    assert len(graph.edges) == size - 1
    assert counts["marks_fills"] == 1
    assert counts["marks_rows"] == size
    assert counts["id_rows"] == size
    assert counts["id_fills"] == (size if shape == "chain" else 2)
    assert cache._frontier_depth == 0
    assert cache._conn.execute("SELECT count(*) FROM temp._read_sets").fetchone() == (0,)


@pytest.mark.parametrize("reader_kind", ["view", "board"])
def test_lineage_read_scope_cleans_up_after_failure(
    tmp_path: Path,
    board: ArenaBoard,
    monkeypatch: pytest.MonkeyPatch,
    reader_kind: str,
) -> None:
    from typing import Any

    from swarm_arena._board.cache import Position

    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan(), run.candidate("a"), run.candidate("b", parents=["a"]))
    _publish(tmp_path, run)
    view = board.snapshot(run.goal.key(), board.frontier())
    cache = view._cache if reader_kind == "view" else board._cache("toy")
    original = cache.candidates
    calls = 0

    def failing(*args: Any, **kwargs: Any) -> list[tuple[Candidate, Position]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected traversal failure")
        return original(*args, **kwargs)

    reader = view if reader_kind == "view" else board
    with monkeypatch.context() as patch:
        patch.setattr(cache, "candidates", failing)
        with pytest.raises(RuntimeError, match="injected traversal failure"):
            reader.lineage("b")
    assert cache._frontier_depth == 0
    assert cache._conn.execute("SELECT count(*) FROM temp._read_sets").fetchone() == (0,)
    assert reader.lineage("b").edges == [("b", "a")]


@pytest.mark.parametrize("foreign_parts", [[], ["foreign"]])
def test_owner_part_wins_foreign_main_key_collision(
    tmp_path: Path,
    board: ArenaBoard,
    foreign_parts: list[str],
) -> None:
    own = FakeRun(str(tmp_path / "runs"), "own")
    draft = own.plan(parts=["a", "b"])
    part_goal = Plan.model_validate(draft.payload["plan"]).parts[0].goal
    own.commit(
        draft,
        own.candidate("own-a", part="a"),
        own.candidate("own-b", part="b"),
        own.score("own-score", "own-a", 1),
        own.learning("own-learning", "viscosity", evidence=["own-a"]),
    )
    foreign = FakeRun(own.run_root, "foreign", goal_id=part_goal.id)
    assert foreign.goal.key() == part_goal.key()
    foreign.commit(
        foreign.plan(parts=foreign_parts),
        foreign.candidate("foreign-main"),
        foreign.learning("foreign-learning", "viscosity", evidence=["foreign-main"]),
        *(foreign.candidate(f"foreign-{part}", part=part) for part in foreign_parts),
    )
    for run in (own, foreign):
        _publish(tmp_path, run)
    board.open(own.run_id, own.run_root, ["toy"])
    frontier = board.frontier()
    direct = board.snapshot(part_goal.key(), frontier)
    sibling = board.snapshot(own.goal.key(), frontier)._sibling_view("a")
    for view in (direct, sibling):
        assert view.part == "a"
        assert view._visible_candidate_ids() == frozenset({"own-a"})
        assert [c.id for c in view.best(view.goal, 5, judge="j")] == ["own-a"]
        assert view.learning_goal == (goal_key_digest(own.goal.key()), own.goal.id)
        assert [item.id for item in view.learnings("toy", goal=view.goal)] == ["own-learning"]
        assert [item.id for item in view.search_learnings("toy", "viscosity", goal=view.goal)] == [
            "own-learning"
        ]
        assert view.count_learnings("toy", goal=view.goal) == 1
        assert view._sibling_view("b")._visible_candidate_ids() == frozenset({"own-b"})
        with pytest.raises(ValueError, match="part outside board snapshot"):
            view._sibling_view("foreign")
        with pytest.raises(KeyError):
            view.get_candidate("foreign-main")
    # Reader-only fallback retains main-first precedence, and owner-main wins too.
    for identity in ("", foreign.run_id):
        board.open(identity, foreign.run_root if identity else "", ["toy"])
        fallback = board.snapshot(part_goal.key(), frontier)
        assert fallback.part is None
        assert fallback._visible_candidate_ids() == frozenset(
            {"foreign-main", *(f"foreign-{part}" for part in foreign_parts)}
        )


def test_past_run_repeat_entries(tmp_path: Path, board: ArenaBoard) -> None:
    from doctest import script_from_examples

    from swarm_arena._board.arena import past_run_repeat_entries
    from swarm_arena.board import BoardView

    own = _seed(tmp_path, "r1")
    other = _seed(tmp_path, "r2")
    other.commit(other.candidate("keyed", repeat_key="key"))
    board.open("r1", own.run_root, ["toy"])
    view = board.snapshot(own.goal.key(), board.frontier())
    other.commit(other.candidate("future", repeat_key="future"))
    board.frontier()
    assert [(key, c.id) for key, c in past_run_repeat_entries(view, "r1")] == [("key", "keyed")]
    namespace: dict[str, object] = {}
    exec(script_from_examples(BoardView.__doc__ or ""), namespace)
    custom = namespace["view"]
    assert isinstance(custom, BoardView)
    assert past_run_repeat_entries(custom, "r1") == []


def test_repeat_hints_respect_owner_part_scope(tmp_path: Path, board: ArenaBoard) -> None:
    from swarm_arena._board.arena import past_run_repeat_entries

    own = FakeRun(str(tmp_path / "runs"), "own")
    draft = own.plan(parts=["a"])
    part_goal = Plan.model_validate(draft.payload["plan"]).parts[0].goal
    own.commit(draft)
    peer = FakeRun(own.run_root, "peer")
    peer.commit(peer.plan(parts=["a"]), peer.candidate("peer-part", part="a", repeat_key="key"))
    foreign = FakeRun(own.run_root, "foreign", goal_id=part_goal.id)
    assert foreign.goal.key() == part_goal.key()
    foreign.commit(foreign.plan(), foreign.candidate("foreign-main", repeat_key="key"))
    for run in (own, peer, foreign):
        _publish(tmp_path, run)
    board.open(own.run_id, own.run_root, ["toy"])
    view = board.snapshot(part_goal.key(), board.frontier())
    assert view._visible_candidate_ids() == frozenset({"peer-part"})
    assert [(key, c.id) for key, c in past_run_repeat_entries(view, own.run_id)] == [
        ("key", "peer-part")
    ]


def test_cross_namespace_groups_keep_candidates_scores_and_siblings(tmp_path: Path) -> None:
    from swarm_arena import PartSpec

    run = FakeRun(str(tmp_path / "runs"), "own")
    parts = [
        PartSpec(name=name, goal=Goal(id=name, namespace=f"other-{name}")) for name in ("a", "b")
    ]
    draft = run.plan(parts=["a", "b"])
    plan = Plan.model_validate(draft.payload["plan"]).model_copy(update={"parts": parts})
    run.commit(
        draft.model_copy(
            update={"payload": {**draft.payload, "plan": plan.model_dump(mode="json")}}
        ),
        run.candidate("main"),
        run.score("score-main", "main", 1),
        run.candidate("part-a", part="a"),
        run.score("score-a", "part-a", 2),
        run.candidate("part-b", part="b"),
        run.score("score-b", "part-b", 3),
    )
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open(run.run_id, run.run_root, ["toy", "other-a", "other-b"])
    try:
        frontier = board.frontier()
        main = board.snapshot(plan.goal.key(), frontier)
        assert main._visible_candidate_ids() == frozenset({"main", "part-a", "part-b"})
        assert main.scores("part-a")[0].value == 2
        for part in parts:
            direct = board.snapshot(part.goal.key(), frontier)
            sibling = main._sibling_view(part.name)
            for view in (direct, sibling):
                assert view.goal == part.goal.key()
                assert view._cache.namespace == part.goal.namespace
                assert [c.id for c in view.best(view.goal, 1, judge="j")] == [f"part-{part.name}"]
                assert view._sibling_view(None).get_candidate("main").id == "main"
                assert view._sibling_view("b").get_candidate("part-b").id == "part-b"
            assert [c.id for c, _ in board._cache(part.goal.namespace).candidates(frontier)] == [
                f"part-{part.name}"
            ]
        assert [c.id for c, _ in board._cache("toy").candidates(frontier)] == ["main"]
    finally:
        board.close()
