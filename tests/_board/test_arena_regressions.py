"""Committed-journal equivalents of the former default-board regressions."""

from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest
from _fakes.board import GOAL, KEY, OTHER, OTHER_KEY, PART, PART_KEY, records
from _fakes.board import candidate as _c
from _fakes.board import learning as _learning
from _fakes.board import score as _score
from _fakes.journals import FakeRun
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Goal, PartSpec, Plan
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.frontier import build_frontier
from swarm_arena._engine.goal import r0_revision_draft
from swarm_arena._errors import ChainError
from swarm_arena._journal.event import EventDraft
from swarm_arena.board import Frontier, GoalKey


@pytest.fixture
def committed(tmp_path: Path) -> Iterator[tuple[ArenaBoard, FakeRun]]:
    run = FakeRun(str(tmp_path / "runs"), "run-1", goal=GOAL)
    plan = Plan(goal=GOAL, teams=[], judges=[], parts=[PartSpec(name="p", goal=PART)])
    other = Plan(goal=OTHER, teams=[], judges=[])
    run.commit(r0_revision_draft(plan, plan.base_dir), r0_revision_draft(other, other.base_dir))
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open(run.run_id, run.run_root, ["n"])
    yield board, run
    board.close()


def test_snapshot_hides_items_after_mark(committed: tuple[ArenaBoard, FakeRun]) -> None:
    board, run = committed
    run.commit(*records(_c("c-1"), _score("s-1", "c-1", 1), _learning("L-1")))
    view = board.snapshot(KEY, board.frontier())
    run.commit(*records(_c("c-2"), _score("s-2", "c-1", 2), _learning("L-2")))
    board.frontier()
    assert view.get_candidate("c-1").id == "c-1"
    assert [s.id for s in view.scores("c-1")] == ["s-1"]
    assert view.get_learning("L-1").id == "L-1"
    for read, id in (
        (view.get_candidate, "c-2"),
        (view.get_learning, "L-2"),
        (view.scores, "c-2"),
        (view.lineage, "c-2"),
    ):
        with pytest.raises(KeyError):
            read(id)


@pytest.mark.parametrize(
    ("goal", "expected"), [(KEY, {"c-main", "c-part"}), (PART_KEY, {"c-part"})]
)
def test_visible_candidate_ids_are_frozen_and_goal_scoped(
    committed: tuple[ArenaBoard, FakeRun], goal: GoalKey, expected: set[str]
) -> None:
    board, run = committed
    run.commit(*records(_c("c-main"), _c("c-part", part="p"), _c("c-other", goal_id="g2")))
    view = board.snapshot(goal, board.frontier())
    assert view._visible_candidate_ids() == frozenset(expected)
    run.commit(*records(_c("later")))
    board.put_candidate(_c("c-main", goal_id="g2"), epoch=1, seq=999)
    board.frontier()
    assert view._visible_candidate_ids() == frozenset(expected)
    assert board.snapshot(goal, Frontier())._visible_candidate_ids() == frozenset()


def test_view_rejects_other_goal(committed: tuple[ArenaBoard, FakeRun]) -> None:
    board, run = committed
    run.commit(*records(_c("other", goal_id="g2")))
    view = board.snapshot(KEY, board.frontier())
    for read in (
        lambda: view.best(OTHER_KEY, 1, judge="program"),
        lambda: view.leaderboard(OTHER_KEY, judge="program"),
        lambda: view.learnings("other"),
        lambda: view.learnings("n", goal=OTHER_KEY),
        lambda: view.search_learnings("n", "hint", goal=OTHER_KEY),
    ):
        with pytest.raises(ValueError, match="outside"):
            read()
    for read in (view.get_candidate, view.scores, view.lineage):
        with pytest.raises(KeyError):
            read("other")


def _ranked(run: FakeRun) -> None:
    run.commit(
        *records(
            _c("c-1", parent_ids=["c-0"]), _c("c-2"), _score("old", "c-2", 100, rubric_hash="old")
        )
    )
    run.commit(*records(_score("s-1", "c-1", 5), _score("s-2", "c-2", 2)))


def test_best_ranks_current_identity_only(committed: tuple[ArenaBoard, FakeRun]) -> None:
    board, run = committed
    _ranked(run)
    board.frontier()
    assert [c.id for c in board.best(KEY, 5, judge="program")] == ["c-1", "c-2"]
    old = build_frontier([(run.run_id, run.run_root, run.heads[-2])], [], [])
    assert [c.id for c in board.snapshot(KEY, old).best(KEY, 5, judge="program")] == ["c-2"]
    run.commit(*records(_score("panel", "c-2", 1, panel_hash="other")))
    board.frontier()
    assert [c.id for c in board.best(KEY, 5, judge="program")] == ["c-2"]
    assert board.best(KEY, 0, judge="program") == []
    assert board.best(KEY, 5, judge="absent") == []
    assert board.best(KEY, 5, judge="program", run_id="other") == []


def test_leaderboard_rows_record_identity(committed: tuple[ArenaBoard, FakeRun]) -> None:
    board, run = committed
    _ranked(run)
    board.frontier()
    rows = board.leaderboard(KEY, judge="program")
    assert [r.rank for r in rows] == [1, 2]
    assert (rows[0].rubric_hash, rows[0].panel_hash, rows[0].run_id) == ("new", "panel", "run-1")
    assert rows[0].built_on == ["c-0"]


@pytest.mark.parametrize("blocked", ["none", "gated", "partial"])
def test_ranking_isolated_by_group_fidelity_and_checks(
    committed: tuple[ArenaBoard, FakeRun], blocked: str
) -> None:
    board, run = committed
    _ranked(run)
    run.commit(*records(_c("part", part="p"), _score("part-score", "part", 99, rubric_hash="part")))
    board.frontier()
    assert [c.id for c in board.best(KEY, 5, judge="program")] == ["c-1", "c-2"]
    assert [c.id for c in board.best(PART_KEY, 5, judge="program")] == ["part"]
    assert board.best(KEY, 5, judge="program", part="p") == board.best(PART_KEY, 5, judge="program")
    changes = (
        {"checks": [{"check": "valid", "passed": False}]}
        if blocked == "gated"
        else {"output": {"status": "ok", "partial": True}}
        if blocked == "partial"
        else {}
    )
    run.commit(*records(_c("new", round=2, fidelity="full", **changes)))
    board.frontier()
    assert board.best(KEY, 5, judge="program") == []
    assert len(board.best(KEY, 5, judge="program", fidelity="")) == 2
    run.commit(*records(_score("new-score", "new", 3)))
    board.frontier()
    assert [c.id for c in board.best(KEY, 5, judge="program")] == (
        [] if blocked != "none" else ["new"]
    )


def test_search_is_case_insensitive_and_filters_tags(committed: tuple[ArenaBoard, FakeRun]) -> None:
    board, run = committed
    run.commit(
        *records(
            _learning("one", title="Fit log(nu)", tags=["fit"]),
            _learning("two", title="LOG scale"),
            _learning("three", title="Elsewhere", text="LOG body"),
        )
    )
    board.frontier()
    assert {x.id for x in board.search_learnings("n", "log")} == {"one", "two", "three"}
    assert [x.id for x in board.search_learnings("n", "log", tags=["fit"])] == ["one"]
    assert board.search_learnings("n", "missing") == []
    assert len(board.search_learnings("n", "LOG", limit=1)) == 1
    assert board.search_learnings("other", "log") == []


@pytest.mark.parametrize("snapshot", [False, True])
def test_search_filters_goal_before_limit(
    committed: tuple[ArenaBoard, FakeRun], snapshot: bool
) -> None:
    board, run = committed
    run.commit(
        *records(
            _learning("own", title="Baseline"), _learning("other", goal_id="g2", title="Baseline")
        )
    )
    frontier = board.frontier()
    reader = board.snapshot(KEY, frontier) if snapshot else board
    assert [x.id for x in reader.search_learnings("n", "baseline", goal=KEY, limit=1)] == ["own"]
    assert len(reader.search_learnings("n", "baseline", limit=1)) == 1


async def test_worker_snapshot_interleaves_loop_writes(
    committed: tuple[ArenaBoard, FakeRun],
) -> None:
    board, run = committed
    run.commit(
        *records(_learning("one"), _c("c-1", parent_ids=["original"]), _score("s-1", "c-1", 1))
    )
    frontier = board.frontier()
    views = []

    async def snapshot() -> None:
        views.append(await anyio.to_thread.run_sync(board.snapshot, KEY, frontier))

    async with anyio.create_task_group() as group:
        for _ in range(8):
            group.start_soon(snapshot)
        board.put_candidate(_c("c-1", parent_ids=["replacement"]), epoch=1, seq=999)
        board.put_learning(_learning("two"), epoch=1, seq=1000)
    for view in views:
        assert [x.id for x in view.learnings("n")] == ["one"]
        assert view.get_candidate("c-1").parent_ids == ["original"]
        assert view.scores("c-1")[0].key == (1.0,)


def test_learnings_relevance_order(committed: tuple[ArenaBoard, FakeRun]) -> None:
    board, run = committed
    run.commit(
        *records(
            *[
                _learning(id, **changes)
                for id, changes in [
                    ("pinned", {"goal_id": "g2", "pinned": True}),
                    ("goal", {"verified": False}),
                    ("verified", {"goal_id": "g2", "verified": True}),
                    ("tag", {"goal_id": "g2", "tags": ["fit"]}),
                    ("new", {"goal_id": "g2"}),
                    ("hidden", {"hidden": True, "pinned": True}),
                ]
            ]
        )
    )
    from _fakes.journals import START

    from swarm_arena._board.human_objects import HumanObject, write_human_objects
    from swarm_arena._store.base import open_store

    write_human_objects(
        open_store(board.config.arena_root),
        "n",
        "curation",
        [
            HumanObject(kind="pin", learning_id="pinned", author="human:a"),
            HumanObject(kind="hide", learning_id="hidden", author="human:a"),
        ],
        now=START,
    )
    board.frontier()
    view = board.snapshot(KEY, board.frontier())
    assert [x.id for x in view.learnings("n")] == ["pinned", "goal", "verified", "new", "tag"]
    assert [x.id for x in view.learnings("n", goal=KEY)] == ["goal"]
    assert [x.id for x in view.learnings("n", tags=["fit"])] == ["tag"]
    assert board.learnings("n", limit=0) == []
    assert board.search_learnings("n", "hint", limit=0) == []
    with pytest.raises(KeyError):
        board.get_learning("hidden")


def test_view_reads_other_goals_learnings_in_namespace(
    committed: tuple[ArenaBoard, FakeRun],
) -> None:
    board, run = committed
    run.commit(*records(_learning("other", goal_id="g2")))
    view = board.snapshot(KEY, board.frontier())
    assert [x.id for x in view.learnings("n")] == ["other"]
    assert view.get_learning("other").goal_id == "g2"
    with pytest.raises(ValueError, match="outside"):
        view.search_learnings("elsewhere", "hint")


def test_frontier_marks_this_run_at_current_head(committed: tuple[ArenaBoard, FakeRun]) -> None:
    board, run = committed
    assert board.frontier() == build_frontier([(run.run_id, run.run_root, run.heads[-1])], [], [])


def test_snapshot_is_independent_and_checks_full_mark(
    committed: tuple[ArenaBoard, FakeRun],
) -> None:
    board, run = committed
    candidate = _c("one", parent_ids=["parent"])
    run.commit(*records(candidate))
    frontier = board.frontier()
    view = board.snapshot(KEY, frontier)
    candidate.parent_ids.append("later")
    frontier.marks.clear()
    view.get_candidate("one").parent_ids.clear()
    run.commit(*records(_c("two")))
    assert view.get_candidate("one").parent_ids == ["parent"]
    assert len(view.frontier.marks) == 1
    with pytest.raises(KeyError):
        view.get_candidate("two")
    corrupt = view.frontier.model_copy(deep=True)
    corrupt.marks[0] = corrupt.marks[0].model_copy(update={"seq": corrupt.marks[0].seq + 1})
    with pytest.raises(ChainError):
        board.snapshot(KEY, corrupt)


def test_lineage_only_traverses_visible_goal_and_terminates_cycles(
    committed: tuple[ArenaBoard, FakeRun],
) -> None:
    board, run = committed
    run.commit(
        *records(
            _c("one", parent_ids=["two", "missing", "other", "future"]),
            _c("two", parent_ids=["one"]),
            _c("other", goal_id="g2"),
        )
    )
    view = board.snapshot(KEY, board.frontier())
    run.commit(*records(_c("future")))
    graph = view.lineage("one")
    assert set(graph.nodes) == {"one", "two"}
    assert graph.edges == [("one", "two"), ("two", "one")]


def test_reopened_cache_equals_ingested_board(
    committed: tuple[ArenaBoard, FakeRun], tmp_path: Path
) -> None:
    board, run = committed
    _ranked(run)
    run.commit(*records(_c("part", part="p"), _score("part-score", "part", 3), _learning("hint")))
    frontier = board.frontier()
    fresh = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "rebuilt"))
    fresh.open(run.run_id, run.run_root, ["n"])
    try:
        for key in (KEY, PART_KEY):
            left, right = [b.snapshot(key, frontier) for b in (board, fresh)]
            assert left.best(key, 10, judge="program") == right.best(key, 10, judge="program")
            assert left.leaderboard(key, judge="program") == right.leaderboard(key, judge="program")
            assert left.learnings("n") == right.learnings("n")
    finally:
        fresh.close()


def test_unknown_goal_errors_hide_values(committed: tuple[ArenaBoard, FakeRun]) -> None:
    board, _ = committed
    key = Goal(id="g", namespace="n", inputs={"SECRET_SENTINEL": "SECRET_SENTINEL"}).key()
    view = board.snapshot(key, board.frontier())
    assert view._visible_candidate_ids() == frozenset()
    with pytest.raises(ValueError, match="outside") as caught:
        view.best(KEY, 1, judge="program")
    assert_secret_absent(caught.value, "SECRET_SENTINEL")


def test_latest_revision_and_recorded_score_order(committed: tuple[ArenaBoard, FakeRun]) -> None:
    board, run = committed
    _ranked(run)
    old_frontier = build_frontier([(run.run_id, run.run_root, run.heads[-2])], [], [])
    assert (
        board.snapshot(KEY, old_frontier).leaderboard(KEY, judge="program")[0].rubric_hash == "old"
    )
    assert (
        board.snapshot(KEY, board.frontier()).leaderboard(KEY, judge="program")[0].rubric_hash
        == "new"
    )


@pytest.mark.parametrize("manifest", [False, True])
def test_part_identity_change_preserves_history_but_not_current_parents(
    committed: tuple[ArenaBoard, FakeRun], manifest: bool
) -> None:
    from swarm_arena._model.goal import FileSet

    board, run = committed
    old = _c("old", part="p")
    run.commit(*records(old, _score("old-score", "old", 100)))
    frozen = board.snapshot(PART_KEY, board.frontier())
    current = PART.model_copy(
        update={"files": FileSet(manifest={"x": "a" * 64})}
        if manifest
        else {"inputs": {"version": 2}}
    )
    plan = Plan(
        goal=GOAL, teams=[], judges=[], revision=1, parts=[PartSpec(name="p", goal=current)]
    )
    run.commit(
        EventDraft(
            type="plan_revision",
            goal_id="g",
            payload={
                "plan": plan.model_dump(mode="json"),
                "origin": "human",
                "first_round": 1,
                "preflight_complete": True,
            },
        )
    )
    empty = board.snapshot(current.key(), board.frontier())
    assert empty.best(current.key(), 1, judge="program") == []
    assert empty._visible_candidate_ids() == frozenset()
    with pytest.raises(KeyError):
        empty.get_candidate("old")
    run.commit(*records(_c("new", part="p", parent_ids=["old"]), _score("new-score", "new", 1)))
    view = board.snapshot(current.key(), board.frontier())
    assert [c.id for c in view.best(current.key(), 5, judge="program")] == ["new"]
    assert view._visible_candidate_ids() == frozenset({"new"})
    assert set(view.lineage("new").nodes) == {"new"}
    assert frozen.get_candidate("old") == old
    historical = board.snapshot(PART_KEY, board.frontier())
    assert historical.get_candidate("old") == old
