from pathlib import Path

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Goal, PartSpec, Plan, Score
from swarm_arena._board.run_board import RunBoard
from swarm_arena._journal.event import Event
from swarm_arena._journal.lease import Head
from swarm_arena._journal.replay import RunState
from swarm_arena.board import Candidate, Frontier, Learning
from swarm_arena.planning import Decision
from swarm_arena.tool import ToolOutput, ToolRequest

KEY = Goal(id="g", namespace="n").key()
OTHER_KEY = Goal(id="g2", namespace="n").key()
PART_KEY = Goal(id="sub", namespace="n").key()
HEAD = Head(epoch=1, seq=9, segment="journal/e1-9.jsonl")


def _c(id: str, **changes: object) -> Candidate:
    return Candidate.model_validate(
        dict(
            id=id,
            run_id="run-1",
            goal_id="g",
            team="fd-solver",
            agent="tuner-1",
            round=1,
            request=ToolRequest(),
            output=ToolOutput(status="ok"),
        )
        | changes
    )


def _learning(id: str, **changes: object) -> Learning:
    return Learning.model_validate(
        dict(
            id=id,
            namespace="n",
            run_id="run-1",
            goal_id="g",
            round=1,
            author="human:alice",
            title="A hint",
            text="Try this",
        )
        | changes
    )


def _score(id: str, candidate: str, score_value: float, **changes: object) -> Score:
    return Score.model_validate(
        dict(
            id=id,
            candidate_id=candidate,
            key=(score_value,),
            judge="program",
            rubric_hash="new",
            panel_hash="panel",
        )
        | changes
    )


def _frontier(seq: int = 9, epoch: int = 1, root: str = "/runs") -> Frontier:
    return Frontier(
        marks=[
            Frontier.Mark(
                run_id="run-1",
                run_root=root,
                epoch=epoch,
                seq=seq,
                segment=f"journal/e{epoch}-{seq}.jsonl",
            )
        ]
    )


@pytest.fixture
def board() -> RunBoard:
    result = RunBoard(run_id="run-1", run_root="/runs", head=lambda: HEAD)
    result.register_goal(KEY, "g", None)
    result.register_goal(OTHER_KEY, "g2", None)
    result.register_goal(PART_KEY, "g", "p")
    return result


def test_snapshot_hides_items_after_mark(board: RunBoard) -> None:
    board.put_candidate(_c("c-1"), epoch=1, seq=4)
    board.put_candidate(_c("c-2"), epoch=1, seq=6)
    board.put_score(_score("s-1", "c-1", 1), epoch=1, seq=5)
    board.put_score(_score("s-2", "c-1", 2), epoch=1, seq=7)
    board.put_learning(_learning("L-1"), epoch=1, seq=5)
    board.put_learning(_learning("L-2"), epoch=1, seq=8)
    view = board.snapshot(KEY, _frontier(seq=5))
    assert view.get_candidate("c-1").id == "c-1"
    assert [s.id for s in view.scores("c-1")] == ["s-1"]
    assert view.get_learning("L-1").id == "L-1"
    for read, id in ((view.get_candidate, "c-2"), (view.get_learning, "L-2")):
        with pytest.raises(KeyError):
            read(id)
    with pytest.raises(KeyError):
        view.scores("c-2")
    with pytest.raises(KeyError):
        view.lineage("c-2")


def test_view_rejects_other_goal(board: RunBoard) -> None:
    board.put_candidate(_c("c-other", goal_id="g2"), epoch=1, seq=1)
    view = board.snapshot(KEY, _frontier(seq=5))
    for read in (
        lambda: view.best(OTHER_KEY, 1, judge="program"),
        lambda: view.leaderboard(OTHER_KEY, judge="program"),
        lambda: view.learnings("other"),
        lambda: view.learnings("n", goal=OTHER_KEY),
        lambda: view.search_learnings("n", "hint", goal=OTHER_KEY),
        lambda: view.get_candidate("c-other"),
        lambda: view.scores("c-other"),
        lambda: view.lineage("c-other"),
    ):
        with pytest.raises(ValueError, match="outside board snapshot"):
            read()


def _ranked_board(board: RunBoard) -> None:
    board.put_candidate(_c("c-1", parent_ids=["c-0"]), epoch=1, seq=1)
    board.put_candidate(_c("c-2"), epoch=1, seq=2)
    board.put_score(_score("s-old", "c-2", 100, rubric_hash="old"), epoch=1, seq=3)
    board.put_score(_score("s-1", "c-1", 5), epoch=1, seq=4)
    board.put_score(_score("s-2", "c-2", 2), epoch=1, seq=5)


def test_best_ranks_current_identity_only(board: RunBoard) -> None:
    _ranked_board(board)
    assert [c.id for c in board.best(KEY, 5, judge="program")] == ["c-1", "c-2"]
    assert [c.id for c in board.snapshot(KEY, _frontier(3)).best(KEY, 5, judge="program")] == [
        "c-2"
    ]
    board.put_score(_score("s-panel", "c-2", 1, panel_hash="other"), epoch=1, seq=6)
    assert [c.id for c in board.best(KEY, 5, judge="program")] == ["c-2"]
    assert board.best(KEY, 0, judge="program") == []
    assert board.best(KEY, 5, judge="absent") == []
    assert board.best(KEY, 5, judge="program", run_id="other") == []


def test_leaderboard_rows_record_identity(board: RunBoard) -> None:
    _ranked_board(board)
    rows = board.leaderboard(KEY, judge="program")
    assert [r.rank for r in rows] == [1, 2]
    assert rows[0].rubric_hash == "new"
    assert rows[0].built_on == ["c-0"]
    assert rows[0].panel_hash == "panel"
    assert rows[0].run_id == "run-1"


def test_ranking_isolated_by_group_fidelity_and_checks(board: RunBoard) -> None:
    _ranked_board(board)
    board.put_candidate(_c("c-part", part="p"), epoch=1, seq=6)
    board.put_score(_score("s-part", "c-part", 99, rubric_hash="part"), epoch=1, seq=7)
    assert [c.id for c in board.best(KEY, 5, judge="program")] == ["c-1", "c-2"]
    assert [c.id for c in board.best(PART_KEY, 5, judge="program")] == ["c-part"]
    assert board.best(KEY, 5, judge="program", part="p") == board.best(PART_KEY, 5, judge="program")
    board.put_candidate(_c("c-new", round=2, fidelity="full"), epoch=1, seq=8)
    assert board.best(KEY, 5, judge="program") == []
    assert len(board.best(KEY, 5, judge="program", fidelity="")) == 2
    board.put_score(_score("s-new", "c-new", 3), epoch=1, seq=9)
    assert [c.id for c in board.best(KEY, 5, judge="program")] == ["c-new"]
    for field in (
        {"checks": [{"check": "valid", "passed": False}]},
        {"output": ToolOutput(status="ok", partial=True)},
    ):
        board.put_candidate(_c("c-new", round=2, fidelity="full", **field), epoch=1, seq=9)
        assert board.best(KEY, 5, judge="program") == []


def test_search_is_case_insensitive_substring_newest_first(board: RunBoard) -> None:
    board.put_learning(_learning("L-1", title="Fit log(nu)", tags=["fit"]), epoch=1, seq=1)
    board.put_learning(_learning("L-2", title="LOG scale"), epoch=1, seq=2)
    board.put_learning(_learning("L-3", title="Elsewhere", text="LOG body"), epoch=1, seq=3)
    assert [x.id for x in board.search_learnings("n", "log")] == ["L-3", "L-2", "L-1"]
    assert [x.id for x in board.search_learnings("n", "log", tags=["fit"])] == ["L-1"]
    assert board.search_learnings("n", "missing") == []
    assert len(board.search_learnings("n", "LOG", limit=1)) == 1
    assert board.search_learnings("other", "log") == []


def test_learnings_relevance_order(board: RunBoard) -> None:
    for seq, (id, changes) in enumerate(
        [
            ("pinned", {"goal_id": "g2", "pinned": True}),
            ("goal", {"verified": False}),
            ("verified", {"goal_id": "g2", "verified": True}),
            ("tag", {"goal_id": "g2", "tags": ["fit"]}),
            ("new", {"goal_id": "g2"}),
            ("hidden", {"hidden": True, "pinned": True}),
        ],
        1,
    ):
        board.put_learning(_learning(id, **changes), epoch=1, seq=seq)
    assert [x.id for x in board.learnings("n", goal=KEY)] == [
        "pinned",
        "goal",
        "verified",
        "new",
        "tag",
    ]
    assert [x.id for x in board.learnings("n", tags=["fit"], goal=KEY)] == [
        "pinned",
        "goal",
        "verified",
        "tag",
        "new",
    ]
    assert board.learnings("n", limit=0) == []
    assert board.search_learnings("n", "hint", limit=0) == []
    with pytest.raises(KeyError):
        board.get_learning("hidden")


def test_view_reads_other_goals_learnings_in_namespace(board: RunBoard) -> None:
    board.put_learning(_learning("L-other", goal_id="g2"), epoch=1, seq=1)
    board.put_learning(_learning("L-ns", namespace="elsewhere"), epoch=1, seq=2)
    view = board.snapshot(KEY, _frontier())
    assert [x.id for x in view.learnings("n")] == ["L-other"]
    assert view.get_learning("L-other").goal_id == "g2"
    with pytest.raises(ValueError, match="outside board snapshot"):
        view.get_learning("L-ns")
    with pytest.raises(ValueError, match="outside board snapshot"):
        view.search_learnings("elsewhere", "hint")


def test_frontier_marks_this_run_at_current_head(board: RunBoard) -> None:
    assert board.frontier().marks == [
        Frontier.Mark(
            run_id="run-1", run_root="/runs", epoch=1, seq=9, segment="journal/e1-9.jsonl"
        )
    ]
    assert RunBoard(run_id="empty", run_root="/runs", head=lambda: None).frontier() == Frontier()


def test_snapshot_is_independent_and_checks_full_mark(board: RunBoard) -> None:
    candidate = _c("c-1", parent_ids=["parent"])
    board.put_candidate(candidate, epoch=1, seq=4)
    frontier = _frontier(5)
    view = board.snapshot(KEY, frontier)
    candidate.parent_ids.append("later")
    frontier.marks.clear()
    view.get_candidate("c-1").parent_ids.clear()
    board.put_candidate(_c("c-2"), epoch=2, seq=6)
    assert view.get_candidate("c-1").parent_ids == ["parent"]
    assert len(view.frontier.marks) == 1
    assert board.snapshot(KEY, _frontier(5)).get_candidate("c-1").id == "c-1"
    with pytest.raises(KeyError):
        board.snapshot(KEY, _frontier(5)).get_candidate("c-2")
    for f in (Frontier(), _frontier(root="/other")):
        with pytest.raises(KeyError):
            board.snapshot(KEY, f).get_candidate("c-1")
    assert board.snapshot(KEY, _frontier(8, epoch=2)).get_candidate("c-2").id == "c-2"


def test_lineage_only_traverses_visible_goal_and_terminates_cycles(board: RunBoard) -> None:
    for seq, candidate in enumerate(
        [
            _c("c-1", parent_ids=["c-2", "c-missing", "c-other", "c-future"]),
            _c("c-2", parent_ids=["c-1"]),
            _c("c-other", goal_id="g2"),
            _c("c-future"),
        ],
        1,
    ):
        board.put_candidate(candidate, epoch=1, seq=seq)
    graph = board.snapshot(KEY, _frontier(3)).lineage("c-1")
    assert set(graph.nodes) == {"c-1", "c-2"}
    assert graph.edges == [("c-1", "c-2"), ("c-2", "c-1")]


def test_from_state_equals_fed_board(tmp_path: Path) -> None:
    plan = Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[],
        judges=[],
        parts=[PartSpec(name="p", goal=Goal(id="sub", namespace="n"))],
    )
    payloads = [
        ("plan_revision", {"plan": plan.model_dump(mode="json"), "origin": "human"}),
        ("tool_end", {"candidate": _c("c-1").model_dump(mode="json")}),
        ("tool_end", {"candidate": _c("c-part", part="p").model_dump(mode="json")}),
        ("score", {"score": _score("s-1", "c-1", 2).model_dump(mode="json")}),
        ("score", {"score": _score("s-2", "c-part", 3).model_dump(mode="json")}),
        ("learning", {"learning": _learning("L-1").model_dump(mode="json")}),
        ("decision", {"round": 1, "decision": Decision(policy="keep").model_dump(mode="json")}),
    ]
    events = [
        Event(
            seq=i,
            ts="2026-10-20T00:00:00Z",
            epoch=1,
            run_id="run-1",
            context_id="run-1",
            task_id="g",
            goal_id="g",
            type=kind,
            payload=payload,
        )
        for i, (kind, payload) in enumerate(payloads, 1)
    ]
    state = RunState.from_events(events)
    root = str(tmp_path)
    rebuilt = RunBoard.from_state(state, run_root=root, head=lambda: HEAD)
    fed = RunBoard(run_id="run-1", run_root=root, head=lambda: HEAD)
    fed.register_goal(KEY, "g", None)
    fed.register_goal(PART_KEY, "g", "p")
    for candidate in state.candidates.values():
        epoch, seq = state.positions[candidate.id]
        fed.put_candidate(candidate, epoch=epoch, seq=seq)
    for score in state.scores.values():
        epoch, seq = state.positions[score.id]
        fed.put_score(score, epoch=epoch, seq=seq)
    for learning in state.learnings.values():
        epoch, seq = state.positions[learning.id]
        fed.put_learning(learning, epoch=epoch, seq=seq)
    fed.put_decision("g", 1, Decision(policy="keep"), epoch=1, seq=7)
    for key in (KEY, PART_KEY):
        for seq in (2, 4, 9):
            left, right = [b.snapshot(key, _frontier(seq, root=root)) for b in (fed, rebuilt)]
            assert left.best(key, 10, judge="program") == right.best(key, 10, judge="program")
            assert left.leaderboard(key, judge="program") == right.leaderboard(key, judge="program")
            assert left.learnings("n") == right.learnings("n")
    assert rebuilt.config.model_dump() == {}


def test_unknown_goal_errors_hide_values(board: RunBoard) -> None:
    secret_key = Goal(id="g", namespace="n", inputs={"SECRET_SENTINEL": "SECRET_SENTINEL"}).key()
    for read in (
        lambda: board.best(secret_key, 1, judge="program"),
        lambda: board.snapshot(secret_key, _frontier()),
    ):
        with pytest.raises(ValueError, match="unregistered board goal") as caught:
            read()
        assert_secret_absent(caught.value, "SECRET_SENTINEL")


def test_from_state_uses_latest_revision_and_recorded_score_order() -> None:
    old_part = Goal(id="sub", namespace="n", inputs={"version": "old"})
    new_part = Goal(id="sub", namespace="n", inputs={"version": "new"})
    plans = [
        Plan(
            goal=Goal(id="g", namespace="n"),
            teams=[],
            judges=[],
            parts=[PartSpec(name="p", goal=part)],
        )
        for part in (old_part, new_part)
    ]
    state = RunState(
        run_id="run-1",
        plans={"g": [(p, "human") for p in plans]},
        candidates={"c-1": _c("c-1")},
        scores={
            "s-new": _score("s-new", "c-1", 1),
            "s-old": _score("s-old", "c-1", 100, rubric_hash="old"),
        },
        positions={"c-1": (1, 1), "s-old": (1, 2), "s-new": (2, 3)},
    )
    board = RunBoard.from_state(state, run_root="/runs", head=lambda: HEAD)
    assert board.snapshot(new_part.key(), _frontier()).goal == new_part.key()
    with pytest.raises(ValueError, match="unregistered board goal"):
        board.snapshot(old_part.key(), _frontier())
    assert board.leaderboard(KEY, judge="program")[0].rubric_hash == "new"
    old_view = board.snapshot(KEY, _frontier(2))
    assert old_view.leaderboard(KEY, judge="program")[0].rubric_hash == "old"
    state.candidates["c-1"].parent_ids.append("mutated")
    assert board.get_candidate("c-1").parent_ids == []
