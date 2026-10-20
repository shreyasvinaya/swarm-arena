from abc import ABC
from collections.abc import Sequence

import pytest

from swarm_arena._board.base import Board, BoardView
from swarm_arena._board.run_board import RunBoard
from swarm_arena._model.board_types import Frontier, LineageGraph, Row
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.planning_types import Decision
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Candidate, Learning, Score


def test_abstract_methods_are_required() -> None:
    assert issubclass(Board, Configurable)
    assert issubclass(BoardView, ABC)
    with pytest.raises(TypeError):
        Board()  # ty: ignore[call-non-callable]
    with pytest.raises(TypeError):
        BoardView()  # ty: ignore[call-non-callable]
    reads = {
        "best",
        "learnings",
        "search_learnings",
        "get_learning",
        "get_candidate",
        "scores",
        "lineage",
        "leaderboard",
    }
    assert reads <= Board.__abstractmethods__
    assert reads == BoardView.__abstractmethods__
    assert {
        "put_candidate",
        "put_score",
        "put_learning",
        "put_decision",
        "frontier",
        "snapshot",
    } <= Board.__abstractmethods__


def test_open_and_close_default_to_no_ops() -> None:
    class Minimal(Board):
        def put_candidate(self, candidate: Candidate, *, epoch: int, seq: int) -> None:
            pass

        def put_score(self, score: Score, *, epoch: int, seq: int) -> None:
            pass

        def put_learning(self, learning: Learning, *, epoch: int, seq: int) -> None:
            pass

        def put_decision(
            self, goal_id: str, round: int, decision: Decision, *, epoch: int, seq: int
        ) -> None:
            pass

        def frontier(self) -> Frontier:
            return Frontier()

        def snapshot(self, goal: GoalKey, frontier: Frontier) -> BoardView:
            raise NotImplementedError

        def best(
            self,
            goal: GoalKey,
            k: int,
            *,
            judge: str,
            run_id: str | None = None,
            part: str | None = None,
            fidelity: str | None = None,
        ) -> list[Candidate]:
            return []

        def learnings(
            self,
            namespace: str,
            *,
            tags: Sequence[str] = (),
            goal: GoalKey | None = None,
            limit: int = 50,
        ) -> list[Learning]:
            return []

        def search_learnings(
            self,
            namespace: str,
            query: str,
            *,
            tags: Sequence[str] = (),
            goal: GoalKey | None = None,
            limit: int = 10,
        ) -> list[Learning]:
            return []

        def get_learning(self, id: str) -> Learning:
            raise KeyError(id)

        def get_candidate(self, id: str) -> Candidate:
            raise KeyError(id)

        def scores(self, candidate_id: str) -> list[Score]:
            return []

        def lineage(self, candidate_id: str) -> LineageGraph:
            return LineageGraph(root=candidate_id, nodes={}, edges=[])

        def leaderboard(self, goal: GoalKey, *, judge: str) -> list[Row]:
            return []

    board = Minimal()
    assert board.open("r", "/runs", ["n"]) is None
    assert board.close() is None
    assert board.config.model_dump() == {}
    assert Board.open is RunBoard.open
    assert Board.close is RunBoard.close
