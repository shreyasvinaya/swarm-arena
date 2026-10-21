import doctest
from abc import ABC
from collections.abc import Sequence
from types import ModuleType

import pytest

from swarm_arena import board as board_facade
from swarm_arena._board import base
from swarm_arena._board.base import Board, BoardView
from swarm_arena._model.board_types import Frontier, LineageGraph, Row
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.planning_types import Decision
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Candidate, Learning, Score


@pytest.mark.parametrize("module", [base, board_facade])
def test_in_memory_examples_are_not_skipped(module: ModuleType) -> None:
    skipped = [
        test.name
        for test in doctest.DocTestFinder().find(module)
        for example in test.examples
        if example.options.get(doctest.SKIP)
    ]
    assert skipped == []


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


def test_count_learnings_default_counts_learnings() -> None:
    namespace: dict[str, object] = {}
    exec(doctest.script_from_examples(BoardView.__doc__ or ""), namespace)
    view = namespace["view"]
    assert isinstance(view, BoardView)
    assert view.count_learnings("n") == 1
    assert view.count_learnings("n", goal=view.goal) == len(view.learnings("n", goal=view.goal))


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
    assert Board.open is Minimal.open
    assert Board.close is Minimal.close
