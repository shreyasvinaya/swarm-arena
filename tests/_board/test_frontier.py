from swarm_arena._board.frontier import build_frontier
from swarm_arena._journal.lease import Head


def test_build_frontier_sorts_marks() -> None:
    f = build_frontier(
        [
            ("r2", "/b", Head(epoch=1, seq=2, segment="journal/e1-2.jsonl")),
            ("r1", "/a", Head(epoch=2, seq=7, segment="journal/e2-7.jsonl")),
        ],
        [],
        [],
    )
    assert [m.run_id for m in f.marks] == ["r1", "r2"]
    assert f.marks[0].epoch == 2


def test_build_frontier_sorts_objects_and_copies_inputs() -> None:
    reviews, curation = ["b", "a"], ["d", "c"]
    f = build_frontier([], reviews, curation)
    reviews.clear()
    curation.clear()
    assert f.reviews == ["a", "b"]
    assert f.curation == ["c", "d"]
    assert build_frontier([], ["a", "b"], ["c", "d"]) == f
