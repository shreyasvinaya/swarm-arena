import pytest

from swarm_arena._board.frontier import build_frontier
from swarm_arena._errors import ChainError
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


def test_build_frontier_is_order_independent() -> None:
    reads = [
        ("r2", "/b", Head(epoch=1, seq=2, segment="journal/e1-2.jsonl")),
        ("r1", "/a", Head(epoch=1, seq=3, segment="journal/e1-3.jsonl")),
    ]
    older = "z/curation/20261018T000000.000Z-00000001.jsonl"
    newer = "a/curation/20261019T000000.000Z-00000001.jsonl"
    first = build_frontier([*reads, reads[0]], [newer, older, older], [newer, older])
    second = build_frontier(list(reversed(reads)), [older, newer], [older, newer])
    assert first.digest() == second.digest()
    assert first.reviews == [older, newer]
    assert len(first.marks) == 2


def test_build_frontier_rejects_conflicting_marks() -> None:
    head = Head(epoch=1, seq=2, segment="journal/e1-2.jsonl")
    with pytest.raises(ChainError, match="conflicting"):
        build_frontier([("r1", "/a", head), ("r1", "/b", head)], [], [])
