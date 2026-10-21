from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from _fakes.board import KEY, seed_board
from _fakes.board import candidate as _c
from _fakes.board import learning as _learning
from _fakes.board import score as _score

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.digest import build_digest, estimate_tokens
from swarm_arena.board import BoardView


@pytest.fixture
def make_board(tmp_path: Path) -> Iterator[Callable[..., ArenaBoard]]:
    boards: list[ArenaBoard] = []

    def make(**items: Any) -> ArenaBoard:
        board = seed_board(tmp_path / str(len(boards)), **items)
        boards.append(board)
        return board

    yield make
    for board in boards:
        board.close()


def _view(make_board: Callable[..., ArenaBoard], count: int, title: str = "日" * 120) -> BoardView:
    board = make_board(
        candidates=[_c("c-1", parent_ids=["c-0"])],
        scores=[_score("s-1", "c-1", 5, value=5)],
        learnings=[
            _learning(f"L-{i:04d}", title=title, pinned=i == 0, round=2 if i >= count - 37 else 1)
            for i in range(count)
        ],
    )
    return board.snapshot(KEY, board.frontier())


@pytest.fixture
def view(make_board: Callable[..., ArenaBoard]) -> BoardView:
    return _view(make_board, 415)


@pytest.fixture
def big_view(make_board: Callable[..., ArenaBoard]) -> BoardView:
    return _view(make_board, 5000, "A short hint")


@pytest.fixture
def non_ascii_view(make_board: Callable[..., ArenaBoard]) -> BoardView:
    return _view(make_board, 5000, "日本🙂e\u0301" * 20)


def test_digest_sections_and_count_line(view: BoardView) -> None:
    d = build_digest(
        view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=1500, k=5
    )
    lines = d.splitlines()
    assert lines[0] == "Leaderboard (program, top 5):"
    assert lines[1].startswith("1. c-")
    assert "fd-solver/tuner-1" in lines[1]
    assert "built on" in lines[1]
    assert "Pinned:" in d
    assert d.index("Pinned:") < d.index("Newest:")
    assert lines[-1] == "412 more learnings, 37 new since the last round. Use search_learnings."


def test_digest_respects_token_cap_for_any_board_size(big_view: BoardView) -> None:
    assert (
        estimate_tokens(
            build_digest(
                big_view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=1500
            )
        )
        <= 1500
    )


def test_digest_cap_holds_for_non_ascii_titles(non_ascii_view: BoardView) -> None:
    d = build_digest(
        non_ascii_view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=1500
    )
    assert len(d.encode("utf-8")) <= 1500
    assert d.splitlines()[0].startswith("Leaderboard")


def test_empty_board_digest(make_board: Callable[..., ArenaBoard]) -> None:
    board = make_board()
    assert (
        build_digest(
            board.snapshot(KEY, board.frontier()),
            goal=KEY,
            run_id="run-1",
            round=3,
            ranking="program",
            cap_tokens=1500,
        )
        == "No results yet. 0 learnings."
    )


def test_estimate_tokens() -> None:
    assert [estimate_tokens(t) for t in ("", "abcde", "é", "日本", "🙂")] == [0, 5, 2, 6, 4]
    with pytest.raises(UnicodeEncodeError):
        estimate_tokens("\ud800")


@pytest.mark.parametrize("cap", [0, 1, 7, 21, 60, 100, 200, 500, 1000, 1500])
def test_tiny_caps_and_count_line_survive(view: BoardView, cap: int) -> None:
    d = build_digest(view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=cap)
    assert len(d.encode("utf-8")) <= cap
    if cap < 60:
        assert "415 more learnings, 37 new since the last round. Use search_learnings.".startswith(
            d
        )
    elif cap >= 100:
        assert d.endswith("new since the last round. Use search_learnings.")


def test_valid_unicode_titles_unchanged_and_hidden_omitted(
    make_board: Callable[..., ArenaBoard],
) -> None:
    title = "日本🙂e\u0301"
    view = _view(make_board, 1, title)
    d = build_digest(view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=1500)
    assert title in d
    assert "unverified" not in d
    board = make_board(learnings=[_learning("L-hidden", hidden=True, title="SECRET_SENTINEL")])
    assert (
        build_digest(
            board.snapshot(KEY, board.frontier()),
            goal=KEY,
            run_id="run-1",
            round=3,
            ranking="program",
            cap_tokens=1500,
        )
        == "No results yet. 0 learnings."
    )


def test_pruning_drops_newest_then_pinned_then_extra_rows(
    make_board: Callable[..., ArenaBoard],
) -> None:
    board = make_board(
        candidates=[_c(f"c-{i}") for i in range(3)],
        scores=[_score(f"s-{i}", f"c-{i}", float(3 - i)) for i in range(3)],
        learnings=[_learning("pinned", pinned=True), _learning("newest", title="新" * 120)],
    )
    view = board.snapshot(KEY, board.frontier())
    d = build_digest(view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=400)
    assert "pinned A hint" in d
    assert "- newest " not in d
    assert "3. c-2" in d
    d = build_digest(view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=200)
    assert "1. c-0" in d
    assert "2. c-1" not in d
    assert "- pinned " not in d


def test_negative_cap_rejected_and_empty_digest_clipped(
    make_board: Callable[..., ArenaBoard],
) -> None:
    board = make_board()
    view = board.snapshot(KEY, board.frontier())
    with pytest.raises(ValueError, match="byte cap must be nonnegative"):
        build_digest(view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=-1)
    assert (
        build_digest(view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=4)
        == "No r"
    )


@pytest.mark.parametrize(("round", "expected"), [(2, 2), (3, 1), (4, 0)])
def test_digest_counts_only_previous_round_of_view_run(
    round: int, expected: int, make_board: Callable[..., ArenaBoard]
) -> None:
    items = [
        _learning("own-r1", round=1),
        _learning("other-r1", goal_id="g2", round=1),
        _learning("own-r2", round=2),
        _learning("other-high", goal_id="g2", round=90),
        _learning("old-high", run_id="old-run", round=100),
        _learning("old-r1", run_id="old-run", round=1),
        _learning("old-r2", run_id="old-run", round=2),
        _learning("hidden-r1", hidden=True, round=1),
        _learning("outside-namespace", namespace="elsewhere", round=1),
    ]
    board = make_board(learnings=items)
    view = board.snapshot(KEY, board.frontier())
    board.put_learning(_learning("after-frontier"), epoch=1, seq=999)
    digest = build_digest(
        view, goal=KEY, run_id="run-1", round=round, ranking="program", cap_tokens=1500
    )
    assert digest.splitlines()[-1] == (
        f"0 more learnings, {expected} new since the last round. Use search_learnings."
    )
    assert "other-high" in digest
    assert "old-high" in digest
    assert "hidden-r1" not in digest
    assert "outside-namespace" not in digest
    assert "after-frontier" not in digest
    assert len(digest.encode("utf-8")) <= 1500


@pytest.mark.parametrize("cap", [*range(201), 1500])
def test_first_round_omits_recent_count_and_keeps_cap(
    cap: int, make_board: Callable[..., ArenaBoard]
) -> None:
    board = make_board(
        learnings=[
            _learning("initial", round=0, title="日本🙂"),
            _learning("older", run_id="old-run", round=99),
        ]
    )
    view = board.snapshot(KEY, board.frontier())
    digest = build_digest(
        view, goal=KEY, run_id="run-1", round=1, ranking="program", cap_tokens=cap
    )
    assert "since the last round" not in digest
    assert "from this round" not in digest
    assert len(digest.encode("utf-8")) <= cap
    if cap == 1500:
        assert digest.splitlines()[-1] == "0 more learnings. Use search_learnings."
    elif cap == 50:
        assert digest == "2 more learnings. Use search_learnings."


def test_digest_owner_without_a_mark_has_no_previous_round_count(
    make_board: Callable[..., ArenaBoard],
) -> None:
    board = make_board(learnings=[_learning("other-run-round", round=1)])
    view = board.snapshot(KEY, board.frontier())
    digest = build_digest(
        view, goal=KEY, run_id="unmarked", round=2, ranking="program", cap_tokens=1500
    )
    assert digest.endswith("0 more learnings, 0 new since the last round. Use search_learnings.")
