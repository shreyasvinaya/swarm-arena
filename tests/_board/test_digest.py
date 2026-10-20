import pytest
from _board.test_run_board import HEAD, KEY, _c, _frontier, _learning, _score

from swarm_arena._board.digest import build_digest, estimate_tokens
from swarm_arena._board.run_board import RunBoard
from swarm_arena.board import BoardView


def _empty_view() -> BoardView:
    board = RunBoard(run_id="run-1", run_root="/runs", head=lambda: HEAD)
    board.register_goal(KEY, "g", None)
    return board.snapshot(KEY, _frontier(10000))


def _view(count: int, title: str = "日" * 120) -> BoardView:
    board = RunBoard(run_id="run-1", run_root="/runs", head=lambda: HEAD)
    board.register_goal(KEY, "g", None)
    board.put_candidate(_c("c-1", parent_ids=["c-0"]), epoch=1, seq=1)
    board.put_score(_score("s-1", "c-1", 5, value=5), epoch=1, seq=2)
    for i in range(count):
        board.put_learning(
            _learning(f"L-{i:04d}", title=title, pinned=i == 0, round=2 if i >= count - 37 else 1),
            epoch=1,
            seq=i + 3,
        )
    return board.snapshot(KEY, _frontier(10000))


@pytest.fixture
def view() -> BoardView:
    return _view(415)


@pytest.fixture
def big_view() -> BoardView:
    return _view(5000, "A short hint")


@pytest.fixture
def non_ascii_view() -> BoardView:
    return _view(5000, "日本🙂e\u0301" * 20)


def test_digest_sections_and_count_line(view: BoardView) -> None:
    d = build_digest(view, goal=KEY, ranking="program", cap_tokens=1500, k=5)
    lines = d.splitlines()
    assert lines[0] == "Leaderboard (program, top 5):"
    assert lines[1].startswith("1. c-")
    assert "fd-solver/tuner-1" in lines[1]
    assert "built on" in lines[1]
    assert "Pinned:" in d
    assert d.index("Pinned:") < d.index("Newest:")
    assert lines[-1] == "412 more learnings, 37 from this round. Use search_learnings."


def test_digest_respects_token_cap_for_any_board_size(big_view: BoardView) -> None:
    assert (
        estimate_tokens(build_digest(big_view, goal=KEY, ranking="program", cap_tokens=1500))
        <= 1500
    )


def test_digest_cap_holds_for_non_ascii_titles(non_ascii_view: BoardView) -> None:
    d = build_digest(non_ascii_view, goal=KEY, ranking="program", cap_tokens=1500)
    assert len(d.encode("utf-8")) <= 1500
    assert d.splitlines()[0].startswith("Leaderboard")


def test_empty_board_digest() -> None:
    assert (
        build_digest(_empty_view(), goal=KEY, ranking="program", cap_tokens=1500)
        == "No results yet. 0 learnings."
    )


def test_estimate_tokens() -> None:
    assert [estimate_tokens(t) for t in ("", "abcde", "é", "日本", "🙂")] == [0, 5, 2, 6, 4]
    with pytest.raises(UnicodeEncodeError):
        estimate_tokens("\ud800")


@pytest.mark.parametrize("cap", [0, 1, 7, 21, 60, 100, 200, 500, 1000, 1500])
def test_tiny_caps_and_count_line_survive(view: BoardView, cap: int) -> None:
    d = build_digest(view, goal=KEY, ranking="program", cap_tokens=cap)
    assert len(d.encode("utf-8")) <= cap
    if cap < 60:
        assert "415 more learnings, 37 from this round. Use search_learnings.".startswith(d)
    elif cap >= 100:
        assert d.endswith("from this round. Use search_learnings.")


def test_valid_unicode_titles_unchanged_and_hidden_omitted() -> None:
    title = "日本🙂e\u0301"
    view = _view(1, title)
    d = build_digest(view, goal=KEY, ranking="program", cap_tokens=1500)
    assert title in d
    assert "unverified" not in d
    board = RunBoard(run_id="run-1", run_root="/runs", head=lambda: HEAD)
    board.register_goal(KEY, "g", None)
    board.put_learning(_learning("L-hidden", hidden=True, title="SECRET_SENTINEL"), epoch=1, seq=1)
    assert (
        build_digest(board.snapshot(KEY, _frontier()), goal=KEY, ranking="program", cap_tokens=1500)
        == "No results yet. 0 learnings."
    )


def test_pruning_drops_newest_then_pinned_then_extra_rows() -> None:
    board = RunBoard(run_id="run-1", run_root="/runs", head=lambda: HEAD)
    board.register_goal(KEY, "g", None)
    for i in range(3):
        board.put_candidate(_c(f"c-{i}"), epoch=1, seq=i + 1)
        board.put_score(_score(f"s-{i}", f"c-{i}", float(3 - i)), epoch=1, seq=i + 4)
    board.put_learning(_learning("pinned", pinned=True), epoch=1, seq=7)
    board.put_learning(_learning("newest", pinned=False, title="新" * 120), epoch=1, seq=8)
    view = board.snapshot(KEY, _frontier())
    d = build_digest(view, goal=KEY, ranking="program", cap_tokens=400)
    assert "pinned A hint" in d
    assert "- newest " not in d
    assert "3. c-2" in d
    d = build_digest(view, goal=KEY, ranking="program", cap_tokens=200)
    assert "1. c-0" in d
    assert "2. c-1" not in d
    assert "- pinned " not in d


def test_negative_cap_rejected_and_empty_digest_clipped() -> None:
    view = _empty_view()
    with pytest.raises(ValueError, match="byte cap must be nonnegative"):
        build_digest(view, goal=KEY, ranking="program", cap_tokens=-1)
    assert build_digest(view, goal=KEY, ranking="program", cap_tokens=4) == "No r"
