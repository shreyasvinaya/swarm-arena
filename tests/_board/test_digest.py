import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from _fakes.board import KEY, seed_board
from _fakes.board import candidate as _c
from _fakes.board import learning as _learning
from _fakes.board import score as _score

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.digest import (
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    build_digest,
    estimate_tokens,
)
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
    assert d.splitlines()[0] == UNTRUSTED_OPEN
    assert any(line.startswith("Leaderboard") for line in d.splitlines())


def test_empty_board_digest(make_board: Callable[..., ArenaBoard]) -> None:
    board = make_board()
    d = build_digest(
        board.snapshot(KEY, board.frontier()),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="program",
        cap_tokens=1500,
    )
    lines = d.splitlines()
    assert lines[0] == UNTRUSTED_OPEN
    assert UNTRUSTED_CLOSE in lines
    assert re.fullmatch(
        r"0 more learnings in namespace \S+ \(0 for this goal\)\. Use search_learnings\.",
        lines[-1],
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
        assert (
            "415 more learnings in namespace n (415 for this goal), "
            "37 new since the last round. Use search_learnings."
        ).startswith(d)
    elif cap >= 200:
        assert d.endswith("new since the last round. Use search_learnings.")


def test_valid_unicode_titles_unchanged_and_hidden_omitted(
    make_board: Callable[..., ArenaBoard],
) -> None:
    title = "日本🙂e\u0301"
    view = _view(make_board, 1, title)
    d = build_digest(view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=1500)
    assert title in d
    assert " · verified" not in d
    board = make_board(learnings=[_learning("L-hidden", hidden=True, title="SECRET_SENTINEL")])
    hidden_digest = build_digest(
        board.snapshot(KEY, board.frontier()),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="program",
        cap_tokens=1500,
    )
    assert "SECRET_SENTINEL" not in hidden_digest
    assert "0 more learnings in namespace n (0 for this goal)" in hidden_digest


def test_pruning_drops_newest_then_pinned_then_extra_rows(
    make_board: Callable[..., ArenaBoard],
) -> None:
    board = make_board(
        candidates=[_c(f"c-{i}") for i in range(3)],
        scores=[_score(f"s-{i}", f"c-{i}", float(3 - i)) for i in range(3)],
        learnings=[_learning("pinned", pinned=True), _learning("newest", title="新" * 120)],
    )
    view = board.snapshot(KEY, board.frontier())
    d = build_digest(view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=600)
    assert "pinned A hint" in d
    assert "- newest " not in d
    assert "3. c-2" in d
    d = build_digest(view, goal=KEY, run_id="run-1", round=3, ranking="program", cap_tokens=320)
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
        == "0 mo"
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
        "0 more learnings in namespace n (5 for this goal), "
        f"{expected} new since the last round. Use search_learnings."
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
        assert (
            digest.splitlines()[-1]
            == "0 more learnings in namespace n (2 for this goal). Use search_learnings."
        )
    elif cap == 50:
        count = "2 more learnings in namespace n (2 for this goal). Use search_learnings."
        assert digest == count.encode()[:cap].decode()


def test_digest_owner_without_a_mark_has_no_previous_round_count(
    make_board: Callable[..., ArenaBoard],
) -> None:
    board = make_board(learnings=[_learning("other-run-round", round=1)])
    view = board.snapshot(KEY, board.frontier())
    digest = build_digest(
        view, goal=KEY, run_id="unmarked", round=2, ranking="program", cap_tokens=1500
    )
    assert digest.endswith(
        "0 more learnings in namespace n (1 for this goal), "
        "0 new since the last round. Use search_learnings."
    )


@pytest.mark.parametrize("n", [10, 2000])
def test_digest_size_does_not_grow(make_board: Callable[..., ArenaBoard], n: int) -> None:
    text = build_digest(
        _view(make_board, n, "A short hint"),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="program",
        cap_tokens=1500,
    )
    assert estimate_tokens(text) <= 1500
    assert re.search(r"^\d+ more learnings in namespace n", text, re.M)
    listed = len(re.findall(r"^- L-", text, re.M))
    assert listed == min(n, 11)
    assert f"{n - listed} more learnings" in text


@pytest.mark.parametrize("cap", [1, 20, 1500])
def test_never_exceeds_cap(make_board: Callable[..., ArenaBoard], cap: int) -> None:
    text = build_digest(
        _view(make_board, 20, "日本🙂e\u0301" * 10 + "abcdef0123456789" * 3),
        goal=KEY,
        run_id="run-1",
        round=2,
        ranking="program",
        cap_tokens=cap,
    )
    assert len(text.encode("utf-8")) <= cap


def test_board_content_inside_markers(make_board: Callable[..., ArenaBoard]) -> None:
    text = build_digest(
        _view(make_board, 3, "A title"),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="program",
        cap_tokens=10000,
    )
    lines = text.splitlines()
    assert lines[0] == UNTRUSTED_OPEN
    assert lines[-2] == UNTRUSTED_CLOSE
    assert any(line.startswith("1. c-1 fd-solver/tuner-1 5 built on c-0") for line in lines[1:-2])
    assert sum("A title" in line for line in lines[1:-2]) == 3
    assert "A title" not in lines[-1]


def test_leaderboard_names_judge_identity(make_board: Callable[..., ArenaBoard]) -> None:
    board = make_board(
        candidates=[_c("c-1")],
        scores=[_score("s-1", "c-1", 2, judge="j", rubric_hash="r1aaaaaabbbb", panel_hash="")],
    )
    text = build_digest(
        board.snapshot(KEY, board.frontier()),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="j",
        cap_tokens=10000,
    )
    assert "Leaderboard g (j, rubric r1aaaaaa, panel -):" in text
    # Null values must use format_value, rather than displaying the ranking key.
    assert "tuner-1 - built on —" in text


def test_pinned_first_then_relevance(make_board: Callable[..., ArenaBoard]) -> None:
    board = make_board(
        learnings=[
            _learning("pin", pinned=True, goal_id="g2", title="Pinned"),
            _learning("verified", verified=True, title="Verified"),
            _learning("newer", title="Newer"),
        ]
    )
    text = build_digest(
        board.snapshot(KEY, board.frontier()),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="j",
        cap_tokens=10000,
    )
    assert text.index("Pinned learnings:") < text.index("- pin Pinned")
    assert text.index("- pin Pinned") < text.index("Top learnings:")
    assert text.index("- verified Verified · verified") < text.index("- newer Newer")
    assert text.count("- pin ") == 1


def test_titles_only_never_text(make_board: Callable[..., ArenaBoard]) -> None:
    board = make_board(learnings=[_learning("l", title="Public title", text="SECRET-TEXT")])
    text = build_digest(
        board.snapshot(KEY, board.frontier()),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="j",
        cap_tokens=10000,
    )
    assert "Public title" in text
    assert "SECRET-TEXT" not in text


def test_covers_whole_namespace(make_board: Callable[..., ArenaBoard]) -> None:
    board = make_board(learnings=[_learning("old", run_id="another", title="Another run")])
    text = build_digest(
        board.snapshot(KEY, board.frontier()),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="j",
        cap_tokens=10000,
    )
    assert "- old Another run" in text


def test_this_goal_first_in_top_learnings(make_board: Callable[..., ArenaBoard]) -> None:
    board = make_board(learnings=[_learning("own"), _learning("other", goal_id="g2")])
    text = build_digest(
        board.snapshot(KEY, board.frontier()),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="j",
        cap_tokens=10000,
    )
    assert text.index("- own ") < text.index("- other ")
    assert "0 more learnings in namespace n (1 for this goal)" in text


def test_hidden_not_listed(make_board: Callable[..., ArenaBoard]) -> None:
    board = make_board(learnings=[_learning("hidden", hidden=True), _learning("visible")])
    text = build_digest(
        board.snapshot(KEY, board.frontier()),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="j",
        cap_tokens=10000,
    )
    assert "- hidden " not in text
    assert "- visible " in text
    assert "(1 for this goal)" in text


@pytest.fixture
def group_board(tmp_path: Path) -> Iterator[ArenaBoard]:
    from _fakes.board import GOAL, records
    from _fakes.journals import START, FakeRun, add_pointer

    from swarm_arena import Goal, PartSpec, Plan, Ref
    from swarm_arena._board.human_objects import HumanObject, write_human_objects
    from swarm_arena._engine.goal import r0_revision_draft
    from swarm_arena._store.base import open_store

    plan = Plan(
        goal=GOAL,
        teams=[],
        judges=[Ref(path="missing:Judge", name="j")],
        parts=[
            PartSpec(
                name="a",
                goal=Goal(id="a", namespace="n"),
                judges=[Ref(path="missing:Judge", name="a")],
            )
        ],
    )
    run = FakeRun(str(tmp_path / "runs"), "run-1", goal=GOAL)
    run.commit(
        r0_revision_draft(plan, plan.base_dir),
        *records(
            _c("main"),
            _c("part", part="a"),
            _score("main-score", "main", 2, judge="j", value=2, rubric_hash="main-rubric"),
            _score("part-score", "part", 3, judge="a", value=3, rubric_hash="part-rubric"),
        ),
    )
    arena = str(tmp_path / "arena")
    add_pointer(arena, run)
    write_human_objects(
        open_store(arena),
        "n",
        "reviews",
        [
            HumanObject(
                kind="review",
                author="human:alice",
                score=_score("reject", "main", 0, judge="human:alice", verdict="reject"),
            ),
            HumanObject(
                kind="review",
                author="human:alice",
                score=_score("accept", "main", 1, judge="human:alice", verdict="accept"),
            ),
        ],
        now=START,
        reviewer="alice",
    )
    board = ArenaBoard(arena_root=arena, cache_dir=str(tmp_path / "cache"))
    board.open("run-1", str(tmp_path / "runs"), ["n"])
    yield board
    board.close()


def test_human_verdict_marked(group_board: ArenaBoard) -> None:
    text = build_digest(
        group_board.snapshot(KEY, group_board.frontier()),
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="j",
        cap_tokens=10000,
    )
    assert " · human: accept" in text
    assert " · human: reject" not in text


def test_part_blocks_use_their_own_judges(group_board: ArenaBoard) -> None:
    view = group_board.snapshot(KEY, group_board.frontier())
    text = build_digest(
        view, goal=KEY, run_id="run-1", round=1, ranking={None: "j", "a": "a"}, cap_tokens=10000
    )
    assert "Part a (a, rubric part-rub, panel panel):\n1. part " in text
    same_judge = build_digest(
        view, goal=KEY, run_id="run-1", round=1, ranking="j", cap_tokens=10000
    )
    assert "Part a" not in same_judge
    omitted = build_digest(
        view, goal=KEY, run_id="run-1", round=1, ranking={None: "j"}, cap_tokens=10000
    )
    assert "Part a" not in omitted


def test_part_view_digest_has_no_part_blocks(group_board: ArenaBoard) -> None:
    from swarm_arena import Goal
    from swarm_arena._engine.round import _SeedView

    key = Goal(id="a", namespace="n").key()
    view = group_board.snapshot(key, group_board.frontier())
    # Real wrapped part view: seed parents must not change digest group metadata.
    for scoped in (view, _SeedView(view, {})):
        text = build_digest(
            scoped,
            goal=key,
            run_id="run-1",
            round=1,
            ranking={None: "j", "a": "a"},
            cap_tokens=10000,
        )
        assert "Leaderboard a (a, rubric part-rub, panel panel):\n1. part " in text
        assert not any(line.startswith("Part ") for line in text.splitlines())
        assert "1. main " not in text


def test_part_rows_pruned_before_main_rows(group_board: ArenaBoard) -> None:
    view = group_board.snapshot(KEY, group_board.frontier())
    text = build_digest(
        view, goal=KEY, run_id="run-1", round=1, ranking={None: "j", "a": "a"}, cap_tokens=10000
    )
    # Removing the part block should leave the main row intact at this cap.
    part_block = next(line for line in text.splitlines() if line.startswith("Part a"))
    part_row = next(line for line in text.splitlines() if line.startswith("1. part"))
    cap = estimate_tokens(text) - estimate_tokens(part_block + "\n" + part_row + "\n")
    pruned = build_digest(
        view, goal=KEY, run_id="run-1", round=1, ranking={None: "j", "a": "a"}, cap_tokens=cap
    )
    assert "1. main " in pruned
    assert "Part a" not in pruned
    assert len(pruned.encode()) <= cap


def test_markers_survive_when_only_markers_and_counts_fit(
    make_board: Callable[..., ArenaBoard],
) -> None:
    view = _view(make_board, 1, "日本🙂")
    count = "1 more learnings in namespace n (1 for this goal). Use search_learnings."
    minimum = "\n".join([UNTRUSTED_OPEN, UNTRUSTED_CLOSE, count])
    text = build_digest(
        view,
        goal=KEY,
        run_id="run-1",
        round=1,
        ranking="program",
        cap_tokens=estimate_tokens(minimum),
    )
    assert text == minimum


def test_missing_group_mapping_omits_leaderboard(make_board: Callable[..., ArenaBoard]) -> None:
    view = _view(make_board, 1)
    text = build_digest(view, goal=KEY, run_id="run-1", round=1, ranking={}, cap_tokens=1500)
    assert "Leaderboard" not in text
    assert "1. c-1" not in text
    assert UNTRUSTED_OPEN in text


def test_goal_scope_rejected_even_without_ranking(make_board: Callable[..., ArenaBoard]) -> None:
    from _fakes.board import OTHER_KEY
    from _fakes.secrecy import assert_secret_absent

    view = _view(make_board, 1)
    # Even an empty mapping cannot bypass the fixed-view goal boundary.
    goal = OTHER_KEY.model_copy(update={"inputs_sha256": "SECRET-SCOPE-SENTINEL"})
    with pytest.raises(ValueError, match="goal outside board snapshot") as caught:
        build_digest(view, goal=goal, run_id="run-1", round=1, ranking={}, cap_tokens=1500)
    assert_secret_absent(caught.value, "SECRET-SCOPE-SENTINEL")
