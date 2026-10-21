"""Exercise posting rules against real fixed-frontier arena views."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from _fakes.board import GOAL, candidate, learning, records, seed_board
from _fakes.journals import FakeRun, add_pointer
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Plan
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.base import BoardView
from swarm_arena._board.learnings import (
    DUPLICATE_JACCARD,
    LEARNING_GUIDE,
    find_duplicates,
    is_verified,
    jaccard,
    title_tokens,
    vet_learning,
)
from swarm_arena._engine.goal import r0_revision_draft
from swarm_arena._errors import SoftError
from swarm_arena.board import Candidate, Learning
from swarm_arena.scoring import CheckResult
from swarm_arena.tool import RunStatus, ToolOutput

OLD_TITLE = "Fitting log(nu) instead of nu converges faster"
TITLE = "Fitting log(nu) instead of nu converges 3x faster"
SECRET = "private-learning-sentinel"


@pytest.fixture
def board(tmp_path: Path) -> Iterator[ArenaBoard]:
    board = seed_board(
        tmp_path,
        candidates=[candidate("c-1")],
        learnings=[
            learning("L-old", title=OLD_TITLE),
            learning("L-other", title="Grid spacing h=0.01 is stable"),
            learning("L-hidden", title="Hidden advice sentinel", hidden=True),
        ],
    )
    yield board
    board.close()


@pytest.fixture
def view(board: ArenaBoard) -> BoardView:
    return board.snapshot(GOAL.key(), board.frontier())


def _vet(
    view: BoardView,
    own: tuple[Candidate | Learning, ...] = (),
    *,
    author: str = "t1/a1",
    title: str = "Independent observation",
    text: str = "t",
    evidence: tuple[str, ...] | list[str] = ("c-1",),
    refines: tuple[str, ...] | list[str] = (),
    duplicate_reason: str | None = None,
) -> bool:
    return vet_learning(
        view,
        own,
        namespace="n",
        author=author,
        title=title,
        text=text,
        evidence=evidence,
        refines=refines,
        duplicate_reason=duplicate_reason,
    )


def test_title_limit_120(view: BoardView) -> None:
    assert _vet(view, title="x" * 120)
    with pytest.raises(SoftError) as caught:
        _vet(view, title="x" * 121)
    assert str(caught.value).endswith(LEARNING_GUIDE)


def test_text_limit_1500(view: BoardView) -> None:
    assert _vet(view, text="x" * 1500)
    with pytest.raises(SoftError) as caught:
        _vet(view, text=SECRET * 1500)
    assert str(caught.value).endswith(LEARNING_GUIDE)
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize(
    "title",
    [
        "",
        "   ",
        *[
            SECRET + char
            for char in ("\n", "\r", "\t", "\x00", "\x7f", "\x85", "\u2028", "\u2029", "\ud800")
        ],
    ],
)
def test_empty_or_multiline_title_rejected(view: BoardView, title: str) -> None:
    with pytest.raises(SoftError) as caught:
        _vet(view, title=title)
    assert str(caught.value).endswith(LEARNING_GUIDE)
    assert_secret_absent(caught.value, SECRET, sanitized=True)


def test_title_validation_uses_production_trimming(view: BoardView) -> None:
    assert _vet(view, title="  " + "é" * 120 + "  ")


def test_evidence_must_be_in_view(tmp_path: Path, view: BoardView, board: ArenaBoard) -> None:
    later = FakeRun(str(tmp_path / "later"), "later-run", goal=GOAL)
    plan = Plan(goal=GOAL, teams=[], judges=[])
    later.commit(
        r0_revision_draft(plan, plan.base_dir), *records(candidate("c-later", run_id="later-run"))
    )
    add_pointer(str(tmp_path / "arena"), later)
    board.frontier()
    assert board.get_candidate("c-later").id == "c-later"
    with pytest.raises(SoftError, match="not in your view"):
        _vet(view, evidence=["c-later"])


def test_own_candidate_counts_as_evidence(view: BoardView) -> None:
    assert _vet(view, (candidate("c-own"),), evidence=["c-own"])
    with pytest.raises(SoftError, match="not in your view"):
        _vet(view, evidence=["c-own"])


def test_evidence_required_except_for_humans(view: BoardView) -> None:
    with pytest.raises(SoftError):
        _vet(view, evidence=[])
    assert not _vet(view, author="human:alice", evidence=[])
    with pytest.raises(SoftError, match="not in your view"):
        _vet(view, author="human:alice", evidence=["c-absent"])


def test_refines_must_be_visible_learning_in_namespace(view: BoardView) -> None:
    assert _vet(view, refines=["L-other"])
    assert _vet(view, (learning("L-own"),), refines=["L-own"])
    for id in ["c-1", "L-absent", "L-hidden"]:
        with pytest.raises(SoftError, match="not in your view"):
            _vet(view, refines=[id])
    with pytest.raises(SoftError, match="not in your view"):
        _vet(view, (learning("L-away", namespace="elsewhere"),), refines=["L-away"])


def test_duplicate_requires_refines_or_reason(view: BoardView) -> None:
    with pytest.raises(SoftError, match="L-old"):
        _vet(view, title=TITLE)
    assert _vet(view, title=TITLE, refines=["L-old"])
    assert _vet(view, title=TITLE, duplicate_reason="new lr")
    for refines, reason in [(["L-other"], None), ([], " \t ")]:
        with pytest.raises(SoftError, match="near-duplicate of L-old"):
            _vet(view, title=TITLE, refines=refines, duplicate_reason=reason)


def test_title_tokens_and_jaccard() -> None:
    assert title_tokens("Fitting LOG(nu), nu; h=0.01") == frozenset(
        {"fitting", "log", "nu", "h", "0", "01"}
    )
    assert jaccard(frozenset(), frozenset()) == 0.0
    assert jaccard(frozenset("ab"), frozenset("cd")) == 0.0


def test_jaccard_threshold_is_inclusive(view: BoardView) -> None:
    assert jaccard(frozenset("abc"), frozenset("abcde")) == DUPLICATE_JACCARD == 0.6
    assert (
        find_duplicates(
            view, [learning("L-boundary", title="a b c d e")], namespace="n", title="a b c"
        )[0].id
        == "L-boundary"
    )


def test_hidden_learning_is_not_a_duplicate(view: BoardView) -> None:
    assert _vet(view, title="Hidden advice sentinel")
    assert _vet(
        view,
        (learning("L-own-hidden", title=TITLE, hidden=True),),
        title=TITLE,
        duplicate_reason="visible old learning",
    )
    assert (
        find_duplicates(
            view,
            [learning("L-own-hidden", title="Hidden advice sentinel", hidden=True)],
            namespace="n",
            title="Hidden advice sentinel",
        )
        == []
    )


def test_own_learning_duplicate_detected(view: BoardView) -> None:
    own = learning("L-own", title=SECRET)
    with pytest.raises(SoftError, match="near-duplicate of L-own") as caught:
        _vet(view, (own,), title=SECRET)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert _vet(view, (own,), title=SECRET, refines=["L-own"])
    assert (
        find_duplicates(
            view, [own.model_copy(update={"namespace": "away"})], namespace="n", title=SECRET
        )
        == []
    )


@pytest.mark.parametrize(
    ("output", "checks", "expected"),
    [
        (ToolOutput(status="ok", partial=True), [], False),
        (ToolOutput(status="ok"), [CheckResult(check="gate", passed=False)], False),
        *[(ToolOutput(status=status), [], False) for status in RunStatus if status != RunStatus.OK],
        (ToolOutput(status="ok"), [CheckResult(check="gate", passed=True)], True),
    ],
)
def test_verified_needs_complete_passing_evidence(
    output: ToolOutput, checks: list[CheckResult], expected: bool
) -> None:
    item = candidate("c", output=output, checks=checks)
    assert is_verified([item]) is expected
    assert not is_verified([])
    assert is_verified([item, candidate("passing")])


def test_duplicate_search_uses_only_top_hit(
    view: BoardView, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    search = Mock(return_value=[learning("L-other", title="Unrelated observation")])
    monkeypatch.setattr(type(view), "search_learnings", search)
    assert find_duplicates(view, [], namespace="n", title=TITLE) == []
    search.assert_called_once_with("n", TITLE, limit=1)
