from collections.abc import Iterator
from pathlib import Path

import pytest
from _fakes.finished import FinishedRun
from _fakes.journals import START, FakeRun, add_pointer

from swarm_arena import AgentSpec, Plan, Ref, Score, TeamSpec
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.cache import LearningFilter
from swarm_arena._board.human_objects import HumanObject, write_human_objects
from swarm_arena._board.ingest import ingest_run
from swarm_arena._board.trace import trace
from swarm_arena._journal.a2a import task_to_json, tool_run_task
from swarm_arena._journal.event import EventDraft
from swarm_arena._model.results import Candidate, CheckResult
from swarm_arena._store.base import open_store
from swarm_arena.tool import RunStatus, ToolOutput, Usage


def _plan(run: FakeRun, revision: int = 0, first_round: int = 1) -> EventDraft:
    draft = run.plan(revision=revision, first_round=first_round, parts=["a"])
    plan = Plan.model_validate(draft.payload["plan"])
    plan = plan.model_copy(
        update={
            "teams": [
                TeamSpec(
                    name="t1",
                    rationale="Explore",
                    part="a",
                    agents=[
                        AgentSpec(
                            name="a1",
                            rationale=f"Approach {revision}",
                            tool=Ref(path="uninstalled_trace_plugin:Tool"),
                            manager=Ref(path="uninstalled_trace_plugin:Manager"),
                        )
                    ],
                )
            ],
            "judges": [Ref(path="uninstalled_trace_plugin:Judge")],
            "ranking": "",
        }
    )
    # A same-name part changes identity at the activation boundary.
    plan.parts[0].goal.inputs["revision"] = revision
    return draft.model_copy(
        update={"payload": {**draft.payload, "plan": plan.model_dump(mode="json")}}
    )


def _candidate(run: FakeRun, cid: str = "c-1", round: int = 1) -> EventDraft:
    draft = run.candidate(
        cid, round=round, part="a", parents=["parent-1", "parent-2"], fingerprint="recorded-tool"
    )
    candidate = Candidate.model_validate(draft.payload["candidate"]).model_copy(
        update={
            "output": ToolOutput(
                status="ok", usage=Usage(input_tokens=17, output_tokens=23, usd=0.00412345)
            ),
            "checks": [
                CheckResult(check="shape", passed=True),
                CheckResult(check="accuracy", passed=False, reasons=["too large", "retry"]),
            ],
        }
    )
    return draft.model_copy(
        update={
            "payload": {
                **draft.payload,
                "candidate": candidate.model_dump(mode="json"),
                "a2a": task_to_json(tool_run_task(candidate, context_id=run.run_id, at=START)),
            }
        }
    )


def _seed(tmp_path: Path) -> FakeRun:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(
        _plan(run),
        _candidate(run),
        run.score("s-1", "c-1", 0.00412345, rubric_hash="rubric-123456", panel_hash="panel-123456"),
        run.learning("L-1", "Improve accuracy", evidence=["c-1"], posted_by_call="call-1"),
    )
    add_pointer(str(tmp_path / "arena"), run)
    return run


@pytest.fixture
def board_with_run(tmp_path: Path) -> Iterator[ArenaBoard]:
    _seed(tmp_path)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        yield board
    finally:
        board.close()


def test_trace_chain_order(board_with_run: ArenaBoard) -> None:
    lines = trace(board_with_run, "c-1")
    kinds = [line.split()[0] for line in lines]
    assert kinds == sorted(kinds, key=["plan", "run", "check", "score", "learning"].index)
    assert kinds[0] == "plan"
    assert lines == [
        "plan     r1 g1 r0 origin=human team=t1 agent=a1 rationale=Approach 0",
        "run      c-1 status=ok partial=False round=1 fidelity=f1 "
        "parents=parent-1,parent-2 usage=in:17 out:23 usd:0.004123",
        "check    shape passed",
        "check    accuracy failed too large; retry",
        "score    j 0.004123 key=(-0.00412345,) rubric=rubric-1 panel=panel-12",
        "learning L-1 Improve accuracy",
    ]


def test_trace_uses_revision_in_effect(tmp_path: Path, board_with_run: ArenaBoard) -> None:
    # Use a second source with the same goal/agent names and its own revisions.
    run = FakeRun(str(tmp_path / "runs"), "r2")
    run.commit(_plan(run), _candidate(run, "c-old"))
    run.commit(_plan(run, revision=1, first_round=2), _candidate(run, "c-new", round=2))
    add_pointer(str(tmp_path / "arena"), run)
    board_with_run.frontier()
    assert "r2 g1 r0 " in trace(board_with_run, "c-old")[0]
    assert trace(board_with_run, "c-new")[0] == (
        "plan     r2 g1 r1 origin=human team=t1 agent=a1 rationale=Approach 1"
    )
    assert "r1 g1 r0 " in trace(board_with_run, "c-1")[0]


def test_trace_future_revision_is_not_yet_applicable(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(_plan(run), _plan(run, revision=1, first_round=3))
    run.commit(_candidate(run, "c-before", round=2), _candidate(run, "c-after", round=3))
    # A later revision cannot rewrite either candidate's producing plan.
    run.commit(_plan(run, revision=2, first_round=1))
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        assert "r0 " in trace(board, "c-before")[0]
        assert "r1 " in trace(board, "c-after")[0]
    finally:
        board.close()


def _review(tmp_path: Path, value: float | None) -> None:
    write_human_objects(
        open_store(str(tmp_path / "arena")),
        "toy",
        "reviews",
        [
            HumanObject(
                kind="review",
                author="human:alice",
                score=Score(
                    id="s-human",
                    candidate_id="c-1",
                    judge="human:alice",
                    key=(1.0, 0.0),
                    value=value,
                    rubric_hash="human-rubric",
                ),
            )
        ],
        reviewer="alice",
        now=START,
    )


def test_trace_includes_human_scores(tmp_path: Path, board_with_run: ArenaBoard) -> None:
    _review(tmp_path, 1.23456)
    lines = trace(board_with_run, "c-1")
    assert "score    human:alice 1.235 key=(1.0, 0.0) rubric=human-ru panel=-" in lines
    assert any(line.startswith("score    j ") for line in lines)


def test_trace_verdict_only_score(tmp_path: Path, board_with_run: ArenaBoard) -> None:
    _review(tmp_path, None)
    assert any(
        line.startswith("score    human:alice - key=(1.0, 0.0)")
        for line in trace(board_with_run, "c-1")
    )


def test_trace_unknown_candidate(board_with_run: ArenaBoard) -> None:
    with pytest.raises(KeyError, match="missing"):
        trace(board_with_run, "missing")


def test_trace_ignores_hints_and_orphans(tmp_path: Path, board_with_run: ArenaBoard) -> None:
    run = FakeRun(str(tmp_path / "other"), "r2")
    draft = _candidate(run, "hint")
    candidate = Candidate.model_validate(draft.payload["candidate"])
    board_with_run.put_candidate(candidate, epoch=1, seq=1)
    board_with_run.put_score(
        Score(id="hint-score", candidate_id="c-1", judge="hint", key=(1.0,)), epoch=1, seq=2
    )
    run.commit(_plan(run))
    run.orphan(_candidate(run, "orphan"), run.score("orphan-score", "c-1", 9, judge="orphan"))
    add_pointer(str(tmp_path / "arena"), run)
    for cid in ["hint", "orphan"]:
        with pytest.raises(KeyError):
            trace(board_with_run, cid)
    assert not any(
        "hint-score" in line or "orphan" in line for line in trace(board_with_run, "c-1")
    )
    assert not any(line.startswith("score    hint ") for line in trace(board_with_run, "c-1"))


def test_trace_keeps_one_frontier_despite_newer_cache(tmp_path: Path) -> None:
    run = _seed(tmp_path)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        expected = trace(board, "c-1")
        head = run.commit(
            run.score("later-score", "c-1", 5, judge="later"),
            run.learning("L-later", "Later learning", evidence=["c-1"]),
        )
        ingest_run(board._cache("toy"), run.store, run.run_id, run.run_root, head)
        assert trace(board, "c-1") == expected
        board.frontier()
        assert "learning L-later Later learning" in trace(board, "c-1")
    finally:
        board.close()


def test_trace_includes_all_visible_citations(tmp_path: Path) -> None:
    run = _seed(tmp_path)
    run.commit(
        run.candidate("c-other"),
        run.learning("L-unrelated", "Unrelated", evidence=["c-other"]),
        *(run.learning(f"L-{i}", f"Citation {i}", evidence=["c-1"]) for i in range(2, 14)),
    )
    human = run.learning("L-human", "Human citation", evidence=["c-1"], author="human:alice")
    write_human_objects(
        open_store(str(tmp_path / "arena")),
        "toy",
        "curation",
        [
            HumanObject(kind="add", author="human:alice", learning=human.payload["learning"]),
            HumanObject(kind="hide", author="human:alice", learning_id="L-2"),
        ],
        now=START,
    )
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        lines = [line for line in trace(board, "c-1") if line.startswith("learning ")]
        assert len(lines) == 13
        assert "learning L-human Human citation" in lines
        assert "learning L-13 Citation 13" in lines
        assert not any("L-unrelated" in line or line.startswith("learning L-2 ") for line in lines)
    finally:
        board.close()


def test_trace_scopes_human_objects_to_candidate_namespace(tmp_path: Path) -> None:
    _seed(tmp_path)
    other = FakeRun(str(tmp_path / "runs"), "other", namespace="alpha")
    other.commit(_plan(other), _candidate(other, "c-alpha"))
    add_pointer(str(tmp_path / "arena"), other)
    _review(tmp_path, None)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["alpha", "toy"])
    try:
        assert "r1 g1 r0 " in trace(board, "c-1")[0]
        assert "other g1 r0 " in trace(board, "c-alpha")[0]
        assert not any(line.startswith("score ") for line in trace(board, "c-alpha"))
    finally:
        board.close()


@pytest.mark.parametrize("status", [RunStatus.OK, RunStatus.ERROR])
def test_trace_unknown_cost_and_empty_parents(tmp_path: Path, status: RunStatus) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(_plan(run), run.candidate("c-1", partial=True, status=status))
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        assert trace(board, "c-1")[1] == (
            f"run      c-1 status={status.value} partial=True round=1 fidelity=f1 "
            "parents=- usage=in:0 out:0 usd:-"
        )
    finally:
        board.close()


async def test_trace_finished_run(finished_run: FinishedRun) -> None:
    board = ArenaBoard(arena_root=finished_run.config.arena_root)
    board.open("", "", [finished_run.namespace])
    try:
        frontier = board.frontier()
        cache = board._cache(finished_run.namespace)
        candidates = cache.candidates(frontier)
        assert candidates
        learning_ids = cache.find(
            frontier, query=None, filters=LearningFilter(), relevance_goal=None, limit=-1
        )
        learnings = cache.learnings_by_id(frontier, learning_ids)
        cited_ids = {id for learning in learnings for id in learning.evidence}
        assert cited_ids & {candidate.id for candidate, _ in candidates}
        for candidate, position in candidates:
            lines = trace(board, candidate.id)
            assert lines[0].startswith(f"plan     {finished_run.run_id} {candidate.goal_id} r0 ")
            assert lines[1].startswith(f"run      {candidate.id} ")
            assert any(line.startswith("score    ") for line in lines)
            assert {line for line in lines if line.startswith("learning ")} == {
                f"learning {learning.id} {learning.title}"
                for learning in learnings
                if candidate.id in learning.evidence
            }
            state = cache._replay(candidate.run_id, through_seq=position[2])
            assert state.fingerprints[candidate.id]
            assert state.positions[candidate.id][1] == position[2]
    finally:
        board.close()
