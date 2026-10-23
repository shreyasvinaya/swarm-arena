from collections import Counter
from collections.abc import Generator, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Any

import pytest
from _fakes.finished import FinishedRun
from _fakes.journals import START, FakeRun, add_pointer

from swarm_arena import AgentSpec, Plan, Ref, Score, TeamSpec
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.cache import ArenaCache, LearningFilter
from swarm_arena._board.human_objects import HumanObject, write_human_objects
from swarm_arena._board.ingest import ingest_run
from swarm_arena._board.trace import trace
from swarm_arena._errors import ChainError
from swarm_arena._journal.a2a import task_to_json, tool_run_task
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.replay import RunState
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.results import Candidate, CheckResult, Learning
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
        board.frontier()
        yield board
    finally:
        board.close()


def test_trace_chain_order(board_with_run: ArenaBoard) -> None:
    lines = trace(board_with_run, "c-1")
    kinds = [line.split()[0] for line in lines]
    assert kinds == sorted(kinds, key=["plan", "run", "check", "score", "llm", "learning"].index)
    assert kinds[0] == "plan"
    assert lines == [
        "plan     r1 g1 r0 origin=human team=t1 agent=a1 rationale=Approach 0",
        "run      c-1 status=ok partial=False round=1 fidelity=f1 "
        "parents=parent-1,parent-2 usage=in:17 out:23 usd:0.004123",
        "check    shape passed",
        "check    accuracy failed too large; retry",
        "score    j 0.004123 key=(-0.00412345,) rubric=rubric-1 panel=panel-12",
        "llm call-1: not found",
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
        board.frontier()
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
    board_with_run.frontier()
    lines = trace(board_with_run, "c-1")
    assert "score    human:alice 1.235 key=(1.0, 0.0) rubric=human-ru panel=-" in lines
    assert any(line.startswith("score    j ") for line in lines)


def test_trace_verdict_only_score(tmp_path: Path, board_with_run: ArenaBoard) -> None:
    _review(tmp_path, None)
    board_with_run.frontier()
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
    board_with_run.frontier()
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
        board.frontier()
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
        board.frontier()
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
        board.frontier()
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
        board.frontier()
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


def _work_board(tmp_path: Path, *, history: int = 0, unrelated: int = 0) -> ArenaBoard:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(
        run.plan(),
        run.candidate("other"),
        *(EventDraft(type=EventType.WATCHDOG) for _ in range(history)),
        run.candidate("target"),
        run.learning("citing", "Target citation", evidence=["target"]),
        *(
            run.learning(f"L-{i}", f"Unrelated {i}", evidence=["other"], text="x" * 1500)
            for i in range(unrelated)
        ),
    )
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    board.frontier()
    return board


def _count_work(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    counts: Counter[str] = Counter()
    apply = RunState._apply
    hydrate = Learning.model_validate_json
    plan = Plan.model_validate_json
    replace = ArenaCache._replace_sets

    def counted_apply(self: RunState, event: Event) -> None:
        counts["folds"] += 1
        apply(self, event)

    def counted_hydrate(
        cls: type[Learning], data: str | bytes | bytearray, **kwargs: Any
    ) -> Learning:
        counts["hydrated"] += 1
        return hydrate(data, **kwargs)

    def counted_plan(cls: type[Plan], data: str | bytes | bytearray, **kwargs: Any) -> Plan:
        counts["plans"] += 1
        return plan(data, **kwargs)

    def counted_sets(
        self: ArenaCache, sets: Sequence[tuple[str, Iterable[tuple[str, int | None]]]]
    ) -> None:
        counts["mark_fills"] += sum(name == "marks" for name, _ in sets)
        replace(self, sets)

    monkeypatch.setattr(RunState, "_apply", counted_apply)
    monkeypatch.setattr(Learning, "model_validate_json", classmethod(counted_hydrate))
    monkeypatch.setattr(Plan, "model_validate_json", classmethod(counted_plan))
    monkeypatch.setattr(ArenaCache, "_replace_sets", counted_sets)
    return counts


@pytest.mark.parametrize("history", [0, 3000])
def test_trace_warm_reads_do_not_refold_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: int
) -> None:
    board = _work_board(tmp_path, history=history)
    try:
        expected = trace(board, "target")
        counts = _count_work(monkeypatch)
        for _ in range(3):
            assert trace(board, "target") == expected
        assert counts["folds"] == 0, dict(counts)
        assert counts["plans"] == 0, dict(counts)
    finally:
        board.close()


@pytest.mark.parametrize("unrelated", [0, 1000])
def test_trace_hydrates_only_citing_learnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unrelated: int
) -> None:
    board = _work_board(tmp_path, unrelated=unrelated)
    try:
        expected = trace(board, "target")
        assert [line for line in expected if line.startswith("learning ")] == [
            "learning citing Target citation"
        ]
        counts = _count_work(monkeypatch)
        for _ in range(3):
            assert trace(board, "target") == expected
        assert counts["hydrated"] == 3, dict(counts)
    finally:
        board.close()


def test_trace_fills_frontier_once_per_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = _work_board(tmp_path)
    try:
        expected = trace(board, "target")
        counts = _count_work(monkeypatch)
        for _ in range(3):
            assert trace(board, "target") == expected
        assert counts["mark_fills"] == 3, dict(counts)
    finally:
        board.close()


def test_trace_releases_frontier_after_hydration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = _work_board(tmp_path)
    try:
        expected = trace(board, "target")
        cache = board._read_state().readers["toy"]

        def fail(frontier: Frontier, ids: Sequence[str]) -> list[Learning]:
            raise RuntimeError("hydration unavailable")

        with monkeypatch.context() as failing:
            failing.setattr(cache, "learnings_by_id", fail)
            with pytest.raises(RuntimeError, match="hydration unavailable"):
                trace(board, "target")
        assert cache._frontier_depth == 0
        assert cache._conn.execute("SELECT count(*) FROM temp._read_sets").fetchone() == (0,)
        counts = _count_work(monkeypatch)
        assert trace(board, "target") == expected
        assert counts["mark_fills"] == 1
    finally:
        board.close()


def test_trace_does_not_hold_board_lock_during_cache_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = _work_board(tmp_path)
    entered, release = ThreadEvent(), ThreadEvent()
    cache = board._read_state().readers["toy"]
    query = cache._citing_learning_ids

    def blocked(frontier: Frontier, candidate_id: str) -> list[str]:
        entered.set()
        assert release.wait(5)
        return query(frontier, candidate_id)

    monkeypatch.setattr(cache, "_citing_learning_ids", blocked)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(trace, board, "target")
            acquired = False
            try:
                assert entered.wait(5)
                acquired = board._lock.acquire(timeout=1)
                assert acquired
            finally:
                if acquired:
                    board._lock.release()
                release.set()
            assert "learning citing Target citation" in result.result(timeout=5)
    finally:
        board.close()


def test_historical_plan_query_matches_core_prefixes(tmp_path: Path) -> None:
    run = _seed(tmp_path)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        old = board.frontier()
        run.commit(_plan(run, revision=1, first_round=3), _candidate(run, "c-before", round=2))
        run.commit(_candidate(run, "c-after", round=3))
        run.commit(_plan(run, revision=2, first_round=1))
        frontier = board.frontier()
        cache = board._cache("toy")
        revisions = board._read_state().plans["r1"].revisions["g1"]
        for candidate, position in cache.candidates(frontier):
            projected = cache._plan_projection(
                frontier,
                candidate.run_id,
                candidate.goal_id,
                through_seq=position[2],
                revisions=revisions,
            )
            replayed = cache._replay(candidate.run_id, through_seq=position[2])
            assert all(plan is revisions[plan.revision] for plan, _ in projected.plans["g1"])
            assert projected.plans == replayed.plans
            assert projected.plan_first_rounds == replayed.plan_first_rounds
            for round in range(5):
                assert projected.plan_for_round(
                    candidate.goal_id, round
                ) == replayed.plan_for_round(candidate.goal_id, round)
        # The mark also bounds a query whose source-prefix argument is newer.
        assert (
            cache._plan_projection(old, "r1", "g1", through_seq=10**6, revisions=revisions).plans
            == cache._replay("r1", through_seq=old.marks[0].seq).plans
        )
        with pytest.raises(ChainError, match="no committed plan"):
            cache._plan_projection(frontier, "r1", "g1", through_seq=0, revisions=revisions)
    finally:
        board.close()


def test_citation_query_uses_fixed_human_membership_and_visibility(tmp_path: Path) -> None:
    run = _seed(tmp_path)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        old = board.frontier()
        cache = board._cache("toy")
        human = Learning.model_validate(
            run.learning(
                "L-human", "Human citation", evidence=["c-1"], author="human:alice"
            ).payload["learning"]
        )
        write_human_objects(
            open_store(str(tmp_path / "arena")),
            "toy",
            "curation",
            [HumanObject(kind="add", author="human:alice", learning=human)],
            now=START,
        )
        added = board.frontier()
        assert cache._citing_learning_ids(old, "c-1") == ["L-1"]
        assert cache._citing_learning_ids(added, "c-1") == ["L-1", "L-human"]
        write_human_objects(
            open_store(str(tmp_path / "arena")),
            "toy",
            "curation",
            [HumanObject(kind="hide", author="human:alice", learning_id="L-1")],
            now=START.replace(year=2027),
        )
        hidden = board.frontier()
        assert cache._citing_learning_ids(hidden, "c-1") == ["L-human"]
        assert cache._citing_learning_ids(added, "c-1") == ["L-1", "L-human"]
        assert cache._citing_learning_ids(hidden, "unknown") == []
        missing_objects = Frontier(marks=hidden.marks, curation=["toy/curation/missing.jsonl"])
        with pytest.raises(ChainError):
            cache._citing_learning_ids(missing_objects, "c-1")
    finally:
        board.close()


def test_trace_is_empty_before_first_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("committed read performed refresh preparation")

    try:
        with monkeypatch.context() as fixed:
            fixed.setattr(board, "frontier", forbidden)
            fixed.setattr(board, "_cache", forbidden)
            fixed.setattr(board, "_read_cache", forbidden)
            with pytest.raises(KeyError, match="c-1"):
                trace(board, "c-1")
        board.frontier()
        assert trace(board, "c-1")[0].startswith("plan     r1 ")
    finally:
        board.close()


def test_trace_uses_one_generation_while_refresh_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = _work_board(tmp_path)
    entered, release = ThreadEvent(), ThreadEvent()
    prepare = board._prepare_published
    capture = board._pin_state
    captures = 0

    def blocked(frontier: Frontier) -> Any:
        entered.set()
        assert release.wait(5)
        return prepare(frontier)

    def counted_capture() -> Any:
        nonlocal captures
        captures += 1
        return capture()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("committed read initialized a cache")

    try:
        expected = trace(board, "target")
        monkeypatch.setattr(board, "_prepare_published", blocked)
        monkeypatch.setattr(board, "_pin_state", counted_capture)
        with ThreadPoolExecutor(max_workers=2) as pool:
            refreshed = pool.submit(board.frontier)
            try:
                assert entered.wait(5)
                with monkeypatch.context() as fixed:
                    fixed.setattr(board, "_cache", forbidden)
                    fixed.setattr(board, "_read_cache", forbidden)
                    result = pool.submit(trace, board, "target")
                    assert result.result(timeout=1) == expected
                    assert captures == 1
            finally:
                release.set()
            refreshed.result(timeout=5)
    finally:
        board.close()


def test_trace_pins_its_generation_against_concurrent_close(tmp_path: Path) -> None:
    _seed(tmp_path)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    board.frontier()
    expected = trace(board, "c-1")
    entered, release = ThreadEvent(), ThreadEvent()
    candidates = ArenaCache.candidates

    def paused(self: ArenaCache, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(5)
        return candidates(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patched, ThreadPoolExecutor(max_workers=2) as pool:
        patched.setattr(ArenaCache, "candidates", paused)
        result = pool.submit(trace, board, "c-1")
        assert entered.wait(5)
        closed = pool.submit(board.close)
        assert not closed.done()
        release.set()
        assert result.result(timeout=5) == expected
        closed.result(timeout=5)
    with pytest.raises(RuntimeError, match="arena board is closed"):
        trace(board, "c-1")


def test_trace_keeps_captured_generation_across_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _seed(tmp_path)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()
        expected = trace(board, "c-1")
        previous = board._read_state()
        run.commit(
            _plan(run, revision=1, first_round=1),
            run.score("later", "c-1", 7, judge="later"),
            run.learning("L-later", "Later citation", evidence=["c-1"]),
        )
        board.frontier()
        pin = board._pin_state
        captures = 0

        @contextmanager
        def captured() -> Generator[Any]:
            nonlocal captures
            captures += 1
            with pin() as published:
                yield previous if captures == 1 else published

        with monkeypatch.context() as fixed:
            fixed.setattr(board, "_pin_state", captured)
            assert trace(board, "c-1") == expected
            assert captures == 1
            current = trace(board, "c-1")
            assert captures == 2
        assert current[0] == expected[0]
        assert "learning L-later Later citation" in current
        assert any(line.startswith("score    later ") for line in current)
    finally:
        board.close()


def test_trace_shows_llm_steps_in_order(tmp_path: Path) -> None:
    from _fakes.trace_calls import _seed as seed_llm

    run = FakeRun(str(tmp_path / "runs"), "llm-run")
    seed_llm(run)
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()
        lines = trace(board, "c-llm")
        positions = [
            next(i for i, line in enumerate(lines) if text in line)
            for text in (
                "plan     ",
                "planner-call",
                "manager-call",
                "run      ",
                "tool-call",
                "check    ",
                "judge-call",
                "judge-child",
                "score    ",
                "learning ",
            )
        ]
        assert positions == sorted(positions)
        assert "tokens=120 cost=$0.000140" in "\n".join(lines)
        assert sum("llm judge-call " in line for line in lines) == 1
        assert "panel=survivin" in "\n".join(lines)
        assert "failed-probe" not in "\n".join(lines)
    finally:
        board.close()


def test_format_llm_step_marks_cache_hits(tmp_path: Path) -> None:
    from _fakes.trace_calls import _call

    from swarm_arena._board.trace import format_llm_step

    run = FakeRun(str(tmp_path / "runs"), "r")
    hit = _call(run, "hit", cache_hit=True, cached_from_call_id="llm-3f9a1c2b-e1-0001")
    assert format_llm_step(hit).endswith("cache-hit-of=llm-3f9a1c2b-e1-0001")
    unknown = hit.model_copy(update={"cached_from_call_id": None, "trace_url": "https://trace/1"})
    assert format_llm_step(unknown).endswith("https://trace/1")
    assert "cache-hit-of=" not in format_llm_step(unknown)


def test_cache_stats_per_agent(tmp_path: Path) -> None:
    from _fakes.trace_calls import _call

    from swarm_arena._board.trace import cache_stats

    run = FakeRun(str(tmp_path / "runs"), "r")
    stats = cache_stats(
        [
            _call(run, "a", team="fd-solver", agent="tuner-1"),
            _call(run, "b", "judge", cache_hit=True),
        ]
    )
    assert stats["fd-solver/tuner-1"].hit_rate == 0.5
    assert stats[":judge"].response_hits == 1
    assert stats[":judge"].calls == 1
    assert cache_stats([_call(run, "zero", usage=Usage())])[":manager"].hit_rate == 0


@pytest.mark.parametrize("reason", [None, "planning complete", "planner failed", "run budget"])
def test_trace_shows_planning_session_calls_for_r0(tmp_path: Path, reason: str | None) -> None:
    from _fakes.trace_calls import _call, _record

    root = str(tmp_path / "producer")
    session = FakeRun(root, "planning-session")
    session.commit(
        EventDraft(type=EventType.RUN_STARTED, payload={"planning": True}),
        EventDraft(type=EventType.GOAL_STARTED, goal_id="g1"),
        _record(_call(session, "session-planner", "planner")),
    )
    if reason is not None:
        session.commit(
            EventDraft(
                type=EventType.GOAL_FINISHED,
                goal_id="g1",
                payload={"reason": reason},
            )
        )
    run = FakeRun(root, "r")
    plan = run.plan()
    plan.payload.update(
        planning_session=session.run_id,
        planner_call_ids=["session-planner"],
        base_dir=str(tmp_path / "plan-source"),
    )
    run.commit(plan, run.candidate("c"))
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()
        lines = trace(board, "c")
        assert any("session-planner planner" in line for line in lines)
        assert "llm session-planner: not found" not in lines
        if reason:
            assert f"planning session planning-session: {reason}" in lines
        retained = board._read_state().plans["r"].revisions["g1"][0]
        assert retained.base_dir == tmp_path / "plan-source"
        session.commit(_record(_call(session, "session-late", "planner")))
        assert trace(board, "c") == lines
        board.frontier()
        assert trace(board, "c") == lines
        run.commit(EventDraft(type=EventType.WATCHDOG))
        board.frontier()
        assert any("session-late" in line for line in trace(board, "c"))
        # An autonomous revision uses its own recorded IDs, without a session link.
        revision = run.plan(revision=1, first_round=2)
        revision.payload.update(planning_session=None, planner_call_ids=["in-run-planner"])
        run.commit(
            _record(_call(run, "in-run-planner", "planner")),
            revision,
            run.candidate("c-next", round=2),
        )
        board.frontier()
        following = "\n".join(trace(board, "c-next"))
        assert "in-run-planner" in following
        assert "session-planner" not in following
    finally:
        board.close()


def test_trace_missing_session_is_safe_and_pinned(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r")
    draft = run.plan()
    draft.payload["planning_session"] = "missing"
    run.commit(draft, run.candidate("c"))
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()
        assert "planning session missing: not found" in trace(board, "c")
    finally:
        board.close()


def test_published_calls_are_source_aware_detached_and_namespace_unique(tmp_path: Path) -> None:
    from _fakes.trace_calls import _call, _record

    boards = []
    try:
        for index in range(2):
            directory = tmp_path / str(index)
            run = FakeRun(str(directory / "runs"), "same-run")
            run.commit(run.plan(), _record(_call(run, "same-call", model=f"local/{index}")))
            add_pointer(str(directory / "arena"), run, "toy")
            add_pointer(str(directory / "arena"), run, "alpha")
            board = ArenaBoard(
                arena_root=str(directory / "arena"), cache_dir=str(directory / "cache")
            )
            board.open("", "", ["toy"])
            board.open("", "", ["alpha"])
            boards.append(board)
            board.frontier()
            with board._read_namespace() as reader:
                calls = reader.llm_calls("same-run")
                assert [c.model for c in calls] == [f"local/{index}"]
                calls[0].tool_calls.append({"id": "mutated"})
                assert not reader.llm_calls("same-run")[0].tool_calls
    finally:
        for board in boards:
            board.close()


def test_trace_cli_core_only_uses_producer_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from contextvars import Context

    from _fakes.trace_calls import _call, _record
    from _fakes.trace_calls import _seed as seed_llm

    from swarm_arena import RunConfig
    from swarm_arena._cli.main import main

    run = FakeRun(str(tmp_path / "producer"), "cli-run")
    seed_llm(run)
    session = FakeRun(run.run_root, "session")
    session.commit(_record(_call(session, "right-session", "planner")))
    from swarm_arena import Goal

    run.goal = Goal(id="queued-goal", namespace="toy")
    revision = run.plan(first_round=1)
    revision.payload["planning_session"] = session.run_id
    run.commit(revision, run.candidate("queued-candidate"))
    add_pointer(str(tmp_path / "arena"), run)
    # The viewer config points somewhere else, with the same session identifier.
    config = RunConfig(
        run_root=str(tmp_path / "other"),
        arena_root=str(tmp_path / "arena"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    wrong = FakeRun(config.run_root, session.run_id)
    wrong.commit(_record(_call(wrong, "wrong-session", "planner")))
    path = tmp_path / "run.toml"
    path.write_text(config.dumps())
    with ThreadPoolExecutor(max_workers=1) as executor:
        code = executor.submit(
            Context().run,
            main,
            ["trace", "queued-candidate", "--namespace", "toy", "--config", str(path)],
        ).result(timeout=10)
    assert code == 0
    output = capsys.readouterr().out
    assert "right-session" in output
    assert "wrong-session" not in output


def test_llm_composite_read_performs_no_refresh_replay_or_authority_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.trace_calls import _seed as seed_llm

    from swarm_arena._board.trace import cache_stats

    run = FakeRun(str(tmp_path / "runs"), "r")
    seed_llm(run)
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("pinned LLM read performed preparation")

        monkeypatch.setattr(board, "frontier", forbidden)
        monkeypatch.setattr(ArenaCache, "_committed_state", forbidden)
        monkeypatch.setattr(RunState, "from_events", forbidden)
        monkeypatch.setattr("swarm_arena._board.arena.read_owner", forbidden)
        assert "planner-call" in "\n".join(trace(board, "c-llm"))
        with board._read_namespace() as reader:
            assert cache_stats(reader.llm_calls("r"))[":manager"].calls == 1
    finally:
        board.close()


def test_run_projection_memo_includes_source_root(tmp_path: Path) -> None:
    from _fakes.trace_calls import _call, _record

    board = ArenaBoard(arena_root=str(tmp_path / "arena"))
    caches = []
    try:
        for name in ("left", "right"):
            directory = tmp_path / name
            run = FakeRun(str(directory / "runs"), "same-run")
            head = run.commit(run.plan(), _record(_call(run, "same-call", model=f"local/{name}")))
            cache = ArenaCache(directory / "toy.sqlite")
            caches.append(cache)
            ingest_run(cache, run.store, run.run_id, run.run_root, head)
            mark = Frontier.Mark(run_id=run.run_id, run_root=run.run_root, **head.model_dump())
            projection = board._plans_at(cache, mark)
            assert projection.calls["same-call"].model == f"local/{name}"
            assert board._plans_at(cache, mark) is projection
    finally:
        for cache in caches:
            cache.close()
        board.close()


@pytest.mark.parametrize("damage", ["missing-segment", "owner", "call"])
def test_unreadable_planning_session_does_not_block_healthy_trace(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, damage: str
) -> None:
    import logging

    from _fakes.trace_calls import _call, _record

    caplog.set_level(logging.DEBUG, logger="swarm_arena.board")
    root = str(tmp_path / "runs")
    session = FakeRun(root, "damaged-session")
    call = _record(_call(session, "planner", "planner"))
    if damage == "call":
        call.payload["call"]["role"] = "session-secret-sentinel"
    first = session.commit(call)
    session.commit(EventDraft(type=EventType.WATCHDOG))
    if damage == "missing-segment":
        (Path(session.store.url) / first.segment).unlink()
    if damage == "owner":
        (Path(session.store.url) / "control/owner.json").write_bytes(b"session-secret-sentinel")
    linked = FakeRun(root, "linked")
    draft = linked.plan()
    draft.payload["planning_session"] = session.run_id
    linked.commit(draft, linked.candidate("c-linked"))
    healthy = FakeRun(root, "healthy")
    healthy.commit(healthy.plan(), healthy.candidate("c-healthy"))
    for run in (linked, healthy):
        add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        assert len(board.frontier().marks) == 2
        assert any("c-healthy" in line for line in trace(board, "c-healthy"))
        assert "planning session damaged-session: not found" in trace(board, "c-linked")
        assert "damaged-session" in caplog.text
        assert "session-secret-sentinel" not in caplog.text
        for _ in range(2):
            board.frontier()
        messages = [r for r in caplog.records if "cannot read planning session" in r.message]
        assert [r.levelno for r in messages] == [logging.WARNING, logging.DEBUG, logging.DEBUG]
        # Moving only the producing run does not create a new session head.
        linked.commit(EventDraft(type=EventType.WATCHDOG))
        board.frontier()
        assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1
        if damage != "owner":
            # A new unreadable session head gets its own warning.
            session.commit(EventDraft(type=EventType.WATCHDOG))
            board.frontier()
            assert sum(r.levelno == logging.WARNING for r in caplog.records) == 2
        assert "session-secret-sentinel" not in caplog.text
    finally:
        board.close()


def test_response_cache_hits_do_not_add_provider_tokens(tmp_path: Path) -> None:
    from _fakes.trace_calls import _call

    from swarm_arena._board.trace import cache_stats

    run = FakeRun(str(tmp_path / "runs"), "r")
    calls = [
        _call(run, "original"),
        _call(
            run,
            "hit",
            cache_hit=True,
            cost_usd=0,
            usage=Usage(input_tokens=1000, cache_read_tokens=1),
        ),
    ]
    row = cache_stats(calls)[":manager"]
    assert row.calls == 2
    assert row.response_hits == 1
    assert row.input_tokens == 25
    assert row.cache_read_tokens == 50
    assert row.cache_write_tokens == 25
    assert row.hit_rate == 0.5


def test_missing_recorded_call_links_are_explicit_and_deduplicated(tmp_path: Path) -> None:
    from _fakes.trace_calls import _candidate as linked_candidate

    run = FakeRun(str(tmp_path / "runs"), "r")
    plan = run.plan()
    plan.payload["planner_call_ids"] = ["planner-missing", "planner-missing"]
    score = run.score("score", "c-llm", 1)
    score.payload["score"]["llm_call_ids"] = ["judge-missing"]
    run.commit(
        plan,
        linked_candidate(run),
        score,
        run.learning(
            "learning", "Missing link", evidence=["c-llm"], posted_by_call="posting-missing"
        ),
    )
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()
        lines = trace(board, "c-llm")
        for identifier in ("planner-missing", "manager-call", "judge-missing", "posting-missing"):
            assert lines.count(f"llm {identifier}: not found") == 1
    finally:
        board.close()


def test_session_read_is_reused_until_producing_mark_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.trace_calls import _call, _record

    from swarm_arena._board import arena

    run = FakeRun(str(tmp_path / "runs"), "r")
    session = FakeRun(run.run_root, "session")
    session.commit(_record(_call(session, "first", "planner")))
    plan = run.plan()
    plan.payload["planning_session"] = session.run_id
    run.commit(plan, run.candidate("c"))
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    original = arena.read_owner
    reads = []

    def owner(store: Any) -> Any:
        if store.url == session.store.url:
            reads.append(1)
        return original(store)

    monkeypatch.setattr(arena, "read_owner", owner)
    try:
        board.frontier()
        expected = trace(board, "c")
        session.commit(_record(_call(session, "later", "planner")))
        board.frontier()
        assert reads == [1]
        assert trace(board, "c") == expected
        run.commit(EventDraft(type=EventType.WATCHDOG))
        board.frontier()
        assert reads == [1, 1]
        assert any("llm later " in line for line in trace(board, "c"))
    finally:
        board.close()


def test_score_call_lookup_uses_score_producing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.trace_calls import _call, _record

    candidate_run = FakeRun(str(tmp_path / "runs"), "candidate-run")
    candidate_run.commit(candidate_run.plan(), candidate_run.candidate("c"))
    score_run = FakeRun(candidate_run.run_root, "score-run")
    score_run.commit(score_run.plan(), _record(_call(score_run, "score-call", "judge")))
    for run in (candidate_run, score_run):
        add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()
        cache = board._read_state().readers["toy"]
        score = Score(
            id="cross-run-score",
            candidate_id="c",
            judge="recorded",
            key=(1,),
            llm_call_ids=["score-call"],
        )
        monkeypatch.setattr(
            cache, "scores", lambda *args, **kwargs: [(score, ("run", "score-run", 2))]
        )
        assert any("llm score-call judge " in line for line in trace(board, "c"))
    finally:
        board.close()


@pytest.mark.parametrize("kind", ["fenced", "terminal", "cancel", "storage", "os", "value"])
def test_session_failure_boundary_preserves_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    import asyncio

    from swarm_arena._board import trace as trace_module
    from swarm_arena._errors import (
        Fenced,
        StorageError,
        _core_category,
        _core_error,
        _SignalCategory,
    )

    run = FakeRun(str(tmp_path / "runs"), "r")
    session = FakeRun(run.run_root, "session")
    session.commit(EventDraft(type=EventType.WATCHDOG))
    plan = run.plan()
    plan.payload["planning_session"] = session.run_id
    run.commit(plan, run.candidate("c"))
    add_pointer(str(tmp_path / "arena"), run)
    failure: BaseException
    if kind == "fenced":
        failure = _core_error(Fenced, "run ownership lost")
    elif kind == "terminal":
        failure = _core_error(
            StorageError, "session read stopped", _category=_SignalCategory.TERMINAL
        )
    elif kind == "cancel":
        failure = asyncio.CancelledError()
    else:
        failure = {"storage": StorageError, "os": OSError, "value": ValueError}[kind](
            "session-failure-secret"
        )

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(trace_module, "_calls_at", fail)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        if kind in {"fenced", "terminal", "cancel"}:
            with pytest.raises(type(failure)) as caught:
                board.frontier()
            if kind == "cancel":
                assert caught.value is failure
            else:
                assert _core_category(caught.value) == _core_category(failure)
        else:
            board.frontier()
            assert "planning session session: not found" in trace(board, "c")
            assert "session-failure-secret" not in caplog.text
    finally:
        board.close()


def test_llm_visibility_uses_retained_source_namespace_provenance(tmp_path: Path) -> None:
    from _fakes.trace_calls import _call, _record

    run = FakeRun(str(tmp_path / "runs"), "private-run", namespace="private")
    run.commit(_record(_call(run, "call", team="private-team", agent="private-agent")))
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["public", "private"])
    try:
        board.frontier()
        with board._read_namespace() as reader:
            with pytest.raises(KeyError):
                reader.llm_calls(run.run_id, namespaces=["public"])
            add_pointer(str(tmp_path / "arena"), run, "public")
            board.frontier()
            with pytest.raises(KeyError):
                reader.llm_calls(run.run_id, namespaces=["public"])
            assert reader.llm_calls(run.run_id, namespaces=["private"])[0].id == "call"
        with board._read_namespace() as reader:
            assert reader.llm_calls(run.run_id, namespaces=["public"])[0].id == "call"
    finally:
        board.close()


def test_llm_visibility_does_not_merge_equal_run_ids_across_sources(tmp_path: Path) -> None:
    from _fakes.trace_calls import _call, _record

    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    for namespace in ("public", "private"):
        run = FakeRun(str(tmp_path / namespace), "same-run", namespace=namespace)
        run.commit(_record(_call(run, "same-call", team=namespace, agent="agent")))
        add_pointer(str(tmp_path / "arena"), run)
    board.open("", "", ["public", "private"])
    try:
        board.frontier()  # The existing frontier policy selects the private pointer first.
        with board._read_namespace() as reader:
            with pytest.raises(KeyError):
                reader.llm_calls("same-run", namespaces=["public"])
            assert reader.llm_calls("same-run", namespaces=["private"])[0].team == "private"
    finally:
        board.close()


def test_session_read_error_is_retried_at_unchanged_producing_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.trace_calls import _call, _record

    from swarm_arena._board import arena
    from swarm_arena._errors import StorageError

    run = FakeRun(str(tmp_path / "runs"), "r")
    session = FakeRun(run.run_root, "session")
    session.commit(_record(_call(session, "planner-call", "planner")))
    plan = run.plan()
    plan.payload["planning_session"] = session.run_id
    run.commit(plan, run.candidate("c"))
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    original = arena.read_owner
    reads: list[int] = []

    def owner(store: Any) -> Any:
        if store.url == session.store.url:
            reads.append(1)
            if len(reads) == 1:
                raise StorageError("TRANSIENT_SESSION_SECRET")
        return original(store)

    monkeypatch.setattr(arena, "read_owner", owner)
    try:
        first = board.frontier()
        assert "planning session session: not found" in trace(board, "c")
        assert board.frontier() == first
        assert any("llm planner-call " in line for line in trace(board, "c"))
        assert reads == [1, 1]
        board.frontier()
        assert reads == [1, 1]
    finally:
        board.close()


@pytest.mark.parametrize(
    ("observed", "label", "expected"),
    [
        (None, "estimated", "1.00 USD (estimated)"),
        (
            0.012345,
            "observed (completeness unverified)",
            "0.0123 USD (observed (completeness unverified))",
        ),
        (0.012345, "observed", "0.0123 USD (observed)"),
    ],
)
def test_trace_shows_gateway_cap(
    tmp_path: Path,
    observed: float | None,
    label: str,
    expected: str,
) -> None:
    from _fakes.gateway_events import gateway_key, reconcile

    run = _seed(tmp_path)
    run.commit(gateway_key())
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()
        with board._read_namespace() as old:
            before = old.trace(["toy"], "c-1")
            assert "gateway key cand-c-1: max_budget 1.00 USD, cap soft" in before
            assert not any(line.startswith("gateway spend") for line in before)
            run.commit(reconcile(observed=observed, label=label))
            assert trace(board, "c-1") == before
            board.frontier()
            assert old.trace(["toy"], "c-1") == before
            after = trace(board, "c-1")
            assert f"gateway spend {expected}" in after
    finally:
        board.close()


def test_gateway_trace_filters_candidate_key_and_producing_run(tmp_path: Path) -> None:
    from _fakes.gateway_events import gateway_key, reconcile

    run = _seed(tmp_path)
    run.commit(gateway_key(), reconcile(key_id="foreign-hash", observed=99, label="observed"))
    other = FakeRun(str(tmp_path / "runs"), "r2")
    key = gateway_key(cap="hard")
    key.payload["run_id"] = "r2"
    other.commit(
        _plan(other), _candidate(other), key, reconcile(observed=99, label="observed", run_id="r2")
    )
    add_pointer(str(tmp_path / "arena"), other)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()
        lines = trace(board, "c-1")
        assert lines[0].startswith("plan     r1 ")
        assert "gateway key cand-c-1: max_budget 1.00 USD, cap soft" in lines
        assert not any(line.startswith("gateway spend") for line in lines)
    finally:
        board.close()


def test_gateway_trace_uses_only_published_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _fakes.gateway_events import gateway_key, reconcile

    run = _seed(tmp_path)
    run.commit(gateway_key(), reconcile())
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    try:
        board.frontier()

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("gateway trace performed preparation")

        monkeypatch.setattr(board, "frontier", forbidden)
        monkeypatch.setattr(ArenaCache, "_committed_state", forbidden)
        monkeypatch.setattr(RunState, "from_events", forbidden)
        monkeypatch.setattr("swarm_arena._board.arena.read_owner", forbidden)
        assert "gateway spend 1.00 USD (estimated)" in trace(board, "c-1")
    finally:
        board.close()


def test_gateway_trace_does_not_merge_equal_run_ids_across_namespaces(tmp_path: Path) -> None:
    from _fakes.gateway_events import gateway_key, reconcile

    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    for namespace, cap in (("public", "hard"), ("private", "soft")):
        run = FakeRun(str(tmp_path / namespace), "same-run", namespace=namespace)
        created = gateway_key(cap=cap)
        created.payload["run_id"] = "same-run"
        run.commit(run.plan(), run.candidate("c-1"), created, reconcile(run_id="same-run"))
        add_pointer(str(tmp_path / "arena"), run)
    board.open("", "", ["public", "private"])
    try:
        board.frontier()
        with board._read_namespace() as reader:
            with pytest.raises(KeyError):
                reader.trace(["public"], "c-1")
            assert "gateway key cand-c-1: max_budget 1.00 USD, cap soft" in reader.trace(
                ["private"], "c-1"
            )
    finally:
        board.close()
