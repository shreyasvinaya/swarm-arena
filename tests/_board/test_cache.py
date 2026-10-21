import gc
import sqlite3
import traceback
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier
from threading import Event as ThreadEvent
from time import monotonic

import pytest
from _fakes.journals import START, FakeRun, candidate_ids
from _fakes.secrecy import assert_secret_absent

from swarm_arena._board.cache import ArenaCache, LearningFilter, fts_query, goal_key_digest
from swarm_arena._board.frontier import build_frontier
from swarm_arena._board.human_objects import HumanObject
from swarm_arena._board.ingest import ingest_run
from swarm_arena._errors import ChainError
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.results import Learning, Score


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(), run.candidate("c-1"))
    cache = ArenaCache(tmp_path / "cache" / "toy.sqlite")
    segs = read_chain(run.store, head)
    cache.ingest("r1", run.run_root, segs)
    cache.ingest("r1", run.run_root, segs)
    assert candidate_ids(cache) == ["c-1"]
    assert cache.run_root("r1") == run.run_root
    assert cache.newest_segment("missing") is None
    assert cache.run_root("missing") is None
    cache.close()
    cache.close()


def test_ingest_keeps_repeat_key(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(), run.candidate("c-1", repeat_key="k1"), run.candidate("c-2"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    assert cache._conn.execute("SELECT repeat_key FROM candidates ORDER BY id").fetchall() == [
        ("k1",),
        (None,),
    ]
    cache.close()


def test_keeps_only_this_namespace(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    for ns in ("deepretro", "toy"):
        run = FakeRun(str(tmp_path / "runs"), ns, namespace=ns)
        head = run.commit(
            run.plan(),
            run.candidate(ns),
            run.score("s-" + ns, ns, 1),
            run.learning("L-" + ns, "Hint", evidence=[ns]),
        )
        ingest_run(cache, run.store, ns, run.run_root, head)
    assert candidate_ids(cache) == ["toy"]
    assert cache._conn.execute("SELECT count(*) FROM goals").fetchone() == (2,)
    assert cache._conn.execute("SELECT count(*) FROM scores").fetchone() == (1,)
    assert cache._conn.execute("SELECT count(*) FROM learnings").fetchone() == (1,)
    cache.close()


def test_rejects_gap_at_junction(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan())
    run.commit(run.candidate("c-1"))
    head = run.commit(run.candidate("c-2"))
    segs = read_chain(run.store, head)
    cache = ArenaCache(tmp_path / "toy.sqlite")
    cache.ingest("r1", run.run_root, segs[:1])
    with pytest.raises(ChainError):
        cache.ingest("r1", run.run_root, segs[2:])
    cache.close()


def test_rejects_segment_with_other_seqs(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(), run.candidate("c-1"))
    segs = read_chain(run.store, head)
    cache = ArenaCache(tmp_path / "toy.sqlite")
    cache.ingest("r1", run.run_root, segs)
    with pytest.raises(ChainError):
        cache.ingest("r1", run.run_root, [(head.segment, segs[0][1][1:])])
    cache.close()


def test_indexed_event_before_plan_raises(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.candidate("c-1"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    with pytest.raises(ChainError):
        ingest_run(cache, run.store, "r1", run.run_root, head)
    cache.close()


def test_failed_segment_rolls_back(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1", goal_id="g0")
    bad = run.candidate("c-1")
    bad.payload["candidate"]["goal_id"] = "g1"
    head = run.commit(run.plan(), bad.model_copy(update={"goal_id": "g1"}))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    with pytest.raises(ChainError):
        ingest_run(cache, run.store, "r1", run.run_root, head)
    for table in ("goals", "plans", "candidates", "segments"):
        assert cache._conn.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
    assert not cache._conn.in_transaction
    valid = FakeRun(str(tmp_path / "runs"), "r2")
    ingest_run(cache, valid.store, "r2", valid.run_root, valid.commit(valid.plan()))
    cache.close()


def test_part_candidates_get_part_key(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(parts=["a"]), run.candidate("c-1", part="a"), run.candidate("c-2"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    rows = cache._conn.execute(
        "SELECT id, goal_key, part_key FROM candidates ORDER BY id"
    ).fetchall()
    main = goal_key_digest(run.goal.key())
    part = goal_key_digest(run.goal.model_copy(update={"id": "g1-a"}).key())
    assert rows == [("c-1", main, part), ("c-2", main, main)]
    cache.close()


@pytest.mark.parametrize("change", ["inputs", "manifest"])
def test_producing_identity_matches_core_and_rebuild(tmp_path: Path, change: str) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    initial = run.commit(run.plan(parts=["a"]), run.candidate("old", part="a"))
    revision = run.plan(revision=1, parts=["a"], first_round=3)
    part = revision.payload["plan"]["parts"][0]["goal"]
    if change == "inputs":
        part["inputs"] = {"changed": True}
    else:
        part["files"] = {"globs": [], "manifest": {"input.txt": "a" * 64}}
    head = run.commit(
        revision, run.candidate("late", part="a", round=2), run.candidate("new", part="a", round=3)
    )
    cache = ArenaCache(tmp_path / "incremental" / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, initial)
    cache.close()
    cache = ArenaCache(tmp_path / "incremental" / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    rebuilt = ArenaCache(tmp_path / "rebuilt" / "toy.sqlite")
    ingest_run(rebuilt, run.store, "r1", run.run_root, head)
    expected = RunState.from_events(
        [e for _, events in read_chain(run.store, head) for e in events]
    )
    rows = cache._conn.execute("SELECT id, part_key FROM candidates ORDER BY id").fetchall()
    assert rows == sorted(
        (cid, goal_key_digest(key)) for cid, key in expected.candidate_goal_keys.items()
    )
    assert (
        rows == rebuilt._conn.execute("SELECT id, part_key FROM candidates ORDER BY id").fetchall()
    )
    assert dict(rows)["old"] == dict(rows)["late"] != dict(rows)["new"]
    assert cache._plan_for_round("r1", "g1", 2, through_seq=head.seq).revision == 0
    assert cache._plan_for_round("r1", "g1", 3, through_seq=head.seq).revision == 1
    assert cache._plan_for_round("r1", "g1", 3, through_seq=initial.seq).revision == 0
    assert cache._conn.execute("SELECT first_round FROM plans ORDER BY revision").fetchall() == [
        (1,),
        (3,),
    ]
    cache.close()
    rebuilt.close()


def test_id_from_other_run_is_skipped_with_warning(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    for rid in ("r1", "r2"):
        run = FakeRun(str(tmp_path / "runs"), rid)
        head = run.commit(
            run.plan(),
            run.candidate("c-1", team=rid),
            run.score("s-1", "c-1", 1),
            run.learning("L-1", "Hint", evidence=["c-1"], tags=["tag"], refines=["L-0"]),
        )
        ingest_run(cache, run.store, rid, run.run_root, head)
        ingest_run(cache, run.store, rid, run.run_root, head)
    assert cache.warnings == [
        f"{kind} {id} of run r2 skipped: the cache holds it from run r1"
        for kind, id in [("candidate", "c-1"), ("score", "s-1"), ("learning", "L-1")]
    ]
    assert cache._conn.execute("SELECT run_id FROM candidates").fetchall() == [("r1",)]
    assert cache._conn.execute("SELECT * FROM learning_tags").fetchall() == [("L-1", "tag")]
    cache.close()


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "toy.sqlite"
    ArenaCache(path).close()
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="reindex"):
        ArenaCache(path)


def test_missing_fts5_error_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("swarm_arena._board.cache._has_fts5", lambda conn: False)
    with pytest.raises(RuntimeError, match="FTS5") as err:
        ArenaCache(tmp_path / "toy.sqlite")
    assert "reindex" not in str(err.value)


def test_two_caches_share_one_file_concurrently(tmp_path: Path) -> None:
    runs = [FakeRun(str(tmp_path / "runs"), rid) for rid in ("r1", "r2")]
    for run in runs:
        for i in range(20):
            drafts = [run.candidate(f"{run.run_id}-{i}-{j}") for j in range(10)]
            run.commit(*([run.plan()] if i == 0 else []), *drafts)
    barrier = Barrier(2)
    path = tmp_path / "fresh" / "toy.sqlite"

    def worker(reverse: bool) -> None:
        barrier.wait(timeout=10)
        cache = ArenaCache(path)
        try:
            for run in runs[::-1] if reverse else runs:
                ingest_run(cache, run.store, run.run_id, run.run_root, run.heads[-1])
        finally:
            cache.close()

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(worker, reverse) for reverse in (False, True)]
        for future in futures:
            future.result(timeout=60)
    cache = ArenaCache(path)
    assert len(candidate_ids(cache)) == 400
    assert cache._conn.execute("PRAGMA user_version").fetchone() == (cache.SCHEMA_VERSION,)
    cache.close()


def test_invalid_payload_is_secret_safe(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    draft = run.candidate("c")
    draft.payload["candidate"]["round"] = "SENTINEL_SECRET"
    head = run.commit(run.plan(), draft)
    cache = ArenaCache(tmp_path / "toy.sqlite")
    with pytest.raises(ChainError) as caught:
        ingest_run(cache, run.store, "r1", run.run_root, head)
    assert_secret_absent(caught.value, "SENTINEL_SECRET")
    assert "SENTINEL_SECRET" not in "".join(traceback.format_exception(caught.value))
    cache.close()


def test_duplicate_id_keeps_first_producing_identity(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    revision = run.plan(revision=1, parts=["a"], first_round=2)
    revision.payload["plan"]["parts"][0]["goal"]["inputs"] = {"changed": True}
    head = run.commit(
        run.plan(parts=["a"]),
        run.candidate("same", part="a"),
        revision,
        run.candidate("same", part="a", round=2),
    )
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    expected = goal_key_digest(run.goal.model_copy(update={"id": "g1-a"}).key())
    assert cache._conn.execute("SELECT part_key FROM candidates").fetchone() == (expected,)
    cache.close()


def test_lock_lifetime_and_failed_open_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    path = tmp_path / "toy.sqlite"
    cache = ArenaCache(path)
    lock_path = Path(str(path) + ".lock")
    with lock_path.open("rb") as handle:
        with pytest.raises(BlockingIOError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache.close()
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert lock_path.exists()
    monkeypatch.setattr("swarm_arena._board.cache._has_fts5", lambda conn: False)
    with pytest.raises(RuntimeError, match="FTS5"):
        ArenaCache(path)
    with lock_path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_removed_part_is_historical_only(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan(parts=["a"]), run.candidate("old", part="a"))
    head = run.commit(run.plan(revision=1, first_round=2))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    assert candidate_ids(cache) == ["old"]
    assert cache._plan_for_round("r1", "g1", 2, through_seq=head.seq).parts == []
    assert [p.name for p in cache._plan_for_round("r1", "g1", 1, through_seq=head.seq).parts] == [
        "a"
    ]
    bad = run.commit(run.candidate("undeclared", part="a", round=2))
    with pytest.raises(ChainError):
        ingest_run(cache, run.store, "r1", run.run_root, bad)
    assert candidate_ids(cache) == ["old"]
    assert cache.newest_segment("r1") == head.segment
    cache.close()


def test_warning_and_learning_children_roll_back(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    first = FakeRun(str(tmp_path / "runs"), "r1")
    ingest_run(
        cache, first.store, "r1", first.run_root, first.commit(first.plan(), first.candidate("c"))
    )
    second = FakeRun(str(tmp_path / "runs"), "r2")
    bad = second.score("bad", "c", 0).model_copy(update={"goal_id": "missing"})
    head = second.commit(
        second.plan(),
        second.candidate("c"),
        second.learning(
            "L", "Hint", evidence=["c"], tags=["tag"], refines=["previous"], supersedes=["other"]
        ),
        bad,
    )
    with pytest.raises(ChainError):
        ingest_run(cache, second.store, "r2", second.run_root, head)
    assert cache.warnings == []
    for table in ("learning_tags", "learning_links", "learning_evidence", "learnings"):
        assert cache._conn.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
    assert not cache._conn.in_transaction
    cache.close()


def test_fake_journal_preserves_producer_payloads_and_core_state(tmp_path: Path) -> None:
    from _fakes.journals import START, add_pointer

    from swarm_arena import Goal
    from swarm_arena._board.pointers import list_pointers
    from swarm_arena._journal.event import EventDraft, EventType
    from swarm_arena._store.base import open_store

    run = FakeRun(
        str(tmp_path / "runs"), "r1", goal=Goal(id="custom", namespace="toy", inputs={"x": 1})
    )
    cid = "c-0123456789abcdef-e1-0007"
    run.commit(
        run.plan(parts=["a"]),
        run.candidate(
            cid,
            part="a",
            fingerprint="fp",
            repeat_key="repeat",
            partial=True,
            passed=False,
            parents=["parent"],
        ),
    )
    learning = run.learning("L-0123456789abcdef-e1-0002", "Hint", posted_by_call="llm-call")
    closed = EventDraft(
        type=EventType.ROUND_CLOSED,
        goal_id="custom",
        round=1,
        payload={"round": 1, "run_elapsed_s": 5.5, "replan": True, "reason": None},
    )
    failed = EventDraft(
        type=EventType.JOIN_FAILED,
        goal_id="custom",
        payload={"attempt": 1, "reasons": ["incomplete"]},
    )
    rejected = EventDraft(
        type=EventType.PLAN_REJECTED,
        goal_id="custom",
        payload={
            "errors": ["invalid"],
            "origin": "human",
            "first_round": 2,
            "join_retry": 1,
            "run_elapsed_s": 9.0,
        },
    )
    counters = EventDraft(
        type=EventType.AGENT_COUNTERS,
        goal_id="custom",
        team="t1",
        agent="a1",
        payload={"turns": 1, "board_reads": 2, "tool_runs": 3},
    )
    partial = counters.model_copy(update={"payload": {"turns": 2, "board_reads": 4}})
    run.commit(learning, counters, partial, closed, failed, rejected)
    revision = run.plan(revision=1, parts=["a"], join_retry=2)
    revision.payload["run_elapsed_s"] = 6.0
    revision.payload["future_metadata"] = {"retained": True}
    head = run.commit(revision)
    add_pointer(str(tmp_path / "arena"), run)
    pointers, warnings = list_pointers(open_store(str(tmp_path / "arena")), "toy")
    assert [(p.run_id, p.run_root) for p in pointers] == [("r1", run.run_root)]
    assert warnings == []
    events = [e for _, batch in read_chain(run.store, head) for e in batch]
    assert all(e.context_id == "r1" and e.run_id == "r1" and e.schema_version == 1 for e in events)
    assert (events[0].ts - START).total_seconds() == 1
    completion = events[1]
    assert completion.task_id == cid
    assert completion.payload["a2a"]["id"] == cid
    assert completion.payload["fingerprint"] == "fp"
    assert learning.task_id == "custom/t1/a1#r1"
    assert learning.payload["learning"]["evidence"] == [cid]
    assert learning.payload["learning"]["posted_by_call"] == "llm-call"
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    state = cache._replay("r1")
    assert state == RunState.from_events(events)
    assert state.id_counters["c"] == 7
    assert cache._replay("r1", through_seq=5).goals["custom"].counters["t1/a1"] == {
        "turns": 2,
        "board_reads": 4,
        "tool_runs": 3,
    }
    assert state.goals["custom"].join_retries == 2
    assert state.goals["custom"].join_phase == "rounds"
    assert state.run_elapsed_s == 9.0
    assert state.plan_first_rounds["custom"] == [1, 2]
    assert cache._conn.execute("SELECT partial, gated_out FROM candidates").fetchone() == (1, 1)
    assert cache._conn.execute("SELECT ts FROM candidates").fetchone()[0].endswith(".000000+00:00")
    assert (
        "future_metadata"
        in cache._conn.execute("SELECT json FROM _events WHERE seq=?", (head.seq,)).fetchone()[0]
    )
    cache.close()


def test_fake_learning_requires_visible_default_evidence(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    with pytest.raises(ValueError, match="evidence"):
        run.learning("L", "No evidence")
    human = run.learning("L", "Human note", author="human:alice")
    assert human.payload["learning"]["evidence"] == []
    assert run.score("s", "c", 2, minimize=False).payload["score"]["key"] == [2.0]
    with pytest.raises(ValueError, match="requires events"):
        run.commit()
    with pytest.raises(ValueError, match="requires events"):
        run.orphan()


def test_one_cache_serializes_whole_transactions(tmp_path: Path) -> None:
    from threading import Event

    from swarm_arena._journal.event import Event as JournalEvent

    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(), run.candidate("c"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    entered, release, reader_started = Event(), Event(), Event()
    original = cache._index

    def index(event: JournalEvent, state: RunState, pending: list[str]) -> None:
        entered.set()
        assert release.wait(timeout=5)
        original(event, state, pending)

    def read() -> list[str]:
        reader_started.set()
        return candidate_ids(cache)

    with pytest.MonkeyPatch.context() as patch, ThreadPoolExecutor(2) as pool:
        patch.setattr(cache, "_index", index)
        writer = pool.submit(ingest_run, cache, run.store, "r1", run.run_root, head)
        try:
            assert entered.wait(timeout=5)
            reader = pool.submit(read)
            assert reader_started.wait(timeout=5)
            assert not reader.done()
        finally:
            release.set()
        assert writer.result(timeout=5) == 2
        assert reader.result(timeout=5) == ["c"]
    cache.close()


@pytest.mark.parametrize("failure", ["fts5", "schema", "chain"])
def test_cache_diagnostics_keep_registered_cli_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from swarm_arena._cli.main import _error_message
    from swarm_arena._errors import _rebuild_core_signal

    path = tmp_path / "toy.sqlite"
    if failure == "fts5":
        monkeypatch.setattr("swarm_arena._board.cache._has_fts5", lambda conn: False)
        with pytest.raises(RuntimeError) as caught:
            ArenaCache(path)
    elif failure == "schema":
        ArenaCache(path).close()
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA user_version=99")
        finally:
            conn.close()
        with pytest.raises(RuntimeError) as caught:
            ArenaCache(path)
    else:
        cache = ArenaCache(path)
        run = FakeRun(str(tmp_path / "runs"), "r1")
        try:
            with pytest.raises(ChainError) as caught:
                ingest_run(cache, run.store, "r1", run.run_root, run.commit(run.candidate("c")))
        finally:
            cache.close()
    error = caught.value
    expected = str(error)
    error.args = ("MUTATED-SECRET",)
    error.add_note("MUTATED-SECRET")
    error.__cause__ = ValueError("MUTATED-SECRET")
    rebuilt = _rebuild_core_signal(error)
    assert rebuilt is not None
    assert type(rebuilt) is type(error)
    assert str(rebuilt) == expected
    assert_secret_absent(rebuilt, "MUTATED-SECRET", sanitized=True)
    assert "MUTATED-SECRET" not in "".join(traceback.format_exception(rebuilt))
    assert _error_message(error) == expected
    if failure == "fts5":
        assert "FTS5" in expected
        assert "reindex" not in expected
    elif failure == "schema":
        assert "swarm-arena reindex" in expected


@pytest.mark.parametrize("size", [50, 100, 200])
def test_incremental_ingest_folds_each_event_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    from swarm_arena._journal.event import Event

    run = FakeRun(str(tmp_path / "runs"), "r1")
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, run.commit(run.plan()))
    folded: list[int] = []
    original = RunState._apply

    def apply(state: RunState, event: Event) -> None:
        folded.append(event.seq)
        original(state, event)

    monkeypatch.setattr(RunState, "_apply", apply)
    try:
        for i in range(size):
            head = run.commit(run.candidate(f"c-{i}"))
            assert ingest_run(cache, run.store, "r1", run.run_root, head) == 1
        assert folded == list(range(2, size + 2))
    finally:
        cache.close()


def test_incremental_replay_catches_up_only_other_connection_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._journal.event import Event

    run = FakeRun(str(tmp_path / "runs"), "r1")
    path = tmp_path / "toy.sqlite"
    cache, other = ArenaCache(path), ArenaCache(path)
    try:
        initial = run.commit(run.plan(parts=["a"]), run.candidate("old", part="a"))
        ingest_run(cache, run.store, "r1", run.run_root, initial)
        revision = run.plan(revision=1, parts=["a"], first_round=2)
        revision.payload["plan"]["parts"][0]["goal"]["inputs"] = {"changed": True}
        middle = run.commit(revision, run.candidate("middle", part="a", round=2))
        ingest_run(other, run.store, "r1", run.run_root, middle)
        # A cached-head hit need not refresh the private replay checkpoint.
        assert ingest_run(cache, run.store, "r1", run.run_root, middle) == 0
        head = run.commit(run.candidate("new", part="a", round=2))
        folded: list[int] = []
        original = RunState._apply

        def apply(state: RunState, event: Event) -> None:
            folded.append(event.seq)
            original(state, event)

        with monkeypatch.context() as patch:
            patch.setattr(RunState, "_apply", apply)
            assert ingest_run(cache, run.store, "r1", run.run_root, head) == 1
        assert folded == [3, 4, 5]
        expected = RunState.from_events(e for _, es in read_chain(run.store, head) for e in es)
        rows = cache._conn.execute("SELECT id, part_key FROM candidates ORDER BY id").fetchall()
        assert rows == sorted(
            (cid, goal_key_digest(key)) for cid, key in expected.candidate_goal_keys.items()
        )
        # Historical lookups remain independent from the retained ingest state.
        snapshot = cache._replay("r1")
        snapshot.plans.clear()
        head = run.commit(run.candidate("next", part="a", round=2))
        assert ingest_run(cache, run.store, "r1", run.run_root, head) == 1
    finally:
        cache.close()
        other.close()


def test_incremental_replay_discards_uncommitted_state_on_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._journal.event import Event, EventType

    run = FakeRun(str(tmp_path / "runs"), "r1")
    cache = ArenaCache(tmp_path / "toy.sqlite")
    initial = run.commit(run.plan(parts=["a"]), run.candidate("old", part="a"))
    ingest_run(cache, run.store, "r1", run.run_root, initial)
    head = run.commit(
        run.plan(revision=1, parts=["a"], first_round=2),
        run.candidate("new", part="a", round=2),
    )
    original = cache._index
    fail = True
    revisions: list[list[int]] = []

    def index(event: Event, state: RunState, pending: list[str]) -> None:
        if event.type == EventType.TOOL_END:
            revisions.append([p.revision for p, _ in state.plans["g1"]])
            if fail:
                raise RuntimeError("injected transaction failure")
        original(event, state, pending)

    monkeypatch.setattr(cache, "_index", index)
    try:
        with pytest.raises(RuntimeError, match="injected transaction failure"):
            ingest_run(cache, run.store, "r1", run.run_root, head)
        assert not cache._conn.in_transaction
        assert cache.newest_segment("r1") == initial.segment
        assert candidate_ids(cache) == ["old"]
        fail = False
        assert ingest_run(cache, run.store, "r1", run.run_root, head) == 2
        assert revisions == [[0, 1], [0, 1]]
        assert candidate_ids(cache) == ["new", "old"]
    finally:
        cache.close()


@pytest.mark.parametrize("existing", [False, True])
def test_segment_extrema_use_sequence_index(tmp_path: Path, existing: bool) -> None:
    path = tmp_path / "toy.sqlite"
    if existing:
        original = ArenaCache(path)
        # Simulate a version-1 database created before the supporting index.
        original._conn.execute("DROP INDEX IF EXISTS segments_run_last_seq")
        original.close()
    cache = ArenaCache(path)
    try:
        for sql in (
            "SELECT epoch,last_seq FROM segments WHERE run_id=? ORDER BY last_seq DESC LIMIT 1",
            "SELECT name FROM segments WHERE run_id=? ORDER BY last_seq DESC LIMIT 1",
            "SELECT max(last_seq) FROM segments WHERE run_id=?",
        ):
            plan = cache._conn.execute("EXPLAIN QUERY PLAN " + sql, ("r",)).fetchall()
            details = " ".join(row[3] for row in plan)
            assert "segments_run_last_seq" in details, details
            assert "TEMP B-TREE" not in details, details
    finally:
        cache.close()


def test_bulk_ingest_sql_work_grows_linearly(tmp_path: Path) -> None:
    from _fakes.journals import START

    from swarm_arena._journal.event import Event, EventType

    work: list[int] = []
    sizes = (500, 1000, 2000)
    for size in sizes:
        cache = ArenaCache(tmp_path / str(size) / "toy.sqlite")
        ticks = 0
        junctions = 0

        def progress() -> int:
            nonlocal ticks
            ticks += 1
            return 0

        def trace(sql: str) -> None:
            nonlocal junctions
            if sql.startswith("SELECT epoch, last_seq FROM segments"):
                junctions += 1

        segments = [
            (
                f"journal/e1-{seq}.jsonl",
                [
                    Event(
                        type=EventType.TAKEOVER,
                        run_id="r",
                        context_id="r",
                        epoch=1,
                        seq=seq,
                        ts=START,
                        payload={},
                    )
                ],
            )
            for seq in range(1, size + 1)
        ]
        try:
            cache._conn.set_progress_handler(progress, 1000)
            cache._conn.set_trace_callback(trace)
            cache.ingest("r", str(tmp_path), segments)
            cache._conn.set_progress_handler(None, 0)
            cache._conn.set_trace_callback(None)
            assert junctions == size
            work.append(ticks * 1000)
        finally:
            cache.close()
    assert work[1] <= work[0] * 2.5, (sizes, work)
    assert work[2] <= work[1] * 2.5, (sizes, work)


def test_items_after_mark_are_hidden(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    h1 = run.commit(
        run.plan(),
        run.candidate("c-1"),
        run.score("s-1", "c-1", 1),
        run.learning("L-1", "First", evidence=["c-1"]),
    )
    h2 = run.commit(
        run.candidate("c-2"),
        run.score("s-2", "c-2", 2),
        run.learning("L-2", "Second", evidence=["c-2"]),
        run.plan(revision=1, first_round=2),
    )
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, h2)
    f1 = build_frontier([("r1", run.run_root, h1)], [], [])
    f2 = build_frontier([("r1", run.run_root, h2)], [], [])
    assert [c.id for c, _ in cache.candidates(f1)] == ["c-1"]
    assert [c.id for c, _ in cache.candidates(f2)] == ["c-1", "c-2"]
    assert [s.id for s, _ in cache.scores(f1)] == ["s-1"]
    assert [s.id for s, _ in cache.scores(f2)] == ["s-1", "s-2"]
    assert [learning.id for learning in cache.learnings_by_id(f1, ["L-1", "L-2"])] == ["L-1"]
    assert [learning.id for learning in cache.learnings_by_id(f2, ["L-1", "L-2"])] == ["L-1", "L-2"]
    assert [(p.revision, origin, seq) for p, origin, seq in cache.plans(f1, "r1", "g1")] == [
        (0, "human", 1),
    ]
    assert [p.revision for p, _, _ in cache.plans(f2, "r1", "g1")] == [0, 1]
    cache.close()


def test_run_without_mark_is_hidden(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    runs = [FakeRun(str(tmp_path / "runs"), rid) for rid in ("r1", "r2")]
    for run in runs:
        cid = "c-" + run.run_id
        head = run.commit(
            run.plan(),
            run.candidate(cid),
            run.score("s-" + run.run_id, cid, 1),
            run.learning("L-" + run.run_id, "Hint", evidence=[cid]),
        )
        ingest_run(cache, run.store, run.run_id, run.run_root, head)
    run = runs[0]
    frontier = build_frontier([("r1", run.run_root, run.heads[-1])], [], [])
    assert cache.membership(frontier) == canonical_json(
        {
            "candidates": ["c-r1"],
            "scores": ["s-r1"],
            "learnings": [["L-r1", False, False]],
        }
    )
    assert cache.plans(frontier, "r2", "g1") == []
    assert cache.candidates(frontier, run_id="r2") == []
    assert cache.membership(Frontier()) == canonical_json(
        {
            "candidates": [],
            "scores": [],
            "learnings": [],
        }
    )
    cache.close()


PIN_NAME = "toy/curation/20261018T000002.000Z-00000001.jsonl"
HIDE_NAME = "toy/curation/20261018T000002.000Z-00000002.jsonl"
REVIEW_NAME = "toy/reviews/alice/20261018T000002.000Z-00000001.jsonl"


def _human_learning(lid: str = "L-h-0000000000000001") -> Learning:
    return Learning(
        id=lid,
        namespace="toy",
        run_id="human",
        goal_id="g1",
        round=0,
        author="human:alice",
        title="Human hint",
        text="Try this",
        tags=["tag"],
        evidence=["c-1"],
        refines=["L-0"],
        supersedes=["L-old"],
    )


def _human_review(sid: str = "s-human") -> HumanObject:
    return HumanObject(
        kind="review",
        author="human:alice",
        score=Score(
            id=sid,
            candidate_id="c-1",
            judge="human:alice",
            rubric_hash="r1",
            key=(1,),
        ),
    )


def test_curation_applies_only_when_named(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    cache.apply_human_objects(
        [PIN_NAME],
        [
            HumanObject(kind="pin", learning_id="L-1", author="human:a"),
        ],
    )
    assert cache.flags(build_frontier([], [], [])) == {}
    assert cache.flags(build_frontier([], [], [PIN_NAME])) == {"L-1": (True, False)}
    cache.apply_human_objects(
        [HIDE_NAME],
        [
            HumanObject(kind="hide", learning_id="L-1", author="human:a"),
        ],
    )
    assert cache.flags(build_frontier([], [], [HIDE_NAME, PIN_NAME])) == {"L-1": (False, True)}
    assert cache.flags(build_frontier([], [PIN_NAME], [])) == {}
    cache.close()


def test_review_scores_come_from_named_objects(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    review = _human_review()
    assert review.score is not None
    head = run.commit(
        run.plan(),
        run.candidate("c-1"),
        EventDraft(
            type=EventType.REVIEW_IMPORTED,
            goal_id="g1",
            payload={"scores": [review.score.model_dump(mode="json")]},
        ),
    )
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    marks = [("r1", run.run_root, head)]
    assert cache.scores(build_frontier(marks, [], [])) == []
    cache.apply_human_objects([REVIEW_NAME], [review])
    assert cache.scores(build_frontier(marks, [], []), judge="human:alice") == []
    frontier = build_frontier(marks, [REVIEW_NAME], [])
    assert [s.id for s, _ in cache.scores(frontier, judge="human:alice")] == ["s-human"]
    assert cache.scores(frontier, judge="j") == []
    assert cache.scores(frontier, candidate_ids=[]) == []
    assert cache.scores(frontier, candidate_ids=["missing"]) == []
    assert cache.scores(frontier, candidate_ids=["c-1", "c-1"]) == cache.scores(frontier)
    assert cache.scores(build_frontier(marks, [], [REVIEW_NAME])) == []
    cache.close()


def test_human_added_learning_needs_its_object(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    learning = _human_learning()
    cache.apply_human_objects(
        [PIN_NAME],
        [
            HumanObject(
                kind="add",
                learning=learning,
                author="human:alice",
            )
        ],
    )
    assert cache.learnings_by_id(Frontier(), [learning.id]) == []
    assert cache.learnings_by_id(build_frontier([], [PIN_NAME], []), [learning.id]) == []
    frontier = build_frontier([], [], [PIN_NAME])
    assert cache.learnings_by_id(frontier, [learning.id]) == [learning]
    cache.apply_human_objects(
        [HIDE_NAME],
        [
            HumanObject(
                kind="hide",
                learning_id=learning.id,
                author="human:alice",
            )
        ],
    )
    hidden = build_frontier([], [], [PIN_NAME, HIDE_NAME])
    assert cache.learnings_by_id(hidden, [learning.id])[0].hidden
    assert not cache.learnings_by_id(frontier, [learning.id])[0].hidden
    assert cache.learnings_by_id(hidden, []) == []
    assert cache.membership(hidden) == canonical_json(
        {
            "candidates": [],
            "scores": [],
            "learnings": [[learning.id, False, True]],
        }
    )
    assert cache._conn.execute("SELECT goal_key FROM learnings").fetchone() == (None,)
    assert cache._conn.execute("SELECT * FROM learning_tags").fetchall() == [(learning.id, "tag")]
    assert cache._conn.execute("SELECT * FROM learning_evidence").fetchall() == [
        (learning.id, "c-1")
    ]
    assert cache._conn.execute(
        "SELECT kind, target_id FROM learning_links ORDER BY kind"
    ).fetchall() == [
        ("refines", "L-0"),
        ("supersedes", "L-old"),
    ]
    cache.close()


def test_membership_is_order_independent(tmp_path: Path) -> None:
    runs = [FakeRun(str(tmp_path / "runs"), rid) for rid in ("r1", "r2")]
    for run in runs:
        cid = "c-" + run.run_id
        run.commit(
            run.plan(),
            run.candidate(cid),
            run.score("s-" + run.run_id, cid, 1),
            run.learning("L-" + run.run_id, "Hint", evidence=[cid]),
        )
    caches = [ArenaCache(tmp_path / str(i) / "toy.sqlite") for i in range(2)]
    pin = HumanObject(kind="pin", learning_id="L-r1", author="human:alice")
    hide = HumanObject(kind="hide", learning_id="L-r1", author="human:alice")
    batches = [(PIN_NAME, pin), (HIDE_NAME, hide), (REVIEW_NAME, _human_review())]
    for i, cache in enumerate(caches):
        for run in runs[:: 1 if i == 0 else -1]:
            ingest_run(cache, run.store, run.run_id, run.run_root, run.heads[-1])
        for name, obj in batches[:: 1 if i == 0 else -1]:
            cache.apply_human_objects([name], [obj])
    frontier = build_frontier(
        [(r.run_id, r.run_root, r.heads[-1]) for r in runs],
        [REVIEW_NAME],
        [HIDE_NAME, PIN_NAME],
    )
    assert caches[0].membership(frontier) == caches[1].membership(frontier)
    assert caches[0].membership(frontier) == canonical_json(
        {
            "candidates": ["c-r1", "c-r2"],
            "scores": ["s-human", "s-r1", "s-r2"],
            "learnings": [["L-r1", False, True], ["L-r2", False, False]],
        }
    )
    for cache in caches:
        cache.close()


def test_positions_are_commit_order(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan(), run.candidate("c-1"))
    head = run.commit(run.candidate("c-2"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    frontier = build_frontier([("r1", run.run_root, head)], [], [])
    positions = [position for _, position in cache.candidates(frontier)]
    assert positions == [
        ((START + timedelta(seconds=seq)).isoformat(timespec="microseconds"), "r1", seq)
        for seq in (2, 3)
    ]
    assert positions[0] < positions[1]
    cache.close()


def test_human_ts_orders_with_journal_ts(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1", start=START - timedelta(seconds=2))
    head = run.commit(run.plan(), run.candidate("c-1"), run.score("s-run", "c-1", 1))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    cache.apply_human_objects([REVIEW_NAME], [_human_review()])
    frontier = build_frontier([("r1", run.run_root, head)], [REVIEW_NAME], [])
    rows = cache.scores(frontier)
    assert [s.id for s, _ in rows] == ["s-run", "s-human"]
    assert rows[0][1] == (
        (START + timedelta(seconds=1)).isoformat(timespec="microseconds"),
        "r1",
        3,
    )
    assert rows[1][1] == (
        (START + timedelta(seconds=2)).isoformat(timespec="microseconds"),
        REVIEW_NAME,
        0,
    )
    assert rows[0][1] < rows[1][1]
    cache.close()


def test_human_objects_are_atomic_idempotent_and_keep_line_order(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    learning = _human_learning()
    names = [PIN_NAME, REVIEW_NAME, PIN_NAME, REVIEW_NAME, PIN_NAME]
    objects = [
        HumanObject(kind="add", learning=learning, author="human:alice"),
        _human_review("s-1"),
        HumanObject(kind="hide", learning_id=learning.id, author="human:alice"),
        _human_review("s-2"),
        HumanObject(kind="pin", learning_id=learning.id, author="human:alice"),
    ]
    assert not cache.has_object(PIN_NAME)
    for _ in range(2):
        cache.apply_human_objects(names, objects)
    assert cache.has_object(PIN_NAME)
    assert cache.has_object(REVIEW_NAME)
    frontier = build_frontier([], [REVIEW_NAME, REVIEW_NAME], [PIN_NAME, PIN_NAME])
    assert cache.flags(frontier) == {learning.id: (True, False)}
    assert [(s.id, p[2]) for s, p in cache.scores(frontier)] == [("s-1", 0), ("s-2", 1)]
    assert cache._conn.execute("SELECT line FROM curation_ops ORDER BY line").fetchall() == [
        (1,),
        (2,),
    ]
    assert cache._conn.execute("SELECT count(*) FROM learning_tags").fetchone() == (1,)
    expected = cache.membership(frontier)
    cache.close()
    cache = ArenaCache(tmp_path / "toy.sqlite")
    assert cache.has_object(PIN_NAME)
    assert cache.membership(frontier) == expected
    cache.close()


@pytest.mark.parametrize("kind", ["name", "folder", "length", "namespace"])
def test_invalid_human_batch_leaves_no_partial_rows(tmp_path: Path, kind: str) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    secret = "SENTINEL_SECRET"
    names = [PIN_NAME, HIDE_NAME]
    objects = [
        HumanObject(kind="add", learning=_human_learning(), author="human:alice"),
        HumanObject(kind="pin", learning_id="L-1", author="human:alice"),
    ]
    if kind == "name":
        names[1] = "toy/curation/" + secret
    elif kind == "folder":
        objects[1] = _human_review()
    elif kind == "length":
        names.pop()
    else:
        objects[1] = HumanObject(
            kind="add",
            author="human:alice",
            learning=_human_learning().model_copy(update={"namespace": secret}),
        )
    with pytest.raises(ValueError, match=r"human object|human learning") as caught:
        cache.apply_human_objects(names, objects)
    assert_secret_absent(caught.value, secret)
    assert secret not in "".join(traceback.format_exception(caught.value))
    for table in ("human_objects", "learnings", "scores", "curation_ops", "learning_tags"):
        assert cache._conn.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
    assert not cache._conn.in_transaction
    cache.close()


@pytest.mark.parametrize(
    "method", ["candidates", "scores", "flags", "learnings_by_id", "plans", "membership"]
)
@pytest.mark.parametrize("bad", ["seq", "epoch", "uncovered", "conflict"])
def test_reads_require_exact_covered_marks(tmp_path: Path, method: str, bad: str) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    old = run.commit(run.plan(), run.candidate("c-1"))
    head = run.commit(run.candidate("c-2"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    frontier = build_frontier([("r1", run.run_root, old)], [], [])
    mark = frontier.marks[0]
    if bad == "seq":
        frontier = frontier.model_copy(
            update={"marks": [mark.model_copy(update={"seq": head.seq})]}
        )
    elif bad == "epoch":
        frontier = frontier.model_copy(update={"marks": [mark.model_copy(update={"epoch": 99})]})
    elif bad == "uncovered":
        frontier = build_frontier([("r1", run.run_root, run.commit(run.candidate("c-3")))], [], [])
    else:
        frontier = frontier.model_copy(
            update={
                "marks": frontier.marks
                + build_frontier([("r1", run.run_root, head)], [], []).marks,
            }
        )
    extra: tuple[object, ...] = ()
    if method == "learnings_by_id":
        extra = (["L-1"],)
    elif method == "plans":
        extra = ("r1", "g1")
    with pytest.raises(ChainError):
        getattr(cache, method)(frontier, *extra)
    cache.close()


@pytest.mark.parametrize("change", ["inputs", "manifest"])
def test_candidate_filters_preserve_fixed_producing_keys(tmp_path: Path, change: str) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    old = run.commit(run.plan(parts=["a"]), run.candidate("old", part="a"), run.candidate("main"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, old)
    frontier = build_frontier([("r1", run.run_root, old)], [], [])
    old_key = goal_key_digest(cache._replay("r1", through_seq=old.seq).candidate_goal_keys["old"])
    before = cache.membership(frontier)
    revision = run.plan(revision=1, parts=["a"], first_round=3)
    goal = revision.payload["plan"]["parts"][0]["goal"]
    if change == "inputs":
        goal["inputs"] = {"changed": True}
    else:
        goal["files"] = {"globs": [], "manifest": {"input.txt": "a" * 64}}
    head = run.commit(
        revision, run.candidate("late", part="a", round=2), run.candidate("new", part="a", round=3)
    )
    ingest_run(cache, run.store, "r1", run.run_root, head)
    current = build_frontier([("r1", run.run_root, head)], [], [])
    new_key = goal_key_digest(cache._replay("r1", through_seq=head.seq).candidate_goal_keys["new"])
    assert cache.membership(frontier) == before
    assert [c.id for c, _ in cache.candidates(frontier, part_key=old_key)] == ["old"]
    assert cache.candidates(frontier, part_key=new_key) == []
    assert [c.id for c, _ in cache.candidates(current, part_key=old_key)] == ["old", "late"]
    assert [c.id for c, _ in cache.candidates(current, part_key=new_key)] == ["new"]
    assert [
        c.id
        for c, _ in cache.candidates(
            current,
            ids=["new", "new", "old"],
            goal_key=goal_key_digest(run.goal.key()),
            run_id="r1",
            part_key=new_key,
        )
    ] == ["new"]
    assert cache.candidates(current, ids=[]) == []
    assert cache.candidates(current, goal_key="missing") == []
    assert cache.candidates(current, run_id="missing") == []
    duplicate = frontier.model_copy(update={"marks": frontier.marks * 2})
    assert cache.membership(duplicate) == before
    assert cache._plan_for_round("r1", "g1", 2, through_seq=head.seq).revision == 0
    assert cache._plan_for_round("r1", "g1", 3, through_seq=head.seq).revision == 1
    assert [p.revision for p, _, _ in cache.plans(frontier, "r1", "g1")] == [0]
    cache.close()


def test_human_objects_concurrent_connections_apply_each_line_once(tmp_path: Path) -> None:
    path = tmp_path / "toy.sqlite"
    caches = [ArenaCache(path), ArenaCache(path)]
    barrier = Barrier(2)
    learning = _human_learning()
    operations = [
        HumanObject(kind="add", learning=learning, author="human:a"),
        HumanObject(kind="pin", learning_id=learning.id, author="human:a"),
        HumanObject(kind="hide", learning_id=learning.id, author="human:a"),
    ]

    def apply(cache: ArenaCache) -> None:
        barrier.wait(timeout=5)
        cache.apply_human_objects([PIN_NAME] * len(operations), operations)

    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(apply, cache) for cache in caches]
            for future in futures:
                future.result(timeout=10)
        assert caches[0].flags(build_frontier([], [], [PIN_NAME])) == {learning.id: (False, True)}
        assert caches[0]._conn.execute("SELECT count(*) FROM curation_ops").fetchone() == (2,)
        assert caches[0]._conn.execute("SELECT count(*) FROM learning_tags").fetchone() == (1,)
    finally:
        for cache in caches:
            cache.close()


def test_human_import_ignores_foreign_namespace(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    foreign = PIN_NAME.replace("toy/", "other/")
    cache.apply_human_objects(
        [foreign],
        [
            HumanObject(
                kind="add",
                learning=_human_learning().model_copy(update={"namespace": "other"}),
                author="human:a",
            )
        ],
    )
    assert not cache.has_object(foreign)
    assert cache.learnings_by_id(build_frontier([], [], [foreign]), [_human_learning().id]) == []
    cache.apply_human_objects([], [])
    cache.close()


def test_learning_flags_come_only_from_named_curation(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    draft = run.learning("L-1", "Hint", evidence=["c-1"])
    draft.payload["learning"].update(pinned=True, hidden=True)
    head = run.commit(run.plan(), run.candidate("c-1"), draft)
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    frontier = build_frontier([("r1", run.run_root, head)], [], [])
    learning = cache.learnings_by_id(frontier, ["L-1"])[0]
    assert not learning.pinned
    assert not learning.hidden
    learning.tags.append("local edit")
    assert cache.learnings_by_id(frontier, ["L-1"])[0].tags == []
    cache.close()


def test_failed_human_transaction_can_be_retried(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    learning = _human_learning()
    objects = [HumanObject(kind="add", learning=learning, author="human:a"), _human_review()]
    # Force a real SQL failure after the learning and its child rows were inserted.
    cache._conn.execute(
        "CREATE TRIGGER refuse_review BEFORE INSERT ON scores "
        "BEGIN SELECT RAISE(ABORT, 'fixture failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="fixture failure"):
        cache.apply_human_objects([PIN_NAME, REVIEW_NAME], objects)
    assert not cache.has_object(PIN_NAME)
    assert not cache.has_object(REVIEW_NAME)
    assert cache._conn.execute("SELECT count(*) FROM learning_tags").fetchone() == (0,)
    assert cache._conn.execute("SELECT count(*) FROM learnings").fetchone() == (0,)
    assert not cache._conn.in_transaction
    cache._conn.execute("DROP TRIGGER refuse_review")
    cache.apply_human_objects([PIN_NAME, REVIEW_NAME], objects)
    assert cache.has_object(PIN_NAME)
    assert cache.has_object(REVIEW_NAME)
    cache.close()


def test_human_and_event_learnings_have_identical_child_rows(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    draft = run.learning(
        "L-run",
        "Shared hint",
        tags=["beta", "alpha"],
        evidence=["c-2", "c-1"],
        refines=["L-0"],
        supersedes=["L-old"],
    )
    head = run.commit(run.plan(), run.candidate("c-1"), run.candidate("c-2"), draft)
    human = Learning.model_validate(draft.payload["learning"]).model_copy(
        update={"id": "L-human", "author": "human:alice"}
    )
    cache = ArenaCache(tmp_path / "toy.sqlite")
    try:
        ingest_run(cache, run.store, "r1", run.run_root, head)
        cache.apply_human_objects(
            [PIN_NAME], [HumanObject(kind="add", learning=human, author="human:alice")]
        )
        for table, columns, expected in (
            ("learning_tags", "tag", [("alpha",), ("beta",)]),
            ("learning_evidence", "candidate_id", [("c-1",), ("c-2",)]),
            ("learning_links", "kind, target_id", [("refines", "L-0"), ("supersedes", "L-old")]),
        ):
            event_rows, human_rows = (
                cache._conn.execute(
                    f"SELECT {columns} FROM {table} WHERE learning_id=? ORDER BY {columns}",
                    (learning_id,),
                ).fetchall()
                for learning_id in ("L-run", "L-human")
            )
            assert event_rows == human_rows == expected
    finally:
        cache.close()


@pytest.fixture
def cache_with_learnings(tmp_path: Path) -> tuple[ArenaCache, FakeRun, Frontier]:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(
        run.plan(parts=["a"]),
        run.candidate("c-1", part="a"),
        run.learning(
            "L-log",
            "Fitting log(nu) instead of nu converges about 3x faster.",
            evidence=["c-1"],
            tags=["a", "b"],
            verified=True,
        ),
        run.learning(
            "L-other", "Other hint", text="t", evidence=["c-1"], tags=["a"], author="t2/a2", round=2
        ),
    )
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, run.run_id, run.run_root, head)
    return cache, run, build_frontier([(run.run_id, run.run_root, head)], [], [])


@pytest.mark.parametrize(
    "query",
    [
        'nu" OR',
        "-lr:(",
        "log(nu)",
        "fd-solver",
        "NEAR(a b)",
        "*",
        '"',
        "AND OR NOT",
        "()",
        "___",
        "a_b",
    ],
)
def test_search_never_raises_on_fts_syntax(
    cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier],
    query: str,
) -> None:
    cache, _, frontier = cache_with_learnings
    assert isinstance(
        cache.search("toy", query, frontier=frontier, goal=None, tags=(), limit=10), list
    )
    cache.close()


def test_search_finds_parenthesised_term(
    cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier],
) -> None:
    cache, _, frontier = cache_with_learnings
    assert cache.search("toy", "log(nu)", frontier=frontier, goal=None, tags=(), limit=1) == [
        "L-log"
    ]
    for query in ("()", "___", "*", '"'):
        assert cache.search("toy", query, frontier=frontier, goal=None, tags=(), limit=10) == []
    cache.close()


def test_fts_query_quotes_tokens() -> None:
    assert fts_query('log(nu) "x"') == '"log" OR "nu" OR "x"'
    assert fts_query("()") is None
    assert fts_query("___") is None
    assert fts_query("fd_solver") == '"fd" OR "solver"'
    assert fts_query("CAFÉ λ") == '"café" OR "λ"'


def test_search_order_ignores_rows_outside_frontier(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    h1 = run.commit(
        run.plan(),
        run.candidate("c"),
        run.learning("L-short", "alpha", evidence=["c"]),
        run.learning("L-long", "Long hint", evidence=["c"], text="alpha " * 3 + "word " * 57),
    )
    frontier = build_frontier([("r1", run.run_root, h1)], [], [])
    a = ArenaCache(tmp_path / "a" / "toy.sqlite")
    b = ArenaCache(tmp_path / "b" / "toy.sqlite")
    ingest_run(a, run.store, "r1", run.run_root, h1)
    h2 = run.commit(
        *(
            run.learning(f"L-later-{i}", "Later", evidence=["c"], text="word " * 250)
            for i in range(200)
        )
    )
    ingest_run(b, run.store, "r1", run.run_root, h2)
    outsider = FakeRun(str(tmp_path / "runs"), "outside")
    oh = outsider.commit(
        outsider.plan(), outsider.learning("L-outsider", "alpha", author="human:x")
    )
    ingest_run(b, outsider.store, outsider.run_id, outsider.run_root, oh)
    # An unnamed human addition also must not affect corpus statistics.
    human = _human_learning().model_copy(update={"title": "alpha"})
    b.apply_human_objects([PIN_NAME], [HumanObject(kind="add", learning=human, author="human:x")])
    for limit in (10, 1):
        assert (
            a.search("toy", "alpha", frontier=frontier, goal=None, tags=(), limit=limit)
            == b.search("toy", "alpha", frontier=frontier, goal=None, tags=(), limit=limit)
            == ["L-short", "L-long"][:limit]
        )
    table = "fts_" + frontier.digest()[:16]
    assert b._conn.execute(f"SELECT count(*) FROM temp.{table}").fetchone() == (2,)
    a.close()
    b.close()


def test_human_added_learning_is_searchable(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    learning = _human_learning().model_copy(update={"text": "zebra"})
    cache.apply_human_objects(
        [PIN_NAME], [HumanObject(kind="add", learning=learning, author="human:x")]
    )
    for frontier, expected in ((Frontier(curation=[PIN_NAME]), [learning.id]), (Frontier(), [])):
        assert (
            cache.search("toy", "zebra", frontier=frontier, goal=None, tags=(), limit=10)
            == expected
        )
    cache.close()


def test_hidden_summary_restores_superseded(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(
        run.plan(),
        run.candidate("c"),
        run.learning("L-b", "Hint", evidence=["c"]),
        run.learning("L-a", "Hint", evidence=["c"]),
        run.learning("L-sum", "Summary", evidence=["c"], supersedes=["L-a"]),
    )
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    f1 = build_frontier([("r1", run.run_root, head)], [], [])
    cache.apply_human_objects(
        [HIDE_NAME], [HumanObject(kind="hide", learning_id="L-sum", author="human:x")]
    )
    f2 = f1.model_copy(update={"curation": [HIDE_NAME]})
    for frontier, expected in ((f1, ["L-b", "L-a"]), (f2, ["L-a", "L-b"]), (f1, ["L-b", "L-a"])):
        ids = cache.find(frontier, query=None, filters=LearningFilter(), relevance_goal=None)
        assert [lid for lid in ids if lid != "L-sum"] == expected
        assert ("L-sum" in ids) == (frontier == f1)
    cache.close()


def test_human_learning_ts_orders_with_journal_ts(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1", start=START - timedelta(seconds=2))
    head = run.commit(run.plan(), run.candidate("c"), run.learning("L-run", "Hint", evidence=["c"]))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    name = "toy/curation/20261018T000000.000Z-00000001.jsonl"
    learning = _human_learning()
    cache.apply_human_objects(
        [name], [HumanObject(kind="add", learning=learning, author="human:x")]
    )
    frontier = build_frontier([("r1", run.run_root, head)], [], [name])
    assert cache.find(frontier, query=None, filters=LearningFilter(), relevance_goal=None) == [
        "L-run",
        learning.id,
    ]
    cache.close()


def test_goal_filter_keeps_human_learning_with_goal_id(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(), run.learning("L-run", "Hint", author="human:x"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    one = _human_learning()
    two = one.model_copy(update={"id": "L-h-two", "goal_id": "g2"})
    cache.apply_human_objects(
        [PIN_NAME] * 2,
        [HumanObject(kind="add", learning=learning, author="human:x") for learning in (one, two)],
    )
    frontier = build_frontier([("r1", run.run_root, head)], [], [PIN_NAME])
    gk = goal_key_digest(run.goal.key())
    for filters, expected in (
        (LearningFilter(goal_key=gk, goal_id="g1"), {"L-run", one.id}),
        (LearningFilter(goal_key=gk), {"L-run"}),
        (LearningFilter(goal_id="g2"), {two.id}),
        (LearningFilter(goal_key="different", goal_id="g1"), {one.id}),
    ):
        assert (
            set(cache.find(frontier, query=None, filters=filters, relevance_goal=None)) == expected
        )
    assert set(
        cache.search("toy", "hint", frontier=frontier, goal=run.goal.key(), tags=(), limit=10)
    ) == {"L-run", one.id}
    cache.close()


def test_relevance_goal_without_goal_filter(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    marks = []
    for gid in ("g1", "g2"):
        run = FakeRun(str(tmp_path / "runs"), gid, goal_id=gid)
        head = run.commit(run.plan(), run.learning("L-" + gid, "Hint", author="human:x"))
        ingest_run(cache, run.store, gid, run.run_root, head)
        marks.append((gid, run.run_root, head))
    frontier = build_frontier(marks, [], [])
    assert cache.find(
        frontier,
        query=None,
        filters=LearningFilter(),
        relevance_goal=(goal_key_digest(run.goal.key()), "g2"),
    ) == ["L-g2", "L-g1"]
    cache.close()


def test_search_orders_by_bm25_then_relevance(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(
        run.plan(),
        run.candidate("c"),
        run.learning("L-verified", "alpha", evidence=["c"], verified=True),
        run.learning("L-newer", "alpha", evidence=["c"]),
        run.learning(
            "L-long", "Long hint", evidence=["c"], verified=True, text="alpha " + "word " * 60
        ),
    )
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    frontier = build_frontier([("r1", run.run_root, head)], [], [])
    cache.apply_human_objects(
        [PIN_NAME], [HumanObject(kind="pin", learning_id="L-long", author="human:x")]
    )
    frontier = frontier.model_copy(update={"curation": [PIN_NAME]})
    assert cache.search("toy", "alpha", frontier=frontier, goal=None, tags=(), limit=10) == [
        "L-verified",
        "L-newer",
        "L-long",
    ]
    cache.close()


def test_relevance_order(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    marks = []
    goal_key = ""
    for gid, specs in (
        ("g1", [("L-old-sup", False, ["a"]), ("L-goal", False, ["a"])]),
        (
            "g2",
            [
                ("L-ns", False, ["a"]),
                ("L-ver", True, ["a"]),
                ("L-tag", False, ["a", "b"]),
                ("L-new", False, ["a"]),
                ("L-pin", False, ["a"]),
            ],
        ),
    ):
        run = FakeRun(str(tmp_path / "runs"), gid, goal_id=gid)
        drafts = [run.plan(), run.candidate("c-" + gid)]
        for lid, verified, tags in specs:
            drafts.append(
                run.learning(
                    lid,
                    "Hint",
                    evidence=["c-" + gid],
                    tags=tags,
                    verified=verified,
                    supersedes=["L-old-sup"] if lid == "L-goal" else [],
                )
            )
        head = run.commit(*drafts)
        ingest_run(cache, run.store, gid, run.run_root, head)
        marks.append((gid, run.run_root, head))
        if gid == "g1":
            goal_key = goal_key_digest(run.goal.key())
    cache.apply_human_objects(
        [PIN_NAME], [HumanObject(kind="pin", learning_id="L-pin", author="human:x")]
    )
    frontier = build_frontier(marks, [], [PIN_NAME])
    assert cache.find(
        frontier,
        query=None,
        filters=LearningFilter(tags=["a", "b"]),
        relevance_goal=(goal_key, "g1"),
    ) == ["L-pin", "L-goal", "L-ver", "L-tag", "L-new", "L-ns", "L-old-sup"]
    cache.close()


def test_hidden_excluded_by_default_and_filterable(
    cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier],
) -> None:
    cache, _, frontier = cache_with_learnings
    cache.apply_human_objects(
        [HIDE_NAME], [HumanObject(kind="hide", learning_id="L-log", author="human:x")]
    )
    frontier = frontier.model_copy(update={"curation": [HIDE_NAME]})
    for hidden, expected in ((False, {"L-other"}), (True, {"L-log"}), (None, {"L-log", "L-other"})):
        assert (
            set(
                cache.find(
                    frontier, query=None, filters=LearningFilter(hidden=hidden), relevance_goal=None
                )
            )
            == expected
        )
    cache.close()


def test_tags_filter_is_any_of(cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier]) -> None:
    cache, _, frontier = cache_with_learnings
    for tags, expected in (
        (["b", "absent"], {"L-log"}),
        (["a", "absent"], {"L-log", "L-other"}),
        (["absent"], set()),
    ):
        assert (
            set(
                cache.find(
                    frontier, query=None, filters=LearningFilter(tags=tags), relevance_goal=None
                )
            )
            == expected
        )
    cache.close()


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (LearningFilter(run="r1"), {"L-log", "L-other"}),
        (LearningFilter(run="other"), set()),
        (LearningFilter(team="t1"), {"L-log"}),
        (LearningFilter(agent="a2"), {"L-other"}),
        (LearningFilter(round=2), {"L-other"}),
        (LearningFilter(part="a"), {"L-log", "L-other"}),
        (LearningFilter(part="b"), set()),
        (LearningFilter(human=False), {"L-log", "L-other"}),
        (LearningFilter(human=True), set()),
        (LearningFilter(verified=True), {"L-log"}),
        (LearningFilter(verified=False), {"L-other"}),
        (LearningFilter(pinned=True), set()),
        (LearningFilter(team="t1", agent="a2"), set()),
    ],
)
def test_filters_run_team_agent_round_part_human_verified(
    cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier],
    filters: LearningFilter,
    expected: set[str],
) -> None:
    cache, _, frontier = cache_with_learnings
    assert set(cache.find(frontier, query=None, filters=filters, relevance_goal=None)) == expected
    assert cache.count(frontier, filters=filters) == len(expected)
    cache.close()


def test_count_matches_find(cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier]) -> None:
    cache, _, frontier = cache_with_learnings
    filters = LearningFilter()
    ids = cache.find(frontier, query=None, filters=filters, relevance_goal=None)
    assert cache.count(frontier, filters=filters) == len(ids) == 2
    assert (
        cache.find(frontier, query=None, filters=filters, relevance_goal=None, offset=1, limit=1)
        == ids[1:2]
    )
    assert cache.find(frontier, query=None, filters=filters, relevance_goal=None, limit=0) == []
    cache.close()


def test_search_rejects_other_namespace(
    cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier],
) -> None:
    cache, _, frontier = cache_with_learnings
    secret = "SECRET-NAMESPACE-INPUT"
    with pytest.raises(ValueError, match="namespace does not match cache") as caught:
        cache.search(secret, "hint", frontier=frontier, goal=None, tags=(), limit=10)
    assert_secret_absent(caught.value, secret)
    cache.close()


@pytest.mark.parametrize("missing", ["mark", "curation", "review"])
def test_search_requires_complete_frontier_coverage(tmp_path: Path, missing: str) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(), run.learning("L-1", "alpha", author="human:x"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    frontier = build_frontier(
        [("r1", run.run_root, head)],
        [REVIEW_NAME] if missing == "review" else [],
        [PIN_NAME] if missing == "curation" else [],
    )
    if missing != "mark":
        ingest_run(cache, run.store, "r1", run.run_root, head)
    for query in ("alpha", "___", None):
        with pytest.raises(ChainError):
            cache.find(frontier, query=query, filters=LearningFilter(), relevance_goal=None)
    assert (
        cache._conn.execute(
            "SELECT name FROM sqlite_temp_master WHERE name LIKE 'fts_%'"
        ).fetchall()
        == []
    )
    ingest_run(cache, run.store, "r1", run.run_root, head)
    if missing == "curation":
        cache.apply_human_objects(
            [PIN_NAME], [HumanObject(kind="pin", learning_id="L-1", author="human:x")]
        )
    if missing == "review":
        cache.apply_human_objects([REVIEW_NAME], [_human_review()])
    assert cache.search("toy", "alpha", frontier=frontier, goal=None, tags=(), limit=10) == ["L-1"]
    cache.close()


def test_frontier_fts_lru_is_connection_local(
    cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier],
) -> None:
    cache, _, frontier = cache_with_learnings
    frontiers = [frontier.model_copy(update={"reviews": [REVIEW_NAME] * i}) for i in range(10)]
    cache.apply_human_objects([REVIEW_NAME], [_human_review()])
    for f in frontiers[:8]:
        cache.search("toy", "hint", frontier=f, goal=None, tags=(), limit=10)
    cache.search("toy", "hint", frontier=frontiers[0], goal=None, tags=(), limit=10)
    for f in frontiers[8:]:
        cache.search("toy", "hint", frontier=f, goal=None, tags=(), limit=10)
    tables = {
        row[0]
        for row in cache._conn.execute("SELECT name FROM sqlite_temp_master WHERE type='table'")
    }
    assert "fts_" + frontiers[0].digest()[:16] in tables
    assert "fts_" + frontiers[1].digest()[:16] not in tables
    assert "fts_" + frontiers[2].digest()[:16] not in tables
    assert sum(name.startswith("fts_") and name.endswith("_config") for name in tables) == 8
    assert (
        cache._conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'fts_%'").fetchall()
        == []
    )
    cache.close()


def test_one_cache_shared_by_threads(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    runs = [FakeRun(str(tmp_path / "runs"), rid) for rid in ("r1", "r2")]
    initial = []
    for run in runs:
        head = run.commit(
            run.plan(),
            run.candidate("c-" + run.run_id),
            run.learning("L-" + run.run_id, "t alpha", evidence=["c-" + run.run_id]),
        )
        ingest_run(cache, run.store, run.run_id, run.run_root, head)
        initial.append((run.run_id, run.run_root, head))
    fixed = build_frontier(initial, [], [])
    expected = cache.find(fixed, query="t", filters=LearningFilter(), relevance_goal=None)
    barrier = Barrier(4)
    done = [ThreadEvent(), ThreadEvent()]

    def write(run: FakeRun, finished: ThreadEvent) -> None:
        barrier.wait(timeout=10)
        for i in range(20):
            drafts = [run.candidate(f"c-{run.run_id}-{i}-{j}") for j in range(10)]
            drafts.append(
                run.learning(
                    f"L-{run.run_id}-{i}",
                    "t later",
                    evidence=[drafts[0].payload["candidate"]["id"]],
                )
            )
            head = run.commit(*drafts)
            ingest_run(cache, run.store, run.run_id, run.run_root, head)
        finished.set()

    def read() -> None:
        barrier.wait(timeout=10)
        deadline = monotonic() + 30
        while not all(finished.is_set() for finished in done):
            assert monotonic() < deadline
            assert len(cache.candidates(fixed)) == 2
            assert (
                cache.find(fixed, query="t", filters=LearningFilter(), relevance_goal=None)
                == expected
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(write, run, finished) for run, finished in zip(runs, done, strict=True)
        ] + [pool.submit(read) for _ in range(2)]
        for future in futures:
            future.result(timeout=60)
    later = build_frontier([(run.run_id, run.run_root, run.heads[-1]) for run in runs], [], [])
    assert len(cache.candidates(later)) == 402
    assert cache.count(later, filters=LearningFilter()) == 42
    assert (
        len(cache.find(later, query="t", filters=LearningFilter(), relevance_goal=None, limit=100))
        == 42
    )
    assert cache.find(fixed, query="t", filters=LearningFilter(), relevance_goal=None) == expected
    assert not cache._conn.in_transaction
    cache.close()


def test_relevance_timestamp_and_id_ties(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    marks = []
    for rid, lid, micros in (("older", "L-a", 0), ("newer", "L-z", 1000)):
        run = FakeRun(
            str(tmp_path / "runs"), rid, start=START.replace(year=9999, microsecond=micros)
        )
        head = run.commit(run.plan(), run.learning(lid, "Hint", author="human:x"))
        ingest_run(cache, run.store, rid, run.run_root, head)
        marks.append((rid, run.run_root, head))
    first = _human_learning("L-human-z")
    second = first.model_copy(update={"id": "L-human-a"})
    cache.apply_human_objects(
        [PIN_NAME] * 2,
        [
            HumanObject(kind="add", learning=learning, author="human:x")
            for learning in (first, second)
        ],
    )
    frontier = build_frontier(marks, [], [PIN_NAME])
    assert cache.find(frontier, query=None, filters=LearningFilter(), relevance_goal=None) == [
        "L-z",
        "L-a",
        "L-human-a",
        "L-human-z",
    ]
    cache.close()


def test_part_filter_uses_evidence_inside_frontier(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    h1 = run.commit(run.plan(parts=["a"]))
    h2 = run.commit(run.candidate("future", part="a"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, h2)
    learning = _human_learning().model_copy(update={"evidence": ["future"], "text": "zebra"})
    cache.apply_human_objects(
        [PIN_NAME], [HumanObject(kind="add", learning=learning, author="human:x")]
    )
    for head, expected in ((h1, []), (h2, [learning.id])):
        frontier = build_frontier([("r1", run.run_root, head)], [], [PIN_NAME])
        assert (
            cache.find(frontier, query=None, filters=LearningFilter(part="a"), relevance_goal=None)
            == expected
        )
    cache.close()


def test_search_corpus_keeps_hidden_rows_and_ignores_filters(
    cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier],
) -> None:
    cache, _, frontier = cache_with_learnings
    cache.apply_human_objects(
        [HIDE_NAME], [HumanObject(kind="hide", learning_id="L-log", author="human:x")]
    )
    frontier = frontier.model_copy(update={"curation": [HIDE_NAME]})
    assert cache.find(frontier, query="log", filters=LearningFilter(), relevance_goal=None) == []
    assert cache.find(
        frontier, query="log", filters=LearningFilter(hidden=True), relevance_goal=None
    ) == ["L-log"]
    assert cache.find(
        frontier, query="b", filters=LearningFilter(hidden=None), relevance_goal=None
    ) == ["L-log"]
    table = "fts_" + frontier.digest()[:16]
    assert cache._conn.execute(f"SELECT title FROM temp.{table} ORDER BY rowid").fetchall() == [
        ("Fitting log(nu) instead of nu converges about 3x faster.",),
        ("Other hint",),
    ]
    cache.close()


def test_summary_outside_frontier_does_not_supersede(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    h1 = run.commit(
        run.plan(),
        run.learning("L-b", "Hint", author="human:x"),
        run.learning("L-a", "Hint", author="human:x"),
    )
    h2 = run.commit(run.learning("L-summary", "Summary", author="human:x", supersedes=["L-a"]))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, h2)
    for head, expected in (
        (h1, ["L-a", "L-b"]),
        (h2, ["L-summary", "L-b", "L-a"]),
        (h1, ["L-a", "L-b"]),
    ):
        frontier = build_frontier([("r1", run.run_root, head)], [], [])
        assert (
            cache.find(frontier, query=None, filters=LearningFilter(), relevance_goal=None)
            == expected
        )
    cache.close()


def test_all_reads_share_indexed_large_frontier(tmp_path: Path) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    names = [f"toy/curation/20261018T000000.000Z-{i:08x}.jsonl" for i in range(40_000)]
    cache.apply_human_objects(
        names, [HumanObject(kind="pin", learning_id="L-absent", author="human:x")] * len(names)
    )
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(
        run.plan(), run.candidate("c"), run.learning("L-1", "rareword", evidence=["c"])
    )
    ingest_run(cache, run.store, "r1", run.run_root, head)
    frontier = build_frontier([("r1", run.run_root, head)], [], names)
    # Exercise the old variable ceiling even on SQLite builds with a higher default.
    cache._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 64)
    assert cache.search("toy", "rareword", frontier=frontier, goal=None, tags=(), limit=1) == [
        "L-1"
    ]
    assert cache.count(frontier, filters=LearningFilter()) == 1
    assert [c.id for c, _ in cache.candidates(frontier)] == ["c"]
    assert cache.scores(frontier) == []
    assert cache.flags(frontier) == {"L-absent": (True, False)}
    assert [learning.id for learning in cache.learnings_by_id(frontier, ["L-1"])] == ["L-1"]
    assert len(cache.plans(frontier, "r1", "g1")) == 1
    assert cache.membership(frontier)
    assert not cache._conn.in_transaction
    assert not hasattr(cache, "_frontier_sql")
    cache.close()


def test_frontier_membership_work_grows_with_names(tmp_path: Path) -> None:
    work = []
    for size in (1_000, 10_000, 50_000):
        cache = ArenaCache(tmp_path / str(size) / "toy.sqlite")
        names = [f"toy/curation/20261018T000000.000Z-{i:08x}.jsonl" for i in range(size)]
        cache.apply_human_objects(
            names, [HumanObject(kind="pin", learning_id="L-absent", author="human:x")] * size
        )
        steps = 0

        def tick() -> int:
            nonlocal steps
            steps += 1000
            return 0

        cache._conn.set_progress_handler(tick, 1000)
        assert cache.flags(Frontier(curation=names)) == {"L-absent": (True, False)}
        cache._conn.set_progress_handler(None, 0)
        work.append(steps)
        cache.close()
    print("indexed frontier VM steps (1k/10k/50k):", work)
    assert 0 < work[1] < 15 * work[0]
    assert work[2] < 8 * work[1]


def test_nested_reads_prepare_frontier_once(
    cache_with_learnings: tuple[ArenaCache, FakeRun, Frontier],
) -> None:
    cache, _, frontier = cache_with_learnings
    trace: list[str] = []
    cache._conn.set_trace_callback(trace.append)
    assert cache.membership(frontier)
    cache._conn.set_trace_callback(None)
    assert trace.count("DELETE FROM _frontier_marks") == 1
    assert trace.count("DELETE FROM _frontier_objects") == 1
    assert cache.membership(Frontier()) == canonical_json(
        {"candidates": [], "scores": [], "learnings": []}
    )
    assert cache.membership(frontier)
    cache.close()


def _seed_search_corpus(cache: ArenaCache, size: int) -> Frontier:
    name = "toy/curation/20261018T000000.000Z-00000000.jsonl"
    names = [name] * size
    objects = [
        HumanObject(
            kind="add",
            author="human:x",
            learning=Learning(
                id=f"L-h-{i:016x}",
                namespace="toy",
                run_id="",
                goal_id="g1",
                round=0,
                author="human:x",
                title="rareword" if i == 0 else "common",
                text="word " * 240,
                tags=["tag"],
            ),
        )
        for i in range(size)
    ]
    cache.apply_human_objects(names, objects)
    return Frontier(curation=[name])


def test_warm_search_and_count_do_not_hydrate_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = ArenaCache(tmp_path / "toy.sqlite")
    frontier = _seed_search_corpus(cache, 1000)
    assert cache.search("toy", "rareword", frontier=frontier, goal=None, tags=(), limit=1) == [
        "L-h-0000000000000000"
    ]

    def refuse(*args: object, **kwargs: object) -> Learning:
        pytest.fail("a cached ID/count read must not hydrate Learning models")

    monkeypatch.setattr(Learning, "model_validate_json", refuse)
    another = frontier.model_copy(update={"curation": frontier.curation * 2})
    assert cache.search("toy", "rareword", frontier=another, goal=None, tags=(), limit=1) == [
        "L-h-0000000000000000"
    ]
    assert cache.search("toy", "rareword", frontier=frontier, goal=None, tags=(), limit=1) == [
        "L-h-0000000000000000"
    ]
    assert cache.count(frontier, filters=LearningFilter(goal_id="absent")) == 0
    assert cache.count(frontier, filters=LearningFilter()) == 1000
    assert (
        len(
            cache.find(
                frontier, query=None, filters=LearningFilter(), relevance_goal=None, limit=10
            )
        )
        == 10
    )
    cache.close()


def test_sql_search_work_and_peak_memory(tmp_path: Path) -> None:
    measurements = []
    for size in (1_000, 10_000, 50_000):
        cache = ArenaCache(tmp_path / str(size) / "toy.sqlite")
        frontier = _seed_search_corpus(cache, size)
        assert cache.search("toy", "rareword", frontier=frontier, goal=None, tags=(), limit=1) == [
            "L-h-0000000000000000"
        ]
        steps = 0

        def tick() -> int:
            nonlocal steps
            steps += 100
            return 0

        cache._conn.set_progress_handler(tick, 100)
        assert cache.search("toy", "rareword", frontier=frontier, goal=None, tags=(), limit=1) == [
            "L-h-0000000000000000"
        ]
        search_work = steps
        steps = 0
        assert cache.count(frontier, filters=LearningFilter()) == size
        count_work = steps
        steps = 0
        assert cache.count(frontier, filters=LearningFilter(goal_id="absent")) == 0
        absent_work = steps
        cache._conn.set_progress_handler(None, 0)
        gc.collect()
        tracemalloc.start()
        try:
            assert cache.search(
                "toy", "rareword", frontier=frontier, goal=None, tags=(), limit=1
            ) == ["L-h-0000000000000000"]
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        measurements.append((search_work, count_work, absent_work, peak))
        assert peak < 2 * 1024 * 1024
        cache.close()
    print("SQL work/peak bytes (search, count, empty count; 1k/10k/50k):", measurements)
    # Fixed one-object membership and one matching row should not scan the corpus.
    assert measurements[2][0] < 5 * max(measurements[0][0], 100)
    assert measurements[2][2] < 5 * max(measurements[0][2], 100)
    # Full counts scale with rows, rather than correlated scans or Python models.
    assert measurements[1][1] < 15 * measurements[0][1]
    assert measurements[2][1] < 8 * measurements[1][1]


def test_nonmatching_null_goal_has_same_relevance_as_other_goal(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(), run.learning("L-z", "Hint", author="human:x"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    human = _human_learning("L-a").model_copy(update={"goal_id": "other"})
    cache.apply_human_objects(
        [PIN_NAME], [HumanObject(kind="add", learning=human, author="human:x")]
    )
    frontier = build_frontier([("r1", run.run_root, head)], [], [PIN_NAME])
    assert cache.find(
        frontier, query=None, filters=LearningFilter(), relevance_goal=("absent", "absent")
    ) == ["L-a", "L-z"]
    cache.close()


def test_scores_join_large_named_frontiers_with_linear_work(tmp_path: Path) -> None:
    work = []
    for size in (1_000, 10_000, 50_000):
        cache = ArenaCache(tmp_path / str(size) / "toy.sqlite")
        names = [f"toy/reviews/alice/20261018T000000.000Z-{i:08x}.jsonl" for i in range(size)]
        cache.apply_human_objects(names, [_human_review(f"s-{i:08x}") for i in range(size)])
        frontier = Frontier(reviews=names)
        cache._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 64)
        steps = 0

        def tick() -> int:
            nonlocal steps
            steps += 1000
            return 0

        cache._conn.set_progress_handler(tick, 1000)
        rows = cache.scores(frontier)
        cache._conn.set_progress_handler(None, 0)
        assert len(rows) == size
        assert rows[0][0].id == "s-00000000"
        assert rows[-1][0].id == f"s-{size - 1:08x}"
        assert cache.scores(Frontier()) == []
        assert [score.id for score, _ in cache.scores(Frontier(reviews=names[:1]))] == [
            "s-00000000"
        ]
        assert not cache._conn.in_transaction
        work.append(steps)
        cache.close()
    print("scores named-frontier VM steps (1k/10k/50k):", work)
    assert 0 < work[1] < 15 * work[0]
    assert work[2] < 8 * work[1]
