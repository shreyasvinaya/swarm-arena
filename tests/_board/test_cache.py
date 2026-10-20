import sqlite3
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from _fakes.journals import FakeRun, candidate_ids
from _fakes.secrecy import assert_secret_absent

from swarm_arena._board.cache import ArenaCache, goal_key_digest
from swarm_arena._board.ingest import ingest_run
from swarm_arena._errors import ChainError
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.replay import RunState


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
