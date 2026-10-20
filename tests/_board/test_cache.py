import sqlite3
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier

import pytest
from _fakes.journals import START, FakeRun, candidate_ids
from _fakes.secrecy import assert_secret_absent

from swarm_arena._board.cache import ArenaCache, goal_key_digest
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
