from pathlib import Path

import pytest
from _fakes.journals import FakeRun, candidate_ids

from swarm_arena._board.cache import ArenaCache
from swarm_arena._board.ingest import _validate_marks, ensure_covered, ingest_run
from swarm_arena._errors import ChainError
from swarm_arena._journal.lease import Head
from swarm_arena._model.board_types import Frontier
from swarm_arena._store.base import Blob


def mark(run: FakeRun, head: Head) -> Frontier.Mark:
    return Frontier.Mark(run_id=run.run_id, run_root=run.run_root, **head.model_dump())


def test_incremental_download_reads_only_new_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan())
    run.commit(run.candidate("c-1"))
    head = run.commit(run.candidate("c-2"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    assert ingest_run(cache, run.store, "r1", run.run_root, head) == 3
    run.commit(run.candidate("c-3"))
    head = run.commit(run.candidate("c-4"))
    reads: list[str] = []
    original = run.store.read

    def read(key: str) -> Blob:
        reads.append(key)
        return original(key)

    monkeypatch.setattr(run.store, "read", read)
    assert ingest_run(cache, run.store, "r1", run.run_root, head) == 2
    assert len(reads) == 2
    assert ingest_run(cache, run.store, "r1", run.run_root, head) == 0
    cache.close()


def test_never_lists_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan())

    def no_list(prefix: str = "") -> list[str]:
        raise AssertionError(prefix)

    monkeypatch.setattr(run.store, "list", no_list)
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    cache.close()


def test_takeover_chain_ignores_fenced_segment(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan(), run.candidate("c-1"))
    run.take_over()
    run.orphan(run.candidate("c-ghost"))
    head = run.commit(run.candidate("c-2"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    assert candidate_ids(cache) == ["c-1", "c-2"]
    cache.close()


def test_fixture_takeover_retries_owner_renewal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(), run.candidate("c-1"))
    original = run.store.replace
    renewed = False

    def replace(key: str, data: bytes, *, if_match: str) -> str:
        nonlocal renewed
        if not renewed:
            renewed = True
            # Change the real owner version between takeover's read and conditional write.
            run._lease.renew()
        return original(key, data, if_match=if_match)

    monkeypatch.setattr(run.store, "replace", replace)
    run.take_over()
    assert renewed
    assert run._lease.doc.owner_id == "replacement"
    assert run._lease.doc.epoch == 2
    assert run._lease.doc.head == head
    run._lease.release()


def test_older_mark_is_covered_by_newer_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan())
    old = run.commit(run.candidate("c-1"))
    head = run.commit(run.candidate("c-2"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)

    def no_store(*args: object) -> None:
        pytest.fail("cached mark must not access storage")

    monkeypatch.setattr("swarm_arena._board.ingest.run_store", no_store)
    ensure_covered(cache, mark(run, old))
    cache.close()


def test_mark_not_on_cached_chain_raises(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan())
    run.take_over()
    orphan = run.orphan(run.candidate("ghost"))
    head = run.commit(run.candidate("real"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    with pytest.raises(ChainError):
        ensure_covered(
            cache,
            Frontier.Mark(
                run_id="r1", run_root=run.run_root, epoch=1, seq=head.seq, segment=orphan
            ),
        )
    cache.close()


def test_ensure_covered_ingests_from_recorded_run_root(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "recorded"), "r1")
    run.commit(run.plan())
    old = run.commit(run.candidate("c-1"))
    run.commit(run.candidate("c-2"))
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ensure_covered(cache, mark(run, old))
    assert candidate_ids(cache) == ["c-1"]
    cache.close()


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("field", ["epoch", "seq"])
def test_rejects_forged_mark_and_head(tmp_path: Path, warm: bool, field: str) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan())
    cache = ArenaCache(tmp_path / "toy.sqlite")
    if warm:
        ingest_run(cache, run.store, "r1", run.run_root, head)
    forged = head.model_copy(update={field: 99})
    with pytest.raises(ChainError):
        ensure_covered(cache, mark(run, forged))
    with pytest.raises(ChainError):
        ingest_run(cache, run.store, "r1", run.run_root, forged)
    cache.close()


def test_rejects_wrong_run_and_conflicting_duplicate_marks(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan())
    cache = ArenaCache(tmp_path / "toy.sqlite")
    with pytest.raises(ChainError):
        ingest_run(cache, run.store, "r2", run.run_root, head)
    m = mark(run, head)
    assert _validate_marks([m, m]) == [m]
    with pytest.raises(ChainError):
        _validate_marks([m, m.model_copy(update={"run_root": "/other"})])
    cache.close()


def test_reader_rejects_unlinked_noninitial_segment(tmp_path: Path) -> None:
    from swarm_arena._journal.reader import read_chain
    from swarm_arena._journal.segment import encode_segment

    run = FakeRun(str(tmp_path / "runs"), "r1")
    first = run.commit(run.plan())
    second = run.commit(run.candidate("c"))
    events = read_chain(run.store, second)[-1][1]
    forged_events = [e.model_copy(update={"epoch": 2}) for e in events]
    run.store.create("journal/e2-2.jsonl", encode_segment(None, forged_events))
    head = Head(epoch=2, seq=2, segment="journal/e2-2.jsonl")
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, first)
    with pytest.raises(ChainError, match="first segment"):
        ingest_run(cache, run.store, "r1", run.run_root, head)
    cache.close()


def test_cached_metadata_cannot_widen_coverage(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan())
    cache = ArenaCache(tmp_path / "toy.sqlite")
    ingest_run(cache, run.store, "r1", run.run_root, head)
    cache._conn.execute("UPDATE segments SET last_seq=100")
    with pytest.raises(ChainError, match="metadata"):
        ensure_covered(cache, mark(run, head))
    cache.close()


def test_download_does_not_hold_cache_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures import ThreadPoolExecutor

    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan())
    cache = ArenaCache(tmp_path / "toy.sqlite")
    original = run.store.read
    with ThreadPoolExecutor(1) as pool:

        def read(key: str) -> Blob:
            assert pool.submit(cache.newest_segment, "r1").result(timeout=5) is None
            return original(key)

        monkeypatch.setattr(run.store, "read", read)
        ingest_run(cache, run.store, "r1", run.run_root, head)
    cache.close()


def test_concurrent_ingest_counts_only_its_insertions(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(run.plan())
    for i in range(10):
        run.commit(run.candidate(f"c-{i}"))
    barrier = Barrier(2)

    def worker() -> int:
        cache = ArenaCache(tmp_path / "toy.sqlite")
        try:
            barrier.wait(timeout=5)
            return ingest_run(cache, run.store, "r1", run.run_root, run.heads[-1])
        finally:
            cache.close()

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(worker) for _ in range(2)]
        assert sum(f.result(timeout=30) for f in futures) == 11


def test_conflicting_marks_keep_registered_cli_diagnostic(tmp_path: Path) -> None:
    from swarm_arena._cli.main import _error_message
    from swarm_arena._errors import _rebuild_core_signal

    run = FakeRun(str(tmp_path / "runs"), "r1")
    original = mark(run, run.commit(run.plan()))
    other = original.model_copy(update={"run_root": "/other"})
    with pytest.raises(ChainError) as caught:
        _validate_marks([original, other])
    expected = str(caught.value)
    caught.value.args = ("MUTATED-SECRET",)
    caught.value.add_note("MUTATED-SECRET")
    assert _rebuild_core_signal(caught.value) is not None
    assert _error_message(caught.value) == expected


def test_run_store_construction_does_not_create_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._board.ingest import run_store

    root = tmp_path / "not-created"

    def no_mkdir(*args: object, **kwargs: object) -> None:
        raise AssertionError("store construction must not create directories")

    monkeypatch.setattr("os.mkdir", no_mkdir)
    store = run_store(str(root), "r1")
    assert store.url == str(root / "r1")
    assert not root.exists()
