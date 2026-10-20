"""Exercise single-writer publication, recovery, expiry, and admission."""

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Lock, get_ident

import anyio
import pytest

from swarm_arena._errors import ChainError, Fenced, StorageError, WriteOutcomeUnknown
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import utc_now
from swarm_arena._store.base import Blob, Store
from swarm_arena._store.local import LocalStore


class FlakyStore(Store):
    def __init__(
        self,
        inner: Store,
        *,
        create_delay_s: float = 0,
        unknown_after_create: int = 0,
        unknown_before_create: int = 0,
        unknown_after_replace: int = 0,
        unknown_before_replace: int = 0,
        after_unknown: Callable[[Store], None] | None = None,
        create_raises: BaseException | None = None,
        replace_raises: BaseException | None = None,
        prewrite: Mapping[str, bytes] | None = None,
    ) -> None:
        self.inner = inner
        self.create_delay_s = create_delay_s
        self.unknown_after_create = unknown_after_create
        self.unknown_before_create = unknown_before_create
        self.unknown_after_replace = unknown_after_replace
        self.unknown_before_replace = unknown_before_replace
        self.after_unknown = after_unknown
        self.create_raises = create_raises
        self.replace_raises = replace_raises
        self.max_concurrent_creates = 0
        self.active_creates = 0
        self.create_calls = 0
        self.replace_calls = 0
        self.replaces_at_first_create: int | None = None
        self.thread_ids: list[int] = []
        self.lock = Lock()
        self.create_started: anyio.Event | None = None
        for key, data in (prewrite or {}).items():
            inner.create(key, data)

    @property
    def url(self) -> str:
        return self.inner.url

    def read(self, key: str) -> Blob:
        self.thread_ids.append(get_ident())
        return self.inner.read(key)

    def _unknown(self) -> None:
        try:
            raise WriteOutcomeUnknown("response lost")
        finally:
            if self.after_unknown is not None:
                callback, self.after_unknown = self.after_unknown, None
                callback(self.inner)

    def create(self, key: str, data: bytes) -> str:
        self.thread_ids.append(get_ident())
        if not key.startswith("journal/"):
            return self.inner.create(key, data)
        with self.lock:
            self.active_creates += 1
            self.create_calls += 1
            if self.replaces_at_first_create is None:
                self.replaces_at_first_create = self.replace_calls
            self.max_concurrent_creates = max(self.max_concurrent_creates, self.active_creates)
        try:
            if self.create_started is not None:
                anyio.from_thread.run_sync(self.create_started.set)
            time.sleep(self.create_delay_s)
            if self.create_raises is not None:
                raise self.create_raises
            if self.unknown_before_create:
                self.unknown_before_create -= 1
                self._unknown()
            etag = self.inner.create(key, data)
            if self.unknown_after_create:
                self.unknown_after_create -= 1
                self._unknown()
            return etag
        finally:
            with self.lock:
                self.active_creates -= 1

    def replace(self, key: str, data: bytes, *, if_match: str) -> str:
        self.thread_ids.append(get_ident())
        self.replace_calls += 1
        if self.replace_raises is not None:
            raise self.replace_raises
        if self.unknown_before_replace:
            self.unknown_before_replace -= 1
            self._unknown()
        etag = self.inner.replace(key, data, if_match=if_match)
        if self.unknown_after_replace:
            self.unknown_after_replace -= 1
            self._unknown()
        return etag

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)

    def list(self, prefix: str) -> list[str]:
        raise AssertionError("journal must not list storage")


def _writer(
    store: Store,
    tmp_path: Path,
    *,
    ttl_s: float = 60,
    sync_interval_s: float = 60,
    group_window_s: float = 0.2,
) -> JournalWriter:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=ttl_s)
    return JournalWriter(
        store,
        lease,
        run_id="r",
        next_seq=1,
        local_path=tmp_path / "journal.local.jsonl",
        sync_interval_s=sync_interval_s,
        group_window_s=group_window_s,
    )


async def test_append_durable_returns_after_head_covers_it(
    store: LocalStore, tmp_path: Path
) -> None:
    async with _writer(store, tmp_path) as w:
        ev = await w.append_durable(EventDraft(type=EventType.RESERVE, payload={"id": "r1"}))
        owner = read_owner(store)
        assert owner is not None
        doc, _ = owner
        assert doc.head is not None
        assert doc.head.seq >= ev.seq == 1


async def test_group_commit_shares_one_segment(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w, anyio.create_task_group() as tg:
        for i in range(5):
            tg.start_soon(w.append_durable, EventDraft(type=EventType.RESERVE, payload={"i": i}))
    assert len(w.committed) == 1


async def test_commits_run_one_at_a_time(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_delay_s=0.05)
    async with _writer(flaky, tmp_path, group_window_s=0.0) as w, anyio.create_task_group() as tg:
        for _ in range(10):
            tg.start_soon(w.append_durable, EventDraft(type=EventType.SETTLE))
    assert flaky.max_concurrent_creates == 1


async def test_every_committed_event_once_in_seq_order(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        for i in range(20):
            w.append(EventDraft(type=EventType.LEARNING, payload={"i": i}))
            if i % 3 == 0:
                await w.commit()
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head is not None
    assert [e.seq for _, evs in read_chain(store, owner[0].head) for e in evs] == list(range(1, 21))


async def test_unknown_create_reuses_same_segment_by_sha(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_after_create=1)
    async with _writer(flaky, tmp_path) as w:
        await w.append_durable(EventDraft(type=EventType.RESERVE))
    assert w.head is not None
    assert w.head.seq == 1


async def test_unknown_advance_that_succeeded_does_not_duplicate(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_after_replace=1)
    async with _writer(flaky, tmp_path) as w:
        await w.append_durable(EventDraft(type=EventType.RESERVE))
        await w.append_durable(EventDraft(type=EventType.SETTLE))
    owner = read_owner(flaky)
    assert owner is not None
    assert owner[0].head is not None
    assert [e.seq for _, evs in read_chain(flaky, owner[0].head) for e in evs] == [1, 2]


async def test_unknown_advance_that_failed_recommits_same_events(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_before_replace=1)
    async with _writer(flaky, tmp_path) as w:
        await w.append_durable(EventDraft(type=EventType.RESERVE))
    owner = read_owner(flaky)
    assert owner is not None
    assert owner[0].head is not None
    assert [e.seq for _, evs in read_chain(flaky, owner[0].head) for e in evs] == [1]


def _take(s: Store) -> None:
    Lease.take_over(s, owner_id="o2", host="h", ttl_s=60, force=True)


async def test_unknown_advance_succeeded_then_takeover_is_fenced(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_after_replace=1, after_unknown=_take)
    async with _writer(flaky, tmp_path) as w:
        with pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert w.fenced.is_set()
        owner = read_owner(flaky)
        assert owner is not None
        version_after_takeover = owner[0].version
    owner = read_owner(flaky)
    assert owner is not None
    doc, _ = owner
    assert (doc.owner_id, doc.epoch, doc.version) == ("o2", 2, version_after_takeover)


async def test_unknown_advance_failed_then_takeover_is_fenced(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_before_replace=1, after_unknown=_take)
    async with _writer(flaky, tmp_path) as w:
        with pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.RESERVE))
    owner = read_owner(flaky)
    assert owner is not None
    assert owner[0].owner_id == "o2"
    assert owner[0].head is None


async def test_storage_error_until_expiry_fails_waiters(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_raises=StorageError("disk full"))
    async with _writer(flaky, tmp_path, ttl_s=0.3) as w:
        with anyio.fail_after(2), pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert w.fenced.is_set()
    assert flaky.replace_calls == flaky.replaces_at_first_create


async def test_chain_error_fails_waiters_at_once(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), prewrite={"journal/e1-1.jsonl": b"junk"})
    async with _writer(flaky, tmp_path) as w:
        with pytest.raises(Fenced) as err:
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert isinstance(err.value.__cause__, ChainError)
        with anyio.fail_after(0.1), pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.SETTLE))
    owner = read_owner(flaky)
    assert owner is not None
    assert owner[0].version == 1


async def test_takeover_fences_writer(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        await w.append_durable(EventDraft(type=EventType.RESERVE))
        Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
        with pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.SETTLE))
        assert w.fenced.is_set()
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head is not None
    assert owner[0].epoch == 2
    assert owner[0].head.seq == 1


async def test_lease_renewed_every_third_of_ttl(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path, ttl_s=0.6):
        await anyio.sleep(0.5)
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].version >= 2


async def test_fenced_when_renewal_fails_until_expiry(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), replace_raises=StorageError("down"))
    async with _writer(flaky, tmp_path, ttl_s=0.3) as w:
        with anyio.fail_after(1.0):
            await w.fenced.wait()


async def test_sync_interval_commits_ordinary_events(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path, sync_interval_s=0.1) as w:
        w.append(EventDraft(type=EventType.LEARNING))
        await anyio.sleep(0.4)
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].head is not None
        assert owner[0].head.seq == 1


async def test_exit_commits_then_releases(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        w.append(EventDraft(type=EventType.LEARNING))
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head is not None
    assert owner[0].head.seq == 1
    assert owner[0].expires_at < utc_now()


async def test_events_also_appended_to_local_file(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        w.append(EventDraft(type=EventType.LEARNING))
    assert len((tmp_path / "journal.local.jsonl").read_text().splitlines()) == 1


async def test_unknown_create_not_found_retries(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_before_create=1)
    async with _writer(flaky, tmp_path, group_window_s=0) as w:
        with anyio.fail_after(2):
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert w.head is not None
        assert w.head.seq == 1
    assert flaky.create_calls == 2


async def test_oserror_retries_until_expiry_without_renewal(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_raises=OSError("disk full"))
    async with _writer(flaky, tmp_path, ttl_s=0.3, group_window_s=0) as w:
        with anyio.fail_after(0.8), pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.RESERVE))
    assert flaky.create_calls == 1
    assert flaky.replace_calls == 0


async def test_store_operations_run_off_loop(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_delay_s=0.1)
    writer = _writer(flaky, tmp_path, group_window_s=0)
    flaky.create_started = anyio.Event()
    flaky.thread_ids.clear()
    async with writer as w:
        w.append(EventDraft(type=EventType.LEARNING))
        async with anyio.create_task_group() as tg:
            tg.start_soon(w.commit)
            with anyio.fail_after(0.08):
                await flaky.create_started.wait()
            assert w.head is None
    assert flaky.thread_ids
    assert get_ident() not in flaky.thread_ids


@pytest.mark.parametrize("after", [True, False])
async def test_unknown_renewal_checks_identity(tmp_path: Path, after: bool) -> None:
    flaky = FlakyStore(
        LocalStore(tmp_path / "run"),
        unknown_after_replace=int(after),
        unknown_before_replace=int(not after),
        after_unknown=_take,
    )
    async with _writer(flaky, tmp_path, ttl_s=0.3) as w:
        with anyio.fail_after(0.8):
            await w.fenced.wait()
        owner = read_owner(flaky)
        assert owner is not None
        version = owner[0].version
    assert read_owner(flaky) == owner
    assert version == 2 + int(after)


async def test_inherited_chain_and_next_sequence(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path, group_window_s=0) as first:
        await first.append_durable(EventDraft(type=EventType.RESERVE))
    lease = Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
    writer = JournalWriter(
        store,
        lease,
        run_id="r",
        next_seq=2,
        local_path=tmp_path / "resumed.jsonl",
        sync_interval_s=60,
        group_window_s=0,
    )
    async with writer as w:
        assert list(w.committed) == list(first.committed)
        event = await w.append_durable(EventDraft(type=EventType.SETTLE))
        assert (event.epoch, event.seq, w.epoch) == (2, 2, 2)
        assert list(w.committed) == ["journal/e1-1.jsonl", "journal/e2-2.jsonl"]


@pytest.mark.parametrize("next_seq", [0, 2])
async def test_wrong_next_sequence_rejected(
    store: LocalStore,
    tmp_path: Path,
    next_seq: int,
) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    with pytest.raises(ValueError, match="next_seq"):
        async with JournalWriter(
            store,
            lease,
            run_id="r",
            next_seq=next_seq,
            local_path=tmp_path / "local.jsonl",
            sync_interval_s=60,
        ):
            pytest.fail("invalid initial sequence accepted")
    assert not (tmp_path / "local.jsonl").exists()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object(), "\ud800"])
async def test_invalid_payload_rejected_before_admission(
    store: LocalStore,
    tmp_path: Path,
    invalid: object,
) -> None:
    from _fakes.secrecy import assert_secret_absent

    secret = "PAYLOAD_PRIVATE_SENTINEL"
    draft = EventDraft.model_construct(type=EventType.LEARNING, payload={secret: [invalid]})
    async with _writer(store, tmp_path) as w:
        with pytest.raises(ValueError, match="payload") as caught:
            w.append(draft)
        assert_secret_absent(caught.value, secret)
        assert w.head is None
        assert not (tmp_path / "journal.local.jsonl").exists()
        event = w.append(EventDraft(type=EventType.LEARNING, payload={"null": None}))
        assert event.seq == 1


async def test_local_and_committed_payloads_are_canonical_snapshots(
    store: LocalStore,
    tmp_path: Path,
) -> None:
    from swarm_arena._model.canonical import canonical_json

    draft = EventDraft(type=EventType.LEARNING, payload={"b": [None, "\u2028"], "a": 1})
    async with _writer(store, tmp_path) as w:
        event = w.append(draft)
        expected = canonical_json(event.model_dump(mode="json")) + b"\n"
        assert event.context_id == event.run_id == "r"
        draft.payload["a"] = 2
        event.payload["a"] = 3
        await w.commit()
        assert w.head is not None
        committed = read_chain(store, w.head)[0][1][0]
        assert canonical_json(committed.model_dump(mode="json")) + b"\n" == expected
    assert (tmp_path / "journal.local.jsonl").read_bytes() == expected


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
async def test_failure_fails_all_waiters_without_secret(
    tmp_path: Path,
    error_type: type[Exception],
) -> None:
    from _fakes.secrecy import assert_secret_absent

    secret = "WRITER_PRIVATE_SENTINEL"
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_raises=error_type(secret))
    failures: list[Fenced] = []
    async with _writer(flaky, tmp_path) as w:

        async def append() -> None:
            with pytest.raises(Fenced) as caught:
                await w.append_durable(EventDraft(type=EventType.RESERVE))
            failures.append(caught.value)

        with anyio.fail_after(1):
            async with anyio.create_task_group() as tg:
                for _ in range(4):
                    tg.start_soon(append)
        for failure in failures:
            assert str(failure) == f"journal writer failed: {error_type.__name__}"
            assert_secret_absent(failure, secret, sanitized=True)
        with pytest.raises(Fenced):
            await w.commit()
        with pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.SETTLE))
    assert len(failures) == 4
    assert flaky.replace_calls == 0
    assert secret.encode() not in (tmp_path / "journal.local.jsonl").read_bytes()
    assert flaky.inner.list("journal/") == []


async def test_cancelled_exit_finishes_commit_and_release(
    store: LocalStore, tmp_path: Path
) -> None:
    with anyio.CancelScope() as scope:
        async with _writer(store, tmp_path) as w:
            w.append(EventDraft(type=EventType.LEARNING))
            scope.cancel()
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head is not None
    assert owner[0].head.seq == 1
    assert owner[0].expires_at < utc_now()


async def test_write_through_append_wakes_writer(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path, group_window_s=0.01) as w:
        w.append(EventDraft(type=EventType.RESERVE))
        with anyio.fail_after(0.5):
            while w.head is None:  # noqa: ASYNC110 - observe publication without requesting a commit
                await anyio.sleep(0.001)
        assert w.head.seq == 1


@pytest.mark.parametrize("succeeded", [True, False])
async def test_unknown_advance_retries_recovery_read_before_another_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    succeeded: bool,
) -> None:
    from swarm_arena._journal.lease import OWNER_KEY

    flaky = FlakyStore(
        LocalStore(tmp_path / "run"),
        unknown_after_replace=int(succeeded),
        unknown_before_replace=int(not succeeded),
    )
    original_read = flaky.read
    read_failures = 1

    def read(key: str) -> Blob:
        nonlocal read_failures
        if key == OWNER_KEY and flaky.replace_calls and read_failures:
            read_failures -= 1
            raise StorageError("read unavailable")
        return original_read(key)

    monkeypatch.setattr(flaky, "read", read)
    async with _writer(flaky, tmp_path, ttl_s=5, group_window_s=0) as w:
        with anyio.fail_after(3):
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert w.head is not None
        assert w.head.seq == 1
        assert flaky.replace_calls == (1 if succeeded else 2)
    assert read_failures == 0


async def test_unknown_advance_rejects_unexpected_same_owner_head(tmp_path: Path) -> None:
    from swarm_arena._journal.lease import OWNER_KEY, Head, OwnerDoc
    from swarm_arena._model.canonical import canonical_json

    def change_head(store: Store) -> None:
        owner = read_owner(store)
        assert owner is not None
        doc, etag = owner
        unexpected = OwnerDoc.model_validate(
            {
                **doc.model_dump(),
                "head": Head(epoch=1, seq=9, segment="journal/e1-9.jsonl"),
            }
        )
        store.replace(OWNER_KEY, canonical_json(unexpected.model_dump(mode="json")), if_match=etag)

    flaky = FlakyStore(
        LocalStore(tmp_path / "run"),
        unknown_after_replace=1,
        after_unknown=change_head,
    )
    writer = _writer(flaky, tmp_path, group_window_s=0)
    async with writer as w:
        with pytest.raises(Fenced) as caught:
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert isinstance(caught.value.__cause__, ChainError)
        assert w.head is None
    assert flaky.replace_calls == 1


async def test_unknown_create_conflicting_bytes_is_fatal(tmp_path: Path) -> None:
    def create_conflict(store: Store) -> None:
        store.create("journal/e1-1.jsonl", b"different bytes")

    flaky = FlakyStore(
        LocalStore(tmp_path / "run"),
        unknown_before_create=1,
        after_unknown=create_conflict,
    )
    async with _writer(flaky, tmp_path, group_window_s=0) as w:
        with anyio.fail_after(0.5), pytest.raises(Fenced) as caught:
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert isinstance(caught.value.__cause__, ChainError)
    assert flaky.create_calls == 1
    assert flaky.replace_calls == 0


async def test_expiry_during_create_prevents_head_advance(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_delay_s=0.2)
    async with _writer(flaky, tmp_path, ttl_s=0.1, group_window_s=0) as w:
        with anyio.fail_after(0.6), pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.RESERVE))
    assert flaky.replace_calls == 0
    assert w.committed == ()


async def test_append_during_commit_forms_consecutive_second_segment(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_delay_s=0.05)
    flaky.create_started = anyio.Event()
    async with _writer(flaky, tmp_path, group_window_s=0) as w:
        async with anyio.create_task_group() as tg:
            tg.start_soon(w.append_durable, EventDraft(type=EventType.RESERVE))
            await flaky.create_started.wait()
            await w.append_durable(EventDraft(type=EventType.SETTLE))
        assert list(w.committed) == ["journal/e1-1.jsonl", "journal/e1-2.jsonl"]
        assert w.head is not None
        assert [e.seq for _, events in read_chain(flaky, w.head) for e in events] == [1, 2]


async def test_cancelled_waiter_does_not_cancel_publication(
    store: LocalStore, tmp_path: Path
) -> None:
    async with _writer(store, tmp_path, group_window_s=0.1) as w:
        with anyio.move_on_after(0.01) as cancelled:
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert cancelled.cancel_called
        await w.commit()
        assert w.head is not None
        assert w.head.seq == 1


async def test_final_commit_failure_is_reported(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_raises=RuntimeError("failure"))
    with pytest.raises(Fenced, match="RuntimeError"):
        async with _writer(flaky, tmp_path) as w:
            w.append(EventDraft(type=EventType.LEARNING))
    assert w.fenced.is_set()
    assert flaky.replace_calls == 0


async def test_local_io_failure_fences_without_admitting_event(
    store: LocalStore,
    tmp_path: Path,
) -> None:
    (tmp_path / "journal.local.jsonl").mkdir()
    async with _writer(store, tmp_path) as w:
        with pytest.raises(Fenced, match="IsADirectoryError"):
            w.append(EventDraft(type=EventType.LEARNING))
        with pytest.raises(Fenced):
            await w.commit()
    assert store.list("journal/") == []
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].version == 1
