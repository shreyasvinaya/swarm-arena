"""Exercise lease authority, conditional writes, recovery, and stored loading."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from threading import Barrier

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

import swarm_arena._journal.lease as lease_mod
from swarm_arena._errors import (
    AlreadyExists,
    Fenced,
    LeaseHeld,
    NotFound,
    PreconditionFailed,
    StorageError,
)
from swarm_arena._journal.lease import OWNER_KEY, Head, Lease, OwnerDoc, read_owner
from swarm_arena._model.base import FrozenModel, StoredModel
from swarm_arena._model.canonical import canonical_json, utc_now
from swarm_arena._store.base import Blob
from swarm_arena._store.local import LocalStore

_T0 = datetime(2026, 10, 18, 13, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lease_mod, "utc_now", lambda: _T0)


def test_acquire_new_creates_epoch_1(store: LocalStore) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    owner = read_owner(store)
    assert owner is not None
    doc, etag = owner
    assert (doc.epoch, doc.version, doc.head, doc.owner_id) == (1, 1, None, "o1")
    assert etag == lease.etag
    assert doc == lease.doc
    assert doc.host == "h"
    assert doc.expires_at == _T0 + timedelta(seconds=60)
    assert lease.ttl_s == 60
    assert OWNER_KEY == "control/owner.json"
    assert store.read(OWNER_KEY).data == canonical_json(doc.model_dump(mode="json"))


def test_second_acquire_fails(store: LocalStore) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    before = store.read(OWNER_KEY)
    with pytest.raises(AlreadyExists):
        Lease.acquire_new(store, owner_id="o2", host="h", ttl_s=60)
    assert store.read(OWNER_KEY) == before


def test_take_over_live_lease_needs_force(store: LocalStore) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    before = store.read(OWNER_KEY)
    with pytest.raises(LeaseHeld):
        Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=False)
    assert store.read(OWNER_KEY) == before


def test_force_take_over_bumps_epoch_keeps_head_and_fences(store: LocalStore) -> None:
    old = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    old.advance(Head(epoch=1, seq=100, segment="journal/e1-100.jsonl"))
    before = (old.doc, old.etag)
    new = Lease.take_over(store, owner_id="o2", host="h2", ttl_s=90, force=True)
    assert new.doc.epoch == 2
    assert new.doc.head == Head(epoch=1, seq=100, segment="journal/e1-100.jsonl")
    assert (new.doc.owner_id, new.doc.host, new.doc.version) == ("o2", "h2", 3)
    assert new.ttl_s == 90
    assert new.doc.expires_at == _T0 + timedelta(seconds=90)
    with pytest.raises(Fenced):
        old.advance(Head(epoch=1, seq=105, segment="journal/e1-105.jsonl"))
    with pytest.raises(Fenced):
        old.renew()
    with pytest.raises(Fenced):
        old.release()
    assert (old.doc, old.etag) == before
    assert read_owner(store) == (new.doc, new.etag)


def test_take_over_after_expiry_without_force(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    monkeypatch.setattr(lease_mod, "utc_now", lambda: _T0 + timedelta(seconds=61))
    assert Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=False).doc.epoch == 2


def test_release_keeps_epoch_and_head(store: LocalStore) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    lease.advance(Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))
    lease.release()
    owner = read_owner(store)
    assert owner is not None
    doc, _ = owner
    assert doc.head is not None
    assert doc.epoch == 1
    assert doc.head.seq == 3
    assert doc.expires_at < utc_now()
    assert doc.expires_at == _T0 - timedelta(seconds=1)
    assert (doc.version, doc.owner_id, doc.host) == (3, "o1", "h")
    assert Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=False).doc.epoch == 2


def test_renew_and_advance_bump_version_and_expiry(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    first = lease.doc.expires_at
    monkeypatch.setattr(lease_mod, "utc_now", lambda: _T0 + timedelta(seconds=5))
    lease.renew()
    assert lease.doc.version == 2
    assert lease.doc.expires_at >= first
    assert lease.doc.expires_at == _T0 + timedelta(seconds=65)
    assert lease.doc.head is None
    monkeypatch.setattr(lease_mod, "utc_now", lambda: _T0 + timedelta(seconds=10))
    head = Head(epoch=1, seq=3, segment="journal/e1-3.jsonl")
    lease.advance(head)
    assert (lease.doc.version, lease.doc.head) == (3, head)
    assert lease.doc.expires_at == _T0 + timedelta(seconds=70)
    assert read_owner(store) == (lease.doc, lease.etag)
    assert store.read(OWNER_KEY).data == canonical_json(lease.doc.model_dump(mode="json"))


def test_concurrent_take_over_has_one_winner(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60).release()
    barrier = Barrier(2)
    original_read = store.read

    def synchronized_read(key: str) -> Blob:
        blob = original_read(key)
        barrier.wait(timeout=5)
        return blob

    def take_over(owner_id: str) -> Lease | Fenced | LeaseHeld:
        try:
            return Lease.take_over(store, owner_id=owner_id, host="h", ttl_s=60, force=False)
        except (Fenced, LeaseHeld) as exc:
            return exc

    with monkeypatch.context() as patch:
        patch.setattr(store, "read", synchronized_read)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(take_over, ["o2", "o3"]))
    winners = [result for result in results if isinstance(result, Lease)]
    assert len(winners) == 1
    assert sum(isinstance(result, Fenced) for result in results) == 1
    assert read_owner(store) == (winners[0].doc, winners[0].etag)
    assert (winners[0].doc.epoch, winners[0].doc.version) == (2, 3)


def test_take_over_with_stale_expected_epoch_is_fenced(store: LocalStore) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
    before = store.read(OWNER_KEY)
    with pytest.raises(Fenced, match=r"another owner took the run \(epoch 2\)"):
        Lease.take_over(store, owner_id="o3", host="h", ttl_s=60, force=True, expect_epoch=1)
    owner = read_owner(store)
    assert owner is not None
    doc, _ = owner
    assert (doc.owner_id, doc.epoch) == ("o2", 2)
    assert store.read(OWNER_KEY) == before


def test_matching_expected_epoch_allows_takeover(store: LocalStore) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    lease = Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True, expect_epoch=1)
    assert lease.doc.epoch == 2


def test_expiry_boundary(store: LocalStore, monkeypatch: pytest.MonkeyPatch) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    assert not Lease.expired(lease.doc)
    monkeypatch.setattr(lease_mod, "utc_now", lambda: lease.doc.expires_at)
    assert Lease.expired(lease.doc)
    assert Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=False).doc.epoch == 2


def test_missing_owner(store: LocalStore) -> None:
    assert read_owner(store) is None
    with pytest.raises(NotFound):
        Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
    assert read_owner(store) is None


@pytest.mark.parametrize("operation", ["advance", "renew", "release", "take_over"])
@pytest.mark.parametrize("failure", [PreconditionFailed, StorageError, OSError])
def test_failed_replace_preserves_local_state(
    store: LocalStore,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure: type[Exception],
) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    before = (lease.doc, lease.etag)
    blob = store.read(OWNER_KEY)
    secret = "private-cas-failure"
    calls: list[tuple[str, bytes, str]] = []

    def fail(key: str, data: bytes, *, if_match: str) -> str:
        calls.append((key, data, if_match))
        if failure is PreconditionFailed:
            from swarm_arena._errors import _core_error

            raise _core_error(PreconditionFailed, secret)
        raise failure(secret)

    monkeypatch.setattr(store, "replace", fail)
    expected = Fenced if failure is PreconditionFailed else failure

    def perform() -> None:
        if operation == "advance":
            lease.advance(Head(epoch=1, seq=1, segment="journal/e1-1.jsonl"))
        elif operation == "take_over":
            Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
        else:
            getattr(lease, operation)()

    with pytest.raises(expected) as caught:
        perform()
    assert (lease.doc, lease.etag) == before
    assert store.read(OWNER_KEY) == blob
    assert len(calls) == 1
    assert (calls[0][0], calls[0][2]) == (OWNER_KEY, before[1])
    proposed = OwnerDoc.model_validate_json(calls[0][1])
    assert proposed.version == 2
    if failure is PreconditionFailed:
        assert_secret_absent(caught.value, secret, sanitized=True)
        if operation == "take_over":
            assert str(caught.value) == "another owner changed the lease"


def test_unknown_replace_outcome_requires_explicit_adoption(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    before = (lease.doc, lease.etag)
    original_replace = store.replace
    head = Head(epoch=1, seq=1, segment="journal/e1-1.jsonl")

    def replace_then_fail(key: str, data: bytes, *, if_match: str) -> str:
        original_replace(key, data, if_match=if_match)
        raise StorageError("unknown write outcome")

    monkeypatch.setattr(store, "replace", replace_then_fail)
    with pytest.raises(StorageError):
        lease.advance(head)
    assert lease.doc is before[0]
    assert lease.etag == before[1]
    owner = read_owner(store)
    assert owner is not None
    doc, etag = owner
    assert doc.head == head
    assert etag != lease.etag
    lease._adopt(doc, etag)
    assert lease.doc is doc
    assert lease.etag == etag
    assert lease.ttl_s == 60


def test_takeover_reads_once(store: LocalStore, monkeypatch: pytest.MonkeyPatch) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    original_read, original_replace = store.read, store.replace
    reads: list[str] = []
    replacements: list[str] = []

    def read(key: str) -> Blob:
        reads.append(key)
        return original_read(key)

    def replace(key: str, data: bytes, *, if_match: str) -> str:
        replacements.append(if_match)
        return original_replace(key, data, if_match=if_match)

    monkeypatch.setattr(store, "read", read)
    monkeypatch.setattr(store, "replace", replace)
    before = original_read(OWNER_KEY)
    Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
    assert reads == [OWNER_KEY]
    assert replacements == [before.etag]


@pytest.mark.parametrize("failure", [StorageError, OSError])
def test_read_owner_propagates_storage_failure(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    def fail(key: str) -> Blob:
        raise failure("read failed")

    monkeypatch.setattr(store, "read", fail)
    with pytest.raises(failure):
        read_owner(store)


@pytest.mark.parametrize(
    "data",
    [
        b'{"private-owner-input":',
        b'{"host":"private-owner-input","epoch":"private-owner-input"}',
        b'{"private-owner-input":"private-owner-input"}',
        b'{"head":{"seq":"private-owner-input"}}',
        b'{"host":"private-owner-input\xff"}',
    ],
)
def test_read_owner_malformed_bytes_are_sanitized(store: LocalStore, data: bytes) -> None:
    store.create(OWNER_KEY, data)
    with pytest.raises(ValidationError) as caught:
        read_owner(store)
    assert_secret_absent(caught.value, "private-owner-input", sanitized=True)


def test_owner_models_are_frozen_and_use_utc_milliseconds() -> None:
    head = Head(epoch=1, seq=1, segment="journal/e1-1.jsonl")
    doc = OwnerDoc(
        version=1,
        epoch=1,
        owner_id="o1",
        host="h",
        expires_at=datetime(
            2026, 10, 18, 18, 30, 0, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30))
        ),
        head=head,
    )
    assert isinstance(head, FrozenModel)
    assert isinstance(doc, StoredModel)
    assert OwnerDoc.schema_kind == "owner"
    assert doc.expires_at == _T0 + timedelta(milliseconds=123)
    assert doc.model_dump(mode="json")["expires_at"] == "2026-10-18T13:00:00.123Z"
    assert OwnerDoc.model_validate_json(canonical_json(doc.model_dump(mode="json"))) == doc
    for model in (head, doc):
        with pytest.raises(ValidationError, match="frozen_instance"):
            model.epoch = 2  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        OwnerDoc.model_validate({**doc.model_dump(), "expires_at": _T0.replace(tzinfo=None)})


def test_read_owner_refuses_newer_schema(store: LocalStore) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    raw = {**lease.doc.model_dump(mode="json"), "schema_version": 2}
    store.replace(OWNER_KEY, canonical_json(raw), if_match=lease.etag)
    with pytest.raises(ValueError, match="newer"):
        read_owner(store)
