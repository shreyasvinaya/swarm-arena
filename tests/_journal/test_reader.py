"""Exercise committed-chain traversal and its authority and validation boundaries."""

from datetime import UTC, datetime

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import ChainError, NotFound, StorageError
from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.lease import OWNER_KEY, Head, OwnerDoc
from swarm_arena._journal.reader import read_chain, read_committed
from swarm_arena._journal.segment import encode_segment, segment_name
from swarm_arena._model.canonical import canonical_json
from swarm_arena._store.base import Blob
from swarm_arena._store.local import LocalStore

_T0 = datetime(2026, 10, 18, 13, 0, tzinfo=UTC)
_SECRET = "private-reader-sentinel"


def _event(seq: int, *, epoch: int = 1) -> Event:
    return Event(
        epoch=epoch,
        seq=seq,
        ts=_T0,
        run_id="r",
        context_id="r",
        type=EventType.TOOL_END,
    )


def _put(
    store: LocalStore,
    link: str | None,
    seqs: list[int],
    epoch: int = 1,
    name: str | None = None,
) -> None:
    store.create(
        name if name is not None else segment_name(epoch, seqs[-1]),
        encode_segment(link, [_event(seq, epoch=epoch) for seq in seqs]),
    )


def _owner(store: LocalStore, head: Head | None, *, epoch: int = 1) -> OwnerDoc:
    doc = OwnerDoc(version=1, epoch=epoch, owner_id="o", host="h", expires_at=_T0, head=head)
    store.create(OWNER_KEY, canonical_json(doc.model_dump(mode="json")))
    return doc


@pytest.fixture
def reads(store: LocalStore, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    keys: list[str] = []
    original = store.read

    def read(key: str) -> Blob:
        keys.append(key)
        return original(key)

    def no_list(prefix: str) -> list[str]:
        pytest.fail("journal traversal must not list storage")

    monkeypatch.setattr(store, "read", read)
    monkeypatch.setattr(store, "list", no_list)
    return keys


def test_reads_chain_oldest_first(store: LocalStore) -> None:
    _put(store, None, [1, 2])
    _put(store, "journal/e1-2.jsonl", [3])
    chain = read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))
    assert [name for name, _ in chain] == ["journal/e1-2.jsonl", "journal/e1-3.jsonl"]
    assert [e.seq for _, evs in chain for e in evs] == [1, 2, 3]


def test_stop_after_is_exclusive(store: LocalStore) -> None:
    _put(store, None, [1])
    _put(store, "journal/e1-1.jsonl", [2])
    chain = read_chain(
        store,
        Head(epoch=1, seq=2, segment="journal/e1-2.jsonl"),
        stop_after="journal/e1-1.jsonl",
    )
    assert [n for n, _ in chain] == ["journal/e1-2.jsonl"]


def test_unlinked_segment_is_never_read(store: LocalStore) -> None:
    _put(store, None, [1, 2])
    _put(store, "journal/e1-2.jsonl", [3, 4, 5])
    _put(store, "journal/e1-2.jsonl", [3], epoch=2)
    chain = read_chain(store, Head(epoch=2, seq=3, segment="journal/e2-3.jsonl"))
    assert [n for n, _ in chain] == ["journal/e1-2.jsonl", "journal/e2-3.jsonl"]


def test_seq_gap_rejected(store: LocalStore) -> None:
    _put(store, None, [1])
    _put(store, "journal/e1-1.jsonl", [3])
    with pytest.raises(ChainError, match="gap"):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))


def test_head_seq_mismatch_rejected(store: LocalStore) -> None:
    _put(store, None, [1, 2])
    with pytest.raises(ChainError, match="head"):
        read_chain(store, Head(epoch=1, seq=5, segment="journal/e1-2.jsonl"))


def test_first_segment_must_start_at_seq_1(store: LocalStore) -> None:
    _put(store, None, [4])
    with pytest.raises(ChainError):
        read_chain(store, Head(epoch=1, seq=4, segment="journal/e1-4.jsonl"))


def test_gap_inside_segment_rejected(store: LocalStore) -> None:
    _put(store, None, [1, 3])
    with pytest.raises(ChainError):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))


def test_link_must_name_previous_seq(store: LocalStore) -> None:
    _put(store, None, [1, 2])
    _put(store, "journal/e1-5.jsonl", [3])
    with pytest.raises(ChainError, match="link"):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))


def test_link_cycle_rejected(store: LocalStore) -> None:
    _put(store, "journal/e1-4.jsonl", [1, 2])
    _put(store, "journal/e1-2.jsonl", [3, 4])
    with pytest.raises(ChainError):
        read_chain(store, Head(epoch=1, seq=4, segment="journal/e1-4.jsonl"))


def test_stop_after_must_be_previous_seq(store: LocalStore) -> None:
    _put(store, None, [1])
    _put(store, "journal/e1-1.jsonl", [3])
    with pytest.raises(ChainError):
        read_chain(
            store,
            Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"),
            stop_after="journal/e1-1.jsonl",
        )


def test_segment_body_must_match_its_name(store: LocalStore) -> None:
    _put(store, None, [1], name="journal/e1-2.jsonl")
    _put(store, "journal/e1-2.jsonl", [3])
    with pytest.raises(ChainError, match="name"):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))


def test_self_link_rejected(store: LocalStore) -> None:
    _put(store, "journal/e1-2.jsonl", [3], name="journal/e1-2.jsonl")
    with pytest.raises(ChainError):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-2.jsonl"))


def test_head_epoch_must_match_name(store: LocalStore, reads: list[str]) -> None:
    _put(store, None, [1])
    with pytest.raises(ChainError, match="head"):
        read_chain(store, Head(epoch=2, seq=1, segment="journal/e1-1.jsonl"))
    assert reads == []


def test_head_body_mismatch_names_head(store: LocalStore) -> None:
    _put(store, None, [1], name="journal/e1-2.jsonl")
    with pytest.raises(ChainError, match="head"):
        read_chain(store, Head(epoch=1, seq=2, segment="journal/e1-2.jsonl"))


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("at_head", [False, True])
def test_every_event_epoch_matches_segment_name(
    store: LocalStore, position: int, at_head: bool
) -> None:
    events = [_event(1), _event(2)]
    events[position] = events[position].model_copy(update={"epoch": 2})
    store.create("journal/e1-2.jsonl", encode_segment(None, events))
    if not at_head:
        _put(store, "journal/e1-2.jsonl", [3])
    seq = 2 if at_head else 3
    with pytest.raises(ChainError, match="epoch"):
        read_chain(store, Head(epoch=1, seq=seq, segment=segment_name(1, seq)))


@pytest.mark.parametrize(
    "name",
    [
        f"../{_SECRET}",
        f"journal/e1-1.jsonl/{_SECRET}",
        "journal/e01-1.jsonl",
        "journal/e1-01.jsonl",
        "journal/e0-1.jsonl",
        "journal/e1-0.jsonl",
        "journal/e1-1.jsonl\n",
    ],
)
@pytest.mark.parametrize("location", ["head", "link", "stop_after"])
def test_noncanonical_names_are_rejected_before_reading(
    store: LocalStore, reads: list[str], name: str, location: str
) -> None:
    _put(store, name if location == "link" else None, [2])
    head = Head(epoch=1, seq=2, segment=name if location == "head" else "journal/e1-2.jsonl")
    with pytest.raises(ChainError) as caught:
        read_chain(store, head, stop_after=name if location == "stop_after" else None)
    assert_secret_absent(caught.value, _SECRET, sanitized=True)
    assert name not in reads
    assert str(caught.value) == f"invalid {location} segment name"


def test_invalid_link_is_rejected_even_when_it_equals_stop_after(
    store: LocalStore, reads: list[str]
) -> None:
    link = f"../{_SECRET}"
    _put(store, link, [2])
    with pytest.raises(ChainError) as caught:
        read_chain(store, Head(epoch=1, seq=2, segment="journal/e1-2.jsonl"), stop_after=link)
    assert_secret_absent(caught.value, _SECRET, sanitized=True)
    assert link not in reads


def test_link_gap_rejected_before_reading_link(store: LocalStore, reads: list[str]) -> None:
    _put(store, "journal/e1-5.jsonl", [3])
    with pytest.raises(
        ChainError, match=r"seq gap: link 'journal/e1-5\.jsonl' ends at seq 5, expected 2"
    ):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))
    assert reads == ["journal/e1-3.jsonl"]


def test_stop_after_does_not_read_or_require_boundary_segment(
    store: LocalStore, reads: list[str]
) -> None:
    _put(store, "journal/e1-1.jsonl", [2])
    chain = read_chain(
        store, Head(epoch=1, seq=2, segment="journal/e1-2.jsonl"), stop_after="journal/e1-1.jsonl"
    )
    assert [name for name, _ in chain] == ["journal/e1-2.jsonl"]
    assert reads == ["journal/e1-2.jsonl"]


def test_stop_after_head_returns_empty_without_reads(store: LocalStore, reads: list[str]) -> None:
    head = Head(epoch=1, seq=1, segment="journal/e1-1.jsonl")
    assert read_chain(store, head, stop_after=head.segment) == []
    assert reads == []


def test_unreachable_stop_after_reads_to_first_segment(store: LocalStore, reads: list[str]) -> None:
    _put(store, None, [1])
    chain = read_chain(
        store, Head(epoch=1, seq=1, segment="journal/e1-1.jsonl"), stop_after="journal/e2-1.jsonl"
    )
    assert [name for name, _ in chain] == ["journal/e1-1.jsonl"]
    assert reads == ["journal/e1-1.jsonl"]


@pytest.mark.parametrize("missing_head", [False, True])
def test_missing_segment_is_chain_error(
    store: LocalStore, reads: list[str], missing_head: bool
) -> None:
    if not missing_head:
        _put(store, "journal/e1-1.jsonl", [2])
    with pytest.raises(ChainError, match="missing") as caught:
        read_chain(store, Head(epoch=1, seq=2, segment="journal/e1-2.jsonl"))
    assert type(caught.value) is ChainError
    assert caught.value.__cause__ is None
    assert reads == (
        ["journal/e1-2.jsonl"] if missing_head else ["journal/e1-2.jsonl", "journal/e1-1.jsonl"]
    )


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b'{"link":null}\n',
        b'{"link":"private-reader-sentinel\xff"}\n',
        b'{"link":"private-reader-sentinel"}\ninvalid\n',
        b'{"link":null,"private-reader-sentinel":"secret"}\n',
        b'{"schema_version":999,"link":"private-reader-sentinel"}\n',
        encode_segment(None, [_event(1)])[:-1],
        encode_segment(None, [_event(1)]) + b"\n",
    ],
)
def test_malformed_segment_becomes_safe_chain_error(store: LocalStore, data: bytes) -> None:
    store.create("journal/e1-1.jsonl", data)
    with pytest.raises(ChainError) as caught:
        read_chain(store, Head(epoch=1, seq=1, segment="journal/e1-1.jsonl"))
    assert_secret_absent(caught.value, _SECRET, sanitized=True)


@pytest.mark.parametrize("key", [OWNER_KEY, "journal/e1-2.jsonl", "journal/e1-1.jsonl"])
def test_storage_error_propagates_unchanged(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    _put(store, None, [1])
    _put(store, "journal/e1-1.jsonl", [2])
    _owner(store, Head(epoch=1, seq=2, segment="journal/e1-2.jsonl"))
    original = store.read
    error = StorageError("temporary read failure")

    def read(name: str) -> Blob:
        if name == key:
            raise error
        return original(name)

    monkeypatch.setattr(store, "read", read)
    with pytest.raises(StorageError) as caught:
        read_committed(store)
    assert caught.value is error


def test_missing_segment_error_suppresses_storage_details(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def read(key: str) -> Blob:
        raise NotFound(_SECRET)

    monkeypatch.setattr(store, "read", read)
    with pytest.raises(ChainError) as caught:
        read_chain(store, Head(epoch=1, seq=1, segment="journal/e1-1.jsonl"))
    assert_secret_absent(caught.value, _SECRET, sanitized=True)


def test_read_committed_without_owner(store: LocalStore, reads: list[str]) -> None:
    assert read_committed(store) is None
    assert reads == [OWNER_KEY]


def test_read_committed_without_head(store: LocalStore, reads: list[str]) -> None:
    doc = _owner(store, None)
    _put(store, None, [1])
    assert read_committed(store) == (doc, [])
    assert reads == [OWNER_KEY]


def test_read_committed_follows_only_one_authority_snapshot(
    store: LocalStore, reads: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(store, None, [1, 2])
    _put(store, "journal/e1-2.jsonl", [3], epoch=2)
    doc = _owner(store, Head(epoch=1, seq=2, segment="journal/e1-2.jsonl"), epoch=2)
    original = store.read

    def read(key: str) -> Blob:
        blob = original(key)
        if key == OWNER_KEY:
            newer = doc.model_copy(
                update={"head": Head(epoch=2, seq=3, segment="journal/e2-3.jsonl"), "version": 2}
            )
            store.replace(
                OWNER_KEY, canonical_json(newer.model_dump(mode="json")), if_match=blob.etag
            )
        return blob

    monkeypatch.setattr(store, "read", read)
    assert read_committed(store) == (doc, [_event(1), _event(2)])
    assert reads == [OWNER_KEY, "journal/e1-2.jsonl"]


def test_read_committed_flattens_all_segments_oldest_first(
    store: LocalStore, reads: list[str]
) -> None:
    _put(store, None, [1, 2])
    _put(store, "journal/e1-2.jsonl", [3], epoch=2)
    _put(store, "journal/e1-2.jsonl", [3, 4, 5])
    doc = _owner(store, Head(epoch=2, seq=3, segment="journal/e2-3.jsonl"), epoch=2)
    assert read_committed(store) == (doc, [_event(1), _event(2), _event(3, epoch=2)])
    assert reads == [OWNER_KEY, "journal/e2-3.jsonl", "journal/e1-2.jsonl"]


@pytest.mark.parametrize(
    "data",
    [
        b"private-reader-sentinel",
        b'{"private-reader-sentinel":"secret"}',
        b'{"schema_version":999,"private-reader-sentinel":"secret"}',
        b'{"head":{"segment":"private-reader-sentinel\xff"}}',
    ],
)
def test_invalid_owner_becomes_safe_chain_error(store: LocalStore, data: bytes) -> None:
    store.create(OWNER_KEY, data)
    with pytest.raises(ChainError, match="owner") as caught:
        read_committed(store)
    assert_secret_absent(caught.value, _SECRET, sanitized=True)


def test_invalid_owner_head_is_not_read(store: LocalStore, reads: list[str]) -> None:
    _owner(store, Head(epoch=1, seq=1, segment=f"../{_SECRET}"))
    with pytest.raises(ChainError, match="head") as caught:
        read_committed(store)
    assert_secret_absent(caught.value, _SECRET, sanitized=True)
    assert reads == [OWNER_KEY]


@pytest.mark.parametrize("first_seq", [1, 4])
def test_unreachable_boundary_validates_null_link_root(store: LocalStore, first_seq: int) -> None:
    _put(store, None, [first_seq])
    head = Head(epoch=1, seq=first_seq, segment=f"journal/e1-{first_seq}.jsonl")
    if first_seq == 4:
        with pytest.raises(ChainError, match="first segment must start at seq 1"):
            read_chain(store, head, stop_after="journal/e2-1.jsonl")
    else:
        chain = read_chain(store, head, stop_after="journal/e2-1.jsonl")
        assert [event.seq for _, batch in chain for event in batch] == [1]
