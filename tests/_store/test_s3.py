"""Exercise conditional storage against a loopback moto server."""

import re
import time
import traceback
from datetime import UTC, datetime
from functools import partial
from typing import Any

import pytest
import s3fs
from _fakes.secrecy import assert_secret_absent
from botocore.exceptions import ClientError, EndpointConnectionError

from swarm_arena._errors import (
    AlreadyExists,
    ConditionalWritesUnsupported,
    NotFound,
    PreconditionFailed,
    StorageError,
    WriteOutcomeUnknown,
    _rebuild_core_signal,
)
from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.lease import Head, Lease
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.segment import encode_segment, segment_name
from swarm_arena._store.base import Blob, open_store
from swarm_arena._store.s3 import S3Store, _S3Error

pytestmark = pytest.mark.s3
SECRET = "s3-private-sentinel-8751"


def test_create_then_read(s3_url: str) -> None:
    store = open_store(s3_url)
    etag = store.create("a/b.json", b"x")
    assert etag.startswith('"')
    assert etag.endswith('"')
    assert store.read("a/b.json") == Blob(b"x", etag)


def test_create_existing_raises_already_exists(s3_url: str) -> None:
    store = open_store(s3_url)
    store.create("k", b"1")
    with pytest.raises(AlreadyExists):
        store.create("k", b"2")
    assert store.read("k").data == b"1"


def test_replace_with_stale_etag_raises_precondition_failed(s3_url: str) -> None:
    store = open_store(s3_url)
    old = store.create("k", b"1")
    new = store.replace("k", b"2", if_match=old)
    assert store.read("k") == Blob(b"2", new)
    with pytest.raises(PreconditionFailed):
        store.replace("k", b"3", if_match=old)
    assert store.read("k").data == b"2"


def test_replace_missing_raises_not_found(s3_url: str) -> None:
    with pytest.raises(NotFound):
        open_store(s3_url).replace("nope", b"", if_match='"x"')


def test_read_missing_raises_not_found(s3_url: str) -> None:
    store = open_store(s3_url)
    with pytest.raises(NotFound):
        store.read("nope")
    assert not store.exists("nope")
    store.create("k", b"x")
    assert store.exists("k")


def test_list_sees_new_key_immediately(s3_url: str) -> None:
    store = open_store(s3_url)
    store.create("ns/runs/r1.json", b"{}")
    assert store.list("ns/runs/") == ["ns/runs/r1.json"]
    store.create("ns/runs/r2.json", b"{}")
    assert store.list("ns/runs/") == ["ns/runs/r1.json", "ns/runs/r2.json"]


def test_child_prefixes_keys(s3_url: str) -> None:
    parent = open_store(s3_url)
    parent.child("run-1").create("x", b"1")
    assert parent.read("run-1/x").data == b"1"


def test_probe_passes(s3_url: str) -> None:
    store = open_store(s3_url)
    assert store.probe_conditional_writes() is None
    keys = store.list("")
    assert len(keys) == 1
    assert re.fullmatch(r"\.swarm-arena-probe/\d{8}T\d{6}\.\d{3}Z-[0-9a-f]{8}", keys[0])
    assert store.read(keys[0]).data == b"4"


def test_lease_and_chain_on_s3(s3_url: str) -> None:
    store = open_store(s3_url).child("run-1")
    lease = Lease.acquire_new(store, owner_id="o", host="h", ttl_s=60.0)
    previous = None
    names = []
    for seq in (1, 2):
        name = segment_name(1, seq)
        event = Event(
            epoch=1,
            seq=seq,
            ts=datetime.now(UTC),
            run_id="r",
            context_id="r",
            type=EventType.TOOL_END,
        )
        store.create(name, encode_segment(previous, [event]))
        lease.advance(Head(epoch=1, seq=seq, segment=name))
        names.append(name)
        previous = name
    assert lease.doc.head is not None
    assert [name for name, _ in read_chain(store, lease.doc.head)] == names


@pytest.mark.parametrize("operation", ["create", "replace"])
def test_conflict_409_retries_then_unknown(
    s3_url: str, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = S3Store(s3_url)
    calls: list[dict[str, str]] = []
    delays: list[float] = []

    def conflict(key: str, data: bytes, **condition: str) -> str:
        calls.append(condition)
        raise _S3Error(409, "ConditionalRequestConflict")

    monkeypatch.setattr(store, "_put_object", conflict)
    monkeypatch.setattr(time, "sleep", delays.append)
    kwargs = {} if operation == "create" else {"if_match": '"old"'}
    with pytest.raises(WriteOutcomeUnknown):
        getattr(store, operation)("k", b"1", **kwargs)
    assert len(calls) == 4
    assert all(call == calls[0] for call in calls)
    assert calls[0] == ({"IfNoneMatch": "*"} if operation == "create" else {"IfMatch": '"old"'})
    assert delays == [0.1, 0.2, 0.4]


def test_old_botocore_sends_no_put(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    store = S3Store(s3_url)
    sent: list[str] = []

    def put(key: str, data: bytes, **condition: str) -> str:
        sent.append(key)
        return "etag"

    monkeypatch.setattr(store, "_conditional_params", lambda: frozenset({"Bucket", "Key", "Body"}))
    monkeypatch.setattr(store, "_put_object", put)
    with pytest.raises(ConditionalWritesUnsupported, match="IfNoneMatch"):
        store.create("k", b"1")
    with pytest.raises(ConditionalWritesUnsupported, match="IfMatch"):
        store.replace("k", b"2", if_match='"x"')
    assert sent == []


def test_locked_botocore_has_conditional_params(s3_url: str) -> None:
    store = open_store(s3_url)
    assert isinstance(store, S3Store)
    assert {"IfNoneMatch", "IfMatch"} <= store._conditional_params()


def test_probe_detects_endpoint_that_ignores_conditions(
    s3_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = S3Store(s3_url)
    real = store._put_object
    monkeypatch.setattr(store, "_put_object", lambda key, data, **condition: real(key, data))
    assert {"IfNoneMatch", "IfMatch"} <= store._conditional_params()
    with pytest.raises(ConditionalWritesUnsupported, match="create"):
        store.probe_conditional_writes()


@pytest.mark.parametrize("operation", ["read", "exists", "list"])
def test_read_path_errors_become_storage_error(
    s3_url: str, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = S3Store(s3_url)

    def fail(method: str, **kwargs: Any) -> dict[str, Any]:
        raise EndpointConnectionError(endpoint_url=SECRET)

    monkeypatch.setattr(store, "_call", fail)
    with pytest.raises(StorageError) as caught:
        getattr(store, operation)("k")
    assert_secret_absent(caught.value, SECRET)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert _rebuild_core_signal(caught.value) is not None


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (412, "PreconditionFailed", AlreadyExists),
        (403, "AccessDenied", StorageError),
        (500, "InternalError", StorageError),
    ],
)
def test_write_maps_response_without_leaking_values(
    s3_url: str,
    monkeypatch: pytest.MonkeyPatch,
    wrapped: bool,
    status: int,
    code: str,
    expected: type[StorageError],
) -> None:
    store = S3Store(s3_url)
    store._conditional_params()
    fs = store._filesystem()

    def fail(method: str, **kwargs: Any) -> dict[str, Any]:
        error = ClientError(
            {
                "Error": {"Code": code, "Message": SECRET},
                "ResponseMetadata": {"HTTPStatusCode": status},
            },
            "PutObject",
        )
        if wrapped:
            raise OSError(SECRET) from error
        raise error

    monkeypatch.setattr(fs, "call_s3", fail)
    with pytest.raises(expected) as caught:
        store.create("k", SECRET.encode())
    assert_secret_absent(caught.value, SECRET)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert _rebuild_core_signal(caught.value) is not None


def test_transport_write_outcome_unknown(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    store = S3Store(s3_url)
    store._conditional_params()

    def fail(method: str, **kwargs: Any) -> dict[str, Any]:
        raise EndpointConnectionError(endpoint_url=SECRET)

    monkeypatch.setattr(store._filesystem(), "call_s3", fail)
    with pytest.raises(WriteOutcomeUnknown) as caught:
        store.create("k", b"1")
    assert_secret_absent(caught.value, SECRET)


def test_list_pagination_and_root_isolation(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    store = S3Store(s3_url)
    store.create("z", b"1")
    store.create("a/b", b"2")
    store.child("other").create("x", b"3")
    real = store._call
    pages: list[str] = []

    def call(method: str, **kwargs: Any) -> dict[str, Any]:
        if method == "list_objects_v2":
            kwargs["MaxKeys"] = 1
            pages.append(kwargs.get("ContinuationToken", ""))
        return real(method, **kwargs)

    monkeypatch.setattr(store, "_call", call)
    assert store.list("") == ["a/b", "other/x", "z"]
    assert len(pages) == 3
    assert all(pages[1:])
    assert store.child("a").list("") == ["b"]
    assert store.list("a") == ["a/b"]


@pytest.mark.parametrize("key", ["", "/x", "../x", "a/./b", "a/../b", "a\x00b"])
def test_bad_keys_do_no_io(s3_url: str, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    store = S3Store(s3_url)

    def no_io() -> Any:
        pytest.fail("invalid key reached client")

    monkeypatch.setattr(store, "_filesystem", no_io)
    for operation in (store.read, store.exists, store.child):
        with pytest.raises(ValueError, match="bad store key"):
            operation(key)
    with pytest.raises(ValueError, match="bad store key"):
        store.create(key, b"1")
    with pytest.raises(ValueError, match="bad store key"):
        store.replace(key, b"1", if_match="x")
    if key:
        with pytest.raises(ValueError, match="bad store key"):
            store.list(key)


def test_constructor_and_child_are_lazy(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**kwargs: Any) -> Any:
        pytest.fail("construction must not build a filesystem")

    monkeypatch.setattr(s3fs, "S3FileSystem", fail)
    store = S3Store(s3_url, client_kwargs={"region_name": "us-east-1"})
    child = store.child("child")
    assert child.url == s3_url + "/child"
    assert child._options == store._options


def test_filesystem_disables_caches_and_model_is_cached(s3_url: str) -> None:
    store = S3Store(s3_url, use_listings_cache=True, skip_instance_cache=False)
    first = store._conditional_params()
    assert store._conditional_params() is first
    assert store._filesystem() is not S3Store(s3_url)._filesystem()
    assert not store._filesystem().dircache.use_listings_cache


@pytest.mark.parametrize("step", ["replace stale", "replace current", "read"])
def test_probe_rejects_each_broken_step(
    s3_url: str, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    store = S3Store(s3_url)
    real_replace = store.replace

    def replace(key: str, data: bytes, *, if_match: str) -> str:
        if step == "replace stale" and if_match == "probe-mismatch":
            return "ignored-condition"
        if step == "replace current" and if_match != "probe-mismatch":
            raise OSError(SECRET)
        return real_replace(key, data, if_match=if_match)

    monkeypatch.setattr(store, "replace", replace)
    if step == "read":
        monkeypatch.setattr(store, "read", lambda key: Blob(b"wrong", "etag"))
    with pytest.raises(ConditionalWritesUnsupported, match=step) as caught:
        store.probe_conditional_writes()
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize(
    "kind",
    [
        StorageError,
        NotFound,
        AlreadyExists,
        PreconditionFailed,
        WriteOutcomeUnknown,
        ConditionalWritesUnsupported,
    ],
)
def test_registered_storage_errors_rebuild_without_mutations(
    s3_url: str,
    monkeypatch: pytest.MonkeyPatch,
    kind: type[StorageError],
) -> None:
    store = S3Store(s3_url)
    store._conditional_params()
    if kind is ConditionalWritesUnsupported:
        monkeypatch.setattr(store, "_conditional_params", frozenset)
    status = {
        NotFound: 404,
        AlreadyExists: 412,
        PreconditionFailed: 412,
        WriteOutcomeUnknown: 409,
    }.get(kind, 403)

    def fail(key: str, data: bytes, **condition: str) -> str:
        raise _S3Error(status, "failure")

    monkeypatch.setattr(store, "_put_object", fail)
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    operation = (
        partial(store.replace, if_match='"old"') if kind is PreconditionFailed else store.create
    )
    with pytest.raises(kind) as caught:
        operation("k", b"data")
    error = caught.value
    safe_args = error.args
    error.args = (SECRET,)
    error.add_note(SECRET)
    error.__cause__ = RuntimeError(SECRET)
    rebuilt = _rebuild_core_signal(error)
    assert type(rebuilt) is kind
    assert rebuilt.args == safe_args
    assert_secret_absent(rebuilt, SECRET)
    assert SECRET not in "".join(traceback.format_exception(rebuilt))
    assert _rebuild_core_signal(kind(SECRET)) is None

    class Impostor(StorageError):
        def __hash__(self) -> int:
            return hash(error)

        def __eq__(self, other: object) -> bool:
            return True

    assert _rebuild_core_signal(Impostor(SECRET)) is None


def test_read_body_failure_closes_stream(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    store = S3Store(s3_url)
    store._conditional_params()
    closed: list[bool] = []

    class Body:
        async def __aenter__(self) -> "Body":
            return self

        async def __aexit__(self, *args: object) -> None:
            closed.append(True)

        async def read(self) -> bytes:
            raise OSError(SECRET)

    monkeypatch.setattr(store, "_call", lambda method, **kwargs: {"Body": Body(), "ETag": "e"})
    with pytest.raises(StorageError) as caught:
        store.read("k")
    assert closed == [True]
    assert_secret_absent(caught.value, SECRET)


def test_conflict_retry_can_succeed(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    store = S3Store(s3_url)
    real = store._put_object
    attempts: list[str] = []

    def put(key: str, data: bytes, **condition: str) -> str:
        attempts.append(key)
        if len(attempts) == 1:
            raise _S3Error(409, "ConditionalRequestConflict")
        return real(key, data, **condition)

    monkeypatch.setattr(store, "_put_object", put)
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    etag = store.create("k", b"1")
    assert len(attempts) == 2
    assert store.read("k") == Blob(b"1", etag)


@pytest.mark.parametrize(
    "root",
    [
        "s3://",
        "s3://user:password@bucket/p",
        "s3://bucket/p?private=value",
        "s3://bucket/p#private",
        "s3://bucket/../p",
    ],
)
def test_invalid_root_is_sanitized(root: str) -> None:
    with pytest.raises(ValueError, match="invalid S3 store root") as caught:
        S3Store(root)
    assert root not in str(caught.value)
    assert caught.value.__cause__ is None


def test_client_initialization_failure_is_sanitized(
    s3_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = S3Store(s3_url, endpoint_url=SECRET)

    def fail(**kwargs: Any) -> Any:
        raise OSError(SECRET)

    monkeypatch.setattr(s3fs, "S3FileSystem", fail)
    with pytest.raises(StorageError) as caught:
        store.create("k", b"1")
    assert_secret_absent(caught.value, SECRET)
    assert _rebuild_core_signal(caught.value) is not None
