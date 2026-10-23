import importlib.metadata
import struct
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import AlreadyExists, NotFound, PreconditionFailed
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.local import LocalStore
from swarm_arena.executor import Job
from swarm_arena.tool import FileRef, ParentRef

try:
    importlib.metadata.version("ray")
except importlib.metadata.PackageNotFoundError:
    pytest.skip("Ray transport tests require the ray extra", allow_module_level=True)

# isort: split
from swarm_arena.ray._files import (
    MIB,
    FrameDiagnostic,
    MemoryStore,
    Unframed,
    check_cap,
    frame,
    inline_inputs,
    needed_hashes,
    unframe,
    uses_object_store,
)


def test_needed_hashes_cover_root_parents_and_candidate() -> None:
    hashes = [sha256_hex(bytes([i])) for i in range(4)]
    stage = Job.Stage(
        root={"a.txt": hashes[0]},
        parents=[
            ParentRef(
                candidate_id="c",
                run_id="r",
                tool_identity="t",
                overlay={"b.txt": hashes[1]},
                files=[FileRef(name="o", path="o", sha256=hashes[2], size=1)],
            )
        ],
        candidate=Job.CandidateFiles(request={}, output={}, files={"c.txt": hashes[3]}),
    )
    assert needed_hashes(stage) == set(hashes)
    assert needed_hashes(Job.Stage()) == set()


def test_inline_inputs_over_cap_raises(tmp_path: Path) -> None:
    artifacts = ArtifactStore(LocalStore(tmp_path))
    sha = artifacts.put_bytes(b"x" * (2 * MIB))
    with pytest.raises(ValueError, match="over the Ray object_store_cap_mib cap of 1 MiB"):
        inline_inputs(Job.Stage(root={"big.bin": sha}), artifacts, cap_bytes=MIB)


def test_inline_inputs_deduplicates(artifacts: ArtifactStore) -> None:
    sha = artifacts.put_bytes(b"ab")
    stage = Job.Stage(root={"a": sha, "b": sha})
    assert inline_inputs(stage, artifacts, cap_bytes=MIB) == {sha: b"ab"}


@pytest.mark.parametrize("what", ["inputs", "outputs"])
def test_check_cap_exact_boundary_and_message(what: Any) -> None:
    check_cap({"a": b"x" * MIB}, cap_bytes=MIB, what=what)
    with pytest.raises(ValueError, match="over the Ray") as caught:
        check_cap({"a": b"x" * MIB, "b": b"!"}, cap_bytes=MIB, what=what)
    assert str(caught.value) == (
        f"{what} of this tool run are {MIB + 1} bytes, over the Ray "
        "object_store_cap_mib cap of 1 MiB; use an s3:// artifact_root"
    )


def test_memory_store_create_is_conditional() -> None:
    store = MemoryStore({"seed": b"input"})
    assert store.created() == {}
    etag = store.create("k", b"1")
    assert etag == sha256_hex(b"1")
    assert store.read("k") == (b"1", etag)
    with pytest.raises(AlreadyExists):
        store.create("k", b"1")
    with pytest.raises(PreconditionFailed):
        store.replace("k", b"2", if_match="wrong")
    assert store.replace("k", b"2", if_match=etag) == sha256_hex(b"2")
    assert store.created() == {"k": b"2"}
    with pytest.raises(NotFound):
        store.read("absent")
    with pytest.raises(NotFound):
        store.replace("absent", b"2", if_match=etag)
    assert store.exists("k")
    assert not store.exists("absent")
    assert store.list("") == ["k", "seed"]


def test_memory_store_child_shares_backing_and_created_tracking() -> None:
    seed = {"nested/input": b"in"}
    store = MemoryStore(seed)
    seed.clear()
    child = store.child("nested")
    assert child.url == store.url + "/nested"
    assert child.read("input").data == b"in"
    child.create("out", b"out")
    assert store.created() == {"nested/out": b"out"}
    assert child.created() == {"out": b"out"}
    assert child.list("") == ["input", "out"]
    assert store.child("nest").list("") == []
    child.created().clear()
    assert child.exists("out")
    assert child.child("deeper").create("x", b"x") == sha256_hex(b"x")
    assert store.read("nested/deeper/x").data == b"x"


@pytest.mark.parametrize("key", ["", "/abs", "a/../b", "./a", "nul\0"])
def test_memory_store_rejects_bad_keys(key: str) -> None:
    store = MemoryStore()
    for operation in (store.read, store.exists, store.child):
        with pytest.raises(ValueError, match="bad store key"):
            operation(key)
    with pytest.raises(ValueError, match="bad store key"):
        store.create(key, b"x")
    with pytest.raises(ValueError, match="bad store key"):
        store.replace(key, b"x", if_match="x")
    with pytest.raises(ValueError, match="bad store key"):
        MemoryStore({key: b"x"})


@pytest.mark.parametrize("diagnostics", [(), (FrameDiagnostic("read_failed", 1),)])
def test_frame_roundtrip(diagnostics: tuple[FrameDiagnostic, ...]) -> None:
    blobs = {sha256_hex(b"a"): b"a", sha256_hex(b"bb"): b"bb"}
    assert unframe(frame(b'{"k": 1}', blobs, diagnostics)) == Unframed(
        b'{"k": 1}', blobs, diagnostics
    )
    assert frame(b"h", blobs) == frame(b"h", dict(reversed(list(blobs.items()))))
    assert unframe(frame(b"", {})) == Unframed(b"", {}, ())


def test_frame_slot_layout() -> None:
    digest = sha256_hex(b"x")
    diagnostics = b'[["read_failed",2]]'
    assert frame(b"h", {digest: b"x"}, [FrameDiagnostic("read_failed", 2)]) == (
        b"SAF1"
        + struct.pack(">Q", 1)
        + b"h"
        + digest.encode("ascii")
        + struct.pack(">Q", 1)
        + b"x"
        + struct.pack(">Q", len(diagnostics))
        + diagnostics
    )


def test_unframe_rejects_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="sha256"):
        unframe(frame(b"{}", {"0" * 64: b"not that"}))


def test_unframe_rejects_duplicate_blob() -> None:
    encoded = frame(b"", {sha256_hex(b"a"): b"a"})
    record = encoded[12:-10]
    with pytest.raises(ValueError, match="invalid frame"):
        unframe(encoded[:12] + record + record + encoded[-10:])


@pytest.mark.parametrize(
    "data",
    [b"", b"BAD!", b"SAF1", b"SAF1" + b"\xff" * 8, b"SAF1" + struct.pack(">Q", 0) + b"\xff" * 8],
)
def test_unframe_rejects_bad_magic_or_length(data: bytes) -> None:
    with pytest.raises(ValueError, match="invalid frame"):
        unframe(data)


def test_unframe_rejects_every_truncation_and_trailing_bytes() -> None:
    data = frame(b"head", {sha256_hex(b"blob"): b"blob"}, [FrameDiagnostic("read_failed", 1)])
    for offset in range(len(data)):
        with pytest.raises(ValueError, match="invalid frame"):
            unframe(data[:offset])
    with pytest.raises(ValueError, match="invalid frame"):
        unframe(data + b"trailing")


@pytest.mark.parametrize(
    "slot",
    [
        b"bad-secret-sentinel",
        b"{}",
        b"[1]",
        b'[["bad-code",1]]',
        b'[["Code",1]]',
        b'[["ok",true]]',
        b'[["ok",0]]',
        b'[["ok",10001]]',
        b'[["ok",1.0]]',
        b'[["ok",1,"text"]]',
        b'[["ok",1]]'[:-1],
        b'[["ok",1]]' + b"\xff",
        b'[["ok",1]]'[:-1] + b',["ok",1]]' * 33,
    ],
)
def test_unframe_rejects_malformed_diagnostics(slot: bytes) -> None:
    with pytest.raises(ValueError, match="invalid frame") as caught:
        unframe(b"SAF1" + struct.pack(">Q", 0) + struct.pack(">Q", len(slot)) + slot)
    assert_secret_absent(caught.value, "bad-secret-sentinel", sanitized=True)


@pytest.mark.parametrize(
    "diagnostic",
    [
        FrameDiagnostic("", 1),
        FrameDiagnostic("A", 1),
        FrameDiagnostic("a" * 65, 1),
        FrameDiagnostic("ok", True),
        FrameDiagnostic("ok", 0),
        FrameDiagnostic("ok", 10001),
    ],
)
def test_frame_rejects_invalid_diagnostic(diagnostic: FrameDiagnostic) -> None:
    with pytest.raises(ValueError, match="invalid frame"):
        frame(b"", {}, [diagnostic])


def test_frame_diagnostic_count_limit() -> None:
    diagnostics = [FrameDiagnostic("x", 1)] * 32
    assert len(unframe(frame(b"", {}, diagnostics)).diagnostics) == 32
    with pytest.raises(ValueError, match="invalid frame"):
        frame(b"", {}, [*diagnostics, diagnostics[0]])
    slot = b'[["x",1]' + b',["x",1]' * 32 + b"]"
    with pytest.raises(ValueError, match="invalid frame"):
        unframe(b"SAF1" + struct.pack(">Q", 0) + struct.pack(">Q", len(slot)) + slot)


def test_uses_object_store() -> None:
    assert not uses_object_store("s3://bucket/a")
    assert uses_object_store("/tmp/artifacts")


@pytest.mark.ray
def test_frame_roundtrips_through_ray_object_store(ray_driver: None) -> None:
    import ray

    blobs = {sha256_hex(b"artifact"): b"artifact"}
    data = frame(b"header", blobs, [FrameDiagnostic("read_failed", 1)])
    assert unframe(ray.get(ray.put(data), timeout=15)) == Unframed(
        b"header", blobs, (FrameDiagnostic("read_failed", 1),)
    )
    assert ray.cluster_resources()["CPU"] == 2


@pytest.mark.parametrize("count", [False, 1.5, "1", None, -1])
def test_frame_rejects_non_integer_or_negative_counts(count: Any) -> None:
    with pytest.raises(ValueError, match="invalid frame diagnostics"):
        frame(b"", {}, [FrameDiagnostic("read_failed", count)])


@pytest.mark.parametrize("code", ["a\n", "_a", "a-b", "secret-sentinel!", "é", 1])
def test_frame_rejects_bad_diagnostic_codes(code: Any) -> None:
    with pytest.raises(ValueError, match="invalid frame diagnostics") as caught:
        frame(b"", {}, [FrameDiagnostic(code, 1)])
    assert_secret_absent(caught.value, "secret-sentinel!", sanitized=True)


def test_unframe_requires_utf8_diagnostics() -> None:
    slot = '[["read_failed",1]]'.encode("utf-16")
    with pytest.raises(ValueError, match="invalid frame diagnostics JSON"):
        unframe(b"SAF1" + struct.pack(">Q", 0) + struct.pack(">Q", len(slot)) + slot)


@pytest.mark.parametrize(
    ("limit", "value", "head", "blobs"),
    [
        ("_MAX_HEAD_BYTES", 1, b"ab", {}),
        ("_MAX_BLOB_BYTES", 1, b"", {sha256_hex(b"ab"): b"ab"}),
        ("_MAX_BLOBS", 0, b"", {sha256_hex(b"a"): b"a"}),
        ("_MAX_DIAGNOSTIC_BYTES", 1, b"", {}),
    ],
)
def test_codec_checks_limits_before_copying(
    limit: str, value: int, head: bytes, blobs: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena.ray import _files

    encoded = frame(head, blobs)
    monkeypatch.setattr(_files, limit, value)
    with pytest.raises(ValueError, match="invalid frame"):
        frame(head, blobs)
    with pytest.raises(ValueError, match="invalid frame"):
        unframe(encoded)


def test_unframe_rejects_total_frame_over_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena.ray import _files

    data = frame(b"", {})
    monkeypatch.setattr(_files, "_MAX_FRAME_BYTES", len(data) - 1)
    with pytest.raises(ValueError, match="invalid frame length"):
        unframe(data)


def test_unframe_rejects_declared_blob_length_before_copying() -> None:
    data = b"SAF1" + struct.pack(">Q", 0) + sha256_hex(b"x").encode() + b"\xff" * 8
    with pytest.raises(ValueError, match="invalid frame length"):
        unframe(data)


def test_frame_rejects_invalid_digest_identifier() -> None:
    with pytest.raises(ValueError, match="invalid frame sha256") as caught:
        frame(b"", {"secret-sentinel": b"x"})
    assert_secret_absent(caught.value, "secret-sentinel", sanitized=True)


@pytest.mark.parametrize("operation", ["init", "create", "replace"])
@pytest.mark.parametrize("view", [False, True])
def test_memory_store_detaches_mutable_buffers(operation: str, view: bool) -> None:
    buffer = bytearray(b"aaaa")
    data: Any = memoryview(buffer) if view else buffer
    store = MemoryStore({"out": data} if operation == "init" else {})
    if operation == "create":
        store.create("out", data)
    elif operation == "replace":
        etag = store.create("out", b"old")
        store.replace("out", data, if_match=etag)
    buffer[:] = b"bbbb"
    blob = store.read("out")
    assert blob.data == b"aaaa"
    assert type(blob.data) is bytes
    assert blob.etag == sha256_hex(b"aaaa")
    if operation != "init":
        assert store.created() == {"out": b"aaaa"}
    store.replace("out", b"next", if_match=blob.etag)


def test_inline_inputs_refuses_aggregate_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._store.base import Blob

    class CountingStore(LocalStore):
        bytes_read = 0

        def read(self, key: str) -> Blob:
            blob = super().read(key)
            self.bytes_read += len(blob.data)
            return blob

    store = CountingStore(tmp_path)
    artifacts = ArtifactStore(store)
    first = artifacts.put_bytes(b"x" * (MIB // 2 + 1))
    second = artifacts.put_bytes(b"y" * (MIB // 2 + 1))
    from swarm_arena.ray import _files

    byte_reads: list[int] = []
    original_read = _files.os.read

    def counting_read(fd: int, size: int) -> bytes:
        data = original_read(fd, size)
        byte_reads.append(len(data))
        return data

    monkeypatch.setattr(_files.os, "read", counting_read)
    with pytest.raises(ValueError, match="over the Ray") as caught:
        inline_inputs(Job.Stage(root={"a": first, "b": second}), artifacts, cap_bytes=MIB)
    assert str(caught.value) == (
        f"inputs of this tool run are {MIB + 2} bytes, over the Ray "
        "object_store_cap_mib cap of 1 MiB; use an s3:// artifact_root"
    )
    assert store.bytes_read == 0
    assert byte_reads == []


@pytest.mark.parametrize("cap_bytes", [0, 1, MIB - 1, 1024 * MIB + 1, 2048 * MIB])
def test_transport_rejects_invalid_object_store_caps(cap_bytes: int) -> None:
    for operation in (
        lambda: inline_inputs(Job.Stage(), ArtifactStore(MemoryStore()), cap_bytes=cap_bytes),
        lambda: check_cap({}, cap_bytes=cap_bytes, what="outputs"),
    ):
        with pytest.raises(ValueError, match="object_store_cap_mib"):
            operation()


def test_transport_cap_endpoints_and_exported_maximum() -> None:
    from swarm_arena.ray._files import MAX_OBJECT_STORE_CAP_MIB

    assert MAX_OBJECT_STORE_CAP_MIB == 1024
    for cap_bytes in (MIB, MAX_OBJECT_STORE_CAP_MIB * MIB):
        check_cap({}, cap_bytes=cap_bytes, what="inputs")
        assert inline_inputs(Job.Stage(), ArtifactStore(MemoryStore()), cap_bytes=cap_bytes) == {}


def test_inline_inputs_rechecks_file_growth_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena.ray import _files

    artifacts = ArtifactStore(LocalStore(tmp_path))
    digest = artifacts.put_bytes(b"x")
    original = _files._input_size
    reads: list[int] = []

    def size_then_grow(store: Any, key: str) -> int:
        size = original(store, key)
        (tmp_path / key).write_bytes(b"x" * (MIB + 1))
        return size

    def read(fd: int, count: int) -> bytes:
        reads.append(count)
        raise AssertionError("over-cap file was read")

    monkeypatch.setattr(_files, "_input_size", size_then_grow)
    monkeypatch.setattr(_files.os, "read", read)
    with pytest.raises(ValueError, match="over the Ray object_store_cap_mib cap of 1 MiB"):
        inline_inputs(Job.Stage(root={"a": digest}), artifacts, cap_bytes=MIB)
    assert reads == []


def test_inline_inputs_verifies_local_hash(tmp_path: Path) -> None:
    from swarm_arena._errors import StorageError

    artifacts = ArtifactStore(LocalStore(tmp_path))
    digest = artifacts.put_bytes(b"x")
    (tmp_path / digest).write_bytes(b"changed")
    with pytest.raises(StorageError, match="artifact failed its hash check"):
        inline_inputs(Job.Stage(root={"a": digest}), artifacts, cap_bytes=MIB)


def test_inline_inputs_supports_immutable_memory_store() -> None:
    artifacts = ArtifactStore(MemoryStore())
    digest = artifacts.put_bytes(b"x")
    assert inline_inputs(Job.Stage(root={"a": digest}), artifacts, cap_bytes=MIB) == {digest: b"x"}
