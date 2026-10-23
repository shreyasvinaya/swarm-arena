from pathlib import Path

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import NotFound, StorageError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._store.artifacts import ArtifactStore

SECRET = "9537d6d3032b2555b86a3df9fb9abdda"


def test_artifacts_put_get_fetch(artifacts: ArtifactStore, tmp_path: Path) -> None:
    h = artifacts.put_bytes(b"hello")
    assert h == sha256_hex(b"hello") == artifacts.put_bytes(b"hello")
    assert artifacts.get_bytes(h) == b"hello"
    (tmp_path / "a.txt").write_bytes(b"hi")
    ref = artifacts.put_file(tmp_path / "a.txt")
    assert (ref.name, ref.sha256, ref.size) == ("a.txt", sha256_hex(b"hi"), 2)
    assert ref.path == "a.txt"
    artifacts.fetch(h, tmp_path / "out" / "x.txt")
    assert (tmp_path / "out" / "x.txt").read_bytes() == b"hello"
    assert (tmp_path / "out" / "x.txt").stat().st_mode & 0o777 == 0o444
    artifacts.fetch(h, tmp_path / "w.txt", read_only=False)
    assert (tmp_path / "w.txt").stat().st_mode & 0o777 == 0o644
    with pytest.raises(NotFound):
        artifacts.get_bytes("0" * 64)


def test_get_bytes_verifies_hash(artifacts: ArtifactStore, artifact_root: str) -> None:
    h = artifacts.put_bytes(b"hello")
    stored = Path(artifact_root) / h
    stored.chmod(0o644)
    stored.write_bytes(SECRET.encode())
    with pytest.raises(StorageError, match="hash") as caught:
        artifacts.get_bytes(h)
    assert_secret_absent(caught.value, SECRET)


def test_fetch_checks_hash_before_writing(
    artifacts: ArtifactStore, artifact_root: str, tmp_path: Path
) -> None:
    h = artifacts.put_bytes(b"hello")
    (Path(artifact_root) / h).write_bytes(b"corrupt")
    dest = tmp_path / "out" / "result"
    with pytest.raises(StorageError, match="hash"):
        artifacts.fetch(h, dest)
    assert not dest.exists()


def test_hash_error_omits_requested_value(artifacts: ArtifactStore, artifact_root: str) -> None:
    root = Path(artifact_root)
    root.mkdir()
    (root / SECRET).write_bytes(b"payload")
    with pytest.raises(StorageError, match="hash") as caught:
        artifacts.get_bytes(SECRET)
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("name", ["swarm", "parents", "candidate"])
def test_put_file_preserves_fileref_path_rules(
    artifacts: ArtifactStore, tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    path.write_bytes(SECRET.encode())
    with pytest.raises(ValueError, match="path") as caught:
        artifacts.put_file(path)
    assert_secret_absent(caught.value, SECRET)


def test_put_file_error_is_sanitized(
    artifacts: ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_read(path: Path) -> bytes:
        raise OSError(SECRET)

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    with pytest.raises(StorageError) as caught:
        artifacts.put_file(tmp_path / "x")
    assert_secret_absent(caught.value, SECRET)


def test_fetch_error_is_sanitized(
    artifacts: ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = artifacts.put_bytes(b"hello")

    def fail_write(root: int, rel: str, data: bytes, *, read_only: bool) -> None:
        raise OSError(SECRET)

    monkeypatch.setattr("swarm_arena._tooling.materialize._write", fail_write)
    with pytest.raises(StorageError) as caught:
        artifacts.fetch(h, tmp_path / "output")
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("kind", ["symlink", "parent_symlink", "fifo"])
def test_put_file_refuses_unsafe_sources(
    artifacts: ArtifactStore, tmp_path: Path, kind: str
) -> None:
    import os

    source = tmp_path / "source.txt"
    source.write_bytes(b"source")
    path = tmp_path / "input.txt"
    if kind == "symlink":
        path.symlink_to(source)
    elif kind == "parent_symlink":
        (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        path = tmp_path / "alias/source.txt"
    else:
        os.mkfifo(path)
    with pytest.raises(StorageError):
        artifacts.put_file(path)


def test_fetch_refuses_symlink_destination(artifacts: ArtifactStore, tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_bytes(b"untouched")
    dest = tmp_path / "out.txt"
    dest.symlink_to(target)
    with pytest.raises(StorageError):
        artifacts.fetch(artifacts.put_bytes(b"overwrite"), dest)
    assert target.read_bytes() == b"untouched"


def test_put_file_refuses_oversized_source_unread(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    path = tmp_path / "big.dat"
    path.touch()
    os.truncate(path, 2**30 + 1)

    def fail(fd: int, size: int) -> bytes:
        pytest.fail("oversized source was read")

    monkeypatch.setattr(os, "read", fail)
    with pytest.raises(StorageError, match="source"):
        artifacts.put_file(path)


def test_fetch_refuses_symlinked_parent(artifacts: ArtifactStore, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StorageError, match="destination"):
        artifacts.fetch(artifacts.put_bytes(b"bytes"), tmp_path / "alias/out")
    assert not (outside / "out").exists()


def test_put_file_fifo_finishes_in_subprocess(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    program = """from pathlib import Path
import sys
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._errors import StorageError
try:
    ArtifactStore(open_store(sys.argv[2])).put_file(Path(sys.argv[1]))
except StorageError:
    pass
else:
    raise AssertionError("FIFO was accepted")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(fifo), str(tmp_path / "artifacts")],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.s3
async def test_nonlocal_put_file_and_fetch(s3_url: str, tmp_path: Path) -> None:
    from swarm_arena._store.base import open_store

    remote = ArtifactStore(open_store(s3_url))
    source = tmp_path / "source.txt"
    source.write_bytes(b"remote bytes")
    ref = remote.put_file(source)
    assert (ref.sha256, ref.size) == (sha256_hex(b"remote bytes"), 12)
    remote.fetch(ref.sha256, tmp_path / "fetched.txt")
    assert (tmp_path / "fetched.txt").read_bytes() == b"remote bytes"
    assert (tmp_path / "fetched.txt").stat().st_mode & 0o777 == 0o444


@pytest.mark.s3
async def test_nonlocal_chunks_bound_before_publication(
    s3_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena._store.base import open_store

    backend = open_store(s3_url)
    remote = ArtifactStore(backend)
    created: list[bytes] = []

    def create(key: str, data: bytes) -> str:
        created.append(data)
        return key

    monkeypatch.setattr(backend, "create", create)
    with pytest.raises(StorageError, match="size cap"):
        remote._put_chunks(iter([b"ab", b"cd"]), max_bytes=3)
    assert created == []
    assert remote._put_chunks(iter([b"ab", b"c"]), max_bytes=3) == (sha256_hex(b"abc"), 3)
    assert created == [b"abc"]


def test_local_chunks_keep_streaming_path(
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(data: bytes) -> str:
        pytest.fail("local streaming used the byte accumulation path")

    monkeypatch.setattr(artifacts, "put_bytes", fail)
    assert artifacts._put_chunks(iter([b"ab", b"c"]), max_bytes=3) == (sha256_hex(b"abc"), 3)
    assert artifacts.get_bytes(sha256_hex(b"abc")) == b"abc"


@pytest.mark.s3
async def test_nonlocal_failed_stream_does_not_publish(
    s3_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Iterator

    from swarm_arena._store.base import open_store

    remote = ArtifactStore(open_store(s3_url))

    def chunks() -> Iterator[bytes]:
        yield b"part"
        raise ValueError("source changed")

    def publish(data: bytes) -> str:
        pytest.fail("partial contents were published")

    monkeypatch.setattr(remote, "put_bytes", publish)
    with pytest.raises(ValueError, match="source changed"):
        remote._put_chunks(chunks(), max_bytes=8)


@pytest.mark.s3
async def test_nonlocal_fetch_checks_digest_cap_and_destination(
    s3_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena._store.base import open_store

    backend = open_store(s3_url)
    remote = ArtifactStore(backend)
    digest = remote.put_bytes(b"hello")
    monkeypatch.setattr("swarm_arena._store.artifacts._MAX_ARTIFACT_BYTES", 4)
    with pytest.raises(StorageError, match="size cap"):
        remote.fetch(digest, tmp_path / "oversized")
    assert not (tmp_path / "oversized").exists()
    monkeypatch.setattr("swarm_arena._store.artifacts._MAX_ARTIFACT_BYTES", 100)
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(StorageError, match="destination"):
        remote.fetch(digest, tmp_path / "link")
    assert outside.read_bytes() == b"untouched"
    etag = backend.read(digest).etag
    backend.replace(digest, SECRET.encode(), if_match=etag)
    with pytest.raises(StorageError, match="hash") as caught:
        remote.fetch(digest, tmp_path / "corrupt")
    assert_secret_absent(caught.value, SECRET)
    assert not (tmp_path / "corrupt").exists()
