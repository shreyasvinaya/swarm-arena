import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import AlreadyExists, NotFound, StorageError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._store.base import Blob
from swarm_arena._store.local import LocalStore

SECRET = "9537d6d3032b2555b86a3df9fb9abdda"


def test_create_then_read(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    assert s.create("a/b.json", b"x") == sha256_hex(b"x")
    assert s.read("a/b.json") == Blob(b"x", sha256_hex(b"x"))


def test_create_is_write_once(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    s.create("k", b"1")
    with pytest.raises(AlreadyExists):
        s.create("k", b"2")
    assert s.read("k").data == b"1"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["k"]


def test_create_race_has_one_winner(tmp_path: Path) -> None:
    barrier = Barrier(8)

    def create(i: int) -> tuple[bytes, str] | None:
        data = f"{i}".encode()
        barrier.wait()
        try:
            return data, LocalStore(tmp_path).create("k", data)
        except AlreadyExists:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(create, range(8)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert results.count(None) == 7
    data, etag = winners[0]
    assert etag == sha256_hex(data)
    assert LocalStore(tmp_path).read("k") == Blob(data, etag)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["k"]


def test_failed_create_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_link(src: Path, dst: Path) -> None:
        raise OSError(SECRET)

    monkeypatch.setattr(os, "link", fail_link)
    s = LocalStore(tmp_path)
    with pytest.raises(OSError, match="create") as caught:
        s.create("k", b"x")
    assert_secret_absent(caught.value, SECRET)
    assert not s.exists("k")
    assert list(tmp_path.iterdir()) == []


def test_fsync_failure_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fsync(fd: int) -> None:
        raise OSError(SECRET)

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="create"):
        LocalStore(tmp_path).create("k", b"x")
    assert list(tmp_path.iterdir()) == []


def test_create_publishes_only_complete_synced_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"complete contents" * 1024
    real_fsync, real_link = os.fsync, os.link
    synced = False

    def record_fsync(fd: int) -> None:
        nonlocal synced
        real_fsync(fd)
        synced = True

    def check_link(src: Path, dst: Path) -> None:
        assert synced
        assert src.read_bytes() == data
        assert not dst.exists()
        real_link(src, dst)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "link", check_link)
    s = LocalStore(tmp_path)
    s.create("k", data)
    assert s.read("k").data == data


def test_temp_collision_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fixed_token(nbytes: int) -> str:
        assert nbytes == 6
        return "0" * 12

    monkeypatch.setattr("swarm_arena._store.local.token_hex", fixed_token)
    temp = tmp_path / ("k.tmp-" + "0" * 12)
    temp.write_bytes(b"another writer")
    with pytest.raises(OSError, match="create"):
        LocalStore(tmp_path).create("k", b"x")
    assert temp.read_bytes() == b"another writer"
    assert not (tmp_path / "k").exists()


def test_read_exists_list_child(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    for k in ("runs/a.json", "runs/b.json", "reviews/x/1.jsonl"):
        s.create(k, b"{}")
    assert s.list("runs/") == ["runs/a.json", "runs/b.json"]
    assert s.exists("runs/a.json")
    assert not s.exists("runs/c")
    s.child("runs").create("c.json", b"1")
    assert s.read("runs/c.json").data == b"1"
    with pytest.raises(NotFound):
        s.read("missing")
    with pytest.raises(NotImplementedError, match="M1b"):
        s.replace("runs/a.json", b"2", if_match="x")


@pytest.mark.parametrize("bad", ["", "/abs", "../x", "a/./b", "a/../b", "a\x00b"])
def test_bad_keys(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="key"):
        LocalStore(tmp_path).create(bad, b"x")


def test_key_errors_are_sanitized(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    key = "../" + SECRET
    for method in (s.read, s.exists, s.child):
        with pytest.raises(ValueError, match="key") as caught:
            method(key)
        assert_secret_absent(caught.value, SECRET)
    with pytest.raises(ValueError, match="key") as caught:
        s.create("../" + SECRET, b"x")
    assert_secret_absent(caught.value, SECRET)


def test_store_errors_are_sanitized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = LocalStore(tmp_path)
    key = SECRET
    with pytest.raises(NotFound) as missing:
        s.read(key)
    assert_secret_absent(missing.value, SECRET)
    s.create(key, b"private contents")
    with pytest.raises(AlreadyExists) as existing:
        s.create(key, b"other contents")
    assert_secret_absent(existing.value, SECRET)

    def fail_read(path: Path) -> bytes:
        raise OSError(SECRET)

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    with pytest.raises(StorageError) as unreadable:
        s.read("k")
    assert_secret_absent(unreadable.value, SECRET)


@pytest.mark.parametrize("operation", ["exists", "list"])
def test_query_errors_are_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    def fail_io(*args: object, **kwargs: object) -> None:
        raise OSError(SECRET)

    s = LocalStore(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists" if operation == "exists" else "rglob", fail_io)
        with pytest.raises(StorageError) as caught:
            getattr(s, operation)("k")
    assert_secret_absent(caught.value, SECRET)


def test_list_excludes_internal_and_nonregular_files(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    for key in ("runs/z", "runs/a", "runs/a.tmp-deadbeef", "runs/a.lock", "other"):
        s.create(key, b"")
    (tmp_path / "runs" / "empty").mkdir()
    os.mkfifo(tmp_path / "runs" / "fifo")
    assert s.list("runs/") == ["runs/a", "runs/z"]
    assert s.list("runs/a") == ["runs/a"]
    assert s.list("") == ["other", "runs/a", "runs/z"]


def test_init_does_no_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "new"

    def fail_io(*args: object, **kwargs: object) -> None:
        pytest.fail("constructor performed filesystem I/O")

    with monkeypatch.context() as patch:
        for method in ("mkdir", "resolve", "stat", "exists", "open"):
            patch.setattr(Path, method, fail_io)
        s = LocalStore(root)
        s.child("runs")
        s.probe_conditional_writes()
    assert not root.exists()
    assert s.list("") == []
    assert not s.exists("k")
