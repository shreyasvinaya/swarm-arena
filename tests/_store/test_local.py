import fcntl
import os
from pathlib import Path
from threading import Barrier

import pytest
from _fakes.cleanup import BoundedThreadPoolExecutor
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import AlreadyExists, NotFound, PreconditionFailed, StorageError
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
        barrier.wait(timeout=300)
        try:
            return data, LocalStore(tmp_path).create("k", data)
        except AlreadyExists:
            return None

    with BoundedThreadPoolExecutor(max_workers=8) as pool:
        try:
            futures = [pool.submit(create, i) for i in range(8)]
            results = [future.result(timeout=60) for future in futures]
        finally:
            barrier.abort()
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
    etag = s.read("runs/a.json").etag
    assert s.replace("runs/a.json", b"2", if_match=etag) == sha256_hex(b"2")
    assert s.read("runs/a.json").data == b"2"


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


def test_replace_with_matching_etag(store: LocalStore) -> None:
    etag = store.create("control/owner.json", b"v1")
    new = store.replace("control/owner.json", b"v2", if_match=etag)
    assert new == sha256_hex(b"v2")
    assert store.read("control/owner.json") == Blob(b"v2", new)


def test_replace_with_stale_etag_fails(store: LocalStore) -> None:
    etag = store.create("k", b"v1")
    store.replace("k", b"v2", if_match=etag)
    with pytest.raises(PreconditionFailed):
        store.replace("k", b"v3", if_match=etag)
    assert store.read("k").data == b"v2"


def test_replace_missing_key_is_not_found(store: LocalStore) -> None:
    with pytest.raises(NotFound):
        store.replace("nope", b"x", if_match="0" * 64)


def _try_replace(store: LocalStore, etag: str, i: int) -> str:
    try:
        store.replace("k", f"v{i + 1}".encode(), if_match=etag)
    except PreconditionFailed:
        return "lost"
    return "ok"


def test_concurrent_replace_has_one_winner(store: LocalStore) -> None:
    etag = store.create("k", b"v0")
    barrier = Barrier(8)

    def replace(i: int) -> str:
        barrier.wait(timeout=300)
        return _try_replace(store, etag, i)

    with BoundedThreadPoolExecutor(8) as pool:
        try:
            futures = [pool.submit(replace, i) for i in range(8)]
            results = [future.result(timeout=60) for future in futures]
        finally:
            barrier.abort()
    assert results.count("ok") == 1 and results.count("lost") == 7  # noqa: PT018
    winner = results.index("ok")
    data = f"v{winner + 1}".encode()
    assert store.read("k") == Blob(data, sha256_hex(data))


def test_replace_leaves_no_temp_files(store: LocalStore, tmp_path: Path) -> None:
    etag = store.create("k", b"v0")
    store.replace("k", b"v1", if_match=etag)
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["k", "k.lock"]


def test_create_takes_no_lock_and_list_skips_lock_files(store: LocalStore, tmp_path: Path) -> None:
    store.create("runs/a.json", b"{}")
    assert sorted(p.name for p in (tmp_path / "run" / "runs").iterdir()) == ["a.json"]
    etag = store.create("runs/b.json", b"{}")
    store.replace("runs/b.json", b"{ }", if_match=etag)
    assert store.list("runs/") == ["runs/a.json", "runs/b.json"]


def test_probe_conditional_writes_passes(store: LocalStore) -> None:
    store.probe_conditional_writes()


@pytest.mark.parametrize("key", ["missing", "missing/parent/key"])
@pytest.mark.parametrize("existing_root", [False, True])
def test_replace_missing_paths_create_no_parents(
    store: LocalStore, tmp_path: Path, key: str, existing_root: bool
) -> None:
    root = tmp_path / "run"
    if existing_root:
        root.mkdir()
    with pytest.raises(NotFound) as caught:
        store.replace(key, SECRET.encode(), if_match=SECRET)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert root.exists() is existing_root
    assert not (root / "missing").exists()
    assert not (root / key).exists()
    assert store.list("") == []


@pytest.mark.parametrize("bad", ["", "/abs", "../x", "a/./b", "a/../b", "a\x00b"])
def test_replace_rejects_bad_keys(store: LocalStore, bad: str) -> None:
    with pytest.raises(ValueError, match="key") as caught:
        store.replace(bad + SECRET if bad else "", SECRET.encode(), if_match=SECRET)
    assert_secret_absent(caught.value, SECRET)


def test_replace_stale_etag_is_sanitized(store: LocalStore, tmp_path: Path) -> None:
    store.create("k", SECRET.encode())
    with pytest.raises(PreconditionFailed) as caught:
        store.replace("k", SECRET.encode(), if_match=SECRET)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert store.read("k").data == SECRET.encode()
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["k", "k.lock"]


def test_replace_temp_collision_preserves_other_writer(
    store: LocalStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    etag = store.create("k", b"v0")

    def fixed_token(nbytes: int) -> str:
        assert nbytes == 6
        return "0" * 12

    temp = tmp_path / "run" / ("k.tmp-" + "0" * 12)
    temp.write_bytes(SECRET.encode())
    monkeypatch.setattr("swarm_arena._store.local.token_hex", fixed_token)
    with pytest.raises(StorageError) as caught:
        store.replace("k", b"v1", if_match=etag)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert temp.read_bytes() == SECRET.encode()
    assert store.read("k").data == b"v0"


@pytest.mark.parametrize(
    "operation",
    ["lock_open", "lock_acquire", "read", "temp_open", "fsync", "replace", "cleanup", "unlock"],
)
def test_replace_io_errors_are_sanitized(
    store: LocalStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    etag = store.create("k", b"v0")

    def fail_io(*args: object, **kwargs: object) -> None:
        raise OSError(SECRET)

    real_flock = fcntl.flock

    def fail_flock(fd: int, mode: int) -> None:
        failing_mode = fcntl.LOCK_EX if operation == "lock_acquire" else fcntl.LOCK_UN
        if mode == failing_mode:
            raise OSError(SECRET)
        real_flock(fd, mode)

    with monkeypatch.context() as patch:
        if operation in {"lock_acquire", "unlock"}:
            patch.setattr(fcntl, "flock", fail_flock)
        elif operation in {"lock_open", "read", "cleanup"}:
            method = {"lock_open": "open", "read": "read_bytes", "cleanup": "unlink"}[operation]
            patch.setattr(Path, method, fail_io)
        else:
            patch.setattr(os, "open" if operation == "temp_open" else operation, fail_io)
        with pytest.raises(StorageError) as caught:
            store.replace("k", SECRET.encode(), if_match=etag)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    committed = operation in {"cleanup", "unlock"}
    assert store.read("k").data == (SECRET.encode() if committed else b"v0")
    assert not list((tmp_path / "run").glob("*.tmp-*"))
    # A later call must still be able to acquire the persistent lock after failure.
    current = store.read("k").etag
    assert store.replace("k", b"recovered", if_match=current) == sha256_hex(b"recovered")


def test_replace_publishes_only_complete_synced_bytes(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    etag = store.create("k", b"v0")
    data = b"complete contents" * 1024
    real_fsync, real_replace = os.fsync, os.replace
    synced = False

    def record_fsync(fd: int) -> None:
        nonlocal synced
        real_fsync(fd)
        synced = True

    def check_replace(src: Path, dst: Path) -> None:
        assert synced
        assert src.read_bytes() == data
        assert dst.read_bytes() == b"v0"
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "replace", check_replace)
    assert store.replace("k", data, if_match=etag) == sha256_hex(data)
    assert store.read("k").data == data
