import traceback
from pathlib import Path

import pytest

from swarm_arena._errors import NotFound, StorageError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._store.artifacts import ArtifactStore


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
    stored.write_bytes(b"SECRET-sentinel")
    with pytest.raises(StorageError, match="hash") as caught:
        artifacts.get_bytes(h)
    assert "SECRET-sentinel" not in "".join(traceback.format_exception(caught.value))


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
    (root / "SECRET-sentinel").write_bytes(b"payload")
    with pytest.raises(StorageError, match="hash") as caught:
        artifacts.get_bytes("SECRET-sentinel")
    assert "SECRET-sentinel" not in str(caught.value)


@pytest.mark.parametrize("name", ["swarm", "parents", "candidate"])
def test_put_file_preserves_fileref_path_rules(
    artifacts: ArtifactStore, tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    path.write_bytes(b"SECRET-sentinel")
    with pytest.raises(ValueError, match="path") as caught:
        artifacts.put_file(path)
    assert "SECRET-sentinel" not in "".join(traceback.format_exception(caught.value))


def test_put_file_error_is_sanitized(
    artifacts: ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_read(path: Path) -> bytes:
        raise OSError("SECRET-sentinel")

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    with pytest.raises(StorageError) as caught:
        artifacts.put_file(tmp_path / "x")
    assert "SECRET-sentinel" not in "".join(traceback.format_exception(caught.value))


def test_fetch_error_is_sanitized(
    artifacts: ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = artifacts.put_bytes(b"hello")

    def fail_write(path: Path, data: bytes) -> int:
        raise OSError("SECRET-sentinel")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    with pytest.raises(StorageError) as caught:
        artifacts.fetch(h, tmp_path / "output")
    assert "SECRET-sentinel" not in "".join(traceback.format_exception(caught.value))
