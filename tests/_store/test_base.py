from pathlib import Path

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import ConditionalWritesUnsupported
from swarm_arena._store.base import Blob, Store, open_store
from swarm_arena._store.local import LocalStore

SECRET = "9537d6d3032b2555b86a3df9fb9abdda"


def test_blob() -> None:
    blob = Blob(b"x", "etag")
    assert tuple(blob) == (b"x", "etag")
    with pytest.raises(AttributeError, match="can't set attribute"):
        blob.data = b"y"  # ty: ignore[invalid-assignment]


def test_store_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Store()  # ty: ignore[call-non-callable]


def test_open_store(tmp_path: Path) -> None:
    s = open_store(str(tmp_path))
    assert isinstance(s, LocalStore)
    assert s.url == str(tmp_path.resolve())
    s.probe_conditional_writes()
    with pytest.raises(ConditionalWritesUnsupported, match="s3"):
        open_store("s3://bucket/p")
    with pytest.raises(ConditionalWritesUnsupported, match="gs"):
        open_store("gs://bucket")


def test_open_store_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert open_store("~/artifacts").url == str(tmp_path / "artifacts")


@pytest.mark.parametrize("scheme", ["s3", "gs", "file", "scheme" + SECRET + "+1.2"])
def test_scheme_errors_are_sanitized(scheme: str) -> None:
    url = f"{scheme}://9537d6d3032b2555b86a3df9fb9abdda:password@bucket/private"
    with pytest.raises(ConditionalWritesUnsupported) as caught:
        open_store(url)
    assert_secret_absent(caught.value, SECRET)


def test_base_child(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    child = Store.child(s, "runs")
    child.create("a", b"x")
    assert s.read("runs/a").data == b"x"
