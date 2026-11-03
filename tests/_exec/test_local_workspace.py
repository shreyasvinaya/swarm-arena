"""Refuse links planted for the package-created workspace root, and keep configured roots."""

import os
import stat
import tempfile
from pathlib import Path

import pytest

from swarm_arena._exec.local_workspace import _workspace_root


def _private(path: Path) -> Path:
    """Create ``path`` as a directory that only this user can change, like a victim's own."""
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def test_default_root_planted_as_a_link_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = _private(tmp_path / "temp")
    target = _private(tmp_path / "victim")
    # Another user of a shared temporary directory plants the per-user root as a link.
    (temp / f"swarm-arena-{os.getuid()}").symlink_to(target)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    with pytest.raises(RuntimeError, match=r"^workspace root is not safe: "):
        _workspace_root(None)
    assert list(target.iterdir()) == []


def test_default_root_that_is_not_a_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = _private(tmp_path / "temp")
    (temp / f"swarm-arena-{os.getuid()}").write_text("")
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    with pytest.raises(RuntimeError, match=r"^workspace root is not safe: "):
        _workspace_root(None)


def test_default_root_under_a_linked_temporary_directory_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = _private(tmp_path / "real")
    # The temporary directory itself may be a link, like /var -> /private/var on macOS.
    (tmp_path / "temp").symlink_to(real)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "temp"))
    old = os.umask(0o022)
    try:
        root = _workspace_root(None)
    finally:
        os.umask(old)
    assert root == real.resolve() / f"swarm-arena-{os.getuid()}"
    info = os.lstat(root)
    assert stat.S_ISDIR(info.st_mode)
    assert stat.S_IMODE(info.st_mode) == 0o700
    assert _workspace_root(None) == root


def test_configured_root_may_be_a_link_to_a_safe_directory(tmp_path: Path) -> None:
    target = _private(tmp_path / "target")
    (tmp_path / "link").symlink_to(target)
    assert _workspace_root(str(tmp_path / "link")) == target.resolve()


def test_configured_root_link_to_an_unsafe_directory_is_refused(tmp_path: Path) -> None:
    target = _private(tmp_path / "target")
    target.chmod(0o777)
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(RuntimeError, match=r"^workspace root is not safe: "):
        _workspace_root(str(tmp_path / "link"))


def test_configured_root_is_created_with_its_parents(tmp_path: Path) -> None:
    root = _workspace_root(str(tmp_path / "a" / "b"))
    assert root == (tmp_path / "a" / "b").resolve()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
