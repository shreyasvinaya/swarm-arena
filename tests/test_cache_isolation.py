"""The shared guard rejects default disk caches in a test working directory."""

from inspect import unwrap
from pathlib import Path

import pytest

from tests import conftest


def test_cwd_disk_cache_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    guard = unwrap(conftest.no_cwd_disk_cache)()
    next(guard)
    cache = tmp_path / ".litellm_cache"
    cache.mkdir()
    try:
        with pytest.raises(AssertionError, match="disk_dir"):
            next(guard)
    finally:
        cache.rmdir()


def test_cwd_disk_cache_guard_accepts_clean_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    guard = unwrap(conftest.no_cwd_disk_cache)()
    next(guard)
    with pytest.raises(StopIteration):
        next(guard)


def test_cwd_guard_does_not_blame_test_for_preexisting_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".litellm_cache"
    cache.mkdir()
    try:
        guard = unwrap(conftest.no_cwd_disk_cache)()
        next(guard)
        with pytest.raises(StopIteration):
            next(guard)
    finally:
        cache.rmdir()


def test_session_cache_check_rejects_preexisting_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".litellm_cache"
    cache.mkdir()
    try:
        with pytest.raises(pytest.exit.Exception, match="pre-existing"):
            conftest.pytest_sessionstart()
    finally:
        cache.rmdir()


def test_session_cache_check_accepts_clean_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    conftest.pytest_sessionstart()
