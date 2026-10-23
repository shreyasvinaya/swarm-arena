"""Optional Ray tests skip missing extras and fail broken installed runtimes."""

import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from _fakes.dependencies import ray_dependency


@pytest.mark.parametrize("ci", [False, True])
def test_missing_ray_distribution_skips(ci: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if ci:
        monkeypatch.setenv("CI", "true")
    else:
        monkeypatch.delenv("CI", raising=False)

    def absent(name: str) -> str:
        assert name == "ray"
        raise importlib.metadata.PackageNotFoundError(name)

    def unexpected(name: str) -> ModuleType:
        raise AssertionError("missing distribution must not be imported")

    monkeypatch.setattr(importlib.metadata, "version", absent)
    monkeypatch.setattr(importlib, "import_module", unexpected)
    with pytest.raises(pytest.skip.Exception, match="ray extra"):
        ray_dependency()


def test_installed_ray_returns_imported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("swarm_arena.ray")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.59.0")
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    assert ray_dependency() is module


@pytest.mark.parametrize("ci", [False, True])
@pytest.mark.parametrize("error_type", [ModuleNotFoundError, ImportError, RuntimeError])
def test_broken_installed_ray_fails(
    ci: bool, error_type: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    if ci:
        monkeypatch.setenv("CI", "true")
    else:
        monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.59.0")

    def broken(name: str) -> ModuleType:
        raise error_type("private-import-detail")

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(pytest.fail.Exception, match="installed Ray runtime failed to import"):
        ray_dependency()


def test_missing_internal_module_fails_with_installed_ray() -> None:
    try:
        importlib.metadata.version("ray")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("internal import regression requires the ray extra")
    code = f"""
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from _fakes.dependencies import ray_dependency
import pytest
sys.modules['swarm_arena.ray._state'] = None
try:
    ray_dependency()
except pytest.fail.Exception:
    pass
except pytest.skip.Exception:
    raise AssertionError('broken installed runtime was skipped') from None
else:
    raise AssertionError('broken internal module imported')
"""
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60
    )
    assert done.returncode == 0, done.stdout + done.stderr
