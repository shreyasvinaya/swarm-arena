"""Shared fixtures, and collection rules for test paths that need an optional extra."""

import importlib.metadata
from pathlib import Path

import pytest

from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store


def _missing(dist: str) -> bool:
    """Return True when the distribution ``dist`` is not installed."""
    try:
        importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return True
    return False


# By distribution, never find_spec: pythonpath = ["tests"] makes tests/ray/ a namespace package.
# -m deselection happens only after import, so a path that imports an extra must not be collected.
collect_ignore: list[str] = []
if _missing("openai-agents"):
    collect_ignore += [
        "llm",
        "testing/test_fake_model.py",
        "e2e/test_rosenbrock_llm.py",
        "e2e/test_autonomous_replan.py",
        "e2e/test_llm_plan_file.py",
        "e2e/test_llm_final_slice.py",
    ]
if _missing("ray"):
    collect_ignore += [
        "ray",
        "e2e/test_rosenbrock_ray.py",
        "e2e/test_ray_cancel.py",
        "e2e/test_ray_dispatch_crash.py",
    ]
if _missing("s3fs"):
    collect_ignore.append("_store/test_s3.py")

# Script fixtures and test helpers are never imported as doctest modules.
collect_ignore_glob = ["*/data/*", "_fakes/*"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def artifact_root(tmp_path: Path) -> str:
    return str(tmp_path / "artifacts")


@pytest.fixture
def artifacts(artifact_root: str) -> ArtifactStore:
    return ArtifactStore(open_store(artifact_root))
