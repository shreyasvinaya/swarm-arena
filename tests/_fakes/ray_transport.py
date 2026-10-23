"""Deny shared artifact/workspace reads around the production Ray transport."""

import importlib
import os
import sys
import tempfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from _fakes.dependencies import ray_dependency
from _fakes.tools import PyTool

from swarm_arena._model.canonical import sha256_hex
from swarm_arena.tool import Command, CompletedRun, FileRef, ToolOutput, ToolRequest

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


class PartialFilesTool(PyTool):
    """Declare a real file during stop collection, before transport strips it."""

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        (workspace / "progress.txt").write_bytes(b"progress")
        (workspace / "swarm/llm_calls.jsonl").write_bytes(b"")
        return super().command(request, workspace)

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        output = super().collect(request, workspace, run)
        return output.model_copy(
            update={
                "data": {**output.data, "declared_files": 1},
                "files": [
                    FileRef(
                        name="progress", path="progress.txt", sha256=sha256_hex(b"progress"), size=8
                    )
                ],
            }
        )


@contextmanager
def _deny_files(
    *, roots: tuple[Path, ...] = (), worker_workspaces: bool = False
) -> Generator[None]:
    active = True
    owner = os.getpid()

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if not active or os.getpid() != owner or event not in {"open", "os.listdir", "os.scandir"}:
            return
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(root) for root in roots) or (
            worker_workspaces and any(part.startswith("swarm-worker-") for part in path.parts)
        ):
            raise PermissionError("test filesystem boundary")

    sys.addaudithook(audit)
    try:
        yield
    finally:
        active = False


def _without_driver_files(payload: bytes, *, driver_roots: tuple[Path, ...] = ()) -> bytes:
    # The wrapper changes only access permissions; execution/framing stays production code.
    from swarm_arena.ray._task import read_payload, run_job

    job, settings, _ = read_payload(payload)
    assert settings.cap_bytes is not None
    # Load the worker runtime before denying the checkout, which may contain
    # its installed environment in .venv during local acceptance.
    importlib.import_module("anyio._backends._asyncio")
    # Import connector code before denying source-data access. FunctionTool's
    # subprocess still needs its documented worker-accessible module root.
    sys.path[:0] = list(settings.import_roots)
    if job.tool is not None and isinstance(function := job.tool.config.get("function"), str):
        importlib.import_module(function.split(":", 1)[0])
    root = Path(job.artifact_root).resolve()
    roots = (root, *driver_roots)
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="swarm-ray-cwd-") as directory:
        os.chdir(directory)
        try:
            with _deny_files(roots=roots):
                for denied in roots:
                    with pytest.raises(PermissionError, match="test filesystem boundary"):
                        (denied / "denial-probe").read_bytes()
                return run_job._function(payload)  # ty: ignore[unresolved-attribute]
        finally:
            os.chdir(previous)


@pytest.fixture
def isolated_ray_filesystem(
    ray_cluster: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Use a real Ray task while refusing cross-boundary Python filesystem reads."""
    ray_dependency()
    from swarm_arena.ray import _executor

    ray = _executor.ray

    driver_roots = (Path.cwd().resolve(), tmp_path.resolve())

    def guarded_task(payload: bytes) -> bytes:
        return _without_driver_files(payload, driver_roots=driver_roots)

    guarded = ray.remote(max_retries=0, max_calls=1)(guarded_task)
    monkeypatch.setattr(_executor, "run_job", guarded)
    probe = tmp_path / "swarm-worker-denial-probe"
    probe.mkdir()
    (probe / "sentinel").write_bytes(b"driver cannot read worker workspaces")
    with _deny_files(worker_workspaces=True):
        with pytest.raises(PermissionError, match="test filesystem boundary"):
            (probe / "sentinel").read_bytes()
        yield
