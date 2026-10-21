"""Materialize a local job and resolve its command, limits, and environment."""

import asyncio
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from swarm_arena._exec.env import build_env
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local_workspace import _DIR_FLAGS
from swarm_arena._model.tooling import ToolRequest
from swarm_arena._outcomes import _classify_outcome
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.materialize import materialize
from swarm_arena._tooling.tool import Tool


def _cwd_fault(workspace: Path, rel: str) -> str | None:
    """Return a note when the command's cwd is not a real directory in the workspace."""
    directories: list[int] = []
    try:
        directories.append(os.open(workspace, _DIR_FLAGS))
        for part in PurePosixPath(rel).parts:
            directories.append(os.open(part, _DIR_FLAGS, dir_fd=directories[-1]))
    except OSError:
        return f"command cwd {rel} is not a directory in the workspace"
    finally:
        for directory in directories:
            os.close(directory)
    return None


@dataclass(frozen=True)
class _Staged:
    """A materialized workspace and the command to run in it, or why it cannot run."""

    tool: Tool | None
    request: ToolRequest | None
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    fault: str | None
    timeout_s: float | None
    grace_s: float


@dataclass
class _Preparation:
    """Own a workspace until all synchronous preparation has stopped using it."""

    job: Job
    workspace: Path
    artifacts: ArtifactStore
    finished: threading.Event = field(default_factory=threading.Event, init=False)

    def run(self) -> _Staged:
        """Prepare once, marking workspace access finished even on a user error."""
        try:
            return _stage(self.job, self.workspace, self.artifacts)
        except (BaseExceptionGroup, asyncio.CancelledError) as error:
            raise _classify_outcome(error) from None
        finally:
            self.finished.set()


def _stage(job: Job, workspace: Path, artifacts: ArtifactStore) -> _Staged:
    """Write the workspace, build the command, and resolve its environment."""
    materialize(job.stage, workspace, artifacts)
    tool: Tool | None = None
    if job.kind == "tool" and job.tool is not None and job.request is not None:
        tool = job.tool.build(Tool)
        try:
            command = tool.command(job.request, workspace)
        except SystemExit:
            return _Staged(
                tool,
                job.request,
                [],
                workspace,
                {},
                "command failed: SystemExit",
                job.timeout_s,
                job.grace_s,
            )
    elif job.kind == "command" and job.command is not None:
        command = job.command
    else:  # Job.__post_init__ already refuses this
        raise ValueError("job has no tool or command for its kind")
    limits = [value for value in (job.timeout_s, command.timeout_s) if value is not None]
    timeout_s = min(limits) if limits else None
    grace_s = min(job.grace_s, command.grace_s)
    argv = list(command.argv)
    if argv[0] == "python":
        argv[0] = command.python or sys.executable
    cwd = workspace / command.cwd if command.cwd is not None else workspace
    fault = _cwd_fault(workspace, command.cwd) if command.cwd is not None else None
    env = build_env(job, command, workspace=workspace, node_env=os.environ)
    return _Staged(
        tool=tool,
        request=job.request,
        argv=argv,
        cwd=cwd,
        env=env,
        fault=fault,
        timeout_s=timeout_s,
        grace_s=grace_s,
    )
