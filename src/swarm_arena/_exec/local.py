"""Run tool and command jobs on this host, each in its own workspace and process group."""

import contextlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Final, cast

import anyio
from pydantic import BaseModel, ConfigDict, Field

from swarm_arena._exec import procinfo
from swarm_arena._exec.env import build_env
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.pool import ResourcePool
from swarm_arena._exec.procinfo import group_alive, host_id
from swarm_arena._exec.supervisor import ProcessSupervisor, RLimits
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import config_or_fields
from swarm_arena._model.tooling import CompletedRun, RunStatus, ToolOutput, ToolRequest
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._tooling.materialize import collect_output_files, materialize
from swarm_arena._tooling.result_file import (
    command_output_from_result,
    note_in_log,
    read_regular_file,
)
from swarm_arena._tooling.tool import Tool

_STDOUT: Final = "swarm/stdout.log"
_STDERR: Final = "swarm/stderr.log"
_STOPPED: Final = frozenset({RunStatus.TIMEOUT, RunStatus.CANCELLED})
_PREFIX_UNSAFE: Final = re.compile(r"[^A-Za-z0-9._-]")
_PREFIX_MAX: Final = 64


def _workspace_root(configured: str | None) -> Path:
    """Create the workspace root if needed; refuse one that another user could change."""
    if configured is None:
        root = Path(tempfile.gettempdir()) / "swarm-arena"
    else:
        root = Path(configured).expanduser()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = root.resolve(strict=True)
        info = root.stat()
    except OSError:
        raise RuntimeError("cannot create the workspace root") from None
    owner_ok = info.st_uid in (os.getuid(), 0)
    others_write = bool(info.st_mode & stat.S_IWOTH) and not info.st_mode & stat.S_ISVTX
    if not stat.S_ISDIR(info.st_mode) or not owner_ok or others_write:
        raise RuntimeError(
            "workspace root must be a directory owned by this user (or root) that other "
            "users cannot write to, unless it is sticky like /tmp"
        )
    return root


def _open_workspace(configured: str | None, job: Job) -> tuple[Path, ArtifactStore]:
    """Open the job's artifact store and create its private workspace directory."""
    artifacts = ArtifactStore(open_store(job.artifact_root))
    root = _workspace_root(configured)
    prefix = _PREFIX_UNSAFE.sub("_", job.id)[:_PREFIX_MAX] + "-"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root)), artifacts


def _remove_workspace(workspace: Path) -> None:
    """Delete a workspace, also when the tool left read-only directories in it."""
    try:
        shutil.rmtree(workspace)
    except FileNotFoundError:
        return
    except OSError:
        _make_directories_writable(workspace)
        shutil.rmtree(workspace, ignore_errors=True)


def _make_directories_writable(top: Path) -> None:
    """Give the owner full access to every real directory below ``top``, never via a symlink."""
    try:
        if not stat.S_ISDIR(top.lstat().st_mode):
            top.unlink()
            return
        top.chmod(0o700)
    except OSError:
        return
    for parent, names, _ in os.walk(top):  # top-down: each directory is fixed before entered
        for name in names:
            path = os.path.join(parent, name)
            with contextlib.suppress(OSError):
                if stat.S_ISDIR(os.lstat(path).st_mode):
                    os.chmod(path, 0o700)


def _group_running(pgid: int) -> bool:
    """Report whether a process group has a member that is not a zombie."""
    if not group_alive(pgid):
        return False
    # A killed member that nobody has reaped yet still answers signal 0 (macOS: with
    # PermissionError), so ask ps for the members' states.
    try:
        done = subprocess.run(
            ["ps", "-A", "-o", "pgid=", "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if done.returncode != 0:
        return True
    for line in done.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == str(pgid) and not fields[1].startswith("Z"):
            return True
    return False


@dataclass(frozen=True)
class _Staged:
    """A materialized workspace and the command to run in it."""

    tool: Tool | None
    request: ToolRequest | None
    argv: list[str]
    cwd: Path
    env: dict[str, str]


def _stage(job: Job, workspace: Path, artifacts: ArtifactStore) -> _Staged:
    """Write the workspace, build the command, and resolve its environment."""
    materialize(job.stage, workspace, artifacts)
    tool: Tool | None = None
    if job.kind == "tool" and job.tool is not None and job.request is not None:
        tool = job.tool.build(Tool)
        command = tool.command(job.request, workspace)
    elif job.kind == "command" and job.command is not None:
        command = job.command
    else:  # Job.__post_init__ already refuses this
        raise ValueError("job has no tool or command for its kind")
    command = command.model_copy(update={"timeout_s": job.timeout_s, "grace_s": job.grace_s})
    argv = list(command.argv)
    if argv[0] == "python":
        argv[0] = command.python or sys.executable
    cwd = workspace / command.cwd if command.cwd is not None else workspace
    env = build_env(job, command, workspace=workspace, node_env=os.environ)
    return _Staged(tool=tool, request=job.request, argv=argv, cwd=cwd, env=env)


def _tag_start(job: Job) -> Callable[[Job.Started], Awaitable[None]] | None:
    """Wrap ``job.on_started`` so the start event carries the job id."""
    commit = job.on_started
    if commit is None:
        return None

    async def on_started(started: Job.Started) -> None:
        await commit(started.model_copy(update={"job_id": job.id}))

    return on_started


class _JobRun:
    """Supervise one staged job, keep its partial output, and upload its results."""

    def __init__(
        self, job: Job, staged: _Staged, workspace: Path, artifacts: ArtifactStore
    ) -> None:
        self._job = job
        self._staged = staged
        self._workspace = workspace
        self._artifacts = artifacts
        self._stderr = workspace / _STDERR
        self._partial: ToolOutput | None = None
        self._provisional: CompletedRun | None = None

    async def run(self, supervisor: ProcessSupervisor, rlimits: RLimits | None) -> ToolOutput:
        """Run the job; on cancellation deliver the partial output, then re-raise."""
        try:
            run = await supervisor.run(
                self._staged.argv,
                cwd=self._staged.cwd,
                env=self._staged.env,
                stdout=self._workspace / _STDOUT,
                stderr=self._stderr,
                timeout_s=self._job.timeout_s,
                grace_s=self._job.grace_s,
                rlimits=rlimits,
                on_started=_tag_start(self._job),
                before_kill=self._before_kill,
            )
            if run.status in _STOPPED:  # collect ran (or was given up) in before_kill
                output = self._partial or await anyio.to_thread.run_sync(
                    self._without_partial, run.status
                )
            else:
                collected = await anyio.to_thread.run_sync(
                    self._collect, run, abandon_on_cancel=True
                )
                output = collected or ToolOutput(status=RunStatus.ERROR)
            return await anyio.to_thread.run_sync(
                self._finish, output, (run.started_at, run.ended_at)
            )
        except anyio.get_cancelled_exc_class():
            with anyio.CancelScope(shield=True):
                await self._deliver_partial()
            raise

    async def _deliver_partial(self) -> None:
        """Finish the cancel-path output and hand it to ``job.on_partial`` (caller shields)."""
        output = self._partial or await anyio.to_thread.run_sync(
            self._without_partial, RunStatus.CANCELLED
        )
        provisional = self._provisional
        times = None if provisional is None else (provisional.started_at, provisional.ended_at)
        # An upload failure must not replace the cancellation; the output goes without logs.
        with contextlib.suppress(Exception):
            output = await anyio.to_thread.run_sync(self._finish, output, times)
        if self._job.on_partial is not None:
            await self._job.on_partial(output)

    async def _before_kill(self, provisional: CompletedRun) -> None:
        """Collect the partial output in a thread the supervisor's deadline can abandon."""
        self._provisional = provisional
        self._partial = await anyio.to_thread.run_sync(
            self._collect, provisional, abandon_on_cancel=True
        )

    def _collect(self, run: CompletedRun) -> ToolOutput | None:
        """Collect the output; on an exception note the traceback and return None."""
        try:
            if self._staged.tool is not None and self._staged.request is not None:
                return self._staged.tool.collect(self._staged.request, self._workspace, run)
            return command_output_from_result(self._workspace, run)
        except Exception:
            note_in_log(run.stderr_path, "collect failed: " + traceback.format_exc())
            return None

    def _without_partial(self, status: RunStatus) -> ToolOutput:
        """Return an empty output for a stopped run whose collect failed or never ran."""
        if self._provisional is not None:
            note_in_log(self._stderr, "no partial output: collect failed or timed out")
        return ToolOutput(status=status)

    def _read_log(self, rel: str) -> bytes:
        """Read a log that is a regular file; anything else is uploaded as empty bytes."""
        try:
            return read_regular_file(self._workspace, rel)
        except ValueError:
            note_in_log(self._stderr, f"{rel} is not a regular file; its log is uploaded empty")
            return b""

    def _finish(self, output: ToolOutput, times: tuple[datetime, datetime] | None) -> ToolOutput:
        """Upload declared files and both logs, then set ``logs_ref`` and the run times."""
        update: dict[str, Any] = {}
        try:
            update["files"] = collect_output_files(
                self._workspace,
                [ref.path for ref in output.files],
                self._artifacts,
                stage=self._job.stage,
            )
        except ValueError as exc:
            note_in_log(self._stderr, f"output files not uploaded: {exc}")
            update |= {"status": RunStatus.ERROR, "files": []}
        # stdout first: a note about it must reach stderr before stderr is read.
        hashes = [self._artifacts.put_bytes(self._read_log(rel)) for rel in (_STDOUT, _STDERR)]
        update["logs_ref"] = "sha256:" + ":".join(hashes)
        if times is not None:
            update["started_at"], update["ended_at"] = times
        return output.model_copy(update=update)


class LocalExecutor(Executor):
    """
    Run jobs on this POSIX host, each in a fresh workspace and its own process group.

    For each job, ``submit`` waits for a slot in the resource pool, creates
    ``<workspace_root>/<job id>-<random>``, stages ``job.stage`` into it, and
    runs the command through the launcher (``python -I -m swarm_arena._launch``)
    under a process-group supervisor: the command starts only after
    ``job.on_started`` returns. A leading ``"python"`` in the command becomes
    ``Command.python`` or ``sys.executable``. The environment is the allowlist
    of ``build_env``; ``Job.timeout_s`` and ``Job.grace_s`` replace the
    command's own values. stdout and stderr go to ``swarm/stdout.log`` and
    ``swarm/stderr.log`` in the workspace and are uploaded to
    ``job.artifact_root``; ``logs_ref`` is ``"sha256:<stdout>:<stderr>"``.

    Construction does no I/O; the pool and the host id are made on the first
    ``submit``.

    Parameters
    ----------
    config : LocalExecutor.Config or None, optional
        Validated settings, mutually exclusive with keyword fields.
    **fields : Any
        Settings passed to ``Config`` when no explicit configuration is supplied.

    Notes
    -----
    swarm-arena must be installed in the interpreter's own environment (a venv
    or its site-packages): the launcher runs as ``python -I``, which reads
    neither ``PYTHONPATH`` nor the user site.

    On a timeout or a cancel, the group gets SIGTERM, then ``grace_s``, then
    ``Tool.collect`` for the partial output, then SIGKILL. A collect that blocks
    is abandoned after ``BEFORE_KILL_TIMEOUT_S`` and is not tried again. The
    cancel-path output goes to ``job.on_partial``. Every diagnostic is appended
    to the stderr log only while it is a regular file, so a tool that replaces
    a log with a FIFO or a symlink cannot stall the executor.

    This is not strong isolation: the tool runs as the same user, with
    resource limits but without a sandbox.

    Examples
    --------
    >>> from swarm_arena.executor import LocalExecutor
    >>> executor = LocalExecutor(max_concurrent=2, keep_workspaces=True)
    >>> executor.config.max_concurrent
    2
    >>> output = await executor.submit(job)  # doctest: +SKIP
    """

    class Config(BaseModel):
        """
        Configure local concurrency, workspaces, and resource limits.

        Parameters
        ----------
        max_concurrent : int or None, default=None
            Most jobs that run at once; None selects ``os.cpu_count()``.
        workspace_root : str or None, default=None
            Directory that holds the job workspaces; None selects
            ``tempfile.gettempdir()/swarm-arena``. It must be owned by this
            user and not writable by other users (unless it is sticky).
        keep_workspaces : bool, default=False
            Keep each workspace after its job instead of removing it.
        rlimits : bool, default=True
            Apply resource limits; ``Resources.memory_gib`` becomes the
            address-space limit.

        Examples
        --------
        >>> from swarm_arena.executor import LocalExecutor
        >>> LocalExecutor.Config().rlimits
        True
        """

        model_config = ConfigDict(extra="forbid", frozen=True)

        max_concurrent: Annotated[int, Field(ge=1)] | None = None
        workspace_root: str | None = None
        keep_workspaces: bool = False
        rlimits: bool = True

    def __init__(self, config: Config | None = None, /, **fields: Any) -> None:
        """Validate configuration without touching the filesystem or processes."""
        super().__init__(config_or_fields(config, fields))
        self._pool: ResourcePool | None = None
        self._host_id: str | None = None
        self._supervisor: ProcessSupervisor | None = None

    @property
    def config(self) -> Config:
        """
        Return the validated executor settings.

        Returns
        -------
        LocalExecutor.Config
            Frozen configuration supplied at construction.

        Examples
        --------
        >>> from swarm_arena.executor import LocalExecutor
        >>> LocalExecutor(keep_workspaces=True).config.keep_workspaces
        True
        """
        return cast(LocalExecutor.Config, super().config)

    def preflight(self) -> None:
        """
        Refuse a platform that is not POSIX.

        Raises
        ------
        RuntimeError
            If ``os.name`` is not ``"posix"``.

        Examples
        --------
        >>> from swarm_arena.executor import LocalExecutor
        >>> LocalExecutor().preflight()
        """
        if os.name != "posix":
            raise RuntimeError("LocalExecutor needs a POSIX system")

    async def capacity(self) -> Resources:
        """
        Return the CPU capacity of this host.

        Returns
        -------
        Resources
            ``cpus`` is ``os.cpu_count()`` (1 when unknown); other fields keep
            their defaults.

        Examples
        --------
        >>> capacity = await LocalExecutor().capacity()  # doctest: +SKIP
        """
        return Resources(cpus=float(os.cpu_count() or 1))

    async def aclose(self) -> None:
        """
        Release nothing; a local executor holds no resource between jobs.

        Examples
        --------
        >>> await executor.aclose()  # doctest: +SKIP
        """

    async def submit(self, job: Job) -> ToolOutput:
        """
        Stage and run a job, and return its output with uploaded files and logs.

        A job that exits on its own is collected with ``Tool.collect`` (tool
        jobs) or from its raw ``swarm/result.json`` (command jobs); an exception
        in collect gives status ERROR, with the traceback in the stderr log.
        After a timeout the partial output collected before SIGKILL is
        returned. A declared output file that cannot be uploaded gives status
        ERROR with no files. The workspace is removed unless
        ``keep_workspaces`` is set.

        Parameters
        ----------
        job : Job
            Staged invocation, limits, environment, and lifecycle callbacks.

        Returns
        -------
        ToolOutput
            Output with ``files`` uploaded, ``logs_ref`` set, and the run's
            start and end times.

        Raises
        ------
        ValueError
            If the job needs more CPUs than this host has (before anything
            starts), or its stage or secrets cannot be read.
        RuntimeError
            If the workspace root cannot be created or is not safe.
        Exception
            Whatever ``job.on_started`` raises; the command then never runs.

        Examples
        --------
        >>> output = await LocalExecutor().submit(job)  # doctest: +SKIP
        >>> output.logs_ref  # doctest: +SKIP
        'sha256:...:...'
        """
        if self._pool is None:
            cpus = os.cpu_count() or 1
            self._pool = ResourcePool(
                cpus=float(cpus), max_concurrent=self.config.max_concurrent or cpus
            )
        async with self._pool.slot(job.resources):
            supervisor = await self._get_supervisor()
            with anyio.CancelScope(shield=True):  # a cancel here must not lose the directory
                workspace, artifacts = await anyio.to_thread.run_sync(
                    _open_workspace, self.config.workspace_root, job
                )
            try:
                staged = await anyio.to_thread.run_sync(_stage, job, workspace, artifacts)
                return await _JobRun(job, staged, workspace, artifacts).run(
                    supervisor, self._rlimits(job)
                )
            finally:
                if not self.config.keep_workspaces:
                    with anyio.CancelScope(shield=True):
                        await anyio.to_thread.run_sync(_remove_workspace, workspace)

    async def confirm_dead(self, started: Job.Started) -> bool | None:
        """
        Report whether a recorded process group on this host has stopped.

        Members that were killed but not yet reaped (zombies) count as dead.

        Parameters
        ----------
        started : Job.Started
            Committed start event of the job.

        Returns
        -------
        bool or None
            True when no live member of the group is left, False while one
            is, and None for a dispatch, another host, or an unknown boot.

        Examples
        --------
        >>> dead = await LocalExecutor().confirm_dead(started)  # doctest: +SKIP
        """
        pgid, recorded = started.pgid, started.host_id
        if started.kind != "process" or pgid is None or pgid <= 1 or recorded is None:
            return None
        if recorded.endswith(":unknown") or recorded != await self._get_host_id():
            return None
        return not await anyio.to_thread.run_sync(_group_running, pgid)

    async def kill_orphan(self, started: Job.Started) -> bool:
        """
        Kill a recorded process group when it is provably the same one.

        The record must be a process start on this host in this boot, and
        the live leader must have the recorded start time.

        Parameters
        ----------
        started : Job.Started
            Committed start event of the job.

        Returns
        -------
        bool
            True when SIGKILL was sent to the group, otherwise False.

        Examples
        --------
        >>> killed = await LocalExecutor().kill_orphan(started)  # doctest: +SKIP
        """
        return await anyio.to_thread.run_sync(procinfo.kill_orphan, started)

    async def _get_host_id(self) -> str:
        """Return this host's id, read once in a thread."""
        known = self._host_id
        if known is None:
            known = await anyio.to_thread.run_sync(host_id)
            self._host_id = known
        return known

    async def _get_supervisor(self) -> ProcessSupervisor:
        """Return the one supervisor, made on first use."""
        supervisor = self._supervisor
        if supervisor is None:
            supervisor = ProcessSupervisor(host_id=await self._get_host_id())
            self._supervisor = supervisor
        return supervisor

    def _rlimits(self, job: Job) -> RLimits | None:
        """Return the job's resource limits, or None when they are turned off."""
        if not self.config.rlimits:
            return None
        memory = job.resources.memory_gib
        return RLimits(memory_bytes=int(memory * 2**30) if memory else None)
