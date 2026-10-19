"""Run tool and command jobs on this host, each in its own workspace and process group."""

import contextlib
import logging
import os
import re
import shutil
import stat
import sys
import tempfile
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final, cast

import anyio
from pydantic import ConfigDict, Field

from swarm_arena._exec import procinfo, supervisor
from swarm_arena._exec.env import build_env
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.pool import ResourcePool
from swarm_arena._exec.procinfo import group_has_live_member, host_id
from swarm_arena._exec.supervisor import ProcessSupervisor, RLimits
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import _ConfigInputModel, config_or_fields
from swarm_arena._model.tooling import CompletedRun, FileRef, RunStatus, ToolOutput, ToolRequest
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._tooling.materialize import collect_output_files, materialize
from swarm_arena._tooling.result_file import command_output_from_result, note_in_log, open_log
from swarm_arena._tooling.tool import Tool

_LOG: Final = logging.getLogger(__name__)
_STDOUT: Final = "swarm/stdout.log"
_STDERR: Final = "swarm/stderr.log"
_STOPPED: Final = frozenset({RunStatus.TIMEOUT, RunStatus.CANCELLED})
_PREFIX_UNSAFE: Final = re.compile(r"[^A-Za-z0-9._-]")
_PREFIX_MAX: Final = 64
_DIR_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_READ_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_CHUNK: Final = 1 << 20
_UNSAFE_ROOT: Final = (
    "workspace root is not safe: it and every parent directory must be owned by this user "
    "or root, and not writable by group or others unless sticky (like /tmp)"
)


def _default_root() -> Path:
    """Return this user's default workspace root, so users of one host never share it."""
    return Path(tempfile.gettempdir()) / f"swarm-arena-{os.getuid()}"


def _only_owner_can_change(info: os.stat_result) -> bool:
    """Return whether only this user or root can add, rename, or remove a directory's entries."""
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (os.getuid(), 0):
        return False
    shared = info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    return not shared or bool(info.st_mode & stat.S_ISVTX)


def _workspace_root(configured: str | None) -> Path:
    """Create the workspace root if needed; refuse one that another user could change."""
    root = Path(configured).expanduser() if configured is not None else _default_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = root.resolve(strict=True)
        safe = all(_only_owner_can_change(os.stat(path)) for path in (root, *root.parents))
    except OSError:
        raise RuntimeError("cannot create the workspace root") from None
    if not safe:
        raise RuntimeError(_UNSAFE_ROOT)
    return root


def _open_workspace(configured: str | None, job: Job) -> tuple[Path, ArtifactStore]:
    """Open the job's artifact store and create its private workspace directory."""
    artifacts = ArtifactStore(open_store(job.artifact_root))
    root = _workspace_root(configured)
    prefix = _PREFIX_UNSAFE.sub("_", job.id)[:_PREFIX_MAX] + "-"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root)), artifacts


def _open_subdirectory(parent: int, name: str) -> int:
    """Open a real subdirectory, first giving it back to the owner if the tool locked it."""
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=parent)
    except PermissionError:
        os.chmod(name, 0o700, dir_fd=parent)  # the caller's lstat saw a directory, not a link
        return os.open(name, _DIR_FLAGS, dir_fd=parent)


def _clear_files(directory: int) -> str | None:
    """Unlink every entry that is not a directory, and return one subdirectory or None."""
    os.fchmod(directory, 0o700)  # a read-only directory refuses unlink
    subdirectory: str | None = None
    for name in os.listdir(directory):
        if stat.S_ISDIR(os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode):
            subdirectory = subdirectory or name
        else:
            os.unlink(name, dir_fd=directory)
    return subdirectory


def _remove_tree(top: Path) -> None:
    """
    Remove a tree without recursion, holding at most two descriptors at a time.

    The walk keeps an explicit stack of names and inodes and opens every child
    from its parent's descriptor, so neither the recursion limit nor PATH_MAX
    applies, and a symlink is unlinked, never followed. Going back up opens
    ``..`` and checks that it is the directory the walk came from.
    """
    if not stat.S_ISDIR(os.lstat(top).st_mode):
        os.unlink(top)
        return
    fd = os.open(top, _DIR_FLAGS)
    try:
        info = os.fstat(fd)
        stack: list[tuple[str, tuple[int, int]]] = [("", (info.st_dev, info.st_ino))]
        while True:
            child = _clear_files(fd)
            if child is not None:
                opened = _open_subdirectory(fd, child)
                os.close(fd)
                fd = opened
                info = os.fstat(fd)
                stack.append((child, (info.st_dev, info.st_ino)))
                continue
            if len(stack) == 1:
                break
            name, _ = stack.pop()
            parent = os.open("..", _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = parent
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != stack[-1][1]:
                raise OSError("workspace tree changed during removal")
            os.rmdir(name, dir_fd=fd)
    finally:
        os.close(fd)
    os.rmdir(top)


_REMOVERS: tuple[Callable[[Path], None], ...] = (shutil.rmtree, _remove_tree)


def _remove_workspace(workspace: Path) -> None:
    """Remove a workspace without ever raising; log a warning for one that is left."""
    for remove in _REMOVERS:
        # RecursionError included: on Python 3.11 rmtree recurses once per directory level.
        with contextlib.suppress(Exception):
            remove(workspace)
        if not os.path.lexists(workspace):
            return
    _LOG.warning("could not remove workspace %s", workspace)


def _open_regular(workspace: Path, rel: str) -> int:
    """Open a regular workspace file read-only through pinned directories, never via a link."""
    parts = PurePosixPath(rel).parts
    if not parts or rel.startswith("/") or ".." in parts:
        raise ValueError(f"{rel} is not a regular file")
    directories: list[int] = []
    try:
        try:
            directories.append(os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NONBLOCK))
            for part in parts[:-1]:
                directories.append(
                    os.open(part, _READ_FLAGS | os.O_DIRECTORY, dir_fd=directories[-1])
                )
            fd = os.open(parts[-1], _READ_FLAGS, dir_fd=directories[-1])
        finally:
            for directory in directories:
                os.close(directory)
    except OSError:
        raise ValueError(f"{rel} is not a regular file") from None
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"{rel} is not a regular file")
    return fd


def _read_tail(fd: int, size: int, limit: int) -> bytes:
    """Read the last ``limit`` of ``size`` bytes at most, even if the file grows meanwhile."""
    offset = max(0, size - limit)
    remaining = size - offset
    chunks: list[bytes] = []
    while remaining > 0:
        chunk = os.pread(fd, min(remaining, _CHUNK), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _truncated(rel: str, dropped: int) -> str:
    """Return the note for a log whose first ``dropped`` bytes are not uploaded."""
    return f"log truncated: {rel}: dropped the first {dropped} bytes"


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
    fault = _cwd_fault(workspace, command.cwd) if command.cwd is not None else None
    env = build_env(job, command, workspace=workspace, node_env=os.environ)
    return _Staged(tool=tool, request=job.request, argv=argv, cwd=cwd, env=env, fault=fault)


class _JobRun:
    """Supervise one staged job, keep its collected output, and upload its results."""

    def __init__(
        self,
        job: Job,
        staged: _Staged,
        workspace: Path,
        artifacts: ArtifactStore,
        config: "LocalExecutor.Config",
    ) -> None:
        self._job = job
        self._staged = staged
        self._workspace = workspace
        self._artifacts = artifacts
        self._max_log = config.max_log_bytes
        self._max_output = config.max_output_bytes
        self._stderr = workspace / _STDERR
        self._started = False  # the launcher reached the start handshake
        self._output: ToolOutput | None = None  # the latest collected output
        self._collected = False  # the collect after a normal exit finished
        self._run: CompletedRun | None = None  # what the supervisor returned
        self._provisional: CompletedRun | None = None  # the run a stop-path collect saw

    async def run(self, sup: ProcessSupervisor, rlimits: RLimits | None) -> ToolOutput:
        """Run the job; on a cancel after the start, deliver its output, then re-raise."""
        try:
            run = await sup.run(
                self._staged.argv,
                cwd=self._staged.cwd,
                env=self._staged.env,
                stdout=self._workspace / _STDOUT,
                stderr=self._stderr,
                timeout_s=self._job.timeout_s,
                grace_s=self._job.grace_s,
                rlimits=rlimits,
                on_started=self._on_started,
                before_kill=self._before_kill,
            )
            self._run = run
            if run.status in _STOPPED:  # collect ran (or was given up) in before_kill
                output = await anyio.to_thread.run_sync(self._stopped_output, run.status)
            else:
                self._output = await run_sync_daemon(self._collect, run)
                self._collected = True
                output = self._output or ToolOutput(status=RunStatus.ERROR)
            return await anyio.to_thread.run_sync(
                self._finish, output, (run.started_at, run.ended_at)
            )
        except anyio.get_cancelled_exc_class():
            if self._started:  # a cancel before the start handshake has no output to give
                with anyio.CancelScope(shield=True):
                    await self._deliver_partial()
            raise

    def fault_output(self, reason: str) -> ToolOutput:
        """Return an ERROR output for a job that cannot start, with the reason in stderr."""
        for rel in (_STDOUT, _STDERR):
            os.close(open_log(self._workspace / rel, truncate=True))
        note_in_log(self._stderr, reason)
        return self._finish(ToolOutput(status=RunStatus.ERROR), None)

    async def _on_started(self, started: Job.Started) -> None:
        """Mark the start, then commit it through ``job.on_started`` with the job id."""
        self._started = True
        commit = self._job.on_started
        if commit is not None:
            await commit(started.model_copy(update={"job_id": self._job.id}))

    async def _before_kill(self, provisional: CompletedRun) -> None:
        """Collect the partial output in a daemon thread the supervisor can stop waiting on."""
        self._provisional = provisional
        self._output = await run_sync_daemon(self._collect, provisional)

    async def _deliver_partial(self) -> None:
        """Hand the best output to ``job.on_partial`` after a cancel (the caller shields)."""
        run = self._run
        if run is not None and run.status not in _STOPPED and not self._collected:
            # The tool exited on its own, but the cancel came before its collect ended.
            cancelled = run.model_copy(update={"status": RunStatus.CANCELLED})
            self._provisional = cancelled
            with anyio.move_on_after(supervisor.BEFORE_KILL_TIMEOUT_S):
                self._output = await run_sync_daemon(self._collect, cancelled)
        output = await anyio.to_thread.run_sync(self._stopped_output, RunStatus.CANCELLED)
        timed = run or self._provisional
        times = None if timed is None else (timed.started_at, timed.ended_at)
        try:
            output = await anyio.to_thread.run_sync(self._finish, output, times)
        except Exception as exc:  # an upload failure must not replace the cancellation
            note = f"uploads failed: {type(exc).__name__}"
            await anyio.to_thread.run_sync(note_in_log, self._stderr, note)
            update: dict[str, Any] = {"files": []}  # never hand over refs that were not stored
            if times is not None:
                update["started_at"], update["ended_at"] = times
            output = output.model_copy(update=update)
        if self._job.on_partial is not None:
            await self._job.on_partial(output)

    def _collect(self, run: CompletedRun) -> ToolOutput | None:
        """Collect the output; on an exception note the traceback and return None."""
        try:
            if self._staged.tool is not None and self._staged.request is not None:
                return self._staged.tool.collect(self._staged.request, self._workspace, run)
            return command_output_from_result(self._workspace, run)
        except Exception:
            note_in_log(run.stderr_path, "collect failed: " + traceback.format_exc())
            return None

    def _stopped_output(self, status: RunStatus) -> ToolOutput:
        """Stamp the stop on the kept output: the stop status and ``partial=True``."""
        if self._output is None:
            if self._provisional is not None:  # a collect was tried and gave nothing
                note_in_log(self._stderr, "no partial output: collect failed or timed out")
            return ToolOutput(status=status)
        return self._output.model_copy(update={"status": status, "partial": True})

    def _read_log(self, rel: str) -> bytes:
        """Read the last ``max_log_bytes`` of a regular log; anything else is empty."""
        try:
            fd = _open_regular(self._workspace, rel)
        except ValueError:
            note_in_log(self._stderr, f"{rel} is not a regular file; its log is uploaded empty")
            return b""
        try:
            size = os.fstat(fd).st_size
            if size > self._max_log:
                dropped = size - self._max_log
                if rel == _STDERR:  # the note lands in this log and pushes out its own length
                    for _ in range(3):
                        line = f"swarm-arena: {_truncated(rel, dropped)}\n"  # note_in_log's form
                        dropped = size + len(line.encode()) - self._max_log
                note_in_log(self._stderr, _truncated(rel, dropped))
                size = os.fstat(fd).st_size
            return _read_tail(fd, size, self._max_log)
        finally:
            os.close(fd)

    def _check_output_sizes(self, files: Sequence[FileRef]) -> None:
        """Raise ValueError naming the declared output whose size crosses the run's cap."""
        total = 0
        for ref in files:
            try:
                fd = _open_regular(self._workspace, ref.path)
            except ValueError:
                continue  # collect_output_files names what is wrong with it
            try:
                total += os.fstat(fd).st_size
            finally:
                os.close(fd)
            if total > self._max_output:
                raise ValueError(f"{ref.path} crosses max_output_bytes")

    def _finish(self, output: ToolOutput, times: tuple[datetime, datetime] | None) -> ToolOutput:
        """Upload declared files, then both logs, one at a time; set ``logs_ref`` and times."""
        update: dict[str, Any] = {}
        try:
            self._check_output_sizes(output.files)
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

    For each job, ``submit`` waits for a slot in the resource pool, creates a
    workspace ``<workspace_root>/<id>-<random>``, stages ``job.stage`` into it,
    and runs the command through the launcher (``python -I -m
    swarm_arena._launch``) under a process-group supervisor: the command starts
    only after ``job.on_started`` returns. In the workspace name, every
    character of the job id outside ``[A-Za-z0-9._-]`` becomes ``_``, and the
    id is cut to 64 characters. A leading ``"python"`` in the command becomes
    ``Command.python`` or ``sys.executable``. The environment is the allowlist
    of ``build_env``; ``Job.timeout_s`` and ``Job.grace_s`` replace the
    command's own values. stdout and stderr go to ``swarm/stdout.log`` and
    ``swarm/stderr.log`` in the workspace; the last ``max_log_bytes`` of each
    are uploaded to ``job.artifact_root``, and ``logs_ref`` is
    ``"sha256:<stdout>:<stderr>"``.

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
    ``Tool.collect`` for the partial output, then SIGKILL. A blocked collect
    delays the SIGKILL by at most ``BEFORE_KILL_TIMEOUT_S`` after the grace
    period: at that deadline the supervisor kills the group, and the collect is
    abandoned and not tried again. Collect runs in a daemon thread, so an
    abandoned one never keeps the interpreter from exiting; it may still write
    into the workspace, even into a kept workspace after ``submit`` returned.
    The cancel-path output goes to ``job.on_partial``. Every diagnostic is
    appended to the stderr log only while it is a regular file, so a tool that
    replaces a log with a FIFO or a symlink cannot stall the executor.

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

    class Config(_ConfigInputModel):
        """
        Configure local concurrency, workspaces, resource limits, and upload caps.

        Validation errors name fields and error types but never echo a value.

        Parameters
        ----------
        max_concurrent : int or None, default=None
            Most jobs that run at once (at least 1); None selects
            ``os.cpu_count()``.
        workspace_root : str or None, default=None
            Directory that holds the job workspaces; None selects
            ``tempfile.gettempdir()/swarm-arena-<uid>``. It and every parent
            must be owned by this user or root and not writable by group or
            others unless sticky (like ``/tmp``).
        keep_workspaces : bool, default=False
            Keep each workspace after its job instead of removing it.
        rlimits : bool, default=True
            Apply resource limits: a soft open-file limit of 4096, and
            ``Resources.memory_gib`` as the address-space limit
            (``RLIMIT_AS``) where the OS enforces it (Linux). macOS refuses
            ``RLIMIT_AS``, so there the memory limit is not applied and the
            launcher notes this in the stderr log.
        max_log_bytes : int, default=8 * 2**20
            Most bytes uploaded per log. A longer log keeps its last
            ``max_log_bytes`` (the tail holds the error), with a
            ``log truncated`` note in the stderr log.
        max_output_bytes : int, default=256 * 2**20
            Most bytes of all declared output files of one run. A run that
            crosses it gets status ERROR and no files, with a note naming the
            file that crossed it.

        Examples
        --------
        >>> from swarm_arena.executor import LocalExecutor
        >>> LocalExecutor.Config().max_log_bytes == 8 * 2**20
        True
        """

        model_config = ConfigDict(extra="forbid", frozen=True)

        max_concurrent: Annotated[int, Field(ge=1)] | None = None
        workspace_root: str | None = None
        keep_workspaces: bool = False
        rlimits: bool = True
        max_log_bytes: Annotated[int, Field(ge=0)] = 8 * 2**20
        max_output_bytes: Annotated[int, Field(ge=0)] = 256 * 2**20

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
        >>> import anyio
        >>> from swarm_arena.executor import LocalExecutor
        >>> anyio.run(LocalExecutor().capacity).cpus >= 1
        True
        """
        return Resources(cpus=float(os.cpu_count() or 1))

    async def aclose(self) -> None:
        """
        Release nothing; a local executor holds no resource between jobs.

        Examples
        --------
        >>> import anyio
        >>> from swarm_arena.executor import LocalExecutor
        >>> anyio.run(LocalExecutor().aclose)
        """

    async def submit(self, job: Job) -> ToolOutput:
        """
        Stage and run a job, and return its output with uploaded files and logs.

        A job that exits on its own is collected with ``Tool.collect`` (tool
        jobs) or from its raw ``swarm/result.json`` (command jobs); an exception
        in collect gives status ERROR, with the traceback in the stderr log.
        After a timeout, the output collected before SIGKILL is returned with
        status TIMEOUT and ``partial=True``. On a cancel after the start
        handshake, ``job.on_partial`` gets the best output there is (status
        CANCELLED, ``partial=True`` when a collect gave one), also when the tool
        had already exited and its collect was still running. A command
        ``cwd`` that is not a directory in the workspace gives status ERROR
        and the command never starts. A declared output file that is missing,
        unsafe, or over ``max_output_bytes`` gives status ERROR with no files.
        Uploaded file references are named after the file's basename, as
        ``collect_output_files`` makes them. The workspace is removed unless
        ``keep_workspaces`` is set; a failed removal is logged as a warning and
        never replaces the output or the cancellation.

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
            starts), its stage is invalid, its tool cannot be built, or its
            secrets file cannot be read.
        RuntimeError
            If the workspace root cannot be created or is not safe.
        swarm_arena._errors.StorageError
            If a stage artifact is missing or corrupt (``NotFound``), or the
            artifact store fails while the logs are uploaded.
        OSError
            If the system cannot create the workspace or start the launcher.
        Exception
            Whatever ``job.on_started`` or ``Tool.command`` raises; the command
            then never runs.

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
            sup = await self._get_supervisor()
            with anyio.CancelScope(shield=True):  # a cancel here must not lose the directory
                workspace, artifacts = await anyio.to_thread.run_sync(
                    _open_workspace, self.config.workspace_root, job
                )
            try:
                staged = await anyio.to_thread.run_sync(_stage, job, workspace, artifacts)
                job_run = _JobRun(job, staged, workspace, artifacts, self.config)
                if staged.fault is not None:
                    return await anyio.to_thread.run_sync(job_run.fault_output, staged.fault)
                return await job_run.run(sup, self._rlimits(job))
            finally:
                if not self.config.keep_workspaces:
                    with anyio.CancelScope(shield=True):
                        await anyio.to_thread.run_sync(_remove_workspace, workspace)

    async def confirm_dead(self, started: Job.Started) -> bool | None:
        """
        Report whether a recorded process group on this host has stopped.

        Members that were killed but not yet reaped (zombies) count as dead.
        When liveness cannot be read, the group counts as alive.

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
        return not await anyio.to_thread.run_sync(group_has_live_member, pgid)

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
        sup = self._supervisor
        if sup is None:
            sup = ProcessSupervisor(host_id=await self._get_host_id())
            self._supervisor = sup
        return sup

    def _rlimits(self, job: Job) -> RLimits | None:
        """Return the job's resource limits, or None when they are turned off."""
        if not self.config.rlimits:
            return None
        memory = job.resources.memory_gib
        return RLimits(memory_bytes=int(memory * 2**30) if memory else None)
