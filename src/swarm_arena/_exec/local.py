"""Run tool and command jobs on this host with bounded resources and private workspaces."""

import logging
import os
from typing import Annotated, Any, Final, cast

import anyio
from pydantic import ConfigDict, Field

from swarm_arena._errors import _core_error
from swarm_arena._exec import procinfo
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.local_run import _NOT_AN_OUTPUT as _NOT_AN_OUTPUT
from swarm_arena._exec.local_run import _JobRun
from swarm_arena._exec.local_stage import _Preparation
from swarm_arena._exec.local_workspace import _open_workspace, _remove_workspace
from swarm_arena._exec.pool import ResourcePool
from swarm_arena._exec.procinfo import group_has_live_member, host_id
from swarm_arena._exec.supervisor import ProcessSupervisor, RLimits
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import _ConfigInputModel, config_or_fields
from swarm_arena._model.tooling import ToolOutput

_LOG: Final = logging.getLogger(__name__)


class LocalExecutor(Executor):
    """
    Run jobs on this POSIX host, each in a fresh workspace and its own process group.

    For each job, ``submit`` waits for a slot in the resource pool, creates a
    workspace ``<workspace_root>/<id>-<random>``, stages ``job.stage`` into it,
    and runs the command through a private launcher process (``python -I``)
    under a process-group supervisor: the command starts only after
    ``job.on_started`` returns. In the workspace name, every
    character of the job id outside ``[A-Za-z0-9._-]`` becomes ``_``, and the
    id is cut to 64 characters. A leading ``"python"`` in the command becomes
    ``Command.python`` or ``sys.executable``. The environment is the allowlist
    of ``build_env``. The smaller of the job and Command limits applies:
    ``None`` means no timeout, and grace is the minimum of both values.
    stdout and stderr go to ``swarm/stdout.log`` and
    ``swarm/stderr.log`` in the workspace; the last ``max_log_bytes`` of each
    are uploaded to ``job.artifact_root``, and ``logs_ref`` is
    ``"sha256:<stdout>:<stderr>"``.

    Construction does no I/O; the pool and the host id are made on the first
    ``submit``.

    Tool construction and command preparation run in a daemon thread.
    Cancellation stops waiting and releases the resource slot. Preparation
    still running at cancellation retains its workspace, with a warning naming
    the path, even if it later returns. Its result is discarded: the executor
    never launches that command or invokes lifecycle callbacks for it.

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
    delays the SIGKILL by at most 30 seconds after the grace
    period: at that deadline the supervisor kills the group, and the collect is
    abandoned and not tried again. Collect runs in a daemon thread, so an
    abandoned one never keeps the interpreter from exiting; it may still write
    into the workspace, even into a kept workspace after ``submit`` returned.
    After normal exit, collect has the same 30-second bound. Cancellation
    rejoins that one collect until its original deadline, keeping its result
    when available; it never starts an overlapping collect. A post-exit
    collect that exceeds the bound gives ERROR and an empty result.
    Artifact uploads are awaited one object at a time under caller supervision.
    Cancellation abandons the current upload and starts no later upload. If
    cancellation interrupts the process, collection, or an upload, collected
    data is delivered without file or log references through ``job.on_partial``;
    no partial-output uploads start after cancellation. The callback is bounded to
    30 seconds. Every diagnostic is
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
            raise _core_error(RuntimeError, "LocalExecutor needs a POSIX system")

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
        jobs) or from its raw ``swarm/result.json`` (command jobs), bounded
        to 30 seconds after collection starts. Cancellation waits for that
        same in-flight collect until its original deadline. A collect
        that raises an ordinary exception or ``SystemExit``, or returns anything
        but a ``ToolOutput`` that passes validation again (``model_copy`` and
        ``model_construct`` skip it), is a
        collect failure: status ERROR after a normal exit, with the reason in
        the stderr log. ``SystemExit`` from ``Tool.command`` also gives ERROR
        without starting the command. KeyboardInterrupt, cancellation,
        exception groups, and Fenced propagate unchanged for the engine or
        direct caller to classify. A stopped run's output has the stop status: after a
        timeout it is the output collected before SIGKILL, with
        ``partial=True`` when that collect returned a valid ``ToolOutput``
        that says it is partial or keeps data or files, also when those
        files are dropped later (over ``max_output_bytes``, or a failed
        upload). A valid but empty output that does not say partial, such
        as a script tool's timeout without ``swarm/result.json``, gives
        ``partial=False``. When collect failed, timed out, or gave an invalid
        result, the output is empty with ``partial=False``, and a note goes to
        the stderr log. On a
        cancel after the start handshake, ``job.on_partial`` gets the best
        collected data (status CANCELLED, no file or log references), also when the tool had already
        exited and its collect was still running. A command
        ``cwd`` that is not a directory in the workspace gives status ERROR
        and the command never starts. A declared output file that is missing,
        unsafe, or over ``max_output_bytes`` gives status ERROR with no files.
        Uploaded file references are named after the file's basename, as
        ``collect_output_files`` makes them. The workspace is removed unless
        ``keep_workspaces`` is set or preparation is still running when cancelled.
        In the latter case, a warning identifies the retained workspace; it is
        never removed automatically, even after a late preparation return.
        Cancellation releases the slot without waiting for user construction
        or command preparation, and their late results cannot start a command
        or invoke ``on_started``/``on_partial``. A failed removal is logged as a
        warning and never replaces the output or the cancellation.

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
            secrets file cannot be read, or its memory limit overflows the
            conversion to bytes.
        RuntimeError
            If the workspace root cannot be created or is not safe.
        StorageError
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
            preparation = _Preparation(job, workspace, artifacts)
            try:
                staged = await run_sync_daemon(preparation.run)
                # Deliver a pending cancel before any launch or lifecycle callback.
                await anyio.lowlevel.checkpoint()
                job_run = _JobRun(job, staged, workspace, artifacts, self.config)
                if staged.fault is not None:
                    return await job_run.fault_output(staged.fault)
                return await job_run.run(sup, self._rlimits(job))
            finally:
                if not preparation.finished.is_set():
                    _LOG.warning("retaining workspace %s: preparation is still running", workspace)
                elif not self.config.keep_workspaces:
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
        >>> import anyio
        >>> from swarm_arena.executor import Job, LocalExecutor
        >>> record = Job.Started(kind="process", job_id="c-1")
        >>> anyio.run(LocalExecutor().confirm_dead, record) is None
        True
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
        >>> import anyio
        >>> from swarm_arena.executor import Job, LocalExecutor
        >>> record = Job.Started(kind="process", job_id="c-1")
        >>> anyio.run(LocalExecutor().kill_orphan, record)
        False
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
        try:
            return RLimits(memory_bytes=int(memory * 2**30) if memory else None)
        except OverflowError:
            raise ValueError("memory_gib exceeds resource limit range") from None
