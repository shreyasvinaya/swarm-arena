"""Run one command in its own process group with a start handshake and a bounded stop."""

import contextlib
import functools
import os
import signal
import subprocess
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import anyio

from swarm_arena._errors import Fenced, _exception_type_name
from swarm_arena._exec.executor import Job
from swarm_arena._exec.procinfo import ExitWatch, group_has_live_member, process_start_time
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.tooling import CompletedRun, RunStatus
from swarm_arena._tooling.result_file import note_in_log, open_log

BEFORE_KILL_TIMEOUT_S: Final = 30.0
"""Seconds ``before_kill`` may run before the group is killed anyway; read at call time."""

_GO: Final = b"go"
_POLL_S: Final = 0.05
_OOM_RETURNCODES: Final = frozenset({-signal.SIGKILL, -signal.SIGSEGV})


class RLimits(FrozenModel):
    """
    Describe the resource limits the launcher applies before it execs the command.

    Parameters
    ----------
    memory_bytes : int or None, default=None
        Address-space limit (``RLIMIT_AS``), from ``Resources.memory_gib``.
        Setting it also lets a SIGKILL or SIGSEGV we did not send count as OOM.
    cpu_seconds : int or None, default=None
        CPU-time limit (``RLIMIT_CPU``).
    nofile : int or None, default=4096
        Soft open-file limit (``RLIMIT_NOFILE``), capped at the hard limit.

    Examples
    --------
    >>> RLimits(memory_bytes=2**30).nofile
    4096
    """

    memory_bytes: int | None = None
    cpu_seconds: int | None = None
    nofile: int | None = 4096


def _launcher_args(rlimits: RLimits | None) -> list[str]:
    """Return the launcher options for the limits that are set."""
    if rlimits is None:
        return []
    pairs = (
        ("--rlimit-as", rlimits.memory_bytes),
        ("--rlimit-cpu", rlimits.cpu_seconds),
        ("--rlimit-nofile", rlimits.nofile),
    )
    return [f"{flag}={value}" for flag, value in pairs if value is not None]


def launch_argv(argv: Sequence[str], *, go_fd: int, rlimits: RLimits | None) -> list[str]:
    """Build the shared isolated launcher command without spawning a process.

    Parameters
    ----------
    argv : Sequence[str]
        Tool executable and arguments.
    go_fd : int
        Inherited descriptor carrying the start handshake.
    rlimits : RLimits or None
        Child resource limits, or no overrides.

    Returns
    -------
    list[str]
        Launcher invocation with the original resource-option spelling.

    Examples
    --------
    >>> launch_argv(["true"], go_fd=5, rlimits=None)[-4:]
    ['--go-fd', '5', '--', 'true']
    """
    return [
        sys.executable,
        "-I",
        "-m",
        "swarm_arena._launch",
        "--go-fd",
        str(go_fd),
        *_launcher_args(rlimits),
        "--",
        *argv,
    ]


def _open_logs(stdout: Path, stderr: Path, fds: list[int]) -> None:
    """Open both logs with ``open_log`` (checked, then truncated), adding each fd to ``fds``."""
    for path in (stdout, stderr):
        fds.append(open_log(path, truncate=True))


def _spawn(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout: int,
    stderr: int,
    go_fd: int,
    procs: list[subprocess.Popen[bytes]],
) -> None:
    """Start the launcher in a new session and add it to ``procs``; it is never reaped here."""
    procs.append(
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            cwd=cwd,
            env=env,
            start_new_session=True,
            pass_fds=(go_fd,),
        )
    )


def _close(fds: list[int], fd: int) -> None:
    """Close ``fd`` once and forget it, so a reused descriptor number is never closed twice."""
    if fd in fds:
        fds.remove(fd)
        os.close(fd)


def _signal_group(pgid: int, sig: int) -> None:
    """
    Signal a process group, ignoring a group that is gone or holds only zombies.

    ``ProcessLookupError`` means no member is left. macOS gives
    ``PermissionError`` for a group whose members are all zombies.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _exit_status(returncode: int, rlimits: RLimits | None) -> RunStatus:
    """Map the exit code of a process that ended on its own to a run status."""
    if returncode == 0:
        return RunStatus.OK
    memory_limited = rlimits is not None and rlimits.memory_bytes is not None
    if memory_limited and returncode in _OOM_RETURNCODES:
        return RunStatus.OOM
    return RunStatus.ERROR


class _Group:
    """
    A launcher's process group, anchored by its unreaped leader.

    The leader is a child of this process. Until it is reaped it holds its pid,
    even as a zombie, so no other process can get that number or lead a group
    with it, and every signal to ``pgid`` reaches only this group. So every
    path signals first and reaps last.
    """

    def __init__(self, proc: subprocess.Popen[bytes], watch: ExitWatch) -> None:
        self.proc = proc
        self.watch = watch
        self.pgid = proc.pid  # start_new_session: the leader's pid is the group id

    def signal(self, sig: int) -> None:
        """Signal the whole group; only call this before ``_kill_and_reap``."""
        _signal_group(self.pgid, sig)


async def _kill_and_reap(proc: subprocess.Popen[bytes], watch: ExitWatch | None = None) -> int:
    """Signal the anchored group before reaping its leader (caller shields)."""
    try:
        _signal_group(proc.pid, signal.SIGKILL)
    finally:
        if watch is not None:
            await watch.wait()
        returncode = await anyio.to_thread.run_sync(proc.wait)
    return returncode


async def _collect(
    before_kill: Callable[[CompletedRun], Awaitable[None]],
    provisional: CompletedRun,
    group: _Group,
) -> str | None:
    """
    Await ``before_kill``; at the deadline SIGKILL the group from a sibling task.

    The SIGKILL does not wait for ``before_kill`` to honour its cancellation, so
    a collector stuck in a thread cannot keep the tool alive. Return the note
    to log (only the exception type on failure), or None on success. Any
    other ``BaseException`` from ``before_kill``, cancellation included, is
    re-raised as itself after the task group, never inside an exception group.
    """
    limit = BEFORE_KILL_TIMEOUT_S
    cancelled = anyio.get_cancelled_exc_class()
    timed_out = False
    failed: str | None = None
    raised: BaseException | None = None
    async with anyio.create_task_group() as tg:

        async def kill_at_deadline() -> None:
            nonlocal timed_out
            await anyio.sleep(limit)
            timed_out = True
            group.signal(signal.SIGKILL)
            tg.cancel_scope.cancel()

        tg.start_soon(kill_at_deadline)
        try:
            await before_kill(provisional)
        except (Fenced, BaseExceptionGroup) as exc:
            raised = exc
        except Exception as exc:
            failed = _exception_type_name(exc)  # the type only: a message can carry secret values
        except BaseException as exc:
            if isinstance(exc, cancelled) and tg.cancel_scope.cancel_called:
                raise  # the deadline's own cancellation: the task group absorbs it
            raised = exc
        tg.cancel_scope.cancel()
    if raised is not None:
        raise raised
    if timed_out:
        return f"before_kill timed out after {limit:g} s"
    if failed is not None:
        return f"before_kill failed: {failed}"
    return None


class ProcessSupervisor:
    """
    Start a command through the launcher in its own process group and stop it safely.

    Parameters
    ----------
    host_id : str
        Host identity written into every ``Job.Started`` (see ``host_id()``).

    Examples
    --------
    >>> import os, sys
    >>> from pathlib import Path
    >>> run = await ProcessSupervisor(host_id="h:b").run(  # doctest: +SKIP
    ...     [sys.executable, "-c", "print('hi')"], cwd=Path("."), env=dict(os.environ),
    ...     stdout=Path("stdout.log"), stderr=Path("stderr.log"), timeout_s=10.0,
    ...     grace_s=1.0, rlimits=None, on_started=None, before_kill=None)
    """

    def __init__(self, *, host_id: str) -> None:
        self._host_id = host_id

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout: Path,
        stderr: Path,
        timeout_s: float | None,
        grace_s: float,
        rlimits: RLimits | None,
        on_started: Callable[[Job.Started], Awaitable[None]] | None,
        before_kill: Callable[[CompletedRun], Awaitable[None]] | None,
    ) -> CompletedRun:
        """
        Run ``argv`` to completion, timeout, or cancellation.

        The command starts as ``python -I -m swarm_arena._launch`` in a new
        session, so its pid is its process group id. The launcher stays inert
        until ``on_started`` returns; only then is "go" written and the command
        exec'd, so every command that runs has a committed start. If
        ``on_started`` raises or is cancelled, the group is killed and reaped
        and the exception propagates; the command never runs.

        On timeout or cancellation the stop sequence runs shielded: SIGTERM to
        the group; a wait until ``grace_s`` passes or no live member is left;
        ``before_kill`` with a provisional result; then, in a ``finally`` that
        always runs, SIGKILL to the group and the reap. At
        ``BEFORE_KILL_TIMEOUT_S`` a sibling task sends the SIGKILL and cancels
        ``before_kill``, so a stuck collector cannot keep the tool alive. A
        ``before_kill`` failure (its exception type only) or timeout is noted
        in the stderr log after the kill. After a normal exit, any members left
        in the group are killed too.

        The leader is reaped only after the last signal to its group. Until
        then it holds its pid, even as a zombie, so the group id cannot be
        reused and every group signal reaches only this run's processes.

        Parameters
        ----------
        argv : Sequence[str]
            Command to exec; a bare ``argv[0]`` is looked up on ``env["PATH"]``.
        cwd : Path
            Working directory of the command.
        env : Mapping[str, str]
            Complete environment of the command.
        stdout : Path
            File that receives standard output. It is opened like
            ``open_log``: no component may be a symlink, and an existing
            file must be a regular file with one link; it is then truncated.
            Pass a resolved path (``/tmp`` and ``/var`` are symlinks on macOS).
        stderr : Path
            File that receives standard error and the supervisor's notes,
            opened like ``stdout``.
        timeout_s : float or None
            Seconds from "go" before the stop sequence runs, or None for none.
        grace_s : float
            Seconds the group gets between SIGTERM and SIGKILL.
        rlimits : RLimits or None
            Limits applied by the launcher, or None for none.
        on_started : Callable[[Job.Started], Awaitable[None]] or None
            Commit the start event (``job_id`` is ``""``; the caller fills it).
        before_kill : Callable[[CompletedRun], Awaitable[None]] or None
            Collect partial output before SIGKILL. It receives a provisional
            result with ``returncode=None`` and status TIMEOUT or CANCELLED.
            It must be cancellable. Blocking work should run in a daemon
            thread that the loop awaits (``LocalExecutor`` uses
            ``swarm_arena._exec.threads.run_sync_daemon``), so cancelling it
            returns at once. A collector that ignores cancellation does not
            delay the SIGKILL, but ``run`` returns only when it returns. Its
            ``Exception`` becomes a note in the stderr log (the type only);
            any other ``BaseException`` leaves ``run`` as itself.

        Returns
        -------
        CompletedRun
            OK for exit code 0; ERROR for other codes and signals; OOM for a
            SIGKILL or SIGSEGV we did not send while ``rlimits.memory_bytes``
            is set; TIMEOUT after the stop sequence. On cancellation the stop
            sequence runs and the cancellation propagates instead.

        Raises
        ------
        OSError
            When a log file cannot be opened safely (``cannot open log file
            safely``) or the launcher cannot start, for example
            ``FileNotFoundError`` for a missing ``cwd``.

        Examples
        --------
        >>> run = await supervisor.run(argv, cwd=cwd, env=env,  # doctest: +SKIP
        ...     stdout=out, stderr=err, timeout_s=60.0, grace_s=5.0,
        ...     rlimits=RLimits(), on_started=commit, before_kill=collect)
        """
        fds: list[int] = []
        procs: list[subprocess.Popen[bytes]] = []
        try:
            await anyio.to_thread.run_sync(_open_logs, stdout, stderr, fds)
            f_out, f_err = fds
            r, w = os.pipe()
            fds += [r, w]
            command = launch_argv(argv, go_fd=r, rlimits=rlimits)
            spawn = functools.partial(
                _spawn,
                command,
                cwd=cwd,
                env=dict(env),
                stdout=f_out,
                stderr=f_err,
                go_fd=r,
                procs=procs,
            )
            await anyio.to_thread.run_sync(spawn)
            [proc] = procs
            for fd in (r, f_out, f_err):
                _close(fds, fd)
            with contextlib.closing(ExitWatch(proc.pid)) as watch:
                return await self._supervise(
                    _Group(proc, watch),
                    w,
                    fds,
                    stdout=stdout,
                    stderr=stderr,
                    timeout_s=timeout_s,
                    grace_s=grace_s,
                    rlimits=rlimits,
                    on_started=on_started,
                    before_kill=before_kill,
                )
        finally:
            for fd in list(fds):
                _close(fds, fd)
            for proc in procs:
                if proc.returncode is None:  # started, but the watch could not be made
                    with anyio.CancelScope(shield=True):
                        await _kill_and_reap(proc)

    async def _supervise(
        self,
        group: _Group,
        w: int,
        fds: list[int],
        *,
        stdout: Path,
        stderr: Path,
        timeout_s: float | None,
        grace_s: float,
        rlimits: RLimits | None,
        on_started: Callable[[Job.Started], Awaitable[None]] | None,
        before_kill: Callable[[CompletedRun], Awaitable[None]] | None,
    ) -> CompletedRun:
        """Run the handshake, wait, and stop sequence for a started launcher."""
        try:
            start_time = await anyio.to_thread.run_sync(process_start_time, group.pgid)
            started = Job.Started(
                kind="process",
                job_id="",
                host_id=self._host_id,
                pid=group.pgid,
                pgid=group.pgid,
                start_time=start_time,
            )
            if on_started is not None:
                await on_started(started)
        except BaseException:
            with anyio.CancelScope(shield=True):
                try:
                    await _kill_and_reap(group.proc, group.watch)
                finally:
                    _close(fds, w)
            raise

        started_at = utc_now()
        stopped: RunStatus | None = None
        try:
            try:
                os.write(w, _GO)
            except BrokenPipeError:
                pass  # the launcher already exited; the reap gives its code, so the status is ERROR
            finally:
                _close(fds, w)
            with anyio.move_on_after(timeout_s) as deadline:
                await group.watch.wait()
            if deadline.cancelled_caught:
                stopped = RunStatus.TIMEOUT
        except anyio.get_cancelled_exc_class():
            stopped = RunStatus.CANCELLED
            raise
        finally:
            with anyio.CancelScope(shield=True):
                if stopped is None:
                    returncode = await _kill_and_reap(group.proc, group.watch)
                else:
                    returncode = await self._stop(
                        group,
                        status=stopped,
                        grace_s=grace_s,
                        stdout=stdout,
                        stderr=stderr,
                        started_at=started_at,
                        before_kill=before_kill,
                    )

        return CompletedRun(
            returncode=returncode,
            status=stopped if stopped is not None else _exit_status(returncode, rlimits),
            stdout_path=stdout,
            stderr_path=stderr,
            started_at=started_at,
            ended_at=utc_now(),
        )

    @staticmethod
    async def _stop(
        group: _Group,
        *,
        status: RunStatus,
        grace_s: float,
        stdout: Path,
        stderr: Path,
        started_at: datetime,
        before_kill: Callable[[CompletedRun], Awaitable[None]] | None,
    ) -> int:
        """Run SIGTERM, grace, ``before_kill``, then SIGKILL in a finally (caller shields)."""
        note: str | None = None
        try:
            group.signal(signal.SIGTERM)
            with anyio.move_on_after(grace_s):
                await group.watch.wait()  # the leader exits but stays unreaped
                # No event exists for "the group has no live member", so poll.
                while await anyio.to_thread.run_sync(  # noqa: ASYNC110
                    group_has_live_member, group.pgid, abandon_on_cancel=True
                ):
                    await anyio.sleep(_POLL_S)
            if before_kill is not None:
                provisional = CompletedRun(
                    returncode=None,
                    status=status,
                    stdout_path=stdout,
                    stderr_path=stderr,
                    started_at=started_at,
                    ended_at=utc_now(),
                )
                note = await _collect(before_kill, provisional, group)
        finally:
            returncode = await _kill_and_reap(group.proc, group.watch)
        if note is not None:
            await anyio.to_thread.run_sync(note_in_log, stderr, note)
        return returncode
