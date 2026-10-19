"""Run one command in its own process group with a start handshake and a bounded stop."""

import contextlib
import os
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from subprocess import DEVNULL
from typing import Final

import anyio
from anyio.abc import Process

from swarm_arena._exec.executor import Job
from swarm_arena._exec.procinfo import group_alive, process_start_time
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.tooling import CompletedRun, RunStatus
from swarm_arena._tooling.result_file import note_in_log

BEFORE_KILL_TIMEOUT_S: Final = 30.0
"""Seconds ``before_kill`` may run before the group is killed anyway; read at call time."""

_GO: Final = b"go"
_POLL_S: Final = 0.05
_OOM_RETURNCODES: Final = frozenset({-signal.SIGKILL, -signal.SIGSEGV})
_LOG_FLAGS: Final = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW  # never through a symlink


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


def _open_logs(stdout: Path, stderr: Path, fds: list[int]) -> None:
    """Create or truncate both log files, appending each descriptor to ``fds`` as it opens."""
    for path in (stdout, stderr):
        fds.append(os.open(path, _LOG_FLAGS, 0o644))


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


async def _collect(
    before_kill: Callable[[CompletedRun], Awaitable[None]], provisional: CompletedRun
) -> str | None:
    """Await ``before_kill`` under its deadline; return the note to log, or None on success."""
    limit = BEFORE_KILL_TIMEOUT_S
    with anyio.move_on_after(limit) as scope:
        try:
            await before_kill(provisional)
        except Exception as exc:
            return f"before_kill failed: {type(exc).__name__}: {exc}"
    if scope.cancelled_caught:
        return f"before_kill timed out after {limit:g} s"
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
        the group; a wait until ``grace_s`` passes or the whole group is gone;
        ``before_kill`` with a provisional result, bounded by
        ``BEFORE_KILL_TIMEOUT_S``; then, in a ``finally`` that always runs,
        SIGKILL to the group, reap, and close. A ``before_kill`` failure or
        timeout is noted in the stderr log only after the kill. After a normal
        exit, any members left in the group are killed too.

        Parameters
        ----------
        argv : Sequence[str]
            Command to exec; a bare ``argv[0]`` is looked up on ``env["PATH"]``.
        cwd : Path
            Working directory of the command.
        env : Mapping[str, str]
            Complete environment of the command.
        stdout : Path
            File that receives standard output; created or truncated.
        stderr : Path
            File that receives standard error and the supervisor's notes.
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
            When a log file cannot be opened or the launcher cannot start, for
            example ``FileNotFoundError`` for a missing ``cwd``.

        Examples
        --------
        >>> run = await supervisor.run(argv, cwd=cwd, env=env,  # doctest: +SKIP
        ...     stdout=out, stderr=err, timeout_s=60.0, grace_s=5.0,
        ...     rlimits=RLimits(), on_started=commit, before_kill=collect)
        """
        fds: list[int] = []
        try:
            await anyio.to_thread.run_sync(_open_logs, stdout, stderr, fds)
            f_out, f_err = fds
            r, w = os.pipe()
            fds += [r, w]
            command = [
                sys.executable,
                "-I",
                "-m",
                "swarm_arena._launch",
                "--go-fd",
                str(r),
                *_launcher_args(rlimits),
                "--",
                *argv,
            ]
            proc = await anyio.open_process(
                command,
                stdin=DEVNULL,
                stdout=f_out,
                stderr=f_err,
                cwd=cwd,
                env=dict(env),
                start_new_session=True,
                pass_fds=(r,),
            )
            for fd in (r, f_out, f_err):
                _close(fds, fd)
            return await self._supervise(
                proc,
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

    async def _supervise(
        self,
        proc: Process,
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
        pgid = proc.pid
        try:
            start_time = await anyio.to_thread.run_sync(process_start_time, proc.pid)
            started = Job.Started(
                kind="process",
                job_id="",
                host_id=self._host_id,
                pid=proc.pid,
                pgid=pgid,
                start_time=start_time,
            )
            if on_started is not None:
                await on_started(started)
        except BaseException:
            with anyio.CancelScope(shield=True):
                try:
                    _signal_group(pgid, signal.SIGKILL)
                    _close(fds, w)
                finally:
                    await proc.wait()
                    await proc.aclose()
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
                await proc.wait()
            if deadline.cancelled_caught:
                stopped = RunStatus.TIMEOUT
        except anyio.get_cancelled_exc_class():
            stopped = RunStatus.CANCELLED
            raise
        finally:
            with anyio.CancelScope(shield=True):
                if stopped is None:
                    returncode = await self._reap(proc, pgid)
                else:
                    returncode = await self._stop(
                        proc,
                        pgid,
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
    async def _reap(proc: Process, pgid: int) -> int:
        """Kill what is left of the group, reap, close, and return the code (caller shields)."""
        try:
            if proc.returncode is None or group_alive(pgid):
                _signal_group(pgid, signal.SIGKILL)
        finally:
            returncode = await proc.wait()
            await proc.aclose()
        return returncode

    @staticmethod
    async def _stop(
        proc: Process,
        pgid: int,
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
            _signal_group(pgid, signal.SIGTERM)
            with anyio.move_on_after(grace_s):
                await proc.wait()  # reap the leader, then wait for the rest of the group
                # No event exists for "the group is empty" (a pidfd covers one process), so poll.
                while group_alive(pgid):  # noqa: ASYNC110
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
                note = await _collect(before_kill, provisional)
        finally:
            try:
                _signal_group(pgid, signal.SIGKILL)
            finally:
                returncode = await proc.wait()
                await proc.aclose()
        if note is not None:
            await anyio.to_thread.run_sync(note_in_log, stderr, note)
        return returncode
