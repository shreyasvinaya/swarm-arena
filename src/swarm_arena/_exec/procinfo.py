"""Identify the host and check or kill process groups recorded in a start event."""

import contextlib
import os
import re
import select
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio

from swarm_arena._exec.executor import Job

_BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
_SYSCTL = "/usr/sbin/sysctl"
_UNKNOWN = "unknown"
# ps prints lstart in the local time of its locale; pin both so a resume on the
# same host reads the same string as the run that recorded it.
_PS_ENV = {"LC_ALL": "C", "TZ": "UTC"}
_PS = "ps"
_PROC = Path("/proc")
_DEAD_STATES = frozenset({b"Z", b"X", b"x"})  # zombie, dead
_GONE = (FileNotFoundError, ProcessLookupError)  # the process or thread exited
_EXIT_POLL_S = 0.05


def _boot_id() -> str:
    """Return an id that changes on every boot, or ``"unknown"``."""
    try:
        boot = _BOOT_ID_FILE.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        boot = ""
    if boot:
        return boot
    try:
        done = subprocess.run(
            [_SYSCTL, "-n", "kern.boottime"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return _UNKNOWN
    match = re.search(r"\bsec\s*=\s*(\d+)", done.stdout) if done.returncode == 0 else None
    return match.group(1) if match else _UNKNOWN


def host_id() -> str:
    """
    Return this host's identity for the current boot.

    Returns
    -------
    str
        ``"<hostname>:<boot>"``. ``boot`` is the Linux ``boot_id``, the
        ``sec`` value of macOS ``kern.boottime``, or ``"unknown"``.

    Examples
    --------
    >>> hostname, boot = host_id().rsplit(":", 1)
    >>> bool(hostname and boot)
    True
    """
    return f"{socket.gethostname()}:{_boot_id()}"


def process_start_time(pid: int) -> str | None:
    """
    Return the start time of a process, used to detect pid reuse.

    Parameters
    ----------
    pid : int
        Process id to inspect.

    Returns
    -------
    str or None
        Stripped ``ps -o lstart= -p <pid>`` output in the C locale and UTC,
        or None when ``ps`` fails or prints nothing (no such process).

    Examples
    --------
    >>> import os
    >>> process_start_time(os.getpid()) == process_start_time(os.getpid())
    True
    """
    try:
        done = subprocess.run(
            [_PS, "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env={**os.environ, **_PS_ENV},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = done.stdout.strip()
    return text if done.returncode == 0 and text else None


def group_alive(pgid: int) -> bool:
    """
    Report whether a process group still has a member.

    Parameters
    ----------
    pgid : int
        Process group id to probe with signal 0.

    Returns
    -------
    bool
        False when no member exists. True otherwise, also when the group
        exists but belongs to another user.

    Examples
    --------
    >>> import os
    >>> group_alive(os.getpgrp())
    True
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_has_live_member(pgid: int) -> bool:
    """
    Report whether a process group has a member that is not a zombie.

    Unlike ``group_alive``, zombie members do not count, so a group whose
    processes have all exited but are not yet reaped reads as empty. Members
    of any user count. When the caller holds the group's leader unreaped (see
    ``ExitWatch``), the zombie keeps the group id reserved, so the answer is
    about that group only; otherwise the id may name a newer group.

    Parameters
    ----------
    pgid : int
        Process group id to inspect.

    Returns
    -------
    bool
        True when a live member exists, and also when liveness cannot be read,
        so callers err toward waiting: ``/proc`` or ``ps`` unavailable, a
        ``/proc`` entry that exists but cannot be read, or ``/proc`` mounted
        with ``hidepid`` (another user's member may be unlisted).
        ``killpg(pgid, 0)`` decides when it can: ``ProcessLookupError`` is no
        member, and on macOS success is a live member (it refuses zombies).
        Linux answers signal 0 for zombies too, so it scans ``/proc``; a
        process is live while any thread in ``/proc/<pid>/task`` is not a
        zombie, even when its main thread has exited. A ``PermissionError``
        on macOS means only zombies or only another user's members are left,
        so ``ps`` tells them apart.

    Examples
    --------
    >>> import os
    >>> group_has_live_member(os.getpgrp())
    True
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        if sys.platform != "linux":
            return _live_in_ps(pgid)
    else:
        if sys.platform != "linux":
            return True
    return _live_in_proc(pgid, _PROC)


def _live_in_ps(pgid: int) -> bool:
    """Read ``ps`` for a process of group ``pgid`` that is not a zombie."""
    try:
        done = subprocess.run(
            [_PS, "-A", "-o", "pgid=", "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env={**os.environ, **_PS_ENV},
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if done.returncode != 0:
        return True
    group = str(pgid)
    for line in done.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == group and not fields[1].startswith("Z"):
            return True
    return False


class _CannotTell(Exception):
    """A ``/proc`` entry exists but cannot be read."""


def _read_proc(path: Path) -> bytes:
    """Read one ``/proc`` file."""
    return path.read_bytes()


def _list_proc(path: Path) -> list[str]:
    """List one ``/proc`` directory."""
    return os.listdir(path)


def _stat_fields(path: Path) -> list[bytes] | None:
    """Return the ``stat`` fields after the last ``)``, or None when the task is gone."""
    try:
        data = _read_proc(path)
    except _GONE:
        return None
    except OSError as exc:
        raise _CannotTell from exc
    return data[data.rfind(b")") + 1 :].split()  # comm may hold spaces and parentheses


def _has_live_thread(pid_dir: Path) -> bool:
    """Report whether any thread of a process is not a zombie (its main thread may be)."""
    try:
        tids = _list_proc(pid_dir / "task")
    except _GONE:
        return False
    except OSError as exc:
        raise _CannotTell from exc
    for tid in tids:
        fields = _stat_fields(pid_dir / "task" / tid / "stat")
        if fields and fields[0] not in _DEAD_STATES:
            return True
    return False


def _proc_hides_pids(proc: Path) -> bool:
    """Report whether ``proc`` may leave out other users' processes (``hidepid``)."""
    try:
        text = _read_proc(proc / "self" / "mountinfo").decode(errors="replace")
    except OSError:
        return True  # cannot tell
    hides: bool | None = None
    for line in text.splitlines():
        head, sep, tail = line.partition(" - ")
        fields, rest = head.split(), tail.split()
        is_proc = sep and len(fields) >= 5 and len(rest) >= 3 and rest[0] == "proc"
        if is_proc and fields[4] == str(proc):  # the last mount on that point is in use
            options = dict(option.partition("=")[::2] for option in rest[2].split(","))
            hides = options.get("hidepid", "0") not in {"0", "off"}
    return True if hides is None else hides


def _live_in_proc(pgid: int, proc: Path) -> bool:
    """
    Scan ``<proc>`` for a process of group ``pgid`` that has a live thread.

    A process counts as live while any of its threads is not a zombie: when
    its main thread exits early (``pthread_exit``), ``<pid>/stat`` reads Z but
    other threads in ``<pid>/task`` still run. The answer is True when the
    scan cannot tell: an entry that exists but cannot be read, or a ``proc``
    mounted with ``hidepid``, where another user's member may be unlisted.
    """
    try:
        if _proc_hides_pids(proc):
            return True
        group = str(pgid).encode()
        for pid in _list_proc(proc):
            if not pid.isdigit():
                continue
            fields = _stat_fields(proc / pid / "stat")
            if fields is None or len(fields) < 3 or fields[2] != group:
                continue
            if fields[0] not in _DEAD_STATES or _has_live_thread(proc / pid):
                return True
    except (_CannotTell, OSError):
        return True
    return False


def _has_exited(pid: int) -> bool:
    """Report, without blocking or reaping, whether the child ``pid`` has exited (Linux)."""
    if sys.platform != "linux":
        raise NotImplementedError
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


class ExitWatch:
    """
    Wait for a child process to exit without reaping it.

    An unreaped child stays a zombie, so its pid, and the process group it
    leads, cannot be reused until its parent reaps it. The watch lets the
    owner learn of the exit and still decide when to reap. Linux uses a pidfd
    and falls back to polling ``waitid(WNOWAIT)`` when pidfds are not
    available; macOS and the BSDs use a kqueue ``NOTE_EXIT`` event.

    Parameters
    ----------
    pid : int
        Unreaped child of this process.

    Examples
    --------
    >>> import subprocess, anyio
    >>> child = subprocess.Popen(["true"])
    >>> watch = ExitWatch(child.pid)
    >>> anyio.run(watch.wait)
    >>> child.returncode is None  # exited, but not reaped
    True
    >>> watch.close()
    >>> child.wait()
    0
    """

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._exited = False
        self._fd: int | None = None
        self._kqueue: Any = None
        if sys.platform == "linux":
            with contextlib.suppress(OSError):  # an old kernel or a seccomp filter: poll waitid
                self._fd = os.pidfd_open(pid)
        else:
            kqueue = select.kqueue()
            event = select.kevent(
                pid,
                filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD,
                fflags=select.KQ_NOTE_EXIT,
            )
            try:
                kqueue.control([event], 0)
            except ProcessLookupError:
                self._exited = True  # already a zombie, since it was never reaped
            except BaseException:
                kqueue.close()
                raise
            self._kqueue = kqueue
            self._fd = kqueue.fileno()

    async def wait(self) -> None:
        """
        Return once the child has exited; never reap it.

        The wait is cancellable and uses no thread. Later calls return at once.

        Examples
        --------
        >>> await watch.wait()  # doctest: +SKIP
        """
        if self._exited:
            return
        if self._fd is not None:
            await anyio.wait_readable(self._fd)
        elif sys.platform == "linux":
            # The fallback has no descriptor to wait on, so it polls.
            while not _has_exited(self._pid):  # noqa: ASYNC110
                await anyio.sleep(_EXIT_POLL_S)
        self._exited = True

    def close(self) -> None:
        """
        Release the watch descriptor; calling it again does nothing.

        Examples
        --------
        >>> watch.close()  # doctest: +SKIP
        """
        if self._kqueue is not None:
            self._kqueue.close()
        elif self._fd is not None:
            os.close(self._fd)
        self._kqueue = None
        self._fd = None


def kill_orphan(started: Job.Started) -> bool:
    """
    Kill a recorded process group, but only when it is provably the same one.

    The record must be a ``"process"`` start whose group leader is its pid,
    made on this host in this boot (a host id ending in ``":unknown"`` never
    matches), with a recorded start time equal to the live process's start
    time. Anything else returns False and nothing is signalled.

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
    >>> from swarm_arena.executor import Job
    >>> kill_orphan(Job.Started(kind="process", job_id="c-1", host_id="other:boot",
    ...                         pid=12345, pgid=12345, start_time="Thu Jan  1 00:00:00 1970"))
    False
    """
    pid, pgid = started.pid, started.pgid
    if started.kind != "process" or pid is None or pgid != pid or pid <= 1:
        return False
    if started.host_id is None or started.host_id.endswith(f":{_UNKNOWN}"):
        return False
    if started.start_time is None or started.host_id != host_id():
        return False
    if process_start_time(pid) != started.start_time:
        return False
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        return False
    return True
