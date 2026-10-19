"""Identify the host and check or kill process groups recorded in a start event."""

import os
import re
import signal
import socket
import subprocess
from pathlib import Path

from swarm_arena._exec.executor import Job

_BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
_SYSCTL = "/usr/sbin/sysctl"
_UNKNOWN = "unknown"
# ps prints lstart in the local time of its locale; pin both so a resume on the
# same host reads the same string as the run that recorded it.
_PS_ENV = {"LC_ALL": "C", "TZ": "UTC"}


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
            ["ps", "-o", "lstart=", "-p", str(pid)],
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
