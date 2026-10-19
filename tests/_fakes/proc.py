"""Process helpers shared by the supervisor, procinfo, and executor tests."""

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest

from swarm_arena._exec.procinfo import _proc_hides_pids


def pid_alive(pid: int) -> bool:
    """Return True while ``pid`` exists (a zombie counts until it is reaped)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_until(pred: Callable[[], bool], timeout_s: float) -> bool:
    """Poll ``pred`` every 0.05 s; return True once it holds, False after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while True:
        if pred():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


async def wait_for_file(path: Path, timeout_s: float) -> None:
    """Poll for ``path`` without blocking the event loop; TimeoutError after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while not await anyio.Path(path).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{path.name} did not appear")
        await anyio.sleep(0.05)


def spawn_group(argv: list[str]) -> subprocess.Popen[bytes]:
    """Start ``argv`` as the leader of a new process group (call through a thread in async code)."""
    return subprocess.Popen(argv, start_new_session=True)


def reap(proc: subprocess.Popen[bytes]) -> None:
    """Kill ``proc`` if it still runs, then reap it, so it is not left as a zombie."""
    if proc.poll() is None:
        proc.kill()
    proc.wait()


# A child whose main thread calls pthread_exit while a worker thread keeps running: Linux then
# shows state Z in /proc/<pid>/stat although the process is alive.
THREADED = (
    "import ctypes, signal, threading, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "def work():\n"
    "    time.sleep({work_s})\n"
    "    open('worker.done', 'w').close()\n"
    "threading.Thread(target=work).start()\n"
    "open('child.ready', 'w').close()\n"
    "ctypes.CDLL(None).pthread_exit(None)\n"
)


def main_thread_state(pid: int) -> str:
    """
    Read the Linux main thread state, or an empty string when unavailable.

    Parameters
    ----------
    pid : int
        Process identifier to inspect.

    Returns
    -------
    str
        Linux state letter, or an empty string.

    Examples
    --------
    >>> main_thread_state(-1)
    ''
    """
    try:
        data = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return ""
    return data[data.rfind(b")") + 1 :].split()[0].decode()


def require_visible_proc() -> None:
    """
    Skip host liveness assertions when Linux hides process information.

    Examples
    --------
    >>> import sys
    >>> if sys.platform != "linux":
    ...     require_visible_proc()
    """
    if sys.platform == "linux" and _proc_hides_pids(Path("/proc")):
        pytest.skip("Linux /proc uses hidepid or its visibility cannot be established")
