"""Process helpers shared by the supervisor, procinfo, and executor tests."""

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import anyio


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
