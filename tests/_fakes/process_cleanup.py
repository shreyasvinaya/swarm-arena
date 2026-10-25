"""Emergency cleanup for children owned by a test, independent of product teardown."""

import os
import signal
import subprocess
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

import pytest

from swarm_arena._exec.procinfo import group_has_live_member, process_start_time

_WAIT_PROCESS = subprocess.Popen.wait


def _leader_missing(pid: int) -> bool:
    """Distinguish a missing leader from an unreadable start-time identity."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _reaped_group_alive(process: subprocess.Popen[Any], start_time: str | None) -> bool:
    """Whether a reaped leader's verified group still has a live member."""
    current = process_start_time(process.pid)
    same_leader = start_time is not None and current == start_time
    leader_gone = current is None and _leader_missing(process.pid)
    return (same_leader or leader_gone) and group_has_live_member(process.pid)


def _left_alive(process: subprocess.Popen[Any], own_group: bool, start_time: str | None) -> bool:
    """Whether a captured child, or a descendant in its own group, is still running."""
    if process.poll() is None:
        return True
    if not own_group:
        return False
    try:
        return _reaped_group_alive(process, start_time)
    except (ProcessLookupError, PermissionError):
        return False


def reap_children(children: list[tuple[subprocess.Popen[Any], bool, str | None]]) -> list[str]:
    """Reap captured children, checking leader identity and liveness before group kills.

    An unreaped leader is signalled directly: its pid, and so its group id,
    cannot be reused while the Popen holds it. A reaped leader with no current
    identity can still have live descendants; their group retains its id until
    its last member exits. A replacement leader or an empty group is never
    signalled. An unavailable start-time lookup alone is not evidence that the
    leader is missing.
    """
    errors: list[str] = []
    for process, own_group, start_time in children:
        try:
            with suppress(ProcessLookupError, PermissionError):
                if own_group and process.poll() is None:
                    # An unreaped leader keeps its pid, so its group id cannot
                    # have been reused: no identity lookup is needed.
                    if group_has_live_member(process.pid):
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                elif own_group and _reaped_group_alive(process, start_time):
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
            # Bypass per-instance wait surrogates used by supervisor failure tests.
            _WAIT_PROCESS(process, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"child cleanup failed: {type(error).__name__}")
    return errors


@pytest.fixture
def owned_processes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Retain children at creation so entry failures cannot skip their emergency reap."""
    children: list[tuple[subprocess.Popen[Any], bool, str | None]] = []
    original = subprocess.Popen

    class OwnedProcess(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            own_group = bool(kwargs.get("start_new_session"))
            index = len(children)
            children.append((self, own_group, None))
            # Identity probes spawn ordinary children; only group leaders need
            # an identity, so capturing the probes cannot recurse.
            if own_group:
                children[index] = (self, own_group, process_start_time(self.pid))

    monkeypatch.setattr(subprocess, "Popen", OwnedProcess)
    try:
        yield
    finally:
        leaked = [
            process for process, own, started in children if _left_alive(process, own, started)
        ]
        errors = reap_children(children)
        assert not leaked, "test left an owned child alive; emergency cleanup ran"
        assert not errors, "; ".join(errors)
