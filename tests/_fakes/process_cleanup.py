"""Emergency cleanup for children owned by a test, independent of product teardown."""

import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProcessIdentity:
    """A process-table identity and its parent/group ownership links."""

    pid: int
    ppid: int
    pgid: int
    started: str


def process_table() -> dict[int, ProcessIdentity]:
    """
    Snapshot process identities and ownership links with one bounded ps call.

    Returns
    -------
    dict of int to ProcessIdentity
        Process ids mapped to start times, parents and groups.

    Examples
    --------
    >>> process_table()[os.getpid()].pid == os.getpid()
    True
    """
    done = subprocess.run(
        ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "pgid=", "-o", "lstart="],
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )
    if done.returncode:
        raise AssertionError("owned process table unavailable")
    result = {}
    for line in done.stdout.splitlines():
        pid, ppid, pgid, started = line.split(maxsplit=3)
        record = ProcessIdentity(int(pid), int(ppid), int(pgid), started.strip())
        result[record.pid] = record
    return result


def snapshot_process_tree(pid: int) -> dict[int, ProcessIdentity]:
    """
    Record a driver's entire transitive descendant tree before sending signals.

    Parameters
    ----------
    pid : int
        Owned driver's process id.

    Returns
    -------
    dict of int to ProcessIdentity
        The driver and all descendants, including separate sessions/groups.

    Examples
    --------
    >>> snapshot_process_tree(os.getpid())[os.getpid()].pid == os.getpid()
    True
    """
    table = process_table()
    descendants = {pid: table[pid]} if pid in table else {}
    pending = [pid]
    while pending:
        parent = pending.pop()
        for child in table.values():
            if child.ppid == parent and child.pid not in descendants:
                descendants[child.pid] = child
                pending.append(child.pid)
    return descendants


def reap_process_tree(
    process: subprocess.Popen[Any],
    started: str | None,
    recorded: dict[int, ProcessIdentity],
    *,
    timeout_s: float = 60,
) -> None:
    """
    Kill owned descendant groups and verify each captured pid identity is gone.

    Parameters
    ----------
    process : subprocess.Popen
        Owned driver, reaped through the shared child helper.
    started : str or None
        Driver's captured start time.
    recorded : dict of int to ProcessIdentity
        Snapshot taken before signalling, possibly retained across driver exit.
    timeout_s : float, optional
        Separate cleanup verification allowance.

    Raises
    ------
    AssertionError
        Any recorded identity remains or the process table cannot be read.

    Examples
    --------
    >>> callable(reap_process_tree)
    True
    """
    signalled: set[int] = set()
    try:
        current = process_table()
        for record in reversed(recorded.values()):
            present = current.get(record.pid)
            if present is None or present.started != record.started:
                continue
            record = present
            with suppress(ProcessLookupError):
                # Never signal the test runner's inherited group if session creation regresses.
                if record.pgid in recorded and record.pgid != os.getpgrp():
                    leader = current.get(record.pgid)
                    original = recorded[record.pgid]
                    if leader is not None and leader.started != original.started:
                        continue
                    if record.pgid not in signalled:
                        os.killpg(record.pgid, signal.SIGKILL)
                        signalled.add(record.pgid)
                else:
                    os.kill(record.pid, signal.SIGKILL)
    finally:
        errors = reap_children([(process, True, started)])
    if errors:
        raise AssertionError("; ".join(errors))
    deadline = time.monotonic() + timeout_s
    while True:
        current = process_table()
        remaining = [
            pid
            for pid, record in recorded.items()
            if pid in current and current[pid].started == record.started
        ]
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise AssertionError("owned process identities remained after cleanup")
        time.sleep(0.01)
