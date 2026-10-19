import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest
from _fakes.proc import wait_until

from swarm_arena._exec import procinfo
from swarm_arena._exec.executor import Job
from swarm_arena._exec.procinfo import group_alive, host_id, kill_orphan, process_start_time


def test_host_id_format_and_stable() -> None:
    assert re.fullmatch(r"[^:]+:.+", host_id())
    assert host_id() == host_id()
    assert not host_id().endswith(":unknown")  # Linux boot_id or macOS kern.boottime


def test_host_id_without_boot_id_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(procinfo, "_BOOT_ID_FILE", tmp_path / "missing")
    monkeypatch.setattr(procinfo, "_SYSCTL", str(tmp_path / "no-sysctl"))
    assert host_id().endswith(":unknown")


def test_process_start_time_and_group_alive() -> None:
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    st = process_start_time(p.pid)
    assert st
    assert st == process_start_time(p.pid)
    assert group_alive(p.pid)
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    assert process_start_time(p.pid) is None
    assert wait_until(lambda: not group_alive(p.pid), 5)


def test_group_alive_counts_a_foreign_group() -> None:
    assert group_alive(1)  # init's group: PermissionError for a normal user, still alive


def test_kill_orphan_checks_host_and_start_time() -> None:
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    good = Job.Started(
        kind="process",
        job_id="c-1",
        host_id=host_id(),
        pid=p.pid,
        pgid=p.pid,
        start_time=process_start_time(p.pid),
    )
    assert not kill_orphan(good.model_copy(update={"start_time": "Thu Jan  1 00:00:00 1970"}))
    assert p.poll() is None
    assert not kill_orphan(good.model_copy(update={"host_id": "other:boot"}))
    assert p.poll() is None
    assert kill_orphan(good)
    assert p.wait(5) == -signal.SIGKILL


@pytest.mark.parametrize(
    "update",
    [
        {"kind": "dispatch"},
        {"start_time": None},
        {"host_id": None},
        {"pid": None},
        {"pgid": None},
        {"pgid": 0},  # killpg(0) would hit our own group
    ],
)
def test_kill_orphan_refuses_incomplete_records(update: dict[str, Any]) -> None:
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        good = Job.Started(
            kind="process",
            job_id="c-1",
            host_id=host_id(),
            pid=p.pid,
            pgid=p.pid,
            start_time=process_start_time(p.pid),
        )
        assert not kill_orphan(good.model_copy(update=update))
        assert p.poll() is None
    finally:
        p.kill()
        p.wait()


def test_kill_orphan_refuses_unknown_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(procinfo, "_BOOT_ID_FILE", tmp_path / "missing")
    monkeypatch.setattr(procinfo, "_SYSCTL", str(tmp_path / "no-sysctl"))
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        started = Job.Started(
            kind="process",
            job_id="c-1",
            host_id=host_id(),  # "<hostname>:unknown" on both sides
            pid=p.pid,
            pgid=p.pid,
            start_time=process_start_time(p.pid),
        )
        assert not kill_orphan(started)
        assert p.poll() is None
    finally:
        p.kill()
        p.wait()
