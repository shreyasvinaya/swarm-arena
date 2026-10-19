import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.proc import pid_alive, wait_until

from swarm_arena._exec import procinfo
from swarm_arena._exec.executor import Job
from swarm_arena._exec.procinfo import (
    ExitWatch,
    group_alive,
    group_has_live_member,
    host_id,
    kill_orphan,
    process_start_time,
)


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


# --- Fix round 1 ---


def _is_zombie(pid: int) -> bool:
    done = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return done.stdout.strip().startswith("Z")


def test_group_has_live_member_ignores_an_unreaped_zombie() -> None:
    p = subprocess.Popen(["sleep", "0.3"], start_new_session=True)
    try:
        assert group_has_live_member(p.pid)
        assert wait_until(lambda: _is_zombie(p.pid), 5)  # exited, not reaped
        assert not group_has_live_member(p.pid)
        assert group_alive(p.pid)  # group_alive keeps its meaning: a zombie counts
    finally:
        p.wait()
    assert not group_has_live_member(p.pid)


def test_group_has_live_member_sees_a_live_child_of_a_zombie_leader(tmp_path: Path) -> None:
    code = (
        "import subprocess; "
        "open('child.pid', 'w').write(str(subprocess.Popen(['sleep', '30']).pid))"
    )
    p = subprocess.Popen([sys.executable, "-c", code], start_new_session=True, cwd=tmp_path)
    try:
        assert wait_until(lambda: _is_zombie(p.pid), 5)
        assert group_has_live_member(p.pid)
        os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)
        assert wait_until(lambda: not group_has_live_member(p.pid), 5)
    finally:
        p.wait()


def fake_proc(root: Path, options: str = "rw,nosuid,nodev,noexec") -> Path:
    """Make an empty fake /proc whose mountinfo mounts it with ``options``."""
    (root / "self").mkdir(parents=True)
    (root / "self/mountinfo").write_text(
        f"22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
        f"25 22 0:23 / {root} rw,relatime shared:13 - proc proc {options}\n"
    )
    return root


def fake_task(proc: Path, pid: str, stat: str, tasks: dict[str, str] | None = None) -> None:
    (proc / pid).mkdir()
    (proc / pid / "stat").write_text(stat)
    if tasks is not None:
        (proc / pid / "task").mkdir()
        for tid, task_stat in tasks.items():
            (proc / pid / "task" / tid).mkdir()
            (proc / pid / "task" / tid / "stat").write_text(task_stat)


def test_proc_scan_reads_state_and_pgrp_after_the_last_paren(tmp_path: Path) -> None:
    proc = fake_proc(tmp_path)
    fake_task(proc, "100", "100 (odd) name) S 1 4242 4242 0 -1\n")  # comm may hold spaces, parens
    fake_task(proc, "101", "101 (zz) Z 1 777 777 0 -1\n", {"101": "101 (zz) Z 1 777 777\n"})
    fake_task(proc, "102", "102 (gone) X 1 888 888 0 -1\n", {})
    (proc / "103").mkdir()  # exited while scanning: no stat file
    assert procinfo._live_in_proc(4242, proc)
    assert not procinfo._live_in_proc(777, proc)
    assert not procinfo._live_in_proc(888, proc)
    assert not procinfo._live_in_proc(5, proc)


def test_proc_scan_counts_a_live_thread_of_an_exited_main_thread(tmp_path: Path) -> None:
    proc = fake_proc(tmp_path)
    tasks = {"200": "200 (py) Z 1 4242 4242\n", "201": "201 (py) S 1 4242 4242\n"}
    fake_task(proc, "200", "200 (py) Z 1 4242 4242 0 -1\n", tasks)  # main thread did pthread_exit
    fake_task(proc, "300", "300 (z) Z 1 777 777 0 -1\n")  # a zombie whose task dir is gone
    assert procinfo._live_in_proc(4242, proc)
    assert not procinfo._live_in_proc(777, proc)


@pytest.mark.parametrize("options", ["rw,hidepid=2", "rw,hidepid=invisible", "rw,hidepid=1"])
def test_proc_scan_cannot_rule_out_hidden_members(tmp_path: Path, options: str) -> None:
    proc = fake_proc(tmp_path, options)
    assert procinfo._live_in_proc(5, proc)  # another user's member may be unlisted


@pytest.mark.parametrize("options", ["rw,hidepid=0", "rw,hidepid=off", "rw"])
def test_proc_scan_trusts_a_proc_without_hidepid(tmp_path: Path, options: str) -> None:
    assert not procinfo._live_in_proc(5, fake_proc(tmp_path, options))


def test_proc_scan_without_mountinfo_cannot_tell(tmp_path: Path) -> None:
    assert procinfo._live_in_proc(5, tmp_path)


@pytest.mark.parametrize("unreadable", ["300/stat", "300/task", "300/task/301/stat"])
def test_proc_scan_unreadable_entry_cannot_tell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unreadable: str
) -> None:
    proc = fake_proc(tmp_path)
    tasks = {"300": "300 (z) Z 1 777 777\n", "301": "301 (z) Z 1 777 777\n"}
    fake_task(proc, "300", "300 (z) Z 1 777 777 0 -1\n", tasks)
    assert not procinfo._live_in_proc(777, proc)
    real_read, real_list = procinfo._read_proc, procinfo._list_proc
    denied = proc / unreadable

    def read(path: Path) -> bytes:
        if path == denied:
            raise PermissionError(13, "Permission denied")
        return real_read(path)

    def listdir(path: Path) -> list[str]:
        if path == denied:
            raise PermissionError(13, "Permission denied")
        return real_list(path)

    monkeypatch.setattr(procinfo, "_read_proc", read)
    monkeypatch.setattr(procinfo, "_list_proc", listdir)
    assert procinfo._live_in_proc(777, proc)  # the pid still exists, but its state is unknown


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


def _main_thread_state(pid: int) -> str:
    try:
        data = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return ""
    return data[data.rfind(b")") + 1 :].split()[0].decode()


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux reports a live process as Z")
def test_live_threads_of_an_exited_main_thread_count(tmp_path: Path) -> None:
    code = THREADED.format(work_s=1.0)
    p = subprocess.Popen([sys.executable, "-c", code], start_new_session=True, cwd=tmp_path)
    try:
        assert wait_until(lambda: _main_thread_state(p.pid) == "Z", 5)
        assert not (tmp_path / "worker.done").exists()
        assert group_has_live_member(p.pid)  # the worker thread still runs
        assert wait_until(lambda: (tmp_path / "worker.done").exists(), 5)
        assert wait_until(lambda: not group_has_live_member(p.pid), 5)
    finally:
        p.kill()
        p.wait()


def _spawn(argv: list[str]) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(argv)


async def test_exit_watch_waits_without_reaping() -> None:
    p = await anyio.to_thread.run_sync(_spawn, ["sleep", "0.3"])
    with contextlib.closing(ExitWatch(p.pid)) as watch:
        with anyio.fail_after(5):
            await watch.wait()
        assert p.returncode is None
        assert pid_alive(p.pid)  # exited but still a zombie
        with anyio.fail_after(1):
            await watch.wait()  # returns at once the second time
    assert await anyio.to_thread.run_sync(p.wait) == 0


async def test_exit_watch_on_an_existing_zombie() -> None:
    p = await anyio.to_thread.run_sync(_spawn, ["true"])
    await anyio.sleep(0.3)  # exited, never reaped
    with contextlib.closing(ExitWatch(p.pid)) as watch, anyio.fail_after(1):
        await watch.wait()
    assert await anyio.to_thread.run_sync(p.wait) == 0


async def test_exit_watch_is_cancellable() -> None:
    p = await anyio.to_thread.run_sync(_spawn, ["sleep", "30"])
    try:
        with contextlib.closing(ExitWatch(p.pid)) as watch:
            with anyio.move_on_after(0.2) as scope:
                await watch.wait()
            assert scope.cancelled_caught
    finally:
        p.kill()
        await anyio.to_thread.run_sync(p.wait)


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="the waitid fallback is Linux only")
async def test_exit_watch_polls_waitid_without_pidfd(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_pidfd(pid: int, flags: int = 0) -> int:
        raise OSError(errno.ENOSYS, "pidfd_open not supported")

    monkeypatch.setattr(os, "pidfd_open", no_pidfd)
    p = await anyio.to_thread.run_sync(_spawn, ["sleep", "0.3"])
    with contextlib.closing(ExitWatch(p.pid)) as watch:
        with anyio.fail_after(5):
            await watch.wait()
        assert p.returncode is None
    assert await anyio.to_thread.run_sync(p.wait) == 0


def test_ps_scan_ignores_zombies() -> None:
    p = subprocess.Popen(["sleep", "0.3"], start_new_session=True)
    try:
        assert procinfo._live_in_ps(p.pid)
        assert wait_until(lambda: _is_zombie(p.pid), 5)
        assert not procinfo._live_in_ps(p.pid)
    finally:
        p.wait()


def test_liveness_reads_as_alive_when_it_cannot_tell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(procinfo, "_PS", str(tmp_path / "no-ps"))
    assert procinfo._live_in_ps(12345)
    assert procinfo._live_in_proc(12345, tmp_path / "no-proc")


def test_group_of_another_user_has_a_live_member() -> None:
    # init's group: signal 0 gives PermissionError to a normal user, and its leader is not a zombie
    assert group_has_live_member(1)
