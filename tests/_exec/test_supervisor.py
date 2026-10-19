import errno
import gc
import itertools
import os
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.proc import THREADED, pid_alive, require_visible_proc, wait_for_file, wait_until

from swarm_arena._exec import supervisor
from swarm_arena._exec.executor import Job
from swarm_arena._exec.procinfo import ExitWatch, host_id
from swarm_arena._exec.supervisor import ProcessSupervisor, RLimits
from swarm_arena._model.tooling import CompletedRun, RunStatus

# A grandchild that ignores SIGTERM (so only the group SIGKILL ends it); the tool waits until the
# grandchild has installed the ignore, writes its pid to gc.pid, and sleeps.
_GRANDCHILD = (
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "open('gc.ready', 'w'); time.sleep(60)"
)
STUBBORN = (
    "import os, subprocess, sys, time\n"
    f"c = subprocess.Popen([sys.executable, '-c', {_GRANDCHILD!r}])\n"
    "while not os.path.exists('gc.ready'): time.sleep(0.01)\n"
    "open('gc.pid', 'w').write(str(c.pid))\n"
    "time.sleep(60)\n"
)


NOFILE = (
    "import resource; open('nofile', 'w').write(str(resource.getrlimit(resource.RLIMIT_NOFILE)[0]))"
)


def sup() -> ProcessSupervisor:
    return ProcessSupervisor(host_id=host_id())


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


async def call(tmp_path: Path, argv: list[str], **kw: Any) -> CompletedRun:
    args: dict[str, Any] = {
        "cwd": tmp_path,
        "env": dict(os.environ),
        "stdout": tmp_path / "stdout.log",
        "stderr": tmp_path / "stderr.log",
        "timeout_s": None,
        "grace_s": 1.0,
        "rlimits": None,
        "on_started": None,
        "before_kill": None,
    }
    args.update(kw)
    return await sup().run(argv, **args)


async def test_exit_codes(tmp_path: Path) -> None:
    ok = await call(tmp_path, py("print('hi')"))
    assert (ok.status, ok.returncode, ok.stdout_path.read_text()) == (RunStatus.OK, 0, "hi\n")
    bad = await call(tmp_path, py("import sys; sys.stderr.write('e'); raise SystemExit(3)"))
    assert (bad.status, bad.returncode, bad.stderr_path.read_text()) == (RunStatus.ERROR, 3, "e")
    assert ok.started_at <= ok.ended_at


async def test_tool_gets_env_and_empty_stdin(tmp_path: Path) -> None:
    code = "import os, sys; print(os.environ['SA_TEST_VAR'], repr(sys.stdin.read()))"
    done = await call(tmp_path, py(code), env={**os.environ, "SA_TEST_VAR": "v1"})
    assert (done.status, done.stdout_path.read_text()) == (RunStatus.OK, "v1 ''\n")


async def test_on_started_receives_started(tmp_path: Path) -> None:
    seen: list[Job.Started] = []

    async def record(s: Job.Started) -> None:
        seen.append(s)

    await call(tmp_path, py("pass"), on_started=record)
    [s] = seen
    assert (s.kind, s.pgid == s.pid, s.host_id) == ("process", True, host_id())
    assert s.start_time


async def test_tool_never_starts_when_on_started_fails(tmp_path: Path) -> None:
    async def fail(s: Job.Started) -> None:
        raise RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        await call(tmp_path, py("open('ran', 'w')"), on_started=fail)
    await anyio.sleep(1)
    assert not (tmp_path / "ran").exists()


async def test_tool_waits_for_on_started(tmp_path: Path) -> None:  # the handshake
    seen: list[bool] = []

    async def slow_commit(s: Job.Started) -> None:
        await anyio.sleep(1)  # longer than the launcher needs to start
        seen.append((tmp_path / "ran").exists())

    done = await call(tmp_path, py("open('ran', 'w')"), on_started=slow_commit)
    assert done.status is RunStatus.OK
    assert seen == [False]
    assert (tmp_path / "ran").exists()


async def test_cancel_in_on_started_kills_launcher(tmp_path: Path) -> None:
    pids: list[int] = []

    async def hang(s: Job.Started) -> None:
        assert s.pid is not None
        pids.append(s.pid)
        await anyio.sleep_forever()

    with anyio.move_on_after(1) as scope:
        await call(tmp_path, py("open('ran', 'w')"), on_started=hang)
    await anyio.sleep(0.5)
    assert scope.cancelled_caught
    assert not (tmp_path / "ran").exists()
    assert wait_until(lambda: not pid_alive(pids[0]), 5)


@pytest.mark.parametrize("pythonpath", [None, "."])  # "." = an allowlisted PYTHONPATH
async def test_workspace_package_cannot_run_before_go(
    tmp_path: Path, pythonpath: str | None
) -> None:  # spec §4.8
    (tmp_path / "swarm_arena").mkdir()
    (tmp_path / "swarm_arena/__init__.py").write_text("open('shadow', 'w')")
    seen: list[bool] = []

    async def slow_fail(s: Job.Started) -> None:
        await anyio.sleep(1)
        seen.append((tmp_path / "shadow").exists())
        raise RuntimeError("commit failed")

    env = {**os.environ, "PYTHONPATH": pythonpath} if pythonpath else dict(os.environ)
    with pytest.raises(RuntimeError, match="commit failed"):
        await call(tmp_path, py("pass"), on_started=slow_fail, env=env)
    await anyio.sleep(0.5)
    assert seen == [False]
    assert not (tmp_path / "shadow").exists()


async def test_launcher_gone_before_go_is_error(
    tmp_path: Path,
) -> None:  # BrokenPipeError on the "go" write
    async def kill_it(s: Job.Started) -> None:
        assert s.pid is not None
        os.kill(s.pid, signal.SIGKILL)
        await anyio.sleep(0.2)

    done = await call(tmp_path, py("open('ran', 'w')"), on_started=kill_it)  # raises nothing
    assert (done.status, done.returncode) == (RunStatus.ERROR, -signal.SIGKILL)
    assert not (tmp_path / "ran").exists()


async def test_partial_collect_after_timeout(tmp_path: Path) -> None:  # exit criterion
    seen: list[tuple[CompletedRun, bool]] = []

    async def before_kill(r: CompletedRun) -> None:
        seen.append((r, (tmp_path / "partial.txt").exists()))

    started = time.monotonic()
    done = await call(
        tmp_path,
        py("open('partial.txt', 'w').write('x'); import time; time.sleep(60)"),
        timeout_s=1.0,
        grace_s=0.5,
        before_kill=before_kill,
    )
    assert done.status is RunStatus.TIMEOUT
    assert time.monotonic() - started < 10
    [(provisional, partial_seen)] = seen
    assert (provisional.status, provisional.returncode, partial_seen) == (
        RunStatus.TIMEOUT,
        None,
        True,
    )
    assert provisional.stderr_path == done.stderr_path
    assert provisional.started_at == done.started_at


async def test_sigterm_handler_runs_within_grace(tmp_path: Path) -> None:
    code = (
        "import signal, sys, time\n"
        "def stop(*_): open('term.txt', 'w').write('t'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "time.sleep(60)\n"
    )
    seen: list[bool] = []

    async def before_kill(r: CompletedRun) -> None:
        seen.append((tmp_path / "term.txt").exists())

    done = await call(tmp_path, py(code), timeout_s=1.0, grace_s=3.0, before_kill=before_kill)
    assert (done.status, done.returncode, seen) == (RunStatus.TIMEOUT, 0, [True])


async def test_descendant_gets_full_grace_after_leader_exits(tmp_path: Path) -> None:
    child = (
        "import signal, sys, time\n"
        "def stop(*_): time.sleep(1.0); open('child_partial.txt', 'w').write('x'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "open('child.ready', 'w')\n"
        "time.sleep(60)\n"
    )
    leader = (  # keeps the default SIGTERM action, so it dies at once
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "while not os.path.exists('child.ready'): time.sleep(0.01)\n"
        "time.sleep(60)\n"
    )
    seen: list[bool] = []

    async def before_kill(r: CompletedRun) -> None:
        seen.append((tmp_path / "child_partial.txt").exists())

    done = await call(tmp_path, py(leader), timeout_s=1.5, grace_s=3.0, before_kill=before_kill)
    assert done.status is RunStatus.TIMEOUT
    assert seen == [True]


async def test_stop_returns_early_when_group_is_gone(tmp_path: Path) -> None:
    require_visible_proc()
    started = time.monotonic()
    done = await call(tmp_path, py("import time; time.sleep(60)"), timeout_s=1.0, grace_s=30.0)
    assert done.status is RunStatus.TIMEOUT
    assert time.monotonic() - started < 10


async def test_grandchildren_killed_on_cancel(tmp_path: Path) -> None:  # exit criterion
    code = (
        "import subprocess, time; c = subprocess.Popen(['sleep', '60']); "
        "open('gc.pid', 'w').write(str(c.pid)); time.sleep(60)"
    )
    statuses: list[RunStatus] = []

    async def before_kill(r: CompletedRun) -> None:
        statuses.append(r.status)

    async def go() -> None:
        await call(tmp_path, py(code), before_kill=before_kill)

    with anyio.move_on_after(30) as scope:
        async with anyio.create_task_group() as tg:
            tg.start_soon(go)
            await wait_for_file(tmp_path / "gc.pid", 10)
            tg.cancel_scope.cancel()
    assert not scope.cancelled_caught
    assert statuses == [RunStatus.CANCELLED]
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)


async def test_raising_before_kill_still_kills_group(tmp_path: Path) -> None:
    async def broken(r: CompletedRun) -> None:
        raise RuntimeError("collector broke")

    done = await stop_after_ready(tmp_path, STUBBORN, grace_s=0.5, before_kill=broken)
    assert done.status is RunStatus.CANCELLED
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)
    assert done.stderr_path.read_text().endswith("swarm-arena: before_kill failed: RuntimeError\n")


async def test_blocked_before_kill_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.5)

    async def blocked(r: CompletedRun) -> None:
        await anyio.sleep_forever()

    started = time.monotonic()
    done = await stop_after_ready(tmp_path, STUBBORN, grace_s=0.5, before_kill=blocked)
    assert done.status is RunStatus.CANCELLED
    assert time.monotonic() - started < 10
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)
    assert "before_kill timed out" in done.stderr_path.read_text()


async def test_before_kill_runs_after_grace_and_before_kill(tmp_path: Path) -> None:
    alive: list[bool] = []

    async def before_kill(r: CompletedRun) -> None:
        alive.append(pid_alive(int((tmp_path / "gc.pid").read_text())))

    done = await stop_after_ready(tmp_path, STUBBORN, grace_s=0.5, before_kill=before_kill)
    assert done.status is RunStatus.CANCELLED
    assert alive == [True]
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)
    assert "swarm-arena" not in done.stderr_path.read_text()  # no note on success


@pytest.mark.parametrize("mode", ["blocked", "raises"])
async def test_fifo_stderr_cannot_block_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.5)
    err = str(tmp_path / "stderr.log")
    code = (
        f"import os; os.remove({err!r}); os.mkfifo({err!r})\n" + STUBBORN
    )  # the test never reads the FIFO

    async def blocked(r: CompletedRun) -> None:
        await anyio.sleep_forever()

    async def broken(r: CompletedRun) -> None:
        raise RuntimeError("collector broke")

    with anyio.fail_after(10):
        done = await stop_after_ready(
            tmp_path,
            code,
            grace_s=0.5,
            before_kill=blocked if mode == "blocked" else broken,
        )
    assert done.status is RunStatus.CANCELLED
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)


async def test_cancel_propagates_when_before_kill_raises(tmp_path: Path) -> None:
    async def broken(r: CompletedRun) -> None:
        raise RuntimeError("collector broke")

    async def go() -> None:
        await call(tmp_path, py(STUBBORN), grace_s=0.5, before_kill=broken)

    with anyio.move_on_after(30) as scope:
        async with anyio.create_task_group() as tg:  # no ExceptionGroup leaves the group
            tg.start_soon(go)
            await wait_for_file(tmp_path / "gc.pid", 10)
            tg.cancel_scope.cancel()
    assert not scope.cancelled_caught
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)
    assert "before_kill failed: RuntimeError" in (tmp_path / "stderr.log").read_text()


async def test_background_child_killed_after_normal_exit(tmp_path: Path) -> None:
    code = (
        "import subprocess; c = subprocess.Popen(['sleep', '60']); "
        "open('bg.pid', 'w').write(str(c.pid))"
    )
    done = await call(tmp_path, py(code))
    assert (done.status, done.returncode) == (RunStatus.OK, 0)
    assert wait_until(lambda: not pid_alive(int((tmp_path / "bg.pid").read_text())), 5)


async def test_oom_status_needs_memory_limit(tmp_path: Path) -> None:
    code = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    assert (
        await call(tmp_path, py(code), rlimits=RLimits(memory_bytes=8 * 2**30))
    ).status is RunStatus.OOM
    assert (await call(tmp_path, py(code))).status is RunStatus.ERROR


async def test_other_signal_is_error_with_memory_limit(tmp_path: Path) -> None:
    code = "import os, signal; os.kill(os.getpid(), signal.SIGABRT)"
    done = await call(tmp_path, py(code), rlimits=RLimits(memory_bytes=8 * 2**30))
    assert (done.status, done.returncode) == (RunStatus.ERROR, -signal.SIGABRT)


async def test_default_rlimits_cap_open_files(tmp_path: Path) -> None:
    code = NOFILE
    done = await call(tmp_path, py(code), rlimits=RLimits())
    assert done.status is RunStatus.OK
    assert 0 < int((tmp_path / "nofile").read_text()) <= 4096


async def test_no_descriptor_or_transport_leak(tmp_path: Path) -> None:  # api-anyio-packaging.md §A
    async def fail(s: Job.Started) -> None:
        raise RuntimeError("commit failed")

    n0 = len(os.listdir("/dev/fd"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        for _ in range(10):
            await call(tmp_path, py("pass"))
        for _ in range(3):
            with pytest.raises(RuntimeError, match="commit failed"):
                await call(tmp_path, py("pass"), on_started=fail)
        for _ in range(2):
            with pytest.raises(FileNotFoundError, match="missing"):
                await call(tmp_path, py("pass"), cwd=tmp_path / "missing")
        gc.collect()
    assert len(os.listdir("/dev/fd")) == n0
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


# --- Fix round 1 ---


def record_signals(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, bool, float]]:
    """Record (signal, leader still unreaped, time) for every group signal the supervisor sends."""
    seen: list[tuple[int, bool, float]] = []
    real = supervisor._signal_group

    def spy(pgid: int, sig: int) -> None:
        # The leader is the group id; a zombie still answers kill(pid, 0), a reaped pid does not.
        seen.append((sig, pid_alive(pgid), time.monotonic()))
        real(pgid, sig)

    monkeypatch.setattr(supervisor, "_signal_group", spy)
    return seen


@pytest.mark.parametrize("case", ["exit", "timeout", "cancel", "on_started_fails"])
async def test_leader_unreaped_until_last_group_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    seen = record_signals(monkeypatch)
    pids: list[int] = []

    async def record(s: Job.Started) -> None:
        assert s.pid is not None
        pids.append(s.pid)
        if case == "on_started_fails":
            raise RuntimeError("commit failed")

    code = "pass" if case == "exit" else "import time; time.sleep(60)"
    timeout_s = 1.0 if case == "timeout" else None
    if case == "on_started_fails":
        with pytest.raises(RuntimeError, match="commit failed"):
            await call(tmp_path, py(code), on_started=record)
    else:
        with anyio.move_on_after(1.0 if case == "cancel" else 30):
            await call(tmp_path, py(code), timeout_s=timeout_s, grace_s=0.5, on_started=record)
    assert seen
    assert seen[-1][0] == signal.SIGKILL
    assert [unreaped for _, unreaped, _ in seen] == [True] * len(seen)  # the pgid stayed anchored
    assert not pid_alive(pids[0])  # reaped once run() is done: no zombie is left


async def test_kill_at_deadline_when_before_kill_ignores_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.5)
    seen = record_signals(monkeypatch)
    entered: list[float] = []

    async def stuck(r: CompletedRun) -> None:
        entered.append(time.monotonic())
        await anyio.to_thread.run_sync(time.sleep, 4.0)  # not abandonable: ignores cancellation

    done = await stop_after_ready(tmp_path, STUBBORN, grace_s=0.5, before_kill=stuck)
    kills = [at for sig, _, at in seen if sig == signal.SIGKILL]
    assert done.status is RunStatus.CANCELLED
    assert 0.4 <= kills[0] - entered[0] < 3.5  # deadline precedes the blocked 4 s collector
    assert "before_kill timed out after 0.5 s" in done.stderr_path.read_text()
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)


async def test_before_kill_failure_note_names_only_the_type(tmp_path: Path) -> None:
    async def leaky(r: CompletedRun) -> None:
        raise ValueError("sentinel-secret")

    done = await stop_after_ready(tmp_path, STUBBORN, grace_s=0.5, before_kill=leaky)
    log = done.stderr_path.read_text()
    assert "before_kill failed: ValueError" in log
    assert "sentinel-secret" not in log


async def test_log_open_refuses_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    with pytest.raises(OSError, match="link"):
        await call(tmp_path, py("open('ran', 'w')"), stdout=tmp_path / "link/stdout.log")
    assert not (real / "stdout.log").exists()
    assert not (tmp_path / "ran").exists()


async def test_log_open_refuses_hardlinked_leaf(tmp_path: Path) -> None:
    (tmp_path / "other.txt").write_text("keep")
    os.link(tmp_path / "other.txt", tmp_path / "stdout.log")
    with pytest.raises(OSError, match=r"stdout\.log"):
        await call(tmp_path, py("open('ran', 'w')"))
    assert (tmp_path / "other.txt").read_text() == "keep"
    assert not (tmp_path / "ran").exists()


async def test_log_open_refuses_fifo_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "stderr.log")  # nobody reads it
    started = time.monotonic()
    with pytest.raises(OSError, match=r"stderr\.log"):
        await call(tmp_path, py("open('ran', 'w')"))
    assert time.monotonic() - started < 5
    assert not (tmp_path / "ran").exists()


async def test_log_rerun_truncates(tmp_path: Path) -> None:
    await call(tmp_path, py("print('a much longer first line')"))
    done = await call(tmp_path, py("print('x')"))
    assert done.stdout_path.read_text() == "x\n"


# --- Fix round 2 ---


async def test_leader_that_exits_during_on_started_stays_anchored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = record_signals(monkeypatch)
    pids: list[int] = []

    async def crash_then_fail(s: Job.Started) -> None:
        assert s.pid is not None
        pids.append(s.pid)
        os.kill(s.pid, signal.SIGKILL)  # the launcher dies while the start is being committed
        await anyio.sleep(0.3)  # long enough for any eager reaper to collect it
        raise RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        await call(tmp_path, py("open('ran', 'w')"), on_started=crash_then_fail)
    assert [(sig, unreaped) for sig, unreaped, _ in seen] == [(signal.SIGKILL, True)]
    assert not pid_alive(pids[0])
    assert not (tmp_path / "ran").exists()


@pytest.mark.parametrize("kind", ["KeyboardInterrupt", "SystemExit", "cancelled"])
async def test_base_exception_from_before_kill_is_not_wrapped(tmp_path: Path, kind: str) -> None:
    expected: type[BaseException] = {
        "KeyboardInterrupt": KeyboardInterrupt,
        "SystemExit": SystemExit,
        "cancelled": anyio.get_cancelled_exc_class(),
    }[kind]

    async def broken(r: CompletedRun) -> None:
        raise expected()

    caught: BaseException | None = None
    try:
        await stop_after_ready(
            tmp_path, STUBBORN, grace_s=0.5, before_kill=broken, propagate_cancel=True
        )
    except BaseException as exc:
        caught = exc
    assert type(caught) is expected
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)


async def test_safety_net_reap_does_not_block_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_watch(pid: int) -> ExitWatch:
        raise OSError(errno.EMFILE, "Too many open files")

    pids: list[int] = []
    real_spawn = supervisor._spawn

    def slow_spawn(*args: Any, **kwargs: Any) -> None:
        real_spawn(*args, **kwargs)
        proc = kwargs["procs"][-1]
        pids.append(proc.pid)
        real_wait = proc.wait

        def slow_wait(timeout: float | None = None) -> int:
            time.sleep(1.0)  # a leader that is slow to die after SIGKILL
            return real_wait(timeout)

        monkeypatch.setattr(proc, "wait", slow_wait)

    monkeypatch.setattr(supervisor, "ExitWatch", no_watch)
    monkeypatch.setattr(supervisor, "_spawn", slow_spawn)
    ticks: list[float] = []

    async def ticker() -> None:
        while True:
            ticks.append(time.monotonic())
            await anyio.sleep(0.05)

    async with anyio.create_task_group() as tg:
        tg.start_soon(ticker)
        await anyio.sleep(0.2)
        with pytest.raises(OSError, match="Too many open files"):
            await call(tmp_path, py("open('ran', 'w')"))
        await anyio.sleep(0.2)
        tg.cancel_scope.cancel()
    assert max(b - a for a, b in itertools.pairwise(ticks)) < 0.5  # the loop kept running
    assert not pid_alive(pids[0])  # killed and reaped
    assert not (tmp_path / "ran").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux reports a live process as Z")
async def test_grace_waits_for_live_threads_of_an_exited_main_thread(tmp_path: Path) -> None:
    require_visible_proc()
    child = THREADED.format(work_s=1.5)
    leader = (  # keeps the default SIGTERM action; the child ignores SIGTERM
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "while not os.path.exists('child.ready'): time.sleep(0.01)\n"
        "time.sleep(60)\n"
    )
    seen: list[bool] = []

    async def before_kill(r: CompletedRun) -> None:
        seen.append((tmp_path / "worker.done").exists())

    done = await stop_after_ready(
        tmp_path, leader, ready="child.ready", grace_s=5.0, before_kill=before_kill
    )
    assert done.status is RunStatus.CANCELLED
    assert seen == [True]  # the grace covered the worker thread, not just the main thread


async def stop_after_ready(
    tmp_path: Path,
    code: str,
    *,
    ready: str = "gc.pid",
    propagate_cancel: bool = False,
    **kwargs: Any,
) -> CompletedRun:
    """Cancel after child readiness and return the provisional stopped run."""
    runs: list[CompletedRun] = []
    failure: BaseException | None = None
    collect = kwargs.pop("before_kill", None)

    async def record(run: CompletedRun) -> None:
        runs.append(run)
        if collect is not None:
            await collect(run)

    async def stop_when_ready(scope: anyio.CancelScope) -> None:
        await wait_for_file(tmp_path / ready, 30)
        scope.cancel()

    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(stop_when_ready, scope)
            try:
                await call(tmp_path, py(code), before_kill=record, **kwargs)
            except BaseException as error:
                failure = error
            finally:
                tasks.cancel_scope.cancel()
    if failure is not None and (
        propagate_cancel or not isinstance(failure, anyio.get_cancelled_exc_class())
    ):
        raise failure
    assert runs, "child must be ready before the stop sequence"
    return runs[-1]


async def test_stop_timing_waits_for_slow_child_startup(tmp_path: Path) -> None:
    code = "import time; time.sleep(1.5)\n" + STUBBORN
    done = await stop_after_ready(tmp_path, code, grace_s=0.1)
    assert (tmp_path / "gc.pid").exists()
    assert done.status is RunStatus.CANCELLED
