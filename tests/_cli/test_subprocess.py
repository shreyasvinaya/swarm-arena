"""Exercise the installed CLI boundary in real child processes."""

import os
import pty
import select
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from _fakes.run import plan
from _fakes.tools import PyTool

from swarm_arena import RunConfig
from swarm_arena._exec.procinfo import group_has_live_member
from swarm_arena._journal.lease import read_owner
from swarm_arena._journal.reader import read_committed
from swarm_arena._model.canonical import utc_now
from swarm_arena._store.local import LocalStore

_ENTRY = "from swarm_arena._cli.main import main; raise SystemExit(main())"
_SECRET = "CLI-MUTATION-SENTINEL"


@pytest.fixture
def cli_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PYTHONUNBUFFERED": "1"}


@pytest.fixture
def inputs(tmp_path: Path, run_config: RunConfig) -> tuple[Path, Path]:
    config_path = tmp_path / "run.toml"
    plan_path = tmp_path / "plan.toml"
    config_path.write_text(run_config.dumps())
    plan_path.write_text(plan(rounds=1).dumps())
    return config_path, plan_path


def _invoke(
    argv: list[str], env: dict[str, str], *, code: str = _ENTRY
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-c", code, *argv],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_mutated_core_config_error_subprocess(cli_env: dict[str, str]) -> None:
    code = f'''
from swarm_arena import RunConfig
from swarm_arena._errors import ConfigError
from swarm_arena._cli import run_cmds
async def fail(*args, **kwargs):
    try:
        RunConfig.loads('reserve_fraction = "{_SECRET}"')
    except ConfigError as error:
        error.args = ("{_SECRET}",)
        error.add_note("{_SECRET}")
        error.__cause__ = ValueError("{_SECRET}")
        raise
from pytest import MonkeyPatch
with MonkeyPatch.context() as patch:
    patch.setattr(run_cmds, "_run_async", fail)
    {_ENTRY}
'''
    result = _invoke(["run", "--yes", "unused.toml", "unused-plan.toml"], cli_env, code=code)
    assert result.returncode == 1
    assert _SECRET not in result.stdout + result.stderr
    assert "reserve_fraction: float_parsing" in result.stderr
    assert "Traceback" not in result.stderr


def test_multi_error_plan_subprocess(inputs: tuple[Path, Path], cli_env: dict[str, str]) -> None:
    config_path, plan_path = inputs
    config = RunConfig.load(config_path)
    config_path.write_text(
        config.model_copy(
            update={"limits": config.limits.model_copy(update={"max_rounds": 1})}
        ).dumps()
    )
    plan_path.write_text(plan(rounds=2).model_copy(update={"ranking": "absent"}).dumps())
    result = _invoke(["run", "--yes", str(config_path), str(plan_path)], cli_env)
    assert result.returncode == 1
    assert result.stderr.startswith("invalid plan:\n")
    assert result.stderr.count("\n- ") == 2


@pytest.mark.parametrize("command", ["status", "stop"])
def test_missing_run_subprocess(tmp_path: Path, cli_env: dict[str, str], command: str) -> None:
    result = _invoke([command, str(tmp_path / "missing")], cli_env)
    assert result.returncode == 1
    assert result.stderr == f"run missing: {'no run' if command == 'status' else 'no live owner'}\n"


def test_missing_resume_owner_subprocess(
    tmp_path: Path, inputs: tuple[Path, Path], cli_env: dict[str, str]
) -> None:
    result = _invoke(["resume", str(tmp_path / "missing"), "--config", str(inputs[0])], cli_env)
    assert result.returncode == 1
    assert result.stderr == "run has no committed owner\n"


@contextmanager
def _child(
    argv: list[str], env: dict[str, str], *, terminal: bool = False, code: str = _ENTRY
) -> Generator[tuple[subprocess.Popen[bytes], int | None]]:
    master, slave = pty.openpty() if terminal else (None, None)
    try:
        with subprocess.Popen(
            [sys.executable, "-B", "-c", code, *argv],
            env=env,
            stdin=slave if terminal else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            try:
                yield process, master
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    process.communicate(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=10)
    finally:
        if master is not None:
            os.close(master)
        if slave is not None:
            os.close(slave)


def _wait_prompt(process: subprocess.Popen[bytes]) -> bytes:
    assert process.stdout is not None
    output = b""
    deadline = time.monotonic() + 20
    while b"Start the run? [y/N] " not in output:
        assert time.monotonic() < deadline, output
        assert process.poll() is None, process.communicate()
        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if ready:
            output += os.read(process.stdout.fileno(), 4096)
    return output


@pytest.mark.parametrize("interrupt", [False, True])
def test_terminal_prompt_subprocess(
    inputs: tuple[Path, Path],
    cli_env: dict[str, str],
    run_config: RunConfig,
    interrupt: bool,
) -> None:
    with _child(["run", *map(str, inputs)], cli_env, terminal=True) as (process, master):
        output = _wait_prompt(process)
        if interrupt:
            process.send_signal(signal.SIGINT)
        else:
            assert master is not None
            os.write(master, b"n\n")
        # Include parent scheduling and interpreter shutdown on loaded CI hosts.
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == (130 if interrupt else 1), (output + stdout, stderr)
        assert stderr == (b"interrupted\n" if interrupt else b"run not approved\n")
    assert not Path(run_config.run_root).exists()
    assert not Path(run_config.artifact_root).exists()


@pytest.mark.parametrize("parent_delayed", [False, True], ids=["on_time", "parent_delayed"])
def test_prompt_sigint_with_reader_bootstrap_delayed(
    inputs: tuple[Path, Path],
    cli_env: dict[str, str],
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    parent_delayed: bool,
) -> None:
    # Simulate an unscheduled reader thread, before Thread.start() can return.
    # Hold the barrier forever: SIGINT must exit without needing its release.
    code = """
import atexit, faulthandler, os, signal, threading, time
from swarm_arena._cli import run_cmds
from swarm_arena._cli.main import main
bootstrap = threading.Thread._bootstrap_inner
def delayed_bootstrap(self: threading.Thread) -> None:
    if self.name == "swarm-arena-daemon-_prompt":
        threading.Event().wait()
    bootstrap(self)
from pytest import MonkeyPatch
with MonkeyPatch.context() as patch:
    patch.setattr(threading.Thread, "_bootstrap_inner", delayed_bootstrap)
    sent = None
    original_print = print
    def prompt_print(*args: object, **kwargs: object) -> None:
        global sent
        original_print(*args, **kwargs)
        if args and args[0] == "Start the run? [y/N] ":
            # Deliver at a known boundary, with the Runner handler installed and
            # before reader startup. No parent/child scheduling race positions it.
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler) and handler is not signal.default_int_handler
            sent = time.monotonic()
            os.kill(os.getpid(), signal.SIGINT)
    patch.setattr(run_cmds, "print", prompt_print, raising=False)
    def exiting() -> None:
        faulthandler.cancel_dump_traceback_later()
        assert sent is not None
        print(f"sigint-to-exit {time.monotonic() - sent}", flush=True)
    atexit.register(exiting)
    # Preserve a stack in TimeoutExpired.stderr before _child's cleanup can send
    # another SIGINT. This watchdog is diagnostic, not the signal-positioning gate.
    faulthandler.dump_traceback_later(10)
    raise SystemExit(main())
"""
    with _child(["run", *map(str, inputs)], cli_env, terminal=True, code=code) as (
        process,
        _,
    ):
        with monkeypatch.context() as patch:
            if parent_delayed:
                # Model the collector being descheduled after select() wakes.
                # Wait for actual child exit, then advance only Popen's clock:
                # no CPU load, wall-clock sleep, or slower child is involved.
                selector_type = vars(subprocess)["_PopenSelector"]
                original_select = selector_type.select
                original_time = time.monotonic
                elapsed = 0.0

                def delayed_select(
                    selector: selectors.BaseSelector, timeout: float | None = None
                ) -> list[tuple[selectors.SelectorKey, int]]:
                    nonlocal elapsed
                    ready = original_select(selector, timeout)
                    assert process.wait(timeout=20) == 130
                    elapsed = 6.0
                    return ready

                patch.setattr(selector_type, "select", delayed_select)
                patch.setattr(subprocess, "_time", lambda: original_time() + elapsed)
            # Allow the parent the same collection budget as other PTY tests.
            # The child-side measurement below keeps the five-second exit bound.
            stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 130, (stdout, stderr)
        assert stderr == b"interrupted\n"
        latency = float(stdout.split(b"sigint-to-exit ")[1])
        record_property("sigint_to_exit_s", latency)
        assert latency < 5, (latency, stdout, stderr)
    assert not Path(run_config.run_root).exists()
    assert not Path(run_config.artifact_root).exists()


@pytest.mark.parametrize("window", ["before_prompt", "await_readable"])
def test_prompt_sigint_at_readiness_boundaries(
    inputs: tuple[Path, Path],
    cli_env: dict[str, str],
    run_config: RunConfig,
    window: str,
) -> None:
    code = f"""
import asyncio, os, signal
from swarm_arena._cli import run_cmds
original_print = print
def prompt_print(*args: object, **kwargs: object) -> None:
    if args and args[0] == "Start the run? [y/N] ":
        assert signal.getsignal(signal.SIGINT) is not signal.default_int_handler
        if {window!r} == "before_prompt":
            os.kill(os.getpid(), signal.SIGINT)
    original_print(*args, **kwargs)
from pytest import MonkeyPatch
with MonkeyPatch.context() as patch:
    patch.setattr(run_cmds, "print", prompt_print, raising=False)
    original_wait = run_cmds.wait_readable
    async def wait(fd: int) -> None:
        # Run this only after wait_readable has registered stdin and yielded.
        asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGINT)
        await original_wait(fd)
    if {window!r} == "await_readable":
        patch.setattr(run_cmds, "wait_readable", wait)
    {_ENTRY}
"""
    with _child(["run", *map(str, inputs)], cli_env, terminal=True, code=code) as (
        process,
        _,
    ):
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 130, (stdout, stderr)
        assert stderr == b"interrupted\n"
    assert not Path(run_config.run_root).exists()
    assert not Path(run_config.artifact_root).exists()


@pytest.mark.parametrize("blocking", [False, True])
def test_prompt_sigint_when_terminal_input_is_flushed(
    inputs: tuple[Path, Path], cli_env: dict[str, str], run_config: RunConfig, blocking: bool
) -> None:
    code = f"""
import os, signal, termios
from swarm_arena._cli import run_cmds
from swarm_arena._cli.main import main
os.set_blocking(0, {blocking!r})
original_wait = run_cmds.wait_readable
async def wait(fd: int) -> None:
    await original_wait(fd)
    # Terminal Ctrl-C can discard the input that made the fd readable.
    termios.tcflush(fd, termios.TCIFLUSH)
    os.kill(os.getpid(), signal.SIGINT)
from pytest import MonkeyPatch
with MonkeyPatch.context() as patch:
    patch.setattr(run_cmds, "wait_readable", wait)
    status = main()
    assert os.get_blocking(0) is {blocking!r}
    raise SystemExit(status)
"""
    with _child(["run", *map(str, inputs)], cli_env, terminal=True, code=code) as (
        process,
        master,
    ):
        output = _wait_prompt(process)
        assert master is not None
        os.write(master, b"y\n")
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 130, (output + stdout, stderr)
        assert stderr == b"interrupted\n"
    assert not Path(run_config.run_root).exists()
    assert not Path(run_config.artifact_root).exists()


def _wait_started(process: subprocess.Popen[bytes], root: Path, *, epoch: int) -> LocalStore:
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        assert process.poll() is None, process.communicate()
        for path in root.glob("*"):
            store = LocalStore(path)
            read = read_committed(store)
            if read is not None and any(
                e.type == "tool_start" and e.epoch == epoch for e in read[1]
            ):
                return store
        time.sleep(0.02)
    pytest.fail("no committed tool_start before timeout")


@pytest.mark.parametrize("command", ["run", "resume"])
def test_sigint_during_active_work_cleans_up(
    inputs: tuple[Path, Path],
    cli_env: dict[str, str],
    run_config: RunConfig,
    command: str,
) -> None:
    config_path, plan_path = inputs
    plan_path.write_text(
        plan(
            rounds=2, tool=PyTool({"code": "import time; time.sleep(600)", "grace_s": 0.1})
        ).dumps()
    )
    argv = ["run", "--yes", str(config_path), str(plan_path)]
    root = Path(run_config.run_root)
    if command == "resume":
        # A real interrupted run supplies committed state for the resume process.
        with _child(argv, cli_env) as (process, _):
            store = _wait_started(process, root, epoch=1)
            process.send_signal(signal.SIGINT)
            process.communicate(timeout=25)
        argv = ["resume", str(next(root.iterdir()))]
    with _child(argv, cli_env) as (process, _):
        store = _wait_started(process, root, epoch=2 if command == "resume" else 1)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=25)
        assert process.returncode == 130, (stdout, stderr)
        assert stderr == b"interrupted\n"
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].expires_at <= utc_now()
    read = read_committed(store)
    assert read is not None
    starts = [e for e in read[1] if e.type == "tool_start"]
    assert starts
    for event in starts:
        assert not group_has_live_member(event.payload["started"]["pgid"])
