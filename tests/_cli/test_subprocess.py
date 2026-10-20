"""Exercise the installed CLI boundary in real child processes."""

import os
import pty
import select
import signal
import subprocess
import sys
import time
from collections.abc import Generator
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
run_cmds._run_async = fail
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
    argv: list[str], env: dict[str, str], *, terminal: bool = False
) -> Generator[tuple[subprocess.Popen[bytes], int | None]]:
    master, slave = pty.openpty() if terminal else (None, None)
    try:
        with subprocess.Popen(
            [sys.executable, "-B", "-c", _ENTRY, *argv],
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
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == (130 if interrupt else 1), (output + stdout, stderr)
        assert stderr == (b"interrupted\n" if interrupt else b"run not approved\n")
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
