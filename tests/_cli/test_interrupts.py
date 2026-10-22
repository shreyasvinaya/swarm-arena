"""CLI interruption ownership preserves cleanup and caller cancellation."""

import asyncio
import signal
import subprocess
import sys
from pathlib import Path

import anyio
import pytest

from swarm_arena._cli.run_cmds import _interrupt_owner, _interruptible, _sigint_wakeup


async def test_repeated_interrupt_keeps_shielded_cleanup_alive() -> None:
    cleaned = False

    async def operation() -> None:
        nonlocal cleaned
        interrupt = _interrupt_owner.get()
        assert interrupt is not None
        try:
            interrupt()
            await anyio.sleep_forever()
        finally:
            with anyio.CancelScope(shield=True):
                interrupt()
                await anyio.lowlevel.checkpoint()
                cleaned = True

    with pytest.raises(KeyboardInterrupt):
        await _interruptible(operation)
    task = asyncio.current_task()
    assert task is not None
    task.uncancel()
    assert cleaned
    assert _interrupt_owner.get() is None


async def test_caller_cancellation_is_preserved() -> None:
    with anyio.move_on_after(0) as scope:
        await _interruptible(anyio.sleep_forever)
    assert scope.cancel_called
    assert _interrupt_owner.get() is None


async def test_nested_prompt_and_grace_restore_outer_callback() -> None:
    calls = []
    arrived = anyio.Event()

    def outer() -> None:
        calls.append("outer")
        arrived.set()

    def grace() -> None:
        calls.append("grace")
        arrived.set()

    async def send() -> None:
        signal.raise_signal(signal.SIGINT)
        with anyio.fail_after(2):
            await arrived.wait()

    before = signal.getsignal(signal.SIGINT)
    with _sigint_wakeup(outer):
        with _sigint_wakeup():
            await send()
        arrived = anyio.Event()
        with _sigint_wakeup(grace):
            await send()
        arrived = anyio.Event()
        await send()
    assert calls == ["outer", "grace", "outer"]
    assert signal.getsignal(signal.SIGINT) is before


def test_third_interrupt_finishes_runner_once(tmp_path: Path) -> None:
    code = """
import asyncio, gc, json, signal
import anyio
from swarm_arena._cli.run_cmds import _interruptible

errors = []
async def operation():
    loop = asyncio.get_running_loop()
    def report(loop, context):
        errors.append(context["message"])
        loop.default_exception_handler(context)
    loop.set_exception_handler(report)
    loop.call_soon(signal.raise_signal, signal.SIGINT)
    try:
        await anyio.sleep_forever()
    finally:
        with anyio.CancelScope(shield=True):
            loop.call_soon(signal.raise_signal, signal.SIGINT)
            loop.call_soon(signal.raise_signal, signal.SIGINT)
            await anyio.sleep_forever()

caught = 0
try:
    anyio.run(_interruptible, operation, backend="asyncio")
except KeyboardInterrupt:
    caught += 1
gc.collect()
print(json.dumps({
    "caught": caught,
    "restored": signal.getsignal(signal.SIGINT) is signal.default_int_handler,
    "errors": errors,
}))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    import json

    assert json.loads(result.stdout) == {"caught": 1, "restored": True, "errors": []}
    assert "never retrieved" not in result.stderr
    assert "Traceback" not in result.stderr


async def test_nested_owner_restore_never_removes_signal_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    remove = loop.remove_signal_handler
    removed = []

    def observe(sig: int) -> bool:
        removed.append(sig)
        return remove(sig)

    def interrupt() -> None:
        pass

    monkeypatch.setattr(loop, "remove_signal_handler", observe)
    before = signal.getsignal(signal.SIGINT)
    with _sigint_wakeup(interrupt):
        with _sigint_wakeup(interrupt):
            pass
        assert removed == []
    assert removed == [signal.SIGINT]
    assert signal.getsignal(signal.SIGINT) is before


@pytest.mark.parametrize("command", ["run", "resume"])
def test_third_interrupt_during_settlement_exits(tmp_path: Path, command: str) -> None:
    import os
    import select
    import time

    code = f"""
import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import anyio
from _fakes.run import plan
from swarm_arena import RunConfig, Swarm
from swarm_arena._budget.ledger import Ledger
from swarm_arena._cli import run_cmds
from swarm_arena._cli.main import main
from swarm_arena._engine.round import EngineDeps
from swarm_arena._engine.run import RunRunner
from swarm_arena._model.results import Candidate

root = Path.cwd()
config = RunConfig(run_root=str(root / 'runs'), artifact_root=str(root / 'artifacts'),
                   arena_root=str(root / 'arena'), settle_timeout_s=2)
example = plan(rounds=1)
(root / 'run.toml').write_text(config.dumps())
(root / 'plan.toml').write_text(example.dumps())
if {command!r} == 'resume':
    queue = RunRunner._queue
    # Persist a queued goal so resume exercises real recovered dispatch and settlement.
    async def queued(self: RunRunner, deps: EngineDeps) -> dict[str, list[Candidate]]:
        return {{}}
    RunRunner._queue = queued
    swarm = Swarm(config)
    anyio.run(swarm.run, example)
    RunRunner._queue = queue
    argv = ['resume', str(root / 'runs' / swarm.run_id)]
else:
    argv = ['run', '--yes', str(root / 'run.toml'), str(root / 'plan.toml')]

first_interrupt = anyio.Event()
wakeup = run_cmds._sigint_wakeup
@contextmanager
def observed_wakeup(on_interrupt: Callable[[], None] | None = None) -> Iterator[None]:
    count = 0
    def observed() -> None:
        nonlocal count
        count += 1
        try:
            assert on_interrupt is not None
            on_interrupt()
        finally:
            if count == 1:
                first_interrupt.set()
            print('received-' + str(count), flush=True)
    with wakeup(observed if on_interrupt is not None else None):
        yield
run_cmds._sigint_wakeup = observed_wakeup
settle = Ledger._settle
async def held_settle(self: Ledger, *args: Any, **kwargs: Any) -> None:
    try:
        with anyio.CancelScope(shield=True):
            print('settlement-ready', flush=True)
            await first_interrupt.wait()
    finally:
        try:
            with anyio.CancelScope(shield=True):
                print('cleanup-ready', flush=True)
                await anyio.sleep_forever()
        finally:
            with anyio.CancelScope(shield=True):
                print('settlement-resumed', flush=True)
                await settle(self, *args, **kwargs)
Ledger._settle = held_settle
handler = signal.getsignal(signal.SIGINT)
status = main(argv)
print('handler-restored=' + str(signal.getsignal(signal.SIGINT) is handler), flush=True)
raise SystemExit(status)
"""
    process = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", code],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        start_new_session=True,
    )
    stdout = process.stdout
    assert stdout is not None
    output = bytearray()

    def until(marker: bytes, timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        while marker not in output:
            assert time.monotonic() < deadline, bytes(output)
            if select.select([stdout], [], [], 0.05)[0]:
                chunk = os.read(stdout.fileno(), 65536)
                assert chunk, bytes(output)
                output.extend(chunk)

    try:
        until(b"settlement-ready")
        process.send_signal(signal.SIGINT)
        until(b"cleanup-ready")
        process.send_signal(signal.SIGINT)
        until(b"received-2")
        deadline = time.monotonic() + 15
        process.send_signal(signal.SIGINT)
        until(b"received-3", timeout=15)
        try:
            rest, _ = process.communicate(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pytest.fail(f"third interrupt did not exit: {bytes(output)!r}")
        output.extend(rest)
        assert process.returncode == 130, bytes(output)
        assert b"settlement-resumed" in output
        assert b"handler-restored=True" in output
        assert b"interrupted" in output
        assert b"never retrieved" not in output
    finally:
        from contextlib import suppress

        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0)
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        stdout.close()
