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
