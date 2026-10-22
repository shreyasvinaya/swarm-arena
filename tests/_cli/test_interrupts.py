"""CLI interruption ownership preserves cleanup and caller cancellation."""

import asyncio
import signal

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
