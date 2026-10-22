"""CLI interruption ownership preserves cleanup and caller cancellation."""

import asyncio

import anyio
import pytest

from swarm_arena._cli.run_cmds import _interrupt_owner, _interruptible


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
    asyncio.current_task().uncancel()
    assert cleaned
    assert _interrupt_owner.get() is None


async def test_caller_cancellation_is_preserved() -> None:
    with anyio.move_on_after(0) as scope:
        await _interruptible(anyio.sleep_forever)
    assert scope.cancel_called
    assert _interrupt_owner.get() is None
