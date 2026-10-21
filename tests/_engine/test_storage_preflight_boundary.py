"""Storage preflight has no settlement deadline and keeps safe cancellation."""

import asyncio
import threading
from typing import Any

import anyio
import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import RunConfig
from swarm_arena._engine import preflight
from swarm_arena._errors import StorageError


async def test_storage_preflight_does_not_use_settlement_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[bool] = []

    class Backend:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        def fail_after(self, delay: float) -> Any:
            pytest.fail("storage probing must not borrow the settlement deadline")

    monkeypatch.setattr(preflight, "anyio", Backend())
    monkeypatch.setattr(preflight, "check_storage", lambda config: probed.append(True))
    await preflight._check_storage(RunConfig(settle_timeout_s=0.001))
    assert probed == [True]


async def test_storage_preflight_failure_is_type_only(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "private-storage-preflight-marker"

    def fail(config: RunConfig) -> None:
        raise OSError(secret)

    monkeypatch.setattr(preflight, "check_storage", fail)
    with pytest.raises(StorageError, match="storage preflight failed: OSError") as caught:
        await preflight._check_storage(RunConfig())
    assert_secret_absent(caught.value, secret)


async def test_storage_preflight_keeps_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []

    def block(config: RunConfig) -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(20)

    monkeypatch.setattr(preflight, "check_storage", block)
    task = asyncio.create_task(preflight._check_storage(RunConfig()))
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(entered.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError), anyio.fail_after(2):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()
