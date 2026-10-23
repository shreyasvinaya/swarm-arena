"""Review storage uses the same authority and cancellation boundary as the engine."""

import asyncio
import threading

import anyio
import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import Fenced, StorageError, _core_error
from swarm_arena._scoring.human_review import _review_io


@pytest.mark.parametrize("kind", ["ordinary", "cancel", "group", "fenced", "core"])
async def test_review_io_classifies_backend_outcomes(kind: str) -> None:
    secret = "private-review-backend-marker"
    errors: dict[str, BaseException] = {
        "ordinary": OSError(secret),
        "cancel": asyncio.CancelledError(secret),
        "group": BaseExceptionGroup(secret, [asyncio.CancelledError(secret), ValueError(secret)]),
        "fenced": Fenced(secret),
        "core": _core_error(Fenced, "run ownership lost"),
    }

    def fail() -> None:
        raise errors[kind]

    with pytest.raises(Fenced if kind == "core" else StorageError) as caught:
        await _review_io(fail)
    assert_secret_absent(caught.value, secret)


async def test_review_io_preserves_caller_cancellation() -> None:
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []

    def block() -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(180)

    task = asyncio.create_task(_review_io(block))
    try:
        with anyio.fail_after(60):
            assert await anyio.to_thread.run_sync(entered.wait, 60)
        task.cancel()
        with pytest.raises(asyncio.CancelledError), anyio.fail_after(30):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()
