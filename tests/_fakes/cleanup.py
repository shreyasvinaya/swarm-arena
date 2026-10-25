"""Observe test-owned task cleanup without waiting for cancellation acknowledgement."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import monotonic
from types import TracebackType
from typing import Any, TypeVar

import anyio

T = TypeVar("T")


async def settle_tasks(
    *tasks: asyncio.Task[Any], timeout_s: float = 60, cancel: bool = True
) -> None:
    """Cancel pending tasks and observe completion under an independent deadline.

    Only completed tasks are gathered. A cleanup failure retains an already
    escaping test failure in its exception chain; callers put thread joins and
    process reaping in a nested finally so neither failure skips those owners.
    """
    owned = set(tasks)
    if not owned:
        return
    for task in owned:
        if cancel and not task.done() and not task.cancelling():
            task.cancel()
    with anyio.CancelScope(shield=True):
        done, pending = await asyncio.wait(owned, timeout=timeout_s)
        await asyncio.gather(*done, return_exceptions=True)
    assert not pending, "test-owned task cleanup did not finish"


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """Join test-owned pool threads under a deadline even if a future fails."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.shutdown(wait=False, cancel_futures=True)
        deadline = monotonic() + 60
        for worker in self._threads:
            worker.join(max(0, deadline - monotonic()))
        if any(worker.is_alive() for worker in self._threads):
            message = "test-owned pool cleanup did not finish"
            if exc_val is not None:
                exc_val.add_note(message)
            else:
                raise AssertionError(message)
        return False


async def observe_task(task: asyncio.Task[T], timeout_s: float = 60) -> T:
    """Observe a task without letting a watchdog wait for its cancellation cleanup."""
    done, _ = await asyncio.wait({task}, timeout=timeout_s)
    assert done, "test-owned task did not finish"
    return task.result()
