"""Run blocking work in daemon threads that the event loop awaits."""

import contextlib
import contextvars
import threading
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import anyio
import anyio.from_thread
import anyio.lowlevel

T = TypeVar("T")


async def run_sync_daemon(fn: Callable[..., T], *args: Any) -> T:
    """
    Run a blocking callable in a new daemon thread and await its outcome.

    Use it for work that may never return, such as a tool's ``collect``: a
    daemon thread cannot keep the interpreter from exiting, unlike the worker
    threads of ``anyio.to_thread.run_sync``. When the awaiting task is
    cancelled, it stops waiting at once and the thread is dropped; the thread
    runs on, and its outcome is discarded. The hand-back never waits for the
    event loop: the thread ends as soon as ``fn`` returns, even while a loop
    that is still open is not running. If the event loop has finished by the
    time the thread ends, the hand-back is skipped.

    ``fn`` runs in a copy of the caller's context
    (``contextvars.copy_context()``), as with ``anyio.to_thread.run_sync``:
    context variables set by the caller are visible, and changes made in the
    thread stay in the copy. Unlike an anyio worker thread, the daemon thread
    has no way back into the event loop: ``anyio.from_thread`` is not
    available in it, and its calls raise ``NoEventLoopError``.

    Parameters
    ----------
    fn : Callable[..., T]
        Blocking callable to run.
    *args : Any
        Positional arguments for ``fn``.

    Returns
    -------
    T
        What ``fn`` returned.

    Raises
    ------
    BaseException
        Whatever ``fn`` raised, re-raised in the awaiting task.

    Examples
    --------
    >>> import anyio
    >>> anyio.run(run_sync_daemon, sum, [1, 2, 3])
    6
    >>> import contextvars
    >>> user = contextvars.ContextVar("user", default="nobody")
    >>> async def main() -> str:
    ...     user.set("ada")
    ...     return await run_sync_daemon(user.get)
    >>> anyio.run(main)
    'ada'
    """
    return await _DaemonCall(fn, *args).result()


def _post(token: anyio.lowlevel.EventLoopToken, callback: Callable[[], object]) -> None:
    """Schedule ``callback`` on the token's loop without waiting for it to run."""
    native = token.native_token
    schedule = getattr(native, "call_soon_threadsafe", None)  # asyncio
    if schedule is None:
        schedule = getattr(native, "run_sync_soon", None)  # trio
    if schedule is None:
        anyio.from_thread.run_sync(callback, token=token)
    else:
        schedule(callback)


class _DaemonCall(Generic[T]):
    """Retain one daemon call's outcome so cancelled waiters can rejoin it."""

    def __init__(self, fn: Callable[..., T], *args: Any) -> None:
        self._done = anyio.Event()
        self._results: list[T] = []
        self._errors: list[BaseException] = []
        token = anyio.lowlevel.current_token()
        context = contextvars.copy_context()

        def work() -> None:
            try:
                self._results.append(context.run(fn, *args))
            except BaseException as exc:
                self._errors.append(exc)
            # A late result after loop shutdown is intentionally discarded.
            with contextlib.suppress(Exception):
                _post(token, self._done.set)

        name = f"swarm-arena-daemon-{getattr(fn, '__name__', 'work')}"
        threading.Thread(target=work, name=name, daemon=True).start()

    async def result(self) -> T:
        """Await the original outcome without starting work again."""
        await self._done.wait()
        if self._errors:
            raise self._errors[0]
        return self._results[0]
