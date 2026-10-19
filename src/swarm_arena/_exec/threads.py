"""Run blocking work in daemon threads that the event loop awaits."""

import contextlib
import contextvars
import threading
from collections.abc import Callable
from typing import Any, TypeVar

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
    runs on, and its outcome is discarded. If the event loop has finished by
    the time the thread ends, the hand-back is skipped.

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
    done = anyio.Event()
    token = anyio.lowlevel.current_token()
    context = contextvars.copy_context()
    results: list[T] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            results.append(context.run(fn, *args))
        except BaseException as exc:  # handed to the awaiting task, which re-raises it
            errors.append(exc)
        # RunFinishedError or RuntimeError: the loop has finished, so nobody is waiting.
        with contextlib.suppress(Exception):
            anyio.from_thread.run_sync(done.set, token=token)

    name = f"swarm-arena-daemon-{getattr(fn, '__name__', 'work')}"
    threading.Thread(target=work, name=name, daemon=True).start()
    await done.wait()
    if errors:
        raise errors[0]
    return results[0]
