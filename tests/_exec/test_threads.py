import contextvars
import threading

import anyio
import pytest

from swarm_arena._exec.threads import run_sync_daemon


async def test_returns_the_result_from_a_daemon_thread() -> None:
    seen: list[tuple[bool, bool]] = []

    def work(a: int, b: int) -> int:
        thread = threading.current_thread()
        seen.append((thread.daemon, thread is threading.main_thread()))
        return a + b

    assert await run_sync_daemon(work, 2, 3) == 5
    assert seen == [(True, False)]


async def test_reraises_the_exception() -> None:
    def broken() -> None:
        raise ValueError("broke")

    with pytest.raises(ValueError, match="broke"):
        await run_sync_daemon(broken)


async def test_cancel_stops_waiting_and_drops_the_thread() -> None:
    release = threading.Event()
    entered = threading.Event()
    finished = threading.Event()
    workers: list[threading.Thread] = []

    def blocked() -> None:
        workers.append(threading.current_thread())
        entered.set()
        try:
            assert release.wait(180), "test did not release the held thread"
        finally:
            finished.set()

    async def cancel_when_entered(scope: anyio.CancelScope) -> None:
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        scope.cancel()

    try:
        with anyio.fail_after(30):
            with anyio.CancelScope() as scope:
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(cancel_when_entered, scope)
                    await run_sync_daemon(blocked)
        assert scope.cancelled_caught
        assert not finished.is_set()  # the awaiting side returned while the thread still ran
    finally:
        release.set()
        try:
            assert await anyio.to_thread.run_sync(finished.wait, 60)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
                assert not worker.is_alive()


def test_hand_back_after_the_loop_closed_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))
    release = threading.Event()
    entered = threading.Event()
    threads: list[threading.Thread] = []

    def blocked() -> int:
        threads.append(threading.current_thread())
        entered.set()
        assert release.wait(180), "test did not release the held thread"
        return 1

    async def main() -> None:
        async def cancel_when_entered(scope: anyio.CancelScope) -> None:
            assert await anyio.to_thread.run_sync(entered.wait, 60)
            scope.cancel()

        with anyio.fail_after(30):
            with anyio.CancelScope() as scope:
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(cancel_when_entered, scope)
                    await run_sync_daemon(blocked)

    try:
        anyio.run(main)  # the loop is closed when this returns
    finally:
        release.set()
        for thread in threads:
            thread.join(60)
    [thread] = threads
    assert not thread.is_alive()
    assert errors == []


def test_hand_back_never_waits_for_an_idle_loop() -> None:
    import asyncio

    release = threading.Event()
    entered = threading.Event()
    threads: list[threading.Thread] = []

    def blocked() -> int:
        threads.append(threading.current_thread())
        entered.set()
        assert release.wait(180), "test did not release the held thread"
        return 1

    async def start_and_abandon() -> None:
        call = asyncio.ensure_future(run_sync_daemon(blocked))
        assert await asyncio.to_thread(entered.wait, 60)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(start_and_abandon())
        # The loop stays open but no longer runs, as between two library runs.
        release.set()
        [thread] = threads
        thread.join(60)
        assert not thread.is_alive(), "the daemon thread waited for an idle loop"
    finally:
        release.set()
        loop.run_until_complete(asyncio.sleep(0))
        for thread in threads:
            thread.join(60)
        loop.close()


_VALUE: contextvars.ContextVar[str] = contextvars.ContextVar("value", default="unset")


async def test_runs_in_a_copy_of_the_callers_context() -> None:
    def work() -> str:
        seen = _VALUE.get()
        _VALUE.set("changed in the thread")
        return seen

    token = _VALUE.set("from the caller")
    try:
        assert await run_sync_daemon(work) == "from the caller"
        assert _VALUE.get() == "from the caller"  # a copy: the thread's change stays there
    finally:
        _VALUE.reset(token)


async def test_from_thread_is_not_available_in_the_daemon_thread() -> None:  # the docstring says so
    import anyio.from_thread

    with pytest.raises(anyio.NoEventLoopError):
        await run_sync_daemon(anyio.from_thread.run_sync, lambda: 1)
