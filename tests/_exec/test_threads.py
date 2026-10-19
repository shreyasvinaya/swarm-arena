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
    finished = threading.Event()

    def blocked() -> None:
        release.wait(10)
        finished.set()

    with anyio.move_on_after(0.2) as scope:
        await run_sync_daemon(blocked)
    assert scope.cancelled_caught
    assert not finished.is_set()  # the awaiting side returned while the thread still ran
    release.set()
    assert await anyio.to_thread.run_sync(finished.wait, 5)


def test_hand_back_after_the_loop_closed_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))
    release = threading.Event()
    threads: list[threading.Thread] = []

    def blocked() -> int:
        threads.append(threading.current_thread())
        release.wait(10)
        return 1

    async def main() -> None:
        with anyio.move_on_after(0.2):
            await run_sync_daemon(blocked)

    anyio.run(main)  # the loop is closed when this returns
    release.set()
    [thread] = threads
    thread.join(5)
    assert not thread.is_alive()
    assert errors == []
