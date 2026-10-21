import asyncio
import threading
from types import SimpleNamespace

import anyio
import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._engine import stops as stops_module
from swarm_arena._engine.stops import RunClock, StopState, round_deadline, watch_stop_files
from swarm_arena._errors import Fenced, StorageError, _core_error, _is_core_signal
from swarm_arena._store.local import LocalStore


async def test_poll_reads_only_own_epoch(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_poll, second_poll, next_poll = anyio.Event(), anyio.Event(), anyio.Event()
    polls = 0

    async def after_poll(interval_s: float) -> None:
        nonlocal polls
        assert interval_s == 0.05
        polls += 1
        if polls == 1:
            first_poll.set()
            await next_poll.wait()
        else:
            second_poll.set()
            await anyio.sleep_forever()

    monkeypatch.setattr(
        stops_module, "anyio", SimpleNamespace(lowlevel=anyio.lowlevel, sleep=after_poll)
    )
    store.create("control/stop-e1", b"")
    store.create("control/stop-now-e1", b"")
    st = StopState()
    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:

            async def poll() -> None:
                await watch_stop_files(store, 2, st, interval_s=0.05)

            tasks.start_soon(poll)
            await first_poll.wait()
            assert not st.stop
            assert not st.stop_now
            store.create("control/stop-now-e2", b"")
            next_poll.set()
            await second_poll.wait()
            tasks.cancel_scope.cancel()
    assert st.stop_now
    assert st.stop


def test_gate_reasons() -> None:
    st = StopState(goal_budget={"g"})
    err = st.gate_for("g")()
    assert err is not None
    assert (err.limit, str(err)) == ("budget", "goal budget used up")
    assert st.gate_for("h")() is None
    st.stop_now = True
    assert (err := st.gate_for("h")()) is not None
    assert err.limit == "stop"


def test_run_clock_counts_only_running_time() -> None:
    assert RunClock(elapsed_before=10.0).elapsed() >= 10.0
    assert RunClock(elapsed_before=86_399.0).left(86_400.0) <= 1.0


def test_round_deadline_single_time_basis() -> None:
    assert round_deadline(1000.0, 5.0, 2.0) == 1002.0
    assert round_deadline(1000.0, 5.0, 86_400.0) == 1005.0


@pytest.mark.parametrize("flag", ["stop", "stop_now", "run_timeout", "run_budget"])
async def test_run_stop_latch_wakes_scheduler_once(flag: str) -> None:
    changed = anyio.Event()
    notifications: list[str] = []

    def notify() -> None:
        notifications.append(flag)
        changed.set()

    state = StopState(on_change=notify)
    setattr(state, flag, True)
    with anyio.fail_after(1):
        await changed.wait()
    setattr(state, flag, True)
    assert notifications == [flag]


@pytest.mark.parametrize("key", ["control/stop-now-e2", "control/stop-e2"])
async def test_cancelled_poll_discards_late_true(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    entered, release, exited = (threading.Event() for _ in range(3))
    calls: list[str] = []
    workers: list[threading.Thread] = []

    def exists(current: str) -> bool:
        calls.append(current)
        if current != key:
            return False
        workers.append(threading.current_thread())
        entered.set()
        try:
            assert release.wait(20)
            return True
        finally:
            exited.set()

    monkeypatch.setattr(store, "exists", exists)
    state = StopState()
    task = asyncio.create_task(watch_stop_files(store, 2, state))
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(entered.wait, 8)
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=2)
        assert done, "polling cancellation waited for storage"
        assert task.cancelled()
        assert not state.stop
        assert not state.stop_now
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            assert await anyio.to_thread.run_sync(exited.wait, 5)
            for worker in workers:
                if worker.name.startswith("swarm-arena-daemon-"):
                    await anyio.to_thread.run_sync(worker.join, 5)
                    assert not worker.is_alive()
    assert not state.stop
    assert not state.stop_now
    assert calls == ([key] if "now" in key else ["control/stop-now-e2", key])


@pytest.mark.parametrize("key", ["control/stop-now-e2", "control/stop-e2"])
@pytest.mark.parametrize("kind", ["fence", "group_fence", "storage"])
async def test_poll_rebuilds_registered_outcomes(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch, key: str, kind: str
) -> None:
    secret = "private-mutated-stop-outcome"
    error = _core_error(StorageError if kind == "storage" else Fenced, "safe stored diagnostic")
    error.args = (secret,)
    error.add_note(secret)
    error.__cause__ = OSError(secret)
    raised: BaseException = error
    if kind == "group_fence":
        raised = BaseExceptionGroup(secret, [asyncio.CancelledError(secret), error])

    def exists(current: str) -> bool:
        if current == key:
            raise raised
        return False

    monkeypatch.setattr(store, "exists", exists)
    with pytest.raises(StorageError if kind == "storage" else Fenced) as caught:
        await watch_stop_files(store, 2, StopState())
    assert caught.value is not error
    assert _is_core_signal(caught.value)
    assert_secret_absent(caught.value, secret)
    assert str(caught.value) == (
        "run ownership lost" if kind != "storage" else "safe stored diagnostic"
    )
