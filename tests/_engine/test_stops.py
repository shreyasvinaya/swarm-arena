from types import SimpleNamespace

import anyio
import pytest

from swarm_arena._engine import stops as stops_module
from swarm_arena._engine.stops import RunClock, StopState, round_deadline, watch_stop_files
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
        stops_module, "anyio", SimpleNamespace(to_thread=anyio.to_thread, sleep=after_poll)
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
