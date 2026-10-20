import anyio

from swarm_arena._engine.stops import RunClock, StopState, round_deadline, watch_stop_files
from swarm_arena._store.local import LocalStore


# tests/_engine/test_stops.py
async def test_poll_reads_only_own_epoch(store: LocalStore) -> None:
    store.create("control/stop-e1", b"")
    st = StopState()
    with anyio.move_on_after(0.3):
        await watch_stop_files(store, 2, st, interval_s=0.05)
    assert not st.stop
    store.create("control/stop-now-e2", b"")
    with anyio.move_on_after(0.3):
        await watch_stop_files(store, 2, st, interval_s=0.05)
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
