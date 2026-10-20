from swarm_arena._engine.control import stop_key


def test_stop_key() -> None:  # tests/_engine/test_control.py
    assert (stop_key(2, now=False), stop_key(2, now=True)) == (
        "control/stop-e2",
        "control/stop-now-e2",
    )
