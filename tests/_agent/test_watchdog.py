import pytest

from swarm_arena import Limits
from swarm_arena._agent.watchdog import Watchdog
from swarm_arena._model.tooling import RunStatus


def test_refused_repeats_over_limit_stop() -> None:
    w = Watchdog(Limits())
    for _ in range(3):
        w.record_refusal()
    assert w.reason is None
    w.record_refusal()
    assert (w.reason, w.limit) == (
        "refused a repeated request more than 3 times",
        "max_refused_repeats",
    )


def test_consecutive_failures_count_error_oom_timeout_only() -> None:
    w = Watchdog(Limits())
    for s in [
        RunStatus.ERROR,
        RunStatus.PROVIDER_FAILED,
        RunStatus.LOST,
        RunStatus.CANCELLED,
        RunStatus.BUDGET_EXCEEDED,
        RunStatus.OOM,
    ]:
        w.record_run(s, passed=False)
    assert w.reason is None
    w.record_run(RunStatus.TIMEOUT, passed=False)
    assert (w.reason, w.limit) == ("3 consecutive failures", "max_consecutive_failures")


def test_ok_resets_failures() -> None:
    w = Watchdog(Limits())
    w.record_run(RunStatus.ERROR, passed=False)
    w.record_run(RunStatus.TIMEOUT, passed=False)
    w.record_run(RunStatus.OK, passed=False)
    w.record_run(RunStatus.OOM, passed=False)
    w.record_run(RunStatus.CANCELLED, passed=False)
    w.record_run(RunStatus.ERROR, passed=False)
    assert w.reason is None
    w.record_run(RunStatus.ERROR, passed=False)
    assert w.reason == "3 consecutive failures"


def test_idle_rounds_skip_starved_rounds() -> None:
    w = Watchdog(Limits())
    w.record_run(RunStatus.OK, passed=False)
    assert w.end_round(ran=True) is None
    assert w.end_round(ran=False) is None
    w.record_run(RunStatus.OK, passed=False)
    assert w.end_round(ran=True) == "no new passing candidate in 2 rounds"
    assert w.limit == "watchdog_idle_rounds"


def test_passing_candidate_resets_idle_rounds_and_round_flag() -> None:
    w = Watchdog(Limits())
    assert w.end_round(ran=True) is None
    w.record_run(RunStatus.OK, passed=True)
    w.record_run(RunStatus.OK, passed=False)
    assert w.end_round(ran=True) is None
    assert w.end_round(ran=True) is None
    assert w.end_round(ran=True) == "no new passing candidate in 2 rounds"


@pytest.mark.parametrize("status", [RunStatus.ERROR, RunStatus.OOM, RunStatus.TIMEOUT])
def test_failures_use_configured_threshold(status: RunStatus) -> None:
    w = Watchdog(Limits(max_consecutive_failures=1))
    w.record_run(status, passed=False)
    assert (w.reason, w.limit) == ("1 consecutive failures", "max_consecutive_failures")


def test_configured_limits_and_first_reason_is_retained() -> None:
    w = Watchdog(Limits(max_refused_repeats=1, watchdog_idle_rounds=1))
    w.record_refusal()
    assert w.reason is None
    assert w.limit is None
    w.record_refusal()
    w.record_run(RunStatus.TIMEOUT, passed=False)
    w.end_round(ran=True)
    assert (w.reason, w.limit) == (
        "refused a repeated request more than 1 times",
        "max_refused_repeats",
    )
