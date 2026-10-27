"""Require every offline LLM check to execute in the upstream canary."""

import os
from collections.abc import Generator
from pathlib import Path

import pytest

_LLM_ROOT = Path(__file__).parent.resolve()


def _required(path: Path) -> bool:
    return "SWARM_ARENA_UPSTREAM_CANARY" in os.environ and path.resolve().is_relative_to(_LLM_ROOT)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Fail skipped setup, calls, teardown and xfails after other report hooks."""
    report = yield
    if _required(item.path) and report.skipped:
        report.outcome = "failed"
        if hasattr(report, "wasxfail"):
            del report.wasxfail
        report.longrepr = f"upstream canary forbids skipped checks: {report.nodeid}"
    return report


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    """Fail import-time skips before pytest counts the collection report."""
    report = yield
    if _required(collector.path) and report.skipped:
        report.outcome = "failed"
        report.longrepr = f"upstream canary forbids skipped imports: {report.nodeid}"
    return report


def pytest_deselected(items: list[pytest.Item]) -> None:
    """Allow only the explicit credentialed live-test exclusion in this canary."""
    # The weekly job has no live credentials; pyproject.toml excludes llm_live.
    # Ubuntu's normal CPython supports all current platform-dependent LLM tests.
    # A selected live test that skips is still a failure, as is any platform skip.
    if any(
        _required(item.path)
        and not (
            item.path == _LLM_ROOT / "test_manager.py"
            and item.name == "test_live_manager_round"
            and item.get_closest_marker("llm_live") is not None
        )
        for item in items
    ):
        pytest.exit("upstream canary forbids deselected checks", returncode=1)
