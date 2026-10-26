"""Required safety statements and complete limits coverage."""

from _docs import page_path, required_facts

from swarm_arena import Limits

REQUIRED_FACTS = [
    "This is not strong isolation",
    "Without a hard cap, the worst-case overspend is the error in the estimates of runs in flight",
    "Total spend can never go over the run cap because of a crash",
    "reserve_fraction",
    "final slice",
    "tranche",
    "watchdog_stopped",
    "rlimits",
    "--yes",
]


def test_required_facts() -> None:
    required_facts("Limits and safety", REQUIRED_FACTS)


def test_limits_page_lists_every_limit() -> None:
    text = page_path("Limits and safety").read_text(encoding="utf-8")
    assert [field for field in Limits.model_fields if f"`{field}`" not in text] == []
