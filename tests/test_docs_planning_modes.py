"""Required facts for the planning modes guide."""

from _docs import required_facts

REQUIRED_FACTS = [
    "Manual",
    "Together",
    "Autonomous",
    "LLMPlanner",
    "CollaborativePlanner",
    "replanner=",
    "Approve",
    "Feedback",
    "Edited",
    "planner_catalog",
    "settable",
    "never settable",
    "set by the core",
    "validate_plan",
    "max_agents_per_fingerprint",
    "only between rounds",
    "swarm-arena plan",
]


def test_required_facts() -> None:
    required_facts("Planning modes", REQUIRED_FACTS)
