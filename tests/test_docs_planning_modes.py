"""Required facts for the planning modes guide."""

from _docs import page_path, required_facts

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


def test_retained_scoring_rules_name_their_scope() -> None:
    # The retained-scoring rule skips human revisions and plans without history.
    text = " ".join(page_path("Planning modes").read_text(encoding="utf-8").split())
    assert "For non-human revisions with admitted history" in text
    assert "New parts inherit main scoring." not in text
