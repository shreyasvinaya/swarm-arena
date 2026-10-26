"""Keep the public roadmap within the reserved design scope."""

from _docs import REPO, page_path, required_facts

REQUIRED_FACTS = [
    "Direct and broadcast messages",
    "Live sharing with no barrier",
    "Outside agents joining",
    "Forum",
    "Ray-sandbox executors",
    "Pareto ranking",
    "A database arena backend",
    "Nested teams",
    "Mid-run release of unused gateway reservations",
    "DeepRetro connector",
    "physika",
    "Release 1.0 after both connectors have run real campaigns",
]


def test_required_facts() -> None:
    required_facts("Roadmap", REQUIRED_FACTS)


def test_roadmap_has_exactly_the_reserved_future_features() -> None:
    spec = (REPO / "docs/superpowers/specs/2026-10-16-swarm-arena-design.md").read_text(
        encoding="utf-8"
    )
    section = spec.split("## 10.", 1)[1].split("## 11.", 1)[0]
    expected = [
        line.split("|")[1].strip()
        for line in section.splitlines()
        if line.startswith("| ") and not line.split("|")[1].strip().startswith("---")
    ]
    text = page_path("Roadmap").read_text(encoding="utf-8")
    actual = [
        line.split("|")[1].strip()
        for line in text.splitlines()
        if line.startswith("| ") and not line.split("|")[1].strip().startswith("---")
    ]
    assert actual == expected
    for reservation in (
        "Plan.flows",
        "context_id",
        "task_id",
        "LLMCall.parent_call_id",
        "Plan.sharing",
        "Learning.tags",
        "Executor",
        "Score.key",
        "Board",
        "schema_version",
        "terminal_accounting",
    ):
        assert reservation in text
