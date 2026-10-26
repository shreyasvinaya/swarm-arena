"""Required facts for the scoring and review guide."""

from _docs import required_facts

REQUIRED_FACTS = [
    "gated_out",
    "Plan.ranking",
    "rubric_hash",
    "panel_hash",
    "anchor",
    "final judge",
    "NaN",
    "ProgramJudge",
    "Lexicographic",
    "ScriptJudge",
    "LLMJudge",
    "temperature 0",
    "review.csv",
    "--top 20",
    "--reviewer alice",
    "human:alice",
    "--by human:alice",
    "NO_RESULT",
    "KeepTopTeams",
    "SuccessiveHalving",
    "Patience",
    "Chain",
    "only passing candidate",
    "starved",
    "last team",
]


def test_required_facts() -> None:
    required_facts("Scoring and review", REQUIRED_FACTS)
