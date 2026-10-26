"""Pin the concepts and importable Python plan example."""

from pathlib import Path

from _docs import docs_bindir, make_workdir, page_path, parse_blocks, required_facts, run_blocks


def test_required_facts() -> None:
    required_facts(
        "Concepts",
        [
            "Goal",
            "RunConfig",
            "Plan",
            "Tool",
            "Check",
            "Judge",
            "Manager",
            "Planner",
            "Policy",
            "Executor",
            "Board",
            "Curator",
            "Swarm",
            "import path",
            "A lambda or local function is rejected",
            "GoalKey",
            "Plan.parts",
            "stable address",
            "model catalog",
            "on by default",
            "shorter than the provider's minimum",
            "response cache",
            "nothing goes to any vendor",
        ],
    )


def test_python_plan_uses_importable_components(tmp_path: Path) -> None:
    blocks = parse_blocks(page_path("Concepts").read_text(encoding="utf-8"))
    files = [block for block in blocks if block.kind == "file"]
    assert [block.arg for block in files] == ["concepts_tools.py"]
    examples = [block for block in blocks if "print(plan.dumps())" in block.code]
    assert len(examples) == 1
    assert "FunctionTool(" in examples[0].code
    assert "ScriptedManager(" in examples[0].code
    assert "Ref.of(lambda x: x)" in examples[0].code
    run_blocks(
        [files[0], examples[0]],
        source="docs/concepts.md",
        workdir=make_workdir(tmp_path, with_examples=False),
        bindir=docs_bindir(),
    )
