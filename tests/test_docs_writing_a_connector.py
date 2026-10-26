"""Verify connector guidance with the installed target-interpreter harness."""

from pathlib import Path

from _docs import docs_bindir, make_workdir, page_path, parse_blocks, required_facts, run_blocks

REQUIRED_FACTS: dict[str, list[str]] = {
    "Writing a connector": [
        "swarm_arena.testing",
        "check_tool",
        "check_check",
        "check_judge",
        "check_manager",
        "FakeModel",
        "preflight()",
        "does no I/O",
        "is_deterministic",
        "editable",
        "estimate",
        "providers",
        "fingerprint",
        "extra_actions",
        "never imported",
        "executor=RayExecutor(",
    ],
}


def test_required_facts() -> None:
    for title, facts in REQUIRED_FACTS.items():
        required_facts(title, facts)


def test_connector_examples_run(tmp_path: Path) -> None:
    path = page_path("Writing a connector")
    blocks = parse_blocks(path.read_text(encoding="utf-8"))
    files = {block.arg for block in blocks if block.kind == "file"}
    assert {"echo_tool.py", "nonempty_check.py", "tests/test_conformance.py"} <= files
    assert any("anyio.run(check_tool, EchoTool()," in block.code for block in blocks)
    assert any("check_check(NonemptyCheck()," in block.code for block in blocks)
    run_blocks(
        blocks,
        source="docs/writing-a-connector.md",
        workdir=make_workdir(tmp_path, with_examples=False),
        bindir=docs_bindir(),
    )
