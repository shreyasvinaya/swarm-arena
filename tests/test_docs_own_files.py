"""Verify the file workspace guide and its complete offline workflows."""

import json
from pathlib import Path

from _docs import docs_bindir, make_workdir, page_path, parse_blocks, required_facts, run_blocks

REQUIRED_FACTS: dict[str, list[str]] = {
    "Use your own files and script": [
        "stable format",
        "swarm/request.json",
        "swarm/result.json",
        "SWARM_ARENA_RESULT",
        "parents/<candidate-id>/",
        "candidate/request.json",
        "candidate/output.json",
        "candidate/files/",
        "swarm/llm_calls.jsonl",
        "SWARM_ARENA_CANDIDATE_ID",
        "overlay rule",
        "max_edit_bytes",
        "256 KiB",
        "never editable",
        "Symlinks are rejected",
    ],
}


def test_required_facts() -> None:
    for title, facts in REQUIRED_FACTS.items():
        required_facts(title, facts)


def test_toml_workflow_runs_and_prints_solution(tmp_path: Path) -> None:
    path = page_path("Use your own files and script")
    blocks = parse_blocks(path.read_text(encoding="utf-8"))
    assert any(
        block.kind == "run"
        and block.lang == "bash"
        and "swarm-arena run --yes examples/own_script/run.toml examples/own_script/plan.toml"
        in block.code
        for block in blocks
    )
    assert any(
        block.lang == "python"
        and block.kind == "none"
        and 'glob.glob("**/.swarm-arena/runs/*/goals/*/solution.json", recursive=True)'
        in block.code
        and "os.path.getmtime" in block.code
        and "print(" in block.code
        for block in blocks
    )
    workdir = make_workdir(tmp_path, with_examples=True)
    run_blocks(blocks, source="docs/own-files.md", workdir=workdir, bindir=docs_bindir())
    [solution_path] = workdir.glob("**/.swarm-arena/runs/*/goals/*/solution.json")
    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    assert solution["source_head"]
    assert solution["status"] == "ok"
    [finalist] = solution["finalists"]
    candidate = finalist["candidate"]
    assert candidate["output"]["status"] == "ok"
    assert candidate["output"]["data"] == {"slope": 2.0, "intercept": 1.0, "fit_error": 0.0}
    assert candidate["checks"]
    assert all(check["passed"] for check in candidate["checks"])
    assert finalist["scores"]
    assert all(score["key"] == [0.0] for score in finalist["scores"])
