"""Verify the shared-arena guide and its bounded command-line workflow."""

import json
from pathlib import Path

from _docs import (
    block_runs,
    docs_bindir,
    make_workdir,
    page_path,
    parse_blocks,
    required_facts,
    run_command,
)

REQUIRED_FACTS = [
    "swarm-arena[s3]",
    "runs/<run-id>.json",
    "reviews/",
    "curation/",
    "control/owner.json",
    "frontier",
    "at most one round",
    "rounds/round-NN.md",
    "digest_tokens",
    "search_learnings",
    "swarm-arena reindex",
    "swarm-arena board serve",
    "swarm-arena board search",
    "swarm-arena board show",
    "swarm-arena learnings pin",
    "swarm-arena learnings hide",
    "swarm-arena learnings add",
    "swarm-arena leaderboard",
    "swarm-arena trace",
    "no shared writer lock",
]


def test_required_facts() -> None:
    required_facts("The arena", REQUIRED_FACTS)


def test_arena_has_no_stale_milestone_or_install_promises() -> None:
    text = page_path("The arena").read_text(encoding="utf-8")
    assert "for M3/M4" not in text
    assert "swarm-arena[s3]" in text


def test_board_serve_is_not_executed(tmp_path: Path) -> None:
    blocks = parse_blocks(page_path("The arena").read_text(encoding="utf-8"))
    servers = [block for block in blocks if "swarm-arena board serve" in block.code]
    assert servers
    assert all(not block_runs(block, workdir=tmp_path, bindir=docs_bindir()) for block in servers)
    commands = "\n".join(block.code for block in blocks if block.kind == "run")
    for command in ("leaderboard", "board search", "board show", "trace", "reindex"):
        assert f"swarm-arena {command}" in commands


def test_arena_example_shares_two_successful_runs(tmp_path: Path) -> None:
    page = page_path("The arena")
    blocks = [
        block
        for block in parse_blocks(page.read_text(encoding="utf-8"))
        if block.kind == "run" and block.lang == "bash"
    ]
    assert len(blocks) == 1
    block = blocks[0]
    workdir = make_workdir(tmp_path, with_examples=True)
    stdout, stderr = run_command(
        ["bash", "-euo", "pipefail", "-c", block.code],
        source=str(page),
        line=block.line,
        workdir=workdir,
        bindir=docs_bindir(),
    )
    assert stderr == ""
    pointers = list(workdir.glob(".swarm-arena/arena/toy/runs/*.json"))
    assert len(pointers) == 2
    run_ids = {json.loads(path.read_text(encoding="utf-8"))["run_id"] for path in pointers}
    assert len(run_ids) == 2
    solutions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in workdir.glob(".swarm-arena/runs/*/goals/rosenbrock/solution.json")
    ]
    assert {solution["run_id"] for solution in solutions} == run_ids
    assert all(
        solution["status"] == "ok" and solution["source_head"] and len(solution["finalists"]) == 2
        for solution in solutions
    )
    rows = [line.split("\t") for line in stdout.splitlines()]
    assert {row[2] for row in rows if len(row) == 6 and row[2] in run_ids} == run_ids
    assert "Rosenbrock baseline" in stdout
    assert "human:alice" in stdout
    assert "toy: 2 runs" in stdout
