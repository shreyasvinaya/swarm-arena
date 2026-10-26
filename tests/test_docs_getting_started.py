"""Pin the first offline run, results, and review exchange in the tutorial."""

import csv
import json
from pathlib import Path

from _docs import docs_bindir, make_workdir, page_path, parse_blocks, required_facts, run_blocks


def test_required_facts() -> None:
    required_facts(
        "Getting started",
        [
            "POSIX",
            "Python 3.11",
            "hello world",
            "swarm-arena run --yes run.toml plan.toml",
            "swarm-arena leaderboard hello",
            "swarm-arena review export",
            "swarm-arena review import",
            "gradient_descent(lr, steps, x0, y0)",
            "random_search(n, radius, seed)",
            "deterministic = false",
            "Score.minimize(f)",
            "SuccessiveHalving",
            "ScriptedManager",
            "AgentsSDKManager",
            "solution.json",
            "source_head",
            "llm",
            "cache",
            "ray",
            "s3",
            "langfuse",
        ],
    )


def test_hello_world_run_view_share(tmp_path: Path) -> None:
    text = page_path("Getting started").read_text(encoding="utf-8")
    headings = ["## Run hello-world", "## View results", "## Share a review", "## Rosenbrock"]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    blocks = parse_blocks(text[: positions[-1]])
    assert [(block.lang, block.arg) for block in blocks if block.kind == "file"] == [
        ("toml", "run.toml"),
        ("toml", "plan.toml"),
    ]
    first_run = next(block for block in blocks if block.kind == "run")
    assert "swarm-arena run --yes run.toml plan.toml" in first_run.code
    viewer = [block for block in blocks if "swarm-arena board serve hello" in block.code]
    assert len(viewer) == 1
    assert viewer[0].lang == "sh"
    assert viewer[0].kind == "none"

    workdir = make_workdir(tmp_path, with_examples=False)
    run_blocks(blocks, source="docs/getting-started.md", workdir=workdir, bindir=docs_bindir())
    solutions = list((workdir / ".swarm-arena/runs").glob("*/goals/hello/solution.json"))
    assert len(solutions) == 1
    solution = json.loads(solutions[0].read_text(encoding="utf-8"))
    assert solution["status"] == "ok"
    assert solution["source_head"]
    assert len(solution["finalists"]) == 1
    candidate = solution["finalists"][0]["candidate"]
    assert candidate["output"]["status"] == "ok"
    assert candidate["output"]["data"] == {"message": "hello world"}
    run_id = (workdir / "hello-run-id.txt").read_text(encoding="utf-8").strip()
    assert run_id == solution["run_id"]
    assert candidate["id"] in (workdir / "hello-leaderboard.txt").read_text(encoding="utf-8")
    assert "imported 1 scores as human:demo" in (workdir / "hello-import.txt").read_text(
        encoding="utf-8"
    )
    leaderboard = (workdir / "hello-human-leaderboard.txt").read_text(encoding="utf-8")
    assert candidate["id"] in leaderboard
    ranking_row = next(line for line in leaderboard.splitlines() if candidate["id"] in line)
    assert ranking_row.split("\t")[4] == "1"
    with (workdir / "review-bundle/review.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["verdict"] == "accept"
    assert rows[0]["score"] == "1"
    assert "human:demo" in leaderboard


def test_rosenbrock_examples_run(tmp_path: Path) -> None:
    text = page_path("Getting started").read_text(encoding="utf-8")
    _, rosenbrock = text.split("## Rosenbrock", 1)
    for filename in ("rosen.py", "plan.toml", "run.toml"):
        assert f"```{{literalinclude}} ../examples/rosenbrock/{filename}" in rosenbrock
    workdir = make_workdir(tmp_path, with_examples=True)
    run_blocks(
        parse_blocks(rosenbrock),
        source="docs/getting-started.md (Rosenbrock)",
        workdir=workdir,
        bindir=docs_bindir(),
    )
    solutions = list((workdir / ".swarm-arena/runs").glob("*/goals/rosenbrock/solution.json"))
    assert len(solutions) == 2  # CLI and Python API each run the plan.
    for path in solutions:
        solution = json.loads(path.read_text(encoding="utf-8"))
        assert solution["status"] == "ok"
        assert solution["source_head"]
        assert len(solution["finalists"]) == 2
        assert all(
            finalist["candidate"]["output"]["status"] == "ok" for finalist in solution["finalists"]
        )
