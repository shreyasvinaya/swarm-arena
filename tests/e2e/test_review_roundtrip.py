"""The installed arena CLI supports a complete offline human review roundtrip."""

import csv
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
from _fakes.finished import FinishedRun, run_again

from swarm_arena._scoring.human_review import CSV_HEADER


def _cli(config_path: Path, *args: str) -> str:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [str(Path(sys.executable).parent / "swarm-arena"), *args, "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=environment,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _ids(output: str) -> list[str]:
    return [line.split("\t")[1] for line in output.splitlines() if "\t" in line]


async def test_review_roundtrip(finished_run: FinishedRun, tmp_path: Path) -> None:
    identifier = "2026-10-21T00-00-00Z_abcdef"
    await run_again(finished_run, run_id=identifier)
    cfg = finished_run.config_path
    assert "no scores for human:alice" in await anyio.to_thread.run_sync(
        _cli, cfg, "leaderboard", identifier, "--by", "human:alice"
    )
    directory = tmp_path / "rev"
    await anyio.to_thread.run_sync(
        _cli, cfg, "review", "export", identifier, "--out", str(directory)
    )
    path = directory / "review.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    exported = [row["id"] for row in rows]
    assert len(exported) > 1
    for index, row in enumerate(rows, 1):
        row.update(verdict="accept", score=str(index))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    await anyio.to_thread.run_sync(
        _cli, cfg, "review", "import", identifier, str(path), "--reviewer", "alice"
    )
    human = await anyio.to_thread.run_sync(
        _cli, cfg, "leaderboard", identifier, "--by", "human:alice"
    )
    ordinary = await anyio.to_thread.run_sync(_cli, cfg, "leaderboard", identifier)
    assert _ids(human) == list(reversed(exported))
    assert _ids(ordinary) == exported


@pytest.mark.parametrize(
    ("document", "heading"), [("README.md", "The arena"), ("docs/arena.md", "Command line")]
)
def test_documented_arena_commands_run_as_written(
    tmp_path: Path, document: str, heading: str
) -> None:
    import re
    import shutil
    import signal
    from contextlib import suppress

    root = Path(__file__).resolve().parents[2]
    section = (root / document).read_text().split(f"## {heading}\n", 1)[1]
    block = section.split("```sh\n", 1)[1].split("```", 1)[0]
    example = tmp_path / "examples" / "rosenbrock"
    shutil.copytree(root / "examples" / "rosenbrock", example)
    environment = dict(os.environ, UV_PROJECT=str(root), UV_NO_SYNC="1")
    environment.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        ["sh", "-eu", "-c", block],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=90)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=60)
    assert process.returncode == 0, stderr
    identifier = re.search(r"^run (\S+) at ", stdout, flags=re.MULTILINE)
    assert identifier is not None
    reviews = tmp_path / ".swarm-arena" / f"review-{identifier[1]}" / "review.csv"
    with reviews.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(row["verdict"] == "accept" for row in rows)
    assert "scores as human:alice" in stdout
    assert "(human:alice," in stdout
    assert "human,pinned" in stdout
