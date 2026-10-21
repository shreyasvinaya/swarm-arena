"""The installed arena CLI supports a complete offline human review roundtrip."""

import csv
import os
import subprocess
import sys
from pathlib import Path

import anyio
from _fakes.finished import FinishedRun, run_again

from swarm_arena._scoring.human_review import CSV_HEADER


def _cli(config_path: Path, *args: str) -> str:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [str(Path(sys.executable).parent / "swarm-arena"), *args, "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=30,
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
