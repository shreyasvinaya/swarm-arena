from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _engine.test_barrier_recovery import _calls, _history

CHECKOUT = Path(__file__).parents[2]


def invoke(mode: str, root: Path, scenario: str, top: int, cutoff: int = 0) -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "_fakes.pr1_r3_final",
            mode,
            str(root),
            scenario,
            str(top),
            str(cutoff),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(CHECKOUT / "tests"), str(CHECKOUT / "src")]),
        },
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert process.returncode == (91 if mode == "crash" else 0), process.stdout + process.stderr


@pytest.mark.parametrize(("scenario", "top"), [("reverse", 2), ("reverse", 1)])
def test_all_final_journal_heads(tmp_path: Path, scenario: str, top: int) -> None:
    control = tmp_path / "control"
    invoke("run", control, scenario, top)
    expected = json.loads((control / "summary.json").read_text())
    history = _history(control)
    start = next(e.seq for e in history if e.type == "converge_started")
    end = next(e.seq for e in history if e.type == "goal_finished")
    heads = sorted(
        {json.loads(line)["head"] for line in (control / "heads.jsonl").read_text().splitlines()}
        & set(range(start, end + 1))
    )
    failures: list[dict[str, Any]] = []
    for head in heads:
        root = tmp_path / f"crash-{head}"
        invoke("crash", root, scenario, top, head)
        before = _history(root)
        completed = [e.payload["final_batch"] for e in before if "final_batch" in e.payload]
        previous = _calls(root) if (root / "judge.log").exists() else []
        invoke("resume", root, scenario, top)
        actual = json.loads((root / "summary.json").read_text())
        repeated = bool(completed and actual["calls"] != previous)
        bad = (
            repeated
            or actual["final"] != expected["final"]
            or (completed and actual["charges"] != expected["charges"])
        )
        detail = {
            "head": head,
            "completed": completed,
            "before_calls": previous,
            "actual": actual,
            "expected": expected,
            "repeated_completed": repeated,
        }
        (root / "comparison.json").write_text(json.dumps(detail, indent=2))
        if bad:
            failures.append(detail)
    assert not failures, json.dumps({"heads": heads, "failures": failures}, indent=2)
