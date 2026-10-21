"""Run the shipped script example and inspect committed candidate provenance.

The private journal reader is used only to inspect the authoritative committed
candidate records; execution and candidate validation use public interfaces.
"""

import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from swarm_arena import Plan, Ref, RunConfig, Swarm
from swarm_arena._journal.reader import read_committed
from swarm_arena._store.local import LocalStore
from swarm_arena.board import Candidate
from swarm_arena.tool import RunStatus

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "own_script"


def _with_roots(cfg: RunConfig, root: Path) -> RunConfig:
    return cfg.model_copy(
        update={
            "run_root": str(root / "runs"),
            "arena_root": str(root / "arena"),
            "artifact_root": str(root / "artifacts"),
        }
    )


def _run_root(cfg: RunConfig) -> Path:
    [root] = [path for path in Path(cfg.run_root).iterdir() if path.is_dir()]
    return root


def _candidate_of(cfg: RunConfig, *, agent: str, round: int) -> Candidate:
    committed = read_committed(LocalStore(_run_root(cfg)))
    assert committed is not None
    [candidate] = [
        Candidate.model_validate(event.payload["candidate"])
        for event in committed[1]
        if event.type == "tool_end" and event.agent == agent and event.round == round
    ]
    return candidate


def _solution(cfg: RunConfig, goal_id: str) -> bytes:
    return (_run_root(cfg) / "goals" / goal_id / "solution.json").read_bytes()


def _tool_workspaces(root: Path) -> list[Path]:
    return [
        request.parent.parent
        for request in root.rglob("swarm/request.json")
        if not (request.parent.parent / "candidate").exists()
    ]


async def test_own_script(tmp_path: Path) -> None:
    cfg = _with_roots(RunConfig.load(EXAMPLE / "run.toml"), tmp_path).model_copy(
        update={
            "executor": Ref(
                path="swarm_arena.executor:LocalExecutor",
                config={"workspace_root": str(tmp_path / "ws"), "keep_workspaces": True},
            )
        }
    )
    result = await Swarm(cfg).run(Plan.load(EXAMPLE / "plan.toml"))
    c1 = _candidate_of(cfg, agent="tuner-1", round=1)
    c2 = _candidate_of(cfg, agent="tuner-2", round=2)
    assert set(c2.overlay) == {"problem/model_a.txt", "problem/model_b.txt"}
    assert c2.overlay["problem/model_a.txt"] == sha256(b"slope = 2.0\n").hexdigest()
    assert c2.overlay["problem/model_b.txt"] == sha256(b"intercept = 1.0\n").hexdigest()
    assert c2.parent_ids == [c1.id]
    assert c1.tool_identity == c2.tool_identity
    assert c1.output.data == {"slope": 2.0, "intercept": 0.0, "fit_error": 1.0}
    assert c2.output.data == {"slope": 2.0, "intercept": 1.0, "fit_error": 0.0}
    assert c1.output.status is c2.output.status is RunStatus.OK
    assert all(check.passed for check in c1.checks + c2.checks)
    assert c1.checks
    assert c2.checks
    hidden = (EXAMPLE / "judge/hidden/points.csv").read_bytes()
    workspaces = _tool_workspaces(tmp_path / "ws")
    assert workspaces
    for ws in workspaces:
        assert not (ws / "judge").exists()
        assert not (ws / "checks").exists()
        assert (ws / "problem/points.txt").read_bytes() != hidden
        assert all(p.read_bytes() != hidden for p in ws.rglob("*") if p.is_file())
    assert json.loads(_solution(cfg, "line-fit"))["status"] == "ok"
    assert [candidate.id for candidate in result["line-fit"]] == [c2.id]
    solution = json.loads(_solution(cfg, "line-fit"))
    assert solution["finalists"][0]["scores"][0]["value"] == 0.0


def _script(root: Path, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script],
        cwd=root,
        env={**os.environ, "SWARM_ARENA_RESULT": str(root / "result.json")},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_script_reads_params_and_assignment_syntax(tmp_path: Path) -> None:
    shutil.copytree(EXAMPLE / "problem", tmp_path / "problem")
    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/request.json").write_text('{"params": {"scale": 2.0}}')
    (tmp_path / "problem/model_a.txt").write_text(" # model\n slope = 1e0 # comment\n")
    (tmp_path / "problem/model_b.txt").write_text("intercept = 0\n")
    done = _script(tmp_path, "problem/run.py")
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    assert json.loads((tmp_path / "result.json").read_bytes()) == {
        "data": {"slope": 2.0, "intercept": 0.0, "fit_error": 1.0}
    }


@pytest.mark.parametrize(
    "assignment",
    ['slope = "PRIVATE_MODEL_SENTINEL"', "slope = nan", "slope = true", "wrong = 1.0"],
)
def test_script_rejects_bad_models_without_echoing_values(tmp_path: Path, assignment: str) -> None:
    shutil.copytree(EXAMPLE / "problem", tmp_path / "problem")
    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/request.json").write_text('{"params": {"scale": 1.0}}')
    (tmp_path / "problem/model_a.txt").write_text(assignment)
    done = _script(tmp_path, "problem/run.py")
    assert done.returncode != 0
    assert "PRIVATE_MODEL_SENTINEL" not in done.stdout + done.stderr
    assert done.stderr.strip() == "invalid line-fit inputs"
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    ("data", "passed"),
    [
        ({"slope": 2.0, "intercept": 1.0, "fit_error": 0.0}, True),
        ({}, False),
        ({"slope": True, "intercept": 1.0, "fit_error": 0.0}, False),
        ({"slope": 2.0, "intercept": 1.0, "fit_error": -1.0}, False),
        ({"slope": 2.0, "intercept": "PRIVATE_CHECK_SENTINEL", "fit_error": 0.0}, False),
    ],
)
def test_check_validates_candidate_data(tmp_path: Path, data: dict[str, Any], passed: bool) -> None:
    shutil.copytree(EXAMPLE / "checks", tmp_path / "checks")
    (tmp_path / "candidate").mkdir()
    (tmp_path / "candidate/output.json").write_text(json.dumps({"data": data}))
    done = _script(tmp_path, "checks/check.py")
    assert done.returncode == 0, done.stderr
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["passed"] is passed
    assert bool(result["reasons"]) is not passed
    assert "PRIVATE_CHECK_SENTINEL" not in json.dumps(result) + done.stdout + done.stderr


def test_judge_scores_private_points_instead_of_public_error(tmp_path: Path) -> None:
    shutil.copytree(EXAMPLE / "judge", tmp_path / "judge")
    (tmp_path / "candidate").mkdir()
    (tmp_path / "candidate/output.json").write_text(
        json.dumps({"data": {"slope": 2.0, "intercept": 0.0, "fit_error": 12345.0}})
    )
    done = _script(tmp_path, "judge/score.py")
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "result.json").read_bytes()) == {"value": 1.0}
