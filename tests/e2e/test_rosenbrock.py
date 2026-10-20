"""Exercise the shipped example through public APIs and the installed CLI."""

import doctest
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from swarm_arena import Goal, Plan, RunConfig, Swarm
from swarm_arena.board import Candidate
from swarm_arena.scoring import CheckContext
from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "rosenbrock"


def _with_roots(config: RunConfig, root: Path) -> RunConfig:
    return config.model_copy(
        update={
            "run_root": str(root / "runs"),
            "arena_root": str(root / "arena"),
            "artifact_root": str(root / "artifacts"),
        }
    )


def _only_run(config: RunConfig) -> str:
    runs = [path.name for path in Path(config.run_root).iterdir() if path.is_dir()]
    assert len(runs) == 1
    return runs[0]


def _events(config: RunConfig) -> list[dict[str, Any]]:
    # Journal internals are used only to assert committed policy/final-score evidence.
    root = Path(config.run_root) / _only_run(config)
    head = json.loads((root / "control/owner.json").read_text())["head"]
    segments: list[list[dict[str, Any]]] = []
    key = head["segment"]
    while key is not None:
        rows = [json.loads(line) for line in (root / key).read_text().splitlines()]
        key = rows[0]["link"]
        segments.append(rows[1:])
    return [event for segment in reversed(segments) for event in segment]


def _decision_events(config: RunConfig) -> list[dict[str, Any]]:
    return [
        event
        for event in _events(config)
        if event["type"] == "decision" and event["goal_id"] == "rosenbrock"
    ]


async def test_rosenbrock(tmp_path: Path) -> None:
    cfg = _with_roots(RunConfig.load(EXAMPLE / "run.toml"), tmp_path)
    out = await Swarm(cfg).run(Plan.load(EXAMPLE / "plan.toml"))
    sol = json.loads(
        (Path(cfg.run_root) / _only_run(cfg) / "goals/rosenbrock/solution.json").read_text()
    )
    assert sol["status"] == "ok"
    assert len(sol["finalists"]) == 2 == len(out["rosenbrock"])
    final_scores = {
        event["payload"]["score"]["id"]: event["payload"]["score"]
        for event in _events(cfg)
        if event["type"] == "score"
        and event["goal_id"] == "rosenbrock"
        and event["payload"]["round"] == 0
        and event["payload"]["score"]["judge"] == "f"
    }
    assert final_scores
    values = []
    batches = set()
    for finalist in sol["finalists"]:
        scores = [score for score in finalist["scores"] if score["id"] in final_scores]
        assert len(scores) == 1
        score = scores[0]
        assert score == final_scores[score["id"]]
        assert score["candidate_id"] == finalist["candidate"]["id"]
        values.append(score["value"])
        batches.add(score["batch_id"])
    assert values == sorted(values)
    assert len(batches) == 1
    assert len({f["candidate"]["id"] for f in sol["finalists"]}) == 2
    assert {f["candidate"]["team"] for f in sol["finalists"]} <= {"descent", "random"}
    assert [c.id for c in out["rosenbrock"]] == [f["candidate"]["id"] for f in sol["finalists"]]
    decisions = _decision_events(cfg)
    assert decisions
    assert any(event["payload"]["decision"]["prune"] for event in decisions)
    assert all(
        reason for event in decisions for reason in event["payload"]["decision"]["prune"].values()
    )


def test_rosenbrock_cli(tmp_path: Path) -> None:
    example = tmp_path / "rosenbrock"
    shutil.copytree(EXAMPLE, example)
    config = _with_roots(RunConfig.load(example / "run.toml"), tmp_path)
    (example / "run.toml").write_text(config.dumps())
    res = subprocess.run(
        [
            str(Path(sys.executable).with_name("swarm-arena")),
            "run",
            "--yes",
            "rosenbrock/run.toml",
            "rosenbrock/plan.toml",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "done: rosenbrock 2 finalists" in res.stdout
    assert (Path(config.run_root) / _only_run(config) / "goals/rosenbrock/solution.json").is_file()


@pytest.fixture
def rosen(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("rosen", EXAMPLE / "rosen.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "rosen", module)
    spec.loader.exec_module(module)
    return module


def _candidate(data: dict[str, Any] | None) -> Candidate:
    # Bypass output validation to exercise the check's defensive malformed-data boundary.
    output = ToolOutput.model_construct(status=RunStatus.OK, data=data)
    return Candidate(
        id="c",
        run_id="r",
        goal_id="rosenbrock",
        team="descent",
        agent="gd-1",
        round=1,
        request=ToolRequest(),
        output=output,
    )


def test_gradient_descent(rosen: ModuleType) -> None:
    result = rosen.gradient_descent(0.001, 2000, -1.5, 2.0)
    assert set(result) == {"x", "y", "f"}
    assert 0 <= result["f"] < 12.5
    assert rosen.gradient_descent(0.001, 0, 1.0, 1.0) == {"x": 1.0, "y": 1.0, "f": 0.0}


def test_random_search(rosen: ModuleType) -> None:
    result = rosen.random_search(100, 2.0, seed=7)
    assert result == rosen.random_search(100, 2.0, seed=7)
    assert set(result) == {"x", "y", "f"}
    assert abs(result["x"]) <= 2
    assert abs(result["y"]) <= 2
    assert result["f"] == pytest.approx(
        (1 - result["x"]) ** 2 + 100 * (result["y"] - result["x"] ** 2) ** 2
    )


@pytest.mark.parametrize(
    ("data", "passed"),
    [
        ({"x": 10.0, "y": -10.0, "f": 100.0}, True),
        ({"x": 10.01, "y": 0.0, "f": 1.0}, False),
        ({"x": 0.0, "y": -10.01, "f": 1.0}, False),
        (None, False),
        ({}, False),
        ({"x": "secret-result", "y": 0.0, "f": 1.0}, False),
        ({"x": True, "y": 0.0, "f": 1.0}, False),
        *[(dict(x=0.0, y=0.0, f=value), False) for value in (math.inf, math.nan)],
        *[(dict(x=value, y=0.0, f=1.0), False) for value in (math.inf, math.nan)],
        *[(dict(x=0.0, y=value, f=1.0), False) for value in (math.inf, math.nan)],
    ],
)
async def test_bounds_check(rosen: ModuleType, data: dict[str, Any] | None, passed: bool) -> None:
    from unittest.mock import Mock

    result = await rosen.BoundsCheck().check(_candidate(data), Mock(spec=CheckContext))
    assert result.check == "bounds"
    assert result.passed is passed
    assert "secret-result" not in str(result)


def test_bounds_config(rosen: ModuleType) -> None:
    config = rosen.BoundsCheck.Config()
    assert config.bound == 10.0
    with pytest.raises(ValueError, match="frozen_instance"):
        config.bound = 2.0
    with pytest.raises(ValueError, match="extra_forbidden"):
        rosen.BoundsCheck.Config(extra="secret-config")


async def test_bounds_custom_bound(rosen: ModuleType) -> None:
    from unittest.mock import Mock

    check = rosen.BoundsCheck({"bound": 1.0})
    result = await check.check(_candidate({"x": 2.0, "y": 0.0, "f": 1.0}), Mock(spec=CheckContext))
    assert not result.passed


def test_score(rosen: ModuleType) -> None:
    score = rosen.score(Goal(id="rosenbrock", namespace="toy"), _candidate({"f": 0.25}))
    assert score.value == 0.25
    assert score.key == (-0.25,)


@pytest.mark.parametrize("data", [None, {}, {"f": "secret-result"}, {"f": math.inf}])
def test_score_missing_or_invalid_data(rosen: ModuleType, data: dict[str, Any] | None) -> None:
    with pytest.raises(ValueError, match="finite numeric f") as error:
        rosen.score(Goal(id="rosenbrock", namespace="toy"), _candidate(data))
    exc = error.value
    surfaces = [
        str(exc),
        repr(exc),
        str(exc.args),
        str(getattr(exc, "__notes__", [])),
        str(exc.__cause__),
        "".join(traceback.format_exception(exc)),
    ]
    assert all("secret-result" not in surface for surface in surfaces)


def test_example_docstrings(rosen: ModuleType) -> None:
    result = doctest.testmod(rosen)
    assert result.attempted > 0
    assert result.failed == 0
