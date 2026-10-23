# ruff: noqa: E402
# Check the optional runtime before importing its modules.
"""Recover an actual Ray task after its dispatching CLI driver dies."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _fakes.dependencies import ray_dependency
from _fakes.tools import WORKER_SCRIPTS
from e2e.test_crash_resume import _committed_events, _wait_for, _wait_run_dir

from swarm_arena import (
    AgentSpec,
    Goal,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    ScriptedManager,
    ScriptTool,
    TeamSpec,
)
from tests.ray.conftest import ray_driver as ray_driver

ray_dependency()

# isort: split
from swarm_arena.ray import RayExecutor
from swarm_arena.ray._state import PENDING_STATES, find_task

pytestmark = pytest.mark.ray


def test_driver_killed_after_dispatch_then_resume(
    ray_cluster: str, ray_driver: Any, tmp_path: Path, run_config: RunConfig
) -> None:
    (tmp_path / "ignore_sigterm.py").write_text(WORKER_SCRIPTS["ignore_sigterm.py"])
    plan = Plan(
        goal=Goal(id="g", namespace="dispatch", files={"globs": ["ignore_sigterm.py"]}),
        teams=[
            TeamSpec(
                name="t",
                rationale="Recovery",
                agents=[
                    AgentSpec(
                        name="a",
                        rationale="Recovery",
                        tool=Ref.of(
                            ScriptTool(command=[sys.executable, "ignore_sigterm.py"], grace_s=1.0)
                        ),
                        manager=Ref.of(ScriptedManager([{"inputs": {}}])),
                    )
                ],
            )
        ],
        rounds=[RoundSpec()],
        judges=[Ref.of(ProgramJudge("_fakes.tools:score_min_y"))],
    )
    config = run_config.model_copy(
        update={
            "executor": Ref.of(RayExecutor()),
            "lease_ttl_s": 3.0,
            "limits": run_config.limits.model_copy(update={"tool_timeout_s": 60}),
        }
    )
    run_file, plan_file = tmp_path / "run.toml", tmp_path / "plan.toml"
    run_file.write_text(config.dumps())
    plan_file.write_text(plan.dumps())
    env = {**os.environ, "RAY_ADDRESS": ray_cluster, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"}
    cli = [sys.executable, str(Path(sys.executable).with_name("swarm-arena"))]
    log = tmp_path / "driver.log"
    with log.open("w") as output:
        driver = subprocess.Popen(
            [*cli, "run", "--yes", str(run_file), str(plan_file)],
            cwd=tmp_path,
            env=env,
            stdout=output,
            stderr=output,
            text=True,
        )
        try:
            run_dir = _wait_run_dir(driver, log)
            _wait_for(lambda: any(e.type == "tool_dispatch" for e in _committed_events(run_dir)))
            intent = next(e for e in _committed_events(run_dir) if e.type == "tool_dispatch")
            started = intent.payload["started"]
            name, driver_id = started["task_name"], started["ray_job_id"]

            def running() -> bool:
                row = find_task(
                    name, dashboard_url=ray_driver.dashboard_url, expected_job_id=driver_id
                )
                return row is not None and row.state not in PENDING_STATES

            _wait_for(running, 60)
            driver.send_signal(signal.SIGKILL)
            assert driver.wait(timeout=60) == -signal.SIGKILL
            resumed = subprocess.run(
                [*cli, "resume", str(run_dir), "--force"],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert resumed.returncode == 0, resumed.stdout + resumed.stderr
            events = _committed_events(run_dir)
            lost = next(e for e in events if e.type == "run_lost")
            assert lost.payload["candidate_id"] == started["job_id"]
            assert lost.payload["confirmed_dead"] is True
            assert lost.payload["task_name"] == name
            assert (
                find_task(name, dashboard_url=ray_driver.dashboard_url, expected_job_id=driver_id)
                is not None
            )
            retries = [e for e in events if e.type == "tool_dispatch" and e.epoch == 2]
            assert len(retries) == 1
            assert retries[0].payload["started"]["ray_job_id"] != driver_id
            ends = [e for e in events if e.type == "tool_end" and e.epoch == 2]
            assert len(ends) == 1
            assert ends[0].payload["candidate"]["output"]["status"] == "timeout"
            assert (
                json.loads((run_dir / "goals/g/solution.json").read_text())["schema_version"] == 1
            )
        finally:
            if driver.poll() is None:
                driver.kill()
            driver.wait(timeout=60)
