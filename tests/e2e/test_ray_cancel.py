# ruff: noqa: E402
# Check the optional runtime before importing its modules.
"""Exercise actual soft and force cancellation through the public executor."""

from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.dependencies import ray_dependency
from _fakes.proc import pid_alive
from _fakes.ray_executor import cancel_and_capture, partial_job, tool_job

from tests.ray.conftest import ray_driver as ray_driver

ray_dependency()

# isort: split
from swarm_arena.ray import RayExecutor
from swarm_arena.ray._state import worker_dead
from swarm_arena.tool import RunStatus

pytestmark = pytest.mark.ray


async def test_soft_cancel_gives_partial_output(ray_driver: Any, tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    out, _ = await cancel_and_capture(
        RayExecutor(),
        partial_job(ready, artifact_root=str(tmp_path / "artifacts")),
        ready=ready,
    )
    assert out.status == RunStatus.CANCELLED
    assert out.partial
    assert out.data["step"] >= 1
    assert out.files == []
    assert out.logs_ref is None
    assert out.llm_calls_file is None


async def test_force_cancel_kills_grandchildren(ray_driver: Any, tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    job = tool_job(
        "",
        tool="StallCollectTool",
        inputs={"pid_file": str(pid_file)},
        artifact_root=str(tmp_path / "artifacts"),
        grace_s=1.0,
    )
    executor = RayExecutor(cancel_margin_s=1.0)
    out, elapsed = await cancel_and_capture(executor, job, ready=pid_file)
    assert elapsed >= 2.0
    with anyio.fail_after(60):
        while (  # noqa: ASYNC110 -- external Ray state
            await anyio.to_thread.run_sync(
                lambda: worker_dead(
                    job.id,
                    dashboard_url=ray_driver.dashboard_url,
                    expected_job_id=executor._driver_job_id,
                )
            )
            is not True
        ):
            await anyio.sleep(0.1)
    pid = int(await anyio.Path(pid_file).read_text())
    with anyio.fail_after(60):
        while pid_alive(pid):  # noqa: ASYNC110 -- external process liveness
            await anyio.sleep(0.05)
    assert out.status == RunStatus.CANCELLED
    assert not out.partial
