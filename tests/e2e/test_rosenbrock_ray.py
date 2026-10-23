# ruff: noqa: E402
# Check the optional runtime before importing its modules.
"""Run the shipped plan and stopped-output contract through Ray's object store."""

import json
from dataclasses import replace
from pathlib import Path

import anyio
import pytest
from _fakes.dependencies import ray_dependency
from _fakes.journals import committed_events
from _fakes.ray_executor import cancel_and_capture, partial_job
from _fakes.ray_transport import isolated_ray_filesystem  # noqa: F401

from swarm_arena import Plan, Ref, RunConfig, Swarm
from swarm_arena.board import ArenaBoard
from swarm_arena.tool import RunStatus

ray_dependency()

# isort: split
from swarm_arena.ray import RayExecutor

pytestmark = [pytest.mark.ray, pytest.mark.usefixtures("isolated_ray_filesystem")]
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "rosenbrock"


async def test_rosenbrock_on_ray(tmp_path: Path) -> None:
    config = RunConfig.load(EXAMPLE / "run.toml").model_copy(
        update={
            "run_root": str(tmp_path / "runs"),
            "arena_root": str(tmp_path / "arena"),
            "artifact_root": str(tmp_path / "artifacts"),
            "executor": Ref(path="swarm_arena.ray:RayExecutor"),
        }
    )
    with anyio.fail_after(180):
        finalists = await Swarm(config).run(Plan.load(EXAMPLE / "plan.toml"))
    runs = await anyio.to_thread.run_sync(lambda: list(Path(config.run_root).iterdir()))
    assert len(runs) == 1
    root = runs[0]
    solution = json.loads((root / "goals/rosenbrock/solution.json").read_text())
    assert solution["status"] == "ok"
    assert len(finalists["rosenbrock"]) == len(solution["finalists"]) == 2
    assert [candidate.id for candidate in finalists["rosenbrock"]] == [
        row["candidate"]["id"] for row in solution["finalists"]
    ]
    types = {event["type"] for event in committed_events(root)}
    assert "tool_dispatch" in types
    assert "tool_start" not in types
    board = ArenaBoard(arena_root=config.arena_root)
    try:
        board.open("", "", ["toy"])
        board.frontier()
        candidates = [
            board.get_candidate(event["payload"]["candidate"]["id"])
            for event in committed_events(root)
            if event["type"] == "tool_end"
        ]
    finally:
        board.close()
    assert candidates
    assert {candidate.id for candidate in candidates} == {
        event["payload"]["candidate_id"]
        for event in committed_events(root)
        if event["type"] == "tool_dispatch"
    }
    for candidate in candidates:
        assert candidate.output.status == RunStatus.OK
        assert candidate.output.logs_ref is not None
        assert (root / "goals/rosenbrock/logs" / f"{candidate.id}.out").is_file()
        assert (root / "goals/rosenbrock/logs" / f"{candidate.id}.err").is_file()


async def test_ray_partial_is_data_only(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    executor = RayExecutor()
    artifacts = tmp_path / "partial-artifacts"
    job = partial_job(ready, artifact_root=str(artifacts))
    assert job.tool is not None
    job = replace(
        job,
        tool=Ref(
            path="_fakes.ray_transport:PartialFilesTool",
            config={**job.tool.config, "grace_s": 30.0},
        ),
        grace_s=30.0,
    )
    try:
        partial, _ = await cancel_and_capture(executor, job, ready=ready)
    finally:
        await executor.aclose()
    assert partial.status == RunStatus.CANCELLED
    assert partial.partial
    assert partial.data["step"] >= 1
    assert partial.data["declared_files"] == 1
    assert partial.files == []
    assert partial.logs_ref is None
    assert partial.llm_calls_file is None
    assert not artifacts.exists() or not list(artifacts.rglob("*"))
