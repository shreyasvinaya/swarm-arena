from collections.abc import Sequence
from pathlib import Path

import anyio
import pytest
from _fakes.engine import with_limits
from _fakes.run import WRITE, events, plan, run_dir, wait_tool_started
from _fakes.tools import PyTool

from swarm_arena import RunConfig
from swarm_arena._engine.swarm import Swarm
from swarm_arena._errors import PlanError
from swarm_arena._model.goal import FileSet
from swarm_arena._model.tooling import Estimate


async def test_approve_false_writes_nothing(run_config: RunConfig, tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("frozen original")
    p = plan()
    p = p.model_copy(
        update={"goal": p.goal.model_copy(update={"files": FileSet(globs=["source.txt"])})}
    )
    p._base_dir = tmp_path

    async def refuse(estimate: Estimate, warnings: Sequence[str]) -> bool:
        assert estimate.usd is not None
        return False

    with pytest.raises(RuntimeError, match="not approved"):
        await Swarm(run_config).run(p, approve=refuse)
    assert not await anyio.to_thread.run_sync(Path(run_config.run_root).exists)
    assert not await anyio.to_thread.run_sync(Path(run_config.artifact_root).exists)


async def test_non_asyncio_backend_raises(
    monkeypatch: pytest.MonkeyPatch, run_config: RunConfig
) -> None:
    def refuse() -> None:
        raise RuntimeError("swarm-arena runs on the asyncio backend only")

    monkeypatch.setattr("swarm_arena._engine.run._require_asyncio", refuse)
    with pytest.raises(RuntimeError, match="asyncio"):
        await Swarm(run_config).run(plan())
    assert not await anyio.to_thread.run_sync(Path(run_config.run_root).exists)


def test_trio_backend_raises(run_config: RunConfig) -> None:
    pytest.importorskip("trio")
    with pytest.raises(RuntimeError, match="asyncio backend only"):
        anyio.run(Swarm(run_config).run, plan(), backend="trio")


async def test_stop_creates_epoch_key_and_queued_goals_no_result(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_concurrent_goals=1)
    swarm = Swarm(cfg)
    slow = PyTool({"code": "import time; time.sleep(2); " + WRITE.format("{'data': {'y': 1.0}}")})
    async with anyio.create_task_group() as tg:
        tg.start_soon(swarm.run, [plan(goal_id="a", tool=slow), plan(goal_id="b")])
        await wait_tool_started(cfg)
        await swarm.stop()
    assert (run_dir(cfg) / "control/stop-e1").exists()
    assert {
        e.goal_id: e.payload["reason"] for e in events(cfg, "no_result") if e.goal_id == "b"
    } == {"b": "stopped"}
    assert [e.payload["now"] for e in events(cfg, "stop_requested")] == [False]


async def test_non_human_origin_is_kept_and_checked(run_config: RunConfig) -> None:
    with pytest.raises(PlanError, match="planner catalog"):
        await Swarm(run_config).run(plan().model_copy(update={"origin": "llm"}))


async def test_approved_snapshot_promotes_original_bytes(
    run_config: RunConfig, tmp_path: Path
) -> None:
    from swarm_arena._store.artifacts import ArtifactStore
    from swarm_arena._store.base import open_store

    source = tmp_path / "source.txt"
    source.write_bytes(b"original bytes")
    p = plan(rounds=1)
    p = p.model_copy(
        update={"goal": p.goal.model_copy(update={"files": FileSet(globs=["source.txt"])})}
    )
    p._base_dir = tmp_path

    async def approve(estimate: Estimate, warnings: Sequence[str]) -> bool:
        source.write_bytes(b"changed after snapshot")
        assert not await anyio.to_thread.run_sync(Path(run_config.artifact_root).exists)
        return True

    await Swarm(run_config).run(p, approve=approve)
    frozen = events(run_config, "plan_revision")[0].payload["plan"]["goal"]["files"]["manifest"]
    assert (
        ArtifactStore(open_store(run_config.artifact_root)).get_bytes(frozen["source.txt"])
        == b"original bytes"
    )


async def test_stop_without_run_refuses(run_config: RunConfig) -> None:
    with pytest.raises(RuntimeError, match="no run has started"):
        await Swarm(run_config).stop()


async def test_stop_now_is_journaled_once(run_config: RunConfig) -> None:
    from _fakes.run import WRITE_THEN_SLEEP

    swarm = Swarm(run_config)
    async with anyio.create_task_group() as tg:
        tg.start_soon(swarm.run, plan(tool=PyTool({"code": WRITE_THEN_SLEEP, "grace_s": 0.1})))
        await wait_tool_started(run_config)
        await swarm.stop(now=True)
        await swarm.stop(now=True)
    assert [e.payload["now"] for e in events(run_config, "stop_requested")] == [True]


async def test_each_new_stop_mode_is_journaled(run_config: RunConfig) -> None:
    from _fakes.run import WRITE_THEN_SLEEP

    swarm = Swarm(run_config)
    with anyio.fail_after(20):
        async with anyio.create_task_group() as tg:
            tg.start_soon(swarm.run, plan(tool=PyTool({"code": WRITE_THEN_SLEEP, "grace_s": 0.1})))
            await wait_tool_started(run_config)
            await swarm.stop()
            while not await anyio.to_thread.run_sync(events, run_config, "stop_requested"):  # noqa: ASYNC110 - committed journal signal
                await anyio.sleep(0.02)
            await swarm.stop(now=True)
    assert [e.payload["now"] for e in events(run_config, "stop_requested")] == [False, True]
