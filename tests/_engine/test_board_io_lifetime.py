"""Lease owners stop while storage and connector workers remain blocked."""

import asyncio
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import anyio
import pytest
from _engine.test_goal import _curation_checkpoint, _plan
from _engine.test_round import _round, _runner
from _fakes.engine import EngineEnv

from swarm_arena._board.arena import ArenaView
from swarm_arena._engine.goal import GoalRunner
from swarm_arena._outcomes import _has_core_fenced


@pytest.mark.parametrize(
    "site",
    [
        "frontier",
        "snapshot",
        "sibling",
        "parents",
        "curator_frontier",
        "curator_snapshot",
        "page_exists",
        "page_create",
    ],
)
@pytest.mark.parametrize("signal", ["stop_now", "fence", "cancel"])
async def test_blocked_io_does_not_retain_goal(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, site: str, signal: str
) -> None:
    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 2.0})
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    loop_thread = threading.get_ident()
    page_key = "goals/g/rounds/round-01.md"
    if site.startswith("curator_"):
        runner = await _curation_checkpoint(env)
        operation: Coroutine[Any, Any, Any] = runner._curate_round(1)
        target, method = env.deps.board, site.removeprefix("curator_")
    elif site.startswith("page_"):
        runner = GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1, rounds=1), replanner=None)
        operation = runner.run()
        target, method = env.store, site.removeprefix("page_")
    else:
        round_runner = _round(
            env,
            [_runner("a")],
            converge=site == "parents",
            seed_parents=("absent",) if site == "parents" else (),
        )
        operation = round_runner.run(1)
        target, method = (
            (ArenaView, "_sibling_view")
            if site == "sibling"
            else (env.deps.board, "snapshot" if site == "parents" else site)
        )
    original = getattr(target, method)

    def blocked(*args: Any, **kwargs: Any) -> Any:
        if site.startswith("page_") and args[0] != page_key:
            return original(*args, **kwargs)
        workers.append(threading.current_thread())
        entered.set()
        try:
            assert threading.get_ident() != loop_thread
            assert release.wait(20), "controlled worker release deadline"
            return original(*args, **kwargs)
        finally:
            exited.set()

    monkeypatch.setattr(target, method, blocked)
    task = asyncio.create_task(operation)
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(entered.wait, 10)
        if signal == "cancel":
            task.cancel()
        elif signal == "fence":
            env.deps.journal.fenced.set()
        else:
            env.deps.stops.stop = env.deps.stops.stop_now = True
        done, _ = await asyncio.wait([task], timeout=7)
        assert done, f"{site} retained its owner after {signal}"
        assert not exited.is_set()
        if signal == "fence":
            error = task.exception()
            assert error is not None
            assert _has_core_fenced(error)
        assert not env.events("curated")
        if site == "page_exists":
            assert not (Path(env.store.url) / page_key).exists()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        assert await anyio.to_thread.run_sync(exited.wait, 5)
        for worker in workers:
            if worker.name.startswith("swarm-arena-daemon-"):
                await anyio.to_thread.run_sync(worker.join, 5)
    assert exited.is_set()
    assert all(
        not worker.is_alive() for worker in workers if worker.name.startswith("swarm-arena-daemon-")
    )
    if site == "page_exists":
        assert not (Path(env.store.url) / page_key).exists()
