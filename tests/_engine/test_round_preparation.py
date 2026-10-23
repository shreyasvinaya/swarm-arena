"""Preparation time does not consume the agent round deadline."""

import asyncio
import threading

import anyio
import pytest
from _engine.test_round import _round, _RoundTime, _scripted
from _fakes.engine import EngineEnv

from swarm_arena._engine import round as round_module
from swarm_arena._model.board_types import Frontier


async def test_round_deadline_starts_after_slow_preparation(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _RoundTime(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = env.deps.board.frontier
    starts: list[float] = []
    deadline = round_module.round_deadline

    def measured(start: float, round_timeout: float, run_left: float) -> float:
        starts.append(start)
        return deadline(start, round_timeout, run_left)

    def blocked() -> Frontier:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(180)
        return original()

    monkeypatch.setattr(round_module, "round_deadline", measured)
    monkeypatch.setattr(env.deps.board, "frontier", blocked)
    runner = _round(env, [_scripted("a", [])])
    task = asyncio.create_task(runner.run(1))
    try:
        with anyio.fail_after(60):
            assert await anyio.to_thread.run_sync(entered.wait, 60)
        clock.advance(1000)
        prepared_at = clock.current_time()
        release.set()
        with anyio.fail_after(60):
            result = await task
        assert starts
        assert starts[0] >= prepared_at
        assert result.reasons == {"t/a": ""}
        assert env.events("round_started")
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()
