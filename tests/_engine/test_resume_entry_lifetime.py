"""Inherited reads retain the validated run clock and confirmed ownership bound."""

import asyncio
import threading
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.leases import force_take_over
from _fakes.run import all_events, plan

from swarm_arena import RunConfig, Swarm
from swarm_arena._engine import run as run_module
from swarm_arena._engine.control import request_stop
from swarm_arena._engine.stops import RunClock
from swarm_arena._errors import Fenced, _is_core_signal
from swarm_arena._journal import writer as writer_module
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._model.canonical import utc_now
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("signal", ["expiry", "takeover", "stop", "cancel", "deadline"])
async def test_inherited_read_ends_without_late_entry(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, signal: str
) -> None:
    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 0.3})
    if signal == "deadline":
        config = config.model_copy(
            update={
                "lease_ttl_s": 60,
                "limits": config.limits.model_copy(update={"run_timeout_s": 100}),
            }
        )
        with monkeypatch.context() as setup:
            setup.setattr(run_module, "RunClock", lambda before: RunClock(before + 95))
            await Swarm(config).run(plan(rounds=1), run_id="r")
    else:
        await Swarm(config).run(plan(rounds=1), run_id="r")
    store = LocalStore(Path(config.run_root) / "r")
    committed = read_committed(store)
    assert committed is not None
    state = RunState.from_events(committed[1])
    remaining = config.limits.run_timeout_s - state.run_elapsed_s
    initial_head = committed[0].head
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = writer_module.read_chain

    def read(*args: Any, **kwargs: Any) -> Any:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(30)
        return original(*args, **kwargs)

    monkeypatch.setattr(writer_module, "read_chain", read)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    task = asyncio.create_task(Swarm(config).resume("r", force=True))
    successor: Lease | None = None
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(entered.wait, 8)
        if signal == "takeover":
            successor = await anyio.to_thread.run_sync(
                partial(force_take_over, store, owner_id="successor", host="local", ttl_s=60)
            )
        elif signal == "stop":
            assert await request_stop(config.run_root, "r", now=True) == 2
        elif signal == "cancel":
            task.cancel()
        bound = remaining + 2 if signal == "deadline" else 4
        done, _ = await asyncio.wait([task], timeout=bound)
        assert done, "inherited read outlived its owner or remaining run allowance"
        if signal == "cancel":
            assert task.cancelled()
        else:
            error = task.exception()
            assert isinstance(error, Fenced if signal in {"expiry", "takeover"} else RuntimeError)
            assert _is_core_signal(error)
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].head == initial_head
        if successor is None:
            assert owner[0].expires_at <= utc_now()
        else:
            assert owner[0] == successor.doc
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
        if successor is not None:
            successor.release()
    assert not any(event.type == "takeover" for event in all_events(config))
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head == initial_head


async def test_preparation_and_replay_share_remaining_allowance(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine import resume as resume_module
    from swarm_arena._engine import stops as stops_module
    from swarm_arena._journal.event import Event
    from swarm_arena._journal.lease import OwnerDoc
    from swarm_arena._store.base import Store

    await Swarm(run_config).run(plan(rounds=1), run_id="r")
    store = LocalStore(Path(run_config.run_root) / "r")
    committed = read_committed(store)
    assert committed is not None
    before = RunState.from_events(committed[1]).run_elapsed_s
    ticks = 0.0
    reads = 0
    original = resume_module._read

    def read(current: Store, doc: OwnerDoc) -> tuple[list[Event], RunState]:
        nonlocal ticks, reads
        result = original(current, doc)
        if reads:
            ticks += 7
        reads += 1
        return result

    monkeypatch.setattr(stops_module, "monotonic", lambda: ticks)
    monkeypatch.setattr(resume_module, "_read", read)
    runner = await resume_module.resume_run(run_config, "r", force=True, replanner=None)
    try:
        assert reads == 3
        assert ticks == 14
        assert runner.resumed is not None
        deps = runner.resumed._services.deps
        assert deps is not None
        assert deps.clock.elapsed() == before + ticks
    finally:
        await runner.aclose()
