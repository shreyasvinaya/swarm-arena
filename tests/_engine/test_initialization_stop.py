"""Lease-held startup stops before a blocked worker can publish later stages."""

import asyncio
import threading
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.run import all_events, plan

from swarm_arena import RunConfig, Swarm
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.base import Board
from swarm_arena._cli.main import _error_message
from swarm_arena._cli.run_cmds import _summaries
from swarm_arena._engine import resume as resume_module
from swarm_arena._engine import run as run_module
from swarm_arena._engine.control import request_stop
from swarm_arena._errors import _is_core_signal
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal import writer as writer_module
from swarm_arena._journal.lease import read_owner
from swarm_arena._model.canonical import utc_now
from swarm_arena._scoring import human_review as review_module
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("site", ["run.toml", "manifest.json", "board_open", "first_commit"])
@pytest.mark.parametrize("signal", ["stop", "stop_now", "deadline"])
async def test_initialization_stops_before_worker_release(
    tmp_path: Path,
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    signal: str,
) -> None:
    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 0.3})
    if site == "first_commit":
        config = config.model_copy(update={"settle_timeout_s": RunConfig().settle_timeout_s})
    cleanup_entered = anyio.Event()
    original_cleanup = run_module._finish_initialization_stop

    async def cleanup(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cleanup_entered.set()
        return await original_cleanup(*args, **kwargs)

    monkeypatch.setattr(run_module, "_finish_initialization_stop", cleanup)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    deadline = anyio.Event()

    class ClockBackend:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        async def sleep(self, seconds: float) -> None:
            assert 0 < seconds <= config.limits.run_timeout_s
            await deadline.wait()

    monkeypatch.setattr(run_module, "anyio", ClockBackend())
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = ArenaBoard.open if site == "board_open" else LocalStore.create

    def block(*args: Any, **kwargs: Any) -> Any:
        if (
            site == "board_open"
            or args[1] == site
            or (
                site == "first_commit"
                and args[1].startswith("journal/e1-")
                and not entered.is_set()
            )
        ):
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(20)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ArenaBoard if site == "board_open" else LocalStore,
        "open" if site == "board_open" else "create",
        block,
    )
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    store = LocalStore(Path(config.run_root) / "r")
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(entered.wait, 10)
        if signal == "deadline":
            deadline.set()
        else:
            assert await request_stop(config.run_root, "r", now=signal == "stop_now") == 1
        if site == "first_commit":
            with anyio.fail_after(10):
                await cleanup_entered.wait()
            # Settle the already-admitted batch within the cleanup allowance.
            release.set()
        done, _ = await asyncio.wait([task], timeout=2)
        assert done, "initialization ignored its stop/deadline while the worker was held"
        if site in {"board_open", "first_commit"}:
            assert task.result() == {"g": []}
            events = all_events(config)
            assert any(e.type == "stop_requested" for e in events)
            for event_type in ("no_result", "goal_finished"):
                matching = [e for e in events if e.type == event_type]
                assert len(matching) == 1
                assert matching[0].goal_id == "g"
                assert matching[0].payload["reason"] == "stopped"
            assert _summaries(config.run_root, "r", ["g"]) == ["no result: g stopped"]
            assert store.exists("review")
        else:
            error = task.exception()
            assert isinstance(error, RuntimeError)
            assert _is_core_signal(error)
            assert _error_message(error) == "run initialization stopped before run_started"
            assert not store.exists("review")
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()
    assert not store.exists("goals/g/rounds/round-01.md")
    if site == "run.toml":
        assert not store.exists("manifest.json")
    if site not in {"board_open", "first_commit"}:
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].head is None
    else:
        monkeypatch.setattr(run_module, "anyio", anyio)
        assert await Swarm(config).resume("r", force=True) == {"g": []}
        assert not any(e.type == "round_started" for e in all_events(config))


async def test_initialization_stop_cleanup_is_bounded(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 0.3})
    board_entered, write_entered, release = (threading.Event() for _ in range(3))
    workers: list[threading.Thread] = []
    original_open, original_create = ArenaBoard.open, LocalStore.create

    def hold(entered: threading.Event) -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(20)

    def open_board(*args: Any, **kwargs: Any) -> Any:
        hold(board_entered)
        return original_open(*args, **kwargs)

    def create(store: LocalStore, key: str, data: bytes) -> str:
        if key == "goals/g/solution.json":
            hold(write_entered)
        return original_create(store, key, data)

    monkeypatch.setattr(ArenaBoard, "open", open_board)
    monkeypatch.setattr(LocalStore, "create", create)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    store = LocalStore(Path(config.run_root) / "r")
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(board_entered.wait, 8)
            await request_stop(config.run_root, "r", now=True)
            assert await anyio.to_thread.run_sync(write_entered.wait, 8)
        done, _ = await asyncio.wait([task], timeout=2)
        assert done, "stop cleanup waited for a blocked solution write"
        error = task.exception()
        assert isinstance(error, RuntimeError)
        assert _error_message(error) == "initialization stop cleanup timed out"
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
    assert not store.exists("review")


@pytest.mark.parametrize(
    "site", ["journal_entry", "review_repair", "board_build", "board_open", "page_repair"]
)
@pytest.mark.parametrize("signal", ["stop_now", "deadline", "cancel"])
async def test_resume_initialization_stops_before_worker_release(
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    signal: str,
) -> None:
    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 0.3})
    await Swarm(config).run(plan(rounds=1), run_id="r")
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    deadline = anyio.Event()
    stages: list[str] = []

    class ClockBackend:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        async def sleep(self, seconds: float) -> None:
            await deadline.wait()

    def hold() -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(20)

    store = LocalStore(Path(config.run_root) / "r")
    initial_owner = read_owner(store)
    assert initial_owner is not None
    original_chain = writer_module.read_chain

    def read_chain(*args: Any, **kwargs: Any) -> Any:
        stages.append("journal_entry")
        if site == "journal_entry":
            hold()
        return original_chain(*args, **kwargs)

    original_repair = review_module._repair_review_objects
    original_build = run_module._start_board
    original_open = ArenaBoard.open
    original_pages = resume_module._repair_round_pages

    async def repair(*args: Any, **kwargs: Any) -> None:
        stages.append("review_repair")
        if site == "review_repair":
            await run_sync_daemon(hold)
        await original_repair(*args, **kwargs)

    async def build(current: RunConfig) -> Board:
        stages.append("board_build")
        if site == "board_build":
            await run_sync_daemon(hold)
        return await original_build(current)

    def open_board(*args: Any, **kwargs: Any) -> None:
        stages.append("board_open")
        if site == "board_open":
            hold()
        original_open(*args, **kwargs)

    async def pages(*args: Any, **kwargs: Any) -> None:
        stages.append("page_repair")
        if site == "page_repair":
            await run_sync_daemon(hold)
        await original_pages(*args, **kwargs)

    monkeypatch.setattr(run_module, "anyio", ClockBackend())
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    monkeypatch.setattr(writer_module, "read_chain", read_chain)
    monkeypatch.setattr(review_module, "_repair_review_objects", repair)
    monkeypatch.setattr(run_module, "_start_board", build)
    monkeypatch.setattr(ArenaBoard, "open", open_board)
    monkeypatch.setattr(resume_module, "_repair_round_pages", pages)
    task = asyncio.create_task(Swarm(config).resume("r", force=True))
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(entered.wait, 8)
        if signal == "cancel":
            task.cancel()
        elif signal == "deadline":
            deadline.set()
        else:
            assert await request_stop(config.run_root, "r", now=True) == 2
        done, _ = await asyncio.wait([task], timeout=2)
        assert done, "resume initialization ignored its stop while storage was held"
        if signal == "cancel":
            assert task.cancelled()
        else:
            error = task.exception()
            assert isinstance(error, RuntimeError)
            assert _is_core_signal(error)
            assert str(error) == "resume initialization stopped"
        owner = read_owner(LocalStore(Path(config.run_root) / "r"))
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
        assert stages[-1] == site
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
    assert stages[-1] == site

    if site == "journal_entry":
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].head == initial_owner[0].head
        assert not any(e.type == "takeover" for e in all_events(config))


@pytest.mark.parametrize("kind", ["ordinary", "group", "cancel"])
async def test_stopped_initialization_contains_segment_failure(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import traceback

    from swarm_arena._errors import StorageError
    from swarm_arena._store.base import Blob

    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 0.5})
    marker = "private-backend-diagnostic"
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    reads: list[str] = []
    original_open, original_read = ArenaBoard.open, LocalStore.read

    def open_board(*args: Any, **kwargs: Any) -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(20)
        original_open(*args, **kwargs)

    def read(store: LocalStore, key: str) -> Blob:
        if entered.is_set() and key.startswith("journal/"):
            reads.append(key)
            if kind == "cancel":
                raise asyncio.CancelledError(marker)
            if kind == "group":
                raise BaseExceptionGroup(marker, [OSError(marker), asyncio.CancelledError(marker)])
            error = OSError(marker)
            error.add_note(marker)
            raise error
        return original_read(store, key)

    monkeypatch.setattr(ArenaBoard, "open", open_board)
    monkeypatch.setattr(LocalStore, "read", read)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    task = asyncio.create_task(Swarm(config).run(plan(rounds=1), run_id="r"))
    try:
        with anyio.fail_after(10):
            assert await anyio.to_thread.run_sync(entered.wait, 8)
        await request_stop(config.run_root, "r", now=True)
        done, _ = await asyncio.wait([task], timeout=2)
        assert done
        assert not task.cancelled()
        error = task.exception()
        assert isinstance(error, StorageError)
        assert _is_core_signal(error)
        assert reads
        assert marker not in "".join(traceback.format_exception(error))
        owner = read_owner(LocalStore(Path(config.run_root) / "r"))
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
