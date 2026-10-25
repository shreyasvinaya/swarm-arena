"""Configured board hooks share safe and bounded run/resume boundaries."""

import threading
import traceback
from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest
from _fakes.board_lifecycle import SECRET, LifecycleBoard
from _fakes.run import plan
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Ref, RunConfig, Swarm


@pytest.fixture(autouse=True)
def reset_board(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    act = LifecycleBoard._act
    workers: list[threading.Thread] = []

    def held_act(board: LifecycleBoard, mode: str) -> None:
        if mode == "block":
            workers.append(threading.current_thread())
            board.entered.set()
            assert board.release.wait(180), "test did not release board"
        else:
            act(board, mode)

    # Keep the shared fake's ten-second backstop outside these longer assertions.
    monkeypatch.setattr(LifecycleBoard, "_act", held_act)
    LifecycleBoard.entered.clear()
    LifecycleBoard.release.clear()
    LifecycleBoard.closed.clear()
    try:
        yield
    finally:
        LifecycleBoard.release.set()
        for worker in workers:
            worker.join(60)
            assert not worker.is_alive()


def configured(config: RunConfig, tmp_path: Path, **modes: str) -> RunConfig:
    return config.model_copy(
        update={
            "board": Ref(
                path="_fakes.board_lifecycle:LifecycleBoard",
                config={
                    "arena_root": config.arena_root,
                    "cache_dir": str(tmp_path / "board-cache"),
                    **modes,
                },
            ),
        }
    )


async def execute(config: RunConfig, operation: str) -> None:
    if operation == "resume":
        await Swarm(config).resume("lifecycle", force=True)
    else:
        await Swarm(config).run(plan(rounds=1), run_id="lifecycle")


@pytest.mark.parametrize("operation", ["run", "resume"])
@pytest.mark.parametrize("phase", ["open", "close"])
async def test_configured_board_lifecycle_hides_every_surface(
    run_config: RunConfig, tmp_path: Path, operation: str, phase: str
) -> None:
    if operation == "resume":
        await execute(run_config, "run")
    config = configured(run_config, tmp_path, **{f"{phase}_mode": "error"})
    with pytest.raises(RuntimeError, match=f"board {phase} failed: ValueError") as caught:
        await execute(config, operation)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert LifecycleBoard.closed.is_set()


@pytest.mark.parametrize("operation", ["run", "resume"])
@pytest.mark.parametrize("phase", ["open", "close"])
@pytest.mark.parametrize("mode", ["core", "fake-fenced"])
async def test_board_lifecycle_preserves_only_registered_signals(
    run_config: RunConfig, tmp_path: Path, operation: str, phase: str, mode: str
) -> None:
    from swarm_arena._errors import Fenced

    if operation == "resume":
        await execute(run_config, "run")
    config = configured(run_config, tmp_path, **{f"{phase}_mode": mode})
    expected = Fenced if mode == "core" else RuntimeError
    with pytest.raises(expected) as caught:
        await execute(config, operation)
    if mode == "core":
        assert type(caught.value) is Fenced
        assert caught.value.args == ()
    else:
        assert str(caught.value) == f"board {phase} failed: Fenced"
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("operation", ["run", "resume"])
async def test_board_close_failure_preserves_primary_open_signal(
    run_config: RunConfig, tmp_path: Path, operation: str
) -> None:
    from swarm_arena._errors import Fenced

    if operation == "resume":
        await execute(run_config, "run")
    config = configured(run_config, tmp_path, open_mode="core", close_mode="error")
    with pytest.raises(Fenced) as caught:
        await execute(config, operation)
    assert caught.value.args == ()
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert LifecycleBoard.closed.is_set()


@pytest.mark.parametrize("operation", ["run", "resume"])
async def test_blocking_board_close_is_bounded(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    if operation == "resume":
        await execute(run_config, "run")
    workers: list[threading.Thread] = []
    close = LifecycleBoard.close

    def tracked_close(self: LifecycleBoard) -> None:
        workers.append(threading.current_thread())
        close(self)

    monkeypatch.setattr(LifecycleBoard, "close", tracked_close)
    config = configured(run_config, tmp_path, close_mode="block").model_copy(
        update={"settle_timeout_s": 2.0}
    )
    completed = anyio.Event()

    async def drive() -> None:
        await execute(config, operation)
        completed.set()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(drive)
            assert await anyio.to_thread.run_sync(LifecycleBoard.entered.wait, 60)
            with anyio.fail_after(30):
                await completed.wait()
            assert not LifecycleBoard.release.is_set()
    finally:
        LifecycleBoard.release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()


@pytest.mark.parametrize("operation", ["run", "resume"])
async def test_blocking_board_open_can_be_cancelled(
    run_config: RunConfig, tmp_path: Path, operation: str
) -> None:
    if operation == "resume":
        await execute(run_config, "run")
    config = configured(run_config, tmp_path, open_mode="block").model_copy(
        update={"settle_timeout_s": 2.0}
    )
    scope = anyio.CancelScope()
    completed = anyio.Event()
    cancelled: list[bool] = []

    async def drive() -> None:
        with scope:
            try:
                await execute(config, operation)
            except anyio.get_cancelled_exc_class():
                cancelled.append(True)
                raise
        completed.set()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(drive)
            assert await anyio.to_thread.run_sync(LifecycleBoard.entered.wait, 60)
            scope.cancel()
            with anyio.fail_after(30):
                await completed.wait()
            assert cancelled == [True]
            assert LifecycleBoard.closed.is_set()
            assert not LifecycleBoard.release.is_set()
    finally:
        LifecycleBoard.release.set()


@pytest.mark.parametrize("operation", ["run", "resume"])
async def test_board_close_failure_preserves_primary_run_failure(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from swarm_arena._engine.round import EngineDeps
    from swarm_arena._engine.run import RunRunner
    from swarm_arena._errors import _core_error
    from swarm_arena._model.results import Candidate

    if operation == "resume":
        await execute(run_config, "run")

    async def fail(self: RunRunner, deps: EngineDeps, epoch: int) -> dict[str, list[Candidate]]:
        raise _core_error(RuntimeError, "vetted primary run failure")

    monkeypatch.setattr(RunRunner, "_drive", fail)
    config = configured(run_config, tmp_path, close_mode="error")
    with pytest.raises(RuntimeError, match=r"^vetted primary run failure$") as caught:
        await execute(config, operation)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert LifecycleBoard.closed.is_set()
