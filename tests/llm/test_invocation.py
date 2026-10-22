"""Shared child-invocation lifecycle contract for LLM components."""

import asyncio
import traceback
from unittest.mock import Mock

import anyio
import pytest

from swarm_arena import Limits
from swarm_arena._budget.llm_scope import LLMScope, current_llm_scope, llm_scope
from swarm_arena._errors import Fenced, _core_error, _stop_cleanup_error
from swarm_arena.config import ModelCatalog, TracingConfig
from swarm_arena.llm._hooks import BudgetExceeded, _sdk_control
from swarm_arena.llm._invocation import _invocation
from swarm_arena.manager import LimitReached


def parent_scope() -> LLMScope:
    return LLMScope(
        role="curator",
        run_id="unbound",
        goal_id="g",
        team="",
        agent="",
        round=1,
        candidate_id=None,
        tranche=Mock(),
        journal=Mock(),
        artifacts=Mock(),
        models=ModelCatalog({}),
        tracing=TracingConfig(),
        cache=None,
        limits=Limits(),
        settle_timeout_s=2,
    )


async def test_child_scope_restored_before_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena.llm import _invocation as module

    parent = parent_scope()
    calls: list[tuple[LLMScope, bool]] = []

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        assert current_llm_scope() is parent
        calls.append((scope, interrupted))

    monkeypatch.setattr(module, "_finish_accounting", finish)
    with llm_scope(parent):
        async with _invocation(parent, name="curator", error_type=ValueError) as child:
            assert current_llm_scope() is child
            assert child is not parent
            assert child.parent is parent
            assert child.tranche is parent.tranche
            assert child.call_ids == []
        assert current_llm_scope() is parent
    assert calls == [(child, False)]
    assert current_llm_scope() is None


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
async def test_body_errors_are_contained_after_drain(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    from swarm_arena.llm import _invocation as module

    drained: list[bool] = []

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        drained.append(interrupted)

    monkeypatch.setattr(module, "_finish_accounting", finish)
    with pytest.raises(error_type, match="LLM curator failed: KeyError") as caught:
        async with _invocation(parent_scope(), name="curator", error_type=error_type):
            raise KeyError("SECRET")
    assert drained == [False]
    assert "SECRET" not in "".join(traceback.format_exception(caught.value))


async def test_ambient_exception_does_not_mark_success_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena.llm import _invocation as module

    drained: list[bool] = []

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        drained.append(interrupted)

    monkeypatch.setattr(module, "_finish_accounting", finish)
    try:
        raise asyncio.CancelledError("outer")
    except asyncio.CancelledError:
        async with _invocation(parent_scope(), name="curator", error_type=ValueError):
            pass
    assert drained == [False]


async def test_budget_carrier_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena.llm import _invocation as module

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        assert not interrupted

    monkeypatch.setattr(module, "_finish_accounting", finish)
    error = _sdk_control(_core_error(LimitReached, "budget", "budget refused"))
    assert isinstance(error, BudgetExceeded)
    with pytest.raises(BudgetExceeded) as caught:
        async with _invocation(parent_scope(), name="curator", error_type=ValueError):
            raise error
    assert caught.value is error


@pytest.mark.parametrize(
    "body_error",
    [None, ValueError("SECRET"), _core_error(LimitReached, "budget", "budget refused")],
)
async def test_cleanup_terminal_wins_over_body_failure(
    monkeypatch: pytest.MonkeyPatch, body_error: Exception | None
) -> None:
    from swarm_arena.llm import _invocation as module

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        raise _stop_cleanup_error()

    monkeypatch.setattr(module, "_finish_accounting", finish)
    with pytest.raises(RuntimeError, match="stop cleanup timed out"):  # noqa: PT012
        async with _invocation(parent_scope(), name="curator", error_type=ValueError):
            if body_error is not None:
                raise body_error


async def test_fence_wins_over_cleanup_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena.llm import _invocation as module

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        raise _stop_cleanup_error()

    monkeypatch.setattr(module, "_finish_accounting", finish)
    with pytest.raises(Fenced):
        async with _invocation(parent_scope(), name="curator", error_type=ValueError):
            raise _core_error(Fenced, "run ownership lost")


@pytest.mark.parametrize("terminal", [False, True])
async def test_real_cancellation_wins_and_drains_without_wait(
    monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    from swarm_arena.llm import _invocation as module

    ready = anyio.Event()
    drained: list[bool] = []

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        drained.append(interrupted)
        if terminal:
            raise _stop_cleanup_error()

    async def body() -> None:
        async with _invocation(parent_scope(), name="curator", error_type=ValueError):
            ready.set()
            await anyio.sleep_forever()

    monkeypatch.setattr(module, "_finish_accounting", finish)
    task = asyncio.create_task(body())
    try:
        with anyio.fail_after(10):
            await ready.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert drained == [True]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_forged_cancellation_is_not_real_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena.llm import _invocation as module

    drained: list[bool] = []

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        drained.append(interrupted)

    monkeypatch.setattr(module, "_finish_accounting", finish)
    with pytest.raises(RuntimeError, match="CancelledError") as caught:
        async with _invocation(parent_scope(), name="curator", error_type=ValueError):
            raise asyncio.CancelledError("SECRET")
    assert "SECRET" not in "".join(traceback.format_exception(caught.value))
    assert drained == [False]


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
async def test_cleanup_process_controls_propagate(
    monkeypatch: pytest.MonkeyPatch, control: type[BaseException]
) -> None:
    from swarm_arena.llm import _invocation as module

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        raise control()

    monkeypatch.setattr(module, "_finish_accounting", finish)
    with pytest.raises(control):
        async with _invocation(parent_scope(), name="curator", error_type=ValueError):
            pass


@pytest.mark.parametrize("timer_kind", ["fail_after", "move_on_after"])
async def test_caller_timer_catches_cancellation_during_accounting(
    monkeypatch: pytest.MonkeyPatch, timer_kind: str
) -> None:
    from swarm_arena.llm import _invocation as module

    entered = False
    timer: anyio.CancelScope

    async def finish(scope: LLMScope, *, interrupted: bool) -> None:
        nonlocal entered
        assert not interrupted
        entered = True
        timer.deadline = anyio.current_time()
        await anyio.sleep_forever()

    async def invoke() -> None:
        async with _invocation(parent_scope(), name="curator", error_type=ValueError):
            pass

    monkeypatch.setattr(module, "_finish_accounting", finish)
    if timer_kind == "fail_after":
        with pytest.raises(TimeoutError), anyio.fail_after(10) as timer:
            await invoke()
    else:
        with anyio.move_on_after(10) as timer:
            await invoke()
        assert timer.cancelled_caught
    assert entered
