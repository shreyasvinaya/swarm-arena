"""Classify extension outcomes without mistaking user failures for cancellation."""

import asyncio
from typing import NoReturn

import anyio

from swarm_arena._errors import (
    Fenced,
    _core_error,
    _exception_type_name,
    _is_core_signal,
    _rebuild_core_signal,
    _with_exception_type,
)


def _has_core_fenced(error: BaseException) -> bool:
    """Find registered ownership loss without trusting exception type alone."""
    pending = [error]
    while pending:
        current = pending.pop()
        if isinstance(current, Fenced) and _is_core_signal(current):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return False


def _classify_outcome(error: BaseException, *, cancelled: bool = False) -> Exception:
    """Preserve control signals and rebuild nonstandard hook failures by type."""
    if cancelled:
        # A hook can catch genuine cancellation and replace its message. The
        # pending caller state wins, independently of that untrusted outcome.
        raise asyncio.CancelledError() from None
    if _has_core_fenced(error):
        raise _core_error(Fenced, "run ownership lost") from None
    rebuilt = _rebuild_core_signal(error)
    if rebuilt is not None:
        return rebuilt
    if isinstance(error, (Fenced, BaseExceptionGroup)) or not isinstance(error, Exception):
        name = _exception_type_name(error)
        return _with_exception_type(
            _core_error(RuntimeError, f"supervised hook failed: {name}"), name
        )
    return error


def _propagate_control(error: BaseException) -> None:
    """Raise control outcomes when a boundary intentionally discards failure details."""
    _classify_outcome(error)


class _Outcome:
    """Snapshot caller cancellation before an asynchronous extension boundary."""

    def __init__(self, *, capture_caller: bool = True) -> None:
        self._task = asyncio.current_task() if capture_caller else None
        self._info = anyio.get_current_task() if capture_caller else None
        self._count = self._task.cancelling() if self._task is not None else 0
        self._pending = self._info.has_pending_cancellation() if self._info is not None else False

    def classify(self, error: BaseException) -> Exception:
        """Let actual pending caller cancellation win over any hook outcome."""
        return _classify_outcome(
            error,
            cancelled=(
                self._pending
                or (self._task is not None and self._task.cancelling() > self._count)
                or (self._info is not None and self._info.has_pending_cancellation())
            ),
        )

    def propagate_control(self, error: BaseException) -> None:
        """Propagate controls using this caller's cancellation snapshot."""
        self.classify(error)

    def contain(self, error: BaseException, error_type: type[Exception], site: str) -> NoReturn:
        """Raise rebuilt core signals or a fresh type-only boundary failure."""
        registered = _is_core_signal(error)
        classified = self.classify(error)
        if registered:
            raise classified from None
        raise _core_error(
            error_type, f"{site} failed: {_exception_type_name(classified)}"
        ) from None
