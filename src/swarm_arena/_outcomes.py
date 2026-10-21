"""Classify extension outcomes without mistaking user failures for cancellation."""

import asyncio

import anyio

from swarm_arena._errors import (
    Fenced,
    _core_error,
    _exception_type_name,
    _rebuild_core_signal,
    _with_exception_type,
)


def _classify_outcome(error: BaseException, *, cancelled: bool = False) -> Exception:
    """Preserve control signals and rebuild nonstandard hook failures by type."""
    if cancelled:
        # A hook can catch genuine cancellation and replace its message. The
        # pending caller state wins, independently of that untrusted outcome.
        raise asyncio.CancelledError() from None
    if isinstance(error, Fenced):
        raise error
    if isinstance(error, BaseExceptionGroup) and error.subgroup(Fenced) is not None:
        # Task groups can wrap ownership loss alongside other hook failures.
        # Preserve the control signal without retaining group or leaf values.
        raise _core_error(Fenced, "run ownership lost") from None
    rebuilt = _rebuild_core_signal(error)
    if rebuilt is not None:
        return rebuilt
    if isinstance(error, BaseExceptionGroup) or not isinstance(error, Exception):
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

    def __init__(self) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._task = task
        self._info = anyio.get_current_task()
        self._count = self._task.cancelling()
        self._pending = self._info.has_pending_cancellation()

    def classify(self, error: BaseException) -> Exception:
        """Let actual pending caller cancellation win over any hook outcome."""
        return _classify_outcome(
            error,
            cancelled=(
                self._pending
                or self._task.cancelling() > self._count
                or self._info.has_pending_cancellation()
            ),
        )

    def propagate_control(self, error: BaseException) -> None:
        """Propagate controls using this caller's cancellation snapshot."""
        self.classify(error)
