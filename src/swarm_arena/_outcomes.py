"""Classify extension outcomes without mistaking user failures for cancellation."""

import asyncio

import anyio

from swarm_arena._errors import Fenced, _core_error, _rebuild_core_signal


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
        return _core_error(RuntimeError, f"supervised hook failed: {type(error).__name__}")
    return error


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
