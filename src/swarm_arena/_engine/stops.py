"""Latch admission stops and measure active run time."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic

import anyio

from swarm_arena._agent.manager import LimitReached
from swarm_arena._engine.control import stop_key
from swarm_arena._errors import StorageError
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._outcomes import _Outcome
from swarm_arena._store.base import Store


class _StopWatcher:
    """Retain one polling task while its owner's cancellation callback changes."""

    def __init__(self, watch: Callable[[], Awaitable[None]]) -> None:
        self.error: BaseException | None = None
        self.on_error: Callable[[], None] | None = None
        self.failed = anyio.Event()
        self.task = asyncio.create_task(self._run(watch), name="stop-file-watcher")

    async def _run(self, watch: Callable[[], Awaitable[None]]) -> None:
        try:
            await watch()
        except BaseException as error:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            self.error = error
            self.failed.set()
            if self.on_error is not None:
                self.on_error()

    async def wait_error(self) -> None:
        await self.failed.wait()
        assert self.error is not None
        raise self.error

    async def close(self) -> None:
        with anyio.CancelScope(shield=True):
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


@dataclass
class StopState:
    """
    Share latched stop signals between the run and its active rounds.

    Parameters
    ----------
    stop : bool, default=False
        Refuse new ordinary work while existing work drains.
    stop_now : bool, default=False
        Cancel ordinary and converge work.
    run_timeout : bool, default=False
        Active run time has expired.
    run_budget : bool, default=False
        Run work budget is exhausted.
    goal_budget : set[str], optional
        Goals whose work budgets are exhausted.
    on_change : Callable or None, default=None
        Notify the scheduler when a run-wide stop flag becomes true.

    Examples
    --------
    >>> state = StopState(goal_budget={"g"})
    >>> state.gate_for("g")().limit
    'budget'
    """

    stop: bool = False
    stop_now: bool = False
    run_timeout: bool = False
    run_budget: bool = False
    goal_budget: set[str] = field(default_factory=set)
    on_change: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    _watcher: _StopWatcher | None = field(default=None, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: object) -> None:
        """Wake scheduling only for a newly latched run-wide stop."""
        latched = (
            name in {"stop", "stop_now", "run_timeout", "run_budget"}
            and value is True
            and not getattr(self, name, False)
        )
        object.__setattr__(self, name, value)
        if latched and (notify := getattr(self, "on_change", None)) is not None:
            notify()

    def gate_for(self, goal_id: str) -> Callable[[], LimitReached | None]:
        """
        Bind a live admission check to one goal.

        Parameters
        ----------
        goal_id : str
            Main goal identifier.

        Returns
        -------
        Callable[[], LimitReached or None]
            A fresh exception for the highest-priority stop, or None.

        Examples
        --------
        >>> StopState().gate_for("g")() is None
        True
        """

        def gate() -> LimitReached | None:
            if self.stop or self.stop_now:
                return LimitReached("stop", "the run is stopping")
            if self.run_timeout:
                return LimitReached("run_timeout_s", "run time used up")
            if self.run_budget:
                return LimitReached("budget", "run budget used up")
            if goal_id in self.goal_budget:
                return LimitReached("budget", "goal budget used up")
            return None

        return gate


class RunClock:
    """
    Measure this active session plus previously recorded running time.

    Parameters
    ----------
    elapsed_before : float
        Accumulated running seconds before this session.

    Examples
    --------
    >>> RunClock(10.0).elapsed() >= 10.0
    True
    """

    def __init__(self, elapsed_before: float) -> None:
        """Start a monotonic running-time interval."""
        self._before = elapsed_before
        self._start = monotonic()

    def _rebase(self, elapsed_before: float) -> None:
        """Adopt the acquired checkpoint without restarting preparation time."""
        self._before = elapsed_before

    def elapsed(self) -> float:
        """
        Return accumulated active seconds, excluding downtime before resume.

        Returns
        -------
        float
            Previous running time plus this monotonic interval.

        Examples
        --------
        >>> RunClock(10.0).elapsed() >= 10.0
        True
        """
        return self._before + monotonic() - self._start

    def left(self, run_timeout_s: float) -> float:
        """
        Return nonnegative remaining active run time.

        Parameters
        ----------
        run_timeout_s : float
            Overall active-time limit.

        Returns
        -------
        float
            Seconds left, clamped at zero.

        Examples
        --------
        >>> RunClock(10.0).left(5.0)
        0.0
        """
        return max(0.0, run_timeout_s - self.elapsed())


def round_deadline(start: float, round_timeout_s: float, run_left_s: float) -> float:
    """
    Bound a round by its own duration and remaining run time.

    Parameters
    ----------
    start : float
        Round start on AnyIO's clock.
    round_timeout_s : float
        Effective round duration.
    run_left_s : float
        Remaining active run seconds.

    Returns
    -------
    float
        Absolute deadline on the same time basis as start.

    Examples
    --------
    >>> round_deadline(1000.0, 5.0, 2.0)
    1002.0
    """
    return start + min(round_timeout_s, run_left_s)


def _stop_exists(store: Store, key: str) -> bool:
    """Validate the storage result before any branch on the event loop."""
    exists = store.exists(key)
    if type(exists) is not bool:
        raise TypeError("stop lookup must return bool")
    return exists


async def watch_stop_files(
    store: Store, epoch: int, state: StopState, *, interval_s: float = 2.0
) -> None:
    """
    Poll only this lease epoch's stop files until cancelled.

    Storage waits are abandonable by the owning supervisor. Late results
    cannot latch a stop or start another lookup.

    Parameters
    ----------
    store : Store
        Run store containing the control directory.
    epoch : int
        Current lease epoch; older stop files are ignored.
    state : StopState
        Shared latched flags; immediate stop also sets stop.
    interval_s : float, default=2.0
        Seconds between polls.

    Raises
    ------
    StorageError
        If storage fails or returns a non-boolean result. Backend diagnostics
        are contained before the owning task group receives the failure.

    Examples
    --------
    >>> await watch_stop_files(store, 2, StopState())  # doctest: +SKIP
    """
    while True:
        for now in (True, False):
            await anyio.lowlevel.checkpoint_if_cancelled()
            outcome = _Outcome()
            try:
                exists = await run_sync_daemon(_stop_exists, store, stop_key(epoch, now=now))
            except BaseException as error:
                outcome.contain(error, StorageError, "stop lookup")
            if exists:
                if now:
                    state.stop_now = True
                state.stop = True
        await anyio.sleep(interval_s)
