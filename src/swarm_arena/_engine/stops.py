"""Latch admission stops and measure active run time."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary, ref

import anyio

from swarm_arena._agent.manager import LimitReached
from swarm_arena._engine.control import stop_key
from swarm_arena._errors import (
    ProviderFailed,
    StorageError,
    _core_category,
    _core_error,
    _rebuild_core_signal,
    _SignalCategory,
)
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._model.context_strings import _context_string
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


if TYPE_CHECKING:
    from swarm_arena._agent.manager import Manager
    from swarm_arena._budget.llm_scope import LLMScope
    from swarm_arena._journal.replay import RunState
    from swarm_arena._journal.writer import JournalWriter
    from swarm_arena._model.catalog import ModelCatalog
    from swarm_arena._model.tooling import ToolOutput
    from swarm_arena._tooling.tool import Tool


_probe: Callable[[str, LLMScope], Awaitable[bool]] | None = None
_services: WeakKeyDictionary[JournalWriter, ProviderStops] = WeakKeyDictionary()


def register_probe(fn: Callable[[str, LLMScope], Awaitable[bool]] | None) -> None:
    """Register the optional adapter's recovery probe.

    Parameters
    ----------
    fn : Callable or None
        Asynchronous provider probe, or None for unconditional recovery.

    Examples
    --------
    >>> previous = _probe
    >>> try:
    ...     register_probe(None)
    ... finally:
    ...     register_probe(previous)
    """
    global _probe
    _probe = fn


def agent_providers(manager: Manager, tool: Tool, models: ModelCatalog) -> frozenset[str]:
    """Resolve an agent's manager aliases and declared tool providers.

    Parameters
    ----------
    manager : Manager
        Built manager supplying model aliases.
    tool : Tool
        Built tool supplying provider identifiers.
    models : ModelCatalog
        Admitted model catalog.

    Returns
    -------
    frozenset[str]
        Detached provider identifiers. Call under supervised preparation.

    Examples
    --------
    >>> from swarm_arena import ScriptedManager, FunctionTool
    >>> from swarm_arena.config import ModelCatalog
    >>> agent_providers(ScriptedManager([]), FunctionTool("math:sqrt"), ModelCatalog({}))
    frozenset()
    """
    providers = {_context_string(p, "provider") for p in tool.providers()}
    for alias in manager.models():
        alias = _context_string(alias, "alias")
        providers.add(_context_string(models.resolve(alias, "manager").provider, "provider"))
    return frozenset(providers)


def failed_providers(output: ToolOutput, tool: Tool) -> frozenset[str]:
    """Resolve explicit or whole-tool provider failures under preparation.

    Parameters
    ----------
    output : ToolOutput
        Detached failed tool output.
    tool : Tool
        Tool whose declarations supply the fallback.

    Returns
    -------
    frozenset[str]
        Normalized identifiers; no upstream message is retained.

    Examples
    --------
    >>> from swarm_arena import FunctionTool
    >>> from swarm_arena._model.tooling import ToolOutput
    >>> failed_providers(ToolOutput(status="provider_failed", failed_provider="openai"),
    ...                  FunctionTool("math:sqrt")) == {"openai"}
    True
    """
    if output.failed_provider is not None:
        return frozenset({_context_string(output.failed_provider, "failed_provider")})
    return frozenset(_context_string(p, "provider") for p in tool.providers())


class ProviderStops:
    """Own run-wide provider generations, causes and cancellation registrations.

    Parameters
    ----------
    journal : JournalWriter
        Entered owner writer. Its identity binds the reporter for child scopes.
    state : RunState or None, optional
        Committed state to restore, independent of today's model catalog.
    standalone : bool, default=False
        Use address-free standalone planning transition events.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> stops = ProviderStops(Mock())
    >>> stops.stopped(), stops.generations()
    ({}, {})
    """

    def __init__(
        self, journal: JournalWriter, *, state: RunState | None = None, standalone: bool = False
    ) -> None:
        self._journal = ref(journal)
        self._standalone = standalone
        self._generations = dict(state.provider_generations) if state is not None else {}
        self._failed = dict(state.provider_failed) if state is not None else {}
        self._causes = (
            {a: dict(c) for a, c in state.provider_causes.items()} if state is not None else {}
        )
        self._providers: dict[str, frozenset[str]] = {}
        self._cancel: dict[str, Callable[[], None]] = {}
        _services[journal] = self

    def _writer(self) -> JournalWriter:
        journal = self._journal()
        if journal is None:
            raise _core_error(RuntimeError, "provider owner is closed")
        return journal

    def _emit(self, transition: str, **payload: Any) -> None:
        self._writer().append(
            EventDraft(
                type=EventType.PROVIDER_FAILED,
                payload={
                    "variant": "standalone" if self._standalone else "transition",
                    "transition": transition,
                    **payload,
                },
            )
        )

    def register(self, address: str, providers: frozenset[str], cancel: Callable[[], None]) -> None:
        """Register active work and immediately observe outstanding failures.

        Parameters
        ----------
        address : str
            Full goal/team/agent identity.
        providers : frozenset[str]
            Validated provider identifiers from supervised preparation.
        cancel : Callable[[], None]
            Core callback carrying shared cancellation provenance.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> stops = ProviderStops(Mock())
        >>> stops.register("g/t/a", frozenset({"openai"}), lambda: None)
        >>> stops.stopped()
        {}
        """
        address = _context_string(address, "address")
        providers = frozenset(_context_string(p, "provider") for p in providers)
        causes = {p: self._failed[p] for p in providers if p in self._failed}
        if causes:
            self._emit("observed", address=address, causes=causes)
            self._causes.setdefault(address, {}).update(causes)
        self._providers[address], self._cancel[address] = providers, cancel
        if self._causes.get(address):
            cancel()

    def unregister(self, address: str) -> None:
        """Remove the active callback while retaining between-round identity.

        Parameters
        ----------
        address : str
            Agent whose active round has ended.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> ProviderStops(Mock()).unregister("g/t/a")
        """
        self._cancel.pop(_context_string(address, "address"), None)

    async def fail(self, provider: str, *, source: str) -> list[str]:
        """Latch and durably report a failure before cancelling matching work.

        Parameters
        ----------
        provider : str
            Core-validated catalog or declared tool provider identifier.
        source : str
            Component identifier, never upstream diagnostic text.

        Returns
        -------
        list[str]
            Sorted affected addresses across all goals.

        Examples
        --------
        >>> stopped = await service.fail("openai", source="manager")  # doctest: +SKIP
        """
        provider = _context_string(provider, "provider")
        source = _context_string(source, "source")
        if source not in {"manager", "tool", "judge", "planner", "curator"}:
            raise _core_error(ValueError, "invalid provider source")
        generation = self._generations.get(provider, 0) + 1
        stopped = sorted(
            {a for a, ps in self._providers.items() if provider in ps}
            | {a for a, cs in self._causes.items() if provider in cs}
        )
        self._emit(
            "failed", provider=provider, generation=generation, source=source, stopped=stopped
        )
        self._generations[provider] = self._failed[provider] = generation
        for address in stopped:
            self._causes.setdefault(address, {})[provider] = generation
        # No await between event admission and the matching live state. Gate
        # readers already refuse new work while durable publication is pending.
        try:
            await self._writer().commit()
        finally:
            for address in stopped:
                if cancel := self._cancel.get(address):
                    cancel()
        return stopped

    def stopped(self) -> dict[str, frozenset[str]]:
        """Snapshot provider-paused addresses and their unresolved causes.

        Returns
        -------
        dict[str, frozenset[str]]
            Empty cause sets denote recovered slots awaiting a fresh round.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> ProviderStops(Mock()).stopped()
        {}
        """
        return {a: frozenset(cs) for a, cs in self._causes.items()}

    def generations(self) -> dict[str, int]:
        """Snapshot currently failed providers' generations for probing.

        Returns
        -------
        dict[str, int]
            Generation that a successful probe must still match.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> ProviderStops(Mock()).generations()
        {}
        """
        return dict(self._failed)

    async def recover(self, provider: str, *, generation: int) -> list[str]:
        """Clear only the generation whose probe succeeded.

        Parameters
        ----------
        provider : str
            Probed identifier, independent of the current catalog.
        generation : int
            Generation captured before starting that probe.

        Returns
        -------
        list[str]
            Addresses whose cause was cleared; other causes remain.

        Examples
        --------
        >>> recovered = await service.recover("openai", generation=1)  # doctest: +SKIP
        """
        provider = _context_string(provider, "provider")
        if self._failed.get(provider) != generation:
            return []
        recovered = sorted(a for a, cs in self._causes.items() if cs.get(provider) == generation)
        self._emit("recovered", provider=provider, generation=generation, recovered=recovered)
        self._failed.pop(provider)
        for address in recovered:
            self._causes[address].pop(provider)
        await self._writer().commit()
        return recovered

    def activate(self, address: str) -> None:
        """Admit a recovered provider-paused slot to its next fresh round.

        Parameters
        ----------
        address : str
            Recovered slot with no unresolved causes.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> ProviderStops(Mock()).activate("g/t/a")
        """
        address = _context_string(address, "address")
        if address in self._causes and not self._causes[address]:
            self._emit("activated", address=address)
            del self._causes[address]

    def release(self, address: str) -> None:
        """Forget provider capacity only after a permanent disposition.

        Parameters
        ----------
        address : str
            Pruned, permanently stopped or completed agent.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> ProviderStops(Mock()).release("g/t/a")
        """
        address = _context_string(address, "address")
        if address in self._causes:
            self._emit("released", address=address)
        self._causes.pop(address, None)
        self._providers.pop(address, None)
        self._cancel.pop(address, None)

    def _release_goal(self, goal_id: str) -> None:
        """Release all current and replayed address identities at goal completion."""
        for address in sorted(self._providers.keys() | self._causes.keys()):
            if address.split("/", 1)[0] == goal_id:
                self.release(address)


def provider_stops(journal: JournalWriter) -> ProviderStops:
    """Find the shared service bound to this writer, including scope children.

    Parameters
    ----------
    journal : JournalWriter
        Current owner's writer.

    Returns
    -------
    ProviderStops
        Existing service, or a new service for a standalone harness owner.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> writer = Mock()
    >>> provider_stops(writer) is provider_stops(writer)
    True
    """
    return _services.get(journal) or ProviderStops(journal)


async def report_provider_failure(
    scope: LLMScope, error: BaseException, *, source: str
) -> list[str]:
    """Report only a validated provider signal through the owner's live latch.

    Parameters
    ----------
    scope : LLMScope
        Component or child scope sharing the entered journal owner.
    error : BaseException
        Registered provider signal, possibly transported through the SDK.
    source : str
        Manager, tool, judge, planner or curator.

    Returns
    -------
    list[str]
        Affected agent addresses, empty for standalone planning.

    Examples
    --------
    >>> addresses = await report_provider_failure(scope, error, source="judge")  # doctest: +SKIP
    """
    classified = _Outcome().classify(error)
    if _core_category(classified) is not _SignalCategory.PROVIDER:
        raise classified from None
    restored = _rebuild_core_signal(classified)
    assert isinstance(restored, ProviderFailed)
    return await provider_stops(scope.journal).fail(restored.provider, source=source)
