"""Approve and execute one run with shared budgets and durable ownership."""

import asyncio
import importlib
import os
import socket
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager, suppress
from fractions import Fraction
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import anyio

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.base import Board
from swarm_arena._board.curator import Curator
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.estimate import final_slice_estimate, make_price_of, worst_case_estimate
from swarm_arena._budget.ledger import Ledger
from swarm_arena._budget.llm_scope import LLMScope, llm_scope
from swarm_arena._budget.tranche import Tranche
from swarm_arena._engine.admission import RunAdmission
from swarm_arena._engine.goal import GoalRunner, r0_revision_draft
from swarm_arena._engine.manifest import RunManifest, build_manifest_async
from swarm_arena._engine.preflight import run_preflight
from swarm_arena._engine.round import EngineDeps
from swarm_arena._engine.solution import solution_view
from swarm_arena._engine.stops import RunClock, StopState, watch_stop_files
from swarm_arena._errors import (
    AlreadyExists,
    Fenced,
    PlanError,
    StorageError,
    _core_category,
    _core_error,
    _exception_type_name,
    _SignalCategory,
    _stop_cleanup_error,
)
from swarm_arena._exec.executor import Executor
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.lease import Head, Lease
from swarm_arena._journal.llm_record import _bound_run, drain
from swarm_arena._journal.reader import _read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.views import ViewWriter
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.base import LLMRole
from swarm_arena._model.canonical import canonical_json, utc_now
from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.goal import Goal
from swarm_arena._model.ids import IdGen, new_run_id
from swarm_arena._model.plan import Plan, Resources, RoundSpec
from swarm_arena._model.results import Candidate
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.tooling import Estimate, ToolRequest
from swarm_arena._outcomes import _has_core_fenced as _has_fenced
from swarm_arena._outcomes import _Outcome
from swarm_arena._planning.planner import Planner
from swarm_arena._planning.provenance import _detach_plan, ref_sites, stamp
from swarm_arena._planning.validate import _goal_key_errors, admit_revision, check_models
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import Store, _read_blob, open_store
from swarm_arena._store.io import _create, _storage_io
from swarm_arena._store.local import LocalStore
from swarm_arena._tooling.tool import Tool, effective_tool_ref

if TYPE_CHECKING:
    from swarm_arena._engine.resume import ResumeState


_run_setups: list[Callable[[RunConfig], None]] = []


def register_run_setup(fn: Callable[[RunConfig], None]) -> None:
    """Register an optional integration's setup for each entered run lifetime.

    Parameters
    ----------
    fn : Callable[[RunConfig], None]
        Synchronous setup, invoked in supervised daemon preparation once per
        start, resume or planning session. Repeated registration is idempotent.

    Examples
    --------
    >>> def setup(config: RunConfig) -> None:
    ...     pass
    >>> register_run_setup(setup)
    >>> register_run_setup(setup)
    >>> sum(fn is setup for fn in _run_setups)
    1
    >>> _run_setups.remove(setup)
    """
    if not any(existing is fn for existing in _run_setups):
        _run_setups.append(fn)


def _bootstrap_registrations(plans: Sequence[Plan], config: RunConfig) -> None:
    """Import authorized registration modules without rebuilding runtime objects."""
    modules = dict.fromkeys(
        site.ref.path.split(":", 1)[0]
        for plan in plans
        for site in ref_sites(plan)
        if site.kind in {"manager", "judge"}
    )
    if config.curator is not None:
        modules[config.curator.path.split(":", 1)[0]] = None
    for module in modules:
        importlib.import_module(module)


async def _setup_run(plans: Sequence[Plan], config: RunConfig) -> None:
    """Bootstrap admitted references, then invoke each setup under the live owner."""
    outcome = _Outcome()
    try:
        await run_sync_daemon(_bootstrap_registrations, plans, config)
        for setup in tuple(_run_setups):
            await run_sync_daemon(setup, config)
    except BaseException as error:
        outcome.contain(error, RuntimeError, "run setup")


async def _watch_fence(journal: JournalWriter, scope: anyio.CancelScope) -> None:
    await journal.fenced.wait()
    scope.cancel()


@asynccontextmanager
async def _supervise_writer(journal: JournalWriter) -> AsyncGenerator[None]:
    """Retain the existing fence watcher through preparation and publication."""
    error: BaseException | None = None
    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as watchers:
            watchers.start_soon(_watch_fence, journal, scope)
            try:
                await anyio.lowlevel.checkpoint_if_cancelled()
                journal._raise_if_fenced()
                yield
            except BaseException as caught:
                error = caught
            finally:
                watchers.cancel_scope.cancel()
    journal._raise_if_fenced()
    if error is not None:
        raise error


@asynccontextmanager
async def _supervise_initialization(
    store: Store, epoch: int, stops: StopState, clock: RunClock | None, config: RunConfig
) -> AsyncGenerator[Callable[[], None]]:
    """Watch stop files and, when a clock is supplied, the remaining run allowance."""
    outcome = _Outcome()
    error: BaseException | None = None
    watcher_scope = anyio.CancelScope()
    with anyio.CancelScope() as preparation:
        stops.on_change = preparation.cancel

        async def watch() -> None:
            with watcher_scope:
                async with anyio.create_task_group() as watchers:
                    watchers.start_soon(watch_stop_files, store, epoch, stops)
                    if clock is not None:
                        await anyio.sleep(clock.left(config.limits.run_timeout_s))
                        stops.run_timeout = True

        def prepared() -> None:
            stops.on_change = None
            watcher_scope.cancel()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(watch)
            try:
                await anyio.lowlevel.checkpoint()
                yield prepared
            except BaseException as caught:
                error = caught
            finally:
                prepared()
                tasks.cancel_scope.cancel()
    # The task group's own cancellation must not hide a concurrent caller cancel.
    outcome.propagate_control(asyncio.CancelledError())
    if error is not None:
        if preparation.cancel_called and isinstance(error, anyio.get_cancelled_exc_class()):
            return
        # Extension boundaries already classify failures; preserve driver errors
        # and cancellation even after nested task groups clear their cancel count.
        raise error


class _RunJournal(JournalWriter):
    """Stamp every durable producer's progress with the shared active clock."""

    clock: RunClock | None = None

    async def __aenter__(self) -> "_RunJournal":
        await super().__aenter__()
        return self

    def append(self, draft: EventDraft) -> Event:
        """Include elapsed active time without changing producer payloads in place."""
        if self.clock is not None:
            draft = draft.model_copy(
                update={"payload": {**draft.payload, "run_elapsed_s": self.clock.elapsed()}}
            )
        return super().append(draft)


def _require_asyncio() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        raise _core_error(RuntimeError, "swarm-arena runs on the asyncio backend only") from None


def _build_executor(config: RunConfig) -> Executor:
    """Build the executor through one replaceable seam shared with resume."""
    return config.executor.build(Executor)


def _startup_models(config: RunConfig, replanner: Planner | None) -> tuple[list[str], list[str]]:
    """Build startup components and normalize their hooks entirely in the worker."""
    uses: dict[LLMRole, list[str]] = {}
    if replanner is not None:
        uses["planner"] = [_context_string(a, "alias") for a in replanner.models()]
    if config.curator is not None:
        curator = config.curator.build(Curator)
        uses["curator"] = [_context_string(a, "alias") for a in curator.models()]
    return check_models(uses, config)


async def _check_startup_models(config: RunConfig, replanner: Planner | None) -> list[str]:
    """Validate startup hooks in an abandonable worker before ownership."""
    outcome = _Outcome()
    try:
        errors, warnings = await run_sync_daemon(_startup_models, config, replanner)
    except BaseException as error:
        error = outcome.classify(error)
        raise _core_error(
            ValueError, f"model checks failed: {_exception_type_name(error)}"
        ) from None
    if errors:
        raise _core_error(ValueError, "\n".join(errors)) from None
    return warnings


def _estimates(plans: Sequence[Plan], config: RunConfig) -> tuple[Amount, dict[str, Amount]]:
    """Bound each address by all its known effective variants and empty inputs."""
    tools: dict[str, Sequence[tuple[Tool, ToolRequest]]] = {}
    for plan in plans:
        rounds = plan.rounds or [RoundSpec() for _ in range(plan.rounds_total(config.limits))]
        teams = [(team, list(enumerate(rounds, 1))) for team in plan.teams]
        if plan.converge.mode == "team" and plan.converge.team is not None:
            teams.append(
                (
                    plan.converge.team,
                    [
                        (number, plan.converge.round or rounds[-1])
                        for number in range(1, len(rounds) + 2)
                    ],
                )
            )
        for team, specs in teams:
            for agent in team.agents:
                tools[f"{plan.goal.id}/{team.name}/{agent.name}"] = [
                    (
                        effective_tool_ref(agent.tool, spec.tool_overrides).build(Tool),
                        ToolRequest(round=number, round_overrides=spec.tool_overrides),
                    )
                    for number, spec in specs
                ]
    price_of = make_price_of(config.models)
    return worst_case_estimate(plans, config, tools, price_of), {
        p.goal.id: final_slice_estimate(p, config, tools, price_of) for p in plans
    }


def _exceeds_limit(amount: Amount, config: RunConfig, fraction: Fraction = Fraction(1)) -> bool:
    """Compare in the ledger's integer tokens and micro-dollars, flooring reserve."""
    limits = config.limits
    cap = Amount(tokens=limits.max_tokens or 0, usd=limits.max_usd or 0)
    return (
        limits.max_tokens is not None
        and amount.tokens > cap.tokens * fraction.numerator // fraction.denominator
    ) or (
        limits.max_usd is not None
        and amount._micro_usd > cap._micro_usd * fraction.numerator // fraction.denominator
    )


def _capacity_warning(plans: Sequence[Plan], capacity: Resources | None) -> list[str]:
    if capacity is None:
        return []
    demand: dict[str, float] = {}
    for plan in plans:
        choices = [plan.teams]
        if plan.converge.mode == "team" and plan.converge.team:
            choices.append([plan.converge.team])
        peaks: dict[str, float] = {}
        for teams in choices:
            total: dict[str, float] = {}
            for team in teams:
                for agent in team.agents:
                    r = agent.resources
                    for key, value in {
                        "cpus": r.cpus,
                        "gpus": r.gpus,
                        "memory_gib": r.memory_gib or 0,
                        **{f"custom:{k}": v for k, v in r.custom.items()},
                    }.items():
                        total[key] = total.get(key, 0) + value * agent.replicates
            for key, value in total.items():
                peaks[key] = max(peaks.get(key, 0), value)
        for key, value in peaks.items():
            demand[key] = demand.get(key, 0) + value
    available = {
        "cpus": capacity.cpus,
        "gpus": capacity.gpus,
        **{f"custom:{k}": v for k, v in capacity.custom.items()},
    }
    if capacity.memory_gib is not None:
        available["memory_gib"] = capacity.memory_gib
    if any(
        value > available.get(key, 0)
        for key, value in demand.items()
        if key != "memory_gib" or key in available
    ):
        return ["peak resource demand exceeds Executor.capacity; work may queue"]
    return []


async def _start_board(config: RunConfig) -> Board:
    """Build the default or configured board with safe connector diagnostics."""
    if config.board is None:
        return ArenaBoard(arena_root=config.arena_root)
    outcome = _Outcome()
    try:
        return await run_sync_daemon(config.board.build, Board)
    except BaseException as error:
        outcome.contain(error, RuntimeError, "board construction")


async def _start_executor(config: RunConfig) -> Executor:
    """Construct an executor within the shared user-code diagnostic boundary."""
    outcome = _Outcome()
    try:
        return await run_sync_daemon(_build_executor, config)
    except BaseException as exc:
        exc = outcome.classify(exc)
        raise _core_error(
            RuntimeError, f"executor construction failed: {_exception_type_name(exc)}"
        ) from None


async def _attach_board_namespaces(deps: EngineDeps, namespaces: set[str]) -> None:
    """Attach admitted namespaces atomically outside revision publication shields."""
    async with deps.board_attach_lock:
        names = deps.board_namespaces | namespaces
        if names != deps.board_namespaces:
            await _open_board(deps.board, deps.config, deps.run_id, sorted(names))
            deps.board_namespaces = names


async def _open_board(
    board: Board, config: RunConfig, run_id: str, namespaces: Sequence[str]
) -> None:
    """Open a board through the same safe, cancellable boundary as construction."""
    outcome = _Outcome()
    try:
        await run_sync_daemon(board.open, run_id, config.run_root, namespaces)
    except BaseException as error:
        outcome.contain(error, RuntimeError, "board open")


async def _close_board(board: Board, config: RunConfig, *, failed: bool) -> None:
    """Bound shielded board cleanup and retain an already-raised primary error."""
    with anyio.move_on_after(config.settle_timeout_s, shield=True) as shutdown:
        outcome = _Outcome()
        try:
            await run_sync_daemon(board.close)
        except BaseException as error:
            if not failed and not shutdown.cancel_called:
                outcome.contain(error, RuntimeError, "board close")


async def _close_executor(executor: Executor, config: RunConfig, *, failed: bool) -> None:
    """Bound shielded shutdown while preserving an already-raised exception."""
    with anyio.move_on_after(config.settle_timeout_s, shield=True) as shutdown:
        outcome = _Outcome()
        try:
            await executor.aclose()
        except BaseException as exc:
            if not failed and not shutdown.cancel_called:
                exc = outcome.classify(exc)
                raise _core_error(
                    RuntimeError, f"executor shutdown failed: {_exception_type_name(exc)}"
                ) from None


def _promote(stage: Path, artifacts: ArtifactStore) -> None:
    staged = ArtifactStore(LocalStore(stage))
    for path in stage.iterdir():
        artifacts.put_bytes(staged.get_bytes(path.name))


def _local_journal_path(run_id: str, epoch: int) -> Path:
    """Locate the local write-ahead journal for one run ownership epoch."""
    return (
        Path(gettempdir()) / f"swarm-arena-{os.getuid()}" / "journals" / f"{run_id}-e{epoch}.jsonl"
    )


class RunRunner:
    """
    Coordinate approval, startup, queued goals and run-wide cancellation.

    Parameters
    ----------
    config : RunConfig
        Human-controlled run services and limits.
    plans : Sequence[Plan]
        Initial plans, retaining each plan's origin and source directory.
    run_id : str or None
        Requested run identifier; None allocates one after approval.
    approve : Callable or None
        Async numeric estimate and warning approval callback.
    replanner : Planner or None
        Optional live revision producer shared by the goals.
    resumed : ResumeState or None, default=None
        Private prepared-service carrier returned by resume_run.

    Examples
    --------
    >>> runner = RunRunner(RunConfig(), [], run_id=None, approve=None, replanner=None)
    >>> runner.run_id is None
    True
    """

    def __init__(
        self,
        config: RunConfig,
        plans: Sequence[Plan],
        *,
        run_id: str | None,
        approve: Callable[[Estimate, Sequence[str]], Awaitable[bool]] | None,
        replanner: Planner | None,
        resumed: "ResumeState | None" = None,
    ) -> None:
        self.config, self.plans = config, list(plans)
        self.run_id, self.approve, self.replanner = run_id, approve, replanner
        self._stop_mode = 0
        self.resumed = resumed
        self._used = False
        self._queue_changed = anyio.Event()

    async def run(self) -> dict[str, list[Candidate]]:
        """
        Admit and approve all goals, then execute under one run lease.

        Returns
        -------
        dict[str, list[Candidate]]
            Finalists in plan order, with empty lists for no-result goals.

        Raises
        ------
        ValueError
            If admission fails or goal identifiers repeat.
        RuntimeError
            If the backend, estimates, approval or preflight refuses the run,
            or ownership is lost. In-flight work is cancelled before returning.

        Notes
        -----
        Preapproval snapshots use a temporary store removed on refusal. Approved
        snapshots are promoted without rereading sources. Estimate hooks run in
        abandonable daemon threads and assume empty, unknown future inputs.
        Board and executor shutdown each use a shielded
        ``config.settle_timeout_s`` deadline.
        On expiry the original result or exception is preserved; shutdown
        failures never replace an exception already raised by the run.

        Examples
        --------
        >>> await runner.run()  # doctest: +SKIP
        {}
        """
        _require_asyncio()
        if self.resumed is not None:
            services = self.resumed._services
            if services.closed or self._used:
                raise _core_error(RuntimeError, "resumed runner is closed or already used")
            self._used = True
            assert services.deps is not None
            failed = False
            try:
                return await self._drive(services.deps, self.resumed.epoch)
            except BaseException as error:
                failed = True
                if _has_fenced(error):
                    raise _core_error(Fenced) from None
                raise
            finally:
                await services._aclose(failed=failed)
        outcome = _Outcome()
        try:
            self.plans = [await run_sync_daemon(_detach_plan, plan) for plan in self.plans]
        except BaseException as error:
            error = outcome.classify(error)
            raise _core_error(PlanError, [f"invalid plan: {_exception_type_name(error)}"]) from None
        seen: set[str] = set()
        for plan in self.plans:
            if plan.goal.id in seen:
                raise _core_error(PlanError, [f"duplicate goal id '{plan.goal.id}'"])
            seen.add(plan.goal.id)
        artifacts = ArtifactStore(open_store(self.config.artifact_root))
        executor: Executor | None = None
        failed = False
        try:
            with TemporaryDirectory(prefix="swarm-arena-approval-") as directory:
                stage = Path(directory)
                staged = ArtifactStore(LocalStore(stage))
                admitted: list[Plan] = []
                warnings: list[str] = []
                for plan in self.plans:
                    if str(plan.base_dir) not in sys.path:
                        sys.path.append(str(plan.base_dir))
                    outcome = _Outcome()
                    try:
                        frozen, notes = await run_sync_daemon(
                            partial(
                                admit_revision,
                                stamp(plan, origin=plan.origin, parent=None),
                                self.config,
                                staged,
                            )
                        )
                    except BaseException as error:
                        classified = outcome.classify(error)
                        if isinstance(classified, PlanError):
                            raise classified from None
                        raise _core_error(
                            PlanError, [f"admission failed: {_exception_type_name(classified)}"]
                        ) from None
                    admitted.append(frozen)
                    warnings.extend(notes)
                if collisions := _goal_key_errors(admitted):
                    raise _core_error(PlanError, collisions)
                self.plans = admitted
                warnings.extend(await _check_startup_models(self.config, self.replanner))
                outcome = _Outcome()
                try:
                    estimate, slices = await run_sync_daemon(_estimates, admitted, self.config)
                except BaseException as exc:
                    exc = outcome.classify(exc)
                    raise _core_error(
                        RuntimeError, f"estimate failed: {_exception_type_name(exc)}"
                    ) from None
                if _exceeds_limit(estimate, self.config):
                    raise _core_error(
                        RuntimeError, "worst-case estimate exceeds the configured run limit"
                    )
                total = sum(slices.values(), Amount())
                if _exceeds_limit(total, self.config, Fraction(str(self.config.reserve_fraction))):
                    raise _core_error(RuntimeError, "reserve cannot hold the final slices")
                executor = await _start_executor(self.config)
                outcome = _Outcome()
                try:
                    capacity = await executor.capacity()
                    invalid_capacity = capacity is not None and not isinstance(capacity, Resources)
                    if not invalid_capacity:
                        if capacity is not None:
                            capacity = await run_sync_daemon(_detach_model, Resources, capacity)
                        warnings.extend(_capacity_warning(admitted, capacity))
                except BaseException as exc:
                    exc = outcome.classify(exc)
                    raise _core_error(
                        RuntimeError, f"executor capacity failed: {_exception_type_name(exc)}"
                    ) from None
                if invalid_capacity:
                    raise _core_error(
                        RuntimeError, f"executor capacity failed: {type(capacity).__name__}"
                    ) from None
                if self.approve is not None:
                    outcome = _Outcome()
                    try:
                        approved = await self.approve(estimate.to_estimate(), warnings)
                    except BaseException as exc:
                        exc = outcome.classify(exc)
                        raise _core_error(
                            RuntimeError, f"approval failed: {_exception_type_name(exc)}"
                        ) from None
                    if type(approved) is not bool:
                        raise _core_error(
                            RuntimeError, f"approval failed: {type(approved).__name__}"
                        ) from None
                    if not approved:
                        raise _core_error(RuntimeError, "run not approved")
                await run_preflight(admitted, self.config, executor)
                self.run_id = self.run_id or new_run_id(utc_now())
                manifest = await build_manifest_async(
                    run_id=self.run_id, config=self.config, plans=admitted, capacity=capacity
                )
                await anyio.to_thread.run_sync(_promote, stage, artifacts)
            return await self._execute(executor, artifacts, estimate, slices, warnings, manifest)
        except BaseException as exc:
            failed = True
            if _has_fenced(exc):
                raise _core_error(Fenced) from None
            raise
        finally:
            if executor is not None:
                await _close_executor(executor, self.config, failed=failed)

    async def _execute(
        self,
        executor: Executor,
        artifacts: ArtifactStore,
        estimate: Amount,
        slices: dict[str, Amount],
        warnings: list[str],
        manifest: dict[str, Any],
    ) -> dict[str, list[Candidate]]:
        from swarm_arena._engine.resume import _supervised_acquire

        assert self.run_id is not None
        config, run_id = self.config, self.run_id
        store = open_store(config.run_root).child(run_id)
        clock, stops = RunClock(0), StopState()
        lease: Lease | None = None
        entered = False
        try:
            lease = await _supervised_acquire(
                store,
                1,
                stops,
                clock,
                config,
                partial(
                    Lease.acquire_new,
                    store,
                    owner_id=uuid4().hex,
                    host=socket.gethostname(),
                    ttl_s=config.lease_ttl_s,
                ),
                stopped="run initialization stopped before run_started",
            )
            local_path = _local_journal_path(run_id, lease.doc.epoch)
            async with _RunJournal(
                store,
                lease,
                run_id=run_id,
                next_seq=1,
                local_path=local_path,
                sync_interval_s=config.sync_interval_s,
                settle_timeout_s=config.settle_timeout_s,
            ) as journal:
                entered = True
                async with _supervise_writer(journal):
                    async with _supervise_initialization(
                        store, lease.doc.epoch, stops, clock, config
                    ) as prepared:
                        journal.clock = clock
                        await _storage_io(_create, store, "run.toml", config.dumps().encode())
                        await _storage_io(
                            _create,
                            store,
                            "manifest.json",
                            canonical_json(
                                RunManifest.model_validate(manifest).model_dump(mode="json")
                            ),
                        )
                        ids = IdGen(run_id, lease.doc.epoch)
                        ledger = Ledger(
                            config.limits,
                            reserve_fraction=config.reserve_fraction,
                            goal_weights=config.goal_weights,
                            journal=journal,
                            ids=ids,
                        )
                        async with _bound_run(
                            run_id,
                            journal=journal,
                            ids=ids,
                            artifacts=artifacts,
                            config=config,
                            stops=stops,
                            clock=clock,
                            state=RunState(),
                        ):
                            journal.append(
                                EventDraft(
                                    type=EventType.RUN_CONFIG,
                                    payload={"config": config.model_dump(mode="json")},
                                )
                            )
                            journal.append(
                                EventDraft(
                                    type=EventType.RUN_STARTED,
                                    payload={
                                        "estimate": estimate.to_estimate().model_dump(mode="json"),
                                        "warnings": warnings,
                                        "manifest": manifest,
                                        "final_slices": {
                                            g: a.model_dump() for g, a in slices.items()
                                        },
                                    },
                                )
                            )
                            await ledger.hold_final_slices(slices)
                            for plan in self.plans:
                                journal.append(r0_revision_draft(plan, plan.base_dir))
                                journal.append(
                                    EventDraft(type=EventType.GOAL_QUEUED, goal_id=plan.goal.id)
                                )
                            head = await journal.commit()
                            assert head is not None
                            views = ViewWriter(store, journal)
                            for plan in self.plans:
                                await views.write(
                                    f"goals/{plan.goal.id}/plan.r0.toml",
                                    plan.dumps().encode(),
                                    source=head,
                                )
                            await _setup_run(self.plans, config)
                            board = await _start_board(config)
                            board_failed = False
                            try:
                                await _open_board(
                                    board,
                                    config,
                                    run_id,
                                    sorted(
                                        {p.goal.namespace for p in self.plans}
                                        | {
                                            part.goal.namespace
                                            for p in self.plans
                                            for part in p.parts
                                        }
                                    ),
                                )
                                deps = EngineDeps(
                                    run_id=run_id,
                                    config=config,
                                    store=store,
                                    journal=journal,
                                    ledger=ledger,
                                    board=board,
                                    executor=executor,
                                    artifacts=artifacts,
                                    ids=ids,
                                    views=views,
                                    stops=stops,
                                    clock=clock,
                                    admission=RunAdmission(
                                        config.limits.max_agents_total, self.plans
                                    ),
                                    board_namespaces={
                                        goal.namespace
                                        for plan in self.plans
                                        for goal in [plan.goal, *(part.goal for part in plan.parts)]
                                    },
                                )
                                prepared()
                                return await self._drive(deps, lease.doc.epoch)
                            except BaseException:
                                board_failed = True
                                raise
                            finally:
                                await _close_board(board, config, failed=board_failed)
                    return await _finish_initialization_stop(store, journal, stops, config)
        finally:
            if lease is not None and not entered:
                with anyio.move_on_after(config.settle_timeout_s, shield=True):
                    with suppress(BaseException):
                        await _storage_io(lease.release)

    async def aclose(self) -> None:
        """
        Release a prepared resume's writer, lease, board and executor once.

        Examples
        --------
        >>> runner = RunRunner(RunConfig(), [], run_id=None, approve=None, replanner=None)
        >>> import anyio
        >>> anyio.run(runner.aclose)
        """
        if self.resumed is not None:
            await self.resumed._services._aclose()

    def _wake_queue(self) -> None:
        self._queue_changed.set()

    async def _drive(self, deps: EngineDeps, epoch: int) -> dict[str, list[Candidate]]:
        assert deps.admission is not None
        deps.admission.on_change = self._wake_queue

        def changed() -> None:
            self._wake_queue()
            if deps.stops.stop or deps.stops.stop_now or deps.stops.run_timeout:
                deps.journal._stop_storage()

        deps.stops.on_change = changed
        changed()
        result: dict[str, list[Candidate]] = {}
        error: BaseException | None = None
        async with _supervise_writer(deps.journal):
            async with anyio.create_task_group() as watchers:
                try:
                    watchers.start_soon(watch_stop_files, deps.store, epoch, deps.stops)
                    watchers.start_soon(self._watch_clock, deps)
                    result = await self._queue(deps)
                    await self._record_stops(deps)
                    with anyio.CancelScope() as completion:

                        def stopping() -> None:
                            deps.journal._stop_storage()
                            completion.deadline = min(
                                completion.deadline,
                                anyio.current_time() + deps.config.settle_timeout_s,
                            )

                        deps.stops.on_change = stopping
                        if self._stopping(deps):
                            stopping()
                        await _finish_run(deps.store, deps.journal)
                    if completion.cancel_called:
                        raise _core_error(RuntimeError, "run completion timed out")
                except BaseException as caught:
                    error = caught
                finally:
                    watchers.cancel_scope.cancel()
            if error is not None:
                raise error
        return result

    async def _watch_clock(self, deps: EngineDeps) -> None:
        await anyio.sleep(deps.clock.left(deps.config.limits.run_timeout_s))
        deps.stops.run_timeout = True

    async def _record_stops(self, deps: EngineDeps) -> None:
        self._stop_mode = await _record_stop(deps.journal, deps.stops, self._stop_mode)

    async def _queued_result(self, deps: EngineDeps, goal_id: str) -> None:
        with _stop_cleanup(deps.config):
            await _finish_queued_goal(deps.journal, deps.ledger, deps.store, deps.views, goal_id)

    def _stopping(self, deps: EngineDeps) -> bool:
        """Recheck live run limits at every goal admission boundary."""
        deps.stops.run_budget |= deps.ledger.run_exhausted()
        if deps.clock.left(deps.config.limits.run_timeout_s) <= 0:
            deps.stops.run_timeout = True
        return (
            deps.stops.stop
            or deps.stops.stop_now
            or deps.stops.run_timeout
            or deps.stops.run_budget
        )

    async def _queue(self, deps: EngineDeps) -> dict[str, list[Candidate]]:
        state = self.resumed.state if self.resumed is not None else None
        order = (
            self.resumed._services.order
            if self.resumed is not None
            else [p.goal.id for p in self.plans]
        )
        plans = {p.goal.id: p for p in self.plans}
        queued = [
            GoalRunner(
                deps,
                goal_id=g,
                plan=plans[g],
                replanner=self.replanner,
                state=state.goals[g] if state is not None else None,
                journal_r0=False,
            )
            for g in order
        ]
        active: dict[str, GoalRunner] = {}
        results: dict[str, list[Candidate]] = {}
        if state is not None:
            head = deps.journal.head
            assert head is not None
            for g, goal in state.goals.items():
                if g not in order:
                    results[g] = [state.candidates[id] for id in goal.final_order]
                    completed = GoalRunner(
                        deps,
                        goal_id=g,
                        plan=state.plans[g][-1][0],
                        replanner=None,
                        state=goal,
                        journal_r0=False,
                    )
                    await completed._write_view(
                        f"goals/{g}/solution.json", solution_view(state, g, head), source=head
                    )
        assert deps.admission is not None

        async def run_goal(runner: GoalRunner) -> None:
            out = await runner.run()
            results[runner.goal_id] = out.finalists
            active.pop(runner.goal_id)
            self._wake_queue()

        error: BaseException | None = None
        async with anyio.create_task_group() as goals:
            try:
                while queued or active:
                    changed = self._queue_changed = anyio.Event()
                    await self._record_stops(deps)
                    if self._stopping(deps):
                        for runner in list(queued):
                            if state is not None and state.goals[runner.goal_id].status != "queued":
                                continue
                            await self._queued_result(deps, runner.goal_id)
                            results[runner.goal_id] = []
                            queued.remove(runner)
                    while queued and len(active) < deps.config.limits.max_concurrent_goals:
                        runner = queued[0]
                        fresh = state is None or state.goals[runner.goal_id].status == "queued"
                        if self._stopping(deps) and fresh:
                            break
                        if (
                            sum(g.active_agents() for g in active.values()) + runner.active_agents()
                            > deps.config.limits.max_agents_total
                        ):
                            break
                        try:
                            await deps.admission.update(runner.goal_id, runner.active_agents())
                        except PlanError:
                            # A revision may have reserved a larger peak during publication.
                            break
                        if self._stopping(deps) and fresh:
                            await deps.admission.release(runner.goal_id)
                            break
                        queued.pop(0)
                        if fresh:
                            deps.ledger.open_goal(runner.goal_id)
                            await deps.journal.append_durable(
                                EventDraft(type=EventType.GOAL_STARTED, goal_id=runner.goal_id)
                            )
                        active[runner.goal_id] = runner
                        goals.start_soon(run_goal, runner)
                    if queued or active:
                        await changed.wait()
            except BaseException as caught:
                error = caught
            finally:
                goals.cancel_scope.cancel()
        if error is not None:
            raise error
        return {p.goal.id: results[p.goal.id] for p in self.plans}


async def _record_stop(journal: JournalWriter, stops: StopState, mode: int = 0) -> int:
    """Record each stronger explicit stop command once."""
    current = 2 if stops.stop_now else 1 if stops.stop else 0
    if current > mode:
        await journal.append_durable(
            EventDraft(type=EventType.STOP_REQUESTED, payload={"now": current == 2})
        )
    return max(mode, current)


@contextmanager
def _stop_cleanup(config: RunConfig) -> Generator[None]:
    """Give both stop cleanup paths one bounded, value-free failure contract."""
    try:
        with anyio.fail_after(config.settle_timeout_s, shield=True):
            yield
    except TimeoutError:
        raise _stop_cleanup_error() from None


async def _finish_queued_goal(
    journal: JournalWriter, ledger: Ledger, store: Store, views: ViewWriter, goal_id: str
) -> None:
    """Publish a queued goal's terminal result under the caller's cleanup bound."""
    journal.append(
        EventDraft(type=EventType.NO_RESULT, goal_id=goal_id, payload={"reason": "stopped"})
    )
    journal.append(
        EventDraft(type=EventType.GOAL_FINISHED, goal_id=goal_id, payload={"reason": "stopped"})
    )
    head = await journal.commit()
    await ledger.end_goal(goal_id)
    assert head is not None
    chain = await _read_chain(store, head)
    state = RunState.from_events(e for _, events in chain for e in events)
    await views.write(
        f"goals/{goal_id}/solution.json", solution_view(state, goal_id, head), source=head
    )


async def _finish_run(store: Store, journal: JournalWriter) -> RunState:
    """Export the committed run without consulting a board."""
    head = await journal.commit()
    assert head is not None
    chain = await _read_chain(store, head)
    state = RunState.from_events(e for _, batch in chain for e in batch)
    await write_review_export(state, store, head)
    return state


async def _finish_initialization_stop(
    store: Store, journal: JournalWriter, stops: StopState, config: RunConfig
) -> dict[str, list[Candidate]]:
    """Close committed queued goals while the initializer still owns its lease."""
    with _stop_cleanup(config):
        head = await journal.commit()
        if head is None:
            raise _core_error(RuntimeError, "run initialization stopped before run_started")
        chain = await _read_chain(store, head)
        events = [event for _, batch in chain for event in batch]
        if not any(event.type == EventType.RUN_STARTED for event in events):
            raise _core_error(RuntimeError, "run initialization stopped before run_started")
        state = RunState.from_events(events)
        ids = IdGen(state.run_id, journal.epoch, counters=state.id_counters)
        ledger = Ledger.replay(state, journal=journal, ids=ids)
        views = ViewWriter(store, journal)
        await _record_stop(journal, stops)
        for goal_id, goal in state.goals.items():
            if goal.status == "queued":
                await _finish_queued_goal(journal, ledger, store, views, goal_id)
        state = await _finish_run(store, journal)
        return {
            goal_id: [state.candidates[candidate_id] for candidate_id in goal.final_order]
            for goal_id, goal in state.goals.items()
        }


async def write_review_export(state: RunState, store: Store, head: Head) -> None:
    """
    Write an immutable review bundle for a committed completion head.

    Parameters
    ----------
    state : RunState
        Replay of exactly the supplied committed head.
    store : Store
        Run-scoped storage.
    head : Head
        Source head naming this immutable export directory.

    Raises
    ------
    StorageError
        If an existing export file has different bytes.

    Examples
    --------
    >>> await write_review_export(state, store, head)  # doctest: +SKIP
    """
    from swarm_arena._scoring.human_review import _review_io, render_review

    files = render_review(state, top=20)

    for name, data in files.items():
        await anyio.lowlevel.checkpoint_if_cancelled()
        key = f"review/e{head.epoch}-{head.seq}/{name}"
        try:
            await _review_io(_create, store, key, data)
        except AlreadyExists:
            existing = await _review_io(_read_blob, store, key)
            if existing.data != data:
                raise _core_error(StorageError, "review export bytes differ") from None


def mark_planning_session(plan: Plan, scope: LLMScope) -> Plan:
    """Copy a plan with its standalone session and completed planner calls.

    Parameters
    ----------
    plan : Plan
        Generated plan whose model fields remain unchanged.
    scope : LLMScope
        Planning scope after its calls have drained.

    Returns
    -------
    Plan
        Copy retaining its base directory and validated private provenance.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> from swarm_arena import Goal
    >>> plan = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])
    >>> mark_planning_session(plan, Mock(run_id="r-p", call_ids=["llm-1"])).planning_session
    'r-p'
    """
    marked = plan.model_copy()
    marked._set_planner_provenance(scope.run_id, scope.call_ids)
    return marked


def _planning_models(config: RunConfig, goal: Goal, planner: Planner) -> tuple[Goal, list[str]]:
    """Detach the human goal and validate only the planner before any writes."""
    goal = _detach_model(Goal, goal)
    aliases = [_context_string(alias, "alias") for alias in planner.models()]
    errors, warnings = check_models({"planner": aliases}, config)
    if errors:
        raise _core_error(ValueError, "\n".join(errors)) from None
    return goal, warnings


@asynccontextmanager
async def planning_session(
    config: RunConfig, goal: Goal, planner: Planner
) -> AsyncGenerator[LLMScope]:
    """Own a journaled planning lifetime without starting an executor or board.

    Parameters
    ----------
    config : RunConfig
        Human model catalog, budgets, storage and settlement settings.
    goal : Goal
        Human goal receiving the planner allocation.
    planner : Planner
        Component declaring its required model aliases.

    Yields
    ------
    LLMScope
        Planner scope under active-time, stop and lease supervision.

    Raises
    ------
    ValueError
        If model checks fail before storage is written.
    RuntimeError
        If initialization or planning stops before successful completion.

    Examples
    --------
    >>> async with planning_session(config, goal, planner) as scope:  # doctest: +SKIP
    ...     result = await planner.plan(goal, config, None)
    """
    from swarm_arena._engine.resume import _supervised_acquire

    _require_asyncio()
    boundary = _Outcome()
    try:
        goal, warnings = await run_sync_daemon(_planning_models, config, goal, planner)
    except BaseException as error:
        boundary.contain(error, ValueError, "planning model checks")
    run_id = new_run_id(utc_now())
    store = await _storage_io(lambda: open_store(config.run_root).child(run_id))
    artifacts = await _storage_io(lambda: ArtifactStore(open_store(config.artifact_root)))
    clock, stops = RunClock(0), StopState()
    lease: Lease | None = None
    entered = completed = False
    try:
        lease = await _supervised_acquire(
            store,
            1,
            stops,
            clock,
            config,
            partial(
                Lease.acquire_new,
                store,
                owner_id=uuid4().hex,
                host=socket.gethostname(),
                ttl_s=config.lease_ttl_s,
            ),
            stopped="planning session stopped",
        )
        async with (
            _supervise_initialization(store, 1, stops, clock, config),
            _RunJournal(
                store,
                lease,
                run_id=run_id,
                next_seq=1,
                local_path=_local_journal_path(run_id, lease.doc.epoch),
                sync_interval_s=config.sync_interval_s,
                settle_timeout_s=config.settle_timeout_s,
            ) as journal,
        ):
            entered = True
            journal.clock = clock
            async with _supervise_writer(journal):
                ids = IdGen(run_id, lease.doc.epoch)
                ledger = Ledger(
                    config.limits,
                    reserve_fraction=config.reserve_fraction,
                    goal_weights=config.goal_weights,
                    journal=journal,
                    ids=ids,
                )
                async with _bound_run(
                    run_id,
                    journal=journal,
                    ids=ids,
                    artifacts=artifacts,
                    config=config,
                    stops=stops,
                    clock=clock,
                    state=RunState(),
                ):
                    journal.append(
                        EventDraft(
                            type=EventType.RUN_CONFIG,
                            payload={"config": config.model_dump(mode="json")},
                        )
                    )
                    journal.append(
                        EventDraft(
                            type=EventType.RUN_STARTED,
                            payload={
                                "planning": True,
                                "warnings": warnings,
                                "final_slices": {goal.id: Amount().model_dump()},
                            },
                        )
                    )
                    await ledger.hold_final_slices({goal.id: Amount()})
                    ledger.open_goal(goal.id)
                    journal.append(EventDraft(type=EventType.GOAL_STARTED, goal_id=goal.id))
                    tranche = Tranche(
                        ledger,
                        f"{goal.id}/:planner",
                        goal.id,
                        size=Amount(
                            tokens=config.limits.llm_tranche_tokens,
                            usd=config.limits.llm_tranche_usd,
                        ),
                    )
                    scope = LLMScope(
                        role="planner",
                        run_id=run_id,
                        goal_id=goal.id,
                        team="",
                        agent="",
                        round=0,
                        candidate_id=None,
                        tranche=tranche,
                        journal=journal,
                        artifacts=artifacts,
                        models=config.models,
                        tracing=config.tracing,
                        cache=config.cache,
                        limits=config.limits,
                        settle_timeout_s=config.settle_timeout_s,
                    )
                    failed = True
                    failure: BaseException | None = None
                    try:
                        await journal.commit()
                        await _storage_io(_create, store, "run.toml", config.dumps().encode())
                        await _setup_run((), config)
                        await anyio.lowlevel.checkpoint_if_cancelled()
                        with llm_scope(scope):
                            yield scope
                        await anyio.lowlevel.checkpoint_if_cancelled()
                        failed = False
                    except BaseException as error:
                        failure = error
                        raise
                    finally:
                        try:
                            if not failed:
                                await drain(scope, wait=True)
                        except BaseException as error:
                            failure = error
                            raise
                        finally:
                            with _stop_cleanup(config):
                                await drain(scope, wait=False)
                                outcome = await tranche.close()
                                reason = (
                                    "planner failed" if failure is not None else "planning complete"
                                )
                                if failure is not None:
                                    try:
                                        classified = boundary.classify(failure)
                                    except anyio.get_cancelled_exc_class():
                                        reason = "stopped"
                                    else:
                                        if _core_category(classified) in {
                                            _SignalCategory.LIMIT,
                                            _SignalCategory.TERMINAL,
                                        }:
                                            reason = "run budget"
                                if outcome.overflow or stops.run_budget:
                                    reason = "run budget"
                                if stops.run_timeout:
                                    reason = "run_timeout_s"
                                if stops.stop or stops.stop_now:
                                    reason = "stop --now" if stops.stop_now else "stop"
                                await _record_stop(journal, stops)
                                journal.append(
                                    EventDraft(
                                        type=EventType.GOAL_FINISHED,
                                        goal_id=goal.id,
                                        payload={"reason": reason},
                                    )
                                )
                                await ledger.end_goal(goal.id)
                                await journal.commit()
                                if outcome.overflow:
                                    ledger._check_usage_overflow()
                    completed = True
        if not completed or stops.stop or stops.stop_now or stops.run_timeout:
            raise _core_error(RuntimeError, "planning session stopped") from None
    finally:
        if lease is not None and not entered:
            with anyio.move_on_after(config.settle_timeout_s, shield=True):
                with suppress(BaseException):
                    await _storage_io(lease.release)
