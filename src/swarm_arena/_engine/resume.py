"""Take ownership of a committed run and retain its reconstructed services."""

import asyncio
import sys
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar
from uuid import uuid4

import anyio

from swarm_arena._board.base import Board
from swarm_arena._budget.ledger import Ledger
from swarm_arena._budget.split import weighted_shares
from swarm_arena._engine import run as run_module
from swarm_arena._engine.admission import RunAdmission
from swarm_arena._engine.goal import _repair_round_pages
from swarm_arena._engine.preflight import _board_arena_root, _check_storage
from swarm_arena._engine.round import EngineDeps
from swarm_arena._engine.stops import RunClock, StopState
from swarm_arena._errors import (
    Fenced,
    PlanError,
    StorageError,
    _core_error,
    _exception_type_name,
)
from swarm_arena._exec.executor import Executor
from swarm_arena._exec.procinfo import host_id
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.lease import Head, Lease, OwnerDoc, read_owner
from swarm_arena._journal.reader import _read_segment, read_chain
from swarm_arena._journal.replay import RunState, continues_on_resume
from swarm_arena._journal.views import ViewWriter
from swarm_arena._model.ids import IdGen
from swarm_arena._model.run_config import RunConfig
from swarm_arena._outcomes import _Outcome
from swarm_arena._planning.planner import Planner
from swarm_arena._planning.validate import _rule_max_agents_total
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import Store, open_store
from swarm_arena._store.io import _storage_io, _validated_io


@dataclass
class ResumeState:
    """
    Retain recovered authority, accounting and the shared service owner.

    Attributes
    ----------
    state : RunState
        Replayed state after committed recovery and lifecycle reopens.
    lease : Lease
        Taken-over authority, maintained by the service owner's writer.
    epoch : int
        New ownership epoch.
    ledger : Ledger
        History-replayed ledger bound to that same live writer.
    orphans_killed : list[str]
        Job identifiers whose executor verified and killed their process group.

    Examples
    --------
    >>> from dataclasses import fields
    >>> fields(ResumeState)[0].name
    'state'
    """

    state: RunState
    lease: Lease
    epoch: int
    ledger: Ledger
    orphans_killed: list[str]
    _services: "_ResumeServices"


def _read(store: Store, doc: OwnerDoc) -> tuple[list[Event], RunState]:
    if doc.head is None:
        raise _core_error(ValueError, "run has no committed journal")
    events = [
        event
        for _, batch in read_chain(store, doc.head, _read=partial(_read_segment, store))
        for event in batch
    ]
    if any(
        event.type == EventType.RUN_STARTED and event.payload.get("planning") for event in events
    ):
        raise _core_error(ValueError, "run is a planning session; it cannot be resumed")
    return events, RunState.from_events(events)


def _check(state: RunState, config: RunConfig) -> list[str]:
    """Reject intrinsic limit incompatibilities without mutating authority."""
    for name in ("max_tokens", "max_usd"):
        old, new = getattr(state.config.limits, name), getattr(config.limits, name)
        if new is not None and (old is None or new < old):
            raise _core_error(ValueError, f"budget cap may only increase: {name}")
    continued = [g for g in state.plans if continues_on_resume(state, g, config.limits)]
    errors: list[str] = []
    for goal_id in state.plans:
        goal = state.goals[goal_id]
        if not goal.finished or goal_id in continued:
            active = state.plan_for_round(goal_id, goal.rounds_closed + 1)
            latest = state.plans[goal_id][-1][0]
            for plan in [active, latest] if active != latest else [active]:
                errors.extend(
                    f"goal '{goal_id}': {error}"
                    for error in _rule_max_agents_total(plan, config, (), other_active_agents=0)
                )
    if errors:
        raise _core_error(PlanError, errors)
    return continued


def _restore_import_paths(state: RunState) -> None:
    """Make recorded component locations available before invoking their hooks."""
    for revisions in state.plans.values():
        directory = str(revisions[0][0].base_dir)
        if directory not in sys.path:
            sys.path.append(directory)


class _ResumeServices:
    """Keep the AnyIO writer context in one task across preparation and execution."""

    def __init__(
        self,
        config: RunConfig,
        run_id: str,
        store: Store,
        lease: Lease,
        *,
        force: bool,
        clock: RunClock,
        validated_head: Head,
        model_warnings: list[str],
    ) -> None:
        self.config, self.run_id, self.store, self.lease = config, run_id, store, lease
        self.force = force
        self.clock = clock
        self.validated_head = validated_head
        self.model_warnings = model_warnings
        self.ready, self.close = anyio.Event(), anyio.Event()
        self.failure: BaseException | None = None
        self.resumed: ResumeState | None = None
        self.deps: EngineDeps | None = None
        self.order: list[str] = []
        self.task: asyncio.Task[None] | None = None
        self.closed = False
        self.run_failed = False
        self.preparation_scope: anyio.CancelScope | None = None
        self.cancel_preparation = False

    async def _serve(self) -> None:
        """Let a cancelled preparing caller interrupt abandonable user hooks."""
        with anyio.CancelScope() as scope:
            self.preparation_scope = scope
            if self.cancel_preparation:
                scope.cancel()
            await self._serve_owned()

    async def _serve_owned(self) -> None:
        """Own entered services and shield their existing bounded cleanup paths."""
        executor: Executor | None = None
        board: Board | None = None
        entered = False
        config, lease = self.config, self.lease
        head = lease.doc.head
        assert head is not None
        try:
            path = run_module._local_journal_path(self.run_id, lease.doc.epoch)
            stops = StopState()
            clock = self.clock
            journal = run_module._RunJournal(
                self.store,
                lease,
                run_id=self.run_id,
                next_seq=head.seq + 1,
                local_path=path,
                sync_interval_s=config.sync_interval_s,
                settle_timeout_s=config.settle_timeout_s,
            )
            if head != self.validated_head:
                validated = False
                async with run_module._supervise_initialization(
                    self.store, lease.doc.epoch, stops, clock, config
                ):
                    # Reuse the writer's confirmed-expiry bound before it owns entry.
                    _, acquired = await journal._retry(
                        partial(_read, self.store, lease.doc), validated=True
                    )
                    clock._rebase(acquired.run_elapsed_s)
                    # Configuration rechecks follow the durable takeover below;
                    # acquiring a newer valid chain must still leave its receipt.
                    validated = True
                if not validated:
                    raise _core_error(RuntimeError, "resume initialization stopped")
            async with (
                run_module._supervise_initialization(
                    self.store, lease.doc.epoch, stops, clock, config
                ) as replay_prepared,
                journal,
            ):
                entered = True
                journal.clock = clock
                # The journal owns release even if supervision entry is cancelled.
                async with run_module._supervise_writer(journal):
                    board_failed = False
                    try:
                        # A successful force CAS must be auditable even if replay now refuses.
                        await journal.append_durable(
                            EventDraft(
                                type=EventType.TAKEOVER,
                                payload={"force": self.force, "warnings": self.model_warnings},
                            )
                        )
                        events, state = await _validated_io(_read, self.store, lease.doc)
                        replay_prepared()
                        async with run_module._supervise_initialization(
                            self.store, lease.doc.epoch, stops, clock, config
                        ) as prepared:
                            from swarm_arena._scoring.human_review import _repair_review_objects

                            arena_root = await run_sync_daemon(_board_arena_root, config)
                            arena = await _storage_io(open_store, arena_root)
                            await _repair_review_objects(state, arena, self.store, lease, journal)
                            continued = _check(state, config)
                            self.order = [
                                g
                                for g in state.plans
                                if not state.goals[g].finished and state.goals[g].status != "queued"
                            ]
                            self.order.extend(continued)
                            self.order.extend(
                                g for g in state.plans if state.goals[g].status == "queued"
                            )
                            _restore_import_paths(state)
                            ids = IdGen(self.run_id, lease.doc.epoch, counters=state.id_counters)
                            ledger = Ledger.replay(state, journal=journal, ids=ids)
                            artifacts = ArtifactStore(
                                await _storage_io(open_store, config.artifact_root)
                            )
                            async with run_module._bound_run(
                                self.run_id,
                                journal=journal,
                                ids=ids,
                                artifacts=artifacts,
                                config=config,
                                stops=stops,
                                clock=clock,
                                state=state,
                            ):
                                executor = await run_module._start_executor(config)
                                killed: list[str] = []
                                starts = {
                                    e.payload["candidate_id"]: e
                                    for e in events
                                    if e.type == EventType.TOOL_START
                                }
                                commands = {
                                    e.payload["job_id"]: e
                                    for e in events
                                    if e.type == EventType.COMMAND_START
                                }
                                for jobs, index, event_type in (
                                    (state.started, starts, EventType.RUN_LOST),
                                    (state.started_commands, commands, EventType.COMMAND_END),
                                ):
                                    for id, started in jobs.items():
                                        outcome = _Outcome()
                                        try:
                                            with anyio.move_on_after(config.settle_timeout_s):
                                                if await executor.kill_orphan(started):
                                                    killed.append(id)
                                        except BaseException as error:
                                            error = outcome.classify(error)
                                            name = _exception_type_name(error)
                                            raise _core_error(
                                                RuntimeError,
                                                f"orphan recovery failed: {name}",
                                            ) from None
                                        source = index[id]
                                        payload = (
                                            {
                                                "candidate_id": id,
                                                "reservation_id": source.payload["reservation_id"],
                                            }
                                            if event_type == EventType.RUN_LOST
                                            else {"job_id": id, "status": "lost"}
                                        )
                                        journal.append(
                                            EventDraft(
                                                type=event_type,
                                                task_id=source.task_id,
                                                goal_id=source.goal_id,
                                                team=source.team,
                                                agent=source.agent,
                                                round=source.round,
                                                payload=payload,
                                            )
                                        )
                                shares = weighted_shares(
                                    ledger.free(),
                                    {g: (config.goal_weights or {}).get(g, 1.0) for g in continued},
                                )
                                for g in continued:
                                    ledger.open_goal(g, pool=shares[g])
                                    journal.append(
                                        EventDraft(
                                            type=EventType.GOAL_STARTED,
                                            goal_id=g,
                                            payload={"pool": shares[g].model_dump()},
                                        )
                                    )
                                if config.limits != state.config.limits:
                                    ledger.change_limits(config.limits)
                                    journal.append(
                                        EventDraft(
                                            type=EventType.LIMITS_CHANGED,
                                            payload={
                                                "old": state.config.limits.model_dump(mode="json"),
                                                "new": config.limits.model_dump(mode="json"),
                                            },
                                        )
                                    )
                                await journal.commit()
                                _, state = await _validated_io(_read, self.store, lease.doc)
                                await run_module._setup_run(
                                    [revisions[-1][0] for revisions in state.plans.values()], config
                                )
                                board = await run_module._start_board(config)
                                await run_module._open_board(
                                    board,
                                    config,
                                    self.run_id,
                                    sorted(
                                        {
                                            goal.namespace
                                            for revisions in state.plans.values()
                                            for plan, _ in revisions
                                            for goal in [
                                                plan.goal,
                                                *(part.goal for part in plan.parts),
                                            ]
                                        }
                                    ),
                                )
                                self.deps = EngineDeps(
                                    run_id=self.run_id,
                                    config=config,
                                    store=self.store,
                                    journal=journal,
                                    ledger=ledger,
                                    board=board,
                                    executor=executor,
                                    artifacts=artifacts,
                                    ids=ids,
                                    views=ViewWriter(self.store, journal),
                                    board_namespaces={
                                        goal.namespace
                                        for revisions in state.plans.values()
                                        for plan, _ in revisions
                                        for goal in [plan.goal, *(part.goal for part in plan.parts)]
                                    },
                                    stops=stops,
                                    clock=clock,
                                    admission=RunAdmission(
                                        config.limits.max_agents_total,
                                        [
                                            plan
                                            for revisions in state.plans.values()
                                            for plan, _ in revisions
                                        ],
                                    ),
                                )
                                await _repair_round_pages(self.deps, state)
                                self.resumed = ResumeState(
                                    state, lease, lease.doc.epoch, ledger, killed, self
                                )
                                prepared()
                                self.ready.set()
                                await self.close.wait()
                        if self.resumed is None:
                            raise _core_error(RuntimeError, "resume initialization stopped")
                    except BaseException:
                        board_failed = True
                        raise
                    finally:
                        if board is not None:
                            await run_module._close_board(
                                board,
                                config,
                                failed=board_failed or self.run_failed,
                            )
            if self.resumed is None:
                raise _core_error(RuntimeError, "resume initialization stopped")
        except BaseException as error:
            self.failure = error
        finally:
            with anyio.move_on_after(config.settle_timeout_s, shield=True):
                try:
                    if not entered:
                        await _storage_io(lease.release)
                except Exception as error:
                    if self.failure is None:
                        self.failure = _core_error(
                            RuntimeError, f"resume cleanup failed: {_exception_type_name(error)}"
                        )
                if executor is not None:
                    try:
                        await run_module._close_executor(
                            executor,
                            config,
                            failed=self.failure is not None or self.run_failed,
                        )
                    except BaseException as error:
                        if self.failure is None:
                            self.failure = error
            self.closed = True
            self.ready.set()

    async def _aclose(self, *, failed: bool = False) -> None:
        self.run_failed = self.run_failed or failed
        self.close.set()
        if failed and self.resumed is None:
            self.cancel_preparation = True
            if self.preparation_scope is not None:
                self.preparation_scope.cancel()
        if self.task is not None:
            with anyio.move_on_after(self.config.settle_timeout_s, shield=True):
                # Timeout bounds the caller without injecting asyncio cancellation
                # into the owner's shielded writer/lease/executor cleanup.
                await asyncio.shield(self.task)
        if self.failure is not None and not failed:
            if run_module._has_fenced(self.failure):
                raise _core_error(Fenced) from None
            raise self.failure


async def _acquire_lease(acquire: Callable[[], Lease], *, settle_timeout_s: float) -> Lease:
    """Release a successful CAS if cancellation wins its delivery handshake."""
    lock = threading.Lock()
    acquired: list[Lease] = []
    abandoned = False

    def work() -> Lease:
        lease = acquire()
        with lock:
            release = abandoned
            if not release:
                acquired.append(lease)
        if release:
            # This daemon owns late cleanup, independent of caller cancellation.
            with suppress(Fenced):
                lease.release()
        return lease

    outcome = _Outcome()
    await anyio.lowlevel.checkpoint_if_cancelled()
    try:
        return await run_sync_daemon(work)
    except BaseException as error:
        with lock:
            abandoned = True
            lease = acquired.pop() if acquired else None
        if lease is not None:
            with anyio.move_on_after(settle_timeout_s, shield=True):
                # Cleanup cannot replace the original outcome, including caller cancellation.
                with suppress(BaseException):
                    await _storage_io(lease.release)
        outcome.contain(error, StorageError, "lease acquisition")


async def _supervised_acquire(
    store: Store,
    epoch: int,
    stops: StopState,
    clock: RunClock | None,
    config: RunConfig,
    acquire: Callable[[], Lease],
    *,
    stopped: str,
) -> Lease:
    """Deliver acquisition under the owner's supervisor or release undelivered ownership."""
    lease: Lease | None = None
    try:
        async with run_module._supervise_initialization(store, epoch, stops, clock, config):
            lease = await _acquire_lease(acquire, settle_timeout_s=config.settle_timeout_s)
            await anyio.lowlevel.checkpoint_if_cancelled()
        if lease is None or stops.stop or stops.stop_now or stops.run_timeout:
            raise _core_error(RuntimeError, stopped)
        return lease
    except BaseException:
        if lease is not None:
            with anyio.move_on_after(config.settle_timeout_s, shield=True):
                with suppress(BaseException):
                    await _storage_io(lease.release)
        raise


_T = TypeVar("_T")


async def _lease_io(operation: Callable[..., _T], *args: Any) -> _T:
    """Read takeover authority under the shared storage outcome boundary."""
    return await _storage_io(operation, *args, site="lease preparation")


async def _take_over(
    config: RunConfig,
    run_id: str,
    store: Store,
    *,
    force: bool,
    validate: Callable[[OwnerDoc], None] | None = None,
    before_acquire: Callable[[], Awaitable[None]] | None = None,
    review: bool = False,
    clock: Callable[[], RunClock | None] | None = None,
    stops: StopState | None = None,
) -> Lease:
    """Retry only same-epoch CAS races, with caller prechecks before each attempt."""
    stops = StopState() if stops is None else stops
    while True:
        owner = await _lease_io(read_owner, store)
        if owner is None:
            if review:
                raise FileNotFoundError(f"run {run_id} has no control/owner.json")
            raise _core_error(ValueError, "run has no committed owner")
        doc, _ = owner
        if doc.head is None:
            raise _core_error(ValueError, f"run {run_id} has no committed events")
        if validate is not None:
            await _validated_io(validate, doc)
        if before_acquire is not None:
            await before_acquire()
        try:
            return await _supervised_acquire(
                store,
                doc.epoch + 1,
                stops,
                clock() if clock is not None else None,
                config,
                partial(
                    Lease.take_over,
                    store,
                    owner_id=uuid4().hex,
                    host=host_id(),
                    ttl_s=config.lease_ttl_s,
                    force=force,
                    expect_epoch=doc.epoch,
                ),
                stopped="review import stopped" if review else "resume initialization stopped",
            )
        except Fenced:
            current = await _lease_io(read_owner, store)
            if current is None or current[0].epoch != doc.epoch:
                epoch = current[0].epoch if current else "missing"
                message = (
                    f"another owner took the run (epoch {epoch})"
                    if review
                    else "a concurrent resume took the run"
                )
                raise _core_error(Fenced, f"run {run_id}: {message}") from None


async def resume_run(
    config: RunConfig, run_id: str, *, force: bool, replanner: Planner | None
) -> run_module.RunRunner:
    """
    Take over a run, recover committed state, and return its live coordinator.

    Parameters
    ----------
    config : RunConfig
        Resume services and limits. Budget caps may increase or become unlimited.
    run_id : str
        Existing run identifier.
    force : bool
        Permit takeover of an unexpired lease.
    replanner : Planner or None
        Optional revision producer for future rounds.

    Returns
    -------
    RunRunner
        Prepared runner owning one writer and executor. Call ``run`` or ``aclose``
        to release its services; preparation already commits recovery events.

    Raises
    ------
    ValueError
        A goal's agent peak exceeds the run-wide limit, no committed state
        exists, a replay payload is invalid, a budget cap decreases, or startup
        model validation refuses the replanner or curator before takeover, or
        the journal belongs to a standalone planning session.
    ChainError
        The committed journal chain is broken or a segment is missing.
    RuntimeError
        A live owner requires force, a competing resume advances the epoch,
        a runtime service fails, or stop/run expiry interrupts preparation.

    Examples
    --------
    >>> runner = await resume_run(config, "run-id", force=False, replanner=None)  # doctest: +SKIP
    >>> results = await runner.run()  # doctest: +SKIP
    """
    await _check_storage(config)
    run_module._require_asyncio()
    store = open_store(config.run_root).child(run_id)

    clock: RunClock | None = None
    stops = StopState()
    validated_head: Head | None = None
    model_warnings: list[str] = []

    def validate(doc: OwnerDoc) -> None:
        nonlocal clock, validated_head
        _, state = _read(store, doc)
        _check(state, config)
        _restore_import_paths(state)
        if clock is None:
            clock = RunClock(state.run_elapsed_s)
        else:
            clock._rebase(state.run_elapsed_s)
        validated_head = doc.head

    async def check_startup_models() -> None:
        nonlocal model_warnings
        model_warnings = await run_module._check_startup_models(config, replanner)

    lease = await _take_over(
        config,
        run_id,
        store,
        force=force,
        validate=validate,
        before_acquire=check_startup_models,
        clock=lambda: clock,
        stops=stops,
    )
    assert clock is not None
    assert validated_head is not None
    services = _ResumeServices(
        config,
        run_id,
        store,
        lease,
        force=force,
        clock=clock,
        validated_head=validated_head,
        model_warnings=model_warnings,
    )
    services.task = asyncio.create_task(services._serve(), name=f"resume-{run_id}")
    try:
        await services.ready.wait()
        if services.failure is not None:
            await services._aclose()
        assert services.resumed is not None
        state = services.resumed.state
        return run_module.RunRunner(
            config,
            [state.plan_for_round(g, state.goals[g].rounds_closed + 1) for g in state.plans],
            run_id=run_id,
            approve=None,
            replanner=replanner,
            resumed=services.resumed,
        )
    except BaseException:
        if not services.closed:
            await services._aclose(failed=True)
        raise
