"""Approve and execute one run with shared budgets and durable ownership."""

import asyncio
import os
import socket
import sys
from collections.abc import Awaitable, Callable, Sequence
from fractions import Fraction
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import anyio

from swarm_arena._board.run_board import RunBoard
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.estimate import final_slice_estimate, make_price_of, worst_case_estimate
from swarm_arena._budget.ledger import Ledger
from swarm_arena._engine.admission import RunAdmission
from swarm_arena._engine.goal import GoalRunner, r0_revision_draft
from swarm_arena._engine.manifest import RunManifest, build_manifest_async
from swarm_arena._engine.preflight import run_preflight
from swarm_arena._engine.round import EngineDeps
from swarm_arena._engine.solution import solution_view
from swarm_arena._engine.stops import RunClock, StopState, watch_stop_files
from swarm_arena._errors import Fenced, PlanError, _core_error, _exception_type_name
from swarm_arena._exec.executor import Executor
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.lease import Lease
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.views import ViewWriter
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import canonical_json, utc_now
from swarm_arena._model.ids import IdGen, new_run_id
from swarm_arena._model.plan import Plan, Resources, RoundSpec
from swarm_arena._model.results import Candidate
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.tooling import Estimate, ToolRequest
from swarm_arena._outcomes import _Outcome
from swarm_arena._planning.planner import Planner
from swarm_arena._planning.provenance import stamp
from swarm_arena._planning.validate import _goal_key_errors, admit_revision
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._store.local import LocalStore
from swarm_arena._tooling.tool import Tool, effective_tool_ref

if TYPE_CHECKING:
    from swarm_arena._engine.resume import ResumeState


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


def _has_fenced(error: BaseException) -> bool:
    return isinstance(error, Fenced) or (
        isinstance(error, BaseExceptionGroup) and any(_has_fenced(e) for e in error.exceptions)
    )


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
        Executor shutdown is shielded for at most ``config.settle_timeout_s``.
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
                    frozen, notes = await anyio.to_thread.run_sync(
                        partial(
                            admit_revision,
                            stamp(plan, origin=plan.origin, parent=None),
                            self.config,
                            staged,
                        )
                    )
                    admitted.append(frozen)
                    warnings.extend(notes)
                if collisions := _goal_key_errors(admitted):
                    raise _core_error(PlanError, collisions)
                self.plans = admitted
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
        assert self.run_id is not None
        config, run_id = self.config, self.run_id
        store = open_store(config.run_root).child(run_id)
        lease = await anyio.to_thread.run_sync(
            partial(
                Lease.acquire_new,
                store,
                owner_id=uuid4().hex,
                host=socket.gethostname(),
                ttl_s=config.lease_ttl_s,
            )
        )
        local_path = _local_journal_path(run_id, lease.doc.epoch)
        clock = RunClock(0)
        async with _RunJournal(
            store,
            lease,
            run_id=run_id,
            next_seq=1,
            local_path=local_path,
            sync_interval_s=config.sync_interval_s,
        ) as journal:
            journal.clock = clock
            await anyio.to_thread.run_sync(store.create, "run.toml", config.dumps().encode())
            await anyio.to_thread.run_sync(
                store.create,
                "manifest.json",
                canonical_json(RunManifest.model_validate(manifest).model_dump(mode="json")),
            )
            ids = IdGen(run_id, lease.doc.epoch)
            ledger = Ledger(
                config.limits,
                reserve_fraction=config.reserve_fraction,
                goal_weights=config.goal_weights,
                journal=journal,
                ids=ids,
            )
            journal.append(
                EventDraft(
                    type=EventType.RUN_CONFIG, payload={"config": config.model_dump(mode="json")}
                )
            )
            journal.append(
                EventDraft(
                    type=EventType.RUN_STARTED,
                    payload={
                        "estimate": estimate.to_estimate().model_dump(mode="json"),
                        "warnings": warnings,
                        "manifest": manifest,
                        "final_slices": {g: a.model_dump() for g, a in slices.items()},
                    },
                )
            )
            await ledger.hold_final_slices(slices)
            for plan in self.plans:
                journal.append(r0_revision_draft(plan, plan.base_dir))
                journal.append(EventDraft(type=EventType.GOAL_QUEUED, goal_id=plan.goal.id))
            head = await journal.commit()
            assert head is not None
            views = ViewWriter(store, journal)
            for plan in self.plans:
                await views.write(
                    f"goals/{plan.goal.id}/plan.r0.toml", plan.dumps().encode(), source=head
                )
            board = RunBoard(run_id=run_id, run_root=config.run_root, head=lambda: journal.head)
            for plan in self.plans:
                board.register_goal(plan.goal.key(), plan.goal.id, None)
                for part in plan.parts:
                    board.register_goal(part.goal.key(), plan.goal.id, part.name)
            await anyio.to_thread.run_sync(
                board.open,
                run_id,
                config.run_root,
                sorted(
                    {p.goal.namespace for p in self.plans}
                    | {part.goal.namespace for p in self.plans for part in p.parts}
                ),
            )
            try:
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
                    stops=StopState(),
                    clock=clock,
                    admission=RunAdmission(config.limits.max_agents_total, self.plans),
                )
                return await self._drive(deps, lease.doc.epoch)
            finally:
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(board.close)

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
        deps.stops.on_change = self._wake_queue
        result: dict[str, list[Candidate]] = {}
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as watchers:
                watchers.start_soon(watch_stop_files, deps.store, epoch, deps.stops)
                watchers.start_soon(self._watch_clock, deps)
                watchers.start_soon(self._watch_fence, deps.journal, scope)
                result = await self._queue(deps)
                await self._record_stops(deps)
                watchers.cancel_scope.cancel()
        if deps.journal.fenced.is_set():
            raise _core_error(Fenced)
        return result

    async def _watch_clock(self, deps: EngineDeps) -> None:
        await anyio.sleep(deps.clock.left(deps.config.limits.run_timeout_s))
        deps.stops.run_timeout = True

    async def _watch_fence(self, journal: JournalWriter, scope: anyio.CancelScope) -> None:
        await journal.fenced.wait()
        scope.cancel()

    async def _record_stops(self, deps: EngineDeps) -> None:
        mode = 2 if deps.stops.stop_now else 1 if deps.stops.stop else 0
        if mode > self._stop_mode:
            await deps.journal.append_durable(
                EventDraft(type=EventType.STOP_REQUESTED, payload={"now": mode == 2})
            )
            self._stop_mode = mode

    async def _queued_result(self, deps: EngineDeps, goal_id: str) -> None:
        deps.journal.append(
            EventDraft(type=EventType.NO_RESULT, goal_id=goal_id, payload={"reason": "stopped"})
        )
        deps.journal.append(
            EventDraft(type=EventType.GOAL_FINISHED, goal_id=goal_id, payload={"reason": "stopped"})
        )
        with anyio.CancelScope(shield=True):
            head = await deps.journal.commit()
            await deps.ledger.end_goal(goal_id)
        assert head is not None
        chain = await anyio.to_thread.run_sync(read_chain, deps.store, head)
        state = RunState.from_events(e for _, events in chain for e in events)
        await deps.views.write(
            f"goals/{goal_id}/solution.json", solution_view(state, goal_id, head), source=head
        )

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
                    await deps.views.write(
                        f"goals/{g}/solution.json", solution_view(state, g, head), source=head
                    )
        assert deps.admission is not None

        async def run_goal(runner: GoalRunner) -> None:
            out = await runner.run()
            results[runner.goal_id] = out.finalists
            active.pop(runner.goal_id)
            self._wake_queue()

        async with anyio.create_task_group() as goals:
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
        return {p.goal.id: results[p.goal.id] for p in self.plans}
