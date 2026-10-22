"""Drive real manager actions and round-barrier judging without a swarm run."""

import atexit
import shutil
import tempfile
import warnings
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from uuid import uuid4

import anyio

from swarm_arena._agent.context import AgentContext
from swarm_arena._agent.manager import LimitReached, Manager
from swarm_arena._agent.repeats import RepeatIndex
from swarm_arena._agent.runtime import AgentCounters, AgentRuntime
from swarm_arena._agent.tool_run import Inflight, command_journal, track_inflight
from swarm_arena._agent.watchdog import Watchdog
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.digest import build_digest
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger
from swarm_arena._budget.llm_scope import LLMScope, llm_scope
from swarm_arena._budget.tranche import Tranche
from swarm_arena._engine.resume import _acquire_lease
from swarm_arena._engine.run import (
    _close_board,
    _open_board,
    _setup_run,
    _supervise_initialization,
    _supervise_writer,
)
from swarm_arena._engine.stops import RunClock, StopState
from swarm_arena._errors import (
    Fenced,
    _core_category,
    _core_error,
    _exception_type_name,
    _is_core_signal,
    _SignalCategory,
)
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.a2a import agent_round_task, agent_task_id, task_to_json
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.lease import Lease
from swarm_arena._journal.llm_record import _bound_run, drain
from swarm_arena._journal.reader import _read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.base import CallRole
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.goal import Goal
from swarm_arena._model.ids import IdGen
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import AgentSpec, Plan, RoundSpec, TeamSpec
from swarm_arena._model.planning_types import RoundReport
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import AgentStatus, Candidate, Learning
from swarm_arena._model.run_config import RunConfig
from swarm_arena._outcomes import _Outcome
from swarm_arena._planning.validate import judge_private_paths
from swarm_arena._scoring.judge import Judge, JudgeContext
from swarm_arena._scoring.runner import judge_round
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import Store
from swarm_arena._store.io import _storage_io
from swarm_arena._store.local import LocalStore
from swarm_arena._tooling.staging import tool_identity
from swarm_arena._tooling.tool import Tool, effective_tool_ref
from swarm_arena.testing._components import _build
from swarm_arena.testing._contracts import _fail, _strict_copy
from swarm_arena.testing._execution import _executor, _failed_runs, _failed_runs_on
from swarm_arena.testing._state import _artifact_root, _artifacts


class _LLMContractError(AssertionError):
    """Identify a core-owned LLM conformance rule without connector text."""


_DRAIN_RULE = "LLM drain rule: component must drain its calls before returning"


@dataclass
class _DrainOutcome:
    """Retain a value-free contract note across owner cancellation rebuilding."""

    violated: bool = False
    noted: BaseException | None = None

    def note(self, error: BaseException) -> None:
        """Attach the rule once to each propagated outcome object."""
        if (self.violated or self.noted is not None) and self.noted is not error:
            BaseException.add_note(error, _DRAIN_RULE)
            self.noted = error
            self.violated = False


@asynccontextmanager
async def _retain_drain_outcome(outcome: _DrainOutcome) -> AsyncGenerator[None]:
    """Keep the note when an enclosing owner rebuilds a cancellation outcome."""
    try:
        yield
    except BaseException as error:
        outcome.note(error)
        raise


async def _assert_drained(scope: LLMScope) -> None:
    """Check self-drain while the owner can still settle a broken component."""
    missing = await drain(scope, wait=False)
    if missing or scope.tranche.pending() != 0:
        raise _LLMContractError(_DRAIN_RULE)


async def _assert_accounted(scope: LLMScope, store: Store) -> None:
    """Compare closed funding with journaled scope costs in exact monetary units."""
    tranche = scope.tranche
    reservations = {batch.id for batch in tranche._batches}
    reservations.update(cid for cid, call in tranche._calls.items() if call.batch is None)
    # Sum whole micro-dollars: a correct round may total more than one Amount can hold.
    charged = sum(tranche._ledger._charges[reservation]._micro_usd for reservation in reservations)
    head = await scope.journal.commit()
    recorded = 0
    if head is not None:
        call_ids = set(scope.call_ids)
        for _, events in await _read_chain(store, head):
            for event in events:
                if event.type == EventType.LLM_CALL and event.payload["call"]["id"] in call_ids:
                    recorded += Amount(usd=event.payload["call"]["cost_usd"])._micro_usd
    if charged != recorded:
        raise _LLMContractError(_DRAIN_RULE)


async def _finish_draining(
    scope: LLMScope, error: BaseException | None, outcome: _DrainOutcome
) -> None:
    """Raise a normal-return violation or annotate the locally captured failure."""
    try:
        await _assert_drained(scope)
    except _LLMContractError:
        if error is None:
            raise
        if not isinstance(error, _LLMContractError):
            outcome.violated = True
            outcome.note(error)


def _scope(
    role: CallRole,
    goal: Goal,
    tranche: Tranche,
    journal: JournalWriter,
    config: RunConfig,
    artifacts: ArtifactStore,
    *,
    run_id: str,
    round: int = 0,
    team: str = "",
    agent: str = "",
) -> LLMScope:
    """Use the harness's entered services and effective run settings."""
    return LLMScope(
        role=role,
        run_id=run_id,
        goal_id=goal.id,
        team=team,
        agent=agent,
        round=round,
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


@asynccontextmanager
async def _judge_scope(goal: Goal, config: RunConfig) -> AsyncGenerator[tuple[LLMScope, IdGen]]:
    """Own an isolated judge run, including initialization and callback cleanup."""
    root = await _storage_io(OfflineHarness._run_folder)
    run_id = "offline-" + uuid4().hex
    store = LocalStore(root / "runs" / run_id)
    config = config.model_copy(
        update={
            "run_root": str(root / "runs"),
            "artifact_root": _artifact_root(),
        }
    )
    lease = await _acquire_lease(
        partial(
            Lease.acquire_new, store, owner_id="testing", host="local", ttl_s=config.lease_ttl_s
        ),
        settle_timeout_s=config.settle_timeout_s,
    )
    ids = IdGen(run_id, lease.doc.epoch)
    stops, clock = StopState(), RunClock(0)
    artifacts = _artifacts()
    drain_outcome = _DrainOutcome()
    completed = False
    entered = False
    try:
        async with (
            _retain_drain_outcome(drain_outcome),
            JournalWriter(
                store,
                lease,
                run_id=run_id,
                next_seq=1,
                local_path=root / "journal.jsonl",
                sync_interval_s=config.sync_interval_s,
                settle_timeout_s=config.settle_timeout_s,
            ) as journal,
        ):
            entered = True
            async with (
                _supervise_writer(journal),
                _supervise_initialization(store, lease.doc.epoch, stops, clock, config),
                _bound_run(
                    run_id,
                    journal=journal,
                    ids=ids,
                    artifacts=artifacts,
                    config=config,
                    stops=stops,
                    clock=clock,
                    state=RunState(),
                ),
            ):
                ledger = Ledger(
                    config.limits,
                    reserve_fraction=config.reserve_fraction,
                    goal_weights=None,
                    journal=journal,
                    ids=ids,
                )
                await journal.append_durable(
                    EventDraft(
                        type=EventType.RUN_CONFIG,
                        payload={"config": config.model_dump(mode="json")},
                    )
                )
                await ledger.hold_final_slices({goal.id: Amount()})
                await journal.append_durable(
                    EventDraft(
                        type=EventType.RUN_STARTED,
                        payload={"final_slices": {goal.id: Amount().model_dump()}},
                    )
                )
                ledger.open_goal(goal.id)
                await journal.append_durable(
                    EventDraft(type=EventType.GOAL_STARTED, goal_id=goal.id)
                )
                await _setup_run((), config)
                tranche = OfflineHarness._tranche(ledger, ":run", goal.id, config.limits)
                scope = _scope("judge", goal, tranche, journal, config, artifacts, run_id=run_id)
                escaping: BaseException | None = None
                try:
                    with llm_scope(scope):
                        yield scope, ids
                except BaseException as error:
                    escaping = error
                    raise
                finally:
                    await _finish_draining(scope, escaping, drain_outcome)
                    if escaping is None:
                        await tranche.close()
                        ledger._check_usage_overflow()
                        await _assert_accounted(scope, store)
                        completed = True
    finally:
        if not entered:
            with anyio.move_on_after(config.settle_timeout_s, shield=True):
                with suppress(BaseException):
                    await _storage_io(lease.release)
    if not completed:
        error = _core_error(RuntimeError, "offline harness stopped")
        drain_outcome.note(error)
        raise error


class OfflineHarness:
    """
    Retain the candidates, learnings, and readable store from an offline run.

    The private ranking convention uses the first supplied judge's component
    name. Without judges, an unscored private ranking makes empty ``best`` reads
    return no rows. Artifacts use the conformance kit's process-local root.
    The temporary run folder remains readable until process exit.

    Attributes
    ----------
    candidates : list[Candidate]
        Tool candidates made in the most recent run.
    learnings : list[Learning]
        Learnings posted in the most recent run.
    manager_errors : list[Exception]
        Original manager failures in round order, with failed-run log notes.
    store : LocalStore
        Readable run journal after ``run_rounds`` has returned.

    Examples
    --------
    >>> harness = OfflineHarness()
    >>> harness.candidates, harness.learnings
    ([], [])
    """

    def __init__(self) -> None:
        """Initialize empty observations without allocating a run folder."""
        self.candidates: list[Candidate] = []
        self.learnings: list[Learning] = []
        self.manager_errors: list[Exception] = []
        self._drain_outcome = _DrainOutcome()
        self.store: LocalStore

    async def run_rounds(
        self,
        manager: Manager,
        *,
        goal: Goal,
        tool: Tool,
        rounds: int,
        limits: Limits,
        config: RunConfig,
        judges: Sequence[Ref] = (),
    ) -> list[RoundReport]:
        """
        Run a manager through real action, journal, budget, and judging services.

        Each round captures a committed frontier and fixed snapshot. Its new
        gated-in candidates are judged and scores indexed before the following
        frontier. ``LimitReached`` ends only that manager round, returning an
        empty report for it. Ordinary manager exceptions are retained in
        ``manager_errors`` with failed-run log notes and journaled by type only;
        their rounds still receive barrier judging and closure, return empty
        reports, and do not prevent later rounds. Cancellation and fencing
        propagate. Returned reports are detached, strictly validated base copies.
        Terminal overflow from settlement or the run ledger ends the harness
        immediately and propagates its registered core control.

        Parameters
        ----------
        manager : Manager
            Manager instance to exercise.
        goal : Goal
            Goal whose files are snapshotted relative to the current directory.
        tool : Tool
            Tool instance rebuilt from its snapshotted reference.
        rounds : int
            Positive number of rounds to execute.
        limits : Limits
            Action limits and optional budget caps; unset budget units are unlimited.
        config : RunConfig
            Run settings; storage is isolated in the temporary run and kit roots.
        judges : Sequence[Ref], default=()
            Private harness judges; the first defines the ranking for ``best``.

        Returns
        -------
        list[RoundReport]
            Detached, strictly validated base ``RoundReport`` copies in round order.

        Raises
        ------
        RuntimeError
            If the active run deadline expires (``offline harness stopped``).

        Notes
        -----
        ``limits.run_timeout_s`` bounds the harness's active work, including
        preparation after lease acquisition. Expiry interrupts the work and
        releases owned services through their bounded cleanup paths.

        Examples
        --------
        >>> from swarm_arena import FunctionTool, Goal, Limits, RunConfig, ScriptedManager
        >>> import anyio
        >>> from functools import partial
        >>> harness = OfflineHarness()
        >>> reports = anyio.run(partial(harness.run_rounds, ScriptedManager([]),  # doctest: +SKIP
        ...     goal=Goal(id="g", namespace="demo"), tool=FunctionTool("my_package:solve"),
        ...     rounds=1, limits=Limits(), config=RunConfig()))
        """
        if rounds < 1:
            raise ValueError("rounds must be positive")
        self.candidates, self.learnings = [], []
        self.manager_errors = []
        self._drain_outcome = _DrainOutcome()
        root = await anyio.to_thread.run_sync(self._run_folder)
        run_id = "offline-" + uuid4().hex
        self.store = LocalStore(root / "runs" / run_id)
        config = config.model_copy(
            update={
                "limits": limits,
                "run_root": str(root / "runs"),
                "artifact_root": _artifact_root(),
            }
        )
        base_ref, _, goal = await anyio.to_thread.run_sync(_build, tool, Tool, goal)
        execution_ref = effective_tool_ref(base_ref, {})
        built_tool = await anyio.to_thread.run_sync(execution_ref.build, Tool)
        built_judges: list[tuple[Ref, Judge]] = []
        for ref in judges:
            snapshot, judge, _ = await anyio.to_thread.run_sync(_build, ref, Judge, goal)
            built_judges.append((snapshot, judge))
        ranking = built_judges[0][0].component_name() if built_judges else ":unscored"
        lease = await anyio.to_thread.run_sync(
            partial(
                Lease.acquire_new,
                self.store,
                owner_id="testing",
                host="local",
                ttl_s=config.lease_ttl_s,
            )
        )
        ids = IdGen(run_id, lease.doc.epoch)
        completed = False
        stops, clock = StopState(), RunClock(0)
        artifacts = _artifacts()
        async with (
            _retain_drain_outcome(self._drain_outcome),
            _executor(None) as executor,
            _failed_runs_on(executor),
            JournalWriter(
                self.store,
                lease,
                run_id=run_id,
                next_seq=1,
                local_path=root / "journal.jsonl",
                sync_interval_s=config.sync_interval_s,
                settle_timeout_s=config.settle_timeout_s,
            ) as journal,
            _supervise_writer(journal),
            _supervise_initialization(self.store, lease.doc.epoch, stops, clock, config),
            _bound_run(
                run_id,
                journal=journal,
                ids=ids,
                artifacts=artifacts,
                config=config,
                stops=stops,
                clock=clock,
                state=RunState(),
            ),
        ):
            ledger = Ledger(
                limits,
                reserve_fraction=config.reserve_fraction,
                goal_weights=None,
                journal=journal,
                ids=ids,
            )
            await journal.append_durable(
                EventDraft(
                    type=EventType.RUN_CONFIG,
                    payload={"config": config.model_dump(mode="json")},
                )
            )
            await ledger.hold_final_slices({goal.id: Amount()})
            await journal.append_durable(
                EventDraft(
                    type=EventType.RUN_STARTED,
                    payload={"final_slices": {goal.id: Amount().model_dump()}},
                )
            )
            ledger.open_goal(goal.id)
            await journal.append_durable(EventDraft(type=EventType.GOAL_STARTED, goal_id=goal.id))
            await _setup_run((), config)
            spec = AgentSpec(
                name="agent",
                rationale="Offline manager check",
                tool=base_ref,
                manager=Ref.of(manager),
            )
            team = TeamSpec(name="team", rationale="Offline manager check", agents=[spec])
            address = f"{goal.id}/team/agent"
            round_spec = RoundSpec()
            plan = Plan(goal=goal, teams=[team], judges=[ref for ref, _ in built_judges])
            await journal.append_durable(
                EventDraft(
                    type=EventType.PLAN_REVISION,
                    goal_id=goal.id,
                    payload={
                        "plan": plan.model_dump(mode="json"),
                        "origin": "human",
                        "first_round": 1,
                    },
                )
            )
            board = ArenaBoard(arena_root=str(root / "arena"), cache_dir=str(root / "cache"))
            board_failed = False
            try:
                await _open_board(board, config, run_id, [goal.namespace])
                view = await _storage_io(
                    board.snapshot,
                    goal.key(),
                    await _storage_io(lambda: _detach_model(Frontier, board.frontier())),
                )
                rt = AgentRuntime(
                    address=address,
                    holder=address,
                    goal_id=goal.id,
                    goal=goal,
                    goal_key=goal.key(),
                    part=None,
                    team=team,
                    spec=spec,
                    round=1,
                    round_spec=round_spec,
                    fidelity=round_spec.effective_fidelity(),
                    tool=built_tool,
                    tool_ref=base_ref,
                    execution_ref=execution_ref,
                    tool_identity=tool_identity(base_ref, None),
                    catalog_entry=None,
                    checks=[],
                    view=view,
                    own=[],
                    digest="",
                    notes="",
                    ledger=ledger,
                    tranche=self._tranche(ledger, address, goal.id, limits),
                    limits=limits,
                    counters=AgentCounters(),
                    repeats=RepeatIndex(run_id),
                    watchdog=Watchdog(limits),
                    journal=journal,
                    board=board,
                    executor=executor,
                    artifacts=artifacts,
                    ids=ids,
                    config=config,
                    resumed=(),
                    rankings={"": ranking},
                    private_paths=judge_private_paths(
                        Plan(
                            goal=goal,
                            teams=[team],
                            judges=[ref for ref, _ in built_judges],
                        )
                    ),
                )
                with track_inflight(Inflight()):
                    reports = await self._rounds(manager, rt, rounds, ranking, built_judges)
                completed = True
            except BaseException:
                board_failed = True
                raise
            finally:
                await _close_board(board, config, failed=board_failed)
        if not completed:
            error = _core_error(RuntimeError, "offline harness stopped")
            self._drain_outcome.note(error)
            raise error
        notes = await anyio.to_thread.run_sync(_failed_runs, executor.outputs)
        if self.manager_errors:
            for error in self.manager_errors:
                for note in notes:
                    error.add_note(note)
        else:
            for note in notes:
                warnings.warn(note, UserWarning, stacklevel=2)
        return reports

    @staticmethod
    def _run_folder() -> Path:
        """Keep the temporary journal readable, using the kit's exit cleanup policy."""
        root = Path(tempfile.mkdtemp(prefix="swarm-arena-manager-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        return root

    @staticmethod
    def _tranche(ledger: Ledger, address: str, goal_id: str, limits: Limits) -> Tranche:
        """Fund a fresh round tranche under the normal holder ceiling."""
        return Tranche(
            ledger,
            address,
            goal_id,
            size=Amount(
                tokens=limits.llm_tranche_tokens,
                usd=limits.llm_tranche_usd,
            ),
        )

    async def _rounds(
        self,
        manager: Manager,
        rt: AgentRuntime,
        rounds: int,
        ranking: str,
        judges: Sequence[tuple[Ref, Judge]],
    ) -> list[RoundReport]:
        """Advance snapshots and judge each completed round before the next."""
        reports: list[RoundReport] = []
        for number in range(1, rounds + 1):
            frontier = await _storage_io(lambda rt=rt: _detach_model(Frontier, rt.board.frontier()))
            await rt.journal.append_durable(
                EventDraft(
                    type=EventType.FRONTIER,
                    goal_id=rt.goal_id,
                    round=number,
                    payload={"round": number, "frontier": frontier.model_dump(mode="json")},
                )
            )
            view = await _storage_io(rt.board.snapshot, rt.goal_key, frontier)
            grants = rt.ledger.open_round(
                rt.goal_id, number, [rt.holder], rounds_left=rounds - number + 1
            )
            await rt.journal.append_durable(
                EventDraft(
                    type=EventType.ROUND_STARTED,
                    goal_id=rt.goal_id,
                    round=number,
                    payload={
                        "round": number,
                        "shares": {h: a.model_dump() for h, a in grants.items()},
                    },
                )
            )
            rt = replace(
                rt,
                round=number,
                view=view,
                own=[],
                counters=AgentCounters(),
                started_runs=0,
                tranche=self._tranche(rt.ledger, rt.holder, rt.goal_id, rt.limits),
                digest=build_digest(
                    view,
                    goal=rt.goal_key,
                    run_id=rt.repeats.run_id,
                    round=number,
                    ranking=ranking,
                    cap_tokens=rt.limits.digest_tokens,
                ),
            )
            manager_error: Exception | None = None
            outcome = _Outcome()
            scope = _scope(
                "manager",
                rt.goal,
                rt.tranche,
                rt.journal,
                rt.config,
                rt.artifacts,
                run_id=rt.repeats.run_id,
                round=number,
                team=rt.team.name,
                agent=rt.spec.name,
            )
            escaping: BaseException | None = None
            try:
                try:
                    with llm_scope(scope):
                        report = await manager.run_round(AgentContext(rt, gate=lambda: None))
                except BaseException as error:
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        outcome.propagate_control(error)
                        raise
                    error = outcome.classify(error)
                    if _is_core_signal(error) and _core_category(error) is _SignalCategory.TERMINAL:
                        raise error from None
                    if rt.journal.fenced.is_set():
                        raise _core_error(Fenced, "run ownership lost") from None
                    if not isinstance(error, LimitReached):
                        self.manager_errors.append(error)
                        manager_error = error
                    report = RoundReport()
            except BaseException as error:
                escaping = error
                raise
            finally:
                self.candidates.extend(c for c in rt.own if isinstance(c, Candidate))
                self.learnings.extend(item for item in rt.own if isinstance(item, Learning))
                # The bound run owns interrupted cleanup; healthy closure stays
                # supervised by the existing run deadline and writer lifetime.
                await _finish_draining(
                    scope,
                    escaping if escaping is not None else manager_error,
                    self._drain_outcome,
                )
                if escaping is None:
                    closed = await rt.tranche.close()
                    if closed.overflow or rt.ledger.run_exhausted():
                        rt.ledger._check_usage_overflow()
                    if manager_error is None:
                        await _assert_accounted(scope, self.store)
                    self._drain_outcome.noted = None
            if any(
                c.requested_by_call is not None and c.requested_by_call not in scope.call_ids
                for c in rt.own
                if isinstance(c, Candidate)
            ):
                raise _LLMContractError(
                    "requested_by_call rule: candidate must name a call in its manager scope"
                )
            if manager_error is not None:
                reason = f"manager error: {_exception_type_name(manager_error)}"
                rt.journal.append(
                    EventDraft(
                        type=EventType.AGENT_STATUS,
                        goal_id=rt.goal_id,
                        team=rt.team.name,
                        agent=rt.spec.name,
                        round=number,
                        task_id=agent_task_id(rt.address, number),
                        payload={
                            "status": AgentStatus.FINISHED.value,
                            "reason": reason,
                            "a2a": task_to_json(
                                agent_round_task(
                                    context_id=rt.repeats.run_id,
                                    address=rt.address,
                                    round=number,
                                    learnings=[
                                        item for item in rt.own if isinstance(item, Learning)
                                    ],
                                    status=AgentStatus.FINISHED,
                                    summary=reason,
                                    at=utc_now(),
                                )
                            ),
                        },
                    )
                )
            if not isinstance(report, RoundReport):
                raise _fail("Manager.run_round() must return a RoundReport")
            valid_report = await run_sync_daemon(_strict_copy, RoundReport, report)
            if valid_report is None:
                raise _fail("Manager.run_round() must return a valid RoundReport")
            reports.append(valid_report)
            candidates = [c for c in rt.own if isinstance(c, Candidate)]
            leaders = [
                view.get_candidate(row.candidate_id)
                for row in view.leaderboard(rt.goal_key, judge=ranking)
            ]
            await self._judge(rt, judges, candidates, leaders)
            await rt.journal.append_durable(
                EventDraft(
                    type=EventType.ROUND_CLOSED,
                    goal_id=rt.goal_id,
                    round=number,
                    payload={"round": number},
                )
            )
        return reports

    @staticmethod
    async def _judge(
        rt: AgentRuntime,
        judges: Sequence[tuple[Ref, Judge]],
        candidates: Sequence[Candidate],
        leaders: Sequence[Candidate],
    ) -> None:
        """Journal and index barrier outcomes with the harness id source."""
        names = {id(judge): ref.component_name() for ref, judge in judges}

        def ctx_for(judge: Judge) -> JudgeContext:
            executor, on_started = command_journal(
                rt.journal,
                rt.executor,
                candidate_id="",
                component=names[id(judge)],
            )
            return JudgeContext(
                goal=rt.goal,
                executor=executor,
                artifact_root=rt.config.artifact_root,
                run_id=rt.repeats.run_id,
                round=rt.round,
                fidelity=rt.fidelity,
                ids=rt.ids,
                on_started=on_started,
                tool_timeout_s=rt.limits.tool_timeout_s,
                secrets_file=rt.config.secrets.file,
            )

        outcome = await judge_round(rt.goal_id, judges, candidates, leaders, ctx_for)
        for score in outcome.scores:
            event = await rt.journal.append_durable(
                EventDraft(
                    type=EventType.SCORE,
                    goal_id=rt.goal_id,
                    round=rt.round,
                    payload={"score": score.model_dump(mode="json")},
                )
            )
            rt.board.put_score(score, epoch=event.epoch, seq=event.seq)
        for error in outcome.errors:
            rt.journal.append(
                EventDraft(
                    type=EventType.JUDGE_ERROR,
                    goal_id=rt.goal_id,
                    round=rt.round,
                    payload=error.model_dump(mode="json"),
                )
            )
