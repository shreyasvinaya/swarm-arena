"""Run one goal through barriers, revisions, convergence, and durable results."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import partial
from inspect import isawaitable, iscoroutinefunction
from pathlib import Path
from typing import Any, Literal, TypeVar

import anyio

from swarm_arena._agent.manager import LimitReached
from swarm_arena._agent.tool_run import command_journal
from swarm_arena._agent.watchdog import Watchdog
from swarm_arena._board.run_board import RunBoard
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.estimate import judge_amount, make_price_of
from swarm_arena._budget.ledger import Reservation
from swarm_arena._budget.llm_scope import LLMScope, llm_scope
from swarm_arena._budget.tranche import Tranche
from swarm_arena._engine.admission import RunAdmission
from swarm_arena._engine.converge import converge
from swarm_arena._engine.parts import kept_best, part_scoring, team_goal
from swarm_arena._engine.preflight import run_preflight
from swarm_arena._engine.round import AgentSlot, EngineDeps, RoundRunner
from swarm_arena._engine.solution import solution_view
from swarm_arena._errors import Fenced, PlanError, _exception_type_name
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.a2a import agent_task_id
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.replay import STOP_REASONS, GoalState, RunState
from swarm_arena._model.base import CallRole
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.plan import Plan, RoundSpec, TeamSpec
from swarm_arena._model.planning_types import Decision, History, RoundState
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import AgentStatus, Candidate
from swarm_arena._model.tooling import RunStatus
from swarm_arena._outcomes import _classify_outcome, _Outcome
from swarm_arena._planning.planner import Planner
from swarm_arena._planning.policy import _INACTIVE, Policy, guard
from swarm_arena._planning.provenance import stamp
from swarm_arena._planning.validate import admit_revision
from swarm_arena._scoring.judge import Judge, JudgeContext
from swarm_arena._scoring.ranking import ranked
from swarm_arena._scoring.runner import (
    JudgeError,
    JudgeOutcome,
    final_judge,
    judge_round,
    prepare_judge_batch,
)

_T = TypeVar("_T")


@dataclass
class GoalOutcome:
    """
    Return a goal's terminal status and ordered finalists.

    Attributes
    ----------
    goal_id : str
        Main goal identifier.
    status : {'finished', 'stopped', 'no_result'}
        Latest lifecycle outcome.
    finalists : list[Candidate]
        Ordered main-goal results; unjoined parts are in the solution view.
    reason : str or None
        Stop or no-result explanation.

    Examples
    --------
    >>> GoalOutcome("g", "no_result", [], "no passing candidate").finalists
    []
    """

    goal_id: str
    status: Literal["finished", "stopped", "no_result"]
    finalists: list[Candidate]
    reason: str | None


def r0_revision_draft(plan: Plan, base_dir: Path) -> EventDraft:
    """
    Build the initial revision event shared with run initialization.

    Parameters
    ----------
    plan : Plan
        Admitted initial plan with its assigned origin.
    base_dir : Path
        Original source directory for later revision admission.

    Returns
    -------
    EventDraft
        Plan, origin, base directory, and empty run planner-call ids.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> plan = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])
    >>> r0_revision_draft(plan, Path("/plans")).payload["planner_call_ids"]
    []
    """
    return EventDraft(
        type=EventType.PLAN_REVISION,
        goal_id=plan.goal.id,
        payload={
            "plan": plan.model_dump(mode="json"),
            "origin": plan.origin,
            "base_dir": str(base_dir),
            "planner_call_ids": [],
            "first_round": 1,
        },
    )


class GoalRunner:
    """
    Drive an admitted goal using shared run services and committed history.

    Parameters
    ----------
    deps : EngineDeps
        Shared services, including the run-wide admission reservation.
    goal_id : str
        Main goal id, also used for part budgets.
    plan : Plan
        Admitted revision applicable to the next unfinished round.
    replanner : Planner or None
        Optional autonomous revision producer.
    history : Sequence[Plan], optional
        Admitted revision history; otherwise restored from the journal.
    state : GoalState or None, default=None
        Resume progress. Full evidence and event positions are read from the
        committed journal at initialization, never inferred from this subset.
    journal_r0 : bool, default=True
        False when the run coordinator already recorded the initial revision.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> from swarm_arena import Goal, RunConfig
    >>> deps = Mock(spec=EngineDeps, config=RunConfig(), admission=None)
    >>> plan = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])
    >>> GoalRunner(deps, goal_id="g", plan=plan, replanner=None).active_agents()
    0
    """

    def __init__(
        self,
        deps: EngineDeps,
        *,
        goal_id: str,
        plan: Plan,
        replanner: Planner | None,
        history: Sequence[Plan] = (),
        state: GoalState | None = None,
        journal_r0: bool = True,
    ) -> None:
        """Copy resume carriers and share the run's admission reservation."""
        self.deps, self.goal_id, self.plan, self.replanner = deps, goal_id, plan, replanner
        if deps.admission is None:
            deps.admission = RunAdmission(deps.config.limits.max_agents_total, [plan])
        self.admission = deps.admission
        self.history = list(history)
        self.state = state.model_copy(deep=True) if state is not None else GoalState()
        self.journal_r0 = journal_r0
        self.rounds_closed = self.state.rounds_closed
        self.select_only = self.state.select_only
        self.join_attempts = self.state.join_attempts
        self.join_retries = self.state.join_retries
        self.join_phase = self.state.join_phase
        self._join_retry_number: int | None = None
        self.join_failures = list(self.state.join_failures)
        self.kept_parts = dict(self.state.kept_parts)
        self.repeat_entries: list[tuple[str, Candidate]] = []
        self.agents: list[AgentSlot] = []
        self.recorded = RunState()
        self._events: list[Event] = []
        self.status: Literal["finished", "stopped", "no_result"] = (
            "stopped"
            if self.state.pending_reason in STOP_REASONS
            else "no_result"
            if self.state.pending_reason is not None
            else "finished"
        )
        self.reason: str | None = self.state.pending_reason
        self.final_parts: dict[str, str | None] | None = None
        self._resumed: dict[str, list[tuple[str, Candidate]]] = {}
        self._frontier = (
            self.state.frontiers.get(self.state.converge_round)
            if self.state.converge_round is not None
            else self.state.frontiers.get(self.state.round)
            if (self.state.round > self.state.rounds_closed)
            else None
        )
        self._recovering_converge = self.state.converge_round is not None
        self._decisions = list(self.state.decisions)

    def active_agents(self) -> int:
        """
        Return the largest remaining ordinary or converge agent count.

        Returns
        -------
        int
            Peak reserved by the goal scheduler, including a future team.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> from swarm_arena import Goal, RunConfig
        >>> deps = Mock(spec=EngineDeps, config=RunConfig(), admission=None)
        >>> plan = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])
        >>> GoalRunner(deps, goal_id="g", plan=plan, replanner=None).active_agents()
        0
        """
        statuses = {f"{s.team.name}/{s.spec.name}": s.status for s in self.agents}
        statuses = {**self.state.agents, **statuses}
        main = sum(
            statuses.get(f"{team.name}/{agent.name}") not in _INACTIVE
            for team in self.plan.teams
            for agent in team.agents
        )
        conv = self.plan.converge
        return max(main, len(conv.team.agents) if conv.mode == "team" and conv.team else 0)

    def _draft(self, type: EventType, *, round_no: int | None = None, **payload: Any) -> EventDraft:
        if self._join_retry_number is not None and type in {
            EventType.PLAN_REVISION,
            EventType.PLAN_REJECTED,
        }:
            payload["join_retry"] = self._join_retry_number
        return EventDraft(
            type=type,
            goal_id=self.goal_id,
            round=round_no if round_no is not None else payload.get("round", 0),
            payload=payload,
        )

    async def _refresh(self) -> None:
        """Fold only a committed head, including the durable fingerprint map."""
        head = await self.deps.journal.commit()
        if head is not None:
            chain = await anyio.to_thread.run_sync(read_chain, self.deps.store, head)
            self._events = [event for _, events in chain for event in events]
            self.recorded = RunState.from_events(self._events)

    async def _plan_view(self) -> None:
        head = self.deps.journal.head
        assert head is not None
        await self.deps.views.write(
            f"goals/{self.goal_id}/plan.r{self.plan.revision}.toml",
            self.plan.dumps().encode(),
            source=head,
        )

    async def _slots(self, teams: Sequence[TeamSpec], *, reuse: bool) -> list[AgentSlot]:
        completed, slots = await self._supervised_step(
            partial(self._prepare_slots, teams, reuse=reuse),
            final=self.plan.converge.team in teams,
        )
        return slots if completed and slots is not None else []

    async def _prepare_slots(self, teams: Sequence[TeamSpec], *, reuse: bool) -> list[AgentSlot]:
        previous = {slot.address: slot for slot in self.agents} if reuse else {}
        slots: list[AgentSlot] = []
        for team in teams:
            for spec in team.agents:
                address = f"{self.goal_id}/{team.name}/{spec.name}"
                key = f"{team.name}/{spec.name}"
                old = previous.get(address)
                goal = team_goal(self.plan, team)
                manager = old.manager if old and old.spec.manager == spec.manager else None
                slots.append(
                    AgentSlot(
                        address=address,
                        team=team,
                        spec=spec,
                        manager=manager,
                        tool_ref=spec.tool,
                        goal=goal,
                        goal_key=goal.key(),
                        part=team.part,
                        notes=old.notes if old else self.state.notes.get(key, ""),
                        watchdog=old.watchdog if old else self._restore_watchdog(key),
                        status=old.status
                        if old
                        else self.state.agents.get(key, AgentStatus.ACTIVE),
                    )
                )
        if self._recovering_converge and self.plan.converge.team in teams:
            self._recovering_converge = False
        board = self.deps.board
        if isinstance(board, RunBoard):
            board.register_goal(self.plan.goal.key(), self.goal_id, None)
            for part in self.plan.parts:
                board.register_goal(part.goal.key(), self.goal_id, part.name)
        return slots

    def _restore_watchdog(self, key: str) -> Watchdog:
        watchdog = Watchdog(self.deps.config.limits)
        started: set[int] = set()
        for event in self._events:
            if event.goal_id != self.goal_id:
                continue
            if f"{event.team}/{event.agent}" == key:
                if event.type == EventType.TOOL_START:
                    started.add(event.round)
                elif event.type == EventType.TOOL_END:
                    candidate = self.recorded.candidates[event.payload["candidate"]["id"]]
                    watchdog.record_run(
                        candidate.output.status,
                        passed=candidate.output.status == RunStatus.OK and not candidate.gated_out,
                    )
                elif event.type == EventType.WATCHDOG and "refused" in event.payload:
                    watchdog.record_refusal()
            if event.type == EventType.ROUND_CLOSED:
                watchdog.end_round(ran=event.payload.get("round", event.round) in started)
        return watchdog

    async def _finish_recovered(self) -> GoalOutcome:
        """Finish a committed selection without repeating agents or final judges."""
        goal = self.recorded.goals[self.goal_id]
        reason = goal.pending_reason
        with anyio.CancelScope(shield=True):
            self.deps.journal.append(self._draft(EventType.GOAL_FINISHED, reason=reason))
            await self.deps.journal.commit()
            await self.deps.ledger.end_goal(self.goal_id)
        await self._refresh()
        head = self.deps.journal.head
        assert head is not None
        await self.deps.views.write(
            f"goals/{self.goal_id}/solution.json",
            solution_view(self.recorded, self.goal_id, head),
            source=head,
        )
        await self.admission.release(self.goal_id)
        goal = self.recorded.goals[self.goal_id]
        return GoalOutcome(
            self.goal_id,
            "no_result"
            if goal.status == "no_result"
            else "stopped"
            if goal.status == "stopped"
            else "finished",
            [self.recorded.candidates[id] for id in goal.final_order],
            reason,
        )

    async def _restore_terminal_decision(self) -> None:
        """Restore this lifecycle/revision's terminal policy and any owed closure."""
        latest: tuple[int, Decision] | None = None
        for event in self._events:
            if event.goal_id != self.goal_id:
                continue
            if event.type == EventType.GOAL_STARTED or (
                event.type == EventType.PLAN_REVISION
                and event.payload["plan"]["revision"] == self.plan.revision
            ):
                latest = None
            elif event.type == EventType.DECISION:
                latest = (
                    event.payload["round"],
                    Decision.model_validate(event.payload["decision"]),
                )
        if latest is not None:
            round, decision = latest
            if decision.no_result:
                self.status, self.reason = "no_result", decision.no_result
                if round > self.rounds_closed:
                    self.rounds_closed = round
                    await self._close_round(round, replan=False)

    def _merge_repeats(self, entries: Sequence[tuple[str, Candidate]]) -> None:
        seen = {candidate.id for _, candidate in self.repeat_entries}
        for key, candidate in entries:
            if candidate.id not in seen:
                self.repeat_entries.append((key, candidate))
                seen.add(candidate.id)

    def _restore_repeats(self) -> None:
        """Keep items after the restarted frontier private to their owner."""
        mark = (
            next((m for m in self._frontier.marks if m.run_id == self.deps.run_id), None)
            if self._frontier
            else None
        )
        boundary = (mark.epoch, mark.seq) if mark is not None else (0, 0)
        for candidate in self.recorded.candidates.values():
            if candidate.goal_id != self.goal_id or candidate.run_id != self.deps.run_id:
                continue
            key = self.state.repeat_keys.get(candidate.id)
            if key is None:
                key = self.recorded.goals.get(self.goal_id, GoalState()).repeat_keys.get(
                    candidate.id
                )
            if key is None:
                continue
            pair = (key, candidate)
            if self._frontier is not None and self.recorded.positions[candidate.id] > boundary:
                self._resumed.setdefault(f"{candidate.team}/{candidate.agent}", []).append(pair)
            else:
                self._merge_repeats([pair])

    def _ranking_name(self, part: str | None) -> str:
        _, refs, name = part_scoring(self.plan, part)
        return name or (refs[0].component_name() if len(refs) == 1 else "")

    def _round_state(self) -> RoundState:
        part_keys = {part.name: part.goal.key() for part in self.plan.parts}
        candidates = [
            c
            for c in self.recorded.candidates.values()
            if c.goal_id == self.goal_id
            and (
                c.part is None
                or (
                    c.part in part_keys
                    and self.recorded.candidate_goal_keys.get(c.id) == part_keys[c.part]
                )
            )
        ]
        boundary: tuple[int, int] | None = None
        if self._recovering_converge and self._frontier is not None:
            # Selection and assigned parents must use the original converge snapshot.
            mark = next((m for m in self._frontier.marks if m.run_id == self.deps.run_id), None)
            boundary = (mark.epoch, mark.seq) if mark is not None else (0, 0)
            candidates = [c for c in candidates if self.recorded.positions[c.id] <= boundary]
        by_id = {c.id: c for c in candidates}
        scores = [
            s
            for s in self.recorded.scores.values()
            if s.candidate_id in by_id
            and (boundary is None or self.recorded.positions[s.id] <= boundary)
        ]
        identities: dict[str | None, tuple[str, str, str]] = {}
        for score in scores:
            part = by_id[score.candidate_id].part
            if part is not None and part not in {p.name for p in self.plan.parts}:
                continue
            if score.judge == self._ranking_name(part):
                identities[part] = score.identity
        return RoundState(
            goal_id=self.goal_id,
            round=self.rounds_closed,
            rounds_total=self.plan.rounds_total(self.deps.config.limits),
            plan=self.plan,
            candidates=candidates,
            scores=scores,
            status={f"{s.team.name}/{s.spec.name}": s.status for s in self.agents},
            ranking_identity=identities,
        )

    def _tranche(self, service: str) -> Tranche:
        limits = self.deps.config.limits
        return Tranche(
            self.deps.ledger,
            f"{self.goal_id}/:{service}",
            self.goal_id,
            size=Amount(tokens=limits.llm_tranche_tokens, usd=limits.llm_tranche_usd),
        )

    def _scope(
        self, role: CallRole, round: int, tranche: Tranche, *, final: bool = False
    ) -> LLMScope:
        deps = self.deps
        return LLMScope(
            role=role,
            run_id=deps.run_id,
            goal_id=self.goal_id,
            team="",
            agent="",
            round=round,
            candidate_id=None,
            tranche=tranche,
            journal=deps.journal,
            artifacts=deps.artifacts,
            models=deps.config.models,
            tracing=deps.config.tracing,
            cache=deps.config.cache,
            force_no_cache_read=final,
        )

    async def _close(self, tranche: Tranche) -> None:
        with anyio.move_on_after(self.deps.config.settle_timeout_s):
            while tranche.pending():  # noqa: ASYNC110 - external call settlement
                await anyio.sleep(0.05)
        # Charges and overflow are already authoritative in the ledger/journal.
        await tranche.close()

    async def _supervised_step(
        self, operation: Callable[[], Awaitable[_T]], *, final: bool
    ) -> tuple[bool, _T | None]:
        """Supervise hooks and contain nonstandard operation failures.

        The engine supports asyncio only. Check the caller's pending state
        independently of group exit: that exit can absorb native cancellation
        racing with the group's own cancellation. External cancellation wins
        with a fresh value-free exception; user cancellation and groups remain
        untrusted operation outcomes handled by callers' Exception boundaries.
        """
        results: list[_T] = []
        errors: list[BaseException] = []
        caller = asyncio.current_task()
        assert caller is not None
        task_info = anyio.get_current_task()
        cancellations_at_entry = caller.cancelling()
        pending_at_entry = task_info.has_pending_cancellation()
        try:
            async with anyio.create_task_group() as group:

                async def invoke() -> None:
                    try:
                        results.append(await operation())
                    except BaseException as error:
                        if not (
                            isinstance(error, anyio.get_cancelled_exc_class())
                            and group.cancel_scope.cancel_called
                        ):
                            errors.append(error)
                    finally:
                        group.cancel_scope.cancel()

                group.start_soon(invoke)
                grace_until: float | None = None
                try:
                    while True:
                        if self.deps.journal.fenced.is_set():
                            raise Fenced()
                        stopped = self._stop()
                        if stopped and not final:
                            break
                        delay = 0.05
                        if final and (self.deps.stops.stop_now or self.deps.stops.run_timeout):
                            if grace_until is None:
                                grace_until = (
                                    anyio.current_time() + self.deps.config.settle_timeout_s
                                )
                            left = grace_until - anyio.current_time()
                            if left <= 0:
                                break
                            delay = min(delay, left)
                        else:
                            left = self.deps.clock.left(self.deps.config.limits.run_timeout_s)
                            if left > 0:
                                delay = min(delay, left)
                        await anyio.sleep(delay)
                finally:
                    group.cancel_scope.cancel()
        finally:
            if (
                pending_at_entry
                or caller.cancelling() > cancellations_at_entry
                or task_info.has_pending_cancellation()
            ):
                raise asyncio.CancelledError() from None
        # Reraise outside the task group so fencing keeps its original type.
        if errors:
            raise _classify_outcome(errors[0]) from None
        return (True, results[0]) if results else (False, None)

    def _judge_context(
        self, ref: Ref, batch: Sequence[Candidate], round: int, *, final: bool = False
    ) -> JudgeContext:
        deps = self.deps
        first = batch[0]
        goal = (
            self.plan.goal
            if first.part is None
            else next(p.goal for p in self.plan.parts if p.name == first.part)
        )
        executor, on_started = command_journal(
            deps.journal, deps.executor, candidate_id=first.id, component=ref.component_name()
        )
        return JudgeContext(
            goal=goal,
            executor=executor,
            artifact_root=deps.config.artifact_root,
            run_id=deps.run_id,
            round=round,
            part=first.part,
            fidelity=first.fidelity,
            ids=deps.ids,
            on_started=on_started,
            secrets_file=deps.config.secrets.file,
            tool_timeout_s=deps.config.limits.tool_timeout_s,
            final=final,
            read_cache=not final,
        )

    def _publish_scores(
        self,
        outcome: JudgeOutcome,
        scope: LLMScope,
        round: int,
        *,
        barrier_batch: dict[str, Any] | None = None,
    ) -> None:
        remaining = len(outcome.scores) + len(outcome.errors)

        def append(kind: EventType, **payload: Any) -> Event:
            nonlocal remaining
            remaining -= 1
            if remaining == 0 and barrier_batch is not None:
                payload["barrier_batch"] = barrier_batch
            return self.deps.journal.append(self._draft(kind, round=round, **payload))

        for score in outcome.scores:
            if not score.llm_call_ids:
                score = score.model_copy(update={"llm_call_ids": list(scope.call_ids)})
            event = append(EventType.SCORE, score=score.model_dump(mode="json"))
            self.deps.board.put_score(score, epoch=event.epoch, seq=event.seq)
        for error in outcome.errors:
            append(EventType.JUDGE_ERROR, **error.model_dump())

    async def _barrier(self, candidates: Sequence[Candidate], round: int) -> bool:
        """Reuse committed batches, funding only missing barrier outcomes."""
        tranche = self._tranche("judges")
        reservations: dict[str, tuple[Reservation, Amount, bool]] = {}
        try:
            completed, funded = await self._supervised_step(
                partial(self._barrier_work, candidates, round, tranche, reservations), final=True
            )
            return bool(completed and funded)
        finally:
            for reservation, amount, dispatched in reservations.values():
                await self.deps.ledger.settle(
                    reservation.id, amount if dispatched else Amount(), estimated=dispatched
                )
            await self._close(tranche)
            self._stop()

    async def _barrier_work(
        self,
        candidates: Sequence[Candidate],
        round: int,
        tranche: Tranche,
        reservations: dict[str, tuple[Reservation, Amount, bool]],
    ) -> bool:
        for part in [None, *(p.name for p in self.plan.parts)]:
            batch = [
                c
                for c in candidates
                if c.part == part
                and not c.gated_out
                and (c.output.status == RunStatus.OK or c.output.partial)
            ]
            if not batch:
                continue
            _, refs, _ = part_scoring(self.plan, part)
            for ref in refs:
                identity: dict[str, Any] = {
                    "goal_id": self.goal_id,
                    "revision": self.plan.revision,
                    "round": round,
                    "part": part,
                    "ref": ref.model_dump(mode="json"),
                    "candidates": sorted(c.id for c in batch),
                }
                key = sha256_hex(canonical_json(identity))
                if any(
                    event.type in {EventType.SCORE, EventType.JUDGE_ERROR}
                    and event.goal_id == self.goal_id
                    and event.round == round
                    and isinstance(record := event.payload.get("barrier_batch"), dict)
                    and record.get("key") == key
                    for event in self._events
                ):
                    continue
                completion: dict[str, Any] = {
                    "key": key,
                    "members": sorted(c.id for c in batch),
                    "anchors": [],
                }
                leaders = [c for c, _ in self._round_state().ranked(part)]
                scope = self._scope("judge", round, tranche)
                preparation = _Outcome()
                try:
                    judge = await run_sync_daemon(ref.build, Judge)
                    prepared = await run_sync_daemon(prepare_judge_batch, judge, batch, leaders)
                    completion.update(
                        members=sorted(c.id for c in prepared[0]), anchors=sorted(prepared[1])
                    )
                    amount = await run_sync_daemon(
                        partial(
                            judge_amount,
                            judge,
                            len(prepared[0]),
                            self.deps.config,
                            make_price_of(self.deps.config.models),
                            llm=False,
                        )
                    )
                except Fenced:
                    raise
                except BaseException as error:
                    error = preparation.classify(error)
                    outcome = JudgeOutcome(
                        scores=[],
                        errors=[
                            JudgeError(
                                candidate_id=c.id,
                                judge=ref.component_name(),
                                reason=f"judge error: {_exception_type_name(error)}",
                            )
                            for c in batch
                        ],
                    )
                    self._publish_scores(outcome, scope, round, barrier_batch=completion)
                    await self._refresh()
                    continue

                def on_admit(admitted: Reservation, amount: Amount = amount) -> None:
                    reservations[admitted.id] = (admitted, amount, False)

                try:
                    reservation = (
                        await self.deps.ledger.reserve(
                            f"{self.goal_id}/:judges",
                            amount,
                            kind="judge",
                            goal_id=self.goal_id,
                            on_admit=on_admit,
                        )
                        if amount != Amount()
                        else None
                    )
                except LimitReached:
                    self.deps.journal.append(
                        self._draft(
                            EventType.RESERVE_EXHAUSTED,
                            **{"round": round, "needed": amount.model_dump()},
                        )
                    )
                    self.select_only = True
                    self.status, self.reason = "stopped", "reserve_exhausted"
                    return False
                if reservation is not None:
                    reservations[reservation.id] = (reservation, amount, True)
                with llm_scope(scope):
                    outcome = await judge_round(
                        self.goal_id,
                        [(ref, judge)],
                        batch,
                        leaders,
                        lambda _, ref=ref, prepared=prepared: self._judge_context(
                            ref, prepared[0], round
                        ),
                        prepared=prepared,
                    )
                if reservation is not None:
                    await self.deps.ledger.settle(reservation.id, amount, estimated=True)
                    reservations.pop(reservation.id)
                self._publish_scores(outcome, scope, round, barrier_batch=completion)
                await self._refresh()
        return True

    async def _final(self, finalists: Sequence[Candidate]) -> list[Candidate]:
        """Rescore within stop grace, falling back only to committed rankings."""
        if not finalists:
            return []
        name = self._ranking_name(None)
        ref = next(ref for ref in self.plan.judges if ref.component_name() == name)
        tranche = self._tranche("final")
        scope = self._scope("judge", 0, tranche, final=True)
        reservation: Reservation | None = None
        dispatched = False
        amount = Amount()

        def on_admit(admitted: Reservation) -> None:
            nonlocal reservation
            reservation = admitted

        async def invoke() -> tuple[list[Candidate], JudgeOutcome]:
            nonlocal amount, dispatched
            with llm_scope(scope):
                judge = await run_sync_daemon(ref.build, Judge)
                amount = await run_sync_daemon(
                    partial(
                        judge_amount,
                        judge,
                        len(finalists),
                        self.deps.config,
                        make_price_of(self.deps.config.models),
                        llm=False,
                    )
                )
                if amount != Amount():
                    await self.deps.ledger.reserve(
                        f"{self.goal_id}/:final",
                        amount,
                        kind="final",
                        goal_id=self.goal_id,
                        on_admit=on_admit,
                    )
                ctx = self._judge_context(ref, finalists, 0, final=True)
                dispatched = True
                return await final_judge(
                    self.goal_id,
                    (ref, judge),
                    finalists,
                    ctx,
                )

        try:
            order: list[Candidate] = []
            try:
                completed, result = await self._supervised_step(invoke, final=True)
                if completed:
                    assert result is not None
                    order, outcome = result
                else:
                    await self._refresh()
                    state = self._round_state()
                    identity = state.ranking_identity.get(None)
                    if identity is not None:
                        order = [
                            c
                            for c, _ in ranked(
                                finalists,
                                state.scores,
                                identity=identity,
                                fidelity=None,
                                allow_partial=self.plan.converge.allow_partial,
                            )
                        ]
                    raise TimeoutError
            except Fenced:
                raise
            except LimitReached:
                self.status, self.reason = "no_result", "final slice used up"
                return []
            except Exception as error:
                outcome = JudgeOutcome(
                    scores=[],
                    errors=[
                        JudgeError(
                            candidate_id=c.id,
                            judge=name,
                            reason=f"judge error: {_exception_type_name(error)}",
                        )
                        for c in finalists
                    ],
                )
            self._publish_scores(outcome, scope, 0)
            if not order and outcome.errors:
                self.status, self.reason = "no_result", "provider_failed"
            return order
        finally:
            if reservation is not None:
                await self.deps.ledger.settle(
                    reservation.id, amount if dispatched else Amount(), estimated=dispatched
                )
            await self._close(tranche)
            self._stop()

    async def _replan(self) -> bool:
        """Supervise the proposal, then atomically recheck capacity and publish."""
        if self.replanner is None:
            return False
        tranche = self._tranche("planner")
        scope = self._scope("planner", self.rounds_closed, tranche)
        state = self._round_state()
        history = History(
            goal_id=self.goal_id,
            revisions=self.history,
            scores=state.scores,
            learnings=[
                item for item in self.recorded.learnings.values() if item.goal_id == self.goal_id
            ],
            decisions=self._decisions,
            budget_left=self.deps.ledger.goal_remaining(self.goal_id),
            rounds_done=self.rounds_closed,
            join_failures=self.join_failures,
        )

        async def propose() -> Plan:
            assert self.replanner is not None
            with llm_scope(scope):
                hook = self.replanner.plan
                proposed = (
                    hook(self.plan.goal, self.deps.config, history)
                    if iscoroutinefunction(hook)
                    else await run_sync_daemon(hook, self.plan.goal, self.deps.config, history)
                )
                if isawaitable(proposed):
                    proposed = await proposed
                if not isinstance(proposed, Plan):
                    raise TypeError("invalid planner result")
                return proposed

        try:
            try:
                completed, proposed = await self._supervised_step(propose, final=False)
            finally:
                await self._close(tranche)
        except Fenced:
            raise
        except Exception as error:
            self.deps.journal.append(
                self._draft(
                    EventType.PLAN_REJECTED,
                    errors=[f"planner error: {_exception_type_name(error)}"],
                    origin="autonomous",
                    first_round=self.rounds_closed + 1,
                )
            )
            return False
        if not completed or self._stop():
            self.deps.journal.append(
                self._draft(
                    EventType.PLAN_REJECTED,
                    errors=["planner stopped"],
                    origin="autonomous",
                    first_round=self.rounds_closed + 1,
                )
            )
            return False
        assert proposed is not None
        try:
            revision = stamp(proposed, origin="autonomous", parent=self.plan)

            async def prepare() -> Plan:
                # Admission can execute extension Config validators.
                admitted, _ = await run_sync_daemon(
                    partial(
                        admit_revision,
                        revision,
                        self.deps.config,
                        self.deps.artifacts,
                        history=self.history,
                        other_active_agents=0,
                    )
                )
                if (
                    self._join_retry_number is not None
                    and admitted.rounds_total(self.deps.config.limits) <= self.rounds_closed
                ):
                    raise PlanError(["join retry has no remaining ordinary rounds"])
                await self._preflight(admitted)
                return admitted

            completed, admitted = await self._supervised_step(prepare, final=False)
            if not completed or admitted is None or self._stop():
                raise PlanError(["revision preparation stopped"])
            peak = max(
                sum(len(t.agents) for t in admitted.teams),
                len(admitted.converge.team.agents)
                if admitted.converge.mode == "team" and admitted.converge.team
                else 0,
            )

            async def publish() -> None:
                self.deps.journal.append(
                    self._draft(
                        EventType.PLAN_REVISION,
                        plan=admitted.model_dump(mode="json"),
                        origin="autonomous",
                        planner_call_ids=list(scope.call_ids),
                        first_round=self.rounds_closed + 1,
                    )
                )
                await self.deps.journal.commit()
                if self.plan.parts != admitted.parts:
                    self.kept_parts, self.join_attempts, self.join_failures = {}, 0, []
                self.plan = admitted
                self.history.append(admitted)

            await self.admission.update(self.goal_id, peak, publish=publish, plan=admitted)
        except PlanError as error:
            self.deps.journal.append(
                self._draft(
                    EventType.PLAN_REJECTED,
                    errors=error.errors,
                    origin="autonomous",
                    first_round=self.rounds_closed + 1,
                )
            )
            return False
        self.agents = await self._slots(self.plan.teams, reuse=True)
        await self._plan_view()
        return True

    async def _preflight(self, plan: Plan) -> None:
        """Convert shared prerequisite failures to an admission refusal."""
        outcome = _Outcome()
        try:
            await run_preflight([plan], self.deps.config, self.deps.executor)
        except Fenced:
            raise
        except BaseException as error:
            error = outcome.classify(error)
            raise PlanError([f"preflight failed: {_exception_type_name(error)}"]) from None

    def _stop(self) -> bool:
        stops = self.deps.stops
        if self.deps.ledger.run_exhausted():
            stops.run_budget = True
        if self.deps.ledger.goal_exhausted(self.goal_id):
            stops.goal_budget.add(self.goal_id)
        if self.deps.clock.left(self.deps.config.limits.run_timeout_s) <= 0:
            stops.run_timeout = True
        reason = (
            "stop --now"
            if stops.stop_now
            else "run_timeout_s"
            if stops.run_timeout
            else "stop"
            if stops.stop
            else "run budget"
            if stops.run_budget
            else "goal budget"
            if self.goal_id in stops.goal_budget
            else None
        )
        if reason is not None:
            if self.status != "no_result" and not (
                self.reason == "reserve_exhausted" and reason in {"run budget", "goal budget"}
            ):
                self.status, self.reason = "stopped", reason
            self.select_only |= stops.stop_now or stops.run_timeout
        return reason is not None

    async def _round_decision(self, round: int) -> Decision:
        """Publish policy and its synchronous effects as one recoverable checkpoint."""
        state = self._round_state()

        async def invoke() -> Decision:
            policy = await run_sync_daemon(self.plan.policy.build, Policy)
            proposed = await run_sync_daemon(policy.after_round, state)
            if isawaitable(proposed):
                proposed = await proposed
            return guard(proposed, state)

        try:
            completed, result = await self._supervised_step(invoke, final=False)
            decision = result if completed and result is not None else Decision()
        except Fenced:
            raise
        except Exception as error:
            decision = guard(Decision(policy=f"policy error: {_exception_type_name(error)}"), state)
        event = self.deps.journal.append(
            self._draft(
                EventType.DECISION,
                **{"round": round, "decision": decision.model_dump(mode="json")},
            )
        )
        self._decisions.append((round, decision))
        self.deps.board.put_decision(
            self.goal_id, round, decision, epoch=event.epoch, seq=event.seq
        )
        for slot in self.agents:
            key = f"{slot.team.name}/{slot.spec.name}"
            reason = decision.prune.get(key, decision.prune.get(slot.team.name))
            if reason is not None:
                slot.status = AgentStatus.PRUNED
                self.deps.journal.append(
                    EventDraft(
                        type=EventType.AGENT_STATUS,
                        task_id=agent_task_id(slot.address, round),
                        goal_id=self.goal_id,
                        team=slot.team.name,
                        agent=slot.spec.name,
                        round=round,
                        payload={"status": "pruned", "reason": reason},
                    )
                )
        for part in self.plan.parts:
            if not any(s.part == part.name and s.status not in _INACTIVE for s in self.agents):
                best = kept_best(state, part.name)
                if best is not None and self.kept_parts.get(part.name) != best.id:
                    self.kept_parts[part.name] = best.id
                    self.deps.journal.append(
                        self._draft(EventType.PART_KEPT, part=part.name, candidate_id=best.id)
                    )
        return decision

    async def _complete_replan(self) -> None:
        """Run a closed round's unpublished replan before any successor managers."""
        await self._refresh()
        goal = self.recorded.goals[self.goal_id]
        if goal.pending_replan is not None and not self._stop():
            if self.replanner is None:
                self.deps.journal.append(
                    self._draft(
                        EventType.PLAN_REJECTED,
                        errors=["replanner unavailable"],
                        origin="autonomous",
                        first_round=goal.pending_replan,
                    )
                )
            else:
                await self._replan()
            # Rejections also complete a boundary and must survive another crash.
            await self._refresh()
        plan = self.recorded.plan_for_round(self.goal_id, self.rounds_closed + 1)
        if plan != self.plan:
            self.plan = plan
            await self.admission.update(self.goal_id, self.active_agents())
            self.agents = await self._slots(self.plan.teams, reuse=True)

    async def _close_round(self, round: int, *, replan: bool) -> None:
        """Commit a completed round with its replan boundary and lifecycle reason."""
        await self.deps.journal.append_durable(
            self._draft(
                EventType.ROUND_CLOSED,
                **{
                    "round": round,
                    "run_elapsed_s": self.deps.clock.elapsed(),
                    "replan": replan,
                    "reason": self.reason,
                },
            )
        )

    async def _run_rounds(self) -> None:
        # A crash after closure must not silently skip the pending replanner.
        await self._complete_replan()
        while self.rounds_closed < self.plan.rounds_total(self.deps.config.limits):
            if self._stop() or self.reason is not None:
                self._merge_repeats([pair for pairs in self._resumed.values() for pair in pairs])
                self._resumed = {}
                break
            round = self.rounds_closed + 1
            decision = next((d for r, d in self._decisions if r == round), None)
            funded = True
            if decision is None:
                spec = self.plan.rounds[round - 1] if self.plan.rounds else RoundSpec()
                recovered = [c for pairs in self._resumed.values() for _, c in pairs]
                out = await RoundRunner(
                    self.deps,
                    goal_id=self.goal_id,
                    plan=self.plan,
                    agents=self.agents,
                    round_spec=spec,
                    rounds_left=self.plan.rounds_total(self.deps.config.limits)
                    - self.rounds_closed,
                    resumed=self._resumed,
                    repeat_entries=self.repeat_entries,
                    frontier=self._frontier,
                ).run(round)
                self._merge_repeats(out.repeat_entries)
                await self._refresh()
                funded = await self._barrier([*recovered, *out.candidates], round)
                self.rounds_closed = round
                if funded:
                    decision = await self._round_decision(round)
            else:
                # Its policy checkpoint proves managers and barrier already completed.
                self._merge_repeats([pair for pairs in self._resumed.values() for pair in pairs])
                self.rounds_closed = round
            self._resumed, self._frontier = {}, None
            if decision is not None and decision.no_result:
                self.status, self.reason = "no_result", decision.no_result
            await self.admission.update(self.goal_id, self.active_agents())
            replan = (
                funded
                and self.status != "no_result"
                and round < self.plan.rounds_total(self.deps.config.limits)
                and not self._stop()
                and self.replanner is not None
            )
            await self._close_round(round, replan=replan)
            if not funded or self.status == "no_result":
                break
            await self._complete_replan()
        await self._refresh()

    async def run(self) -> GoalOutcome:
        """
        Execute rounds and publish a result after its terminal journal commit.

        Stop reasons are refreshed through convergence and final judging.
        Pending replans, policy decisions and ordinary slot preparation are
        abandoned on any stop. After an immediate stop or run deadline,
        barrier judging, converge slot preparation and final judging have
        ``settle_timeout_s`` to finish; on timeout, selection uses only previously
        journaled ranking scores. A graceful stop that preempts the policy
        journals the default ``Decision()`` without pruning. Candidates left
        unjudged when a round's barrier is cut short stay unranked on resume;
        later rounds do not rejudge those closed-round candidates.
        Policy checkpoints complete an unfinished round without repeating its
        managers or barrier. Round closure commits before replanning; revisions
        and rejections identify the first round to which they apply. Recovery
        completes a pending replan before starting that round. Final selection
        records its lifecycle reason independently of the live stop flags.
        A terminal policy checkpoint restores its effect and completes any
        owed round closure before constructing managers or judges. New revisions
        pass bounded preflight before publication. Recovery replays already
        admitted revisions.

        Returns
        -------
        GoalOutcome
            Finalists, or an explicit reason when nothing passes.

        Examples
        --------
        >>> outcome = await runner.run()  # doctest: +SKIP
        """
        await self.admission.update(self.goal_id, self.active_agents())
        if self.journal_r0:
            self.deps.journal.append(r0_revision_draft(self.plan, self.plan.base_dir))
            await self.deps.journal.commit()
            await self._plan_view()
        await self._refresh()
        if self.state.selection_recorded and not self.state.finished:
            return await self._finish_recovered()
        if not self.history:
            self.history = [p for p, _ in self.recorded.plans.get(self.goal_id, [])] or [self.plan]
        await self._restore_terminal_decision()
        self._restore_repeats()
        if self.status != "no_result":
            self.agents = await self._slots(self.plan.teams, reuse=False)
            if not self.state.converge_started and self.join_phase != "replan":
                await self._run_rounds()
        finalists: list[Candidate] = []
        if self.status != "no_result":
            self._stop()
            if self.join_phase != "replan":
                self.join_phase = "converge"
                self.deps.journal.append(
                    self._draft(
                        EventType.CONVERGE_STARTED, select_only=self.select_only, reason=self.reason
                    )
                )
                await self.deps.journal.commit()
            finalists = await converge(self, select_only=self.select_only)
        self._stop()
        if not finalists and not (self.final_parts and any(self.final_parts.values())):
            if self.status != "no_result":
                self.status, self.reason = "no_result", "no passing candidate"
            self.deps.journal.append(self._draft(EventType.NO_RESULT, reason=self.reason))
        else:
            self.deps.journal.append(
                self._draft(
                    EventType.FINAL_ORDER,
                    candidate_ids=[c.id for c in finalists],
                    parts=self.final_parts,
                    reason=self.reason,
                )
            )
        # Replay returns the pool at goal_finished; do not suspend before applying it live.
        with anyio.CancelScope(shield=True):
            await self.deps.journal.append_durable(
                self._draft(EventType.GOAL_FINISHED, reason=self.reason)
            )
            await self.deps.ledger.end_goal(self.goal_id)
        await self._refresh()
        head = self.deps.journal.head
        assert head is not None
        await self.deps.views.write(
            f"goals/{self.goal_id}/solution.json",
            solution_view(self.recorded, self.goal_id, head),
            source=head,
        )
        await self.admission.release(self.goal_id)
        return GoalOutcome(self.goal_id, self.status, finalists, self.reason)
