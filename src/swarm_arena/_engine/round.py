"""Run parallel manager rounds against one durable, fixed frontier."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anyio

from swarm_arena._agent.context import AgentContext, _prepare_metadata, _ToolMetadata
from swarm_arena._agent.manager import LimitReached, Manager, RoundReport
from swarm_arena._agent.repeats import COUNTED, RepeatIndex
from swarm_arena._agent.runtime import AgentCounters, AgentRuntime
from swarm_arena._agent.tool_run import Inflight, track_inflight
from swarm_arena._agent.watchdog import Watchdog
from swarm_arena._board.arena import ArenaView, past_run_repeat_entries
from swarm_arena._board.base import Board, BoardView
from swarm_arena._board.digest import build_digest
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger
from swarm_arena._budget.llm_scope import LLMScope, llm_scope
from swarm_arena._budget.tranche import Tranche
from swarm_arena._engine.admission import RunAdmission
from swarm_arena._engine.parts import part_scoring, team_goal
from swarm_arena._engine.stops import RunClock, StopState, round_deadline
from swarm_arena._errors import Fenced, _core_error, _rebuild_core_signal
from swarm_arena._exec.executor import Executor
from swarm_arena._exec.supervisor import BEFORE_KILL_TIMEOUT_S
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.a2a import agent_round_task, agent_task_id, task_to_json
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.views import ViewWriter
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.board_types import Frontier, LineageGraph, Row
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.data import finite_json
from swarm_arena._model.goal import Goal, GoalKey
from swarm_arena._model.ids import IdGen
from swarm_arena._model.plan import AgentSpec, Plan, RoundSpec, TeamSpec
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import AgentStatus, Candidate, Learning, Score
from swarm_arena._model.run_config import RunConfig
from swarm_arena._planning.catalog_check import find_entry
from swarm_arena._planning.validate import judge_private_paths
from swarm_arena._scoring.check import Check
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import Store
from swarm_arena._tooling.staging import tool_identity
from swarm_arena._tooling.tool import Tool, effective_tool_ref


@dataclass(kw_only=True)
class EngineDeps:
    """
    Bundle shared run services without owning their lifetime.

    Attributes
    ----------
    run_id : str
        Durable run identifier.
    config : RunConfig
        Admitted run configuration.
    store : Store
        Run store.
    journal : JournalWriter
        Entered durable writer.
    ledger : Ledger
        Shared run budgets.
    board : Board
        Live committed-result index.
    executor : Executor
        Tool executor.
    artifacts : ArtifactStore
        Content-addressed artifact storage.
    ids : IdGen
        Epoch-aware identifiers.
    views : ViewWriter
        Derived view publisher.
    stops : StopState
        Latched run and goal stops.
    clock : RunClock
        Accumulated active run time.
    admission : RunAdmission or None, default=None
        Shared scheduler capacity service; standalone goals initialize it.

    Examples
    --------
    >>> from dataclasses import fields
    >>> fields(EngineDeps)[0].name
    'run_id'
    """

    run_id: str
    config: RunConfig
    store: Store
    journal: JournalWriter
    ledger: Ledger
    board: Board
    executor: Executor
    artifacts: ArtifactStore
    ids: IdGen
    views: ViewWriter
    stops: StopState
    clock: RunClock
    admission: RunAdmission | None = None


@dataclass
class AgentSlot:
    """
    Carry one admitted agent and its notes and watchdog between rounds.

    Attributes
    ----------
    address : str
        Main-goal/team/agent address.
    team : TeamSpec
        Team definition.
    spec : AgentSpec
        Agent definition.
    manager : Manager
        Built manager instance.
    tool_ref : Ref
        Admitted base tool reference, before round overrides.
    goal : Goal
        Main or part goal.
    goal_key : GoalKey
        Snapshotted goal identity.
    part : str or None
        Own ranking group.
    notes : str
        Mutable notes carried into the next round.
    watchdog : Watchdog
        Cross-round progress tracker.
    status : AgentStatus
        Current lifecycle status.

    Examples
    --------
    >>> from dataclasses import fields
    >>> fields(AgentSlot)[0].name
    'address'
    """

    address: str
    team: TeamSpec
    spec: AgentSpec
    manager: Manager
    tool_ref: Ref
    goal: Goal
    goal_key: GoalKey
    part: str | None
    notes: str
    watchdog: Watchdog
    status: AgentStatus


@dataclass
class RoundOutcome:
    """
    Return new results and repeat entries at the round barrier.

    Attributes
    ----------
    round : int
        Completed round number.
    candidates : list[Candidate]
        New candidates in slot order.
    learnings : list[Learning]
        New learnings in slot order.
    statuses : dict[str, AgentStatus]
        Team/agent lifecycle statuses.
    reasons : dict[str, str]
        Sanitized per-agent explanations.
    repeat_entries : list[tuple[str, Candidate]]
        Recovered pairs followed by new pairs, first candidate id kept.

    Examples
    --------
    >>> RoundOutcome(1, [], [], {}, {}, []).candidates
    []
    """

    round: int
    candidates: list[Candidate]
    learnings: list[Learning]
    statuses: dict[str, AgentStatus]
    reasons: dict[str, str]
    repeat_entries: list[tuple[str, Candidate]]


def _past_run_entries(view: BoardView, run_id: str) -> list[tuple[str, Candidate]]:
    """Keep the newest counted, complete other-run candidate for each repeat key."""
    newest = {}
    for key, candidate in past_run_repeat_entries(view, run_id):
        if candidate.output.status in COUNTED and not candidate.output.partial:
            newest[key] = candidate
    return list(newest.items())


class _SeedView(BoardView):
    """Add only assigned, frontier-covered parents to identifier reads."""

    def __init__(self, view: BoardView, parents: Mapping[str, BoardView]) -> None:
        self._view, self._parents = view, dict(parents)
        self.part = getattr(view, "part", None)
        self.goal, self.frontier = view.goal, view.frontier

    def best(
        self,
        goal: GoalKey,
        k: int,
        *,
        judge: str,
        run_id: str | None = None,
        part: str | None = None,
        fidelity: str | None = None,
    ) -> list[Candidate]:
        """Keep ranking reads restricted to the agent's own group."""
        return self._view.best(goal, k, judge=judge, run_id=run_id, part=part, fidelity=fidelity)

    def learnings(
        self,
        namespace: str,
        *,
        tags: Sequence[str] = (),
        goal: GoalKey | None = None,
        limit: int = 50,
    ) -> list[Learning]:
        """Delegate namespace learning reads."""
        return self._view.learnings(namespace, tags=tags, goal=goal, limit=limit)

    def search_learnings(
        self,
        namespace: str,
        query: str,
        *,
        tags: Sequence[str] = (),
        goal: GoalKey | None = None,
        limit: int = 10,
    ) -> list[Learning]:
        """Delegate scoped learning search."""
        return self._view.search_learnings(namespace, query, tags=tags, goal=goal, limit=limit)

    def count_learnings(self, namespace: str, *, goal: GoalKey | None = None) -> int:
        """Preserve the wrapped snapshot's optional count capability."""
        return self._view.count_learnings(namespace, goal=goal)

    def get_learning(self, id: str) -> Learning:
        """Delegate learning lookup."""
        return self._view.get_learning(id)

    def get_candidate(self, id: str) -> Candidate:
        """Read an assigned parent or an ordinary own-group candidate."""
        return self._parents.get(id, self._view).get_candidate(id)

    def scores(self, candidate_id: str) -> list[Score]:
        """Read scores through the same fixed parent snapshot."""
        return self._parents.get(candidate_id, self._view).scores(candidate_id)

    def lineage(self, candidate_id: str) -> LineageGraph:
        """Delegate lineage without widening the underlying snapshot."""
        return self._parents.get(candidate_id, self._view).lineage(candidate_id)

    def leaderboard(self, goal: GoalKey, *, judge: str) -> list[Row]:
        """Keep leaderboard reads restricted to the own group."""
        return self._view.leaderboard(goal, judge=judge)

    def _visible_candidate_ids(self) -> frozenset[str]:
        return self._view._visible_candidate_ids() | self._parents.keys()


class _RoundContext(AgentContext):
    """Refresh the run's budget latches after every completed tool action."""

    def __init__(
        self,
        rt: AgentRuntime,
        *,
        gate: Callable[[], LimitReached | None],
        refresh: Callable[[], None],
        metadata: _ToolMetadata,
        feedback_for: Callable[[Candidate], Mapping[str, str]],
    ) -> None:
        self._bind(rt, gate=gate, metadata=metadata, feedback_for=feedback_for)
        self._refresh = refresh
        self._running_tools = 0

    async def run_tool(
        self,
        inputs: Mapping[str, Any],
        *,
        parents: Sequence[str] = (),
        replicate: int | None = None,
        edits: Mapping[str, str] | None = None,
    ) -> Candidate:
        """Apply normal action accounting and refresh budget exhaustion on exit."""
        self._running_tools += 1
        try:
            return await super().run_tool(inputs, parents=parents, replicate=replicate, edits=edits)
        finally:
            self._running_tools -= 1
            self._refresh()


class RoundRunner:
    """
    Run independent manager tasks against one committed round frontier.

    Each slot prepares its components under live stop and deadline supervision.
    A failed or abandoned preparation ends only that slot; its manager never
    starts. User constructors and tool metadata hooks run in abandonable daemon
    workers. Agent-status records retain the actual check policy, producing
    group, revision and new candidate ids. Assigned parents and historical
    candidates use that policy; unknown history hides feedback.

    Parameters
    ----------
    deps : EngineDeps
        Shared run services.
    goal_id : str
        Main goal identifier, also used for part budgets and events.
    plan : Plan
        Admitted, snapshotted plan revision.
    agents : Sequence[AgentSlot]
        Slots in deterministic output order. Pruned, stopped, and watchdog-stopped
        slots are not scheduled.
    round_spec : RoundSpec
        Current effort and tool overrides.
    rounds_left : int
        Ordinary rounds remaining, including this one.
    resumed : Mapping[str, Sequence[tuple[str, Candidate]]], optional
        Own committed pairs keyed by team/agent, private until the barrier.
    repeat_entries : Sequence[tuple[str, Candidate]], optional
        Earlier-round repeat pairs.
    frontier : Frontier or None, default=None
        Recorded frontier of a restarted round; its shares are already replayed.
    converge : bool, default=False
        Charge the final holder and ignore ordinary admission stops.
    seed_parents : Sequence[str], optional
        Ordered converge parent identifiers, prepended to every tool request.
    started_runs : Mapping[str, int], optional
        Restored durable starts by team/agent, including lost runs. Without an
        explicit count, a restarted round counts its committed tool_start events.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> plan = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])
    >>> runner = RoundRunner(Mock(spec=EngineDeps), goal_id="g", plan=plan,
    ...                      agents=[], round_spec=RoundSpec(), rounds_left=1)
    >>> runner.goal_id, runner.seed_parents
    ('g', ())
    """

    def __init__(
        self,
        deps: EngineDeps,
        *,
        goal_id: str,
        plan: Plan,
        agents: Sequence[AgentSlot],
        round_spec: RoundSpec,
        rounds_left: int,
        resumed: Mapping[str, Sequence[tuple[str, Candidate]]] = {},
        repeat_entries: Sequence[tuple[str, Candidate]] = (),
        frontier: Frontier | None = None,
        converge: bool = False,
        seed_parents: Sequence[str] = (),
        started_runs: Mapping[str, int] = {},
    ) -> None:
        """Copy incoming round and resume carriers before launching tasks."""
        self.deps, self.goal_id, self.plan = deps, goal_id, plan
        self.agents = tuple(agents)
        self.round_spec, self.rounds_left = round_spec, rounds_left
        self.resumed = {key: tuple(pairs) for key, pairs in resumed.items()}
        self.repeat_entries = tuple(repeat_entries)
        self.frontier, self.converge = frontier, converge
        self.seed_parents = tuple(dict.fromkeys(seed_parents)) if converge else ()
        self.started_runs = dict(started_runs)
        self._counters: dict[str, dict[str, int]] = {}

    def _refresh_budget(self) -> None:
        if self.deps.ledger.run_exhausted():
            self.deps.stops.run_budget = True
        if self.deps.ledger.goal_exhausted(self.goal_id):
            self.deps.stops.goal_budget.add(self.goal_id)

    def _gate(self) -> LimitReached | None:
        self._refresh_budget()
        if self.converge:
            return LimitReached("stop", "the run is stopping") if self.deps.stops.stop_now else None
        return self.deps.stops.gate_for(self.goal_id)()

    def _draft(self, type: EventType, round_no: int, **payload: Any) -> EventDraft:
        return EventDraft(type=type, goal_id=self.goal_id, round=round_no, payload=payload)

    async def _frontier(self, round: int, agents: Sequence[AgentSlot]) -> Frontier:
        if self.frontier is not None:
            return self.frontier
        frontier = await anyio.to_thread.run_sync(self.deps.board.frontier)
        shares = (
            {}
            if self.converge
            else self.deps.ledger.open_round(
                self.goal_id, round, [slot.address for slot in agents], self.rounds_left
            )
        )
        payload: dict[str, Any] = {
            "round": round,
            "run_elapsed_s": self.deps.clock.elapsed(),
            "shares": {holder: amount.model_dump() for holder, amount in shares.items()},
        }
        if self.converge:
            payload["converge"] = True
        self.deps.journal.append(self._draft(EventType.ROUND_STARTED, round, **payload))
        await self.deps.journal.append_durable(
            self._draft(
                EventType.FRONTIER,
                round,
                round=round,
                frontier=frontier.model_dump(mode="json"),
            )
        )
        return frontier

    async def _runtime(
        self,
        slot: AgentSlot,
        round: int,
        frontier: Frontier,
        private_paths: frozenset[str],
        parents: Mapping[str, BoardView],
    ) -> AgentRuntime:
        deps = self.deps
        view = await anyio.to_thread.run_sync(deps.board.snapshot, slot.goal_key, frontier)
        past = _past_run_entries(view, deps.run_id)
        siblings = {}
        if isinstance(view, ArenaView):
            for part in [None, *(p.name for p in self.plan.parts)]:
                sibling = await anyio.to_thread.run_sync(view._sibling_view, part)
                siblings[part or ""] = sibling
        if parents:
            view = _SeedView(view, parents)
        checks, judges, ranking = part_scoring(self.plan, slot.part)
        if slot.part is None:
            ranking = self.plan.ranking_name()
        elif not ranking and len(judges) == 1:
            ranking = judges[0].component_name()
        execution_ref = await run_sync_daemon(
            effective_tool_ref, slot.tool_ref, self.round_spec.tool_overrides
        )
        tool = await run_sync_daemon(execution_ref.build, Tool)
        entry = await run_sync_daemon(
            find_entry, slot.tool_ref, "tool", deps.config.planner_catalog
        )
        holder = f"{self.goal_id}/:final" if self.converge else slot.address
        key = f"{slot.team.name}/{slot.spec.name}"
        pairs = self.resumed.get(key, ())
        return AgentRuntime(
            address=slot.address,
            holder=holder,
            goal_id=self.goal_id,
            goal=team_goal(self.plan, slot.team),
            goal_key=slot.goal_key,
            part=slot.part,
            team=slot.team,
            spec=slot.spec,
            round=round,
            round_spec=self.round_spec,
            fidelity=self.round_spec.effective_fidelity(),
            tool=tool,
            tool_ref=slot.tool_ref,
            execution_ref=execution_ref,
            tool_identity=await run_sync_daemon(tool_identity, slot.tool_ref, entry),
            catalog_entry=entry,
            checks=[
                (ref.component_name(), await run_sync_daemon(ref.build, Check)) for ref in checks
            ],
            view=view,
            own=[candidate for _, candidate in pairs],
            digest=build_digest(
                view,
                goal=slot.goal_key,
                run_id=deps.run_id,
                round=round,
                ranking=ranking,
                cap_tokens=deps.config.limits.digest_tokens,
            ),
            notes=slot.notes,
            ledger=deps.ledger,
            tranche=Tranche(
                deps.ledger,
                holder,
                self.goal_id,
                size=Amount(
                    usd=deps.config.limits.llm_tranche_usd,
                    tokens=deps.config.limits.llm_tranche_tokens,
                ),
            ),
            limits=deps.config.limits,
            counters=AgentCounters(**self._counters.get(key, {})),
            repeats=RepeatIndex(deps.run_id, [*self.repeat_entries, *pairs, *past]),
            watchdog=slot.watchdog,
            journal=deps.journal,
            board=deps.board,
            executor=deps.executor,
            artifacts=deps.artifacts,
            ids=deps.ids,
            config=deps.config,
            resumed=tuple(c for _, c in pairs),
            private_paths=private_paths,
            rankings=await run_sync_daemon(self._rankings),
            sibling_views=siblings,
            seed_parents=self.seed_parents,
            started_runs=self.started_runs.get(key, 0),
        )

    def _rankings(self) -> dict[str, str]:
        """Resolve every live group's ranking within supervised preparation."""
        rankings = {"": self.plan.ranking_name()}
        for part in self.plan.parts:
            _, judges, ranking = part_scoring(self.plan, part.name)
            rankings[part.name] = ranking or (
                judges[0].component_name() if len(judges) == 1 else ""
            )
        return rankings

    async def _feedback_policies(self) -> dict[str, dict[str, str]]:
        """Recover actual producing policies without re-running old check hooks.

        Candidate ids identify their producing group and plan revision. If a
        producer crashed before its status committed, or an older journal has
        no policy snapshot, its candidate feedback remains private.
        """
        if self.deps.journal.head is None:
            return {}
        chain = await anyio.to_thread.run_sync(read_chain, self.deps.store, self.deps.journal.head)
        policies: dict[str, dict[str, str]] = {}
        for _, events in chain:
            for event in events:
                if event.type != EventType.AGENT_STATUS or event.goal_id != self.goal_id:
                    continue
                policy = event.payload.get("feedback_policy")
                if not isinstance(policy, dict):
                    continue
                checks, ids = policy.get("checks"), policy.get("candidates")
                if not isinstance(checks, dict) or not isinstance(ids, list):
                    continue
                modes = {name: mode for name, mode in checks.items() if isinstance(mode, str)}
                for id in ids:
                    if isinstance(id, str):
                        policies[id] = modes
        return policies

    async def _context(
        self, rt: AgentRuntime, policies: Mapping[str, Mapping[str, str]]
    ) -> _RoundContext:
        """Prepare hooks under slot supervision, keeping all view reads on-loop."""
        historical = set(rt.view._visible_candidate_ids())
        historical.update(item.id for item in rt.own if isinstance(item, Candidate))
        historical.update(
            candidate.id for candidate in rt.repeats.candidates_of(f"{rt.team.name}/{rt.spec.name}")
        )
        metadata = await run_sync_daemon(_prepare_metadata, rt.tool, rt.checks)

        def feedback_for(candidate: Candidate) -> Mapping[str, str]:
            if candidate.run_id != self.deps.run_id or candidate.goal_id != self.goal_id:
                return {}
            if candidate.id in historical:
                return policies.get(candidate.id, {})
            if (
                candidate.round == rt.round
                and candidate.part == rt.part
                and candidate.team == rt.team.name
                and candidate.agent == rt.spec.name
            ):
                return metadata.feedback
            return {}

        return _RoundContext(
            rt,
            gate=self._gate,
            refresh=self._refresh_budget,
            metadata=metadata,
            feedback_for=feedback_for,
        )

    async def _parent_views(self, frontier: Frontier) -> dict[str, BoardView]:
        parents: dict[str, BoardView] = {}
        if not self.seed_parents:
            return parents
        for goal in [self.plan.goal, *(part.goal for part in self.plan.parts)]:
            view = await anyio.to_thread.run_sync(self.deps.board.snapshot, goal.key(), frontier)
            for id in self.seed_parents:
                try:
                    view.get_candidate(id)
                except (KeyError, ValueError):
                    continue
                parents[id] = view
        if len(parents) != len(self.seed_parents):
            raise ValueError("assigned parent is outside the round frontier")
        return parents

    async def _watch(
        self,
        inflight: Inflight,
        scopes: Sequence[anyio.CancelScope],
        contexts: Sequence[_RoundContext | None],
        deadline: float,
    ) -> None:
        drain_until: float | None = None
        cancel_until: float | None = None
        while True:
            self._refresh_budget()
            if self.deps.journal.fenced.is_set():
                raise _core_error(Fenced, "run ownership lost")
            now = anyio.current_time()
            stops = self.deps.stops
            if self.deps.clock.left(self.deps.config.limits.run_timeout_s) <= 0:
                stops.run_timeout = True
            immediate = stops.stop_now or now >= deadline
            if not self.converge:
                immediate |= stops.run_timeout
                if stops.stop:
                    inflight.close()
                if (stops.run_budget or self.goal_id in stops.goal_budget) and drain_until is None:
                    inflight.close()
                    drain_until = now + inflight.max_grace_s()
            if cancel_until is None and (
                immediate or (drain_until is not None and now >= drain_until)
            ):
                inflight.close()
                # Collection is owned by the executor and may outlive process grace.
                cancel_until = now + inflight.max_grace_s() + 5.0
                inflight.cancel_all()
            if cancel_until is not None:
                for scope, ctx in zip(scopes, contexts, strict=True):
                    if ctx is None:
                        scope.cancel()
            if cancel_until is not None and now >= cancel_until:
                for scope, ctx in zip(scopes, contexts, strict=True):
                    if (
                        ctx is None
                        or not ctx._running_tools
                        or now >= cancel_until + BEFORE_KILL_TIMEOUT_S
                    ):
                        scope.cancel()
                if now >= cancel_until + BEFORE_KILL_TIMEOUT_S:
                    return
            wake_at = now + 0.5
            if cancel_until is None:
                wake_at = min(wake_at, deadline)
                if drain_until is not None:
                    wake_at = min(wake_at, drain_until)
            elif now < cancel_until:
                wake_at = min(wake_at, cancel_until)
            else:
                wake_at = min(wake_at, cancel_until + BEFORE_KILL_TIMEOUT_S)
            await anyio.sleep_until(wake_at)

    async def _restore(self, round: int) -> None:
        """Recover only the latest matching round's committed action allowance."""
        if (self.frontier is None and not self.converge) or self.deps.journal.head is None:
            return
        chain = await anyio.to_thread.run_sync(read_chain, self.deps.store, self.deps.journal.head)
        events = [event for _, batch in chain for event in batch]
        if self.converge and self.frontier is None:
            state = RunState.from_events(events)
            goal = state.goals[self.goal_id]
            starts = [
                e
                for e in events
                if e.goal_id == self.goal_id
                and e.type == EventType.ROUND_STARTED
                and e.payload.get("converge", False)
                and e.payload["round"] == round
            ]
            if (
                goal.converge_round != round
                or goal.selection_recorded
                or not starts
                or starts[-1].epoch == self.deps.journal.epoch
                or any(
                    e.goal_id == self.goal_id
                    and e.type == EventType.JOIN_FAILED
                    and e.seq > starts[-1].seq
                    for e in events
                )
            ):
                return
            self.frontier = goal.frontiers[round]
            mark = next((m for m in self.frontier.marks if m.run_id == self.deps.run_id), None)
            boundary = (mark.epoch, mark.seq) if mark is not None else (0, 0)
            self.repeat_entries = tuple(
                pair for pair in self.repeat_entries if state.positions[pair[1].id] <= boundary
            )
            recovered: dict[str, list[tuple[str, Candidate]]] = {}
            for candidate in state.candidates.values():
                if (
                    candidate.goal_id == self.goal_id
                    and candidate.round == round
                    and state.positions[candidate.id] > boundary
                ):
                    key = goal.repeat_keys.get(candidate.id)
                    if key is not None:
                        recovered.setdefault(f"{candidate.team}/{candidate.agent}", []).append(
                            (key, candidate)
                        )
            self.resumed = {key: tuple(pairs) for key, pairs in recovered.items()}
        start = max(
            (
                e.seq
                for e in events
                if e.goal_id == self.goal_id
                and e.type == EventType.ROUND_STARTED
                and e.payload["round"] == round
                and bool(e.payload.get("converge", False)) == self.converge
            ),
            default=0,
        )
        admitted: dict[str, set[str]] = {}
        started: dict[str, set[str]] = {}
        for event in events:
            if event.seq <= start or event.goal_id != self.goal_id or event.round != round:
                continue
            key = f"{event.team}/{event.agent}"
            counters = self._counters.setdefault(key, {})
            if event.type == EventType.AGENT_COUNTERS:
                counters.update(
                    {
                        name: event.payload[name]
                        for name in ("turns", "board_reads", "tool_runs")
                        if name in event.payload
                    }
                )
            elif event.type == EventType.LEARNING:
                counters["learnings"] = counters.get("learnings", 0) + 1
            elif event.type in {EventType.TOOL_START, EventType.RUN_LOST, EventType.TOOL_END}:
                id = (
                    event.payload["candidate"]["id"]
                    if event.type == EventType.TOOL_END
                    else event.payload["candidate_id"]
                )
                admitted.setdefault(key, set()).add(id)
                if event.type == EventType.TOOL_START:
                    started.setdefault(key, set()).add(id)
        for key, ids in admitted.items():
            self._counters[key].setdefault("tool_runs", len(ids))
        self.started_runs = {**{key: len(ids) for key, ids in started.items()}, **self.started_runs}

    async def run(self, round: int) -> RoundOutcome:
        """
        Commit the frontier, run managers concurrently, and settle their tranches.

        Parameters
        ----------
        round : int
            Round number, including a distinct converge round when applicable.

        Returns
        -------
        RoundOutcome
            New results, sanitized statuses, and barrier-visible repeat pairs.

        Raises
        ------
        Fenced
            Ownership was lost; propagated bare even from parallel tasks.

        Examples
        --------
        >>> outcome = await runner.run(1)  # doctest: +SKIP
        """
        await self._restore(round)
        agents = [
            slot
            for slot in self.agents
            if slot.status
            not in {AgentStatus.PRUNED, AgentStatus.STOPPED, AgentStatus.WATCHDOG_STOPPED}
        ]
        frontier = await self._frontier(round, agents)
        deadline = round_deadline(
            anyio.current_time(),
            self.deps.config.limits.effective_round_timeout_s(),
            self.deps.clock.left(self.deps.config.limits.run_timeout_s),
        )
        private_paths = judge_private_paths(self.plan)
        parents = await self._parent_views(frontier)
        feedback_policies = await self._feedback_policies()
        runtimes: list[AgentRuntime | None] = [None for _ in agents]
        inflight = Inflight()
        scopes = [anyio.CancelScope() for _ in agents]
        contexts: list[_RoundContext | None] = [None for _ in agents]
        reasons: dict[str, str] = {}
        summaries: dict[str, str] = {}
        done = anyio.Event()
        remaining = len(agents)

        async def manage(index: int, slot: AgentSlot) -> None:
            nonlocal remaining
            key = f"{slot.team.name}/{slot.spec.name}"
            rt: AgentRuntime | None = None
            try:
                with scopes[index], track_inflight(inflight):
                    await anyio.lowlevel.checkpoint()
                    rt = await self._runtime(slot, round, frontier, private_paths, parents)
                    runtimes[index] = rt
                    ctx = await self._context(rt, feedback_policies)
                    contexts[index] = ctx
                    with llm_scope(
                        LLMScope(
                            role="manager",
                            run_id=self.deps.run_id,
                            goal_id=self.goal_id,
                            team=slot.team.name,
                            agent=slot.spec.name,
                            round=round,
                            candidate_id=None,
                            tranche=rt.tranche,
                            journal=self.deps.journal,
                            artifacts=self.deps.artifacts,
                            models=self.deps.config.models,
                            tracing=self.deps.config.tracing,
                            cache=self.deps.config.cache,
                        )
                    ):
                        report = await slot.manager.run_round(ctx)
                        if not isinstance(report, RoundReport) or not isinstance(
                            report.summary, str
                        ):
                            raise TypeError("invalid manager report")
                        summaries[key] = finite_json(report.summary)
            except Fenced:
                raise
            except Exception as error:
                if self.deps.journal.fenced.is_set() or (
                    isinstance(error, BaseExceptionGroup) and error.subgroup(Fenced) is not None
                ):
                    raise _core_error(Fenced, "run ownership lost") from None
                if contexts[index] is None:
                    reasons[key] = f"agent error: {type(error).__name__}"
                else:
                    reasons[key] = (
                        "limit reached"
                        if isinstance(_rebuild_core_signal(error), LimitReached)
                        else f"manager error: {type(error).__name__}"
                    )
            finally:
                if contexts[index] is None:
                    reasons.setdefault(key, "runtime preparation cancelled")
                if rt is not None:
                    slot.notes = rt.notes
                remaining -= 1
                if remaining == 0:
                    done.set()

        if not agents:
            done.set()
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(self._watch, inflight, scopes, contexts, deadline)
                for index, slot in enumerate(agents):
                    tasks.start_soon(manage, index, slot)
                await done.wait()
                tasks.cancel_scope.cancel()
        except BaseExceptionGroup as errors:
            if self.deps.journal.fenced.is_set() or errors.subgroup(Fenced) is not None:
                raise _core_error(Fenced, "run ownership lost") from None
            raise
        if self.deps.journal.fenced.is_set():
            raise _core_error(Fenced, "run ownership lost")
        with anyio.move_on_after(self.deps.config.settle_timeout_s):
            while any(rt.tranche.pending() for rt in runtimes if rt is not None):  # noqa: ASYNC110 - external call settlement
                await anyio.sleep(0.05)
        for rt in runtimes:
            if rt is not None:
                await rt.tranche.close()
        self._refresh_budget()
        candidates: list[Candidate] = []
        learnings: list[Learning] = []
        statuses: dict[str, AgentStatus] = {}
        for slot, rt, ctx in zip(agents, runtimes, contexts, strict=True):
            key = f"{slot.team.name}/{slot.spec.name}"
            own_learnings: list[Learning] = []
            slot.status = AgentStatus.FINISHED
            if rt is not None:
                own_learnings = [item for item in rt.own if isinstance(item, Learning)]
                recovered_ids = {candidate.id for candidate in rt.resumed}
                candidates.extend(
                    item
                    for item in rt.own
                    if isinstance(item, Candidate)
                    and (self.converge or item.id not in recovered_ids)
                )
                learnings.extend(own_learnings)
                reason = rt.watchdog.end_round(ran=rt.started_runs > 0)
                if (
                    rt.counters.tool_runs > self._counters.get(key, {}).get("tool_runs", 0)
                    and not rt.started_runs
                ):
                    slot.status = AgentStatus.STARVED
                if reason is not None:
                    slot.status = AgentStatus.WATCHDOG_STOPPED
                    reasons[key] = reason
                    self.deps.journal.append(
                        self._agent_draft(slot, EventType.WATCHDOG, round, reason=reason)
                    )
            statuses[key] = slot.status
            reasons.setdefault(key, "")
            a2a = agent_round_task(
                context_id=self.deps.run_id,
                address=slot.address,
                round=round,
                learnings=own_learnings,
                status=slot.status,
                summary=reasons[key] or summaries.get(key, ""),
                at=utc_now(),
            )
            self.deps.journal.append(
                self._agent_draft(
                    slot,
                    EventType.AGENT_STATUS,
                    round,
                    status=slot.status.value,
                    reason=reasons[key],
                    a2a=task_to_json(a2a),
                    feedback_policy=(
                        {
                            "revision": self.plan.revision,
                            "part": slot.part,
                            "checks": dict(ctx._feedback),
                            "candidates": [
                                item.id
                                for item in rt.own
                                if isinstance(item, Candidate) and item.id not in recovered_ids
                            ],
                        }
                        if ctx is not None and rt is not None
                        else None
                    ),
                )
            )
        entries: dict[str, tuple[str, Candidate]] = {}
        for pair in [
            *(pair for pairs in self.resumed.values() for pair in pairs),
            *(pair for rt in runtimes if rt is not None for pair in rt.repeats.added),
        ]:
            entries.setdefault(pair[1].id, pair)
        return RoundOutcome(round, candidates, learnings, statuses, reasons, list(entries.values()))

    def _agent_draft(
        self, slot: AgentSlot, type: EventType, round: int, **payload: Any
    ) -> EventDraft:
        return EventDraft(
            type=type,
            task_id=agent_task_id(slot.address, round),
            goal_id=self.goal_id,
            team=slot.team.name,
            agent=slot.spec.name,
            round=round,
            payload=payload,
        )
