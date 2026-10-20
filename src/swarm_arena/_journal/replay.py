"""Reconstruct resumable run state solely from committed journal events."""

import re
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast, get_args

from swarm_arena._errors import ChainError, _core_error
from swarm_arena._exec.executor import Job
from swarm_arena._journal.event import Event, EventType
from swarm_arena._model.base import FrozenModel, Origin
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import Plan
from swarm_arena._model.planning_types import Decision
from swarm_arena._model.results import AgentStatus, Candidate, Learning, Score
from swarm_arena._model.run_config import RunConfig

STOP_REASONS = frozenset(
    {"stop", "stop --now", "run_timeout_s", "run budget", "goal budget", "reserve_exhausted"}
)
_LEDGER_TYPES = frozenset(
    {
        EventType.RUN_CONFIG,
        EventType.RUN_STARTED,
        EventType.GOAL_STARTED,
        EventType.ROUND_STARTED,
        EventType.RESERVE,
        EventType.SETTLE,
        EventType.GOAL_FINISHED,
        EventType.LIMITS_CHANGED,
    }
)
_ID = re.compile(r"(c|L|s|r|llm|x|k)-[0-9a-f]{16}-e[0-9]+-([0-9]{4,})")


class GoalState(FrozenModel):
    """
    Hold replayed progress and the state needed to resume a goal.

    ``end_reason``, ``final_order`` and the converge lifecycle fields describe
    the latest lifecycle only; earlier final orders and end reasons remain in
    the journal. ``round`` and ``rounds_closed`` describe ordinary rounds.
    A converge-team round uses ``converge_round`` and its own frontier without
    resetting finalization. Repeat keys retain candidate history across rounds.

    Parameters
    ----------
    status : str, default="queued"
        Queued, running, finished, stopped, or no_result.
    round : int, default=0
        Latest ordinary round started.
    rounds_closed : int, default=0
        Latest ordinary round completed.
    agents : dict[str, AgentStatus], default={}
        Latest recorded team/agent statuses.
    decisions : list[tuple[int, Decision]], default=[]
        Policy decisions with their round numbers.
    frontiers : dict[int, Frontier], default={}
        Snapshots for ordinary and converge rounds.
    end_reason : str or None, default=None
        Latest goal_finished reason.
    no_result_reason : str or None, default=None
        Latest no_result reason, separate from the goal_finished reason.
    pending_reason : str or None, default=None
        Lifecycle termination reason retained through takeover before completion.
    pending_replan : int or None, default=None
        First round awaiting the replan recorded by its preceding round closure.
    preflight_revisions : list[int], default=[]
        Revisions with durably completed prerequisite checks.
    replanned_rounds : list[int], default=[]
        Round boundaries with an already committed revision or rejection.
    notes : dict[str, str], default={}
        Most recent saved notes by team/agent.
    repeat_keys : dict[str, str], default={}
        Candidate repeat keys, populated by the tool pipeline's fold.
    final_order : list[str], default=[]
        Latest final candidate order.
    final_parts : dict[str, str or None] or None, default=None
        Latest final selection by part.
    kept_parts : dict[str, str], default={}
        Part selections, populated by the converge pipeline's fold.
    join_attempts : int, default=0
        Failed joins, populated by the converge pipeline's fold.
    join_retries : int, default=0
        Committed retry proposals, counted independently of failed joins.
    join_phase : {'converge', 'replan', 'rounds'}, default='converge'
        Owed join transition, retained across ordinary-round closure.
    join_failures : list[str], default=[]
        Ordered failure reasons in this lifecycle and partition.
    select_only : bool, default=False
        Whether convergence can only select existing candidates.
    converge_started : bool, default=False
        Whether the current lifecycle has entered convergence.
    converge_round : int or None, default=None
        Converge-team round to resume, separate from the ordinary round.
    counters : dict[str, dict[str, int]], default={}
        Latest turns, board_reads and admitted tool_runs in the current round.
    finished : bool, default=False
        Whether this lifecycle has a committed goal_finished event.
    selection_recorded : bool, default=False
        Whether final_order or no_result has committed in this lifecycle.

    Examples
    --------
    >>> GoalState().status
    'queued'
    """

    finished: bool = False
    selection_recorded: bool = False
    status: Literal["queued", "running", "finished", "stopped", "no_result"] = "queued"
    round: int = 0
    rounds_closed: int = 0
    agents: dict[str, AgentStatus] = {}
    decisions: list[tuple[int, Decision]] = []
    frontiers: dict[int, Frontier] = {}
    end_reason: str | None = None
    no_result_reason: str | None = None
    pending_reason: str | None = None
    pending_replan: int | None = None
    preflight_revisions: list[int] = []
    replanned_rounds: list[int] = []
    notes: dict[str, str] = {}
    repeat_keys: dict[str, str] = {}
    final_order: list[str] = []
    final_parts: dict[str, str | None] | None = None
    kept_parts: dict[str, str] = {}
    join_attempts: int = 0
    join_retries: int = 0
    join_phase: Literal["converge", "replan", "rounds"] = "converge"
    join_failures: list[str] = []
    select_only: bool = False
    converge_started: bool = False
    converge_round: int | None = None
    counters: dict[str, dict[str, int]] = {}


class ReservationState(FrozenModel):
    """
    Retain a reservation and its latest settlement without recomputing costs.

    Parameters
    ----------
    reservation : dict[str, Any]
        Original reservation payload.
    state : {'open', 'settled', 'estimated'}, default='open'
        Latest settlement state.
    actual : dict[str, Any] or None, default=None
        Settled amount, absent while open.

    Examples
    --------
    >>> ReservationState(reservation={"id": "r-1"}).state
    'open'
    """

    reservation: dict[str, Any]
    state: Literal["open", "settled", "estimated"] = "open"
    actual: dict[str, Any] | None = None


@dataclass
class RunState:
    """
    Collect comparable run state folded from committed events in sequence order.

    Fields contain plans, goals, board items and their journal positions, open
    processes, reservations, stop state, elapsed time, epoch, and id counters.
    ``ledger_events`` retains whole budget events for history-only accounting.
    ``manifest`` retains the run_started metadata independently of its event.
    ``candidate_goal_keys`` retains each candidate's producing GoalKey, derived
    from its applicable committed revision when tool_end is folded. Historical
    evidence is never relabeled by a later same-name part revision.
    ``llm_request_sha256`` is reserved for the later LLM event fold.

    Examples
    --------
    >>> state = RunState.from_events([])
    >>> (state.run_elapsed_s, state.started, state.stop)
    (0.0, {}, 'none')
    """

    run_id: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    config: RunConfig = field(default_factory=RunConfig)
    plans: dict[str, list[tuple[Plan, Origin]]] = field(default_factory=dict)
    plan_first_rounds: dict[str, list[int]] = field(default_factory=dict)
    goals: dict[str, GoalState] = field(default_factory=dict)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    candidate_goal_keys: dict[str, GoalKey] = field(default_factory=dict)
    fingerprints: dict[str, str | None] = field(default_factory=dict)
    scores: dict[str, Score] = field(default_factory=dict)
    learnings: dict[str, Learning] = field(default_factory=dict)
    positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    reservations: dict[str, ReservationState] = field(default_factory=dict)
    ledger_events: list[Event] = field(default_factory=list)
    started: dict[str, Job.Started] = field(default_factory=dict)
    started_commands: dict[str, Job.Started] = field(default_factory=dict)
    id_counters: dict[str, int] = field(default_factory=dict)
    stop: Literal["none", "stop", "stop-now"] = "none"
    llm_request_sha256: dict[str, str] = field(default_factory=dict)
    run_elapsed_s: float = 0.0
    epoch: int = 0
    lost: set[str] = field(default_factory=set)
    _finished_goals: set[str] = field(default_factory=set, repr=False, compare=False)

    def plan_for_round(self, goal_id: str, round: int) -> Plan:
        """
        Select the latest committed revision applicable to an ordinary round.

        Parameters
        ----------
        goal_id : str
            Goal with recorded revisions.
        round : int
            One-based round about to execute.

        Returns
        -------
        Plan
            Latest applicable revision, or the initial revision.

        Examples
        --------
        >>> from swarm_arena import Goal, Plan
        >>> plan = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])
        >>> state = RunState(plans={"g": [(plan, "human")]})
        >>> state.plan_for_round("g", 1) == plan
        True
        """
        revisions = self.plans[goal_id]
        starts = self.plan_first_rounds.get(goal_id, [1] * len(revisions))
        return next(
            (
                plan
                for (plan, _), start in reversed(list(zip(revisions, starts, strict=True)))
                if start <= round
            ),
            revisions[0][0],
        )

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> "RunState":
        """
        Fold committed events without I/O, component imports, or input mutation.

        Parameters
        ----------
        events : Iterable[Event]
            Committed events; replay sorts them by sequence number and rejects
            duplicate sequence numbers before applying any event.

        Returns
        -------
        RunState
            Independent state suitable for resume, status, and view construction.

        Raises
        ------
        ChainError
            If sequence numbers repeat, including identical duplicate events.
            The error identifies only the duplicate sequence number.
        ValueError
            If a consumed payload is invalid; the error contains no payload values.

        Examples
        --------
        >>> RunState.from_events(iter([])) == RunState.from_events([])
        True
        """
        ordered = sorted(events, key=lambda event: event.seq)
        for previous, event in pairwise(ordered):
            if previous.seq == event.seq:
                raise ChainError(f"duplicate journal seq {event.seq}")
        state = cls()
        for event in ordered:
            try:
                state._apply(deepcopy(event))
            except (ValueError, TypeError, KeyError):
                raise _core_error(ValueError, "invalid journal replay payload") from None
        return state

    def _apply(self, event: Event) -> None:
        """Apply one owned event copy and retain the recorded budget history."""
        payload = event.payload
        self.run_id = event.run_id
        self.epoch = event.epoch
        self.run_elapsed_s = max(self.run_elapsed_s, payload.get("run_elapsed_s", 0.0))
        self._count_ids(event.task_id)
        self._count_ids(payload)
        if event.type in _LEDGER_TYPES:
            self.ledger_events.append(event)
        if event.goal_id:
            self._goal_event(event)

        match event.type:
            case EventType.RUN_STARTED:
                self.manifest = deepcopy(payload.get("manifest", {}))
            case EventType.RUN_CONFIG:
                self.config = RunConfig.model_validate(payload["config"])
            case EventType.LIMITS_CHANGED:
                limits = Limits.model_validate(payload["new"])
                self.config = self.config.model_copy(update={"limits": limits})
            case EventType.PLAN_REVISION:
                revisions = self.plans.setdefault(event.goal_id, [])
                origin = payload["origin"]
                if origin not in get_args(Origin):
                    raise _core_error(ValueError, "invalid plan origin")
                if revisions:
                    plan = Plan._from_data(payload["plan"], base_dir=revisions[0][0].base_dir)
                elif payload.get("base_dir") is not None:
                    plan = Plan._from_data(payload["plan"], base_dir=Path(payload["base_dir"]))
                else:
                    plan = Plan.model_validate(payload["plan"])
                if revisions and revisions[-1][0].parts != plan.parts:
                    self.goals[event.goal_id] = self.goals[event.goal_id].model_copy(
                        update={"kept_parts": {}, "join_attempts": 0, "join_failures": []}
                    )
                revisions.append((plan, cast(Origin, origin)))
                goal = self.goals[event.goal_id]
                first_round = payload.get(
                    "first_round",
                    1 if len(revisions) == 1 else max(goal.round, goal.rounds_closed) + 1,
                )
                self.plan_first_rounds.setdefault(event.goal_id, []).append(first_round)
            case EventType.RESERVE:
                reservation = deepcopy(payload["reservation"])
                self.reservations[reservation["id"]] = ReservationState(reservation=reservation)
            case EventType.SETTLE:
                id = payload["id"]
                self.reservations[id] = self.reservations[id].model_copy(
                    update={
                        "state": "estimated" if payload["estimated"] else "settled",
                        "actual": deepcopy(payload["actual"]),
                    }
                )
            case EventType.TOOL_START:
                self.started[payload["candidate_id"]] = Job.Started.model_validate(
                    payload["started"]
                )
            case EventType.TOOL_END:
                candidate = Candidate.model_validate(payload["candidate"])
                self.candidates[candidate.id] = candidate
                plan = self.plan_for_round(candidate.goal_id, candidate.round)
                producing_goal = (
                    plan.goal
                    if candidate.part is None
                    else {part.name: part.goal for part in plan.parts}[candidate.part]
                )
                self.candidate_goal_keys[candidate.id] = producing_goal.key()
                fingerprint = payload.get("fingerprint")
                if fingerprint is not None and not isinstance(fingerprint, str):
                    raise _core_error(ValueError, "invalid fingerprint")
                self.fingerprints[candidate.id] = fingerprint
                self.positions[candidate.id] = (event.epoch, event.seq)
                self.started.pop(candidate.id, None)
                if "repeat_key" in payload:
                    self.goals.setdefault(event.goal_id, GoalState()).repeat_keys[candidate.id] = (
                        payload["repeat_key"]
                    )
            case EventType.RUN_LOST:
                id = payload["candidate_id"]
                self.lost.add(id)
                self.started.pop(id, None)
            case EventType.COMMAND_START:
                self.started_commands[payload["job_id"]] = Job.Started.model_validate(
                    payload["started"]
                )
            case EventType.COMMAND_END:
                self.started_commands.pop(payload["job_id"], None)
            case EventType.SCORE:
                score = Score.model_validate(payload["score"])
                self.scores[score.id] = score
                self.positions[score.id] = (event.epoch, event.seq)
            case EventType.LEARNING:
                learning = Learning.model_validate(payload["learning"])
                self.learnings[learning.id] = learning
                self.positions[learning.id] = (event.epoch, event.seq)
            case EventType.STOP_REQUESTED:
                if payload["now"]:
                    self.stop = "stop-now"
                elif self.stop != "stop-now":
                    self.stop = "stop"
            case EventType.TAKEOVER:
                self.stop = "none"

    def _goal_event(self, event: Event) -> None:
        """Fold goal progress while keeping converge rounds distinct from main rounds."""
        goal = self.goals.setdefault(event.goal_id, GoalState())
        payload = event.payload
        updates: dict[str, Any] = {}
        address = f"{event.team}/{event.agent}"
        match event.type:
            case EventType.GOAL_QUEUED:
                updates["status"] = "queued"
            case EventType.GOAL_STARTED:
                updates["status"] = "running"
                updates["finished"] = False
                if event.goal_id in self._finished_goals:
                    updates.update(_reset_converge())
                    updates.update(
                        kept_parts={},
                        join_attempts=0,
                        join_failures=[],
                        join_retries=0,
                        join_phase="converge",
                    )
                    updates["end_reason"] = None
                    updates["pending_replan"] = None
                    self._finished_goals.remove(event.goal_id)
            case EventType.GOAL_FINISHED:
                reason = payload.get("reason")
                updates["finished"] = True
                updates["end_reason"] = reason
                if goal.status != "no_result":
                    updates["status"] = "stopped" if reason in STOP_REASONS else "finished"
                self._finished_goals.add(event.goal_id)
            case EventType.NO_RESULT:
                updates["selection_recorded"] = True
                updates["status"] = "no_result"
                updates["no_result_reason"] = payload.get("reason")
                updates["pending_reason"] = payload.get("reason")
            case EventType.ROUND_STARTED:
                updates["counters"] = {}
                if payload.get("converge", False):
                    updates["converge_round"] = payload["round"]
                else:
                    updates.update(_reset_converge())
                    updates["round"] = payload["round"]
            case EventType.ROUND_CLOSED:
                updates["rounds_closed"] = max(
                    goal.rounds_closed, payload.get("round", event.round or goal.round)
                )
                first_round = updates["rounds_closed"] + 1
                if payload.get("replan", False) and first_round not in goal.replanned_rounds:
                    updates["pending_replan"] = first_round
                updates["pending_reason"] = payload.get("reason", goal.pending_reason)
            case EventType.PLAN_REVISION | EventType.PLAN_REJECTED:
                if payload.get("preflight_complete", False):
                    updates["preflight_revisions"] = [
                        *goal.preflight_revisions,
                        payload["plan"]["revision"],
                    ]
                retry = payload.get("join_retry")
                if retry is not None:
                    updates["join_retries"] = retry
                    updates["join_phase"] = (
                        "rounds" if event.type == EventType.PLAN_REVISION else "replan"
                    )
                    updates["converge_round"] = None
                    if event.type == EventType.PLAN_REVISION:
                        updates["converge_started"] = False
                first_round = payload.get(
                    "first_round",
                    max(goal.round, goal.rounds_closed) + 1 if self.plans.get(event.goal_id) else 1,
                )
                if type(first_round) is not int or first_round < 1:
                    raise _core_error(ValueError, "invalid revision first_round")
                updates["replanned_rounds"] = [*goal.replanned_rounds, first_round]
                if goal.pending_replan == first_round:
                    updates["pending_replan"] = None
            case EventType.FRONTIER:
                updates["frontiers"] = {
                    **goal.frontiers,
                    payload["round"]: Frontier.model_validate(payload["frontier"]),
                }
            case EventType.AGENT_STATUS:
                updates["agents"] = {**goal.agents, address: AgentStatus(payload["status"])}
            case EventType.AGENT_COUNTERS:
                updates["counters"] = {
                    **goal.counters,
                    address: {
                        **goal.counters.get(address, {}),
                        "turns": payload["turns"],
                        "board_reads": payload["board_reads"],
                        **({"tool_runs": payload["tool_runs"]} if "tool_runs" in payload else {}),
                    },
                }
            case EventType.NOTES_SAVED:
                updates["notes"] = {**goal.notes, address: payload["text"]}
            case EventType.DECISION:
                updates["decisions"] = [
                    *goal.decisions,
                    (payload["round"], Decision.model_validate(payload["decision"])),
                ]
            case EventType.PART_KEPT:
                updates["kept_parts"] = {
                    **goal.kept_parts,
                    payload["part"]: payload["candidate_id"],
                }
            case EventType.JOIN_FAILED:
                updates["join_phase"] = "replan"
                updates["converge_round"] = None
                updates["counters"] = {}
                updates["join_attempts"] = payload["attempt"]
                updates["join_failures"] = [*goal.join_failures, *payload["reasons"]]
            case EventType.CONVERGE_STARTED:
                updates["join_phase"] = "converge"
                updates["converge_started"] = True
                updates["select_only"] = payload["select_only"]
                updates["pending_reason"] = payload.get("reason")
            case EventType.RESERVE_EXHAUSTED:
                updates["select_only"] = True
            case EventType.FINAL_ORDER:
                updates["selection_recorded"] = True
                updates["final_order"] = payload["candidate_ids"]
                updates["final_parts"] = payload["parts"]
                updates["pending_reason"] = payload.get("reason")
        if updates:
            self.goals[event.goal_id] = goal.model_copy(update=updates)

    def _count_ids(self, value: Any) -> None:
        """Track maxima for complete generated ids, including ids from other run tags."""
        if isinstance(value, str):
            match = _ID.fullmatch(value)
            if match:
                prefix, counter = match.groups()
                self.id_counters[prefix] = max(self.id_counters.get(prefix, 0), int(counter))
        elif isinstance(value, dict):
            for key, item in value.items():
                self._count_ids(key)
                self._count_ids(item)
        elif isinstance(value, list):
            for item in value:
                self._count_ids(item)


def _reset_converge() -> dict[str, Any]:
    """Start a fresh ordinary round or continued lifecycle."""
    return {
        "selection_recorded": False,
        "no_result_reason": None,
        "pending_reason": None,
        "converge_started": False,
        "converge_round": None,
        "select_only": False,
        "final_order": [],
        "final_parts": None,
        "counters": {},
    }


def continues_on_resume(state: RunState, goal_id: str, limits: Limits) -> bool:
    """
    Check whether a stopped goal has another ordinary round to run.

    Parameters
    ----------
    state : RunState
        Committed state before the resume's limits_changed event.
    goal_id : str
        Goal whose latest lifecycle and plan revision are inspected.
    limits : Limits
        Limits requested for the resumed run.

    Returns
    -------
    bool
        True for a stopped goal with rounds left after an explicit stop or a
        limits change. Finished and no-result goals do not continue.

    Examples
    --------
    >>> continues_on_resume(RunState(), "absent", Limits())
    False
    """
    goal = state.goals.get(goal_id)
    revisions = state.plans.get(goal_id)
    return bool(
        goal is not None
        and revisions
        and goal.status == "stopped"
        and goal.rounds_closed < revisions[-1][0].rounds_total(limits)
        and (goal.end_reason in {"stop", "stop --now"} or limits != state.config.limits)
    )
