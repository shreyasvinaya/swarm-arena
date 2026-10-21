"""Define journal event types, drafts, and versioned committed records."""

from enum import StrEnum
from typing import ClassVar

from pydantic import Field

from swarm_arena._model.base import FrozenModel, StoredModel
from swarm_arena._model.canonical import UtcDatetime
from swarm_arena._model.data import FiniteJsonObject


class EventType(StrEnum):
    """
    Identify a journal record's meaning across supported milestones.

    Examples
    --------
    >>> EventType.COMMAND_START.value
    'command_start'
    """

    RUN_STARTED = "run_started"
    RUN_CONFIG = "run_config"
    LIMITS_CHANGED = "limits_changed"
    TAKEOVER = "takeover"
    STOP_REQUESTED = "stop_requested"
    GOAL_QUEUED = "goal_queued"
    GOAL_STARTED = "goal_started"
    GOAL_FINISHED = "goal_finished"
    PLAN_REVISION = "plan_revision"
    PLAN_REJECTED = "plan_rejected"
    FRONTIER = "frontier"
    ROUND_STARTED = "round_started"
    AGENT_STATUS = "agent_status"
    AGENT_COUNTERS = "agent_counters"
    NOTES_SAVED = "notes_saved"
    RESERVE = "reserve"
    SETTLE = "settle"
    TOOL_DISPATCH = "tool_dispatch"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    COMMAND_START = "command_start"
    COMMAND_END = "command_end"
    RUN_LOST = "run_lost"
    SCORE = "score"
    JUDGE_ERROR = "judge_error"
    LEARNING = "learning"
    DECISION = "decision"
    NO_RESULT = "no_result"
    PART_KEPT = "part_kept"
    JOIN_FAILED = "join_failed"
    RESERVE_EXHAUSTED = "reserve_exhausted"
    PROVIDER_FAILED = "provider_failed"
    WATCHDOG = "watchdog"
    ROUND_CLOSED = "round_closed"
    CONVERGE_STARTED = "converge_started"
    FINAL_ORDER = "final_order"
    SOLUTION_WRITTEN = "solution_written"
    LLM_CALL = "llm_call"
    GATEWAY_KEY = "gateway_key"
    RECONCILE = "reconcile"
    CURATED = "curated"
    REVIEW_IMPORTED = "review_imported"


WRITE_THROUGH: frozenset[EventType] = frozenset(
    {
        EventType.RESERVE,
        EventType.SETTLE,
        EventType.TOOL_DISPATCH,
        EventType.TOOL_START,
        EventType.TOOL_END,
        EventType.COMMAND_START,
        EventType.FRONTIER,
        EventType.ROUND_CLOSED,
    }
)
"""Event types that must commit before dependent work starts."""


class EventDraft(FrozenModel):
    """
    Validate event content before the journal assigns its durable position.

    Parameters
    ----------
    type : EventType
        Meaning of the event.
    goal_id : str, default=""
        Goal identifier, empty for a run-level event.
    team : str, default=""
        Team name, when applicable.
    agent : str, default=""
        Agent name, when applicable.
    round : int, default=0
        Round number, zero outside a round.
    task_id : str, default=""
        Associated task identifier, when applicable.
    payload : dict[str, Any], default={}
        Finite JSON object; nested nulls are preserved.

    Examples
    --------
    >>> draft = EventDraft(type=EventType.RESERVE, payload={"tokens": 100, "usd": None})
    >>> draft.type in WRITE_THROUGH
    True
    """

    type: EventType
    goal_id: str = ""
    team: str = ""
    agent: str = ""
    round: int = 0
    task_id: str = ""
    payload: FiniteJsonObject = Field(default_factory=dict)


class Event(StoredModel):
    """
    Store one positioned journal event with a UTC millisecond timestamp.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version, upgraded on loading.
    epoch : int
        Writer's ownership epoch.
    seq : int
        Run-wide sequence number.
    ts : datetime
        Aware timestamp, normalized to UTC and truncated to milliseconds.
    run_id : str
        Run identifier.
    context_id : str
        Event context identifier, assigned the run identifier by the writer.
    task_id : str, default=""
        Associated task identifier, when applicable.
    goal_id : str, default=""
        Goal identifier, empty for a run-level event.
    team : str, default=""
        Team name, when applicable.
    agent : str, default=""
        Agent name, when applicable.
    round : int, default=0
        Round number, zero outside a round.
    type : EventType
        Meaning of the event.
    payload : dict[str, Any], default={}
        Finite JSON object; nested nulls are preserved.

    Examples
    --------
    >>> from swarm_arena._model.canonical import utc_now
    >>> event = Event(epoch=1, seq=1, ts=utc_now(), run_id="r", context_id="r",
    ...               type=EventType.RUN_STARTED)
    >>> Event.model_validate_json(event.model_dump_json()) == event
    True
    """

    schema_kind: ClassVar[str] = "event"
    epoch: int
    seq: int
    ts: UtcDatetime
    run_id: str
    context_id: str
    task_id: str = ""
    goal_id: str = ""
    team: str = ""
    agent: str = ""
    round: int = 0
    type: EventType
    payload: FiniteJsonObject = Field(default_factory=dict)
