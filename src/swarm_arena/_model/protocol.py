"""Define dependency-free A2A v1.0.1 records and execution-state mapping."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from typing import Final

from swarm_arena._model.base import JsonValue
from swarm_arena._model.canonical import _normalize_utc
from swarm_arena._model.tooling import RunStatus


class Role(StrEnum):
    """
    Identify an A2A message sender using proto enum names.

    Examples
    --------
    >>> from swarm_arena.protocol import Role
    >>> Role.AGENT.value
    'ROLE_AGENT'
    """

    UNSPECIFIED = "ROLE_UNSPECIFIED"
    USER = "ROLE_USER"
    AGENT = "ROLE_AGENT"


class TaskState(StrEnum):
    """
    Describe an A2A task state using proto enum names.

    Examples
    --------
    >>> from swarm_arena.protocol import TaskState
    >>> TaskState.COMPLETED.value
    'TASK_STATE_COMPLETED'
    """

    UNSPECIFIED = "TASK_STATE_UNSPECIFIED"
    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    COMPLETED = "TASK_STATE_COMPLETED"
    FAILED = "TASK_STATE_FAILED"
    CANCELED = "TASK_STATE_CANCELED"
    REJECTED = "TASK_STATE_REJECTED"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"


class _Unset(Enum):
    """Distinguish absent data from JSON null, including across pickle copies."""

    VALUE = "unset"


UNSET: Final = _Unset.VALUE


@dataclass(frozen=True, slots=True, kw_only=True)
class Part:
    """
    Carry exactly one A2A content value and optional descriptive metadata.

    Presence counts empty strings and bytes as content. JSON null is represented
    by ``data=None``; omitted data uses a private sentinel.

    Parameters
    ----------
    text : str or None, default=None
        Text content.
    raw : bytes or None, default=None
        Inline binary content.
    url : str or None, default=None
        External content URL.
    data : JsonValue, optional
        Structured content, including null. Omit this argument for absent data;
        the displayed ellipsis denotes omission, not a content value.
    media_type : str or None, default=None
        Content media type.
    filename : str or None, default=None
        Suggested filename.
    metadata : Mapping[str, JsonValue], optional
        Extension metadata; defaults to an independent empty mapping.

    Examples
    --------
    >>> from swarm_arena.protocol import Part
    >>> Part(data=None).data is None
    True
    """

    text: str | None = None
    raw: bytes | None = None
    url: str | None = None
    data: JsonValue | _Unset = UNSET
    media_type: str | None = None
    filename: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require exactly one present content field."""
        if (
            sum(
                (
                    self.text is not None,
                    self.raw is not None,
                    self.url is not None,
                    self.data is not UNSET,
                )
            )
            != 1
        ):
            raise ValueError("Part needs exactly one content field")


@dataclass(frozen=True, slots=True, kw_only=True)
class Message:
    """
    Group nonempty A2A message parts with sender and task references.

    Parameters
    ----------
    message_id : str
        Message identifier.
    role : Role
        Sender role.
    parts : tuple[Part, ...]
        At least one content part.
    context_id : str or None, default=None
        Run context identifier.
    task_id : str or None, default=None
        Associated task identifier.
    reference_task_ids : tuple[str, ...], default=()
        Tasks referenced by this message.
    metadata : Mapping[str, JsonValue], optional
        Extension metadata; defaults to an independent empty mapping.
    extensions : tuple[str, ...], default=()
        Extension URIs.

    Examples
    --------
    >>> from swarm_arena.protocol import Message, Part, Role
    >>> Message(message_id="m1", role=Role.USER, parts=(Part(text="Solve this"),)).role
    <Role.USER: 'ROLE_USER'>
    """

    message_id: str
    role: Role
    parts: tuple[Part, ...]
    context_id: str | None = None
    task_id: str | None = None
    reference_task_ids: tuple[str, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require at least one message part."""
        if not self.parts:
            raise ValueError("Message needs at least one part")


@dataclass(frozen=True, slots=True, kw_only=True)
class Artifact:
    """
    Group nonempty A2A result parts with an artifact identifier.

    Parameters
    ----------
    artifact_id : str
        Artifact identifier.
    parts : tuple[Part, ...]
        At least one result part.
    name : str or None, default=None
        Display name.
    description : str or None, default=None
        Artifact description.
    metadata : Mapping[str, JsonValue], optional
        Extension metadata; defaults to an independent empty mapping.
    extensions : tuple[str, ...], default=()
        Extension URIs.

    Examples
    --------
    >>> from swarm_arena.protocol import Artifact, Part
    >>> Artifact(artifact_id="a1", parts=(Part(data={"score": 1}),)).artifact_id
    'a1'
    """

    artifact_id: str
    parts: tuple[Part, ...]
    name: str | None = None
    description: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require at least one artifact part."""
        if not self.parts:
            raise ValueError("Artifact needs at least one part")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskStatus:
    """
    Record a task state at an aware UTC timestamp with millisecond precision.

    Parameters
    ----------
    state : TaskState
        Current protocol state.
    timestamp : datetime
        Aware time normalized to UTC and truncated to milliseconds.
    message : Message or None, default=None
        Optional progress message.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> from swarm_arena.protocol import TaskState, TaskStatus
    >>> TaskStatus(state=TaskState.WORKING, timestamp=datetime(2026, 1, 1, tzinfo=UTC)).state
    <TaskState.WORKING: 'TASK_STATE_WORKING'>
    """

    state: TaskState
    timestamp: datetime
    message: Message | None = None

    def __post_init__(self) -> None:
        """Normalize aware timestamps without relaxing frozen field assignment."""
        object.__setattr__(self, "timestamp", _normalize_utc(self.timestamp))


@dataclass(frozen=True, slots=True, kw_only=True)
class Task:
    """
    Store an A2A work item with its status, artifacts, and message history.

    Parameters
    ----------
    id : str
        Task identifier.
    context_id : str
        Run context identifier.
    status : TaskStatus
        Current status.
    artifacts : tuple[Artifact, ...], default=()
        Task results.
    history : tuple[Message, ...], default=()
        Retained message history.
    metadata : Mapping[str, JsonValue], optional
        Extension metadata; defaults to an independent empty mapping.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> from swarm_arena.protocol import Task, TaskState, TaskStatus
    >>> status = TaskStatus(state=TaskState.SUBMITTED,
    ...     timestamp=datetime(2026, 1, 1, tzinfo=UTC))
    >>> Task(id="t1", context_id="r", status=status).artifacts
    ()
    """

    id: str
    context_id: str
    status: TaskStatus
    artifacts: tuple[Artifact, ...] = ()
    history: tuple[Message, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


def task_state_for(status: RunStatus) -> tuple[TaskState, dict[str, str]]:
    """Map execution outcomes to A2A states and failure-reason metadata."""
    if status is RunStatus.OK:
        return TaskState.COMPLETED, {}
    if status is RunStatus.CANCELLED:
        return TaskState.CANCELED, {}
    return TaskState.FAILED, {"swarm_arena.reason": status.value}
