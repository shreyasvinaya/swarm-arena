"""Build deterministic A2A records and serialize their proto JSON field names."""

from base64 import b64encode
from collections.abc import Mapping, Sequence
from dataclasses import fields
from datetime import datetime
from enum import Enum
from typing import Any

from swarm_arena._model.canonical import format_utc
from swarm_arena._model.protocol import UNSET, task_state_for
from swarm_arena._model.results import AgentStatus, Candidate, Learning
from swarm_arena.protocol import Artifact, Message, Part, Role, Task, TaskState, TaskStatus


def agent_task_id(address: str, round: int) -> str:
    """
    Identify an agent's task within one round.

    Parameters
    ----------
    address : str
        Full goal/team/agent address.
    round : int
        Round number.

    Returns
    -------
    str
        Address followed by the round suffix.

    Examples
    --------
    >>> agent_task_id("g/t/a", 2)
    'g/t/a#r2'
    """
    return f"{address}#r{round}"


def tool_run_task(candidate: Candidate, *, context_id: str, at: datetime) -> Task:
    """
    Record a tool outcome with its parents on a request history message.

    Parameters
    ----------
    candidate : Candidate
        Committed tool result and request.
    context_id : str
        Run identifier.
    at : datetime
        Aware output or event timestamp; no wall clock is consulted.

    Returns
    -------
    Task
        Candidate task with the mapped execution state and reason metadata.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> from swarm_arena.testing import make_candidate
    >>> from swarm_arena.tool import RunStatus, ToolOutput
    >>> candidate = make_candidate(ToolOutput(status=RunStatus.OK), id="c-1")
    >>> tool_run_task(candidate, context_id="r", at=datetime(2026, 1, 1, tzinfo=UTC)).id
    'c-1'
    """
    state, metadata = task_state_for(candidate.output.status)
    request = Message(
        message_id=f"{candidate.id}:request",
        role=Role.USER,
        parts=(Part(data=candidate.request.model_dump(mode="json")),),
        context_id=context_id,
        task_id=candidate.id,
        reference_task_ids=tuple(candidate.parent_ids),
    )
    return Task(
        id=candidate.id,
        context_id=context_id,
        status=TaskStatus(state=state, timestamp=at),
        history=(request,),
        metadata=metadata,
    )


def agent_round_task(
    *,
    context_id: str,
    address: str,
    round: int,
    learnings: Sequence[Learning],
    status: AgentStatus,
    summary: str,
    at: datetime,
) -> Task:
    """
    Record an agent round with one structured artifact per learning.

    Parameters
    ----------
    context_id : str
        Run identifier.
    address : str
        Full goal/team/agent address.
    round : int
        Round number.
    learnings : Sequence[Learning]
        Learnings in recorded order.
    status : AgentStatus
        Agent's round status.
    summary : str
        Text for the task status message.
    at : datetime
        Aware event timestamp; no wall clock is consulted.

    Returns
    -------
    Task
        Round task with learning JSON and titles, plus a status message.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> task = agent_round_task(context_id="r", address="g/t/a", round=1,
    ...     learnings=[], status=AgentStatus.FINISHED, summary="done",
    ...     at=datetime(2026, 1, 1, tzinfo=UTC))
    >>> task.status.state.value
    'TASK_STATE_COMPLETED'
    """
    state = {
        AgentStatus.ACTIVE: TaskState.WORKING,
        AgentStatus.FINISHED: TaskState.COMPLETED,
        AgentStatus.PRUNED: TaskState.CANCELED,
        AgentStatus.STOPPED: TaskState.CANCELED,
        AgentStatus.STARVED: TaskState.FAILED,
        AgentStatus.WATCHDOG_STOPPED: TaskState.FAILED,
    }[status]
    task_id = agent_task_id(address, round)
    message = Message(
        message_id=f"{task_id}:status",
        role=Role.AGENT,
        parts=(Part(text=summary),),
        context_id=context_id,
        task_id=task_id,
    )
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=state, timestamp=at, message=message),
        artifacts=tuple(
            Artifact(
                artifact_id=learning.id,
                parts=(Part(data=learning.model_dump(mode="json")), Part(text=learning.title)),
            )
            for learning in learnings
        ),
        metadata={"swarm_arena.reason": status.value} if state == TaskState.FAILED else {},
    )


def task_to_json(task: Task) -> dict[str, Any]:
    """
    Serialize an A2A task using proto JSON names and UTC milliseconds.

    Parameters
    ----------
    task : Task
        Record to serialize without mutation.

    Returns
    -------
    dict[str, Any]
        Independent JSON tree with camelCase fields, enum names, and base64
        bytes. Absent fields are omitted; present null data is retained.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> task = Task(id="t", context_id="r", status=TaskStatus(
    ...     state=TaskState.COMPLETED, timestamp=datetime(2026, 1, 1, tzinfo=UTC)))
    >>> task_to_json(task)["status"]
    {'state': 'TASK_STATE_COMPLETED', 'timestamp': '2026-01-01T00:00:00.000Z'}
    """
    return _wire(task)


def _wire(value: Any) -> Any:
    """Copy protocol records to their JSON representation without altering data keys."""
    if isinstance(value, Task | TaskStatus | Message | Artifact | Part):
        result: dict[str, Any] = {}
        for field in fields(value):
            item = getattr(value, field.name)
            if item is UNSET or (item is None and field.name != "data"):
                continue
            if field.name != "data" and isinstance(item, tuple | Mapping) and not item:
                continue
            first, *rest = field.name.split("_")
            name = first + "".join(word.title() for word in rest)
            result[name] = _wire(item)
        return result
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return format_utc(value)
    if isinstance(value, bytes):
        return b64encode(value).decode("ascii")
    if isinstance(value, Mapping):
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_wire(item) for item in value]
    return value
