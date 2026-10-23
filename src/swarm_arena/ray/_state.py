"""Read conservative Ray state evidence within one caller-owned deadline.

These blocking helpers do not initialize Ray, retry, or mutate cluster state.
Each SDK request gets a finite timeout. SDK address discovery and transports
can still block outside that timeout: async lifecycle callers must supervise
the whole call with ``run_sync_daemon`` and their remaining allowance, including
first-use initialization. A late response cannot start the next lookup or
produce evidence. Polling and any subsequent action belong to the caller.
"""

import math
from dataclasses import is_dataclass
from time import monotonic
from typing import Annotated, Final

from pydantic import Field, field_validator
from ray.util.state import list_tasks, list_workers

from swarm_arena._model.base import FrozenModel
from swarm_arena._outcomes import _propagate_control

PENDING_STATES: Final[frozenset[str]] = frozenset(
    {
        "PENDING_ARGS_AVAIL",
        "PENDING_NODE_ASSIGNMENT",
        "PENDING_OBJ_STORE_MEM_AVAIL",
        "PENDING_ARGS_FETCH",
        "SUBMITTED_TO_WORKER",
        "PENDING_ACTOR_TASK_ARGS_FETCH",
        "PENDING_ACTOR_TASK_ORDERING_OR_CONCURRENCY",
        "GETTING_AND_PINNING_ARGS",
    }
)
TERMINAL_STATES: Final[frozenset[str]] = frozenset({"FINISHED", "FAILED"})
_RUNNING_STATES: Final = frozenset({"RUNNING", "RUNNING_IN_RAY_GET", "RUNNING_IN_RAY_WAIT"})
_RPC_TIMEOUT_S: Final = 5


class TaskInfo(FrozenModel):
    """Retain a detached task attempt and its observed worker/node identity.

    Parameters
    ----------
    task_id : str
        Ray task identifier.
    attempt_number : int
        Nonnegative Ray attempt number; never inferred from row order.
    name : str
        Exact dispatch name, compared case-sensitively after the SDK filter.
    state : str
        Recognized pending, running or terminal state; NIL is unknown.
    worker_id, node_id : str or None
        Observed identities. Missing identities do not prove never-started work.

    Examples
    --------
    >>> TaskInfo(task_id='task', attempt_number=0, name='dispatch',
    ...          state='RUNNING', worker_id='worker', node_id='node').attempt_number
    0
    """

    task_id: Annotated[str, Field(strict=True, min_length=1)]
    attempt_number: Annotated[int, Field(strict=True, ge=0)]
    name: Annotated[str, Field(strict=True, min_length=1)]
    state: Annotated[str, Field(strict=True)]
    worker_id: Annotated[str, Field(strict=True)] | None
    node_id: Annotated[str, Field(strict=True)] | None

    @field_validator("task_id")
    @classmethod
    def _task_id(cls, value: str) -> str:
        if _absent_id(value):
            raise ValueError("invalid task_id")
        return value

    @field_validator("worker_id", "node_id")
    @classmethod
    def _optional_id(cls, value: str | None) -> str | None:
        return None if value is None or _absent_id(value) else value

    @field_validator("state")
    @classmethod
    def _state(cls, value: str) -> str:
        if value not in PENDING_STATES | _RUNNING_STATES | TERMINAL_STATES:
            raise ValueError("invalid state")
        return value


def _absent_id(value: str) -> bool:
    return not value.strip() or set(value.lower()) == {"f"}


def _deadline(deadline: float | None) -> float:
    return monotonic() + _RPC_TIMEOUT_S if deadline is None else deadline


def _timeout(deadline: float) -> int:
    if not math.isfinite(deadline):
        return 0
    # Ray 2.59's ListApiOptions floors to whole seconds with a minimum of one.
    # Never let that minimum extend a caller's subsecond remaining allowance.
    return max(0, math.floor(min(_RPC_TIMEOUT_S, deadline - monotonic())))


def _rows(field: str, value: str, deadline: float) -> list[object] | None:
    timeout = _timeout(deadline)
    if timeout == 0:
        return None
    lookup = list_tasks if field == "name" else list_workers
    rows = lookup(
        filters=[(field, "=", value)],
        detail=True,
        timeout=timeout,
        limit=2,
        raise_on_missing_output=True,
    )
    # Two rows already prove ambiguity. A one-row limit would silently hide it;
    # raise_on_missing_output also refuses upstream truncation/partial failures.
    if monotonic() >= deadline or not isinstance(rows, list) or len(rows) != 1:
        return None
    return rows


def _matches(info: TaskInfo, expected: TaskInfo) -> bool:
    if (info.name, info.task_id, info.attempt_number) != (
        expected.name,
        expected.task_id,
        expected.attempt_number,
    ):
        return False
    if expected.worker_id is not None and info.worker_id != expected.worker_id:
        return False
    if expected.node_id is not None and info.node_id != expected.node_id:
        return False
    if expected.state in TERMINAL_STATES:
        return info.state == expected.state
    return not (expected.state in _RUNNING_STATES and info.state in PENDING_STATES)


def _find(task_name: str, deadline: float, expected: TaskInfo | None) -> TaskInfo | None:
    if type(task_name) is not str or not task_name:
        return None
    rows = _rows("name", task_name, deadline)
    if rows is None:
        return None
    row = rows[0]
    if not is_dataclass(row) or isinstance(row, type):
        return None
    info = TaskInfo.model_validate(
        {field: getattr(row, field) for field in TaskInfo.model_fields}, strict=True
    )
    if info.name != task_name or (expected is not None and not _matches(info, expected)):
        return None
    return info if monotonic() < deadline else None


def find_task(
    task_name: str,
    *,
    deadline: float | None = None,
    expected: TaskInfo | None = None,
) -> TaskInfo | None:
    """Look up one exact named attempt, returning unknown for ambiguous evidence.

    Parameters
    ----------
    task_name : str
        Unique dispatch name. Case-insensitive SDK matches are checked exactly.
    deadline : float or None, default=None
        Absolute ``time.monotonic()`` deadline shared with the caller's other
        work. None gives this operation five seconds. Every RPC is capped at
        five seconds; less than one remaining second permits no new RPC.
    expected : TaskInfo or None, default=None
        Previously owned/observed attempt. Known task, attempt, worker and node
        identities must match; terminal and running state cannot regress.

    Returns
    -------
    TaskInfo or None
        Detached evidence, or unknown on missing, malformed, incomplete,
        ambiguous, stale, unavailable or late evidence. Multiple attempts are
        ambiguous, even with ``expected``; no newest-row guess is made.

    Notes
    -----
    Names must be unique per dispatch. With only a name, this discovers the
    unique visible attempt; pass ``expected`` to bind later reads to it. The
    eventually consistent API cannot rule out an unseen dispatch/name reuse.
    Ordinary SDK failures become unknown; registered controls propagate.

    Examples
    --------
    An exhausted allowance needs no Ray connection:

    >>> find_task('dispatch', deadline=0) is None
    True
    """
    try:
        return _find(task_name, _deadline(deadline), expected)
    except Exception as error:
        _propagate_control(error)
        return None


def worker_dead(
    task_name: str,
    *,
    deadline: float | None = None,
    expected: TaskInfo | None = None,
) -> bool | None:
    """Join a terminal attempt to positive, identity-matched worker evidence.

    Parameters
    ----------
    task_name : str
        Unique dispatch name.
    deadline : float or None, default=None
        Absolute monotonic deadline for all task and worker calls together;
        None gives the operation five seconds. See ``find_task``.
    expected : TaskInfo or None, default=None
        Previously owned/observed attempt to match, as in ``find_task``.

    Returns
    -------
    bool or None
        False for a recognized nonterminal task or matching live worker.
        True only for a terminal task with a matching dead worker and a fresh
        matching terminal task read. None for every unknown case, including a
        terminal task with absent worker/node fields or a missing worker row.

    Notes
    -----
    Worker listing retains the SDK's completeness checks, unlike ``get_worker``.
    The final task read rejects a changed identity or a regressed state during
    the join. Missing fields never prove that a task was not started. Evidence
    is a snapshot, not an atomic cluster transaction or a process-group probe.

    Examples
    --------
    >>> worker_dead('dispatch', deadline=0) is None
    True
    """
    try:
        until = _deadline(deadline)
        info = _find(task_name, until, expected)
        if info is None:
            return None
        if info.state not in TERMINAL_STATES:
            return False
        if info.worker_id is None or info.node_id is None:
            return None
        rows = _rows("worker_id", info.worker_id, until)
        if rows is None:
            return None
        worker = rows[0]
        worker_id = getattr(worker, "worker_id", None)
        node_id = getattr(worker, "node_id", None)
        alive = getattr(worker, "is_alive", None)
        if (
            not is_dataclass(worker)
            or isinstance(worker, type)
            or type(worker_id) is not str
            or type(node_id) is not str
            or worker_id != info.worker_id
            or node_id != info.node_id
            or type(alive) is not bool
        ):
            return None
        if monotonic() >= until:
            return None
        if alive:
            return False
        return True if _find(task_name, until, info) is not None else None
    except Exception as error:
        _propagate_control(error)
        return None
