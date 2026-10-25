"""Read conservative Ray State REST evidence from an explicitly bound dashboard.

Each operation shares one monotonic deadline. HTTP sockets are closed when the
caller times out; an uninterruptible OS resolver can leave a daemon until DNS
returns, but cannot subsequently open a socket or start another lookup. No SDK
address discovery, global internal-KV mutation or environment fallback occurs.
"""

import contextlib
import json
import math
import socket
import ssl
import threading
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from ipaddress import ip_address
from time import monotonic
from typing import Annotated, Any, Final
from urllib.parse import urlencode, urlsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from pydantic import Field, field_validator

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
_MAX_RESPONSE_BYTES: Final = 1024 * 1024


class TaskInfo(FrozenModel):
    """Retain a detached task attempt and its observed worker/node identity.

    Parameters
    ----------
    job_id : str
        Dispatching driver job identifier, used to bind recovery evidence.
    task_id : str
        Ray task identifier.
    attempt_number : int
        Nonnegative Ray attempt number; never inferred from row order.
    name : str
        Exact dispatch name, compared case-sensitively after the REST filter.
    state : str
        Recognized pending, running or terminal state; NIL is unknown.
    worker_id, node_id : str or None
        Observed identities. Missing identities do not prove never-started work.

    Examples
    --------
    >>> TaskInfo(job_id='job', task_id='task', attempt_number=0, name='dispatch',
    ...          state='RUNNING', worker_id='worker', node_id='node').attempt_number
    0
    """

    job_id: Annotated[str, Field(strict=True, min_length=1)]
    task_id: Annotated[str, Field(strict=True, min_length=1)]
    attempt_number: Annotated[int, Field(strict=True, ge=0)]
    name: Annotated[str, Field(strict=True, min_length=1)]
    state: Annotated[str, Field(strict=True)]
    worker_id: Annotated[str, Field(strict=True)] | None
    node_id: Annotated[str, Field(strict=True)] | None

    @field_validator("task_id", "job_id")
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


class _Transport:
    """Own sockets and a shrinking allowance for one HTTP exchange."""

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.lock = threading.Lock()
        self.stopped = False
        self.sockets: list[socket.socket] = []

    def remaining(self) -> float:
        left = self.deadline - monotonic()
        if self.stopped or left <= 0:
            raise TimeoutError
        return left

    def track(self, sock: socket.socket) -> socket.socket:
        with self.lock:
            if self.stopped:
                sock.close()
                raise TimeoutError
            self.sockets.append(sock)
        return sock

    def stop(self) -> None:
        with self.lock:
            self.stopped = True
            for sock in self.sockets:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
                sock.close()
            self.sockets.clear()

    def refresh(self) -> None:
        left = self.remaining()
        with self.lock:
            for sock in self.sockets:
                if sock.fileno() >= 0:
                    sock.settimeout(left)

    def connect(self, address: tuple[str, int], *args: Any) -> socket.socket:
        host, port = address
        self.remaining()
        try:
            literal = ip_address(host)
        except ValueError:
            addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        else:
            family = socket.AF_INET if literal.version == 4 else socket.AF_INET6
            addresses = [(family, socket.SOCK_STREAM, 0, "", (host, port))]
        for family, kind, protocol, _, destination in addresses:
            self.remaining()
            sock = self.track(socket.socket(family, kind, protocol))
            try:
                sock.settimeout(self.remaining())
                sock.connect(tuple(destination))
                return sock
            except OSError:
                sock.close()
        raise OSError


class _Connection(HTTPConnection):
    def __init__(self, transport: _Transport, host: str, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self.transport = transport
        self._create_connection = transport.connect

    def send(self, data: Any) -> None:
        if self.sock is None:
            self.connect()
        self.transport.refresh()
        super().send(data)

    def getresponse(self) -> HTTPResponse:
        self.transport.refresh()
        return super().getresponse()


class _TLSConnection(_Connection, HTTPSConnection):
    _context: ssl.SSLContext

    def connect(self) -> None:
        HTTPConnection.connect(self)
        assert self.sock is not None
        wrapped = self._context.wrap_socket(
            self.sock, server_hostname=self.host, do_handshake_on_connect=False
        )
        self.sock = self.transport.track(wrapped)
        self.transport.refresh()
        wrapped.do_handshake()


class _Handler(HTTPHandler, HTTPSHandler):
    def __init__(self, transport: _Transport) -> None:
        super().__init__()
        self.transport = transport

    def http_open(self, req: Request) -> Any:
        return self.do_open(lambda host, **kw: _Connection(self.transport, host, **kw), req)

    def https_open(self, req: Request) -> Any:
        return self.do_open(lambda host, **kw: _TLSConnection(self.transport, host, **kw), req)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise OSError


def _request(url: str, deadline: float) -> Any:
    transport = _Transport(min(deadline, monotonic() + _RPC_TIMEOUT_S))
    done = threading.Event()
    result: list[Any] = []
    errors: list[BaseException] = []

    def read() -> None:
        try:
            try:
                from ray._private.authentication.http_token_authentication import (
                    get_auth_headers_if_auth_enabled,
                )
            except ImportError:
                headers = {}
            else:
                headers = get_auth_headers_if_auth_enabled({})
            # Token loading can involve file I/O; it shares the caller's deadline.
            # A late loader must never start an HTTP exchange after abandonment.
            transport.remaining()
            request = Request(url, headers=headers)
            opener = build_opener(ProxyHandler({}), _Handler(transport), _NoRedirect())
            with opener.open(request, timeout=transport.remaining()) as response:
                body = bytearray()
                while True:
                    transport.refresh()
                    chunk = response.read1(min(65536, _MAX_RESPONSE_BYTES + 1 - len(body)))
                    body.extend(chunk)
                    if len(body) > _MAX_RESPONSE_BYTES:
                        raise ValueError
                    if not chunk:
                        break
                transport.remaining()
                result.append(json.loads(body))
        except BaseException as error:
            errors.append(error)
        finally:
            transport.stop()
            done.set()

    worker = threading.Thread(target=read, daemon=True, name="ray-state-read")
    transport.remaining()
    worker.start()
    try:
        allowance = max(0, transport.deadline - monotonic())
        if not done.wait(allowance) or monotonic() >= transport.deadline:
            return None
        if errors:
            raise errors[0]
        return result[0]
    finally:
        transport.stop()


def _dashboard_url(value: str | None) -> str | None:
    if type(value) is not str or not value or any(char.isspace() for char in value):
        return None
    url = urlsplit(value if "://" in value else "http://" + value)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        return None
    # Validate the port before opening a connection.
    _ = url.port
    return f"{url.scheme}://{url.netloc}"


def _rows(
    field: str,
    value: str,
    deadline: float,
    dashboard_url: str,
    expected_job_id: str | None = None,
) -> list[dict[str, Any]] | None:
    left = deadline - monotonic()
    if not math.isfinite(deadline) or left <= 0:
        return None
    filters = [(field, "=", value)]
    if field == "name" and expected_job_id is not None:
        filters.append(("job_id", "=", expected_job_id))
    # Ray 2.59 StateApiClient._make_param(ListApiOptions(...)); the server
    # timeout has a one-second floor, but the local socket deadline does not.
    params = {
        "limit": 2,
        "timeout": max(1, int(min(_RPC_TIMEOUT_S, left) * 0.8)),
        "detail": True,
        "exclude_driver": True,
        "server_timeout_multiplier": 0.8,
        "filter_keys": [key for key, _, _ in filters],
        "filter_predicates": [op for _, op, _ in filters],
        "filter_values": [val for _, _, val in filters],
    }
    resource = "tasks" if field == "name" else "workers"
    reply = _request(f"{dashboard_url}/api/v0/{resource}?{urlencode(params, doseq=True)}", deadline)
    if monotonic() >= deadline or not isinstance(reply, dict) or reply.get("result") is not True:
        return None
    payload = reply.get("data", {}).get("result")
    if (
        not isinstance(payload, dict)
        or payload.get("partial_failure_warning")
        or payload.get("warnings")
    ):
        return None
    # Task status drops are included in total but absent from num_after_truncation.
    # Without a job filter even another job's drops invalidate the observation.
    total, truncated, filtered = (
        payload.get(key) for key in ("total", "num_after_truncation", "num_filtered")
    )
    rows = payload.get("result")
    if (
        any(type(count) is not int or count < 0 for count in (total, truncated, filtered))
        or total != truncated
        or filtered != 1
        or filtered > truncated
        or not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], dict)
    ):
        return None
    return rows


def _matches(info: TaskInfo, expected: TaskInfo) -> bool:
    if (info.name, info.job_id, info.task_id, info.attempt_number) != (
        expected.name,
        expected.job_id,
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


def _find(
    task_name: str,
    deadline: float,
    expected: TaskInfo | None,
    dashboard_url: str | None,
    expected_job_id: str | None,
) -> TaskInfo | None:
    if type(task_name) is not str or not task_name:
        return None
    dashboard = _dashboard_url(dashboard_url)
    if dashboard is None:
        return None
    if expected_job_id is not None and (
        type(expected_job_id) is not str or _absent_id(expected_job_id)
    ):
        return None
    job_id = (
        expected_job_id if expected_job_id is not None else (expected.job_id if expected else None)
    )
    rows = _rows("name", task_name, deadline, dashboard, job_id)
    if rows is None:
        return None
    row = rows[0]
    info = TaskInfo.model_validate(
        {field: row[field] for field in TaskInfo.model_fields}, strict=True
    )
    if job_id is not None and info.job_id != job_id:
        return None
    if info.name != task_name or (expected is not None and not _matches(info, expected)):
        return None
    return info if monotonic() < deadline else None


def find_task(
    task_name: str,
    *,
    dashboard_url: str | None = None,
    expected_job_id: str | None = None,
    deadline: float | None = None,
    expected: TaskInfo | None = None,
) -> TaskInfo | None:
    """Look up one exact named attempt, returning unknown for ambiguous evidence.

    Parameters
    ----------
    task_name : str
        Unique dispatch name. Case-insensitive REST matches are checked exactly.
    dashboard_url : str or None, default=None
        Connected RayContext dashboard URL (host:port or HTTP(S) origin).
        Missing URL means unknown; there is no address discovery or fallback.
    expected_job_id : str or None, default=None
        Dispatching driver job ID. Filters tasks and scopes dropped events;
        also checked exactly. Defaults to expected.job_id when available.
    deadline : float or None, default=None
        Absolute ``time.monotonic()`` deadline shared with the caller's other
        work. None gives this operation five seconds. Every HTTP exchange is
        capped at five seconds with shrinking connect/read timeouts.
    expected : TaskInfo or None, default=None
        Previously owned/observed attempt. Known job, task, attempt, worker and node
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
    Ordinary REST failures become unknown; registered controls propagate.
    Partial responses and dropped task events mean unknown. Without a job
    filter, drops from unrelated jobs may prevent confirmation indefinitely.

    Examples
    --------
    An exhausted allowance needs no Ray connection:

    >>> find_task('dispatch', deadline=0) is None
    True
    """
    try:
        return _find(task_name, _deadline(deadline), expected, dashboard_url, expected_job_id)
    except Exception as error:
        _propagate_control(error)
        return None


def worker_dead(
    task_name: str,
    *,
    dashboard_url: str | None = None,
    expected_job_id: str | None = None,
    deadline: float | None = None,
    expected: TaskInfo | None = None,
) -> bool | None:
    """Join a terminal attempt to positive, identity-matched worker evidence.

    Parameters
    ----------
    task_name : str
        Unique dispatch name.
    dashboard_url : str or None, default=None
        Connected RayContext dashboard URL (host:port or HTTP(S) origin).
        Missing URL means unknown; there is no address discovery or fallback.
    expected_job_id : str or None, default=None
        Dispatching driver job ID. Filters tasks and scopes dropped events;
        also checked exactly. Defaults to expected.job_id when available.
    deadline : float or None, default=None
        Absolute monotonic deadline for all task and worker calls together;
        None gives the operation five seconds. See ``find_task``.
    expected : TaskInfo or None, default=None
        Previously owned/observed attempt to match, as in ``find_task``.

    Returns
    -------
    bool or None
        False for a recognized nonterminal task without a worker or a matching
        live worker. Nonterminal with a dead/missing worker is unknown.
        True only for a terminal task with a matching dead worker and a fresh
        matching terminal task read. None for every unknown case, including a
        terminal task with absent worker/node fields or a missing worker row.

    Notes
    -----
    Worker listing applies the same completeness checks as task listing.
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
        info = _find(task_name, until, expected, dashboard_url, expected_job_id)
        if info is None:
            return None
        if info.state not in TERMINAL_STATES and info.worker_id is None:
            return False
        if info.worker_id is None or info.node_id is None:
            return None
        rows = _rows("worker_id", info.worker_id, until, _dashboard_url(dashboard_url) or "")
        if rows is None:
            return None
        worker = rows[0]
        worker_id = worker.get("worker_id")
        node_id = worker.get("node_id")
        alive = worker.get("is_alive")
        if (
            type(worker_id) is not str
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
        if info.state not in TERMINAL_STATES:
            return None
        return (
            True
            if _find(task_name, until, info, dashboard_url, expected_job_id) is not None
            else None
        )
    except Exception as error:
        _propagate_control(error)
        return None
