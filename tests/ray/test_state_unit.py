# ruff: noqa: E402
# Check the optional runtime before importing its modules.
"""Conservative state normalization without starting a Ray cluster."""

import asyncio
import dataclasses
import threading
from functools import partial
from typing import Any
from urllib.parse import parse_qs, urlsplit

import anyio
import pytest
from _fakes.dependencies import ray_dependency
from _fakes.secrecy import assert_secret_absent

ray_dependency()

# isort: split
from swarm_arena._agent.manager import LimitReached
from swarm_arena._errors import Fenced, _core_error, _stop_cleanup_error
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena.ray import _state
from swarm_arena.ray._state import PENDING_STATES, TERMINAL_STATES, TaskInfo

find_task = partial(_state.find_task, dashboard_url="http://127.0.0.1:12345")
worker_dead = partial(_state.worker_dead, dashboard_url="http://127.0.0.1:12345")


@dataclasses.dataclass
class TaskRow:
    job_id: Any = "job"
    task_id: Any = "task"
    attempt_number: Any = 0
    name: Any = "dispatch"
    state: Any = "FINISHED"
    worker_id: Any = "worker"
    node_id: Any = "node"


@dataclasses.dataclass
class WorkerRow:
    worker_id: Any = "worker"
    node_id: Any = "node"
    is_alive: Any = False


@pytest.fixture
def rpc(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tasks": [TaskRow()],
        "workers": [WorkerRow()],
        "calls": [],
        "now": 100.0,
    }

    def tasks(**kwargs: Any) -> list[TaskRow]:
        data["calls"].append(("tasks", kwargs))
        return data["tasks"]

    def workers(**kwargs: Any) -> list[WorkerRow]:
        data["calls"].append(("workers", kwargs))
        return data["workers"]

    data["list_tasks"] = tasks
    data["list_workers"] = workers

    def request(url: str, deadline: float) -> dict[str, Any]:
        parsed = urlsplit(url)
        params = parse_qs(parsed.query)
        rows = data["list_" + parsed.path.rsplit("/", 1)[1]](
            filters=list(
                zip(
                    params["filter_keys"],
                    params["filter_predicates"],
                    params["filter_values"],
                    strict=True,
                )
            ),
            detail=params["detail"] == ["True"],
            timeout=int(params["timeout"][0]),
            limit=int(params["limit"][0]),
        )
        return {
            "result": True,
            "data": {
                "result": {
                    "total": len(rows),
                    "num_after_truncation": len(rows),
                    "num_filtered": len(rows),
                    "result": [
                        dataclasses.asdict(row) if dataclasses.is_dataclass(row) else row
                        for row in rows
                    ],
                    **data.get("incomplete", {}).get("list_" + parsed.path.rsplit("/", 1)[1], {}),
                }
            },
        }

    monkeypatch.setattr(_state, "_request", request)
    monkeypatch.setattr(_state, "monotonic", lambda: data["now"])
    return data


def test_find_normalizes_json_and_checks_rpc_options(rpc: dict[str, Any]) -> None:
    info = find_task("dispatch", deadline=110)
    assert info == TaskInfo(**dataclasses.asdict(TaskRow()))
    assert rpc["calls"] == [
        (
            "tasks",
            {
                "filters": [("name", "=", "dispatch")],
                "detail": True,
                "timeout": 4,
                "limit": 2,
            },
        )
    ]
    rpc["tasks"][0].worker_id = "changed"
    assert info is not None
    assert info.worker_id == "worker"
    with pytest.raises(ValueError, match="frozen"):
        info.state = "FAILED"  # ty: ignore[invalid-assignment]


def test_state_sets() -> None:
    assert (
        frozenset(
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
        == PENDING_STATES
    )
    assert frozenset({"FINISHED", "FAILED"}) == TERMINAL_STATES


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [TaskRow(), TaskRow()],
        [TaskRow(), TaskRow(task_id="reused-name")],
        [TaskRow(), TaskRow(attempt_number=1)],
        [TaskRow(name="DISPATCH")],
        [TaskRow(task_id="")],
        [TaskRow(task_id="f" * 48)],
        [TaskRow(task_id=None)],
        [TaskRow(attempt_number=None)],
        [TaskRow(attempt_number=-1)],
        [TaskRow(attempt_number=True)],
        [TaskRow(attempt_number="0")],
        [TaskRow(state="NIL")],
        [TaskRow(state="CANCELLED")],
        [TaskRow(state="new-state")],
        [TaskRow(worker_id=4)],
        [TaskRow(node_id=[])],
        [None],
    ],
)
def test_missing_ambiguous_or_malformed_task_is_unknown(
    rpc: dict[str, Any],
    rows: list[Any],
) -> None:
    rpc["tasks"] = rows
    assert find_task("dispatch") is None
    assert worker_dead("dispatch") is None
    assert all(kind == "tasks" for kind, _ in rpc["calls"])


@pytest.mark.parametrize(
    "state",
    sorted(
        PENDING_STATES
        | {
            "RUNNING",
            "RUNNING_IN_RAY_GET",
            "RUNNING_IN_RAY_WAIT",
        }
    ),
)
def test_nonterminal_is_not_dead(rpc: dict[str, Any], state: str) -> None:
    rpc["tasks"] = [TaskRow(state=state, worker_id=None, node_id=None)]
    assert worker_dead("dispatch") is False
    assert len(rpc["calls"]) == 1


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES))
@pytest.mark.parametrize("alive", [True, False])
def test_terminal_requires_matching_worker(
    rpc: dict[str, Any],
    state: str,
    alive: bool,
) -> None:
    rpc["tasks"] = [TaskRow(state=state)]
    rpc["workers"] = [WorkerRow(is_alive=alive)]
    assert worker_dead("dispatch") is (not alive)
    assert rpc["calls"][1][1]["filters"] == [("worker_id", "=", "worker")]
    if not alive:
        assert [kind for kind, _ in rpc["calls"]] == ["tasks", "workers", "tasks"]


@pytest.mark.parametrize("field", ["worker_id", "node_id"])
@pytest.mark.parametrize("missing", [None, "", "f" * 56])
def test_terminal_missing_identity_never_means_never_started(
    rpc: dict[str, Any],
    field: str,
    missing: str | None,
) -> None:
    setattr(rpc["tasks"][0], field, missing)
    assert worker_dead("dispatch") is None
    assert len(rpc["calls"]) == 1


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [WorkerRow(), WorkerRow()],
        [WorkerRow(worker_id="other")],
        [WorkerRow(node_id="other")],
        [WorkerRow(node_id=None)],
        [WorkerRow(is_alive=None)],
        [WorkerRow(is_alive=0)],
        [WorkerRow(is_alive="false")],
        [None],
    ],
)
def test_invalid_worker_evidence_is_unknown(rpc: dict[str, Any], rows: list[Any]) -> None:
    rpc["workers"] = rows
    assert worker_dead("dispatch") is None


@pytest.mark.parametrize(
    "changes",
    [
        {"task_id": "other"},
        {"job_id": "other"},
        {"attempt_number": 1},
        {"worker_id": "other"},
        {"node_id": "other"},
        {"name": "other"},
        {"state": "RUNNING"},
        {"state": "FAILED"},
    ],
)
def test_expected_identity_rejects_stale_or_reused_task(
    rpc: dict[str, Any],
    changes: dict[str, Any],
) -> None:
    expected = TaskInfo(**dataclasses.asdict(TaskRow()))
    rpc["tasks"] = [dataclasses.replace(TaskRow(), **changes)]
    assert find_task("dispatch", expected=expected) is None
    assert worker_dead("dispatch", expected=expected) is None


def test_pending_identity_can_gain_worker_but_running_cannot_regress(rpc: dict[str, Any]) -> None:
    expected = TaskInfo(
        **dataclasses.asdict(
            TaskRow(
                state="PENDING_NODE_ASSIGNMENT",
                worker_id=None,
                node_id=None,
            )
        )
    )
    assert worker_dead("dispatch", expected=expected) is True
    expected = expected.model_copy(update={"state": "RUNNING"})
    rpc["tasks"] = [TaskRow(state="PENDING_NODE_ASSIGNMENT")]
    assert find_task("dispatch", expected=expected) is None


@pytest.mark.parametrize(
    "replacement",
    [
        [],
        [TaskRow(attempt_number=1)],
        [TaskRow(task_id="other")],
        [TaskRow(worker_id="other")],
        [TaskRow(node_id="other")],
        [TaskRow(state="RUNNING")],
        [TaskRow(), TaskRow(task_id="other")],
    ],
)
def test_death_rechecks_task_after_worker_lookup(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    replacement: list[TaskRow],
) -> None:
    def workers(**kwargs: Any) -> list[WorkerRow]:
        rpc["tasks"] = replacement
        return [WorkerRow()]

    rpc["list_workers"] = workers
    assert worker_dead("dispatch") is None


@pytest.mark.parametrize("deadline", [99, 100, float("nan"), float("inf")])
def test_no_rpc_without_finite_allowance(rpc: dict[str, Any], deadline: float) -> None:
    assert find_task("dispatch", deadline=deadline) is None
    assert worker_dead("dispatch", deadline=deadline) is None
    assert not rpc["calls"]


def test_calls_share_shrinking_deadline(
    rpc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def tasks(**kwargs: Any) -> list[TaskRow]:
        rpc["calls"].append(("tasks", kwargs))
        rpc["now"] += 2
        return [TaskRow()]

    def workers(**kwargs: Any) -> list[WorkerRow]:
        rpc["calls"].append(("workers", kwargs))
        rpc["now"] += 2
        return [WorkerRow()]

    rpc["list_tasks"] = tasks
    rpc["list_workers"] = workers
    assert worker_dead("dispatch", deadline=107) is True
    assert [kw["timeout"] for _, kw in rpc["calls"]] == [4, 4, 2]
    rpc["calls"].clear()
    rpc["now"] = 100
    assert worker_dead("dispatch") is None
    assert [kw["timeout"] for _, kw in rpc["calls"]] == [4, 2, 1]


@pytest.mark.parametrize("site", ["list_tasks", "list_workers"])
def test_late_rpc_result_is_unknown(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    site: str,
) -> None:
    def late(**kwargs: Any) -> list[Any]:
        rpc["now"] = 111
        return [TaskRow()] if site == "list_tasks" else [WorkerRow()]

    rpc[site] = late
    assert worker_dead("dispatch", deadline=110) is None


@pytest.mark.parametrize("site", ["list_tasks", "list_workers"])
@pytest.mark.parametrize("failure", [TimeoutError, RuntimeError, ValueError])
def test_rpc_failures_are_unknown_without_exposing_values(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    site: str,
    failure: type[Exception],
) -> None:
    def fail(**kwargs: Any) -> Any:
        raise failure("state-SENTINEL-secret")

    rpc[site] = fail
    assert worker_dead("dispatch") is None
    assert "state-SENTINEL-secret" not in repr(capsys.readouterr())


@pytest.mark.parametrize("site", ["list_tasks", "list_workers"])
@pytest.mark.parametrize("kind", ["terminal", "fenced", "nested_terminal", "nested_fenced"])
def test_registered_controls_propagate(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    kind: str,
) -> None:
    error = _stop_cleanup_error() if "terminal" in kind else _core_error(Fenced, "ownership lost")
    injected = (
        ExceptionGroup("state-SENTINEL-secret", [error]) if kind.startswith("nested") else error
    )

    def fail(**kwargs: Any) -> Any:
        raise injected

    rpc[site] = fail
    with pytest.raises(type(error)) as caught:
        worker_dead("dispatch")
    assert_secret_absent(caught.value, "state-SENTINEL-secret")


@pytest.mark.parametrize(
    "error",
    [
        Fenced("state-SENTINEL-secret"),
        LimitReached("stop", "state-SENTINEL-secret"),
        RuntimeError("secret"),
    ],
)
def test_unregistered_lookalikes_are_unknown(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail(**kwargs: Any) -> Any:
        raise error

    rpc["list_tasks"] = fail
    assert find_task("dispatch") is None


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()])
def test_process_controls_pass_through(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    def fail(**kwargs: Any) -> Any:
        raise error

    rpc["list_tasks"] = fail
    with pytest.raises(type(error)):
        find_task("dispatch")


async def test_held_rpc_is_abandonable_and_cannot_start_next_lookup(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def held(**kwargs: Any) -> list[TaskRow]:
        entered.set()
        assert release.wait(10)
        rpc["now"] = 120
        return [TaskRow()]

    def lookup() -> bool | None:
        try:
            return worker_dead("dispatch", deadline=110)
        finally:
            finished.set()

    rpc["list_tasks"] = held
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(run_sync_daemon, lookup)
            with anyio.fail_after(5):
                assert await anyio.to_thread.run_sync(entered.wait, 5)
            group.cancel_scope.cancel()
    finally:
        release.set()
        assert await anyio.to_thread.run_sync(finished.wait, 5)
    assert not rpc["calls"]


@pytest.mark.parametrize("site", ["list_tasks", "list_workers"])
@pytest.mark.parametrize(
    "incomplete",
    [
        {"partial_failure_warning": "state-SENTINEL-secret", "total": 1, "num_after_truncation": 1},
        {"total": 2, "num_after_truncation": 1},  # one dropped status event
        {"warnings": ["data unavailable"]},
        {"num_filtered": 2},
        {"total": True},
    ],
)
def test_rest_incomplete_responses_stay_unknown(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    incomplete: dict[str, Any],
) -> None:
    rpc["incomplete"] = {site: incomplete}
    assert worker_dead("dispatch") is None


def test_eventually_visible_evidence_is_not_cached(rpc: dict[str, Any]) -> None:
    rpc["tasks"] = []
    assert find_task("dispatch") is None
    rpc["tasks"] = [TaskRow(state="RUNNING")]
    observed = find_task("dispatch")
    assert observed is not None
    rpc["tasks"] = [TaskRow()]
    rpc["workers"] = []
    assert worker_dead("dispatch", expected=observed) is None
    rpc["workers"] = [WorkerRow()]
    assert worker_dead("dispatch", expected=observed) is True


def test_explicit_dashboard_and_job_identity(rpc: dict[str, Any]) -> None:
    assert (
        find_task("dispatch", dashboard_url="http://127.0.0.1:12345", expected_job_id="other")
        is None
    )


@pytest.mark.parametrize("workers", [[], [WorkerRow()], [WorkerRow(is_alive=True)]])
def test_nonterminal_worker_evidence(rpc: dict[str, Any], workers: list[WorkerRow]) -> None:
    rpc["tasks"] = [TaskRow(state="RUNNING")]
    rpc["workers"] = workers
    assert worker_dead("dispatch") is (False if workers and workers[0].is_alive else None)


@pytest.mark.parametrize("mode", ["held", "refused"])
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_endpoint_deadline(mode: str, scheme: str) -> None:
    import socket
    import time

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        if mode == "held":
            server.listen()
        start = time.monotonic()
        assert (
            _state.find_task(
                "dispatch",
                dashboard_url=f"{scheme}://127.0.0.1:{server.getsockname()[1]}",
                deadline=start + 0.2,
            )
            is None
        )
        assert time.monotonic() - start < 1


@pytest.mark.parametrize(
    "url", [None, "", "ray://127.0.0.1:12345", "http://host/path", "http://user:pass@host"]
)
def test_no_dashboard_discovery(url: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import os

    import ray._private.services
    from ray.experimental import internal_kv

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("state lookup attempted address discovery or global KV mutation")

    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(type(os.environ), "__getitem__", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(ray._private.services, "get_ray_address_from_environment", forbidden)
    monkeypatch.setattr(internal_kv, "_initialize_internal_kv", forbidden)
    assert _state.find_task("dispatch", dashboard_url=url) is None
    assert _state.worker_dead("dispatch", dashboard_url=url) is None


def test_expected_job_filter_is_sent_and_checked(rpc: dict[str, Any]) -> None:
    assert find_task("dispatch", expected_job_id="job") is not None
    assert rpc["calls"][0][1]["filters"] == [("name", "=", "dispatch"), ("job_id", "=", "job")]
    assert find_task("dispatch", expected_job_id="other") is None
    expected = TaskInfo(**dataclasses.asdict(TaskRow()))
    assert find_task("dispatch", expected=expected) is not None
    assert rpc["calls"][-1][1]["filters"][-1] == ("job_id", "=", "job")


@pytest.mark.parametrize(
    "mode", ["success", "dropped", "oversize", "redirect", "drip_headers", "drip_body"]
)
def test_loopback_rest_transport(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from ray.util.state.api import StateApiClient
    from ray.util.state.common import ListApiOptions

    entered = threading.Event()
    release = threading.Event()
    paths: list[str] = []
    failures: list[BaseException] = []
    body = json.dumps(
        {
            "result": True,
            "data": {
                "result": {
                    "total": 2 if mode == "dropped" else 1,
                    "num_after_truncation": 1,
                    "num_filtered": 1,
                    "result": [dataclasses.asdict(TaskRow())],
                }
            },
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            entered.set()
            try:
                if mode == "redirect":
                    self.send_response(302)
                    self.send_header("Location", "/wrong-cluster")
                    self.end_headers()
                    return
                if mode == "drip_headers":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                else:
                    self.send_response(200)
                    self.end_headers()
                if mode in {"drip_headers", "drip_body"}:
                    while not release.wait(0.01):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                    return
                self.wfile.write(
                    b" " * (_state._MAX_RESPONSE_BYTES + 1) if mode == "oversize" else body
                )
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as error:
                failures.append(error)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        server.timeout = 3
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        # Invalid ambient addresses must never influence the explicitly bound origin.
        monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:1")
        monkeypatch.setenv("RAY_API_SERVER_ADDRESS", "http://127.0.0.1:1")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("no_proxy", "")
        import builtins
        import os

        from ray.experimental import internal_kv

        accesses: list[str] = []
        original_getitem = type(os.environ).__getitem__
        original_open = builtins.open

        def getenv(self: Any, key: str) -> str:
            if key in {"RAY_ADDRESS", "RAY_API_SERVER_ADDRESS"}:
                accesses.append(key)
                raise AssertionError("ambient cluster address read")
            return original_getitem(self, key)

        def open_file(file: Any, *args: Any, **kwargs: Any) -> Any:
            if "ray_current_cluster" in str(file):
                accesses.append("cluster file")
                raise AssertionError("local cluster file read")
            return original_open(file, *args, **kwargs)

        def initialize_kv(*args: Any, **kwargs: Any) -> None:
            accesses.append("internal KV")
            raise AssertionError("global KV mutation")

        monkeypatch.setattr(type(os.environ), "__getitem__", getenv)
        monkeypatch.setattr(builtins, "open", open_file)
        monkeypatch.setattr(internal_kv, "_initialize_internal_kv", initialize_kv)
        start = time.monotonic()
        try:
            info = _state.find_task(
                "dispatch",
                dashboard_url=f"127.0.0.1:{server.server_port}",
                expected_job_id="job",
                deadline=start + 0.3,
            )
            assert entered.is_set()
            assert not accesses
            assert (info is not None) is (mode == "success")
            assert time.monotonic() - start < 1.5
            assert len(paths) == 1
            expected = StateApiClient._make_param(
                ListApiOptions(
                    limit=2,
                    timeout=1,
                    detail=True,
                    filters=[("name", "=", "dispatch"), ("job_id", "=", "job")],
                )
            )
            from urllib.parse import urlencode

            assert parse_qs(urlsplit(paths[0]).query) == parse_qs(urlencode(expected, doseq=True))
        finally:
            release.set()
            thread.join(3)
        assert not thread.is_alive()
        assert not failures


def test_dns_deadline_prevents_late_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket
    import time

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    connections: list[object] = []
    original_stop = _state._Transport.stop

    def resolve(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(3)
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 1))]

    def stop(self: Any) -> None:
        original_stop(self)
        if threading.current_thread().name == "ray-state-read":
            finished.set()

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: connections.append(a))
    monkeypatch.setattr(_state._Transport, "stop", stop)
    start = time.monotonic()
    try:
        assert (
            _state.find_task("dispatch", dashboard_url="http://held.invalid", deadline=start + 0.2)
            is None
        )
        assert entered.is_set()
        assert time.monotonic() - start < 1
    finally:
        release.set()
        assert finished.wait(3)
    assert not connections
