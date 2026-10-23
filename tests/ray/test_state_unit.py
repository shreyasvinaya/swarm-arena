"""Conservative state normalization without starting a Ray cluster."""

import asyncio
import dataclasses
import threading
from typing import Any

import anyio
import pytest
from _fakes.secrecy import assert_secret_absent

pytest.importorskip("swarm_arena.ray")

# isort: split
from swarm_arena._agent.manager import LimitReached
from swarm_arena._errors import Fenced, _core_error, _stop_cleanup_error
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena.ray import _state
from swarm_arena.ray._state import PENDING_STATES, TERMINAL_STATES, TaskInfo, find_task, worker_dead


@dataclasses.dataclass
class TaskRow:
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

    monkeypatch.setattr(_state, "list_tasks", tasks)
    monkeypatch.setattr(_state, "list_workers", workers)
    monkeypatch.setattr(_state, "monotonic", lambda: data["now"])
    return data


def test_find_normalizes_dataclass_and_checks_rpc_options(rpc: dict[str, Any]) -> None:
    info = find_task("dispatch", deadline=110)
    assert info == TaskInfo(**dataclasses.asdict(TaskRow()))
    assert rpc["calls"] == [
        (
            "tasks",
            {
                "filters": [("name", "=", "dispatch")],
                "detail": True,
                "timeout": 5,
                "limit": 2,
                "raise_on_missing_output": True,
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
        [dataclasses.asdict(TaskRow())],
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
    assert rpc["calls"][1][1]["raise_on_missing_output"] is True
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
        [dataclasses.asdict(WorkerRow())],
    ],
)
def test_invalid_worker_evidence_is_unknown(rpc: dict[str, Any], rows: list[Any]) -> None:
    rpc["workers"] = rows
    assert worker_dead("dispatch") is None


@pytest.mark.parametrize(
    "changes",
    [
        {"task_id": "other"},
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

    monkeypatch.setattr(_state, "list_workers", workers)
    assert worker_dead("dispatch") is None


@pytest.mark.parametrize("deadline", [99, 100, 100.9, float("nan"), float("inf")])
def test_no_rpc_without_finite_sdk_allowance(rpc: dict[str, Any], deadline: float) -> None:
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

    monkeypatch.setattr(_state, "list_tasks", tasks)
    monkeypatch.setattr(_state, "list_workers", workers)
    assert worker_dead("dispatch", deadline=107) is True
    assert [kw["timeout"] for _, kw in rpc["calls"]] == [5, 5, 3]
    rpc["calls"].clear()
    rpc["now"] = 100
    assert worker_dead("dispatch") is None
    assert [kw["timeout"] for _, kw in rpc["calls"]] == [5, 3, 1]


@pytest.mark.parametrize("site", ["list_tasks", "list_workers"])
def test_late_rpc_result_is_unknown(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    site: str,
) -> None:
    def late(**kwargs: Any) -> list[Any]:
        rpc["now"] = 111
        return [TaskRow()] if site == "list_tasks" else [WorkerRow()]

    monkeypatch.setattr(_state, site, late)
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

    monkeypatch.setattr(_state, site, fail)
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

    monkeypatch.setattr(_state, site, fail)
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

    monkeypatch.setattr(_state, "list_tasks", fail)
    assert find_task("dispatch") is None


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()])
def test_process_controls_pass_through(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    def fail(**kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(_state, "list_tasks", fail)
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

    monkeypatch.setattr(_state, "list_tasks", held)
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
        {"total": 2, "num_after_truncation": 1},
    ],
)
def test_sdk_incomplete_responses_stay_unknown(
    rpc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    incomplete: dict[str, Any],
) -> None:
    from ray.util.state import api

    client = object.__new__(api.StateApiClient)
    monkeypatch.setattr(
        client,
        "_make_http_get_request",
        lambda **kwargs: {
            **incomplete,
            "result": [],
        },
    )
    monkeypatch.setattr(api, "StateApiClient", lambda **kwargs: client)
    monkeypatch.setattr(_state, site, getattr(api, site))
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
