import contextlib
import io
import json
import os
from http.client import BadStatusLine, HTTPException
from pathlib import Path
from typing import Any

import pytest
from _fakes import ray_cluster as helper


class Head:
    pid = 123456789

    def __init__(self, *, failed: bool = False) -> None:
        self.failed = failed

    def poll(self) -> int | None:
        return 1 if self.failed else None

    def wait(self, timeout: float | None = None) -> int:
        return 1 if self.failed else 0


def fake_launch(
    monkeypatch: pytest.MonkeyPatch, *, failed: bool = False
) -> list[tuple[list[str], dict[str, Any]]]:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def popen(args: list[str], **kwargs: Any) -> Head:
        calls.append((args, kwargs))
        if failed:
            kwargs["stderr"].write("startup diagnostic marker\n")
            kwargs["stderr"].flush()
        return Head(failed=failed)

    monkeypatch.setattr(helper.subprocess, "Popen", popen)
    monkeypatch.setattr(
        helper.socket, "create_connection", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(helper, "_state_ready", lambda port, timeout: True, raising=False)
    monkeypatch.setattr(helper, "_stop_group", lambda process: None)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    return calls


def test_cluster_owns_group_ports_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_launch(monkeypatch)
    with helper.local_ray_cluster() as address:
        assert os.environ["RAY_ADDRESS"] == address
        assert os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert "--block" in args
    assert "--include-dashboard=true" in args
    assert "--dashboard-host=127.0.0.1" in args
    assert kwargs["start_new_session"] is True
    assert "--num-cpus=2" in args
    gcs_port = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--port="))
    client_port = next(
        arg.split("=", 1)[1] for arg in args if arg.startswith("--ray-client-server-port=")
    )
    dashboard_port = next(
        arg.split("=", 1)[1] for arg in args if arg.startswith("--dashboard-port=")
    )
    agent_ports = {
        next(arg.split("=", 1)[1] for arg in args if arg.startswith(flag))
        for flag in ("--dashboard-agent-listen-port=", "--dashboard-agent-grpc-port=")
    }
    assert len({gcs_port, client_port, dashboard_port, *agent_ports}) == 5
    root = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--temp-dir="))
    assert Path(root).parent == Path("/tmp")


def test_cluster_refuses_xdist_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_launch(monkeypatch)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    with pytest.raises(pytest.skip.Exception, match="xdist"), helper.local_ray_cluster():
        pass
    assert calls == []


def test_cluster_start_failure_shows_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_launch(monkeypatch, failed=True)
    with (
        pytest.raises(pytest.fail.Exception, match="startup diagnostic marker"),
        helper.local_ray_cluster(),
    ):
        pass


def test_cluster_cleanup_preserves_primary_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_launch(monkeypatch)
    monkeypatch.setattr(helper, "_stop_group", lambda process: "cleanup diagnostic")

    with pytest.raises(LookupError, match="primary diagnostic"), helper.local_ray_cluster():
        raise LookupError("primary diagnostic")
    assert "cleanup diagnostic" in capsys.readouterr().err


def test_cluster_cleanup_failure_after_success_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_launch(monkeypatch)
    monkeypatch.setattr(helper, "_stop_group", lambda process: "cleanup diagnostic")
    with (
        pytest.raises(pytest.fail.Exception, match="cleanup diagnostic"),
        helper.local_ray_cluster(),
    ):
        pass


def test_group_cleanup_escalates_only_its_own_group(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal

    signals: list[tuple[int, int]] = []
    waits = iter([False, True])
    monkeypatch.setattr(helper.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(helper, "_wait_group_exit", lambda pgid, timeout: next(waits))
    process: Any = Head()
    assert helper._stop_group(process) is None
    assert signals == [(Head.pid, signal.SIGTERM), (Head.pid, signal.SIGKILL)]


def test_cluster_tolerates_log_directory_cleanup_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_launch(monkeypatch)
    original = helper.tempfile.TemporaryDirectory
    calls: list[bool] = []

    def temporary_directory(**kwargs: Any) -> Any:
        calls.append(kwargs.get("ignore_cleanup_errors", False))
        return original(**kwargs)

    monkeypatch.setattr(helper.tempfile, "TemporaryDirectory", temporary_directory)
    with helper.local_ray_cluster():
        pass
    assert calls == [True]


def test_readiness_waits_for_state_api(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_launch(monkeypatch)
    responses = iter([False, True])
    attempts: list[tuple[int, float]] = []

    def state_ready(port: int, timeout: float) -> bool:
        attempts.append((port, timeout))
        return next(responses)

    monkeypatch.setattr(helper, "_state_ready", state_ready, raising=False)
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: None)
    with helper.local_ray_cluster():
        assert len(attempts) == 2
        assert attempts[0][0] == attempts[1][0]
        assert all(0 < timeout <= 1 for _, timeout in attempts)


def test_unready_state_api_exhausts_single_startup_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_launch(monkeypatch)
    stopped: list[int] = []
    clock = [100.0]

    def unready(port: int, timeout: float) -> bool:
        clock[0] += 45
        return False

    monkeypatch.setattr(helper, "_state_ready", unready, raising=False)
    monkeypatch.setattr(helper.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(helper, "_stop_group", lambda process: stopped.append(process.pid))
    with (
        pytest.raises(pytest.fail.Exception, match="readiness timed out"),
        helper.local_ray_cluster(),
    ):
        pytest.fail("yielded before State API readiness")
    assert stopped == [Head.pid]


@pytest.mark.parametrize(
    ("body", "ready"),
    [
        ({"result": True, "data": {"result": {"result": []}}}, True),
        ({"result": False, "data": {"result": {"result": []}}}, False),
        ({"result": True}, False),
        ([], False),
    ],
)
def test_state_readiness_requires_successful_state_response(
    monkeypatch: pytest.MonkeyPatch,
    body: Any,
    ready: bool,
) -> None:
    calls: list[tuple[str, float]] = []

    def open_url(url: str, *, timeout: float) -> io.BytesIO:
        calls.append((url, timeout))
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(helper, "urlopen", open_url, raising=False)
    assert helper._state_ready(12345, 0.5) is ready
    assert calls == [("http://127.0.0.1:12345/api/v0/tasks?limit=1&timeout=1", 0.5)]


@pytest.mark.parametrize("error", [HTTPException, BadStatusLine])
def test_state_readiness_protocol_failure(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error("bad response")

    monkeypatch.setattr(helper, "urlopen", fail)
    assert helper._state_ready(12345, 0.5) is False
