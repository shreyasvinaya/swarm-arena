"""Own a small CLI-started Ray head for the test process and its child drivers."""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TextIO

import pytest


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _group_exists(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _stop_group(process: subprocess.Popen[str]) -> str | None:
    """Stop only this head's process group, returning any cleanup diagnostic."""
    try:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        if not _wait_group_exit(process.pid, 5):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            if not _wait_group_exit(process.pid, 5):
                return "Ray test head process group did not exit after SIGKILL"
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Ray test head cleanup failed: {type(error).__name__}"
    return None


def _stderr_tail(stderr: TextIO) -> str:
    stderr.flush()
    stderr.seek(0, os.SEEK_END)
    stderr.seek(max(0, stderr.tell() - 8192))
    return "\n".join(stderr.read().splitlines()[-20:])


@contextmanager
def local_ray_cluster() -> Generator[str]:
    """Start one bounded, two-CPU loopback head and stop its group on every exit."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("Ray cluster tests require sequential execution without xdist")
    with socket.socket() as gcs, socket.socket() as client, socket.socket() as dashboard:
        gcs.bind(("127.0.0.1", 0))
        client.bind(("127.0.0.1", 0))
        dashboard.bind(("127.0.0.1", 0))
        dashboard_port = dashboard.getsockname()[1]
        port, client_port = gcs.getsockname()[1], client.getsockname()[1]
    address = f"127.0.0.1:{port}"
    tests_root = str(Path(__file__).resolve().parents[1])
    with (
        tempfile.TemporaryDirectory(
            prefix="sa-ray-", dir="/tmp", ignore_cleanup_errors=True
        ) as root,
        pytest.MonkeyPatch.context() as patch,
        (Path(root) / "stdout.log").open("w+", encoding="utf-8", errors="replace") as stdout,
        (Path(root) / "stderr.log").open("w+", encoding="utf-8", errors="replace") as stderr,
    ):
        patch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
        patch.setenv("RAY_USAGE_STATS_ENABLED", "0")
        patch.setenv("RAY_ADDRESS", address)
        # This loopback-only test head does not exercise token provisioning.
        patch.setenv("RAY_AUTH_MODE", "disabled")
        patch.setenv(
            "PYTHONPATH", os.pathsep.join(filter(None, [tests_root, os.getenv("PYTHONPATH")]))
        )
        endpoint = os.getenv("SWARM_ARENA_TEST_S3_ENDPOINT")
        if endpoint:
            patch.setenv("AWS_ENDPOINT_URL_S3", endpoint)
        command = [
            sys.executable,
            "-u",
            "-m",
            "ray.scripts.scripts",
            "start",
            "--head",
            "--block",
            f"--port={port}",
            f"--ray-client-server-port={client_port}",
            "--include-dashboard=false",
            f"--dashboard-port={dashboard_port}",
            "--node-ip-address=127.0.0.1",
            "--num-cpus=2",
            "--object-store-memory=134217728",
            '--resources={"llm_keys":1}',
            f"--temp-dir={root}",
            "--disable-usage-stats",
        ]
        try:
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, text=True, start_new_session=True
            )
        except OSError as error:
            pytest.fail(
                f"Ray test head start failed: {type(error).__name__}\n{_stderr_tail(stderr)}"
            )
        print(f"Ray test head process group: {process.pid}", flush=True)
        failure: BaseException | None = None
        try:
            deadline = time.monotonic() + 90
            while True:
                if process.poll() is not None:
                    pytest.fail(f"Ray test head start failed\n{_stderr_tail(stderr)}")
                try:
                    with socket.create_connection(("127.0.0.1", client_port), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        pytest.fail(f"Ray test head readiness timed out\n{_stderr_tail(stderr)}")
                    time.sleep(0.05)
            yield address
        except BaseException as error:
            failure = error
            raise
        finally:
            diagnostic = _stop_group(process)
            if diagnostic:
                if failure is None:
                    pytest.fail(diagnostic)
                print(diagnostic, file=sys.stderr)
