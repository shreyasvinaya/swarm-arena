"""Own a small CLI-started Ray head for the test process and its child drivers."""

import os
import socket
import subprocess
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def local_ray_cluster() -> Generator[str]:
    """Start one bounded, two-CPU loopback head and stop it on every exit."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"127.0.0.1:{port}"
    tests_root = str(Path(__file__).resolve().parents[1])
    with (
        tempfile.TemporaryDirectory(prefix="sa-ray-", dir="/tmp") as root,
        pytest.MonkeyPatch.context() as patch,
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
        command = [sys.executable, "-m", "ray.scripts.scripts"]
        try:
            subprocess.run(
                [
                    *command,
                    "start",
                    "--head",
                    f"--port={port}",
                    "--node-ip-address=127.0.0.1",
                    "--num-cpus=2",
                    "--object-store-memory=134217728",
                    '--resources={"llm_keys":1}',
                    f"--temp-dir={root}",
                    "--disable-usage-stats",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
            with socket.create_connection(("127.0.0.1", port), timeout=10):
                pass
            yield address
        finally:
            subprocess.run(
                [*command, "stop", "--force"],
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
            )
