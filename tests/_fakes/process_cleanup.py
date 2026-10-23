"""Emergency cleanup for children owned by a test, independent of product teardown."""

import os
import signal
import subprocess
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

import pytest

_WAIT_PROCESS = subprocess.Popen.wait


def reap_children(children: list[tuple[subprocess.Popen[Any], bool]]) -> list[str]:
    """Kill retained child groups or individual children and bound every reap."""
    errors: list[str] = []
    for process, own_group in children:
        try:
            # A leader may already be reaped while its held descendants remain.
            # Only a group created by this test is eligible for group signalling.
            with suppress(ProcessLookupError, PermissionError):
                if own_group:
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
            # Bypass per-instance wait surrogates used by supervisor failure tests.
            _WAIT_PROCESS(process, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"child cleanup failed: {type(error).__name__}")
    return errors


@pytest.fixture
def owned_processes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Retain children at creation so entry failures cannot skip their emergency reap."""
    children: list[tuple[subprocess.Popen[Any], bool]] = []
    original = subprocess.Popen

    class OwnedProcess(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            children.append((self, bool(kwargs.get("start_new_session"))))

    monkeypatch.setattr(subprocess, "Popen", OwnedProcess)
    try:
        yield
    finally:
        leaked = [process for process, _ in children if process.poll() is None]
        errors = reap_children(children)
        assert not leaked, "test left an owned child alive; emergency cleanup ran"
        assert not errors, "; ".join(errors)
