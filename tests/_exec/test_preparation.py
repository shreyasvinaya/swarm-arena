"""Subprocess bounds for preparation that never returns or returns after cancel."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("release", ["late", "never"])
def test_blocked_constructor_is_cancellable(tmp_path: Path, release: str) -> None:
    _probe(tmp_path, "constructor", release)


@pytest.mark.parametrize("release", ["late", "never"])
def test_blocked_command_is_cancellable(tmp_path: Path, release: str) -> None:
    _probe(tmp_path, "command", release)


def _probe(tmp_path: Path, where: str, release: str) -> None:
    script = Path(__file__).with_name("data") / "blocked_preparation.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path), where, release],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "cancelled; slot reused; no late launch"


@pytest.mark.parametrize("where", ["constructor", "command"])
@pytest.mark.parametrize("failure", ["error", "interrupt", "cancel"])
async def test_preparation_failures_propagate_and_release_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, failure: str
) -> None:
    import anyio
    from _fakes.tools import PyTool, make_job

    from swarm_arena import Ref
    from swarm_arena.executor import LocalExecutor
    from swarm_arena.tool import ToolRequest

    error: BaseException
    if failure == "interrupt":
        error = KeyboardInterrupt()
    elif failure == "cancel":
        error = anyio.get_cancelled_exc_class()()
    else:
        error = RuntimeError("preparation refused")

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(PyTool, "__init__" if where == "constructor" else "command", fail)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    job = make_job(
        kind="tool",
        command=None,
        tool=Ref(path="_fakes.tools:PyTool", config={"code": "pass"}),
        request=ToolRequest(),
        artifact_root=str(tmp_path / "artifacts"),
    )
    with pytest.raises(type(error)) as caught:
        await executor.submit(job)
    assert caught.value is error
    assert list((tmp_path / "ws").iterdir()) == []
