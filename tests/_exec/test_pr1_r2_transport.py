"""Keep low-level executor transport distinct from engine failure policy."""

import asyncio
from pathlib import Path

import pytest
from _fakes.tools import PyTool, make_job

from swarm_arena import Ref
from swarm_arena._errors import Fenced, _core_error
from swarm_arena.executor import LocalExecutor
from swarm_arena.tool import ToolRequest


def failure(kind: str) -> BaseException:
    """Create the exact object that the executor must transport."""
    if kind == "cancel":
        return asyncio.CancelledError("transport")
    if kind == "fenced":
        return Fenced("transport")
    if kind == "core_fenced":
        return _core_error(Fenced, "run ownership lost")
    if kind == "group":
        return ExceptionGroup("transport", [ValueError("transport")])
    return BaseExceptionGroup(
        "transport", [asyncio.CancelledError("transport"), ExceptionGroup("inner", [Fenced()])]
    )


@pytest.mark.parametrize("site", ["constructor", "command", "collect"])
@pytest.mark.parametrize("kind", ["cancel", "fenced", "core_fenced", "group", "mixed"])
async def test_executor_preserves_hook_exception_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str, kind: str
) -> None:
    error = failure(kind)

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(PyTool, "__init__" if site == "constructor" else site, fail)
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


@pytest.mark.parametrize("kind", ["cancel", "fenced", "core_fenced", "group", "mixed"])
async def test_before_kill_transports_exception_identity(tmp_path: Path, kind: str) -> None:
    from _fakes.proc import pid_alive, wait_until

    from swarm_arena.tool import CompletedRun
    from tests._exec.test_supervisor import STUBBORN, stop_after_ready

    error = failure(kind)

    async def broken(run: CompletedRun) -> None:
        raise error

    with pytest.raises(type(error)) as caught:
        await stop_after_ready(
            tmp_path, STUBBORN, grace_s=0.5, before_kill=broken, propagate_cancel=True
        )
    assert caught.value is error
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 60)
