"""Cluster-independent job builders and bounded cancellation ownership helpers."""

import dataclasses
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
from _fakes.proc import wait_for_file
from _fakes.tools import WORKER_SCRIPTS, make_job

from swarm_arena import Ref
from swarm_arena.executor import Executor, Job
from swarm_arena.tool import ToolOutput, ToolRequest


def tool_job(
    code: str, *, tool: str = "PyTool", inputs: dict[str, Any] | None = None, **kw: Any
) -> Job:
    return make_job(
        id=f"executor-{uuid4().hex}",
        kind="tool",
        command=None,
        tool=Ref(path=f"_fakes.tools:{tool}", config={"code": code}),
        request=ToolRequest(inputs=inputs or {}),
        timeout_s=90,
        **kw,
    )


def script_job(script: str = "report_env.py", *, args: list[str] | None = None, **kw: Any) -> Job:
    code = f"import sys; sys.argv = {([script] + (args or []))!r}\n" + WORKER_SCRIPTS[script]
    return tool_job(code, **kw)


def partial_job(ready: Path, **kw: Any) -> Job:
    return script_job("partial_on_sigterm.py", args=["--ready-file", str(ready)], grace_s=2, **kw)


async def cancel_and_capture(
    executor: Executor, job: Job, *, ready: Path
) -> tuple[ToolOutput, float]:
    got: list[ToolOutput] = []
    scopes: list[anyio.CancelScope] = []
    finished = anyio.Event()

    async def partial(output: ToolOutput) -> None:
        got.append(output)

    async def submit() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await executor.submit(dataclasses.replace(job, on_partial=partial))
        finished.set()

    with anyio.fail_after(90):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(submit)
            try:
                await wait_for_file(ready, 60)
                start = time.monotonic()
                scopes[0].cancel()
                await finished.wait()
            finally:
                tasks.cancel_scope.cancel()
    assert len(got) == 1
    return got[0], time.monotonic() - start
