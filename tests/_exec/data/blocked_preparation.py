"""Probe preparation cancellation in a disposable interpreter."""

import logging
import os
import sys
import threading
import types
from pathlib import Path

import anyio
from _fakes.tools import PyTool, make_job

from swarm_arena import Ref, Resources
from swarm_arena.executor import Job, LocalExecutor
from swarm_arena.tool import Command, ToolOutput, ToolRequest

ROOT = Path(sys.argv[1])
WHERE, RELEASE = sys.argv[2:]
ENTERED = threading.Event()
UNBLOCK = threading.Event()
THREADS: list[threading.Thread] = []
WARNINGS: list[str] = []


def block() -> None:
    THREADS.append(threading.current_thread())
    ENTERED.set()
    UNBLOCK.wait()


class BlockedTool(PyTool):
    def __init__(self, config: PyTool.Config) -> None:
        super().__init__(config)
        if WHERE == "constructor":
            block()

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        if WHERE == "command":
            block()
        # A released callback still owns a real workspace until it returns.
        (workspace / "late_write").write_text("safe")
        return super().command(request, workspace)


class Capture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        WARNINGS.append(record.getMessage())


MODULE = types.ModuleType("preparation_probe")
MODULE.__dict__["BlockedTool"] = BlockedTool
sys.modules[MODULE.__name__] = MODULE
logging.getLogger("swarm_arena._exec").addHandler(Capture())


async def main() -> None:
    executor = LocalExecutor(max_concurrent=1, workspace_root=str(ROOT / "ws"))
    callbacks: list[str] = []
    scopes: list[anyio.CancelScope] = []
    cancelled = anyio.Event()

    async def on_started(started: Job.Started) -> None:
        callbacks.append("started")

    async def on_partial(output: ToolOutput) -> None:
        callbacks.append("partial")

    job = make_job(
        kind="tool",
        command=None,
        tool=Ref(
            path="preparation_probe:BlockedTool",
            config={"code": f"open({str(ROOT / 'launched')!r}, 'w').close()"},
        ),
        request=ToolRequest(),
        resources=Resources(cpus=float(os.cpu_count() or 1)),
        artifact_root=str(ROOT / "artifacts"),
        on_started=on_started,
        on_partial=on_partial,
    )

    async def submit() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await executor.submit(job)
        assert scope.cancelled_caught
        cancelled.set()

    with anyio.fail_after(12):
        async with anyio.create_task_group() as group:
            group.start_soon(submit)
            while not ENTERED.is_set():  # noqa: ASYNC110
                await anyio.sleep(0.01)
            scopes[0].cancel()
            with anyio.fail_after(5):
                await cancelled.wait()
            [workspace] = (ROOT / "ws").iterdir()
            assert workspace.is_dir()
            assert any(str(workspace) in note and "preparation" in note for note in WARNINGS)
            assert all(thread.daemon for thread in THREADS)
            assert callbacks == []
            # This needs the entire CPU reservation and the only concurrency slot.
            out = await executor.submit(
                make_job(
                    id="next",
                    resources=job.resources,
                    artifact_root=job.artifact_root,
                    command=Command(
                        argv=[
                            "python",
                            "-c",
                            "import os; open(os.environ['SWARM_ARENA_RESULT'], 'w').write('{}')",
                        ]
                    ),
                )
            )
            assert out.status.value == "ok"
            if RELEASE == "late":
                UNBLOCK.set()
                while any(thread.is_alive() for thread in THREADS):  # noqa: ASYNC110
                    await anyio.sleep(0.01)
                assert (workspace / "late_write").read_text() == "safe"
            assert callbacks == []
            assert not (ROOT / "launched").exists()
    await executor.aclose()
    print("cancelled; slot reused; no late launch", flush=True)


anyio.run(main)
