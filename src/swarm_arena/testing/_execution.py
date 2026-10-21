"""Watch executor runs and retain stderr diagnostics for non-ok outcomes."""

import contextlib
from collections.abc import AsyncGenerator, Sequence
from typing import Final

import anyio

from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.plan import Resources
from swarm_arena._model.tooling import RunStatus, ToolOutput, parse_logs_ref
from swarm_arena._scoring.check import _prepare_command_output
from swarm_arena.testing._state import _artifacts

_LOG_TAIL_BYTES: Final = 2048
_NOT_OK: Final = tuple(status for status in RunStatus if status is not RunStatus.OK)


class _Watched(Executor):
    """Pass jobs to another executor and keep every output, to report failed runs."""

    def __init__(self, inner: Executor) -> None:
        super().__init__()
        self._inner = inner
        self.outputs: list[tuple[str, object]] = []

    async def submit(self, job: Job) -> ToolOutput:
        """Submit the job to the wrapped executor and keep its output."""
        out = await self._inner.submit(job)
        detached = await run_sync_daemon(_prepare_command_output, out)
        self.outputs.append((job.id, detached))
        return out

    async def capacity(self) -> Resources | None:
        """Return the wrapped executor's capacity."""
        return await self._inner.capacity()

    async def aclose(self) -> None:
        """Leave closing to the owner of the wrapped executor."""

    async def confirm_dead(self, started: Job.Started) -> bool | None:
        """Ask the wrapped executor whether a recorded job has stopped."""
        return await self._inner.confirm_dead(started)

    async def kill_orphan(self, started: Job.Started) -> bool:
        """Ask the wrapped executor to stop an orphan."""
        return await self._inner.kill_orphan(started)


@contextlib.asynccontextmanager
async def _executor(executor: Executor | None) -> AsyncGenerator[_Watched, None]:
    """Wrap the caller's executor, or a new LocalExecutor that is closed afterwards."""
    if executor is not None:
        yield _Watched(executor)
        return
    owned = LocalExecutor()
    try:
        yield _Watched(owned)
    finally:
        await owned.aclose()


def _stderr_tail(out: ToolOutput) -> str:
    """Read the last bytes of a run's stderr log from this process's artifact store."""
    logs_ref = getattr(out, "logs_ref", None)  # model_construct can leave it unset or wrong
    if not isinstance(logs_ref, str):
        return "(no stderr log)"
    try:
        _, stderr_sha = parse_logs_ref(logs_ref)
    except ValueError:
        return "(no stderr log)"
    try:
        data = _artifacts().get_bytes(stderr_sha)
    except Exception:
        return "(the stderr log cannot be read)"
    return data[-_LOG_TAIL_BYTES:].decode(errors="replace")


def _failed_runs(
    outputs: Sequence[tuple[str, object]], statuses: Sequence[RunStatus] = _NOT_OK
) -> list[str]:
    """Describe each run with one of ``statuses`` and the end of its log (the tool's own text)."""
    notes = []
    for job_id, out in outputs:
        if not isinstance(out, ToolOutput):
            continue
        status = getattr(out, "status", None)  # model_construct can leave any object here
        failed = next((member for member in statuses if status == member), None)
        if failed is not None:
            tail = _stderr_tail(out)
            notes.append(
                f"run {job_id} ended with status {failed}; the end of its stderr log:\n{tail}"
            )
    return notes


@contextlib.asynccontextmanager
async def _failed_runs_on(runner: _Watched) -> AsyncGenerator[None, None]:
    """Add a note for each run that is not ok to any error from the block, then re-raise it."""
    try:
        yield
    except Exception as error:  # not BaseException: a cancellation stays as it is
        with anyio.CancelScope(shield=True):  # bounded: it reads the stored logs' tails
            notes = await anyio.to_thread.run_sync(_failed_runs, runner.outputs, _NOT_OK)
        for note in notes:
            error.add_note(note)
        raise
