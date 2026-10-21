"""Supervise one local job, collect its result once, and upload bounded outputs."""

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import anyio

from swarm_arena._errors import Fenced, _exception_type_name
from swarm_arena._exec import supervisor
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local_stage import _Staged
from swarm_arena._exec.supervisor import ProcessSupervisor, RLimits
from swarm_arena._exec.threads import _DaemonCall, run_sync_daemon
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.tooling import CompletedRun, FileRef, RunStatus, ToolOutput, format_logs_ref
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.io import _storage_io, _validated_io
from swarm_arena._tooling.materialize import _stage_layout, collect_output_files
from swarm_arena._tooling.paths import output_paths
from swarm_arena._tooling.result_file import (
    _OUTPUT_BYTE_LIMIT,
    MAX_DECLARED_FILES,
    NOTE_PREFIX,
    command_output_from_result,
    note_in_log,
    open_log,
    open_regular_file,
)

if TYPE_CHECKING:
    from swarm_arena._exec.local import LocalExecutor

_STDOUT: Final = "swarm/stdout.log"
_STDERR: Final = "swarm/stderr.log"
_STOPPED: Final = frozenset({RunStatus.TIMEOUT, RunStatus.CANCELLED})
_CHUNK: Final = 1 << 20
_NOT_AN_OUTPUT: Final = "collect failed: Tool.collect() did not return a ToolOutput"


def _read_tail(fd: int, size: int, limit: int) -> bytes:
    """Read the last ``limit`` of ``size`` bytes at most, even if the file grows meanwhile."""
    offset = max(0, size - limit)
    remaining = size - offset
    chunks: list[bytes] = []
    while remaining > 0:
        chunk = os.pread(fd, min(remaining, _CHUNK), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _truncated(rel: str, dropped: int) -> str:
    """Return the note for a log whose first ``dropped`` bytes are not uploaded."""
    return f"log truncated: {rel}: dropped the first {dropped} bytes"


class _JobRun:
    """Supervise one staged job, keep its collected output, and upload its results."""

    def __init__(
        self,
        job: Job,
        staged: _Staged,
        workspace: Path,
        artifacts: ArtifactStore,
        config: "LocalExecutor.Config",
    ) -> None:
        self._job = job
        self._staged = staged
        self._workspace = workspace
        self._artifacts = artifacts
        self._max_log = config.max_log_bytes
        self._max_output = config.max_output_bytes
        self._stderr = workspace / _STDERR
        self._started = False  # the launcher reached the start handshake
        self._output: ToolOutput | None = None  # the latest collected output
        self._post_collect: _DaemonCall[ToolOutput | None] | None = None
        self._collect_deadline = 0.0
        self._collected = False  # the collect after a normal exit finished
        self._run: CompletedRun | None = None  # what the supervisor returned
        self._provisional: CompletedRun | None = None  # the run a stop-path collect saw

    async def run(self, sup: ProcessSupervisor, rlimits: RLimits | None) -> ToolOutput:
        """Run the job; on a cancel after the start, deliver its output, then re-raise."""
        uploading = False
        try:
            run = await sup.run(
                self._staged.argv,
                cwd=self._staged.cwd,
                env=self._staged.env,
                stdout=self._workspace / _STDOUT,
                stderr=self._stderr,
                timeout_s=self._staged.timeout_s,
                grace_s=self._staged.grace_s,
                rlimits=rlimits,
                on_started=self._on_started,
                before_kill=self._before_kill,
            )
            self._run = run
            if run.status in _STOPPED:  # collect ran (or was given up) in before_kill
                output = await anyio.to_thread.run_sync(self._stopped_output, run.status)
            else:
                self._collect_deadline = anyio.current_time() + supervisor.BEFORE_KILL_TIMEOUT_S
                self._post_collect = _DaemonCall(self._collect, run)
                await self._wait_post_collect()
                output = self._output or ToolOutput(status=RunStatus.ERROR)
            uploading = True
            return await self._finish(output, (run.started_at, run.ended_at))
        except anyio.get_cancelled_exc_class():
            if self._started:  # a cancel before the start handshake has no output to give
                with anyio.CancelScope(shield=True):
                    await self._deliver_partial(upload=not uploading)
            raise

    async def fault_output(self, reason: str) -> ToolOutput:
        """Upload an unstarted job's diagnostics under the caller's supervision."""
        await _validated_io(self._fault_logs, reason)
        return await self._finish(ToolOutput(status=RunStatus.ERROR), None)

    def _fault_logs(self, reason: str) -> None:
        """Prepare local diagnostics without publishing any artifacts."""
        for rel in (_STDOUT, _STDERR):
            os.close(open_log(self._workspace / rel, truncate=True))
        note_in_log(self._stderr, reason)

    async def _on_started(self, started: Job.Started) -> None:
        """Mark the start, then commit it through ``job.on_started`` with the job id."""
        self._started = True
        commit = self._job.on_started
        if commit is not None:
            await commit(started.model_copy(update={"job_id": self._job.id}))

    async def _before_kill(self, provisional: CompletedRun) -> None:
        """Collect the partial output in a daemon thread the supervisor can stop waiting on."""
        self._provisional = provisional
        self._output = await run_sync_daemon(self._collect, provisional)

    async def _deliver_partial(self, *, upload: bool) -> None:
        """Hand the best output to ``job.on_partial`` after a cancel (the caller shields)."""
        run = self._run
        if run is not None and run.status not in _STOPPED and not self._collected:
            # Rejoin the same collect, with its original deadline; never call it twice.
            self._provisional = run.model_copy(update={"status": RunStatus.CANCELLED})
            await self._wait_post_collect()
        output = await anyio.to_thread.run_sync(self._stopped_output, RunStatus.CANCELLED)
        timed = run or self._provisional
        times = None if timed is None else (timed.started_at, timed.ended_at)
        # Preserve collected data, but never publish refs from an abandoned upload.
        update: dict[str, Any] = {"files": [], "logs_ref": None}
        if times is not None:
            update["started_at"], update["ended_at"] = times
        partial = output.model_copy(update=update)
        if upload:
            try:
                with anyio.move_on_after(supervisor.BEFORE_KILL_TIMEOUT_S) as uploads:
                    partial = await self._finish(output, times)
                if uploads.cancel_called:
                    await anyio.to_thread.run_sync(note_in_log, self._stderr, "uploads timed out")
            except Exception as exc:  # upload failure must not replace caller cancellation
                note = f"uploads failed: {_exception_type_name(exc)}"
                await anyio.to_thread.run_sync(note_in_log, self._stderr, note)
        if self._job.on_partial is not None:
            with anyio.move_on_after(supervisor.BEFORE_KILL_TIMEOUT_S):
                await self._job.on_partial(partial)

    async def _wait_post_collect(self) -> None:
        """Wait for the one post-exit collect until its original deadline."""
        assert self._post_collect is not None
        with anyio.move_on_after(max(0.0, self._collect_deadline - anyio.current_time())) as scope:
            self._output = await self._post_collect.result()
            self._collected = True
        if scope.cancelled_caught:
            await anyio.to_thread.run_sync(note_in_log, self._stderr, "collect timed out")

    def _collect(self, run: CompletedRun) -> ToolOutput | None:
        """Convert ordinary collection failures; transport nonstandard outcomes unchanged."""
        token = _OUTPUT_BYTE_LIMIT.set(self._max_output)
        try:
            output: object
            if self._staged.tool is not None and self._staged.request is not None:
                output = self._staged.tool.collect(self._staged.request, self._workspace, run)
            else:
                output = command_output_from_result(self._workspace, run)
        except (Fenced, BaseExceptionGroup):
            raise
        except (Exception, SystemExit) as error:
            note_in_log(run.stderr_path, f"collect failed: {_exception_type_name(error)}")
            return None
        finally:
            _OUTPUT_BYTE_LIMIT.reset(token)
        try:
            if not isinstance(output, ToolOutput):
                raise TypeError(_NOT_AN_OUTPUT)
            if len(output.files) > MAX_DECLARED_FILES:
                note_in_log(run.stderr_path, "too many declared files")
                return None
            # model_copy and model_construct skip validation (the finite-JSON rule too).
            return _detach_model(ToolOutput, output)
        except (Fenced, BaseExceptionGroup):
            raise
        except (Exception, SystemExit):
            note_in_log(run.stderr_path, _NOT_AN_OUTPUT)  # the reason only: no values
            return None

    def _stopped_output(self, status: RunStatus) -> ToolOutput:
        """
        Stamp the stop on the output: the stop status, and ``partial``.

        ``partial`` is True when the stop-path collect returned a valid
        ``ToolOutput`` that says it is partial or keeps data or files (spec
        §5: a valid result is kept as partial; without one, the data is
        empty). A collect that keeps data but says ``partial=False`` is still
        partial, and files that ``_finish`` drops later do not change it. A
        valid but empty output that does not say partial is not partial. A
        failed, abandoned, or invalid collect gives an empty output with
        ``partial=False`` and a note.
        """
        output = self._output
        if output is None:
            if self._provisional is not None:  # a collect was tried and gave nothing
                note_in_log(self._stderr, "no partial output: collect failed or timed out")
            return ToolOutput(status=status)
        partial = output.partial or bool(output.data or output.files)
        return output.model_copy(update={"status": status, "partial": partial})

    def _read_log(self, rel: str) -> bytes:
        """Read the last ``max_log_bytes`` of a regular log; anything else is empty."""
        try:
            fd = open_regular_file(self._workspace, rel)
        except ValueError:
            note_in_log(self._stderr, f"{rel} is not a regular file; its log is uploaded empty")
            return b""
        try:
            size = os.fstat(fd).st_size
            if size > self._max_log:
                dropped = size - self._max_log
                if rel == _STDERR:  # the note lands in this log and pushes out its own length
                    for _ in range(3):
                        line = f"{NOTE_PREFIX}{_truncated(rel, dropped)}\n"  # note_in_log's form
                        dropped = size + len(line.encode()) - self._max_log
                note_in_log(self._stderr, _truncated(rel, dropped))
                size = os.fstat(fd).st_size
            return _read_tail(fd, size, self._max_log)
        finally:
            os.close(fd)

    def _check_output_sizes(self, files: Sequence[FileRef]) -> None:
        """Raise ValueError naming the declared output whose size crosses the run's cap."""
        if len(files) > MAX_DECLARED_FILES:
            raise ValueError("too many declared files")
        total = 0
        for ref in files:
            try:
                fd = open_regular_file(self._workspace, ref.path)
            except ValueError:
                continue  # collect_output_files names what is wrong with it
            try:
                total += os.fstat(fd).st_size
            finally:
                os.close(fd)
            if total > self._max_output:
                raise ValueError(f"{ref.path} crosses max_output_bytes")

    def _output_paths(self, files: Sequence[FileRef]) -> list[str]:
        """Check the complete declaration before the first per-object upload."""
        self._check_output_sizes(files)
        layout = _stage_layout(self._job.stage)
        return output_paths(
            [ref.path for ref in files],
            effective_files=[*layout.files, *layout.metadata],
            directories=sorted(layout.directories),
        )

    def _upload_file(self, path: str, left: int) -> FileRef | str:
        """Upload one checked output; retain trusted path diagnostics as data."""
        try:
            [ref] = collect_output_files(
                self._workspace,
                [path],
                self._artifacts,
                stage=self._job.stage,
                max_bytes=left,
            )
        except ValueError as error:
            # collect_output_files has already replaced backend details with
            # a fixed diagnostic naming only the declared workspace path.
            return str(error)
        return ref

    def _upload_log(self, rel: str) -> str:
        """Read and publish exactly one bounded log in the storage worker."""
        return self._artifacts.put_bytes(self._read_log(rel))

    async def _finish(
        self, output: ToolOutput, times: tuple[datetime, datetime] | None
    ) -> ToolOutput:
        """Await each object separately so abandonment cannot publish the next one."""
        update: dict[str, Any] = {}
        try:
            paths = await _validated_io(self._output_paths, output.files)
            files: list[FileRef] = []
            left = self._max_output
            for path in paths:
                ref = await _storage_io(self._upload_file, path, left)
                if isinstance(ref, str):
                    raise ValueError(ref)
                files.append(ref)
                left -= ref.size
            update["files"] = files
        except ValueError as exc:
            await anyio.to_thread.run_sync(
                note_in_log, self._stderr, f"output files not uploaded: {exc}"
            )
            update |= {"status": RunStatus.ERROR, "files": []}
        # stdout first: a note about it must reach stderr before stderr is read.
        hashes = []
        for rel in (_STDOUT, _STDERR):
            hashes.append(await _storage_io(self._upload_log, rel))
        update["logs_ref"] = format_logs_ref(hashes[0], hashes[1])
        if times is not None:
            update["started_at"], update["ended_at"] = times
        return output.model_copy(update=update)
