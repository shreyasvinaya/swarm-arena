"""Shared staged preparation, one-shot collection and per-object publication."""

import contextlib
import os
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Generic, NoReturn, TypeVar

import anyio

from swarm_arena._errors import Fenced, _exception_type_name
from swarm_arena._exec import supervisor
from swarm_arena._exec.env import build_env
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local_workspace import _DIR_FLAGS
from swarm_arena._exec.supervisor import ProcessSupervisor, RLimits
from swarm_arena._exec.threads import _DaemonCall
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.tooling import (
    Command,
    CompletedRun,
    FileRef,
    RunStatus,
    ToolOutput,
    ToolRequest,
    format_logs_ref,
    is_stored_edit_ref,
)
from swarm_arena._outcomes import _classify_outcome, _Outcome, _propagate_control, _raise_terminal
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.io import _storage_io, _validated_io
from swarm_arena._tooling.materialize import (
    _open_output_file,
    _stage_layout,
    collect_output_files,
    materialize,
)
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
from swarm_arena._tooling.tool import Tool


def _cwd_fault(workspace: Path, rel: str) -> str | None:
    """Return a note when the command's cwd is not a real directory in the workspace."""
    directories: list[int] = []
    try:
        directories.append(os.open(workspace, _DIR_FLAGS))
        for part in PurePosixPath(rel).parts:
            directories.append(os.open(part, _DIR_FLAGS, dir_fd=directories[-1]))
    except OSError:
        return f"command cwd {rel} is not a directory in the workspace"
    finally:
        for directory in directories:
            os.close(directory)
    return None


@dataclass(frozen=True)
class _Staged:
    """A materialized workspace and the command to run in it, or why it cannot run."""

    tool: Tool | None
    request: ToolRequest | None
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    fault: str | None
    timeout_s: float | None
    grace_s: float


@dataclass
class _Preparation:
    """Own a workspace until all synchronous preparation has stopped using it."""

    job: Job
    workspace: Path
    artifacts: ArtifactStore
    finished: threading.Event = field(default_factory=threading.Event, init=False)

    def run(self) -> _Staged:
        """Prepare once, marking workspace access finished even on a user error."""
        try:
            return prepare_job(self.job, self.workspace, self.artifacts)
        finally:
            self.finished.set()


def prepare_job(job: Job, workspace: Path, artifacts: ArtifactStore) -> _Staged:
    """Materialize a job and resolve its command and effective limits.

    Parameters
    ----------
    job : Job
        Invocation carrying live edit contents, stage and upper bounds.
    workspace : Path
        Private workspace retained while preparation is in flight.
    artifacts : ArtifactStore
        Backend for staged input contents.

    Returns
    -------
    _Staged
        Tool, live request, resolved argv/cwd/env/fault and minimum limits.

    Examples
    --------
    >>> staged = prepare_job(job, workspace, artifacts)  # doctest: +SKIP
    """
    materialize(job.stage, workspace, artifacts)
    tool: Tool | None = None
    if job.kind == "tool" and job.tool is not None and job.request is not None:
        tool = job.tool.build(Tool)
        try:
            command = tool.command(job.request, workspace)
        except SystemExit:
            return _Staged(
                tool,
                job.request,
                [],
                workspace,
                {},
                "command failed: SystemExit",
                job.timeout_s,
                job.grace_s,
            )
    elif job.kind == "command" and job.command is not None:
        command = job.command
    else:  # Job.__post_init__ already refuses this
        raise ValueError("job has no tool or command for its kind")
    command = _detach_model(Command, command)
    limits = [value for value in (job.timeout_s, command.timeout_s) if value is not None]
    timeout_s = min(limits) if limits else None
    grace_s = min(job.grace_s, command.grace_s)
    argv = resolve_argv(command)
    cwd = workspace / command.cwd if command.cwd is not None else workspace
    fault = _cwd_fault(workspace, command.cwd) if command.cwd is not None else None
    env = build_env(job, command, workspace=workspace, node_env=os.environ)
    return _Staged(
        tool=tool,
        request=job.request,
        argv=argv,
        cwd=cwd,
        env=env,
        fault=fault,
        timeout_s=timeout_s,
        grace_s=grace_s,
    )


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


_T = TypeVar("_T")


class _WorkspaceCall(Generic[_T]):
    """Retain a publication workspace until its worker finishes or is revoked."""

    def __init__(self, operation: Callable[..., _T], args: tuple[Any, ...]) -> None:
        self.operation = operation
        self.args = args
        self.finished = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._revoked = False

    def run(self) -> _T:
        """Claim workspace access unless the awaiting owner has already left."""
        with self._lock:
            if self._revoked:
                raise RuntimeError("workspace operation abandoned")
            self._started = True
        try:
            return self.operation(*self.args)
        finally:
            self.finished.set()

    def abandon(self) -> None:
        """Revoke queued work; an already-running worker retains ownership."""
        with self._lock:
            if not self._started:
                self._revoked = True
                self.finished.set()


class _JobRun:
    """Supervise one staged job, keep its collected output, and upload its results."""

    def __init__(
        self,
        job: Job,
        staged: _Staged,
        workspace: Path,
        artifacts: ArtifactStore,
        *,
        max_log_bytes: int,
        max_output_bytes: int,
    ) -> None:
        self._job = job
        self._staged = staged
        self._workspace = workspace
        self._artifacts = artifacts
        self._max_log = max_log_bytes
        self._max_output = max_output_bytes
        self._stderr = workspace / _STDERR
        self._started = False  # the launcher reached the start handshake
        self._output: ToolOutput | None = None  # the latest collected output
        self._post_collect: _DaemonCall[ToolOutput | None] | None = None
        self._collect_deadline = 0.0
        self._collected = False  # the collect after a normal exit finished
        self._collect_error: BaseException | None = None
        self.collect_finished = threading.Event()
        self.collect_finished.set()  # no collecting worker owns the workspace yet
        self._publication: _WorkspaceCall[Any] | None = None
        self._run: CompletedRun | None = None  # what the supervisor returned
        self._provisional: CompletedRun | None = None  # the run a stop-path collect saw

    async def run(self, sup: ProcessSupervisor, rlimits: RLimits | None) -> ToolOutput:
        """Run the job; on a cancel after the start, deliver its output, then re-raise."""
        outcome = _Outcome()
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
            if self._collect_error is not None:
                # The process supervisor may log an ordinary callback exception.
                # Registered terminal collection failures still belong to the owner.
                _raise_terminal(_classify_outcome(self._collect_error))
            if run.status in _STOPPED:  # collect ran (or was given up) in before_kill
                output = await anyio.to_thread.run_sync(self._stopped_output, run.status)
            else:
                self._collect_deadline = anyio.current_time() + supervisor.BEFORE_KILL_TIMEOUT_S
                self._post_collect = self._start_collect(run)
                await self._wait_post_collect()
                output = self._output or ToolOutput(status=RunStatus.ERROR)
            return await self._finish(output, (run.started_at, run.ended_at))
        except (Exception, anyio.get_cancelled_exc_class()) as error:
            await self._raise_failure(error, outcome)

    async def _raise_failure(self, error: BaseException, outcome: _Outcome) -> NoReturn:
        """Give the caller its own cancellation before propagating a job failure."""
        try:
            # Supervisor cleanup can hide scope cancellation behind a shield.
            # Outside that shield, obtain the caller's cancellation itself rather
            # than letting classification manufacture one from a collector error.
            await anyio.lowlevel.checkpoint()
            raise error
        except anyio.get_cancelled_exc_class():
            if self._started:
                with anyio.CancelScope(shield=True):
                    try:
                        await self._deliver_partial()
                    except BaseException as late_error:
                        # Classify cleanup against the original awaiting snapshot,
                        # then re-raise the captured, scope-attributable cancellation.
                        with contextlib.suppress(anyio.get_cancelled_exc_class()):
                            outcome.classify(late_error)
            raise

    async def fault_output(self, reason: str) -> ToolOutput:
        """Upload an unstarted job's diagnostics under the caller's supervision."""
        await self._workspace_io(self._fault_logs, reason, validation=True)
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
        try:
            call = self._start_collect(provisional)
            self._output = await call.result()
        except anyio.get_cancelled_exc_class():
            raise
        except BaseException as error:
            # Only a hand-back actually received by this owner can survive
            # supervisor cleanup. Abandoned workers cannot set this latch.
            self._collect_error = error
            raise

    def _start_collect(self, run: CompletedRun) -> _DaemonCall[ToolOutput | None]:
        """Reserve workspace ownership, undoing it if no daemon can be started."""
        self.collect_finished.clear()
        try:
            return _DaemonCall(self._collect_owned, run)
        except BaseException:
            self.collect_finished.set()
            raise

    async def _deliver_partial(self) -> None:
        """Deliver collected data after cancellation without starting artifact uploads."""
        run = self._run
        try:
            if run is not None and run.status not in _STOPPED and not self._collected:
                # Rejoin the same collect, with its original deadline; never call it twice.
                self._provisional = run.model_copy(update={"status": RunStatus.CANCELLED})
                await self._wait_post_collect()
        finally:
            output = await anyio.to_thread.run_sync(self._stopped_output, RunStatus.CANCELLED)
            timed = run or self._provisional
            times = None if timed is None else (timed.started_at, timed.ended_at)
            # Preserve collected data, but never publish refs from an abandoned upload.
            update: dict[str, Any] = {"files": [], "logs_ref": None}
            if times is not None:
                update["started_at"], update["ended_at"] = times
            partial = output.model_copy(update=update)
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

    def _collect_owned(self, run: CompletedRun) -> ToolOutput | None:
        """Release workspace ownership only after collection and validation finish."""
        try:
            return self._collect(run)
        finally:
            self.collect_finished.set()

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
            _raise_terminal(_classify_outcome(error))
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
        except (Exception, SystemExit) as error:
            _raise_terminal(_classify_outcome(error))
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
        paths = output_paths(
            [ref.path for ref in files],
            effective_files=[*layout.files, *layout.metadata],
            directories=sorted(layout.directories),
        )
        listings: dict[Path, dict[str, list[str]]] = {}
        for path in paths:
            os.close(_open_output_file(self._workspace, path, listings))
        return paths

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
            _propagate_control(error)
            # collect_output_files has already replaced backend details with
            # a fixed diagnostic naming only the declared workspace path.
            return str(error)
        return ref

    def _upload_log(self, rel: str) -> str:
        """Read and publish exactly one bounded log in the storage worker."""
        return self._artifacts.put_bytes(self._read_log(rel))

    @property
    def publication_finished(self) -> bool:
        """Whether every publication worker has released workspace access."""
        return self._publication is None or self._publication.finished.is_set()

    async def _workspace_io(
        self, operation: Callable[..., _T], *args: Any, validation: bool = False
    ) -> _T:
        """Track one supervised workspace operation, revoking it if not yet started."""
        call = _WorkspaceCall(operation, args)
        self._publication = call
        try:
            if validation:
                return await _validated_io(call.run)
            return await _storage_io(call.run)
        finally:
            call.abandon()

    async def _finish(
        self, output: ToolOutput, times: tuple[datetime, datetime] | None
    ) -> ToolOutput:
        """Await each object separately so abandonment cannot publish the next one."""
        update: dict[str, Any] = {}
        try:
            paths = await self._workspace_io(self._output_paths, output.files, validation=True)
            files: list[FileRef] = []
            left = self._max_output
            for path in paths:
                ref = await self._workspace_io(self._upload_file, path, left)
                if isinstance(ref, str):
                    raise ValueError(ref)
                files.append(ref)
                left -= ref.size
            update["files"] = files
        except ValueError as exc:
            _propagate_control(exc)
            await anyio.to_thread.run_sync(
                note_in_log, self._stderr, f"output files not uploaded: {exc}"
            )
            update |= {"status": RunStatus.ERROR, "files": []}
        # stdout first: a note about it must reach stderr before stderr is read.
        hashes = []
        for rel in (_STDOUT, _STDERR):
            hashes.append(await self._workspace_io(self._upload_log, rel))
        update["logs_ref"] = format_logs_ref(hashes[0], hashes[1])
        if times is not None:
            update["started_at"], update["ended_at"] = times
        return output.model_copy(update=update)


def resolve_argv(command: Command) -> list[str]:
    """Resolve a leading Python executable without changing the command.

    Parameters
    ----------
    command : Command
        Validated invocation and optional interpreter override.

    Returns
    -------
    list[str]
        Fresh argv with a leading ``python`` resolved to the override or host.

    Examples
    --------
    >>> resolve_argv(Command(argv=["python", "x.py"], python="/opt/py"))
    ['/opt/py', 'x.py']
    """
    argv = list(command.argv)
    if argv[0] == "python":
        argv[0] = command.python or sys.executable
    return argv


def live_request(request: ToolRequest, artifacts: ArtifactStore) -> ToolRequest:
    """Restore transported edit references to strictly decoded live contents.

    Parameters
    ----------
    request : ToolRequest
        Transported request or already live request. Edit order is retained.
    artifacts : ArtifactStore
        Backend whose verified bytes supply reserved SHA-256 references.

    Returns
    -------
    ToolRequest
        Copy with live edits and every other field unchanged.

    Raises
    ------
    ValueError
        If a reference cannot be restored; diagnostics contain no stored values.

    Examples
    --------
    >>> from swarm_arena._store.base import open_store
    >>> request = ToolRequest(edits={"p.txt": "sha256:hello"})
    >>> live_request(request, ArtifactStore(open_store("unused"))).edits
    {'p.txt': 'sha256:hello'}
    """
    edits: dict[str, str] = {}
    for path, value in request.edits.items():
        if not is_stored_edit_ref(value):
            edits[path] = value
            continue
        try:
            edits[path] = artifacts.get_bytes(value[7:]).decode("utf-8")
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            _propagate_control(error)
            raise ValueError(
                f"edit {path!r}: stored reference could not be restored: "
                f"{_exception_type_name(error)}"
            ) from None
    return request.model_copy(update={"edits": edits})
