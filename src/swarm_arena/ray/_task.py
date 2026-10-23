"""Run one framed job in a one-shot Ray worker, retaining the shared job owner."""

import asyncio
import atexit
import contextlib
import dataclasses
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Annotated

import anyio
import ray
from pydantic import Field, field_validator

from swarm_arena._errors import _exception_type_name
from swarm_arena._exec import supervisor
from swarm_arena._exec.executor import Job
from swarm_arena._exec.job_steps import _JobRun, _Staged, live_request, prepare_job
from swarm_arena._exec.supervisor import ProcessSupervisor, RLimits, launch_argv
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.tooling import CompletedRun, RunStatus, ToolOutput
from swarm_arena._outcomes import _classify_outcome, _Outcome, _raise_terminal
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._tooling.function_tool import FunctionTool, import_root
from swarm_arena._tooling.output import detach_tool_output
from swarm_arena._tooling.result_file import note_in_log, open_log
from swarm_arena._tooling.tool import Tool
from swarm_arena.ray._files import MIB, FrameDiagnostic, MemoryStore, check_cap, frame, unframe


class TaskSettings(FrozenModel):
    """Configure the private worker transport and subprocess polling.

    Parameters
    ----------
    poll_s : float, default=0.2
        Finite positive subprocess polling interval.
    cap_bytes : int or None, default=None
        Object-store cap in whole MiB, from 1 through 1024; None uses the
        artifact backend directly.
    import_roots : tuple[str, ...], default=()
        Driver-resolved module roots, also present on the worker node.

    Examples
    --------
    >>> TaskSettings().poll_s
    0.2
    """

    poll_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.2
    cap_bytes: Annotated[int, Field(strict=True)] | None = None
    import_roots: tuple[str, ...] = ()

    @field_validator("cap_bytes")
    @classmethod
    def _cap(cls, value: int | None) -> int | None:
        if value is not None:
            check_cap({}, cap_bytes=value, what="inputs")
        return value


class _Payload(FrozenModel):
    """Keep job transport fields separate from worker settings and callbacks."""

    job: Job = Field(repr=False)
    settings: TaskSettings


def job_payload(
    job: Job, settings: TaskSettings, *, blobs: Mapping[str, bytes] | None = None
) -> bytes:
    """Frame a job without its callbacks, retaining only supplied secret values.

    Parameters
    ----------
    job : Job
        Driver-owned job; request edits serialize as stored references.
    settings : TaskSettings
        Worker settings; node/file secrets are not read here.
    blobs : Mapping[str, bytes] or None, default=None
        Verified staged input artifacts for object-store transport.

    Returns
    -------
    bytes
        Private payload. Never log, journal or place it in runtime_env.

    Examples
    --------
    >>> from swarm_arena import Resources
    >>> from swarm_arena.tool import Command
    >>> job = Job(id='x', kind='command', command=Command(argv=['true']),
    ...     stage=Job.Stage(), timeout_s=None, grace_s=1, resources=Resources(),
    ...     env={}, secret_env=(), secrets_file=None, artifact_root='artifacts')
    >>> read_payload(job_payload(job, TaskSettings()))[0].id
    'x'
    """
    try:
        head = (
            _Payload(job=job, settings=settings)
            .model_dump_json(exclude={"job": {"on_started", "on_partial"}})
            .encode()
        )
        if settings.cap_bytes is not None:
            check_cap(blobs or {}, cap_bytes=settings.cap_bytes, what="inputs")
        return frame(head, blobs or {})
    except (ValueError, TypeError, RecursionError):
        raise ValueError("invalid worker payload") from None


def read_payload(payload: bytes) -> tuple[Job, TaskSettings, dict[str, bytes]]:
    """Decode and validate a job frame without importing connector code.

    Parameters
    ----------
    payload : bytes
        Framed job, settings and optional input artifacts.

    Returns
    -------
    tuple[Job, TaskSettings, dict[str, bytes]]
        Job with no callbacks, settings and verified input blobs.

    Raises
    ------
    ValueError
        If the frame or job is invalid; contents are never echoed.

    Examples
    --------
    >>> read_payload(b'invalid')
    Traceback (most recent call last):
        ...
    ValueError: invalid worker payload
    """
    try:
        decoded = unframe(payload)
        content = _Payload.model_validate_json(decoded.head)
        if decoded.diagnostics:
            raise ValueError("input diagnostics are not supported")
        if content.settings.cap_bytes is not None:
            check_cap(decoded.blobs, cap_bytes=content.settings.cap_bytes, what="inputs")
        return (
            dataclasses.replace(content.job, on_started=None, on_partial=None),
            content.settings,
            decoded.blobs,
        )
    except (ValueError, TypeError, RecursionError):
        raise ValueError("invalid worker payload") from None


def driver_import_roots(job: Job) -> tuple[str, ...]:
    """Resolve a FunctionTool's module root on the driver.

    Parameters
    ----------
    job : Job
        Job whose effective tool reference is already selected. This can run
        connector construction/import code; callers must supervise it in a daemon.

    Returns
    -------
    tuple[str, ...]
        Function module root, or empty for all other jobs.

    Examples
    --------
    >>> from swarm_arena import Resources
    >>> from swarm_arena.tool import Command
    >>> job = Job(id='x', kind='command', command=Command(argv=['true']),
    ...     stage=Job.Stage(), timeout_s=None, grace_s=1, resources=Resources(),
    ...     env={}, secret_env=(), secrets_file=None, artifact_root='artifacts')
    >>> driver_import_roots(job)
    ()
    """
    if job.kind == "tool" and job.tool is not None:
        tool = job.tool.build(Tool)
        if isinstance(tool, FunctionTool):
            return (import_root(tool.config.function.split(":", 1)[0]),)
    return ()


class _WorkerSupervisor(ProcessSupervisor):
    """Adapt only process mechanics; the shared job owner still collects/publishes."""

    def __init__(self, poll_s: float) -> None:
        self.poll_s = poll_s

    async def _wait(self, process: subprocess.Popen[bytes], deadline: float | None) -> bool:
        """Poll the child so Python can deliver Ray's main-thread interrupt."""
        while True:
            await anyio.lowlevel.checkpoint()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return process.poll() is not None
            try:
                process.wait(
                    timeout=self.poll_s if remaining is None else min(self.poll_s, remaining)
                )
                return True
            except subprocess.TimeoutExpired:
                pass

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout: Path,
        stderr: Path,
        timeout_s: float | None,
        grace_s: float,
        rlimits: RLimits | None,
        on_started: Callable[[Job.Started], Awaitable[None]] | None,
        before_kill: Callable[[CompletedRun], Awaitable[None]] | None,
    ) -> CompletedRun:
        """Launch in the worker group; always reap before handing back an outcome."""
        fds: list[int] = []
        process: subprocess.Popen[bytes] | None = None
        started_at = utc_now()
        status = RunStatus.ERROR
        try:
            for path in (stdout, stderr):
                fds.append(open_log(path, truncate=True))
            r, w = os.pipe()
            fds.extend((r, w))
            process = subprocess.Popen(  # noqa: ASYNC220 -- worker owns synchronous signal delivery
                launch_argv(argv, go_fd=r, rlimits=rlimits),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=fds[0],
                stderr=fds[1],
                pass_fds=(r,),
            )
            start = time.monotonic()
            if on_started is not None:
                await on_started(Job.Started(kind="process", job_id=""))
            started_at = utc_now()
            os.close(r)
            fds.remove(r)
            try:
                with contextlib.suppress(BrokenPipeError):
                    os.write(w, b"go")
            finally:
                os.close(w)
                fds.remove(w)
            try:
                done = await self._wait(process, None if timeout_s is None else start + timeout_s)
                if done:
                    status = RunStatus.OK if process.returncode == 0 else RunStatus.ERROR
                    if (
                        rlimits is not None
                        and rlimits.memory_bytes is not None
                        and process.returncode
                        in {
                            -signal.SIGKILL,
                            -signal.SIGSEGV,
                        }
                    ):
                        status = RunStatus.OOM
                else:
                    status = RunStatus.TIMEOUT
            except anyio.get_cancelled_exc_class():
                with anyio.CancelScope(shield=True):
                    await self._stop_child(
                        process,
                        grace_s,
                        before_kill,
                        CompletedRun(
                            returncode=None,
                            status=RunStatus.CANCELLED,
                            stdout_path=stdout,
                            stderr_path=stderr,
                            started_at=started_at,
                            ended_at=utc_now(),
                        ),
                    )
                raise
            if status == RunStatus.TIMEOUT:
                with anyio.CancelScope(shield=True):
                    await self._stop_child(
                        process,
                        grace_s,
                        before_kill,
                        CompletedRun(
                            returncode=None,
                            status=status,
                            stdout_path=stdout,
                            stderr_path=stderr,
                            started_at=started_at,
                            ended_at=utc_now(),
                        ),
                    )
                # Deliver a cancel held by the stop shield before any publication.
                await anyio.lowlevel.checkpoint()
            return CompletedRun(
                returncode=process.returncode,
                status=status,
                stdout_path=stdout,
                stderr_path=stderr,
                started_at=started_at,
                ended_at=utc_now(),
            )
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
            for fd in fds:
                os.close(fd)

    async def _stop_child(
        self,
        process: subprocess.Popen[bytes],
        grace_s: float,
        before_kill: Callable[[CompletedRun], Awaitable[None]] | None,
        provisional: CompletedRun,
    ) -> None:
        """Give grace and one bounded daemon collect, then kill before logging."""
        note: str | None = None
        try:
            if process.poll() is None:
                process.terminate()
            await self._wait(process, time.monotonic() + grace_s)
            if before_kill is not None:
                limit = supervisor.BEFORE_KILL_TIMEOUT_S
                try:
                    with anyio.move_on_after(limit) as scope:
                        await before_kill(provisional)
                    if scope.cancelled_caught:
                        note = f"before_kill timed out after {limit:g} s"
                except Exception as error:
                    # The shared owner latches received controls and propagates
                    # them after this cleanup; never adopt an abandoned result.
                    note = f"before_kill failed: {_exception_type_name(error)}"
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        if note is not None:
            note_in_log(provisional.stderr_path, note)


class _WorkerRun(_JobRun):
    """Keep hook-raised process controls distinct from a real Ray interrupt."""

    def _collect_owned(self, run: CompletedRun) -> ToolOutput | None:
        try:
            try:
                return self._collect(run)
            except BaseException as error:
                classified = _classify_outcome(error)
                _raise_terminal(classified)
                note_in_log(run.stderr_path, f"collect failed: {_exception_type_name(error)}")
                return None
        finally:
            self.collect_finished.set()


class _Worker:
    """Retain workspace owners through preparation, collection and publication."""

    def __init__(
        self, job: Job, workspace: Path, artifacts: ArtifactStore, settings: TaskSettings
    ) -> None:
        self.job = dataclasses.replace(job, on_started=None, on_partial=self._partial)
        self.workspace = workspace
        self.artifacts = artifacts
        self.settings = settings
        self.prepared = threading.Event()
        self.job_run: _WorkerRun | None = None
        self.partial = ToolOutput(status=RunStatus.CANCELLED)

    async def _partial(self, output: ToolOutput) -> None:
        self.partial = output

    def _prepare(self) -> _Staged | ToolOutput:
        try:
            sys.path[:0] = list(self.settings.import_roots)
            if self.job.request is not None:
                try:
                    request = live_request(self.job.request, self.artifacts)
                except ValueError as error:
                    _raise_terminal(_classify_outcome(error))
                    return _error(str(error))
                self.job = dataclasses.replace(self.job, request=request)
            staged = prepare_job(self.job, self.workspace, self.artifacts)
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                staged.env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
            return staged
        except BaseException as error:
            _Outcome(capture_caller=False).contain(error, RuntimeError, "worker preparation")
        finally:
            self.prepared.set()

    @property
    def finished(self) -> bool:
        return self.prepared.is_set() and (
            self.job_run is None
            or (self.job_run.collect_finished.is_set() and self.job_run.publication_finished)
        )

    async def run(self) -> ToolOutput:
        outcome = _Outcome()
        try:
            staged = await run_sync_daemon(self._prepare)
            await anyio.lowlevel.checkpoint()
            if isinstance(staged, ToolOutput):
                return staged
            self.job_run = _WorkerRun(
                self.job,
                staged,
                self.workspace,
                self.artifacts,
                max_log_bytes=8 * MIB,
                max_output_bytes=256 * MIB,
            )
            if staged.fault is not None:
                return await self.job_run.fault_output(staged.fault)
            memory = self.job.resources.memory_gib
            return await self.job_run.run(
                _WorkerSupervisor(self.settings.poll_s),
                RLimits(memory_bytes=int(memory * 2**30) if memory else None),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            outcome.propagate_control(error)
            return _error(f"worker failed: {_exception_type_name(error)}")


def _error(message: str) -> ToolOutput:
    return ToolOutput(status=RunStatus.ERROR, data={"swarm_arena_error": message})


def _execute(worker: _Worker) -> ToolOutput:
    """Own main-thread SIGINT through loop shutdown, independent of asyncio's handler."""
    interrupted = False
    active: tuple[asyncio.AbstractEventLoop, anyio.CancelScope] | None = None
    result: ToolOutput | None = None

    def interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True
        owner = active
        if owner is not None:
            owner[0].call_soon_threadsafe(owner[1].cancel)

    async def run() -> ToolOutput:
        nonlocal active, result
        with anyio.CancelScope() as scope:
            active = (asyncio.get_running_loop(), scope)
            try:
                if interrupted:
                    scope.cancel()
                result = await worker.run()
            finally:
                active = None
        # An interrupt after completion cannot replace the accepted result.
        return worker.partial if result is None else result

    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        return anyio.run(run)
    finally:
        signal.signal(signal.SIGINT, previous)


def run_in_worker(
    job: Job,
    workspace: Path,
    artifacts: ArtifactStore,
    *,
    poll_s: float,
    import_roots: Sequence[str] = (),
) -> ToolOutput:
    """Run a job with Ray's subprocess mechanics without needing a cluster.

    Parameters
    ----------
    job : Job
        Tool or command job, with live edits or stored edit references.
    workspace : Path
        Private directory retained by the caller if a daemon is abandoned.
    artifacts : ArtifactStore
        Worker-accessible artifacts.
    poll_s : float
        Finite positive child polling interval (normally 0.2 seconds).
    import_roots : Sequence[str], default=()
        Module roots prepended to this one-shot worker's sys.path.

    Returns
    -------
    ToolOutput
        Validated result, including a once-collected timeout/cancel partial.

    Notes
    -----
    Uses its own asyncio loop and temporarily owns SIGINT; call from synchronous
    main-thread code. Soft cancellation returns only collected data, without
    file, log or report references or new uploads. Both cancellation and timeout
    preserve the shielded grace and once-only collection. Subprocesses stay
    in the worker's process group. Ray owns descendant cleanup when its
    one-shot worker exits; this function only terminates/reaps the direct child.
    The caller owns the workspace. ``run_job`` removes it only after all
    preparation, collection and publication workers have released it, or at
    one-shot worker exit if an abandoned daemon still owns it.

    Examples
    --------
    >>> output = run_in_worker(job, workspace, artifacts, poll_s=0.2)  # doctest: +SKIP
    """
    settings = TaskSettings(poll_s=poll_s, import_roots=tuple(import_roots))
    return _execute(_Worker(job, workspace, artifacts, settings))


_CAP_ERROR = "outputs exceed the Ray object_store_cap_mib cap; use an s3:// artifact_root"


class _OutputStore(MemoryStore):
    """Reject a distinct output before adding it to the resident blob set."""

    def __init__(self, blobs: Mapping[str, bytes], cap: int) -> None:
        super().__init__(blobs)
        self.cap = cap
        self.total = 0
        self.exceeded = False

    def create(self, key: str, data: bytes) -> str:
        with self._lock:
            if not self.exists(key):
                if self.total + len(data) > self.cap:
                    self.exceeded = True
                    raise ValueError(_CAP_ERROR)
                result = super().create(key, data)
                self.total += len(data)
                return result
            return super().create(key, data)


class _OutputArtifacts(ArtifactStore):
    """Bound the transient per-file buffer as well as the resident output set."""

    def __init__(self, store: _OutputStore) -> None:
        super().__init__(store)
        self.memory = store

    def _put_chunks(self, chunks: Iterable[bytes], *, max_bytes: int = 2**30) -> tuple[str, int]:
        def bounded() -> Iterator[bytes]:
            size = 0
            for chunk in chunks:
                size += len(chunk)
                if size > self.memory.cap:
                    self.memory.exceeded = True
                    raise ValueError(_CAP_ERROR)
                yield chunk

        return super()._put_chunks(bounded(), max_bytes=max_bytes)


@ray.remote(max_retries=0, max_calls=1)
def run_job(payload: bytes) -> bytes:
    """Execute a private job frame in a one-shot Ray worker.

    Parameters
    ----------
    payload : bytes
        Output of ``job_payload``; never logged or journaled.

    Returns
    -------
    bytes
        ToolOutput JSON, newly created blobs and registered collection diagnostics
        in separate frame sections. A transport cap failure returns no blobs.

    Examples
    --------
    >>> result = ray.get(run_job.remote(payload))  # doctest: +SKIP
    """
    try:
        return _run_frame(payload)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        classified = _classify_outcome(error)
        _raise_terminal(classified)
        output = _error(f"worker failed: {_exception_type_name(classified)}")
        return frame(output.model_dump_json().encode(), {})


def _run_frame(payload: bytes) -> bytes:
    """Own the temporary workspace across decode, execution and encoding."""
    job, settings, blobs = read_payload(payload)
    memory = None if settings.cap_bytes is None else _OutputStore(blobs, settings.cap_bytes)
    artifacts = (
        ArtifactStore(open_store(job.artifact_root)) if memory is None else _OutputArtifacts(memory)
    )
    workspace = Path(tempfile.mkdtemp(prefix="swarm-worker-")).resolve()
    worker = _Worker(job, workspace, artifacts, settings)
    raised: BaseException | None = None
    try:
        output = detach_tool_output(_execute(worker))
        created: dict[str, bytes] = {}
        if memory is not None and output.status != RunStatus.CANCELLED:
            created = memory.created()
            if memory.exceeded:
                output, created = _error(_CAP_ERROR), {}
            else:
                try:
                    check_cap(created, cap_bytes=memory.cap, what="outputs")
                except ValueError:
                    output, created = _error(_CAP_ERROR), {}
        return frame(
            output.model_dump_json().encode(),
            created,
            [FrameDiagnostic(code, count) for code, count in output._collection_diagnostics],
        )
    except BaseException as error:
        raised = error
        raise
    finally:
        if worker.finished:
            try:
                shutil.rmtree(workspace)
            except Exception:
                if raised is None:
                    raise
                # Cleanup must not replace an escaping ownership/control failure.
        else:
            atexit.register(shutil.rmtree, workspace, ignore_errors=True)
