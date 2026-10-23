"""Dispatch one-shot Ray jobs with bounded recovery and owned artifact publication."""

import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from functools import partial
from threading import Lock
from time import monotonic
from typing import Annotated, Any, Final, cast

import anyio
import ray
import ray.exceptions
from pydantic import ConfigDict, Field

from swarm_arena._errors import StorageError, _core_error, _exception_type_name
from swarm_arena._exec.executor import Executor, Job, current_schedule_timeout
from swarm_arena._exec.threads import _DaemonCall, run_sync_daemon
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import _ConfigInputModel, config_or_fields
from swarm_arena._model.tooling import RunStatus, ToolOutput
from swarm_arena._outcomes import _classify_outcome, _Outcome, _propagate_control
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._store.io import _storage_io
from swarm_arena._tooling.output import with_collection_diagnostics
from swarm_arena.ray._files import (
    MAX_OBJECT_STORE_CAP_MIB,
    MIB,
    check_cap,
    inline_inputs,
    unframe,
    uses_object_store,
)
from swarm_arena.ray._state import PENDING_STATES, find_task, worker_dead
from swarm_arena.ray._task import TaskSettings, driver_import_roots, job_payload, run_job

_SCHEDULE_POLL_S: Final = 1.0
_STATE_RPC_S: Final = 5.0
_connection_lock = Lock()
_connection_address: str | None = None


def task_options(job: Job, config: "RayExecutor.Config") -> dict[str, Any]:
    """Map declared resources and remaining node-secret requirements to Ray options."""
    resources = dict(job.resources.custom)
    node_secrets = (
        set(job.secret_env) - set(job.drop_env) - job.secret_values.keys() - job.env.keys()
    )
    if config.secret_resource is not None and node_secrets:
        resources[config.secret_resource] = 1
    options: dict[str, Any] = {
        "name": job.id,
        "num_cpus": job.resources.cpus,
        "num_gpus": job.resources.gpus,
        "resources": resources,
    }
    if job.resources.memory_gib is not None:
        options["memory"] = int(job.resources.memory_gib * 1024**3)
    return options


def request_shape(options: Mapping[str, Any]) -> dict[str, float]:
    """Normalize nonzero scheduling quantities to node-resource names."""
    result = {
        target: float(options[source])
        for source, target in (("num_cpus", "CPU"), ("num_gpus", "GPU"), ("memory", "memory"))
        if options.get(source)
    }
    result.update(
        {key: float(value) for key, value in options.get("resources", {}).items() if value}
    )
    return result


def fits_any_node(request: Mapping[str, float], nodes: Sequence[Mapping[str, float]]) -> bool:
    """Test total capacity on an individual alive node, independent of occupancy."""
    return any(all(node.get(key, 0.0) >= value for key, value in request.items()) for node in nodes)


def _error(error: Exception) -> ToolOutput:
    status = (
        RunStatus.OOM if isinstance(error, ray.exceptions.OutOfMemoryError) else RunStatus.ERROR
    )
    return ToolOutput(status=status, data={"swarm_arena_error": _exception_type_name(error)})


def _prepare(job: Job, config: "RayExecutor.Config") -> ray.ObjectRef:
    """Finish reads, connector imports and serialization before dispatch intent."""
    cap = config.object_store_cap_mib * MIB if uses_object_store(job.artifact_root) else None
    blobs = None
    if cap is not None:
        try:
            blobs = inline_inputs(
                job.stage, ArtifactStore(open_store(job.artifact_root)), cap_bytes=cap
            )
        except ValueError as error:
            _propagate_control(error)
            # The codec's cap error includes sizes; expose only the field and remedy.
            raise _core_error(
                ValueError,
                "Ray inputs exceed object_store_cap_mib or are invalid; use an s3:// artifact_root",
            ) from None
    payload = job_payload(
        job,
        TaskSettings(poll_s=config.poll_s, cap_bytes=cap, import_roots=driver_import_roots(job)),
        blobs=blobs,
    )
    return ray.put(payload)


def _decode(raw: bytes, cap_bytes: int) -> tuple[ToolOutput, dict[str, bytes]]:
    """Revalidate the wire output and restore only registered collection diagnostics."""
    decoded = unframe(raw)
    check_cap(decoded.blobs, cap_bytes=cap_bytes, what="outputs")
    output = ToolOutput.model_validate_json(decoded.head)
    return with_collection_diagnostics(
        output, [(d.code, d.count) for d in decoded.diagnostics]
    ), decoded.blobs


def _partial(raw: bytes, cap_bytes: int) -> ToolOutput:
    """Keep data from a completed or stopped result without publishing abandoned refs."""
    output, _ = _decode(raw, cap_bytes)
    return output.model_copy(
        update={
            "status": RunStatus.CANCELLED,
            "partial": output.partial or bool(output.data or output.files),
            "files": [],
            "logs_ref": None,
            "llm_calls_file": None,
        }
    )


def _put_blob(artifacts: ArtifactStore, digest: str, data: bytes) -> None:
    """Publish exactly one verified blob; no subsequent write can start after abandonment."""
    if artifacts.put_bytes(data) != digest:
        raise ValueError("Ray output artifact hash mismatch")


class RayExecutor(Executor):
    """Run tool and command jobs on an existing Ray cluster.

    Construction performs no I/O. Each executor connects lazily, commits dispatch
    intent before submission and binds recovery queries to its connected dashboard.
    Closing an executor leaves the process-wide Ray connection available to others.

    Parameters
    ----------
    config : RayExecutor.Config or None, optional
        Validated settings; cannot be combined with keyword fields.
    **fields : Any
        Configuration fields when no explicit config is supplied.

    Examples
    --------
    >>> from swarm_arena.ray import RayExecutor
    >>> executor = RayExecutor(poll_s=0.1, secret_resource="llm_keys")
    >>> executor.config.object_store_cap_mib
    256
    """

    class Config(_ConfigInputModel):
        """Configure cluster connection, polling, cancellation and transport.

        Parameters
        ----------
        address : str, default='auto'
            Existing Ray cluster address. Ray Client addresses are refused.
        poll_s : float, default=0.2
            Finite positive worker subprocess polling interval in seconds.
        cancel_margin_s : float, default=10.0
            Finite positive extra soft-cancel wait and recovery operation allowance.
        object_store_cap_mib : int, default=256
            Maximum input or output blobs per job, from 1 through 1024 MiB.
        secret_resource : str or None, default=None
            Node resource requested when unresolved node/file secret names remain.

        Examples
        --------
        >>> from swarm_arena.ray import RayExecutor
        >>> RayExecutor.Config().address
        'auto'
        """

        model_config = ConfigDict(extra="forbid", frozen=True)
        address: str = "auto"
        poll_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.2
        cancel_margin_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 10.0
        object_store_cap_mib: Annotated[
            int, Field(strict=True, ge=1, le=MAX_OBJECT_STORE_CAP_MIB)
        ] = 256
        secret_resource: str | None = None

    def __init__(self, config: Config | None = None, /, **fields: Any) -> None:
        """Validate settings without connecting to Ray."""
        super().__init__(config_or_fields(config, fields))
        self._init_lock = anyio.Lock()
        self._init_call: _DaemonCall[tuple[str | None, str | None]] | None = None
        self._initialized = False
        self._dashboard_url: str | None = None
        self._driver_job_id: str | None = None
        self._nodes_lock = anyio.Lock()
        self._nodes_call: _DaemonCall[list[dict[str, Any]]] | None = None
        self._nodes_snapshot: list[Mapping[str, float]] | None = None
        self._nodes_refresh_at = float("-inf")

    @property
    def config(self) -> Config:
        """Return frozen validated settings.

        Returns
        -------
        RayExecutor.Config
            Settings supplied at construction.

        Examples
        --------
        >>> from swarm_arena.ray import RayExecutor
        >>> RayExecutor(cancel_margin_s=2).config.cancel_margin_s
        2.0
        """
        return cast(RayExecutor.Config, super().config)

    def preflight(self) -> None:
        """Validate platform and address without contacting the cluster.

        Raises
        ------
        RuntimeError
            On a non-POSIX platform.
        ValueError
            For a Ray Client address.

        Examples
        --------
        >>> from swarm_arena.ray import RayExecutor
        >>> try:
        ...     RayExecutor(address="ray://head:10001").preflight()
        ... except ValueError as error:
        ...     print(str(error))
        Ray Client (ray://) is not used for long runs; ...
        """
        if os.name != "posix":
            raise _core_error(RuntimeError, "RayExecutor requires POSIX")
        if self.config.address.startswith("ray://"):
            raise _core_error(
                ValueError,
                "Ray Client (ray://) is not used for long runs; "
                "run the driver as a Ray Job with swarm-arena submit",
            )

    def _connect(self) -> tuple[str | None, str | None]:
        global _connection_address
        # Serialize process-wide ownership even across distinct executor daemons.
        with _connection_lock:
            connected = ray.is_initialized()
            if not connected:
                _connection_address = None
            elif self.config.address != "auto":
                if _connection_address is None:
                    raise _core_error(RuntimeError, "Ray is already initialized in this process")
                if _connection_address != self.config.address:
                    raise _core_error(RuntimeError, "Ray connected cluster does not match address")
            context = ray.init(address=self.config.address, ignore_reinit_error=True)
            if not connected:
                # Keep the accepted spelling; Ray may normalize loopback addresses.
                _connection_address = self.config.address
            return context.dashboard_url, ray.get_runtime_context().get_job_id()

    async def _init(self) -> None:
        self.preflight()
        async with self._init_lock:
            if self._initialized:
                return
            if self._init_call is None:
                self._init_call = _DaemonCall(self._connect)
            # Abandon only this wait. Later callers rejoin the same attempt under
            # their own allowance; only an active caller may adopt its result.
            outcome = _Outcome()
            try:
                dashboard, job_id = await self._init_call.result()
                await anyio.lowlevel.checkpoint_if_cancelled()
            except anyio.get_cancelled_exc_class() as error:
                classified = outcome.classify(error)
                self._init_call = None
                outcome.contain(classified, RuntimeError, "Ray initialization")
            except Exception as error:
                self._init_call = None
                outcome.contain(error, RuntimeError, "Ray initialization")
            self._dashboard_url, self._driver_job_id = dashboard, job_id
            self._initialized = True
            self._init_call = None

    async def submit(self, job: Job) -> ToolOutput:
        """Dispatch a job once and return its validated, published result.

        Parameters
        ----------
        job : Job
            Staged tool or command, resources and local lifecycle callbacks.

        Returns
        -------
        ToolOutput
            Completed output, or a value-free failure without retrying the job.

        Notes
        -----
        Cancellation soft-cancels, waits, then force-cancels in a shield. These
        steps and ``job.on_partial`` share one deadline: grace plus the margin.
        Only that callback receives cancelled data; no artifact upload begins.
        Pending caller cancellation retains precedence over callback failures.
        Hard process controls synchronously request force cancellation before
        propagating. The private run scope bounds only infeasible requests.

        Examples
        --------
        >>> from swarm_arena.ray import RayExecutor
        >>> output = await RayExecutor().submit(job)  # doctest: +SKIP
        """
        await self._init()
        outcome = _Outcome()
        try:
            payload = await run_sync_daemon(_prepare, job, self.config)
            options = task_options(job, self.config)
        except anyio.get_cancelled_exc_class() as error:
            return _error(outcome.classify(error))
        except StorageError as error:
            outcome.propagate_control(error)
            raise
        except ValueError as error:
            outcome.propagate_control(error)
            return ToolOutput(
                status=RunStatus.ERROR,
                data={
                    "swarm_arena_error": (
                        "invalid Ray inputs or object_store_cap_mib; "
                        "use an s3:// artifact_root for large files"
                    )
                },
            )
        except Exception as error:
            outcome.propagate_control(error)
            return _error(error)
        if job.on_started is not None:
            await job.on_started(
                Job.Started(
                    kind="dispatch", job_id=job.id, task_name=job.id, ray_job_id=self._driver_job_id
                )
            )
        await anyio.lowlevel.checkpoint_if_cancelled()
        try:
            ref = run_job.options(**options).remote(payload)
        except Exception as error:
            outcome.propagate_control(error)
            return _error(error)
        try:
            return await self._submitted(ref, job, options)
        except (KeyboardInterrupt, SystemExit, GeneratorExit, BaseExceptionGroup):
            # Hard process controls cannot await cleanup. Queue the force request
            # on this thread before preserving the original process control.
            with suppress(Exception):
                ray.cancel(ref, force=True)
            raise

    async def _submitted(self, ref: Any, job: Job, options: Mapping[str, Any]) -> ToolOutput:
        """Own result publication and bounded cleanup after exactly one dispatch."""
        outcome = _Outcome()
        cleanup_started = False
        deadline: float | None = None
        try:
            raw, timed_out = await self._wait(ref, job, options)
            if timed_out is not None:
                cleanup_started = True
                deadline = anyio.current_time() + job.grace_s + self.config.cancel_margin_s
                await self._cancel(ref, job, deadline=deadline)
                return timed_out
            assert raw is not None
            output, blobs = await run_sync_daemon(
                _decode, raw, self.config.object_store_cap_mib * MIB
            )
            artifacts = ArtifactStore(open_store(job.artifact_root))
            for digest, data in blobs.items():
                await _storage_io(_put_blob, artifacts, digest, data, site="Ray output publication")
            await anyio.lowlevel.checkpoint_if_cancelled()
            return output
        except anyio.get_cancelled_exc_class() as error:
            if deadline is None:
                deadline = anyio.current_time() + job.grace_s + self.config.cancel_margin_s
            classified: Exception | None = None
            try:
                classified = outcome.classify(error)
            except anyio.get_cancelled_exc_class():
                pass
            with anyio.CancelScope(shield=True):
                try:
                    output = ToolOutput(status=RunStatus.CANCELLED)
                    if not cleanup_started:
                        cleanup_started = True
                        output = await self._cancel(ref, job, deadline=deadline)
                except Exception as cleanup_error:
                    if classified is not None:
                        outcome.propagate_control(cleanup_error)
                    # The original caller cancellation owns this outcome.
                    output = ToolOutput(status=RunStatus.CANCELLED)
                if classified is None and job.on_partial is not None:
                    try:
                        with anyio.move_on_after(max(0.0, deadline - anyio.current_time())):
                            await job.on_partial(output)
                    except (Exception, anyio.get_cancelled_exc_class()) as late_error:
                        # Caller cancellation wins; callback errors are dropped,
                        # as in LocalExecutor, through the shared outcome rule.
                        # Raise fresh cancellation outside this handler so the
                        # callback's exception context cannot change scope ownership.
                        with suppress(anyio.get_cancelled_exc_class()):
                            _classify_outcome(late_error, cancelled=True)
            if classified is not None:
                return _error(classified)
            raise anyio.get_cancelled_exc_class()() from None
        except Exception as error:
            cleanup_error: Exception | None = None
            if not cleanup_started:
                try:
                    await self._cancel(ref, job)
                except Exception as caught:
                    cleanup_error = caught
            outcome.propagate_control(error)
            if cleanup_error is not None:
                outcome.propagate_control(cleanup_error)
            if isinstance(error, StorageError):
                raise error from None
            return _error(error)

    async def _wait(
        self, ref: Any, job: Job, options: Mapping[str, Any]
    ) -> tuple[bytes | None, ToolOutput | None]:
        raw: bytes | None = None
        timed_out: ToolOutput | None = None
        error: Exception | None = None
        timeout = current_schedule_timeout()
        timeout = 600.0 if timeout is None else timeout
        async with anyio.create_task_group() as tasks:

            async def receive() -> None:
                nonlocal raw, error
                outcome = _Outcome()
                try:
                    raw = await ref
                except anyio.get_cancelled_exc_class() as caught:
                    error = outcome.classify(caught)
                except Exception as caught:
                    error = caught
                else:
                    tasks.cancel_scope.cancel()
                    return
                tasks.cancel_scope.cancel()

            async def watch() -> None:
                nonlocal timed_out, error
                try:
                    timed_out = await self._watch(job.id, request_shape(options), timeout)
                except Exception as caught:
                    error = caught
                tasks.cancel_scope.cancel()

            tasks.start_soon(receive)
            tasks.start_soon(watch)
        if error is not None:
            raise error
        return raw, timed_out

    async def _watch(
        self, name: str, request: Mapping[str, float], allowance_s: float
    ) -> ToolOutput:
        infeasible_since: float | None = None
        outcome = _Outcome()
        while True:
            now = monotonic()
            try:
                with anyio.move_on_after(_STATE_RPC_S) as poll:
                    info = await run_sync_daemon(
                        partial(
                            find_task,
                            name,
                            dashboard_url=self._dashboard_url,
                            expected_job_id=self._driver_job_id,
                            deadline=now + _STATE_RPC_S,
                        )
                    )
                    pending = info is None or info.state in PENDING_STATES
                    totals = await self._node_totals() if pending else None
                if not poll.cancelled_caught and not pending:
                    # One-shot tasks cannot return to pending. Stay dormant until
                    # receive finishes or the owner cancels, outside the RPC timer.
                    await anyio.sleep_forever()
                if poll.cancelled_caught or totals is None or fits_any_node(request, totals):
                    infeasible_since = None
                else:
                    observed = monotonic()
                    if infeasible_since is None:
                        infeasible_since = observed
                    if observed - infeasible_since >= allowance_s:
                        return ToolOutput(
                            status=RunStatus.ERROR,
                            data={
                                "swarm_arena_error": (
                                    "no node met the resource request within "
                                    f"schedule_timeout_s={allowance_s:g}"
                                )
                            },
                        )
            except anyio.get_cancelled_exc_class() as error:
                outcome.propagate_control(error)
                infeasible_since = None
            except Exception as error:
                outcome.propagate_control(error)
                infeasible_since = None
            await anyio.sleep(_SCHEDULE_POLL_S)

    async def _node_totals(self) -> list[Mapping[str, float]] | None:
        """Share one rate-limited, single-flight resource read across pending jobs."""
        async with self._nodes_lock:
            if self._nodes_call is None:
                if monotonic() < self._nodes_refresh_at:
                    return self._nodes_snapshot
                self._nodes_refresh_at = monotonic() + _SCHEDULE_POLL_S
                self._nodes_call = _DaemonCall(ray.nodes)
            outcome = _Outcome()
            try:
                nodes = await self._nodes_call.result()
                snapshot: list[Mapping[str, float]] = [
                    dict(node["Resources"]) for node in nodes if node["Alive"] is True
                ]
            except anyio.get_cancelled_exc_class() as error:
                # Genuine cancellation abandons only this waiter, never starts
                # another RPC while the shared call is still running.
                classified = outcome.classify(error)
                self._nodes_call = None
                self._nodes_snapshot = None
                raise classified from None
            except Exception:
                self._nodes_call = None
                self._nodes_snapshot = None
                raise
            self._nodes_call = None
            self._nodes_snapshot = snapshot
            return self._nodes_snapshot

    async def _cancel(self, ref: Any, job: Job, *, deadline: float | None = None) -> ToolOutput:
        if deadline is None:
            deadline = anyio.current_time() + job.grace_s + self.config.cancel_margin_s
        output = ToolOutput(status=RunStatus.CANCELLED)
        errors: list[Exception] = []
        with anyio.CancelScope(shield=True):
            outcome = _Outcome()
            process_control = False
            try:
                with anyio.move_on_after(max(0.0, deadline - anyio.current_time())):
                    await run_sync_daemon(ray.cancel, ref)
                    output = await run_sync_daemon(
                        _partial, await ref, self.config.object_store_cap_mib * MIB
                    )
            except anyio.get_cancelled_exc_class() as error:
                errors.append(outcome.classify(error))
            except Exception as error:
                errors.append(error)
            except (KeyboardInterrupt, SystemExit, GeneratorExit, BaseExceptionGroup):
                process_control = True
                raise
            finally:
                if not process_control:
                    try:
                        with anyio.move_on_after(max(0.0, deadline - anyio.current_time())):
                            await run_sync_daemon(partial(ray.cancel, ref, force=True))
                    except anyio.get_cancelled_exc_class() as error:
                        errors.append(outcome.classify(error))
                    except Exception as error:
                        errors.append(error)
            for error in errors:
                outcome.propagate_control(error)
        return output

    async def capacity(self) -> Resources | None:
        """Read the cluster's total logical resource capacity.

        Returns
        -------
        Resources or None
            CPU, GPU, memory and custom capacity; None on ordinary RPC failure.
            Ray node/accelerator labels and object-store memory are excluded.

        Examples
        --------
        >>> from swarm_arena.ray import RayExecutor
        >>> capacity = await RayExecutor().capacity()  # doctest: +SKIP
        """
        await self._init()
        outcome = _Outcome()
        try:
            with anyio.move_on_after(_STATE_RPC_S) as scope:
                values = await run_sync_daemon(ray.cluster_resources)
            if scope.cancelled_caught:
                return None
            return Resources(
                cpus=values.get("CPU", 0.0),
                gpus=values.get("GPU", 0.0),
                memory_gib=values["memory"] / 1024**3 if values.get("memory") else None,
                custom={
                    key: value
                    for key, value in values.items()
                    if key not in {"CPU", "GPU", "memory", "object_store_memory"}
                    and not key.startswith(("node:", "accelerator_type:"))
                },
            )
        except anyio.get_cancelled_exc_class() as error:
            outcome.propagate_control(error)
            return None
        except Exception as error:
            outcome.propagate_control(error)
            return None

    async def aclose(self) -> None:
        """Release executor ownership without shutting down shared Ray state.

        Returns
        -------
        None
            The shared Ray connection remains available to other callers.

        Examples
        --------
        >>> import anyio
        >>> from swarm_arena.ray import RayExecutor
        >>> anyio.run(RayExecutor().aclose)
        """

    async def confirm_dead(self, started: Job.Started) -> bool | None:
        """Confirm a dispatched worker's death using the original driver's identity.

        Parameters
        ----------
        started : Job.Started
            Recorded dispatch, including its original Ray job id when available.

        Returns
        -------
        bool or None
            True only with affirmative worker-death evidence; None for unknown
            evidence or a non-dispatch record. First connection and lock waits
            consume the same ``cancel_margin_s`` allowance as state queries.

        Examples
        --------
        >>> import anyio
        >>> from swarm_arena.executor import Job
        >>> from swarm_arena.ray import RayExecutor
        >>> anyio.run(RayExecutor().confirm_dead, Job.Started(kind="process", job_id="x")) is None
        True
        """
        return await self._recovery(started, poll=False)

    async def kill_orphan(self, started: Job.Started) -> bool:
        """Wait boundedly for Ray's dead-driver fate sharing; never kill directly.

        Parameters
        ----------
        started : Job.Started
            Recorded dispatch whose worker should have died with its driver.

        Returns
        -------
        bool
            True only when death is confirmed within ``cancel_margin_s``.

        Examples
        --------
        >>> import anyio
        >>> from swarm_arena.executor import Job
        >>> from swarm_arena.ray import RayExecutor
        >>> anyio.run(RayExecutor().kill_orphan, Job.Started(kind="process", job_id="x"))
        False
        """
        return await self._recovery(started, poll=True) is True

    async def _recovery(self, started: Job.Started, *, poll: bool) -> bool | None:
        if started.kind != "dispatch" or not started.task_name:
            return None
        deadline = monotonic() + self.config.cancel_margin_s
        outcome = _Outcome()
        try:
            with anyio.move_on_after(self.config.cancel_margin_s):
                await self._init()
                while True:
                    result = await run_sync_daemon(
                        partial(
                            worker_dead,
                            started.task_name,
                            dashboard_url=self._dashboard_url,
                            expected_job_id=started.ray_job_id,
                            deadline=deadline,
                        )
                    )
                    if monotonic() >= deadline:
                        return None
                    if result is True or not poll:
                        return result
                    await anyio.sleep(min(1.0, max(0.0, deadline - monotonic())))
        except anyio.get_cancelled_exc_class() as error:
            outcome.propagate_control(error)
        except Exception as error:
            outcome.propagate_control(error)
        return None
