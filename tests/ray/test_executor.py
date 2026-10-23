# ruff: noqa: E402
# Check the optional runtime before importing its modules.
"""Executor policy tests need no cluster; marked cases exercise the shared head."""

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import anyio
import pytest
from _fakes.dependencies import ray_dependency
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import make_job

ray_dependency()

# isort: split
import ray
import ray.exceptions

from swarm_arena import Resources
from swarm_arena._errors import Fenced, StorageError, _core_error, _stop_cleanup_error
from swarm_arena._exec.executor import current_schedule_timeout, schedule_timeout_scope
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._tooling.output import collection_warnings
from swarm_arena.executor import Job
from swarm_arena.ray import RayExecutor
from swarm_arena.ray import _executor as module
from swarm_arena.ray._executor import fits_any_node, request_shape, task_options
from swarm_arena.ray._files import MAX_OBJECT_STORE_CAP_MIB, MIB, FrameDiagnostic, frame
from swarm_arena.ray._task import read_payload
from swarm_arena.tool import FileRef, RunStatus, ToolOutput


class _Ref:
    def __init__(self, result: bytes | BaseException | None) -> None:
        self.result = result
        self.waiting = anyio.Event()
        self.release = anyio.Event()
        if result is not None:
            self.release.set()

    async def wait(self) -> bytes:
        self.waiting.set()
        await self.release.wait()
        if isinstance(self.result, BaseException):
            raise self.result
        assert isinstance(self.result, bytes)
        return self.result

    def __await__(self) -> Generator[Any, None, bytes]:
        return self.wait().__await__()


class _RunJob:
    def __init__(self, ref: _Ref) -> None:
        self.ref = ref
        self.kw: dict[str, Any] = {}
        self.payloads: list[bytes] = []
        self.arguments: list[Any] = []

    def options(self, **kw: Any) -> "_RunJob":
        self.kw = kw
        return self

    def remote(self, payload: Any) -> _Ref:
        self.arguments.append(payload)
        self.payloads.append(payload.data if isinstance(payload, _Payload) else payload)
        return self.ref


class _Payload:
    def __init__(self, data: bytes) -> None:
        self.data = data


def _frame(output: ToolOutput | None = None, blobs: dict[str, bytes] | None = None) -> bytes:
    return frame((output or ToolOutput(status="ok")).model_dump_json().encode(), blobs or {})


@pytest.fixture
def fake_ray(monkeypatch: pytest.MonkeyPatch) -> _RunJob:
    run_job = _RunJob(_Ref(_frame()))
    connected = [False]

    def init(**kw: Any) -> Any:
        connected[0] = True
        return SimpleNamespace(dashboard_url="head:8265")

    monkeypatch.setattr(module, "_connection_address", None, raising=False)
    monkeypatch.setattr(ray, "is_initialized", lambda: connected[0])
    monkeypatch.setattr(module, "run_job", run_job)
    monkeypatch.setattr(ray, "init", Mock(side_effect=init))
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: "01000000", gcs_address="configured-head"),
    )
    monkeypatch.setattr(ray, "put", Mock(side_effect=_Payload))
    monkeypatch.setattr(ray, "nodes", lambda: [{"Alive": True, "Resources": {"CPU": 2.0}}])
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 2.0})
    monkeypatch.setattr(ray, "cancel", Mock())
    monkeypatch.setattr(module, "find_task", lambda *args, **kw: None)
    monkeypatch.setattr(module, "worker_dead", lambda *args, **kw: None)
    return run_job


def test_preflight_rejects_ray_client_address() -> None:
    with pytest.raises(ValueError, match="Ray Client"):
        RayExecutor(address="ray://head:10001").preflight()


def test_preflight_rejects_non_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module.os, "name", "nt")
    with pytest.raises(RuntimeError, match="POSIX"):
        RayExecutor().preflight()


@pytest.mark.parametrize("field", ["poll_s", "cancel_margin_s"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), "secret-config"])
def test_config_positive_finite(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field) as caught:
        RayExecutor(**{field: value})
    assert_secret_absent(caught.value, "secret-config")


@pytest.mark.parametrize("value", [0, -1, MAX_OBJECT_STORE_CAP_MIB + 1, 1.5, True, "secret-cap"])
def test_config_bounds_cap(value: Any) -> None:
    with pytest.raises(ValueError, match="object_store_cap_mib") as caught:
        RayExecutor(object_store_cap_mib=value)
    assert_secret_absent(caught.value, "secret-cap")


def test_task_options_maps_resources() -> None:
    job = make_job(resources=Resources(cpus=2, gpus=1, memory_gib=1.5, custom={"deepretro": 1}))
    assert task_options(job, RayExecutor.Config()) == {
        "name": job.id,
        "num_cpus": 2,
        "num_gpus": 1,
        "memory": 1_610_612_736,
        "resources": {"deepretro": 1},
    }


@pytest.mark.parametrize("source", ["node", "drop", "supplied", "env"])
def test_secret_resource_uses_only_remaining_node_secrets(source: str) -> None:
    kw: dict[str, Any] = {"secret_env": ("OPENAI_API_KEY",)}
    if source == "drop":
        kw["drop_env"] = ("OPENAI_API_KEY",)
    if source == "supplied":
        kw["secret_values"] = {"OPENAI_API_KEY": "sentinel-value"}
    if source == "env":
        kw["env"] = {"OPENAI_API_KEY": "sentinel-value"}
    options = task_options(make_job(**kw), RayExecutor.Config(secret_resource="llm_keys"))
    assert options["resources"] == ({"llm_keys": 1} if source == "node" else {})
    assert "runtime_env" not in options
    assert "sentinel-value" not in repr(options)


def test_request_shape_and_fits_any_node() -> None:
    assert request_shape({"num_cpus": 2, "num_gpus": 0, "memory": 12, "resources": {"x": 1}}) == {
        "CPU": 2,
        "memory": 12,
        "x": 1,
    }
    assert request_shape({}) == {}
    nodes = [{"CPU": 4.0, "memory": 8e9, "llm_keys": 1.0}]
    assert fits_any_node({"CPU": 4.0, "llm_keys": 1.0}, nodes)
    assert not fits_any_node({"CPU": 5.0}, nodes)
    assert not fits_any_node({"no_such_node": 1.0}, nodes)
    assert not fits_any_node({"CPU": 1.0}, [])


async def test_init_runs_once_in_worker_thread(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[bool] = []

    def init(**kw: Any) -> Any:
        assert kw == {"address": "configured-head", "ignore_reinit_error": True}
        threads.append(threading.current_thread() is threading.main_thread())
        return SimpleNamespace(dashboard_url="head:8265")

    monkeypatch.setattr(ray, "init", init)
    executor = RayExecutor(address="configured-head")
    async with anyio.create_task_group() as tasks:
        for _ in range(4):
            tasks.start_soon(executor.capacity)
    assert threads == [False]


async def test_dispatch_intent_before_submit(fake_ray: _RunJob, tmp_path: Path) -> None:
    seen: list[Job.Started] = []

    async def started(value: Job.Started) -> None:
        assert fake_ray.payloads == []
        seen.append(value)

    job = make_job(artifact_root=str(tmp_path), on_started=started)
    assert (await RayExecutor().submit(job)).status == RunStatus.OK
    assert seen == [
        Job.Started(kind="dispatch", job_id=job.id, task_name=job.id, ray_job_id="01000000")
    ]
    assert len(fake_ray.payloads) == 1
    assert read_payload(fake_ray.payloads[0])[0].on_started is None


async def test_job_without_on_started_is_submitted(fake_ray: _RunJob, tmp_path: Path) -> None:
    assert (
        await RayExecutor().submit(make_job(artifact_root=str(tmp_path)))
    ).status == RunStatus.OK
    assert len(fake_ray.payloads) == 1


@pytest.mark.parametrize("cancel", [False, True])
async def test_on_started_failure_submits_nothing(
    fake_ray: _RunJob, tmp_path: Path, cancel: bool
) -> None:
    with anyio.CancelScope() as scope:

        async def started(value: Job.Started) -> None:
            if cancel:
                scope.cancel()
            else:
                raise RuntimeError("commit failed")

        if cancel:
            await RayExecutor().submit(make_job(artifact_root=str(tmp_path), on_started=started))
        else:
            with pytest.raises(RuntimeError, match="commit failed"):
                await RayExecutor().submit(
                    make_job(artifact_root=str(tmp_path), on_started=started)
                )
    assert fake_ray.payloads == []


async def test_inputs_over_cap_never_dispatched(fake_ray: _RunJob, tmp_path: Path) -> None:
    artifacts = ArtifactStore(open_store(str(tmp_path)))
    digest = artifacts.put_bytes(b"x" * (2 * MIB))
    started: list[Job.Started] = []

    async def on_started(value: Job.Started) -> None:
        started.append(value)

    out = await RayExecutor(object_store_cap_mib=1).submit(
        make_job(
            stage=Job.Stage(root={"large": digest}),
            artifact_root=str(tmp_path),
            on_started=on_started,
        )
    )
    assert out.status == RunStatus.ERROR
    assert "object_store_cap_mib" in out.data["swarm_arena_error"]
    assert not fake_ray.payloads
    assert not started


async def test_output_blobs_and_diagnostics(fake_ray: _RunJob, tmp_path: Path) -> None:
    data = b"result"
    digest = sha256_hex(data)
    output = ToolOutput(
        status="ok", files=[FileRef(name="output", path="output", sha256=digest, size=len(data))]
    )
    fake_ray.ref.result = frame(
        output.model_dump_json().encode(),
        {digest: data},
        [
            FrameDiagnostic("llm_calls_symlink", 1),
            FrameDiagnostic("unknown_code", 1),
        ],
    )
    out = await RayExecutor().submit(make_job(artifact_root=str(tmp_path)))
    assert out.files == output.files
    assert ArtifactStore(open_store(str(tmp_path))).get_bytes(digest) == data
    assert collection_warnings(out) == [
        "swarm/llm_calls.jsonl has a symlink component; not collected"
    ]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ray.exceptions.OutOfMemoryError("secret-ray-error"), RunStatus.OOM),
        (ray.exceptions.RayError("secret-ray-error"), RunStatus.ERROR),
        (ValueError("secret-ray-error"), RunStatus.ERROR),
    ],
)
async def test_result_errors_are_safe_outputs(
    fake_ray: _RunJob, tmp_path: Path, error: Exception, status: RunStatus
) -> None:
    fake_ray.ref.result = error
    out = await RayExecutor().submit(make_job(artifact_root=str(tmp_path)))
    assert out.status == status
    assert type(error).__name__ in out.data["swarm_arena_error"]
    assert "secret-ray-error" not in repr(out)


@pytest.mark.parametrize("error", [_core_error(Fenced, "lost"), _stop_cleanup_error()])
@pytest.mark.parametrize("nested", [False, True])
async def test_result_controls_propagate_without_group(
    fake_ray: _RunJob, tmp_path: Path, error: Exception, nested: bool
) -> None:
    fake_ray.ref.result = ExceptionGroup("hook", [error]) if nested else error
    with pytest.raises(type(error)):
        await RayExecutor().submit(make_job(artifact_root=str(tmp_path)))


@pytest.mark.parametrize("mode", ["partial", "timeout", "soft_error", "ref_error", "bad_frame"])
async def test_cancel_sequence_and_data_only_partial(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    fake_ray.ref = _Ref(None)
    order: list[str] = []
    got: list[ToolOutput] = []
    digest = sha256_hex(b"unpublished")

    def cancel(ref: _Ref, *, force: bool = False) -> None:
        order.append("force" if force else "soft")
        if force or mode == "timeout":
            return
        if mode == "soft_error":
            raise ValueError("secret-cancel")
        ref.result = (
            ValueError("secret-ref")
            if mode == "ref_error"
            else b"invalid"
            if mode == "bad_frame"
            else _frame(
                ToolOutput(
                    status="cancelled",
                    partial=True,
                    data={"step": 1},
                    files=[FileRef(name="out", path="out", sha256=digest, size=11)],
                ),
                {digest: b"unpublished"},
            )
        )
        ref.release.set()

    async def partial(out: ToolOutput) -> None:
        order.append("partial")
        got.append(out)

    monkeypatch.setattr(ray, "cancel", cancel)
    with anyio.fail_after(60):
        async with anyio.create_task_group() as tasks:

            async def submit() -> None:
                with anyio.CancelScope() as scope:
                    scopes.append(scope)
                    await RayExecutor(cancel_margin_s=0.05).submit(
                        make_job(
                            artifact_root=str(tmp_path),
                            grace_s=0,
                            on_partial=partial,
                        )
                    )

            scopes: list[anyio.CancelScope] = []
            tasks.start_soon(submit)
            await fake_ray.ref.waiting.wait()
            scopes[0].cancel()
    assert order == ["soft", "force", "partial"]
    assert len(got) == 1
    assert got[0].status == RunStatus.CANCELLED
    assert got[0].partial is (mode == "partial")
    assert got[0].files == []
    assert got[0].llm_calls_file is None
    assert not [path async for path in anyio.Path(tmp_path).iterdir()]


async def test_schedule_timeout_cancels_without_partial(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ray.ref = _Ref(None)
    partial = Mock()
    monkeypatch.setattr(module, "_SCHEDULE_POLL_S", 0.01)
    with anyio.fail_after(60), schedule_timeout_scope(0.02):
        out = await RayExecutor(cancel_margin_s=0.02).submit(
            make_job(
                artifact_root=str(tmp_path),
                grace_s=0,
                resources=Resources(custom={"missing": 1}),
                on_partial=partial,
            )
        )
    assert out.status == RunStatus.ERROR
    assert (
        out.data["swarm_arena_error"]
        == "no node met the resource request within schedule_timeout_s=0.02"
    )
    assert current_schedule_timeout() is None
    assert [call.kwargs.get("force", False) for call in cast(Mock, ray.cancel).call_args_list] == [
        False,
        True,
    ]
    partial.assert_not_called()


async def test_capacity_filters_internal_resources(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ray,
        "cluster_resources",
        lambda: {
            "CPU": 2.0,
            "GPU": 1.0,
            "memory": 2**30,
            "object_store_memory": 2**20,
            "node:127.0.0.1": 1.0,
            "accelerator_type:X": 1.0,
            "llm_keys": 1.0,
        },
    )
    assert await RayExecutor().capacity() == Resources(
        cpus=2, gpus=1, memory_gib=1, custom={"llm_keys": 1}
    )


@pytest.mark.parametrize("method", ["confirm_dead", "kill_orphan"])
async def test_recovery_uses_connected_dashboard_and_original_job(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    calls: list[dict[str, Any]] = []

    def dead(name: str, **kw: Any) -> bool:
        assert name == "old-dispatch"
        calls.append(kw)
        return True

    monkeypatch.setattr(module, "worker_dead", dead)
    started = Job.Started(
        kind="dispatch", job_id="old", task_name="old-dispatch", ray_job_id="02000000"
    )
    assert await getattr(RayExecutor(), method)(started) is True
    assert calls[0]["dashboard_url"] == "head:8265"
    assert calls[0]["expected_job_id"] == "02000000"
    assert isinstance(calls[0]["deadline"], float)
    cast(Mock, ray.cancel).assert_not_called()


@pytest.mark.parametrize("method", ["confirm_dead", "kill_orphan"])
@pytest.mark.parametrize("held", ["init", "state", "lock"])
async def test_recovery_allowance_includes_init_lock_and_rpc(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch, method: str, held: str
) -> None:
    workers: list[threading.Thread] = []
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def block(*args: Any, **kw: Any) -> Any:
        workers.append(threading.current_thread())
        entered.set()
        try:
            assert release.wait(300)
            return SimpleNamespace(dashboard_url="head:8265") if held == "init" else True
        finally:
            done.set()

    executor = RayExecutor(cancel_margin_s=0.05)
    if held == "init":
        monkeypatch.setattr(ray, "init", block)
    elif held == "state":
        monkeypatch.setattr(module, "worker_dead", block)
    else:
        await executor._init_lock.acquire()
    try:
        with anyio.fail_after(30):
            result = await getattr(executor, method)(
                Job.Started(kind="dispatch", job_id="x", task_name="x")
            )
        assert result is (None if method == "confirm_dead" else False)
        if held != "lock":
            assert entered.is_set()
    finally:
        release.set()
        try:
            if held == "lock":
                executor._init_lock.release()
            else:
                assert await anyio.to_thread.run_sync(done.wait, 60)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
            assert all(not worker.is_alive() for worker in workers)


async def test_aclose_does_not_shutdown(fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch) -> None:
    shutdown = Mock()
    monkeypatch.setattr(ray, "shutdown", shutdown)
    await RayExecutor().aclose()
    shutdown.assert_not_called()


async def test_schedule_clock_resets_when_request_becomes_feasible(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ray.ref = _Ref(None)
    samples = iter([(1.0, 2.0), (4.0, 4.0), (5.0, 2.0), (6.0, 2.0), (8.0, 2.0)])
    clock = [0.0]
    seen: list[float] = []

    def nodes() -> list[dict[str, Any]]:
        clock[0], cpus = next(samples)
        seen.append(clock[0])
        return [{"Alive": True, "Resources": {"CPU": cpus}}]

    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module, "_SCHEDULE_POLL_S", 0.0)
    monkeypatch.setattr(ray, "nodes", nodes)
    with anyio.fail_after(60), schedule_timeout_scope(2):
        out = await RayExecutor(cancel_margin_s=0.01).submit(
            make_job(
                artifact_root=str(tmp_path),
                resources=Resources(cpus=3),
                grace_s=0,
            )
        )
    assert out.status == RunStatus.ERROR
    assert seen == [1, 4, 5, 6, 8]


async def test_cancel_during_upload_starts_no_later_publication(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers: list[threading.Thread] = []
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    blobs = {sha256_hex(data): data for data in [b"first", b"second"]}
    fake_ray.ref.result = _frame(ToolOutput(status="ok", data={"kept": True}), blobs)
    writes: list[str] = []
    partials: list[ToolOutput] = []
    scopes: list[anyio.CancelScope] = []

    def put(artifacts: ArtifactStore, data: bytes) -> str:
        workers.append(threading.current_thread())
        writes.append(sha256_hex(data))
        entered.set()
        try:
            assert release.wait(300)
            return sha256_hex(data)
        finally:
            done.set()

    async def partial(out: ToolOutput) -> None:
        partials.append(out)

    async def submit() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await RayExecutor(cancel_margin_s=0.1).submit(
                make_job(
                    artifact_root=str(tmp_path),
                    on_partial=partial,
                )
            )

    monkeypatch.setattr(ArtifactStore, "put_bytes", put)
    try:
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(submit)
                assert await anyio.to_thread.run_sync(entered.wait, 60)
                scopes[0].cancel()
        assert len(writes) == 1
        assert len(partials) == 1
        assert partials[0].data == {"kept": True}
        assert partials[0].files == []
    finally:
        release.set()
        try:
            assert await anyio.to_thread.run_sync(done.wait, 60)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
            assert all(not worker.is_alive() for worker in workers)
    assert len(writes) == 1


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit, GeneratorExit])
async def test_preparation_process_controls_pass_through(
    fake_ray: _RunJob,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    error = failure("process-control")

    def fail(job: Job) -> tuple[str, ...]:
        raise error

    monkeypatch.setattr(module, "driver_import_roots", fail)
    with pytest.raises(failure) as caught:
        await RayExecutor().submit(make_job(artifact_root=str(tmp_path)))
    assert caught.value is error
    assert fake_ray.payloads == []
    cast(Mock, ray.cancel).assert_not_called()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("site", ["prepare", "state", "soft", "force", "upload"])
async def test_boundaries_preserve_terminal_controls(
    fake_ray: _RunJob,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    site: str,
) -> None:
    from swarm_arena._errors import _core_category, _SignalCategory

    terminal = _stop_cleanup_error()

    def fail(*args: Any, **kw: Any) -> Any:
        raise ExceptionGroup("hook", [terminal]) if nested else terminal

    if site == "prepare":
        monkeypatch.setattr(module, "driver_import_roots", fail)
    elif site == "state":
        monkeypatch.setattr(module, "worker_dead", fail)
    elif site == "upload":
        fake_ray.ref.result = _frame(blobs={sha256_hex(b"out"): b"out"})
        monkeypatch.setattr(ArtifactStore, "put_bytes", fail)
    else:
        fake_ray.ref = _Ref(None)
        monkeypatch.setattr(module, "_SCHEDULE_POLL_S", 0.001)

        def cancel(ref: _Ref, *, force: bool = False) -> None:
            if force == (site == "force"):
                fail()
            elif site == "force":
                ref.result = _frame()
                ref.release.set()

        monkeypatch.setattr(ray, "cancel", cancel)
    executor = RayExecutor(cancel_margin_s=0.01)
    operation = (
        executor.confirm_dead(Job.Started(kind="dispatch", job_id="x", task_name="x"))
        if site == "state"
        else executor.submit(
            make_job(
                artifact_root=str(tmp_path),
                grace_s=0,
                resources=Resources(custom={"missing": 1}),
            )
        )
    )
    with anyio.fail_after(60), schedule_timeout_scope(0.001), pytest.raises(RuntimeError) as caught:
        await operation
    assert _core_category(caught.value) is _SignalCategory.TERMINAL


@pytest.mark.parametrize("failure", [Fenced("secret-lookalike"), ValueError("secret-lookalike")])
async def test_ordinary_state_failures_are_unknown(
    fake_ray: _RunJob,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    def fail(*args: Any, **kw: Any) -> None:
        raise failure

    monkeypatch.setattr(module, "worker_dead", fail)
    assert (
        await RayExecutor().confirm_dead(Job.Started(kind="dispatch", job_id="x", task_name="x"))
        is None
    )


class _SpyRunJob:
    def __init__(self) -> None:
        self.options_seen: dict[str, Any] = {}
        self.payloads: list[ray.ObjectRef] = []

    def options(self, **kw: Any) -> "_SpyRunJob":
        self.options_seen = kw
        return self

    def remote(self, payload: ray.ObjectRef) -> Any:
        from swarm_arena.ray._task import run_job

        self.payloads.append(payload)
        return run_job.options(**self.options_seen).remote(payload)


@ray.remote(max_retries=0, max_calls=1)
def _crash_run_job(payload: bytes) -> bytes:
    os._exit(1)


@pytest.mark.ray
async def test_live_dispatch_intent_and_command_submission(
    ray_driver: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from _fakes.ray_executor import script_job

    from swarm_arena.tool import Command

    spy = _SpyRunJob()
    monkeypatch.setattr(module, "run_job", spy)
    seen: list[Job.Started] = []

    async def started(value: Job.Started) -> None:
        assert spy.payloads == []
        seen.append(value)

    executor = RayExecutor()
    with anyio.fail_after(60):
        out = await executor.submit(script_job(artifact_root=str(tmp_path), on_started=started))
        assert out.status == RunStatus.OK
        out = await executor.submit(
            make_job(
                id="executor-command",
                artifact_root=str(tmp_path),
                timeout_s=None,
                command=Command(
                    argv=[
                        sys.executable,
                        "-c",
                        "import json, os; "
                        "json.dump({}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))",
                    ]
                ),
            )
        )
        assert out.status == RunStatus.OK
    assert len(seen) == 1
    assert seen[0].ray_job_id == ray.get_runtime_context().get_job_id()
    assert executor._dashboard_url == ray_driver.dashboard_url
    assert len(spy.payloads) == 2
    assert all(isinstance(payload, ray.ObjectRef) for payload in spy.payloads)
    assert "runtime_env" not in spy.options_seen


@pytest.mark.ray
async def test_worker_crash_maps_to_error(
    ray_driver: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.ray_executor import script_job

    monkeypatch.setattr(module, "run_job", _crash_run_job)
    with anyio.fail_after(60):
        out = await RayExecutor().submit(script_job(artifact_root=str(tmp_path)))
    assert out.status == RunStatus.ERROR
    assert out.data["swarm_arena_error"].startswith("WorkerCrashedError")


@pytest.mark.ray
async def test_submit_roundtrips_files_through_object_store(
    ray_driver: Any, tmp_path: Path
) -> None:
    from _fakes.ray_executor import tool_job

    artifacts = ArtifactStore(open_store(str(tmp_path)))
    digest = await anyio.to_thread.run_sync(artifacts.put_bytes, b"input")
    code = (
        "import json, os; from pathlib import Path; "
        "assert Path('input.txt').read_bytes() == b'input'; "
        "Path('output.txt').write_bytes(b'result'); "
        "json.dump({'data': {}, 'files': ['output.txt']}, "
        "open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
    )
    with anyio.fail_after(60):
        out = await RayExecutor().submit(
            tool_job(code, artifact_root=str(tmp_path), stage=Job.Stage(root={"input.txt": digest}))
        )
    assert out.status == RunStatus.OK
    assert await anyio.to_thread.run_sync(artifacts.get_bytes, out.files[0].sha256) == b"result"


@pytest.mark.ray
async def test_core_built_function_job_imports_on_worker(
    ray_driver: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.tools import _write_module

    from swarm_arena import FunctionTool
    from swarm_arena.testing import check_tool

    await anyio.to_thread.run_sync(_write_module, tmp_path / "mods" / "modx_c.py")
    monkeypatch.syspath_prepend(str(tmp_path / "mods"))
    with anyio.fail_after(60):
        out = await check_tool(FunctionTool("modx_c:f"), {"a": 1}, executor=RayExecutor())
    assert out.status == RunStatus.OK
    assert out.data == {"a": 1}


@pytest.mark.ray
async def test_secret_values_never_in_task_metadata(ray_driver: Any, tmp_path: Path) -> None:
    import ray.util.state
    from _fakes.ray_executor import script_job

    sentinel = "sk-virt-executor-sentinel"
    job = script_job(artifact_root=str(tmp_path), secret_values={"OPENAI_API_KEY": sentinel})
    executor = RayExecutor()
    with anyio.fail_after(60):
        assert (await executor.submit(job)).status == RunStatus.OK
        while True:
            rows = await anyio.to_thread.run_sync(
                lambda: ray.util.state.list_tasks(
                    address="http://" + ray_driver.dashboard_url,
                    filters=[("name", "=", job.id)],
                    detail=True,
                    timeout=60,
                )
            )
            if rows:
                break
            await anyio.sleep(0.1)
    assert sentinel not in repr(rows)


@pytest.mark.ray
async def test_capacity_reports_cluster(ray_driver: Any) -> None:
    cap = await RayExecutor().capacity()
    assert cap is not None
    assert cap.cpus == 2.0
    assert cap.custom == {"llm_keys": 1.0}


@pytest.mark.ray
async def test_confirm_dead_dispatch_after_finish(ray_driver: Any, tmp_path: Path) -> None:
    from _fakes.ray_executor import script_job

    executor = RayExecutor()
    job = script_job(artifact_root=str(tmp_path))
    with anyio.fail_after(60):
        await executor.submit(job)
        started = Job.Started(
            kind="dispatch", job_id=job.id, task_name=job.id, ray_job_id=executor._driver_job_id
        )
        while await executor.confirm_dead(started) is not True:  # noqa: ASYNC110 -- external state
            await anyio.sleep(0.1)
    assert (
        await executor.confirm_dead(Job.Started(kind="process", job_id="x", pid=1, pgid=1)) is None
    )


@pytest.mark.ray
async def test_kill_orphan_bounded_for_unknown_task(ray_driver: Any) -> None:
    from uuid import uuid4

    with anyio.fail_after(30):
        assert (
            await RayExecutor(cancel_margin_s=0.2).kill_orphan(
                Job.Started(
                    kind="dispatch",
                    job_id="unknown",
                    task_name=f"unknown-{uuid4().hex}",
                )
            )
            is False
        )


@pytest.mark.ray
async def test_schedule_timeout_fails_run(ray_driver: Any, tmp_path: Path) -> None:
    from _fakes.ray_executor import script_job

    with anyio.fail_after(30), schedule_timeout_scope(1.0):
        out = await RayExecutor(cancel_margin_s=1).submit(
            script_job(
                artifact_root=str(tmp_path),
                grace_s=0.5,
                resources=Resources(custom={"no_such_node": 1}),
            )
        )
    assert out.status == RunStatus.ERROR
    assert "schedule_timeout_s=1" in out.data["swarm_arena_error"]


@pytest.mark.ray
async def test_cancel_while_pending_never_runs_tool(
    ray_driver: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.ray_executor import script_job, tool_job

    from swarm_arena.ray._state import PENDING_STATES, TERMINAL_STATES, find_task

    marker = tmp_path / "ran"
    job = tool_job(
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
        resources=Resources(custom={"no_such_node": 1}),
        artifact_root=str(tmp_path / "pending"),
        grace_s=1,
    )
    executor = RayExecutor(cancel_margin_s=10)
    cancel = Mock(wraps=ray.cancel)
    monkeypatch.setattr(ray, "cancel", cancel)
    scopes: list[anyio.CancelScope] = []
    outputs: list[ToolOutput] = []
    finished = anyio.Event()

    async def submit() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            outputs.append(await executor.submit(job))
        finished.set()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(submit)
        try:
            with anyio.fail_after(60):
                while True:
                    pending = await anyio.to_thread.run_sync(
                        lambda: find_task(
                            job.id,
                            dashboard_url=ray_driver.dashboard_url,
                            expected_job_id=executor._driver_job_id,
                        )
                    )
                    if pending is not None and pending.state in PENDING_STATES:
                        break
                    await anyio.sleep(0.1)
            assert not finished.is_set()
            assert not await anyio.Path(marker).exists()
            # The cancel sequence proves the policy; this is a CI safety bound.
            with anyio.fail_after(60):
                scopes[0].cancel()
                await finished.wait()
        finally:
            tasks.cancel_scope.cancel()

    assert scopes[0].cancelled_caught
    assert outputs == []
    assert [call.kwargs.get("force", False) for call in cancel.call_args_list] == [False, True]
    assert cancel.call_args_list[0].args == cancel.call_args_list[1].args
    with anyio.fail_after(60):
        while True:
            terminal = await anyio.to_thread.run_sync(
                lambda: find_task(
                    job.id,
                    dashboard_url=ray_driver.dashboard_url,
                    expected_job_id=executor._driver_job_id,
                )
            )
            if terminal is not None and terminal.state in TERMINAL_STATES:
                break
            await anyio.sleep(0.1)
    assert terminal.task_id == pending.task_id
    assert terminal.attempt_number == pending.attempt_number
    assert not await anyio.Path(marker).exists()
    with anyio.fail_after(60):
        assert (
            await executor.submit(script_job(artifact_root=str(tmp_path / "later")))
        ).status == RunStatus.OK
    assert not await anyio.Path(marker).exists()


@pytest.mark.ray
async def test_cancel_partial_matches_local_executor(ray_driver: Any, tmp_path: Path) -> None:
    from _fakes.ray_executor import cancel_and_capture, partial_job

    from swarm_arena.executor import LocalExecutor

    local, _ = await cancel_and_capture(
        LocalExecutor(), partial_job(tmp_path / "local"), ready=tmp_path / "local"
    )
    remote, _ = await cancel_and_capture(
        RayExecutor(), partial_job(tmp_path / "remote"), ready=tmp_path / "remote"
    )
    assert (
        (local.status, local.partial)
        == (remote.status, remote.partial)
        == (RunStatus.CANCELLED, True)
    )
    assert local.data["step"] >= 1
    assert remote.data["step"] >= 1


@pytest.mark.ray
@pytest.mark.parametrize("cancel_pending", [False, True])
async def test_feasible_queue_and_pending_cancel(
    ray_driver: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_pending: bool,
) -> None:
    import time

    from _fakes.proc import wait_for_file
    from _fakes.ray_executor import script_job, tool_job

    from swarm_arena.ray._state import PENDING_STATES, TERMINAL_STATES, find_task

    ready, release, marker = (tmp_path / name for name in ("ready", "release", "ran"))
    hold_code = (
        "import json, os, time; from pathlib import Path; "
        f"Path({str(ready)!r}).touch()\n"
        "deadline = time.monotonic() + 300\n"
        f"while not Path({str(release)!r}).exists():\n"
        "    assert time.monotonic() < deadline, 'tool was not released'\n"
        "    time.sleep(0.05)\n"
        "json.dump({}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
    )
    holding = tool_job(hold_code, resources=Resources(cpus=2), artifact_root=str(tmp_path / "a"))
    queued = (
        script_job(artifact_root=str(tmp_path / "b"))
        if not cancel_pending
        else tool_job(
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
            artifact_root=str(tmp_path / "b"),
            grace_s=0.5,
        )
    )
    executor = RayExecutor(cancel_margin_s=1)
    seen_pending = anyio.Event()
    finished = anyio.Event()
    scopes: list[anyio.CancelScope] = []
    outputs: list[ToolOutput] = []
    queued_at = [0.0]

    def inspect(name: str, **kw: Any) -> Any:
        info = find_task(name, **kw)
        if (
            name == queued.id
            and info is not None
            and info.state in PENDING_STATES
            and time.monotonic() - queued_at[0] > 1.0
        ):
            anyio.from_thread.run_sync(seen_pending.set, token=token)
        return info

    token = anyio.lowlevel.current_token()
    monkeypatch.setattr(module, "find_task", inspect)

    async def submit_queued() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            outputs.append(await executor.submit(queued))
        finished.set()

    with anyio.fail_after(90), schedule_timeout_scope(0.5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(executor.submit, holding)
            try:
                await wait_for_file(ready, 60)
                queued_at[0] = time.monotonic()
                tasks.start_soon(submit_queued)
                await seen_pending.wait()
                assert not finished.is_set()
                if cancel_pending:
                    scopes[0].cancel()
                    await finished.wait()
                await anyio.Path(release).touch()
                await finished.wait()
            finally:
                await anyio.Path(release).touch()
                tasks.cancel_scope.cancel()
    if cancel_pending:
        with anyio.fail_after(30):
            while True:
                info = await anyio.to_thread.run_sync(
                    lambda: find_task(
                        queued.id,
                        dashboard_url=ray_driver.dashboard_url,
                        expected_job_id=executor._driver_job_id,
                    )
                )
                if info is not None and info.state in TERMINAL_STATES:
                    break
                await anyio.sleep(0.1)
        # An actual follow-up task crosses the released-capacity boundary.
        with anyio.fail_after(60):
            assert (
                await executor.submit(script_job(artifact_root=str(tmp_path / "c")))
            ).status == RunStatus.OK
        assert not await anyio.Path(marker).exists()
        assert outputs == []
    else:
        assert len(outputs) == 1
        assert outputs[0].status == RunStatus.OK


@pytest.mark.ray
@pytest.mark.s3
@pytest.mark.skipif(
    not os.environ.get("SWARM_ARENA_TEST_S3_ENDPOINT"),
    reason="needs a worker-reachable S3 endpoint before the session",
)
async def test_s3_artifact_root_sends_only_hashes(
    ray_driver: Any,
    s3_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _fakes.ray_executor import tool_job

    artifacts = ArtifactStore(open_store(s3_url))
    digest = await anyio.to_thread.run_sync(artifacts.put_bytes, b"x" * MIB)
    spy = _SpyRunJob()
    monkeypatch.setattr(module, "run_job", spy)
    code = (
        "import json, os; from pathlib import Path; "
        f"assert len(Path('input.txt').read_bytes()) == {MIB}; "
        "json.dump({'data': {'read': True}}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
    )
    with anyio.fail_after(60):
        out = await RayExecutor().submit(
            tool_job(code, artifact_root=s3_url, stage=Job.Stage(root={"input.txt": digest}))
        )
    assert out.status == RunStatus.OK
    payloads = await anyio.to_thread.run_sync(ray.get, spy.payloads)
    assert max(map(len, payloads)) < 64 * 1024
    assert read_payload(payloads[0])[2] == {}


@pytest.mark.parametrize("site", ["prepare", "ref"])
async def test_hook_cancellation_is_an_ordinary_failure(
    fake_ray: _RunJob,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
) -> None:
    error = asyncio.CancelledError("secret-forged-cancel")

    def fail(job: Job) -> tuple[str, ...]:
        raise error

    if site == "prepare":
        monkeypatch.setattr(module, "driver_import_roots", fail)
    else:
        fake_ray.ref.result = error
    with anyio.fail_after(60):
        out = await RayExecutor(cancel_margin_s=0.01).submit(make_job(artifact_root=str(tmp_path)))
    assert out.status == RunStatus.ERROR
    assert "CancelledError" in out.data["swarm_arena_error"]
    assert "secret-forged-cancel" not in repr(out)


@pytest.mark.parametrize("method", ["capacity", "confirm_dead", "kill_orphan"])
async def test_query_hook_cancellation_is_an_ordinary_failure(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError("secret-query-error")

    monkeypatch.setattr(ray, "cluster_resources", fail)
    monkeypatch.setattr(module, "worker_dead", fail)
    executor = RayExecutor()
    args = (
        () if method == "capacity" else (Job.Started(kind="dispatch", job_id="x", task_name="x"),)
    )
    result = await getattr(executor, method)(*args)
    assert result is (False if method == "kill_orphan" else None)


@pytest.mark.parametrize("method", ["capacity", "confirm_dead", "kill_orphan"])
async def test_query_preserves_caller_cancellation(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    workers: list[threading.Thread] = []
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def held(*args: Any, **kwargs: Any) -> Any:
        workers.append(threading.current_thread())
        entered.set()
        try:
            assert release.wait(300)
        finally:
            done.set()

    monkeypatch.setattr(ray, "cluster_resources", held)
    monkeypatch.setattr(module, "worker_dead", held)
    executor = RayExecutor()
    args = (
        () if method == "capacity" else (Job.Started(kind="dispatch", job_id="x", task_name="x"),)
    )
    returned = False

    async def query() -> None:
        nonlocal returned
        await getattr(executor, method)(*args)
        returned = True

    try:
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(query)
                assert await anyio.to_thread.run_sync(entered.wait, 60)
                tasks.cancel_scope.cancel()
        assert not returned
    finally:
        release.set()
        try:
            assert await anyio.to_thread.run_sync(done.wait, 60)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
            assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("site", ["prepare", "upload"])
async def test_core_storage_errors_propagate(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    seen: list[Job.Started] = []

    async def started(value: Job.Started) -> None:
        seen.append(value)

    def fail(*args: Any, **kw: Any) -> Any:
        if site == "prepare":
            raise StorageError("artifact input unavailable")
        raise OSError("private backend failure")

    if site == "prepare":
        monkeypatch.setattr(module, "inline_inputs", fail)
    else:
        fake_ray.ref.result = _frame(blobs={sha256_hex(b"result"): b"result"})
        monkeypatch.setattr(module, "_put_blob", fail)
    with pytest.raises(StorageError):
        await RayExecutor().submit(make_job(artifact_root=str(tmp_path), on_started=started))
    assert len(seen) == len(fake_ray.payloads) == (0 if site == "prepare" else 1)


@pytest.mark.parametrize("complete_before_rejoin", [False, True])
async def test_abandoned_init_rejoins_without_another_attempt(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete_before_rejoin: bool
) -> None:
    workers: list[threading.Thread] = []
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    calls: list[int] = []
    outputs: list[ToolOutput] = []
    executor = RayExecutor(cancel_margin_s=0.05)

    def init(**kw: Any) -> Any:
        workers.append(threading.current_thread())
        calls.append(threading.get_ident())
        entered.set()
        try:
            assert release.wait(300)
            return SimpleNamespace(dashboard_url="head:8265")
        finally:
            done.set()

    async def submit() -> None:
        outputs.append(await executor.submit(make_job(artifact_root=str(tmp_path))))

    monkeypatch.setattr(ray, "init", init)
    try:
        with anyio.fail_after(60):
            assert not await executor.kill_orphan(
                Job.Started(kind="dispatch", job_id="old", task_name="old")
            )
            assert entered.is_set()
            assert not fake_ray.payloads
            if complete_before_rejoin:
                release.set()
                assert await anyio.to_thread.run_sync(done.wait, 60)
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(submit)
                tasks.start_soon(submit)
                await anyio.lowlevel.checkpoint()
                release.set()
        assert len(outputs) == 2
        assert all(out.status == RunStatus.OK for out in outputs)
        assert len(calls) == 1
    finally:
        release.set()
        try:
            assert await anyio.to_thread.run_sync(done.wait, 60)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
            assert all(not worker.is_alive() for worker in workers)


async def test_failed_init_can_be_retried_by_a_later_caller(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def init(**kw: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("private connection failure")
        return SimpleNamespace(dashboard_url="head:8265")

    monkeypatch.setattr(ray, "init", init)
    executor = RayExecutor()
    with pytest.raises(RuntimeError) as caught:
        await executor.capacity()
    assert_secret_absent(caught.value, "private connection failure")
    assert (await executor.submit(make_job(artifact_root=str(tmp_path)))).status == RunStatus.OK
    assert calls == 2


@pytest.mark.parametrize("kind", ["local", "ray"])
@pytest.mark.parametrize(
    "error",
    [None, RuntimeError("partial failed"), _core_error(Fenced, "lost"), _stop_cleanup_error()],
)
async def test_partial_callback_error_matches_local_executor(
    fake_ray: _RunJob,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    error: Exception | None,
) -> None:
    import dataclasses

    from _fakes.proc import wait_for_file
    from _fakes.ray_executor import partial_job

    from swarm_arena.executor import LocalExecutor

    fake_ray.ref = _Ref(None)
    scopes: list[anyio.CancelScope] = []
    errors: list[Exception] = []
    delivered: list[ToolOutput] = []
    ready = tmp_path / "ready"
    executor = (
        LocalExecutor(workspace_root=str(tmp_path / "workspaces"))
        if kind == "local"
        else RayExecutor()
    )

    def cancel(ref: _Ref, *, force: bool = False) -> None:
        if not force:
            ref.result = _frame(ToolOutput(status="cancelled", partial=True, data={"step": 1}))
            ref.release.set()

    async def partial(out: ToolOutput) -> None:
        delivered.append(out)
        if error is not None:
            raise error

    async def submit() -> None:
        try:
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                await executor.submit(
                    dataclasses.replace(
                        partial_job(ready, artifact_root=str(tmp_path / "artifacts")),
                        on_partial=partial,
                    )
                )
        except Exception as caught:
            errors.append(caught)

    monkeypatch.setattr(ray, "cancel", cancel)
    with anyio.fail_after(90):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(submit)
            if kind == "local":
                await wait_for_file(ready, 60)
            else:
                await fake_ray.ref.waiting.wait()
            scopes[0].cancel()
    assert len(delivered) == 1
    assert delivered[0].status == RunStatus.CANCELLED
    assert delivered[0].partial
    assert delivered[0].data["step"] >= 1
    assert delivered[0].files == []
    assert delivered[0].logs_ref is None
    assert delivered[0].llm_calls_file is None
    assert not errors
    assert scopes[0].cancelled_caught


async def test_payload_is_put_off_loop_before_dispatch_intent(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    puts: list[_Payload] = []

    def put(data: bytes) -> _Payload:
        assert threading.current_thread() is not threading.main_thread()
        payload = _Payload(data)
        puts.append(payload)
        return payload

    async def started(value: Job.Started) -> None:
        assert len(puts) == 1
        assert not fake_ray.arguments

    monkeypatch.setattr(ray, "put", put)
    out = await RayExecutor().submit(make_job(artifact_root=str(tmp_path), on_started=started))
    assert out.status == RunStatus.OK
    assert fake_ray.arguments == puts


async def test_abandoned_payload_put_never_dispatches(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers: list[threading.Thread] = []
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    started = Mock()
    scopes: list[anyio.CancelScope] = []

    def put(data: bytes) -> _Payload:
        workers.append(threading.current_thread())
        entered.set()
        try:
            assert release.wait(300)
            return _Payload(data)
        finally:
            done.set()

    async def submit() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await RayExecutor().submit(make_job(artifact_root=str(tmp_path), on_started=started))

    monkeypatch.setattr(ray, "put", put)
    try:
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(submit)
                assert await anyio.to_thread.run_sync(entered.wait, 60)
                scopes[0].cancel()
        started.assert_not_called()
        assert not fake_ray.arguments
    finally:
        release.set()
        try:
            if entered.is_set():
                assert await anyio.to_thread.run_sync(done.wait, 60)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
            assert all(not worker.is_alive() for worker in workers)
    await anyio.lowlevel.checkpoint()
    assert not fake_ray.arguments


async def test_cancel_force_and_partial_share_one_deadline(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers: list[threading.Thread] = []
    fake_ray.ref = _Ref(None)
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    scopes: list[anyio.CancelScope] = []
    delivered: list[ToolOutput] = []
    clock = [100.0]
    allowances: list[float] = []
    loop = asyncio.get_running_loop()
    move_on_after = anyio.move_on_after

    def observed_allowance(delay: float, **kwargs: Any) -> Any:
        allowances.append(delay)
        return move_on_after(delay, **kwargs)

    class ClockedAnyIO(SimpleNamespace):
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

    monkeypatch.setattr(
        module,
        "anyio",
        ClockedAnyIO(current_time=lambda: clock[0], move_on_after=observed_allowance),
    )

    def cancel(ref: _Ref, *, force: bool = False) -> None:
        if force:
            workers.append(threading.current_thread())
            entered.set()
            clock[0] = 101.0
            try:
                assert release.wait(300)
            finally:
                done.set()
        else:
            clock[0] = 100.2
            ref.result = _frame()
            loop.call_soon_threadsafe(ref.release.set)

    async def partial(out: ToolOutput) -> None:
        delivered.append(out)
        await anyio.sleep_forever()

    async def submit() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await RayExecutor(cancel_margin_s=0.5).submit(
                make_job(artifact_root=str(tmp_path), grace_s=0.2, on_partial=partial)
            )

    monkeypatch.setattr(ray, "cancel", cancel)
    try:
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(submit)
                await fake_ray.ref.waiting.wait()
                allowances.clear()
                scopes[0].cancel()
        # Soft cancel, force and partial receive only the remainder of the same 0.7 s.
        assert allowances == pytest.approx([0.7, 0.5, 0.0])
        assert len(delivered) == 1
        assert not release.is_set()
        assert await anyio.to_thread.run_sync(entered.wait, 60)
    finally:
        release.set()
        try:
            if entered.is_set():
                assert await anyio.to_thread.run_sync(done.wait, 60)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
            assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit, GeneratorExit])
@pytest.mark.parametrize("site", ["wait", "decode", "cancel"])
async def test_dispatched_process_control_forces_synchronously(
    fake_ray: _RunJob,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
    site: str,
) -> None:
    error = failure("process control")
    forced: list[bool] = []

    async def wait(self: RayExecutor, *args: Any) -> Any:
        if site == "wait":
            raise error
        if site == "cancel":
            raise ValueError("transport failed")
        return _frame(), None

    def decode(*args: Any) -> Any:
        raise error

    def cancel(ref: _Ref, *, force: bool = False) -> None:
        if force:
            forced.append(threading.current_thread() is threading.main_thread())
        elif site == "cancel":
            raise error

    monkeypatch.setattr(RayExecutor, "_wait", wait)
    monkeypatch.setattr(ray, "cancel", cancel)
    if site == "decode":
        monkeypatch.setattr(module, "_decode", decode)
    with pytest.raises(failure) as caught:
        await RayExecutor().submit(make_job(artifact_root=str(tmp_path)))
    assert caught.value is error
    assert forced == [True]


async def test_running_task_stops_the_scheduling_watch(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeping = anyio.Event()
    deadlines: list[float] = []
    original_sleep = anyio.sleep_forever
    find = Mock(return_value=SimpleNamespace(state="RUNNING"))

    async def sleep_forever() -> None:
        deadlines.append(anyio.current_effective_deadline())
        sleeping.set()
        await original_sleep()

    monkeypatch.setattr(module, "find_task", find)
    monkeypatch.setattr(module, "_STATE_RPC_S", 0.03)
    monkeypatch.setattr(module.anyio, "sleep_forever", sleep_forever)
    with anyio.fail_after(60) as allowance:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(RayExecutor()._watch, "running", {"CPU": 1}, 10)
            await sleeping.wait()
            tasks.cancel_scope.cancel()
    assert deadlines == [allowance.deadline]
    assert find.call_count == 1


@pytest.mark.parametrize("held", [False, True])
async def test_pending_jobs_share_rate_limited_node_snapshot(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch, held: bool
) -> None:
    workers: list[threading.Thread] = []
    clock = [0.0]
    calls: list[float] = []
    counts = [0, 0]
    sleeping = [anyio.Event(), anyio.Event()]
    advance = [anyio.Event(), anyio.Event()]
    release = threading.Event()
    completed: list[threading.Event] = []
    original_sleep = anyio.sleep

    def nodes() -> list[dict[str, Any]]:
        workers.append(threading.current_thread())
        done = threading.Event()
        completed.append(done)
        calls.append(clock[0])
        try:
            if held:
                assert release.wait(300)
            return [{"Alive": True, "Resources": {"CPU": 2.0}}]
        finally:
            done.set()

    async def sleep(seconds: float) -> None:
        if seconds != 1.0:
            await original_sleep(seconds)
            return
        phase = int(clock[0] / 2)
        counts[phase] += 1
        if counts[phase] == 4:
            sleeping[phase].set()
        await advance[phase].wait()

    monkeypatch.setattr(ray, "nodes", nodes)
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module, "_STATE_RPC_S", 0.03)
    monkeypatch.setattr(module.anyio, "sleep", sleep)
    executor = RayExecutor()
    try:
        with anyio.fail_after(60):
            async with anyio.create_task_group() as tasks:
                for index in range(4):
                    tasks.start_soon(executor._watch, str(index), {"CPU": 1}, 10)
                await sleeping[0].wait()
                first_count = len(calls)
                clock[0] = 2.0
                advance[0].set()
                await sleeping[1].wait()
                tasks.cancel_scope.cancel()
        assert first_count == 1
        assert len(calls) == (1 if held else 2)
    finally:
        release.set()
        try:
            for done in completed:
                assert await anyio.to_thread.run_sync(done.wait, 60)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
            assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("address", ["localhost:6379", "127.0.0.1:6379", "auto"])
async def test_owned_connection_reuses_configured_address(
    fake_ray: _RunJob, tmp_path: Path, address: str
) -> None:
    job = make_job(artifact_root=str(tmp_path))
    # The fake runtime stores a different GCS spelling from either explicit input.
    for requested in (address, address, "auto"):
        assert (await RayExecutor(address=requested).submit(job)).status == RunStatus.OK
    assert len(fake_ray.arguments) == 3
    with pytest.raises(RuntimeError) as caught:
        await RayExecutor(address="private-other-head").submit(job)
    assert_secret_absent(caught.value, "private-other-head")
    assert_secret_absent(caught.value, address)
    assert len(fake_ray.arguments) == 3


async def test_foreign_connection_refuses_explicit_address_but_auto_reuses(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    job = make_job(artifact_root=str(tmp_path))
    for address in ("configured-head", "auto", "configured-head"):
        if address == "auto":
            assert (await RayExecutor().submit(job)).status == RunStatus.OK
        else:
            with pytest.raises(RuntimeError, match="Ray is already initialized in this process"):
                await RayExecutor(address=address).submit(job)
    assert len(fake_ray.arguments) == 1


async def test_disconnected_process_can_claim_a_new_address(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await RayExecutor(address="first-private-head").capacity()
    assert first is not None
    assert first.cpus == 2
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    second = await RayExecutor(address="second-private-head").capacity()
    assert second is not None
    assert second.cpus == 2


async def test_concurrent_executors_serialize_connection_ownership(
    fake_ray: _RunJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers: list[threading.Thread] = []
    entered, second_entered, release = threading.Event(), threading.Event(), threading.Event()
    connected = [False]
    calls: list[str] = []
    errors: list[Exception] = []
    connect = RayExecutor._connect

    def init(*, address: str, **kw: Any) -> Any:
        calls.append(address)
        entered.set()
        assert release.wait(300)
        connected[0] = True
        return SimpleNamespace(dashboard_url="head:8265")

    def connect_started(self: RayExecutor) -> tuple[str | None, str | None]:
        workers.append(threading.current_thread())
        if self.config.address == "second-private-head":
            second_entered.set()
        return connect(self)

    async def capacity(address: str) -> None:
        try:
            result = await RayExecutor(address=address).capacity()
            assert result is not None
            assert result.cpus == 2
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(ray, "init", init)
    monkeypatch.setattr(ray, "is_initialized", lambda: connected[0])
    monkeypatch.setattr(RayExecutor, "_connect", connect_started)
    try:
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(capacity, "configured-head")
                assert await anyio.to_thread.run_sync(entered.wait, 30)
                tasks.start_soon(capacity, "second-private-head")
                assert await anyio.to_thread.run_sync(second_entered.wait, 30)
                release.set()
        assert calls == ["configured-head"]
        assert len(errors) == 1
        assert_secret_absent(errors[0], "second-private-head")
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
        assert all(not worker.is_alive() for worker in workers)


@pytest.mark.ray
@pytest.mark.parametrize("foreign", [False, True])
def test_live_process_connection_ownership(ray_cluster: str, foreign: bool) -> None:
    # A separate driver avoids changing the shared ray_driver fixture's connection.
    script = textwrap.dedent("""\
        import os
        import sys
        import anyio
        import ray
        from swarm_arena.ray import RayExecutor

        async def main() -> None:
            # Exercise the typed spelling without the fixture's address override.
            os.environ.pop("RAY_ADDRESS", None)
            cluster, mode = sys.argv[1:]
            address = "localhost:" + cluster.rsplit(":", 1)[1]
            if mode == "foreign":
                ray.init(address=cluster)
                try:
                    await RayExecutor(address=cluster).capacity()
                except RuntimeError as error:
                    assert str(error) == "Ray is already initialized in this process"
                else:
                    raise AssertionError("foreign connection was claimed")
            else:
                for _ in range(2):
                    assert (await RayExecutor(address=address).capacity()).cpus == 2
                try:
                    await RayExecutor(address="other-private-head:1234").capacity()
                except RuntimeError as error:
                    assert "other-private-head" not in str(error)
                    assert address not in str(error)
                else:
                    raise AssertionError("a different address was silently ignored")
            assert (await RayExecutor().capacity()).cpus == 2
            print("ownership verified", flush=True)

        try:
            anyio.run(main)
        finally:
            ray.shutdown()
        """)
    result = subprocess.run(
        [sys.executable, "-c", script, ray_cluster, "foreign" if foreign else "owned"],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ownership verified" in result.stdout


async def test_schedule_cleanup_does_not_restart_after_a_force_control(
    fake_ray: _RunJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []

    async def wait(self: RayExecutor, *args: Any) -> Any:
        return None, ToolOutput(status=RunStatus.ERROR)

    def cancel(ref: _Ref, *, force: bool = False) -> None:
        calls.append(force)
        if force:
            raise _stop_cleanup_error()

    monkeypatch.setattr(RayExecutor, "_wait", wait)
    monkeypatch.setattr(ray, "cancel", cancel)
    with pytest.raises(RuntimeError):
        await RayExecutor().submit(make_job(artifact_root=str(tmp_path)))
    assert calls == [False, True]
