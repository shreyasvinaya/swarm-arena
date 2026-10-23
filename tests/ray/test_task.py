# ruff: noqa: E402
# Check the optional runtime before importing its modules.
"""Worker mechanics run locally; only transport/lifetime tests need the shared head."""

import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from _fakes.dependencies import ray_dependency
from _fakes.proc import pid_alive, wait_until
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import RELEASE, WORKER_SCRIPTS, _write_module, make_job

from swarm_arena._exec import supervisor
from swarm_arena._model.ref import Ref
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._tooling.result_file import request_json
from swarm_arena._tooling.staging import edit_hashes, put_edits
from swarm_arena.executor import Job, LocalExecutor
from swarm_arena.tool import Command, CompletedRun, RunStatus, ToolOutput, ToolRequest

ray_dependency()

# isort: split
import ray

from swarm_arena.ray._files import MIB, MemoryStore, frame, inline_inputs, unframe
from swarm_arena.ray._task import (
    TaskSettings,
    _WorkerSupervisor,
    driver_import_roots,
    job_payload,
    read_payload,
    run_in_worker,
    run_job,
)


def _job(script: str = "report_env.py", *, args: list[str] | None = None, **kw: Any) -> Job:
    code = f"import sys; sys.argv = {([script] + (args or []))!r}\n" + WORKER_SCRIPTS[script]
    return _tool_job(code, **kw)


def _tool_job(code: str, *, tool: str = "PyTool", **kw: Any) -> Job:
    kw.setdefault("timeout_s", 60)
    return make_job(
        kind="tool",
        command=None,
        tool=Ref(path=f"_fakes.tools:{tool}", config={"code": code}),
        request=ToolRequest(),
        **kw,
    )


def _function_job(path: str, *, inputs: dict[str, Any] | None = None) -> Job:
    return make_job(
        kind="tool",
        command=None,
        tool=Ref(path="swarm_arena:FunctionTool", config={"function": path}),
        request=ToolRequest(inputs=inputs or {}),
        stage=Job.Stage(request_json=request_json(ToolRequest(inputs=inputs or {}))),
        timeout_s=60,
    )


def _run(job: Job, workspace: Path, artifacts: ArtifactStore | None = None) -> ToolOutput:
    return run_in_worker(job, workspace, artifacts or ArtifactStore(MemoryStore()), poll_s=0.2)


def _remote(job: Job, *, cap: int = 256 * MIB, blobs: dict[str, bytes] | None = None) -> bytes:
    return ray.get(
        run_job.remote(job_payload(job, TaskSettings(cap_bytes=cap), blobs=blobs)), timeout=60
    )


def _pid_written(path: Path) -> bool:
    """Check pid contents, not just file creation before write_text flushes."""
    return path.exists() and path.read_text().isdigit()


def _timeout_after_ready(monkeypatch: pytest.MonkeyPatch, ready: Callable[[], bool]) -> None:
    """Inject execution timeout only after readiness; keep the real grace wait."""
    original = _WorkerSupervisor._wait
    timed_out = False

    async def wait(
        self: _WorkerSupervisor, process: subprocess.Popen[bytes], deadline: float | None
    ) -> bool:
        nonlocal timed_out
        if not timed_out:
            assert deadline is not None
            assert await anyio.to_thread.run_sync(wait_until, ready, 60)
            assert process.poll() is None
            timed_out = True
            return False
        return await original(self, process, deadline)

    monkeypatch.setattr(_WorkerSupervisor, "_wait", wait)


def test_run_in_worker_ok(tmp_path: Path) -> None:
    assert _run(_job(), tmp_path).status == RunStatus.OK


def test_timeout_keeps_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready = tmp_path / "ready"
    _timeout_after_ready(monkeypatch, ready.exists)
    out = _run(
        _job("partial_on_sigterm.py", args=["--ready-file", str(ready)], grace_s=2), tmp_path
    )
    assert out.status == RunStatus.TIMEOUT
    assert out.partial
    assert out.data["step"] >= 1


def test_tool_ignoring_sigterm_killed_after_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = tmp_path / "pid"
    _timeout_after_ready(monkeypatch, lambda: _pid_written(pid))
    out = _run(_job("ignore_sigterm.py", args=["--pid-file", str(pid)], grace_s=0.2), tmp_path)
    assert out.status == RunStatus.TIMEOUT
    assert not pid_alive(int(pid.read_text()))


def test_blocked_collect_cannot_stop_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workers: list[threading.Thread] = []
    entered_at: list[float] = []
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.2)
    from _fakes.tools import BlockingCollectTool, PyTool

    from swarm_arena.ray._task import _execute, _Worker

    def collect(self: Any, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        workers.append(threading.current_thread())
        entered_at.append(time.monotonic())
        assert RELEASE.wait(300)
        return PyTool.collect(self, request, workspace, run)

    monkeypatch.setattr(BlockingCollectTool, "collect", collect)
    RELEASE.clear()
    pid = tmp_path / "pid"
    _timeout_after_ready(monkeypatch, lambda: _pid_written(pid))
    code = WORKER_SCRIPTS["ignore_sigterm.py"]
    code = f"import sys; sys.argv = ['tool', '--pid-file', {str(pid)!r}]\n" + code
    worker = _Worker(
        _tool_job(code, tool="BlockingCollectTool", grace_s=0.2),
        tmp_path,
        ArtifactStore(MemoryStore()),
        TaskSettings(),
    )
    try:
        out = _execute(worker)
        assert len(entered_at) == 1
        assert time.monotonic() - entered_at[0] < 30
        assert not RELEASE.is_set()
        assert worker.job_run is not None
        assert not worker.job_run.collect_finished.is_set()
        assert (out.status, out.partial) == (RunStatus.TIMEOUT, False)
        assert not pid_alive(int(pid.read_text()))
    finally:
        RELEASE.set()
        if worker.job_run is not None:
            assert worker.job_run.collect_finished.wait(60)
        for thread in workers:
            thread.join(60)
            assert not thread.is_alive()


def test_fifo_stderr_cannot_block_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid = tmp_path / "pid"
    _timeout_after_ready(monkeypatch, lambda: _pid_written(pid))
    out = _run(
        _job(
            "ignore_sigterm.py",
            args=["--pid-file", str(pid), "--fifo-stderr"],
            grace_s=0.2,
        ),
        tmp_path,
    )
    assert out.status == RunStatus.TIMEOUT
    assert not pid_alive(int(pid.read_text()))


def test_swarm_arena_env_and_cuda_reach_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    out = _run(_job(env={"SWARM_ARENA_CANDIDATE_ID": "c-x-e1-0001"}), tmp_path)
    assert out.data["env"]["SWARM_ARENA_CANDIDATE_ID"] == "c-x-e1-0001"
    assert out.data["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_secrets_read_on_worker_never_in_payload(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("SA_TEST_SECRET=from-file\n")
    job = _job(secret_env=("SA_TEST_SECRET",), secrets_file=str(secrets))
    assert _run(job, tmp_path).data["env"]["SA_TEST_SECRET"] == "from-file"
    assert b"from-file" not in job_payload(job, TaskSettings())
    supplied = dataclasses.replace(job, secret_values={"SA_TEST_SECRET": "driver-key"})
    restored, _, _ = read_payload(job_payload(supplied, TaskSettings()))
    assert restored.secret_values == {"SA_TEST_SECRET": "driver-key"}
    assert _run(restored, tmp_path).data["env"]["SA_TEST_SECRET"] == "driver-key"


def test_import_roots_make_function_importable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    root = tmp_path / "mods"
    _write_module(root / "modx_w.py")
    out = run_in_worker(
        _function_job("modx_w:f", inputs={"a": 1}),
        tmp_path / "ws",
        ArtifactStore(MemoryStore()),
        poll_s=0.2,
        import_roots=[str(root)],
    )
    assert out.status == RunStatus.OK
    assert out.data == {"a": 1}


def test_driver_import_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "mods"
    _write_module(root / "modx_d.py")
    monkeypatch.syspath_prepend(str(root))
    assert driver_import_roots(_function_job("modx_d:f")) == (str(root.resolve()),)
    assert driver_import_roots(_job()) == ()
    assert driver_import_roots(make_job()) == ()


def test_run_job_options() -> None:
    options = cast(Any, run_job)._default_options
    assert options["max_retries"] == 0
    assert options["max_calls"] == 1


@pytest.mark.ray
def test_soft_cancel_returns_partial_through_ray(ray_driver: None, tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    ref = run_job.remote(
        job_payload(
            _cancel_job(ready),
            TaskSettings(cap_bytes=256 * MIB),
        )
    )
    try:
        assert wait_until(ready.exists, 60)
        ray.cancel(ref)
        decoded = unframe(ray.get(ref, timeout=60))
        out = ToolOutput.model_validate_json(decoded.head)
        assert out.status == RunStatus.CANCELLED
        assert out.partial
        assert out.data["step"] >= 1
        assert out.files == []
        assert out.logs_ref is None
        assert out.llm_calls_file is None
        assert decoded.blobs == {}
    finally:
        ray.cancel(ref, force=True)


@pytest.mark.ray
def test_tool_in_worker_process_group(ray_driver: None) -> None:
    data = ToolOutput.model_validate_json(unframe(_remote(_job())).head).data
    assert data["pgid"] == data["parent_pgid"]


@pytest.mark.ray
def test_grandchild_cleaned_after_finish(ray_driver: None, tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    try:
        _remote(
            _job("spawn_grandchild.py", args=["--pid-file", str(pid_file), "--exit-after", "0.5"])
        )
        assert wait_until(lambda: not pid_alive(int(pid_file.read_text())), 60)
    finally:
        if pid_file.exists() and pid_alive(pid := int(pid_file.read_text())):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.ray
def test_output_over_cap_is_error(ray_driver: None) -> None:
    code = """import json, os
open('big.bin', 'wb').write(b'x' * (2 * 2**20))
json.dump({'data': {}, 'files': ['big.bin']}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))
"""
    decoded = unframe(_remote(_tool_job(code), cap=MIB))
    out = ToolOutput.model_validate_json(decoded.head)
    assert out.status == RunStatus.ERROR
    assert "object_store_cap_mib" in out.data["swarm_arena_error"]
    assert decoded.blobs == {}


async def _noop(value: Any) -> None:
    pass


def test_payload_has_no_callables() -> None:
    payload = job_payload(_job(on_started=_noop, on_partial=_noop), TaskSettings())
    head = json.loads(unframe(payload).head)
    assert "on_started" not in head["job"]
    assert "on_partial" not in head["job"]
    job, settings, blobs = read_payload(payload)
    assert job.on_started is None
    assert job.on_partial is None
    assert settings == TaskSettings()
    assert blobs == {}


EDITS = {"p.txt": "hello", "q.txt": "sha256:hello"}


def _edit_spy_job(
    tmp_path: Path, edits: dict[str, str], *, code: str = "pass", timeout_s: float | None = None
) -> tuple[Job, ArtifactStore]:
    artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))
    put_edits(edits, artifacts)
    job = _tool_job(
        code,
        tool="EditSpyTool",
        timeout_s=timeout_s,
        grace_s=0.5,
        artifact_root=str(tmp_path / "artifacts"),
    )
    return dataclasses.replace(
        job, request=ToolRequest(edits=edits), stage=Job.Stage(root=edit_hashes(edits))
    ), artifacts


@pytest.mark.ray
@pytest.mark.parametrize("object_store", [True, False])
@pytest.mark.parametrize(
    ("code", "timeout_s", "status"),
    [
        ("pass", None, RunStatus.OK),
        ("import time; time.sleep(60)", 1.0, RunStatus.TIMEOUT),
    ],
)
async def test_edit_contents_same_on_ray_as_local(
    ray_driver: None,
    local_executor: LocalExecutor,
    tmp_path: Path,
    object_store: bool,
    code: str,
    timeout_s: float | None,
    status: RunStatus,
) -> None:
    job, artifacts = _edit_spy_job(tmp_path, EDITS, code=code, timeout_s=timeout_s)
    blobs = inline_inputs(job.stage, artifacts, cap_bytes=256 * MIB) if object_store else None
    payload = job_payload(
        job, TaskSettings(cap_bytes=256 * MIB if object_store else None), blobs=blobs
    )
    sent, _, _ = read_payload(payload)
    assert sent.request is not None
    assert sent.request.edits == {p: f"sha256:{h}" for p, h in edit_hashes(EDITS).items()}
    ref = run_job.remote(payload)
    remote_bytes = await anyio.to_thread.run_sync(lambda: ray.get(ref, timeout=60))
    remote = ToolOutput.model_validate_json(unframe(remote_bytes).head)
    local = await local_executor.submit(job)
    assert remote.status == local.status == status
    assert remote.data == local.data == {"command_edits": EDITS, "collect_edits": EDITS}


def test_unrestorable_edit_reference_never_starts_tool(tmp_path: Path) -> None:
    job, _ = _edit_spy_job(tmp_path, {"p.txt": "hello"})
    sent, _, _ = read_payload(job_payload(job, TaskSettings()))
    out = _run(sent, tmp_path / "ws")
    assert out.status == RunStatus.ERROR
    assert out.data["swarm_arena_error"].startswith("edit 'p.txt': stored reference")
    assert not (tmp_path / "ws/edit_spy.json").exists()


@pytest.mark.parametrize("poll", [0, -1, float("nan"), float("inf")])
def test_invalid_poll_is_value_free(poll: float) -> None:
    with pytest.raises(ValueError, match="poll_s"):
        TaskSettings(poll_s=poll)


@pytest.mark.parametrize("cap", [0, MIB - 1, 1025 * MIB, True])
def test_invalid_cap_is_value_free(cap: Any) -> None:
    with pytest.raises(ValueError, match="cap_bytes"):
        TaskSettings(cap_bytes=cap)


def test_payload_errors_are_value_free() -> None:
    secret = "payload-SENTINEL-secret"
    with pytest.raises(ValueError, match="invalid worker payload") as caught:
        read_payload(frame(json.dumps({"job": secret}).encode(), {}))
    assert_secret_absent(caught.value, secret)


def test_effective_command_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready = tmp_path / "ready"
    _timeout_after_ready(monkeypatch, ready.exists)
    original = _WorkerSupervisor.run

    async def run(self: _WorkerSupervisor, *args: Any, **kwargs: Any) -> CompletedRun:
        assert kwargs["timeout_s"] == 1
        assert kwargs["grace_s"] == 0.2
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(_WorkerSupervisor, "run", run)
    job = _tool_job(WORKER_SCRIPTS["partial_on_sigterm.py"], timeout_s=60, grace_s=60)
    job = dataclasses.replace(
        job,
        tool=Ref(
            path="_fakes.tools:PyTool",
            config={
                "code": f"import sys; sys.argv = ['tool', '--ready-file', {str(ready)!r}]\n"
                + WORKER_SCRIPTS["partial_on_sigterm.py"],
                "timeout_s": 1,
                "grace_s": 0.2,
            },
        ),
    )
    out = _run(job, tmp_path)
    assert out.status == RunStatus.TIMEOUT
    assert out.partial


def test_normal_collect_is_bounded_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workers: list[threading.Thread] = []
    entered_at: list[float] = []
    from swarm_arena._exec.job_steps import _JobRun

    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.2)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls: list[bool] = []

    def collect(self: Any, run: Any) -> ToolOutput:
        workers.append(threading.current_thread())
        entered_at.append(time.monotonic())
        calls.append(threading.current_thread().daemon)
        entered.set()
        try:
            assert release.wait(300)
            assert tmp_path.exists()
            return ToolOutput(status=RunStatus.OK)
        finally:
            finished.set()

    monkeypatch.setattr(_JobRun, "_collect", collect)
    try:
        out = _run(_tool_job("pass"), tmp_path)
        assert len(entered_at) == 1
        assert time.monotonic() - entered_at[0] < 30
        assert not release.is_set()
        assert not finished.is_set()
        assert entered.is_set()
        assert out.status == RunStatus.ERROR
        assert calls == [True]
    finally:
        release.set()
        assert finished.wait(60)
        for thread in workers:
            thread.join(60)
            assert not thread.is_alive()


@pytest.mark.ray
def test_worker_report_diagnostics_use_frame_slot(ray_driver: None) -> None:
    code = (
        "import json, os; os.mkfifo('swarm/llm_calls.jsonl'); "
        "json.dump({'data': {}}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
    )
    decoded = unframe(_remote(_tool_job(code)))
    out = ToolOutput.model_validate_json(decoded.head)
    assert out.status == RunStatus.OK
    assert out.data == {}
    assert [(d.code, d.count) for d in decoded.diagnostics] == [("llm_calls_special_file", 1)]
    assert "diagnostic" not in decoded.head.decode()


@pytest.mark.parametrize("phase", ["prepare", "collect"])
@pytest.mark.parametrize(
    "kind", ["ordinary", "raw_fenced", "raw_limit", "terminal", "nested", "fenced", "interrupt"]
)
def test_hook_outcomes_preserve_authority_and_secrecy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    kind: str,
) -> None:
    from _fakes.tools import PyTool

    from swarm_arena._errors import Fenced, _core_error, _SignalCategory
    from swarm_arena.manager import LimitReached

    secret = "hook-SENTINEL-secret"
    errors: dict[str, BaseException] = {
        "ordinary": ValueError(secret),
        "raw_fenced": Fenced(secret),
        "raw_limit": LimitReached("bad", secret),
        "terminal": _core_error(RuntimeError, "cleanup failed", _category=_SignalCategory.TERMINAL),
        "fenced": _core_error(Fenced, "run ownership lost"),
        "interrupt": KeyboardInterrupt(secret),
    }
    errors["nested"] = BaseExceptionGroup(secret, [errors["terminal"]])

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise errors[kind]

    monkeypatch.setattr(PyTool, "command" if phase == "prepare" else "collect", fail)
    artifacts = ArtifactStore(MemoryStore())
    if kind in {"terminal", "nested", "fenced"}:
        with pytest.raises(
            (RuntimeError, Fenced), match=r"cleanup failed|ownership lost"
        ) as caught:
            _run(_tool_job("pass"), tmp_path, artifacts)
        assert_secret_absent(caught.value, secret)
    else:
        out = _run(_tool_job("pass"), tmp_path, artifacts)
        assert out.status == RunStatus.ERROR
        assert secret not in out.model_dump_json()
    for path in tmp_path.rglob("*.log"):
        assert secret not in path.read_text()


@pytest.mark.parametrize("nested", [False, True])
def test_stop_collect_control_survives_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    from _fakes.tools import PyTool

    from swarm_arena._errors import _core_error, _SignalCategory

    pid = tmp_path / "pid"
    _timeout_after_ready(monkeypatch, lambda: _pid_written(pid))

    def fail(*args: Any, **kwargs: Any) -> Any:
        terminal = _core_error(RuntimeError, "cleanup failed", _category=_SignalCategory.TERMINAL)
        raise BaseExceptionGroup("SENTINEL", [terminal]) if nested else terminal

    monkeypatch.setattr(PyTool, "collect", fail)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        _run(
            _job("ignore_sigterm.py", args=["--pid-file", str(pid)], grace_s=0),
            tmp_path,
        )
    assert not pid_alive(int(pid.read_text()))
    assert "SENTINEL" not in (tmp_path / "swarm/stderr.log").read_text()


def test_broken_handshake_still_reaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena.ray import _task

    original = os.write

    def broken(fd: int, data: bytes) -> int:
        if data == b"go":
            raise BrokenPipeError
        return original(fd, data)

    monkeypatch.setattr(_task.os, "write", broken)
    assert _run(_tool_job("pass"), tmp_path).status == RunStatus.ERROR


def test_incremental_output_cap_and_duplicate_blobs() -> None:
    from swarm_arena.ray._task import _OutputArtifacts, _OutputStore

    store = _OutputStore({}, MIB)
    artifacts = _OutputArtifacts(store)
    digest = artifacts.put_bytes(b"x" * MIB)
    assert artifacts.put_bytes(b"x" * MIB) == digest
    with pytest.raises(ValueError, match="object_store_cap_mib"):
        artifacts.put_bytes(b"y")
    assert store.total == MIB
    assert len(store.created()) == 1
    consumed: list[int] = []

    def chunks() -> Any:
        for i in range(10):
            consumed.append(i)
            yield b"z" * MIB

    with pytest.raises(ValueError, match="object_store_cap_mib"):
        _OutputArtifacts(_OutputStore({}, MIB))._put_chunks(chunks())
    assert consumed == [0, 1]


async def test_cancel_preparation_never_launches_and_retains_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers: list[threading.Thread] = []
    import asyncio

    from _fakes.tools import PyTool

    from swarm_arena.ray._task import _Worker

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    ready = anyio.Event()
    original = PyTool.command

    def held(self: Any, request: Any, workspace: Path) -> Command:
        workers.append(threading.current_thread())
        entered.set()
        loop.call_soon_threadsafe(ready.set)
        try:
            assert release.wait(300)
            (workspace / "late").touch()
            return original(self, request, workspace)
        finally:
            finished.set()

    monkeypatch.setattr(PyTool, "command", held)
    worker = _Worker(
        _tool_job("open('launched', 'w').close()"),
        tmp_path,
        ArtifactStore(MemoryStore()),
        TaskSettings(),
    )
    task = asyncio.create_task(worker.run())
    try:
        with anyio.fail_after(60):
            await ready.wait()
        task.cancel()
        with anyio.fail_after(30):
            await asyncio.wait({task})
        with pytest.raises(asyncio.CancelledError):
            task.result()
        assert not release.is_set()
        assert not finished.is_set()
        assert not worker.finished
        assert not (tmp_path / "launched").exists()
    finally:
        release.set()
        assert await anyio.to_thread.run_sync(finished.wait, 60)
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert entered.is_set()
    assert (tmp_path / "late").exists()
    assert not (tmp_path / "launched").exists()


async def test_cancel_publication_is_data_only_and_no_next_object(
    tmp_path: Path,
) -> None:
    workers: list[threading.Thread] = []
    import asyncio

    from swarm_arena.ray._task import _Worker

    loop = asyncio.get_running_loop()
    ready = anyio.Event()
    release, finished = threading.Event(), threading.Event()
    writes: list[str] = []

    class HeldStore(MemoryStore):
        def create(self, key: str, data: bytes) -> str:
            workers.append(threading.current_thread())
            writes.append(key)
            loop.call_soon_threadsafe(ready.set)
            try:
                assert release.wait(300)
                return super().create(key, data)
            finally:
                finished.set()

    worker = _Worker(
        _tool_job(
            "import json, os; json.dump({'data': {'a': 1}}, "
            "open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
        ),
        tmp_path,
        ArtifactStore(HeldStore()),
        TaskSettings(),
    )
    task = asyncio.create_task(worker.run())
    try:
        with anyio.fail_after(60):
            await ready.wait()
        task.cancel()
        with anyio.fail_after(30):
            await asyncio.wait({task})
        with pytest.raises(asyncio.CancelledError):
            task.result()
        assert not release.is_set()
        assert not finished.is_set()
        assert worker.partial.status == RunStatus.CANCELLED
        assert worker.partial.data == {"a": 1}
        assert worker.partial.files == []
        assert worker.partial.logs_ref is None
        assert worker.partial.llm_calls_file is None
        assert not worker.finished
        assert len(writes) == 1
    finally:
        release.set()
        assert await anyio.to_thread.run_sync(finished.wait, 60)
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert len(writes) == 1


@pytest.mark.parametrize("nested", [False, True])
async def test_pending_cancel_beats_collector_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
) -> None:
    workers: list[threading.Thread] = []
    import asyncio

    from swarm_arena._errors import _core_error, _SignalCategory
    from swarm_arena.ray._task import _Worker, _WorkerRun

    ready = anyio.Event()
    loop = asyncio.get_running_loop()
    release, finished = threading.Event(), threading.Event()

    def held(self: Any, run: Any) -> ToolOutput:
        loop.call_soon_threadsafe(ready.set)
        try:
            assert release.wait(300)
            terminal = _core_error(
                RuntimeError, "cleanup failed", _category=_SignalCategory.TERMINAL
            )
            raise BaseExceptionGroup("secret", [terminal]) if nested else terminal
        finally:
            finished.set()

    monkeypatch.setattr(_WorkerRun, "_collect", held)
    worker = _Worker(_tool_job("pass"), tmp_path, ArtifactStore(MemoryStore()), TaskSettings())
    task = asyncio.create_task(worker.run())
    try:
        with anyio.fail_after(60):
            await ready.wait()
        task.cancel()
        release.set()
        with anyio.fail_after(60):
            await asyncio.wait({task})
        with pytest.raises(asyncio.CancelledError):
            task.result()
        assert worker.partial.status == RunStatus.CANCELLED
    finally:
        release.set()
        assert await anyio.to_thread.run_sync(finished.wait, 60)
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_run_job_retains_abandoned_collector_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers: list[threading.Thread] = []
    entered_at: list[float] = []
    import atexit

    from swarm_arena.ray import _task

    hooks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def register(callback: Any, *args: Any, **kwargs: Any) -> Any:
        hooks.append((callback, args, kwargs))
        return callback

    monkeypatch.setattr(atexit, "register", register)
    workspace = tmp_path / "worker"
    workspace.mkdir()
    release, finished = threading.Event(), threading.Event()
    calls: list[bool] = []

    def collect(self: Any, run: Any) -> ToolOutput:
        workers.append(threading.current_thread())
        entered_at.append(time.monotonic())
        calls.append(threading.current_thread().daemon)
        try:
            assert release.wait(300)
            (workspace / "late-write").touch()
            return ToolOutput(status=RunStatus.OK)
        finally:
            finished.set()

    monkeypatch.setattr(_task.tempfile, "mkdtemp", lambda **kw: str(workspace))
    monkeypatch.setattr(_task._WorkerRun, "_collect", collect)
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.2)
    try:
        wire = cast(Any, run_job)._function(
            job_payload(_tool_job("pass"), TaskSettings(cap_bytes=MIB))
        )
        assert len(entered_at) == 1
        assert time.monotonic() - entered_at[0] < 30
        assert not release.is_set()
        assert not finished.is_set()
        assert ToolOutput.model_validate_json(unframe(wire).head).status == RunStatus.ERROR
        assert workspace.exists()
        assert calls == [True]
    finally:
        release.set()
        assert finished.wait(60)
        for thread in workers:
            thread.join(60)
            assert not thread.is_alive()
    assert (workspace / "late-write").exists()
    assert len(hooks) == 1
    callback, args, kwargs = hooks[0]
    assert kwargs == {"ignore_errors": True}
    callback(*args, **kwargs)
    assert not workspace.exists()
    callback(*args, **kwargs)  # already removed at exit is harmless


def test_worker_report_bytes_and_registered_diagnostics(tmp_path: Path) -> None:
    code = (
        "import json, os; open('swarm/llm_calls.jsonl', 'w').write('{}\\n'); "
        "json.dump({'data': {'value': 1}}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
    )
    wire = cast(Any, run_job)._function(job_payload(_tool_job(code), TaskSettings(cap_bytes=MIB)))
    decoded = unframe(wire)
    output = ToolOutput.model_validate_json(decoded.head)
    assert output.status == RunStatus.OK
    assert output.llm_calls_file is not None
    assert decoded.blobs[output.llm_calls_file.sha256] == b"{}\n"
    assert decoded.diagnostics == ()


def test_process_options_and_oom_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena import Resources
    from swarm_arena.ray import _task

    calls: list[dict[str, Any]] = []
    original_spawn = _task.subprocess.Popen
    original_argv = _task.launch_argv

    def spawn(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original_spawn(*args, **kwargs)

    def argv(args: Any, *, go_fd: int, rlimits: Any) -> list[str]:
        assert rlimits.memory_bytes == 2**30
        # macOS refuses RLIMIT_AS; test exit classification independently.
        return original_argv(args, go_fd=go_fd, rlimits=supervisor.RLimits())

    monkeypatch.setattr(_task.subprocess, "Popen", spawn)
    monkeypatch.setattr(_task, "launch_argv", argv)
    out = _run(
        make_job(
            command=Command(
                argv=[
                    sys.executable,
                    "-c",
                    "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
                ]
            ),
            resources=Resources(memory_gib=1),
            timeout_s=60,
        ),
        tmp_path,
    )
    assert out.status == RunStatus.OOM
    assert len(calls) == 1
    assert not {"preexec_fn", "start_new_session", "process_group"}.intersection(calls[0])
    assert len(calls[0]["pass_fds"]) == 1


@pytest.mark.parametrize("site", ["open_store", "mkdtemp", "rmtree"])
def test_remote_boundary_errors_are_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
) -> None:
    from swarm_arena.ray import _task

    secret = "boundary-SENTINEL-secret"

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError(secret)

    if site == "open_store":
        monkeypatch.setattr(_task, site, fail)
    elif site == "mkdtemp":
        monkeypatch.setattr(_task.tempfile, site, fail)
    else:
        monkeypatch.setattr(_task.shutil, site, fail)
        workspace = tmp_path / "worker"
        workspace.mkdir()
        monkeypatch.setattr(_task.tempfile, "mkdtemp", lambda **kw: str(workspace))
    job = _tool_job("pass", artifact_root=str(tmp_path / "artifacts"))
    wire = cast(Any, run_job)._function(job_payload(job, TaskSettings()))
    output = ToolOutput.model_validate_json(unframe(wire).head)
    assert output.status == RunStatus.ERROR
    assert secret not in wire.decode()
    assert "ValueError" in output.data["swarm_arena_error"]


@pytest.mark.parametrize("kind", ["terminal", "nested", "fenced", "raw"])
def test_remote_boundary_keeps_control_authority(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from swarm_arena._errors import Fenced, _core_error, _SignalCategory
    from swarm_arena.ray import _task

    terminal = _core_error(RuntimeError, "cleanup failed", _category=_SignalCategory.TERMINAL)
    errors = {
        "terminal": terminal,
        "nested": BaseExceptionGroup("SENTINEL", [terminal]),
        "fenced": _core_error(Fenced, "run ownership lost"),
        "raw": Fenced("SENTINEL"),
    }

    def fail(root: str) -> Any:
        raise errors[kind]

    monkeypatch.setattr(_task, "open_store", fail)
    payload = job_payload(_job(), TaskSettings())
    if kind == "raw":
        wire = cast(Any, run_job)._function(payload)
        assert ToolOutput.model_validate_json(unframe(wire).head).status == RunStatus.ERROR
        assert b"SENTINEL" not in wire
    else:
        with pytest.raises(
            (RuntimeError, Fenced), match=r"cleanup failed|ownership lost"
        ) as caught:
            cast(Any, run_job)._function(payload)
        assert_secret_absent(caught.value, "SENTINEL")


def _cancel_job(ready: Path) -> Job:
    code = WORKER_SCRIPTS["partial_on_sigterm.py"].replace(
        "{'data': {'step': n}}", "{'data': {'step': n}, 'files': ['out.txt']}"
    )
    prefix = (
        "import sys; from pathlib import Path; "
        f"sys.argv = ['tool', '--ready-file', {str(ready)!r}]; "
        "Path('out.txt').write_text('artifact'); "
        "Path('swarm/llm_calls.jsonl').write_text('{}\\n'); print('tool log', flush=True)\n"
    )
    return _tool_job(prefix + code, timeout_s=None, grace_s=2)


@pytest.mark.parametrize("foreign", [False, True])
def test_interrupt_main_owns_handler_and_returns_data_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign: bool
) -> None:
    import _thread

    from swarm_arena.ray import _task

    foreign_calls: list[int] = []

    def foreign_handler(signum: int, frame: Any) -> None:
        foreign_calls.append(signum)
        raise KeyboardInterrupt

    previous = signal.getsignal(signal.SIGINT)
    installed = foreign_handler if foreign else signal.default_int_handler
    signal.signal(signal.SIGINT, installed)
    ready = tmp_path / "ready"
    original_wait = _task._WorkerSupervisor._wait
    original_run = anyio.run
    owned_before_loop: list[bool] = []
    interrupted = False

    def run(*args: Any, **kwargs: Any) -> Any:
        owned_before_loop.append(signal.getsignal(signal.SIGINT) is not installed)
        return original_run(*args, **kwargs)

    async def wait(self: Any, process: Any, deadline: Any) -> bool:
        nonlocal interrupted
        if not interrupted:
            assert await anyio.to_thread.run_sync(wait_until, ready.exists, 60)
            interrupted = True
            _thread.interrupt_main()
        return await original_wait(self, process, deadline)

    monkeypatch.setattr(_task._WorkerSupervisor, "_wait", wait)
    monkeypatch.setattr(anyio, "run", run)
    store = MemoryStore()
    try:
        out = _run(_cancel_job(ready), tmp_path, ArtifactStore(store))
        assert signal.getsignal(signal.SIGINT) is installed
        assert owned_before_loop == [True]
        assert foreign_calls == []
        assert out.status == RunStatus.CANCELLED
        assert out.partial
        assert out.data["step"] >= 1
        assert out.files == []
        assert out.logs_ref is None
        assert out.llm_calls_file is None
        assert store.created() == {}
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.mark.parametrize("cancel_kind", ["scope", "interrupt"])
def test_cancel_during_timeout_grace_keeps_one_collect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_kind: str
) -> None:
    import _thread

    from _fakes.tools import PyTool

    from swarm_arena.ray import _task

    ready, term, release = (tmp_path / name for name in ("ready", "term", "release"))
    code = f"""import json, os, signal, time
from pathlib import Path
stopped = False
def stop(signum, frame):
    global stopped
    stopped = True
signal.signal(signal.SIGTERM, stop)
Path('out.txt').write_text('artifact')
Path('swarm/llm_calls.jsonl').write_text('{{}}\\n')
print('tool log', flush=True)
Path({str(ready)!r}).touch()
while not stopped:
    time.sleep(0.01)
Path({str(term)!r}).touch()
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
result = {{'data': {{'step': 4}}, 'files': ['out.txt']}}
Path(os.environ['SWARM_ARENA_RESULT']).write_text(json.dumps(result))
"""
    store = MemoryStore()
    worker = _task._Worker(
        _tool_job(code, timeout_s=60, grace_s=300), tmp_path, ArtifactStore(store), TaskSettings()
    )
    original_wait = _task._WorkerSupervisor._wait
    original_collect = PyTool.collect
    waits: list[str] = []
    collects: list[RunStatus] = []
    processes: list[Any] = []
    scope: anyio.CancelScope | None = None

    def collect(self: Any, request: Any, workspace: Path, run: Any) -> ToolOutput:
        collects.append(run.status)
        assert release.exists()
        return original_collect(self, request, workspace, run)

    async def wait(self: Any, process: Any, deadline: Any) -> bool:
        processes.append(process)
        marker = ready if not waits else term
        assert await anyio.to_thread.run_sync(wait_until, marker.exists, 60)
        if not waits:
            waits.append("execution")
            return False  # enter timeout cleanup after the child's readiness event
        waits.append("grace")
        if cancel_kind == "scope":
            assert scope is not None
            scope.cancel()
        else:
            _thread.interrupt_main()
        # The child cannot produce a result until this grace wait survives cancel.
        await anyio.lowlevel.checkpoint()
        await anyio.lowlevel.checkpoint()
        assert process.poll() is None
        assert collects == []
        release.touch()
        return await original_wait(self, process, deadline)

    async def run_scoped() -> ToolOutput:
        nonlocal scope
        with anyio.CancelScope() as scope:
            await worker.run()
        assert scope.cancelled_caught
        return worker.partial

    monkeypatch.setattr(_task._WorkerSupervisor, "_wait", wait)
    monkeypatch.setattr(PyTool, "collect", collect)
    out = anyio.run(run_scoped) if cancel_kind == "scope" else _task._execute(worker)
    assert waits == ["execution", "grace"]
    assert out.status == RunStatus.CANCELLED
    assert out.partial
    assert out.data == {"step": 4}
    assert out.files == []
    assert out.logs_ref is None
    assert out.llm_calls_file is None
    assert collects == [RunStatus.TIMEOUT]
    assert store.created() == {}
    assert all(process.poll() is not None for process in processes)


def test_interrupt_after_loop_completion_keeps_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import _thread

    from swarm_arena.ray import _task

    expected = ToolOutput(status=RunStatus.OK, data={"finished": True})
    original = anyio.run

    async def completed(self: Any) -> ToolOutput:
        return expected

    def late_interrupt(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        _thread.interrupt_main()
        return output

    monkeypatch.setattr(_task._Worker, "run", completed)
    monkeypatch.setattr(anyio, "run", late_interrupt)
    assert _run(_job(), tmp_path) is expected


def test_soft_cancel_during_publication_omits_completed_blobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import _thread

    from swarm_arena.ray import _task

    original = _task._WorkerRun._workspace_io
    uploads: list[str] = []

    async def interrupt_after_file(self: Any, operation: Any, *args: Any, **kw: Any) -> Any:
        result = await original(self, operation, *args, **kw)
        if operation.__name__ == "_upload_file":
            uploads.append(args[0])
            _thread.interrupt_main()
            await anyio.lowlevel.checkpoint()
            await anyio.lowlevel.checkpoint()
        return result

    monkeypatch.setattr(_task._WorkerRun, "_workspace_io", interrupt_after_file)
    code = (
        "import json, os; open('out.txt', 'w').write('artifact'); "
        "print('tool log', flush=True); "
        "json.dump({'data': {'step': 7}, 'files': ['out.txt']}, "
        "open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
    )
    decoded = unframe(_task._run_frame(job_payload(_tool_job(code), TaskSettings(cap_bytes=MIB))))
    output = ToolOutput.model_validate_json(decoded.head)
    assert uploads == ["out.txt"]
    assert output.status == RunStatus.CANCELLED
    assert output.partial
    assert output.data == {"step": 7}
    assert output.files == []
    assert output.logs_ref is None
    assert output.llm_calls_file is None
    assert decoded.blobs == {}


def test_drop_env_payload_roundtrip_and_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "real-provider-SENTINEL")
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-SENTINEL")
    job = make_job(
        command=Command(
            argv=[
                sys.executable,
                "-c",
                "import os; assert 'GEMINI_API_KEY' not in os.environ; "
                "assert os.environ['OPENAI_API_KEY'] == 'sk-v'; "
                "open(os.environ['SWARM_ARENA_RESULT'], 'w').write('{\"data\": {}}')",
            ],
            env=["GEMINI_API_KEY", "OPENAI_API_KEY"],
        ),
        artifact_root=str(tmp_path / "artifacts"),
        drop_env=("GEMINI_API_KEY", "OPENAI_API_KEY"),
        secret_values={"OPENAI_API_KEY": "sk-v"},
        timeout_s=60,
    )
    restored, _, _ = read_payload(job_payload(job, TaskSettings()))
    assert restored.drop_env == job.drop_env
    assert _run(restored, tmp_path).status == RunStatus.OK
