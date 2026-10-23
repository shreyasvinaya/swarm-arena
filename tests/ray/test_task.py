"""Worker mechanics run locally; only transport/lifetime tests need the shared head."""

import dataclasses
import importlib.metadata
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
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
from swarm_arena.tool import Command, RunStatus, ToolOutput, ToolRequest

try:
    importlib.metadata.version("ray")
except importlib.metadata.PackageNotFoundError:
    pytest.skip("Ray worker tests require the ray extra", allow_module_level=True)

# isort: split
import ray

from swarm_arena.ray._files import MIB, MemoryStore, frame, inline_inputs, unframe
from swarm_arena.ray._task import (
    TaskSettings,
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
    )


def _run(job: Job, workspace: Path, artifacts: ArtifactStore | None = None) -> ToolOutput:
    return run_in_worker(job, workspace, artifacts or ArtifactStore(MemoryStore()), poll_s=0.2)


def _remote(job: Job, *, cap: int = 256 * MIB, blobs: dict[str, bytes] | None = None) -> bytes:
    return ray.get(
        run_job.remote(job_payload(job, TaskSettings(cap_bytes=cap), blobs=blobs)), timeout=45
    )


def test_run_in_worker_ok(tmp_path: Path) -> None:
    assert _run(_job(), tmp_path).status == RunStatus.OK


def test_timeout_keeps_partial_output(tmp_path: Path) -> None:
    out = _run(_job("partial_on_sigterm.py", timeout_s=1.0, grace_s=2.0), tmp_path)
    assert out.status == RunStatus.TIMEOUT
    assert out.partial
    assert out.data["step"] >= 1


def test_tool_ignoring_sigterm_killed_after_grace(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    out = _run(
        _job("ignore_sigterm.py", args=["--pid-file", str(pid)], timeout_s=1, grace_s=0.2), tmp_path
    )
    assert out.status == RunStatus.TIMEOUT
    assert not pid_alive(int(pid.read_text()))


def test_blocked_collect_cannot_stop_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.2)
    from swarm_arena.ray._task import _execute, _Worker

    RELEASE.clear()
    pid = tmp_path / "pid"
    code = WORKER_SCRIPTS["ignore_sigterm.py"]
    code = f"import sys; sys.argv = ['tool', '--pid-file', {str(pid)!r}]\n" + code
    worker = _Worker(
        _tool_job(code, tool="BlockingCollectTool", timeout_s=1, grace_s=0.2),
        tmp_path,
        ArtifactStore(MemoryStore()),
        TaskSettings(),
    )
    try:
        out = _execute(worker)
        assert (out.status, out.partial) == (RunStatus.TIMEOUT, False)
        assert not pid_alive(int(pid.read_text()))
    finally:
        RELEASE.set()
        if worker.job_run is not None:
            assert worker.job_run.collect_finished.wait(5)


def test_fifo_stderr_cannot_block_kill(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    out = _run(
        _job(
            "ignore_sigterm.py",
            args=["--pid-file", str(pid), "--fifo-stderr"],
            timeout_s=1,
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
            _job(
                "partial_on_sigterm.py",
                args=["--ready-file", str(ready)],
                timeout_s=None,
                grace_s=2,
            ),
            TaskSettings(cap_bytes=256 * MIB),
        )
    )
    try:
        assert wait_until(ready.exists, 30)
        ray.cancel(ref)
        out = ToolOutput.model_validate_json(unframe(ray.get(ref, timeout=40)).head)
        assert out.status == RunStatus.CANCELLED
        assert out.partial
        assert out.data["step"] >= 1
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
        assert wait_until(lambda: not pid_alive(int(pid_file.read_text())), 10)
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
    remote_bytes = await anyio.to_thread.run_sync(lambda: ray.get(ref, timeout=45))
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


def test_effective_command_bounds(tmp_path: Path) -> None:
    job = _tool_job(WORKER_SCRIPTS["partial_on_sigterm.py"], timeout_s=60, grace_s=60)
    job = dataclasses.replace(
        job,
        tool=Ref(
            path="_fakes.tools:PyTool",
            config={
                "code": "import sys; sys.argv = ['tool']\n"
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
    from swarm_arena._exec.job_steps import _JobRun

    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.2)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls: list[bool] = []

    def collect(self: Any, run: Any) -> ToolOutput:
        calls.append(threading.current_thread().daemon)
        entered.set()
        try:
            assert release.wait(30)
            assert tmp_path.exists()
            return ToolOutput(status=RunStatus.OK)
        finally:
            finished.set()

    monkeypatch.setattr(_JobRun, "_collect", collect)
    try:
        out = _run(_tool_job("pass"), tmp_path)
        assert entered.is_set()
        assert out.status == RunStatus.ERROR
        assert calls == [True]
    finally:
        release.set()
        assert finished.wait(5)


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

    def fail(*args: Any, **kwargs: Any) -> Any:
        terminal = _core_error(RuntimeError, "cleanup failed", _category=_SignalCategory.TERMINAL)
        raise BaseExceptionGroup("SENTINEL", [terminal]) if nested else terminal

    monkeypatch.setattr(PyTool, "collect", fail)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        _run(
            _job("ignore_sigterm.py", args=["--pid-file", str(pid)], timeout_s=1, grace_s=0),
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
    import asyncio

    from _fakes.tools import PyTool

    from swarm_arena.ray._task import _Worker

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    ready = anyio.Event()
    original = PyTool.command

    def held(self: Any, request: Any, workspace: Path) -> Command:
        entered.set()
        loop.call_soon_threadsafe(ready.set)
        try:
            assert release.wait(30)
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
        with anyio.fail_after(10):
            await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not worker.finished
        assert not (tmp_path / "launched").exists()
    finally:
        release.set()
        assert await anyio.to_thread.run_sync(finished.wait, 5)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert entered.is_set()
    assert (tmp_path / "late").exists()
    assert not (tmp_path / "launched").exists()


async def test_cancel_publication_is_data_only_and_no_next_object(
    tmp_path: Path,
) -> None:
    import asyncio

    from swarm_arena.ray._task import _Worker

    loop = asyncio.get_running_loop()
    ready = anyio.Event()
    release, finished = threading.Event(), threading.Event()
    writes: list[str] = []

    class HeldStore(MemoryStore):
        def create(self, key: str, data: bytes) -> str:
            writes.append(key)
            loop.call_soon_threadsafe(ready.set)
            try:
                assert release.wait(30)
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
        with anyio.fail_after(10):
            await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker.partial.status == RunStatus.CANCELLED
        assert worker.partial.data == {"a": 1}
        assert worker.partial.files == []
        assert worker.partial.logs_ref is None
        assert worker.partial.llm_calls_file is None
        assert not worker.finished
        assert len(writes) == 1
    finally:
        release.set()
        assert await anyio.to_thread.run_sync(finished.wait, 5)
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
    import asyncio

    from swarm_arena._errors import _core_error, _SignalCategory
    from swarm_arena.ray._task import _Worker, _WorkerRun

    ready = anyio.Event()
    loop = asyncio.get_running_loop()
    release, finished = threading.Event(), threading.Event()

    def held(self: Any, run: Any) -> ToolOutput:
        loop.call_soon_threadsafe(ready.set)
        try:
            assert release.wait(30)
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
        with anyio.fail_after(10):
            await ready.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker.partial.status == RunStatus.CANCELLED
    finally:
        release.set()
        assert await anyio.to_thread.run_sync(finished.wait, 5)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_run_job_retains_abandoned_collector_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena.ray import _task

    workspace = tmp_path / "worker"
    workspace.mkdir()
    release, finished = threading.Event(), threading.Event()
    calls: list[bool] = []

    def collect(self: Any, run: Any) -> ToolOutput:
        calls.append(threading.current_thread().daemon)
        try:
            assert release.wait(30)
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
        assert ToolOutput.model_validate_json(unframe(wire).head).status == RunStatus.ERROR
        assert workspace.exists()
        assert calls == [True]
    finally:
        release.set()
        assert finished.wait(5)
    assert (workspace / "late-write").exists()


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
    )
    restored, _, _ = read_payload(job_payload(job, TaskSettings()))
    assert restored.drop_env == job.drop_env
    assert _run(restored, tmp_path).status == RunStatus.OK
