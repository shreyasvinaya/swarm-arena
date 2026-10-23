"""Shared executor preparation, transport restoration and publication contracts."""

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import PyTool, make_job

from swarm_arena import Ref
from swarm_arena._errors import Fenced, StorageError, _core_error, _SignalCategory
from swarm_arena._exec.job_steps import _JobRun, live_request, prepare_job, resolve_argv
from swarm_arena._model.canonical import sha256_hex, utc_now
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.staging import edit_hashes, put_edits
from swarm_arena.executor import Job
from swarm_arena.testing import read_logs, read_output_file
from swarm_arena.tool import Command, CompletedRun, FileRef, RunStatus, ToolOutput, ToolRequest

EDITS = {"p.txt": "hello", "q.txt": "sha256:hello"}
SECRET = "f810e652-43d2-4555-8280-90e395b4ee31"


def _job(tmp_path: Path, **fields: Any) -> Job:
    return make_job(
        artifact_root=str(tmp_path / "artifacts"),
        **{"command": Command(argv=["python", "-c", "pass"]), **fields},
    )


def _completed(ws: Path) -> CompletedRun:
    (ws / "swarm/stdout.log").write_bytes(b"o")
    (ws / "swarm/stderr.log").write_bytes(b"e")
    return CompletedRun(
        status=RunStatus.OK,
        returncode=0,
        stdout_path=ws / "swarm/stdout.log",
        stderr_path=ws / "swarm/stderr.log",
        started_at=utc_now(),
        ended_at=utc_now(),
    )


def test_prepare_job_tool_kind_returns_command_and_tool(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    request = ToolRequest(edits=EDITS)
    job = _job(
        tmp_path,
        kind="tool",
        command=None,
        request=request,
        tool=Ref.of(PyTool({"code": "pass", "timeout_s": 2, "grace_s": 1})),
        timeout_s=5,
        grace_s=3,
    )
    staged = prepare_job(job, tmp_path / "ws", artifacts)
    assert isinstance(staged.tool, PyTool)
    assert staged.request is request
    assert (staged.timeout_s, staged.grace_s) == (2, 1)
    assert staged.argv == [sys.executable, "-c", "pass"]


def test_prepare_job_command_kind_uses_job_command(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    job = _job(tmp_path, timeout_s=5, grace_s=1)
    staged = prepare_job(job, tmp_path / "ws", artifacts)
    assert staged.tool is None
    assert (staged.timeout_s, staged.grace_s) == (5, 1)
    assert staged.argv == [sys.executable, "-c", "pass"]


def test_resolve_argv_replaces_leading_python() -> None:
    assert resolve_argv(Command(argv=["python", "x.py"], python="/opt/py"))[0] == "/opt/py"
    assert resolve_argv(Command(argv=["python", "x.py"]))[0] == sys.executable
    assert resolve_argv(Command(argv=["/bin/echo"]))[0] == "/bin/echo"


def test_live_request_restores_transported_edits(artifacts: ArtifactStore) -> None:
    put_edits(EDITS, artifacts)
    sent = ToolRequest.model_validate_json(
        ToolRequest(inputs={"n": 1}, edits=EDITS).model_dump_json()
    )
    assert sent.edits == {p: f"sha256:{h}" for p, h in edit_hashes(EDITS).items()}
    live = live_request(sent, artifacts)
    assert (live.edits, live.inputs) == (EDITS, {"n": 1})
    assert list(live.edits) == list(EDITS)
    assert live_request(live, artifacts) == live


def test_live_request_keeps_live_content(artifacts: ArtifactStore) -> None:
    assert live_request(ToolRequest(edits=EDITS), artifacts).edits == EDITS


@pytest.mark.parametrize("kind", ["missing", "encoding", "hash"])
def test_live_request_unrestorable_reference_raises(
    artifacts: ArtifactStore,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = sha256_hex(b"absent") if kind == "missing" else artifacts.put_bytes(b"\xff\xfe")
    if kind == "hash":

        def fail(_: str) -> bytes:
            raise StorageError(SECRET)

        monkeypatch.setattr(artifacts, "get_bytes", fail)
    with pytest.raises(
        ValueError, match=r"^edit 'p.txt': stored reference could not be restored"
    ) as caught:
        live_request(ToolRequest(edits={"p.txt": f"sha256:{digest}"}), artifacts)
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("kind", ["terminal", "fenced", "raw-fenced", "raw-limit", "cancel"])
@pytest.mark.parametrize("nested", [False, True])
def test_live_request_preserves_control_authority(
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    nested: bool,
) -> None:
    from swarm_arena._agent.manager import LimitReached

    if kind == "terminal":
        error = _core_error(RuntimeError, "terminal test", _category=_SignalCategory.TERMINAL)
    elif kind == "fenced":
        error = _core_error(Fenced, "run ownership lost")
    elif kind == "raw-fenced":
        error = Fenced(SECRET)
    elif kind == "raw-limit":
        error = LimitReached("turns", SECRET)
    else:
        error = asyncio.CancelledError(SECRET)
    if nested:
        error = BaseExceptionGroup("untrusted", [error])

    def fail(_: str) -> bytes:
        raise error

    monkeypatch.setattr(artifacts, "get_bytes", fail)
    expected = {
        "terminal": RuntimeError,
        "fenced": Fenced,
        "cancel": ValueError,
    }.get(kind, ValueError)
    with pytest.raises(expected) as caught:
        live_request(ToolRequest(edits={"p.txt": "sha256:" + "0" * 64}), artifacts)
    assert_secret_absent(caught.value, SECRET)


async def test_collect_output_command_kind_parses_result_json(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    job = _job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    run = _completed(ws)
    (ws / "swarm/result.json").write_text('{"x": 1}')
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=1024)
    collected = runner._collect(run)
    assert collected is not None
    assert collected.data == {"x": 1}
    out = await runner._finish(ToolOutput(status=RunStatus.OK), (run.started_at, run.ended_at))
    assert out.logs_ref == f"sha256:{sha256_hex(b'o')}:{sha256_hex(b'e')}"
    assert (out.started_at, out.ended_at) == (run.started_at, run.ended_at)


@pytest.mark.s3
async def test_nonlocal_outputs_read_back_through_public_readers(
    tmp_path: Path,
    s3_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena._store.base import open_store

    artifacts = ArtifactStore(open_store(s3_url))
    job = _job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    run = _completed(ws)
    (ws / "answer.txt").write_bytes(b"answer")
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=6)
    output = ToolOutput(
        status=RunStatus.OK,
        files=[FileRef(name="answer", path="answer.txt", sha256="0" * 64, size=0)],
    )
    out = await runner._finish(output, (run.started_at, run.ended_at))
    assert out.status is RunStatus.OK
    monkeypatch.setattr("swarm_arena.testing._artifacts._artifacts", lambda: artifacts)
    assert read_output_file(out, "answer.txt") == b"answer"
    assert read_logs(out) == (b"o", b"e")


@pytest.mark.parametrize("kind", ["terminal", "fenced", "raw-fenced", "raw-limit"])
@pytest.mark.parametrize("nested", [False, True])
async def test_file_publication_preserves_control_authority(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    nested: bool,
) -> None:
    from swarm_arena._agent.manager import LimitReached

    job = _job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    run = _completed(ws)
    (ws / "answer").write_bytes(b"a")
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=1)
    output = ToolOutput(
        status=RunStatus.OK, files=[FileRef(name="answer", path="answer", sha256="0" * 64, size=0)]
    )
    error: BaseException
    if kind == "terminal":
        error = _core_error(ValueError, "terminal test", _category=_SignalCategory.TERMINAL)
    elif kind == "fenced":
        error = _core_error(Fenced, "run ownership lost")
    elif kind == "raw-fenced":
        error = Fenced(SECRET)
    else:
        error = LimitReached("turns", SECRET)
    if nested:
        error = BaseExceptionGroup(SECRET, [error])

    def fail(*args: Any, **kw: Any) -> tuple[str, int]:
        raise error

    monkeypatch.setattr(artifacts, "_put_chunks", fail)
    if kind in ("terminal", "fenced"):
        with pytest.raises(ValueError if kind == "terminal" else Fenced) as caught:
            await runner._finish(output, (run.started_at, run.ended_at))
        assert_secret_absent(caught.value, SECRET)
    else:
        out = await runner._finish(output, (run.started_at, run.ended_at))
        assert out.status is RunStatus.ERROR
        assert not out.files
        assert SECRET not in run.stderr_path.read_text()


@pytest.mark.s3
async def test_nonlocal_aggregate_cap_precedes_publication(
    tmp_path: Path,
    s3_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena._store.base import open_store

    artifacts = ArtifactStore(open_store(s3_url))
    job = _job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    _completed(ws)
    for path in ("one", "two"):
        (ws / path).write_bytes(b"123")
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=5)
    files = [FileRef(name=p, path=p, sha256="0" * 64, size=0) for p in ("one", "two")]
    stored: list[bytes] = []
    put = artifacts.put_bytes

    def publish(data: bytes) -> str:
        stored.append(data)
        return put(data)

    monkeypatch.setattr(artifacts, "put_bytes", publish)
    out = await runner._finish(ToolOutput(status=RunStatus.OK, files=files), None)
    assert out.status is RunStatus.ERROR
    assert out.files == []
    assert len(stored) == 2  # logs only, no first-file publication
    assert b"123" not in stored
    assert b"two crosses max_output_bytes" in stored[-1]


@pytest.mark.parametrize("invalid", ["missing", "symlink", "directory"])
async def test_invalid_later_output_precedes_any_file_publication(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    job = _job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    _completed(ws)
    (ws / "one").write_bytes(b"1")
    if invalid == "symlink":
        (ws / "two").symlink_to(ws / "one")
    elif invalid == "directory":
        (ws / "two").mkdir()
    uploads: list[bool] = []
    original = artifacts._put_chunks

    def publish(*args: Any, **kw: Any) -> tuple[str, int]:
        uploads.append(True)
        return original(*args, **kw)

    monkeypatch.setattr(artifacts, "_put_chunks", publish)
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=10)
    files = [FileRef(name=p, path=p, sha256="0" * 64, size=0) for p in ("one", "two")]
    out = await runner._finish(ToolOutput(status=RunStatus.OK, files=files), None)
    assert out.status is RunStatus.ERROR
    assert not out.files
    assert uploads == []


@pytest.mark.s3
async def test_cancel_nonlocal_publication_starts_no_next_object(
    tmp_path: Path,
    s3_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import anyio

    from swarm_arena._store.base import open_store

    backend = open_store(s3_url)
    artifacts = ArtifactStore(backend)
    job = _job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    _completed(ws)
    for path in ("one", "two"):
        (ws / path).write_bytes(path.encode())
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=6)
    files = [FileRef(name=p, path=p, sha256="0" * 64, size=0) for p in ("one", "two")]
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    published: list[bytes] = []
    create = backend.create

    def publish(key: str, data: bytes) -> str:
        published.append(data)
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(30)
        return create(key, data)

    monkeypatch.setattr(backend, "create", publish)
    task = asyncio.create_task(runner._finish(ToolOutput(status=RunStatus.OK, files=files), None))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 10)
        task.cancel()
        with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
            await task
        assert published == [b"one"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
    assert published == [b"one"]


async def test_live_request_await_is_abandonable(
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import anyio

    from swarm_arena._store.io import _validated_io

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []

    def read(digest: str) -> bytes:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(30)
        raise ValueError(SECRET)

    monkeypatch.setattr(artifacts, "get_bytes", read)
    task = asyncio.create_task(
        _validated_io(
            live_request,
            ToolRequest(edits={"p.txt": "sha256:" + "0" * 64}),
            artifacts,
        )
    )
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 5)
        task.cancel()
        with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()


@pytest.mark.parametrize("stop", ["normal", "cancel", "timeout"])
async def test_abandoned_collector_retains_its_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    import threading

    import anyio

    from swarm_arena._exec import supervisor
    from swarm_arena.executor import LocalExecutor

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    workspaces: list[Path] = []
    calls: list[bool] = []

    def collect(self: _JobRun, run: CompletedRun) -> ToolOutput:
        calls.append(True)
        workers.append(threading.current_thread())
        workspaces.append(self._workspace)
        entered.set()
        assert release.wait(30)
        (self._workspace / "late-collect").write_text("retained")
        return ToolOutput(status=RunStatus.OK, data={"kept": 1})

    monkeypatch.setattr(_JobRun, "_collect", collect)
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.2)
    job = _job(tmp_path, command=Command(argv=["true"]), grace_s=0)
    if stop == "timeout":
        job = _job(
            tmp_path,
            timeout_s=0.2,
            grace_s=0,
            command=Command(argv=["python", "-c", "import time; time.sleep(60)"]),
        )
    executor = LocalExecutor(workspace_root=str(tmp_path / "workspaces"))
    task = asyncio.create_task(executor.submit(job))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 10)
        with anyio.fail_after(5):
            if stop == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                output = await task
                assert output.status == (
                    RunStatus.TIMEOUT if stop == "timeout" else RunStatus.ERROR
                )
        assert calls == [True]
        assert workspaces[0].is_dir(), "workspace removed while an abandoned collect still uses it"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
        await executor.aclose()
    assert (workspaces[0] / "late-collect").read_text() == "retained"


@pytest.mark.parametrize("site", ["collect", "validation"])
@pytest.mark.parametrize("kind", ["terminal", "limit", "ordinary"])
async def test_collection_keeps_terminal_authority_and_ordinary_containment(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    kind: str,
) -> None:
    from swarm_arena._agent.manager import LimitReached
    from swarm_arena._exec import job_steps
    from swarm_arena._exec.threads import run_sync_daemon

    job = _job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    run = _completed(ws)
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=10)
    if kind == "terminal":
        error = _core_error(RuntimeError, "terminal test", _category=_SignalCategory.TERMINAL)
    elif kind == "limit":
        error = _core_error(LimitReached, "turns", "limit test")
    else:
        error = RuntimeError(SECRET)

    def fail(*args: Any, **kw: Any) -> ToolOutput:
        raise error

    monkeypatch.setattr(
        job_steps, "command_output_from_result", lambda *args: ToolOutput(status=RunStatus.OK)
    )
    monkeypatch.setattr(
        job_steps, "command_output_from_result" if site == "collect" else "_detach_model", fail
    )
    if kind == "terminal":
        with pytest.raises(RuntimeError, match="terminal test"):
            await run_sync_daemon(runner._collect, run)
    else:
        assert await run_sync_daemon(runner._collect, run) is None
        assert SECRET not in run.stderr_path.read_text()


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_timeout_collection_preserves_terminal_after_process_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    import anyio

    from swarm_arena.executor import LocalExecutor

    collected: list[bool] = []
    started = anyio.Event()

    async def on_started(record: Job.Started) -> None:
        started.set()

    def collect(self: _JobRun, run: CompletedRun) -> ToolOutput:
        collected.append(True)
        raise _core_error(RuntimeError, "terminal test", _category=_SignalCategory.TERMINAL)

    monkeypatch.setattr(_JobRun, "_collect", collect)
    job = _job(
        tmp_path,
        timeout_s=0.2 if stop == "timeout" else 30,
        grace_s=0,
        command=Command(argv=["python", "-c", "import time; time.sleep(60)"]),
        on_started=on_started,
    )
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    task = asyncio.create_task(executor.submit(job))
    try:
        with anyio.fail_after(5):
            if stop == "cancel":
                await started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(RuntimeError, match="terminal test"):
                    await task
    finally:
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
        await executor.aclose()
    assert collected == [True]
    assert list((tmp_path / "ws").iterdir()) == []


@pytest.mark.parametrize("kind", ["terminal", "fenced", "group", "ordinary"])
@pytest.mark.parametrize("cancellation", ["native", "scope", "move_on_after", "fail_after"])
@pytest.mark.parametrize("collection", ["normal", "timeout"])
async def test_caller_cancellation_owns_collector_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    cancellation: str,
    collection: str,
) -> None:
    import threading

    import anyio

    from swarm_arena._exec import job_steps
    from swarm_arena.executor import LocalExecutor

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    partials: list[ToolOutput] = []
    error: BaseException = _core_error(
        RuntimeError, "terminal test", _category=_SignalCategory.TERMINAL
    )
    if kind == "fenced":
        error = _core_error(Fenced, "run ownership lost")
    elif kind == "group":
        error = ExceptionGroup("outer", [ExceptionGroup("inner", [error])])
    elif kind == "ordinary":
        error = RuntimeError(SECRET)

    def collect(*args: Any) -> ToolOutput:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(30)
        raise error

    async def received(output: ToolOutput) -> None:
        partials.append(output)

    monkeypatch.setattr(job_steps, "command_output_from_result", collect)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    scopes: list[anyio.CancelScope] = []
    after_scope: list[bool] = []

    async def submit() -> None:
        if cancellation == "fail_after":
            owner = anyio.fail_after(30)
        elif cancellation == "move_on_after":
            owner = anyio.move_on_after(30)
        else:
            owner = anyio.CancelScope()
        # CancelScope is itself a context manager, while fail_after yields one.
        with owner as scope:
            scopes.append(scope)
            await executor.submit(
                _job(
                    tmp_path,
                    command=Command(
                        argv=["true"]
                        if collection == "normal"
                        else [sys.executable, "-c", "import time; time.sleep(60)"]
                    ),
                    timeout_s=None if collection == "normal" else 0.2,
                    grace_s=0,
                    on_partial=received,
                )
            )
            pytest.fail("caller cancellation became a normal result")
        after_scope.append(True)

    task = asyncio.create_task(submit())
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 10)
        [scope] = scopes
        if cancellation == "native":
            task.cancel()
        elif cancellation == "scope":
            scope.cancel()
        else:
            scope.deadline = anyio.current_time()
        with anyio.fail_after(5):
            release.set()
            if cancellation == "native":
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert not scope.cancelled_caught
            elif cancellation == "fail_after":
                with pytest.raises(TimeoutError):
                    await task
                assert scope.cancelled_caught
            else:
                await task
                assert scope.cancelled_caught
                assert after_scope == [True]
        assert len(partials) == 1
        assert partials[0].status is RunStatus.CANCELLED
        assert partials[0].files == []
        assert partials[0].logs_ref is None
        assert len(workers) == 1
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
        await executor.aclose()


@pytest.mark.parametrize("stop", [False, True])
async def test_collect_thread_start_failure_releases_workspace(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch, stop: bool
) -> None:
    from swarm_arena._exec import threads

    job = _job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    completed = _completed(ws)
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=10, max_output_bytes=10)

    async def run(*args: Any, **kwargs: Any) -> CompletedRun:
        return completed

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("cannot start thread")

    from unittest.mock import Mock

    sup = Mock(run=run)
    monkeypatch.setattr(threads.threading.Thread, "start", fail)
    invocation = runner._before_kill(completed) if stop else runner.run(sup, None)
    with pytest.raises(RuntimeError, match="cannot start thread"):
        await invocation
    assert runner.collect_finished.is_set()


async def test_abandoned_stop_collector_cannot_latch_late_terminal(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from unittest.mock import Mock

    import anyio

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    job = _job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    completed = _completed(ws).model_copy(update={"status": RunStatus.TIMEOUT})
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=10, max_output_bytes=10)

    def collect(self: _JobRun, run: CompletedRun) -> ToolOutput:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(30)
        raise _core_error(RuntimeError, "late terminal", _category=_SignalCategory.TERMINAL)

    async def run(*args: Any, **kwargs: Any) -> CompletedRun:
        callback = asyncio.create_task(kwargs["before_kill"](completed))
        assert await anyio.to_thread.run_sync(entered.wait, 10)
        callback.cancel()
        with pytest.raises(asyncio.CancelledError):
            await callback
        release.set()
        # Model the supervisor's reap window: collector ends after abandonment,
        # but before run hands back TIMEOUT to the owner.
        assert await anyio.to_thread.run_sync(runner.collect_finished.wait, 5)
        return completed

    monkeypatch.setattr(_JobRun, "_collect", collect)
    try:
        with anyio.fail_after(10):
            output = await runner.run(Mock(run=run), None)
        assert output.status is RunStatus.TIMEOUT
        assert not output.data
        assert not output.partial
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()


async def test_abandoned_publication_retains_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import anyio

    from swarm_arena.executor import LocalExecutor

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    workspaces: list[Path] = []
    read = _JobRun._read_log

    def held(self: _JobRun, rel: str) -> bytes:
        workers.append(threading.current_thread())
        workspaces.append(self._workspace)
        entered.set()
        assert release.wait(30)
        (self._workspace / "late-publication").write_text("retained")
        return read(self, rel)

    monkeypatch.setattr(_JobRun, "_read_log", held)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    task = asyncio.create_task(executor.submit(_job(tmp_path, command=Command(argv=["true"]))))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 10)
        task.cancel()
        with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
            await task
        assert workspaces[0].is_dir(), "workspace removed while publication still owns it"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
        await executor.aclose()
    assert len(workers) == 1
    assert (workspaces[0] / "late-publication").read_text() == "retained"


async def test_publication_cancel_before_worker_entry_revokes_workspace_use(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import anyio

    from swarm_arena._exec.job_steps import _WorkspaceCall

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    writes: list[bool] = []
    job = _job(tmp_path)
    ws = tmp_path / "ws"
    runner = _JobRun(
        job, prepare_job(job, ws, artifacts), ws, artifacts, max_log_bytes=10, max_output_bytes=10
    )
    original = _WorkspaceCall.run

    def delayed(self: _WorkspaceCall[Any]) -> Any:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(30)
        return original(self)

    def write() -> None:
        writes.append(True)

    monkeypatch.setattr(_WorkspaceCall, "run", delayed)
    task = asyncio.create_task(runner._workspace_io(write))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 10)
        task.cancel()
        with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
            await task
        assert runner.publication_finished
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(10):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
    assert writes == []


@pytest.mark.parametrize("site", ["normal", "stop"])
async def test_pending_scope_cancel_wins_collector_handback(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    from unittest.mock import Mock

    import anyio

    job = _job(tmp_path)
    ws = tmp_path / "ws"
    runner = _JobRun(
        job, prepare_job(job, ws, artifacts), ws, artifacts, max_log_bytes=10, max_output_bytes=10
    )
    completed = _completed(ws)
    scope = anyio.CancelScope()

    async def handback() -> None:
        scope.cancel()
        raise _core_error(RuntimeError, "terminal test", _category=_SignalCategory.TERMINAL)

    call = Mock(result=handback)
    monkeypatch.setattr(runner, "_start_collect", lambda run: call)

    async def run(*args: Any, **kwargs: Any) -> CompletedRun:
        if site == "stop":
            with anyio.CancelScope(shield=True):
                await kwargs["before_kill"](completed)
        return completed

    with scope:
        await runner.run(Mock(run=run), None)
        pytest.fail("pending cancellation became a collector outcome")
    assert scope.cancelled_caught
