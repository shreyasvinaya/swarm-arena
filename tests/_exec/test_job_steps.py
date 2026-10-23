"""Shared executor preparation, transport restoration and publication contracts."""

import asyncio
import sys
from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.cleanup import observe_task, settle_tasks
from _fakes.proc import wait_for_file
from _fakes.process_cleanup import owned_processes  # noqa: F401
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

pytestmark = pytest.mark.usefixtures("owned_processes")

EDITS = {"p.txt": "hello", "q.txt": "sha256:hello"}
SECRET = "f810e652-43d2-4555-8280-90e395b4ee31"


def _job(tmp_path: Path, **fields: Any) -> Job:
    return make_job(
        artifact_root=str(tmp_path / "artifacts"),
        **{"command": Command(argv=["python", "-c", "pass"]), **fields},
    )


@pytest.fixture
async def timeout_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Path]:
    """Arm the short execution timer only after the child publishes readiness."""
    from swarm_arena._exec import supervisor

    ready = tmp_path / "child-ready"
    waiting = anyio.Event()
    scopes: list[anyio.CancelScope] = []

    class ReadyTimeout:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        @contextmanager
        def move_on_after(
            self, delay: float | None, shield: bool = False
        ) -> Generator[anyio.CancelScope]:
            if delay == 0.2 and not scopes:
                with anyio.move_on_after(None, shield=shield) as scope:
                    scopes.append(scope)
                    waiting.set()
                    yield scope
            else:
                with anyio.move_on_after(delay, shield=shield) as scope:
                    yield scope

    async def arm() -> None:
        await waiting.wait()
        await wait_for_file(ready, 60)
        scopes[0].deadline = anyio.current_time() + 0.2

    monkeypatch.setattr(supervisor, "anyio", ReadyTimeout())
    async with anyio.create_task_group() as group:
        group.start_soon(arm)
        try:
            yield ready
        finally:
            group.cancel_scope.cancel()


def _held_child(ready: Path) -> Command:
    """Keep a ready child alive until process cleanup, with a finite backstop."""
    return Command(
        argv=[
            sys.executable,
            "-c",
            f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(300)",
        ]
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
        assert release.wait(300)
        return create(key, data)

    monkeypatch.setattr(backend, "create", publish)
    task = asyncio.create_task(runner._finish(ToolOutput(status=RunStatus.OK, files=files), None))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        task.cancel()
        with anyio.fail_after(30), pytest.raises(asyncio.CancelledError):
            await observe_task(task)
        assert published == [b"one"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        try:
            await settle_tasks(task)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
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
        assert release.wait(300)
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
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        task.cancel()
        with anyio.fail_after(30), pytest.raises(asyncio.CancelledError):
            await observe_task(task)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        try:
            await settle_tasks(task)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
                assert not worker.is_alive()


@pytest.mark.parametrize("stop", ["normal", "cancel", "timeout"])
async def test_abandoned_collector_retains_its_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
    timeout_ready: Path,
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
        assert release.wait(300)
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
            command=_held_child(timeout_ready),
        )
    executor = LocalExecutor(workspace_root=str(tmp_path / "workspaces"))
    task = asyncio.create_task(executor.submit(job))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        with anyio.fail_after(30):
            if stop == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observe_task(task)
            else:
                output = await observe_task(task)
                assert output.status == (
                    RunStatus.TIMEOUT if stop == "timeout" else RunStatus.ERROR
                )
        assert calls == [True]
        assert workspaces[0].is_dir(), "workspace removed while an abandoned collect still uses it"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        try:
            await settle_tasks(task)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
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
        job_steps, "command_output_from_result" if site == "collect" else "detach_tool_output", fail
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
    timeout_ready: Path,
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
        timeout_s=0.2 if stop == "timeout" else None,
        grace_s=0,
        command=_held_child(timeout_ready),
        on_started=on_started,
    )
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    task = asyncio.create_task(executor.submit(job))
    try:
        with anyio.fail_after(60):
            await started.wait()
            await wait_for_file(timeout_ready, 60)
        with anyio.fail_after(30):
            if stop == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observe_task(task)
            else:
                with pytest.raises(RuntimeError, match="terminal test"):
                    await observe_task(task)
    finally:
        if not task.done():
            task.cancel()
        try:
            await settle_tasks(task)
        finally:
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
    timeout_ready: Path,
) -> None:
    import threading

    import anyio

    from swarm_arena._exec import job_steps
    from swarm_arena.executor import LocalExecutor

    entered, finished = anyio.Event(), anyio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
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
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(300)
        raise error

    async def received(output: ToolOutput) -> None:
        partials.append(output)

    monkeypatch.setattr(job_steps, "command_output_from_result", collect)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    scopes: list[anyio.CancelScope] = []
    after_scope: list[bool] = []

    async def submit() -> None:
        if cancellation == "fail_after":
            owner = anyio.fail_after(300)
        elif cancellation == "move_on_after":
            owner = anyio.move_on_after(300)
        else:
            owner = anyio.CancelScope()
        # CancelScope is itself a context manager, while fail_after yields one.
        with owner as scope:
            scopes.append(scope)
            await executor.submit(
                _job(
                    tmp_path,
                    command=Command(argv=["true"])
                    if collection == "normal"
                    else _held_child(timeout_ready),
                    timeout_s=None if collection == "normal" else 0.2,
                    grace_s=0,
                    on_partial=received,
                )
            )
            pytest.fail("caller cancellation became a normal result")
        after_scope.append(True)

    task = asyncio.create_task(submit())
    task.add_done_callback(lambda _: finished.set())
    try:
        with anyio.fail_after(60):
            await entered.wait()
        [scope] = scopes
        if cancellation == "native":
            task.cancel()
        elif cancellation == "scope":
            scope.cancel()
        else:
            scope.deadline = anyio.current_time()
        release.set()
        # A watchdog around `await task` would send it a second native cancel.
        # Await only notification; inspect the job's own outcome outside the timer.
        with anyio.fail_after(60):
            await finished.wait()
        if cancellation == "native":
            with pytest.raises(asyncio.CancelledError):
                task.result()
            assert not scope.cancelled_caught
        elif cancellation == "fail_after":
            with pytest.raises(TimeoutError):
                task.result()
            assert scope.cancelled_caught
        else:
            task.result()
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
        try:
            await settle_tasks(task)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
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
        assert release.wait(300)
        raise _core_error(RuntimeError, "late terminal", _category=_SignalCategory.TERMINAL)

    async def run(*args: Any, **kwargs: Any) -> CompletedRun:
        callback = asyncio.create_task(kwargs["before_kill"](completed))
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        callback.cancel()
        with pytest.raises(asyncio.CancelledError):
            await callback
        release.set()
        # Model the supervisor's reap window: collector ends after abandonment,
        # but before run hands back TIMEOUT to the owner.
        assert await anyio.to_thread.run_sync(runner.collect_finished.wait, 60)
        return completed

    monkeypatch.setattr(_JobRun, "_collect", collect)
    try:
        with anyio.fail_after(30):
            output = await runner.run(Mock(run=run), None)
        assert output.status is RunStatus.TIMEOUT
        assert not output.data
        assert not output.partial
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
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
        assert release.wait(300)
        (self._workspace / "late-publication").write_text("retained")
        return read(self, rel)

    monkeypatch.setattr(_JobRun, "_read_log", held)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    task = asyncio.create_task(executor.submit(_job(tmp_path, command=Command(argv=["true"]))))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        task.cancel()
        with anyio.fail_after(30), pytest.raises(asyncio.CancelledError):
            await observe_task(task)
        assert workspaces[0].is_dir(), "workspace removed while publication still owns it"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        try:
            await settle_tasks(task)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
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
        assert release.wait(300)
        return original(self)

    def write() -> None:
        writes.append(True)

    monkeypatch.setattr(_WorkspaceCall, "run", delayed)
    task = asyncio.create_task(runner._workspace_io(write))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        task.cancel()
        with anyio.fail_after(30), pytest.raises(asyncio.CancelledError):
            await observe_task(task)
        assert runner.publication_finished
    finally:
        release.set()
        if not task.done():
            task.cancel()
        try:
            await settle_tasks(task)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
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


async def test_run_coroutine_closes_without_awaiting_generator_exit(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    from unittest.mock import Mock

    job = _job(tmp_path)
    ws = tmp_path / "ws"
    runner = _JobRun(
        job, prepare_job(job, ws, artifacts), ws, artifacts, max_log_bytes=10, max_output_bytes=10
    )

    async def suspended(*args: Any, **kwargs: Any) -> CompletedRun:
        await asyncio.sleep(0)
        pytest.fail("supervisor resumed after coroutine close")

    invocation = runner.run(Mock(run=suspended), None)
    try:
        assert invocation.send(None) is None
        invocation.close()
    finally:
        invocation.close()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cancellation", ["native", "scope"])
async def test_process_control_error_passes_through_pending_cancel(
    tmp_path: Path, artifacts: ArtifactStore, error_type: type[BaseException], cancellation: str
) -> None:
    import contextlib
    from unittest.mock import AsyncMock, Mock

    import anyio

    job = _job(tmp_path)
    ws = tmp_path / "ws"
    runner = _JobRun(
        job, prepare_job(job, ws, artifacts), ws, artifacts, max_log_bytes=10, max_output_bytes=10
    )
    error = error_type("process control")
    failure_handler = AsyncMock(wraps=runner._raise_failure)
    runner._raise_failure = failure_handler
    caught: list[BaseException] = []

    async def invoke() -> None:
        with anyio.CancelScope() as scope:

            async def interrupted(*args: Any, **kwargs: Any) -> CompletedRun:
                if cancellation == "native":
                    task = asyncio.current_task()
                    assert task is not None
                    task.cancel()
                else:
                    scope.cancel()
                raise error

            try:
                await runner.run(Mock(run=interrupted), None)
            except BaseException as propagated:
                caught.append(propagated)
            finally:
                # Consume the pending cancellation within its owner task/scope.
                with contextlib.suppress(asyncio.CancelledError):
                    await anyio.lowlevel.checkpoint()

    await asyncio.create_task(invoke())
    assert caught == [error]
    assert caught[0] is error
    failure_handler.assert_not_awaited()


def _report_job(tmp_path: Path, *, code: str = "pass") -> Job:
    """Build a real tool job for optional report collection regressions."""
    return _job(
        tmp_path,
        kind="tool",
        tool=Ref.of(PyTool({"code": code})),
        request=ToolRequest(),
        command=None,
    )


@pytest.mark.parametrize("size", [17, 17 * 1024 * 1024, 1024 * 1024 * 1024 + 1])
async def test_upload_outputs_sets_llm_calls_file(
    tmp_path: Path, artifacts: ArtifactStore, size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _report_job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    run = _completed(ws)
    report = ws / "swarm/llm_calls.jsonl"
    with report.open("wb") as stream:
        stream.truncate(size)
    puts: list[bytes] = []
    put = artifacts.put_bytes

    def capture(data: bytes) -> str:
        puts.append(data)
        return put(data)

    monkeypatch.setattr(artifacts, "put_bytes", capture)
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=1024)
    output = await runner._finish(
        ToolOutput(status=RunStatus.OK, data={"valid": True}), (run.started_at, run.ended_at)
    )
    assert output.data == {"valid": True}
    assert output.status == RunStatus.OK
    if size > 1024 * 1024 * 1024:
        assert output.llm_calls_file is None
        assert output._collection_diagnostics == (("llm_calls_hash_limit", 1),)
    else:
        assert output.llm_calls_file == FileRef(
            name="llm_calls.jsonl",
            path="llm_calls.jsonl",
            sha256=sha256_hex(b"\0" * size),
            size=size,
        )
    assert len(puts) == (3 if size <= 16 * 1024 * 1024 else 2)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "missing"])
async def test_optional_report_refusal_preserves_result(
    tmp_path: Path, artifacts: ArtifactStore, kind: str
) -> None:
    import os

    job = _report_job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    run = _completed(ws)
    path = ws / "swarm/llm_calls.jsonl"
    if kind == "symlink":
        path.symlink_to(ws / "missing")
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=1024)
    output = await runner._finish(ToolOutput(status=RunStatus.OK, data={"valid": True}), None)
    assert output.status == RunStatus.OK
    assert output.data == {"valid": True}
    assert output.llm_calls_file is None
    assert bool(output._collection_diagnostics) == (kind != "missing")
    assert "diagnostic" not in output.model_dump_json()
    assert "diagnostic" not in repr(output)
    assert run.status == RunStatus.OK


async def test_cancelled_report_read_never_starts_upload(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.llm_run import HeldOperation

    from swarm_arena._exec import job_steps

    job = _report_job(tmp_path)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    _completed(ws)
    (ws / "swarm/llm_calls.jsonl").write_bytes(b'{"model":"a/b"}')
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=1024)
    held = HeldOperation()
    collect = job_steps.collect_llm_calls
    puts: list[bytes] = []
    put = artifacts.put_bytes

    def hold(workspace: Path) -> Any:
        held.wait()
        return collect(workspace)

    def capture(data: bytes) -> str:
        puts.append(data)
        return put(data)

    monkeypatch.setattr(job_steps, "collect_llm_calls", hold)
    monkeypatch.setattr(artifacts, "put_bytes", capture)
    task = asyncio.create_task(runner._finish(ToolOutput(status=RunStatus.OK), None))
    try:
        await held.ready()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await held.finish()
        assert puts == [b"o", b"e"]
    finally:
        held.release.set()
        task.cancel()
        try:
            await settle_tasks(task)
        finally:
            await held.finish()


async def test_partial_delivery_strips_report_reference(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    received: list[ToolOutput] = []

    async def partial(output: ToolOutput) -> None:
        received.append(output)

    job = _job(tmp_path, on_partial=partial)
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=1024)
    runner._output = ToolOutput(
        status=RunStatus.OK,
        data={"keep": True},
        llm_calls_file=FileRef(
            name="llm_calls.jsonl",
            path="llm_calls.jsonl",
            sha256="0" * 64,
            size=1,
        ),
    )
    await runner._deliver_partial()
    assert received[0].data == {"keep": True}
    assert received[0].llm_calls_file is None
    assert received[0].files == []
    assert received[0].logs_ref is None


@pytest.mark.parametrize("site", ["read", "upload"])
async def test_abandoned_report_operation_retains_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    import threading

    import anyio

    from swarm_arena._exec import job_steps
    from swarm_arena.executor import LocalExecutor

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    workspaces: list[Path] = []
    collect = job_steps.collect_llm_calls
    upload = _JobRun._upload_report

    def hold(workspace: Path) -> None:
        workers.append(threading.current_thread())
        workspaces.append(workspace)
        entered.set()
        assert release.wait(300)
        (workspace / "late-report").write_text("retained")

    def held_read(workspace: Path) -> Any:
        hold(workspace)
        return collect(workspace)

    def held_upload(self: _JobRun, data: bytes, expected: str) -> None:
        hold(self._workspace)
        upload(self, data, expected)

    if site == "read":
        monkeypatch.setattr(job_steps, "collect_llm_calls", held_read)
    else:
        monkeypatch.setattr(_JobRun, "_upload_report", held_upload)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    command = Command(
        argv=[
            "python",
            "-c",
            "from pathlib import Path; Path('swarm/llm_calls.jsonl').write_text('{}')",
        ]
    )
    task = asyncio.create_task(executor.submit(_report_job(tmp_path, code=command.argv[2])))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        task.cancel()
        with anyio.fail_after(30), pytest.raises(asyncio.CancelledError):
            await observe_task(task)
        assert workspaces[0].is_dir(), "workspace removed while report operation still owns it"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        try:
            await settle_tasks(task)
        finally:
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
                assert not worker.is_alive()
            await executor.aclose()
    assert len(workers) == 1
    assert (workspaces[0] / "late-report").read_text() == "retained"


async def test_command_jobs_do_not_collect_optional_llm_report(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._exec import job_steps

    job = _job(tmp_path)
    assert job.kind == "command"
    ws = tmp_path / "ws"
    staged = prepare_job(job, ws, artifacts)
    _completed(ws)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("command/check jobs must not inspect an LLM report")

    monkeypatch.setattr(job_steps, "collect_llm_calls", forbidden)
    runner = _JobRun(job, staged, ws, artifacts, max_log_bytes=1024, max_output_bytes=1024)
    output = await runner._finish(ToolOutput(status=RunStatus.OK), None)
    assert output.llm_calls_file is None
