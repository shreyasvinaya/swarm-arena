"""Executor uploads stay inside the caller's supervised storage lifetime."""

import asyncio
import threading
from pathlib import Path

import anyio
import pytest
from _fakes.run import all_events, plan

from swarm_arena import FunctionTool, RunConfig, Swarm
from swarm_arena._journal.lease import read_owner
from swarm_arena._model.canonical import sha256_hex, utc_now
from swarm_arena._store.local import LocalStore


async def test_cancel_abandons_executor_log_storage(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 2.0})
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    publications: list[str] = []
    original = LocalStore.create
    empty_hash = sha256_hex(b"")
    armed = True

    def create(store: LocalStore, key: str, data: bytes) -> str:
        nonlocal armed
        if key == empty_hash:
            publications.append(key)
            if armed:
                armed = False
                workers.append(threading.current_thread())
                entered.set()
                assert release.wait(180)
        return original(store, key, data)

    monkeypatch.setattr(LocalStore, "create", create)
    task = asyncio.create_task(
        Swarm(config).run(
            plan(rounds=1, tool=FunctionTool("_fakes.tools:square", grace_s=0)), run_id="r"
        )
    )
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=30)
        assert done, "caller cancellation still waits for executor artifact storage"
        assert task.cancelled()
        owner = read_owner(LocalStore(Path(config.run_root) / "r"))
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
        head = owner[0].head
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(60):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                if worker.name.startswith("swarm-arena-daemon-"):
                    await anyio.to_thread.run_sync(worker.join, 60)
                    assert not worker.is_alive()
    assert publications == [empty_hash], "abandoned stdout upload started another publication"
    owner = read_owner(LocalStore(Path(config.run_root) / "r"))
    assert owner is not None
    assert owner[0].head == head
    assert not any(event.type == "tool_end" for event in all_events(config))


@pytest.mark.parametrize("site", ["file", "log", "fault"])
async def test_upload_cancellation_never_starts_another_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
) -> None:
    from collections.abc import Iterable

    from _fakes.tools import WRITE, make_job

    from swarm_arena import ScriptTool
    from swarm_arena._model.ref import Ref
    from swarm_arena._model.tooling import Command, RunStatus, ToolOutput, ToolRequest
    from swarm_arena.executor import Job, LocalExecutor

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    publications: list[str] = []
    partials: list[ToolOutput] = []
    create, artifact = LocalStore.create, LocalStore._create_artifact

    def block() -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(180)

    def upload_log(store: LocalStore, key: str, data: bytes) -> str:
        publications.append("log")
        if site != "file" and not entered.is_set():
            block()
        return create(store, key, data)

    def upload_file(store: LocalStore, chunks: Iterable[bytes]) -> tuple[str, int]:
        publications.append("file")
        if site == "file" and not entered.is_set():
            block()
        return artifact(store, chunks)

    async def partial(output: ToolOutput) -> None:
        partials.append(output)

    monkeypatch.setattr(LocalStore, "create", upload_log)
    monkeypatch.setattr(LocalStore, "_create_artifact", upload_file)
    code = (
        "from pathlib import Path; Path('one.txt').write_text('one'); "
        "Path('two.txt').write_text('two'); "
        + WRITE.format("{'data': {'kept': 1}, 'files': ['one.txt', 'two.txt']}")
    )
    job = make_job(
        id="upload",
        kind="tool",
        tool=Ref.of(ScriptTool(command=["python", "-c", code])),
        request=ToolRequest(),
        stage=Job.Stage(),
        artifact_root=str(tmp_path / "artifacts"),
        on_partial=partial,
    )
    if site == "fault":
        job = make_job(
            id="fault",
            command=Command(argv=["true"], cwd="missing"),
            stage=Job.Stage(),
            artifact_root=str(tmp_path / "artifacts"),
            on_partial=partial,
        )
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    task = asyncio.create_task(executor.submit(job))
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        before = list(publications)
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=30)
        assert done, "executor retained a cancelled upload"
        assert task.cancelled()
        if site == "fault":
            assert not partials
        else:
            [output] = partials
            assert output.status == RunStatus.CANCELLED
            assert output.data == {"kept": 1}
            assert output.partial
            assert output.files == []
            assert output.logs_ref is None
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(60):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                if worker.name.startswith("swarm-arena-daemon-"):
                    await anyio.to_thread.run_sync(worker.join, 60)
                    assert not worker.is_alive()
        await executor.aclose()
    assert publications == before


@pytest.mark.parametrize("failure", ["ordinary", "group", "cancel", "fenced"])
async def test_log_upload_contains_backend_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from _fakes.secrecy import assert_secret_absent
    from _fakes.tools import make_job

    from swarm_arena._errors import Fenced, StorageError
    from swarm_arena._model.tooling import Command
    from swarm_arena.executor import Job, LocalExecutor

    secret = "private-upload-sentinel"
    error = {
        "ordinary": OSError(secret),
        "group": BaseExceptionGroup(secret, [asyncio.CancelledError(secret)]),
        "cancel": asyncio.CancelledError(secret),
        "fenced": Fenced(secret),
    }[failure]

    def fail(store: LocalStore, key: str, data: bytes) -> str:
        raise error

    monkeypatch.setattr(LocalStore, "create", fail)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    job = make_job(
        id="failure",
        command=Command(argv=["true"]),
        stage=Job.Stage(),
        artifact_root=str(tmp_path / "artifacts"),
    )
    try:
        with pytest.raises(StorageError) as caught:
            await executor.submit(job)
        assert_secret_absent(caught.value, secret)
    finally:
        await executor.aclose()


async def test_cancel_before_upload_keeps_data_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import Any

    from _fakes.tools import make_job

    from swarm_arena._exec import local_run
    from swarm_arena._model.tooling import Command, RunStatus, ToolOutput
    from swarm_arena.executor import Job, LocalExecutor

    started = anyio.Event()
    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    publications: list[str] = []
    partials: list[ToolOutput] = []
    create = LocalStore.create

    async def on_started(record: Job.Started) -> None:
        started.set()

    async def on_partial(output: ToolOutput) -> None:
        partials.append(output)

    def collect(*args: Any) -> ToolOutput:
        return ToolOutput(status=RunStatus.CANCELLED, data={"kept": 1})

    def upload(store: LocalStore, key: str, data: bytes) -> str:
        publications.append(key)
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(180)
        return create(store, key, data)

    monkeypatch.setattr(local_run._JobRun, "_collect", collect)
    monkeypatch.setattr(LocalStore, "create", upload)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    job = make_job(
        id="partial",
        command=Command(argv=["python", "-c", "import time; time.sleep(60)"]),
        stage=Job.Stage(),
        artifact_root=str(tmp_path / "artifacts"),
        grace_s=0,
        on_started=on_started,
        on_partial=on_partial,
    )
    task = asyncio.create_task(executor.submit(job))
    try:
        with anyio.fail_after(60):
            await started.wait()
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=30)
        assert done, "cancelled executor started a partial upload"
        assert task.cancelled()
        [output] = partials
        assert output.data == {"kept": 1}
        assert output.partial
        assert output.files == []
        assert output.logs_ref is None
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(60):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
                assert not worker.is_alive()
        await executor.aclose()
    assert publications == []


@pytest.mark.parametrize("stop", ["cancel", "fence"])
async def test_public_run_abandons_partial_uploads(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    from _fakes.leases import force_take_over
    from _fakes.run import WRITE_THEN_SLEEP, wait_tool_started
    from _fakes.tools import PyTool

    from swarm_arena._errors import Fenced
    from swarm_arena._journal.writer import JournalWriter

    config = run_config.model_copy(update={"lease_ttl_s": 3, "settle_timeout_s": 2.0})
    release = threading.Event()
    fenced = anyio.Event()
    workers: list[threading.Thread] = []
    publications: list[str] = []
    create, fail = LocalStore.create, JournalWriter._fail

    def upload(store: LocalStore, key: str, data: bytes) -> str:
        if len(key) == 64:
            publications.append(key)
            workers.append(threading.current_thread())
            assert release.wait(180)
        return create(store, key, data)

    def record_failure(writer: JournalWriter, error: Exception) -> None:
        fail(writer, error)
        if isinstance(error, Fenced):
            fenced.set()

    monkeypatch.setattr(LocalStore, "create", upload)
    monkeypatch.setattr(JournalWriter, "_fail", record_failure)
    task = asyncio.create_task(
        Swarm(config).run(
            plan(rounds=1, tool=PyTool({"code": WRITE_THEN_SLEEP, "grace_s": 0})), run_id="r"
        )
    )
    successor = None
    try:
        store = await wait_tool_started(config)
        if stop == "cancel":
            task.cancel()
        else:
            successor = await anyio.to_thread.run_sync(
                lambda: force_take_over(store, owner_id="successor", host="test", ttl_s=60)
            )
            with anyio.fail_after(60):
                await fenced.wait()
        done, _ = await asyncio.wait([task], timeout=30)
        assert done, "partial upload escaped the owning run's lifetime"
        if stop == "cancel":
            assert task.cancelled()
        else:
            assert isinstance(task.exception(), Fenced)
        owner = read_owner(store)
        assert owner is not None
        if stop == "cancel":
            assert owner[0].expires_at <= utc_now()
        else:
            assert successor is not None
            assert owner[0].owner_id == successor.doc.owner_id
        head = owner[0].head
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with anyio.fail_after(60):
            await asyncio.gather(task, return_exceptions=True)
            for worker in workers:
                await anyio.to_thread.run_sync(worker.join, 60)
                assert not worker.is_alive()
        if successor is not None:
            successor.release()
    assert publications == [], "an upload started after cancellation or confirmed fencing"
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head == head
    assert not any(event.type == "tool_end" for event in all_events(config))
