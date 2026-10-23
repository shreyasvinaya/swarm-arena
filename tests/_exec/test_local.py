import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.proc import pid_alive, reap, require_visible_proc, spawn_group, wait_until
from _fakes.read_counts import count_reads
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import (
    COLLECTING,
    RELEASE,
    WRITE,
    BadCollectTool,
    BlockingCollectTool,
    GhostFileTool,
    OkCollectTool,
    PyTool,
    SlowCollectTool,
    UnreadFilesTool,
    make_job,
)
from pydantic import ValidationError

from swarm_arena._errors import NotFound, StorageError
from swarm_arena._exec import local_run, local_stage, local_workspace, supervisor
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._exec.procinfo import host_id, process_start_time
from swarm_arena._exec.supervisor import ProcessSupervisor, RLimits
from swarm_arena._model.canonical import sha256_hex, utc_now
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Ref
from swarm_arena._model.tooling import (
    Command,
    CompletedRun,
    FileRef,
    RunStatus,
    ToolOutput,
    ToolRequest,
)
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store


def tool_job(
    tmp_path: Path,
    code: str,
    *,
    tool_cls: type[PyTool] = PyTool,
    tool_timeout_in_config: float | None = None,
    **kw: Any,
) -> Job:
    fields: dict[str, Any] = {
        "id": "c-1",
        "kind": "tool",
        "tool": Ref.of(tool_cls({"code": code, "timeout_s": tool_timeout_in_config})),
        "command": None,
        "request": ToolRequest(),
        "stage": Job.Stage(),
        "timeout_s": 60.0,
        "grace_s": 1.0,
        "artifact_root": str(tmp_path / "artifacts"),
    }
    return make_job(**{**fields, **kw})


def command_job(tmp_path: Path, command: Command, **kw: Any) -> Job:
    fields: dict[str, Any] = {
        "id": "k-1",
        "command": command,
        "timeout_s": 60.0,
        "artifact_root": str(tmp_path / "artifacts"),
    }
    return make_job(**{**fields, **kw})


def blocking_job(tmp_path: Path, code: str, **kw: Any) -> Job:
    return tool_job(tmp_path, code, tool_cls=BlockingCollectTool, **kw)


def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(open_store(str(tmp_path / "artifacts")))


def log(tmp_path: Path, out: ToolOutput, which: int) -> bytes:  # 1 = stdout, 2 = stderr
    assert out.logs_ref is not None
    return store(tmp_path).get_bytes(out.logs_ref.split(":")[which])


@pytest.fixture
def kept_executor(tmp_path: Path) -> LocalExecutor:
    return LocalExecutor(workspace_root=str(tmp_path / "ws"), keep_workspaces=True)


def _workspace_log(tmp_path: Path, which: int) -> bytes:
    """Inspect local diagnostics when cancellation has ended artifact publication."""
    [workspace] = (tmp_path / "ws").iterdir()
    return (workspace / "swarm" / ("stdout.log" if which == 1 else "stderr.log")).read_bytes()


async def test_tool_never_runs_when_start_commit_fails(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:  # exit criterion
    async def fail(s: Job.Started) -> None:
        raise RuntimeError("commit failed")

    marker = tmp_path / "ran"
    with pytest.raises(RuntimeError, match="commit failed"):
        await local_executor.submit(
            tool_job(tmp_path, f"open({str(marker)!r}, 'w')", on_started=fail)
        )
    await anyio.sleep(1)
    assert not marker.exists()
    assert list((tmp_path / "ws").iterdir()) == []


async def test_on_started_gets_the_job_id(local_executor: LocalExecutor, tmp_path: Path) -> None:
    seen: list[Job.Started] = []

    async def record(s: Job.Started) -> None:
        seen.append(s)

    out = await local_executor.submit(
        tool_job(tmp_path, WRITE.format("{'data': {}}"), on_started=record)
    )
    [s] = seen
    assert (out.status, s.job_id, s.kind, s.host_id, s.pid == s.pgid) == (
        RunStatus.OK,
        "c-1",
        "process",
        host_id(),
        True,
    )


async def test_runs_a_tool_and_uploads_logs(local_executor: LocalExecutor, tmp_path: Path) -> None:
    out = await local_executor.submit(
        tool_job(tmp_path, "print('hello'); " + WRITE.format("{'data': {'f': 1.5}}"))
    )
    assert (out.status, out.data) == (RunStatus.OK, {"f": 1.5})
    artifacts = store(tmp_path)
    assert out.logs_ref is not None
    _, out_h, err_h = out.logs_ref.split(":")
    assert out.logs_ref.startswith("sha256:")
    assert artifacts.get_bytes(out_h) == b"hello\n"
    assert artifacts.get_bytes(err_h) == b""
    assert out.started_at is not None
    assert out.ended_at is not None
    assert out.started_at <= out.ended_at


async def test_python_argv0_replaced(local_executor: LocalExecutor, tmp_path: Path) -> None:
    code = WRITE.format("{'exe': __import__('sys').executable}")
    out = await local_executor.submit(command_job(tmp_path, Command(argv=["python", "-c", code])))
    assert out.data["exe"] == sys.executable
    # Framework Python on macOS resolves interpreter symlinks in sys.executable.
    # Record invocation through the configured path instead of assuming its spelling survives.
    marker = tmp_path / "python-called"
    interpreter = tmp_path / "custom-python"
    interpreter.write_text(
        f"#!/bin/sh\nprintf called > {shlex.quote(str(marker))}\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    interpreter.chmod(0o755)
    link = tmp_path / "py"
    link.symlink_to(interpreter)
    out2 = await local_executor.submit(
        command_job(tmp_path, Command(argv=["python", "-c", code], python=str(link)))
    )
    assert marker.read_text() == "called"
    assert out2.data["exe"] == sys.executable


async def test_command_cwd_is_inside_workspace(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:
    artifacts = store(tmp_path)
    digest = artifacts.put_bytes(b"x")
    code = WRITE.format("{'cwd': os.path.basename(os.getcwd())}")
    job = command_job(
        tmp_path,
        Command(argv=["python", "-c", code], cwd="problem"),
        stage=Job.Stage(root={"problem/input.txt": digest}),
    )
    assert (await local_executor.submit(job)).data == {"cwd": "problem"}


async def test_job_timeout_wins_and_partial_kept(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:
    # OkCollectTool's output does not depend on how fast the tool starts, so the 1 s
    # timeout cannot race start-up under load; the data shows partial is kept.
    with anyio.fail_after(30):
        out = await local_executor.submit(
            tool_job(
                tmp_path,
                "import time; time.sleep(180)",
                tool_cls=OkCollectTool,
                timeout_s=1.0,
                tool_timeout_in_config=100.0,
            )
        )
    assert (out.status, out.partial, out.data) == (RunStatus.TIMEOUT, True, {"k": 1})


async def test_cancel_delivers_partial_through_on_partial(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:
    got: list[ToolOutput] = []

    async def on_partial(o: ToolOutput) -> None:
        got.append(o)

    code = (
        WRITE.format("{'data': {'step': 2}}") + "; open('ready', 'w'); import time; time.sleep(60)"
    )
    job = tool_job(tmp_path, code, on_partial=on_partial)

    async def ready() -> None:
        deadline = time.monotonic() + 60
        while not [p async for p in anyio.Path(tmp_path / "ws").glob("*/ready")]:
            if time.monotonic() >= deadline:
                raise TimeoutError("ready did not appear")
            await anyio.sleep(0.05)

    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(local_executor.submit, job)
            await ready()
            tg.cancel_scope.cancel()
    assert [(o.status, o.partial, o.data) for o in got] == [
        (RunStatus.CANCELLED, True, {"step": 2})
    ]
    assert got[0].logs_ref is None
    assert got[0].files == []
    assert list((tmp_path / "ws").iterdir()) == []


async def test_output_files_uploaded_or_error(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:
    code = (
        "import os; os.makedirs('out'); open('out/p.txt', 'w').write('payload'); "
        + WRITE.format("{'data': {}, 'files': ['out/p.txt']}")
    )
    out = await local_executor.submit(tool_job(tmp_path, code))
    [ref] = out.files
    assert (out.status, ref.path, ref.size) == (RunStatus.OK, "out/p.txt", 7)
    assert store(tmp_path).get_bytes(ref.sha256) == b"payload"

    missing = WRITE.format("{'data': {}, 'files': ['out/missing.txt']}")
    bad = await local_executor.submit(tool_job(tmp_path, missing))
    assert (bad.status, bad.files) == (RunStatus.ERROR, [])

    ghost = await local_executor.submit(tool_job(tmp_path, "pass", tool_cls=GhostFileTool))
    assert (ghost.status, ghost.files) == (RunStatus.ERROR, [])
    assert b"out/ghost.txt is missing" in log(tmp_path, ghost, 2)


async def test_env_allowlist(
    local_executor: LocalExecutor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VISIBLE", "v")
    monkeypatch.setenv("HIDDEN", "h")
    code = WRITE.format(
        "{k: os.environ.get(k) for k in ('VISIBLE', 'HIDDEN', 'SWARM_ARENA_RUN_ID')}"
    )
    out = await local_executor.submit(
        command_job(
            tmp_path,
            Command(argv=["python", "-c", code], env=["VISIBLE"]),
            env={"SWARM_ARENA_RUN_ID": "local"},
        )
    )
    assert out.data == {"VISIBLE": "v", "HIDDEN": None, "SWARM_ARENA_RUN_ID": "local"}


async def test_workspaces_removed_unless_kept(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    code = (  # a read-only tree must not stop the cleanup
        "import os; os.makedirs('ro/sub'); open('ro/sub/f', 'w').close(); "
        "os.chmod('ro/sub', 0o500); os.chmod('ro', 0o500); " + WRITE.format("{'data': {}}")
    )
    out = await LocalExecutor(workspace_root=str(ws)).submit(tool_job(tmp_path, code))
    assert out.status is RunStatus.OK
    assert list(ws.iterdir()) == []

    kept = tmp_path / "kept"
    job = tool_job(tmp_path, WRITE.format("{'data': {}}"))
    out = await LocalExecutor(workspace_root=str(kept), keep_workspaces=True).submit(job)
    [workspace] = list(kept.iterdir())
    assert out.status is RunStatus.OK
    assert workspace.name.startswith(f"{job.id}-")
    assert (workspace / "swarm" / "stdout.log").is_file()


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o720])
async def test_unsafe_workspace_root_refused(tmp_path: Path, mode: int) -> None:
    ws = tmp_path / "shared"
    ws.mkdir()
    ws.chmod(mode)
    with pytest.raises(RuntimeError, match="workspace root"):
        await LocalExecutor(workspace_root=str(ws)).submit(tool_job(tmp_path, "pass"))
    assert list(ws.iterdir()) == []
    ws.chmod(0o1000 | mode)  # sticky, like /tmp: other users cannot rename or remove our entries
    job = tool_job(tmp_path, WRITE.format("{'data': {}}"))
    assert (await LocalExecutor(workspace_root=str(ws)).submit(job)).status is RunStatus.OK


async def test_workspace_root_parent_must_be_safe(tmp_path: Path) -> None:
    parent = tmp_path / "open"
    parent.mkdir()
    (parent / "ws").mkdir(mode=0o700)
    parent.chmod(0o775)  # a group member could rename ws and put their own directory there
    with pytest.raises(RuntimeError, match="workspace root"):
        await LocalExecutor(workspace_root=str(parent / "ws")).submit(tool_job(tmp_path, "pass"))
    assert list((parent / "ws").iterdir()) == []
    parent.chmod(0o755)
    job = tool_job(tmp_path, WRITE.format("{'data': {}}"))
    out = await LocalExecutor(workspace_root=str(parent / "ws")).submit(job)
    assert out.status is RunStatus.OK


async def test_default_workspace_root_is_per_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    job = tool_job(tmp_path, WRITE.format("{'data': {}}"))
    out = await LocalExecutor(keep_workspaces=True).submit(job)
    assert out.status is RunStatus.OK
    [workspace] = list((tmp_path / f"swarm-arena-{os.getuid()}").iterdir())
    assert workspace.name.startswith("c-1-")


async def test_job_id_is_sanitized_in_the_workspace_name(tmp_path: Path) -> None:
    root = tmp_path / "kept"
    ex = LocalExecutor(workspace_root=str(root), keep_workspaces=True)
    for job_id in ("../escaped", "x" * 100):
        out = await ex.submit(tool_job(tmp_path, WRITE.format("{'data': {}}"), id=job_id))
        assert out.status is RunStatus.OK
    names = sorted(p.name for p in root.iterdir())
    assert len(names) == 2
    assert names[0].startswith(".._escaped-")
    assert names[1].startswith("x" * 64 + "-")
    assert not names[1].startswith("x" * 65)
    assert all((root / name).parent == root for name in names)
    assert sorted(os.listdir(tmp_path)) == ["artifacts", "kept"]


async def test_collect_exception_becomes_error(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:
    out = await local_executor.submit(tool_job(tmp_path, "pass", tool_cls=BadCollectTool))
    assert out.status is RunStatus.ERROR
    assert log(tmp_path, out, 2) == b"swarm-arena: collect failed: RuntimeError\n"
    assert out.started_at is not None  # the executor stamps the run times itself
    assert out.ended_at is not None
    assert out.started_at <= out.ended_at


@pytest.mark.parametrize("fifo_stderr", [False, True])
async def test_blocked_collect_cannot_stop_kill(
    local_executor: LocalExecutor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fifo_stderr: bool,
) -> None:
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.5)
    pids: list[int] = []

    async def ready(started: Job.Started) -> None:
        # Everything the test needs is in place before "go", so before the timeout clock
        # starts: the launcher execs the tool in place, so its pid is the tool's pid, and
        # the stderr log is swapped for a FIFO here instead of by the tool after start-up.
        assert started.pid is not None
        pids.append(started.pid)
        if fifo_stderr:
            [workspace] = list((tmp_path / "ws").iterdir())
            (workspace / "swarm/stderr.log").unlink()
            os.mkfifo(workspace / "swarm/stderr.log")

    code = "import time; time.sleep(60)"
    RELEASE.clear()  # a set event from an earlier case would not block collect
    collect = BlockingCollectTool.collect
    entered = threading.Event()
    workers: list[threading.Thread] = []

    def held_collect(*args: Any, **kwargs: Any) -> ToolOutput:
        workers.append(threading.current_thread())
        entered.set()
        RELEASE.wait()
        return collect(*args, **kwargs)

    monkeypatch.setattr(BlockingCollectTool, "collect", held_collect)
    task = asyncio.create_task(
        local_executor.submit(
            blocking_job(tmp_path, code, timeout_s=1.0, grace_s=0.2, on_started=ready)
        )
    )
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        done, _ = await asyncio.wait([task], timeout=30)
        assert done, "kill waited for the held collection worker"
        assert not RELEASE.is_set()
        out = task.result()
        assert (out.status, out.partial) == (RunStatus.TIMEOUT, False)
        [pid] = pids
        assert wait_until(lambda: not pid_alive(pid), 60)
        if not fifo_stderr:
            assert b"before_kill timed out" in log(tmp_path, out, 2)
            assert b"no partial output" in log(tmp_path, out, 2)
            assert out.started_at is not None
            assert out.ended_at is not None
            assert out.started_at <= out.ended_at
        else:
            assert log(tmp_path, out, 2) == b""
    finally:
        RELEASE.set()
        if not task.done():
            task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 60)
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 60)
            assert not worker.is_alive()


async def test_fifo_log_does_not_hang(local_executor: LocalExecutor, tmp_path: Path) -> None:
    code = (
        "import os; os.remove('swarm/stdout.log'); os.mkfifo('swarm/stdout.log'); "
        + WRITE.format("{'data': {}}")
    )
    with anyio.fail_after(60):
        out = await local_executor.submit(tool_job(tmp_path, code))
    assert out.logs_ref is not None
    assert out.logs_ref.split(":")[1] == sha256_hex(b"")
    assert b"swarm/stdout.log is not a regular file" in log(tmp_path, out, 2)


async def test_fifo_stderr_with_invalid_result_does_not_hang(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:
    code = (
        "import os; os.remove('swarm/stderr.log'); os.mkfifo('swarm/stderr.log')"  # no result.json
    )
    with anyio.fail_after(60):  # the "missing result" note goes through note_in_log: dropped
        out = await local_executor.submit(tool_job(tmp_path, code))
    assert out.status is RunStatus.ERROR
    assert out.logs_ref is not None
    assert out.logs_ref.split(":")[2] == sha256_hex(b"")


async def test_command_job_without_on_started_runs(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:
    job = make_job(
        command=Command(argv=["python", "-c", WRITE.format("{'passed': True}")]),
        artifact_root=str(tmp_path / "artifacts"),
    )
    assert job.on_started is None
    out = await local_executor.submit(job)
    assert (out.status, out.data) == (RunStatus.OK, {"passed": True})


async def test_resources_and_concurrency(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    big = tool_job(tmp_path, f"open({str(marker)!r}, 'w')", resources=Resources(cpus=10_000))
    with pytest.raises(ValueError, match="cpus"):
        await LocalExecutor(workspace_root=str(tmp_path / "ws")).submit(big)
    assert not marker.exists()
    assert not (tmp_path / "ws").exists()

    ex = LocalExecutor(workspace_root=str(tmp_path / "ws"), max_concurrent=1)
    code = "import time; s = time.time(); time.sleep(0.3); e = time.time(); " + WRITE.format(
        "{'data': {'s': s, 'e': e}}"
    )
    outs: list[ToolOutput] = []

    async def one(i: int) -> None:
        outs.append(await ex.submit(tool_job(tmp_path, code, id=f"c-{i}")))

    async with anyio.create_task_group() as tg:
        for i in range(3):
            tg.start_soon(one, i)
    spans = sorted((o.data["s"], o.data["e"]) for o in outs)
    assert len(spans) == 3
    assert all(o.status is RunStatus.OK for o in outs)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in pairwise(spans))


async def test_capacity_preflight_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ex = LocalExecutor(workspace_root=str(tmp_path / "never"))
    assert not (tmp_path / "never").exists()  # __init__ does no I/O
    assert await ex.capacity() == Resources(cpus=float(os.cpu_count() or 1))
    assert ex.config == LocalExecutor.Config(workspace_root=str(tmp_path / "never"))
    assert LocalExecutor(LocalExecutor.Config(keep_workspaces=True)).config.keep_workspaces
    with pytest.raises(TypeError, match="config or fields"):
        LocalExecutor(LocalExecutor.Config(), rlimits=False)
    with pytest.raises(ValidationError, match="max_concurrent"):
        LocalExecutor(max_concurrent=0)
    assert ex.config.max_log_bytes == 8 * 2**20
    assert ex.config.max_output_bytes == 256 * 2**20
    await ex.aclose()
    ex.preflight()
    monkeypatch.setattr(os, "name", "nt")  # the last step, so no Path is built while it holds
    with pytest.raises(RuntimeError, match="POSIX"):
        ex.preflight()


async def test_confirm_dead_and_kill_orphan(local_executor: LocalExecutor) -> None:
    require_visible_proc()  # under hidepid, group_has_live_member answers "alive" to be safe
    me = await anyio.to_thread.run_sync(host_id)
    proc = await anyio.to_thread.run_sync(spawn_group, ["sleep", "30"])
    try:
        started = Job.Started(
            kind="process",
            job_id="c-1",
            host_id=me,
            pid=proc.pid,
            pgid=proc.pid,
            start_time=await anyio.to_thread.run_sync(process_start_time, proc.pid),
        )
        assert await local_executor.confirm_dead(started) is False  # alive
        assert await local_executor.kill_orphan(started) is True
        deadline = time.monotonic() + 60
        while not await local_executor.confirm_dead(started):  # an unreaped zombie is dead
            assert time.monotonic() < deadline, "the killed group still looks alive"
            await anyio.sleep(0.05)
        assert pid_alive(proc.pid)  # still a zombie: nothing has reaped it yet
        await anyio.to_thread.run_sync(reap, proc)
        assert await local_executor.confirm_dead(started) is True
        assert await local_executor.kill_orphan(started) is False  # gone: nothing to kill
        other = started.model_copy(update={"host_id": "other:x"})
        assert await local_executor.confirm_dead(other) is None
        dispatch = Job.Started(kind="dispatch", job_id="c-1", task_name="t")
        assert await local_executor.confirm_dead(dispatch) is None
    finally:
        await anyio.to_thread.run_sync(reap, proc)


async def test_confirm_dead_unknown_boot_is_none(
    local_executor: LocalExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:  # this host's own id ends in ':unknown', so the record matches it
    unknown = "somehost:unknown"
    monkeypatch.setattr(local_executor, "_host_id", unknown)
    started = Job.Started(
        kind="process", job_id="c-1", host_id=unknown, pid=os.getpid(), pgid=os.getpgrp()
    )
    assert await local_executor.confirm_dead(started) is None


@pytest.mark.parametrize(
    ("staged", "declared"),
    [("p/run.py", "P/RUN.py"), ("problem/run.py", "Problem/out.txt")],
)
async def test_folded_output_vs_staged_path_fails_producing_run(
    local_executor: LocalExecutor, tmp_path: Path, staged: str, declared: str
) -> None:
    artifacts = store(tmp_path)
    digest = await anyio.to_thread.run_sync(artifacts.put_bytes, b"staged-secret")
    code = (
        "from pathlib import Path; "
        f"p = Path({declared!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text('output-secret'); " + WRITE.format(repr({"data": {}, "files": [declared]}))
    )
    output = await local_executor.submit(
        tool_job(tmp_path, code, stage=Job.Stage(root={staged: digest}))
    )
    assert output.status == RunStatus.ERROR
    assert output.files == []
    diagnostic = log(tmp_path, output, 2).decode()
    assert staged in diagnostic
    assert declared in diagnostic
    assert "staged-secret" not in diagnostic
    assert "output-secret" not in diagnostic


async def cancel_when(
    submit: Callable[[], Coroutine[Any, Any, object]], ready: Callable[[], Awaitable[bool]]
) -> None:
    """Start ``submit``, wait (asynchronously) until ``ready()``, then cancel it."""
    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(submit)
            # The conditions are files and threading.Events, which have no anyio event: poll.
            while not await ready():  # noqa: ASYNC110
                await anyio.sleep(0.02)
            tg.cancel_scope.cancel()


def ready_file(ws: Path) -> Callable[[], Awaitable[bool]]:
    async def ready() -> bool:
        return bool([p async for p in anyio.Path(ws).glob("*/ready")])

    return ready


@pytest.mark.parametrize(("budget", "kept"), [(30.0, True), (0.2, False)])
async def test_cancel_after_normal_exit_keeps_the_result(
    kept_executor: LocalExecutor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: float,
    kept: bool,
) -> None:
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", budget)  # bounds the original collect
    got: list[ToolOutput] = []

    async def on_partial(o: ToolOutput) -> None:
        got.append(o)

    async def collecting() -> bool:
        return COLLECTING.is_set()

    COLLECTING.clear()
    code = WRITE.format("{'data': {'v': 7}}")  # exits 0 with a valid result
    job = tool_job(tmp_path, code, tool_cls=SlowCollectTool, on_partial=on_partial)
    await cancel_when(lambda: kept_executor.submit(job), collecting)
    [out] = got
    assert out.status is RunStatus.CANCELLED
    assert (out.partial, out.data) == ((True, {"v": 7}) if kept else (False, {}))
    assert out.started_at is not None
    assert out.ended_at is not None
    assert out.started_at <= out.ended_at
    if not kept:
        assert b"no partial output" in _workspace_log(tmp_path, 2)
    assert out.logs_ref is None


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_stop_facts_are_stamped_on_the_partial(
    local_executor: LocalExecutor, tmp_path: Path, stop: str
) -> None:  # OkCollectTool returns status OK and partial False whatever happened
    code = "open('ready', 'w'); import time; time.sleep(60)"
    if stop == "timeout":
        out = await local_executor.submit(
            tool_job(tmp_path, code, tool_cls=OkCollectTool, timeout_s=1.0, grace_s=0.2)
        )
        assert (out.status, out.partial, out.data) == (RunStatus.TIMEOUT, True, {"k": 1})
        return
    got: list[ToolOutput] = []

    async def on_partial(o: ToolOutput) -> None:
        got.append(o)

    job = tool_job(tmp_path, code, tool_cls=OkCollectTool, grace_s=0.2, on_partial=on_partial)
    await cancel_when(lambda: local_executor.submit(job), ready_file(tmp_path / "ws"))
    assert [(o.status, o.partial, o.data) for o in got] == [(RunStatus.CANCELLED, True, {"k": 1})]


async def test_cancel_during_staging_delivers_nothing(
    local_executor: LocalExecutor,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    got: list[ToolOutput] = []
    entered = threading.Event()
    release = threading.Event()
    threads: list[threading.Thread] = []
    original = PyTool.command

    def blocked(tool: PyTool, request: ToolRequest, workspace: Path) -> Command:
        threads.append(threading.current_thread())
        entered.set()
        release.wait(180)
        return original(tool, request, workspace)

    async def on_partial(o: ToolOutput) -> None:
        got.append(o)

    async def staging() -> bool:
        return entered.is_set()

    monkeypatch.setattr(PyTool, "command", blocked)
    marker = tmp_path / "ran"
    job = tool_job(tmp_path, f"open({str(marker)!r}, 'w')", on_partial=on_partial)
    try:
        await cancel_when(lambda: local_executor.submit(job), staging)
        [workspace] = (tmp_path / "ws").iterdir()
        assert workspace.is_dir()
        assert f"retaining workspace {workspace}: preparation is still running" in caplog.text
    finally:
        release.set()
        for thread in threads:
            await anyio.to_thread.run_sync(thread.join, 60)
            assert not thread.is_alive()
    assert got == []
    assert not marker.exists()
    assert workspace.is_dir()


async def test_cancel_skips_unavailable_uploads_and_sends_no_unstored_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    got: list[ToolOutput] = []

    async def on_partial(o: ToolOutput) -> None:
        got.append(o)

    def down(self: ArtifactStore, data: bytes) -> str:
        raise StorageError("store down")

    monkeypatch.setattr(ArtifactStore, "put_bytes", down)
    code = (
        "import os; os.makedirs('out'); open('out/p.txt', 'w').write('x'); "
        + WRITE.format("{'data': {'s': 1}, 'files': ['out/p.txt']}")
        + "; open('ready', 'w'); import time; time.sleep(60)"
    )
    ws = tmp_path / "kept"
    ex = LocalExecutor(workspace_root=str(ws), keep_workspaces=True)
    job = tool_job(tmp_path, code, grace_s=0.2, on_partial=on_partial)
    await cancel_when(lambda: ex.submit(job), ready_file(ws))
    [out] = got
    assert (out.status, out.partial, out.data, out.files) == (
        RunStatus.CANCELLED,
        True,
        {"s": 1},
        [],
    )
    assert out.logs_ref is None
    assert out.started_at is not None
    [workspace] = list(ws.iterdir())
    assert "uploads failed" not in (workspace / "swarm" / "stderr.log").read_text()


async def test_missing_cwd_is_a_job_error(local_executor: LocalExecutor, tmp_path: Path) -> None:
    out = await local_executor.submit(
        command_job(tmp_path, Command(argv=["python", "-c", "pass"], cwd="missing/dir"))
    )
    assert (out.status, out.files) == (RunStatus.ERROR, [])
    assert b"command cwd missing/dir is not a directory in the workspace" in log(tmp_path, out, 2)
    assert log(tmp_path, out, 1) == b""
    assert list((tmp_path / "ws").iterdir()) == []


async def test_missing_stage_artifact_raises_and_cleans_up(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:  # a stage artifact that is not in the store
    job = command_job(
        tmp_path, Command(argv=["true"]), stage=Job.Stage(root={"problem/in.txt": "e" * 64})
    )
    with pytest.raises(NotFound):
        await local_executor.submit(job)
    assert list((tmp_path / "ws").iterdir()) == []


@pytest.mark.parametrize("entry", ["fields", "model_validate_json"])
def test_config_errors_hide_values(entry: str) -> None:
    secret = "81fc2389-90d2-4331-8e22-cc91703ac8ed"

    def build() -> object:
        if entry == "fields":
            return LocalExecutor(workspace_root={"token": secret})
        return LocalExecutor.Config.model_validate_json(json.dumps({"workspace_root": [secret]}))

    with pytest.raises(ValidationError, match="workspace_root") as caught:
        build()
    assert_secret_absent(caught.value, secret, sanitized=True)
    with pytest.raises(ValidationError, match="max_concurrent") as caught:
        LocalExecutor(max_concurrent=secret)
    assert_secret_absent(caught.value, secret, sanitized=True)


async def test_rlimits_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[RLimits | None] = []
    original = ProcessSupervisor.run

    async def spy(self: ProcessSupervisor, argv: Sequence[str], **kw: Any) -> CompletedRun:
        seen.append(kw["rlimits"])
        return await original(self, argv, **kw)

    monkeypatch.setattr(ProcessSupervisor, "run", spy)
    code = "import resource; " + WRITE.format(
        "{'nofile': resource.getrlimit(resource.RLIMIT_NOFILE)[0]}"
    )
    need = Resources(memory_gib=0.5)
    job = command_job(tmp_path, Command(argv=["python", "-c", code]), resources=need)
    on = await LocalExecutor(workspace_root=str(tmp_path / "ws")).submit(job)
    off = await LocalExecutor(workspace_root=str(tmp_path / "ws"), rlimits=False).submit(job)
    assert seen == [RLimits(memory_bytes=2**29), None]
    assert (on.status, off.status) == (RunStatus.OK, RunStatus.OK)
    assert 0 < on.data["nofile"] <= 4096


@pytest.mark.skipif(
    sys.platform != "linux", reason="macOS refuses RLIMIT_AS; the launcher notes it in stderr"
)
async def test_memory_gib_sets_the_address_space_limit(tmp_path: Path) -> None:
    code = "import resource; " + WRITE.format("{'as': resource.getrlimit(resource.RLIMIT_AS)[0]}")
    job = command_job(
        tmp_path, Command(argv=["python", "-c", code]), resources=Resources(memory_gib=0.5)
    )
    out = await LocalExecutor(workspace_root=str(tmp_path / "ws")).submit(job)
    assert (out.status, out.data) == (RunStatus.OK, {"as": 2**29})


async def test_sparse_log_upload_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = 1 << 20  # every driver read path is counted while submit runs
    read: list[int] = []
    code = (
        WRITE.format("{'data': {}}") + "; import os; os.truncate('swarm/stdout.log', "
        f"{2 * cap}); f = open('swarm/stdout.log', 'r+b'); f.seek({2 * cap - 9}); "
        "f.write(b'TAIL-MARK'); f.close()"
    )
    ex = LocalExecutor(workspace_root=str(tmp_path / "ws"), max_log_bytes=cap)
    with count_reads(monkeypatch, read):
        out = await ex.submit(tool_job(tmp_path, code))
    total = sum(read)
    stdout = log(tmp_path, out, 1)
    assert len(stdout) == cap
    assert stdout.endswith(b"TAIL-MARK")
    note = f"log truncated: swarm/stdout.log: dropped the first {cap} bytes".encode()
    assert note in log(tmp_path, out, 2)
    assert cap <= total <= cap + 64 * 1024  # the tail and small files, never the whole 2x log


async def test_stderr_tail_keeps_its_own_truncation_note(tmp_path: Path) -> None:
    cap = 4096
    code = "import sys; sys.stderr.write('e' * 10000); " + WRITE.format("{'data': {}}")
    ex = LocalExecutor(workspace_root=str(tmp_path / "ws"), max_log_bytes=cap)
    out = await ex.submit(tool_job(tmp_path, code))
    stderr = log(tmp_path, out, 2)
    assert len(stderr) == cap
    # The tool wrote no final newline, so the note follows its bytes on the same line.
    note = re.search(
        rb"swarm-arena: log truncated: swarm/stderr\.log: dropped the first (\d+) "
        rb"bytes\n\Z",
        stderr,
    )
    assert note is not None
    assert int(note[1]) == 10000 + len(note[0]) - cap  # exact: the note's own line counts


async def test_outputs_over_the_cap_are_an_error(tmp_path: Path) -> None:
    code = (
        "import os; os.makedirs('out'); open('out/a.txt', 'w').write('a' * 600); "
        "open('out/b.txt', 'w').close(); os.truncate('out/b.txt', 2**30); "  # sparse 1 GiB
        + WRITE.format("{'data': {}}")
    )
    ex = LocalExecutor(workspace_root=str(tmp_path / "ws"), max_output_bytes=1024)
    with anyio.fail_after(60):
        out = await ex.submit(tool_job(tmp_path, code, tool_cls=UnreadFilesTool))
    assert (out.status, out.files) == (RunStatus.ERROR, [])
    assert b"out/b.txt crosses max_output_bytes" in log(tmp_path, out, 2)


async def test_upload_reads_are_bounded_by_the_cap_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # collect_output_files gets max_output_bytes
    monkeypatch.setattr(local_run._JobRun, "_check_output_sizes", lambda self, files: None)
    code = (
        "import os; os.makedirs('out'); open('out/a.txt', 'w').write('a' * 600); "
        "open('out/b.txt', 'w').close(); os.truncate('out/b.txt', 2**30); "  # sparse 1 GiB
        + WRITE.format("{'data': {}}")
    )
    ex = LocalExecutor(workspace_root=str(tmp_path / "ws"), max_output_bytes=1024)
    read: list[int] = []
    with anyio.fail_after(60), count_reads(monkeypatch, read):
        out = await ex.submit(tool_job(tmp_path, code, tool_cls=UnreadFilesTool))
    assert (out.status, out.files) == (RunStatus.ERROR, [])
    assert b"output files not uploaded: out/b.txt crosses the size cap" in log(tmp_path, out, 2)
    assert sum(read) < 64 * 1024  # out/a.txt and the logs, never the sparse 1 GiB file


def remove_tree_flat(top: Path) -> None:
    """
    Remove any tree one level at a time, independent of the code under test.

    Each step moves the grandchildren up into ``top`` and removes the emptied
    children, so no path is ever deeper than two names and nothing recurses.
    """
    count = 0
    top.chmod(0o700)
    while True:
        children = list(top.iterdir())
        directories = [p for p in children if p.is_dir() and not p.is_symlink()]
        for path in children:
            if path not in directories:
                path.unlink()
        if not directories:
            break
        for directory in directories:
            directory.chmod(0o700)
            for path in list(directory.iterdir()):
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o700)  # renaming a directory updates its ".." entry
                    count += 1
                    path.rename(top / f"up-{count}")
                else:
                    path.unlink()
            directory.rmdir()
    top.rmdir()


@pytest.fixture
def deep_tree_guard(tmp_path: Path) -> Iterator[None]:
    """Remove any workspace a deep-tree test leaves, even when the test fails.

    pytest's own basetemp cleanup uses the recursive shutil.rmtree, which dies
    with RecursionError on Python 3.11, so a leftover would break later runs.
    """
    try:
        yield
    finally:
        ws = tmp_path / "ws"
        if ws.exists():
            for leftover in list(ws.iterdir()):
                remove_tree_flat(leftover)


@pytest.mark.usefixtures("deep_tree_guard")
async def test_deep_read_only_tree_is_removed(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:
    code = WRITE.format("{'data': {'deep': True}}") + (
        "\nimport os\nfor _ in range(1200):\n    os.mkdir('d'); os.chdir('d')\n"
        "open('f', 'w').close(); os.chmod('.', 0o500)\n"
    )
    with anyio.fail_after(60):
        out = await local_executor.submit(tool_job(tmp_path, code))
    assert (out.status, out.data) == (RunStatus.OK, {"deep": True})
    assert list((tmp_path / "ws").iterdir()) == []


async def test_failed_cleanup_never_replaces_output_or_cancel(
    local_executor: LocalExecutor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(path: Path) -> None:
        raise RecursionError("too deep")

    monkeypatch.setattr(local_workspace, "_REMOVERS", (broken,))
    out = await local_executor.submit(tool_job(tmp_path, WRITE.format("{'data': {'v': 1}}")))
    assert (out.status, out.data) == (RunStatus.OK, {"v": 1})
    [left] = list((tmp_path / "ws").iterdir())
    assert f"could not remove workspace {left}" in caplog.text

    code = "open('ready', 'w'); import time; time.sleep(60)"
    with anyio.move_on_after(30) as scope:  # the cancel still propagates (no ExceptionGroup)
        await cancel_when(
            lambda: local_executor.submit(tool_job(tmp_path, code, grace_s=0.2)),
            ready_file(tmp_path / "ws"),
        )
    assert not scope.cancelled_caught


_EXIT_SCRIPT = """
import sys
from typing import Any
import anyio
from _fakes.tools import BlockingCollectTool, RELEASE, make_job
from swarm_arena._exec import supervisor
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._model.ref import Ref
from swarm_arena._model.tooling import ToolOutput, ToolRequest

supervisor.BEFORE_KILL_TIMEOUT_S = 0.5
RELEASE.clear()
collect = BlockingCollectTool.collect

def held_collect(*args: Any, **kwargs: Any) -> ToolOutput:
    RELEASE.wait()  # never released in this interpreter
    return collect(*args, **kwargs)

BlockingCollectTool.collect = held_collect


async def main() -> None:
    tool = Ref.of(BlockingCollectTool({"code": "import time; time.sleep(60)"}))
    job = make_job(kind="tool", command=None, tool=tool, request=ToolRequest(),
                   timeout_s=0.5, grace_s=0.1, artifact_root=sys.argv[2])
    out = await LocalExecutor(workspace_root=sys.argv[1]).submit(job)
    print(out.status.value, flush=True)


anyio.run(main)
"""


def test_blocked_collect_does_not_keep_the_interpreter_alive(tmp_path: Path) -> None:
    tests_dir = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(tests_dir)}
    done = subprocess.run(
        [sys.executable, "-c", _EXIT_SCRIPT, str(tmp_path / "ws"), str(tmp_path / "artifacts")],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert (done.returncode, done.stdout.strip()) == (0, "timeout"), done.stderr


@pytest.fixture
def wide_tree(tmp_path: Path) -> Iterator[Path]:
    """Give a tree root that is removed in teardown, even when the test fails."""
    top = tmp_path / "tree"
    top.mkdir()
    try:
        yield top
    finally:
        if os.path.lexists(top):
            remove_tree_flat(top)


def test_fallback_remover_is_linear(
    wide_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = 20_000
    for i in range(n):
        os.mkdir(wide_tree / f"d{i}")
    (wide_tree / "d0" / "f").write_text("x")
    (wide_tree / "d0").chmod(0o500)  # makes shutil.rmtree fail into this fallback
    (wide_tree / "d1" / "g").write_text("y")
    (wide_tree / "d1").chmod(0o000)  # can only be opened after an unlock
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("k")
    outside.chmod(0o500)
    (wide_tree / "d2" / "link").symlink_to(outside)  # must be unlinked, never followed
    calls = {"listdir": 0, "stat": 0}
    real_listdir, real_stat = os.listdir, os.stat
    limit = 4 * (n + 10)  # a quadratic walk crosses this at once and fails fast

    def listdir(path: Any) -> list[str]:
        calls["listdir"] += 1
        if calls["listdir"] > limit:
            raise RuntimeError("listdir called too often")
        return real_listdir(path)

    def stat_(*args: Any, **kw: Any) -> os.stat_result:
        calls["stat"] += 1
        if calls["stat"] > limit:
            raise RuntimeError("stat called too often")
        return real_stat(*args, **kw)

    try:
        with pytest.raises(PermissionError):
            shutil.rmtree(wide_tree / "d0")  # the reason the fallback runs at all
        with monkeypatch.context() as patch:
            patch.setattr(os, "listdir", listdir)
            patch.setattr(os, "stat", stat_)
            local_workspace._remove_tree(wide_tree)
        assert not os.path.lexists(wide_tree)
        # Linear in the number of directories, by call counts, not by the wall clock.
        assert calls["listdir"] <= n + 2  # each directory listed exactly once
        assert calls["stat"] <= 2 * (n + 10)
        assert (outside / "keep").read_text() == "k"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o500
    finally:
        outside.chmod(0o700)


def test_fallback_remover_unlocks_only_refused_directories(
    wide_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # a chmod costs ~8 ms deep in an APFS tree, so only on EACCES/EPERM
    (wide_tree / "a/b/c").mkdir(parents=True)
    for rel in ("a/f", "a/b/f", "a/b/c/f"):
        (wide_tree / rel).write_text("x")
    (wide_tree / "files_locked").mkdir()
    (wide_tree / "files_locked/f").write_text("x")
    (wide_tree / "files_locked").chmod(0o500)  # unlink in it is refused
    (wide_tree / "dirs_locked/sub").mkdir(parents=True)
    (wide_tree / "dirs_locked").chmod(0o500)  # only rmdir of its child is refused
    (wide_tree / "closed").mkdir()
    (wide_tree / "closed/f").write_text("x")
    (wide_tree / "closed").chmod(0o000)  # cannot even be opened: unlocked from its parent
    fchmods: list[int] = []
    real_fchmod = os.fchmod

    def fchmod(fd: int, mode: int) -> None:
        fchmods.append(mode)
        real_fchmod(fd, mode)

    monkeypatch.setattr(os, "fchmod", fchmod)
    local_workspace._remove_tree(wide_tree)
    assert not os.path.lexists(wide_tree)
    assert fchmods == [0o700, 0o700]  # files_locked and dirs_locked; never top, a, b or c


@pytest.mark.usefixtures("deep_tree_guard")
@pytest.mark.parametrize("mode", [0o000, 0o500])
async def test_tool_that_locks_its_own_workspace_root_is_removed(
    local_executor: LocalExecutor, tmp_path: Path, mode: int
) -> None:
    code = "import os; os.makedirs('out/deep'); open('out/deep/f', 'w').close(); "
    code += f"os.chmod('out/deep', 0o500); os.chmod('.', {mode:#o})"
    with anyio.fail_after(30):
        await local_executor.submit(tool_job(tmp_path, code))
    assert list((tmp_path / "ws").iterdir()) == []


def test_unlock_uses_a_no_follow_chmod_where_the_os_has_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # macOS and the BSDs
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def chmod(*args: Any, **kw: Any) -> None:
        calls.append((args, kw))

    with monkeypatch.context() as patch:
        patch.setattr(os, "chmod", chmod)
        patch.setattr(os, "supports_follow_symlinks", {chmod})
        patch.setattr(os, "supports_dir_fd", {chmod})
        unlocked = local_workspace._unlock_directory(7, "d")
    assert unlocked is True
    assert calls == [(("d", 0o700), {"dir_fd": 7, "follow_symlinks": False})]


def test_unlock_uses_o_path_and_proc_on_linux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:  # Linux has no fchmodat(AT_SYMLINK_NOFOLLOW) for directories
    o_path = 0o10000000
    opened: list[tuple[str, int, int | None]] = []
    chmods: list[tuple[Any, ...]] = []
    closed: list[int] = []

    def open_(name: str, flags: int, dir_fd: int | None = None) -> int:
        opened.append((name, flags, dir_fd))
        return 99

    def chmod(*args: Any, **kw: Any) -> None:
        chmods.append(args)

    with monkeypatch.context() as patch:
        patch.setattr(os, "supports_follow_symlinks", set())
        patch.setattr(os, "O_PATH", o_path, raising=False)
        patch.setattr(local_workspace, "_PROC_FD", str(tmp_path))  # stands in for /proc/self/fd
        patch.setattr(os, "open", open_)
        patch.setattr(os, "chmod", chmod)
        patch.setattr(os, "close", closed.append)
        unlocked = local_workspace._unlock_directory(7, "d")
    assert unlocked is True
    assert opened == [("d", o_path | os.O_NOFOLLOW | os.O_DIRECTORY, 7)]
    assert chmods == [(f"{tmp_path}/99", 0o700)]
    assert closed == [99]


def test_unlock_without_a_safe_method_skips_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:  # never fall back to a chmod that follows links
    chmods: list[tuple[Any, ...]] = []
    with monkeypatch.context() as patch:
        patch.setattr(os, "supports_follow_symlinks", set())
        patch.delattr(os, "O_PATH", raising=False)
        patch.setattr(os, "chmod", lambda *args, **kw: chmods.append(args))
        with caplog.at_level(logging.WARNING):
            unlocked = local_workspace._unlock_directory(7, "d")
    assert unlocked is False
    assert chmods == []
    assert "cannot unlock a locked directory without following symlinks: d" in caplog.text


def _not_a_tool_output(kind: str) -> Any:
    if kind == "dict":
        return {"status": "ok"}
    if kind == "none":
        return None
    if kind == "construct":  # skips validation, so a NaN gets past the finite-JSON rule
        return ToolOutput.model_construct(status=RunStatus.OK, data={"x": float("nan")})
    return ToolOutput(status=RunStatus.OK).model_copy(update={"data": {"x": float("inf")}})


@pytest.mark.parametrize("kind", ["dict", "none", "construct", "copy"])
@pytest.mark.parametrize("path", ["exit", "timeout", "cancel"])
async def test_collect_must_return_a_valid_tool_output(
    kept_executor: LocalExecutor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    path: str,
) -> None:
    monkeypatch.setattr(
        OkCollectTool, "collect", lambda self, request, workspace, run: _not_a_tool_output(kind)
    )
    if path == "cancel":
        got: list[ToolOutput] = []

        async def on_partial(o: ToolOutput) -> None:
            got.append(o)

        code = "open('ready', 'w'); import time; time.sleep(60)"
        job = tool_job(tmp_path, code, tool_cls=OkCollectTool, grace_s=0.2, on_partial=on_partial)
        await cancel_when(lambda: kept_executor.submit(job), ready_file(tmp_path / "ws"))
        [out] = got
        status = RunStatus.CANCELLED
    elif path == "timeout":
        code = "import time; time.sleep(60)"
        job = tool_job(tmp_path, code, tool_cls=OkCollectTool, timeout_s=0.5, grace_s=0.2)
        out = await kept_executor.submit(job)
        status = RunStatus.TIMEOUT
    else:
        out = await kept_executor.submit(tool_job(tmp_path, "pass", tool_cls=OkCollectTool))
        status = RunStatus.ERROR
    # The collect gave no valid ToolOutput, so partial is False on every path.
    assert (out.status, out.partial, out.data, out.files) == (status, False, {}, [])
    diagnostic = _workspace_log(tmp_path, 2) if path == "cancel" else log(tmp_path, out, 2)
    assert b"collect failed: Tool.collect() did not return a ToolOutput" in diagnostic
    if path == "cancel":
        assert out.logs_ref is None
    assert out.started_at is not None


# --- A stopped run is partial when its valid collect output kept something or says so ---


async def _cancel_when_ready(
    executor: LocalExecutor, ws: Path, job_kw: dict[str, Any], tmp_path: Path, code: str
) -> ToolOutput:
    """Cancel only after the tool wrote its ``ready`` file: no race with start-up."""
    got: list[ToolOutput] = []

    async def on_partial(o: ToolOutput) -> None:
        got.append(o)

    job = tool_job(
        tmp_path,
        code + "; open('ready', 'w'); import time; time.sleep(60)",
        grace_s=0.2,
        on_partial=on_partial,
        **job_kw,
    )
    await cancel_when(lambda: executor.submit(job), ready_file(ws))
    [out] = got
    return out


@pytest.mark.parametrize(
    ("result", "partial", "kept"),
    [
        (None, False, ({}, [])),  # no result.json: the data is empty and not partial (spec 5)
        ("{'data': {}}", True, ({}, [])),  # a valid result, even an empty one, is partial
        ("{'data': {}, 'files': ['out/x.txt']}", True, ({}, [])),
        ("{'data': {'step': 3}}", True, ({"step": 3}, [])),
    ],
)
async def test_stopped_run_is_partial_only_with_a_valid_result(
    kept_executor: LocalExecutor,
    tmp_path: Path,
    result: str | None,
    partial: bool,
    kept: tuple[dict[str, Any], list[str]],
) -> None:
    write = "" if result is None else "; " + WRITE.format(result)
    code = "import os; os.makedirs('out'); open('out/x.txt', 'w').write('x')" + write
    out = await _cancel_when_ready(kept_executor, tmp_path / "ws", {}, tmp_path, code)
    assert (out.status, out.partial) == (RunStatus.CANCELLED, partial)
    assert (out.data, [ref.path for ref in out.files]) == kept
    assert out.logs_ref is None
    stderr = _workspace_log(tmp_path, 2)
    assert b"no partial output" not in stderr  # the collect itself did not fail
    assert (b"swarm/result.json is missing" in stderr) is (result is None)


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
@pytest.mark.parametrize("says_partial", [False, True])
async def test_valid_empty_collect_output_is_partial_only_if_it_says_so(
    kept_executor: LocalExecutor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
    says_partial: bool,
) -> None:  # the collect output does not depend on the tool's speed, so a timeout cannot race
    empty = ToolOutput(status=RunStatus.OK, partial=says_partial)
    monkeypatch.setattr(OkCollectTool, "collect", lambda self, request, workspace, run: empty)
    if stop == "timeout":
        job = tool_job(
            tmp_path,
            "import time; time.sleep(60)",
            tool_cls=OkCollectTool,
            timeout_s=0.5,
            grace_s=0.2,
        )
        out, status = await kept_executor.submit(job), RunStatus.TIMEOUT
    else:
        kw = {"tool_cls": OkCollectTool}
        out = await _cancel_when_ready(kept_executor, tmp_path / "ws", kw, tmp_path, "pass")
        status = RunStatus.CANCELLED
    assert (out.status, out.partial, out.data, out.files) == (status, says_partial, {}, [])
    diagnostic = _workspace_log(tmp_path, 2) if stop == "cancel" else log(tmp_path, out, 2)
    assert b"no partial output" not in diagnostic
    if stop == "cancel":
        assert out.logs_ref is None


@pytest.mark.parametrize("drop", ["cap", "upload"])
async def test_cancel_keeps_partial_when_files_cannot_be_uploaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drop: str
) -> None:
    ws = tmp_path / "kept"

    def collect(
        self: PyTool, request: ToolRequest, workspace: Path, run: CompletedRun
    ) -> ToolOutput:
        # A custom connector can declare a valid file before final upload checks.
        # Built-in collection rejects an over-budget envelope before hashing now.
        return ToolOutput(
            status=run.status,
            files=[FileRef(name="x.txt", path="out/x.txt", sha256=sha256_hex(b"x"), size=1)],
        )

    monkeypatch.setattr(PyTool, "collect", collect)
    if drop == "upload":

        def down(self: ArtifactStore, data: bytes) -> str:
            raise StorageError("store down")

        monkeypatch.setattr(ArtifactStore, "put_bytes", down)
    ex = LocalExecutor(
        workspace_root=str(ws), keep_workspaces=True, max_output_bytes=0 if drop == "cap" else 1024
    )
    code = "import os; os.makedirs('out'); open('out/x.txt', 'w').write('x')"
    out = await _cancel_when_ready(ex, ws, {}, tmp_path, code)
    assert (out.status, out.partial, out.data, out.files) == (RunStatus.CANCELLED, True, {}, [])
    assert out.logs_ref is None
    [workspace] = list(ws.iterdir())
    assert (workspace / "out/x.txt").read_bytes() == b"x"


@pytest.mark.parametrize(
    ("job_timeout", "command_timeout", "expected"),
    [(9.0, 2.0, 2.0), (2.0, 9.0, 2.0), (None, 2.0, 2.0), (2.0, None, 2.0), (None, None, None)],
)
@pytest.mark.parametrize(("job_grace", "command_grace"), [(9.0, 2.0), (2.0, 9.0)])
@pytest.mark.parametrize("kind", ["tool", "command"])
async def test_command_limits_tighten_job_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    job_timeout: float | None,
    command_timeout: float | None,
    expected: float | None,
    job_grace: float,
    command_grace: float,
    kind: str,
) -> None:
    seen: list[tuple[float | None, float]] = []
    command = Command(
        argv=["python", "-c", "pass"], timeout_s=command_timeout, grace_s=command_grace
    )
    monkeypatch.setattr(PyTool, "command", lambda self, request, workspace: command)
    original = ProcessSupervisor.run

    async def record(self: ProcessSupervisor, *args: Any, **kw: Any) -> CompletedRun:
        seen.append((kw["timeout_s"], kw["grace_s"]))
        return await original(self, *args, **kw)

    monkeypatch.setattr(ProcessSupervisor, "run", record)
    job = tool_job(tmp_path, "pass") if kind == "tool" else command_job(tmp_path, command)
    await LocalExecutor(workspace_root=str(tmp_path / "ws")).submit(
        replace(job, timeout_s=job_timeout, grace_s=job_grace)
    )
    assert seen == [(expected, min(job_grace, command_grace))]


@pytest.mark.parametrize("where", ["command", "collect"])
@pytest.mark.parametrize("stop", [False, True])
async def test_user_system_exit_is_a_job_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
    stop: bool,
) -> None:
    secret = "d68e94c4-477f-4935-aaf0-70960e8b3c7f"

    def exit_user(*args: Any, **kw: Any) -> Any:
        raise SystemExit(secret)

    monkeypatch.setattr(PyTool, where, exit_user)
    job = tool_job(
        tmp_path,
        "import time; time.sleep(60)" if stop else "pass",
        timeout_s=0.2 if stop else 30.0,
        grace_s=0.0,
    )
    # Catch it in the test task too: a regression must fail without exiting pytest.
    try:
        out = await LocalExecutor(workspace_root=str(tmp_path / "ws")).submit(job)
    except SystemExit:
        pytest.fail("user SystemExit escaped submit")
    assert out.status == (RunStatus.TIMEOUT if stop and where == "collect" else RunStatus.ERROR)
    assert not out.partial
    assert b"SystemExit" in log(tmp_path, out, 2)
    if where == "command":
        assert secret.encode() not in log(tmp_path, out, 2)


@pytest.mark.parametrize("where", ["command", "collect"])
def test_user_keyboard_interrupt_still_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
) -> None:
    def interrupt(*args: Any, **kw: Any) -> Any:
        raise KeyboardInterrupt

    job = tool_job(tmp_path, "pass")
    workspace, artifacts = local_workspace._open_workspace(str(tmp_path / "ws"), job)
    monkeypatch.setattr(PyTool, where, interrupt)
    if where == "command":
        with pytest.raises(KeyboardInterrupt):
            local_stage._stage(job, workspace, artifacts)
    else:
        staged = local_stage._stage(job, workspace, artifacts)
        run = CompletedRun(
            returncode=0,
            status=RunStatus.OK,
            stdout_path=workspace / "swarm/stdout.log",
            stderr_path=workspace / "swarm/stderr.log",
            started_at=utc_now(),
            ended_at=utc_now(),
        )
        runner = local_run._JobRun(
            job, staged, workspace, artifacts, max_log_bytes=1024, max_output_bytes=1024
        )
        with pytest.raises(KeyboardInterrupt):
            runner._collect(run)


async def test_cancel_waits_for_the_original_collect_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = threading.Event(), threading.Event()
    calls: list[RunStatus] = []
    got: list[ToolOutput] = []
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 5.0)

    def collect(*args: Any) -> ToolOutput:
        calls.append(args[-1].status)
        entered.set()
        release.wait(180)
        return ToolOutput(status=RunStatus.OK, data={"kept": 1})

    async def on_partial(out: ToolOutput) -> None:
        got.append(out)

    monkeypatch.setattr(PyTool, "collect", collect)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))

    async def worker(*, task_status: anyio.abc.TaskStatus[anyio.CancelScope]) -> None:
        with anyio.CancelScope() as scope:
            task_status.started(scope)
            await executor.submit(tool_job(tmp_path, "pass", on_partial=on_partial))
        assert scope.cancelled_caught

    try:
        with anyio.fail_after(60):
            async with anyio.create_task_group() as group:
                scope = await group.start(worker)
                while not entered.is_set():  # noqa: ASYNC110 - waiting on a thread event
                    await anyio.sleep(0.01)
                scope.cancel()
                await anyio.wait_all_tasks_blocked()
                release.set()
        assert calls == [RunStatus.OK]
        assert [(o.status, o.partial, o.data) for o in got] == [
            (RunStatus.CANCELLED, True, {"kept": 1})
        ]
    finally:
        release.set()


async def test_hung_post_exit_collect_releases_pool_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    calls = 0
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.2)

    def collect(*args: Any) -> ToolOutput:
        nonlocal calls
        calls += 1
        if calls == 1:
            release.wait(180)
        return ToolOutput(status=RunStatus.OK, data={"kept": 1})

    monkeypatch.setattr(PyTool, "collect", collect)
    executor = LocalExecutor(max_concurrent=1, workspace_root=str(tmp_path / "ws"))
    try:
        with anyio.fail_after(60):
            out = await executor.submit(tool_job(tmp_path, "pass"))
            assert (out.status, out.partial, out.data) == (RunStatus.ERROR, False, {})
            assert b"collect timed out" in log(tmp_path, out, 2)
            assert (await executor.submit(tool_job(tmp_path, "pass"))).status == RunStatus.OK
    finally:
        release.set()


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_files_only_collect_counts_as_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    def collect(
        self: PyTool, request: ToolRequest, workspace: Path, run: CompletedRun
    ) -> ToolOutput:
        (workspace / "out").mkdir(exist_ok=True)
        (workspace / "out/x.txt").write_bytes(b"x")
        return ToolOutput(
            status=RunStatus.OK,
            files=[FileRef(name="x.txt", path="out/x.txt", sha256=sha256_hex(b"x"), size=1)],
        )

    monkeypatch.setattr(PyTool, "collect", collect)
    executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
    if stop == "cancel":
        out = await _cancel_when_ready(executor, tmp_path / "ws", {}, tmp_path, "pass")
        expected = RunStatus.CANCELLED
    else:
        out = await executor.submit(
            tool_job(tmp_path, "import time; time.sleep(60)", timeout_s=0.2, grace_s=0.0)
        )
        expected = RunStatus.TIMEOUT
    assert (out.status, out.partial, out.data) == (expected, True, {})
    if stop == "cancel":
        assert out.files == []
        assert out.logs_ref is None
    else:
        assert [ref.path for ref in out.files] == ["out/x.txt"]
        assert store(tmp_path).get_bytes(out.files[0].sha256) == b"x"


@pytest.mark.parametrize("extra", [0, 1])
async def test_output_total_at_the_size_cap(tmp_path: Path, extra: int) -> None:
    code = (
        "import os; os.makedirs('out'); open('out/a', 'wb').write(b'a' * 4); "
        f"open('out/b', 'wb').write(b'b' * {6 + extra}); "
        + WRITE.format("{'data': {}, 'files': ['out/a', 'out/b']}")
    )
    out = await LocalExecutor(workspace_root=str(tmp_path / "ws"), max_output_bytes=10).submit(
        tool_job(tmp_path, code)
    )
    assert out.status == (RunStatus.ERROR if extra else RunStatus.OK)
    assert len(out.files) == (0 if extra else 2)
    if not extra:
        assert [store(tmp_path).get_bytes(ref.sha256) for ref in out.files] == [b"aaaa", b"bbbbbb"]


def test_memory_limit_overflow_is_sanitized(tmp_path: Path) -> None:
    job = command_job(tmp_path, Command(argv=["true"]), resources=Resources(memory_gib=1e300))
    with pytest.raises(ValueError, match="memory_gib") as caught:
        LocalExecutor()._rlimits(job)
    assert str(caught.value) == "memory_gib exceeds resource limit range"


def test_memory_overflow_exception_cannot_expose_values(tmp_path: Path) -> None:
    secret = "04c2f5db-ef64-4a48-a0df-606244b2758e"

    class OverflowingMemory(float):
        def __mul__(self, other: Any) -> float:
            raise OverflowError(secret)

    resources = Resources.model_construct(memory_gib=OverflowingMemory(1.0))
    job = command_job(tmp_path, Command(argv=["true"]), resources=resources)
    with pytest.raises(ValueError, match="memory_gib") as caught:
        LocalExecutor()._rlimits(job)
    assert_secret_absent(caught.value, secret)
