import contextlib
import io
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Coroutine, Generator, Iterator, Sequence
from itertools import pairwise
from pathlib import Path
from traceback import format_exception
from typing import Any

import anyio
import pytest
from _fakes.proc import pid_alive, reap, spawn_group, wait_until
from _fakes.tools import (
    COLLECTING,
    RELEASE,
    STAGING,
    WRITE,
    BadCollectTool,
    BlockingCollectTool,
    GhostFileTool,
    OkCollectTool,
    PyTool,
    SlowCollectTool,
    SlowCommandTool,
    UnreadFilesTool,
    make_job,
)
from pydantic import ValidationError

from swarm_arena._errors import NotFound, StorageError
from swarm_arena._exec import local, supervisor
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._exec.procinfo import host_id, process_start_time
from swarm_arena._exec.supervisor import ProcessSupervisor, RLimits
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Ref
from swarm_arena._model.tooling import Command, CompletedRun, RunStatus, ToolOutput, ToolRequest
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
        "timeout_s": 30.0,
        "grace_s": 1.0,
        "artifact_root": str(tmp_path / "artifacts"),
    }
    return make_job(**{**fields, **kw})


def command_job(tmp_path: Path, command: Command, **kw: Any) -> Job:
    fields: dict[str, Any] = {
        "id": "k-1",
        "command": command,
        "timeout_s": 30.0,
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
    link = tmp_path / "py"
    link.symlink_to(sys.executable)
    out2 = await local_executor.submit(
        command_job(tmp_path, Command(argv=["python", "-c", code], python=str(link)))
    )
    assert out2.data["exe"] == str(link)


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
    code = WRITE.format("{'data': {'step': 1}}") + "; import time; time.sleep(60)"
    started = time.monotonic()
    out = await local_executor.submit(
        tool_job(tmp_path, code, timeout_s=1.0, tool_timeout_in_config=100.0)
    )
    assert (out.status, out.partial, out.data) == (RunStatus.TIMEOUT, True, {"step": 1})
    assert time.monotonic() - started < 20


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
        deadline = time.monotonic() + 10
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
    assert got[0].logs_ref is not None
    assert got[0].logs_ref.startswith("sha256:")
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


async def test_workspace_root_parent_must_be_safe(tmp_path: Path) -> None:  # M5
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
) -> None:  # M6
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    job = tool_job(tmp_path, WRITE.format("{'data': {}}"))
    out = await LocalExecutor(keep_workspaces=True).submit(job)
    assert out.status is RunStatus.OK
    [workspace] = list((tmp_path / f"swarm-arena-{os.getuid()}").iterdir())
    assert workspace.name.startswith("c-1-")


async def test_job_id_is_sanitized_in_the_workspace_name(tmp_path: Path) -> None:  # M11
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
    assert b"collect broke" in log(tmp_path, out, 2)
    assert out.started_at is not None  # M11: the executor stamps the run times itself
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
    fifo = (
        "import os; os.remove('swarm/stderr.log'); os.mkfifo('swarm/stderr.log'); "
        if fifo_stderr
        else ""
    )
    pid_file = str(tmp_path / "pid")
    code = (
        fifo + f"import os, time; open({pid_file!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    )
    RELEASE.clear()  # C40: a set event from an earlier case would not block collect
    try:
        with anyio.fail_after(10):  # collect blocks for 30 s unless it is abandoned
            out = await local_executor.submit(
                blocking_job(tmp_path, code, timeout_s=1.0, grace_s=0.2)
            )
        assert (out.status, out.partial) == (RunStatus.TIMEOUT, False)
        assert wait_until(lambda: not pid_alive(int((tmp_path / "pid").read_text())), 5)
        if not fifo_stderr:
            assert b"before_kill timed out" in log(tmp_path, out, 2)
            assert b"no partial output" in log(tmp_path, out, 2)  # M11
            assert out.started_at is not None
            assert out.ended_at is not None
            assert out.started_at <= out.ended_at
        else:
            assert log(tmp_path, out, 2) == b""
    finally:
        RELEASE.set()


async def test_fifo_log_does_not_hang(local_executor: LocalExecutor, tmp_path: Path) -> None:
    code = (
        "import os; os.remove('swarm/stdout.log'); os.mkfifo('swarm/stdout.log'); "
        + WRITE.format("{'data': {}}")
    )
    with anyio.fail_after(10):
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
    with anyio.fail_after(10):  # the "missing result" note goes through note_in_log: dropped
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
    monkeypatch.setattr(os, "name", "nt")  # C43: the last step, so no Path is built while it holds
    with pytest.raises(RuntimeError, match="POSIX"):
        ex.preflight()


async def test_confirm_dead_and_kill_orphan(local_executor: LocalExecutor) -> None:
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
        deadline = time.monotonic() + 5
        while not await local_executor.confirm_dead(started):  # an unreaped zombie is dead (C43)
            assert time.monotonic() < deadline, "the killed group still looks alive"
            await anyio.sleep(0.05)
        assert pid_alive(proc.pid)  # still a zombie: nothing has reaped it yet
        await anyio.to_thread.run_sync(reap, proc)  # C41
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
) -> None:  # M11: this host's own id ends in ':unknown', so the record matches it
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


# --- Fix round 1 ---


def assert_secret_safe(error: BaseException, secret: str) -> None:
    for text in (str(error), repr(error), repr(error.args), "".join(format_exception(error))):
        assert secret not in text
    if isinstance(error, ValidationError):
        assert secret not in repr(error.errors())
        assert secret not in error.json()
    assert error.__cause__ is None


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
    local_executor: LocalExecutor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: float,
    kept: bool,
) -> None:  # L1
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", budget)  # bounds the re-collect
    got: list[ToolOutput] = []

    async def on_partial(o: ToolOutput) -> None:
        got.append(o)

    async def collecting() -> bool:
        return COLLECTING.is_set()

    COLLECTING.clear()
    code = WRITE.format("{'data': {'v': 7}}")  # exits 0 with a valid result
    job = tool_job(tmp_path, code, tool_cls=SlowCollectTool, on_partial=on_partial)
    await cancel_when(lambda: local_executor.submit(job), collecting)
    [out] = got
    assert out.status is RunStatus.CANCELLED
    assert (out.partial, out.data) == ((True, {"v": 7}) if kept else (False, {}))
    assert out.started_at is not None
    assert out.ended_at is not None
    assert out.started_at <= out.ended_at
    if not kept:
        assert b"no partial output" in log(tmp_path, out, 2)
    assert list((tmp_path / "ws").iterdir()) == []


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_stop_facts_are_stamped_on_the_partial(
    local_executor: LocalExecutor, tmp_path: Path, stop: str
) -> None:  # L2: OkCollectTool returns status OK and partial False whatever happened
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
    local_executor: LocalExecutor, tmp_path: Path
) -> None:  # M8
    got: list[ToolOutput] = []

    async def on_partial(o: ToolOutput) -> None:
        got.append(o)

    async def staging() -> bool:
        return STAGING.is_set()

    STAGING.clear()
    marker = tmp_path / "ran"
    job = tool_job(
        tmp_path, f"open({str(marker)!r}, 'w')", tool_cls=SlowCommandTool, on_partial=on_partial
    )
    await cancel_when(lambda: local_executor.submit(job), staging)
    await anyio.sleep(0.5)
    assert got == []
    assert not marker.exists()
    assert list((tmp_path / "ws").iterdir()) == []


async def test_failed_cancel_path_upload_sends_no_unstored_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # M3
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
    assert "uploads failed: StorageError" in (workspace / "swarm" / "stderr.log").read_text()


async def test_missing_cwd_is_a_job_error(local_executor: LocalExecutor, tmp_path: Path) -> None:
    out = await local_executor.submit(  # M2
        command_job(tmp_path, Command(argv=["python", "-c", "pass"], cwd="missing/dir"))
    )
    assert (out.status, out.files) == (RunStatus.ERROR, [])
    assert b"command cwd missing/dir is not a directory in the workspace" in log(tmp_path, out, 2)
    assert log(tmp_path, out, 1) == b""
    assert list((tmp_path / "ws").iterdir()) == []


async def test_missing_stage_artifact_raises_and_cleans_up(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:  # M2 / Raises: a stage artifact that is not in the store
    job = command_job(
        tmp_path, Command(argv=["true"]), stage=Job.Stage(root={"problem/in.txt": "e" * 64})
    )
    with pytest.raises(NotFound):
        await local_executor.submit(job)
    assert list((tmp_path / "ws").iterdir()) == []


@pytest.mark.parametrize("entry", ["fields", "model_validate_json"])
def test_config_errors_hide_values(entry: str) -> None:  # M1
    secret = "SECRET_SENTINEL_local_executor"

    def build() -> object:
        if entry == "fields":
            return LocalExecutor(workspace_root={"token": secret})
        return LocalExecutor.Config.model_validate_json(json.dumps({"workspace_root": [secret]}))

    with pytest.raises(ValidationError, match="workspace_root") as caught:
        build()
    assert_secret_safe(caught.value, secret)
    with pytest.raises(ValidationError, match="max_concurrent") as caught:
        LocalExecutor(max_concurrent=secret)
    assert_secret_safe(caught.value, secret)


async def test_rlimits_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # M11
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
async def test_memory_gib_sets_the_address_space_limit(tmp_path: Path) -> None:  # M11
    code = "import resource; " + WRITE.format("{'as': resource.getrlimit(resource.RLIMIT_AS)[0]}")
    job = command_job(
        tmp_path, Command(argv=["python", "-c", code]), resources=Resources(memory_gib=0.5)
    )
    out = await LocalExecutor(workspace_root=str(tmp_path / "ws")).submit(job)
    assert (out.status, out.data) == (RunStatus.OK, {"as": 2**29})


async def test_sparse_log_upload_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = 1 << 20  # L3, K4: every driver read path is counted while submit runs
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


class _CountingStream:
    """Wrap a file object and record the size of everything read through it."""

    def __init__(self, stream: Any, sink: list[int]) -> None:
        self._stream = stream
        self._sink = sink

    def read(self, *args: Any) -> Any:
        data = self._stream.read(*args)
        self._sink.append(len(data))
        return data

    def __enter__(self) -> "_CountingStream":
        self._stream.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stream.__exit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextlib.contextmanager
def count_reads(monkeypatch: pytest.MonkeyPatch, sink: list[int]) -> Generator[None]:
    """Count bytes read through os.read, os.pread, os.fdopen streams, and io.open streams."""
    with monkeypatch.context() as patch:
        real_read, real_pread, real_fdopen, real_open = os.read, os.pread, os.fdopen, io.open

        def read(fd: int, n: int) -> bytes:
            data = real_read(fd, n)
            sink.append(len(data))
            return data

        def pread(fd: int, n: int, offset: int) -> bytes:
            data = real_pread(fd, n, offset)
            sink.append(len(data))
            return data

        def fdopen(*args: Any, **kw: Any) -> Any:
            return _CountingStream(real_fdopen(*args, **kw), sink)

        def open_(*args: Any, **kw: Any) -> Any:
            return _CountingStream(real_open(*args, **kw), sink)

        patch.setattr(os, "read", read)
        patch.setattr(os, "pread", pread)
        patch.setattr(os, "fdopen", fdopen)
        patch.setattr(io, "open", open_)
        yield


async def test_stderr_tail_keeps_its_own_truncation_note(tmp_path: Path) -> None:  # L3
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


async def test_outputs_over_the_cap_are_an_error(tmp_path: Path) -> None:  # L3
    code = (
        "import os; os.makedirs('out'); open('out/a.txt', 'w').write('a' * 600); "
        "open('out/b.txt', 'w').close(); os.truncate('out/b.txt', 2**30); "  # sparse 1 GiB
        + WRITE.format("{'data': {}}")
    )
    ex = LocalExecutor(workspace_root=str(tmp_path / "ws"), max_output_bytes=1024)
    with anyio.fail_after(20):
        out = await ex.submit(tool_job(tmp_path, code, tool_cls=UnreadFilesTool))
    assert (out.status, out.files) == (RunStatus.ERROR, [])
    assert b"out/b.txt crosses max_output_bytes" in log(tmp_path, out, 2)


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
) -> None:  # L4
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
) -> None:  # L4
    def broken(path: Path) -> None:
        raise RecursionError("too deep")

    monkeypatch.setattr(local, "_REMOVERS", (broken,))
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
import anyio
from _fakes.tools import BlockingCollectTool, RELEASE, make_job
from swarm_arena._exec import supervisor
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._model.ref import Ref
from swarm_arena._model.tooling import ToolRequest

supervisor.BEFORE_KILL_TIMEOUT_S = 0.5
RELEASE.clear()  # collect blocks for 30 s


async def main() -> None:
    tool = Ref.of(BlockingCollectTool({"code": "import time; time.sleep(60)"}))
    job = make_job(kind="tool", command=None, tool=tool, request=ToolRequest(),
                   timeout_s=0.5, grace_s=0.1, artifact_root=sys.argv[2])
    out = await LocalExecutor(workspace_root=sys.argv[1]).submit(job)
    print(out.status.value, flush=True)


anyio.run(main)
"""


def test_blocked_collect_does_not_keep_the_interpreter_alive(tmp_path: Path) -> None:  # L5
    tests_dir = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(tests_dir)}
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", _EXIT_SCRIPT, str(tmp_path / "ws"), str(tmp_path / "artifacts")],
        capture_output=True,
        text=True,
        env=env,
        timeout=25,
        check=False,
    )
    assert (done.returncode, done.stdout.strip()) == (0, "timeout"), done.stderr
    assert time.monotonic() - started < 15  # the blocked collect thread would hold it for 30 s


# --- Fix round 2 ---


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
) -> None:  # K1, K2
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
            started = time.monotonic()
            local._remove_tree(wide_tree)
            elapsed = time.monotonic() - started
        assert not os.path.lexists(wide_tree)
        assert calls["listdir"] <= n + 2  # each directory listed exactly once
        assert calls["stat"] <= 2 * (n + 10)
        assert elapsed < 20
        assert (outside / "keep").read_text() == "k"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o500
    finally:
        outside.chmod(0o700)


def test_unlock_uses_a_no_follow_chmod_where_the_os_has_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # K2: macOS and the BSDs
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def chmod(*args: Any, **kw: Any) -> None:
        calls.append((args, kw))

    with monkeypatch.context() as patch:
        patch.setattr(os, "chmod", chmod)
        patch.setattr(os, "supports_follow_symlinks", {chmod})
        patch.setattr(os, "supports_dir_fd", {chmod})
        unlocked = local._unlock_directory(7, "d")
    assert unlocked is True
    assert calls == [(("d", 0o700), {"dir_fd": 7, "follow_symlinks": False})]


def test_unlock_uses_o_path_and_proc_on_linux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:  # K2: Linux has no fchmodat(AT_SYMLINK_NOFOLLOW) for directories
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
        patch.setattr(local, "_PROC_FD", str(tmp_path))  # stands in for /proc/self/fd
        patch.setattr(os, "open", open_)
        patch.setattr(os, "chmod", chmod)
        patch.setattr(os, "close", closed.append)
        unlocked = local._unlock_directory(7, "d")
    assert unlocked is True
    assert opened == [("d", o_path | os.O_NOFOLLOW | os.O_DIRECTORY, 7)]
    assert chmods == [(f"{tmp_path}/99", 0o700)]
    assert closed == [99]


def test_unlock_without_a_safe_method_skips_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:  # K2: never fall back to a chmod that follows links
    chmods: list[tuple[Any, ...]] = []
    with monkeypatch.context() as patch:
        patch.setattr(os, "supports_follow_symlinks", set())
        patch.delattr(os, "O_PATH", raising=False)
        patch.setattr(os, "chmod", lambda *args, **kw: chmods.append(args))
        with caplog.at_level(logging.WARNING):
            unlocked = local._unlock_directory(7, "d")
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
    local_executor: LocalExecutor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    path: str,
) -> None:  # K3
    monkeypatch.setattr(
        OkCollectTool, "collect", lambda self, request, workspace, run: _not_a_tool_output(kind)
    )
    if path == "cancel":
        got: list[ToolOutput] = []

        async def on_partial(o: ToolOutput) -> None:
            got.append(o)

        code = "open('ready', 'w'); import time; time.sleep(60)"
        job = tool_job(tmp_path, code, tool_cls=OkCollectTool, grace_s=0.2, on_partial=on_partial)
        await cancel_when(lambda: local_executor.submit(job), ready_file(tmp_path / "ws"))
        [out] = got
        status = RunStatus.CANCELLED
    elif path == "timeout":
        code = "import time; time.sleep(60)"
        job = tool_job(tmp_path, code, tool_cls=OkCollectTool, timeout_s=0.5, grace_s=0.2)
        out = await local_executor.submit(job)
        status = RunStatus.TIMEOUT
    else:
        out = await local_executor.submit(tool_job(tmp_path, "pass", tool_cls=OkCollectTool))
        status = RunStatus.ERROR
    # Nothing valid was kept, so partial is False on every path (fix round 3 ruling).
    assert (out.status, out.partial, out.data, out.files) == (status, False, {}, [])
    assert b"collect failed: Tool.collect() did not return a ToolOutput" in log(tmp_path, out, 2)
    assert out.started_at is not None


@pytest.mark.parametrize(
    ("result", "partial"),
    [
        (None, False),  # no result.json: the valid output keeps no data
        ("{'data': {}, 'files': ['out/x.txt']}", True),  # a file is kept, so it is partial
        ("{'data': {'step': 3}}", True),
    ],
)
async def test_stopped_run_is_partial_only_when_something_is_kept(
    local_executor: LocalExecutor, tmp_path: Path, result: str | None, partial: bool
) -> None:  # fix round 3: partial=True only when the collect kept data or files
    write = "" if result is None else WRITE.format(result) + "; "
    code = (
        "import os; os.makedirs('out'); open('out/x.txt', 'w').write('x'); "
        + write
        + "import time; time.sleep(60)"
    )
    out = await local_executor.submit(tool_job(tmp_path, code, timeout_s=0.5, grace_s=0.2))
    assert (out.status, out.partial) == (RunStatus.TIMEOUT, partial)
    kept = (out.data, [ref.path for ref in out.files])
    if result is None:
        assert kept == ({}, [])
    elif partial and not out.data:
        assert kept == ({}, ["out/x.txt"])
    else:
        assert kept == ({"step": 3}, [])
