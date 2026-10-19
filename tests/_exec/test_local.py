import os
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.proc import pid_alive, reap, spawn_group, wait_until
from _fakes.tools import (
    RELEASE,
    WRITE,
    BadCollectTool,
    BlockingCollectTool,
    GhostFileTool,
    PyTool,
    make_job,
)
from pydantic import ValidationError

from swarm_arena._exec import supervisor
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._exec.procinfo import host_id, process_start_time
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Ref
from swarm_arena._model.tooling import Command, RunStatus, ToolOutput, ToolRequest
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


async def test_unsafe_workspace_root_refused(tmp_path: Path) -> None:
    ws = tmp_path / "shared"
    ws.mkdir()
    ws.chmod(0o777)
    with pytest.raises(RuntimeError, match="workspace root"):
        await LocalExecutor(workspace_root=str(ws)).submit(tool_job(tmp_path, "pass"))
    assert list(ws.iterdir()) == []
    ws.chmod(0o1777)  # sticky, like /tmp: other users cannot rename or remove our entries
    job = tool_job(tmp_path, WRITE.format("{'data': {}}"))
    assert (await LocalExecutor(workspace_root=str(ws)).submit(job)).status is RunStatus.OK


async def test_collect_exception_becomes_error(
    local_executor: LocalExecutor, tmp_path: Path
) -> None:
    out = await local_executor.submit(tool_job(tmp_path, "pass", tool_cls=BadCollectTool))
    assert out.status is RunStatus.ERROR
    assert b"collect broke" in log(tmp_path, out, 2)


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
        unknown = started.model_copy(update={"host_id": f"{me.rsplit(':', 1)[0]}:unknown"})
        assert await local_executor.confirm_dead(unknown) is None
        dispatch = Job.Started(kind="dispatch", job_id="c-1", task_name="t")
        assert await local_executor.confirm_dead(dispatch) is None
    finally:
        await anyio.to_thread.run_sync(reap, proc)


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
