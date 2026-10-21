"""Enforce executor-owned collection budgets before hashing or model expansion."""

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from _fakes.read_counts import count_reads
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import make_job

from swarm_arena import Ref, ScriptTool
from swarm_arena._exec import local_run, local_stage, local_workspace
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.canonical import sha256_hex, utc_now
from swarm_arena._tooling import result_file
from swarm_arena.executor import Job, LocalExecutor
from swarm_arena.tool import Command, CompletedRun, FileRef, RunStatus, ToolOutput, ToolRequest


def script_job(tmp_path: Path, size: int) -> Job:
    code = (
        "import os, pathlib; pathlib.Path('answer').touch(); "
        f"os.truncate('answer', {size}); "
        'pathlib.Path(\'swarm/result.json\').write_text(\'{"data": {}, "files": ["answer"]}\')'
    )
    return make_job(
        id="bounded",
        kind="tool",
        tool=Ref.of(ScriptTool(command=["python", "-c", code])),
        request=ToolRequest(),
        stage=Job.Stage(),
        artifact_root=str(tmp_path / "artifacts"),
    )


def runner_for(tmp_path: Path, limit: int) -> tuple[local_run._JobRun, CompletedRun]:
    job = make_job(
        id="bounded",
        kind="command",
        command=Command(argv=["true"]),
        stage=Job.Stage(),
        artifact_root=str(tmp_path / "artifacts"),
    )
    workspace, artifacts = local_workspace._open_workspace(str(tmp_path / "ws"), job)
    staged = local_stage._stage(job, workspace, artifacts)
    run = CompletedRun(
        returncode=0,
        status=RunStatus.OK,
        stdout_path=workspace / "swarm/stdout.log",
        stderr_path=workspace / "swarm/stderr.log",
        started_at=utc_now(),
        ended_at=utc_now(),
    )
    return local_run._JobRun(
        job, staged, workspace, artifacts, LocalExecutor.Config(max_output_bytes=limit)
    ), run


async def test_script_byte_budget_refuses_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashed: list[str] = []

    def hash_file(workspace: Path, rel: str, *, max_bytes: int) -> tuple[str, int]:
        hashed.append(rel)
        return sha256_hex(b""), (workspace / rel).stat().st_size

    monkeypatch.setattr(result_file, "_hash_regular_file", hash_file)
    out = await LocalExecutor(workspace_root=str(tmp_path / "ws"), max_output_bytes=1).submit(
        script_job(tmp_path, 2)
    )
    assert hashed == []
    assert (out.status, out.files) == (RunStatus.ERROR, [])


async def test_script_byte_budget_above_old_default_reaches_hash_and_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 256 * 2**20 + 1
    limit = size + 7
    seen: list[tuple[str, int | None]] = []
    file = FileRef(name="answer", path="answer", sha256=sha256_hex(b""), size=size)

    def hash_file(workspace: Path, rel: str, *, max_bytes: int) -> tuple[str, int]:
        assert (workspace / rel).stat().st_size == size
        seen.append(("hash", max_bytes))
        return file.sha256, size

    def upload(*args: Any, max_bytes: int | None, **kw: Any) -> list[FileRef]:
        seen.append(("upload", max_bytes))
        return [file]

    monkeypatch.setattr(result_file, "_hash_regular_file", hash_file)
    monkeypatch.setattr(local_run, "collect_output_files", upload)
    out = await LocalExecutor(workspace_root=str(tmp_path / "ws"), max_output_bytes=limit).submit(
        script_job(tmp_path, size)
    )
    assert (out.status, out.files) == (RunStatus.OK, [file])
    assert seen == [("hash", limit), ("upload", limit)]


async def test_script_real_hash_and_upload_use_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 3 * 2**20 + 13
    reads: list[int] = []
    with count_reads(monkeypatch, reads):
        out = await LocalExecutor(
            workspace_root=str(tmp_path / "ws"), max_output_bytes=size
        ).submit(script_job(tmp_path, size))
    assert out.status == RunStatus.OK
    assert out.files[0].size == size
    assert max(reads) <= 2**20
    # Both collection and upload read actual bytes; no sparse-file hashing shortcut.
    assert sum(reads) >= 2 * size


@pytest.mark.parametrize(
    "failure", [None, ValueError, SystemExit, KeyboardInterrupt, asyncio.CancelledError]
)
async def test_collection_context_is_daemon_local_and_always_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException] | None,
) -> None:
    runner, run = runner_for(tmp_path, 17)
    seen: list[tuple[int | None, bool]] = []
    error = None if failure is None else failure()

    def collect(*args: Any) -> ToolOutput:
        seen.append((result_file._OUTPUT_BYTE_LIMIT.get(), threading.current_thread().daemon))
        if error is not None:
            raise error
        return ToolOutput(status=RunStatus.OK)

    def invoke() -> None:
        token = result_file._OUTPUT_BYTE_LIMIT.set(91)
        try:
            if failure in (KeyboardInterrupt, asyncio.CancelledError):
                with pytest.raises(failure) as caught:
                    runner._collect(run)
                assert caught.value is error
            else:
                output = runner._collect(run)
                assert (output is None) == (failure is not None)
            assert result_file._OUTPUT_BYTE_LIMIT.get() == 91
        finally:
            result_file._OUTPUT_BYTE_LIMIT.reset(token)

    monkeypatch.setattr(local_run, "command_output_from_result", collect)
    parent = result_file._OUTPUT_BYTE_LIMIT.get()
    await run_sync_daemon(invoke)
    assert seen == [(17, True)]
    assert result_file._OUTPUT_BYTE_LIMIT.get() == parent


async def test_custom_file_count_refused_before_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, run = runner_for(tmp_path, 0)
    file = FileRef(name="empty", path="empty", sha256=sha256_hex(b""), size=0)
    secret = "779d465b-5453-4473-97ce-5b3e54605b9f"
    output = ToolOutput.model_construct(
        status=RunStatus.OK, files=[file] * 4097, data={"token": secret}
    )
    dumped: list[bool] = []
    original = ToolOutput.model_dump

    def dump(self: ToolOutput, *args: Any, **kw: Any) -> dict[str, Any]:
        dumped.append(True)
        return original(self, *args, **kw)

    monkeypatch.setattr(ToolOutput, "model_dump", dump)
    monkeypatch.setattr(local_run, "command_output_from_result", lambda *args: output)
    assert await run_sync_daemon(runner._collect, run) is None
    assert dumped == []
    stderr = run.stderr_path.read_text()
    assert "too many declared files" in stderr
    assert secret not in stderr


@pytest.mark.parametrize("count", [4096, 4097])
def test_custom_file_count_bound_precedes_open_at_zero_byte_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    runner, _ = runner_for(tmp_path, 0)
    files = [
        FileRef(
            name="779d465b-5453-4473-97ce-5b3e54605b9f",
            path=f"f{i}",
            sha256=sha256_hex(b""),
            size=0,
        )
        for i in range(count)
    ]
    opened: list[str] = []
    original = local_run.open_regular_file
    empty = tmp_path / "empty"
    empty.touch()

    def open_file(workspace: Path, rel: str) -> int:
        opened.append(rel)
        return original(tmp_path, "empty")

    monkeypatch.setattr(local_run, "open_regular_file", open_file)
    if count > 4096:
        with pytest.raises(ValueError, match="too many declared files") as caught:
            runner._check_output_sizes(files)
        assert_secret_absent(caught.value, "779d465b-5453-4473-97ce-5b3e54605b9f")
        assert opened == []
    else:
        runner._check_output_sizes(files)
        assert len(opened) == count
