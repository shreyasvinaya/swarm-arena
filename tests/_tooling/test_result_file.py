import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import pytest

from swarm_arena._model.canonical import sha256_hex
from swarm_arena._tooling.result_file import (
    ToolResultFile,
    command_output_from_result,
    note_in_log,
    read_regular_file,
    read_result_json,
    request_json,
    symlinked_component,
    tool_output_from_result,
)
from swarm_arena.tool import CompletedRun, RunStatus, ToolRequest


def run(workspace: Path, status: RunStatus, rc: int | None) -> CompletedRun:
    (workspace / "swarm").mkdir(parents=True, exist_ok=True)
    stdout = workspace / "swarm/stdout.log"
    stderr = workspace / "swarm/stderr.log"
    stdout.touch()
    stderr.touch()
    return CompletedRun(
        returncode=rc,
        status=status,
        stdout_path=stdout,
        stderr_path=stderr,
        started_at=datetime(2026, 10, 18, tzinfo=UTC),
        ended_at=datetime(2026, 10, 18, 0, 1, tzinfo=UTC),
    )


def write_result(workspace: Path, data: dict[str, Any]) -> None:
    (workspace / "swarm").mkdir(parents=True, exist_ok=True)
    (workspace / "swarm/result.json").write_text(json.dumps(data))


def prepare_unsafe_result(ws: Path, kind: str) -> None:
    ws.mkdir()
    outside = ws.parent / "outside"
    write_result(outside, {"data": {}})
    if kind == "symlinked_swarm_dir":
        (ws / "swarm").symlink_to(outside / "swarm", target_is_directory=True)
    else:
        (ws / "swarm").mkdir()
        p = ws / "swarm/result.json"
        if kind == "fifo":
            os.mkfifo(p)
        else:
            p.symlink_to(outside / "swarm/result.json")


def prepare_log(p: Path, kind: str) -> None:
    if kind == "regular":
        p.write_text("x\n")
    elif kind == "fifo":
        os.mkfifo(p)
    elif kind == "symlink":
        target = p.parent / "target"
        target.write_text("t")
        p.symlink_to(target)


def assert_log(p: Path, kind: str) -> None:
    if kind == "regular":
        assert p.read_text() == "x\nswarm-arena: m\n"
    elif kind == "missing":
        assert p.read_text() == "swarm-arena: m\n"
    elif kind == "symlink":
        assert (p.parent / "target").read_text() == "t"


def test_request_json_shape() -> None:
    req = ToolRequest(inputs={"lr": 0.1}, round=2, round_overrides={"epochs": 100}, replicate=1)
    assert request_json(req) == {
        "schema_version": 1,
        "params": {"lr": 0.1},
        "round": 2,
        "round_overrides": {"epochs": 100},
        "replicate": 1,
    }


def test_ok_with_valid_result(tmp_path: Path) -> None:
    write_result(
        tmp_path,
        {"data": {"f": 1.5}, "usage": {"input_tokens": 10, "usd": 0.01}, "files": ["out/p.txt"]},
    )
    (tmp_path / "out").mkdir()
    (tmp_path / "out/p.txt").write_text("p")
    out = tool_output_from_result(tmp_path, run(tmp_path, RunStatus.OK, 0))
    assert (out.status, out.partial, out.data, out.usage.input_tokens) == (
        RunStatus.OK,
        False,
        {"f": 1.5},
        10,
    )
    assert [(f.path, f.sha256) for f in out.files] == [("out/p.txt", sha256_hex(b"p"))]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "swarm/result.json is missing"),
        ("{nope", "not valid JSON"),
        ('{"data": [1]}', "data"),
        ('{"data": {}, "usgae": {}}', "<key>"),
        ('{"schema_version": 2, "data": {}}', "newer"),
        ('{"data": {}, "files": ["swarm/x"]}', "reserved"),
        ('{"data": {}, "files": ["out/none.txt"]}', "out/none.txt"),
        ('{"data": {"f": NaN}}', "non-finite number"),
        ('{"data": {}, "failed_provider": ""}', "failed_provider"),
        ('{"data": {"f": 1e999}}', "non-finite number"),
        ('{"data": {"g": [-1e999]}}', "non-finite number"),
        ('{"data": {}, "usage": {"usd": 1e999}}', "non-finite number"),
    ],
)
def test_exit_zero_without_valid_result_is_error(
    tmp_path: Path, content: str | None, reason: str
) -> None:  # Review Focus 3
    if content is not None:
        (tmp_path / "swarm").mkdir(exist_ok=True)
        (tmp_path / "swarm/result.json").write_text(content)
    r = run(tmp_path, RunStatus.OK, 0)
    out = tool_output_from_result(tmp_path, r)
    assert (out.status, out.data, out.files) == (RunStatus.ERROR, {}, [])
    assert "swarm-arena: " in r.stderr_path.read_text()
    assert reason in r.stderr_path.read_text()


@pytest.mark.parametrize("status", [RunStatus.TIMEOUT, RunStatus.CANCELLED])
def test_killed_run_with_result_is_partial(tmp_path: Path, status: RunStatus) -> None:
    write_result(tmp_path, {"data": {"step": 3}})
    out = tool_output_from_result(tmp_path, run(tmp_path, status, None))
    assert (out.status, out.partial, out.data) == (status, True, {"step": 3})
    empty = tool_output_from_result(
        tmp_path / "none", run(tmp_path / "none", status, None)
    )  # no result file
    assert (empty.partial, empty.data) == (False, {})


def test_error_keeps_usage_only(tmp_path: Path) -> None:
    write_result(tmp_path, {"data": {"x": 1}, "usage": {"output_tokens": 7}})
    out = tool_output_from_result(tmp_path, run(tmp_path, RunStatus.ERROR, 1))
    assert (out.status, out.data, out.usage.output_tokens) == (RunStatus.ERROR, {}, 7)


@pytest.mark.parametrize(("status", "rc"), [(RunStatus.OK, 0), (RunStatus.ERROR, 1)])
def test_failed_provider_key_is_provider_failed(
    tmp_path: Path, status: RunStatus, rc: int
) -> None:  # roadmap R4.2
    write_result(tmp_path, {"data": {"x": 1}, "usage": {"usd": 0.2}, "failed_provider": "openai"})
    out = tool_output_from_result(tmp_path, run(tmp_path, status, rc))
    assert (out.status, out.failed_provider, out.partial, out.data, out.files, out.usage.usd) == (
        RunStatus.PROVIDER_FAILED,
        "openai",
        False,
        {},
        [],
        0.2,
    )
    killed = tool_output_from_result(
        tmp_path, run(tmp_path, RunStatus.TIMEOUT, None)
    )  # the kill status wins
    assert (killed.status, killed.partial, killed.data, killed.failed_provider) == (
        RunStatus.TIMEOUT,
        True,
        {"x": 1},
        None,
    )
    write_result(tmp_path, {"data": {}})
    assert tool_output_from_result(tmp_path, run(tmp_path, RunStatus.OK, 0)).failed_provider is None


def test_command_output_is_the_raw_object(tmp_path: Path) -> None:
    write_result(tmp_path, {"passed": True, "reasons": []})
    assert command_output_from_result(tmp_path, run(tmp_path, RunStatus.OK, 0)).data == {
        "passed": True,
        "reasons": [],
    }
    missing = command_output_from_result(tmp_path / "none", run(tmp_path / "none", RunStatus.OK, 0))
    assert (missing.status, missing.data) == (RunStatus.ERROR, {})


def test_command_output_rejects_non_finite(tmp_path: Path) -> None:
    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/result.json").write_text(
        '{"passed": true, "x": 1e999}'
    )  # raw text: json.dumps cannot write 1e999
    r = run(tmp_path, RunStatus.OK, 0)
    out = command_output_from_result(tmp_path, r)
    assert (out.status, out.data) == (
        RunStatus.ERROR,
        {},
    )
    assert "non-finite number" in r.stderr_path.read_text()


@pytest.mark.parametrize("kind", ["fifo", "symlink", "symlinked_swarm_dir"])
async def test_result_must_be_a_regular_file(tmp_path: Path, kind: str) -> None:
    ws = tmp_path / "ws"
    await anyio.to_thread.run_sync(prepare_unsafe_result, ws, kind)
    r = await anyio.to_thread.run_sync(run, ws, RunStatus.OK, 0)
    with anyio.fail_after(5):  # a FIFO opened without O_NONBLOCK would block forever
        out = await anyio.to_thread.run_sync(tool_output_from_result, ws, r, abandon_on_cancel=True)
    assert out.status is RunStatus.ERROR
    logged = await anyio.to_thread.run_sync(r.stderr_path.read_text)
    if kind == "symlinked_swarm_dir":
        # The diagnostic must also refuse the replaced parent directory.
        assert logged == ""
    else:
        assert "not a regular file" in logged


@pytest.mark.parametrize("kind", ["regular", "missing", "fifo", "symlink"])
async def test_note_in_log_never_blocks_or_follows(tmp_path: Path, kind: str) -> None:
    p = tmp_path / "stderr.log"
    await anyio.to_thread.run_sync(prepare_log, p, kind)
    with anyio.fail_after(5):
        await anyio.to_thread.run_sync(
            note_in_log, p, "m", abandon_on_cancel=True
        )  # returns None, raises nothing

    await anyio.to_thread.run_sync(assert_log, p, kind)


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "null",
        '"private"',
        "3",
        '{"data": {}, "usage": {"usd": "NaN"}}',
        '{"data": {}, "usage": {"usd": "Infinity"}}',
    ],
)
def test_result_rejects_nonobjects_and_coerced_nonfinite(tmp_path: Path, content: str) -> None:
    r = run(tmp_path, RunStatus.OK, 0)
    (tmp_path / "swarm/result.json").write_text(content)
    out = tool_output_from_result(tmp_path, r)
    assert (out.status, out.data, out.files) == (RunStatus.ERROR, {}, [])


@pytest.mark.parametrize(
    "status", [RunStatus.OK, RunStatus.ERROR, RunStatus.OOM, RunStatus.TIMEOUT, RunStatus.CANCELLED]
)
@pytest.mark.parametrize("valid", [True, False])
def test_result_status_matrix(tmp_path: Path, status: RunStatus, valid: bool) -> None:
    if valid:
        write_result(
            tmp_path,
            {
                "data": {"x": 1},
                "usage": {"usd": 0.2},
                "files": ["p.txt"],
                "failed_provider": "openai",
            },
        )
        (tmp_path / "p.txt").write_text("p")
    r = run(tmp_path, status, 0 if status == RunStatus.OK else None)
    out = tool_output_from_result(tmp_path, r)
    assert (out.started_at, out.ended_at) == (r.started_at, r.ended_at)
    if valid:
        assert out.usage.usd == 0.2
        if status in (RunStatus.TIMEOUT, RunStatus.CANCELLED):
            assert (out.status, out.partial, out.data, len(out.files), out.failed_provider) == (
                status,
                True,
                {"x": 1},
                1,
                None,
            )
        elif status == RunStatus.OOM:
            assert (out.status, out.partial, out.data, out.files, out.failed_provider) == (
                status,
                False,
                {},
                [],
                None,
            )
        else:
            assert (out.status, out.partial, out.data, out.files, out.failed_provider) == (
                RunStatus.PROVIDER_FAILED,
                False,
                {},
                [],
                "openai",
            )
    else:
        assert (out.status, out.partial, out.data, out.files, out.usage.usd) == (
            RunStatus.ERROR if status == RunStatus.OK else status,
            False,
            {},
            [],
            None,
        )


@pytest.mark.parametrize(
    "status", [RunStatus.OK, RunStatus.ERROR, RunStatus.OOM, RunStatus.TIMEOUT, RunStatus.CANCELLED]
)
def test_command_preserves_status_and_raw_data(tmp_path: Path, status: RunStatus) -> None:
    write_result(tmp_path, {"passed": True, "failed_provider": "not-a-tool-envelope"})
    r = run(tmp_path, status, None)
    out = command_output_from_result(tmp_path, r)
    assert out.status == status
    assert out.data == {"passed": True, "failed_provider": "not-a-tool-envelope"}
    assert (out.started_at, out.ended_at, out.partial, out.failed_provider) == (
        r.started_at,
        r.ended_at,
        False,
        None,
    )


@pytest.mark.parametrize(
    "content",
    [
        '{"data": [], "usage": {"usd": "SENTINEL_SECRET"}}',
        '{"data": {}, "SENTINEL_SECRET": "SENTINEL_SECRET"}',
        '{"data": {}, "usage": {"SENTINEL_SECRET": 1}}',
        '{"data": {}, "files": [123], "failed_provider": {"SENTINEL_SECRET": 1}}',
        '{"schema_version": "SENTINEL_SECRET", "data": {}}',
        '{"data": {"SENTINEL_SECRET": NaN}}',
        '{"data": "SENTINEL_SECRET"',
    ],
)
def test_result_diagnostics_hide_values_and_dynamic_keys(tmp_path: Path, content: str) -> None:
    r = run(tmp_path, RunStatus.OK, 0)
    (tmp_path / "swarm/result.json").write_text(content)
    out = tool_output_from_result(tmp_path, r)
    assert out.status == RunStatus.ERROR
    message = r.stderr_path.read_text()
    assert "swarm-arena: " in message
    assert "SENTINEL_SECRET" not in message


def test_json_decode_errors_are_sanitized(tmp_path: Path) -> None:
    r = run(tmp_path, RunStatus.OK, 0)
    (tmp_path / "swarm/result.json").write_bytes(b'{"SENTINEL_SECRET": "\xff"}')
    with pytest.raises(ValueError, match="not valid JSON") as exc:
        read_result_json(tmp_path)
    assert "SENTINEL_SECRET" not in "".join(traceback.format_exception(exc.value))
    assert command_output_from_result(tmp_path, r).status == RunStatus.ERROR


def test_read_helpers_and_model(tmp_path: Path) -> None:
    write_result(tmp_path, {"data": {}})
    assert read_result_json(tmp_path) == {"data": {}}
    assert read_regular_file(tmp_path, "swarm/result.json") == b'{"data": {}}'
    assert symlinked_component(tmp_path, "swarm/result.json") is None
    (tmp_path / "link").symlink_to(tmp_path / "swarm", target_is_directory=True)
    assert symlinked_component(tmp_path, "link/result.json") == "link"
    (tmp_path / "leaf").symlink_to(tmp_path / "swarm/result.json")
    assert symlinked_component(tmp_path, "leaf") == "leaf"
    assert symlinked_component(tmp_path, "missing/result.json") is None
    result = ToolResultFile(data={})
    assert result.schema_kind == "tool_result"
    assert result.schema_version == 1
    assert result.files == []
    assert result.usage.input_tokens == 0
    assert result.failed_provider is None


@pytest.mark.parametrize("rel", ["missing", "swarm", "../outside", "/absolute", "", "bad\0path"])
def test_read_regular_file_refuses_unsafe_or_nonregular(tmp_path: Path, rel: str) -> None:
    (tmp_path / "swarm").mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        read_regular_file(tmp_path, rel)


@pytest.mark.parametrize("kind", ["fifo", "symlink", "symlinked_swarm_dir"])
async def test_declared_files_must_be_regular(tmp_path: Path, kind: str) -> None:
    ws = tmp_path / "ws"
    await anyio.to_thread.run_sync(prepare_unsafe_result, ws, kind)

    # Use the unsafe entry as a declared artifact, separate from the result envelope.
    def collect() -> None:
        (ws / "swarm").rename(ws / "out")
        write_result(ws, {"data": {}, "files": ["out/result.json"]})
        r = run(ws, RunStatus.OK, 0)
        out = tool_output_from_result(ws, r)
        assert out.status == RunStatus.ERROR
        assert "not a regular file" in r.stderr_path.read_text()

    with anyio.fail_after(5):
        await anyio.to_thread.run_sync(collect, abandon_on_cancel=True)


def test_log_drops_oserrors_and_nonregular(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    note_in_log(tmp_path / "bad\0path", "invalid path")
    note_in_log(tmp_path, "directory")
    note_in_log(tmp_path / "missing/log", "missing directory")
    p = tmp_path / "log"
    p.write_text("original")

    def fail_write(fd: int, data: bytes) -> int:
        raise OSError("SENTINEL_SECRET")

    monkeypatch.setattr(os, "write", fail_write)
    assert note_in_log(p, "m") is None
    assert p.read_text() == "original"


def test_log_drops_fifo_with_reader(tmp_path: Path) -> None:
    p = tmp_path / "fifo"
    os.mkfifo(p)
    reader = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
    try:
        note_in_log(p, "m")
        assert os.read(reader, 1024) == b""
    finally:
        os.close(reader)


def test_read_refuses_parent_replaced_after_component_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._tooling import result_file

    workspace = tmp_path / "workspace"
    parent = workspace / "out"
    parent.mkdir(parents=True)
    (parent / "data").write_bytes(b"safe")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data").write_bytes(b"SENTINEL_SECRET")
    original = result_file.symlinked_component

    def swap(base: Path, rel: str) -> str | None:
        found = original(base, rel)
        parent.rename(workspace / "old-out")
        parent.symlink_to(outside, target_is_directory=True)
        return found

    monkeypatch.setattr(result_file, "symlinked_component", swap)
    with pytest.raises(ValueError, match="not a regular file"):
        read_regular_file(workspace, "out/data")


@pytest.mark.parametrize("existing", [True, False])
def test_log_refuses_symlinked_parent(tmp_path: Path, existing: bool) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "stderr.log"
    if existing:
        target.write_text("original\n")
    (workspace / "swarm").symlink_to(outside, target_is_directory=True)

    assert note_in_log(workspace / "swarm/stderr.log", "m") is None

    if existing:
        assert target.read_text() == "original\n"
    else:
        assert not target.exists()


def test_log_refuses_hardlinked_file(tmp_path: Path) -> None:
    target = tmp_path / "outside.log"
    target.write_text("original\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "stderr.log"
    path.hardlink_to(target)

    assert note_in_log(path, "m") is None
    assert target.read_text() == path.read_text() == "original\n"


@pytest.mark.parametrize("pinned", [False, True])
def test_log_parent_replacement_cannot_redirect_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned: bool
) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "swarm"
    parent.mkdir(parents=True)
    (parent / "stderr.log").write_text("original\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "stderr.log"
    target.write_text("outside\n")
    original_open = os.open
    swapped = False

    def swap_open(
        path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal swapped
        # Swap immediately before opening swarm, or after its fd was pinned.
        if Path(path).name == ("stderr.log" if pinned else "swarm"):
            parent.rename(workspace / "old-swarm")
            parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_open)
    assert note_in_log(parent / "stderr.log", "m") is None
    assert swapped
    assert target.read_text() == "outside\n"
    expected = "original\nswarm-arena: m\n" if pinned else "original\n"
    assert (workspace / "old-swarm/stderr.log").read_text() == expected


def test_log_sanitizes_control_characters(tmp_path: Path) -> None:
    path = tmp_path / "stderr.log"
    message = "first\nforged\r\x1b[31m\t\0\x7f\x85\u2028\u2029last café"
    assert note_in_log(path, message) is None
    logged = path.read_text()
    assert len(logged.splitlines()) == 1
    assert logged.startswith("swarm-arena: first")
    assert logged.endswith("last café\n")
    assert not any(char in logged[:-1] for char in "\n\r\x1b\t\0\x7f\x85\u2028\u2029")


@pytest.mark.parametrize("failure", [None, "open", "fstat", "write"])
def test_log_closes_all_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    parent = tmp_path / "swarm"
    parent.mkdir()
    path = parent / "stderr.log"
    opened: list[int] = []
    original_open = os.open
    original_fstat = os.fstat

    def track_open(
        path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if failure == "open" and Path(path).name == "stderr.log":
            raise OSError("SENTINEL_SECRET")
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(fd)
        return fd

    def fail_fstat(fd: int) -> os.stat_result:
        raise OSError("SENTINEL_SECRET")

    def fail_write(fd: int, data: bytes) -> int:
        raise OSError("SENTINEL_SECRET")

    monkeypatch.setattr(os, "open", track_open)
    if failure == "fstat":
        monkeypatch.setattr(os, "fstat", fail_fstat)
    if failure == "write":
        monkeypatch.setattr(os, "write", fail_write)

    assert note_in_log(path, "m") is None
    assert opened
    for fd in opened:
        with pytest.raises(OSError, match="Bad file descriptor"):
            original_fstat(fd)


def test_log_supports_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "swarm").mkdir()
    note_in_log(Path("swarm/stderr.log"), "relative")
    note_in_log(Path("stderr.log"), "leaf")
    assert (tmp_path / "swarm/stderr.log").read_text() == "swarm-arena: relative\n"
    assert (tmp_path / "stderr.log").read_text() == "swarm-arena: leaf\n"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("out.txt", "OUT.txt"),
        ("caf\u00e9.txt", "cafe\u0301.txt"),
        ("out.txt", "./out.txt"),
        ("out.txt", "out.txt"),
    ],
)
def test_folded_output_duplicates_fail_producing_run(
    tmp_path: Path, first: str, second: str
) -> None:
    secret = "secret-output-sentinel"
    (tmp_path / first).write_text(secret)
    (tmp_path / second).write_text(secret)
    write_result(tmp_path, {"data": {"value": secret}, "files": [first, second]})
    completed = run(tmp_path, RunStatus.OK, 0)
    out = tool_output_from_result(tmp_path, completed)
    assert (out.status, out.data, out.files) == (RunStatus.ERROR, {}, [])
    diagnostic = completed.stderr_path.read_text()
    assert "duplicate output path" in diagnostic
    assert first in diagnostic
    assert second in diagnostic
    assert secret not in diagnostic
