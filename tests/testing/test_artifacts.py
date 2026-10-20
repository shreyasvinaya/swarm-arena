"""Public access to verified output artifacts and complete binary logs."""

from functools import partial
from pathlib import Path

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import ScriptTool, testing
from swarm_arena._errors import NotFound, StorageError
from swarm_arena._model.tooling import format_logs_ref, parse_logs_ref
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena.executor import LocalExecutor
from swarm_arena.testing import _conformance
from swarm_arena.testing._conformance import _artifact_root, _artifacts
from swarm_arena.tool import FileRef, RunStatus, ToolOutput

SECRET = "47a8e694-f6b1-46ad-8519-157058fb3cdc"


@pytest.fixture(autouse=True)
def isolated_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_conformance, "_root", str(tmp_path / "artifacts"))


def output_with_artifacts() -> ToolOutput:
    files = [
        FileRef(name="same.bin", path=path, sha256=_artifacts().put_bytes(data), size=len(data))
        for path, data in (("a/same.bin", b"\x00\xffa"), ("b/same.bin", b"b"))
    ]
    stdout = _artifacts().put_bytes(b"stdout\x00\xff")
    stderr = _artifacts().put_bytes(b"stderr\xfe")
    return ToolOutput(status=RunStatus.OK, files=files, logs_ref=format_logs_ref(stdout, stderr))


def test_read_output_file_uses_exact_path_and_returns_bytes() -> None:
    out = output_with_artifacts()
    assert testing.read_output_file(out, "a/same.bin") == b"\x00\xffa"
    assert testing.read_output_file(out, "b/same.bin") == b"b"
    with pytest.raises(ValueError, match="output file"):
        testing.read_output_file(out, "same.bin")


def test_read_logs_returns_stdout_then_stderr_bytes() -> None:
    assert testing.read_logs(output_with_artifacts()) == (b"stdout\x00\xff", b"stderr\xfe")


@pytest.mark.parametrize("value", [None, SECRET, "sha256:" + SECRET + ":" + SECRET, 12])
def test_read_logs_rejects_missing_or_invalid_reference_without_values(value: object) -> None:
    out = ToolOutput.model_construct(status=RunStatus.OK, logs_ref=value)
    with pytest.raises(ValueError, match="logs_ref") as caught:
        testing.read_logs(out)
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("kind", ["file", "logs"])
@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_readers_verify_artifacts(kind: str, failure: str) -> None:
    out = output_with_artifacts()
    digest = out.files[0].sha256
    if kind == "logs":
        assert out.logs_ref is not None
        digest, _ = parse_logs_ref(out.logs_ref)
    path = Path(_artifact_root()) / digest
    if failure == "missing":
        path.unlink()
    else:
        path.write_bytes(SECRET.encode())
    read = (
        partial(testing.read_output_file, out, "a/same.bin")
        if kind == "file"
        else partial(testing.read_logs, out)
    )
    with pytest.raises((NotFound, StorageError)) as caught:
        read()
    assert_secret_absent(caught.value, SECRET)


async def test_read_artifacts_from_real_run_after_workspace_cleanup(
    tmp_path: Path, local_executor: LocalExecutor
) -> None:
    code = (
        "import json, pathlib, sys; "
        "pathlib.Path('result.bin').write_bytes(bytes([0, 255, 10])); "
        "sys.stdout.buffer.write(b'out\\x00'); sys.stderr.buffer.write(b'err\\xff'); "
        "pathlib.Path('swarm/result.json').write_text("
        "json.dumps({'data': {}, 'files': ['result.bin']}))"
    )
    out = await testing.check_tool(
        ScriptTool(command=["python", "-c", code]), {}, executor=local_executor
    )
    assert out.status == RunStatus.OK
    assert testing.read_output_file(out, "result.bin") == b"\x00\xff\n"
    assert testing.read_logs(out) == (b"out\x00", b"err\xff")


def test_artifact_helpers_are_public() -> None:
    for name in ("read_output_file", "read_logs"):
        assert name in testing.__all__
        assert getattr(testing, name).__module__ == "swarm_arena.testing"


@pytest.mark.parametrize("value", [None, SECRET, "sha256:" + SECRET + ":" + SECRET, 12])
def test_stderr_tail_rejects_malformed_reference_before_storage(
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> ArtifactStore:
        pytest.fail("malformed reference reached storage")

    monkeypatch.setattr(_conformance, "_artifacts", forbidden)
    out = ToolOutput.model_construct(status=RunStatus.ERROR, logs_ref=value)
    assert _conformance._stderr_tail(out) == "(no stderr log)"


def test_stderr_tail_reads_only_stderr_and_handles_missing_log() -> None:
    stderr = _artifacts().put_bytes(b"e" * 3000)
    # Stdout is deliberately missing: diagnostic tails must never read it.
    out = ToolOutput(status=RunStatus.ERROR, logs_ref=format_logs_ref("0" * 64, stderr))
    assert _conformance._stderr_tail(out) == "e" * 2048
    missing = ToolOutput(status=RunStatus.ERROR, logs_ref=format_logs_ref("0" * 64, "1" * 64))
    assert _conformance._stderr_tail(missing) == "(the stderr log cannot be read)"
