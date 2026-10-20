import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.tooling import (
    Command,
    CompletedRun,
    Estimate,
    FileRef,
    ParentRef,
    RunStatus,
    ToolOutput,
    ToolRequest,
    Usage,
    check_parent_id,
    is_stored_edit_ref,
)

H = "a" * 64


def test_run_status_values() -> None:
    assert [s.value for s in RunStatus] == [
        "ok",
        "error",
        "timeout",
        "oom",
        "budget_exceeded",
        "cancelled",
        "provider_failed",
        "lost",
    ]


def test_usage_add_and_total() -> None:
    u = Usage(
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=3,
        cache_write_tokens=2,
        cache_write_5m_tokens=2,
    )
    assert u.total_tokens == 20  # the 5m/1h fields split cache_write_tokens
    assert (u + u).input_tokens == 20
    assert (u + u).cache_write_5m_tokens == 4
    assert (Usage() + Usage(usd=0.5)).usd == 0.5
    assert (Usage() + Usage()).usd is None


def test_tool_request_edits_serialize_as_hashes() -> None:
    dumped = ToolRequest(edits={"a.txt": "hi"}).model_dump(mode="json")
    assert dumped["edits"] == {"a.txt": "sha256:" + sha256_hex(b"hi")}
    assert ToolRequest.model_validate(dumped).model_dump(mode="json") == dumped  # no double hashing


def test_tool_request_edit_content_with_sha_prefix() -> (
    None
):  # only an exact stored reference is kept
    for content in ("sha256:hello", "sha256:" + "A" * 64):
        dumped = ToolRequest(edits={"a.txt": content}).model_dump(mode="json")
        assert dumped["edits"] == {"a.txt": "sha256:" + sha256_hex(content.encode())}
        assert ToolRequest.model_validate(dumped).model_dump(mode="json") == dumped


def test_tool_request_reserved_form_is_a_reference() -> None:
    ref = "sha256:" + "a" * 64
    dumped = ToolRequest(edits={"a.txt": ref}).model_dump(mode="json")
    assert dumped["edits"] == {"a.txt": ref}  # dumped unchanged by definition
    assert ToolRequest.model_validate(dumped).model_dump(mode="json") == dumped  # stable


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "/abs.txt",
        "swarm/result.json",
        "PARENTS/c1/x",
        "./p/a.txt",
        "p//a.txt",
        "a\x00b",
        "problem/\udcc5\udca1olver.py",  # a surrogateescape spelling
        "",
    ],
)
def test_tool_request_edit_keys_are_checked_paths(path: str) -> None:
    secret = "42cf8095-6b54-51af-84f8-5198d5b65e3b"
    for build in (
        lambda: ToolRequest(edits={path: secret}),
        lambda: ToolRequest.model_validate({"edits": {path: secret}}),
    ):
        with pytest.raises(ValidationError, match="edits") as caught:
            build()
        assert_secret_absent(caught.value, secret, sanitized=True)
    assert ToolRequest(edits={"p/a.txt": secret}).edits == {"p/a.txt": secret}


def test_parent_ref_rules() -> None:
    base = {"candidate_id": "c1", "run_id": "r", "tool_identity": "T"}
    for bad in (
        {"candidate_id": ".."},
        {"candidate_id": "a/b"},
        {"candidate_id": ""},
        {"candidate_id": "../../outside"},
        {"overlay": {"/abs": H}},
        {"overlay": {"../x": H}},
        {"overlay": {"swarm/x": H}},
        {"overlay": {"./a": H}},
        {"overlay": {"a": "nothex"}},
    ):
        with pytest.raises(ValidationError):
            ParentRef.model_validate({**base, **bad})


def test_tool_output_and_command() -> None:
    out = ToolOutput(
        status=RunStatus.OK, started_at=datetime(2026, 10, 18, 13, 52, 1, 123456, tzinfo=UTC)
    )
    assert out.model_dump(mode="json")["started_at"] == "2026-10-18T13:52:01.123Z"
    assert pickle.loads(pickle.dumps(out)) == out  # pickles by public path
    c = Command(argv=["python", "run.py"])
    assert (c.cwd, c.env, c.timeout_s, c.python, c.grace_s) == (None, [], None, None, 10.0)
    for bad in (
        {"argv": []},
        {"argv": ["x"], "env": ["BAD-NAME"]},
        {"argv": ["x"], "timeout_s": 0},
    ):
        with pytest.raises(ValidationError):
            Command.model_validate(bad)


def test_file_ref_rules() -> None:
    for bad in (
        {"name": "a", "path": "/a", "sha256": H, "size": 1},
        {"name": "a", "path": "a", "sha256": "x", "size": 1},
        {"name": "a", "path": "a", "sha256": H, "size": -1},
    ):
        with pytest.raises(ValidationError):
            FileRef.model_validate(bad)


def test_usage_sums_every_field() -> None:
    u = Usage(
        input_tokens=1,
        output_tokens=2,
        cache_read_tokens=3,
        cache_write_tokens=4,
        cache_write_5m_tokens=1,
        cache_write_1h_tokens=3,
        usd=0.25,
    )
    assert u + u == Usage(
        input_tokens=2,
        output_tokens=4,
        cache_read_tokens=6,
        cache_write_tokens=8,
        cache_write_5m_tokens=2,
        cache_write_1h_tokens=6,
        usd=0.5,
    )
    assert (u + Usage()).usd == 0.25
    assert (Usage(usd=0) + Usage()).usd == 0
    assert u.total_tokens == 10


@pytest.mark.parametrize("content", ["", "λ\n", "sha256:" + H + "\n", "sha256:" + H[:-1]])
def test_tool_request_hashes_content_in_all_modes(content: str) -> None:
    request = ToolRequest(edits={"a": content})
    expected = {"a": "sha256:" + sha256_hex(content.encode())}
    assert request.model_dump()["edits"] == expected
    assert request.model_dump(mode="json")["edits"] == expected
    assert (
        ToolRequest.model_validate_json(request.model_dump_json()).model_dump()["edits"] == expected
    )
    assert request.edits == {"a": content}


@pytest.mark.parametrize("value", ["c1", "A-1_2.3", "x", "a.."])
def test_check_parent_id_accepts_component(value: str) -> None:
    assert check_parent_id(value) == value
    assert ParentRef(candidate_id=value, run_id="r", tool_identity="T").candidate_id == value


@pytest.mark.parametrize("value", [".", "..", "a/b", "a\\b", "", "a\n", "x\x00", "_x"])
def test_check_parent_id_rejects_component(value: str) -> None:
    with pytest.raises(ValueError, match="not one path component"):
        check_parent_id(value)


@pytest.mark.parametrize("suffix", [H, "A" * 64, H + "\n", H[:-1], H + "a", "hello"])
def test_is_stored_edit_ref_is_exact(suffix: str) -> None:
    value = "sha256:" + suffix
    assert is_stored_edit_ref(value) is (suffix == H)
    request = ToolRequest(edits={"a": value})
    expected = value if suffix == H else "sha256:" + sha256_hex(value.encode())
    assert request.model_dump()["edits"] == {"a": expected}


@pytest.mark.parametrize(
    "bad",
    [
        {"cwd": "/abs"},
        {"cwd": "../x"},
        {"cwd": "swarm"},
        {"env": ["1BAD"]},
        {"env": ["X\n"]},
        {"env": [""]},
        {"timeout_s": -1},
        {"grace_s": -1},
    ],
)
def test_command_rejects_invalid_fields(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Command.model_validate({"argv": ["x"], **bad})


def test_command_normalizes_cwd_and_accepts_env_names() -> None:
    command = Command(
        argv=["x"], cwd="./problem//sub", env=["_X", "a1", "PATH"], timeout_s=0.1, grace_s=0
    )
    assert command.cwd == "problem/sub"
    assert command.env == ["_X", "a1", "PATH"]
    assert command.grace_s == 0


def test_completed_run_and_estimate() -> None:
    now = datetime(2026, 10, 18, 13, 52, 1, 123456, tzinfo=UTC)
    run = CompletedRun(
        returncode=None,
        status=RunStatus.CANCELLED,
        stdout_path=Path("stdout"),
        stderr_path=Path("stderr"),
        started_at=now,
        ended_at=now,
    )
    assert run.started_at.microsecond == 123000
    assert run.model_dump(mode="json")["ended_at"] == "2026-10-18T13:52:01.123Z"
    assert CompletedRun.model_validate_json(run.model_dump_json()) == run
    assert pickle.loads(pickle.dumps(run)) == run
    assert Estimate().model_dump() == {"seconds": None, "tokens": None, "usd": None}
    assert Estimate(seconds=1, tokens=2, usd=3).model_dump() == {
        "seconds": 1.0,
        "tokens": 2,
        "usd": 3.0,
    }


def test_tool_output_and_completed_run_reject_naive_times() -> None:
    now = datetime(2026, 10, 18, tzinfo=UTC)
    naive = now.replace(tzinfo=None)
    for field in ("started_at", "ended_at"):
        with pytest.raises(ValidationError):
            ToolOutput.model_validate({"status": "ok", field: naive})
        with pytest.raises(ValidationError):
            CompletedRun.model_validate(
                {
                    "returncode": 0,
                    "status": "ok",
                    "stdout_path": "out",
                    "stderr_path": "err",
                    "started_at": now,
                    "ended_at": now,
                    field: naive,
                }
            )


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("nested", [False, True])
def test_tool_output_rejects_nonfinite_data_without_values(number: float, nested: bool) -> None:
    secret = "9bceb674-b923-57a8-b838-b8af9fc884bf"
    data = {secret: [{"loss": number, "note": secret}]} if nested else {secret: number}
    with pytest.raises(ValidationError) as caught:
        ToolOutput(status=RunStatus.OK, data=data)
    error = caught.value
    assert error.errors()[0]["loc"] == ("data",)
    assert_secret_absent(error, secret, sanitized=True)


def test_tool_output_preserves_finite_json_data() -> None:
    data = {"nested": [{"loss": 0.25}, None, True, "ok", 3], "empty": {}}
    assert ToolOutput(status=RunStatus.OK, data=data).data == data
