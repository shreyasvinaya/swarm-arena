"""Bounded, hostile-input-safe cooperating tool call reports."""

import os
from pathlib import Path
from typing import Any

import pytest

from swarm_arena._model.canonical import sha256_hex
from swarm_arena._tooling.llm_calls import (
    LLM_CALLS_PATH,
    MAX_LLM_CALLS_BYTES,
    hash_regular_file,
    llm_calls_source,
    may_call_llm,
    parse_llm_calls,
)


def test_parse_valid_rows() -> None:
    rows, warnings = parse_llm_calls(
        b'{"model":"openai/gpt-6","usage":{"input_tokens":10},"cost_usd":0.01}\n'
        b'{"schema_version":1,"model":"local-llm","litellm_call_id":"abc"}\n'
    )
    assert warnings == []
    assert [r.schema_version for r in rows] == [1, 1]
    assert rows[0].provider == "openai"
    assert rows[1].provider == "unknown"
    assert rows[1].litellm_call_id == "abc"


def test_parse_skips_bad_lines() -> None:
    rows, warnings = parse_llm_calls(b'{"model":"a/b"}\nnot json\n{"model":"a/b","cost_usd":NaN}\n')
    assert len(rows) == 1
    assert len(warnings) == 2
    assert "line 2" in warnings[0]
    assert "line 3" in warnings[1]


def test_parse_row_cap() -> None:
    rows, warnings = parse_llm_calls(b'{"model":"a/b"}\n' * 10_001)
    assert len(rows) == 10_000
    assert warnings == ["swarm/llm_calls.jsonl has more than 10000 rows; rows after 10000 skipped"]


def test_bad_rows_count_toward_cap_and_blanks_do_not() -> None:
    rows, warnings = parse_llm_calls(b"\n \n" + b"x\n" * 10_000 + b'{"model":"a/b"}')
    assert rows == []
    assert "more than 10000 rows" in " ".join(warnings)
    assert "9980 more bad lines skipped" in " ".join(warnings)


def test_parse_ignores_unknown_keys() -> None:
    rows, warnings = parse_llm_calls(b'{"model":"a/b","vendor_field":1,"owner_epoch":"999"}\n')
    assert not warnings
    assert rows[0].model == "a/b"
    assert "owner_epoch" not in rows[0].model_dump()


def test_hostile_llm_calls_file_never_crashes() -> None:
    data = b"\n".join(
        [
            b"",
            b"[1,2]",
            b'{"model":"a/b","usage":{"input_tokens":-5}}',
            b'{"model":7}',
            b'{"model":"a/b","cost_usd":Infinity}',
            b"{" * 100_000,
            b'{"model":"ok/x"}',
        ]
    )
    rows, warnings = parse_llm_calls(data)
    assert [r.model for r in rows] == ["ok/x"]
    assert len(warnings) == 5


@pytest.mark.parametrize(
    "fragment",
    [
        '"latency_ms":-1',
        '"latency_ms":NaN',
        '"schema_version":2',
        '"schema_version":true',
        '"request":{"secret":NaN}',
        '"response":{"secret":"\\ud800"}',
        '"usage":{"usd":Infinity}',
    ],
)
def test_invalid_row_fields_are_value_free(fragment: str) -> None:
    rows, warnings = parse_llm_calls(('{"model":"SENTINEL/x",' + fragment + "}").encode())
    assert rows == []
    assert len(warnings) == 1
    assert "SENTINEL" not in warnings[0]
    assert "secret" not in warnings[0]


def test_parse_bad_line_warnings_capped() -> None:
    rows, warnings = parse_llm_calls(b"x\n" * 1000)
    assert rows == []
    assert len(warnings) == 21
    assert warnings[-1] == "swarm/llm_calls.jsonl: 980 more bad lines skipped"


def test_parse_refuses_oversize_bytes() -> None:
    rows, warnings = parse_llm_calls(b"x" * (MAX_LLM_CALLS_BYTES + 1))
    assert rows == []
    assert len(warnings) == 1
    assert "over the 16 MiB limit" in warnings[0]


@pytest.mark.parametrize(
    ("keyed", "rows", "may_call", "source"),
    [
        (True, 3, True, "gateway"),
        (False, 2, True, "reported"),
        (False, 0, True, "unknown"),
        (False, 0, False, "none"),
    ],
)
def test_llm_calls_source(keyed: bool, rows: int, may_call: bool, source: str) -> None:
    assert llm_calls_source(keyed=keyed, reported_rows=rows, may_call=may_call) == source


def test_may_call_llm() -> None:
    assert may_call_llm(frozenset({"openai"}), ())
    assert may_call_llm(frozenset(), ["GEMINI_API_KEY"])
    assert not may_call_llm(frozenset(), ["MY_TOKEN"])


def test_hash_regular_file(tmp_path: Path) -> None:
    (tmp_path / "swarm").mkdir()
    data = b"x" * (3 * 1024 * 1024 + 1)
    (tmp_path / LLM_CALLS_PATH).write_bytes(data)
    assert hash_regular_file(tmp_path, LLM_CALLS_PATH) == (sha256_hex(data), len(data))
    (tmp_path / "swarm/link.jsonl").symlink_to(tmp_path / LLM_CALLS_PATH)
    with pytest.raises(ValueError, match="is not a regular file"):
        hash_regular_file(tmp_path, "swarm/link.jsonl")


@pytest.mark.parametrize("kind", ["fifo", "parent_symlink", "directory"])
def test_hash_refuses_special_paths(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "swarm"
    target.mkdir()
    if kind == "fifo":
        os.mkfifo(target / "llm_calls.jsonl")
    elif kind == "directory":
        (target / "llm_calls.jsonl").mkdir()
    else:
        (target / "llm_calls.jsonl").write_bytes(b"x")
        (tmp_path / "alias").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="is not a regular file"):
        hash_regular_file(
            tmp_path, "alias/llm_calls.jsonl" if kind == "parent_symlink" else LLM_CALLS_PATH
        )


def test_streamed_hash_reads_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._tooling import llm_calls

    data = b"x" * (3 * 1024 * 1024 + 1)
    (tmp_path / "report").write_bytes(data)
    read = os.read
    sizes: list[int] = []

    def capture(fd: int, size: int) -> bytes:
        sizes.append(size)
        return read(fd, size)

    monkeypatch.setattr(os, "read", capture)
    assert llm_calls.hash_regular_file(tmp_path, "report") == (sha256_hex(data), len(data))
    assert max(sizes) <= 1024 * 1024
    assert len(sizes) >= 4


def test_hard_hash_ceiling_refuses_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._tooling.llm_calls import MAX_LLM_CALLS_HASH_BYTES, collect_llm_calls

    (tmp_path / "swarm").mkdir()
    with (tmp_path / LLM_CALLS_PATH).open("wb") as stream:
        stream.truncate(MAX_LLM_CALLS_HASH_BYTES + 1)

    def forbidden(fd: int, size: int) -> bytes:
        pytest.fail("over-ceiling file must not be read")

    monkeypatch.setattr(os, "read", forbidden)
    assert collect_llm_calls(tmp_path) == (None, None, (("llm_calls_hash_limit", 1),))


@pytest.mark.parametrize("change", ["grow", "shrink", "overwrite"])
def test_changing_report_has_no_ref_or_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from swarm_arena._tooling import llm_calls

    (tmp_path / "swarm").mkdir()
    path = tmp_path / LLM_CALLS_PATH
    path.write_bytes(b"x" * (2 * 1024 * 1024))
    read = os.read
    total = 0
    changed = False
    ceiling = 3 * 1024 * 1024
    monkeypatch.setattr(llm_calls, "MAX_LLM_CALLS_HASH_BYTES", ceiling)

    def changing(fd: int, size: int) -> bytes:
        nonlocal total, changed
        chunk = read(fd, size)
        total += len(chunk)
        if not changed:
            changed = True
            with path.open("r+b") as stream:
                if change == "grow":
                    stream.truncate(ceiling + 1)
                elif change == "shrink":
                    stream.truncate(1)
                else:
                    stream.write(b"z")
        return chunk

    monkeypatch.setattr(os, "read", changing)
    assert llm_calls.collect_llm_calls(tmp_path) == (None, None, (("llm_calls_changed", 1),))
    assert total <= ceiling


@pytest.mark.parametrize("kind", ["terminal", "fenced", "unreadable", "raw_fenced"])
@pytest.mark.parametrize("nested", [False, True])
def test_collection_keeps_controls_and_hides_optional_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, nested: bool
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._errors import Fenced, _core_error, _SignalCategory
    from swarm_arena._tooling import llm_calls

    error: BaseException = (
        _core_error(RuntimeError, "terminal", _category=_SignalCategory.TERMINAL)
        if kind == "terminal"
        else _core_error(Fenced, "run ownership lost")
        if kind == "fenced"
        else Fenced("SECRET")
        if kind == "raw_fenced"
        else PermissionError("SECRET")
    )
    if nested:
        error = BaseExceptionGroup("SECRET", [error])

    def fail(*args: object, **kwargs: object) -> Any:
        raise error

    monkeypatch.setattr(llm_calls, "_inspect_path", fail)
    if kind in {"terminal", "fenced"}:
        with pytest.raises(RuntimeError if kind == "terminal" else Fenced) as caught:
            llm_calls.collect_llm_calls(tmp_path)
        assert_secret_absent(caught.value, "SECRET")
    else:
        assert llm_calls.collect_llm_calls(tmp_path) == (None, None, (("llm_calls_unreadable", 1),))


def test_diagnostics_survive_detach_without_public_fields() -> None:
    from swarm_arena._tooling.output import (
        collection_warnings,
        detach_tool_output,
        with_collection_diagnostics,
    )
    from swarm_arena.tool import RunStatus, ToolOutput

    output = with_collection_diagnostics(
        ToolOutput(status=RunStatus.OK, data={"keep": True}),
        [
            ("llm_calls_unreadable", 2),
            ("unknown_secret", 1),
            ("llm_calls_unreadable", 3),
        ],
    )
    detached = detach_tool_output(output.model_copy(deep=True))
    assert detached._collection_diagnostics == (("llm_calls_unreadable", 5),)
    assert detached.data == {"keep": True}
    assert collection_warnings(detached) == [
        "swarm/llm_calls.jsonl could not be read; not collected (5 occurrences)"
    ]
    assert "diagnostic" not in repr(detached)
    assert "diagnostic" not in detached.model_dump_json()
    for count in (True, 0, 10_001):
        with pytest.raises(ValueError, match="invalid collection diagnostics"):
            with_collection_diagnostics(output, [("llm_calls_unreadable", count)])
    with pytest.raises(ValueError, match="invalid collection diagnostics"):
        with_collection_diagnostics(output, [("llm_calls_unreadable", 1)] * 33)
