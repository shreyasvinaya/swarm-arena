"""Bounded, hostile-input-safe cooperating tool call reports."""

import os
from pathlib import Path
from typing import Any

import pytest
from _fakes.cleanup import observe_task, settle_tasks

from swarm_arena._model.canonical import sha256_hex
from swarm_arena._tooling.llm_calls import (
    LLM_CALLS_PATH,
    MAX_LLM_CALLS_BYTES,
    LLMCallRow,
    hash_regular_file,
    llm_calls_source,
    may_call_llm,
    parse_llm_calls,
)


def _parsed(data: bytes) -> tuple[list[LLMCallRow], list[str]]:
    rows, warnings = parse_llm_calls(data)
    return list(rows), warnings


def test_parse_valid_rows() -> None:
    rows, warnings = _parsed(
        b'{"model":"openai/gpt-6","usage":{"input_tokens":10},"cost_usd":0.01}\n'
        b'{"schema_version":1,"model":"local-llm","litellm_call_id":"abc"}\n'
    )
    assert warnings == []
    assert [r.schema_version for r in rows] == [1, 1]
    assert rows[0].provider == "openai"
    assert rows[1].provider == "unknown"
    assert rows[1].litellm_call_id == "abc"


def test_parse_skips_bad_lines() -> None:
    rows, warnings = _parsed(b'{"model":"a/b"}\nnot json\n{"model":"a/b","cost_usd":NaN}\n')
    assert len(rows) == 1
    assert len(warnings) == 2
    assert "line 2" in warnings[0]
    assert "line 3" in warnings[1]


def test_parse_row_cap() -> None:
    rows, warnings = _parsed(b'{"model":"a/b"}\n' * 10_001)
    assert len(rows) == 10_000
    assert warnings == ["swarm/llm_calls.jsonl has more than 10000 rows; rows after 10000 skipped"]


def test_bad_rows_count_toward_cap_and_blanks_do_not() -> None:
    rows, warnings = _parsed(b"\n \n" + b"x\n" * 10_000 + b'{"model":"a/b"}')
    assert rows == []
    assert "more than 10000 rows" in " ".join(warnings)
    assert "9980 more bad lines skipped" in " ".join(warnings)


def test_parse_ignores_unknown_keys() -> None:
    rows, warnings = _parsed(b'{"model":"a/b","vendor_field":1,"owner_epoch":"999"}\n')
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
    rows, warnings = _parsed(data)
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
    rows, warnings = _parsed(('{"model":"SENTINEL/x",' + fragment + "}").encode())
    assert rows == []
    assert len(warnings) == 1
    assert "SENTINEL" not in warnings[0]
    assert "secret" not in warnings[0]


def test_parse_bad_line_warnings_capped() -> None:
    rows, warnings = _parsed(b"x\n" * 1000)
    assert rows == []
    assert len(warnings) == 21
    assert warnings[-1] == "swarm/llm_calls.jsonl: 980 more bad lines skipped"


def test_parse_refuses_oversize_bytes() -> None:
    rows, warnings = _parsed(b"x" * (MAX_LLM_CALLS_BYTES + 1))
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
    for pairs in (
        [("llm_calls_unreadable", True)],
        [("llm_calls_unreadable", 0)],
        [("llm_calls_unreadable", 10_001)],
        [("llm_calls_unreadable", 1)] * 33,
    ):
        checked = with_collection_diagnostics(output, pairs)
        assert checked.data == output.data
        assert collection_warnings(checked) == [
            "swarm/llm_calls.jsonl: invalid collection diagnostics"
        ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "m" * 257),
        ("provider", "p" * 65),
        ("litellm_call_id", "i" * 129),
        ("model", "a\nb"),
        ("cost_usd", True),
        ("latency_ms", True),
        ("usage", {"input_tokens": True}),
        ("usage", {"usd": True}),
        ("usage", {"output_tokens": 1.5}),
        ("cost_usd", "1"),
    ],
)
def test_report_fields_are_bounded_and_strict(field: str, value: Any) -> None:
    import json

    data = json.dumps({"model": "a/b", field: value}).encode()
    rows, warnings = parse_llm_calls(data)
    assert list(rows) == []
    assert warnings == ["swarm/llm_calls.jsonl line 1: invalid row; line skipped"]


def test_report_line_limit_skips_without_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    load = json.loads
    lengths: list[int] = []

    def bounded(data: Any, *args: Any, **kwargs: Any) -> Any:
        lengths.append(len(data))
        assert len(data) <= 64 * 1024
        return load(data, *args, **kwargs)

    monkeypatch.setattr(json, "loads", bounded)
    data = b'{"model":"a/b","request":"' + b"x" * (64 * 1024) + b'"}\n{"model":"ok"}'
    rows, warnings = parse_llm_calls(data)
    assert [row.model for row in rows] == ["ok"]
    assert lengths == [len(b'{"model":"ok"}')]
    assert len(warnings) == 1


async def test_valid_full_report_has_bounded_memory_and_loop_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import json
    import threading
    import tracemalloc
    from collections.abc import Iterator

    import anyio

    from swarm_arena._exec.threads import run_sync_daemon

    probe, _ = parse_llm_calls(b'{"model":"a/b"}')
    assert isinstance(probe, Iterator), "rows must not accumulate decoded payloads"
    prefix = b'{"model":"a/b","request":['
    line = prefix + b"{}," * 21000 + b"{}]}"
    line = line.ljust(65535, b" ") + b"\n"
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    parsing = asyncio.Event()
    release = threading.Event()
    order: list[str] = []
    workers: list[threading.Thread] = []
    loads = json.loads

    def gated_loads(data: Any, *args: Any, **kwargs: Any) -> Any:
        if not workers:
            assert threading.get_ident() != loop_thread, "JSON decoding must run off-loop"
            workers.append(threading.current_thread())
            order.append("parsing")
            loop.call_soon_threadsafe(parsing.set)
            assert release.wait(180), "event loop did not release the parser"
            order.append("released")
        return loads(data, *args, **kwargs)

    async def progress() -> None:
        with anyio.fail_after(60):
            await parsing.wait()
        order.append("loop progressed")
        release.set()

    monkeypatch.setattr(json, "loads", gated_loads)
    tracemalloc.start()
    observer = asyncio.create_task(progress())
    try:
        data = line * 256
        assert len(data) == 16 * 1024 * 1024
        rows, warnings = parse_llm_calls(data)
        count = 0
        while (row := await run_sync_daemon(next, rows, None)) is not None:
            assert len(row.request) == 21001
            count += 1
        peak = tracemalloc.get_traced_memory()[1]
        assert count == 256
        assert warnings == []
        assert peak < 64 * 1024 * 1024, peak
        await observe_task(observer)
        assert order == ["parsing", "loop progressed", "released"]
    finally:
        release.set()
        try:
            await settle_tasks(observer)
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    for worker in workers:
                        await asyncio.to_thread(worker.join, 60)
                    assert all(not worker.is_alive() for worker in workers)
            finally:
                tracemalloc.stop()


def test_growing_report_read_stops_at_pinned_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._tooling.llm_calls import collect_llm_calls

    (tmp_path / "swarm").mkdir()
    path = tmp_path / LLM_CALLS_PATH
    path.write_bytes(b"x" * 10)
    read = os.read
    total = 0

    def grow(fd: int, size: int) -> bytes:
        nonlocal total
        if total == 0:
            with path.open("ab") as stream:
                stream.write(b"x" * (2 * 1024 * 1024))
        data = read(fd, size)
        total += len(data)
        return data

    monkeypatch.setattr(os, "read", grow)
    assert collect_llm_calls(tmp_path) == (None, None, (("llm_calls_changed", 1),))
    assert total <= 10 + 4096


def test_bad_carrier_is_bounded_and_preserves_valid_output() -> None:
    from swarm_arena._tooling.output import collection_warnings, detach_tool_output
    from swarm_arena.tool import RunStatus, ToolOutput

    class LyingList(list[Any]):
        iterations = 0

        def __len__(self) -> int:
            return 0

        def __iter__(self) -> Any:
            for _ in range(100):
                self.iterations += 1
                yield ("llm_calls_changed", 1)

    carrier = LyingList()
    output = ToolOutput(status=RunStatus.OK, data={"keep": True})
    object.__setattr__(output, "_collection_diagnostics", carrier)
    detached = detach_tool_output(output)
    assert carrier.iterations <= 33
    assert detached.status == RunStatus.OK
    assert detached.data == {"keep": True}
    assert len(collection_warnings(detached)) == 1
    assert "invalid collection diagnostics" in collection_warnings(detached)[0]


def test_carrier_does_not_change_output_or_candidate_equality() -> None:
    from swarm_arena._model.results import Candidate
    from swarm_arena._tooling.output import with_collection_diagnostics
    from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest

    plain = ToolOutput(status=RunStatus.OK)
    decorated = with_collection_diagnostics(plain, [("llm_calls_changed", 1)])
    assert decorated == plain
    candidate = Candidate(
        id="c",
        run_id="r",
        goal_id="g",
        team="t",
        agent="a",
        round=1,
        output=decorated,
        request=ToolRequest(),
    )
    assert candidate == Candidate.model_validate_json(candidate.model_dump_json())


@pytest.mark.parametrize("case", ["head_oversize", "growth", "valid", "empty"])
def test_report_artifact_s3_fetch_is_bounded(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    pytest.importorskip("fsspec.asyn")

    import asyncio
    from contextlib import nullcontext
    from unittest.mock import Mock

    from swarm_arena._errors import StorageError
    from swarm_arena._store.artifacts import ArtifactStore
    from swarm_arena._store.s3 import S3Store
    from swarm_arena._tooling.llm_calls import _read_report_artifact
    from swarm_arena.tool import FileRef

    data = b"" if case == "empty" else b"{}"
    ref = FileRef(
        name="llm_calls.jsonl", path="llm_calls.jsonl", sha256=sha256_hex(data), size=len(data)
    )
    reads: list[int] = []
    gets: list[dict[str, Any]] = []

    class Body:
        def __init__(self) -> None:
            self.remaining = data + b"x" if case == "growth" else data

        async def __aenter__(self) -> "Body":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def read(self, size: int) -> bytes:
            reads.append(size)
            chunk, self.remaining = self.remaining[:size], self.remaining[size:]
            return chunk

    def call(method: str, **kwargs: Any) -> Any:
        if method == "head_object":
            return {"ContentLength": 17 * 1024 * 1024 if case == "head_oversize" else ref.size}
        assert method == "get_object"
        gets.append(kwargs)
        return {"Body": Body()}

    store = Mock(spec=S3Store)
    store._bucket = "bucket"
    store._key.side_effect = lambda key: key
    store._reading.side_effect = lambda method: nullcontext()
    store._call.side_effect = call

    def sync(loop: Any, fn: Any) -> bytes:
        return asyncio.run(fn())

    monkeypatch.setattr("fsspec.asyn.sync", sync)
    if case in {"head_oversize", "growth"}:
        with pytest.raises(StorageError, match="size mismatch"):
            _read_report_artifact(ArtifactStore(store), ref)
    else:
        assert _read_report_artifact(ArtifactStore(store), ref) == data
    if case == "head_oversize":
        assert gets == reads == []
    else:
        assert reads == ([ref.size + 1, 1] if case == "valid" else [ref.size + 1])
        assert gets[0].get("Range") == (f"bytes=0-{ref.size}" if ref.size else None)


@pytest.mark.parametrize("status", ["ok", "error"])
def test_report_reserves_gateway_observation_namespace(status: str) -> None:
    import json

    data = json.dumps(
        {"model": "SECRET/model", "status": status, "litellm_call_id": "gw:PRIVATE"}
    ).encode()
    rows, warnings = _parsed(data)
    assert rows == []
    assert warnings == ["swarm/llm_calls.jsonl line 1: invalid row; line skipped"]
