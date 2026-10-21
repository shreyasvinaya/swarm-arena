"""Verify committed view metadata and conditional publication."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident
from typing import Literal

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import JsonValue

from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.lease import Head, Lease
from swarm_arena._journal.views import ViewWriter, embed_source_head, read_source_head
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import canonical_json
from swarm_arena._store.base import Blob
from swarm_arena._store.local import LocalStore

H1 = Head(epoch=1, seq=10, segment="journal/e1-10.jsonl")
KEY = "goals/g/solution.json"
SECRET = "private-view-sentinel"
Format = Literal["json", "toml", "md"]


class ObservedStore(LocalStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.calls: list[tuple[str, int]] = []
        self.rivals: list[bytes] = []

    def read(self, key: str) -> Blob:
        if key == KEY:
            self.calls.append(("read", get_ident()))
        return super().read(key)

    def create(self, key: str, data: bytes) -> str:
        if key == KEY:
            self.calls.append(("create", get_ident()))
        return super().create(key, data)

    def replace(self, key: str, data: bytes, *, if_match: str) -> str:
        if key == KEY:
            self.calls.append(("replace", get_ident()))
            if self.rivals:
                current = super().read(key)
                super().replace(key, self.rivals.pop(0), if_match=current.etag)
        return super().replace(key, data, if_match=if_match)


@dataclass
class ViewEnv:
    store: ObservedStore
    writer: JournalWriter
    views: ViewWriter
    h1: Head
    h2: Head


@pytest.fixture
async def view_env(tmp_path: Path) -> AsyncIterator[ViewEnv]:
    store = ObservedStore(tmp_path / "run")
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    async with JournalWriter(
        store,
        lease,
        run_id="r",
        next_seq=1,
        local_path=tmp_path / "journal.local.jsonl",
        sync_interval_s=60,
        group_window_s=0,
    ) as writer:
        await writer.append_durable(EventDraft(type=EventType.RESERVE))
        h1 = writer.head
        await writer.append_durable(EventDraft(type=EventType.SETTLE))
        h2 = writer.head
        assert h1 is not None
        assert h2 is not None
        assert writer.committed == (h1.segment, h2.segment)
        yield ViewEnv(store, writer, ViewWriter(store, writer), h1, h2)


def test_embed_formats() -> None:
    assert json.loads(embed_source_head("json", b'{"a": 1}', H1))["source_head"] == {
        "epoch": 1,
        "seq": 10,
        "segment": "journal/e1-10.jsonl",
    }
    assert (
        embed_source_head("toml", b"a = 1\n", H1).splitlines()[0]
        == b"# swarm-arena source_head: e1/10/journal/e1-10.jsonl"
    )
    assert (
        embed_source_head("md", b"# x\n", H1).splitlines()[0]
        == b"<!-- swarm-arena source_head: e1/10/journal/e1-10.jsonl -->"
    )
    cases: list[tuple[Format, bytes]] = [("json", b"{}"), ("toml", b""), ("md", b"")]
    for fmt, body in cases:
        assert read_source_head(fmt, embed_source_head(fmt, body, H1)) == H1


def test_json_embedding_is_canonical_and_preserves_payload() -> None:
    body: dict[str, JsonValue] = {"z": [None, True, 1, 1.0, "λ"], "a": 1}
    expected = canonical_json({**body, "source_head": H1.model_dump()})
    assert embed_source_head("json", json.dumps(body).encode(), H1) == expected
    assert embed_source_head("json", expected, H1) == expected


@pytest.mark.parametrize(("fmt", "body"), [("toml", b'a = "\xce\xbb"\n'), ("md", b"# x\r\n")])
def test_text_embedding_preserves_body(fmt: Format, body: bytes) -> None:
    assert embed_source_head(fmt, body, H1).split(b"\n", 1)[1] == body


@pytest.mark.parametrize(
    "body",
    [
        b"[]",
        b"null",
        b"1",
        b'{"x": NaN}',
        b'{"x": Infinity}',
        b'{"x": 1e999}',
        b'{"x": "\\ud800"}',
        b'{"x": 1, "x": 2}',
        b'{"x": "\xff"}',
        b'{"' + SECRET.encode() + b'":',
    ],
)
def test_invalid_json_embedding_is_safe(body: bytes) -> None:
    with pytest.raises(ValueError, match="invalid view") as caught:
        embed_source_head("json", body, H1)
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize(
    ("fmt", "body"),
    [("toml", SECRET.encode() + b" = ["), ("toml", b"\xff"), ("md", b"\xff")],
)
def test_invalid_text_embedding_is_safe(fmt: Format, body: bytes) -> None:
    with pytest.raises(ValueError, match="invalid view") as caught:
        embed_source_head(fmt, body, H1)
    assert_secret_absent(caught.value, SECRET, sanitized=True)


def test_invalid_source_embedding_is_safe() -> None:
    source = Head(epoch=1, seq=10, segment=SECRET)
    with pytest.raises(ValueError, match="invalid view") as caught:
        embed_source_head("json", b"{}", source)
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize(
    ("fmt", "body"),
    [
        ("json", b"{}"),
        ("json", b"[]"),
        ("json", b"null"),
        ("json", b'{"source_head": null}'),
        ("json", b'{"source_head": "private-view-sentinel"}'),
        ("json", b'{"source_head":'),
        ("toml", b"a = 1"),
        ("toml", b"# swarm-arena source_head: e01/10/journal/e1-10.jsonl\n"),
        ("toml", b"# swarm-arena source_head: e1/10/journal/e1-11.jsonl\n"),
        ("toml", b"# swarm-arena source_head: e1/10/journal/e1-10.jsonl\nbad = ["),
        ("md", b"<!-- swarm-arena source_head: e1/10/journal/e1-10.jsonl --> trailing"),
        ("md", b"<!-- swarm-arena source_head: e1/10/journal/e1-10.jsonl"),
        ("md", b"\n<!-- swarm-arena source_head: e1/10/journal/e1-10.jsonl -->"),
    ],
)
def test_read_malformed_or_missing_head_returns_none(fmt: Format, body: bytes) -> None:
    assert read_source_head(fmt, body) is None


@pytest.mark.parametrize("fmt", ["json", "toml", "md"])
def test_read_rejects_invalid_utf8_anywhere(fmt: Format) -> None:
    body = b"{}" if fmt == "json" else b""
    encoded = embed_source_head(fmt, body, H1)
    assert read_source_head(fmt, encoded + b"\xff" + SECRET.encode()) is None


@pytest.mark.parametrize(
    "update",
    [
        {"epoch": True},
        {"seq": "10"},
        {"seq": 10.0},
        {"epoch": 0},
        {"seq": 11},
        {"segment": "journal/e01-10.jsonl"},
        {"segment": SECRET},
        {"extra": SECRET},
    ],
)
def test_read_rejects_invalid_head_fields(update: dict[str, object]) -> None:
    head = {**H1.model_dump(), **update}
    assert read_source_head("json", json.dumps({"source_head": head}).encode()) is None


def test_read_rejects_invalid_json_body_with_valid_head() -> None:
    header = json.dumps(H1.model_dump()).encode()
    for suffix in [b'"x": NaN', b'"x": "\\ud800"', b'"x": 1, "x": 2']:
        assert read_source_head("json", b'{"source_head":' + header + b"," + suffix + b"}") is None


async def test_same_bytes_twice_is_noop(view_env: ViewEnv) -> None:
    await view_env.views.write("goals/g/solution.json", b"{}", source=view_env.h1)
    await view_env.views.write("goals/g/solution.json", b"{}", source=view_env.h1)
    assert [op for op, _ in view_env.store.calls] == ["create", "create", "read"]


async def test_later_head_replaces_earlier(view_env: ViewEnv) -> None:
    await view_env.views.write("goals/g/solution.json", b'{"v": 1}', source=view_env.h1)
    await view_env.views.write("goals/g/solution.json", b'{"v": 2}', source=view_env.h2)
    assert json.loads(view_env.store.read("goals/g/solution.json").data)["v"] == 2


async def test_earlier_head_never_replaces_later(view_env: ViewEnv) -> None:
    await view_env.views.write("goals/g/solution.json", b'{"v": 2}', source=view_env.h2)
    await view_env.views.write("goals/g/solution.json", b'{"v": 1}', source=view_env.h1)
    assert json.loads(view_env.store.read("goals/g/solution.json").data)["v"] == 2
    assert not any(op == "replace" for op, _ in view_env.store.calls)


async def test_uncommitted_source_head_refused(view_env: ViewEnv) -> None:
    with pytest.raises(RuntimeError, match="not committed"):
        await view_env.views.write(
            "goals/g/solution.json",
            b"{}",
            source=Head(epoch=1, seq=99, segment="journal/e1-99.jsonl"),
        )
    assert view_env.store.calls == []


@pytest.mark.parametrize("change", [{"epoch": 2}, {"seq": 99}, {"segment": SECRET}])
async def test_mismatched_source_refused(view_env: ViewEnv, change: dict[str, object]) -> None:
    source = view_env.h1.model_copy(update=change)
    with pytest.raises(RuntimeError, match="not committed") as caught:
        await view_env.views.write(KEY, b"{}", source=source)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert view_env.store.calls == []


@pytest.mark.parametrize("suffix", ["JSON", "Toml", "MD", "txt", "", "json.bak"])
async def test_format_requires_exact_lowercase_suffix(view_env: ViewEnv, suffix: str) -> None:
    key = f"goals/g/{SECRET}.{suffix}"
    with pytest.raises(ValueError, match="suffix") as caught:
        await view_env.views.write(key, SECRET.encode(), source=view_env.h1)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert not view_env.store.exists(key)


@pytest.mark.parametrize(("fmt", "body"), [("toml", b"a = 1\n"), ("md", b"# x\n")])
async def test_write_text_formats(view_env: ViewEnv, fmt: Format, body: bytes) -> None:
    key = f"goals/g/view.{fmt}"
    await view_env.views.write(key, body, source=view_env.h1)
    await view_env.views.write(key, body + b"\n", source=view_env.h2)
    assert read_source_head(fmt, view_env.store.read(key).data) == view_env.h2


@pytest.mark.parametrize("existing", ["bad", "no_head", "unknown", "mismatch", "same"])
async def test_unproven_or_same_head_view_untouched(view_env: ViewEnv, existing: str) -> None:
    head = view_env.h1.model_dump()
    if existing == "unknown":
        head = Head(epoch=2, seq=1, segment="journal/e2-1.jsonl").model_dump()
    elif existing == "mismatch":
        head["seq"] = 0
    elif existing == "same":
        head = view_env.h2.model_dump()
    body = json.dumps({"source_head": head, "v": 1}).encode()
    if existing == "bad":
        body = b"\xff" + SECRET.encode()
    elif existing == "no_head":
        body = b"{}"
    view_env.store.create(KEY, body)
    await view_env.views.write(KEY, b'{"v": 2}', source=view_env.h2)
    assert view_env.store.read(KEY).data == body
    assert not any(op == "replace" for op, _ in view_env.store.calls)


async def test_store_io_runs_off_loop(view_env: ViewEnv) -> None:
    loop_thread = get_ident()
    await view_env.views.write(KEY, b"{}", source=view_env.h1)
    await view_env.views.write(KEY, b"{}", source=view_env.h2)
    assert {op for op, _ in view_env.store.calls} == {"create", "read", "replace"}
    assert all(thread != loop_thread for _, thread in view_env.store.calls)


@pytest.mark.parametrize("rival", ["earlier", "same", "later", "unknown", "bad"])
async def test_cas_failure_rereads_and_rechecks(view_env: ViewEnv, rival: str) -> None:
    await view_env.views.write(KEY, b'{"v": 1}', source=view_env.h1)
    heads = {"earlier": view_env.h1, "same": view_env.h2, "unknown": H1}
    await view_env.writer.append_durable(EventDraft(type=EventType.LEARNING))
    h3 = view_env.writer.head
    assert h3 is not None
    heads["later"] = h3
    competing = b"{}" if rival == "bad" else embed_source_head("json", b'{"v": 3}', heads[rival])
    view_env.store.rivals = [competing]
    view_env.store.calls.clear()
    await view_env.views.write(KEY, b'{"v": 2}', source=view_env.h2)
    assert [op for op, _ in view_env.store.calls].count("read") == 2
    assert [op for op, _ in view_env.store.calls].count("replace") == (
        2 if rival == "earlier" else 1
    )
    expected = (
        embed_source_head("json", b'{"v": 2}', view_env.h2) if rival == "earlier" else competing
    )
    assert view_env.store.read(KEY).data == expected


async def test_second_cas_failure_stops_retrying(view_env: ViewEnv) -> None:
    await view_env.views.write(KEY, b'{"v": 1}', source=view_env.h1)
    rivals = [embed_source_head("json", json.dumps({"v": n}).encode(), view_env.h1) for n in (3, 4)]
    last_rival = rivals[-1]
    view_env.store.rivals = rivals
    view_env.store.calls.clear()
    await view_env.views.write(KEY, b'{"v": 2}', source=view_env.h2)
    assert [op for op, _ in view_env.store.calls] == [
        "create",
        "read",
        "replace",
        "read",
        "replace",
    ]
    assert view_env.store.read(KEY).data == last_rival
