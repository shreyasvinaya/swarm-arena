"""Exercise bounded consumers with real short-reading async S3 bodies."""

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import StorageError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.s3 import S3Store
from swarm_arena._tooling.llm_calls import _read_report_artifact
from swarm_arena.tool import FileRef

SECRET = "private-response-value"


@pytest.mark.parametrize("consumer", ["get_bytes", "fetch", "report"])
@pytest.mark.parametrize(
    "case", ["valid", "empty", "truncated", "transport_truncated", "growth", "corrupt", "head"]
)
def test_s3_consumers_read_short_chunks_under_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, consumer: str, case: str
) -> None:
    pytest.importorskip("fsspec.asyn")
    response_module = pytest.importorskip("aiobotocore.response")
    protocol_module = pytest.importorskip("aiohttp.base_protocol")
    streams_module = pytest.importorskip("aiohttp.streams")
    expected = b"" if case == "empty" else b'{"model":"openai/review","cost_usd":0.000001}\n'
    cap = len(expected)
    ref = FileRef(
        name="llm_calls.jsonl",
        path="llm_calls.jsonl",
        sha256=sha256_hex(expected),
        size=len(expected),
    )
    payload = expected
    if case in {"truncated", "transport_truncated"}:
        payload = expected[:-1]
    elif case == "growth":
        payload += SECRET.encode() * 10
    elif case == "corrupt":
        payload = (SECRET.encode() * 10)[: len(expected)]
    monkeypatch.setattr("swarm_arena._store.artifacts._MAX_ARTIFACT_BYTES", cap)
    calls: list[tuple[str, dict[str, Any]]] = []
    reads: list[int] = []
    chunks: list[int] = []

    class ObservedBody(response_module.AioStreamingBody):
        async def read(self, amt: int) -> bytes:
            assert amt > 0, "bounded consumers must never read all or read zero bytes"
            reads.append(amt)
            chunk = await super().read(amt)
            chunks.append(len(chunk))
            return chunk

    class Response:
        def __init__(self) -> None:
            self.closed = False
            self.body: Any = None

        async def __aenter__(self) -> Any:
            loop = asyncio.get_running_loop()
            protocol = protocol_module.BaseProtocol(loop)
            protocol.connection_made(Mock(spec=asyncio.Transport))
            reader = streams_module.StreamReader(protocol, limit=65536, loop=loop)
            raw = SimpleNamespace(content=reader, close=self.close, url=SECRET)
            length = len(expected) if case == "transport_truncated" else len(payload)
            self.body = ObservedBody(raw, length)
            position = 0

            def deliver() -> None:
                nonlocal position
                if self.closed:
                    return
                if position < len(payload):
                    reader.feed_data(payload[position : position + 5])
                    position += 5
                    loop.call_soon(deliver)
                else:
                    reader.feed_eof()

            deliver()
            return await self.body.__aenter__()

        async def __aexit__(self, *args: Any) -> None:
            await self.body.__aexit__(*args)

        def close(self) -> None:
            self.closed = True

    response = Response()
    store = S3Store("s3://offline-bucket")

    def call(method: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((method, kwargs))
        if method == "head_object":
            return {"ContentLength": len(expected) + (case == "head")}
        assert method == "get_object"
        return {"Body": response}

    def sync(loop: Any, fn: Callable[[], Coroutine[Any, Any, bytes]]) -> bytes:
        return asyncio.run(fn())

    monkeypatch.setattr(store, "_call", call)
    monkeypatch.setattr(store, "_filesystem", lambda: SimpleNamespace(loop=None))
    monkeypatch.setattr("fsspec.asyn.sync", sync)
    artifacts = ArtifactStore(store)
    destination = tmp_path / "artifact"

    def read() -> bytes:
        if consumer == "get_bytes":
            return artifacts.get_bytes(ref.sha256)
        if consumer == "report":
            return _read_report_artifact(artifacts, ref)
        artifacts.fetch(ref.sha256, destination)
        return destination.read_bytes()

    if case in {"valid", "empty"}:
        assert read() == expected
    else:
        with pytest.raises(StorageError) as caught:
            read()
        assert_secret_absent(caught.value, SECRET)
        assert not destination.exists()
        if case == "corrupt":
            assert "hash" in str(caught.value)
        elif case == "truncated":
            assert ("size" if consumer == "report" else "hash") in str(caught.value)
    if case == "head":
        assert [method for method, _ in calls] == ["head_object"]
        assert reads == []
        return
    assert [method for method, _ in calls] == ["head_object", "get_object"]
    limit = ref.size if consumer == "report" else cap
    assert calls[1][1].get("Range") == (f"bytes=0-{limit}" if expected else None)
    assert response.closed
    assert response.body.tell() == min(len(payload), limit + 1)
    assert all(0 < size <= limit + 1 for size in reads)
    assert sum(chunks) <= limit + 1
    if case == "growth":
        assert response.body.tell() == limit + 1
    elif case == "valid":
        assert len([size for size in chunks if size]) > 2
        assert chunks[-1] == 0  # A short read is not EOF; consume through the empty read.
