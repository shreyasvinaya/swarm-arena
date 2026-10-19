import io
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


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


@pytest.fixture
def bytes_read(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the size of every read through os.read, os.pread, os.fdopen and io.open."""
    sink: list[int] = []
    real_read, real_pread, real_fdopen, real_open = os.read, os.pread, os.fdopen, io.open

    def read(fd: int, n: int) -> bytes:
        data = real_read(fd, n)
        sink.append(len(data))
        return data

    def pread(fd: int, n: int, offset: int) -> bytes:
        data = real_pread(fd, n, offset)
        sink.append(len(data))
        return data

    monkeypatch.setattr(os, "read", read)
    monkeypatch.setattr(os, "pread", pread)
    monkeypatch.setattr(os, "fdopen", lambda *a, **kw: _CountingStream(real_fdopen(*a, **kw), sink))
    monkeypatch.setattr(io, "open", lambda *a, **kw: _CountingStream(real_open(*a, **kw), sink))
    return sink


@pytest.fixture
def sparse() -> Callable[[Path, int], None]:
    """Return a maker of sparse files: the size is set, but no data blocks are written."""

    def make(path: Path, size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        os.truncate(path, size)

    return make
