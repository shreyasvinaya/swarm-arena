"""Shared bounded-read accounting for executor and tooling tests."""

import contextlib
import io
import os
from collections.abc import Generator
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


@contextlib.contextmanager
def count_reads(monkeypatch: pytest.MonkeyPatch, sink: list[int]) -> Generator[None]:
    """
    Count bytes read through descriptor and file-object reads.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patches restored when the context exits.
    sink : list[int]
        Destination for each read's byte count.

    Yields
    ------
    None
        All four read entry points are instrumented.

    Examples
    --------
    >>> with pytest.MonkeyPatch.context() as patch:
    ...     counts = []
    ...     with count_reads(patch, counts):
    ...         pass
    >>> counts
    []
    """
    with monkeypatch.context() as patch:
        real_read, real_pread, real_fdopen, real_open = os.read, os.pread, os.fdopen, io.open

        def read(fd: int, n: int) -> bytes:
            data = real_read(fd, n)
            sink.append(len(data))
            return data

        def pread(fd: int, n: int, offset: int) -> bytes:
            data = real_pread(fd, n, offset)
            sink.append(len(data))
            return data

        def fdopen(*args: Any, **kw: Any) -> Any:
            return _CountingStream(real_fdopen(*args, **kw), sink)

        def open_(*args: Any, **kw: Any) -> Any:
            return _CountingStream(real_open(*args, **kw), sink)

        patch.setattr(os, "read", read)
        patch.setattr(os, "pread", pread)
        patch.setattr(os, "fdopen", fdopen)
        patch.setattr(io, "open", open_)
        yield
