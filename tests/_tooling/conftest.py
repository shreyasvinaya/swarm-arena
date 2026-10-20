import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from _fakes.read_counts import count_reads


@pytest.fixture
def bytes_read(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Count reads through the shared descriptor and stream instrumentation."""
    sink: list[int] = []
    with count_reads(monkeypatch, sink):
        yield sink


@pytest.fixture
def sparse() -> Callable[[Path, int], None]:
    """Return a maker of sparse files: the size is set, but no data blocks are written."""

    def make(path: Path, size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        os.truncate(path, size)

    return make
