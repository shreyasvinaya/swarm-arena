"""Configured review roots are built in a supervised worker."""

import threading
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Ref, RunConfig
from swarm_arena.board import ArenaBoard
from swarm_arena.scoring import HumanReview


async def test_import_contains_root_constructor_off_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "private-review-constructor-marker"
    threads: list[int] = []

    def fail(self: ArenaBoard, *args: Any, **kwargs: Any) -> None:
        threads.append(threading.get_ident())
        raise OSError(secret)

    config = RunConfig(board=Ref.of(ArenaBoard(arena_root=str(tmp_path))))
    monkeypatch.setattr(ArenaBoard, "__init__", fail)
    with pytest.raises(RuntimeError, match="board construction failed: OSError") as caught:
        await HumanReview("alice").import_csv("r", tmp_path / "unused.csv", config=config)
    assert threads and threading.get_ident() not in threads
    assert_secret_absent(caught.value, secret)
