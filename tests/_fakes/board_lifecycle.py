"""Importable configured board with hostile and blocking lifecycle hooks."""

from collections.abc import Sequence
from threading import Event
from typing import ClassVar, cast

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._errors import Fenced, _core_error

SECRET = "board-lifecycle-private-sentinel"


class LifecycleBoard(ArenaBoard):
    entered: ClassVar[Event] = Event()
    release: ClassVar[Event] = Event()
    closed: ClassVar[Event] = Event()

    class Config(ArenaBoard.Config):  # ty: ignore[invalid-attribute-override]
        open_mode: str = "ok"
        close_mode: str = "ok"

    def _act(self, mode: str) -> None:
        if mode == "block":
            self.entered.set()
            assert self.release.wait(10), "test did not release board"
        elif mode != "ok":
            if mode == "core":
                error = _core_error(Fenced)
                error.args = (SECRET,)
            elif mode == "fake-fenced":
                error = Fenced(SECRET)
            else:
                error = ValueError(SECRET)
            error.add_note(SECRET)
            raise error from RuntimeError(SECRET)

    def open(self, run_id: str, run_root: str, namespaces: Sequence[str]) -> None:
        self._act(cast(LifecycleBoard.Config, self.config).open_mode)
        super().open(run_id, run_root, namespaces)

    def close(self) -> None:
        self.closed.set()
        super().close()
        self._act(cast(LifecycleBoard.Config, self.config).close_mode)
