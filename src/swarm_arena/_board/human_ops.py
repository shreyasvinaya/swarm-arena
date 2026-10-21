"""Share validated human curation between command-line and viewer callers."""

from collections.abc import Sequence
from functools import partial
from typing import Literal

from swarm_arena._board.arena import ArenaBoard, _namespace_frontier
from swarm_arena._board.human_objects import HumanObject, write_human_objects
from swarm_arena._board.learnings import is_verified
from swarm_arena._board.pointers import _validate_namespace
from swarm_arena._errors import _core_error
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.ids import human_learning_id
from swarm_arena._model.results import Learning
from swarm_arena._store.base import open_store


def _prepare_human_operation(
    board: ArenaBoard,
    namespace: str,
    kind: Literal["add", "pin", "hide"],
    *,
    author: str,
    learning_id: str | None = None,
    title: str = "",
    text: str = "",
    goal_id: str = "",
    tags: Sequence[str] = (),
    evidence: Sequence[str] = (),
) -> HumanObject:
    """Validate against one published generation, including hidden target membership."""
    namespace = _validate_namespace(namespace)
    with board._pin_state() as state:
        if namespace not in state.namespaces:
            raise _core_error(ValueError, "namespace outside board")
        cache = state.readers.get(namespace)
        frontier = _namespace_frontier(state.frontier, namespace)
        if kind != "add":
            if cache is None or not cache.learnings_by_id(frontier, [learning_id or ""]):
                raise _core_error(ValueError, f"unknown learning {learning_id}")
            return HumanObject(kind=kind, learning_id=learning_id, author=f"human:{author}")
        candidates = [] if cache is None else cache.candidates(frontier, ids=list(evidence))
        found = {candidate.id: candidate for candidate, _ in candidates}
        for identifier in evidence:
            if identifier not in found:
                raise _core_error(ValueError, f"unknown candidate {identifier}")
        learning = Learning(
            id=human_learning_id(),
            namespace=namespace,
            run_id="",
            goal_id=goal_id,
            round=0,
            author=f"human:{author}",
            title=title,
            text=text,
            evidence=list(evidence),
            tags=list(tags),
            verified=is_verified(list(found.values())),
        )
        return HumanObject(kind="add", learning=learning, author=learning.author)


async def _apply_human_operation(
    board: ArenaBoard,
    namespace: str,
    kind: Literal["add", "pin", "hide"],
    *,
    author: str,
    learning_id: str | None = None,
    title: str = "",
    text: str = "",
    goal_id: str = "",
    tags: Sequence[str] = (),
    evidence: Sequence[str] = (),
) -> HumanObject:
    """Create one immutable curation object after successful generation-local validation.

    Callers publish ``board.frontier()`` on a worker before invoking this helper.
    A cancelled validation never advances to storage publication. This operation
    takes no run lease and can be reused by the viewer's async request boundary.
    """
    operation = await run_sync_daemon(
        partial(
            _prepare_human_operation,
            board,
            namespace,
            kind,
            author=author,
            learning_id=learning_id,
            title=title,
            text=text,
            goal_id=goal_id,
            tags=tags,
            evidence=evidence,
        )
    )
    arena = await run_sync_daemon(open_store, board.config.arena_root)
    await run_sync_daemon(
        partial(
            write_human_objects,
            arena,
            namespace,
            "curation",
            [operation],
            now=utc_now(),
        )
    )
    return operation
