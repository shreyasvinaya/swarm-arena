"""Download committed chain extensions outside the arena cache lock."""

from collections.abc import Sequence

from swarm_arena._board.cache import ArenaCache, _validate_head
from swarm_arena._errors import ChainError, _core_error
from swarm_arena._journal.lease import Head
from swarm_arena._journal.reader import read_chain
from swarm_arena._model.board_types import Frontier
from swarm_arena._store.base import Store, open_store


def run_store(run_root: str, run_id: str) -> Store:
    """
    Open the recorded run location without consulting a newer arena pointer.

    Parameters
    ----------
    run_root : str
        Recorded local or S3 storage root.
    run_id : str
        Run identifier beneath that root.

    Returns
    -------
    Store
        Run-scoped store.

    Examples
    --------
    >>> store = run_store('/runs', 'r1')  # doctest: +SKIP
    """
    return open_store(run_root).child(run_id)


def ingest_run(cache: ArenaCache, store: Store, run_id: str, run_root: str, head: Head) -> int:
    """
    Fetch only uncached committed segments and index the chain extension.

    Parameters
    ----------
    cache : ArenaCache
        Namespace cache.
    store : Store
        Run-scoped store.
    run_id : str
        Expected run identifier.
    run_root : str
        Recorded root for this read.
    head : Head
        Committed authority selected by the caller.

    Returns
    -------
    int
        Number of new events stored by this call, including unindexed types.

    Raises
    ------
    ChainError
        If the complete head or its ancestry contradicts the cached chain.

    Examples
    --------
    >>> ingest_run(cache, store, 'r1', '/runs', head)  # doctest: +SKIP
    2
    """
    if cache._covered(run_id, head):
        return 0
    boundary = cache.newest_segment(run_id)
    segments = read_chain(store, head, stop_after=boundary)
    count = cache._ingest(run_id, run_root, segments)
    if not cache._covered(run_id, head):
        raise _core_error(ChainError, "head was not covered by downloaded chain")
    return count


def ensure_covered(cache: ArenaCache, mark: Frontier.Mark) -> None:
    """
    Validate a recorded mark and fetch exactly its missing committed prefix.

    Parameters
    ----------
    cache : ArenaCache
        Namespace cache, possibly ahead of the requested mark.
    mark : Frontier.Mark
        Complete recorded run, root, epoch, sequence, and segment identity.

    Raises
    ------
    ChainError
        If metadata or ancestry do not match, including on cache hits.

    Examples
    --------
    >>> ensure_covered(cache, mark)  # doctest: +SKIP
    """
    head = Head(epoch=mark.epoch, seq=mark.seq, segment=mark.segment)
    if not cache._covered(mark.run_id, head):
        ingest_run(cache, run_store(mark.run_root, mark.run_id), mark.run_id, mark.run_root, head)


def _validate_marks(marks: Sequence[Frontier.Mark]) -> list[Frontier.Mark]:
    """Deduplicate identical marks and refuse conflicting marks before any I/O."""
    unique: dict[str, Frontier.Mark] = {}
    for mark in marks:
        _validate_head(Head(epoch=mark.epoch, seq=mark.seq, segment=mark.segment))
        previous = unique.get(mark.run_id)
        if previous is not None and previous != mark:
            raise _core_error(ChainError, "conflicting frontier marks for run")
        unique[mark.run_id] = mark
    return [unique[run_id] for run_id in sorted(unique)]
