"""Read only the journal chain selected by a single committed head."""

from collections.abc import Callable
from functools import partial
from typing import Literal

from swarm_arena._errors import ChainError, NotFound, _core_error
from swarm_arena._journal.event import Event
from swarm_arena._journal.lease import Head, OwnerDoc, read_owner
from swarm_arena._journal.segment import decode_segment, parse_segment_name
from swarm_arena._outcomes import _Outcome
from swarm_arena._store.base import Store, _read_blob
from swarm_arena._store.io import _contain_storage, _validated_io


def _parse_name(name: str, location: Literal["head", "link", "stop_after"]) -> tuple[int, int]:
    """Reject untrusted names before storage access or diagnostic interpolation."""
    try:
        return parse_segment_name(name)
    except ValueError:
        raise _core_error(ChainError, f"invalid {location} segment name") from None


def read_chain(
    store: Store,
    head: Head,
    *,
    stop_after: str | None = None,
    _read: Callable[[str], bytes] | None = None,
) -> list[tuple[str, list[Event]]]:
    """
    Follow a committed head's links and return validated segments oldest first.

    Segment names, event epochs, and consecutive sequence numbers must agree.
    Each link is checked before reading its target, including the exclusive
    boundary. Storage is never listed. Async callers offload this synchronous
    operation with ``_read_chain`` when holding a lease.

    Parameters
    ----------
    store : Store
        Run-scoped store containing immutable journal segments.
    head : Head
        Committed segment and its epoch and last sequence number.
    stop_after : str or None, default=None
        Already known segment to exclude, stopping before reading it. If it is
        not encountered, traversal ends at the first segment's null link.
    _read : callable or None, default=None
        Internal backend-byte boundary; validation remains owned by this reader.

    Returns
    -------
    list[tuple[str, list[Event]]]
        Segment keys and their events, ordered from oldest to newest. An empty
        list is returned when the head is the exclusive boundary.

    Raises
    ------
    ChainError
        If a segment is missing, malformed, or inconsistent with its name,
        head, or predecessor.
    StorageError
        If another storage failure occurs; callers may retry the read.

    Examples
    --------
    >>> head = Head(epoch=1, seq=2, segment="journal/e1-2.jsonl")
    >>> chain = read_chain(store, head)  # doctest: +SKIP
    >>> [event.seq for _, events in chain for event in events]  # doctest: +SKIP
    [1, 2]
    """
    epoch, last_seq = _parse_name(head.segment, "head")
    if (head.epoch, head.seq) != (epoch, last_seq):
        raise _core_error(ChainError, "head epoch or seq does not match its segment name")
    if stop_after is not None:
        _parse_name(stop_after, "stop_after")

    chain: list[tuple[str, list[Event]]] = []
    name = head.segment
    while name != stop_after:
        try:
            data = store.read(name).data if _read is None else _read(name)
        except NotFound:
            raise _core_error(ChainError, f"missing segment '{name}'") from None
        try:
            link, events = decode_segment(data)
        except ValueError:
            raise _core_error(ChainError, f"invalid segment '{name}'") from None
        label = "head segment" if name == head.segment else "segment"
        if events[-1].seq != last_seq:
            raise _core_error(
                ChainError,
                f"{label} '{name}' ends at seq {events[-1].seq}, its name says {last_seq}",
            )
        if any(event.epoch != epoch for event in events):
            raise _core_error(ChainError, f"{label} '{name}' has an event epoch mismatch")

        chain.append((name, events))
        first_seq = events[0].seq
        if link is None:
            if first_seq != 1:
                raise _core_error(ChainError, "seq gap: first segment must start at seq 1")
            break
        epoch, last_seq = _parse_name(link, "link")
        if last_seq != first_seq - 1:
            raise _core_error(
                ChainError,
                f"seq gap: link '{link}' ends at seq {last_seq}, expected {first_seq - 1}",
            )
        name = link

    chain.reverse()
    return chain


def read_committed(store: Store) -> tuple[OwnerDoc, list[Event]] | None:
    """
    Read authority once and return exactly the events committed by that head.

    A concurrent head advance does not affect this read. The owner's current
    epoch may be newer than its inherited head following takeover. Async callers
    offload this synchronous operation with ``anyio.to_thread.run_sync``.

    Parameters
    ----------
    store : Store
        Run-scoped store holding the owner document and journal.

    Returns
    -------
    tuple[OwnerDoc, list[Event]] or None
        Authority snapshot and events oldest first, or None when no owner
        exists. An owner with no head has an empty event list.

    Raises
    ------
    ChainError
        If the owner document or its committed chain is invalid.
    StorageError
        If another storage failure prevents reading.

    Examples
    --------
    >>> committed = read_committed(store)  # doctest: +SKIP
    >>> owner, events = committed  # doctest: +SKIP
    >>> [event.seq for event in events]  # doctest: +SKIP
    [1, 2]
    """
    try:
        owner = read_owner(store)
    except ValueError:
        raise _core_error(ChainError, "invalid owner document") from None
    if owner is None:
        return None
    doc, _ = owner
    if doc.head is None:
        return doc, []
    return doc, [event for _, events in read_chain(store, doc.head) for event in events]


def _read_segment(store: Store, key: str) -> bytes:
    """Contain and detach one backend read, leaving chain validation outside."""
    try:
        return _read_blob(store, key).data
    except NotFound as error:
        _Outcome(capture_caller=False).contain(error, NotFound, "journal read")
    except BaseException as error:
        _contain_storage(_Outcome(capture_caller=False), error, "journal read")


async def _read_chain(store: Store, head: Head) -> list[tuple[str, list[Event]]]:
    """Read under caller supervision without wrapping trusted chain diagnostics."""
    return await _validated_io(
        partial(read_chain, store, head, _read=partial(_read_segment, store))
    )
