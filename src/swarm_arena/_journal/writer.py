"""Serialize journal publication and lease maintenance in one asynchronous task."""

from collections.abc import Callable, Sequence
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Literal, TypeVar

import anyio
from anyio.abc import TaskGroup

from swarm_arena._errors import (
    AlreadyExists,
    ChainError,
    Fenced,
    NotFound,
    StorageError,
    WriteOutcomeUnknown,
    _core_error,
    _exception_type_name,
)
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.event import WRITE_THROUGH, Event, EventDraft
from swarm_arena._journal.lease import Head, Lease, read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.segment import encode_segment, segment_name
from swarm_arena._model.canonical import canonical_json, utc_now
from swarm_arena._outcomes import _Outcome
from swarm_arena._store.base import Store

_T = TypeVar("_T")


class JournalWriter:
    """
    Publish queued events and maintain ownership through one writer task.

    The inherited-chain read is abandonable under the owner's supervision.
    Publication and lease operations run in AnyIO workers. Durable appends share a
    group window and wait for the authority head to cover their sequence. A
    transient storage failure suspends renewals and retries only until the
    last confirmed expiry. Fatal failures wake all waiters with ``Fenced``.
    Use this object from its owning event loop and within its async context.

    Parameters
    ----------
    store : Store
        Run-scoped synchronous storage.
    lease : Lease
        Acquired lease, exclusively owned by this writer while entered.
    run_id : str
        Run and context identifier stamped on each event.
    next_seq : int
        Next sequence, exactly one greater than the inherited head or one.
    local_path : Path
        Local append-only canonical JSONL mirror, flushed before admission.
    sync_interval_s : float
        Maximum interval before ordinary queued events trigger a commit.
    group_window_s : float, default=0.2
        Delay from the first durable wake-up to group publication.

    Attributes
    ----------
    fenced : anyio.Event
        Set when ownership expires, changes, or the writer fails.
    epoch : int
        Ownership epoch of the supplied lease.

    Examples
    --------
    >>> writer = JournalWriter(store, lease, run_id="r", next_seq=1,
    ...     local_path=Path("journal.local.jsonl"), sync_interval_s=60)  # doctest: +SKIP
    >>> async with writer:  # doctest: +SKIP
    ...     event = await writer.append_durable(EventDraft(type="reserve"))
    """

    def __init__(
        self,
        store: Store,
        lease: Lease,
        *,
        run_id: str,
        next_seq: int,
        local_path: Path,
        sync_interval_s: float,
        group_window_s: float = 0.2,
    ) -> None:
        self._store = store
        self._lease = lease
        self._run_id = run_id
        self._next_seq = next_seq
        self._local_path = local_path
        self._sync_interval_s = sync_interval_s
        self._group_window_s = group_window_s
        self._head = lease.doc.head
        if type(next_seq) is not int or next_seq != (self._head.seq if self._head else 0) + 1:
            raise ValueError("next_seq must immediately follow the inherited head")
        self.epoch = lease.doc.epoch
        self.fenced = anyio.Event()
        self._committed: list[str] = []
        self._queue: list[Event] = []
        self._waiters: dict[int, list[anyio.Event]] = {}
        self._wake = anyio.Event()
        self._done = anyio.Event()
        self._group_at: float | None = None
        self._sync_at: float | None = None
        self._immediate = False
        self._closing = False
        self._entered = False
        self._failure: Exception | None = None
        self._tasks: TaskGroup | None = None

    async def __aenter__(self) -> "JournalWriter":
        """Load the inherited chain and start the sole publication task."""
        if self._entered or self._closing:
            raise _core_error(RuntimeError, "journal writer context cannot be reused")
        if self._head is not None:
            try:
                outcome = _Outcome()
                try:
                    chain = await self._retry(
                        partial(read_chain, self._store, self._head), abandonable=True
                    )
                except BaseException as error:
                    outcome.contain(error, StorageError, "journal initialization")
                self._committed = [name for name, _ in chain]
            except Exception as error:
                self._fail(error)
                self._raise_if_fenced()
        self._tasks = anyio.create_task_group()
        await self._tasks.__aenter__()
        self._entered = True
        self._tasks.start_soon(self._run)
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Shield the final publication and release, skipping writes after fencing."""
        was_fenced = self.fenced.is_set()
        self._closing = True
        self._wake.set()
        try:
            with anyio.CancelScope(shield=True):
                await self._done.wait()
        finally:
            assert self._tasks is not None
            await self._tasks.__aexit__(None, None, None)
            self._entered = False
        if not was_fenced and (not exc or exc[0] is None):
            self._raise_if_fenced()

    @property
    def head(self) -> Head | None:
        """
        Return the last head confirmed by this writer.

        Returns
        -------
        Head or None
            Published head, absent before the first committed event.

        Examples
        --------
        >>> writer.head.seq  # doctest: +SKIP
        1
        """
        return self._head

    @property
    def committed(self) -> Sequence[str]:
        """
        Return committed segment names including the inherited chain.

        Returns
        -------
        Sequence[str]
            Immutable snapshot ordered oldest first.

        Examples
        --------
        >>> list(writer.committed)  # doctest: +SKIP
        ['journal/e1-1.jsonl']
        """
        return tuple(self._committed)

    def append(self, draft: EventDraft) -> Event:
        """
        Validate, stamp, mirror, and queue an event without awaiting durability.

        Write-through event types wake the group timer automatically. Other
        events publish on the sync timer, an explicit commit, or context exit.
        Invalid payloads consume no sequence and write no local line.

        Parameters
        ----------
        draft : EventDraft
            Finite JSON event content; revalidated before queue admission.

        Returns
        -------
        Event
            Stamped event, independent of the queued payload snapshot.

        Raises
        ------
        ValueError
            If the draft is invalid or cannot be canonically encoded.
        Fenced
            If the writer failed or lost ownership.
        RuntimeError
            If the writer is outside its active context.

        Examples
        --------
        >>> draft = EventDraft(type="learning", payload={"n": None})
        >>> event = writer.append(draft)  # doctest: +SKIP
        """
        self._check_open()
        event = Event.model_validate(
            {
                **{name: getattr(draft, name) for name in EventDraft.model_fields},
                "epoch": self.epoch,
                "seq": self._next_seq,
                "ts": utc_now(),
                "run_id": self._run_id,
                "context_id": self._run_id,
            }
        )
        line = canonical_json(event.model_dump(mode="json")) + b"\n"
        queued = Event.model_validate_json(line)
        try:
            self._mirror(line)
        except Exception as error:
            self._fail(error)
            self._raise_if_fenced()
        self._queue.append(queued)
        self._next_seq += 1
        if self._sync_at is None:
            self._sync_at = anyio.current_time() + self._sync_interval_s
        if event.type in WRITE_THROUGH:
            self._request_group()
        self._wake.set()
        return event

    async def append_durable(self, draft: EventDraft) -> Event:
        """
        Append an event and wait until the authority head covers its sequence.

        Parameters
        ----------
        draft : EventDraft
            Event content to validate and publish.

        Returns
        -------
        Event
            Stamped event with confirmed durable publication.

        Raises
        ------
        Fenced
            If ownership expires, changes, or publication fails permanently.

        Examples
        --------
        >>> event = await writer.append_durable(EventDraft(type="reserve"))  # doctest: +SKIP
        """
        event = self.append(draft)
        self._request_group()
        await self._wait_for(event.seq)
        return event

    async def commit(self) -> Head | None:
        """
        Request immediate publication of all events queued at this call.

        Returns
        -------
        Head or None
            Confirmed head covering the requested events; empty commits do
            not create a segment or replace the owner document.

        Raises
        ------
        Fenced
            If the writer failed or lost ownership.

        Examples
        --------
        >>> head = await writer.commit()  # doctest: +SKIP
        """
        self._check_open()
        target = self._next_seq - 1
        if target > (self._head.seq if self._head else 0):
            self._immediate = True
            self._wake.set()
            await self._wait_for(target)
        return self._head

    def _mirror(self, line: bytes) -> None:
        """Flush the synchronous local mirror before exposing an admitted event."""
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        with self._local_path.open("ab") as stream:
            stream.write(line)
            stream.flush()

    def _check_open(self) -> None:
        """Reject failed and inactive writers before admission or waiting."""
        self._raise_if_fenced()
        if not self._entered or self._closing:
            raise _core_error(RuntimeError, "journal writer is not active")

    def _request_group(self) -> None:
        """Start a group window only on the first outstanding durable request."""
        if self._group_at is None:
            self._group_at = anyio.current_time() + self._group_window_s
        self._wake.set()

    async def _wait_for(self, seq: int) -> None:
        """Wait for publication or failure without leaving cancelled waiters behind."""
        self._raise_if_fenced()
        if self._head is not None and self._head.seq >= seq:
            return
        event = anyio.Event()
        waiters = self._waiters.setdefault(seq, [])
        waiters.append(event)
        try:
            await event.wait()
            self._raise_if_fenced()
        finally:
            waiters.remove(event)
            if not waiters:
                self._waiters.pop(seq, None)

    def _fail(self, error: Exception) -> None:
        """Latch the first failure and wake all consumers and the writer."""
        if not self.fenced.is_set():
            self._failure = error
            self.fenced.set()
        for waiters in self._waiters.values():
            for event in waiters:
                event.set()
        self._wake.set()

    def _raise_if_fenced(self) -> None:
        """Expose only exception types and explicitly trusted core storage causes."""
        if self.fenced.is_set():
            error = self._failure
            cause = error if isinstance(error, StorageError) else None
            name = _exception_type_name(error) if error is not None else "NoneType"
            raise _core_error(Fenced, f"journal writer failed: {name}") from cause

    def _remaining(self) -> float:
        """Compute lifetime using only the last confirmed owner document."""
        return (self._lease.doc.expires_at - utc_now()).total_seconds()

    async def _retry(self, operation: Callable[[], _T], *, abandonable: bool = False) -> _T:
        """Retry transient core I/O without renewal, bounded by confirmed expiry."""
        while True:
            self._raise_if_fenced()
            if self._remaining() <= 0:
                raise _core_error(Fenced, "journal lease expired")
            try:
                if abandonable:
                    await anyio.lowlevel.checkpoint_if_cancelled()
                    with anyio.move_on_after(self._remaining()) as lifetime:
                        result = await run_sync_daemon(operation)
                    if lifetime.cancel_called or self._remaining() <= 0:
                        raise _core_error(Fenced, "journal lease expired")
                    return result
                return await anyio.to_thread.run_sync(operation)
            except ChainError:
                raise
            except (OSError, StorageError):
                await anyio.sleep(max(0, min(1.0, self._remaining())))

    async def _update_owner(
        self, kind: Literal["advance", "renew", "release"], head: Head | None = None
    ) -> None:
        """Recover unknown updates by reading identity before adopting any ETag."""
        old = self._lease.doc
        expected = head if kind == "advance" else old.head
        uncertain = False

        def update() -> None:
            nonlocal uncertain
            if not uncertain:
                try:
                    if kind == "advance":
                        assert head is not None
                        self._lease.advance(head)
                    elif kind == "renew":
                        self._lease.renew()
                    else:
                        self._lease.release()
                    return
                except WriteOutcomeUnknown:
                    uncertain = True
            owner = read_owner(self._store)
            if owner is None:
                raise _core_error(Fenced, "journal owner disappeared")
            doc, etag = owner
            if (doc.epoch, doc.owner_id) != (old.epoch, old.owner_id):
                raise _core_error(Fenced, "journal owner changed")
            if doc.head != old.head and doc.head != expected:
                raise ChainError("unexpected journal head during recovery")
            self._lease._adopt(doc, etag)
            uncertain = False
            if doc.head == expected and doc.version > old.version:
                return
            raise StorageError("journal owner update not confirmed")

        await self._retry(update)

    async def _publish(self) -> None:
        """Freeze one batch and confirm its immutable bytes before advancing authority."""
        events = list(self._queue)
        name = segment_name(self.epoch, events[-1].seq)
        data = encode_segment(self._head.segment if self._head else None, events)
        digest = sha256(data).digest()
        new_head = Head(epoch=self.epoch, seq=events[-1].seq, segment=name)
        self._sync_at = self._group_at = None
        self._immediate = False

        def create() -> None:
            try:
                self._store.create(name, data)
            except (AlreadyExists, WriteOutcomeUnknown):
                try:
                    stored = self._store.read(name)
                except NotFound:
                    raise StorageError("journal segment creation not confirmed") from None
                if sha256(stored.data).digest() != digest:
                    raise ChainError("journal segment bytes differ") from None

        await self._retry(create)
        await self._update_owner("advance", new_head)
        self._head = new_head
        self._committed.append(name)
        self._queue = [event for event in self._queue if event.seq > new_head.seq]
        for seq, waiters in self._waiters.items():
            if seq <= new_head.seq:
                for event in waiters:
                    event.set()

    async def _run(self) -> None:
        """Own every commit, renewal, and release, including shielded shutdown."""
        with anyio.CancelScope(shield=True):
            try:
                while not self.fenced.is_set():
                    remaining = self._remaining()
                    if remaining <= 0:
                        raise _core_error(Fenced, "journal lease expired")
                    now = anyio.current_time()
                    deadlines = [
                        value for value in (self._group_at, self._sync_at) if value is not None
                    ]
                    if self._queue and (
                        self._closing or self._immediate or any(now >= value for value in deadlines)
                    ):
                        await self._publish()
                        continue
                    if self._closing:
                        await self._update_owner("release")
                        return
                    renew_in = remaining - self._lease.ttl_s * 2 / 3
                    if renew_in <= 0:
                        await self._update_owner("renew")
                        continue
                    delay = min([renew_in, *(max(0, value - now) for value in deadlines)])
                    with anyio.move_on_after(delay):
                        await self._wake.wait()
                    self._wake = anyio.Event()
            except Exception as error:
                self._fail(error)
            finally:
                self._done.set()
