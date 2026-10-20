"""Maintain the run's lease and committed head through conditional storage writes."""

from datetime import datetime, timedelta
from typing import ClassVar

from swarm_arena._errors import Fenced, LeaseHeld, NotFound, PreconditionFailed
from swarm_arena._model.base import FrozenModel, StoredModel
from swarm_arena._model.canonical import UtcDatetime, canonical_json, utc_now
from swarm_arena._store.base import Store

OWNER_KEY = "control/owner.json"


class Head(FrozenModel):
    """
    Identify the newest committed journal segment and its last event.

    Parameters
    ----------
    epoch : int
        Ownership epoch that wrote the segment.
    seq : int
        Last event sequence number in the segment.
    segment : str
        Relative key of the committed segment.

    Examples
    --------
    >>> Head(epoch=1, seq=3, segment="journal/e1-3.jsonl").seq
    3
    """

    epoch: int
    seq: int
    segment: str


class OwnerDoc(StoredModel):
    """
    Store ownership and the journal's single commit point.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version, upgraded on loading.
    version : int
        Document revision, incremented on every conditional replacement.
    epoch : int
        Ownership epoch, incremented on takeover.
    owner_id : str
        Current owner's identifier.
    host : str
        Current owner's host identifier.
    expires_at : datetime
        Aware expiry time, normalized to UTC milliseconds.
    head : Head or None, default=None
        Newest committed segment, absent before the first commit.

    Examples
    --------
    >>> doc = OwnerDoc(version=1, epoch=1, owner_id="o1", host="h",
    ...                expires_at=utc_now())
    >>> OwnerDoc.model_validate_json(doc.model_dump_json()) == doc
    True
    """

    schema_kind: ClassVar[str] = "owner"
    version: int
    epoch: int
    owner_id: str
    host: str
    expires_at: UtcDatetime
    head: Head | None = None


def read_owner(store: Store) -> tuple[OwnerDoc, str] | None:
    """
    Read and validate the owner document together with its storage version.

    Parameters
    ----------
    store : Store
        Run-scoped synchronous store.

    Returns
    -------
    tuple[OwnerDoc, str] or None
        Validated authority and ETag, or None only when the key is absent.

    Raises
    ------
    ValueError
        If stored data is malformed or uses an unsupported schema version.
    StorageError
        If reading fails for a reason other than an absent key.

    Examples
    --------
    >>> owner = read_owner(store)  # doctest: +SKIP
    >>> owner[0].epoch  # doctest: +SKIP
    1
    """
    try:
        blob = store.read(OWNER_KEY)
    except NotFound:
        return None
    return OwnerDoc.model_validate_json(blob.data), blob.etag


class Lease:
    """
    Serialize lease changes against the last confirmed owner ETag.

    One writer owns this synchronous object and serializes its operations.
    Async consumers offload calls to an AnyIO worker. Failed writes leave the
    local document and ETag unchanged, including when the storage outcome is
    unknown; recovery may explicitly adopt a confirmed authority read.

    Parameters
    ----------
    store : Store
        Run-scoped synchronous store.
    doc : OwnerDoc
        Last confirmed owner document.
    etag : str
        Storage version of that document.
    ttl_s : float
        Lease lifetime in seconds after acquisition, renewal, or head advance.

    Attributes
    ----------
    doc : OwnerDoc
        Last confirmed owner document.
    etag : str
        Storage version used for the next conditional write.
    ttl_s : float
        Lease lifetime in seconds.

    Examples
    --------
    >>> lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)  # doctest: +SKIP
    >>> lease.renew()  # doctest: +SKIP
    """

    def __init__(self, store: Store, doc: OwnerDoc, etag: str, *, ttl_s: float) -> None:
        self._store = store
        self.doc = doc
        self.etag = etag
        self.ttl_s = ttl_s

    @classmethod
    def acquire_new(cls, store: Store, *, owner_id: str, host: str, ttl_s: float) -> "Lease":
        """
        Create the first ownership epoch only when no owner document exists.

        Parameters
        ----------
        store : Store
            Run-scoped synchronous store.
        owner_id : str
            New owner's identifier.
        host : str
            New owner's host identifier.
        ttl_s : float
            Lease lifetime in seconds.

        Returns
        -------
        Lease
            Confirmed lease at epoch 1 and document version 1.

        Raises
        ------
        AlreadyExists
            If a lease document already exists, even when expired.

        Examples
        --------
        >>> lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)  # doctest: +SKIP
        >>> lease.doc.epoch  # doctest: +SKIP
        1
        """
        doc = OwnerDoc(
            version=1,
            epoch=1,
            owner_id=owner_id,
            host=host,
            expires_at=utc_now() + timedelta(seconds=ttl_s),
        )
        etag = store.create(OWNER_KEY, canonical_json(doc.model_dump(mode="json")))
        return cls(store, doc, etag, ttl_s=ttl_s)

    @classmethod
    def take_over(
        cls,
        store: Store,
        *,
        owner_id: str,
        host: str,
        ttl_s: float,
        force: bool,
        expect_epoch: int | None = None,
    ) -> "Lease":
        """
        Take ownership with one authority read and one conditional replacement.

        Preserve the committed head while incrementing the epoch and document
        version. A failed conditional write is never retried here.

        Parameters
        ----------
        store : Store
            Run-scoped synchronous store.
        owner_id : str
            New owner's identifier.
        host : str
            New owner's host identifier.
        ttl_s : float
            Lease lifetime in seconds.
        force : bool
            Permit takeover before the current lease expires.
        expect_epoch : int or None, default=None
            Required epoch from a preceding authority read, when supplied.

        Returns
        -------
        Lease
            Confirmed ownership in the next epoch.

        Raises
        ------
        NotFound
            If there is no owner document.
        Fenced
            If the expected epoch differs or the conditional replacement fails.
        LeaseHeld
            If the current lease is live and force is false.

        Examples
        --------
        >>> lease = Lease.take_over(store, owner_id="o2", host="h", ttl_s=60,
        ...                         force=False, expect_epoch=1)  # doctest: +SKIP
        """
        owner = read_owner(store)
        if owner is None:
            raise NotFound("run owner not found")
        doc, etag = owner
        if expect_epoch is not None and doc.epoch != expect_epoch:
            raise Fenced(f"another owner took the run (epoch {doc.epoch})")
        if not force and not cls.expired(doc):
            raise LeaseHeld(f"run lease is held by {doc.owner_id} (epoch {doc.epoch})")
        proposed = OwnerDoc.model_validate(
            {
                **doc.model_dump(),
                "version": doc.version + 1,
                "epoch": doc.epoch + 1,
                "owner_id": owner_id,
                "host": host,
                "expires_at": utc_now() + timedelta(seconds=ttl_s),
            }
        )
        lease = cls(store, doc, etag, ttl_s=ttl_s)
        lease._replace(proposed)
        return lease

    @staticmethod
    def expired(doc: OwnerDoc) -> bool:
        """
        Check whether the document's expiry is at or before the current time.

        Parameters
        ----------
        doc : OwnerDoc
            Owner document to inspect without storage access.

        Returns
        -------
        bool
            Whether the lease has expired by this process's UTC clock.

        Examples
        --------
        >>> doc = OwnerDoc(version=1, epoch=1, owner_id="o1", host="h",
        ...                expires_at=utc_now() - timedelta(seconds=1))
        >>> Lease.expired(doc)
        True
        """
        return doc.expires_at <= utc_now()

    def advance(self, head: Head) -> None:
        """
        Publish a committed head and renew the lease in one conditional write.

        Parameters
        ----------
        head : Head
            New committed journal head.

        Raises
        ------
        Fenced
            If another write changed the owner document.

        Examples
        --------
        >>> lease.advance(Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))  # doctest: +SKIP
        """
        self._write(head=head, expires_at=utc_now() + timedelta(seconds=self.ttl_s))

    def renew(self) -> None:
        """
        Extend the lease lifetime while preserving the epoch and committed head.

        Raises
        ------
        Fenced
            If another write changed the owner document.

        Examples
        --------
        >>> lease.renew()  # doctest: +SKIP
        """
        self._write(head=self.doc.head, expires_at=utc_now() + timedelta(seconds=self.ttl_s))

    def release(self) -> None:
        """
        Expire the lease one second ago while preserving its epoch and head.

        Raises
        ------
        Fenced
            If another write changed the owner document.

        Examples
        --------
        >>> lease.release()  # doctest: +SKIP
        """
        self._write(head=self.doc.head, expires_at=utc_now() - timedelta(seconds=1))

    def _write(self, *, head: Head | None, expires_at: datetime) -> None:
        """Validate the next document before attempting its conditional write."""
        doc = OwnerDoc.model_validate(
            {
                **self.doc.model_dump(),
                "version": self.doc.version + 1,
                "head": head,
                "expires_at": expires_at,
            }
        )
        self._replace(doc)

    def _replace(self, doc: OwnerDoc) -> None:
        """Adopt proposed state only after storage confirms the replacement."""
        try:
            etag = self._store.replace(
                OWNER_KEY, canonical_json(doc.model_dump(mode="json")), if_match=self.etag
            )
        except PreconditionFailed:
            raise Fenced("another owner changed the lease") from None
        self._adopt(doc, etag)

    def _adopt(self, doc: OwnerDoc, etag: str) -> None:
        """Adopt state confirmed by a successful write or the writer's recovery read."""
        self.doc = doc
        self.etag = etag
