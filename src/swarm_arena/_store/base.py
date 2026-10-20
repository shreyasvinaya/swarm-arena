"""Define conditional storage operations and select the local backend."""

import re
from abc import ABC, abstractmethod
from typing import NamedTuple

from swarm_arena._errors import ConditionalWritesUnsupported


class Blob(NamedTuple):
    """Pair stored bytes with their version identifier.

    Parameters
    ----------
    data : bytes
        Stored contents.
    etag : str
        Version identifier; local stores use the SHA-256 digest.

    Examples
    --------
    >>> Blob(b"x", "version").data
    b'x'
    """

    data: bytes
    etag: str


def _check_key(key: str) -> str:
    """Validate raw key segments before pathlib can normalize them."""
    if not key or key.startswith("/") or "\x00" in key or {".", ".."} & set(key.split("/")):
        raise ValueError("bad store key: require a non-empty relative path without dot segments")
    return key


class Store(ABC):
    """Define synchronous storage with conditional writes only.

    Async callers run these operations through ``anyio.to_thread.run_sync``.
    Implementations expose their root through ``url``. No unconditional put
    operation is provided.

    Examples
    --------
    >>> store = open_store("./artifacts")  # doctest: +SKIP
    >>> store.create("example", b"hello")  # doctest: +SKIP
    """

    @property
    @abstractmethod
    def url(self) -> str:
        """Return the store's root location.

        Returns
        -------
        str
            Local absolute path or backend URL.

        Examples
        --------
        >>> open_store("/tmp/artifacts").url  # doctest: +SKIP
        '/tmp/artifacts'
        """

    @abstractmethod
    def read(self, key: str) -> Blob:
        """Read a key's contents and version.

        Parameters
        ----------
        key : str
            Relative POSIX storage key.

        Returns
        -------
        Blob
            Contents and version identifier.

        Raises
        ------
        NotFound
            If the key does not exist.

        Examples
        --------
        >>> store.read("example").data  # doctest: +SKIP
        b'hello'
        """

    @abstractmethod
    def create(self, key: str, data: bytes) -> str:
        """Create a key only if it does not already exist.

        Parameters
        ----------
        key : str
            Relative POSIX storage key.
        data : bytes
            Complete contents to publish.

        Returns
        -------
        str
            Version identifier of the created object.

        Raises
        ------
        AlreadyExists
            If another object already occupies the key.

        Examples
        --------
        >>> etag = store.create("example", b"hello")  # doctest: +SKIP
        """

    @abstractmethod
    def replace(self, key: str, data: bytes, *, if_match: str) -> str:
        """Replace an existing key only when its version matches.

        Parameters
        ----------
        key : str
            Relative POSIX storage key.
        data : bytes
            Replacement contents.
        if_match : str
            Required current version identifier.

        Returns
        -------
        str
            Version identifier of the replacement.

        Raises
        ------
        NotFound
            If the key does not exist.
        PreconditionFailed
            If the current version differs from ``if_match``.
        StorageError
            If the conditional replacement fails.

        Examples
        --------
        >>> store.replace("example", b"updated", if_match=etag)  # doctest: +SKIP
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether a key exists.

        Parameters
        ----------
        key : str
            Relative POSIX storage key.

        Returns
        -------
        bool
            Whether the key exists.

        Examples
        --------
        >>> store.exists("example")  # doctest: +SKIP
        True
        """

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """List sorted regular-file keys beginning with a prefix.

        Parameters
        ----------
        prefix : str
            Literal key prefix; an empty string selects all keys.

        Returns
        -------
        list[str]
            Matching keys, excluding temporary and lock files.

        Notes
        -----
        Use listing for runs, reviews, and curation, never to discover the
        committed journal chain.

        Examples
        --------
        >>> store.list("runs/")  # doctest: +SKIP
        ['runs/a.json']
        """

    def child(self, prefix: str) -> "Store":
        """Open a store rooted below this store.

        Parameters
        ----------
        prefix : str
            Non-empty relative POSIX key for the child root.

        Returns
        -------
        Store
            Store whose keys are relative to the child root.

        Examples
        --------
        >>> runs = store.child("runs")  # doctest: +SKIP
        """
        return open_store(f"{self.url.rstrip('/')}/{_check_key(prefix)}")

    def probe_conditional_writes(self) -> None:
        """Probe support for conditional writes, doing nothing for local stores.

        Remote backends override this method to verify their guarantees.

        Examples
        --------
        >>> store.probe_conditional_writes()  # doctest: +SKIP
        """
        return None


def open_store(url: str) -> Store:
    """Open local storage or reject an unsupported URL scheme.

    Parameters
    ----------
    url : str
        Local path, optionally starting with a tilde. URL schemes, including
        ``file://``, are refused. S3 support requires the ``swarm-arena[s3]``
        extra planned for M2.

    Returns
    -------
    Store
        Local store with an expanded, resolved root.

    Raises
    ------
    ConditionalWritesUnsupported
        If the location includes a URL scheme.

    Examples
    --------
    >>> store = open_store("~/artifacts")  # doctest: +SKIP
    """
    from pathlib import Path

    from swarm_arena._store.local import LocalStore

    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*)://", url)
    if match is not None:
        # Only fixed, recognized scheme names may appear in diagnostics.
        scheme = match.group(1).lower()
        if scheme == "s3":
            raise ConditionalWritesUnsupported("s3:// needs the swarm-arena[s3] extra (M2)")
        label = f"{scheme}://" if scheme in {"s3", "gs", "file"} else "URL"
        raise ConditionalWritesUnsupported(
            f"{label} has no conditional writes; swarm-arena refuses to run on it"
        )
    return LocalStore(Path(url).expanduser().resolve())
