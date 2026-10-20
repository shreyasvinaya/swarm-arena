"""Publish complete local objects with an exclusive temporary file and hard link."""

import contextlib
import hashlib
import os
from collections.abc import Iterable
from pathlib import Path
from secrets import token_hex

from swarm_arena._errors import AlreadyExists, NotFound, StorageError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._store.base import Blob, Store, _check_key


class LocalStore(Store):
    """Store write-once objects below a local directory.

    Construction performs no filesystem I/O. Directories are created lazily
    when an object is written. The backend requires POSIX hard-link support.

    Parameters
    ----------
    root : Path
        Local directory containing the store's keys.

    Examples
    --------
    >>> store = LocalStore(Path("/tmp/artifacts"))
    >>> etag = store.create("example", b"hello")  # doctest: +SKIP
    """

    def __init__(self, root: Path) -> None:
        """Remember the root without touching the filesystem."""
        self._root = root

    @property
    def url(self) -> str:
        """Return the resolved absolute path of the root.

        Returns
        -------
        str
            Absolute local path.

        Examples
        --------
        >>> LocalStore(Path("/tmp/artifacts")).url  # doctest: +SKIP
        '/tmp/artifacts'
        """
        return str(self._root.resolve())

    def read(self, key: str) -> Blob:
        """Read an object's bytes and compute its SHA-256 version.

        Parameters
        ----------
        key : str
            Non-empty relative POSIX key without dot segments.

        Returns
        -------
        Blob
            Bytes and their SHA-256 digest.

        Raises
        ------
        NotFound
            If the key is missing.
        StorageError
            If reading the object fails.

        Examples
        --------
        >>> store.read("example").data  # doctest: +SKIP
        b'hello'
        """
        path = self._root / _check_key(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise NotFound("store key not found") from None
        except OSError:
            raise StorageError("store read failed") from None
        return Blob(data, sha256_hex(data))

    def create(self, key: str, data: bytes) -> str:
        """Publish bytes atomically if the destination key is absent.

        Parameters
        ----------
        key : str
            Non-empty relative POSIX key without dot segments.
        data : bytes
            Complete object contents.

        Returns
        -------
        str
            SHA-256 digest of the published bytes.

        Raises
        ------
        AlreadyExists
            If the destination already exists.
        OSError
            If writing, syncing, linking, or temporary-file cleanup fails.

        Examples
        --------
        >>> etag = store.create("example", b"hello")  # doctest: +SKIP
        """
        path = self._root / _check_key(key)
        temp = path.with_name(f"{path.name}.tmp-{token_hex(6)}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temp, path)
                except FileExistsError:
                    raise AlreadyExists("store key already exists") from None
            finally:
                temp.unlink(missing_ok=True)
        except OSError:
            raise OSError("store create failed") from None
        return sha256_hex(data)

    def _create_artifact(self, chunks: Iterable[bytes]) -> tuple[str, int]:
        """Hash and publish one artifact with memory bounded by the input chunk size."""
        temp = self._root / f".artifact.tmp-{token_hex(16)}"
        digest = hashlib.sha256()
        size = 0
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                for chunk in chunks:
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            key = digest.hexdigest()
            with contextlib.suppress(FileExistsError):
                os.link(temp, self._root / key)
            return key, size
        except OSError:
            raise StorageError("artifact source or destination failed") from None
        finally:
            temp.unlink(missing_ok=True)

    def replace(self, key: str, data: bytes, *, if_match: str) -> str:
        """Refuse conditional replacement until M1b.

        Parameters
        ----------
        key : str
            Object key.
        data : bytes
            Proposed replacement contents.
        if_match : str
            Required current version.

        Returns
        -------
        str
            Replacement version, once implemented.

        Raises
        ------
        NotImplementedError
            Always in M1a.

        Examples
        --------
        >>> store.replace("example", b"updated", if_match="old")  # doctest: +SKIP
        """
        raise NotImplementedError("Store.replace arrives in M1b")

    def exists(self, key: str) -> bool:
        """Check whether the key exists on disk.

        Parameters
        ----------
        key : str
            Non-empty relative POSIX key without dot segments.

        Returns
        -------
        bool
            Whether the key exists.

        Examples
        --------
        >>> store.exists("example")  # doctest: +SKIP
        True
        """
        path = self._root / _check_key(key)
        try:
            return path.exists()
        except OSError:
            raise StorageError("store existence check failed") from None

    def list(self, prefix: str) -> list[str]:
        """List sorted regular-file keys matching a literal prefix.

        Parameters
        ----------
        prefix : str
            Literal prefix, possibly empty or ending in a slash.

        Returns
        -------
        list[str]
            Root-relative keys, excluding temporary and lock file names.

        Examples
        --------
        >>> store.list("runs/")  # doctest: +SKIP
        ['runs/a.json']
        """
        try:
            return sorted(
                key
                for path in self._root.rglob("*")
                if ".tmp-" not in path.name and not path.name.endswith(".lock") and path.is_file()
                if (key := path.relative_to(self._root).as_posix()).startswith(prefix)
            )
        except OSError:
            raise StorageError("store listing failed") from None

    def child(self, prefix: str) -> Store:
        """Create a child store without touching the filesystem.

        Parameters
        ----------
        prefix : str
            Non-empty relative POSIX key without dot segments.

        Returns
        -------
        Store
            Local store below this root.

        Examples
        --------
        >>> runs = LocalStore(Path("artifacts")).child("runs")
        """
        return LocalStore(self._root / _check_key(prefix))
