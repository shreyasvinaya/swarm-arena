"""Store immutable file contents by SHA-256 and verify every retrieval."""

import contextlib
from pathlib import Path

from pydantic import ValidationError

from swarm_arena._errors import AlreadyExists, StorageError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.tooling import FileRef
from swarm_arena._store.base import Store


class ArtifactStore:
    """Address artifact contents by their SHA-256 digest.

    Parameters
    ----------
    store : Store
        Backend holding one object per digest. Construction performs no I/O.

    Examples
    --------
    >>> from swarm_arena._store.base import open_store
    >>> artifacts = ArtifactStore(open_store("./artifacts"))  # doctest: +SKIP
    >>> digest = artifacts.put_bytes(b"hello")  # doctest: +SKIP
    """

    def __init__(self, store: Store) -> None:
        """Remember the backend without accessing it."""
        self._store = store

    def put_bytes(self, data: bytes) -> str:
        """Store contents once under their SHA-256 digest.

        Parameters
        ----------
        data : bytes
            Complete artifact contents.

        Returns
        -------
        str
            Lowercase SHA-256 digest; an existing key counts as success.

        Examples
        --------
        >>> digest = artifacts.put_bytes(b"hello")  # doctest: +SKIP
        """
        digest = sha256_hex(data)
        with contextlib.suppress(AlreadyExists):
            self._store.create(digest, data)
        return digest

    def put_file(self, path: Path) -> FileRef:
        """Snapshot a file and return its basename-based reference.

        Callers can set a workspace-relative path with ``model_copy``. The
        basename must satisfy ``FileRef.path`` rules, including reserved names.

        Parameters
        ----------
        path : Path
            Source file to read.

        Returns
        -------
        FileRef
            Reference with name and path equal to the source basename.

        Raises
        ------
        ValueError
            If the basename is invalid as a file-reference path.
        StorageError
            If reading the source fails.

        Examples
        --------
        >>> ref = artifacts.put_file(Path("results/output.txt"))  # doctest: +SKIP
        >>> ref.path  # doctest: +SKIP
        'output.txt'
        """
        try:
            data = path.read_bytes()
        except OSError:
            raise StorageError("artifact source read failed") from None
        digest = sha256_hex(data)
        try:
            ref = FileRef(name=path.name, path=path.name, sha256=digest, size=len(data))
        except ValidationError:
            raise ValueError("invalid artifact file reference: path") from None
        self.put_bytes(data)
        return ref

    def get_bytes(self, sha256: str) -> bytes:
        """Read artifact contents and verify their digest.

        Parameters
        ----------
        sha256 : str
            Expected digest and backend key.

        Returns
        -------
        bytes
            Verified artifact contents.

        Raises
        ------
        NotFound
            If the artifact is missing.
        StorageError
            If its contents fail the hash check.

        Examples
        --------
        >>> artifacts.get_bytes(digest)  # doctest: +SKIP
        b'hello'
        """
        data = self._store.read(sha256).data
        if sha256_hex(data) != sha256:
            raise StorageError("artifact failed its hash check")
        return data

    def fetch(self, sha256: str, dest: Path, *, read_only: bool = True) -> None:
        """Write verified contents to a destination with explicit permissions.

        Parameters
        ----------
        sha256 : str
            Digest of the artifact to retrieve.
        dest : Path
            Destination file; missing parent directories are created.
        read_only : bool, default=True
            Set mode 0444 when true, otherwise 0644.

        Raises
        ------
        StorageError
            If verification or destination writing fails.

        Examples
        --------
        >>> artifacts.fetch(digest, Path("workspace/input.txt"))  # doctest: +SKIP
        """
        data = self.get_bytes(sha256)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            dest.chmod(0o444 if read_only else 0o644)
        except OSError:
            raise StorageError("artifact destination write failed") from None
