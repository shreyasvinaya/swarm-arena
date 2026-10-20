"""Store immutable file contents by SHA-256 and verify every retrieval."""

import contextlib
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from swarm_arena._errors import AlreadyExists, StorageError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.tooling import FileRef
from swarm_arena._store.base import Store
from swarm_arena._store.local import LocalStore

_MAX_ARTIFACT_BYTES: Final = 2**30


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
        Reads refuse symlinks in every component and nonregular sources, and
        cap each file at 1 GiB before reading. Hashing and publication use
        1 MiB chunks and atomic, write-once local storage.

        Parameters
        ----------
        path : Path
            Regular source file, at most 1 GiB, without symlinked components.

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
        >>> from tempfile import TemporaryDirectory
        >>> from swarm_arena._store.base import open_store
        >>> with TemporaryDirectory() as tmp:
        ...     root = Path(tmp).resolve()
        ...     artifacts = ArtifactStore(open_store(str(root / "artifacts")))
        ...     _ = (root / "output.txt").write_bytes(b"hello")
        ...     ref = artifacts.put_file(root / "output.txt")
        ...     (ref.path, ref.size, artifacts.get_bytes(ref.sha256))
        ('output.txt', 5, b'hello')
        """
        from swarm_arena._tooling.result_file import _chunks, open_regular_file

        try:
            FileRef(name=path.name, path=path.name, sha256="0" * 64, size=0)
        except ValidationError:
            raise ValueError("invalid artifact file reference: path") from None
        absolute = path.absolute()
        try:
            fd = open_regular_file(
                Path(absolute.anchor), absolute.relative_to(absolute.anchor).as_posix()
            )
            try:
                digest, size = self._put_chunks(
                    _chunks(fd, path.name, _MAX_ARTIFACT_BYTES, step=2**20)
                )
            finally:
                os.close(fd)
        except (OSError, ValueError):
            raise StorageError("artifact source read failed") from None
        return FileRef(name=path.name, path=path.name, sha256=digest, size=size)

    def _put_chunks(self, chunks: Iterable[bytes]) -> tuple[str, int]:
        """Publish local artifact chunks without adding a streaming Store API."""
        if not isinstance(self._store, LocalStore):
            raise StorageError("streamed artifacts require local storage")
        return self._store._create_artifact(chunks)

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

        Reads are capped at 1 GiB and refuse symlinks or special files. A
        destination is published atomically through pinned directories; no
        symlinked directory or leaf is followed and hardlinks are replaced.

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
        >>> from tempfile import TemporaryDirectory
        >>> from swarm_arena._store.base import open_store
        >>> with TemporaryDirectory() as tmp:
        ...     root = Path(tmp).resolve()
        ...     artifacts = ArtifactStore(open_store(str(root / "artifacts")))
        ...     digest = artifacts.put_bytes(b"hello")
        ...     artifacts.fetch(digest, root / "workspace/input.txt")
        ...     (root / "workspace/input.txt").read_bytes()
        b'hello'
        """
        from swarm_arena._tooling.materialize import _directory, _write
        from swarm_arena._tooling.result_file import read_regular_file

        if not isinstance(self._store, LocalStore):
            raise StorageError("artifact fetch requires local storage")
        try:
            data = read_regular_file(Path(self._store.url), sha256, max_bytes=_MAX_ARTIFACT_BYTES)
            if sha256_hex(data) != sha256:
                raise StorageError("artifact failed its hash check")
            absolute = dest.absolute()
            root = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with _directory(root, absolute.parts[1:-1]) as directory:
                    _write(directory, absolute.name, data, read_only=read_only)
            finally:
                os.close(root)
        except (OSError, ValueError):
            raise StorageError("artifact destination or source failed") from None
