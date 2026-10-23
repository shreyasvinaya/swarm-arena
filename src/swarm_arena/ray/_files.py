"""Transport bounded, hash-verified artifacts and diagnostics without pickle."""

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Final, Literal, NamedTuple
from uuid import uuid4

from swarm_arena._errors import AlreadyExists, NotFound, PreconditionFailed, StorageError
from swarm_arena._exec.executor import Job
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import Blob, Store, _check_key
from swarm_arena._tooling.result_file import _chunks, open_regular_file

MIB: Final = 1024 * 1024
_MAX_HEAD_BYTES: Final = 16 * MIB
MAX_OBJECT_STORE_CAP_MIB: Final = 1024
_MAX_BLOB_BYTES: Final = MAX_OBJECT_STORE_CAP_MIB * MIB
_MAX_BLOBS: Final = 100_000
_MAX_DIAGNOSTICS: Final = 32
_MAX_DIAGNOSTIC_BYTES: Final = 4096
_MAX_FRAME_BYTES: Final = (
    12 + _MAX_HEAD_BYTES + _MAX_BLOBS * 72 + _MAX_BLOB_BYTES + 8 + _MAX_DIAGNOSTIC_BYTES
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


@dataclass(frozen=True)
class FrameDiagnostic:
    """Carry a diagnostic code and bounded occurrence count in a private frame.

    Parameters
    ----------
    code : str
        Lowercase identifier, at most 64 ASCII characters. Domain consumers
        separately recognize the codes they accept.
    count : int
        Occurrences, from 1 through 10,000; booleans are invalid.

    Examples
    --------
    >>> FrameDiagnostic("read_failed", 1).count
    1
    """

    code: str
    count: int


class Unframed(NamedTuple):
    """Return a decoded private transport frame.

    Parameters
    ----------
    head : bytes
        Opaque header, interpreted by the consumer.
    blobs : dict[str, bytes]
        Verified artifacts, keyed by SHA-256 hex digest.
    diagnostics : tuple[FrameDiagnostic, ...]
        Shape-validated diagnostic identifiers and counts.

    Examples
    --------
    >>> unframe(frame(b"header", {})) == Unframed(b"header", {}, ())
    True
    """

    head: bytes
    blobs: dict[str, bytes]
    diagnostics: tuple[FrameDiagnostic, ...]


class MemoryStore(Store):
    """Keep worker artifacts in memory with conditional writes and child views.

    Parameters
    ----------
    blobs : Mapping[str, bytes] or None, default=None
        Initial objects, copied at construction and excluded from ``created``.
        Artifact callers use SHA-256 keys; ordinary Store keys also work.

    Examples
    --------
    >>> store = MemoryStore({"input": b"in"})
    >>> _ = store.create("output", b"out")
    >>> store.created()
    {'output': b'out'}
    """

    def __init__(self, blobs: Mapping[str, bytes] | None = None) -> None:
        """Copy initial contents without marking them as worker outputs."""
        self._blobs = {_check_key(key): bytes(data) for key, data in (blobs or {}).items()}
        self._created: dict[str, bytes] = {}
        self._prefix = ""
        self._url = f"memory://{uuid4().hex}"
        self._lock = RLock()

    @property
    def url(self) -> str:
        """Return the private in-memory root identifier.

        Returns
        -------
        str
            Memory URL; it is not accepted by public ``open_store``.

        Examples
        --------
        >>> MemoryStore().url.startswith("memory://")
        True
        """
        return self._url

    def created(self) -> dict[str, bytes]:
        """Copy objects created after initialization, relative to this view.

        Returns
        -------
        dict[str, bytes]
            New objects with their latest contents; excludes seeded objects.

        Examples
        --------
        >>> MemoryStore({"input": b"x"}).created()
        {}
        """
        with self._lock:
            return {
                key[len(self._prefix) :]: data
                for key, data in self._created.items()
                if key.startswith(self._prefix)
            }

    def read(self, key: str) -> Blob:
        """Read an object and its SHA-256 version.

        Parameters
        ----------
        key : str
            Relative storage key.

        Returns
        -------
        Blob
            Stored contents and their digest.

        Raises
        ------
        NotFound
            If no object occupies the key.

        Examples
        --------
        >>> MemoryStore({"a": b"x"}).read("a").data
        b'x'
        """
        key = self._prefix + _check_key(key)
        with self._lock:
            if key not in self._blobs:
                raise NotFound("memory object not found")
            data = self._blobs[key]
            return Blob(data, sha256_hex(data))

    def create(self, key: str, data: bytes) -> str:
        """Create an absent object atomically.

        Parameters
        ----------
        key : str
            Relative storage key.
        data : bytes
            Contents to store.

        Returns
        -------
        str
            SHA-256 version of the contents.

        Raises
        ------
        AlreadyExists
            If the key is already occupied.

        Examples
        --------
        >>> store = MemoryStore()
        >>> store.create("a", b"x") == sha256_hex(b"x")
        True
        """
        key = self._prefix + _check_key(key)
        with self._lock:
            if key in self._blobs:
                raise AlreadyExists("memory object already exists")
            data = bytes(data)
            self._blobs[key] = self._created[key] = data
            return sha256_hex(data)

    def replace(self, key: str, data: bytes, *, if_match: str) -> str:
        """Replace an object only when its current digest matches.

        Parameters
        ----------
        key : str
            Relative storage key.
        data : bytes
            Replacement contents.
        if_match : str
            Required current SHA-256 version.

        Returns
        -------
        str
            SHA-256 version of the replacement.

        Raises
        ------
        NotFound
            If the object is absent.
        PreconditionFailed
            If the current version does not match.

        Examples
        --------
        >>> store = MemoryStore({"a": b"x"})
        >>> store.replace("a", b"y", if_match=sha256_hex(b"x")) == sha256_hex(b"y")
        True
        """
        key = self._prefix + _check_key(key)
        with self._lock:
            if key not in self._blobs:
                raise NotFound("memory object not found")
            if sha256_hex(self._blobs[key]) != if_match:
                raise PreconditionFailed("memory object version mismatch")
            data = bytes(data)
            self._blobs[key] = data
            if key in self._created:
                self._created[key] = data
            return sha256_hex(data)

    def exists(self, key: str) -> bool:
        """Check for an object relative to this view.

        Parameters
        ----------
        key : str
            Relative storage key.

        Returns
        -------
        bool
            Whether the key is occupied.

        Examples
        --------
        >>> MemoryStore({"a": b"x"}).exists("a")
        True
        """
        key = self._prefix + _check_key(key)
        with self._lock:
            return key in self._blobs

    def list(self, prefix: str) -> list[str]:
        """List sorted view-relative keys with a literal prefix.

        Parameters
        ----------
        prefix : str
            Relative key prefix; empty selects every object in this view.

        Returns
        -------
        list[str]
            Matching keys in lexical order.

        Examples
        --------
        >>> MemoryStore({"b": b"", "a": b""}).list("")
        ['a', 'b']
        """
        if prefix:
            _check_key(prefix)
        with self._lock:
            return sorted(
                key[len(self._prefix) :]
                for key in self._blobs
                if key.startswith(self._prefix + prefix)
            )

    def child(self, prefix: str) -> "MemoryStore":
        """Create a child view sharing conditional writes and output tracking.

        Parameters
        ----------
        prefix : str
            Nonempty relative path below this view.

        Returns
        -------
        MemoryStore
            Shared view whose keys are relative to the child root.

        Examples
        --------
        >>> store = MemoryStore({"dir/a": b"x"})
        >>> store.child("dir").read("a").data
        b'x'
        """
        prefix = _check_key(prefix).rstrip("/")
        child = MemoryStore()
        child._blobs, child._created, child._lock = self._blobs, self._created, self._lock
        child._prefix = self._prefix + prefix + "/"
        child._url = self.url + "/" + prefix
        return child


def uses_object_store(artifact_root: str) -> bool:
    """Select inline transport for artifact roots outside S3.

    Parameters
    ----------
    artifact_root : str
        Driver artifact root.

    Returns
    -------
    bool
        True for local artifact roots.

    Examples
    --------
    >>> uses_object_store("/tmp/artifacts"), uses_object_store("s3://bucket/root")
    (True, False)
    """
    return not artifact_root.startswith("s3://")


def needed_hashes(stage: Job.Stage) -> set[str]:
    """Find every distinct artifact referenced by a staged job.

    Parameters
    ----------
    stage : Job.Stage
        Root, parent and candidate file references.

    Returns
    -------
    set[str]
        Required SHA-256 digests.

    Examples
    --------
    >>> needed_hashes(Job.Stage(root={"a": "0" * 64})) == {"0" * 64}
    True
    """
    hashes = set(stage.root.values())
    for parent in stage.parents:
        hashes.update(parent.overlay.values())
        hashes.update(file.sha256 for file in parent.files)
    if stage.candidate is not None:
        hashes.update(stage.candidate.files.values())
    return hashes


def inline_inputs(
    stage: Job.Stage, artifacts: ArtifactStore, *, cap_bytes: int
) -> dict[str, bytes]:
    """Read verified staged artifacts, enforcing the aggregate input cap.

    Parameters
    ----------
    stage : Job.Stage
        Job's staged file references.
    artifacts : ArtifactStore
        Driver artifact backend.
    cap_bytes : int
        Maximum aggregate bytes of distinct artifacts, in whole MiB from
        1 through ``MAX_OBJECT_STORE_CAP_MIB`` (1024). All sizes are checked
        before reading; local reads are bounded again against file growth.

    Returns
    -------
    dict[str, bytes]
        Contents keyed by verified digest.

    Raises
    ------
    ValueError
        If the cap is outside the supported range or aggregate input size
        exceeds it.
    NotFound
        If a referenced artifact is missing, including after size preflight.
    StorageError
        If an artifact is not a readable regular file or fails its hash check.

    Examples
    --------
    >>> inline_inputs(Job.Stage(), ArtifactStore(MemoryStore()), cap_bytes=MIB)
    {}
    """
    _validate_cap(cap_bytes)
    store = artifacts._store
    digests = sorted(needed_hashes(stage))
    sizes = {digest: _input_size(store, digest) for digest in digests}
    _check_cap(sum(sizes.values()), cap_bytes=cap_bytes, what="inputs")
    blobs: dict[str, bytes] = {}
    total = 0
    for digest in digests:
        if isinstance(store, MemoryStore):
            data = store.read(digest).data
            _check_cap(total + len(data), cap_bytes=cap_bytes, what="inputs")
        else:
            fd = _open_local_input(Path(store.url), digest)
            try:
                _check_cap(total + os.fstat(fd).st_size, cap_bytes=cap_bytes, what="inputs")
                try:
                    data = b"".join(_chunks(fd, digest, cap_bytes - total, step=MIB))
                except ValueError:
                    _check_cap(total + os.fstat(fd).st_size, cap_bytes=cap_bytes, what="inputs")
                    raise StorageError("artifact input read failed") from None
            finally:
                os.close(fd)
        if sha256_hex(data) != digest:
            raise StorageError("artifact failed its hash check")
        total += len(data)
        blobs[digest] = data
    return blobs


def _open_local_input(root: Path, digest: str) -> int:
    """Keep missing and damaged artifacts on the storage error path."""
    try:
        return open_regular_file(root, digest)
    except ValueError:
        try:
            os.lstat(root / digest)
        except FileNotFoundError:
            raise NotFound("store key not found") from None
        except OSError:
            raise StorageError("artifact input read failed") from None
        raise StorageError("artifact input read failed") from None


def _input_size(store: Store, digest: str) -> int:
    """Inspect an artifact's size without reading its contents."""
    if isinstance(store, MemoryStore):
        with store._lock:
            key = store._prefix + _check_key(digest)
            if key not in store._blobs:
                raise NotFound("memory object not found")
            return len(store._blobs[key])
    fd = _open_local_input(Path(store.url), digest)
    try:
        return os.fstat(fd).st_size
    finally:
        os.close(fd)


def _validate_cap(cap_bytes: int) -> None:
    if (
        type(cap_bytes) is not int
        or not MIB <= cap_bytes <= MAX_OBJECT_STORE_CAP_MIB * MIB
        or cap_bytes % MIB
    ):
        raise ValueError("object_store_cap_mib must be an integer from 1 through 1024")


def _check_cap(n: int, *, cap_bytes: int, what: Literal["inputs", "outputs"]) -> None:
    if n > cap_bytes:
        cap_mib = cap_bytes // MIB
        raise ValueError(
            f"{what} of this tool run are {n} bytes, over the Ray object_store_cap_mib "
            f"cap of {cap_mib} MiB; use an s3:// artifact_root"
        )


def check_cap(
    blobs: Mapping[str, bytes], *, cap_bytes: int, what: Literal["inputs", "outputs"]
) -> None:
    """Check one direction of a tool run's object-store byte budget.

    Parameters
    ----------
    blobs : Mapping[str, bytes]
        Distinct files, including output logs when checking outputs.
    cap_bytes : int
        Maximum aggregate content bytes, in whole MiB from 1 through 1024.
    what : {"inputs", "outputs"}
        Direction named in the fixed cap diagnostic.

    Raises
    ------
    ValueError
        If the cap is outside the supported range or aggregate content
        exceeds it.

    Examples
    --------
    >>> check_cap({"a": b"x"}, cap_bytes=MIB, what="inputs")
    """
    _validate_cap(cap_bytes)
    _check_cap(sum(map(len, blobs.values())), cap_bytes=cap_bytes, what=what)


def _validate_diagnostics(diagnostics: Sequence[FrameDiagnostic]) -> None:
    if len(diagnostics) > _MAX_DIAGNOSTICS:
        raise ValueError("invalid frame diagnostics")
    for item in diagnostics:
        if (
            not isinstance(item, FrameDiagnostic)
            or not isinstance(item.code, str)
            or _CODE.fullmatch(item.code) is None
            or type(item.count) is not int
            or not 1 <= item.count <= 10_000
        ):
            raise ValueError("invalid frame diagnostics")


def frame(
    head: bytes, blobs: Mapping[str, bytes], diagnostics: Sequence[FrameDiagnostic] = ()
) -> bytes:
    """Encode an opaque header, sorted blobs and a diagnostic JSON section.

    Parameters
    ----------
    head : bytes
        Opaque header, at most 16 MiB.
    blobs : Mapping[str, bytes]
        At most 100,000 digest-keyed blobs, at most 1 GiB in total.
    diagnostics : Sequence[FrameDiagnostic], default=()
        At most 32 shape-valid code/count pairs.

    Returns
    -------
    bytes
        SAF1 frame with unsigned 8-byte big-endian section lengths.

    Raises
    ------
    ValueError
        If a size, digest identifier or diagnostic shape is invalid.

    Examples
    --------
    >>> unframe(frame(b"{}", {})).head
    b'{}'
    """
    if len(head) > _MAX_HEAD_BYTES or len(blobs) > _MAX_BLOBS:
        raise ValueError("invalid frame length")
    _validate_diagnostics(diagnostics)
    slot = json.dumps([[d.code, d.count] for d in diagnostics], separators=(",", ":")).encode()
    if len(slot) > _MAX_DIAGNOSTIC_BYTES:
        raise ValueError("invalid frame diagnostics length")
    total = 0
    for digest, blob in blobs.items():
        if _HEX.fullmatch(digest) is None:
            raise ValueError("invalid frame sha256")
        total += len(blob)
        if total > _MAX_BLOB_BYTES:
            raise ValueError("invalid frame length")
    parts = [b"SAF1", len(head).to_bytes(8, "big"), head]
    for digest in sorted(blobs):
        blob = blobs[digest]
        parts.extend((digest.encode("ascii"), len(blob).to_bytes(8, "big"), blob))
    parts.extend((len(slot).to_bytes(8, "big"), slot))
    return b"".join(parts)


def unframe(data: bytes) -> Unframed:
    """Decode bounded frame sections and verify every blob before returning.

    Parameters
    ----------
    data : bytes
        Complete SAF1 frame. Diagnostics follow the blobs as length-prefixed
        compact JSON pairs. Their bounded length starts with a zero byte,
        distinguishing the section from a blob's ASCII hexadecimal digest.

    Returns
    -------
    Unframed
        Header, verified blobs and shape-validated diagnostics.

    Raises
    ------
    ValueError
        On invalid magic, lengths, digest, diagnostics, duplicate blobs or
        trailing bytes. Messages never include frame contents.

    Examples
    --------
    >>> unframe(frame(b"h", {}, [FrameDiagnostic("read_failed", 2)])).diagnostics
    (FrameDiagnostic(code='read_failed', count=2),)
    """
    if len(data) > _MAX_FRAME_BYTES:
        raise ValueError("invalid frame length")
    if data[:4] != b"SAF1":
        raise ValueError("invalid frame magic")
    position = 4

    def section(limit: int) -> bytes:
        nonlocal position
        if len(data) - position < 8:
            raise ValueError("invalid frame length")
        size = int.from_bytes(data[position : position + 8], "big")
        position += 8
        if size > limit or size > len(data) - position:
            raise ValueError("invalid frame length")
        result = data[position : position + size]
        position += size
        return result

    head = section(_MAX_HEAD_BYTES)
    blobs: dict[str, bytes] = {}
    total = 0
    previous = ""
    while position < len(data) and data[position] in b"0123456789abcdef":
        if len(blobs) >= _MAX_BLOBS or len(data) - position < 64:
            raise ValueError("invalid frame length")
        raw_digest = data[position : position + 64]
        position += 64
        try:
            digest = raw_digest.decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("invalid frame sha256") from None
        if _HEX.fullmatch(digest) is None or digest <= previous:
            raise ValueError("invalid frame sha256 order")
        blob = section(_MAX_BLOB_BYTES - total)
        total += len(blob)
        if sha256_hex(blob) != digest:
            raise ValueError("frame sha256 mismatch")
        blobs[digest] = blob
        previous = digest
    slot = section(_MAX_DIAGNOSTIC_BYTES)
    if position != len(data):
        raise ValueError("invalid frame trailing bytes")
    try:
        pairs = json.loads(slot.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ValueError("invalid frame diagnostics JSON") from None
    if (
        not isinstance(pairs, list)
        or len(pairs) > _MAX_DIAGNOSTICS
        or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
    ):
        raise ValueError("invalid frame diagnostics")
    diagnostics = tuple(FrameDiagnostic(pair[0], pair[1]) for pair in pairs)
    _validate_diagnostics(diagnostics)
    return Unframed(head, blobs, diagnostics)
