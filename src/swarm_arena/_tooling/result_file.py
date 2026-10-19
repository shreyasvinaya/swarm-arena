"""Read result envelopes and workspace artifacts without following file links."""

import contextlib
import errno
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final, NoReturn

from pydantic import Field, ValidationError

from swarm_arena._model.base import StoredModel
from swarm_arena._model.migrate import CURRENT
from swarm_arena._model.tooling import (
    CompletedRun,
    FileRef,
    RunStatus,
    ToolOutput,
    ToolRequest,
    Usage,
)
from swarm_arena._tooling.paths import output_paths

_RESULT = "swarm/result.json"
_CHUNK: Final = 1 << 20
_MAX_READ: Final = 1 << 30  # one read(2) call; macOS refuses more than INT_MAX
MAX_RESULT_JSON_BYTES: Final = 16 * 2**20
"""Most bytes ``read_result_json`` reads from ``swarm/result.json``."""
MAX_DECLARED_FILES_BYTES: Final = 256 * 2**20
"""Most bytes of all files one result declares (the ``LocalExecutor`` default cap)."""


class ToolResultFile(StoredModel):
    """
    Validate the result envelope written by a tool program.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version; omission means version one.
    data : dict[str, Any]
        Structured tool result.
    usage : Usage, default=Usage()
        Token counts and reported cost.
    files : list[str], default=[]
        Declared output paths relative to the workspace.
    failed_provider : str or None, default=None
        Nonempty provider name reporting a provider failure.

    Examples
    --------
    >>> ToolResultFile(data={"x": 1}).schema_version
    1
    """

    schema_kind: ClassVar[str] = "tool_result"
    data: dict[str, Any]
    usage: Usage = Usage()
    files: list[str] = []
    failed_provider: str | None = Field(default=None, min_length=1)


def request_json(request: ToolRequest) -> dict[str, Any]:
    """
    Build the script request object with tool inputs under ``params``.

    Parameters
    ----------
    request : ToolRequest
        Tool invocation to stage.

    Returns
    -------
    dict[str, Any]
        Versioned parameters, round, overrides, and replicate index.

    Examples
    --------
    >>> from swarm_arena.tool import ToolRequest
    >>> request_json(ToolRequest(inputs={"n": 2}))["params"]
    {'n': 2}
    """
    return {
        "schema_version": 1,
        "params": request.inputs,
        "round": request.round,
        "round_overrides": request.round_overrides,
        "replicate": request.replicate,
    }


def symlinked_component(base: Path, rel: str) -> str | None:
    """Return the first symlink component below base, or None if none exists."""
    current = base
    parts: list[str] = []
    for part in PurePosixPath(rel).parts:
        parts.append(part)
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(mode):
            return "/".join(parts)
    return None


def open_regular_file(workspace: Path, rel: str) -> int:
    """
    Open a regular workspace file read-only without following any symlink component.

    Each directory is pinned by its descriptor while the next component is
    opened with ``O_NOFOLLOW | O_NONBLOCK``, so a parent swapped for a symlink
    cannot redirect the open, and a FIFO never blocks it.

    Parameters
    ----------
    workspace : Path
        Trusted workspace root.
    rel : str
        Relative path, including internal paths such as ``swarm/result.json``.

    Returns
    -------
    int
        A read-only descriptor of a regular file; the caller closes it.

    Raises
    ------
    ValueError
        ``<rel> is not a regular file`` if the path escapes the workspace, is
        missing, contains a symlink, cannot be opened, or is not a regular file.

    Examples
    --------
    >>> import os, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     size = (Path(tmp) / "out.txt").write_text("hi")
    ...     fd = open_regular_file(Path(tmp), "out.txt")
    ...     os.read(fd, 10)
    ...     os.close(fd)
    b'hi'
    """
    try:
        parts = PurePosixPath(rel).parts
        if not parts or rel.startswith("/") or ".." in parts or "\0" in rel:
            raise ValueError
        if symlinked_component(workspace, rel) is not None:
            raise ValueError
        # Pin each directory while opening the next component. The lstat walk
        # alone would allow a parent to be replaced by a symlink before open.
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with contextlib.ExitStack() as stack:
            directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NONBLOCK)
            stack.callback(os.close, directory)
            for part in parts[:-1]:
                directory = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
                stack.callback(os.close, directory)
            fd = os.open(parts[-1], flags, dir_fd=directory)
    except (OSError, ValueError):
        raise ValueError(f"{rel} is not a regular file") from None
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError:
        regular = False
    if not regular:
        os.close(fd)
        raise ValueError(f"{rel} is not a regular file")
    return fd


def _chunks(fd: int, rel: str, max_bytes: int | None, *, step: int | None) -> Iterator[bytes]:
    """
    Yield a file's bytes, refusing it unread when ``fstat`` shows it is over the cap.

    Reads go through ``os.read`` and never past ``max_bytes``. Each read asks
    for ``step`` bytes, or with no step for the size ``fstat`` reported, so a
    whole file usually arrives in one buffer. A file that grows past the cap
    after the first check is refused when the cap is reached.
    """
    limit = sys.maxsize if max_bytes is None else max_bytes
    size = os.fstat(fd).st_size
    if size > limit:
        raise ValueError(f"{rel} crosses the size cap")
    total = 0
    while total < limit:
        want = step if step is not None else max(size - total, _CHUNK)
        chunk = os.read(fd, min(want, limit - total, _MAX_READ))
        if not chunk:
            return
        total += len(chunk)
        yield chunk
    if os.fstat(fd).st_size > limit:
        raise ValueError(f"{rel} crosses the size cap")


def read_regular_file(workspace: Path, rel: str, *, max_bytes: int | None = None) -> bytes:
    """
    Read a regular workspace file without following any symlink component.

    The file is opened through ``open_regular_file``. Its ``fstat`` size is
    checked before any read: a file over ``max_bytes`` is refused unread, so a
    sparse file is never read whole, and no read goes past the cap even if
    the file grows meanwhile.

    Parameters
    ----------
    workspace : Path
        Trusted workspace root.
    rel : str
        Relative path, including internal paths such as ``swarm/result.json``.
    max_bytes : int or None, optional
        Most bytes to read. Every read of a tool-written file passes a cap;
        None reads the whole file, for human-authored sources only.

    Returns
    -------
    bytes
        File contents.

    Raises
    ------
    ValueError
        ``<rel> is not a regular file`` if the path escapes the workspace, is
        missing, contains a symlink, cannot be read, or is not a regular file;
        ``<rel> crosses the size cap`` if it is larger than ``max_bytes``.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     size = (Path(tmp) / "out.txt").write_text("hello")
    ...     read_regular_file(Path(tmp), "out.txt", max_bytes=5)
    b'hello'
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     size = (Path(tmp) / "out.txt").write_text("hello")
    ...     read_regular_file(Path(tmp), "out.txt", max_bytes=4)
    Traceback (most recent call last):
    ...
    ValueError: out.txt crosses the size cap
    """
    fd = open_regular_file(workspace, rel)
    try:
        return b"".join(_chunks(fd, rel, max_bytes, step=None))
    except OSError:
        raise ValueError(f"{rel} is not a regular file") from None
    finally:
        os.close(fd)


def _hash_regular_file(workspace: Path, rel: str, *, max_bytes: int) -> tuple[str, int]:
    """Hash a regular workspace file in chunks; return its SHA-256 digest and size."""
    fd = open_regular_file(workspace, rel)
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in _chunks(fd, rel, max_bytes, step=_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    except OSError:
        raise ValueError(f"{rel} is not a regular file") from None
    finally:
        os.close(fd)
    return digest.hexdigest(), size


def _regular_file_size(workspace: Path, rel: str) -> int:
    """Return a regular workspace file's ``fstat`` size without reading it."""
    fd = open_regular_file(workspace, rel)
    try:
        return os.fstat(fd).st_size
    except OSError:
        raise ValueError(f"{rel} is not a regular file") from None
    finally:
        os.close(fd)


def _reject(value: str) -> NoReturn:
    """Reject JSON's nonstandard non-finite constants without echoing input."""
    raise ValueError("non-finite number in swarm/result.json")


def _finite_float(value: str) -> float:
    """Reject overflow in otherwise valid JSON float literals."""
    number = float(value)
    if not math.isfinite(number):
        _reject(value)
    return number


def read_result_json(workspace: Path) -> dict[str, Any]:
    """
    Read a JSON object from a regular ``swarm/result.json`` file.

    Parameters
    ----------
    workspace : Path
        Workspace containing the result file.

    Returns
    -------
    dict[str, Any]
        Raw object with finite JSON numbers.

    Raises
    ------
    ValueError
        If the file is missing, unsafe, unreadable, larger than
        ``MAX_RESULT_JSON_BYTES`` (16 MiB), invalid JSON, not an object, or
        contains a non-finite number.

    Examples
    --------
    >>> result = read_result_json(Path("workspace"))  # doctest: +SKIP
    """
    # Check links before existence: a dangling symlink is unsafe, not missing.
    try:
        if symlinked_component(workspace, _RESULT) is not None:
            raise ValueError(f"{_RESULT} is not a regular file")
        (workspace / _RESULT).lstat()
    except FileNotFoundError:
        raise ValueError(f"{_RESULT} is missing") from None
    except OSError:
        raise ValueError(f"{_RESULT} is not a regular file") from None
    raw = read_regular_file(workspace, _RESULT, max_bytes=MAX_RESULT_JSON_BYTES)
    try:
        text = raw.decode("utf-8")
        result = json.loads(text, parse_constant=_reject, parse_float=_finite_float)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"{_RESULT} is not valid JSON: invalid encoding or syntax") from None
    except RecursionError:
        raise ValueError(f"{_RESULT} is not valid JSON: nesting too deep") from None
    except ValueError as exc:
        # parse hooks supply this fixed reason; int conversion can also reject
        # extremely long numeric literals without exposing their contents.
        if str(exc) == "non-finite number in swarm/result.json":
            raise ValueError("non-finite number in swarm/result.json") from None
        raise ValueError(f"{_RESULT} is not valid JSON: invalid number") from None
    if not isinstance(result, dict):
        raise ValueError(f"{_RESULT} must contain a JSON object")
    return result


def _validation_reason(exc: ValidationError) -> str:
    """Format locations already sanitized by the shared model boundary."""
    reasons = [
        (".".join(str(part) for part in error["loc"]) + ": " if error["loc"] else "")
        + error["type"]
        for error in exc.errors(include_input=False, include_url=False, include_context=False)
    ]
    return f"{_RESULT}: " + "; ".join(reasons)


def _load_result(workspace: Path) -> tuple[ToolResultFile, list[FileRef]]:
    """Validate the envelope and all artifacts before accepting any result."""
    raw = read_result_json(workspace)
    version = raw.get("schema_version", 1)
    if (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version > CURRENT["tool_result"]
    ):
        raise ValueError("tool_result schema_version is newer than this swarm-arena supports")
    try:
        result = ToolResultFile.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(_validation_reason(exc)) from None
    # Pydantic may coerce strings such as "NaN" to floats after JSON parsing.
    if result.usage.usd is not None and not math.isfinite(result.usage.usd):
        raise ValueError("non-finite number in swarm/result.json")
    # The result parser has no original stage. The executor's upload collector
    # performs the full staged-layout check with its required stage argument.
    paths = output_paths(result.files, effective_files=())
    total = 0
    for rel in paths:  # every size before any read, so an oversized set is never read
        total += _regular_file_size(workspace, rel)
        if total > MAX_DECLARED_FILES_BYTES:
            raise ValueError(
                f"declared files are too large: {rel} crosses MAX_DECLARED_FILES_BYTES"
            )
    files: list[FileRef] = []
    left = MAX_DECLARED_FILES_BYTES
    for rel in paths:  # one file at a time, each bounded by what is left of the cap
        digest, size = _hash_regular_file(workspace, rel, max_bytes=left)
        left -= size
        files.append(FileRef(name=PurePosixPath(rel).name, path=rel, sha256=digest, size=size))
    return result, files


def tool_output_from_result(workspace: Path, run: CompletedRun) -> ToolOutput:
    """
    Collect a tool envelope according to process status and artifact validity.

    Every read is bounded: ``swarm/result.json`` by ``MAX_RESULT_JSON_BYTES``
    (16 MiB), and the declared files together by ``MAX_DECLARED_FILES_BYTES``
    (256 MiB). Their sizes are summed before any of them is read, so a result
    whose files cross the cap is invalid unread; the files are hashed one at a
    time.

    Parameters
    ----------
    workspace : Path
        Completed tool workspace.
    run : CompletedRun
        Process status, timestamps, and log paths.

    Returns
    -------
    ToolOutput
        Valid data for successful or partial runs; usage only for errors and
        provider failures. An invalid successful result becomes an error. A
        timeout or cancel keeps a valid result, even an empty one, with
        ``partial=True``; without one the output is empty with
        ``partial=False``, and the reason goes to the stderr log.

    Examples
    --------
    >>> output = tool_output_from_result(Path("workspace"), run)  # doctest: +SKIP
    """
    status = run.status
    try:
        result, files = _load_result(workspace)
    except ValueError as exc:
        if status in (RunStatus.OK, RunStatus.TIMEOUT, RunStatus.CANCELLED):
            note_in_log(run.stderr_path, str(exc))  # why nothing was kept
        if status == RunStatus.OK:
            status = RunStatus.ERROR
        return ToolOutput(status=status, started_at=run.started_at, ended_at=run.ended_at)

    failed_provider = None
    if status in (RunStatus.OK, RunStatus.ERROR) and result.failed_provider is not None:
        status = RunStatus.PROVIDER_FAILED
        failed_provider = result.failed_provider
    partial = status in (RunStatus.TIMEOUT, RunStatus.CANCELLED)
    keep_data = status == RunStatus.OK or partial
    return ToolOutput(
        status=status,
        partial=partial,
        data=result.data if keep_data else {},
        files=files if keep_data else [],
        usage=result.usage,
        failed_provider=failed_provider,
        started_at=run.started_at,
        ended_at=run.ended_at,
    )


def command_output_from_result(workspace: Path, run: CompletedRun) -> ToolOutput:
    """
    Collect a command's raw JSON object without tool-envelope interpretation.

    Parameters
    ----------
    workspace : Path
        Completed command workspace.
    run : CompletedRun
        Process status, timestamps, and log paths.

    Returns
    -------
    ToolOutput
        Raw JSON data and the process status. A successful process with no
        valid result becomes an error with a diagnostic in its stderr log.

    Examples
    --------
    >>> output = command_output_from_result(Path("workspace"), run)  # doctest: +SKIP
    """
    status = run.status
    try:
        data = read_result_json(workspace)
    except ValueError as exc:
        data = {}
        if status == RunStatus.OK:
            status = RunStatus.ERROR
            note_in_log(run.stderr_path, str(exc))
    return ToolOutput(status=status, data=data, started_at=run.started_at, ended_at=run.ended_at)


_NOT_SINGLE_REGULAR = "not a regular file with one link"


def open_log(path: Path, *, truncate: bool) -> int:
    """
    Open a log file for writing without following links or blocking.

    Every component is opened from its parent's descriptor with
    ``O_NOFOLLOW | O_NONBLOCK`` (directories also ``O_DIRECTORY``), so a
    symlinked component, or one replaced during the walk, cannot redirect the
    open. The leaf is created if missing and is never truncated before it
    passes the checks. A FIFO without a reader fails at once instead of
    blocking.

    Parameters
    ----------
    path : Path
        Log path. Every component must be a real directory, not a symlink.
    truncate : bool
        True empties the file after the checks; False opens it in append mode.

    Returns
    -------
    int
        A blocking, non-inheritable write descriptor that the caller closes.

    Raises
    ------
    OSError
        ``cannot open log file safely: <reason>`` with ``filename`` set to
        ``path``, when a component is a symlink or missing, or the leaf is not
        a regular file with exactly one link (a hardlink, FIFO, or directory).

    Examples
    --------
    >>> import os, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     log = Path(tmp).resolve() / "stderr.log"
    ...     fd = open_log(log, truncate=True)
    ...     written = os.write(fd, b"hi")
    ...     os.close(fd)
    ...     log.read_text()
    'hi'
    """
    fds: list[int] = []
    try:
        parts = path.parts[1:] if path.anchor else path.parts
        if not parts:
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR))
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
        directory = os.open(path.anchor or ".", directory_flags)
        fds.append(directory)
        for part in parts[:-1]:
            directory = os.open(part, directory_flags, dir_fd=directory)
            fds.append(directory)
        leaf_flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(
            parts[-1], leaf_flags if truncate else leaf_flags | os.O_APPEND, 0o644, dir_fd=directory
        )
        fds.append(fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError(errno.EINVAL, _NOT_SINGLE_REGULAR)
        if truncate:
            os.ftruncate(fd, 0)
        os.set_blocking(fd, True)
        fds.remove(fd)
        return fd
    except (OSError, ValueError) as exc:
        code = exc.errno if isinstance(exc, OSError) and exc.errno else errno.EINVAL
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else "invalid path"
        raise OSError(code, f"cannot open log file safely: {reason}", os.fspath(path)) from None
    finally:
        for fd in reversed(fds):
            with contextlib.suppress(OSError):
                os.close(fd)


def note_in_log(path: Path, message: str) -> None:
    """
    Append a diagnostic to a regular log file, dropping unsafe or failed writes.

    Parameters
    ----------
    path : Path
        Log path, opened through ``open_log``: a missing leaf is created;
        symlinks in any component, hardlinked files, and special files are
        refused, and the note is dropped.
    message : str
        Diagnostic without the prefix or trailing newline. Nonprintable
        characters, including controls and line separators, become spaces.

    Examples
    --------
    >>> note_in_log(Path("stderr.log"), "result is missing")  # doctest: +SKIP
    """
    try:
        fd = open_log(path, truncate=False)
    except OSError:
        return
    try:
        sanitized = "".join(char if char.isprintable() else " " for char in message)
        os.write(fd, f"swarm-arena: {sanitized}\n".encode())
    except OSError:
        pass
    finally:
        os.close(fd)
