"""Read result envelopes and workspace artifacts without following file links."""

import contextlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, NoReturn

from pydantic import Field, ValidationError

from swarm_arena._model.base import StoredModel
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.migrate import CURRENT
from swarm_arena._model.tooling import (
    CompletedRun,
    FileRef,
    RunStatus,
    ToolOutput,
    ToolRequest,
    Usage,
)
from swarm_arena._tooling.paths import check_relpath

_RESULT = "swarm/result.json"


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


def read_regular_file(workspace: Path, rel: str) -> bytes:
    """
    Read a regular workspace file without following any symlink component.

    Parameters
    ----------
    workspace : Path
        Trusted workspace root.
    rel : str
        Relative path, including internal paths such as ``swarm/result.json``.

    Returns
    -------
    bytes
        File contents.

    Raises
    ------
    ValueError
        If the path escapes the workspace, is missing, contains a symlink,
        cannot be read, or is not a regular file.

    Examples
    --------
    >>> data = read_regular_file(Path("workspace"), "out.txt")  # doctest: +SKIP
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
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError
                return stream.read()
    except (OSError, ValueError):
        raise ValueError(f"{rel} is not a regular file") from None


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
        If the file is missing, unsafe, unreadable, invalid JSON, not an object,
        or contains a non-finite number.

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
    raw = read_regular_file(workspace, _RESULT)
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
    """Keep schema field names and list indices; redact dynamic locations."""
    reasons: list[str] = []
    for error in exc.errors(include_input=False, include_url=False, include_context=False):
        loc = error["loc"]
        safe: list[str] = []
        for index, segment in enumerate(loc):
            if isinstance(segment, int):
                safe.append(str(segment))
            elif (index == 0 and segment in ToolResultFile.model_fields) or (
                index == 1 and loc[0] == "usage" and segment in Usage.model_fields
            ):
                safe.append(segment)
            else:
                safe.append("<key>")
        reasons.append(f"{'.'.join(safe) or 'schema_version'}: {error['type']}")
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
    files: list[FileRef] = []
    for index, path in enumerate(result.files):
        try:
            rel = check_relpath(path)
        except ValueError:
            raise ValueError(
                f"files.{index}: invalid path (absolute, parent, empty, NUL, or reserved)"
            ) from None
        data = read_regular_file(workspace, rel)
        files.append(
            FileRef(name=PurePosixPath(rel).name, path=rel, sha256=sha256_hex(data), size=len(data))
        )
    return result, files


def tool_output_from_result(workspace: Path, run: CompletedRun) -> ToolOutput:
    """
    Collect a tool envelope according to process status and artifact validity.

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
        provider failures. An invalid successful result becomes an error.

    Examples
    --------
    >>> output = tool_output_from_result(Path("workspace"), run)  # doctest: +SKIP
    """
    status = run.status
    try:
        result, files = _load_result(workspace)
    except ValueError as exc:
        if status == RunStatus.OK:
            status = RunStatus.ERROR
            note_in_log(run.stderr_path, str(exc))
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


def note_in_log(path: Path, message: str) -> None:
    """
    Append a diagnostic to a regular log file, dropping unsafe or failed writes.

    Parameters
    ----------
    path : Path
        Log path. A missing leaf is created. Symlinks in any component,
        hardlinked files, and special files are refused. Directory descriptors
        pin the parent walk so replacing a component cannot redirect the write.
    message : str
        Diagnostic without the prefix or trailing newline. Nonprintable
        characters, including controls and line separators, become spaces.

    Examples
    --------
    >>> note_in_log(Path("stderr.log"), "result is missing")  # doctest: +SKIP
    """
    fds: list[int] = []
    try:
        parts = path.parts[1:] if path.anchor else path.parts
        if not parts:
            return
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
        directory = os.open(path.anchor or ".", directory_flags)
        fds.append(directory)
        for part in parts[:-1]:
            directory = os.open(part, directory_flags, dir_fd=directory)
            fds.append(directory)
        fd = os.open(
            parts[-1],
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o644,
            dir_fd=directory,
        )
        fds.append(fd)
        info = os.fstat(fd)
        if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            sanitized = "".join(char if char.isprintable() else " " for char in message)
            os.write(fd, f"swarm-arena: {sanitized}\n".encode())
    except (OSError, ValueError):
        pass
    finally:
        for fd in reversed(fds):
            with contextlib.suppress(OSError):
                os.close(fd)
