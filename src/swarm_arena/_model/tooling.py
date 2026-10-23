"""Define tool requests, outputs, usage, and command execution data."""

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final

from pydantic import AfterValidator, Field, PrivateAttr, TypeAdapter, field_serializer

from swarm_arena._model.base import FrozenModel
from swarm_arena._model.canonical import UtcDatetime, sha256_hex
from swarm_arena._model.data import FiniteJsonObject, Manifest, Sha256, TomlJsonObject
from swarm_arena._tooling.paths import check_key, check_relpath

_Sha256 = Sha256
TimeoutS = Annotated[float, Field(gt=0)]
GraceS = Annotated[float, Field(ge=0)]
EnvName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
DEFAULT_GRACE_S: Final = 10.0
_Relpath = Annotated[str, AfterValidator(check_relpath)]


def check_parent_id(candidate_id: str) -> str:
    """
    Validate a parent identifier as one workspace path component.

    Parameters
    ----------
    candidate_id : str
        Identifier used for the parent's staged directory.

    Returns
    -------
    str
        Validated, unchanged identifier.

    Raises
    ------
    ValueError
        If the identifier is not one permitted path component.

    Examples
    --------
    >>> from swarm_arena._model.tooling import check_parent_id
    >>> check_parent_id("c1")
    'c1'
    """
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate_id) is None:
        raise ValueError(f"parent id {candidate_id!r}: not one path component")
    return candidate_id


def is_stored_edit_ref(value: str) -> bool:
    """
    Return whether a value has the exact reserved stored-reference form.

    Parameters
    ----------
    value : str
        Edit value to examine without interpreting it as live content.

    Returns
    -------
    bool
        Whether the value is ``sha256:`` followed by 64 lowercase hex digits.

    Examples
    --------
    >>> from swarm_arena._model.tooling import is_stored_edit_ref
    >>> is_stored_edit_ref("sha256:" + "a" * 64)
    True
    """
    return re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


class RunStatus(StrEnum):
    """
    Describe the execution outcome of a tool run.

    Examples
    --------
    >>> from swarm_arena.tool import RunStatus
    >>> RunStatus.OK.value
    'ok'
    """

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"
    PROVIDER_FAILED = "provider_failed"
    LOST = "lost"


class Usage(FrozenModel):
    """
    Record token consumption and optional monetary cost.

    ``input_tokens`` counts input tokens neither read from nor written to the prompt cache.

    Parameters
    ----------
    input_tokens : int, default=0
        Uncached input tokens, excluding cache reads and writes.
    output_tokens : int, default=0
        Generated output tokens.
    cache_read_tokens : int, default=0
        Input tokens read from the prompt cache.
    cache_write_tokens : int, default=0
        Input tokens written to the prompt cache, including both lifetime splits.
    cache_write_5m_tokens : int, default=0
        Five-minute subset of cache-write tokens.
    cache_write_1h_tokens : int, default=0
        One-hour subset of cache-write tokens.
    usd : float or None, default=None
        Reported dollar cost, or None when unknown.

    Examples
    --------
    >>> from swarm_arena.tool import Usage
    >>> Usage(input_tokens=10, output_tokens=5).total_tokens
    15
    """

    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0
    cache_read_tokens: Annotated[int, Field(ge=0)] = 0
    cache_write_tokens: Annotated[int, Field(ge=0)] = 0
    cache_write_5m_tokens: Annotated[int, Field(ge=0)] = 0
    cache_write_1h_tokens: Annotated[int, Field(ge=0)] = 0
    usd: Annotated[float, Field(ge=0)] | None = None

    @property
    def total_tokens(self) -> int:
        """
        Sum token consumption without counting cache-write subsets twice.

        Returns
        -------
        int
            Input, output, cache-read, and total cache-write tokens.

        Examples
        --------
        >>> from swarm_arena.tool import Usage
        >>> Usage(cache_write_tokens=5, cache_write_5m_tokens=5).total_tokens
        5
        """
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: "Usage") -> "Usage":
        """
        Add corresponding counters and all known costs.

        Parameters
        ----------
        other : Usage
            Usage to combine with this value.

        Returns
        -------
        Usage
            Summed counters, with unknown cost only when both costs are unknown.

        Examples
        --------
        >>> from swarm_arena.tool import Usage
        >>> (Usage(input_tokens=2) + Usage(input_tokens=3)).input_tokens
        5
        """
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_write_5m_tokens=self.cache_write_5m_tokens + other.cache_write_5m_tokens,
            cache_write_1h_tokens=self.cache_write_1h_tokens + other.cache_write_1h_tokens,
            usd=None
            if self.usd is None and other.usd is None
            else (self.usd or 0.0) + (other.usd or 0.0),
        )


class Estimate(FrozenModel):
    """
    Describe an estimated run cost or budget amount.

    Parameters
    ----------
    seconds : float or None, default=None
        Estimated execution time in seconds, if known.
    tokens : int or None, default=None
        Estimated token consumption, if known.
    usd : float or None, default=None
        Estimated dollar cost, if known.

    Examples
    --------
    >>> from swarm_arena.tool import Estimate
    >>> Estimate(seconds=2.0).tokens is None
    True
    """

    seconds: Annotated[float, Field(ge=0)] | None = None
    tokens: Annotated[int, Field(ge=0)] | None = None
    usd: Annotated[float, Field(ge=0)] | None = None


class FileRef(FrozenModel):
    """
    Describe a content-addressed output file.

    Parameters
    ----------
    name : str
        Logical artifact name.
    path : str
        Workspace-relative path, normalized during validation.
    sha256 : str
        Lowercase SHA-256 digest of the file bytes.
    size : int
        File size in bytes, at least zero.

    Examples
    --------
    >>> from swarm_arena.tool import FileRef
    >>> FileRef(name="result", path="./result.txt", sha256="a" * 64, size=0).path
    'result.txt'
    """

    name: str
    path: _Relpath
    sha256: _Sha256
    size: Annotated[int, Field(ge=0)]


class ParentRef(FrozenModel):
    """
    Describe a parent candidate staged under its own directory.

    Parameters
    ----------
    candidate_id : str
        Parent identifier consisting of one safe path component.
    run_id : str
        Run that produced the parent.
    tool_identity : str
        Identity of the tool that produced the parent.
    data : dict[str, Any], default={}
        Parent's structured output.
    overlay : dict[str, str], default={}
        Full effective overlay, mapping normalized paths to SHA-256 digests.
    files : list[FileRef], default=[]
        Parent output artifacts.

    Examples
    --------
    >>> from swarm_arena.tool import ParentRef
    >>> ParentRef(candidate_id="c1", run_id="r", tool_identity="T").overlay
    {}
    """

    candidate_id: Annotated[str, AfterValidator(check_parent_id)]
    run_id: str
    tool_identity: str
    data: FiniteJsonObject = {}
    overlay: Manifest = {}
    files: list[FileRef] = []


class ToolRequest(FrozenModel):
    """
    Describe a tool invocation and serialize edited contents as hashes.

    ``sha256:`` + 64 lowercase hex is the reserved stored-reference form.
    In any ToolRequest it means a reference (as ``swarm/`` is a reserved path),
    so it is dumped unchanged by definition. Live-content entry points
    (``validate_edits``, ``edit_hashes``, ``put_edits``, and so
    ``testing.make_candidate``) refuse it as content.

    Parameters
    ----------
    inputs : dict[str, Any], default={}
        Structured inputs to validate against the tool's input model.
    parents : list[ParentRef], default=[]
        Declared parent candidates.
    replicate : int, default=0
        Replicate index for this request.
    round : int, default=0
        Round index recorded in the staged request.
    round_overrides : dict[str, Any], default={}
        Tool configuration overrides for this round.
    edits : dict[str, str], default={}
        Workspace paths mapped to live contents or reserved stored references.
        Each key must pass ``check_key``: a relative path in normal form,
        strict UTF-8, with no ``..``, NUL, or reserved top-level name.

    Examples
    --------
    >>> from swarm_arena.tool import ToolRequest
    >>> request = ToolRequest(edits={"a.txt": "hi"})
    >>> dumped = request.model_dump(mode="json")
    >>> dumped["edits"]["a.txt"].startswith("sha256:")
    True
    >>> ToolRequest.model_validate(dumped).model_dump(mode="json") == dumped
    True
    """

    inputs: FiniteJsonObject = {}
    parents: list[ParentRef] = []
    replicate: int = 0
    round: int = 0
    round_overrides: TomlJsonObject = {}
    edits: dict[Annotated[str, AfterValidator(check_key)], str] = {}

    @field_serializer("edits")
    def _serialize_edits(self, edits: dict[str, str]) -> dict[str, str]:
        """Hash live contents and preserve only exact stored references in all modes."""
        return {
            path: content
            if is_stored_edit_ref(content)
            else "sha256:" + sha256_hex(content.encode())
            for path, content in edits.items()
        }


def format_logs_ref(stdout_sha: str, stderr_sha: str) -> str:
    """
    Format the ordered stdout and stderr artifact digests.

    Parameters
    ----------
    stdout_sha : str
        Lowercase SHA-256 digest of stdout.
    stderr_sha : str
        Lowercase SHA-256 digest of stderr.

    Returns
    -------
    str
        Reference carrying both validated digests.

    Raises
    ------
    ValueError
        If either digest is invalid, without echoing its value.

    Examples
    --------
    >>> parse_logs_ref(format_logs_ref("a" * 64, "b" * 64)) == ("a" * 64, "b" * 64)
    True
    """
    try:
        adapter = TypeAdapter(Sha256)
        adapter.validate_python(stdout_sha, strict=True)
        adapter.validate_python(stderr_sha, strict=True)
    except Exception:
        raise ValueError("invalid logs_ref") from None
    return f"sha256:{stdout_sha}:{stderr_sha}"


def parse_logs_ref(ref: str) -> tuple[str, str]:
    """
    Parse an ordered stdout and stderr artifact reference.

    Parameters
    ----------
    ref : str
        Reference returned by ``format_logs_ref``.

    Returns
    -------
    tuple[str, str]
        Stdout digest followed by stderr digest.

    Raises
    ------
    ValueError
        If the reference is malformed, without echoing its value.

    Examples
    --------
    >>> parse_logs_ref("sha256:bad:ref")
    Traceback (most recent call last):
    ...
    ValueError: invalid logs_ref
    """
    if not isinstance(ref, str):
        raise ValueError("invalid logs_ref")
    parts = ref.split(":")
    if len(parts) != 3 or parts[0] != "sha256":
        raise ValueError("invalid logs_ref")
    format_logs_ref(parts[1], parts[2])
    return parts[1], parts[2]


class ToolOutput(FrozenModel):
    """
    Record a tool run's outcome and collected results.

    Parameters
    ----------
    status : RunStatus
        Execution outcome, independent of result quality.
    partial : bool, default=False
        Whether the output is an incomplete result.
    data : dict[str, Any], default={}
        Structured result data containing only finite JSON; NaN, infinity,
        circular structures and non-serializable objects are rejected.
    files : list[FileRef], default=[]
        Collected output artifacts.
    usage : Usage, default=Usage()
        Token consumption and reported cost.
    logs_ref : str or None, default=None
        Transport log hashes or the run-relative log reference.
    llm_calls_file : FileRef or None, default=None
        Optional cooperating-tool report, collected from swarm/llm_calls.jsonl.
        Files over 16 MiB and at most 1 GiB are fully hashed but not uploaded.
    started_at : datetime or None, default=None
        Aware start time, normalized to UTC milliseconds.
    ended_at : datetime or None, default=None
        Aware completion time, normalized to UTC milliseconds.
    failed_provider : str or None, default=None
        Failed provider name; None leaves the provider unspecified.

    Raises
    ------
    ValidationError
        If ``data`` cannot be encoded as finite JSON, or another field is invalid.

    Examples
    --------
    >>> from swarm_arena.tool import RunStatus, ToolOutput
    >>> ToolOutput(status=RunStatus.OK).partial
    False
    >>> ToolOutput(status=RunStatus.OK, data={"count": 2}).data
    {'count': 2}
    """

    status: RunStatus
    partial: bool = False
    data: FiniteJsonObject = {}
    files: list[FileRef] = []
    usage: Usage = Usage()
    logs_ref: str | None = None
    llm_calls_file: FileRef | None = None
    _collection_diagnostics: tuple[tuple[str, int], ...] = PrivateAttr(default=())
    started_at: UtcDatetime | None = None
    ended_at: UtcDatetime | None = None
    failed_provider: str | None = None

    def __eq__(self, other: object) -> bool:
        """Compare output values independently of transient collection diagnostics."""
        if not isinstance(other, ToolOutput):
            return super().__eq__(other)
        left, right = self.model_copy(), other.model_copy()
        for output in (left, right):
            output.__pydantic_private__ = {
                key: value
                for key, value in (output.__pydantic_private__ or {}).items()
                if key != "_collection_diagnostics"
            }
        return FrozenModel.__eq__(left, right)


class Command(FrozenModel):
    """
    Describe a command to execute inside a staged workspace.

    Parameters
    ----------
    argv : list[str]
        Non-empty argument vector.
    cwd : str or None, default=None
        Relative working directory, or None for the workspace root.
    env : list[str], default=[]
        Allowlisted environment variable names; values come from the node.
    timeout_s : float or None, default=None
        Positive execution timeout in seconds, if set.
    python : str or None, default=None
        Interpreter replacing a leading ``python`` argument.
    grace_s : float, default=10.0
        Nonnegative shutdown grace period in seconds.

    Examples
    --------
    >>> from swarm_arena.tool import Command
    >>> Command(argv=["python", "run.py"]).grace_s
    10.0
    """

    argv: Annotated[list[str], Field(min_length=1)]
    cwd: _Relpath | None = None
    env: list[EnvName] = []
    timeout_s: TimeoutS | None = None
    python: str | None = None
    grace_s: GraceS = DEFAULT_GRACE_S


class CompletedRun(FrozenModel):
    """
    Record a completed process and its local log files.

    Parameters
    ----------
    returncode : int or None
        Process return code, or None when unavailable.
    status : RunStatus
        Execution outcome from the process supervisor.
    stdout_path : Path
        Local stdout log path.
    stderr_path : Path
        Local stderr log path.
    started_at : datetime
        Aware start time, normalized to UTC milliseconds.
    ended_at : datetime
        Aware completion time, normalized to UTC milliseconds.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> from pathlib import Path
    >>> from swarm_arena.tool import CompletedRun, RunStatus
    >>> now = datetime(2026, 10, 18, tzinfo=UTC)
    >>> run = CompletedRun(returncode=0, status=RunStatus.OK,
    ...     stdout_path=Path("stdout"), stderr_path=Path("stderr"),
    ...     started_at=now, ended_at=now)
    >>> run.returncode
    0
    """

    returncode: int | None
    status: RunStatus
    stdout_path: Path
    stderr_path: Path
    started_at: UtcDatetime
    ended_at: UtcDatetime
