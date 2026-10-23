"""Collect and record bounded informational LLM reports from cooperating tools."""

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Literal

from pydantic import AfterValidator, ConfigDict, Field, model_validator

from swarm_arena._journal.llm_record import record_call
from swarm_arena._model.base import JsonValue, StoredModel
from swarm_arena._model.data import finite_json
from swarm_arena._model.results import LLMCall
from swarm_arena._model.tooling import FileRef, Usage
from swarm_arena._outcomes import _propagate_control
from swarm_arena._tooling.result_file import _chunks, _inspect_path, open_regular_file

LLM_CALLS_PATH: Final = "swarm/llm_calls.jsonl"
LLM_CALLS_REF_PATH: Final = "llm_calls.jsonl"
MAX_LLM_CALLS_BYTES: Final = 16 * 1024 * 1024
MAX_LLM_CALLS_HASH_BYTES: Final = 1024 * 1024 * 1024
MAX_LLM_CALL_ROWS: Final = 10_000
MAX_LLM_CALL_WARNINGS: Final = 20
PROVIDER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "FIREWORKS_AI_API_KEY",
        "MOONSHOT_API_KEY",
        "OPENROUTER_API_KEY",
        "MISTRAL_API_KEY",
        "COHERE_API_KEY",
        "GROQ_API_KEY",
        "XAI_API_KEY",
        "TOGETHERAI_API_KEY",
        "AZURE_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    }
)


class LLMCallRow(StoredModel):
    """Validate one informational report, dropping unrecognized fields.

    Parameters
    ----------
    model : str
        Actual model identifier. Its prefix supplies a missing provider.
    provider : str or None, default=None
        Provider identifier; an unprefixed model defaults to ``unknown``.
    usage : Usage, default=Usage()
        Nonnegative token counts and optional reported usage cost.
    cost_usd : float, default=0
        Finite nonnegative observed cost; never independently settled.
    latency_ms : float or None, default=None
        Finite nonnegative latency.
    status : {'ok', 'error'}, default='ok'
        Reported outcome.
    error : str or None, default=None
        Input diagnostic; the recorder retains only a fixed error description.
    litellm_call_id : str or None, default=None
        Optional run-wide deduplication identity.
    request, response : JsonValue, default=None
        Optional finite JSON payloads, stored only with tracing consent.

    Examples
    --------
    >>> LLMCallRow(model="openai/gpt-6").provider
    'openai'
    """

    schema_kind: ClassVar[str] = "llm_call_row"
    model_config = ConfigDict(extra="ignore")
    model: str
    provider: str | None = None
    usage: Usage = Usage()
    cost_usd: Annotated[float, Field(ge=0)] = 0.0
    latency_ms: Annotated[float, Field(ge=0)] | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    litellm_call_id: str | None = None
    request: Annotated[JsonValue, AfterValidator(finite_json)] = None
    response: Annotated[JsonValue, AfterValidator(finite_json)] = None

    @model_validator(mode="after")
    def _provider(self) -> "LLMCallRow":
        """Fill a missing provider only after the report fields validate."""
        if self.provider is None:
            object.__setattr__(
                self, "provider", self.model.split("/", 1)[0] if "/" in self.model else "unknown"
            )
        return self


def oversize_warning(size: int) -> str:
    """Describe a report that exceeds the parse/upload limit.

    Parameters
    ----------
    size : int
        Verified file size in bytes.

    Returns
    -------
    str
        Fixed diagnostic with a numeric byte count.

    Examples
    --------
    >>> oversize_warning(17825792)
    'swarm/llm_calls.jsonl is 17825792 bytes, over the 16 MiB limit; not collected'
    """
    return f"{LLM_CALLS_PATH} is {size} bytes, over the 16 MiB limit; not collected"


def parse_llm_calls(data: bytes) -> tuple[list[LLMCallRow], list[str]]:
    """Parse bounded JSONL, skipping invalid rows with value-free diagnostics.

    Parameters
    ----------
    data : bytes
        UTF-8 report; at most 16 MiB is parsed.

    Returns
    -------
    tuple[list[LLMCallRow], list[str]]
        Valid rows and bounded warnings; blank lines do not consume the row cap.

    Examples
    --------
    >>> rows, warnings = parse_llm_calls(b'{"model":"local"}')
    >>> rows[0].provider, warnings
    ('unknown', [])
    """
    if len(data) > MAX_LLM_CALLS_BYTES:
        return [], [oversize_warning(len(data))]
    rows: list[LLMCallRow] = []
    warnings: list[str] = []
    seen = bad = 0
    for number, line in enumerate(data.split(b"\n"), 1):
        if not line.strip():
            continue
        if seen == MAX_LLM_CALL_ROWS:
            warnings.append(f"{LLM_CALLS_PATH} has more than 10000 rows; rows after 10000 skipped")
            break
        seen += 1
        reason = "invalid JSON"
        try:
            raw = json.loads(line.decode("utf-8"))
            reason = "invalid row"
            row = LLMCallRow.model_validate(raw)
        except Exception as error:
            _propagate_control(error)
            bad += 1
            if bad <= MAX_LLM_CALL_WARNINGS:
                warnings.append(f"{LLM_CALLS_PATH} line {number}: {reason}; line skipped")
        else:
            rows.append(row)
    if bad > MAX_LLM_CALL_WARNINGS:
        warnings.append(f"{LLM_CALLS_PATH}: {bad - MAX_LLM_CALL_WARNINGS} more bad lines skipped")
    return rows, warnings


class _ReportFileError(ValueError):
    """Carry only one core-defined optional-file refusal code."""

    def __init__(self, code: str) -> None:
        super().__init__("LLM report file refused")
        self.code = code


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    """Compare file identity, size and mutation timestamps across the pinned read."""
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_report(workspace: Path, rel: str, *, retain: bool) -> tuple[str, int, bytes | None]:
    """Hash at most 1 GiB, retaining at most 16 MiB on the same pinned descriptor."""
    fd = open_regular_file(workspace, rel)
    try:
        before = os.fstat(fd)
        if before.st_size > MAX_LLM_CALLS_HASH_BYTES:
            raise _ReportFileError("llm_calls_hash_limit")
        keep = retain and before.st_size <= MAX_LLM_CALLS_BYTES
        data = bytearray()
        digest = hashlib.sha256()
        total = 0
        try:
            for chunk in _chunks(fd, rel, max_bytes=MAX_LLM_CALLS_HASH_BYTES, step=1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
                if keep and total <= MAX_LLM_CALLS_BYTES:
                    data.extend(chunk)
                else:
                    keep = False
                    data.clear()
        except ValueError as error:
            _propagate_control(error)
            raise _ReportFileError("llm_calls_changed") from None
        if _stamp(before) != _stamp(os.fstat(fd)) or total != before.st_size:
            raise _ReportFileError("llm_calls_changed")
        return digest.hexdigest(), total, bytes(data) if keep else None
    finally:
        os.close(fd)


def hash_regular_file(workspace: Path, rel: str) -> tuple[str, int]:
    """Hash an unchanged pinned regular file with a finite full-read ceiling.

    Parameters
    ----------
    workspace : Path
        Trusted workspace root.
    rel : str
        Workspace-relative file; no symlink component is permitted.

    Returns
    -------
    tuple[str, int]
        Full SHA-256 and fstat size, computed in at most 1 MiB chunks.

    Raises
    ------
    ValueError
        If unsafe, changed, unreadable or larger than 1 GiB. No prefix hash returns.

    Examples
    --------
    >>> digest, size = hash_regular_file(workspace, LLM_CALLS_PATH)  # doctest: +SKIP
    """
    digest, size, _ = _read_report(workspace, rel, retain=False)
    return digest, size


def collect_llm_calls(
    workspace: Path,
) -> tuple[FileRef | None, bytes | None, tuple[tuple[str, int], ...]]:
    """Read an optional report without publishing or hiding backend failures.

    Parameters
    ----------
    workspace : Path
        Tool workspace, retained for the duration of its supervised read.

    Returns
    -------
    tuple[FileRef or None, bytes or None, tuple[tuple[str, int], ...]]
        Real full hash/size, uploadable bytes, and private refusal diagnostics.
        Missing files return three empty values; files over 16 MiB have no bytes.

    Examples
    --------
    >>> ref, data, diagnostics = collect_llm_calls(workspace)  # doctest: +SKIP
    """
    try:
        link, info = _inspect_path(workspace, LLM_CALLS_PATH)
        if link is not None:
            raise _ReportFileError("llm_calls_symlink")
        if info is None:
            return None, None, ()
        if not stat.S_ISREG(info.st_mode):
            raise _ReportFileError("llm_calls_special_file")
        digest, size, data = _read_report(workspace, LLM_CALLS_PATH, retain=True)
    except BaseException as error:
        _propagate_control(error)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        code = error.code if type(error) is _ReportFileError else "llm_calls_unreadable"
        return None, None, ((code, 1),)
    return (
        FileRef(name=LLM_CALLS_REF_PATH, path=LLM_CALLS_REF_PATH, sha256=digest, size=size),
        data,
        (),
    )


def may_call_llm(providers: frozenset[str], secret_env: Sequence[str]) -> bool:
    """Recognize declared providers or known provider credential names.

    Parameters
    ----------
    providers : frozenset[str]
        Providers declared by the effective tool.
    secret_env : Sequence[str]
        Effective credential allowlist names, never values.

    Returns
    -------
    bool
        Whether the tool may call an LLM.

    Examples
    --------
    >>> may_call_llm(frozenset(), ["OPENAI_API_KEY"])
    True
    """
    return bool(providers or PROVIDER_KEYS.intersection(secret_env))


def llm_calls_source(
    *, keyed: bool, reported_rows: int, may_call: bool
) -> Literal["gateway", "reported", "none", "unknown"]:
    """Choose the observed call source in gateway, report, unknown, none order.

    Parameters
    ----------
    keyed : bool
        Whether the core keyed this run through the gateway.
    reported_rows : int
        Number of valid reported rows, including already deduplicated rows.
    may_call : bool
        Whether effective providers or credentials indicate possible LLM use.

    Returns
    -------
    {'gateway', 'reported', 'none', 'unknown'}
        Evidence source; the core cannot infer tool-owned Langfuse traces.

    Examples
    --------
    >>> llm_calls_source(keyed=False, reported_rows=0, may_call=True)
    'unknown'
    """
    return (
        "gateway"
        if keyed
        else "reported"
        if reported_rows > 0
        else "unknown"
        if may_call
        else "none"
    )


async def record_reported(
    rows: Sequence[LLMCallRow], *, metadata: Mapping[str, Any]
) -> list[LLMCall]:
    """Record external observations without funding or adopting a current scope.

    Parameters
    ----------
    rows : Sequence[LLMCallRow]
        Validated tool observations; no row supplies core routing authority.
    metadata : Mapping[str, Any]
        Trusted core run/goal/team/agent/round/candidate fields and current writer
        owner_epoch. Provider reservation and call identities are never forwarded.

    Returns
    -------
    list[LLMCall]
        Newly recorded calls, excluding duplicates and rejected late callbacks.

    Examples
    --------
    >>> import asyncio
    >>> asyncio.run(record_reported([], metadata={}))
    []
    """
    meta = {
        key: metadata[key]
        for key in ("run_id", "goal_id", "team", "agent", "round", "candidate_id", "owner_epoch")
        if key in metadata
    }
    meta["role"] = "tool"
    calls: list[LLMCall] = []
    for row in rows:
        call = await record_call(
            metadata=meta,
            model=row.model,
            provider=row.provider or "unknown",
            usage=row.usage,
            cost_usd=row.cost_usd,
            cache_hit=False,
            latency_ms=row.latency_ms,
            status=row.status,
            error=row.error,
            request=row.request,
            response=row.response,
            litellm_call_id=row.litellm_call_id,
            tool_call_ids=(),
            settle=False,
            _external_observation=True,
        )
        if call is not None:
            calls.append(call)
    return calls
