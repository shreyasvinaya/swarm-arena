"""Collect and record bounded informational LLM reports from cooperating tools."""

import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Literal, cast

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, model_validator

from swarm_arena._errors import StorageError
from swarm_arena._journal.llm_record import record_call
from swarm_arena._model.base import JsonValue, StoredModel
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.data import finite_json
from swarm_arena._model.results import LLMCall
from swarm_arena._model.tooling import FileRef, Usage
from swarm_arena._outcomes import _propagate_control
from swarm_arena._store.artifacts import ArtifactStore, _read_bounded_body
from swarm_arena._store.local import LocalStore
from swarm_arena._store.s3 import S3Store
from swarm_arena._tooling.result_file import (
    _chunks,
    _inspect_path,
    open_regular_file,
    read_regular_file,
)

LLM_CALLS_PATH: Final = "swarm/llm_calls.jsonl"
LLM_CALLS_REF_PATH: Final = "llm_calls.jsonl"
MAX_LLM_CALLS_BYTES: Final = 16 * 1024 * 1024
MAX_LLM_CALL_LINE_BYTES: Final = 64 * 1024
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


def _identifier(value: str) -> str:
    """Reject control characters in tool-supplied identifiers."""
    if value and not value.isprintable():
        raise ValueError("invalid report identifier")
    return value


def _reported_call_id(value: str) -> str:
    """Keep tool content out of the core gateway observation namespace."""
    if value.startswith("gw:"):
        raise ValueError("reserved report identifier")
    return _identifier(value)


def _reported_usage(value: Any) -> Usage:
    """Validate nested usage without coercing booleans or numeric strings."""
    return Usage.model_validate(value, strict=True)


class LLMCallRow(StoredModel):
    """Validate one informational report, dropping unrecognized fields.

    Parameters
    ----------
    model : str
        Printable model identifier, at most 256 characters. Its prefix supplies
        a missing provider.
    provider : str or None, default=None
        Printable identifier, at most 64 characters; an unprefixed model defaults
        to ``unknown``.
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
        Optional printable tool-observation identity, at most 128 characters.
        The ``gw:`` prefix is reserved for core gateway observations and refused.
        Deduplication is separate from funded component calls.
    request, response : JsonValue, default=None
        Optional finite JSON payloads, stored only with tracing consent.

    Examples
    --------
    >>> LLMCallRow(model="openai/gpt-6").provider
    'openai'
    """

    schema_kind: ClassVar[str] = "llm_call_row"
    model_config = ConfigDict(extra="ignore", strict=True)
    model: Annotated[str, Field(max_length=256), AfterValidator(_identifier)]
    provider: Annotated[str, Field(max_length=64), AfterValidator(_identifier)] | None = None
    usage: Annotated[Usage, BeforeValidator(_reported_usage)] = Usage()
    cost_usd: Annotated[float, Field(ge=0, strict=True)] = 0.0
    latency_ms: Annotated[float, Field(ge=0, strict=True)] | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    litellm_call_id: (
        Annotated[str, Field(max_length=128), AfterValidator(_reported_call_id)] | None
    ) = None
    request: Annotated[JsonValue, AfterValidator(finite_json)] = None
    response: Annotated[JsonValue, AfterValidator(finite_json)] = None

    @model_validator(mode="after")
    def _provider(self) -> "LLMCallRow":
        """Fill a missing provider only after the report fields validate."""
        if self.provider is None:
            provider = self.model.split("/", 1)[0] if "/" in self.model else "unknown"
            if len(provider) > 64:
                raise ValueError("invalid report provider")
            object.__setattr__(self, "provider", provider)
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


def parse_llm_calls(data: bytes) -> tuple[Iterator[LLMCallRow], list[str]]:
    """Iterate bounded JSONL without retaining decoded rows or payloads.

    Parameters
    ----------
    data : bytes
        UTF-8 report, at most 16 MiB; each line is at most 64 KiB.

    Returns
    -------
    tuple[Iterator[LLMCallRow], list[str]]
        Single-pass lazy rows and a bounded warning list populated as consumed.
        Advance the iterator in a supervised worker, then record each row before
        advancing again. Do not accumulate decoded payloads in the driver.

    Examples
    --------
    >>> rows, warnings = parse_llm_calls(b'{"model":"local"}')
    >>> [row.provider for row in rows], warnings
    (['unknown'], [])
    """
    if len(data) > MAX_LLM_CALLS_BYTES:
        return iter(()), [oversize_warning(len(data))]
    warnings: list[str] = []
    return _rows(data, warnings), warnings


def _rows(data: bytes, warnings: list[str]) -> Iterator[LLMCallRow]:
    """Slice one bounded line at a time, never splitting the whole report."""
    seen = bad = number = offset = 0
    while offset < len(data):
        end = data.find(b"\n", offset)
        if end < 0:
            end = len(data)
        number += 1
        size = end - offset
        line = data[offset:end] if size <= MAX_LLM_CALL_LINE_BYTES else None
        offset = end + 1
        if line is not None and not line.strip():
            continue
        if seen == MAX_LLM_CALL_ROWS:
            warnings.append(f"{LLM_CALLS_PATH} has more than 10000 rows; rows after 10000 skipped")
            break
        seen += 1
        reason = "invalid row" if line is None else "invalid JSON"
        try:
            if line is None:
                raise ValueError("report line too long")
            raw = json.loads(line.decode("utf-8"))
            reason = "invalid row"
            row = LLMCallRow.model_validate(raw)
            del raw
        except Exception as error:
            _propagate_control(error)
            bad += 1
            if bad <= MAX_LLM_CALL_WARNINGS:
                warnings.append(f"{LLM_CALLS_PATH} line {number}: {reason}; line skipped")
        else:
            yield row
            del row
    if bad > MAX_LLM_CALL_WARNINGS:
        warnings.append(f"{LLM_CALLS_PATH}: {bad - MAX_LLM_CALL_WARNINGS} more bad lines skipped")


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
            for chunk in _chunks(fd, rel, max_bytes=before.st_size, step=1024 * 1024):
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


def _read_report_artifact(artifacts: ArtifactStore, ref: FileRef) -> bytes:
    """Read and verify one report under its declared size and the 16 MiB ceiling.

    Local reads are pinned and bounded before allocation; S3 checks HEAD and a
    bounded GET consumed through EOF or one excess byte to detect growth.
    Generic byte-only stores have
    no pre-read bound in their API, so their returned object is checked before
    copying, hashing or parsing. Call only from the owner's storage worker.
    """
    if ref.size > MAX_LLM_CALLS_BYTES:
        raise StorageError("report artifact crosses the size cap")
    store = artifacts._store
    if isinstance(store, LocalStore):
        data = read_regular_file(store._root, ref.sha256, max_bytes=ref.size)
    elif isinstance(store, S3Store):
        from fsspec.asyn import sync

        key = store._key(ref.sha256)
        with store._reading("head_object"):
            head = store._call("head_object", Bucket=store._bucket, Key=key)
        if head["ContentLength"] != ref.size:
            raise StorageError("report artifact size mismatch")
        with store._reading("get_object"):
            response = store._call(
                "get_object",
                Bucket=store._bucket,
                Key=key,
                **({"Range": f"bytes=0-{ref.size}"} if ref.size else {}),
            )

            async def consume() -> bytes:
                async with response["Body"] as body:
                    return await _read_bounded_body(body, ref.size)

            data = cast(bytes, sync(store._filesystem().loop, consume))
    else:
        data = store.read(ref.sha256).data
    if not isinstance(data, bytes):
        raise TypeError("invalid report artifact bytes")
    if memoryview(data).nbytes != ref.size:
        raise StorageError("report artifact size mismatch")
    data = data if type(data) is bytes else memoryview(data).tobytes()
    if sha256_hex(data) != ref.sha256:
        raise StorageError("report artifact hash mismatch")
    return data


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
    rows: Sequence[LLMCallRow], *, metadata: Mapping[str, Any], call_ids: list[str] | None = None
) -> list[LLMCall]:
    """Record external observations without funding or adopting a current scope.

    Parameters
    ----------
    rows : Sequence[LLMCallRow]
        Validated tool observations; no row supplies core routing authority.
    metadata : Mapping[str, Any]
        Trusted core run/goal/team/agent/round/candidate fields and current writer
        owner_epoch. Provider reservation and call identities are never forwarded.

    call_ids : list[str] or None, default=None
        Caller-owned links appended immediately after each admitted call, so
        cancellation preserves the already journaled prefix.

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
            if call_ids is not None:
                call_ids.append(call.id)
            calls.append(call)
    return calls
