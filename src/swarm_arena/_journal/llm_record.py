"""Record and settle calls within the lifetime of their entered run services.

The core knows no provider SDK. A cache identity is a SHA-256 of the adapter's
effective cache key, namespace and generating response identity, not a prompt
hash. An adapter without proof of that identity must omit it.
"""

import asyncio
import math
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, TypeVar

import anyio
from pydantic import JsonValue

from swarm_arena._agent.manager import LimitReached
from swarm_arena._budget.amount import USD_LIMIT, Amount
from swarm_arena._budget.llm_scope import LLMScope, current_llm_scope
from swarm_arena._engine.stops import RunClock, StopState
from swarm_arena._errors import Fenced, StorageError, _core_error, _core_signal, _stop_cleanup_error
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.a2a import agent_task_id
from swarm_arena._journal.event import EventDraft
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.detach import _detach_model, _plain
from swarm_arena._model.ids import IdGen
from swarm_arena._model.limits import Limits
from swarm_arena._model.results import LLMCall
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.settings import TracingConfig
from swarm_arena._model.tooling import Usage
from swarm_arena._outcomes import _has_core_fenced, _Outcome
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.io import _storage_io

_T = TypeVar("_T")
_NO_REPORT = "estimated: no LiteLLM record within settle_timeout_s; charged in full"
_UNKNOWN_SPEND = "estimated: provider spend unknown; charged in full"
_OVERFLOW = "usage_overflow: charged at the supported USD limit"


@dataclass
class _Pending:
    scope: LLMScope
    call_id: str
    reservation_id: str
    estimate: Amount
    model_alias: str
    model: str
    provider: str
    state: Literal["open", "claimed", "abandoned", "done"] = "open"
    event: anyio.Event = field(default_factory=anyio.Event)
    workers: set[anyio.CancelScope] = field(default_factory=set)
    journaled: bool = False
    actual: Amount | None = None
    overflow: bool = False
    settled: bool = False
    error: Exception | None = None


@dataclass
class _Binding:
    journal: JournalWriter
    ids: IdGen
    artifacts: ArtifactStore
    tracing: TracingConfig
    loop: asyncio.AbstractEventLoop
    stops: StopState
    clock: RunClock
    limits: Limits
    settle_timeout_s: float
    seen_litellm: dict[str, str] = field(default_factory=dict)
    recorded: set[str] = field(default_factory=set)
    originals: dict[str, str | None] = field(default_factory=dict)
    pending: dict[str, _Pending] = field(default_factory=dict)
    call_reservations: dict[str, str] = field(default_factory=dict)
    scopes: dict[int, LLMScope] = field(default_factory=dict)
    drained: set[int] = field(default_factory=set)
    finished_scopes: set[int] = field(default_factory=set)
    undrained_scopes: dict[int, int] = field(default_factory=dict)
    workers: set[anyio.CancelScope] = field(default_factory=set)
    active: bool = True
    closing: bool = False
    terminal: Exception | None = None
    overflow_notified: set[int] = field(default_factory=set)

    def track_scope(self, scope: LLMScope) -> None:
        """Count each funded scope once until all its accounting has drained."""
        if id(scope) not in self.scopes:
            self.scopes[id(scope)] = scope
            tranche = id(scope.tranche)
            self.undrained_scopes[tranche] = self.undrained_scopes.get(tranche, 0) + 1

    def finish_scope(self, scope: LLMScope) -> None:
        """Release one scope's ownership without closing its shared tranche."""
        if id(scope) not in self.finished_scopes:
            self.finished_scopes.add(id(scope))
            self.undrained_scopes[id(scope.tranche)] -= 1

    def check(self) -> None:
        """Preserve the writer's actual terminal cause, including cleanup expiry."""
        self.journal._raise_if_fenced()
        if self.terminal is not None:
            raise self.terminal from None

    def original(self, identity: str, call_id: str) -> None:
        """Keep ambiguous producer identities unlinked, regardless of callback order."""
        if identity not in self.originals:
            self.originals[identity] = call_id
        elif self.originals[identity] != call_id:
            self.originals[identity] = None


_bindings: dict[str, _Binding] = {}


def bind_run(
    run_id: str,
    *,
    journal: JournalWriter,
    ids: IdGen,
    artifacts: ArtifactStore,
    tracing: TracingConfig,
    committed: Sequence[LLMCall],
    request_sha256: Mapping[str, str],
    cache_identity: Mapping[str, str] | None = None,
    stops: StopState | None = None,
    clock: RunClock | None = None,
    limits: Limits | None = None,
    settle_timeout_s: float = 60,
) -> None:
    """Bind entered services and restore committed deduplication and cache indexes.

    Parameters
    ----------
    run_id : str
        Owned run identifier. An active binding cannot be replaced.
    journal : JournalWriter
        Entered writer on the current asyncio loop.
    ids : IdGen
        Run's shared identifier generator.
    artifacts : ArtifactStore
        Run payload store.
    tracing : TracingConfig
        Payload storage policy.
    committed : Sequence[LLMCall]
        Core-owned committed calls, in journal order.
    request_sha256 : Mapping[str, str]
        Replayed request digests; these do not establish cache provenance.
    cache_identity : Mapping[str, str] or None, optional
        Call ID to opaque generating-response digest, restored from replay.
    stops : StopState or None, optional
        The owner's live stop state. Standalone bindings get an empty state.
    clock : RunClock or None, optional
        The owner's existing active clock, never reset by fresh/resume owners.
    limits : Limits or None, optional
        The owner's run limits, or standalone defaults.
    settle_timeout_s : float, default=60
        Report and post-stop accounting allowance.

    Notes
    -----
    Engine/session owners supply their existing stops, clock, limits and timeout
    and use ``_bound_run`` for drain/close/unbind ordering. A low-level caller
    assumes that same cleanup responsibility. No registry entry is installed
    until every replay index has been validated.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> async def example() -> None:
    ...     bind_run("demo", journal=Mock(spec=JournalWriter), ids=IdGen("demo", 1),
    ...              artifacts=Mock(spec=ArtifactStore), tracing=TracingConfig(),
    ...              committed=[], request_sha256={})
    ...     unbind_run("demo")
    >>> anyio.run(example)
    """
    run_id = _context_string(run_id, "run_id")
    if run_id in _bindings:
        raise _core_error(ValueError, f"run {run_id} is already bound")
    if not math.isfinite(settle_timeout_s) or settle_timeout_s <= 0:
        raise _core_error(ValueError, "invalid settle_timeout_s")
    journal._check_open()
    binding = _Binding(
        journal,
        ids,
        artifacts,
        tracing,
        asyncio.get_running_loop(),
        stops if stops is not None else StopState(),
        clock if clock is not None else RunClock(0),
        limits if limits is not None else Limits(),
        settle_timeout_s,
    )
    identities = cache_identity or {}
    for call in committed:
        if call.run_id != run_id or call.id in binding.recorded:
            raise _core_error(ValueError, "invalid committed LLM call identity")
        binding.recorded.add(call.id)
        if call.reservation_id is not None:
            binding.call_reservations[call.id] = call.reservation_id
        if call.litellm_call_id is not None:
            binding.seen_litellm[call.litellm_call_id] = call.id
        _digest(request_sha256.get(call.id))
        identity = _digest(identities.get(call.id))
        if identity is not None and call.status == "ok" and not call.cache_hit:
            binding.original(identity, call.id)
    _bindings[run_id] = binding


def unbind_run(run_id: str) -> None:
    """Invalidate the binding and abandon callbacks before services close.

    Parameters
    ----------
    run_id : str
        Run to forget. Repeated calls are harmless.

    Examples
    --------
    >>> unbind_run("not-bound")
    """
    binding = _bindings.pop(_context_string(run_id, "run_id"), None)
    if binding is not None:
        binding.active = False
        for worker in tuple(binding.workers):
            worker.cancel()
        for entry in binding.pending.values():
            entry.event.set()


def open_call(
    scope: LLMScope,
    *,
    reservation_id: str,
    estimate: Amount,
    model_alias: str,
    model: str,
    provider: str,
) -> str:
    """Register an admitted reservation and allocate its stable call identifier.

    Parameters
    ----------
    scope : LLMScope
        Live component scope on the bound owner's loop.
    reservation_id : str
        Returned durable call reservation.
    estimate : Amount
        Full conservative charge when authoritative usage never arrives.
    model_alias : str
        Approved catalog alias.
    model : str
        Selected model identifier.
    provider : str
        Selected provider identifier.

    Returns
    -------
    str
        Preallocated call identifier, retained even by an estimated placeholder.

    Examples
    --------
    This runs after ``await scope.tranche.reserve_call(estimate)``. The
    returned identifier is included as ``metadata['call_id']`` in the request.
    >>> IdGen("demo", 1).next("llm").startswith("llm-")
    True
    """
    run_id = _context_string(scope.run_id, "run_id")
    binding = _bindings.get(run_id)
    if binding is None:
        raise _core_error(ValueError, f"run {run_id} is not bound")
    if asyncio.get_running_loop() is not binding.loop:
        raise _core_error(ValueError, "open_call requires the bound owner loop")
    binding.check()
    if binding.closing or id(scope) in binding.drained:
        raise _core_error(ValueError, "LLM scope is drained")
    if scope.journal is not binding.journal or scope.artifacts is not binding.artifacts:
        raise _core_error(ValueError, "LLM scope does not belong to the bound run")
    rid = _context_string(reservation_id, "reservation_id")
    alias = _context_string(model_alias, "model_alias")
    model = _context_string(model, "model")
    provider = _context_string(provider, "provider")
    if rid in binding.pending:
        raise _core_error(ValueError, "LLM reservation is already open")
    # Amount is core-owned at reservation admission; no model hooks run here.
    if type(estimate) is not Amount or estimate.tokens < 0 or estimate.usd < 0:
        raise _core_error(ValueError, "invalid call estimate")
    cid = binding.ids.next("llm")
    binding.pending[rid] = _Pending(scope, cid, rid, estimate, alias, model, provider)
    binding.call_reservations[cid] = rid
    binding.track_scope(scope)
    return cid


def pending_scope(metadata: Mapping[str, Any]) -> LLMScope | None:
    """Find a live reservation's scope without relying on a task-local context.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        Run and reservation identifiers from the adapter.

    Returns
    -------
    LLMScope or None
        Original scope, or None for absent, drained or unbound reservations.

    Examples
    --------
    >>> pending_scope({"run_id": "absent", "reservation_id": "r"}) is None
    True
    """
    run_id = _context_string(metadata.get("run_id", ""), "run_id")
    rid = _context_string(metadata.get("reservation_id", ""), "reservation_id")
    binding = _bindings.get(run_id)
    entry = binding.pending.get(rid) if binding is not None else None
    return (
        entry.scope
        if binding is not None and entry is not None and id(entry.scope) not in binding.drained
        else None
    )


def _digest(value: str | None) -> str | None:
    if value is None:
        return None
    value = _context_string(value, "cache/request digest")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise _core_error(ValueError, "invalid LLM call digest")
    return value


async def _watch_owner(binding: _Binding, scope: anyio.CancelScope) -> None:
    """Bound foreign callback tasks by the same stop state and active clock."""
    grace: float | None = None
    while True:
        if not binding.active:
            scope.cancel()
            return
        left = binding.clock.left(binding.limits.run_timeout_s)
        if binding.stops.stop or binding.stops.stop_now or binding.stops.run_timeout or left <= 0:
            binding.journal._stop_storage()
            if grace is None:
                grace = anyio.current_time() + binding.settle_timeout_s
            left = grace - anyio.current_time()
            if left <= 0:
                binding.terminal = _stop_cleanup_error()
                scope.cancel()
                return
        await anyio.sleep(min(0.05, max(0.001, left)))


async def _watch_writer(binding: _Binding, scope: anyio.CancelScope) -> None:
    await binding.journal.fenced.wait()
    scope.cancel()


async def _accounting_wait(binding: _Binding, operation: Callable[[], Awaitable[_T]]) -> _T:
    """Cancel a waiter across inner accounting shields without undoing admission."""
    task = asyncio.ensure_future(operation())
    try:
        # Keep the cancelling owner's original scope marker; the child may
        # rebuild its own cancellation through a storage containment boundary.
        return await asyncio.shield(task)
    finally:
        if not task.done():
            task.cancel()
            with anyio.move_on_after(binding.settle_timeout_s, shield=True):
                with suppress(asyncio.CancelledError):
                    await task


async def _owned(
    binding: _Binding,
    operation: Callable[[], Awaitable[_T]],
    *,
    entry: _Pending | None = None,
    accounting: bool = False,
) -> tuple[bool, _T | None]:
    """Await one operation with owner supervision and no adoption after abandonment."""
    if not binding.active:
        return False, None
    binding.check()
    outcome = _Outcome()
    completed = False
    result: _T | None = None
    error: BaseException | None = None
    with anyio.CancelScope() as scope:
        binding.workers.add(scope)
        if entry is not None:
            entry.workers.add(scope)
        try:
            async with anyio.create_task_group() as watchers:
                watchers.start_soon(_watch_owner, binding, scope)
                watchers.start_soon(_watch_writer, binding, scope)
                try:
                    await anyio.lowlevel.checkpoint_if_cancelled()
                    result = (
                        await _accounting_wait(binding, operation)
                        if accounting
                        else await operation()
                    )
                    completed = True
                except BaseException as caught:
                    error = caught
                finally:
                    watchers.cancel_scope.cancel()
        finally:
            binding.workers.discard(scope)
            if entry is not None:
                entry.workers.discard(scope)
    # Preserve an enclosing timeout's marker so its owner can classify expiry.
    if isinstance(error, anyio.get_cancelled_exc_class()) and not scope.cancel_called:
        raise error
    # A concurrent native caller cancellation wins over our watcher cancellation.
    outcome.propagate_control(asyncio.CancelledError())
    if not binding.active:
        return False, None
    binding.check()
    if error is not None:
        if scope.cancel_called and isinstance(error, anyio.get_cancelled_exc_class()):
            return False, None
        raise error
    return completed and not scope.cancel_called, result


async def _prepare(operation: Callable[[], _T]) -> _T:
    outcome = _Outcome()
    try:
        return await run_sync_daemon(operation)
    except BaseException as error:
        outcome.contain(error, ValueError, "LLM record preparation")


def _metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Normalize only documented identifier fields in a preparation worker."""
    fields = (
        "run_id",
        "reservation_id",
        "call_id",
        "goal_id",
        "team",
        "agent",
        "round",
        "role",
        "candidate_id",
        "model_alias",
        "trace_url",
        "parent_call_id",
    )
    return {
        key: _context_string(value, key)
        for key in fields
        if (value := metadata.get(key)) is not None
    }


@dataclass(frozen=True)
class _Prepared:
    call: LLMCall
    request: bytes
    response: bytes
    request_sha: str
    cache_identity: str | None
    actual: Amount
    overflow: bool


def _prepare_call(
    meta: dict[str, str], cid: str, entry: _Pending | None, kwargs: dict[str, Any]
) -> _Prepared:
    """Detach and validate the complete record before uploads or a dedup claim."""
    for flag in ("settle", "charge_reservation"):
        if type(kwargs[flag]) is not bool:
            raise ValueError(f"invalid {flag}")
    usage = _detach_model(Usage, kwargs["usage"])
    cost = _plain(kwargs["cost_usd"])
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ValueError("invalid cost_usd")
    cache_hit = kwargs["cache_hit"]
    if type(cache_hit) is not bool:
        raise ValueError("invalid cache_hit")
    estimated = kwargs["charge_reservation"] and entry is not None
    overflow = not cache_hit and cost > USD_LIMIT and not estimated
    if estimated:
        usage = Usage(input_tokens=entry.estimate.tokens)
        actual = entry.estimate
    else:
        actual = Amount(
            tokens=0 if cache_hit else usage.total_tokens,
            usd=0 if cache_hit else USD_LIMIT if overflow else cost,
        )
    fields: dict[str, Any] = {
        key: meta[key]
        for key in (
            "run_id",
            "goal_id",
            "team",
            "agent",
            "role",
            "candidate_id",
            "model_alias",
            "trace_url",
            "parent_call_id",
            "reservation_id",
        )
        if key in meta
    }
    fields["round"] = int(meta.get("round", "0"))
    if entry is not None:
        if cid != entry.call_id:
            raise ValueError("call_id does not match reservation_id")
        for name in ("run_id", "goal_id", "team", "agent", "round", "role", "candidate_id"):
            expected = getattr(entry.scope, name)
            if name in fields and fields[name] != expected:
                raise ValueError("metadata does not match the pending scope")
            fields[name] = expected
        fields["model_alias"] = entry.model_alias
    status = _context_string(kwargs["status"], "status")
    if status not in ("ok", "error"):
        raise ValueError("invalid status")
    call = LLMCall(
        **fields,
        id=cid,
        model=_context_string(kwargs["model"], "model"),
        provider=_context_string(kwargs["provider"], "provider"),
        tool_calls=[
            {"id": _context_string(id, "tool_call_id"), "name": _context_string(name, "tool_name")}
            for id, name in kwargs["tool_call_ids"]
        ],
        usage=usage,
        cost_usd=actual.usd,
        cache_hit=cache_hit,
        latency_ms=_plain(kwargs["latency_ms"]),
        status="error" if estimated or overflow else status,
        error=_UNKNOWN_SPEND
        if estimated
        else _OVERFLOW
        if overflow
        else ("provider call failed" if status == "error" else None),
        litellm_call_id=None
        if kwargs["litellm_call_id"] is None
        else _context_string(kwargs["litellm_call_id"], "litellm_call_id"),
    )
    request = canonical_json(_plain(kwargs["request"]))
    response = canonical_json(_plain(kwargs["response"]))
    return _Prepared(
        call,
        request,
        response,
        sha256_hex(request),
        _digest(kwargs["cache_identity"]),
        actual,
        overflow,
    )


def _put_payload(artifacts: ArtifactStore, content: bytes) -> str:
    """Validate backend-returned hashes in the worker that performed the upload."""
    digest = _context_string(artifacts.put_bytes(content), "artifact digest")
    if digest != sha256_hex(content):
        raise ValueError("artifact digest mismatch")
    return digest


def _duplicate(binding: _Binding, cid: str, lid: str | None, entry: _Pending | None) -> bool:
    return (
        not binding.active
        or binding.closing
        or cid in binding.recorded
        or (lid is not None and lid in binding.seen_litellm)
        or (
            entry is not None and (entry.state == "abandoned" or id(entry.scope) in binding.drained)
        )
    )


def _append(
    binding: _Binding, call: LLMCall, request_sha: str | None, identity: str | None
) -> None:
    binding.check()
    outcome = _Outcome()
    try:
        binding.journal.append(
            EventDraft(
                type="llm_call",
                goal_id=call.goal_id,
                team=call.team,
                agent=call.agent,
                round=call.round,
                task_id=agent_task_id(f"{call.goal_id}/{call.team}/{call.agent}", call.round)
                if call.team and call.agent
                else "",
                payload={
                    "call": call.model_dump(mode="json"),
                    "request_sha256": request_sha,
                    "cache_identity": identity,
                },
            )
        )
    except BaseException as error:
        outcome.contain(error, ValueError, "LLM journal append")
    binding.recorded.add(call.id)
    if call.litellm_call_id is not None:
        binding.seen_litellm[call.litellm_call_id] = call.id
    if identity is not None and call.status == "ok" and not call.cache_hit:
        binding.original(identity, call.id)


async def _settle(binding: _Binding, entry: _Pending) -> None:
    """Retry only outstanding accounting; the tranche owns admission idempotency."""
    if entry.settled or entry.actual is None:
        return
    outcome = _Outcome()
    try:
        # Protection from caller cancellation is bounded independently of a
        # provider report wait. The writer retains its own lease/stop limits.
        with anyio.move_on_after(binding.settle_timeout_s, shield=True) as cleanup:
            completed, result = await _owned(
                binding,
                partial(
                    entry.scope.tranche._settle_call,
                    entry.reservation_id,
                    entry.actual,
                    overflow=True,
                )
                if entry.overflow
                else partial(entry.scope.tranche.settle_call, entry.reservation_id, entry.actual),
                accounting=True,
            )
        if cleanup.cancel_called:
            binding.journal._raise_if_fenced()
            binding.terminal = _stop_cleanup_error()
            raise binding.terminal
        if not completed:
            return
        assert result is not None
        entry.settled = True
        if result.overflow:
            binding.overflow_notified.add(id(entry.scope.tranche))
            raise _core_signal(
                LimitReached("usage_overflow", "reported usage exceeds the supported range")
            )
    except BaseException as error:
        outcome.contain(error, StorageError, "LLM settlement")


async def record_call(
    *,
    metadata: Mapping[str, Any],
    model: str,
    provider: str,
    usage: Usage,
    cost_usd: float,
    cache_hit: bool,
    latency_ms: float | None,
    status: Literal["ok", "error"],
    error: str | None,
    request: JsonValue,
    response: JsonValue,
    litellm_call_id: str | None,
    tool_call_ids: Sequence[tuple[str, str]],
    settle: bool,
    charge_reservation: bool = False,
    cache_identity: str | None = None,
) -> LLMCall | None:
    """Prepare, journal and settle one call, suppressing duplicate or late reports.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        String tracking fields, including run_id, reservation_id and call_id.
    model, provider : str
        Actual model and provider identifiers.
    usage : Usage
        Authoritative usage; retained for display on a zero-charge cache hit.
    cost_usd : float
        Reported nonnegative cost, normalized once to whole micro-dollars.
    cache_hit : bool
        Whether this response was served from cache.
    latency_ms : float or None
        Nonnegative observed latency.
    status : {"ok", "error"}
        Provider outcome.
    error : str or None
        Provider diagnostic input. Raw text is never retained or displayed.
    request, response : JsonValue
        Finite JSON payloads, serialized in supervised preparation workers.
    litellm_call_id : str or None
        Adapter deduplication identifier.
    tool_call_ids : Sequence[tuple[str, str]]
        Requested action identifiers and names.
    settle : bool
        Settle a matching pending reservation at the recorded amount.
    charge_reservation : bool, default=False
        Charge the complete estimate for unknown provider spend.
    cache_identity : str or None, optional
        SHA-256 of the namespace, effective key and generating response identity.
        Missing/ambiguous provenance leaves cached_from_call_id unset.

    Returns
    -------
    LLMCall or None
        Newly journaled call, or None for a duplicate or abandoned callback.

    Notes
    -----
    Callback tasks are marshaled to the binding's loop and retain its live stop,
    clock and writer supervision. Each payload upload is separately awaited.
    A journaled call is not a promise of durability after a fence or crash.

    Examples
    --------
    The call identifier is allocated before dispatch and appears once in the
    scope, including when ``drain`` records its conservative placeholder.
    >>> pending_scope({"run_id": "absent", "reservation_id": "missing"}) is None
    True
    """
    # Routing data is copied off-loop; all serialization and model work after
    # routing is additionally bounded by the live owner's supervision.
    meta = await _prepare(partial(_metadata, metadata))
    run_id = meta.get("run_id", "")
    binding = _bindings.get(run_id)
    if binding is None:
        if meta.get("call_id") or meta.get("reservation_id") or litellm_call_id is not None:
            return None
        raise _core_error(ValueError, f"run {run_id} is not bound")
    kwargs = dict(
        model=model,
        provider=provider,
        usage=usage,
        cost_usd=cost_usd,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        status=status,
        error=error,
        request=request,
        response=response,
        litellm_call_id=litellm_call_id,
        tool_call_ids=tool_call_ids,
        settle=settle,
        charge_reservation=charge_reservation,
        cache_identity=cache_identity,
    )
    if asyncio.get_running_loop() is not binding.loop:
        future = asyncio.run_coroutine_threadsafe(_record(binding, meta, kwargs), binding.loop)
        return await asyncio.wrap_future(future)
    return await _record(binding, meta, kwargs)


async def _record(
    binding: _Binding, meta: dict[str, str], kwargs: dict[str, Any]
) -> LLMCall | None:
    entry = binding.pending.get(meta.get("reservation_id", ""))
    cid = meta.get("call_id") or (entry.call_id if entry is not None else binding.ids.next("llm"))
    reservation = binding.call_reservations.get(cid)
    if reservation is not None and meta.get("reservation_id") != reservation:
        raise _core_error(ValueError, "LLM call requires its matching reservation")
    lid = kwargs["litellm_call_id"]
    if lid is not None:
        lid = _context_string(lid, "litellm_call_id")
    if _duplicate(binding, cid, lid, entry):
        return None
    binding.check()
    try:
        ready, prepared = await _owned(
            binding,
            partial(_prepare, partial(_prepare_call, meta, cid, entry, kwargs)),
            entry=entry,
        )
        if not ready or _duplicate(binding, cid, lid, entry):
            return None
        assert prepared is not None
        call = prepared.call
        if binding.tracing.store_payloads:
            refs: list[str] = []
            for content in (prepared.request, prepared.response):
                ready, digest = await _owned(
                    binding,
                    partial(
                        _storage_io, _put_payload, binding.artifacts, content, site="LLM payload"
                    ),
                    entry=entry,
                )
                if not ready or _duplicate(binding, cid, lid, entry):
                    return None
                assert digest is not None
                refs.append(digest)
            call = call.model_copy(update={"request_ref": refs[0], "response_ref": refs[1]})
        if call.cache_hit and prepared.cache_identity is not None:
            call = call.model_copy(
                update={"cached_from_call_id": binding.originals.get(prepared.cache_identity)}
            )
        # No await from the final duplicate check through journal admission.
        if entry is not None:
            entry.state = "claimed"
        _append(binding, call, prepared.request_sha, prepared.cache_identity)
        scope = entry.scope if entry is not None else current_llm_scope()
        if scope is not None and scope.run_id == call.run_id and call.id not in scope.call_ids:
            scope.call_ids.append(call.id)
        if entry is not None:
            entry.journaled = True
            entry.actual = prepared.actual
            entry.overflow = prepared.overflow
            if kwargs["settle"]:
                await _settle(binding, entry)
        return call
    except BaseException as caught:
        # Every boundary above already contains untrusted failures. Cancellation
        # never becomes an error-bearing placeholder or a terminal-control forgery.
        if _has_core_fenced(caught):
            binding.terminal = _core_error(Fenced, "run ownership lost")
        if (
            entry is not None
            and isinstance(caught, Exception)
            and not (isinstance(caught, LimitReached) and entry.settled)
        ):
            entry.error = caught
        raise
    finally:
        if entry is not None:
            if entry.state == "claimed":
                entry.state = "done"
            entry.event.set()


def abandon_call(scope: LLMScope, reservation_id: str) -> None:
    """Mark a cancelled call for immediate conservative drain without I/O.

    Parameters
    ----------
    scope : LLMScope
        Exact originating scope, not an equivalent replacement.
    reservation_id : str
        Reservation whose callback will not arrive.

    Examples
    --------
    Cancellation calls this before ``await drain(scope, wait=False)``. It does
    not allocate another call identifier or perform any settlement itself.
    >>> pending_scope({"run_id": "absent"}) is None
    True
    """
    binding = _bindings.get(_context_string(scope.run_id, "run_id"))
    rid = _context_string(reservation_id, "reservation_id")
    entry = binding.pending.get(rid) if binding is not None else None
    if entry is not None and entry.scope is scope and not entry.journaled:
        entry.state = "abandoned"
        for worker in tuple(entry.workers):
            worker.cancel()
        entry.event.set()


def _placeholder(binding: _Binding, entry: _Pending) -> None:
    if entry.call_id in binding.recorded:
        raise _core_error(ValueError, "LLM call is already journaled")
    scope = entry.scope
    call = LLMCall(
        id=entry.call_id,
        run_id=scope.run_id,
        goal_id=scope.goal_id,
        team=scope.team,
        agent=scope.agent,
        round=scope.round,
        role=scope.role,
        candidate_id=scope.candidate_id,
        model_alias=entry.model_alias,
        model=entry.model,
        provider=entry.provider,
        reservation_id=entry.reservation_id,
        usage=Usage(input_tokens=entry.estimate.tokens),
        cost_usd=entry.estimate.usd,
        status="error",
        error=_NO_REPORT,
    )
    _append(binding, call, None, None)
    entry.journaled = True
    entry.actual = entry.estimate
    if call.id not in scope.call_ids:
        scope.call_ids.append(call.id)


async def drain(scope: LLMScope, *, wait: bool) -> list[str]:
    """Finish a scope, conservatively recording missing calls within owned bounds.

    Parameters
    ----------
    scope : LLMScope
        Exact scope whose calls are complete. It cannot open more calls after drain.
    wait : bool
        Wait up to settle_timeout_s for reports; abandoned calls never wait.

    Returns
    -------
    list[str]
        Identifiers of placeholders actually appended by this drain.

    Notes
    -----
    Claims without a journaled call can be recovered only while the writer is
    usable. Journaled calls retry their recorded accounting idempotently.
    The first safe recorder error is raised after all recoverable calls settle.
    Terminal publication failure retains conservative reservations and its cause.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> async def example() -> list[str]:
    ...     return await drain(SimpleNamespace(run_id="absent"), wait=False)
    >>> anyio.run(example)
    []
    """
    binding = _bindings.get(_context_string(scope.run_id, "run_id"))
    if binding is None:
        return []
    entries = [entry for entry in binding.pending.values() if entry.scope is scope]
    binding.track_scope(scope)
    if not entries:
        binding.drained.add(id(scope))
        binding.finish_scope(scope)
        return []
    binding.check()
    if wait:
        with anyio.move_on_after(scope.settle_timeout_s):
            for entry in entries:
                if entry.state != "abandoned":
                    await _owned(binding, entry.event.wait)
    binding.drained.add(id(scope))
    placeholders: list[str] = []
    failure: Exception | None = None
    with anyio.move_on_after(scope.settle_timeout_s, shield=True) as cleanup:
        for entry in entries:
            binding.check()
            for worker in tuple(entry.workers):
                worker.cancel()
            try:
                if not entry.journaled:
                    entry.state = "claimed"
                    _placeholder(binding, entry)
                    placeholders.append(entry.call_id)
                await _settle(binding, entry)
            except Exception as error:
                failure = failure or error
            finally:
                entry.state = "done"
                entry.event.set()
            failure = failure or entry.error
            entry.error = None
    if cleanup.cancel_called:
        binding.terminal = _stop_cleanup_error()
    binding.check()
    if all(entry.settled for entry in entries):
        binding.finish_scope(scope)
    if failure is not None:
        raise failure from None
    return placeholders


@asynccontextmanager
async def _bound_run(
    run_id: str,
    *,
    journal: JournalWriter,
    ids: IdGen,
    artifacts: ArtifactStore,
    config: RunConfig,
    stops: StopState,
    clock: RunClock,
    state: RunState,
) -> AsyncGenerator[None]:
    """Bind once, then drain/close within usable services and unbind on every exit."""
    bind_run(
        run_id,
        journal=journal,
        ids=ids,
        artifacts=artifacts,
        tracing=config.tracing,
        committed=list(state.llm_calls.values()),
        request_sha256=state.llm_request_sha256,
        cache_identity=state.llm_cache_identity,
        stops=stops,
        clock=clock,
        limits=config.limits,
        settle_timeout_s=config.settle_timeout_s,
    )
    binding = _bindings[run_id]
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        binding.closing = True
        failure: BaseException | None = None
        try:
            if binding.active:
                with anyio.move_on_after(config.settle_timeout_s, shield=True) as cleanup:
                    for scope in binding.scopes.values():
                        try:
                            await drain(scope, wait=False)
                        except BaseException as error:
                            failure = failure or error
                    tranches = {
                        id(scope.tranche): scope.tranche for scope in binding.scopes.values()
                    }
                    for key, tranche in tranches.items():
                        if binding.undrained_scopes[key]:
                            continue
                        try:
                            completed, result = await _owned(
                                binding, tranche.close, accounting=True
                            )
                            if (
                                completed
                                and result is not None
                                and result.overflow
                                and key not in binding.overflow_notified
                            ):
                                binding.overflow_notified.add(key)
                                raise _core_signal(
                                    LimitReached(
                                        "usage_overflow",
                                        "reported usage exceeds the supported range",
                                    )
                                )
                        except BaseException as error:
                            failure = failure or error
                    binding.check()
                if cleanup.cancel_called:
                    failure = _stop_cleanup_error()
        except BaseException as error:
            failure = failure or error
        finally:
            if _bindings.get(run_id) is binding:
                unbind_run(run_id)
        if failure is not None and not failed:
            raise failure from None
