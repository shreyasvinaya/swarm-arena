"""Own per-tool virtual keys and ingest settlement-free gateway observations."""

import logging
import math
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from contextvars import Context, ContextVar
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Annotated, Any, Final, Literal, TypeVar, get_args

import anyio
from pydantic import AwareDatetime, Field

from swarm_arena._budget.ledger import Reservation
from swarm_arena._exec.executor import Job
from swarm_arena._gateway.adapter import GatewayAdapter, GatewayError, VirtualKey
from swarm_arena._gateway.labels import _observation_label
from swarm_arena._gateway.reports import GatewayKeyAction, GatewayReadStatus
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.llm_record import record_call
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.results import LLMCall
from swarm_arena._model.settings import GatewayConfig
from swarm_arena._outcomes import _Outcome
from swarm_arena._tooling.llm_calls import PROVIDER_KEYS

REPLACED_KEYS: Final = frozenset({"OPENAI_API_KEY", "ANTHROPIC_API_KEY"})
_CLEANUP_TIMEOUT_S: Final = 30.0
_CLOCK_SKEW: Final = timedelta(minutes=15)
_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")
_read_deadline: ContextVar[float] = ContextVar("gateway_read_deadline", default=math.inf)


def needs_key(providers: frozenset[str], secret_env: Sequence[str]) -> bool:
    """Decide whether declared providers or replaceable names require a key.

    Parameters
    ----------
    providers : frozenset of str
        Effective tool providers.
    secret_env : Sequence of str
        Effective environment allowlist, including configured command names.

    Returns
    -------
    bool
        Whether a virtual key is required.

    Examples
    --------
    >>> needs_key(frozenset(), ["OPENAI_API_KEY"])
    True
    """
    return bool(providers or REPLACED_KEYS.intersection(secret_env))


def unreplaced_provider_keys(secret_env: Sequence[str]) -> list[str]:
    """List unique provider names for which gateway replacement is unavailable.

    Parameters
    ----------
    secret_env : Sequence of str
        Effective environment allowlist.

    Returns
    -------
    list of str
        Sorted names for an identifier-only warning.

    Examples
    --------
    >>> unreplaced_provider_keys(["GEMINI_API_KEY", "OPENAI_API_KEY"])
    ['GEMINI_API_KEY']
    """
    return sorted(set(secret_env).intersection(PROVIDER_KEYS - REPLACED_KEYS))


def key_alias(candidate_id: str) -> str:
    """Build the stable candidate alias.

    Parameters
    ----------
    candidate_id : str
        Tool invocation identity.

    Returns
    -------
    str
        Gateway alias.

    Examples
    --------
    >>> key_alias("c-1")
    'cand-c-1'
    """
    return f"cand-{candidate_id}"


def key_metadata(
    *, candidate_id: str, run_id: str, goal_id: str, team: str, agent: str
) -> dict[str, Any]:
    """Build gateway attribution without secret values.

    Parameters
    ----------
    candidate_id, run_id, goal_id, team, agent : str
        Core-owned invocation identifiers.

    Returns
    -------
    dict of str to Any
        Attribution, searchable tags and spend-log metadata.

    Examples
    --------
    >>> key_metadata(candidate_id="c", run_id="r", goal_id="g", team="t",
    ...              agent="a")["tags"]
    ['swarm-arena', 'run:r', 'goal:g', 'candidate:c']
    """
    identity = dict(
        candidate_id=candidate_id, run_id=run_id, goal_id=goal_id, team=team, agent=agent
    )
    return {
        **identity,
        "tags": ["swarm-arena", f"run:{run_id}", f"goal:{goal_id}", f"candidate:{candidate_id}"],
        "spend_logs_metadata": identity.copy(),
    }


def gateway_call_id(key_id: str, request_id: str) -> str:
    """Identify a gateway observation independently of client-selected call IDs.

    Parameters
    ----------
    key_id, request_id : str
        Verified key hash and primary spend-row identity.

    Returns
    -------
    str
        Canonical ``gw:`` SHA-256 identity in the tool-observation namespace.

    Examples
    --------
    >>> len(gateway_call_id("hash", "request"))
    67
    """
    return "gw:" + sha256_hex(canonical_json([key_id, request_id]))


def gateway_job(job: Job, *, url: str, secret: str) -> Job:
    """Route a copied job through the gateway and strip node provider keys.

    Parameters
    ----------
    job : Job
        Original invocation, left unchanged.
    url : str
        Gateway base URL.
    secret : str
        Ephemeral virtual key; retained only in ``secret_values``.

    Returns
    -------
    Job
        Invocation with gateway URL/value overlays and provider drops.

    Examples
    --------
    >>> from swarm_arena import Resources
    >>> from swarm_arena.tool import Command
    >>> job = Job(id="x", kind="command", command=Command(argv=["true"]),
    ...     stage=Job.Stage(), timeout_s=1, grace_s=0, resources=Resources(),
    ...     env={}, secret_env=(), secrets_file=None, artifact_root="artifacts")
    >>> gateway_job(job, url="http://gateway/", secret="example").env["OPENAI_BASE_URL"]
    'http://gateway/v1'
    """
    base = url.rstrip("/")
    return replace(
        job,
        env={**job.env, "OPENAI_BASE_URL": f"{base}/v1", "ANTHROPIC_BASE_URL": base},
        secret_values={**job.secret_values, **dict.fromkeys(REPLACED_KEYS, secret)},
        secret_env=tuple(name for name in job.secret_env if name not in PROVIDER_KEYS),
        drop_env=tuple(sorted(PROVIDER_KEYS.union(job.drop_env))),
    )


class KeyRecord(FrozenModel):
    """Retain nonsecret key ownership and a conservative spend-query bound.

    Parameters
    ----------
    candidate_id, key_id, alias, reservation_id : str
        Tool, gateway and charge identities; never the bearer value.
    max_budget_usd : float
        Full reserved dollar amount.
    cap : {'hard', 'soft'}
        Verified capability label, supplied by preflight.
    run_id : str
        Run owning the key.
    owner_epoch : int
        Creation epoch; ingestion always uses the live writer's epoch instead.
    goal_id, team, agent : str
        Core-owned attribution.
    round : int
        Tool invocation round.
    created_at : datetime
        UTC lower bound captured before sending create.
    closed_at : datetime or None, default=None
        Adapter-confirmed closure; does not prove final spend completeness.

    Examples
    --------
    >>> record = KeyRecord(candidate_id="c", key_id="hash", alias="cand-c",
    ...     reservation_id="res", max_budget_usd=1, cap="soft", run_id="r",
    ...     owner_epoch=1, goal_id="g", team="t", agent="a", round=1,
    ...     created_at=datetime(2026, 1, 1, tzinfo=UTC))
    >>> record.closed_at is None
    True
    """

    candidate_id: str
    key_id: str
    alias: str
    reservation_id: str
    max_budget_usd: Annotated[float, Field(ge=0)]
    cap: Literal["hard", "soft"]
    run_id: str
    owner_epoch: Annotated[int, Field(ge=1)]
    goal_id: str
    team: str
    agent: str
    round: int
    created_at: AwareDatetime
    closed_at: AwareDatetime | None = None


def _virtual(record: KeyRecord) -> VirtualKey:
    return VirtualKey(key_id=record.key_id, alias=record.alias)


class KeyManager:
    """Manage keys under an entered writer and the caller's run supervision.

    Parameters
    ----------
    adapter : GatewayAdapter
        Bounded adapter owning undelivered creates and late-response cleanup.
    config : GatewayConfig
        Key lifetime and output estimate.
    cap : {'hard', 'soft'}
        Preflight capability label.
    journal : JournalWriter
        Entered writer retained for its live epoch and failure supervision.
    settle_timeout_s : float
        Finite nonnegative optional first-row polling allowance. It never bounds
        healthy durable recorder settlement or supplies a fresh cleanup window.
    run_id : str
        Explicit run binding, checked against open and restored events.

    Notes
    -----
    Callers retain their stop/deadline supervisor and own a returned key until
    close/revoke. Pre-return compensation has at most 30 seconds from open entry,
    capped by the inherited deadline and the writer's remaining closing allowance.
    It writes no compensation event. Expired/failed cleanup falls back to key expiry.
    Revoke never shields or restarts the caller's cleanup budget.

    Examples
    --------
    >>> import inspect
    >>> "run_id" in inspect.signature(KeyManager).parameters
    True
    """

    def __init__(
        self,
        adapter: GatewayAdapter,
        config: GatewayConfig,
        *,
        cap: Literal["hard", "soft"],
        journal: JournalWriter,
        settle_timeout_s: float,
        run_id: str,
    ) -> None:
        if not math.isfinite(settle_timeout_s) or settle_timeout_s < 0:
            raise ValueError("settle_timeout_s must be finite and nonnegative")
        self._adapter = adapter
        self._config = config
        self._cap = cap
        self._journal = journal
        self._run_id = run_id
        self._settle_timeout_s = settle_timeout_s
        self._records: dict[str, KeyRecord] = {}
        self._boundary_reads: dict[str, datetime] = {}

    @asynccontextmanager
    async def _owned(self) -> AsyncGenerator[None]:
        # Import lazily: engine construction will itself consume this module.
        from swarm_arena._engine.run import _supervise_writer

        self._journal._check_open()
        async with _supervise_writer(self._journal):
            yield
            self._journal._check_open()

    async def _call(self, operation: Callable[[], Awaitable[_T]], site: str) -> _T:
        outcome = _Outcome()
        try:
            return await operation()
        except BaseException as error:
            outcome.propagate_control(error)
            raise GatewayError(f"{site} failed") from None

    async def _compensate(self, key: VirtualKey, deadline: float) -> None:
        deadline = min(deadline, self._journal._closing_at)
        if self._journal._io_scope is not None:
            deadline = min(deadline, self._journal._io_scope.deadline)
        if deadline <= anyio.current_time():
            _LOG.warning("gateway key cleanup expired")
            return
        with anyio.CancelScope(deadline=deadline, shield=True) as cleanup:
            try:
                await self._call(lambda: self._adapter.block_key(key), "gateway block")
            except GatewayError:
                _LOG.warning("gateway key cleanup failed")
        if cleanup.cancelled_caught:
            _LOG.warning("gateway key cleanup expired")

    async def open(
        self,
        *,
        candidate_id: str,
        reservation: Reservation,
        run_id: str,
        goal_id: str,
        team: str,
        agent: str,
        round: int,
    ) -> tuple[KeyRecord, str]:
        """Create and journal a key, transferring ownership only on return.

        Parameters
        ----------
        candidate_id, run_id, goal_id, team, agent : str
            Trusted invocation attribution; run_id must match this manager.
            Use one create attempt per candidate_id; already held candidates
            are refused even when their key is closed.
        reservation : Reservation
            Existing full tool charge, never settled by this manager.
        round : int
            Invocation round.

        Returns
        -------
        tuple of KeyRecord and str
            Registered record and ephemeral bearer value.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(KeyManager.open)
        True
        """
        if run_id != self._run_id:
            raise ValueError("gateway run binding mismatch")
        if any(record.candidate_id == candidate_id for record in self._records.values()):
            raise ValueError("candidate already has a gateway key")
        deadline = min(
            anyio.current_effective_deadline(), anyio.current_time() + _CLEANUP_TIMEOUT_S
        )
        key: VirtualKey | None = None
        outcome = _Outcome()
        try:
            async with self._owned():
                created_at = datetime.now(UTC)
                key, secret = await self._call(
                    lambda: self._adapter.create_key(
                        alias=key_alias(candidate_id),
                        max_budget_usd=reservation.amount.usd,
                        duration_s=self._config.key_duration_s,
                        metadata=key_metadata(
                            candidate_id=candidate_id,
                            run_id=run_id,
                            goal_id=goal_id,
                            team=team,
                            agent=agent,
                        ),
                        default_estimated_output_tokens=self._config.default_estimated_output_tokens,
                    ),
                    "gateway create",
                )
                await anyio.lowlevel.checkpoint_if_cancelled()
                self._journal._check_open()
                record = KeyRecord(
                    candidate_id=candidate_id,
                    key_id=key.key_id,
                    alias=key.alias,
                    reservation_id=reservation.id,
                    max_budget_usd=reservation.amount.usd,
                    cap=self._cap,
                    run_id=run_id,
                    owner_epoch=self._journal.epoch,
                    goal_id=goal_id,
                    team=team,
                    agent=agent,
                    round=round,
                    created_at=created_at,
                )
                self._journal.append(
                    self._draft(record, {"action": "created", **record.model_dump(mode="json")})
                )
            self._records[record.key_id] = record
            return record, secret
        except BaseException as error:
            cleanup_error: BaseException | None = None
            if key is not None:
                try:
                    await self._compensate(key, deadline)
                except BaseException as caught:
                    cleanup_error = caught
            outcome.propagate_control(error)
            if cleanup_error is not None:
                outcome.propagate_control(cleanup_error)
            raise GatewayError("gateway open failed") from None

    def _draft(self, record: KeyRecord, payload: dict[str, Any]) -> EventDraft:
        return EventDraft(
            type=EventType.GATEWAY_KEY,
            goal_id=record.goal_id,
            team=record.team,
            agent=record.agent,
            round=record.round,
            task_id=record.candidate_id,
            payload=payload,
        )

    def _check_record(self, record: KeyRecord) -> None:
        if record.run_id != self._run_id:
            raise ValueError("gateway run binding mismatch")

    async def close(self, record: KeyRecord) -> KeyRecord:
        """Block a key and journal only adapter-confirmed closure.

        Parameters
        ----------
        record : KeyRecord
            Owned key record.

        Returns
        -------
        KeyRecord
            Confirmed updated record, or unchanged record with a safe warning.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(KeyManager.close)
        True
        """
        self._check_record(record)
        async with self._owned():
            try:
                await self._call(lambda: self._adapter.block_key(_virtual(record)), "gateway block")
                closed_at = await self._call(
                    lambda: self._adapter.closed_at(_virtual(record)), "gateway closure"
                )
            except GatewayError:
                _LOG.warning("gateway key closure unconfirmed")
                return record
            if closed_at is None:
                _LOG.warning("gateway key closure unconfirmed")
                return record
            updated = KeyRecord.model_validate({**record.model_dump(), "closed_at": closed_at})
            self._journal.append(
                self._draft(
                    record,
                    {
                        "action": "closed",
                        "key_id": record.key_id,
                        "closed_at": updated.model_dump(mode="json")["closed_at"],
                    },
                )
            )
            self._records[record.key_id] = updated
            return updated

    async def revoke(self, record: KeyRecord) -> None:
        """Block a key without reading closure or writing to the journal.

        Parameters
        ----------
        record : KeyRecord
            Key whose external cleanup remains owned by the caller. Caller
            supplies any shield and remaining cleanup deadline, even after fencing.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(KeyManager.revoke)
        True
        """
        self._check_record(record)
        await anyio.lowlevel.checkpoint_if_cancelled()
        await self._call(lambda: self._adapter.block_key(_virtual(record)), "gateway revoke")

    def _revoke_later(self, record: KeyRecord) -> None:
        """Hand one external-only block attempt to an independent bounded owner."""
        self._check_record(record)
        adapter, key = self._adapter, _virtual(record)
        timeout_s = _CLEANUP_TIMEOUT_S

        async def block() -> None:
            with anyio.move_on_after(timeout_s):
                try:
                    await adapter.block_key(key)
                except Exception:
                    _LOG.warning("gateway key cleanup failed")

        threading.Thread(
            target=Context().run,
            args=(anyio.run, block),
            daemon=True,
            name="swarm-arena-gateway-block",
        ).start()

    async def read(self, records: Sequence[KeyRecord], *, wait_first: bool) -> list[LLMCall]:
        """Read bounded spend windows and record informational tool observations.

        Parameters
        ----------
        records : Sequence of KeyRecord
            Keys attributed through trusted records, never row metadata.
        wait_first : bool
            Poll immediately and then at most once per second until rows arrive
            or the first-row deadline expires. Duplicate rows also end the wait.

        Returns
        -------
        list of LLMCall
            Newly recorded calls. Failed/incomplete reads warn and return no
            observations; no estimate or ledger balance is changed. Invalid model
            and provider labels become "unknown". An ordinary recording failure
            warns once for that row and does not prevent later rows from recording.
            Cancellation and process-control exceptions propagate unchanged.

        Notes
        -----
        Query bounds include 15 minutes of clock skew on both sides, in addition
        to second rounding and the one-second lower overlap. The key filter
        selects observations and canonical identities deduplicate this overlap.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(KeyManager.read).parameters["wait_first"].kind.name
        'KEYWORD_ONLY'
        """
        calls, _ = await self.read_with_status(records, wait_first=wait_first)
        return calls

    async def read_with_status(
        self, records: Sequence[KeyRecord], *, wait_first: bool
    ) -> tuple[list[LLMCall], GatewayReadStatus]:
        """Return informational observations and an operation-local read status.

        Parameters
        ----------
        records : Sequence of KeyRecord
            Trusted keys to read with the same ownership and polling rules as
            :meth:`read`.
        wait_first : bool
            Whether to poll for the first visible spend rows.

        Returns
        -------
        tuple of list of LLMCall and str
            Newly recorded observations and a value-free status: ``ok`` for a
            successful read (including empty or duplicate-only reads), ``failed``
            for a failed gateway operation, or ``incomplete`` for polling expiry
            or an ordinary per-row recording failure. Success does not certify
            terminal accounting or final spend completeness. Controls propagate.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(KeyManager.read_with_status).parameters["wait_first"].kind.name
        'KEYWORD_ONLY'
        """
        for record in records:
            self._check_record(record)
        if not records:
            return [], "ok"
        deadline = anyio.current_time() + self._settle_timeout_s
        async with self._owned():
            while True:
                observations = []
                try:
                    # Only external polling has this timer; recorder storage below
                    # remains under its existing live owner/settlement supervision.
                    with anyio.move_on_at(
                        min(
                            _read_deadline.get(),
                            deadline if wait_first and self._settle_timeout_s else math.inf,
                        )
                    ) as poll:
                        end = datetime.now(UTC)
                        end = end.replace(microsecond=0) + timedelta(seconds=1) + _CLOCK_SKEW
                        for record in records:
                            start = (
                                record.created_at.astimezone(UTC).replace(microsecond=0)
                                - timedelta(seconds=1)
                                - _CLOCK_SKEW
                            )
                            rows = await self._call(
                                partial(
                                    self._adapter.spend_rows, _virtual(record), start=start, end=end
                                ),
                                "gateway read",
                            )
                            if any(row.key_id != record.key_id for row in rows):
                                raise GatewayError("gateway read identity mismatch")
                            observations.extend((record, row) for row in rows)
                    if poll.cancelled_caught:
                        _LOG.warning("gateway spend read timed out")
                        return [], "incomplete"
                except GatewayError:
                    _LOG.warning("gateway spend read failed")
                    return [], "failed"
                if observations or not wait_first or anyio.current_time() >= deadline:
                    break
                await anyio.sleep(min(1.0, max(0, deadline - anyio.current_time())))
            calls: list[LLMCall] = []
            status: GatewayReadStatus = "ok"
            for record, row in observations:
                outcome = _Outcome()
                try:
                    model = _observation_label(row.model, 256)
                    provider = (
                        _observation_label(model.split("/", 1)[0], 64)
                        if "/" in model
                        else "unknown"
                    )
                    call = await record_call(
                        metadata={
                            "run_id": self._run_id,
                            "owner_epoch": str(self._journal.epoch),
                            "goal_id": record.goal_id,
                            "team": record.team,
                            "agent": record.agent,
                            "round": str(record.round),
                            "role": "tool",
                            "candidate_id": record.candidate_id,
                        },
                        model=model,
                        provider=provider,
                        usage=row.usage,
                        cost_usd=row.cost_usd,
                        cache_hit=False,
                        latency_ms=None,
                        status="ok",
                        error=None,
                        request=None,
                        response=None,
                        litellm_call_id=gateway_call_id(row.key_id, row.request_id),
                        tool_call_ids=(),
                        settle=False,
                        _external_observation=True,
                    )
                except Exception as error:
                    outcome.propagate_control(error)
                    _LOG.warning("gateway spend row recording failed")
                    status = "incomplete"
                    continue
                if call is not None:
                    calls.append(call)
            return calls, status

    def restore(self, events: Iterable[Event]) -> None:
        """Rebuild key records from a trusted committed journal snapshot.

        Parameters
        ----------
        events : Iterable of Event
            Committed events for this run. Legacy creation events use their event
            timestamp and epoch; reads include a one-second lower-bound overlap
            and 15 minutes of clock skew on both sides.
            Restore recorder dedup indexes separately before calling ``read``.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(KeyManager.restore).return_annotation is None
        True
        """
        restored: dict[str, KeyRecord] = {}
        for event in events:
            if event.run_id != self._run_id:
                raise ValueError("gateway run binding mismatch")
            if event.type != EventType.GATEWAY_KEY:
                continue
            payload = dict(event.payload)
            action = payload.pop("action", None)
            # The engine stamps every event with its run clock; it is not key state.
            payload.pop("run_elapsed_s", None)
            if action == "created":
                payload.setdefault("run_id", event.run_id)
                payload.setdefault("owner_epoch", event.epoch)
                payload.setdefault("created_at", event.ts)
                record = KeyRecord.model_validate(payload)
                self._check_record(record)
                restored[record.key_id] = record
            elif action == "closed":
                key_id = payload.get("key_id")
                if key_id not in restored:
                    raise ValueError("gateway closure has no creation record")
                record = restored[key_id]
                restored[key_id] = KeyRecord.model_validate(
                    {**record.model_dump(), "closed_at": payload.get("closed_at")}
                )
            elif action not in get_args(GatewayKeyAction):
                raise ValueError("invalid gateway key action")
        self._records = restored

    def records(self, *, goal_id: str | None = None) -> list[KeyRecord]:
        """Return a detached list of known records, optionally filtered by goal.

        Parameters
        ----------
        goal_id : str or None, default=None
            Goal filter; None includes every goal.

        Returns
        -------
        list of KeyRecord
            Immutable records in creation order.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(KeyManager.records).parameters["goal_id"].default is None
        True
        """
        return [
            record
            for record in self._records.values()
            if goal_id is None or record.goal_id == goal_id
        ]
