"""Reconcile gateway observations under existing or taken-over run authority."""

from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any, Final, TypeVar
from uuid import uuid4

import anyio

from swarm_arena._budget.amount import Amount, _micro_units
from swarm_arena._engine.preflight import check_storage
from swarm_arena._engine.resume import _check_gateway_binding, _take_over
from swarm_arena._engine.run import (
    _close_stop_watcher,
    _gateway_deadline,
    _local_journal_path,
    _supervise_initialization,
    _supervise_writer,
)
from swarm_arena._engine.stops import StopState
from swarm_arena._errors import ConfigError, Fenced, LeaseHeld, _core_error
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._gateway.adapter import GatewayAdapter, GatewayError, ProfileReport, SpendRow
from swarm_arena._gateway.keys import _CLOCK_SKEW, KeyManager, KeyRecord, _virtual, gateway_call_id
from swarm_arena._gateway.labels import _observation_label
from swarm_arena._gateway.reports import (
    GatewayKeyAction,
    GatewayKeyFailure,
    KeyReport,
    ReconcileReport,
    _report_usd,
)
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.lease import Lease, OwnerDoc
from swarm_arena._journal.llm_record import _bound_run, record_call
from swarm_arena._journal.reader import _read_segment, read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.ids import IdGen
from swarm_arena._model.results import LLMCall
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.settings import GatewayConfig
from swarm_arena._model.tooling import Usage
from swarm_arena._outcomes import _Outcome
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._store.io import _storage_io, _validated_io

OBSERVED: Final = "observed"
UNVERIFIED: Final = "observed (completeness unverified)"
ESTIMATED: Final = "estimated"
_T = TypeVar("_T")
_ALLOWANCE_REASON: Final = "reconcile time allowance exhausted"


class _AllowanceExpired(GatewayError):
    """Distinguish work that never started from an external operation failure."""


async def _external(operation: Callable[[], Awaitable[_T]], deadline: float) -> _T:
    """Bound external work only; durable recording keeps its owner's lifetime."""
    if anyio.current_time() >= deadline:
        raise _AllowanceExpired(_ALLOWANCE_REASON)
    outcome = _Outcome()
    started = False
    try:
        with anyio.move_on_at(deadline) as wait:
            await anyio.lowlevel.checkpoint()
            started = True
            result = await operation()
        if wait.cancel_called:
            if not started:
                raise _AllowanceExpired(_ALLOWANCE_REASON)
            raise GatewayError("gateway operation timed out")
        return result
    except _AllowanceExpired:
        raise
    except BaseExceptionGroup as error:
        if error.subgroup((KeyboardInterrupt, SystemExit, GeneratorExit)) is not None:
            raise
        outcome.propagate_control(error)
        raise GatewayError("gateway operation failed") from None
    except (Exception, anyio.get_cancelled_exc_class()) as error:
        outcome.propagate_control(error)
        raise GatewayError("gateway operation failed") from None


def _key_event(
    journal: JournalWriter, record: KeyRecord, action: GatewayKeyAction, **fields: Any
) -> None:
    payload = {
        "action": action,
        "key_id": record.key_id,
        "candidate_id": record.candidate_id,
        **fields,
    }
    if action in {"read_failed", "close_failed"}:
        payload = GatewayKeyFailure.model_validate(payload).model_dump(mode="json")
    journal.append(
        EventDraft(
            type=EventType.GATEWAY_KEY,
            goal_id=record.goal_id,
            team=record.team,
            agent=record.agent,
            round=record.round,
            task_id=record.candidate_id,
            payload=payload,
        )
    )


def _signature(model: str, provider: str, usage: Usage, units: int) -> tuple[Any, ...]:
    """Ignore redundant USD, client IDs and metadata when comparing observations."""
    return model, provider, usage.model_dump(exclude={"usd"}), units


def _labels(row: SpendRow) -> tuple[str, str]:
    model = _observation_label(row.model, 256)
    provider = _observation_label(model.split("/", 1)[0], 64) if "/" in model else "unknown"
    return model, provider


def _observations(
    record: KeyRecord, committed: Sequence[LLMCall], rows: Sequence[SpendRow]
) -> tuple[int, list[SpendRow]]:
    """Merge committed and incoming canonical identities before integer summation."""
    known: dict[str, tuple[Any, ...]] = {}
    for call in committed:
        if (
            call.candidate_id != record.candidate_id
            or call.role != "tool"
            or call.reservation_id is not None
            or call.litellm_call_id is None
            or not call.litellm_call_id.startswith("gw:")
        ):
            continue
        if call.status != "ok":
            raise GatewayError("unsupported gateway observation")
        signature = _signature(call.model, call.provider, call.usage, _micro_units(call.cost_usd))
        previous = known.setdefault(call.litellm_call_id, signature)
        if previous != signature:
            raise GatewayError("conflicting gateway observations")
    fresh: list[SpendRow] = []
    for row in rows:
        if row.key_id != record.key_id:
            raise GatewayError("gateway observation identity mismatch")
        try:
            units = Amount(usd=row.cost_usd)._micro_usd
        except ValueError:
            raise GatewayError("unsupported gateway observation") from None
        identity = gateway_call_id(row.key_id, row.request_id)
        signature = _signature(*_labels(row), row.usage, units)
        if identity in known:
            if known[identity] != signature:
                raise GatewayError("conflicting gateway observations")
        else:
            known[identity] = signature
            fresh.append(row)
    return sum(signature[-1] for signature in known.values()), fresh


async def _record_row(record: KeyRecord, row: SpendRow, journal: JournalWriter) -> LLMCall | None:
    model, provider = _labels(row)
    return await record_call(
        metadata={
            "run_id": record.run_id,
            "owner_epoch": str(journal.epoch),
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


async def reconcile_keys(
    events: Sequence[Event],
    *,
    run_id: str,
    journal: JournalWriter,
    adapter: GatewayAdapter,
    executor: Executor,
    gateway: GatewayConfig,
    profile: ProfileReport | None,
    now: Callable[[], datetime] = utc_now,
    _deadline: float | None = None,
    _manager: KeyManager | None = None,
) -> ReconcileReport:
    """Report known keys without changing reservations or inferring missing spend.

    Parameters
    ----------
    events : Sequence of Event
        Captured committed history, including prior external observations.
    run_id : str
        Explicit identity, retained even for an empty history.
    journal : JournalWriter
        Entered, supervised owner with an active recorder binding.
    adapter : GatewayAdapter
        Gateway operations and explicit accounting capability.
    executor : Executor
        Conservative process or dispatch death confirmation.
    gateway : GatewayConfig
        Required duration wait and key policy.
    profile : ProfileReport or None
        Enforced duration and batch interval evidence, or unavailable evidence.
    now : callable, default=utc_now
        Aware wall clock for closure waits and fixed query windows.
    _deadline : float or None, optional
        Private overall external-work deadline, at most the writer settlement
        timeout and capped before inherited, closing and I/O deadlines. Each
        key receives at most request_timeout_s within that bound. It never
        wraps healthy recorder or journal storage.

    _manager : KeyManager or None, optional
        Existing run key owner; maintenance constructs one when absent.

    Returns
    -------
    ReconcileReport
        Per-key observations or full estimated reservations; the shared payload
        is also appended to the journal. A later call can add retained late rows
        for keys not yet deleted; deleted keys reuse committed observations.

    Examples
    --------
    >>> ReconcileReport(run_id="r", keys=[]).spend_usd()
    0.0
    """
    journal._check_open()
    if journal._run_id != run_id or any(event.run_id != run_id for event in events):
        raise _core_error(ConfigError, "run_id")
    overall = _gateway_deadline(journal)
    if _deadline is not None:
        overall = min(overall, _deadline)
    manager = (
        _manager
        if _manager is not None
        else KeyManager(
            adapter, gateway, cap="soft", journal=journal, settle_timeout_s=0, run_id=run_id
        )
    )
    manager.restore(events)
    records = manager.records()
    if len({r.candidate_id for r in records}) != len(records):
        raise _core_error(ValueError, "gateway candidate has multiple keys")
    state = await _validated_io(RunState.from_events, events)
    starts: dict[str, Job.Started] = {}
    closed = {r.key_id: r.closed_at for r in records}
    by_key = {r.key_id: r for r in records}
    deleted: set[str] = set()
    for event in events:
        if event.type in {EventType.TOOL_START, EventType.TOOL_DISPATCH}:
            starts[event.payload["candidate_id"]] = Job.Started.model_validate(
                event.payload["started"]
            )
        elif event.type == EventType.GATEWAY_KEY:
            key_id = event.payload.get("key_id")
            if event.payload.get("action") in {"closed", "deleted"} and (
                not isinstance(key_id, str) or key_id not in by_key
            ):
                raise _core_error(ValueError, "gateway lifecycle has no creation record")
            if event.payload.get("action") == "closed":
                assert isinstance(key_id, str)
                closure = KeyRecord.model_validate(
                    {**by_key[key_id].model_dump(), "closed_at": event.payload["closed_at"]}
                ).closed_at
                if closure is not None:
                    previous = closed[key_id]
                    closed[key_id] = min(previous, closure) if previous is not None else closure
            elif event.payload.get("action") == "deleted":
                assert isinstance(key_id, str)
                deleted.add(key_id)
    reports: list[KeyReport] = []
    for record in records:
        if record.key_id in deleted:
            retained: float | None = None
            retained_reasons: list[str] = []
            try:
                units, _ = _observations(record, list(state.llm_calls.values()), [])
                retained = _report_usd(units)
            except (GatewayError, ValueError):
                retained_reasons.append("unsupported gateway observation")
            reports.append(
                KeyReport(
                    candidate_id=record.candidate_id,
                    key_id=record.key_id,
                    alias=record.alias,
                    reserved_usd=record.max_budget_usd,
                    observed_usd=retained,
                    label=ESTIMATED
                    if retained is None
                    else OBSERVED
                    if adapter.terminal_accounting
                    else UNVERIFIED,
                    reasons=retained_reasons,
                    deleted=True,
                )
            )
            continue
        deadline = min(overall, _gateway_deadline(journal, timeout_s=gateway.request_timeout_s))
        reasons: list[str] = []
        key = _virtual(record)
        try:
            matches = await _external(partial(adapter.key_matches, key, run_id=run_id), deadline)
        except _AllowanceExpired:
            reasons.append(_ALLOWANCE_REASON)
        except GatewayError:
            reasons.append("gateway key identity unavailable")
        else:
            if matches is not True:
                reasons.append(
                    "gateway key identity mismatch"
                    if matches is False
                    else "gateway key identity unavailable"
                )
        if reasons:
            reports.append(
                KeyReport(
                    candidate_id=record.candidate_id,
                    key_id=record.key_id,
                    alias=record.alias,
                    reserved_usd=record.max_budget_usd,
                    observed_usd=None,
                    label=ESTIMATED,
                    reasons=reasons,
                    deleted=False,
                )
            )
            continue
        start = starts.get(record.candidate_id)
        dead: bool | None = True
        if start is not None:
            try:
                dead = await _external(partial(executor.confirm_dead, start), deadline)
            except _AllowanceExpired:
                reasons.append(_ALLOWANCE_REASON)
            except GatewayError:
                dead = None
        if dead is not True:
            reasons.append("tool not confirmed dead")
        closure = closed[record.key_id]
        if closure is None and record.key_id not in deleted:
            try:
                await _external(partial(adapter.block_key, key), deadline)
                closure = await _external(partial(adapter.closed_at, key), deadline)
            except _AllowanceExpired:
                if _ALLOWANCE_REASON not in reasons:
                    reasons.append(_ALLOWANCE_REASON)
            except GatewayError:
                _key_event(journal, record, "close_failed", error="gateway closure unconfirmed")
            else:
                if closure is not None:
                    _key_event(journal, record, "closed", closed_at=closure.isoformat())
                else:
                    _key_event(journal, record, "close_failed", error="gateway closure unconfirmed")
        if closure is None and _ALLOWANCE_REASON not in reasons:
            reasons.append("key closure not confirmed by the gateway")
        if (
            profile is None
            or profile.request_timeout_s is None
            or profile.request_timeout_s > gateway.request_timeout_s
        ):
            reasons.append("gateway maximum request duration not verified")
        interval = None if profile is None else profile.batch_write_interval_s
        if interval is None:
            reasons.append("gateway batch write interval unknown")
        instant = now()
        if (
            closure is not None
            and interval is not None
            and (instant - closure).total_seconds() < gateway.request_timeout_s + interval
        ):
            reasons.append("request_timeout_s has not passed since closure")
        observed: float | None = None
        if not reasons:
            try:
                rows = await _external(
                    partial(
                        adapter.spend_rows,
                        key,
                        start=record.created_at.astimezone(UTC).replace(microsecond=0)
                        - timedelta(seconds=1)
                        - _CLOCK_SKEW,
                        end=instant.astimezone(UTC).replace(microsecond=0)
                        + timedelta(seconds=1)
                        + _CLOCK_SKEW,
                    ),
                    deadline,
                )
            except _AllowanceExpired:
                reasons.append(_ALLOWANCE_REASON)
            except GatewayError:
                reasons.append("gateway spend read failed")
            else:
                try:
                    units, fresh = _observations(record, list(state.llm_calls.values()), rows)
                    observed = _report_usd(units)
                except GatewayError as error:
                    # These messages are constructed locally, never adapter data.
                    reasons.append(str(error))
                except ValueError:
                    reasons.append("unsupported gateway observation")
                else:
                    for row in fresh:
                        if await _record_row(record, row, journal) is None:
                            reasons.append("gateway observation recording incomplete")
                            observed = None
                            break
            if reasons and _ALLOWANCE_REASON not in reasons:
                _key_event(journal, record, "read_failed", error="gateway read incomplete")
            elif not reasons:
                # Publication must cover both closure and observations before
                # deleting their upstream recovery source. This is storage work.
                await journal.commit()
                try:
                    await _external(partial(adapter.delete_key, key), deadline)
                except _AllowanceExpired:
                    reasons.append(_ALLOWANCE_REASON)
                except GatewayError:
                    _key_event(journal, record, "read_failed", error="gateway read incomplete")
                else:
                    deleted.add(record.key_id)
                    _key_event(journal, record, "deleted")
        reports.append(
            KeyReport(
                candidate_id=record.candidate_id,
                key_id=record.key_id,
                alias=record.alias,
                reserved_usd=record.max_budget_usd,
                observed_usd=observed,
                label=ESTIMATED
                if observed is None
                else OBSERVED
                if adapter.terminal_accounting
                else UNVERIFIED,
                reasons=reasons,
                deleted=record.key_id in deleted,
            )
        )
    report = ReconcileReport(run_id=run_id, keys=reports)
    journal.append(EventDraft(type=EventType.RECONCILE, payload=report.model_dump(mode="json")))
    return report


async def take_run(
    config: RunConfig,
    run_id: str,
    *,
    owner_id: str,
    force: bool,
    _stops: StopState | None = None,
) -> tuple[Lease, list[Event], JournalWriter]:
    """Take an expired lease with the shared supervised compare-and-swap helper.

    Parameters
    ----------
    config : RunConfig
        Storage roots and lease settings; conditional writes are probed first.
    run_id : str
        Existing run, including one with an owner but no committed head.
    owner_id : str
        Reconciler's new owner identity.
    force : bool
        Whether to permit takeover of an unexpired owner. Reconcile passes False.
    _stops : StopState or None, optional
        Private retained poller shared with the calling maintenance lifetime.

    Returns
    -------
    tuple of Lease, list of Event, JournalWriter
        Acquired authority, its committed history, and an unentered writer.
        The caller must enter the writer or release the lease on entry failure.

    Examples
    --------
    >>> import inspect
    >>> inspect.iscoroutinefunction(take_run)
    True
    """
    await _validated_io(check_storage, config)
    store = await _storage_io(lambda: open_store(config.run_root).child(run_id))

    def validate(doc: OwnerDoc) -> None:
        events = (
            []
            if doc.head is None
            else [
                event
                for _, batch in read_chain(store, doc.head, _read=partial(_read_segment, store))
                for event in batch
            ]
        )
        state = RunState.from_events(events)
        if events and state.run_id != run_id:
            raise _core_error(ConfigError, "run_id")
        _check_gateway_binding(state, config)

    stops = StopState() if _stops is None else _stops
    lease: Lease | None = None
    delivered = False
    hard = False
    try:
        lease = await _take_over(
            config,
            run_id,
            store,
            force=force,
            review=True,
            stops=stops,
            allow_empty_head=True,
            owner_id=owner_id,
            validate=validate,
        )
        head = lease.doc.head
        writer = JournalWriter(
            store,
            lease,
            run_id=run_id,
            next_seq=1 if head is None else head.seq + 1,
            local_path=_local_journal_path(run_id, lease.doc.epoch),
            sync_interval_s=config.sync_interval_s,
            settle_timeout_s=config.settle_timeout_s,
        )
        events: list[Event] = []
        completed = False
        async with _supervise_initialization(
            store, lease.doc.epoch, stops, None, config, retain=True
        ):
            if head is not None:
                chain = await writer._retry(
                    partial(read_chain, store, head, _read=partial(_read_segment, store)),
                    validated=True,
                )
                events = [event for _, batch in chain for event in batch]
            state = await _validated_io(RunState.from_events, events)
            if events and state.run_id != run_id:
                raise _core_error(ConfigError, "run_id")
            _check_gateway_binding(state, config)
            completed = True
        if not completed:
            raise _core_error(RuntimeError, "reconcile stopped")
        delivered = True
        return lease, events, writer
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        hard = True
        raise
    except BaseExceptionGroup as error:
        hard = error.subgroup((KeyboardInterrupt, SystemExit, GeneratorExit)) is not None
        raise
    finally:
        if not hard:
            if lease is not None and not delivered:
                with anyio.move_on_after(config.settle_timeout_s, shield=True):
                    with suppress(Fenced):
                        await _storage_io(lease.release)
            if _stops is None or not delivered:
                await _close_stop_watcher(stops)


async def reconcile_run(
    config: RunConfig, run_id: str, adapter: GatewayAdapter, executor: Executor
) -> ReconcileReport:
    """Reconcile a stopped run under its lease, without reviving paid work.

    Parameters
    ----------
    config : RunConfig
        Run storage, gateway, and settlement settings.
    run_id : str
        Existing run identifier.
    adapter : GatewayAdapter
        Gateway management adapter; preflight failure leaves reports estimated.
    executor : Executor
        Executor for conservative death evidence. Its caller owns shutdown.

    Returns
    -------
    ReconcileReport
        Committed observations; reservations and historical run time are unchanged.

    Raises
    ------
    RuntimeError
        A live owner holds the lease or reconciliation was stopped.

    Examples
    --------
    >>> import inspect
    >>> inspect.iscoroutinefunction(reconcile_run)
    True
    """
    gateway = config.llm_gateway
    if gateway is None:
        raise _core_error(ConfigError, "llm_gateway")
    stops = StopState()
    try:
        lease, events, writer = await take_run(
            config, run_id, owner_id=uuid4().hex, force=False, _stops=stops
        )
    except LeaseHeld:
        raise _core_error(
            RuntimeError,
            f"run {run_id} has a live owner; retry after its lease expires",
        ) from None
    entered = False
    hard = False
    report: ReconcileReport | None = None
    try:
        if lease.doc.head is None:
            return ReconcileReport(run_id=run_id, keys=[])
        store = lease._store
        async with (
            _supervise_initialization(store, lease.doc.epoch, stops, None, config, retain=True),
            writer,
        ):
            entered = True
            async with _supervise_writer(writer):
                writer.append(EventDraft(type=EventType.TAKEOVER, payload={"force": False}))
                state = await _validated_io(RunState.from_events, events)
                artifacts = await _storage_io(
                    lambda: ArtifactStore(open_store(config.artifact_root))
                )
                async with _bound_run(
                    run_id,
                    journal=writer,
                    ids=IdGen(run_id, lease.doc.epoch, counters=state.id_counters),
                    artifacts=artifacts,
                    config=config,
                    stops=stops,
                    clock=None,
                    state=state,
                    _maintenance_lease=lease,
                ):
                    deadline = _gateway_deadline(writer)
                    try:
                        profile = await _external(
                            adapter.preflight,
                            min(
                                deadline,
                                _gateway_deadline(writer, timeout_s=gateway.request_timeout_s),
                            ),
                        )
                    except GatewayError:
                        profile = None
                    report = await reconcile_keys(
                        events,
                        run_id=run_id,
                        journal=writer,
                        adapter=adapter,
                        executor=executor,
                        gateway=gateway,
                        profile=profile,
                        _deadline=deadline,
                    )
                    await writer.commit()
        if report is None or stops.stop or stops.stop_now:
            raise _core_error(RuntimeError, "reconcile stopped")
        return report
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        hard = True
        raise
    except BaseExceptionGroup as error:
        hard = error.subgroup((KeyboardInterrupt, SystemExit, GeneratorExit)) is not None
        raise
    finally:
        if not hard:
            if not entered:
                with anyio.move_on_after(config.settle_timeout_s, shield=True):
                    with suppress(Fenced):
                        await _storage_io(lease.release)
            await _close_stop_watcher(stops)
