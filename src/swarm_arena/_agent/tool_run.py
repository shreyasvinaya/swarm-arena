"""Validate, reserve, execute, and durably publish one agent tool request."""

import asyncio
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

import anyio

from swarm_arena._agent.manager import LimitReached
from swarm_arena._agent.repeats import _is_deterministic, repeat_key
from swarm_arena._agent.runtime import AgentRuntime
from swarm_arena._board.leaderboard import format_value
from swarm_arena._budget.amount import USD_LIMIT, Amount
from swarm_arena._budget.ledger import Reservation
from swarm_arena._budget.llm_scope import current_requesting_call
from swarm_arena._errors import (
    ConfigError,
    Fenced,
    SoftError,
    _core_error,
    _core_signal,
    _exception_type_name,
    _is_core_signal,
)
from swarm_arena._exec.env import tool_env_vars
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.a2a import agent_task_id, task_to_json, tool_run_task
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.data import check_utf8
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import _validate_config
from swarm_arena._model.results import Candidate, CheckResult
from swarm_arena._model.tooling import RunStatus, ToolOutput, ToolRequest, parse_logs_ref
from swarm_arena._outcomes import _classify_outcome, _has_core_fenced, _Outcome
from swarm_arena._scoring.check import CheckContext, _prepare_command_output
from swarm_arena._scoring.runner import run_checks
from swarm_arena._store.base import open_store
from swarm_arena._store.io import _create, _storage_io
from swarm_arena._tooling.result_file import request_json
from swarm_arena._tooling.staging import (
    parent_ref,
    plan_tool_stage,
    put_edits,
    tool_root,
    validate_edits,
)
from swarm_arena._tooling.tool import effective_tool_timeout, tool_grace

FINGERPRINT_TIMEOUT_S = 10.0
"""Maximum seconds to await the optional fingerprint hook before recording null."""


class Inflight:
    """
    Track cancellable tool submissions and latch closed round admission.

    Attributes
    ----------
    closed : bool
        Whether new reservations must be refused. Budget closure still permits
        previously reserved submissions to start until drain grace expires.

    Examples
    --------
    >>> registry = Inflight()
    >>> registry.close()
    >>> registry.closed
    True
    """

    def __init__(self) -> None:
        """Start with open admission and no submissions."""
        self.closed = False
        self._starts_closed = False
        self._cancel_generation = 0
        self._scopes: dict[anyio.CancelScope, float] = {}

    def add(
        self,
        scope: anyio.CancelScope,
        *,
        grace_s: float = 0.0,
        admitted_generation: int | None = None,
    ) -> None:
        """
        Admit a submission unless the round has closed.

        Parameters
        ----------
        scope : anyio.CancelScope
            Scope controlling one executor submission.
        grace_s : float, default=0.0
            Effective job grace in seconds; tracked from reservation admission
            through executor collection, including durable reservation writes.
        admitted_generation : int or None, default=None
            Cancellation generation of a reserved action allowed to drain after
            budget closure. Other closure kinds still refuse its start.

        Raises
        ------
        LimitReached
            If admission has closed.

        Examples
        --------
        >>> import anyio
        >>> async def demo() -> bool:
        ...     registry = Inflight()
        ...     with anyio.CancelScope() as scope:
        ...         registry.add(scope)
        ...         registry.discard(scope)
        ...     return registry.closed
        >>> anyio.run(demo)
        False
        """
        if admitted_generation is not None:
            self._check_admitted(admitted_generation)
        elif self.closed:
            raise _core_signal(LimitReached("stop", "the round is stopping"))
        self._scopes[scope] = grace_s

    def discard(self, scope: anyio.CancelScope) -> None:
        """
        Remove a submission, including one already removed.

        Parameters
        ----------
        scope : anyio.CancelScope
            Submission scope to remove.

        Examples
        --------
        The :meth:`add` example admits and discards a scope in memory.
        """
        self._scopes.pop(scope, None)

    def max_grace_s(self) -> float:
        """
        Return the largest grace of currently admitted submissions.

        Returns
        -------
        float
            Maximum job grace, or zero when idle. Closing or cancelling does not
            remove entries; the executor retains its collection allowance after
            grace, and discard occurs only when submit exits.

        Examples
        --------
        >>> Inflight().max_grace_s()
        0.0
        """
        return max(self._scopes.values(), default=0.0)

    def close(self, *, budget: bool = False) -> None:
        """
        Refuse future admission without cancelling existing work.

        Parameters
        ----------
        budget : bool, default=False
            Let already reserved actions start during budget drain. A later
            non-budget close permanently revokes this allowance.

        Examples
        --------
        >>> registry = Inflight()
        >>> registry.close()
        >>> registry.closed
        True
        """
        self.closed = True
        self._starts_closed |= not budget

    def _check_admitted(self, generation: int) -> None:
        """Refuse reserved starts after hard closure or cancellation."""
        if self._starts_closed or generation != self._cancel_generation:
            raise _core_signal(LimitReached("stop", "the round is stopping"))

    def cancel_all(self) -> None:
        """
        Cancel current submissions without changing the admission latch.

        Examples
        --------
        >>> registry = Inflight()
        >>> registry.cancel_all()
        >>> registry.closed
        False
        """
        self._cancel_generation += 1
        for scope in tuple(self._scopes):
            scope.cancel()

    def _add_completion(self, scope: anyio.CancelScope, *, grace_s: float, generation: int) -> bool:
        """Supervise completion checks even after a graceful admission close."""
        if generation != self._cancel_generation:
            return False
        self._scopes[scope] = grace_s
        return True


_inflight: ContextVar[Inflight | None] = ContextVar("tool_inflight", default=None)


@contextmanager
def track_inflight(registry: Inflight) -> Generator[None]:
    """
    Bind a round's submission registry to this task and its child tasks.

    Parameters
    ----------
    registry : Inflight
        Shared admission and cancellation registry.

    Yields
    ------
    None
        Restores the prior registry on exit.

    Examples
    --------
    >>> with track_inflight(Inflight()):
    ...     pass
    """
    token = _inflight.set(registry)
    try:
        yield
    finally:
        _inflight.reset(token)


class _CommandJournal(Executor):
    """Forward command jobs and record successful returns in their journal."""

    def __init__(self, journal: JournalWriter, executor: Executor) -> None:
        super().__init__()
        self._journal = journal
        self._executor = executor

    async def submit(self, job: Job) -> ToolOutput:
        output = await self._executor.submit(job)
        output = await _prepare_sync(_prepare_command_output, output)
        self._journal.append(
            EventDraft(
                type=EventType.COMMAND_END,
                task_id=job.id,
                payload={"job_id": job.id, "status": output.status.value},
            )
        )
        return output

    async def capacity(self) -> Resources | None:
        return await self._executor.capacity()

    async def aclose(self) -> None:
        await self._executor.aclose()


def command_journal(
    journal: JournalWriter, executor: Executor, *, candidate_id: str, component: str
) -> tuple[Executor, Callable[[Job.Started], Awaitable[None]]]:
    """
    Wrap command completion and provide a durable start callback.

    Parameters
    ----------
    journal : JournalWriter
        Writer owning the run lease.
    executor : Executor
        Underlying command executor.
    candidate_id : str
        Candidate being checked or judged.
    component : str
        Plan name of the check or judge.

    Returns
    -------
    tuple[Executor, Callable[[Job.Started], Awaitable[None]]]
        Executor wrapper and the callback to place on command jobs.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> wrapped, started = command_journal(Mock(), Mock(), candidate_id="c1", component="check")
    >>> isinstance(wrapped, Executor), callable(started)
    (True, True)
    """
    inflight = _inflight.get()
    generation = None if inflight is None else inflight._cancel_generation

    async def on_started(started: Job.Started) -> None:
        started = await _prepare_sync(_prepare_started, started, generation=generation)
        await journal.append_durable(
            EventDraft(
                type=EventType.COMMAND_START,
                task_id=started.job_id,
                payload={
                    "job_id": started.job_id,
                    "started": started.model_dump(mode="json"),
                    "candidate_id": candidate_id,
                    "component": component,
                },
            )
        )

    return _CommandJournal(journal, executor), on_started


def _draft(rt: AgentRuntime, type: EventType, **payload: Any) -> EventDraft:
    """Stamp agent context without resolving or copying secrets."""
    return EventDraft(
        type=type,
        goal_id=rt.goal_id,
        team=rt.team.name,
        agent=rt.spec.name,
        round=rt.round,
        payload=payload,
    )


def _prepare_started(value: object) -> Job.Started:
    """Detach executor start-event serializers before touching the journal."""
    if not isinstance(value, Job.Started):
        raise TypeError("invalid started event")
    return _detach_model(Job.Started, value)


def _prepare_output(value: object) -> ToolOutput:
    """Detach executor output serializers and validate before core completion."""
    if not isinstance(value, ToolOutput):
        raise TypeError("invalid tool output")
    return _detach_model(ToolOutput, value)


_T = TypeVar("_T")


def _request_hook(hook: Callable[..., _T], *args: Any) -> _T:
    """Contain user hook failures separately from safe core validation errors."""
    try:
        return hook(*args)
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        error = _classify_outcome(error)
        raise _core_signal(
            SoftError(f"invalid tool request: {_exception_type_name(error)}")
        ) from None


async def _prepare_sync(
    hook: Callable[..., _T],
    *args: Any,
    generation: int | None = None,
    storage: bool = False,
    admitted_generation: int | None = None,
) -> _T:
    """Supervise preparation; admitted commands may finish during graceful stop."""
    inflight = _inflight.get()
    with anyio.CancelScope() as scope:
        if inflight is not None:
            if generation is None:
                inflight.add(scope, grace_s=0, admitted_generation=admitted_generation)
            elif not inflight._add_completion(scope, grace_s=0, generation=generation):
                raise _core_signal(LimitReached("stop", "the round is stopping"))
        outcome = _Outcome()
        try:
            result = await (_storage_io(hook, *args) if storage else run_sync_daemon(hook, *args))
        except BaseException as error:
            raise outcome.classify(error) from None
        finally:
            if inflight is not None:
                inflight.discard(scope)
    if scope.cancelled_caught:
        raise _core_signal(LimitReached("stop", "the round is stopping"))
    return result


def _parents(rt: AgentRuntime, parents: Sequence[str]) -> list[Candidate]:
    """Resolve visible parents exclusively on the board's owning loop."""
    resolved: list[Candidate] = []
    if len(set(parents)) != len(parents):
        raise _core_signal(SoftError("duplicate parent ids"))
    for id in parents:
        candidate = next((c for c in rt.own if isinstance(c, Candidate) and c.id == id), None)
        if candidate is None:
            try:
                candidate = rt.view.get_candidate(id)
            except (KeyError, ValueError):
                raise _core_signal(SoftError(f"unknown parent: {id}")) from None
        resolved.append(candidate.model_copy(deep=True))
    return resolved


def _request(
    rt: AgentRuntime,
    inputs: Mapping[str, Any],
    parents: Sequence[Candidate],
    replicate: int | None,
    edits: Mapping[str, str] | None,
) -> ToolRequest:
    """Validate inputs and visible parents before any budget or count change."""
    model = _request_hook(rt.tool.get_input_model)
    try:
        validated = _validate_config(model, dict(inputs), where="tool inputs")
        return ToolRequest(
            inputs=validated.model_dump(mode="json"),
            parents=[parent_ref(c) for c in parents],
            round=rt.round,
            round_overrides=rt.round_spec.tool_overrides,
            replicate=0 if replicate is None else replicate,
            edits=dict(edits or {}),
        )
    except ConfigError as error:
        raise _core_signal(SoftError(str(error))) from None
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        error = _classify_outcome(error)
        raise _core_signal(
            SoftError(f"invalid tool request: {_exception_type_name(error)}")
        ) from None


def _stage_settings(
    rt: AgentRuntime,
    request: ToolRequest,
) -> tuple[dict[str, str], Sequence[str], frozenset[str]]:
    """Validate root and edit permissions, rebuilding core refusal snapshots."""
    try:
        files = dict(_request_hook(rt.tool._own_files))
        root = tool_root(rt.goal.files.manifest, files)
        protected = frozenset(_request_hook(rt.tool._protected, root.keys())) | rt.private_paths
        editable = tuple(_request_hook(rt.tool.editable))
        if (
            any(type(path) is not str or type(digest) is not str for path, digest in files.items())
            or any(type(path) is not str for path in protected)
            or any(type(path) is not str for path in editable)
        ):
            raise TypeError("invalid stage settings")
        validate_edits(
            request.edits,
            editable=editable,
            protected=protected,
            root=root.keys(),
            max_edit_bytes=rt.limits.max_edit_bytes,
        )
        return files, editable, protected
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        error = _classify_outcome(error)
        if _is_core_signal(error) and isinstance(error, (SoftError, LimitReached)):
            raise error from None
        raise _core_signal(
            SoftError(f"invalid tool stage: {_exception_type_name(error)}")
        ) from None


def _prepare_request(
    rt: AgentRuntime,
    inputs: Mapping[str, Any],
    parents: Sequence[Candidate],
    replicate: int | None,
    edits: Mapping[str, str] | None,
) -> tuple[ToolRequest, tuple[dict[str, str], Sequence[str], frozenset[str]], bool]:
    """Return core request data and detached settings from synchronous user hooks."""
    request = _request(rt, inputs, parents, replicate, edits)
    settings = _stage_settings(rt, request)
    deterministic = _request_hook(_is_deterministic, rt.tool, request.model_copy(deep=True))
    return request, settings, deterministic


def _stage(
    rt: AgentRuntime,
    request: ToolRequest,
    settings: tuple[dict[str, str], Sequence[str], frozenset[str]],
) -> tuple[Job.Stage, dict[str, str], list[str]]:
    """Plan before count/reserve and expose only fresh core refusal snapshots."""
    files, editable, protected = settings
    try:
        return plan_tool_stage(
            goal_files=rt.goal.files.manifest,
            tool_files=files,
            editable=editable,
            protected=protected,
            private_paths=rt.private_paths,
            parents=request.parents,
            receiving_identity=rt.tool_identity,
            edits=request.edits,
            request_json=request_json(request),
        )
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        error = _classify_outcome(error)
        if _is_core_signal(error) and isinstance(error, (SoftError, LimitReached)):
            raise error from None
        raise _core_signal(
            SoftError(f"invalid tool stage: {_exception_type_name(error)}")
        ) from None


async def _logs(rt: AgentRuntime, id: str, ref: str) -> str:
    """Read and publish each log through a separate supervised storage boundary."""
    stdout_sha, stderr_sha = parse_logs_ref(ref)
    stdout = await _storage_io(rt.artifacts.get_bytes, stdout_sha)
    stderr = await _storage_io(rt.artifacts.get_bytes, stderr_sha)
    store = await _storage_io(lambda: open_store(rt.config.run_root).child(rt.repeats.run_id))
    stem = f"goals/{rt.goal_id}/logs/{id}"
    await _storage_io(_create, store, stem + ".out", stdout)
    await _storage_io(_create, store, stem + ".err", stderr)
    return stem


@dataclass
class _ToolReservation:
    """Retain admission ownership until dispatch or normal completion."""

    admitted: Reservation | None = None
    started: bool = False


async def _refund_undispatched(rt: AgentRuntime, owner: _ToolReservation) -> None:
    """Book zero once before dispatch, bounding even the ledger's inner shield."""
    if owner.admitted is None or owner.started or rt.journal.fenced.is_set():
        return
    with anyio.CancelScope(shield=True):
        try:
            # Native timeout cancellation crosses the ledger durability shield.
            # The writer retains any admitted settlement after this waiter exits.
            async with asyncio.timeout(rt.config.settle_timeout_s):
                await rt.ledger.settle(owner.admitted.id, Amount())
        except TimeoutError:
            pass


async def execute_tool_run(
    rt: AgentRuntime,
    inputs: Mapping[str, Any],
    *,
    parents: Sequence[str],
    replicate: int | None,
    edits: Mapping[str, str] | None,
    read_ids: Sequence[str] = (),
) -> Candidate:
    """
    Execute one validated request with durable reservation and completion.

    User hooks, validators, and serializers run in abandonable daemon threads.
    Detached preparation precedes atomic repeat and run-count admission.
    Internal round cancellation retains partial output; outer cancellation
    before completion propagates. Optional fingerprinting receives a deep copy
    and waits at most ten seconds outside the completion shield. Interrupted
    fingerprinting records null and completes settlement while still owner.
    Fencing prevents completion and settlement. Connector failures become
    error candidates with type-only warnings.
    Executor outputs are detached under round supervision before completion.
    Partial output delivered after cancellation gets at most the configured
    settlement allowance for detachment; interruption produces a safe error.

    Parameters
    ----------
    rt : AgentRuntime
        Loop-owned agent services and round state.
    inputs : Mapping[str, Any]
        Inputs validated against the effective tool's input model.
    parents : Sequence[str]
        Ordered visible parent candidate identifiers. Runtime seed_parents are
        prepended; one stable deduplication preserves the first occurrence.
    replicate : int or None
        Requested nondeterministic replicate, or automatic allocation.
    edits : Mapping[str, str] or None
        Live editable-file contents.
    read_ids : Sequence[str], default=()
        Read identifiers snapshotted at action entry, stamped before journaling
        and indexing. Linked results retain their original read_ids unchanged;
        resumed results keep their journaled provenance.

    Returns
    -------
    Candidate
        Linked or recovered result, or a newly committed candidate.

    Raises
    ------
    SoftError
        Invalid request, stage, or refused repeat.
    LimitReached
        Run, repeat, budget, or round admission limit reached.
    Fenced
        The writer lost ownership; no further writes are permitted.

    Examples
    --------
    >>> candidate = await execute_tool_run(rt, {"x": 1.0}, parents=(),
    ...     replicate=None, edits=None)  # doctest: +SKIP
    """
    parents = tuple(dict.fromkeys((*rt.seed_parents, *parents)))
    request, settings, deterministic = await _prepare_sync(
        _prepare_request, rt, inputs, _parents(rt, parents), replicate, edits
    )
    allocation_request = request
    index: int | None = None
    try:
        if not deterministic:
            index = rt.repeats.next_replicate(rt, request, requested=replicate)
            request = request.model_copy(update={"replicate": index})
        stage, overlay, warnings = _stage(rt, request, settings)
        key = repeat_key(
            goal_key=rt.goal_key, tool_ref=rt.tool_ref, request=request, deterministic=deterministic
        )
        outcome = rt.repeats.lookup(
            key,
            agent=f"{rt.team.name}/{rt.spec.name}",
            deterministic=deterministic,
            resumed={candidate.id for candidate in rt.resumed},
        )
        if outcome.action == "refuse":
            assert outcome.candidate is not None
            rt.watchdog.record_refusal()
            rt.journal.append(_draft(rt, EventType.WATCHDOG, refused=outcome.candidate.id))
            if rt.watchdog.reason is not None:
                raise _core_signal(LimitReached("max_refused_repeats", rt.watchdog.reason))
            raise _core_signal(SoftError(f"repeat refused: {outcome.candidate.id}"))
        if outcome.action in {"return_own", "link"}:
            assert outcome.candidate is not None
            return outcome.candidate
        if rt.counters.tool_runs >= rt.limits.max_tool_runs_per_round:
            raise _core_signal(
                LimitReached("max_tool_runs_per_round", "tool runs used up for this round")
            )
        rt.counters.tool_runs += 1
        rt.journal.append(
            _draft(
                rt,
                EventType.AGENT_COUNTERS,
                turns=rt.counters.turns,
                board_reads=rt.counters.board_reads,
                tool_runs=rt.counters.tool_runs,
            ).model_copy(update={"task_id": agent_task_id(rt.address, rt.round)})
        )
        if outcome.hint is not None:
            judge = rt.rankings.get(rt.part or "")
            scores = [score for score in rt.view.scores(outcome.hint.id) if score.judge == judge]
            detail = f"{judge}={format_value(scores[-1].value)}" if scores else "unscored"
            warnings.append(f"past-run match {outcome.hint.id}: {detail}")

        owner = _ToolReservation()
        try:
            with anyio.CancelScope() as admission_scope:
                try:
                    return await _run(
                        rt,
                        request,
                        key,
                        stage,
                        overlay,
                        warnings,
                        tuple(read_ids),
                        owner,
                        admission_scope,
                    )
                finally:
                    inflight = _inflight.get()
                    if inflight is not None:
                        inflight.discard(admission_scope)
            raise _core_signal(LimitReached("stop", "the round is stopping"))
        except BaseException as error:
            if not _has_core_fenced(error):
                await _refund_undispatched(rt, owner)
            raise
    finally:
        if index is not None:
            rt.repeats.release(rt, allocation_request, index)


async def _run(
    rt: AgentRuntime,
    request: ToolRequest,
    key: str,
    stage: Job.Stage,
    overlay: dict[str, str],
    warnings: list[str],
    read_ids: Sequence[str],
    owner: _ToolReservation,
    admission_scope: anyio.CancelScope,
) -> Candidate:
    """Own reservation, executor lifecycle, and the shielded completion sequence."""
    id = rt.ids.next("c")
    inflight = _inflight.get()
    if inflight is None:
        raise _core_error(RuntimeError, "tool run requires track_inflight")
    generation = inflight._cancel_generation
    outcome = _Outcome()
    started = False
    admission_refusal: LimitReached | None = None
    partial: list[ToolOutput] = []
    output = ToolOutput(status=RunStatus.ERROR)
    amount = Amount.from_estimate(rt.limits.default_run_reservation)
    providers = False
    failed = False

    def estimate() -> tuple[Amount, bool]:
        value = rt.tool.estimate(request)
        return (
            amount if value is None else Amount.from_estimate(value),
            bool(rt.tool.providers()),
        )

    try:
        amount, providers = await _prepare_sync(estimate)
    except BaseException as error:
        error = outcome.classify(error)
        if _is_core_signal(error) and isinstance(error, LimitReached):
            raise error from None
        warnings.append(f"tool error: {_exception_type_name(error)}")
        failed = True
    if inflight.closed or generation != inflight._cancel_generation:
        raise _core_signal(LimitReached("stop", "the round is stopping"))

    grace_s = tool_grace(rt.tool)

    def on_admit(reservation: Reservation) -> None:
        # The ledger calls this synchronously after booking, before durability.
        if not failed:  # failed estimation already rules out dispatch
            inflight.add(admission_scope, grace_s=grace_s, admitted_generation=generation)
        owner.admitted = reservation

    res = await rt.ledger.reserve(
        rt.holder, amount, kind="tool", goal_id=rt.goal_id, on_admit=on_admit
    )
    await anyio.lowlevel.checkpoint()

    async def on_started(event: Job.Started) -> None:
        nonlocal started, admission_refusal
        try:
            event = await _prepare_sync(_prepare_started, event, admitted_generation=generation)
            inflight._check_admitted(generation)
        except LimitReached as error:
            admission_refusal = error
            raise
        await rt.journal.append_durable(
            _draft(
                rt,
                EventType.TOOL_START,
                candidate_id=id,
                reservation_id=res.id,
                started=event.model_dump(mode="json"),
            ).model_copy(update={"task_id": id})
        )
        started = owner.started = True
        rt.started_runs += 1

    async def on_partial(value: ToolOutput) -> None:
        partial.append(value)

    try:
        if not failed:
            for edit in [{path: content} for path, content in request.edits.items()] or [{}]:
                try:
                    await _prepare_sync(
                        put_edits, edit, rt.artifacts, storage=True, admitted_generation=generation
                    )
                except LimitReached as error:
                    if inflight.closed or generation != inflight._cancel_generation:
                        admission_refusal = error
                    raise
            job = Job(
                id=id,
                kind="tool",
                tool=rt.execution_ref,
                request=request,
                stage=stage,
                timeout_s=effective_tool_timeout(
                    rt.tool,
                    round_timeout_s=rt.round_spec.tool_timeout_s,
                    limit_s=rt.limits.tool_timeout_s,
                ),
                grace_s=grace_s,
                resources=rt.spec.resources,
                env=tool_env_vars(
                    run_id=rt.repeats.run_id,
                    goal_id=rt.goal_id,
                    candidate_id=id,
                    team=rt.team.name,
                    agent=rt.spec.name,
                    round=rt.round,
                ),
                secret_env=(),
                secrets_file=rt.config.secrets.file,
                artifact_root=rt.config.artifact_root,
                on_started=on_started,
                on_partial=on_partial,
            )
            with anyio.CancelScope() as run_scope:
                try:
                    inflight.add(run_scope, grace_s=job.grace_s, admitted_generation=generation)
                    inflight.discard(admission_scope)
                except LimitReached as error:
                    admission_refusal = error
                    raise
                try:
                    output = await rt.executor.submit(job)
                finally:
                    inflight.discard(run_scope)
            if run_scope.cancelled_caught:
                output = partial[0] if partial else ToolOutput(status=RunStatus.CANCELLED)
    except BaseException as error:
        inflight.discard(admission_scope)
        refused = error is admission_refusal
        error = outcome.classify(error)
        if refused and _is_core_signal(error) and isinstance(error, LimitReached) and not started:
            raise error from None
        warnings.append(f"tool error: {_exception_type_name(error)}")
        output = ToolOutput(status=RunStatus.ERROR)

    # A stopped executor may still deliver partial output during collection.
    # Give that data a bounded detachment phase, outside all completion shields.
    deadline = (
        anyio.current_time() + rt.config.settle_timeout_s
        if generation != inflight._cancel_generation
        else float("inf")
    )
    with anyio.CancelScope(deadline=deadline) as output_scope:
        inflight._add_completion(output_scope, grace_s=0, generation=inflight._cancel_generation)
        try:
            output = await run_sync_daemon(_prepare_output, output)
        except BaseException as error:
            outcome.propagate_control(error)
            warnings.append("tool error: invalid output")
            output = ToolOutput(status=RunStatus.ERROR)
        finally:
            inflight.discard(output_scope)
    if output_scope.cancelled_caught:
        warnings.append(f"tool error: {anyio.get_cancelled_exc_class().__name__}")
        output = ToolOutput(status=RunStatus.ERROR)

    # Deliver an outer cancellation before accounting and supervised log publication.
    await anyio.lowlevel.checkpoint()
    if rt.journal.fenced.is_set():
        raise _core_error(Fenced, "run ownership lost")
    actual = Amount()
    estimated = False
    usage = output.usage
    overflow = usage.usd is not None and usage.usd > USD_LIMIT
    if overflow:
        warnings.append("tool error: usage exceeds supported range")
        output = output.model_copy(update={"status": RunStatus.ERROR})
        actual = Amount(tokens=max(0, usage.total_tokens), usd=USD_LIMIT)
    else:
        try:
            reported = Amount(tokens=usage.total_tokens, usd=usage.usd or 0.0)
            if started:
                actual = reported
                estimated = providers and usage.usd is None and usage.total_tokens == 0
        except BaseException as error:
            error = outcome.classify(error)
            warnings.append(f"tool error: {_exception_type_name(error)}")
            output = ToolOutput(status=RunStatus.ERROR)
            actual, estimated = Amount(), started and providers
    if output.logs_ref is not None:
        logs_ref = None
        deadline = (
            anyio.current_time() + rt.config.settle_timeout_s
            if generation != inflight._cancel_generation
            else float("inf")
        )
        with anyio.CancelScope(deadline=deadline) as logs_scope:
            inflight._add_completion(logs_scope, grace_s=0, generation=inflight._cancel_generation)
            try:
                logs_ref = await _logs(rt, id, output.logs_ref)
            except BaseException as error:
                outcome.propagate_control(error)
                warnings.append(f"tool error: {_exception_type_name(error)}")
            finally:
                inflight.discard(logs_scope)
        if logs_ref is None:
            output = output.model_copy(update={"status": RunStatus.ERROR, "logs_ref": None})
        else:
            output = output.model_copy(update={"logs_ref": logs_ref})
    # Optional fingerprint work must remain cancellable before durable completion.
    fingerprint: str | None = None
    if output.status == RunStatus.OK or (
        output.partial and output.status not in {RunStatus.ERROR, RunStatus.LOST}
    ):
        try:
            with anyio.fail_after(FINGERPRINT_TIMEOUT_S):
                value = await run_sync_daemon(rt.tool.fingerprint, output.model_copy(deep=True))
                if not isinstance(value, str):
                    raise TypeError("invalid fingerprint")
                fingerprint = check_utf8(value)
        except anyio.get_cancelled_exc_class() as error:
            # The tool has completed: abandon optional fingerprint work, then
            # finish its shielded publication/settlement. The caller's scope
            # still owns cancellation; the ownership check below still wins.
            warnings.append(f"fingerprint error: {_exception_type_name(error)}")
        except BaseException as error:
            error = outcome.classify(error)
            warnings.append(f"fingerprint error: {_exception_type_name(error)}")
    if rt.journal.fenced.is_set():
        raise _core_error(Fenced, "run ownership lost")
    candidate = Candidate(
        id=id,
        run_id=rt.repeats.run_id,
        goal_id=rt.goal_id,
        part=rt.part,
        team=rt.team.name,
        agent=rt.spec.name,
        round=rt.round,
        fidelity=rt.fidelity,
        tool_identity=rt.tool_identity,
        request=request,
        output=output,
        parent_ids=[parent.candidate_id for parent in request.parents],
        overlay=overlay,
        requested_by_call=current_requesting_call(),
        read_ids=list(read_ids),
        warnings=warnings,
    )
    checks = []
    checks_to_run = () if not started and output.status == RunStatus.CANCELLED else rt.checks
    for name, check in checks_to_run:
        executor, on_command_started = command_journal(
            rt.journal,
            rt.executor,
            candidate_id=id,
            component=name,
        )
        context = CheckContext(
            goal=rt.goal,
            executor=executor,
            artifact_root=rt.config.artifact_root,
            run_id=rt.repeats.run_id,
            round=rt.round,
            part=rt.part,
            on_started=on_command_started,
            secrets_file=rt.config.secrets.file,
            ids=rt.ids,
            tool_timeout_s=rt.limits.tool_timeout_s,
        )
        interrupted = False
        try:
            with anyio.CancelScope() as check_scope:
                if not inflight._add_completion(
                    check_scope, grace_s=tool_grace(rt.tool), generation=generation
                ):
                    interrupted = True
                else:
                    try:
                        checks.extend(await run_checks(candidate, [(name, check)], context))
                    finally:
                        inflight.discard(check_scope)
            interrupted |= check_scope.cancelled_caught
        except anyio.get_cancelled_exc_class():
            interrupted = True
        if interrupted:
            checks.append(CheckResult(check=name, passed=False, reasons=["check interrupted"]))
    with anyio.CancelScope(shield=True):
        if rt.journal.fenced.is_set():
            raise _core_error(Fenced, "run ownership lost")
        candidate = candidate.model_copy(update={"checks": checks})
        # Python-form validation sees non-finite data before JSON could turn it into null.
        Candidate.model_validate(candidate.model_dump(mode="python", warnings=False))
        rt.watchdog.record_run(
            candidate.output.status,
            passed=not candidate.gated_out and candidate.output.status == RunStatus.OK,
        )
        end = _draft(
            rt,
            EventType.TOOL_END,
            candidate=candidate.model_dump(mode="json"),
            repeat_key=key,
            fingerprint=fingerprint,
            a2a=task_to_json(tool_run_task(candidate, context_id=rt.repeats.run_id, at=utc_now())),
        ).model_copy(update={"task_id": id})
        event = await rt.journal.append_durable(end)
        if overflow:
            await rt.ledger.settle_overflow(res.id, actual)
        else:
            await rt.ledger.settle(res.id, actual, estimated=estimated)
        rt.board.put_candidate(candidate, epoch=event.epoch, seq=event.seq)
        rt.own.append(candidate)
        rt.repeats.add(key, candidate)
    if candidate.output.status == RunStatus.PROVIDER_FAILED:
        from swarm_arena._engine.stops import failed_providers

        failed = await _prepare_sync(lambda: failed_providers(candidate.output, rt.tool))
        with anyio.CancelScope(shield=True):
            for provider in sorted(failed):
                await rt.stops.fail(provider, source="tool")
    return candidate
