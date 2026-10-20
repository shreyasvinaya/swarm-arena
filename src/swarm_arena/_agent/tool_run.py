"""Validate, reserve, execute, and durably publish one agent tool request."""

from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

import anyio

from swarm_arena._agent.manager import LimitReached
from swarm_arena._agent.runtime import AgentRuntime
from swarm_arena._budget.amount import USD_LIMIT, Amount
from swarm_arena._budget.llm_scope import current_requesting_call
from swarm_arena._errors import ConfigError, Fenced, SoftError
from swarm_arena._exec.env import tool_env_vars
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.a2a import task_to_json, tool_run_task
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import _validate_config
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import RunStatus, ToolOutput, ToolRequest, parse_logs_ref
from swarm_arena._scoring.check import CheckContext
from swarm_arena._scoring.runner import run_checks
from swarm_arena._store.base import open_store
from swarm_arena._tooling.result_file import request_json
from swarm_arena._tooling.staging import (
    parent_ref,
    plan_tool_stage,
    put_edits,
    tool_root,
    validate_edits,
)
from swarm_arena._tooling.tool import effective_tool_timeout, tool_grace


class Inflight:
    """
    Track cancellable tool submissions and latch closed round admission.

    Attributes
    ----------
    closed : bool
        Whether new submissions and queued starts must be refused.

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
        self._scopes: dict[anyio.CancelScope, float] = {}

    def add(self, scope: anyio.CancelScope, *, grace_s: float = 0.0) -> None:
        """
        Admit a submission unless the round has closed.

        Parameters
        ----------
        scope : anyio.CancelScope
            Scope controlling one executor submission.
        grace_s : float, default=0.0
            Effective job grace in seconds; retained through executor collection.

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
        if self.closed:
            raise LimitReached("stop", "the round is stopping")
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

    def close(self) -> None:
        """
        Refuse future admission without cancelling existing work.

        Examples
        --------
        >>> registry = Inflight()
        >>> registry.close()
        >>> registry.closed
        True
        """
        self.closed = True

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
        for scope in tuple(self._scopes):
            scope.cancel()


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

    async def on_started(started: Job.Started) -> None:
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


_T = TypeVar("_T")


def _request_hook(hook: Callable[..., _T], *args: Any) -> _T:
    """Contain user hook failures separately from safe core validation errors."""
    try:
        return hook(*args)
    except Fenced:
        raise
    except Exception as error:
        raise SoftError(f"invalid tool request: {type(error).__name__}") from None


def _request(
    rt: AgentRuntime,
    inputs: Mapping[str, Any],
    parents: Sequence[str],
    replicate: int | None,
    edits: Mapping[str, str] | None,
) -> ToolRequest:
    """Validate inputs and visible parents before any budget or count change."""
    resolved: list[Candidate] = []
    if len(set(parents)) != len(parents):
        raise SoftError("duplicate parent ids")
    for id in parents:
        candidate = next((c for c in rt.own if isinstance(c, Candidate) and c.id == id), None)
        if candidate is None:
            try:
                candidate = rt.view.get_candidate(id)
            except (KeyError, ValueError):
                raise SoftError(f"unknown parent: {id}") from None
        resolved.append(candidate)
    model = _request_hook(rt.tool.get_input_model)
    try:
        validated = _validate_config(model, dict(inputs), where="tool inputs")
        return ToolRequest(
            inputs=validated.model_dump(mode="json"),
            parents=[parent_ref(c) for c in resolved],
            round=rt.round,
            round_overrides=rt.round_spec.tool_overrides,
            replicate=0 if replicate is None else replicate,
            edits=dict(edits or {}),
        )
    except Fenced:
        raise
    except ConfigError as error:
        raise SoftError(str(error)) from None
    except Exception as error:
        raise SoftError(f"invalid tool request: {type(error).__name__}") from None


def _stage_settings(
    rt: AgentRuntime,
    request: ToolRequest,
) -> tuple[dict[str, str], Sequence[str], frozenset[str]]:
    """Validate root and edit permissions before allocating a replicate."""
    try:
        files = _request_hook(rt.tool._own_files)
        root = tool_root(rt.goal.files.manifest, files)
        protected = _request_hook(rt.tool._protected, root.keys()) | rt.private_paths
        editable = _request_hook(rt.tool.editable)
        validate_edits(
            request.edits,
            editable=editable,
            protected=protected,
            root=root.keys(),
            max_edit_bytes=rt.limits.max_edit_bytes,
        )
        return files, editable, protected
    except (SoftError, Fenced):
        raise
    except Exception as error:
        raise SoftError(f"invalid tool stage: {type(error).__name__}") from None


def _stage(
    rt: AgentRuntime,
    request: ToolRequest,
    settings: tuple[dict[str, str], Sequence[str], frozenset[str]],
) -> tuple[Job.Stage, dict[str, str], list[str]]:
    """Plan with the final request before counting or reserving the run."""
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
    except (SoftError, Fenced):
        raise
    except Exception as error:
        raise SoftError(f"invalid tool stage: {type(error).__name__}") from None


def _logs(rt: AgentRuntime, id: str, ref: str) -> str:
    """Read both verified executor blobs before publishing run-relative logs."""
    stdout_sha, stderr_sha = parse_logs_ref(ref)
    stdout, stderr = rt.artifacts.get_bytes(stdout_sha), rt.artifacts.get_bytes(stderr_sha)
    store = open_store(rt.config.run_root).child(rt.repeats.run_id)
    stem = f"goals/{rt.goal_id}/logs/{id}"
    store.create(stem + ".out", stdout)
    store.create(stem + ".err", stderr)
    return stem


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

    Input, stage, repeat, and run-count admission occur without an await.
    Internal round cancellation retains partial output; outer cancellation
    propagates. Fencing prevents completion and settlement. Connector failures
    become error candidates with type-only warnings.

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
    request = _request(rt, inputs, parents, replicate, edits)
    allocation_request = request
    settings = _stage_settings(rt, request)
    index: int | None = None
    try:
        deterministic = _request_hook(rt.tool.is_deterministic, request)
        if not deterministic:
            index = rt.repeats.next_replicate(rt, request, requested=replicate)
            request = request.model_copy(update={"replicate": index})
        stage, overlay, warnings = _stage(rt, request, settings)
        key = _request_hook(rt.repeats.key, rt, request)
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
                raise LimitReached("max_refused_repeats", rt.watchdog.reason)
            raise SoftError(f"repeat refused: {outcome.candidate.id}")
        if outcome.action in {"return_own", "link"}:
            assert outcome.candidate is not None
            return outcome.candidate
        if rt.counters.tool_runs >= rt.limits.max_tool_runs_per_round:
            raise LimitReached("max_tool_runs_per_round", "tool runs used up for this round")
        rt.counters.tool_runs += 1
        return await _run(rt, request, key, stage, overlay, warnings, tuple(read_ids))
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
) -> Candidate:
    """Own reservation, executor lifecycle, and the shielded completion sequence."""
    id = rt.ids.next("c")
    inflight = _inflight.get()
    if inflight is None:
        raise RuntimeError("tool run requires track_inflight")
    started = False
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
        amount, providers = await run_sync_daemon(estimate)
    except Fenced:
        raise
    except Exception as error:
        warnings.append(f"tool error: {type(error).__name__}")
        failed = True
    res = await rt.ledger.reserve(rt.holder, amount, kind="tool", goal_id=rt.goal_id)

    async def on_started(event: Job.Started) -> None:
        nonlocal started
        if inflight.closed:
            raise LimitReached("stop", "the round is stopping")
        await rt.journal.append_durable(
            _draft(
                rt,
                EventType.TOOL_START,
                candidate_id=id,
                reservation_id=res.id,
                started=event.model_dump(mode="json"),
            )
        )
        started = True
        rt.started_runs += 1

    async def on_partial(value: ToolOutput) -> None:
        partial.append(value)

    try:
        if not failed:
            await anyio.to_thread.run_sync(put_edits, request.edits, rt.artifacts)
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
                grace_s=tool_grace(rt.tool),
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
                inflight.add(run_scope, grace_s=job.grace_s)
                try:
                    output = await rt.executor.submit(job)
                finally:
                    inflight.discard(run_scope)
            if run_scope.cancelled_caught:
                output = partial[0] if partial else ToolOutput(status=RunStatus.CANCELLED)
    except Fenced:
        raise
    except LimitReached:
        with anyio.CancelScope(shield=True):
            await rt.ledger.settle(res.id, Amount())
        raise
    except Exception as error:
        warnings.append(f"tool error: {type(error).__name__}")
        output = ToolOutput(status=RunStatus.ERROR)

    # Deliver an outer cancellation before entering the completion shield.
    await anyio.lowlevel.checkpoint()
    with anyio.CancelScope(shield=True):
        if rt.journal.fenced.is_set():
            raise Fenced("run ownership lost")
        try:
            if not isinstance(output, ToolOutput):
                raise TypeError("invalid tool output")
            output = ToolOutput.model_validate(output.model_dump(warnings=False))
        except Fenced:
            raise
        except Exception:
            warnings.append("tool error: invalid output")
            output = ToolOutput(status=RunStatus.ERROR)
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
            except Fenced:
                raise
            except Exception as error:
                warnings.append(f"tool error: {type(error).__name__}")
                output = ToolOutput(status=RunStatus.ERROR)
                actual, estimated = Amount(), started and providers
        if output.logs_ref is not None:
            try:
                logs_ref = await anyio.to_thread.run_sync(_logs, rt, id, output.logs_ref)
                output = output.model_copy(update={"logs_ref": logs_ref})
            except Fenced:
                raise
            except Exception as error:
                warnings.append(f"tool error: {type(error).__name__}")
                output = output.model_copy(update={"status": RunStatus.ERROR, "logs_ref": None})
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
            checks.extend(await run_checks(candidate, [(name, check)], context))
        candidate = candidate.model_copy(update={"checks": checks})
        rt.watchdog.record_run(
            candidate.output.status,
            passed=not candidate.gated_out and candidate.output.status == RunStatus.OK,
        )
        end = _draft(
            rt,
            EventType.TOOL_END,
            candidate=candidate.model_dump(mode="json"),
            repeat_key=key,
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
    return candidate
