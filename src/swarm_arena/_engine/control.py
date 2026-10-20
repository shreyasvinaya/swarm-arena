"""Request epoch-scoped stops and report committed run progress."""

from contextlib import suppress
from typing import Literal

import anyio

from swarm_arena._budget.amount import Amount
from swarm_arena._errors import AlreadyExists, _core_error
from swarm_arena._journal.event import EventType
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.plan import Plan
from swarm_arena._model.results import Candidate
from swarm_arena._scoring.ranking import ranked
from swarm_arena._store.base import open_store


def stop_key(epoch: int, *, now: bool) -> str:
    """
    Return the graceful or immediate stop key for one lease epoch.

    Parameters
    ----------
    epoch : int
        Current lease epoch.
    now : bool
        Select immediate cancellation instead of graceful admission closure.

    Returns
    -------
    str
        Run-store-relative control key.

    Examples
    --------
    >>> stop_key(2, now=True)
    'control/stop-now-e2'
    """
    return f"control/stop{'-now' if now else ''}-e{epoch}"


class GoalStatus(FrozenModel):
    """
    Summarize one goal using committed progress and stamped ranking scores.

    Parameters
    ----------
    status : str
        Replayed lifecycle status.
    round : int
        Latest ordinary round started.
    rounds_total : int
        Current plan's ordinary round count under replayed limits.
    candidates : int
        Number of committed candidates, including parts and gated results.
    best : float or None
        Current main group's best ranking score.value. None when absent or
        non-scalar, including parts-only or unjoined results. Tuple keys rank
        candidates but their coordinates are never substituted for the value.

    Examples
    --------
    >>> GoalStatus(status="queued", round=0, rounds_total=3, candidates=0, best=None).best
    """

    status: str
    round: int
    rounds_total: int
    candidates: int
    best: float | None


class StatusReport(FrozenModel):
    """
    Describe a single authority snapshot and its committed replay state.

    Parameters
    ----------
    run_id : str
        Run identifier.
    epoch : int
        Epoch from the owner authority snapshot, even before its first event.
    owner_live : bool
        Whether the authority lease has not expired.
    head_seq : int
        Last committed event sequence, or zero before the first commit.
    stop : {'none', 'stop', 'stop-now'}
        Journaled stop state, independent of pending control files.
    goals : dict[str, GoalStatus]
        Goal summaries in replay order.
    spent : Amount
        Charged reservations, including conservative and overflow settlements.
    lost : int
        Number of distinct lost tool runs.

    Examples
    --------
    >>> report = StatusReport(run_id="r", epoch=1, owner_live=True, head_seq=0,
    ...     stop="none", goals={}, spent=Amount(), lost=0)
    >>> report.head_seq
    0
    """

    run_id: str
    epoch: int
    owner_live: bool
    head_seq: int
    stop: Literal["none", "stop", "stop-now"]
    goals: dict[str, GoalStatus]
    spent: Amount
    lost: int


def _request_stop(run_root: str, run_id: str, now: bool) -> int:
    """Create only the requested epoch's key and verify authority afterwards."""
    store = open_store(run_root).child(run_id)
    first = read_owner(store)
    if first is None or Lease.expired(first[0]):
        raise _core_error(RuntimeError, f"run {run_id}: no live owner")
    epoch = first[0].epoch
    with suppress(AlreadyExists):
        store.create(stop_key(epoch, now=now), b"")
    second = read_owner(store)
    if second is not None and second[0].epoch != epoch:
        raise _core_error(
            RuntimeError,
            f"run {run_id}: the epoch changed from {epoch} to {second[0].epoch} "
            "during stop; run stop again",
        )
    if second is None or Lease.expired(second[0]):
        raise _core_error(RuntimeError, f"run {run_id}: no live owner")
    return epoch


async def request_stop(run_root: str, run_id: str, *, now: bool) -> int:
    """
    Request a graceful or immediate stop from the current live owner.

    Parameters
    ----------
    run_root : str
        Root containing run directories.
    run_id : str
        Run to stop.
    now : bool
        Select immediate cancellation instead of graceful admission closure.

    Returns
    -------
    int
        Owner epoch whose control file was created. Repeated requests succeed.

    Raises
    ------
    RuntimeError
        If no live owner exists or ownership changes during the request.
        A changed epoch requires a new request for its own control key.

    Examples
    --------
    >>> import anyio
    >>> from functools import partial
    >>> anyio.run(partial(request_stop, "runs", "r", now=False))  # doctest: +SKIP
    1
    """
    return await anyio.to_thread.run_sync(_request_stop, run_root, run_id, now)


def _best(state: RunState, plan: Plan, candidates: list[Candidate]) -> float | None:
    """Rank only main candidates, resolving names from metadata or stamped scores."""
    if state.goals[plan.goal.id].final_parts is not None:
        return None
    pool = [candidate for candidate in candidates if candidate.part is None]
    ids = {candidate.id for candidate in pool}
    scores = [score for score in state.scores.values() if score.candidate_id in ids]
    name = plan.ranking or (plan.judges[0].name if len(plan.judges) == 1 else None)
    identity = next(
        (score.identity for score in reversed(scores) if not name or score.judge == name), None
    )
    if identity is None:
        return None
    pairs = ranked(
        pool, scores, identity=identity, fidelity=None, allow_partial=plan.converge.allow_partial
    )
    return pairs[0][1].value if pairs else None


def _spent(state: RunState) -> Amount:
    """Reproduce authoritative reservation charges, retaining overflow actuals."""
    overflow = {
        event.payload["id"]
        for event in state.ledger_events
        if event.type == EventType.SETTLE and event.payload.get("usage_overflow", False)
    }
    spent = Amount()
    for id, entry in state.reservations.items():
        conservative = entry.state in ("open", "estimated") or entry.reservation.get(
            "keep_charged", False
        )
        raw = entry.actual if id in overflow or not conservative else entry.reservation["amount"]
        spent += Amount.model_validate(raw)
    return spent


def _read_status(run_root: str, run_id: str) -> StatusReport:
    """Read one authority and follow only its committed chain without imports."""
    store = open_store(run_root).child(run_id)
    owner = read_owner(store)
    if owner is None:
        raise _core_error(RuntimeError, f"run {run_id}: no run")
    doc, _ = owner
    chain = read_chain(store, doc.head) if doc.head is not None else []
    state = RunState.from_events(event for _, events in chain for event in events)
    goals: dict[str, GoalStatus] = {}
    for goal_id, goal in state.goals.items():
        revisions = state.plans.get(goal_id, [])
        plan = revisions[-1][0] if revisions else None
        candidates = [
            c for c in state.candidates.values() if c.goal_id == goal_id and c.run_id == run_id
        ]
        goals[goal_id] = GoalStatus(
            status=goal.status,
            round=goal.round,
            rounds_total=plan.rounds_total(state.config.limits) if plan is not None else 0,
            candidates=len(candidates),
            best=_best(state, plan, candidates) if plan else None,
        )
    return StatusReport(
        run_id=run_id,
        epoch=doc.epoch,
        owner_live=not Lease.expired(doc),
        head_seq=doc.head.seq if doc.head is not None else 0,
        stop=state.stop,
        goals=goals,
        spent=_spent(state),
        lost=len(state.lost),
    )


async def read_status(run_root: str, run_id: str) -> StatusReport:
    """
    Read status exclusively from one committed journal snapshot.

    Parameters
    ----------
    run_root : str
        Root containing run directories.
    run_id : str
        Run to inspect.

    Returns
    -------
    StatusReport
        Authority liveness, committed progress, ranking value and charged spend.
        Components are never imported and cached view files are never read.
        Pending stop files do not affect the journal's latched stop state.
        Store, lease and journal work is offloaded from the event loop.

    Raises
    ------
    RuntimeError
        If the run has no authority document or its committed chain is invalid.

    Examples
    --------
    >>> import anyio
    >>> anyio.run(read_status, "runs", "r").head_seq  # doctest: +SKIP
    5
    """
    return await anyio.to_thread.run_sync(_read_status, run_root, run_id)


def format_status(report: StatusReport) -> str:
    """
    Format authority, goal progress and charged spend as plain text.

    Parameters
    ----------
    report : StatusReport
        Validated snapshot to render.

    Returns
    -------
    str
        Header, one line per goal and a final spend line, without a trailing newline.

    Examples
    --------
    >>> report = StatusReport(run_id="r", epoch=1, owner_live=False, head_seq=0,
    ...     stop="none", goals={}, spent=Amount(), lost=0)
    >>> format_status(report).splitlines()[0]
    'run r  epoch 1  owner expired  head 0  stop none'
    """
    lines = [
        f"run {report.run_id}  epoch {report.epoch}  "
        f"owner {'live' if report.owner_live else 'expired'}  "
        f"head {report.head_seq}  stop {report.stop}"
    ]
    for goal_id, goal in report.goals.items():
        lines.append(
            f"goal {goal_id}  {goal.status}  round {goal.round}/{goal.rounds_total}  "
            f"candidates {goal.candidates}  best {goal.best if goal.best is not None else '-'}"
        )
    lines.append(f"spent {report.spent.tokens} tokens, ${report.spent.usd:.2f}  lost {report.lost}")
    return "\n".join(lines)
