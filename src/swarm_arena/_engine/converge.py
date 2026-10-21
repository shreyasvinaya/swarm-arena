"""Select finalists from an already ordered candidate ranking."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from swarm_arena._engine.parts import join_failure_reasons, join_parents, kept_best
from swarm_arena._engine.round import RoundRunner
from swarm_arena._journal.event import EventType
from swarm_arena._model.plan import RoundSpec
from swarm_arena._model.results import Candidate, Score
from swarm_arena._model.tooling import RunStatus
from swarm_arena._scoring.ranking import newest_fidelity, ranked

if TYPE_CHECKING:
    from swarm_arena._engine.goal import GoalRunner


def select_finalists(
    ranked: Sequence[tuple[Candidate, Score]],
    *,
    top_n: int,
    distinct: bool,
    fingerprint: Callable[[Candidate], str | None],
) -> list[Candidate]:
    """
    Retain leading candidates, optionally skipping duplicate fingerprints.

    Parameters
    ----------
    ranked : Sequence[tuple[Candidate, Score]]
        Eligible candidate-score pairs in the authoritative ranking order.
    top_n : int
        Maximum number retained; nonpositive values select none.
    distinct : bool
        Whether repeated fingerprints are skipped.
    fingerprint : Callable[[Candidate], str or None]
        Result fingerprint supplied by the caller's tool adapter. Called only
        when distinct selection is requested, until enough finalists are found.
        None means unknown and is always retained as distinct.

    Returns
    -------
    list[Candidate]
        Candidates in their original ranking order.

    Examples
    --------
    >>> select_finalists([], top_n=2, distinct=True, fingerprint=lambda c: c.id)
    []
    """
    if top_n <= 0:
        return []
    finalists: list[Candidate] = []
    seen: set[str] = set()
    for candidate, _ in ranked:
        if distinct:
            key = fingerprint(candidate)
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
        finalists.append(candidate)
        if len(finalists) == top_n:
            break
    return finalists


async def _retry_join(goal: "GoalRunner") -> bool:
    """Complete the durable retry transition before another convergence pass."""
    while (
        goal.join_retries < goal.plan.converge.max_join_attempts
        and goal.rounds_closed < goal.deps.config.limits.max_rounds
        and goal.replanner is not None
        and not goal._stop()
    ):
        goal._join_retry_number = goal.join_retries + 1
        try:
            accepted = await goal._replan()
        finally:
            goal._join_retry_number = None
        await goal._refresh()
        restored = goal.recorded.goals[goal.goal_id]
        goal.join_retries, goal.join_phase = restored.join_retries, restored.join_phase
        if not accepted:
            continue
        goal._frontier = None
        goal._recovering_converge = False
        await goal._run_rounds()
        if goal.status == "no_result":
            return False
        goal.join_phase = "converge"
        goal.deps.journal.append(
            goal._draft(
                EventType.CONVERGE_STARTED, select_only=goal.select_only, reason=goal.reason
            )
        )
        await goal.deps.journal.commit()
        return True
    goal._stop()
    if goal.select_only:
        goal.join_phase = "converge"
        return True
    reasons = next(
        (
            event.payload["reasons"]
            for event in reversed(goal._events)
            if event.goal_id == goal.goal_id and event.type == EventType.JOIN_FAILED
        ),
        goal.join_failures,
    )
    goal.status, goal.reason = "no_result", "; ".join(reasons)
    return False


async def converge(goal: "GoalRunner", *, select_only: bool) -> list[Candidate]:
    """
    Select or join candidates and obtain one uncached final ranking.

    Parts stopped in select-only mode are retained independently in
    ``goal.final_parts``. Join failures may replan and run remaining rounds.
    Failed joins owe a durable retry proposal; accepted retries owe ordinary
    rounds before another converge pass. Refused proposals consume retry
    allowance independently of the failed-join count. An immediate stop or
    deadline during a join retains the best parts before join-failure handling.
    Fingerprints come only from committed records.

    Parameters
    ----------
    goal : GoalRunner
        Goal with completed ordinary-round evidence.
    select_only : bool
        Suppress the converge team, retaining best parts when unjoined.

    Returns
    -------
    list[Candidate]
        Main-goal finalists in final order, limited by top_n and distinct.

    Examples
    --------
    >>> finalists = await converge(runner, select_only=True)  # doctest: +SKIP
    """
    recovered = goal._recover_final()
    if recovered is not None:
        return select_finalists(
            [(candidate, Score.maximize(0)) for candidate in recovered],
            top_n=goal.plan.converge.top_n,
            distinct=goal.plan.converge.distinct,
            fingerprint=lambda c: goal.recorded.fingerprints.get(c.id),
        )
    while True:
        if goal.join_phase == "replan" and not await _retry_join(goal):
            return []
        goal._stop()
        select_only |= goal.select_only
        state = goal._round_state()
        spec = goal.plan.converge
        if goal.plan.parts and select_only:
            goal.final_parts = {}
            for part in goal.plan.parts:
                candidate = kept_best(state, part.name)
                kept = goal.kept_parts.get(part.name)
                goal.final_parts[part.name] = (
                    candidate.id
                    if candidate
                    else kept
                    if any(c.id == kept for c in state.candidates)
                    else None
                )
            return []
        identity = state.ranking_identity.get(None)
        ranking = (
            ranked(
                (c for c in state.candidates if c.part is None),
                state.scores,
                identity=identity,
                fidelity=None,
                allow_partial=spec.allow_partial,
            )
            if identity is not None
            else []
        )
        survivors = select_finalists(
            ranking,
            top_n=spec.top_n,
            distinct=spec.distinct,
            fingerprint=lambda c: goal.recorded.fingerprints.get(c.id),
        )
        pool = survivors
        if spec.mode == "team" and not select_only and spec.team is not None:
            parents = (
                join_parents(state, parents_per_part=spec.parents_per_part)
                if goal.plan.parts
                else survivors
            )
            round_spec = spec.round or (goal.plan.rounds[-1] if goal.plan.rounds else RoundSpec())
            agents = await goal._slots([spec.team], reuse=False)
            out = await RoundRunner(
                goal.deps,
                goal_id=goal.goal_id,
                plan=goal.plan,
                agents=agents,
                supervisor=goal,
                round_spec=round_spec,
                rounds_left=1,
                converge=True,
                repeat_entries=goal.repeat_entries,
                seed_parents=[c.id for c in parents],
            ).run(goal.rounds_closed + 1)
            goal._merge_repeats(out.repeat_entries)
            await goal._refresh()
            goal._stop()
            select_only |= goal.select_only
            if goal.plan.parts and select_only:
                continue
            funded = await goal._barrier(out.candidates, goal.rounds_closed + 1)
            goal._stop()
            select_only |= goal.select_only
            if goal.plan.parts and select_only:
                continue
            pool = [
                c
                for c in out.candidates
                if not c.gated_out
                and (
                    (c.output.status == RunStatus.OK and not c.output.partial)
                    or (spec.allow_partial and c.output.partial)
                )
            ]
            if not funded:
                # Unfunded converge outputs cannot bypass the ordinary judges.
                pool = []
                select_only = True
                if goal.plan.parts:
                    continue
            if goal.plan.parts and not pool:
                reasons = join_failure_reasons(out.candidates) or ["no passing join candidate"]
                goal.join_attempts += 1
                goal.join_failures.extend(reasons)
                goal.deps.journal.append(
                    goal._draft(
                        EventType.JOIN_FAILED,
                        attempt=goal.join_attempts,
                        reasons=reasons,
                    )
                )
                await goal._refresh()
                goal.join_phase = "replan"
                continue
            if not goal.plan.parts:
                pool.extend(survivors)
            fidelity = newest_fidelity(pool)
            pool = [c for c in pool if c.fidelity == fidelity]
        order = await goal._final(pool)
        # Use final order, not an earlier score identity, for the last selection.
        return select_finalists(
            [(candidate, Score.maximize(0)) for candidate in order],
            top_n=spec.top_n,
            distinct=spec.distinct,
            fingerprint=lambda c: goal.recorded.fingerprints.get(c.id),
        )
