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


async def converge(goal: "GoalRunner", *, select_only: bool) -> list[Candidate]:
    """
    Select or join candidates and obtain one uncached final ranking.

    Parts stopped in select-only mode are retained independently in
    ``goal.final_parts``. Join failures may replan and run remaining rounds.
    Fingerprints come exclusively from committed producer records.

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
    retries = 0
    retry_limit = max(0, goal.plan.converge.max_join_attempts - goal.join_attempts)
    while True:
        state = goal._round_state()
        spec = goal.plan.converge
        if goal.plan.parts and select_only:
            goal.final_parts = {}
            for part in goal.plan.parts:
                candidate = kept_best(state, part.name)
                goal.final_parts[part.name] = (
                    candidate.id if candidate else goal.kept_parts.get(part.name)
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
                round_spec=round_spec,
                rounds_left=1,
                converge=True,
                repeat_entries=goal.repeat_entries,
                seed_parents=[c.id for c in parents],
            ).run(goal.rounds_closed + 1)
            goal._merge_repeats(out.repeat_entries)
            await goal._refresh()
            funded = await goal._barrier(out.candidates, goal.rounds_closed + 1)
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
                if (
                    retries < retry_limit
                    and goal.rounds_closed < goal.deps.config.limits.max_rounds
                    and not goal.deps.ledger.goal_exhausted(goal.goal_id)
                    and not goal._stop()
                    and await goal._replan()
                    and goal.rounds_closed < goal.plan.rounds_total(goal.deps.config.limits)
                ):
                    retries += 1
                    await goal._run_rounds()
                    if goal.status == "no_result":
                        return []
                    select_only = goal.select_only
                    goal.deps.journal.append(
                        goal._draft(EventType.CONVERGE_STARTED, select_only=select_only)
                    )
                    await goal.deps.journal.commit()
                    continue
                goal.status, goal.reason = "no_result", "; ".join(reasons)
                return []
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
