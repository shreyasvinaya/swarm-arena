"""Build a versioned, canonical solution view from committed replay state."""

import math
from statistics import mean
from typing import Any, ClassVar, Literal

from swarm_arena._engine.parts import part_scoring
from swarm_arena._journal.lease import Head
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.views import embed_source_head
from swarm_arena._model.base import StoredModel
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.data import FiniteJsonObject
from swarm_arena._model.plan import Plan
from swarm_arena._model.results import Candidate, Score
from swarm_arena._scoring.ranking import groups, ranked


class SolutionDocument(StoredModel):
    """
    Validate the private stored solution artifact, including its JSON subtrees.

    Candidate and score dumps retain nulls. The manifest here is run metadata,
    separate from the path-to-digest Manifest alias used for file snapshots.
    Readers load with model_validate_json to apply registered migrations and
    refuse newer formats.

    Examples
    --------
    >>> SolutionDocument.schema_kind
    'solution'
    """

    schema_kind: ClassVar[str] = "solution"
    run_id: str
    goal_id: str
    status: Literal["ok", "no_result", "unjoined"]
    reason: str | None
    finalists: list[FiniteJsonObject]
    parts: list[FiniteJsonObject]
    replicate_stats: list[FiniteJsonObject]
    best_of_n: FiniteJsonObject | None
    plan_revisions: list[FiniteJsonObject]
    manifest: FiniteJsonObject
    source_head: Head


def _ranking_pairs(
    state: RunState, plan: Plan, candidates: list[Candidate]
) -> list[tuple[Candidate, Score]]:
    """Use the newest stamped identity of each group's ranking judge."""
    pairs: list[tuple[Candidate, Score]] = []
    for group in groups(plan):
        pool = [candidate for candidate in candidates if candidate.part == group]
        ids = {candidate.id for candidate in pool}
        _, _, name = part_scoring(plan, group)
        scores = [score for score in state.scores.values() if score.candidate_id in ids]
        identity = next(
            (score.identity for score in reversed(scores) if not name or score.judge == name),
            None,
        )
        if identity is not None:
            pairs.extend(
                ranked(
                    pool,
                    scores,
                    identity=identity,
                    fidelity=None,
                    allow_partial=plan.converge.allow_partial,
                )
            )
    return pairs


def _entry(state: RunState, candidate: Candidate | None) -> dict[str, Any]:
    """Serialize a selected candidate and its recorded provenance without imports."""
    return {
        "candidate": candidate.model_dump(mode="json") if candidate is not None else None,
        "scores": [
            score.model_dump(mode="json")
            for score in state.scores.values()
            if candidate is not None and score.candidate_id == candidate.id
        ],
        "lineage": list(candidate.parent_ids) if candidate is not None else [],
    }


def _replicate_stats(pairs: list[tuple[Candidate, Score]]) -> list[dict[str, Any]]:
    """Summarize finite ranking values only for groups with explicit replicates."""
    grouped: dict[tuple[str, str], list[tuple[Candidate, float]]] = {}
    for candidate, score in pairs:
        if score.value is not None and math.isfinite(score.value):
            key = candidate.tool_identity, sha256_hex(canonical_json(candidate.request.inputs))
            grouped.setdefault(key, []).append((candidate, score.value))
    result: list[dict[str, Any]] = []
    for (tool_identity, inputs_sha256), members in grouped.items():
        if not any(candidate.request.replicate > 0 for candidate, _ in members):
            continue
        values = [value for _, value in members]
        result.append(
            {
                "tool_identity": tool_identity,
                "inputs_sha256": inputs_sha256,
                "n": len(values),
                "mean": mean(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return result


def solution_view(state: RunState, goal_id: str, source: Head) -> bytes:
    """
    Render a canonical solution artifact from replayed final selection events.

    Parameters
    ----------
    state : RunState
        Committed state through the supplied source head, with plan revisions.
    goal_id : str
        Goal whose final order or unjoined parts are rendered.
    source : Head
        Committed source head embedded in the artifact.

    Returns
    -------
    bytes
        Validated solution StoredModel serialized as canonical UTF-8 JSON.
        Ranks are one-based. Each finalist's plan_revision is the zero-based
        final evaluation revision, and plan_revisions contains all recorded
        plans with journal-authoritative origins. Lineage lists direct parents.
        Statistics use each group's newest ranking identity and fidelity;
        best_of_n reports the first finalist's ranking value when available.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan
    >>> from swarm_arena._journal.replay import GoalState
    >>> state = RunState(run_id="r", goals={"g": GoalState(status="no_result")},
    ...     plans={"g": [(Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[]), "human")]})
    >>> head = Head(epoch=1, seq=1, segment="journal/e1-1.jsonl")
    >>> b'"status":"no_result"' in solution_view(state, "g", head)
    True
    """
    goal = state.goals[goal_id]
    revisions = state.plans[goal_id]
    plan = revisions[-1][0]
    status: Literal["ok", "no_result", "unjoined"] = (
        "no_result"
        if goal.status == "no_result"
        else "unjoined"
        if goal.final_parts is not None
        else "ok"
    )
    selected = [state.candidates[id] for id in goal.final_order] if status == "ok" else []
    candidates = [
        c for c in state.candidates.values() if c.run_id == state.run_id and c.goal_id == goal_id
    ]
    pairs = _ranking_pairs(state, plan, candidates)
    ranking_scores = {candidate.id: score for candidate, score in pairs}
    best = ranking_scores.get(selected[0].id) if selected else None
    document = SolutionDocument(
        run_id=state.run_id,
        goal_id=goal_id,
        status=status,
        reason=goal.no_result_reason if status == "no_result" else goal.end_reason,
        finalists=[
            {"rank": rank, **_entry(state, candidate), "plan_revision": len(revisions) - 1}
            for rank, candidate in enumerate(selected, 1)
        ],
        parts=[
            {"part": part.name, **_entry(state, state.candidates[id] if id else None)}
            for part in plan.parts
            for id in [(goal.final_parts or {}).get(part.name)]
        ]
        if status == "unjoined"
        else [],
        replicate_stats=_replicate_stats(pairs),
        best_of_n={"candidate_id": selected[0].id, "value": best.value}
        if best is not None and best.value is not None
        else None,
        plan_revisions=[
            {**revision.model_dump(mode="json"), "origin": origin} for revision, origin in revisions
        ],
        manifest=state.manifest,
        source_head=source,
    )
    return embed_source_head("json", canonical_json(document.model_dump(mode="json")), source)
