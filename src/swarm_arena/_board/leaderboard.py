"""Build leaderboard rows from selected candidates and recorded score identities.

Callers select current or historical producing GoalKeys through the cache/view
before calling these helpers. Part names label those selected groups; they do
not establish producing identity. Reading recorded plans never loads plugins.
"""

from collections.abc import Iterable, Mapping, Sequence

from swarm_arena._board.cache import Position
from swarm_arena._engine.parts import part_scoring
from swarm_arena._model.board_types import Row
from swarm_arena._model.plan import Plan
from swarm_arena._model.results import Candidate, Score
from swarm_arena._scoring.ranking import groups, ranked_groups


def pick_identity(
    scores: Sequence[tuple[Score, Position]], judge: str, *, run_id: str | None
) -> tuple[str, str, str] | None:
    """
    Pick a judge's newest recorded identity, preferring the requested run.

    Parameters
    ----------
    scores : Sequence[tuple[Score, Position]]
        Scores of the selected group's candidates. A journal Position's second
        field is its source run ID; a human Position contains its object name.
    judge : str
        Recorded judge name to match exactly.
    run_id : str or None
        Prefer scores recorded by this run if it has any for the named judge;
        otherwise fall back to all supplied matching scores.

    Returns
    -------
    tuple[str, str, str] or None
        Judge, rubric hash, and panel hash, or None when no score matches.

    Examples
    --------
    >>> pick_identity([], "quality", run_id=None) is None
    True
    """
    matching = [pair for pair in scores if pair[0].judge == judge]
    preferred = [pair for pair in matching if pair[1][1] == run_id] if run_id is not None else []
    newest = max(preferred or matching, key=lambda pair: pair[1], default=None)
    return newest[0].identity if newest is not None else None


def group_rows(
    candidates: Sequence[tuple[Candidate, Position]],
    scores: Sequence[tuple[Score, Position]],
    *,
    judge: str | Mapping[str | None, str],
    run_id: str | None = None,
    identity_run: str | None = None,
    fidelity: str | None = None,
    min_improvement: float | None = None,
) -> dict[str | None, list[Row]]:
    """
    Build independently ranked rows for each selected main-goal or part group.

    Parameters
    ----------
    candidates : Sequence[tuple[Candidate, Position]]
        Candidate set already filtered by current or historical producing keys.
        Positions define commit order for ties and first-place improvement.
    scores : Sequence[tuple[Score, Position]]
        Recorded scores; scores outside the selected candidate set are ignored
        before identity choice. Latest matching Position wins per candidate.
    judge : str or Mapping[str or None, str]
        One judge for all groups, or recorded names per group. Missing names
        omit a group. Each ``human:`` judge ranks all fidelities together.
    run_id : str or None, default=None
        Limit ranked candidates to this run, independently of identity choice.
    identity_run : str or None, default=None
        Prefer this source run's identity among all supplied selected candidates
        in each group, including when ``run_id`` limits the ranked rows.
    fidelity : str or None, default=None
        Requested fidelity, or the highest-round fidelity per ranked group.
    min_improvement : float or None, default=None
        First-place replacement requires a strictly better full key and at least
        this improvement in its first component. Other rows retain key order.

    Returns
    -------
    dict[str or None, list[Row]]
        Nonempty groups with ranks starting at one and complete provenance.

    Examples
    --------
    >>> group_rows([], [], judge="quality")
    {}
    """
    ordered = sorted(candidates, key=lambda pair: pair[1])
    by_id = {candidate.id: candidate for candidate, _ in ordered}
    by_group: dict[str | None, list[tuple[Score, Position]]] = {}
    for score, position in sorted(scores, key=lambda pair: pair[1]):
        candidate = by_id.get(score.candidate_id)
        if candidate is not None:
            by_group.setdefault(candidate.part, []).append((score, position))
    identities: dict[str | None, tuple[str, str, str]] = {}
    latest: dict[str, Score] = {}
    for part, recorded in by_group.items():
        name = judge if isinstance(judge, str) else judge.get(part)
        if name is None:
            continue
        identity = pick_identity(recorded, name, run_id=identity_run)
        if identity is not None:
            identities[part] = identity
            latest.update(
                (score.candidate_id, score) for score, _ in recorded if score.identity == identity
            )
    selected = [c for c, _ in ordered if run_id is None or c.run_id == run_id]
    ranked = {}
    for part, identity in identities.items():
        ranked.update(
            ranked_groups(
                [c for c in selected if c.part == part],
                latest.values(),
                identity={part: identity},
                fidelity=fidelity,
                every_fidelity=identity[0].startswith("human:"),
            )
        )
    positions = {c.id: position for c, position in ordered}
    result: dict[str | None, list[Row]] = {}
    for part, eligible in ranked.items():
        if min_improvement is not None and not identities[part][0].startswith("human:"):
            leader: tuple[Candidate, Score] | None = None
            for pair in sorted(eligible, key=lambda pair: positions[pair[0].id]):
                if leader is None or (
                    pair[1].key > leader[1].key
                    and pair[1].key[0] - leader[1].key[0] >= min_improvement
                ):
                    leader = pair
            if leader is not None:
                eligible = [leader] + [pair for pair in eligible if pair[0].id != leader[0].id]
        result[part] = [
            Row(
                rank=rank,
                candidate_id=candidate.id,
                run_id=candidate.run_id,
                goal_id=candidate.goal_id,
                part=candidate.part,
                team=candidate.team,
                agent=candidate.agent,
                fidelity=candidate.fidelity,
                key=score.key,
                value=score.value,
                judge=score.judge,
                rubric_hash=score.rubric_hash,
                panel_hash=score.panel_hash,
                built_on=candidate.parent_ids,
                partial=candidate.output.partial,
            )
            for rank, (candidate, score) in enumerate(eligible, 1)
        ]
    return result


def flatten(groups: Mapping[str | None, list[Row]]) -> list[Row]:
    """
    Flatten rows with the main group first and then parts by name.

    Parameters
    ----------
    groups : Mapping[str or None, list[Row]]
        Independently ranked main-goal and part rows.

    Returns
    -------
    list[Row]
        Rows in deterministic group order, preserving each group's own ranks.

    Examples
    --------
    >>> flatten({})
    []
    """
    return [
        row
        for part in sorted(groups, key=lambda part: (part is not None, part or ""))
        for row in groups[part]
    ]


def group_judges(
    plan: Plan, *, candidates: Sequence[Candidate] = (), scores: Iterable[Score] = ()
) -> dict[str | None, str]:
    """
    Resolve group ranking names from explicit settings or recorded sole judges.

    Parameters
    ----------
    plan : Plan
        Revision applicable to the current or historical view. Part settings
        inherit through ``part_scoring`` without component or catalog imports.
    candidates : Sequence[Candidate], default=()
        Candidates already selected by the view's producing GoalKeys. Only their
        scores can resolve an unnamed judge; part names alone do not select them.
    scores : Iterable[Score], default=()
        Recorded scores for name resolution. An unnamed sole judge resolves only
        when its group has exactly one nonempty, non-human recorded judge name.

    Returns
    -------
    dict[str or None, str]
        Explicit names and unambiguously resolved sole judges. Missing or
        ambiguous recorded names are omitted; no implementation is imported.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan, Ref
    >>> plan = Plan(goal=Goal(id="g", namespace="n"), teams=[],
    ...             judges=[Ref(path="missing:Judge", name="quality")])
    >>> group_judges(plan)
    {None: 'quality'}
    """
    by_id = {candidate.id: candidate for candidate in candidates}
    recorded: dict[str | None, set[str]] = {}
    for score in scores:
        candidate = by_id.get(score.candidate_id)
        if candidate is not None and score.judge and not score.judge.startswith("human:"):
            recorded.setdefault(candidate.part, set()).add(score.judge)
    result: dict[str | None, str] = {}
    for part in groups(plan):
        _, refs, name = part_scoring(plan, part)
        if name:
            result[part] = name
        elif len(refs) == 1:
            names = recorded.get(part, set())
            if len(names) == 1:
                result[part] = next(iter(names))
    return result


def format_value(value: float | None) -> str:
    """
    Format a human-facing score, including verdict-only reviews.

    Parameters
    ----------
    value : float or None
        Optional numeric score value.

    Returns
    -------
    str
        Four-significant-digit display, or a dash for an absent value.

    Examples
    --------
    >>> format_value(0.0041), format_value(None)
    ('0.0041', '-')
    """
    return "-" if value is None else f"{value:.4g}"
