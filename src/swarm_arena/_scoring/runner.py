"""Run ordered checks and judging batches with safe, journal-ready outcomes."""

from collections.abc import Callable, Mapping, Sequence
from copy import copy

from swarm_arena._errors import Fenced
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._scoring.judge import Judge, JudgeContext, is_finite_score, stamp_scores


class JudgeError(FrozenModel):
    """
    Record one candidate's failure in a judging batch.

    Parameters
    ----------
    candidate_id : str
        Candidate that could not be scored.
    judge : str
        Plan name of the judge.
    reason : str
        Safe diagnostic without exception values.

    Examples
    --------
    >>> JudgeError(candidate_id='c-1', judge='quality', reason='no score').reason
    'no score'
    """

    candidate_id: str
    judge: str
    reason: str


class JudgeOutcome(FrozenModel):
    """
    Collect stamped scores and ordered candidate errors for journaling.

    Parameters
    ----------
    scores : list[Score]
        Finite scores stamped with the judging context's id source.
    errors : list[JudgeError]
        At most one error per candidate per judge, in batch order.

    Examples
    --------
    >>> JudgeOutcome(scores=[], errors=[]).model_dump()
    {'scores': [], 'errors': []}
    """

    scores: list[Score]
    errors: list[JudgeError]


async def run_checks(
    candidate: Candidate, checks: Sequence[tuple[str, Check]], ctx: CheckContext
) -> list[CheckResult]:
    """
    Run every check in plan order and stamp its plan name.

    Each check receives its own shallow context copy, sharing the executor and
    id source. Invalid results fail closed after lax model revalidation. Check
    exceptions expose only their types, while fencing always propagates.

    Parameters
    ----------
    candidate : Candidate
        Tool result to check.
    checks : Sequence[tuple[str, Check]]
        Plan names paired with check instances, in execution order.
    ctx : CheckContext
        Execution settings copied for each check.

    Returns
    -------
    list[CheckResult]
        One validated, plan-named result for every check.

    Raises
    ------
    Fenced
        If a command's start callback reports lost ownership.

    Examples
    --------
    >>> import asyncio
    >>> from swarm_arena import Goal
    >>> from swarm_arena._model.tooling import RunStatus, ToolOutput, ToolRequest
    >>> from unittest.mock import Mock
    >>> candidate = Candidate(id='c-1', run_id='r', goal_id='g', team='t', agent='a',
    ...     round=1, request=ToolRequest(), output=ToolOutput(status=RunStatus.OK))
    >>> context = CheckContext(goal=Goal(id='g', namespace='example'),
    ...     executor=Mock(), artifact_root='/artifacts')
    >>> asyncio.run(run_checks(candidate, [], context))
    []
    """
    results: list[CheckResult] = []
    for name, check in checks:
        try:
            result = await check.check(candidate, copy(ctx))
        except Fenced:
            raise
        except Exception as error:
            results.append(
                CheckResult(
                    check=name, passed=False, reasons=[f"check error: {type(error).__name__}"]
                )
            )
            continue
        try:
            if not isinstance(result, CheckResult):
                raise TypeError("invalid check result")
            valid = CheckResult.model_validate(result.model_dump(warnings=False))
        except Fenced:
            raise
        except Exception:
            valid = CheckResult(check=name, passed=False, reasons=["check error: invalid result"])
        results.append(valid.model_copy(update={"check": name}))
    return results


def prepare_judge_batch(
    judge: Judge, candidates: Sequence[Candidate], leaders: Sequence[Candidate]
) -> tuple[list[Candidate], frozenset[str]]:
    """
    Prepare the exact eligible and unique-anchor batch before funding it.

    The caller contains exceptions from the judge's anchor hook. Prepared
    batches passed to ``judge_round`` do not invoke that hook again.

    Parameters
    ----------
    judge : Judge
        Judge supplying the anchor count.
    candidates : Sequence[Candidate]
        New candidates, filtered by checks.
    leaders : Sequence[Candidate]
        Ranked leaders supplying the anchor prefix.

    Returns
    -------
    tuple[list[Candidate], frozenset[str]]
        Ordered batch and anchor ids.

    Examples
    --------
    >>> from swarm_arena import ProgramJudge
    >>> prepare_judge_batch(ProgramJudge("builtins:abs"), [], [])
    ([], frozenset())
    """
    batch = [candidate for candidate in candidates if not candidate.gated_out]
    if not batch:
        return [], frozenset()
    anchors = leaders[: judge.anchor_count()]
    members = {candidate.id for candidate in batch}
    for anchor in anchors:
        if anchor.id not in members:
            batch.append(anchor)
            members.add(anchor.id)
    return batch, frozenset(anchor.id for anchor in anchors)


async def _judge_batch(
    ref: Ref,
    judge: Judge,
    batch: Sequence[Candidate],
    ctx: JudgeContext,
    *,
    leaders: Sequence[Candidate] | None = None,
) -> JudgeOutcome:
    """Contain judge hooks and malformed output before building safe outcomes."""
    name = ref.component_name()
    raised: str | None = None
    seen: set[str] = set()
    nonfinite: set[str] = set()
    try:
        if leaders is not None:
            batch, ctx.anchors = prepare_judge_batch(judge, batch, leaders)
        scores = await judge.judge(ctx.goal, batch, ctx)
        if (
            not isinstance(scores, Sequence)
            or isinstance(scores, (str, bytes, bytearray))
            or any(not isinstance(score, Score) for score in scores)
        ):
            raise TypeError("invalid judge result")
        members = {candidate.id for candidate in batch}
        accepted: list[Score] = []
        for score in scores:
            if score.candidate_id not in members or score.candidate_id in seen:
                continue
            seen.add(score.candidate_id)
            if not is_finite_score(score):
                nonfinite.add(score.candidate_id)
            elif score.candidate_id not in ctx.errors:
                accepted.append(score)
        stamped = (
            stamp_scores(accepted, judge=judge, judge_name=name, ids=ctx.ids, batch_id=ctx.batch_id)
            if accepted
            else []
        )
    except Fenced:
        raise
    except Exception as error:
        stamped = []
        raised = f"{type(error).__name__}: judge raised"

    errors: list[JudgeError] = []
    for candidate in batch:
        if candidate.id in ctx.errors:
            reason = ctx.errors[candidate.id]
        elif raised is not None:
            reason = raised
        elif candidate.id in nonfinite:
            reason = "non-finite score"
        elif candidate.id not in seen:
            reason = "no score"
        else:
            continue
        errors.append(JudgeError(candidate_id=candidate.id, judge=name, reason=reason))
    return JudgeOutcome(scores=stamped, errors=errors)


async def judge_round(
    goal_id: str,
    judges: Sequence[tuple[Ref, Judge]],
    candidates: Sequence[Candidate],
    leaders: Sequence[Candidate],
    ctx_for: Callable[[Judge], JudgeContext],
    *,
    prepared: tuple[list[Candidate], frozenset[str]] | None = None,
) -> JudgeOutcome:
    """
    Judge gated-in candidates once per judge, adding its leader anchors.

    Empty gated-in batches skip all judges. Foreign scores and later duplicate
    scores are discarded before finite-key validation and stamping. Each judge
    gets one fresh context from ``ctx_for``; its batch and anchor ids are set on
    that supplied context. Hook failures and malformed returns fail the whole
    invocation with type-only errors; fencing and cancellation propagate.

    Parameters
    ----------
    goal_id : str
        Goal identifier used in batch ids.
    judges : Sequence[tuple[Ref, Judge]]
        Configured references paired with built judges, in plan order.
    candidates : Sequence[Candidate]
        New candidates; failed checks exclude a candidate from this batch.
    leaders : Sequence[Candidate]
        Ordered leaders whose prefix supplies each judge's anchors.
    ctx_for : Callable[[Judge], JudgeContext]
        Factory returning a new execution context for each judge.
    prepared : tuple or None, default=None
        Already funded batch and anchor ids for a single judge. The caller
        contains preparation-hook failures before providing this value.

    Returns
    -------
    JudgeOutcome
        All accepted scores and per-judge errors in execution order.

    Raises
    ------
    Fenced
        If a command's start callback reports lost ownership.

    Examples
    --------
    >>> import asyncio
    >>> asyncio.run(judge_round('g', [], [], [], lambda judge: None)).model_dump()
    {'scores': [], 'errors': []}
    """
    eligible = [candidate for candidate in candidates if not candidate.gated_out]
    scores: list[Score] = []
    errors: list[JudgeError] = []
    if not eligible:
        return JudgeOutcome(scores=scores, errors=errors)
    for ref, judge in judges:
        ctx = ctx_for(judge)
        ctx.batch_id = f"{goal_id}-{ref.component_name()}-{eligible[0].id}"
        ctx.anchors = frozenset() if prepared is None else prepared[1]
        outcome = await _judge_batch(
            ref,
            judge,
            eligible if prepared is None else prepared[0],
            ctx,
            leaders=leaders if prepared is None else None,
        )
        scores.extend(outcome.scores)
        errors.extend(outcome.errors)
    return JudgeOutcome(scores=scores, errors=errors)


async def final_judge(
    goal_id: str,
    ranking: tuple[Ref, Judge],
    finalists: Sequence[Candidate],
    ctx: JudgeContext,
) -> tuple[list[Candidate], JudgeOutcome]:
    """
    Rescore all finalists in one uncached batch and return order and outcomes.

    Final order compares score keys descending, then candidate ids ascending,
    across every finalist's round and fidelity. The supplied context is marked
    final, cache reads are disabled, and anchors are cleared. Judge and metadata
    hook failures or malformed returns produce an empty order and safe errors.

    Parameters
    ----------
    goal_id : str
        Goal identifier used in the batch id.
    ranking : tuple[Ref, Judge]
        Configured reference and built ranking judge.
    finalists : Sequence[Candidate]
        All finalists to judge together; an empty sequence skips the judge.
    ctx : JudgeContext
        Final execution context and the only source of score ids.

    Returns
    -------
    tuple[list[Candidate], JudgeOutcome]
        Scored finalists in final order, plus scores and errors for journaling.

    Raises
    ------
    Fenced
        If a command's start callback reports lost ownership.

    Examples
    --------
    >>> import asyncio
    >>> from swarm_arena import Goal, ProgramJudge
    >>> from unittest.mock import Mock
    >>> judge = ProgramJudge('builtins:abs')
    >>> context = JudgeContext(goal=Goal(id='g', namespace='example'),
    ...     executor=Mock(), artifact_root='/artifacts')
    >>> order, outcome = asyncio.run(final_judge('g', (Ref.of(judge), judge), [], context))
    >>> (order, outcome.model_dump(), context.final, context.read_cache)
    ([], {'scores': [], 'errors': []}, True, False)
    """
    ctx.final = True
    ctx.read_cache = False
    ctx.anchors = frozenset()
    if not finalists:
        return [], JudgeOutcome(scores=[], errors=[])
    ref, judge = ranking
    ctx.batch_id = f"{goal_id}-{ref.component_name()}-{finalists[0].id}"
    outcome = await _judge_batch(ref, judge, finalists, ctx)
    scores = {score.candidate_id: score for score in outcome.scores}
    order = sorted(
        (candidate for candidate in finalists if candidate.id in scores), key=lambda c: c.id
    )
    order.sort(key=lambda candidate: scores[candidate.id].key, reverse=True)
    return order, outcome


def for_manager(candidate: Candidate, feedback: Mapping[str, str]) -> Candidate:
    """
    Copy a candidate with private check feedback removed for its manager.

    Parameters
    ----------
    candidate : Candidate
        Candidate whose stored checks must remain unchanged.
    feedback : Mapping[str, str]
        Plan check names mapped to feedback modes. ``pass_fail`` hides both
        reasons and structured details; other modes retain the check feedback.

    Returns
    -------
    Candidate
        Candidate copy with filtered checks and unchanged execution output.

    Examples
    --------
    >>> from swarm_arena._model.tooling import RunStatus, ToolOutput, ToolRequest
    >>> candidate = Candidate(id='c-1', run_id='r', goal_id='g', team='t', agent='a',
    ...     round=1, request=ToolRequest(), output=ToolOutput(status=RunStatus.OK),
    ...     checks=[CheckResult(check='private', passed=False, reasons=['why'], details={'n': 1})])
    >>> for_manager(candidate, {'private': 'pass_fail'}).checks[0].model_dump()
    {'check': 'private', 'passed': False, 'reasons': [], 'details': {}}
    >>> candidate.checks[0].reasons
    ['why']
    """
    checks = [
        result.model_copy(update={"reasons": [], "details": {}})
        if feedback.get(result.check) == "pass_fail"
        else result.model_copy()
        for result in candidate.checks
    ]
    return candidate.model_copy(update={"checks": checks})
