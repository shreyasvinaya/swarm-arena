"""Compute synchronous budget estimates, including user component hooks.

These functions call estimate/models/providers hooks synchronously. The engine
must offload this phase to its abandonable daemon worker before starting a run.
"""

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

from swarm_arena._agent.manager import Manager
from swarm_arena._budget.amount import Amount, _usd_from_micro_units
from swarm_arena._model.catalog import ModelCatalog, ModelSpec
from swarm_arena._model.plan import Plan, TeamSpec
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.tooling import ToolRequest
from swarm_arena._scoring.judge import Judge
from swarm_arena._scoring.lexicographic import Lexicographic
from swarm_arena._scoring.program_judge import ProgramJudge
from swarm_arena._tooling.tool import Tool

PriceSource = Callable[[str], ModelSpec.Prices | None]
_price_sources: list[PriceSource] = []


def register_price_source(source: PriceSource) -> None:
    """
    Register a fallback model-price lookup after catalog prices.

    Parameters
    ----------
    source : PriceSource
        Synchronous hook taking a provider/model identifier and returning USD
        prices per million tokens, or None. Sources run in registration order.

    Examples
    --------
    >>> register_price_source(lambda model: None)
    """
    _price_sources.append(source)


def make_price_of(models: ModelCatalog) -> Callable[[str], ModelSpec.Prices | None]:
    """
    Build an alias lookup preferring catalog prices over registered sources.

    Parameters
    ----------
    models : ModelCatalog
        Approved aliases and their provider/model identifiers.

    Returns
    -------
    Callable[[str], ModelSpec.Prices | None]
        Synchronous lookup returning None when no source knows the price.

    Examples
    --------
    >>> catalog = ModelCatalog({"a": ModelSpec(model="openai/example",
    ...     roles=["manager"], prices=ModelSpec.Prices(input=1, output=2))})
    >>> make_price_of(catalog)("a").output
    2.0
    """

    def price_of(alias: str) -> ModelSpec.Prices | None:
        spec = models.root[alias]
        if spec.prices is not None:
            return spec.prices
        for source in _price_sources:
            prices = source(spec.model)
            if prices is not None:
                return prices
        return None

    return price_of


def _scale(amount: Amount, factor: int | Fraction) -> Amount:
    """Scale exact units, rounding once and refusing unsupported USD results."""
    return Amount(
        tokens=round(amount.tokens * factor),
        usd=_usd_from_micro_units(round(amount._micro_usd * factor)),
    )


def _model_amount(
    aliases: Sequence[str],
    calls: int,
    config: RunConfig,
    price_of: Callable[[str], ModelSpec.Prices | None],
    *,
    cache_write: bool = False,
) -> Amount:
    """Sum repeated model aliases with decimal-rational price arithmetic."""
    total = Amount()
    for alias in aliases:
        tokens = calls * config.models.root[alias].max_tokens
        prices = price_of(alias)
        micros = (
            Fraction(0)
            if prices is None
            else tokens
            * max(
                Fraction(str(prices.input)) * (2 if cache_write else 1),
                Fraction(str(prices.output)),
            )
        )
        total += Amount(tokens=tokens, usd=_usd_from_micro_units(round(micros)))
    return total


def judge_amount(
    judge: Judge,
    n: int,
    config: RunConfig,
    price_of: Callable[[str], ModelSpec.Prices | None],
    *,
    llm: bool = True,
) -> Amount:
    """
    Estimate a judging batch using the first matching paid-call rule.

    Parameters
    ----------
    judge : Judge
        Synchronous model/provider hooks identify paid calls. ProgramJudge is
        free; Lexicographic recursively sums its members.
    n : int
        Number of candidates in the batch.
    config : RunConfig
        Model catalog, fallback reservation and tranche sizes.
    price_of : Callable[[str], ModelSpec.Prices | None]
        Synchronous alias-price lookup. Missing prices contribute zero USD.
    llm : bool, default=True
        Include model-based judges. False leaves their funding to the scope's
        tranche, while retaining provider-only judge reservations.

    Returns
    -------
    Amount
        Model judges use the componentwise maximum of one tranche and repeated
        model terms of twice the batch size, priced at max(2*input, output).
        Provider-only judges use n default reservations; all others are free.

    Examples
    --------
    >>> judge_amount(ProgramJudge("example:score"), 3, RunConfig(), lambda a: None)
    Amount(tokens=0, usd=0.0)
    """
    if isinstance(judge, ProgramJudge):
        return Amount()
    if isinstance(judge, Lexicographic):
        return sum(
            (judge_amount(j, n, config, price_of, llm=llm) for j in judge._get_members()), Amount()
        )
    aliases = judge.models()
    if aliases:
        if not llm:
            return Amount()
        modeled = _model_amount(aliases, 2 * n, config, price_of, cache_write=True)
        return Amount(
            tokens=max(config.limits.llm_tranche_tokens, modeled.tokens),
            usd=_usd_from_micro_units(
                max(
                    Amount(usd=config.limits.llm_tranche_usd)._micro_usd,
                    modeled._micro_usd,
                )
            ),
        )
    if judge.providers():
        return _scale(Amount.from_estimate(config.limits.default_run_reservation), n)
    return Amount()


def _team_amount(
    plan: Plan,
    teams: Sequence[TeamSpec],
    config: RunConfig,
    tools: Mapping[str, Tool],
    price_of: Callable[[str], ModelSpec.Prices | None],
) -> Amount:
    """Call synchronous tool/manager hooks for one round of the supplied teams."""
    total = Amount()
    for team in teams:
        for agent in team.agents:
            tool = tools[f"{plan.goal.id}/{team.name}/{agent.name}"]
            estimate = tool.estimate(ToolRequest())
            run = config.limits.default_run_reservation if estimate is None else estimate
            total += _scale(Amount.from_estimate(run), config.limits.max_tool_runs_per_round)
            manager = agent.manager.build(Manager)
            total += _model_amount(
                manager.models(),
                config.limits.max_turns_per_round,
                config,
                price_of,
            )
    return total


def final_slice_estimate(
    plan: Plan,
    config: RunConfig,
    tools: Mapping[str, Tool],
    price_of: Callable[[str], ModelSpec.Prices | None],
) -> Amount:
    """
    Estimate convergence retries and the complete final ranking batch.

    Parameters
    ----------
    plan : Plan
        Plan supplying convergence settings and ranking judge.
    config : RunConfig
        Limits and model catalog.
    tools : Mapping[str, Tool]
        Built tools keyed by agent address, including converge agents.
    price_of : Callable[[str], ModelSpec.Prices | None]
        Synchronous model alias-price lookup.

    Returns
    -------
    Amount
        In team mode, one plus join attempts times the converge team's tool
        and manager costs, plus final judging of agents*tool runs+top_n.
        Select mode includes only final judging of top_n.

    Notes
    -----
    Calls component construction and user estimate/models/providers hooks
    synchronously; the engine offloads this phase to a daemon worker.

    Examples
    --------
    >>> from swarm_arena._model.goal import Goal
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[],
    ...             judges=[ProgramJudge("example:score")])
    >>> final_slice_estimate(plan, RunConfig(), {}, lambda a: None)
    Amount(tokens=0, usd=0.0)
    """
    total = Amount()
    n_final = plan.converge.top_n
    if plan.converge.mode == "team" and plan.converge.team is not None:
        team = plan.converge.team
        total = _scale(
            _team_amount(plan, [team], config, tools, price_of),
            1 + plan.converge.max_join_attempts,
        )
        n_final += len(team.agents) * config.limits.max_tool_runs_per_round
    ranking = plan.ranking_name()
    ref = next(ref for ref in plan.judges if ref.component_name() == ranking)
    return total + judge_amount(ref.build(Judge), n_final, config, price_of, llm=True)


def worst_case_estimate(
    plans: Sequence[Plan],
    config: RunConfig,
    tools: Mapping[str, Tool],
    price_of: Callable[[str], ModelSpec.Prices | None],
) -> Amount:
    """
    Estimate all planned rounds and final slices, including the run reserve.

    Parameters
    ----------
    plans : Sequence[Plan]
        Initial goal plans.
    config : RunConfig
        Limits, model catalog and reserved fraction.
    tools : Mapping[str, Tool]
        Built tools keyed by agent address across all plans.
    price_of : Callable[[str], ModelSpec.Prices | None]
        Synchronous model alias-price lookup.

    Returns
    -------
    Amount
        Round costs plus final slices, divided by 1-reserve_fraction. Tokens
        and micro-dollars round to the nearest integer with ties to even.

    Raises
    ------
    ValueError
        If the estimate exceeds the supported Amount dollar range.

    Notes
    -----
    Calls component construction and user estimate/models/providers hooks
    synchronously; the engine offloads this phase to a daemon worker.

    Examples
    --------
    >>> worst_case_estimate([], RunConfig(), {}, lambda a: None)
    Amount(tokens=0, usd=0.0)
    """
    total = Amount()
    for plan in plans:
        total += _scale(
            _team_amount(plan, plan.teams, config, tools, price_of),
            plan.rounds_total(config.limits),
        )
        total += final_slice_estimate(plan, config, tools, price_of)
    return _scale(total, 1 / (1 - Fraction(str(config.reserve_fraction))))
