from collections.abc import Iterator, Sequence

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import PyTool

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    Goal,
    Limits,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    Score,
    TeamSpec,
)
from swarm_arena._budget import estimate as estimate_module
from swarm_arena._budget.amount import USD_LIMIT, Amount
from swarm_arena._budget.estimate import (
    final_slice_estimate,
    judge_amount,
    make_price_of,
    register_price_source,
    worst_case_estimate,
)
from swarm_arena._model.catalog import ModelCatalog, ModelSpec
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import Estimate, ToolRequest
from swarm_arena._scoring.judge import Judge, JudgeContext
from swarm_arena._scoring.lexicographic import Lexicographic
from swarm_arena._scoring.script_judge import ScriptJudge
from swarm_arena._tooling.tool import Tool


class _FixedEstimateTool(PyTool):
    def __init__(self, usd: float) -> None:
        super().__init__({"code": "pass"})
        self._usd = usd

    def estimate(self, request: ToolRequest) -> Estimate:
        return Estimate(usd=self._usd)


class _FreeTool(_FixedEstimateTool):
    def __init__(self) -> None:
        super().__init__(0)

    def estimate(self, request: ToolRequest) -> Estimate:
        return Estimate()


class _NoEstimateTool(PyTool):
    def __init__(self) -> None:
        super().__init__({"code": "pass"})


class _FakeLLMJudge(Judge):
    name = "llm"

    def __init__(self, models: tuple[str, ...]) -> None:
        super().__init__()
        self._models = models

    def models(self) -> tuple[str, ...]:
        return self._models

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return []


def _provider_judge() -> Judge:
    return Ref(path="_fakes.tools:ProviderJudge").build(Judge)


def _script_judge() -> Judge:
    return ScriptJudge(command=["true"], direction="min")


def _plan(
    *,
    teams: int = 1,
    agents_per_team: int = 1,
    rounds: int = 1,
    ranking_judge: Judge | None = None,
    top_n: int = 2,
    models: bool = False,
) -> Plan:
    judge = ProgramJudge("_fakes.tools:rosen_score") if ranking_judge is None else ranking_judge
    return Plan(
        goal=Goal(id="g1", namespace="test"),
        teams=[_team(f"t{i}", agents_per_team, models=models) for i in range(teams)],
        judges=[Ref.of(judge)],
        ranking=judge.name,
        rounds=[RoundSpec() for _ in range(rounds)],
        converge=ConvergeSpec(top_n=top_n),
    )


def _team(name: str, agents: int, *, models: bool = False) -> TeamSpec:
    return TeamSpec(
        name=name,
        rationale="Explore",
        agents=[
            AgentSpec(
                name=f"a{i}",
                rationale="Explore",
                tool=Ref(path="_fakes.tools:PyTool"),
                manager=Ref(
                    path="_fakes.tools:NoopManager", config={"models": ["opus"]} if models else {}
                ),
            )
            for i in range(agents)
        ],
    )


def _plan_with_converge_team(
    *,
    agents: int = 2,
    max_join_attempts: int = 1,
    ranking_judge: Judge | None = None,
    top_n: int = 2,
    models: bool = False,
) -> Plan:
    plan = _plan(ranking_judge=ranking_judge, top_n=top_n)
    converge = ConvergeSpec(
        mode="team",
        team=_team("final", agents, models=models),
        top_n=top_n,
        max_join_attempts=max_join_attempts,
    )
    return plan.model_copy(update={"converge": converge})


def _tools(plan: Plan, tool: Tool) -> dict[str, Tool]:
    teams = [*plan.teams]
    if plan.converge.team is not None:
        teams.append(plan.converge.team)
    return {f"{plan.goal.id}/{team.name}/{a.name}": tool for team in teams for a in team.agents}


def _cfg(
    *,
    limits: Limits | None = None,
    alias: str = "opus",
    max_tokens: int = 8000,
    input: float = 5,
    output: float = 25,
) -> RunConfig:
    return RunConfig(
        limits=limits or Limits(),
        models=ModelCatalog(
            {
                alias: ModelSpec(
                    model="openai/example",
                    roles=["manager", "judge"],
                    max_tokens=max_tokens,
                    prices=ModelSpec.Prices(input=input, output=output),
                )
            }
        ),
    )


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    previous = estimate_module._price_sources[:]
    estimate_module._price_sources.clear()
    yield
    estimate_module._price_sources[:] = previous


def test_worst_case_tool_runs_plus_reserve() -> None:
    plan = _plan(teams=2, agents_per_team=2, rounds=3)
    total = worst_case_estimate(
        [plan], RunConfig(), _tools(plan, _FixedEstimateTool(usd=0.5)), lambda a: None
    )
    assert total.usd == pytest.approx(3 * 4 * 3 * 0.5 / 0.85)


def test_no_estimate_uses_default_run_reservation() -> None:
    plan = _plan()
    total = worst_case_estimate(
        [plan], RunConfig(), _tools(plan, _NoEstimateTool()), lambda a: None
    )
    assert total.tokens == round(3 * 100_000 / 0.85)
    assert total.usd == pytest.approx(3 * 1.0 / 0.85)


def test_manager_tokens_priced_at_max_of_input_output() -> None:
    plan, cfg = _plan(models=True), _cfg()
    total = worst_case_estimate([plan], cfg, _tools(plan, _FreeTool()), make_price_of(cfg.models))
    assert total.tokens == round(20 * 8000 / 0.85)
    assert total.usd == pytest.approx(20 * 8000 * 25 / 1e6 / 0.85)


def test_final_slice_covers_join_attempts() -> None:
    plan = _plan_with_converge_team(agents=2, max_join_attempts=1)
    assert final_slice_estimate(
        plan, RunConfig(), _tools(plan, _FixedEstimateTool(usd=0.5)), lambda a: None
    ).usd == pytest.approx(2 * 2 * 3 * 0.5)


def test_final_slice_covers_converge_managers() -> None:
    plan, cfg = _plan_with_converge_team(models=True), _cfg()
    total = final_slice_estimate(plan, cfg, _tools(plan, _FreeTool()), make_price_of(cfg.models))
    assert total.tokens == 2 * 2 * 20 * 8000
    assert total.usd == pytest.approx(2 * 2 * 20 * 8000 * 25 / 1e6)


def test_final_slice_final_judge_batch_in_team_mode() -> None:
    plan = _plan_with_converge_team(agents=2, ranking_judge=_provider_judge(), top_n=2)
    assert final_slice_estimate(
        plan, RunConfig(), _tools(plan, _FreeTool()), lambda a: None
    ).usd == pytest.approx((2 * 3 + 2) * 1.0)


def test_final_slice_adds_the_final_judge() -> None:
    plan = _plan(ranking_judge=_provider_judge(), top_n=2)
    assert final_slice_estimate(
        plan, RunConfig(), _tools(plan, _FixedEstimateTool(usd=0.5)), lambda a: None
    ) == Amount(tokens=200_000, usd=2.0)


def test_judge_amount() -> None:
    cfg = RunConfig()

    def price_of(alias: str) -> ModelSpec.Prices | None:
        return None

    assert judge_amount(ProgramJudge("_fakes.tools:rosen_score"), 3, cfg, price_of) == Amount()
    assert judge_amount(_script_judge(), 3, cfg, price_of) == Amount()
    assert judge_amount(_provider_judge(), 3, cfg, price_of) == Amount(tokens=300_000, usd=3.0)
    assert judge_amount(_FakeLLMJudge(models=("opus",)), 3, cfg, price_of, llm=False) == Amount()
    assert judge_amount(
        Lexicographic([ProgramJudge("_fakes.tools:rosen_score"), _provider_judge()]),
        3,
        cfg,
        price_of,
    ) == Amount(tokens=300_000, usd=3.0)


def test_judge_amount_prices_llm_judge_models() -> None:
    cfg = _cfg(
        alias="j",
        max_tokens=1000,
        input=1,
        output=2,
        limits=Limits(llm_tranche_tokens=1_000, llm_tranche_usd=0.001),
    )
    judge = _FakeLLMJudge(models=("j", "j"))
    amount = judge_amount(judge, 3, cfg, make_price_of(cfg.models))
    assert amount.tokens == 12_000
    assert amount.usd == pytest.approx(12_000 * 2 / 1e6)
    assert judge_amount(judge, 3, cfg, make_price_of(cfg.models), llm=False) == Amount()
    big = _cfg(alias="j", max_tokens=1000, input=1, output=2)
    assert judge_amount(judge, 3, big, make_price_of(big.models)) == Amount(tokens=25_000, usd=0.25)


def test_registered_source_used_after_catalog_prices() -> None:
    prices = ModelSpec.Prices(input=1, output=2)
    register_price_source(lambda model: prices if model == "x/y" else None)
    catalog = ModelCatalog(
        {
            "a": ModelSpec(model="x/y", roles=["judge"]),
            "b": ModelSpec(
                model="x/y", roles=["judge"], prices=ModelSpec.Prices(input=3, output=4)
            ),
            "c": ModelSpec(model="x/z", roles=["judge"]),
        }
    )
    price_of = make_price_of(catalog)
    assert price_of("a") == prices
    assert price_of("b") == catalog.root["b"].prices
    assert price_of("c") is None


def test_worst_case_includes_final_slice_and_default_rounds() -> None:
    plan = _plan_with_converge_team(ranking_judge=_provider_judge())
    plan = plan.model_copy(update={"rounds": []})
    cfg = RunConfig()
    tools = _tools(plan, _FixedEstimateTool(0.5))
    amount = worst_case_estimate([plan], cfg, tools, lambda a: None)
    assert amount == Amount(tokens=round(800_000 / 0.85), usd=round((4.5 + 6 + 8) / 0.85, 6))


def test_zero_estimate_and_missing_price_still_count_tokens() -> None:
    plan, cfg = _plan(models=True), _cfg()
    amount = worst_case_estimate([plan], cfg, _tools(plan, _FreeTool()), lambda a: None)
    assert amount == Amount(tokens=round(160_000 / 0.85))


def test_estimate_refuses_out_of_range_result_without_values() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="supported USD range") as exc:
        worst_case_estimate(
            [plan], RunConfig(), _tools(plan, _FixedEstimateTool(USD_LIMIT)), lambda a: None
        )
    assert_secret_absent(exc.value, str(USD_LIMIT))


def test_judge_cache_write_price_and_componentwise_maximum() -> None:
    cfg = _cfg(
        alias="j",
        max_tokens=1000,
        input=5,
        output=2,
        limits=Limits(llm_tranche_tokens=25_000, llm_tranche_usd=0.001),
    )
    amount = judge_amount(_FakeLLMJudge(("j",)), 3, cfg, make_price_of(cfg.models))
    assert amount == Amount(tokens=25_000, usd=0.06)
