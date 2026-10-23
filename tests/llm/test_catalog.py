import os
import subprocess
import sys

import pytest
from agents.extensions.models.litellm_model import LitellmModel
from agents.models.interface import Model

from swarm_arena._budget.amount import Amount
from swarm_arena._model.base import LLMRole
from swarm_arena._model.catalog import ModelCatalog, ModelSpec
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.tooling import Usage
from swarm_arena.llm import _catalog
from swarm_arena.llm._catalog import (
    build_model,
    call_cost,
    call_estimate,
    check_capabilities,
    price_of_model,
    prices_for,
    register_fake_model,
    supports,
)


def test_cost_map_pinned_at_import() -> None:
    env = dict(os.environ)
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import swarm_arena.llm, os; print(os.environ['LITELLM_LOCAL_MODEL_COST_MAP'])",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert proc.stdout.splitlines()[0] == "True"


def test_cost_map_repinned_when_litellm_imported_first() -> None:
    env = dict(os.environ)
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    env["LITELLM_MODEL_COST_MAP_URL"] = "http://127.0.0.1:9/"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import litellm; litellm.model_cost['zz-test'] = {}; "
            "import swarm_arena.llm; print('zz-test' in litellm.model_cost)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert proc.stdout.strip() == "False"


@pytest.fixture
def spec_in1_out2() -> ModelSpec:
    return ModelSpec(model="fake/m", roles=["manager"], prices=ModelSpec.Prices(input=1, output=2))


def test_price_of_model_from_pinned_map() -> None:
    p = price_of_model("anthropic/claude-sonnet-4-6")
    assert p is not None
    assert (p.input, p.cache_read, p.cache_write) == (3.0, 0.3, 3.75)
    assert p.output > p.input


def test_price_of_unknown_model_is_none() -> None:
    assert price_of_model("fake/m") is None


def test_catalog_prices_override_litellm_cost() -> None:
    spec = ModelSpec(
        model="gemini/gemini-3-flash",
        roles=["judge", "curator"],
        prices=ModelSpec.Prices(input=0.3, output=2.5),
    )
    assert call_cost(
        spec, Usage(input_tokens=1_000_000, output_tokens=1_000_000), litellm_cost=99.0
    ) == pytest.approx(2.8)


def test_litellm_cost_used_without_catalog_prices() -> None:
    assert (
        call_cost(
            ModelSpec(model="openai/gpt-6", roles=["manager"]),
            Usage(input_tokens=10),
            litellm_cost=0.123,
        )
        == 0.123
    )


def test_missing_catalog_cache_prices_are_conservative(spec_in1_out2: ModelSpec) -> None:
    usage = Usage(input_tokens=250_000, cache_read_tokens=500_000, cache_write_tokens=250_000)
    assert call_cost(spec_in1_out2, usage, litellm_cost=None) == pytest.approx(1.25)


def test_call_estimate_formula(spec_in1_out2: ModelSpec) -> None:
    assert call_estimate(spec_in1_out2, prompt_chars=3_000, max_tokens=8_000) == Amount(
        tokens=9_000, usd=0.018
    )
    assert call_estimate(spec_in1_out2, prompt_chars=4, max_tokens=1).tokens == 3


def test_call_estimate_covers_an_all_cache_write_prompt(spec_in1_out2: ModelSpec) -> None:
    est = call_estimate(spec_in1_out2, prompt_chars=3_000, max_tokens=8_000)
    assert est.usd >= call_cost(
        spec_in1_out2, Usage(cache_write_tokens=1_000, output_tokens=8_000), litellm_cost=None
    )


def test_call_estimate_covers_one_hour_cache_writes() -> None:
    spec = ModelSpec(
        model="anthropic/claude-sonnet-4-6",
        roles=["manager"],
        prompt_cache=ModelSpec.PromptCache(ttl="1h"),
    )
    p = price_of_model(spec.model)
    assert p is not None
    assert (
        call_estimate(spec, prompt_chars=3_000, max_tokens=8_000).usd
        >= (1_000 * 2 * p.input + 8_000 * p.output) / 1e6
    )


@pytest.mark.parametrize(
    ("five", "hour", "total", "micros"),
    [(100, 900, 1000, 5775), (100, 0, 1000, 5775), (1000, 0, 1000, 3750), (0, 0, 1000, 6000)],
)
def test_provider_fallback_preserves_ttl_split(
    five: int, hour: int, total: int, micros: int
) -> None:
    spec = ModelSpec(model="anthropic/claude-sonnet-4-6", roles=["manager"])
    usage = Usage(cache_write_tokens=total, cache_write_5m_tokens=five, cache_write_1h_tokens=hour)
    assert Amount(usd=call_cost(spec, usage, litellm_cost=None))._micro_usd == micros


def test_catalog_override_ignores_provider_ttl_tariffs() -> None:
    spec = ModelSpec(
        model="anthropic/claude-sonnet-4-6",
        roles=["judge"],
        prices=ModelSpec.Prices(input=1, output=2, cache_write=0.5, cache_read=0),
    )
    assert (
        call_cost(
            spec,
            Usage(cache_write_tokens=1000, cache_write_1h_tokens=1000, cache_read_tokens=1000),
            litellm_cost=99,
        )
        == 0.0005
    )
    assert prices_for(spec) == spec.prices


@pytest.mark.parametrize(("tokens", "micros"), [(1, 0), (3, 2), (5, 2)])
def test_cost_rounds_combined_rational_charge_once(tokens: int, micros: int) -> None:
    spec = ModelSpec(
        model="fake/m", roles=["judge"], prices=ModelSpec.Prices(input=0.1, output=0.4)
    )
    assert (
        Amount(
            usd=call_cost(spec, Usage(input_tokens=tokens, output_tokens=tokens), litellm_cost=None)
        )._micro_usd
        == micros
    )


def test_manager_model_needs_function_calling() -> None:
    config = RunConfig(
        models=ModelCatalog(
            {"r1": ModelSpec(model="deepseek/deepseek-reasoner", roles=["manager"])}
        )
    )
    errors, _ = check_capabilities({"manager": ["r1"]}, config)
    assert any("function calling" in e for e in errors)


@pytest.mark.parametrize("role", ["judge", "planner", "curator"])
def test_structured_role_without_response_schema_warns_two_stage(
    role: LLMRole, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_catalog.litellm, "supports_response_schema", lambda **kw: False)
    config = RunConfig(models=ModelCatalog({"m": ModelSpec(model="openai/gpt-6", roles=[role])}))
    errors, warnings = check_capabilities({role: ["m", "m"]}, config)
    assert errors == []
    assert len(warnings) == 1
    assert "two-stage" in warnings[0]


def test_build_model_litellm_and_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    monkeypatch.setattr(_catalog, "_FAKE_MODELS", {})
    fake = Mock(spec=Model)
    register_fake_model("m", fake)
    assert build_model(ModelSpec(model="fake/m", roles=["judge"])) is fake
    assert isinstance(build_model(ModelSpec(model="openai/gpt-6", roles=["manager"])), LitellmModel)
    with pytest.raises(ValueError, match="not registered"):
        build_model(ModelSpec(model="fake/missing", roles=["manager"]))
    assert supports(ModelSpec(model="fake/m", roles=["manager"]), "function_calling")
    assert supports(ModelSpec(model="fake/m", roles=["judge"]), "response_schema")


def test_price_lookup_prefix_fallback_and_zero_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _catalog.litellm,
        "model_cost",
        {
            "m": {"input_cost_per_token": 0.0000001, "output_cost_per_token": 0.0000002},
            "free": {"input_cost_per_token": 0, "output_cost_per_token": 0},
        },
    )
    assert price_of_model("provider/m") == ModelSpec.Prices(input=0.1, output=0.2)
    assert price_of_model("provider/free") is None
    assert prices_for(ModelSpec(model="provider/m", roles=["judge"])) == price_of_model(
        "provider/m"
    )


def test_estimate_respects_expensive_catalog_write_price() -> None:
    spec = ModelSpec(
        model="fake/m", roles=["manager"], prices=ModelSpec.Prices(input=1, output=2, cache_write=7)
    )
    assert call_estimate(spec, prompt_chars=3000, max_tokens=1000) == Amount(tokens=2000, usd=0.009)
    assert (
        call_cost(spec, Usage(cache_write_tokens=1000, output_tokens=1000), litellm_cost=None)
        == 0.009
    )


def test_unknown_uncapped_model_costs_zero_without_provider_report() -> None:
    spec = ModelSpec(model="fake/unpriced", roles=["manager"])
    assert call_cost(spec, Usage(input_tokens=10), litellm_cost=None) == 0


@pytest.mark.parametrize(("cost", "micros"), [(0.0000005, 0), (0.0000015, 2), (0.0000025, 2)])
def test_authoritative_cost_uses_core_rounding(cost: float, micros: int) -> None:
    spec = ModelSpec(model="fake/m", roles=["manager"])
    assert Amount(usd=call_cost(spec, Usage(), litellm_cost=cost))._micro_usd == micros


def test_cost_is_not_capped_by_reservation() -> None:
    spec = ModelSpec(model="fake/m", roles=["manager"])
    estimate = call_estimate(spec, prompt_chars=0, max_tokens=1)
    assert call_cost(spec, Usage(output_tokens=10), litellm_cost=1) > estimate.usd


def test_unregistered_fake_error_omits_model_value() -> None:
    from _fakes.secrecy import assert_secret_absent

    sentinel = "secret-provider-setting"
    with pytest.raises(ValueError, match="not registered") as error:
        build_model(ModelSpec(model=f"fake/{sentinel}", roles=["manager"]))
    assert_secret_absent(error.value, sentinel)
