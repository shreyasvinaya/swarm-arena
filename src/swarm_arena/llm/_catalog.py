"""Price and construct approved models using LiteLLM's pinned bundled facts."""

import os
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

from swarm_arena._budget.amount import Amount, _usd_from_micro_units
from swarm_arena._budget.estimate import register_price_source
from swarm_arena._model.base import LLMRole
from swarm_arena._model.catalog import ModelSpec
from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.tooling import Usage
from swarm_arena._planning.validate import register_model_check

_already_local = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"
_already_imported = "litellm" in sys.modules
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# isort: split
import litellm  # noqa: E402 - the environment pin must precede this import
from agents import Model  # noqa: E402
from agents.extensions.models.litellm_model import LitellmModel  # noqa: E402

if _already_imported and not _already_local:
    litellm.model_cost = litellm.get_model_cost_map(url=litellm.model_cost_map_url)

_FAKE_MODELS: dict[str, Model] = {}


def _price_entry(model: str) -> Mapping[str, Any]:
    """Read the pinned entry, with the provider-free spelling as fallback."""
    model = _context_string(model, "model")
    return litellm.model_cost.get(model, litellm.model_cost.get(model.partition("/")[2], {}))


def price_of_model(model: str) -> ModelSpec.Prices | None:
    """
    Look up bundled provider prices in USD per million tokens.

    Parameters
    ----------
    model : str
        Provider/model identifier, with an unprefixed fallback lookup.

    Returns
    -------
    ModelSpec.Prices or None
        Known prices, retaining the five-minute write tariff; None when absent
        or both ordinary input and output prices are zero.

    Examples
    --------
    >>> price_of_model("fake/unknown") is None
    True
    """
    entry = _price_entry(model)

    def price(field: str) -> float | None:
        value = entry.get(field)
        return None if value is None else float(Fraction(str(value)) * 1_000_000)

    input_price = price("input_cost_per_token") or 0.0
    output_price = price("output_cost_per_token") or 0.0
    if input_price == output_price == 0:
        return None
    return ModelSpec.Prices(
        input=input_price,
        output=output_price,
        cache_read=price("cache_read_input_token_cost"),
        cache_write=price("cache_creation_input_token_cost"),
    )


def prices_for(spec: ModelSpec) -> ModelSpec.Prices | None:
    """
    Prefer explicit catalog prices over the bundled provider map.

    Parameters
    ----------
    spec : ModelSpec
        Approved model specification.

    Returns
    -------
    ModelSpec.Prices or None
        Catalog prices, otherwise known provider prices.

    Examples
    --------
    >>> prices_for(ModelSpec(model="fake/m", roles=["judge"])) is None
    True
    """
    return spec.prices if spec.prices is not None else price_of_model(spec.model)


def _cost_micros(spec: ModelSpec, usage: Usage) -> Fraction:
    """Sum exact token-price products before one micro-dollar rounding.

    Catalog write prices apply to every TTL. Provider fallback retains known
    5m/1h subsets. Unsplit writes use the maximum of both provider tariffs and
    twice ordinary input, even when the requested TTL was five minutes.
    """
    prices = prices_for(spec)
    if prices is None:
        return Fraction(0)
    input_price = Fraction(str(prices.input))
    read = input_price if prices.cache_read is None else Fraction(str(prices.cache_read))
    write = 2 * input_price if prices.cache_write is None else Fraction(str(prices.cache_write))
    cost = usage.input_tokens * input_price + usage.output_tokens * Fraction(str(prices.output))
    cost += usage.cache_read_tokens * read
    if spec.prices is not None:
        return cost + usage.cache_write_tokens * write
    one_hour = _price_entry(spec.model).get("cache_creation_input_token_cost_above_1hr")
    hour = max(2 * input_price, write) if one_hour is None else Fraction(str(one_hour)) * 1_000_000
    remainder = usage.cache_write_tokens - usage.cache_write_5m_tokens - usage.cache_write_1h_tokens
    if remainder < 0:
        raise ValueError("cache write subsets exceed cache_write_tokens")
    return (
        cost
        + usage.cache_write_5m_tokens * write
        + usage.cache_write_1h_tokens * hour
        + remainder * max(write, hour, 2 * input_price)
    )


def call_cost(spec: ModelSpec, usage: Usage, *, litellm_cost: float | None) -> float:
    """
    Calculate one normalized actual charge without capping it at an estimate.

    Parameters
    ----------
    spec : ModelSpec
        Catalog prices override even an authoritative provider response cost.
    usage : Usage
        Validated counters; ordinary input excludes cache reads and writes.
    litellm_cost : float or None
        Authoritative response cost, used only without catalog prices.

    Returns
    -------
    float
        Dollars rounded once to whole micro-dollars, ties to even. Without
        catalog or provider prices and without response cost, returns zero.

    Raises
    ------
    ValueError
        If the charge is negative, nonfinite, unsupported or the TTL split is invalid.

    Examples
    --------
    >>> spec = ModelSpec(model="fake/m", roles=["judge"],
    ...                  prices=ModelSpec.Prices(input=1, output=2))
    >>> call_cost(spec, Usage(input_tokens=1_000_000), litellm_cost=99)
    1.0
    """
    if spec.prices is None and litellm_cost is not None:
        if litellm_cost < 0:
            raise ValueError("invalid response cost")
        return Amount(usd=litellm_cost).usd
    return _usd_from_micro_units(round(_cost_micros(spec, usage)))


def call_estimate(spec: ModelSpec, *, prompt_chars: int, max_tokens: int) -> Amount:
    """
    Estimate a complete call with conservative cache-write input pricing.

    Parameters
    ----------
    spec : ModelSpec
        Catalog specification and optional prices.
    prompt_chars : int
        Full serialized input length, including tools and schema.
    max_tokens : int
        Output token allowance.

    Returns
    -------
    Amount
        Ceil(prompt_chars/3) input tokens plus the output allowance. Each input
        token costs max(2*input, cache_write), covering both cache lifetimes.
        Character conversion is an estimate, not a token bound.

    Examples
    --------
    >>> call_estimate(ModelSpec(model="fake/m", roles=["judge"]),
    ...               prompt_chars=4, max_tokens=10)
    Amount(tokens=12, usd=0.0)
    """
    inputs = (prompt_chars + 2) // 3
    prices = prices_for(spec)
    micros = Fraction(0)
    if prices is not None:
        micros = inputs * max(
            2 * Fraction(str(prices.input)), Fraction(str(prices.cache_write or 0))
        )
        micros += max_tokens * Fraction(str(prices.output))
    return Amount(tokens=inputs + max_tokens, usd=_usd_from_micro_units(round(micros)))


def supports(spec: ModelSpec, feature: Literal["function_calling", "response_schema"]) -> bool:
    """
    Check a bundled provider capability, allowing all offline fake models.

    Parameters
    ----------
    spec : ModelSpec
        Approved model specification.
    feature : {'function_calling', 'response_schema'}
        Capability to query.

    Returns
    -------
    bool
        Whether the provider supports the requested feature.

    Examples
    --------
    >>> supports(ModelSpec(model="fake/m", roles=["judge"]), "response_schema")
    True
    """
    if spec.model.startswith("fake/"):
        return True
    check = (
        litellm.supports_function_calling
        if feature == "function_calling"
        else litellm.supports_response_schema
    )
    return bool(check(model=spec.model))


def check_capabilities(
    uses: Mapping[LLMRole, Sequence[str]], config: RunConfig
) -> tuple[list[str], list[str]]:
    """
    Validate manager tools and warn about structured-output fallback.

    Parameters
    ----------
    uses : Mapping[LLMRole, Sequence[str]]
        Role-to-alias declarations from admitted components.
    config : RunConfig
        Human-approved model catalog.

    Returns
    -------
    tuple[list[str], list[str]]
        Function-calling errors and two-stage structured-output warnings.
        Unknown or disallowed aliases are handled by core alias validation.

    Examples
    --------
    >>> check_capabilities({}, RunConfig())
    ([], [])
    """
    errors: list[str] = []
    warnings: list[str] = []
    for role, aliases in uses.items():
        role = _context_string(role, "role")
        for alias in dict.fromkeys(_context_string(a, "alias") for a in aliases):
            spec = config.models.root.get(alias)
            if spec is None or role not in spec.roles:
                continue
            if role == "manager" and not supports(spec, "function_calling"):
                errors.append(f"alias {alias!r} for role 'manager' requires function calling")
            elif role in {"judge", "planner", "curator"} and not supports(spec, "response_schema"):
                warnings.append(f"alias {alias!r} for role {role!r} uses the two-stage fallback")
    return errors, warnings


def register_fake_model(name: str, model: Model) -> None:
    """
    Register an offline model instance under a fake-provider name.

    Parameters
    ----------
    name : str
        Name after the ``fake/`` prefix.
    model : agents.Model
        Offline instance returned unchanged by the factory.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> register_fake_model("example", Mock(spec=Model))
    >>> _ = _FAKE_MODELS.pop("example")
    """
    _FAKE_MODELS[_context_string(name, "name")] = model


def build_model(spec: ModelSpec) -> Model:
    """
    Construct a LiteLLM adapter or return a registered offline model.

    Parameters
    ----------
    spec : ModelSpec
        Approved provider/model specification.

    Returns
    -------
    agents.Model
        Registered fake instance or a new LitellmModel adapter.

    Raises
    ------
    ValueError
        If a fake model has not been registered.

    Examples
    --------
    >>> isinstance(build_model(ModelSpec(model="openai/gpt-6", roles=["manager"])), LitellmModel)
    True
    """
    model = _context_string(spec.model, "model")
    if model.startswith("fake/"):
        name = model.removeprefix("fake/")
        if name not in _FAKE_MODELS:
            raise ValueError("fake model is not registered")
        return _FAKE_MODELS[name]
    return LitellmModel(model=model)


register_price_source(price_of_model)
register_model_check(check_capabilities)
