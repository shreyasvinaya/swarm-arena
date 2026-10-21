import json
import math
import random
import sys
from collections.abc import Callable
from fractions import Fraction
from operator import add, sub
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena._budget import amount as amount_module
from swarm_arena._budget.amount import MICRO_USD, Amount
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.tooling import Estimate


def test_fits_in_respects_unset_caps() -> None:
    assert Amount(tokens=10**9, usd=5).fits_in(
        Amount(tokens=1, usd=5), cap_tokens=False, cap_usd=True
    )
    assert not Amount(usd=5.000001).fits_in(Amount(usd=5), cap_tokens=False, cap_usd=True)


def test_arithmetic() -> None:
    assert Amount(tokens=3, usd=1.5) - Amount(tokens=1, usd=0.5) == Amount(tokens=2, usd=1.0)


def test_frozen_defaults_and_serialization() -> None:
    amount = Amount()
    assert isinstance(amount, FrozenModel)
    assert amount.model_dump() == {"tokens": 0, "usd": 0.0}
    assert MICRO_USD == 1e-6
    normalized = Amount(tokens=-3, usd=-0.1234567)
    assert Amount.model_validate_json(normalized.model_dump_json()) == normalized
    with pytest.raises(ValidationError, match="frozen_instance"):
        amount.tokens = 1  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("usd", "expected"),
    [
        (0.1234564, 0.123456),
        (0.1234567, 0.123457),
        (-0.1234564, -0.123456),
        (-0.1234567, -0.123457),
        (0.0000005, 0.0),
        (0.0000015, 0.000002),
    ],
)
def test_normalizes_micro_dollars(usd: float, expected: float) -> None:
    assert Amount(usd=usd).usd == expected


def test_addition_and_repeated_subtraction_are_exact() -> None:
    assert Amount(usd=0.1) + Amount(usd=0.2) == Amount(usd=0.3)
    charged = Amount()
    for _ in range(6):
        charged = charged + Amount(tokens=1, usd=0.01)
    assert charged == Amount(tokens=6, usd=0.06)
    remaining = Amount(tokens=100, usd=1)
    for _ in range(100):
        remaining = remaining - Amount(tokens=1, usd=0.01)
    assert remaining == Amount()


def test_replay_subtraction_preserves_negative_amounts() -> None:
    assert Amount(tokens=1, usd=0.01) - Amount(tokens=3, usd=0.02) == Amount(tokens=-2, usd=-0.01)


@pytest.mark.parametrize(
    ("cap_tokens", "cap_usd"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_fits_in_checks_only_capped_units(cap_tokens: bool, cap_usd: bool) -> None:
    assert Amount(tokens=10, usd=2).fits_in(
        Amount(tokens=10, usd=2), cap_tokens=cap_tokens, cap_usd=cap_usd
    )
    assert Amount(tokens=11, usd=2).fits_in(
        Amount(tokens=10, usd=2), cap_tokens=cap_tokens, cap_usd=cap_usd
    ) is (not cap_tokens)
    assert Amount(tokens=10, usd=2.000001).fits_in(
        Amount(tokens=10, usd=2), cap_tokens=cap_tokens, cap_usd=cap_usd
    ) is (not cap_usd)


@pytest.mark.parametrize(
    ("estimate", "expected"),
    [
        (Estimate(), Amount()),
        (Estimate(seconds=1, tokens=7), Amount(tokens=7)),
        (Estimate(usd=0.1234567), Amount(usd=0.123457)),
        (Estimate(tokens=7, usd=0.5), Amount(tokens=7, usd=0.5)),
    ],
)
def test_from_estimate(estimate: Estimate, expected: Amount) -> None:
    assert Amount.from_estimate(estimate) == expected


@pytest.mark.parametrize(
    ("cap_tokens", "cap_usd"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_to_estimate_cap_flags(cap_tokens: bool, cap_usd: bool) -> None:
    estimate = Amount(tokens=7, usd=0.5).to_estimate(cap_tokens=cap_tokens, cap_usd=cap_usd)
    assert estimate == Estimate(tokens=7 if cap_tokens else None, usd=0.5 if cap_usd else None)
    assert Amount(tokens=7, usd=0.5).to_estimate() == Estimate(tokens=7, usd=0.5)
    assert Amount().to_estimate() == Estimate(tokens=0, usd=0)


@pytest.mark.parametrize("field", ["tokens", "usd"])
def test_to_estimate_rejects_capped_negative_units(field: str) -> None:
    amount = Amount.model_validate({field: -1})
    with pytest.raises(ValueError, match=field):
        amount.to_estimate()
    assert amount.to_estimate(cap_tokens=False, cap_usd=False) == Estimate()


@pytest.mark.parametrize("field", ["tokens", "usd"])
@pytest.mark.parametrize(
    "value", [-1, -0.0000001, float("nan"), float("inf"), float("-inf"), "private-estimate-value"]
)
def test_from_estimate_defensively_validates_malformed_objects(field: str, value: Any) -> None:
    malformed = Estimate.model_construct(**{field: value})
    with pytest.raises(ValueError, match=field) as caught:
        Amount.from_estimate(malformed)
    assert_secret_absent(caught.value, "private-estimate-value", sanitized=True)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), "private-amount-value"]
)
@pytest.mark.parametrize(
    "validate",
    [
        pytest.param(lambda value: Amount(usd=value), id="constructor"),
        pytest.param(lambda value: Amount.model_validate({"usd": value}), id="model_validate"),
        pytest.param(
            lambda value: Amount.model_validate_json(json.dumps({"usd": value})),
            id="model_validate_json",
        ),
    ],
)
def test_invalid_usd_is_sanitized(value: Any, validate: Callable[[Any], Amount]) -> None:
    with pytest.raises(ValidationError) as caught:
        validate(value)
    assert_secret_absent(caught.value, "private-amount-value", sanitized=True)


def test_supported_usd_range_preserves_micro_units_and_replay() -> None:
    assert amount_module.USD_LIMIT == 1_000_000_000
    limit_units = 1_000_000_000_000_000
    rng = random.Random(8401)
    samples = [
        -limit_units,
        -limit_units + 1,
        -1,
        0,
        1,
        limit_units - 1,
        limit_units,
        *(rng.randint(-limit_units, limit_units) for _ in range(2_000)),
    ]
    for units in samples:
        usd = units / 1_000_000
        assert round(float(usd) * 1_000_000) == units
        amount = Amount(usd=usd)
        assert Fraction(str(amount.usd)) * 1_000_000 == units
        replayed = Amount.model_validate_json(amount.model_dump_json())
        assert Fraction(str(replayed.usd)) * 1_000_000 == units


@pytest.mark.parametrize(
    "usd",
    [
        math.nextafter(1_000_000_000.0, math.inf),
        1_000_000_000.000001,
        1e10,
        1e12,
        sys.float_info.max,
    ],
)
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(
    "validate",
    [
        pytest.param(lambda usd: Amount(usd=usd), id="constructor"),
        pytest.param(lambda usd: Amount.model_validate({"usd": usd}), id="model_validate"),
        pytest.param(
            lambda usd: Amount.model_validate_json(json.dumps({"usd": usd})),
            id="model_validate_json",
        ),
    ],
)
def test_out_of_range_usd_is_refused_before_rounding(
    usd: float, sign: int, validate: Callable[[float], Amount]
) -> None:
    value = sign * usd
    with pytest.raises(ValueError, match="amount exceeds the supported USD range") as caught:
        validate(value)
    assert_secret_absent(caught.value, str(value), sanitized=True)


@pytest.mark.parametrize(
    ("left", "right", "operation"),
    [
        (1_000_000_000.0, 0.000001, add),
        (-1_000_000_000.0, -0.000001, add),
        (1_000_000_000.0, -0.000001, sub),
        (-1_000_000_000.0, 0.000001, sub),
        (1_000_000_000.0, 1_000_000_000.0, add),
        (-1_000_000_000.0, 1_000_000_000.0, sub),
    ],
)
def test_arithmetic_refuses_out_of_range_result(
    left: float, right: float, operation: Callable[[Amount, Amount], Amount]
) -> None:
    with pytest.raises(ValueError, match="amount exceeds the supported USD range") as caught:
        operation(Amount(usd=left), Amount(usd=right))
    for value in (left, right):
        assert_secret_absent(caught.value, str(value), sanitized=True)


@pytest.mark.parametrize("sign", [-1, 1])
def test_tiny_debits_and_credits_at_range_ends_are_exact(sign: int) -> None:
    balance = Amount(usd=sign * 1_000_000_000)
    debit = Amount(usd=sign * 0.000001)
    for count in range(1, 1_002):
        balance = balance - debit
        assert Fraction(str(balance.usd)) * 1_000_000 == sign * (1_000_000_000_000_000 - count)
    for count in range(1_000, -1, -1):
        balance = balance + debit
        assert Fraction(str(balance.usd)) * 1_000_000 == sign * (1_000_000_000_000_000 - count)
    assert balance - balance == Amount()
    assert balance + Amount(usd=-balance.usd) == Amount()


def test_from_estimate_refuses_unsupported_pool() -> None:
    with pytest.raises(ValueError, match="amount exceeds the supported USD range"):
        Amount.from_estimate(Estimate(usd=1e10))


def test_range_diagnostic_survives_sanitizer_without_value() -> None:
    with pytest.raises(ValidationError) as caught:
        Amount(usd=1_234_567_890.123456)
    assert caught.value.errors() == [
        {
            "type": "value_error",
            "loc": ("usd",),
            "msg": "amount exceeds the supported USD range",
            "input": None,
        }
    ]
    assert_secret_absent(caught.value, "1234567890.123456", sanitized=True)
