from fractions import Fraction
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._budget.amount import Amount
from swarm_arena._budget.split import equal_shares, round_pool, weighted_shares


def test_equal_shares_never_exceed_total() -> None:
    shares = equal_shares(Amount(tokens=100, usd=1.0), ["a", "b", "c"])
    assert all(s == Amount(tokens=33, usd=0.333333) for s in shares.values())
    assert sum(s.usd for s in shares.values()) <= 1.0


def test_weighted_shares() -> None:
    assert weighted_shares(Amount(usd=8.0), {"g1": 3.0, "g2": 1.0}) == {
        "g1": Amount(usd=6.0),
        "g2": Amount(usd=2.0),
    }


def test_round_pool_divides_by_rounds_left() -> None:
    assert round_pool(Amount(tokens=90, usd=0.9), 3) == Amount(tokens=30, usd=0.3)


def test_zero_weight_and_empty_holders() -> None:
    assert equal_shares(Amount(usd=1.0), []) == {}
    with pytest.raises(ValueError, match="holder g"):
        weighted_shares(Amount(usd=1.0), {"g": 0.0})


def test_empty_weights_and_single_holder() -> None:
    total = Amount(tokens=17, usd=0.123456)
    assert weighted_shares(total, {}) == {}
    assert equal_shares(total, ["a"]) == {"a": total}
    assert weighted_shares(total, {"a": 0.1}) == {"a": total}
    assert round_pool(total, 1) == total


def test_remainders_stay_in_parent() -> None:
    total = Amount(tokens=100, usd=1)
    distributed = Amount()
    for share in equal_shares(total, ["a", "b", "c"]).values():
        distributed = distributed + share
    assert total - distributed == Amount(tokens=1, usd=0.000001)
    shares = weighted_shares(Amount(tokens=11, usd=0.000011), {"a": 2, "b": 1})
    assert shares == {"a": Amount(tokens=7, usd=0.000007), "b": Amount(tokens=3, usd=0.000003)}
    assert round_pool(Amount(tokens=11, usd=0.000011), 3) == Amount(tokens=3, usd=0.000003)


@pytest.mark.parametrize("total_units", [0, 1, 2, 7, 100, 1_000_001, 2**53 + 1])
@pytest.mark.parametrize(
    "weights",
    [
        {"a": 0.1, "b": 0.2},
        {"a": 1e308, "b": 1e308},
        {"a": 5e-324, "b": 5e-324},
        {"a": 1e-200, "b": 1e200},
    ],
)
def test_weights_use_exact_floor_arithmetic(total_units: int, weights: dict[str, float]) -> None:
    # Large tokens catch accidental float division without relying on USD float sums.
    total = Amount(tokens=total_units, usd=0.000007)
    shares = weighted_shares(total, weights)
    rational_weights = {key: Fraction(str(weight)) for key, weight in weights.items()}
    denominator = sum(rational_weights.values())
    for key, weight in rational_weights.items():
        assert shares[key] == Amount(
            tokens=int(total.tokens * weight // denominator),
            usd=int(7 * weight // denominator) / 1_000_000,
        )
    assert sum(share.tokens for share in shares.values()) <= total.tokens
    assert sum(round(share.usd * 1_000_000) for share in shares.values()) <= 7


def test_signed_pools_are_floored() -> None:
    total = Amount(tokens=-5, usd=-0.000005)
    expected = {"a": Amount(tokens=-3, usd=-0.000003), "b": Amount(tokens=-3, usd=-0.000003)}
    assert equal_shares(total, ["a", "b"]) == expected
    assert weighted_shares(total, {"a": 1, "b": 1}) == expected
    assert round_pool(total, 2) == expected["a"]


@pytest.mark.parametrize("usd", [1e100, 1e308, -1e100])
@pytest.mark.parametrize("count", [3, 7])
def test_large_finite_pools_never_round_shares_up(usd: float, count: int) -> None:
    total = Amount(usd=usd)
    holders = [str(index) for index in range(count)]
    equal = equal_shares(total, holders)
    weighted = weighted_shares(total, dict.fromkeys(holders, 1.0))
    micro_total = round(Fraction(str(total.usd)) * 1_000_000)
    assert equal == weighted
    assert equal[holders[0]] == round_pool(total, count)
    assert all(
        round(Fraction(str(share.usd)) * 1_000_000) <= micro_total // count
        for share in equal.values()
    )


@pytest.mark.parametrize(
    "weight", [0.0, -1.0, float("nan"), float("inf"), float("-inf"), "private-weight-value"]
)
def test_invalid_weight_names_holder_without_value(weight: Any) -> None:
    with pytest.raises(ValueError, match="goal/team/agent") as caught:
        weighted_shares(Amount(usd=1), {"goal/team/agent": weight})
    assert_secret_absent(caught.value, "private-weight-value", sanitized=True)
    if isinstance(weight, float):
        assert str(weight) not in str(caught.value)


@pytest.mark.parametrize("rounds", [0, -1, 1.5, True, "private-round-value"])
def test_invalid_round_count_is_sanitized(rounds: Any) -> None:
    with pytest.raises(ValueError, match="rounds_left") as caught:
        round_pool(Amount(usd=1), rounds)
    assert_secret_absent(caught.value, "private-round-value", sanitized=True)


def test_duplicate_holders_are_rejected() -> None:
    with pytest.raises(ValueError, match="holder"):
        equal_shares(Amount(usd=1), ["a", "a"])
