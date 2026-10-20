"""Divide budget pools with integer floors and retain unallocated remainders."""

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction

from swarm_arena._budget.amount import _MICROS_PER_USD, Amount


def _floored_amount(tokens: int, micro_usd: int) -> Amount:
    """Keep float serialization from rounding a large floored share upward."""
    amount = Amount(tokens=tokens, usd=micro_usd / _MICROS_PER_USD)
    if amount._micro_usd > micro_usd:
        amount = Amount(tokens=tokens, usd=math.nextafter(amount.usd, -math.inf))
    return amount


def weighted_shares(total: Amount, weights: Mapping[str, float]) -> dict[str, Amount]:
    """
    Split a pool by positive finite weights, flooring each unit independently.

    Parameters
    ----------
    total : Amount
        Pool to divide.
    weights : Mapping[str, float]
        Holder identifiers and their positive finite relative weights.

    Returns
    -------
    dict[str, Amount]
        Shares in mapping order; an empty mapping produces no shares.
        Unallocated tokens and micro-dollars remain in the parent pool.

    Raises
    ------
    ValueError
        If a holder's weight is nonnumeric, nonpositive, or non-finite.
        The diagnostic identifies only the holder.

    Notes
    -----
    Decimal representations of weights are converted to exact fractions before
    summation and division, avoiding floating-point overflow and floor drift.
    If a very large share cannot be represented at micro-dollar precision as a
    float, its dollar representation is rounded downward.

    Examples
    --------
    >>> weighted_shares(Amount(usd=8), {"g1": 3.0, "g2": 1.0})
    {'g1': Amount(tokens=0, usd=6.0), 'g2': Amount(tokens=0, usd=2.0)}
    """
    rational_weights: dict[str, Fraction] = {}
    for holder, weight in weights.items():
        if (
            not isinstance(weight, int | float)
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(f"invalid weight for holder {holder}")
        rational_weights[holder] = Fraction(str(weight))
    weight_sum = sum(rational_weights.values(), Fraction())
    return {
        holder: _floored_amount(
            tokens=total.tokens * weight // weight_sum,
            micro_usd=total._micro_usd * weight // weight_sum,
        )
        for holder, weight in rational_weights.items()
    }


def equal_shares(total: Amount, holders: Sequence[str]) -> dict[str, Amount]:
    """
    Divide a pool equally, flooring tokens and whole micro-dollars.

    Parameters
    ----------
    total : Amount
        Pool to divide.
    holders : Sequence[str]
        Distinct holder identifiers.

    Returns
    -------
    dict[str, Amount]
        Equal shares in holder order, or an empty dictionary for no holders.
        The remainder stays in the parent pool.

    Raises
    ------
    ValueError
        If holder identifiers are duplicated.

    Examples
    --------
    >>> equal_shares(Amount(tokens=5, usd=0.000005), ["a", "b"])
    {'a': Amount(tokens=2, usd=2e-06), 'b': Amount(tokens=2, usd=2e-06)}
    """
    if len(set(holders)) != len(holders):
        raise ValueError("duplicate holder identifiers")
    if not holders:
        return {}
    share = round_pool(total, len(holders))
    return dict.fromkeys(holders, share)


def round_pool(goal_left: Amount, rounds_left: int) -> Amount:
    """
    Allocate one round's floored share of the remaining goal pool.

    Parameters
    ----------
    goal_left : Amount
        Remaining goal budget.
    rounds_left : int
        Positive number of remaining rounds, including this round.

    Returns
    -------
    Amount
        Tokens and whole micro-dollars divided with floor rounding.

    Raises
    ------
    ValueError
        If rounds_left is not a positive integer.

    Examples
    --------
    >>> round_pool(Amount(tokens=90, usd=0.9), 3)
    Amount(tokens=30, usd=0.3)
    """
    if isinstance(rounds_left, bool) or not isinstance(rounds_left, int) or rounds_left <= 0:
        raise ValueError("rounds_left must be a positive integer")
    return _floored_amount(
        tokens=goal_left.tokens // rounds_left,
        micro_usd=goal_left._micro_usd // rounds_left,
    )
