"""Represent signed budget amounts with whole micro-dollar arithmetic."""

from fractions import Fraction

from pydantic import field_validator

from swarm_arena._model.base import FrozenModel
from swarm_arena._model.tooling import Estimate

MICRO_USD = 1e-6
USD_LIMIT = 1_000_000_000
_MICROS_PER_USD = 1_000_000


def _micro_units(usd: float) -> int:
    """Round decimal dollars to micro-dollars, with ties going to the even integer."""
    return round(Fraction(str(usd)) * _MICROS_PER_USD)


def _usd_from_micro_units(units: int) -> float:
    """Refuse unsupported integer balances before converting to float dollars."""
    if abs(units) > USD_LIMIT * _MICROS_PER_USD:
        raise ValueError("amount exceeds the supported USD range")
    return units / _MICROS_PER_USD


class Amount(FrozenModel):
    """
    Store signed token and dollar balances for budgeting and replay.

    Parameters
    ----------
    tokens : int, default=0
        Signed token count.
    usd : float, default=0.0
        Finite dollar amount between -1,000,000,000 and 1,000,000,000 inclusive,
        rounded to the nearest micro-dollar with ties to even.

    Raises
    ------
    ValueError
        If the dollar amount is outside the supported range, before rounding.

    Notes
    -----
    Arithmetic uses integer micro-dollars and refuses results outside the
    supported range. Within this range, float serialization preserves every
    whole micro-dollar exactly on round trip. Negative balances are retained
    so replay and settlement can represent overspending. Estimate conversion
    enforces the stricter nonnegative contract for capped units.

    Examples
    --------
    >>> Amount(usd=0.1) + Amount(usd=0.2) == Amount(usd=0.3)
    True
    >>> Amount(tokens=1) - Amount(tokens=2)
    Amount(tokens=-1, usd=0.0)
    """

    tokens: int = 0
    usd: float = 0.0

    @field_validator("usd")
    @classmethod
    def _normalize_usd(cls, value: float) -> float:
        """Normalize only after the safe base has validated a finite float."""
        if abs(value) > USD_LIMIT:
            raise ValueError("amount exceeds the supported USD range")
        return _usd_from_micro_units(_micro_units(value))

    @property
    def _micro_usd(self) -> int:
        """Return the normalized dollar amount as integer micro-units."""
        return _micro_units(self.usd)

    def __add__(self, other: "Amount") -> "Amount":
        """
        Add both units using integer arithmetic.

        Parameters
        ----------
        other : Amount
            Amount to add.

        Returns
        -------
        Amount
            Sum of the balances.

        Raises
        ------
        ValueError
            If the sum is outside the supported dollar range.

        Examples
        --------
        >>> Amount(tokens=1, usd=0.01) + Amount(tokens=2, usd=0.02)
        Amount(tokens=3, usd=0.03)
        """
        return Amount(
            tokens=self.tokens + other.tokens,
            usd=_usd_from_micro_units(self._micro_usd + other._micro_usd),
        )

    def __sub__(self, other: "Amount") -> "Amount":
        """
        Subtract both units without clamping negative balances.

        Parameters
        ----------
        other : Amount
            Amount to subtract.

        Returns
        -------
        Amount
            Signed difference of the balances.

        Raises
        ------
        ValueError
            If the difference is outside the supported dollar range.

        Examples
        --------
        >>> Amount(tokens=1, usd=0.01) - Amount(tokens=2, usd=0.02)
        Amount(tokens=-1, usd=-0.01)
        """
        return Amount(
            tokens=self.tokens - other.tokens,
            usd=_usd_from_micro_units(self._micro_usd - other._micro_usd),
        )

    def fits_in(self, other: "Amount", *, cap_tokens: bool, cap_usd: bool) -> bool:
        """
        Check whether every capped unit fits within another amount.

        Parameters
        ----------
        other : Amount
            Available balance.
        cap_tokens : bool
            Whether to compare token balances.
        cap_usd : bool
            Whether to compare dollar balances.

        Returns
        -------
        bool
            Whether all enabled comparisons succeed.

        Examples
        --------
        >>> Amount(tokens=100, usd=1).fits_in(
        ...     Amount(usd=1), cap_tokens=False, cap_usd=True
        ... )
        True
        """
        return (not cap_tokens or self.tokens <= other.tokens) and (
            not cap_usd or self._micro_usd <= other._micro_usd
        )

    @classmethod
    def from_estimate(cls, e: Estimate) -> "Amount":
        """
        Convert a validated cost estimate, treating missing units as zero.

        Parameters
        ----------
        e : Estimate
            Estimated cost. Explicitly malformed model instances are revalidated.

        Returns
        -------
        Amount
            Numeric token and normalized dollar amounts.

        Raises
        ------
        ValueError
            If an estimate field violates its nonnegative, finite contract
            or its dollar amount exceeds the supported range.

        Examples
        --------
        >>> Amount.from_estimate(Estimate(tokens=10))
        Amount(tokens=10, usd=0.0)
        """
        validated = Estimate.model_validate(
            {"seconds": e.seconds, "tokens": e.tokens, "usd": e.usd}
        )
        return cls(
            tokens=0 if validated.tokens is None else validated.tokens,
            usd=0.0 if validated.usd is None else validated.usd,
        )

    def to_estimate(self, *, cap_tokens: bool = True, cap_usd: bool = True) -> Estimate:
        """
        Convert to an estimate, using None for uncapped units.

        Parameters
        ----------
        cap_tokens : bool, default=True
            Whether the estimate contains a numeric token amount.
        cap_usd : bool, default=True
            Whether the estimate contains a numeric dollar amount.

        Returns
        -------
        Estimate
            Nonnegative capped amounts and None for uncapped units.

        Raises
        ------
        ValueError
            If a capped unit is negative.

        Examples
        --------
        >>> Amount(tokens=10, usd=0.5).to_estimate(cap_tokens=False)
        Estimate(seconds=None, tokens=None, usd=0.5)
        """
        return Estimate(
            tokens=self.tokens if cap_tokens else None,
            usd=self.usd if cap_usd else None,
        )
