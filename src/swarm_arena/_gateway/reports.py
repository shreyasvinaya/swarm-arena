"""Share gateway lifecycle vocabulary and reconciliation payloads with replay."""

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from swarm_arena._budget.amount import _micro_units
from swarm_arena._errors import _core_error
from swarm_arena._model.base import FrozenModel

GatewayId = Annotated[str, Field(strict=True, min_length=1, max_length=256)]
GatewayUsd = Annotated[float, Field(strict=True, ge=0)]
ReportLabel = Literal["observed", "observed (completeness unverified)", "estimated"]
GatewayKeyAction = Literal["created", "closed", "deleted", "read_failed", "close_failed"]
GatewayReadStatus = Literal["ok", "no_rows", "failed", "incomplete"]


def _report_usd(units: int) -> float:
    """Convert an unbounded report total only if every micro-dollar survives."""
    try:
        usd = units / 1_000_000
    except OverflowError:
        raise _core_error(ValueError, "gateway report total is not representable") from None
    if not math.isfinite(usd) or _micro_units(usd) != units:
        raise _core_error(ValueError, "gateway report total is not representable")
    return usd


class GatewayKeyFailure(FrozenModel):
    """Identify a value-free failed key operation for later reconciliation.

    Parameters
    ----------
    action : {'read_failed', 'close_failed'}
        Lifecycle operation that remains unconfirmed.
    key_id, candidate_id : str
        Nonsecret key and invocation identities.
    error : str
        Fixed diagnostic without external values.

    Examples
    --------
    >>> GatewayKeyFailure(action="read_failed", key_id="k", candidate_id="c",
    ...     error="gateway read failed").action
    'read_failed'
    """

    action: Literal["read_failed", "close_failed"]
    key_id: GatewayId
    candidate_id: GatewayId
    error: Literal["gateway read failed", "gateway read incomplete", "gateway closure unconfirmed"]


class KeyReport(FrozenModel):
    """Validate one key's report without changing ledger settlement.

    Parameters
    ----------
    candidate_id, key_id, alias : str
        Nonsecret identifiers, bounded to 256 characters.
    reserved_usd : float
        Finite nonnegative reservation, used when spend remains estimated.
    observed_usd : float or None
        Finite nonnegative observed spend; None exactly for estimated reports.
    label : str
        Observed, observed (completeness unverified), or estimated.
    reasons : list of str
        Value-free reasons for the reporting decision.
    deleted : bool
        Whether key deletion was confirmed.

    Examples
    --------
    >>> row = KeyReport(candidate_id="c", key_id="k", alias="cand-c", reserved_usd=1,
    ...     observed_usd=None, label="estimated", reasons=[], deleted=False)
    >>> row.observed_usd is None
    True
    """

    candidate_id: GatewayId
    key_id: GatewayId
    alias: GatewayId
    reserved_usd: GatewayUsd
    observed_usd: GatewayUsd | None
    label: ReportLabel
    reasons: list[Annotated[str, Field(strict=True)]]
    deleted: Annotated[bool, Field(strict=True)]

    @model_validator(mode="after")
    def _consistent_observation(self) -> Self:
        """Keep the reported amount and its confidence label consistent."""
        if (self.observed_usd is None) != (self.label == "estimated"):
            raise ValueError("gateway spend conflicts with report label")
        return self


class ReconcileReport(FrozenModel):
    """Validate the full payload emitted by a reconciliation producer.

    Parameters
    ----------
    run_id : str
        Producing run; replay also checks the event envelope identity.
    keys : list of KeyReport
        Per-key observations, with the same validation used during replay.
    total_usd : float or None
        Exact sum of observations/full estimates. If that sum is not representable,
        use the exact full-reservation sum, or None if that is also unrepresentable.
    total_label : str
        Least certain per-key label; estimated when the observed sum is unsupported.
    total_reasons : list of str
        Fixed reason for an unsupported aggregate. Totals are derived from keys;
        explicitly supplied inconsistent totals are refused.

    Examples
    --------
    >>> ReconcileReport(run_id="r", keys=[]).total_usd
    0.0
    """

    run_id: GatewayId
    keys: list[KeyReport]
    total_usd: GatewayUsd | None = None
    total_label: ReportLabel = "estimated"
    total_reasons: list[Literal["unsupported gateway observation"]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _run_envelope(cls, value: Any) -> Any:
        """Accept the engine's clock stamp without making it part of the report."""
        if isinstance(value, Mapping):
            return {key: item for key, item in value.items() if key != "run_elapsed_s"}
        return value

    @model_validator(mode="after")
    def _total(self) -> Self:
        """Preserve per-key facts even when their aggregate cannot be displayed."""
        label: ReportLabel = (
            "estimated"
            if any(key.label == "estimated" for key in self.keys)
            else "observed (completeness unverified)"
            if any(key.label == "observed (completeness unverified)" for key in self.keys)
            else "observed"
        )
        reasons: list[str] = []
        try:
            total: float | None = _report_usd(
                sum(
                    _micro_units(key.reserved_usd if key.observed_usd is None else key.observed_usd)
                    for key in self.keys
                )
            )
        except ValueError:
            label = "estimated"
            reasons = ["unsupported gateway observation"]
            try:
                total = _report_usd(sum(_micro_units(key.reserved_usd) for key in self.keys))
            except ValueError:
                total = None
        for name, value in (
            ("total_usd", total),
            ("total_label", label),
            ("total_reasons", reasons),
        ):
            if name in self.model_fields_set and getattr(self, name) != value:
                raise ValueError("gateway total conflicts with key reports")
            object.__setattr__(self, name, value)
        return self

    def spend_usd(self) -> float | None:
        """Sum observed spend where available, otherwise full reservations.

        Returns
        -------
        float or None
            Exact display total, using full reservations if observed aggregation
            is unsupported. None means even the full estimate is unrepresentable.
            Inspect total_label and total_reasons; this never changes settlement.

        Examples
        --------
        >>> ReconcileReport(run_id="r", keys=[]).spend_usd()
        0.0
        """
        return self.total_usd
