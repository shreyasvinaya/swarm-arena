"""Share gateway lifecycle vocabulary and reconciliation payloads with replay."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from swarm_arena._model.base import FrozenModel

GatewayId = Annotated[str, Field(strict=True, min_length=1, max_length=256)]
GatewayUsd = Annotated[float, Field(strict=True, ge=0)]
ReportLabel = Literal["observed", "observed (completeness unverified)", "estimated"]
GatewayKeyAction = Literal["created", "closed", "read_failed", "close_failed"]
GatewayReadStatus = Literal["ok", "failed", "incomplete"]


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

    Examples
    --------
    >>> ReconcileReport(run_id="r", keys=[]).model_dump(mode="json")
    {'run_id': 'r', 'keys': []}
    """

    run_id: GatewayId
    keys: list[KeyReport]

    def spend_usd(self) -> float:
        """Sum observed spend where available, otherwise full reservations.

        Returns
        -------
        float
            Display total only; never a replacement for ledger settlement.

        Examples
        --------
        >>> ReconcileReport(run_id="r", keys=[]).spend_usd()
        0.0
        """
        return sum(
            (
                key.reserved_usd if key.observed_usd is None else key.observed_usd
                for key in self.keys
            ),
            0.0,
        )
