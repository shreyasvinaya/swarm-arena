"""Define private gateway records and the asynchronous adapter boundary."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Any, Protocol

from pydantic import AwareDatetime, Field, field_validator

from swarm_arena._model.base import FrozenModel
from swarm_arena._model.data import FiniteJsonObject
from swarm_arena._model.tooling import Usage


class GatewayError(RuntimeError):
    """
    Report a gateway failure using only method, route and status or error type.

    Transport implementations construct the safe message and suppress raw causes;
    query strings, bearer values and response bodies must never be included.

    Examples
    --------
    >>> str(GatewayError("GET /model/info: 503"))
    'GET /model/info: 503'
    """


class ProfileReport(FrozenModel):
    """
    Record strict-profile evidence without inferring unavailable guarantees.

    Parameters
    ----------
    verified : bool
        Whether every profile requirement was verified.
    failures : list of str
        Ordered, value-free rule diagnostics (model identifiers are allowed).
    batch_write_interval_s : float or None
        Finite nonnegative configured interval, not a maximum spend-log lag.
    request_timeout_s : float or None
        Finite nonnegative enforced total duration, or None when unverified.

    Examples
    --------
    >>> report = ProfileReport(verified=False, failures=["unverified"],
    ...     batch_write_interval_s=10, request_timeout_s=None)
    >>> report.verified
    False
    """

    verified: bool
    failures: list[str]
    batch_write_interval_s: Annotated[float, Field(ge=0, strict=True)] | None
    request_timeout_s: Annotated[float, Field(ge=0, strict=True)] | None


class VirtualKey(FrozenModel):
    """
    Retain durable key identity without storing its secret bearer value.

    Parameters
    ----------
    key_id : str
        Gateway token hash, safe to persist.
    alias : str
        Candidate alias supplied when creating the key.

    Examples
    --------
    >>> VirtualKey(key_id="hash", alias="cand-1").model_dump()
    {'key_id': 'hash', 'alias': 'cand-1'}
    """

    key_id: str
    alias: str


class SpendRow(FrozenModel):
    """
    Preserve one validated spend row and its independent upstream call identity.

    Parameters
    ----------
    request_id : str
        Required nonempty primary spend-row ID. Together with key identity, this
        identifies the observation; the upstream call ID is not a dedup key.
    litellm_call_id : str or None
        Nullable, potentially client-selected upstream call ID.
    model : str
        Model identifier reported by the gateway.
    usage : Usage
        Validated nonnegative token counters and optional cost. The HTTP adapter
        reads cache counts from ``metadata.additional_usage_values``; missing
        counts follow Usage's zero/unknown convention.
    cost_usd : float
        Finite nonnegative row cost.
    ts : datetime
        Aware start timestamp, normalized to UTC.
    metadata : dict of str to Any
        Finite JSON metadata, validated through the shared JSON boundary.

    Examples
    --------
    >>> row = SpendRow(request_id="row-1", litellm_call_id=None, model="chat",
    ...     usage=Usage(input_tokens=3), cost_usd=0.001,
    ...     ts=datetime(2026, 10, 23, tzinfo=UTC), metadata={})
    >>> row.request_id, row.ts.tzinfo is UTC
    ('row-1', True)
    """

    request_id: Annotated[str, Field(min_length=1, strict=True)]
    litellm_call_id: str | None
    model: str
    usage: Usage
    cost_usd: Annotated[float, Field(ge=0, strict=True)]
    ts: AwareDatetime
    metadata: FiniteJsonObject

    @field_validator("request_id")
    @classmethod
    def _primary_identity(cls, value: str) -> str:
        """Reject blank identity without changing a meaningful upstream ID."""
        if not value.strip():
            raise ValueError("request_id must be nonempty")
        return value

    @field_validator("usage", mode="before")
    @classmethod
    def _validated_usage(cls, value: Any) -> Any:
        """Revalidate copied or constructed Usage instances at the row boundary."""
        if isinstance(value, Usage):
            return value.model_dump(mode="python")
        return value

    @field_validator("ts")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        """Normalize an already validated aware timestamp."""
        return value.astimezone(UTC)


class GatewayAdapter(Protocol):
    """
    Describe gateway operations without exposing a public adapter API.

    Attributes
    ----------
    terminal_accounting : bool
        Whether the adapter can prove final spend completeness. LiteLLM is False;
        neither key closure nor a batch interval alone establishes completeness.

    Examples
    --------
    >>> import inspect
    >>> inspect.iscoroutinefunction(GatewayAdapter.preflight)
    True
    """

    terminal_accounting: bool

    async def preflight(self) -> ProfileReport:
        """
        Inspect gateway health, models and available strict-profile evidence.

        Returns
        -------
        ProfileReport
            Verified requirements and ordered failures.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(GatewayAdapter.preflight).return_annotation is ProfileReport
        True
        """
        ...

    async def create_key(
        self,
        *,
        alias: str,
        max_budget_usd: float,
        duration_s: float,
        metadata: Mapping[str, Any],
        default_estimated_output_tokens: int,
    ) -> tuple[VirtualKey, str]:
        """
        Create a key and return its durable identity separately from its secret.

        Parameters
        ----------
        alias : str
            Candidate alias.
        max_budget_usd : float
            Full reservation for this tool run.
        duration_s : float
            Key expiration interval.
        metadata : Mapping of str to Any
            Run and candidate attribution.
        default_estimated_output_tokens : int
            Reservation estimate, never an enforced output cap.

        Returns
        -------
        tuple of VirtualKey and str
            Durable identity and ephemeral bearer value, respectively.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(GatewayAdapter.create_key).parameters['alias'].kind.name
        'KEYWORD_ONLY'
        """
        ...

    async def block_key(self, key: VirtualKey) -> None:
        """
        Prevent the key from authorizing new requests.

        Parameters
        ----------
        key : VirtualKey
            Durable key identity.

        Examples
        --------
        >>> import inspect
        >>> list(inspect.signature(GatewayAdapter.block_key).parameters)
        ['self', 'key']
        """
        ...

    async def closed_at(self, key: VirtualKey) -> datetime | None:
        """
        Return API-confirmed closure time without implying final spend.

        Parameters
        ----------
        key : VirtualKey
            Durable key identity.

        Returns
        -------
        datetime or None
            Aware UTC closure time, or None when closure is unconfirmed.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(GatewayAdapter.closed_at)
        True
        """
        ...

    async def spend_rows(
        self, key: VirtualKey, *, start: datetime, end: datetime
    ) -> list[SpendRow]:
        """
        Read spend for a fixed inclusive UTC window under one operation deadline.

        Parameters
        ----------
        key : VirtualKey
            Durable key identity; deletion does not preclude historical reads.
        start : datetime
            Required aware UTC lower bound, including the caller's overlap.
        end : datetime
            Required aware UTC upper bound, fixed before pagination begins.

        Returns
        -------
        list of SpendRow
            Validated rows from every page. Incomplete or capped responses fail
            with GatewayError; a successful read does not prove terminal spend.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(GatewayAdapter.spend_rows).parameters['start'].kind.name
        'KEYWORD_ONLY'
        """
        ...

    async def delete_key(self, key: VirtualKey) -> None:
        """
        Delete the key without discarding its historical spend identity.

        Parameters
        ----------
        key : VirtualKey
            Durable key identity.

        Examples
        --------
        >>> import inspect
        >>> list(inspect.signature(GatewayAdapter.delete_key).parameters)
        ['self', 'key']
        """
        ...
