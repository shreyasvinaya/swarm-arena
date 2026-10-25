"""Define private gateway records and the asynchronous adapter boundary."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Any, Protocol, Self

from pydantic import AfterValidator, AwareDatetime, Field, field_validator, model_validator

from swarm_arena._model.base import FrozenModel
from swarm_arena._model.data import FiniteJsonObject
from swarm_arena._model.tooling import Usage


def _nonblank(value: str) -> str:
    """Reject blank identifiers without changing meaningful identity text."""
    if not value.strip():
        raise ValueError("identifier must be nonblank")
    return value


_KeyText = Annotated[
    str, Field(min_length=1, max_length=256, strict=True), AfterValidator(_nonblank)
]


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
    failures : tuple of str
        Immutable ordered, value-free diagnostics (model identifiers are allowed).
        Empty exactly when verified; a verified report also requires a duration.
    batch_write_interval_s : float or None
        Finite nonnegative configured interval, not a maximum spend-log lag.
    request_timeout_s : float or None
        Finite nonnegative enforced total duration, or None when unverified.

    Examples
    --------
    >>> report = ProfileReport(verified=False, failures=("unverified",),
    ...     batch_write_interval_s=10, request_timeout_s=None)
    >>> report.verified
    False
    """

    verified: bool
    failures: tuple[str, ...]
    batch_write_interval_s: Annotated[float, Field(ge=0, strict=True)] | None
    request_timeout_s: Annotated[float, Field(ge=0, strict=True)] | None

    @model_validator(mode="after")
    def _consistent_evidence(self) -> Self:
        """Require the decision, failures and duration evidence to agree."""
        if self.verified != (not self.failures):
            raise ValueError("profile verification conflicts with failures")
        if self.verified and self.request_timeout_s is None:
            raise ValueError("verified profile requires request_timeout_s")
        return self


class VirtualKey(FrozenModel):
    """
    Retain durable key identity without storing its secret bearer value.

    Parameters
    ----------
    key_id : str
        Gateway token hash, safe to persist; strict nonblank text, 1-256 characters.
    alias : str
        Candidate alias supplied when creating the key; strict nonblank text,
        1-256 characters. Identifiers are preserved, not trimmed.

    Examples
    --------
    >>> VirtualKey(key_id="hash", alias="cand-1").model_dump()
    {'key_id': 'hash', 'alias': 'cand-1'}
    """

    key_id: _KeyText
    alias: _KeyText


class SpendRow(FrozenModel):
    """
    Preserve one validated spend row and its independent upstream call identity.

    Parameters
    ----------
    key_id : str
        The row's own ``api_key`` hash, not the queried key substituted by the
        caller. Strict nonblank text, 1-256 characters. Adapters reject a row
        whose key identity differs from the query's VirtualKey.
    request_id : str
        Required nonempty primary spend-row ID. Together with key identity, this
        identifies the observation; the upstream call ID is not a dedup key.
    litellm_call_id : str or None
        Nullable, potentially client-selected upstream call ID.
    model : str
        Model identifier reported by the gateway.
    usage : Usage
        Strictly validated nonnegative integer token counters and optional cost.
        ``usd`` must be None or equal to authoritative ``cost_usd``. The HTTP adapter
        reads cache counts from ``metadata.additional_usage_values``; missing
        counts follow Usage's zero/unknown convention.
    cost_usd : float
        Authoritative finite nonnegative row cost; never add ``usage.usd`` to it.
    ts : datetime or str
        Aware start datetime or timezone-aware ISO string, normalized to UTC.
        Unix numbers and numeric strings are rejected.
    metadata : dict of str to Any
        Finite JSON metadata, validated through the shared JSON boundary.

    Examples
    --------
    >>> row = SpendRow(key_id="hash", request_id="row-1", litellm_call_id=None, model="chat",
    ...     usage=Usage(input_tokens=3), cost_usd=0.001,
    ...     ts=datetime(2026, 10, 23, tzinfo=UTC), metadata={})
    >>> row.request_id, row.ts.tzinfo is UTC
    ('row-1', True)
    """

    key_id: _KeyText
    request_id: Annotated[str, Field(min_length=1, strict=True), AfterValidator(_nonblank)]
    litellm_call_id: str | None
    model: str
    usage: Usage
    cost_usd: Annotated[float, Field(ge=0, strict=True)]
    ts: AwareDatetime
    metadata: FiniteJsonObject

    @field_validator("usage", mode="before")
    @classmethod
    def _validated_usage(cls, value: Any) -> Usage:
        """Revalidate copied or constructed Usage instances at the row boundary."""
        if isinstance(value, Usage):
            # Serialization of invalid copied fields can warn with their values.
            value = dict(vars(value))
        return Usage.model_validate(value, strict=True)

    @field_validator("ts", mode="before")
    @classmethod
    def _timestamp_form(cls, value: Any) -> datetime:
        """Accept datetime/ISO forms without Pydantic's epoch-number inference."""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                raise ValueError("ts must be an aware datetime or ISO timestamp") from None
        if not isinstance(value, datetime):
            raise ValueError("ts must be an aware datetime or ISO timestamp")
        return value

    @field_validator("ts")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        """Normalize an already validated aware timestamp."""
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _consistent_cost(self) -> Self:
        """Prevent duplicate cost carriers from disagreeing."""
        if self.usage.usd is not None and self.usage.usd != self.cost_usd:
            raise ValueError("usage.usd conflicts with cost_usd")
        return self


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

    async def key_matches(self, key: VirtualKey, *, run_id: str) -> bool | None:
        """Check authoritative run metadata and alias before reconciliation mutates a key.

        Parameters
        ----------
        key : VirtualKey
            Journaled hash and expected alias.
        run_id : str
            Addressed run identity, already checked against the journal.

        Returns
        -------
        bool or None
            True for a matching run and alias, False for a mismatch, or None
            when the key or authoritative metadata is unavailable.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(GatewayAdapter.key_matches)
        True
        """
        ...

    async def block_key(self, key: VirtualKey) -> None:
        """
        Prevent new requests with an idempotent key operation.

        Parameters
        ----------
        key : VirtualKey
            Durable key identity.

        Returns
        -------
        None
            Success, including a key already blocked, deleted or missing.

        Raises
        ------
        GatewayError
            Closure could not be confirmed, including transient transport/server
            failures. Adapters normalize already-closed responses to success;
            consumers never parse exception messages to decide idempotency.

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
            Aware UTC confirmation time when the API confirms blocked or deleted,
            including an archived HTTP 200 response with ``status="deleted"`` or
            a definitive key-info 404. Use the observation time when the API has
            no reliable closure timestamp; never fabricate an earlier time.
            None means the key is still open or closure is unconfirmed.

        Raises
        ------
        GatewayError
            A transport, server or malformed-response failure prevents inspection.
            Such a failure is not proof of closure or key absence.

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
            Each row retains its upstream ``api_key`` as ``key_id``. The adapter
            requires ``row.key_id == key.key_id`` and raises GatewayError on a
            mismatch; never substitute the queried key or return a partial total.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(GatewayAdapter.spend_rows).parameters['start'].kind.name
        'KEYWORD_ONLY'
        """
        ...

    async def delete_key(self, key: VirtualKey) -> None:
        """
        Delete the key with an idempotent operation, retaining spend identity.

        Parameters
        ----------
        key : VirtualKey
            Durable key identity.

        Returns
        -------
        None
            Success, including a key already blocked, deleted or missing. A
            blocked but existing key is still deleted; closed is not absent.

        Raises
        ------
        GatewayError
            Deletion could not be confirmed, including transient transport/server
            failures. Already-deleted responses are normalized to success, so
            retries after a crash need no exception-message parsing.

        Examples
        --------
        >>> import inspect
        >>> list(inspect.signature(GatewayAdapter.delete_key).parameters)
        ['self', 'key']
        """
        ...
