"""Validate safe gateway records and freeze the downstream adapter signature."""

import inspect
import traceback
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from swarm_arena._gateway.adapter import (
    GatewayAdapter,
    GatewayError,
    ProfileReport,
    SpendRow,
    VirtualKey,
)
from swarm_arena._model.tooling import Usage


def _row(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "request_id": "row-1",
        "litellm_call_id": None,
        "model": "chat-model",
        "usage": Usage(input_tokens=3, output_tokens=2, cache_read_tokens=1),
        "cost_usd": 0.001,
        "ts": datetime(2026, 10, 23, tzinfo=UTC),
        "metadata": {"additional_usage_values": {"cache_read_input_tokens": 1}},
    }
    data.update(changes)
    return data


def test_virtual_key_holds_no_secret() -> None:
    assert set(VirtualKey.model_fields) == {"key_id", "alias"}
    key = VirtualKey(key_id="hash", alias="cand-1")
    assert key.model_dump() == {"key_id": "hash", "alias": "cand-1"}
    with pytest.raises(ValidationError):
        key.alias = "changed"  # ty: ignore[invalid-assignment]


def test_spend_row_preserves_primary_and_nullable_upstream_identity() -> None:
    a = SpendRow(**_row())
    b = SpendRow(**_row(request_id="row-2", litellm_call_id="shared"))
    c = SpendRow(**_row(request_id="row-3", litellm_call_id="shared"))
    assert a.litellm_call_id is None
    assert b.request_id != c.request_id
    assert b.litellm_call_id == c.litellm_call_id
    assert SpendRow.model_validate_json(a.model_dump_json()) == a
    assert a.usage.cache_read_tokens == 1
    assert a.metadata["additional_usage_values"]["cache_read_input_tokens"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": ""},
        {"request_id": "   "},
        {"request_id": 1},
        {"request_id": None},
        {"cost_usd": -1},
        {"cost_usd": float("nan")},
        {"cost_usd": float("inf")},
        {"cost_usd": True},
        {"usage": {"input_tokens": -1}},
        {"usage": {"usd": float("nan")}},
        {"usage": Usage().model_copy(update={"input_tokens": -1})},
        {"ts": datetime(2026, 10, 23)},
        {"ts": "2026-10-23"},
        {"metadata": {"x": float("inf")}},
        {"metadata": {"x": object()}},
        {"metadata": {"x": (1, 2)}},
    ],
)
def test_spend_row_rejects_invalid_domains(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SpendRow(**_row(**change))


def test_primary_identity_is_required() -> None:
    data = _row()
    del data["request_id"]
    with pytest.raises(ValidationError):
        SpendRow.model_validate(data)


def test_spend_row_normalizes_aware_timestamp_to_utc() -> None:
    value = datetime(2026, 10, 23, 5, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    row = SpendRow(**_row(ts=value))
    assert row.ts == datetime(2026, 10, 23, tzinfo=UTC)
    assert row.ts.tzinfo is UTC


@pytest.mark.parametrize("field", ["batch_write_interval_s", "request_timeout_s"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_profile_numeric_domains(field: str, value: Any) -> None:
    data = {
        "verified": False,
        "failures": ["unverified"],
        "batch_write_interval_s": None,
        "request_timeout_s": None,
        field: value,
    }
    with pytest.raises(ValidationError):
        ProfileReport.model_validate(data)


def test_profile_optional_and_zero_timings() -> None:
    report = ProfileReport(
        verified=True, failures=[], batch_write_interval_s=0, request_timeout_s=None
    )
    assert report.verified
    assert report.batch_write_interval_s == 0


@pytest.mark.parametrize("entry", ["constructor", "python", "json"])
def test_validation_errors_never_expose_values(entry: str) -> None:
    secret = "sentinel-secret"
    data = _row(cost_usd=secret, metadata={secret: object()})
    invalid_json = '{"request_id": "' + secret + '", broken'

    def validate() -> SpendRow:
        if entry == "constructor":
            return SpendRow(**data)
        if entry == "python":
            return SpendRow.model_validate(data)
        return SpendRow.model_validate_json(invalid_json)

    with pytest.raises(ValidationError) as caught:
        validate()
    error = caught.value
    surfaces = [
        str(error),
        repr(error),
        str(error.args),
        str(error.errors()),
        error.json(),
        str(getattr(error, "__notes__", None)),
        str(error.__cause__),
        "".join(traceback.format_exception(error)),
    ]
    assert all(secret not in surface for surface in surfaces)


def test_protocol_signatures_and_gateway_error_base() -> None:
    assert issubclass(GatewayError, RuntimeError)
    assert str(GatewayError("GET /model/info: ValueError")) == "GET /model/info: ValueError"
    expected = {
        "preflight": ["self"],
        "create_key": [
            "self",
            "alias",
            "max_budget_usd",
            "duration_s",
            "metadata",
            "default_estimated_output_tokens",
        ],
        "block_key": ["self", "key"],
        "closed_at": ["self", "key"],
        "spend_rows": ["self", "key", "start", "end"],
        "delete_key": ["self", "key"],
    }
    for name, names in expected.items():
        method = getattr(GatewayAdapter, name)
        assert inspect.iscoroutinefunction(method)
        assert list(inspect.signature(method).parameters) == names
    for name in ("start", "end"):
        param = inspect.signature(GatewayAdapter.spend_rows).parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty
    for name, param in inspect.signature(GatewayAdapter.create_key).parameters.items():
        if name != "self":
            assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert GatewayAdapter.__annotations__["terminal_accounting"] is bool


def test_unrepresentable_utc_timestamp_is_a_safe_validation_error() -> None:
    value = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(ValidationError):
        SpendRow(**_row(ts=value))
