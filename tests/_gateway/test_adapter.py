"""Validate safe gateway records and freeze the downstream adapter signature."""

import inspect
import json
import traceback
import warnings
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
        "key_id": "hash",
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


def _validate_entry(
    model: type[SpendRow] | type[ProfileReport],
    data: dict[str, Any],
    entry: str,
) -> SpendRow | ProfileReport:
    if entry == "json":
        return model.model_validate_json(json.dumps(data))
    return model.model_validate(data)


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
        verified=True, failures=(), batch_write_interval_s=0, request_timeout_s=0
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


@pytest.mark.parametrize("construction", ["copy", "construct"])
def test_usage_revalidation_has_no_serializer_warnings(construction: str) -> None:
    secret = "rejected-usage-sentinel"
    data = {"input_tokens": secret}
    usage = (
        Usage().model_copy(update=data)
        if construction == "copy"
        else Usage.model_construct(input_tokens=secret)
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError) as caught:
            SpendRow(**_row(usage=usage))
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
    assert all(secret not in text for text in surfaces)
    assert all(secret not in str(warning.message) for warning in captured)
    assert not captured


@pytest.mark.parametrize(
    ("verified", "failures", "duration"),
    [
        (True, ["unverified"], 10.0),
        (True, [], None),
        (False, [], None),
        (False, [], 10.0),
    ],
)
@pytest.mark.parametrize("entry", ["python", "json"])
def test_profile_rejects_inconsistent_evidence(
    verified: bool,
    failures: list[str],
    duration: float | None,
    entry: str,
) -> None:
    data = {
        "verified": verified,
        "failures": failures,
        "request_timeout_s": duration,
        "batch_write_interval_s": None,
    }
    with pytest.raises(ValidationError):
        _validate_entry(ProfileReport, data, entry)


def test_profile_invariant_error_is_value_free() -> None:
    secret = "profile-evidence-sentinel"
    data = {
        "verified": True,
        "failures": [secret],
        "request_timeout_s": 300,
        "batch_write_interval_s": None,
    }
    with pytest.raises(ValidationError) as caught:
        ProfileReport.model_validate(data)
    error = caught.value
    assert secret not in str(error.errors())
    assert secret not in error.json()
    assert secret not in "".join(traceback.format_exception(error))


def test_profile_failures_are_immutable_and_detached() -> None:
    failures = ["unverified"]
    report = ProfileReport.model_validate(
        {
            "verified": False,
            "failures": failures,
            "batch_write_interval_s": None,
            "request_timeout_s": None,
        }
    )
    assert isinstance(report.failures, tuple)
    failures.clear()
    assert report.failures == ("unverified",)
    assert not hasattr(report.failures, "clear")
    assert ProfileReport.model_validate_json(report.model_dump_json()) == report
    assert json.loads(report.model_dump_json())["failures"] == ["unverified"]


@pytest.mark.parametrize(
    "field",
    [
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_write_5m_tokens",
        "cache_write_1h_tokens",
    ],
)
@pytest.mark.parametrize("value", [True, "5", 5.0])
@pytest.mark.parametrize("entry", ["python", "json"])
def test_spend_usage_rejects_coerced_token_counts(field: str, value: Any, entry: str) -> None:
    data = _row(usage={field: value}, ts="2026-10-23T00:00:00Z")
    with pytest.raises(ValidationError):
        _validate_entry(SpendRow, data, entry)


@pytest.mark.parametrize(("value", "cost"), [("0.001", 0.001), (True, 1.0)])
def test_spend_usage_rejects_coerced_usd(value: Any, cost: float) -> None:
    with pytest.raises(ValidationError):
        SpendRow(**_row(usage={"usd": value}, cost_usd=cost))


def test_spend_row_retains_its_own_key_identity() -> None:
    a = SpendRow(**_row(key_id="hash-a"))
    b = SpendRow(**_row(key_id="hash-b"))
    assert a.key_id != b.key_id
    assert a.request_id == b.request_id
    assert SpendRow.model_validate_json(a.model_dump_json()).key_id == "hash-a"


def test_spend_row_requires_key_identity() -> None:
    data = _row()
    data.pop("key_id", None)
    with pytest.raises(ValidationError):
        SpendRow.model_validate(data)


@pytest.mark.parametrize("field", ["key_id", "alias"])
@pytest.mark.parametrize("value", ["", " \t\n", 1, True, None, "x" * 257])
def test_virtual_key_rejects_invalid_identity(field: str, value: Any) -> None:
    data = {"key_id": "hash", "alias": "cand-1", field: value}
    with pytest.raises(ValidationError):
        VirtualKey.model_validate(data)


@pytest.mark.parametrize("value", ["", " \t\n", 1, True, None, "x" * 257])
def test_spend_row_rejects_invalid_key_identity(value: Any) -> None:
    with pytest.raises(ValidationError):
        SpendRow(**_row(key_id=value))


@pytest.mark.parametrize("value", [0, 1700000000, 1700000000000, 1700000000.5, "1700000000"])
@pytest.mark.parametrize("entry", ["python", "json"])
def test_spend_timestamp_rejects_epoch_values(value: Any, entry: str) -> None:
    data = _row(ts=value, usage={})
    with pytest.raises(ValidationError):
        _validate_entry(SpendRow, data, entry)


@pytest.mark.parametrize("value", ["2026-10-23T05:30:00+05:30", "2026-10-23T00:00:00Z"])
def test_spend_timestamp_accepts_aware_iso_strings(value: str) -> None:
    row = SpendRow(**_row(ts=value))
    assert row.ts == datetime(2026, 10, 23, tzinfo=UTC)
    assert row.ts.tzinfo is UTC


@pytest.mark.parametrize("usage", [{"usd": 5.0}, Usage(usd=5.0)])
def test_spend_row_rejects_conflicting_cost_sources(usage: Any) -> None:
    with pytest.raises(ValidationError):
        SpendRow(**_row(usage=usage))


@pytest.mark.parametrize("usd", [None, 0.001])
def test_spend_row_authoritative_cost_accepts_unknown_or_matching_usage(usd: float | None) -> None:
    row = SpendRow(**_row(usage={"input_tokens": 5, "usd": usd}))
    assert row.cost_usd == 0.001
    assert row.usage.usd == usd
    assert row.usage.input_tokens == 5


def test_protocol_defines_idempotent_close_and_delete_outcomes() -> None:
    for method in (GatewayAdapter.block_key, GatewayAdapter.delete_key):
        assert inspect.signature(method).return_annotation is None
        doc = inspect.getdoc(method) or ""
        assert "idempotent" in doc
        assert "already blocked, deleted or missing" in doc
        assert "GatewayError" in doc
    closure = inspect.getdoc(GatewayAdapter.closed_at) or ""
    assert 'status="deleted"' in closure
    assert "404" in closure
    assert "confirmation time" in closure
    spend = inspect.getdoc(GatewayAdapter.spend_rows) or ""
    assert "row.key_id == key.key_id" in spend
