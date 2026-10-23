"""Pin profile failures to supported LiteLLM settings and their actual semantics."""

from copy import deepcopy
from typing import Any

import pytest
from _fakes.gateway import ABSENT_FAILURES, strict_models, strict_settings

from swarm_arena._gateway.profile import (
    BATCH_WRITE_INTERVAL,
    BUDGET_RESERVATIONS,
    FAIL_CLOSED,
    MAX_REQUEST_DURATION,
    MAX_TOKENS_BOUND,
    PASS_THROUGH,
    RESERVED_ROUTES_ONLY,
    VIRTUAL_KEYS_REQUIRED,
    check_profile,
)


def _pinned_merge(absent: list[str], *texts: str) -> tuple[str, ...]:
    order = [
        "budget reservations are disabled",
        "spend enforcement",
        "no price",
        "no max output",
        "request max_tokens",
        "routes without",
        "no maximum",
        "maximum request",
        "virtual keys",
        "pass-through",
        "gateway settings",
    ]
    return tuple(
        sorted(
            [*absent, *texts],
            key=lambda text: next(i for i, fragment in enumerate(order) if fragment in text),
        )
    )


def test_setting_names_match_pinned_facts() -> None:
    assert (
        BUDGET_RESERVATIONS,
        FAIL_CLOSED,
        BATCH_WRITE_INTERVAL,
        PASS_THROUGH,
        RESERVED_ROUTES_ONLY,
    ) == (
        "disable_budget_reservation",
        "fail_closed_budget_enforcement",
        "proxy_batch_write_at",
        "pass_through_endpoints",
        "allowed_routes",
    )
    assert MAX_REQUEST_DURATION is VIRTUAL_KEYS_REQUIRED is MAX_TOKENS_BOUND is None


def test_strict_profile_verified() -> None:
    report = check_profile(
        strict_settings(request_timeout_s=300, batch_write_interval_s=10),
        strict_models(),
        request_timeout_s=600.0,
    )
    assert report.verified is False
    assert report.failures == tuple(ABSENT_FAILURES)
    assert report.batch_write_interval_s == 10.0
    assert report.request_timeout_s is None


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ({"disable_budget_reservation": True}, "budget reservations are disabled"),
        (
            {"fail_closed_budget_enforcement": False},
            "spend enforcement does not fail closed when spend counters are stale",
        ),
        ({"allowed_routes": ["llm_api_routes"]}, "routes without budget reservations are enabled"),
        (
            {"pass_through_endpoints": [{"path": "/custom", "target": "sentinel-secret"}]},
            "pass-through endpoints are configured",
        ),
    ],
)
def test_each_profile_rule(change: dict[str, Any], failure: str) -> None:
    report = check_profile(strict_settings(**change), strict_models(), request_timeout_s=600.0)
    assert not report.verified
    assert report.failures == _pinned_merge(ABSENT_FAILURES, failure)


@pytest.mark.parametrize("failure", ABSENT_FAILURES)
def test_absent_setting_always_fails(failure: str) -> None:
    settings = strict_settings(
        request_timeout=1,
        master_key="sentinel-secret",
        default_estimated_output_tokens=1,
        max_tokens=1,
    )
    assert failure in check_profile(settings, strict_models(), request_timeout_s=600.0).failures


def test_unpriced_and_unbounded_models_fail() -> None:
    cheap, long = strict_models()[0], strict_models()[0]
    cheap["model_name"], long["model_name"] = "cheap", "long"
    cheap["model_info"].pop("input_cost_per_token")
    long["model_info"]["max_output_tokens"] = None
    assert check_profile(strict_settings(), [cheap, long], request_timeout_s=600.0).failures == (
        _pinned_merge(
            ABSENT_FAILURES,
            "model cheap: no price (input and output cost per token)",
            "model long: no max output token bound",
        )
    )


@pytest.mark.parametrize("batch", [None, -1, float("nan"), float("inf"), True, "10", {}, []])
def test_unknown_batch_interval_is_none_not_failure(batch: Any) -> None:
    report = check_profile(
        strict_settings(batch_write_interval_s=batch), strict_models(), request_timeout_s=600.0
    )
    assert report.failures == tuple(ABSENT_FAILURES)
    assert report.batch_write_interval_s is None


def test_unavailable_settings_do_not_invent_switch_states() -> None:
    models = strict_models()
    models[0]["model_info"] = {}
    report = check_profile(None, models, request_timeout_s=600.0)
    assert report.failures == _pinned_merge(
        ABSENT_FAILURES,
        "model chat-model: no price (input and output cost per token)",
        "model chat-model: no max output token bound",
        "gateway settings introspection unavailable",
    )
    assert report.verified is False
    assert report.request_timeout_s is report.batch_write_interval_s is None
    assert "gateway settings introspection unavailable" not in ABSENT_FAILURES


@pytest.mark.parametrize("value", [None, False, 0, 1, "true", {}, []])
def test_fail_closed_requires_literal_true(value: Any) -> None:
    report = check_profile(
        strict_settings(fail_closed_budget_enforcement=value),
        strict_models(),
        request_timeout_s=600.0,
    )
    assert "spend enforcement does not fail closed when spend counters are stale" in report.failures


@pytest.mark.parametrize("value", [False, None])
def test_reservation_opt_out_is_inverted(value: Any) -> None:
    report = check_profile(
        strict_settings(disable_budget_reservation=value), strict_models(), request_timeout_s=600.0
    )
    assert report.failures == tuple(ABSENT_FAILURES)


@pytest.mark.parametrize(
    "value", [None, True, "on", "llm_api_routes", {}, [], ["/responses"], ["/custom"], ["*"], [1]]
)
def test_routes_need_explicit_supported_allowlist(value: Any) -> None:
    report = check_profile(
        strict_settings(allowed_routes=value), strict_models(), request_timeout_s=600.0
    )
    assert "routes without budget reservations are enabled" in report.failures


def test_management_routes_do_not_prevent_the_restricted_profile() -> None:
    routes = [
        "/v1/chat/completions",
        "/key/generate",
        "/key/block",
        "/key/info",
        "/key/delete",
        "/spend/logs/v2",
        "/model/info",
        "/v1/model/info",
        "/health/liveliness",
    ]
    assert check_profile(
        strict_settings(allowed_routes=routes), strict_models(), request_timeout_s=600.0
    ).failures == tuple(ABSENT_FAILURES)


@pytest.mark.parametrize("value", [None, False, "off", {}])
def test_unknown_pass_through_is_not_disabled(value: Any) -> None:
    report = check_profile(
        strict_settings(pass_through_endpoints=value), strict_models(), request_timeout_s=600.0
    )
    assert "pass-through endpoints are configured" in report.failures


@pytest.mark.parametrize(
    "field", ["input_cost_per_token", "output_cost_per_token", "max_output_tokens"]
)
@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf"), True, "10", {}])
def test_model_domains_are_not_coerced(field: str, value: Any) -> None:
    models = strict_models()
    models[0]["model_info"][field] = value
    report = check_profile(strict_settings(), models, request_timeout_s=600.0)
    text = (
        "no max output token bound"
        if field == "max_output_tokens"
        else "no price (input and output cost per token)"
    )
    assert f"model chat-model: {text}" in report.failures


def test_legacy_context_limit_and_estimate_are_not_output_bounds() -> None:
    models = strict_models()
    models[0]["model_info"] = {
        "max_tokens": 4096,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    }
    report = check_profile(
        strict_settings(default_estimated_output_tokens=4096), models, request_timeout_s=600.0
    )
    assert "model chat-model: no max output token bound" in report.failures
    assert "routes without budget reservations are enabled" in report.failures
    assert ABSENT_FAILURES[0] in report.failures


def test_model_failures_keep_pinned_order_and_inputs_unchanged() -> None:
    models = [{"model_name": "first", "model_info": {}}, {"model_name": "second", "model_info": {}}]
    settings = strict_settings()
    before = deepcopy((settings, models))
    report = check_profile(settings, models, request_timeout_s=600.0)
    assert report.failures[:4] == (
        "model first: no price (input and output cost per token)",
        "model second: no price (input and output cost per token)",
        "model first: no max output token bound",
        "model second: no max output token bound",
    )
    assert (settings, models) == before


def test_unreadable_values_are_never_diagnostics() -> None:
    secret = "sentinel-secret"
    settings = {key: secret for key in strict_settings()}
    models = [{"model_name": {secret: secret}, "model_info": {"max_output_tokens": secret}}]
    report = check_profile(settings, models, request_timeout_s=600.0)
    assert secret not in str(report.failures)


@pytest.mark.parametrize("mode", ["embedding", "image_generation", None, {}])
def test_non_chat_model_modes_do_not_establish_route_eligibility(mode: Any) -> None:
    models = strict_models()
    models[0]["model_info"]["mode"] = mode
    report = check_profile(strict_settings(), models, request_timeout_s=600)
    assert report.failures == _pinned_merge(
        ABSENT_FAILURES, "routes without budget reservations are enabled"
    )


@pytest.mark.parametrize("info", [None, [], "sentinel-secret"])
def test_malformed_model_info_has_safe_model_failures(info: Any) -> None:
    report = check_profile(
        None, [{"model_name": "chat", "model_info": info}], request_timeout_s=600
    )
    assert report.failures == _pinned_merge(
        ABSENT_FAILURES,
        "model chat: no price (input and output cost per token)",
        "model chat: no max output token bound",
        "gateway settings introspection unavailable",
    )


def test_unavailable_settings_keep_exact_absent_order_with_valid_models() -> None:
    report = check_profile(None, strict_models(), request_timeout_s=600)
    assert report.failures == (*ABSENT_FAILURES, "gateway settings introspection unavailable")


def test_missing_settings_are_not_the_unavailable_settings_path() -> None:
    report = check_profile({}, strict_models(), request_timeout_s=600)
    assert report.failures == _pinned_merge(
        ABSENT_FAILURES,
        "spend enforcement does not fail closed when spend counters are stale",
        "routes without budget reservations are enabled",
        "pass-through endpoints are configured",
    )
    assert report.batch_write_interval_s is None


def test_free_input_is_priced_when_output_has_a_positive_cost() -> None:
    models = strict_models()
    models[0]["model_info"]["input_cost_per_token"] = 0
    report = check_profile(strict_settings(batch_write_interval_s=0), models, request_timeout_s=600)
    assert report.failures == tuple(ABSENT_FAILURES)
    assert report.batch_write_interval_s == 0


@pytest.mark.parametrize("model", [None, "model-secret-sentinel", 42, []])
@pytest.mark.parametrize("settings", [None, strict_settings()])
def test_non_mapping_model_entries_fail_without_value_leaks(model: Any, settings: Any) -> None:
    report = check_profile(settings, [model], request_timeout_s=600)
    expected = [
        "model <unknown>: no price (input and output cost per token)",
        "model <unknown>: no max output token bound",
        *ABSENT_FAILURES,
    ]
    if settings is None:
        expected.append("gateway settings introspection unavailable")
    assert list(report.failures) == expected
    assert not report.verified
    assert "model-secret-sentinel" not in repr(report)
