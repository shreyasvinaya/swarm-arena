"""Check strict-profile evidence for LiteLLM Proxy 1.107.0rc2 without I/O."""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from swarm_arena._gateway.adapter import ProfileReport
from swarm_arena._gateway.labels import _observation_label

BUDGET_RESERVATIONS: Final[str | None] = "disable_budget_reservation"
FAIL_CLOSED: Final[str | None] = "fail_closed_budget_enforcement"
MAX_REQUEST_DURATION: Final[str | None] = None
BATCH_WRITE_INTERVAL: Final[str | None] = "proxy_batch_write_at"
VIRTUAL_KEYS_REQUIRED: Final[str | None] = None
PASS_THROUGH: Final[str | None] = "pass_through_endpoints"
MAX_TOKENS_BOUND: Final[str | None] = None
RESERVED_ROUTES_ONLY: Final[str | None] = "allowed_routes"

# Only the source-verified chat routes are accepted; broad route groups and
# provider/native pass-through paths do not establish reservation coverage.
_CHAT_ROUTES: Final = frozenset({"/chat/completions", "/v1/chat/completions"})
_MANAGEMENT_ROUTES: Final = frozenset(
    {
        "/key/generate",
        "/key/block",
        "/key/info",
        "/key/delete",
        "/spend/logs/v2",
        "/model/info",
        "/v1/model/info",
        "/health/liveliness",
    }
)


def _number(value: Any) -> float | None:
    """Read a finite nonnegative JSON number without coercing strings or bools."""
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _model_info(model: object) -> Mapping[str, Any]:
    """Read validated model metadata, treating malformed shapes as unknown."""
    value = model.get("model_info") if isinstance(model, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _model_name(model: object) -> str:
    """Keep model identifiers, never representations of malformed values."""
    name = model.get("model_name") if isinstance(model, Mapping) else None
    return _observation_label(name, 256, fallback="<unknown>")


def _reserved_routes(value: Any, models: Sequence[object]) -> bool:
    """Require a narrow route list and reject known non-reserving model modes."""
    if not isinstance(value, list) or not value or not models:
        return False
    if any(type(route) is not str for route in value):
        return False
    routes = set(value)
    if not routes <= _CHAT_ROUTES | _MANAGEMENT_ROUTES or not routes & _CHAT_ROUTES:
        return False
    for model in models:
        info = _model_info(model)
        # Missing price metadata has its own model diagnostic. Known zero-cost
        # models bypass monetary reservation even on an otherwise eligible route.
        if info.get("mode", "chat") != "chat":
            return False
        if (
            _number(info.get("input_cost_per_token"))
            == _number(info.get("output_cost_per_token"))
            == 0
        ):
            return False
    return True


def check_profile(
    settings: Mapping[str, Any] | None,
    models: Sequence[object],
    *,
    request_timeout_s: float,
) -> ProfileReport:
    """
    List strict-profile failures in the pinned requirement order.

    Parameters
    ----------
    settings : Mapping of str to Any or None
        Flattened general and LiteLLM settings, or None when introspection is
        unavailable. None never implies that an unreadable switch is disabled.
    models : Sequence of object
        ``/model/info`` data entries. Non-mapping entries receive both model
        failures under the fixed identifier ``<unknown>``. Model metadata supplies
        per-token prices and advertised output limits; neither proves a per-key cap.
    request_timeout_s : float
        Caller's gateway allowance. The pinned proxy has no enforced total
        duration to compare with it; the report therefore always returns None
        for its duration, independently of this allowance.

    Returns
    -------
    ProfileReport
        Always unverified on 1.107.0rc2: total request duration, virtual-key-only
        access and per-key output enforcement are absent. Unknown batch timing
        is None and adds no failure. With unavailable settings, append a fixed
        introspection diagnostic after model and absent-rule failures.

    Notes
    -----
    Complete fixture settings are not a production introspection mechanism.
    An empty custom pass-through list passes that individual rule only; the
    separate route allowlist must also exclude native and DB-backed routes.
    A model price of zero is valid price data, but two zero token prices cannot
    establish reservation coverage. No upstream request or input mutation occurs.

    Examples
    --------
    >>> report = check_profile(None, [], request_timeout_s=600)
    >>> report.verified, report.request_timeout_s
    (False, None)
    >>> report.failures[-1]
    'gateway settings introspection unavailable'
    """
    failures: list[str] = []
    if settings is not None:
        disabled = settings.get(BUDGET_RESERVATIONS) if BUDGET_RESERVATIONS else True
        if disabled is not False and disabled is not None:
            failures.append("budget reservations are disabled")
        if FAIL_CLOSED is None or settings.get(FAIL_CLOSED) is not True:
            failures.append("spend enforcement does not fail closed when spend counters are stale")

    for model in models:
        info = _model_info(model)
        if any(
            _number(info.get(field)) is None
            for field in ("input_cost_per_token", "output_cost_per_token")
        ):
            failures.append(
                f"model {_model_name(model)}: no price (input and output cost per token)"
            )
    for model in models:
        maximum = _model_info(model).get("max_output_tokens")
        if type(maximum) is not int or maximum <= 0:
            failures.append(f"model {_model_name(model)}: no max output token bound")

    failures.append("request max_tokens is not bounded per key")
    if settings is not None and (
        RESERVED_ROUTES_ONLY is None
        or not _reserved_routes(settings.get(RESERVED_ROUTES_ONLY), models)
    ):
        failures.append("routes without budget reservations are enabled")
    failures.append("no maximum request duration is enforced")
    failures.append("virtual keys are not required")
    if settings is not None and (PASS_THROUGH is None or settings.get(PASS_THROUGH) != []):
        failures.append("pass-through endpoints are configured")
    if settings is None:
        failures.append("gateway settings introspection unavailable")

    batch = None
    if settings is not None and BATCH_WRITE_INTERVAL is not None:
        batch = _number(settings.get(BATCH_WRITE_INTERVAL))
    return ProfileReport(
        verified=not failures,
        failures=tuple(failures),
        batch_write_interval_s=batch,
        request_timeout_s=None,
    )
