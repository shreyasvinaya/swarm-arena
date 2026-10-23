"""Complete settings fixtures for the pinned, still-unverified LiteLLM profile."""

from typing import Any

ABSENT_FAILURES: list[str] = [
    "request max_tokens is not bounded per key",
    "no maximum request duration is enforced",
    "virtual keys are not required",
]


def strict_settings(**overrides: Any) -> dict[str, Any]:
    """
    Build complete settings without inventing absent gateway guarantees.

    Parameters
    ----------
    **overrides : Any
        Actual setting names to replace. ``batch_write_interval_s`` aliases
        ``proxy_batch_write_at``. ``request_timeout_s`` is ignored because the
        pinned proxy has no corresponding enforced total-duration setting.

    Returns
    -------
    dict of str to Any
        A fresh flattened mapping; no secret or absent-setting placeholder.

    Examples
    --------
    >>> strict_settings(batch_write_interval_s=5)["proxy_batch_write_at"]
    5
    """
    settings: dict[str, Any] = {
        "disable_budget_reservation": False,
        "fail_closed_budget_enforcement": True,
        "proxy_batch_write_at": 10.0,
        "allowed_routes": ["/chat/completions", "/v1/chat/completions"],
        "pass_through_endpoints": [],
    }
    # No supported total-duration setting exists in the pinned proxy.
    overrides.pop("request_timeout_s", None)
    if "batch_write_interval_s" in overrides:
        settings["proxy_batch_write_at"] = overrides.pop("batch_write_interval_s")
    settings.update(overrides)
    return settings


def strict_models() -> list[dict[str, Any]]:
    """
    Build priced chat-model metadata with an advertised output maximum.

    Returns
    -------
    list of dict
        Fresh model-info entries; the advertised maximum is not a per-key cap.

    Examples
    --------
    >>> strict_models()[0]["model_info"]["max_output_tokens"]
    4096
    """
    return [
        {
            "model_name": "chat-model",
            "litellm_params": {"model": "openai/chat-model"},
            "model_info": {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000002,
                "max_output_tokens": 4096,
                "mode": "chat",
            },
        }
    ]
