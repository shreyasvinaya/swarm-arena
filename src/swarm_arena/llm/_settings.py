"""Build metered per-call settings and protect them through SDK merges."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from agents import ModelSettings

from swarm_arena._budget.llm_scope import LLMScope
from swarm_arena._model.catalog import ModelSpec
from swarm_arena.llm._cache import _bypass

_RESPONSE_CONTROLS = {
    "cache",
    "caching",
    "preset_cache_key",
    "cache_key",
    "cache_namespace",
    "use-cache",
    "no-cache",
    "no-store",
    "litellm_params",
}
_PROMPT_CONTROLS = {"cache_control_injection_points", "enable_prompt_caching", "cache_control"}

# The accepted spellings and provider mappings are pinned alongside the upstream facts.
_THINKING_CONTROLS = frozenset(
    {
        "thinking",
        "reasoning",
        "reasoning_effort",
        "thinkingConfig",
        "thinking_config",
        "budget_tokens",
        "thinkingBudget",
        "thinking_budget",
        "thinkingLevel",
        "thinking_level",
        "enable_thinking",
        "reasoning_budget",
        "low_effort",
        "reasoningConfig",
        "maxReasoningEffort",
        "effort",
    }
)
_THINKING_CONTAINERS = (
    "extra_body",
    "generationConfig",
    "generation_config",
    "config",
    "chat_template_kwargs",
    "additionalModelRequestFields",
    "additional_model_request_fields",
    "output_config",
)


def injection_points(spec: ModelSpec) -> list[dict[str, Any]] | None:
    """
    Select the provider's Claude prompt-cache breakpoints and lifetime.

    Parameters
    ----------
    spec : ModelSpec
        Approved model and prompt-cache configuration.

    Returns
    -------
    list[dict[str, Any]] or None
        System and last-message points for Claude on Anthropic, Bedrock or
        Vertex, otherwise None. One-hour points carry explicit control blocks.

    Examples
    --------
    >>> injection_points(ModelSpec(model="anthropic/claude", roles=["manager"]))
    [{'location': 'message', 'role': 'system'}, {'location': 'message', 'index': -1}]
    """
    if (
        not spec.prompt_cache.enabled
        or spec.provider not in {"anthropic", "bedrock", "vertex_ai"}
        or "claude" not in spec.model.lower()
    ):
        return None
    points: list[dict[str, Any]] = [
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1},
    ]
    if spec.prompt_cache.ttl == "1h":
        for point in points:
            point["control"] = {"type": "ephemeral", "ttl": "1h"}
    return points


def model_settings(
    spec: ModelSpec,
    *,
    alias: str,
    scope: LLMScope,
    reservation_id: str,
    call_id: str,
    namespace: str,
    read: bool,
    write: bool,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ModelSettings:
    """
    Build usage-enabled settings for one reserved, preallocated LLM call.

    Parameters
    ----------
    spec : ModelSpec
        Approved model specification and output limit.
    alias : str
        Catalog alias, separate from the provider model name.
    scope : LLMScope
        Calling component's tracking identifiers and read suppression.
    reservation_id, call_id : str
        Funding reservation and preallocated journal call identifier.
    namespace : str
        Effective response-cache namespace.
    read, write : bool
        Resolved response-cache permissions.
    max_tokens : int or None, optional
        Output limit override capped by the catalog limit.
    temperature : float or None, optional
        Sampling temperature; sent with drop_params for unsupported models.

    Returns
    -------
    agents.ModelSettings
        String-valued metadata, usage reporting and authoritative extra args.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> scope = Mock(spec=LLMScope, run_id="r", goal_id="g", team="t", agent="a",
    ...              round=1, role="manager", force_no_cache_read=False)
    >>> s = model_settings(ModelSpec(model="openai/gpt-6", roles=["manager"]),
    ...     alias="gpt", scope=scope, reservation_id="res", call_id="llm-1",
    ...     namespace="physika", read=False, write=False)
    >>> s.metadata["round"], s.max_tokens, s.include_usage
    ('1', 4096, True)
    """
    extra: dict[str, Any] = {}
    points = injection_points(spec)
    if points is not None:
        extra["cache_control_injection_points"] = points
    read = read and not scope.force_no_cache_read
    if (read or write) and not _bypass.get():
        extra["cache"] = {
            "use-cache": True,
            "no-cache": not read,
            "no-store": not write,
            "namespace": namespace,
        }
    if temperature is not None:
        extra["drop_params"] = True
    return ModelSettings(
        max_tokens=min(max_tokens or spec.max_tokens, spec.max_tokens),
        include_usage=True,
        temperature=temperature,
        metadata={
            "run_id": scope.run_id,
            "goal_id": scope.goal_id,
            "team": scope.team,
            "agent": scope.agent,
            "round": str(scope.round),
            "role": scope.role,
            "reservation_id": reservation_id,
            "call_id": call_id,
            "model_alias": alias,
        },
        extra_args=extra or None,
    )


def _thinking(settings: ModelSettings) -> bool:
    """Check SDK reasoning and provider escape hatches before cache selection."""
    if settings.reasoning is not None:
        return True

    def enabled(value: Mapping[str, Any]) -> bool:
        # An explicit thinking control bypasses caching even when requesting its
        # disabled variant: native defaults and provider interpretation may differ.
        if any(value.get(key) is not None for key in _THINKING_CONTROLS):
            return True
        return any(
            isinstance(nested := value.get(key), Mapping) and enabled(nested)
            for key in _THINKING_CONTAINERS
        )

    body = settings.extra_body
    return enabled(settings.extra_args or {}) or (isinstance(body, Mapping) and enabled(body))


def merge_model_settings(
    caller: ModelSettings,
    per_call: ModelSettings,
    *,
    streamed: bool = False,
    thinking: bool = False,
) -> ModelSettings:
    """
    Merge caller settings and then enforce metering and cache controls.

    Parameters
    ----------
    caller : agents.ModelSettings
        SDK-resolved caller settings, including provider escape hatches.
    per_call : agents.ModelSettings
        Authoritative result of model_settings for this reserved call.
    streamed, thinking : bool, default=False
        Additional adapter-known reasons to bypass both cache directions.
        SDK reasoning and extra-args/body thinking are detected automatically.

    Returns
    -------
    agents.ModelSettings
        Detached top-level settings; caller inputs are not changed. Metered
        tracking metadata and prompt/response controls win after every merge.

    Notes
    -----
    The metered adapter must use this as its final merge before delegation.
    The SDK forwards extra_args metadata after its metadata field, so the
    escape-hatch metadata and LiteLLM preset keys are removed explicitly.

    Examples
    --------
    >>> caller = ModelSettings(extra_args={"cache": {"namespace": "other"}, "seed": 1})
    >>> merge_model_settings(caller, ModelSettings()).extra_args
    {'seed': 1}
    """
    merged = caller.resolve(per_call)
    extra = dict(merged.extra_args or {})
    streamed = streamed or bool(extra.get("stream"))
    disabled = streamed or thinking or _thinking(merged) or _bypass.get()
    protected = (
        _RESPONSE_CONTROLS
        | _PROMPT_CONTROLS
        | {
            "metadata",
            "litellm_metadata",
            "max_tokens",
            "temperature",
            "include_usage",
            "drop_params",
        }
    )
    extra = {key: value for key, value in extra.items() if key not in protected}
    body = merged.extra_body
    if isinstance(body, Mapping):
        body = {key: value for key, value in body.items() if key not in protected}
    nested = extra.get("extra_body")
    if isinstance(nested, Mapping):
        extra["extra_body"] = {key: value for key, value in nested.items() if key not in protected}
    for key, value in (per_call.extra_args or {}).items():
        if key != "cache" or not disabled:
            extra[key] = value
    if merged.temperature is not None:
        extra["drop_params"] = True
    metadata = {**(caller.metadata or {}), **(per_call.metadata or {})}
    return replace(merged, metadata=metadata or None, extra_args=extra or None, extra_body=body)
