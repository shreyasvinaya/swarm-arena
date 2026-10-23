"""Build metered per-call settings and protect them through SDK merges."""

from collections.abc import Mapping
from dataclasses import fields, replace
from math import isfinite
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

# Explicit thinking examples are pinned alongside the upstream facts. Cache
# eligibility depends on the fail-closed allowlist, not completeness of this set.
_THINKING_CONTROLS = frozenset(
    {
        "think",
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

_OUTPUT_LIMIT_ALIASES = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "maxCompletionTokens",
        "max_response_output_tokens",
        "max_output_tokens",
        "maxOutputTokens",
        "maxTokens",
        "maxTokenCount",
        "max_tokens_to_sample",
        "max_new_tokens",
        "max_gen_len",
        "num_predict",
        "max_length",
    }
)
_OUTPUT_ENVELOPES = {
    "generationConfig": "maxOutputTokens",
    "generation_config": "max_output_tokens",
    "inferenceConfig": "maxTokens",
    "inference_config": "max_tokens",
    "textGenerationConfig": "maxTokenCount",
    "config": "max_output_tokens",
    "parameters": "maxOutputTokens",
    "options": "num_predict",
}

# Pinned non-thinking request controls. New SDK fields and provider keys are
# ineligible until reviewed; the thinking facts are regression evidence only.
_CACHE_SAFE_MODEL_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "tool_choice",
        "parallel_tool_calls",
        "truncation",
        "max_tokens",
        "verbosity",
        "metadata",
        "store",
        "prompt_cache_retention",
        "include_usage",
        "top_logprobs",
        "preserve_raw_usage",
        "timeout",
        "extra_args",
        "extra_body",
    }
)
_CACHE_SAFE_SCALARS = (
    frozenset(
        {
            "temperature",
            "top_p",
            "topP",
            "top_k",
            "topK",
            "frequency_penalty",
            "presence_penalty",
            "parallel_tool_calls",
            "truncation",
            "verbosity",
            "store",
            "prompt_cache_retention",
            "include_usage",
            "top_logprobs",
            "logprobs",
            "seed",
            "n",
            "timeout",
            "num_retries",
            "max_retries",
            "preserve_raw_usage",
            "drop_params",
            "caching",
            "preset_cache_key",
            "cache_key",
            "cache_namespace",
            "enable_prompt_caching",
            "response_mime_type",
        }
    )
    | _OUTPUT_LIMIT_ALIASES
)
_CACHE_SAFE_ENVELOPES = frozenset(
    (*_THINKING_CONTAINERS, *_OUTPUT_ENVELOPES, "extra_args", "litellm_params")
)
_CACHE_OBJECT_FIELDS = {
    "cache": frozenset({"use-cache", "no-cache", "no-store", "namespace", "ttl"}),
    "stream_options": frozenset({"include_usage"}),
    "cache_control": frozenset({"type", "ttl"}),
}


def _cache_scalar(value: Any) -> bool:
    """Recognize ordinary scalar parameter values without inspecting payload data."""
    return value is None or isinstance(value, str | bool | int | float)


def _cache_object(value: Any, allowed: frozenset[str]) -> bool:
    """Check a fixed, non-generation parameter object's field names and values."""
    return isinstance(value, Mapping) and all(
        key in allowed and _cache_scalar(item) for key, item in value.items()
    )


def _cache_response_format(value: Any) -> bool:
    """Keep schema property names separate from provider request parameter names."""
    if not isinstance(value, Mapping) or value.keys() - {"type", "json_schema"}:
        return False
    if not _cache_scalar(value.get("type")):
        return False
    schema = value.get("json_schema")
    return schema is None or (
        isinstance(schema, Mapping)
        and not schema.keys() - {"name", "description", "strict", "schema"}
        and all(_cache_scalar(item) for key, item in schema.items() if key != "schema")
        and isinstance(schema.get("schema", {}), Mapping)
    )


def _cache_parameters(parameters: Mapping[str, Any]) -> bool:
    """Fail closed on unknown keys, including inside known provider envelopes."""
    pending = [parameters]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            return False
        seen.add(id(current))
        for key, value in current.items():
            if key in _CACHE_SAFE_SCALARS:
                if not _cache_scalar(value):
                    return False
            elif key == "stop":
                if not isinstance(value, str) and not (
                    isinstance(value, list) and all(isinstance(item, str) for item in value)
                ):
                    return False
            elif key == "logit_bias":
                if not isinstance(value, Mapping) or not all(
                    isinstance(token, str | int)
                    and not isinstance(token, bool)
                    and type(bias) in (int, float)
                    and (not isinstance(bias, float) or isfinite(bias))
                    for token, bias in value.items()
                ):
                    return False
            elif key in _CACHE_SAFE_ENVELOPES:
                if not isinstance(value, Mapping):
                    return False
                pending.append(value)
            elif key in _CACHE_OBJECT_FIELDS:
                if not _cache_object(value, _CACHE_OBJECT_FIELDS[key]):
                    return False
            elif key in {"metadata", "litellm_metadata"}:
                if not isinstance(value, Mapping) or not all(
                    isinstance(name, str) and isinstance(item, str) for name, item in value.items()
                ):
                    return False
            elif key == "response_format":
                if not _cache_response_format(value):
                    return False
            elif key == "tool_choice":
                if not isinstance(value, str) and not (
                    isinstance(value, Mapping)
                    and not value.keys() - {"type", "name", "function"}
                    and all(
                        _cache_scalar(item) for name, item in value.items() if name != "function"
                    )
                    and _cache_object(value.get("function", {}), frozenset({"name"}))
                ):
                    return False
            elif key == "cache_control_injection_points":
                if not isinstance(value, list):
                    return False
                for point in value:
                    if (
                        not isinstance(point, Mapping)
                        or not _cache_object(
                            {name: item for name, item in point.items() if name != "control"},
                            frozenset({"location", "role", "index"}),
                        )
                        or not _cache_object(point.get("control", {}), frozenset({"type", "ttl"}))
                    ):
                        return False
            elif key == "mock_response":
                if not isinstance(value, str):
                    return False
            elif key == "client":
                # A transport handle, not a provider generation-options object.
                continue
            elif key == "stream":
                if value is not False and value is not None:
                    return False
            else:
                return False
    return True


def _cache_settings(settings: ModelSettings) -> bool:
    """Check every active SDK field against the pinned cache-safe inventory."""
    for field in fields(settings):
        value = getattr(settings, field.name)
        if value is None:
            continue
        if field.name not in _CACHE_SAFE_MODEL_FIELDS:
            return False
        if field.name in {"extra_args", "extra_body"}:
            if not isinstance(value, Mapping) or not _cache_parameters(value):
                return False
        elif not _cache_parameters({field.name: value}):
            return False
    return True


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


def _output_limits(
    value: Mapping[str, Any], cap: int | None, *, native_key: str | None = None
) -> dict[str, Any]:
    """Remove caller aliases and retain the cap in replacing provider envelopes."""
    result: dict[str, Any] = {}
    for key, option in value.items():
        if key in _OUTPUT_LIMIT_ALIASES:
            continue
        if isinstance(option, Mapping) and key in (*_THINKING_CONTAINERS, *_OUTPUT_ENVELOPES):
            option = _output_limits(option, cap, native_key=_OUTPUT_ENVELOPES.get(key))
        result[key] = option
    # A native generation object can replace the generated object wholesale.
    # Keep its authoritative limit even when the caller supplied only temperature.
    if native_key is not None and cap is not None:
        result[native_key] = cap
    return result


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
    Output-limit aliases are removed from escape hatches; replacing native
    generation envelopes receive the authoritative per-call output limit.
    Only pinned non-thinking, non-streaming request parameters are cacheable;
    unknown SDK fields or provider keys disable both directions without removal.

    Examples
    --------
    >>> caller = ModelSettings(extra_args={"cache": {"namespace": "other"}, "seed": 1})
    >>> merge_model_settings(caller, ModelSettings()).extra_args
    {'seed': 1}
    """
    merged = caller.resolve(per_call)
    extra = dict(merged.extra_args or {})
    streamed = streamed or bool(extra.get("stream"))
    disabled = (
        streamed
        or thinking
        or _thinking(merged)
        or _bypass.get()
        or not _cache_settings(caller)
        or not _cache_settings(merged)
    )
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
    extra = _output_limits(extra, per_call.max_tokens)
    if isinstance(body, Mapping):
        body = _output_limits(body, per_call.max_tokens)
    metadata = {**(caller.metadata or {}), **(per_call.metadata or {})}
    return replace(merged, metadata=metadata or None, extra_args=extra or None, extra_body=body)
