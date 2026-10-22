"""Adapt LiteLLM callbacks to the live owner's single call recorder."""

import asyncio
import logging
from collections.abc import Mapping
from functools import partial
from threading import RLock
from typing import Any, cast

import litellm
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger
from pydantic import BaseModel

from swarm_arena._budget.amount import _MICROS_PER_USD, USD_LIMIT, _usd_from_micro_units
from swarm_arena._budget.llm_scope import LLMScope
from swarm_arena._engine.run import register_run_setup
from swarm_arena._errors import _exception_type_name
from swarm_arena._journal import llm_record as _recorder
from swarm_arena._journal.llm_record import pending_scope, record_call
from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.data import finite_json
from swarm_arena._model.detach import _plain
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.settings import CachePolicy
from swarm_arena._model.tooling import Usage
from swarm_arena._outcomes import _Outcome, _raise_terminal
from swarm_arena.llm._cache import cache_identity, require_cache_extra, response_cache
from swarm_arena.llm._catalog import _cost_micros, call_cost
from swarm_arena.llm._langfuse import setup_langfuse

_log = logging.getLogger("swarm_arena.llm")
_install_lock = RLock()
_installed_policy: CachePolicy | None = None
_CACHE_KEY = "_swarm_arena_effective_cache_key"


def _field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _counter(obj: Any, name: str, default: int = 0) -> int:
    value = _field(obj, name)
    if value is None:
        return default
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def request_payload(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """
    Select only the seven request-artifact fields from callback arguments.

    Parameters
    ----------
    kwargs : Mapping[str, Any]
        LiteLLM callback envelope. Optional parameters override top-level fields.

    Returns
    -------
    dict[str, Any]
        Detached request data, excluding credentials, headers and tracking data.

    Examples
    --------
    >>> request_payload({"model": "m", "messages": [], "api_key": "private"})
    {'model': 'm', 'messages': []}
    >>> request_payload({"optional_params": {"temperature": 0}})
    {'temperature': 0}
    """
    result = {key: kwargs[key] for key in ("model", "messages") if key in kwargs}
    optional = kwargs.get("optional_params") or {}
    for key in ("tools", "tool_choice", "temperature", "max_tokens", "response_format"):
        if key in optional:
            result[key] = optional[key]
        elif key in kwargs:
            result[key] = kwargs[key]
    return _plain(result)


def usage_of(response_obj: Any) -> Usage:
    """
    Convert provider counters without counting prompt-cache tokens twice.

    Parameters
    ----------
    response_obj : Any
        Response mapping or SDK object with a usage field. Missing counters are zero.

    Returns
    -------
    Usage
        Fresh input, output, cache reads and writes including both TTL subsets.

    Raises
    ------
    ValueError
        For invalid counters or TTL subsets exceeding total cache writes.

    Examples
    --------
    >>> usage_of({"usage": {"prompt_tokens": 12, "completion_tokens": 3,
    ...           "prompt_tokens_details": {"cached_tokens": 4}}}).total_tokens
    15
    """
    usage = _field(response_obj, "usage")
    details = _field(usage, "prompt_tokens_details")
    read = _counter(details, "cached_tokens")
    write = _counter(
        details,
        "cache_creation_tokens",
        _counter(
            details,
            "cache_write_tokens",
            _counter(
                usage, "cache_creation_tokens", _counter(usage, "cache_creation_input_tokens")
            ),
        ),
    )
    split = _field(details, "cache_creation_token_details")
    if split is None:
        split = _field(usage, "cache_creation_token_details")
    five, hour = (
        _counter(split, "ephemeral_5m_input_tokens"),
        _counter(split, "ephemeral_1h_input_tokens"),
    )
    if five + hour > write:
        raise ValueError("invalid cache_creation_token_details")
    return Usage(
        input_tokens=max(0, _counter(usage, "prompt_tokens") - read - write),
        output_tokens=_counter(usage, "completion_tokens"),
        cache_read_tokens=read,
        cache_write_tokens=write,
        cache_write_5m_tokens=five,
        cache_write_1h_tokens=hour,
    )


def failure_usage(exc: BaseException) -> Usage | None:
    """
    Read authoritative usage from an exception's original upstream response.

    Parameters
    ----------
    exc : BaseException
        Post-call exception, such as LiteLLM ModifyResponseException.

    Returns
    -------
    Usage or None
        Validated original usage, or None when no authoritative usage exists.
        An exception class alone never proves that a failed call was free.

    Examples
    --------
    >>> failure_usage(RuntimeError("not a usage report")) is None
    True
    """
    response = getattr(exc, "original_response", None)
    return usage_of(response) if _field(response, "usage") is not None else None


def _routing(kwargs: Mapping[str, Any]) -> tuple[dict[str, str], str | None]:
    # Routing must not invoke mapping hooks before we know the owner. LiteLLM
    # supplies dictionaries; bypass subclass methods and normalize their keys.
    if not issubclass(type(kwargs), dict):
        raise ValueError("invalid callback kwargs")
    envelope = cast(dict[str, Any], kwargs)
    params = dict.get(envelope, "litellm_params")
    if params is None:
        return {}, None
    if not issubclass(type(params), dict):
        raise ValueError("invalid litellm_params")
    metadata = dict.get(params, "metadata")
    if metadata is None:
        return {}, None
    if not issubclass(type(metadata), dict):
        raise ValueError("invalid metadata")
    plain = {
        str.__str__(key): value for key, value in dict.items(metadata) if issubclass(type(key), str)
    }
    if "run_id" not in plain:
        return {}, None
    meta = _recorder._metadata(plain)
    lid = dict.get(envelope, "litellm_call_id")
    return meta, None if lid is None else _context_string(lid, "litellm_call_id")


def _event(
    kwargs: Mapping[str, Any],
    response: Any,
    start: Any,
    end: Any,
    meta: dict[str, str],
    lid: str | None,
    scope: LLMScope | None,
    failure: bool,
) -> dict[str, Any]:
    """Detach every provider-owned value inside the owner's preparation worker."""
    error = kwargs.get("exception") if failure else None
    usage = (
        (failure_usage(error) if isinstance(error, BaseException) else None)
        if failure
        else (usage_of(response) if _field(response, "usage") is not None else None)
    )
    unknown = usage is None
    usage = usage or Usage()
    if failure:
        response = getattr(error, "original_response", None)
    hit = kwargs.get("cache_hit")
    if hit is None:
        hit = False
    if type(hit) is not bool:
        raise ValueError("invalid cache_hit")
    if failure:
        hit = False
    cost = _plain(kwargs.get("response_cost")) if not failure else None
    spec = scope.models.root.get(meta.get("model_alias", "")) if scope is not None else None
    if hit or unknown:
        cost = 0.0
    elif spec is not None:
        if spec.prices is not None or cost is None:
            micros = round(_cost_micros(spec, usage))
            # Preserve the exact pricing/rounding before bounded conversion.
            # A finite over-range marker feeds the recorder's existing seam;
            # only that seam caps the record and makes accounting terminal.
            cost = (
                float(USD_LIMIT + 1)
                if micros > USD_LIMIT * _MICROS_PER_USD
                else _usd_from_micro_units(micros)
            )
        elif not (isinstance(cost, (int, float)) and cost > USD_LIMIT):
            cost = call_cost(spec, usage, litellm_cost=cost)
    elif cost is None:
        cost = 0.0
    params = kwargs.get("litellm_params") or {}
    standard = kwargs.get("standard_logging_object") or {}
    provider = (
        kwargs.get("custom_llm_provider")
        or params.get("custom_llm_provider")
        or standard.get("custom_llm_provider")
        or "unknown"
    )
    choices = _field(response, "choices", []) or []
    message = _field(choices[0], "message") if choices else None
    tools = [
        (
            _context_string(_field(tool, "id"), "tool_call_id"),
            _context_string(_field(_field(tool, "function"), "name"), "tool_name"),
        )
        for tool in (_field(message, "tool_calls", []) or [])
    ]
    identity = cache_identity(kwargs, response, effective_key=kwargs.get(_CACHE_KEY))
    payload = response.model_dump(mode="json") if isinstance(response, BaseModel) else response
    latency = (
        (end - start).total_seconds() * 1000 if start is not None and end is not None else None
    )
    return dict(
        metadata=meta,
        model=_context_string(kwargs.get("model", "unknown"), "model"),
        provider=_context_string(provider, "provider"),
        usage=usage,
        cost_usd=cost,
        cache_hit=hit,
        latency_ms=_plain(latency),
        status="error" if failure else "ok",
        error=(
            f"provider call failed ({_exception_type_name(error)})"
            if isinstance(error, BaseException)
            else None
        ),
        request=finite_json(request_payload(kwargs)),
        response=finite_json(_plain(payload)),
        litellm_call_id=lid,
        tool_call_ids=tools,
        settle=True,
        charge_reservation=unknown and not hit,
        cache_identity=identity,
    )


async def _on_owner(
    binding: _recorder._Binding,
    meta: dict[str, str],
    lid: str | None,
    kwargs: Mapping[str, Any],
    response: Any,
    start: Any,
    end: Any,
    failure: bool,
) -> None:
    if not _recorder._owns_callback(binding, meta):
        return
    entry = binding.pending.get(meta.get("reservation_id", ""))
    cid = meta.get("call_id", entry.call_id if entry is not None else "")
    if _recorder._duplicate(binding, cid, lid, entry):
        return
    scope = pending_scope(meta)
    pricing_meta = dict(meta)
    if entry is not None:
        pricing_meta["model_alias"] = entry.model_alias
    outcome = _Outcome()
    try:
        ready, event = await _recorder._owned(
            binding,
            partial(
                _recorder._prepare,
                partial(_event, kwargs, response, start, end, pricing_meta, lid, scope, failure),
            ),
            entry=entry,
        )
        if ready:
            assert event is not None
            await record_call(**event)
    except BaseException as error:
        classified = outcome.classify(error)
        # The writer latch preserves cleanup expiry as well as actual fencing.
        # A registered sanitized hook failure alone is not terminal authority.
        binding.check()
        _raise_terminal(classified)
        _log.error("LLM callback failed (%s)", _exception_type_name(classified))


async def _callback(
    kwargs: Mapping[str, Any],
    response: Any,
    start: Any,
    end: Any,
    *,
    failure: bool,
) -> None:
    outcome = _Outcome()
    try:
        meta, lid = await _recorder._prepare(partial(_routing, kwargs))
    except BaseException as error:
        classified = outcome.classify(error)
        _log.error("LLM callback routing failed (%s)", _exception_type_name(classified))
        return
    if not meta.get("run_id"):
        return
    binding = _recorder._bindings.get(meta["run_id"])
    if binding is None:
        return
    operation = _on_owner(binding, meta, lid, kwargs, response, start, end, failure)
    if asyncio.get_running_loop() is binding.loop:
        await operation
    else:
        future = asyncio.run_coroutine_threadsafe(operation, binding.loop)
        await asyncio.wrap_future(future)


class CallLogger(CustomLogger):
    """
    Record callbacks using the calling run's catalog and supervised lifetime.

    Notes
    -----
    Holds no catalog or per-run state. Unknown failure spend keeps the complete
    reservation; cache hits keep display usage while charging zero. Ordinary
    adaptation errors are logged by type only. Core terminal failures propagate
    and remain observable through the owner's recorder/drain path.

    Examples
    --------
    >>> isinstance(CallLogger(), CustomLogger)
    True
    """

    def __init__(self) -> None:
        super().__init__()

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """
        Record a successful provider response or response-cache hit.

        Parameters
        ----------
        kwargs : dict[str, Any]
            LiteLLM envelope with tracking metadata in litellm_params.
        response_obj : Any
            Provider response including usage and optional tool calls.
        start_time, end_time : Any
            Datetimes used to calculate milliseconds, or None.

        Examples
        --------
        >>> asyncio.run(CallLogger().async_log_success_event({}, None, None, None))
        """
        await _callback(kwargs, response_obj, start_time, end_time, failure=False)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """
        Record authoritative failure usage or keep the full reservation charged.

        Parameters
        ----------
        kwargs : dict[str, Any]
            LiteLLM envelope including exception and tracking metadata.
        response_obj : Any
            Usually None; authoritative failure usage comes from the exception.
        start_time, end_time : Any
            Datetimes used to calculate milliseconds, or None.

        Examples
        --------
        >>> asyncio.run(CallLogger().async_log_failure_event({}, None, None, None))
        """
        await _callback(kwargs, response_obj, start_time, end_time, failure=True)


def _capture_cache_keys(cache: Cache) -> None:
    """Capture the key at its actual upstream route, never from callback fields."""
    get_key = cache.get_cache_key

    def capture(**kwargs: Any) -> str:
        key = get_key(**kwargs)
        logging_obj = kwargs.get("litellm_logging_obj")
        details = getattr(logging_obj, "model_call_details", None)
        if type(details) is dict and type(key) is str:
            details[_CACHE_KEY] = key
        return key

    cast(Any, cache).get_cache_key = capture


def install(config: RunConfig) -> None:
    """
    Install one logger and, when configured, one process-wide response cache.

    Parameters
    ----------
    config : RunConfig
        Run settings. Catalogs remain per scope; None leaves cache installation alone.

    Raises
    ------
    ValueError
        If a different response-cache policy has already been installed.
    ImportError
        If the selected response cache requires missing optional dependencies.

    Examples
    --------
    >>> from unittest.mock import patch
    >>> with patch.object(litellm, "callbacks", []):
    ...     install(RunConfig())
    ...     install(RunConfig())
    ...     sum(isinstance(callback, CallLogger) for callback in litellm.callbacks)
    1
    """
    setup_langfuse(config.tracing)
    global _installed_policy
    with _install_lock:
        if config.cache is not None:
            if _installed_policy is not None and config.cache != _installed_policy:
                raise ValueError("one response-cache policy per process")
            if _installed_policy is None:
                require_cache_extra(config.cache)
                cache = response_cache(config.cache)
                _capture_cache_keys(cache)
                litellm.cache = cache
                _installed_policy = config.cache.model_copy(deep=True)
        if not any(isinstance(callback, CallLogger) for callback in litellm.callbacks):
            litellm.callbacks.append(CallLogger())


register_run_setup(install)
