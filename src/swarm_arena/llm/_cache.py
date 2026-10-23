"""Select response-cache permissions, construct backends and prove origins."""

import hashlib
import importlib.util
import json
import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit

from litellm.caching.caching import Cache, CacheMode
from litellm.types.caching import LiteLLMCacheType

from swarm_arena._errors import _exception_type_name
from swarm_arena._model.base import LLMRole
from swarm_arena._model.settings import CachePolicy

_bypass: ContextVar[bool] = ContextVar("response_cache_bypass", default=False)


@contextmanager
def bypass_response_cache() -> Generator[None]:
    """
    Disable both response-cache directions for calls in this task's context.

    Yields
    ------
    None
        Child tasks inherit the bypass; the previous value is restored on exit.

    Notes
    -----
    Provider probes use this complete bypass. Anchors and final judging use
    the scope's read-only suppression instead, preserving configured writes.

    Examples
    --------
    >>> with bypass_response_cache():
    ...     cache_switches(CachePolicy(backend="disk"), "judge",
    ...                    force_no_read=False, streamed=False, thinking=False)
    (False, False)
    """
    token = _bypass.set(True)
    try:
        yield
    finally:
        _bypass.reset(token)


def cache_switches(
    policy: CachePolicy | None,
    role: LLMRole,
    *,
    force_no_read: bool,
    streamed: bool,
    thinking: bool,
) -> tuple[bool, bool]:
    """
    Resolve the role's read and write permissions for a single call.

    Parameters
    ----------
    policy : CachePolicy or None
        Human-selected backend and permissions, or no response cache.
    role : LLMRole
        Calling component's role.
    force_no_read : bool
        Suppress reads for anchors and final judging, retaining writes.
    streamed, thinking : bool
        Disable both directions when response replay would lose data.

    Returns
    -------
    tuple[bool, bool]
        Read and write permissions; missing role entries disable both.

    Examples
    --------
    >>> cache_switches(CachePolicy(backend="disk"), "planner",
    ...                force_no_read=False, streamed=False, thinking=False)
    (False, True)
    """
    if policy is None or streamed or thinking or _bypass.get():
        return False, False
    switch = policy.roles.get(role)
    return (switch.read and not force_no_read, switch.write) if switch else (False, False)


def cache_namespace(policy: CachePolicy | None, goal_namespace: str) -> str:
    """
    Choose the explicit cache namespace or the calling goal's namespace.

    Parameters
    ----------
    policy : CachePolicy or None
        Optional response-cache configuration.
    goal_namespace : str
        Calling goal's namespace.

    Returns
    -------
    str
        Explicit override, when present, otherwise the goal namespace.

    Examples
    --------
    >>> cache_namespace(None, "physika")
    'physika'
    """
    return (
        policy.namespace if policy is not None and policy.namespace is not None else goal_namespace
    )


def require_cache_extra(policy: CachePolicy) -> None:
    """
    Require cache-hit dependencies and the selected backend's package.

    Parameters
    ----------
    policy : CachePolicy
        Cache backend to check without constructing a connection.

    Raises
    ------
    ImportError
        Naming the missing module and the cache extra installation command.

    Examples
    --------
    >>> from unittest.mock import patch
    >>> with patch("importlib.util.find_spec", return_value=None):
    ...     require_cache_extra(CachePolicy(backend="disk"))
    Traceback (most recent call last):
        ...
    ImportError: missing fastapi; uv add "swarm-arena[llm,cache] @ git+https://github.com/shreyasvinaya/swarm-arena"
    """
    backend = {"disk": "diskcache", "s3": "boto3", "redis": "redis"}[policy.backend]
    for module in ("fastapi", "orjson", backend):
        if importlib.util.find_spec(module) is None:
            raise ImportError(
                f"missing {module}; "
                'uv add "swarm-arena[llm,cache] @ git+https://github.com/shreyasvinaya/swarm-arena"'
            )


def response_cache(policy: CachePolicy) -> Cache:
    """
    Construct a response cache with per-call opt-in and the configured TTL.

    Parameters
    ----------
    policy : CachePolicy
        Disk directory, S3 location or Redis URL environment identifier.

    Returns
    -------
    litellm.caching.caching.Cache
        Cache in ``default_off`` mode; its owner installs and closes it.

    Raises
    ------
    ImportError
        If the cache extra is incomplete.
    ValueError
        If s3_path is malformed or the Redis environment variable is absent.
    RuntimeError
        With a type-only diagnostic if backend construction fails.

    Examples
    --------
    >>> from unittest.mock import patch
    >>> with patch("swarm_arena.llm._cache.Cache") as factory:
    ...     with patch("swarm_arena.llm._cache.require_cache_extra"):
    ...         cache = response_cache(CachePolicy(backend="disk", disk_dir="cache"))
    ...     factory.call_args.kwargs["mode"] == CacheMode.default_off
    True
    """
    require_cache_extra(policy)
    kwargs: dict[str, Any] = {
        "type": LiteLLMCacheType(policy.backend),
        "mode": CacheMode.default_off,
        "ttl": policy.ttl_s,
    }
    if policy.backend == "disk":
        if policy.disk_dir is not None:
            kwargs["disk_cache_dir"] = policy.disk_dir
    elif policy.backend == "s3":
        try:
            location = urlsplit(policy.s3_path or "")
            if (
                location.scheme != "s3"
                or not location.netloc
                or any(char in location.netloc for char in "@:")
                or location.query
                or location.fragment
            ):
                raise ValueError
        except ValueError:
            raise ValueError("invalid s3_path") from None
        kwargs.update(s3_bucket_name=location.netloc, s3_path=location.path.lstrip("/"))
    else:
        name = policy.redis_url_env
        if name is None or name not in os.environ:
            raise ValueError(f"missing Redis environment variable {name}") from None
        kwargs["url"] = os.environ[name]
    try:
        return Cache(**kwargs)
    except Exception as error:
        raise RuntimeError(
            f"response cache construction failed ({_exception_type_name(error)})"
        ) from None


def cache_identity(
    kwargs: Mapping[str, Any], response: Any, *, effective_key: str | None = None
) -> str | None:
    """
    Hash a proven effective cache key and its generating response identity.

    Parameters
    ----------
    kwargs : Mapping[str, Any]
        LiteLLM callback kwargs, including litellm_params.preset_cache_key.
    response : Any
        Upstream response object or dictionary retaining the generating id.

    effective_key : str or None, optional
        Key captured from the exact outbound request or actual cache write.
        Supply it when a miss callback omits its key; never reconstruct it from
        normalized callback fields.

    Returns
    -------
    str or None
        SHA-256 of namespace, effective key and generating response id, suitable
        for ``record_call(cache_identity=...)`` and replay. Missing or conflicting
        evidence returns None, never an inferred request-payload origin.

    Notes
    -----
    The pinned cache prefixes the effective key with its namespace. Hits retain
    the generating response id; a no-read refresh replaces that id. No binding,
    run clock, cache backend or stored payload is read or changed here. The
    callback adapter must call this inside its supervised preparation boundary.

    Examples
    --------
    >>> cache_identity({}, {"id": "response-1"}) is None
    True
    >>> kw = {"litellm_params": {"preset_cache_key": "physika:key"}}
    >>> len(cache_identity(kw, {"id": "response-1"}))
    64
    """
    params = kwargs.get("litellm_params")
    preset = params.get("preset_cache_key") if isinstance(params, Mapping) else None
    if isinstance(response, Mapping):
        response_id, hidden = response.get("id"), response.get("_hidden_params", {})
    else:
        response_id = getattr(response, "id", None)
        hidden = getattr(response, "_hidden_params", {})
    response_key = hidden.get("cache_key") if isinstance(hidden, Mapping) else None
    keys = [key for key in (effective_key, preset, response_key) if key is not None]
    if not keys or any(type(key) is not str or not key or key != keys[0] for key in keys):
        return None
    if type(response_id) is not str or not response_id:
        return None
    key = keys[0]
    namespace = key.rpartition(":")[0]
    identity = json.dumps([namespace, key, response_id], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()
