"""Bounded stdlib transport for the private LiteLLM Proxy adapter.

The pinned proxy cannot verify the strict profile. Socket timeouts bound each
blocking I/O attempt; a single abandonable operation deadline also covers body
reads, parsing and every pagination request. No transport worker publishes core
state. An undelivered create result has a separate, data-only cleanup owner.
"""

import contextlib
import contextvars
import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.client import HTTPException
from typing import Annotated, Any, TypeVar
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import anyio
from pydantic import ConfigDict, Field

from swarm_arena._errors import ConfigError, _exception_type_name, _is_core_control
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._gateway.adapter import GatewayError, ProfileReport, SpendRow, VirtualKey
from swarm_arena._gateway.profile import check_profile
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.data import FiniteJsonObject
from swarm_arena._model.settings import GatewayConfig
from swarm_arena._model.tooling import Usage
from swarm_arena._outcomes import _Outcome

_PAGE_SIZE = 50
_MAX_PAGES = 200
_MAX_ROWS = 10_000
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_OPERATION_BYTES = 16 * 1024 * 1024
_MISSING = object()
_T = TypeVar("_T")
_Count = Annotated[int, Field(strict=True, ge=0)]
_Text = Annotated[str, Field(strict=True, min_length=1)]


class _Health(FrozenModel):
    """Validate the JSON string returned by liveliness through the safe base."""

    value: str


class _Model(FrozenModel):
    """Keep only the model identifier and strict-profile metadata."""

    model_config = ConfigDict(extra="ignore")

    model_name: _Text
    model_info: FiniteJsonObject


class _Models(FrozenModel):
    """Validate the model-info list envelope."""

    model_config = ConfigDict(extra="ignore")

    data: list[_Model]


class _CreateInput(FrozenModel):
    """Validate creation before any remote side effect can occur."""

    alias: Annotated[str, Field(strict=True, min_length=1, max_length=256)]
    max_budget_usd: Annotated[float, Field(strict=True, ge=0)]
    duration_s: Annotated[float, Field(strict=True, gt=0)]
    metadata: FiniteJsonObject
    default_estimated_output_tokens: Annotated[int, Field(strict=True, ge=1)]


class _Created(FrozenModel):
    """Separate the ephemeral bearer from its durable hash."""

    model_config = ConfigDict(extra="ignore")

    key: _Text = Field(repr=False)
    token_id: _Text


class _Blocked(FrozenModel):
    """Require an affirmative block response."""

    model_config = ConfigDict(extra="ignore")

    blocked: Annotated[bool, Field(strict=True)]


class _Info(FrozenModel):
    """Represent actual key-info closure evidence."""

    model_config = ConfigDict(extra="ignore")

    blocked: Annotated[bool, Field(strict=True)] | None
    status: _Text


class _KeyInfo(FrozenModel):
    """Validate live or archived key information."""

    model_config = ConfigDict(extra="ignore")

    info: _Info


class _Deleted(FrozenModel):
    """Validate the acknowledged hash list."""

    model_config = ConfigDict(extra="ignore")

    deleted_keys: list[_Text]


class _Cache(FrozenModel):
    """Use zero for unavailable cache counts, without claiming observed savings."""

    model_config = ConfigDict(extra="ignore")

    cache_read_input_tokens: _Count | None = None
    cache_creation_input_tokens: _Count | None = None


class _Spend(FrozenModel):
    """Validate the pinned spend-log row before converting it."""

    model_config = ConfigDict(extra="ignore")

    api_key: _Text
    request_id: _Text
    litellm_call_id: str | None
    model: str
    prompt_tokens: _Count
    completion_tokens: _Count
    spend: Annotated[float, Field(strict=True, ge=0)]
    startTime: str
    metadata: FiniteJsonObject

    def normalized(self) -> SpendRow:
        """Produce a fully validated core-owned record."""
        raw_cache = self.metadata.get("additional_usage_values")
        cache = _Cache.model_validate({} if raw_cache is None else raw_cache)
        read = cache.cache_read_input_tokens or 0
        write = cache.cache_creation_input_tokens or 0
        # LiteLLM's normalized prompt count includes cached input. Usage splits it.
        return SpendRow(
            key_id=self.api_key,
            request_id=self.request_id,
            litellm_call_id=self.litellm_call_id,
            model=self.model,
            cost_usd=self.spend,
            ts=self.startTime,
            metadata=self.metadata,
            usage=Usage(
                input_tokens=self.prompt_tokens - read - write,
                output_tokens=self.completion_tokens,
                cache_read_tokens=read,
                cache_write_tokens=write,
            ),
        )


class _Page(FrozenModel):
    """Require all completeness fields; absent flags cannot imply completeness."""

    model_config = ConfigDict(extra="ignore")

    data: list[_Spend]
    total: _Count
    page: Annotated[int, Field(strict=True, ge=1)]
    page_size: Annotated[int, Field(strict=True, ge=1)]
    total_pages: _Count
    total_is_capped: Annotated[bool, Field(strict=True)]


class _NoRedirect(HTTPRedirectHandler):
    """Do not forward master authorization to a redirect target."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


@dataclass
class _Operation:
    """One deadline, byte allowance and abandonment flag for a complete read."""

    deadline: float
    method: str
    path: str
    consumed: int = 0
    abandoned: threading.Event = field(default_factory=threading.Event)

    def remaining(self, *, retain_response: bool = False) -> float:
        """Refuse the next I/O when the owner has left or its deadline elapsed."""
        remaining = self.deadline - time.monotonic()
        if (self.abandoned.is_set() and not retain_response) or remaining <= 0:
            raise TimeoutError
        return remaining


class _CreateOwner:
    """Atomically hand a known key to the caller or to cleanup, never both."""

    def __init__(self, cleanup: Callable[[VirtualKey], None]) -> None:
        self._lock = threading.Lock()
        self._key: VirtualKey | None = None
        self._abandoned = False
        self._cleanup = cleanup

    def publish(self, key: VirtualKey) -> None:
        """Retain ownership even when the daemon's eventual result is discarded."""
        with self._lock:
            abandoned = self._abandoned
            if not abandoned:
                self._key = key
        if abandoned:
            self._cleanup(key)

    def abandon(self) -> None:
        """Detach cleanup without extending the cancelled caller's lifetime."""
        with self._lock:
            self._abandoned = True
            key, self._key = self._key, None
        if key is not None:
            threading.Thread(
                target=self._cleanup, args=(key,), daemon=True, name="swarm-arena-gateway-block"
            ).start()


class LiteLLMProxy:
    """
    Access the pinned proxy using bounded, abandonable stdlib HTTP operations.

    Parameters
    ----------
    url : str
        HTTP(S) proxy base URL, optionally with a path prefix.
    master_key : str
        Ephemeral admin bearer. It is never part of a returned durable record.
    request_timeout_s : float
        Profile's requested model-call allowance, not a transport deadline.
    http_timeout_s : float, default=30
        Total allowance per adapter operation, including every page. Socket I/O
        receives only the remaining allowance. Late known creates get one
        independent best-effort block attempt, bounded by this same allowance,
        without delaying the departed caller or publishing core state.

    Notes
    -----
    There are no retries or redirects. Limits are 4 MiB per response, 16 MiB per
    operation, 200 pages and 10,000 rows. A capped/inconsistent result raises
    GatewayError rather than returning partial spend. ``terminal_accounting``
    remains False and unavailable settings never verify the strict profile.

    Examples
    --------
    >>> proxy = LiteLLMProxy(url='http://localhost:4000', master_key='example',
    ...                      request_timeout_s=600)
    >>> proxy.terminal_accounting
    False
    """

    terminal_accounting: bool = False

    def __init__(
        self,
        *,
        url: str,
        master_key: str,
        request_timeout_s: float,
        http_timeout_s: float = 30.0,
    ) -> None:
        try:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in (request_timeout_s, http_timeout_s)
            ):
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise ConfigError("invalid gateway url or timeout") from None
        self._url = url.rstrip("/")
        self._master_key = master_key
        self._request_timeout_s = request_timeout_s
        self._http_timeout_s = http_timeout_s

    def _request(
        self,
        operation: _Operation,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
        retain_response: bool = False,
    ) -> Any:
        """Read one bounded JSON reply, with no redirects, retries or logging."""
        operation.method, operation.path = method, path
        timeout = operation.remaining()
        data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        request = Request(
            self._url + path + ("?" + urlencode(query) if query else ""),
            data=data,
            headers={
                "Authorization": f"Bearer {self._master_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            response = build_opener(_NoRedirect()).open(request, timeout=timeout)
        except HTTPError as error:
            error.close()
            if path == "/key/info" and error.code == 404:
                return _MISSING
            raise
        with response:
            if not 200 <= response.status < 300:
                raise HTTPError(request.full_url, response.status, "", response.headers, None)
            content_length = response.headers.get("Content-Length")
            limit = min(_MAX_RESPONSE_BYTES, _MAX_OPERATION_BYTES - operation.consumed)
            if content_length is not None and (
                int(content_length) < 0 or int(content_length) > limit
            ):
                raise ValueError
            chunks: list[bytes] = []
            size = 0
            while True:
                operation.remaining(retain_response=retain_response)
                chunk = response.read1(min(65536, limit - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError
                chunks.append(chunk)
            if content_length is not None and size != int(content_length):
                raise ValueError
            operation.consumed += size
        return json.loads(b"".join(chunks))

    def _info(self, operation: _Operation, key: VirtualKey) -> _Info | None:
        """Read closure evidence without confusing a JSON null with a missing row."""
        raw = self._request(operation, "GET", "/key/info", query={"key": key.key_id})
        return None if raw is _MISSING else _KeyInfo.model_validate(raw).info

    def _mutation(self, operation: _Operation, path: str, key: VirtualKey) -> None:
        """Confirm uncertain mutation outcomes using key-info under the same deadline."""
        blocking = path == "/key/block"
        try:
            raw = self._request(
                operation,
                "POST",
                path,
                body={"key": key.key_id} if blocking else {"keys": [key.key_id]},
            )
            if blocking:
                if not _Blocked.model_validate(raw).blocked:
                    raise ValueError
            elif key.key_id not in _Deleted.model_validate(raw).deleted_keys:
                raise ValueError
        except (
            OSError,
            HTTPException,
            ValueError,
            TypeError,
            OverflowError,
            RecursionError,
        ) as error:
            # Genuine controls are never converted into an idempotent outcome.
            # Groups and cancellation do not enter this narrow transport catch.
            if _is_core_control(error):
                raise
            info = self._info(operation, key)
            if info is None or info.status == "deleted" or (blocking and info.blocked is True):
                return
            # An open key is unconfirmed; a blocked key still needs deletion.
            operation.method, operation.path = "POST", path
            raise

    async def _run(self, method: str, path: str, work: Callable[[_Operation], _T]) -> _T:
        """Supervise an entire operation and normalize errors on the awaiting loop."""
        operation = _Operation(time.monotonic() + self._http_timeout_s, method, path)
        outcome = _Outcome()
        try:
            with anyio.fail_after(self._http_timeout_s):
                result = await run_sync_daemon(work, operation)
                operation.remaining()
                return result
        except BaseException as error:
            classified = outcome.classify(error)
            if _is_core_control(classified):
                raise classified from None
            if isinstance(classified, HTTPError):
                detail = f"HTTP {classified.code}"
            else:
                detail = _exception_type_name(classified)
            raise GatewayError(f"{operation.method} {operation.path}: {detail}") from None
        finally:
            operation.abandoned.set()

    def _block_abandoned(self, key: VirtualKey) -> None:
        """Use a detached, bounded owner with no inherited engine context."""
        # Reuse the same operation supervisor, including the deadline on headers
        # and DNS that a socket timeout alone cannot enforce. Cleanup cannot
        # replace the departed caller's outcome. Unknown outcomes rely on expiry.
        with contextlib.suppress(BaseException):
            contextvars.Context().run(anyio.run, self.block_key, key)

    async def preflight(self) -> ProfileReport:
        """
        Read health and models; report unavailable settings without inventing them.

        Returns
        -------
        ProfileReport
            Unverified pinned profile, with unknown request and batch timing.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(LiteLLMProxy.preflight)
        True
        """

        def work(operation: _Operation) -> ProfileReport:
            _Health(value=self._request(operation, "GET", "/health/liveliness"))
            models = _Models.model_validate(self._request(operation, "GET", "/model/info"))
            return check_profile(
                None,
                [model.model_dump() for model in models.data],
                request_timeout_s=self._request_timeout_s,
            )

        return await self._run("GET", "/health/liveliness", work)

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
        Create a key, transferring cleanup ownership only on successful delivery.

        Parameters
        ----------
        alias : str
            Candidate alias.
        max_budget_usd : float
            Full reservation in dollars.
        duration_s : float
            Finite positive expiry interval in seconds, rounded up to whole seconds.
        metadata : Mapping of str to Any
            Finite JSON attribution supplied by the owner.
        default_estimated_output_tokens : int
            Top-level upstream estimate; it is not an enforced output cap.

        Returns
        -------
        tuple of VirtualKey and str
            Durable identity and ephemeral secret. On an uncertain outcome, do
            not dispatch or retry blindly; retain the charge and rely on expiry.
            A known undelivered result is used only for best-effort blocking.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(LiteLLMProxy.create_key).parameters['alias'].kind.name
        'KEYWORD_ONLY'
        """
        owner = _CreateOwner(self._block_abandoned)
        delivered = False

        def work(operation: _Operation) -> tuple[VirtualKey, str]:
            inputs = _CreateInput(
                alias=alias,
                max_budget_usd=max_budget_usd,
                duration_s=duration_s,
                metadata=dict(metadata),
                default_estimated_output_tokens=default_estimated_output_tokens,
            )
            if not inputs.alias.strip():
                raise ValueError
            created = _Created.model_validate(
                self._request(
                    operation,
                    "POST",
                    "/key/generate",
                    body={
                        "key_alias": inputs.alias,
                        "max_budget": inputs.max_budget_usd,
                        "duration": f"{math.ceil(inputs.duration_s)}s",
                        "metadata": inputs.metadata,
                        "default_estimated_output_tokens": inputs.default_estimated_output_tokens,
                    },
                    retain_response=True,
                )
            )
            key = VirtualKey(key_id=created.token_id, alias=inputs.alias)
            owner.publish(key)
            return key, created.key

        try:
            result = await self._run("POST", "/key/generate", work)
            delivered = True
            return result
        finally:
            if not delivered:
                owner.abandon()

    async def block_key(self, key: VirtualKey) -> None:
        """
        Block future authorizations with the durable key hash.

        Already blocked, deleted or definitively missing keys succeed. A failed
        mutation requires key-info confirmation; an unconfirmed outcome raises
        GatewayError. Confirmation shares the original operation deadline.

        Parameters
        ----------
        key : VirtualKey
            Identity to block.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(LiteLLMProxy.block_key)
        True
        """

        def work(operation: _Operation) -> None:
            self._mutation(operation, "/key/block", key)

        await self._run("POST", "/key/block", work)

    async def closed_at(self, key: VirtualKey) -> datetime | None:
        """
        Timestamp confirmed blocking, archived deletion or missing-key closure.

        Parameters
        ----------
        key : VirtualKey
            Identity to inspect.

        Returns
        -------
        datetime or None
            Current UTC observation time when closed, otherwise None. It is not
            a spend-completeness guarantee or the historical closure instant.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(LiteLLMProxy.closed_at)
        True
        """

        def work(operation: _Operation) -> datetime | None:
            info = self._info(operation, key)
            return utc_now() if info is None or info.blocked or info.status == "deleted" else None

        return await self._run("GET", "/key/info", work)

    async def spend_rows(
        self, key: VirtualKey, *, start: datetime, end: datetime
    ) -> list[SpendRow]:
        """
        Read all pages in one fixed inclusive UTC window and operation allowance.

        Parameters
        ----------
        key : VirtualKey
            Durable hash, including after key deletion.
        start : datetime
            Aware lower bound, rounded down to seconds. The caller supplies its
            one-second creation overlap; this layer does not apply it twice.
        end : datetime
            Aware upper bound, rounded up to seconds and fixed across pages.

        Returns
        -------
        list of SpendRow
            Validated rows, preserving each row's own api_key as key_id and
            rejecting a missing or mismatched hash. Primary and nullable upstream
            identities remain distinct. cost_usd is authoritative; usage.usd is
            None. Missing cache counts become zero (unknown), not observed savings.
            Only timezone-aware ISO timestamps are accepted from the wire.

        Raises
        ------
        GatewayError
            Invalid dates, capped/incomplete/inconsistent envelopes, resource
            limits, or HTTP/transport/parse errors. No partial list is returned.

        Examples
        --------
        >>> import inspect
        >>> inspect.signature(LiteLLMProxy.spend_rows).parameters['start'].kind.name
        'KEYWORD_ONLY'
        """

        def work(operation: _Operation) -> list[SpendRow]:
            if (
                start.tzinfo is None
                or end.tzinfo is None
                or start.utcoffset() is None
                or end.utcoffset() is None
            ):
                raise ValueError
            lower, upper = start.astimezone(UTC), end.astimezone(UTC)
            if lower > upper:
                raise ValueError
            lower = lower.replace(microsecond=0)
            if upper.microsecond:
                upper = upper.replace(microsecond=0) + timedelta(seconds=1)
            dates = {
                "start_date": lower.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": upper.strftime("%Y-%m-%d %H:%M:%S"),
            }
            rows: list[SpendRow] = []
            expected: tuple[int, int] | None = None
            for number in range(1, _MAX_PAGES + 1):
                page = _Page.model_validate(
                    self._request(
                        operation,
                        "GET",
                        "/spend/logs/v2",
                        query={
                            "api_key": key.key_id,
                            **dates,
                            "page": number,
                            "page_size": _PAGE_SIZE,
                        },
                    )
                )
                counts = (page.total, page.total_pages)
                if (
                    page.total_is_capped
                    or page.page != number
                    or page.page_size != _PAGE_SIZE
                    or page.total > _MAX_ROWS
                    or page.total_pages > _MAX_PAGES
                    or page.total_pages != (page.total + _PAGE_SIZE - 1) // _PAGE_SIZE
                    or (expected is not None and expected != counts)
                    or len(page.data) != min(_PAGE_SIZE, max(0, page.total - len(rows)))
                ):
                    raise ValueError
                expected = counts
                if any(row.api_key != key.key_id for row in page.data):
                    raise ValueError
                rows.extend(row.normalized() for row in page.data)
                if number >= page.total_pages:
                    if len(rows) != page.total:
                        raise ValueError
                    return rows
            raise ValueError

        return await self._run("GET", "/spend/logs/v2", work)

    async def delete_key(self, key: VirtualKey) -> None:
        """
        Delete one key by hash without erasing its spend identity.

        Already deleted or definitively missing keys succeed. An existing
        blocked key must still be deleted. A failed mutation requires key-info
        confirmation under the same deadline; unconfirmed outcomes raise GatewayError.

        Parameters
        ----------
        key : VirtualKey
            Identity to delete.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(LiteLLMProxy.delete_key)
        True
        """

        def work(operation: _Operation) -> None:
            self._mutation(operation, "/key/delete", key)

        await self._run("POST", "/key/delete", work)


def adapter_for(config: GatewayConfig, environ: Mapping[str, str]) -> LiteLLMProxy:
    """
    Build the private adapter using an environment-supplied master bearer.

    Parameters
    ----------
    config : GatewayConfig
        Gateway address, key variable identifier and requested model allowance.
    environ : Mapping of str to str
        Environment values, never logged or persisted by the adapter.

    Returns
    -------
    LiteLLMProxy
        Configured adapter; construction performs no I/O.

    Raises
    ------
    ConfigError
        The named environment variable is absent or empty.

    Examples
    --------
    >>> adapter_for(GatewayConfig(url='http://localhost:4000'),
    ...             {'LITELLM_MASTER_KEY': 'example'}).terminal_accounting
    False
    """
    master_key = environ.get(config.master_key_env)
    if not master_key:
        raise ConfigError(f"set {config.master_key_env} for the LLM gateway") from None
    return LiteLLMProxy(
        url=config.url, master_key=master_key, request_timeout_s=config.request_timeout_s
    )
