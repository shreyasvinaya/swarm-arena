"""Bounded stdlib transport for the private LiteLLM Proxy adapter.

The pinned proxy cannot verify the strict profile. One deadline bounds the caller
and shuts down active sockets across connect, TLS, headers and body reads. DNS
runs in the daemon HTTP worker; an abandoned lookup ends at the OS resolver
timeout. No worker publishes core state. Undelivered creates have a data-only
cleanup owner.
"""

import contextlib
import contextvars
import hashlib
import json
import math
import socket
import ssl
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.client import HTTPConnection, HTTPException, HTTPResponse, HTTPSConnection
from ipaddress import ip_address
from types import TracebackType
from typing import Annotated, Any, Literal, Self, TypeVar
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

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
    token: _Text


class _Info(FrozenModel):
    """Represent actual key-info closure evidence."""

    model_config = ConfigDict(extra="ignore")

    blocked: Annotated[bool, Field(strict=True)] | None
    status: _Text


class _KeyInfo(FrozenModel):
    """Validate live or archived key information."""

    model_config = ConfigDict(extra="ignore")

    key: _Text
    info: _Info


class _NotFound(FrozenModel):
    """Recognize only the pinned key-info ProxyException, never an ingress 404."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["not_found_error"]
    param: Literal["key"]


class _MissingKey(FrozenModel):
    """Validate the bounded error envelope before treating the key as absent."""

    model_config = ConfigDict(extra="ignore")

    error: _NotFound


class _Deleted(FrozenModel):
    """Validate the acknowledged hash list."""

    model_config = ConfigDict(extra="ignore")

    deleted_keys: list[_Text]


class _Cache(FrozenModel):
    """Use zero for unavailable cache counts, without claiming observed savings."""

    model_config = ConfigDict(extra="ignore")

    cache_read_input_tokens: _Count | None = None
    cache_creation_input_tokens: _Count | None = None


class _Attribution(FrozenModel):
    """Keep only the bounded candidate identity needed by reconciliation."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: Annotated[str, Field(strict=True, min_length=1, max_length=256)] | None = None


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
        raw_attribution = self.metadata.get("spend_logs_metadata")
        attribution = _Attribution.model_validate(
            {} if raw_attribution is None else raw_attribution
        )
        metadata: dict[str, Any] = {}
        if raw_cache is not None:
            metadata["additional_usage_values"] = cache.model_dump(exclude_unset=True)
        if attribution.candidate_id is not None:
            metadata["spend_logs_metadata"] = {"candidate_id": attribution.candidate_id}
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
            metadata=metadata,
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
    lock: Any = field(default_factory=threading.Lock)
    transport: "_Transport | None" = None

    def remaining(self, *, retain_response: bool = False) -> float:
        """Refuse the next I/O when the owner has left or its deadline elapsed."""
        remaining = self.deadline - time.monotonic()
        if (self.abandoned.is_set() and not retain_response) or remaining <= 0:
            raise TimeoutError
        return remaining

    def abandon(self) -> None:
        """Release ordinary sockets immediately; retained creates keep their deadline."""
        self.abandoned.set()
        with self.lock:
            transport = self.transport
        if transport is not None and not transport.retain_response:
            transport.stop()


class _Transport:
    """Own interruptible network resources for one request, including headers."""

    def __init__(self, operation: _Operation, retain_response: bool) -> None:
        self.operation = operation
        self._retain_response = retain_response
        self._sent = False
        self._lock = threading.Lock()
        self._stopped = False
        self._sockets: list[socket.socket] = []
        self._timer = threading.Timer(operation.remaining(), self.stop)
        self._timer.daemon = True

    @property
    def retain_response(self) -> bool:
        """Retain only a create whose request has begun sending, never queued DNS."""
        return self._retain_response and self._sent

    def begin_request(self) -> None:
        """Order send admission against abandonment before writing any bytes."""
        with self.operation.lock:
            self.operation.remaining(retain_response=self.retain_response)
            self._sent = True
        self.refresh_timeout()

    def __enter__(self) -> Self:
        with self.operation.lock:
            self.operation.remaining()
            self.operation.transport = self
        self._timer.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._timer.cancel()
        self.stop()
        self._timer.join()
        with self.operation.lock:
            self.operation.transport = None

    def stop(self) -> None:
        """Wake blocked header/body/TLS I/O; an OS DNS lookup owns no socket."""
        with self._lock:
            self._stopped = True
            for sock in self._sockets:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
                sock.close()
            self._sockets.clear()

    def track(self, sock: socket.socket) -> socket.socket:
        """Close a socket arriving after abandonment instead of starting more I/O."""
        with self._lock:
            if self._stopped:
                sock.close()
                raise TimeoutError
            self._sockets.append(sock)
        return sock

    def refresh_timeout(self) -> None:
        """Shrink the socket allowance before send, headers and each body read."""
        remaining = self.operation.remaining(retain_response=self.retain_response)
        with self._lock:
            for sock in self._sockets:
                if sock.fileno() >= 0:
                    sock.settimeout(remaining)

    def addresses(self, host: str, port: int) -> list[Any]:
        """Resolve inside the existing daemon HTTP worker, before opening a socket.

        The operation deadline bounds the caller, not the OS resolver. A lookup
        that outlives an abandoned wait leaves only this daemon thread without
        a socket until the OS resolver timeout. Once resolution returns, check
        abandonment/deadline before opening any socket or sending a request.
        """
        self.operation.remaining()
        try:
            address = ip_address(host)
        except ValueError:
            addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            self.operation.remaining()
            return addresses
        family = socket.AF_INET if address.version == 4 else socket.AF_INET6
        return [(family, socket.SOCK_STREAM, 0, "", (host, port))]

    def connect(self, address: tuple[str, int], *args: Any) -> socket.socket:
        """Share the remaining allowance across resolution and address attempts."""
        host, port = address
        for family, kind, protocol, _, destination in self.addresses(host, port):
            sock = self.track(socket.socket(family, kind, protocol))
            try:
                sock.settimeout(self.operation.remaining(retain_response=self.retain_response))
                sock.connect(tuple(destination))
                return sock
            except OSError:
                sock.close()
        raise OSError


class _HTTPConnection(HTTPConnection):
    """Install deadline-owned sockets in the stdlib HTTP connection."""

    def __init__(self, transport: _Transport, host: str, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._transport = transport
        self._create_connection = transport.connect

    def send(self, data: Any) -> None:
        if self.sock is None:
            self.connect()
        self._transport.begin_request()
        super().send(data)

    def getresponse(self) -> HTTPResponse:
        self._transport.refresh_timeout()
        return super().getresponse()


class _HTTPSConnection(HTTPSConnection):
    """Register the TLS socket before starting a potentially blocking handshake."""

    _context: ssl.SSLContext

    def __init__(self, transport: _Transport, host: str, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._transport = transport
        self._create_connection = transport.connect

    def send(self, data: Any) -> None:
        if self.sock is None:
            self.connect()
        self._transport.begin_request()
        super().send(data)

    def getresponse(self) -> HTTPResponse:
        self._transport.refresh_timeout()
        return super().getresponse()

    def connect(self) -> None:
        HTTPConnection.connect(self)
        assert self.sock is not None
        wrapped = self._context.wrap_socket(
            self.sock, server_hostname=self.host, do_handshake_on_connect=False
        )
        self.sock = self._transport.track(wrapped)
        wrapped.settimeout(
            self._transport.operation.remaining(retain_response=self._transport.retain_response)
        )
        wrapped.do_handshake()


class _HTTPHandler(HTTPHandler):
    """Keep urllib request handling while supplying deadline-aware connections."""

    def __init__(self, transport: _Transport) -> None:
        super().__init__()
        self._transport = transport

    def http_open(self, req: Request) -> Any:
        return self.do_open(lambda host, **kw: _HTTPConnection(self._transport, host, **kw), req)


class _HTTPSHandler(HTTPSHandler):
    """Use normal certificate verification with interruptible TLS handshakes."""

    def __init__(self, transport: _Transport) -> None:
        super().__init__()
        self._transport = transport

    def https_open(self, req: Request) -> Any:
        return self.do_open(lambda host, **kw: _HTTPSConnection(self._transport, host, **kw), req)


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
    There are no retries, redirects or environment/system proxies. Hostname
    resolution runs in the daemon HTTP worker; numeric IPs bypass resolution.
    An abandoned DNS lookup owns no socket and ends at the OS resolver timeout.
    A watchdog shuts down sockets at the operation deadline, including during
    headers and TLS. Limits are 4 MiB per response, 16 MiB per operation, 200
    pages and 10,000 wire rows. Capped/inconsistent results raise GatewayError.
    ``terminal_accounting`` remains False; unavailable settings never verify
    the strict profile.

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
        with _Transport(operation, retain_response) as transport:
            opener = build_opener(
                ProxyHandler({}), _NoRedirect(), _HTTPHandler(transport), _HTTPSHandler(transport)
            )
            missing = False
            try:
                response = opener.open(request, timeout=timeout)
            except HTTPError as error:
                if path != "/key/info" or error.code != 404:
                    error.close()
                    raise
                response = error
                missing = True
            with response:
                status = response.status
                if not isinstance(status, int):
                    raise ValueError
                if not missing and not 200 <= status < 300:
                    raise HTTPError(request.full_url, status, "", response.headers, None)
                content_length = response.headers.get("Content-Length")
                limit = min(_MAX_RESPONSE_BYTES, _MAX_OPERATION_BYTES - operation.consumed)
                if content_length is not None and (
                    int(content_length) < 0 or int(content_length) > limit
                ):
                    raise ValueError
                chunks: list[bytes] = []
                size = 0
                while True:
                    transport.refresh_timeout()
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
            payload = json.loads(b"".join(chunks))
            if missing:
                _MissingKey.model_validate(payload)
                return _MISSING
            return payload

    def _info(self, operation: _Operation, key: VirtualKey) -> _Info | None:
        """Read closure evidence without confusing a JSON null with a missing row."""
        raw = self._request(operation, "GET", "/key/info", query={"key": key.key_id})
        if raw is _MISSING:
            return None
        reply = _KeyInfo.model_validate(raw)
        if reply.key != key.key_id:
            raise ValueError
        return reply.info

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
                reply = _Blocked.model_validate(raw)
                if not reply.blocked or reply.token != key.key_id:
                    raise ValueError
            elif _Deleted.model_validate(raw).deleted_keys != [key.key_id]:
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
            operation.abandon()

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
            if created.key in created.token_id:
                raise ValueError
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
        Timestamp confirmed blocking, expiry, revocation, deletion or absence.

        Parameters
        ----------
        key : VirtualKey
            Identity to inspect.

        Returns
        -------
        datetime or None
            Current UTC observation time when closed, otherwise None. It is not
            a spend-completeness guarantee or the historical closure instant.
            Only a matching key-info echo or the bounded pinned missing-key
            envelope establishes closure. Other 404s raise GatewayError.

        Examples
        --------
        >>> import inspect
        >>> inspect.iscoroutinefunction(LiteLLMProxy.closed_at)
        True
        """

        def work(operation: _Operation) -> datetime | None:
            info = self._info(operation, key)
            return (
                utc_now()
                if info is None or info.blocked or info.status in {"deleted", "expired", "revoked"}
                else None
            )

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
            Metadata retains only spend_logs_metadata.candidate_id (at most 256
            characters) and the two cache counters in additional_usage_values.
            Identical request_id observations collapse within this read;
            conflicts reject the entire read. No dedup state survives a read.

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
            rows: dict[str, SpendRow] = {}
            fingerprints: dict[str, bytes] = {}
            seen = 0
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
                    or len(page.data) != min(_PAGE_SIZE, max(0, page.total - seen))
                ):
                    raise ValueError
                expected = counts
                if any(row.api_key != key.key_id for row in page.data):
                    raise ValueError
                seen += len(page.data)
                for raw_row in page.data:
                    row = raw_row.normalized()
                    # Compare validated wire observations before dropping metadata;
                    # keep only their digest, never the upstream request payloads.
                    fingerprint = hashlib.sha256(
                        json.dumps(raw_row.model_dump(), sort_keys=True, allow_nan=False).encode()
                    ).digest()
                    previous = fingerprints.get(row.request_id)
                    if previous is not None and previous != fingerprint:
                        raise ValueError
                    fingerprints[row.request_id] = fingerprint
                    rows[row.request_id] = row
                if number >= page.total_pages:
                    if seen != page.total:
                        raise ValueError
                    return list(rows.values())
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
