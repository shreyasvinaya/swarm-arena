"""Complete settings fixtures for the pinned, still-unverified LiteLLM profile."""

import hashlib
import json
import math
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Self
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

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


class FakeGateway:
    """
    Serve the pinned proxy's supported wire shapes on loopback only.

    Parameters
    ----------
    strict : bool, default=True
        Supply priced model metadata. This never makes the profile verified.
    batch_write_interval_s : float or None, default=0.0
        Retained fixture setting; no supported config endpoint exposes it.
    max_request_duration_s : float or None, default=0.1
        Retained fixture setting, never advertised as an upstream guarantee.

    Attributes
    ----------
    requests : list
        Method, query-free path and decoded JSON/query, in arrival order.
    keys : dict
        Hash-indexed key state, including secret, alias, blocked and deleted.
        Deleted keys remain archived for key-info. A missing block route/key
        returns 404; only the separate key-info reply establishes absence.
    fail, delay, malformed : dict, dict, set
        Per-route HTTP status, response delay and malformed-JSON injection.
    failure_bodies, raw_responses : dict, dict
        Error JSON overrides and exact response bytes, including ingress 404s.
    entered, hold : dict
        Per-route events set after handling and gates held before responding.
    page_overrides, row_overrides : dict
        Envelope-by-page and per-row mutations for hostile response tests.
    chat_visible_after_s : float
        Delay before chat spend is visible. ``reveal_spend`` is the event-driven
        alternative for tests that must not depend on wall-clock latency.

    Examples
    --------
    >>> gateway = FakeGateway()
    >>> gateway.keys, gateway.requests
    ({}, [])
    """

    def __init__(
        self,
        *,
        strict: bool = True,
        batch_write_interval_s: float | None = 0.0,
        max_request_duration_s: float | None = 0.1,
    ) -> None:
        self.strict = strict
        self.batch_write_interval_s = batch_write_interval_s
        self.max_request_duration_s = max_request_duration_s
        self.master_key = "sk-master-sentinel"
        self.url = ""
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.authorizations: list[str | None] = []
        self.keys: dict[str, dict[str, Any]] = {}
        self.fail: dict[str, int] = {}
        self.failure_bodies: dict[str, Any] = {}
        self.raw_responses: dict[str, bytes] = {}
        self.delay: dict[str, float] = {}
        self.malformed: set[str] = set()
        self.entered: dict[str, threading.Event] = {}
        self.hold: dict[str, threading.Event] = {}
        self.page_overrides: dict[int, dict[str, Any]] = {}
        self.row_overrides: dict[str, Any] = {}
        self.responses: dict[str, Any] = {}
        self.response_headers: dict[str, dict[str, str]] = {}
        self.chat_visible_after_s = 0.0
        self._spend: list[tuple[float, dict[str, Any]]] = []
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        """Start a loopback server owned by this context."""
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(5)

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                self._handle()

            def do_POST(self) -> None:
                self._handle()

            def _handle(self) -> None:
                parsed = urlsplit(self.path)
                path = parsed.path
                if self.command == "POST":
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                else:
                    body = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
                with gateway._lock:
                    gateway.requests.append((self.command, path, body))
                    gateway.authorizations.append(self.headers.get("Authorization"))
                    status, data, headers = gateway._route(
                        self.command, path, body, self.headers.get("Authorization", "")
                    )
                if path in gateway.entered:
                    gateway.entered[path].set()
                gate = gateway.hold.get(path)
                if gate is not None:
                    while not gateway._stopping.is_set() and not gate.wait(0.05):
                        pass
                gateway._stopping.wait(gateway.delay.get(path, 0.0))
                raw = b"{not json" if path in gateway.malformed else json.dumps(data).encode()
                raw = gateway.raw_responses.get(path, raw)
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    for name, value in {
                        **headers,
                        **gateway.response_headers.get(path, {}),
                    }.items():
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(raw)
                except (OSError, TimeoutError):
                    pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release held replies and join all server-owned threads."""
        self._stopping.set()
        for event in self.hold.values():
            event.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            assert not self._thread.is_alive()

    def add_spend(
        self,
        key_id: str,
        *,
        model: str,
        cost_usd: float,
        tokens: tuple[int, int],
        call_id: str | None,
        visible_after_s: float = 0.0,
        ts: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a spend row with independent primary and optional upstream identities.

        Parameters
        ----------
        key_id : str
            Durable key hash.
        model : str
            Provider model identifier.
        cost_usd : float
            Row cost.
        tokens : tuple of int
            Total prompt and completion tokens.
        call_id : str or None
            Upstream call identifier, allowed to repeat or be absent.
        visible_after_s : float, default=0
            Visibility delay; ``reveal_spend`` can release rows deterministically.
        ts : datetime or None, default=None
            Aware timestamp, defaulting to current UTC.
        metadata : dict or None, default=None
            Raw upstream metadata, including optional additional_usage_values.

        Examples
        --------
        >>> gateway = FakeGateway()
        >>> gateway.add_spend('hash', model='m', cost_usd=0.01, tokens=(1, 2), call_id=None)
        """
        with self._lock:
            self._spend.append(
                (
                    time.monotonic() + visible_after_s,
                    {
                        "api_key": key_id,
                        "request_id": uuid4().hex,
                        "litellm_call_id": call_id,
                        "model": model,
                        "prompt_tokens": tokens[0],
                        "completion_tokens": tokens[1],
                        "spend": cost_usd,
                        "startTime": (ts or datetime.now(UTC)).isoformat(),
                        "metadata": {} if metadata is None else metadata,
                    },
                )
            )

    def reveal_spend(self) -> None:
        """
        Make every pending row visible without a sleep.

        Examples
        --------
        >>> gateway = FakeGateway()
        >>> gateway.reveal_spend()
        """
        with self._lock:
            self._spend = [(0.0, row) for _, row in self._spend]

    def _route(
        self,
        method: str,
        path: str,
        body: dict[str, Any],
        authorization: str,
    ) -> tuple[int, Any, dict[str, str]]:
        if path in self.fail:
            return self.fail[path], self.failure_bodies.get(path, {"error": self.master_key}), {}
        if path == "/v1/chat/completions" and method == "POST":
            key_id = next(
                (k for k, v in self.keys.items() if authorization == f"Bearer {v['secret']}"), None
            )
            if (
                key_id is None
                or self.keys[key_id]["blocked"]
                or self.keys[key_id]["deleted"]
                or self.keys[key_id].get("status") in {"expired", "revoked"}
            ):
                return 401, {"error": "unauthorized"}, {}
            call_id = uuid4().hex
            model = body.get("model", "chat-model")
            self.add_spend(
                key_id,
                model=model,
                cost_usd=0.01,
                tokens=(10, 2),
                call_id=call_id,
                visible_after_s=self.chat_visible_after_s,
            )
            return (
                200,
                {
                    "id": call_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
                {"x-litellm-call-id": call_id},
            )
        if authorization != f"Bearer {self.master_key}" and path != "/health/liveliness":
            return 401, {"error": "unauthorized"}, {}
        if path in self.responses:
            return 200, self.responses[path], {}
        if path == "/health/liveliness" and method == "GET":
            return 200, "I'm alive!", {}
        if path == "/model/info" and method == "GET":
            models = strict_models()
            if not self.strict:
                models[0]["model_info"]["input_cost_per_token"] = None
            return 200, {"data": models}, {}
        if path == "/key/generate" and method == "POST":
            secret = "sk-" + uuid4().hex
            key_id = hashlib.sha256(secret.encode()).hexdigest()
            metadata = dict(body["metadata"])
            metadata["default_estimated_output_tokens"] = body["default_estimated_output_tokens"]
            self.keys[key_id] = {
                **body,
                "alias": body["key_alias"],
                "secret": secret,
                "blocked": False,
                "deleted": False,
                "metadata": metadata,
            }
            return 200, {"key": secret, "token_id": key_id, "key_alias": body["key_alias"]}, {}
        if path in {"/key/block", "/key/info"}:
            key_id = body.get("key")
            row = self.keys.get(key_id)
            if row is None:
                if path == "/key/info" and method == "GET":
                    return (
                        404,
                        {
                            "error": {
                                "message": "Key not found in database",
                                "type": "not_found_error",
                                "param": "key",
                                "code": "404",
                            }
                        },
                        {},
                    )
                return 404, {"error": "missing"}, {}
            if path == "/key/block" and method == "POST":
                if row["deleted"]:
                    return 404, {"error": "missing"}, {}
                row["blocked"] = True
                return 200, {"token": key_id, "blocked": True}, {}
            if path == "/key/info" and method == "GET":
                return (
                    200,
                    {
                        "key": key_id,
                        "info": {
                            "blocked": row["blocked"],
                            "status": "deleted"
                            if row["deleted"]
                            else "revoked"
                            if row["blocked"]
                            else row.get("status", "active"),
                        },
                    },
                    {},
                )
        if path == "/key/delete" and method == "POST":
            for key_id in body["keys"]:
                if key_id in self.keys:
                    self.keys[key_id]["deleted"] = True
            return 200, {"deleted_keys": body["keys"]}, {}
        if path == "/spend/logs/v2" and method == "GET":
            if "start_date" not in body or "end_date" not in body:
                return 400, {"error": "dates required"}, {}
            start, end = (
                datetime.strptime(body[name], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                for name in ("start_date", "end_date")
            )
            rows = [
                {**row, **self.row_overrides}
                for visible, row in self._spend
                if visible <= time.monotonic()
                and row["api_key"] == body["api_key"]
                and start <= datetime.fromisoformat(row["startTime"]) <= end
            ]
            page, size = int(body.get("page", 1)), int(body.get("page_size", 50))
            data = {
                "data": rows[(page - 1) * size : page * size],
                "total": len(rows),
                "page": page,
                "page_size": size,
                "total_pages": math.ceil(len(rows) / size),
                "total_is_capped": False,
            }
            data.update(self.page_overrides.get(page, {}))
            return 200, data, {}
        return 404, {"detail": "Not Found"}, {}
