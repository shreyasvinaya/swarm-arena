"""Serve escaped, generation-consistent arena pages without external resources."""

import html
import math
import re
import secrets
import time
from collections.abc import Sequence
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from typing import Literal
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

import anyio

from swarm_arena._board.arena import ArenaBoard, _namespace_frontier, _Published
from swarm_arena._board.cache import ArenaCache, LearningFilter
from swarm_arena._board.human_ops import _apply_human_operation
from swarm_arena._board.leaderboard import format_value
from swarm_arena._board.pointers import _validate_namespace
from swarm_arena._board.trace import _lines
from swarm_arena._errors import _core_error
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.results import Candidate, Learning

_PAGE_SIZE = 50
_MAX_FORM = 65536


def _e(value: object) -> str:
    """Escape every dynamic HTML text and attribute value."""
    return html.escape(str(value), quote=True)


def _link(kind: str, identifier: str) -> str:
    """Quote an identifier as one path component before HTML escaping."""
    return f'<a href="/{kind}/{_e(quote(identifier, safe=""))}">{_e(identifier)}</a>'


def _markers(item: Learning) -> str:
    """Describe frontier-local curation and evidence flags."""
    return ", ".join(
        name
        for name, enabled in (
            ("verified", item.verified),
            ("human", item.from_human),
            ("pinned", item.pinned),
            ("hidden", item.hidden),
        )
        if enabled
    )


def _learning(
    state: _Published, namespaces: Sequence[str], identifier: str
) -> tuple[str, Learning]:
    """Find even a hidden item without weakening public board filtering."""
    for ns in namespaces:
        cache = state.readers.get(ns)
        if cache is not None:
            items = cache.learnings_by_id(_namespace_frontier(state.frontier, ns), [identifier])
            if items:
                return ns, items[0]
    raise KeyError(identifier)


def _candidate(
    state: _Published, namespaces: Sequence[str], identifier: str
) -> tuple[ArenaCache, Frontier, Candidate, int]:
    """Resolve explicit historical evidence within one captured generation."""
    for ns in namespaces:
        cache = state.readers.get(ns)
        if cache is not None:
            frontier = _namespace_frontier(state.frontier, ns)
            pairs = cache.candidates(frontier, ids=[identifier])
            if pairs:
                item, position = pairs[0]
                return cache, frontier, item, position[2]
    raise KeyError(identifier)


def _index(
    state: _Published, namespaces: Sequence[str], params: dict[str, list[str]], run_id: str | None
) -> str:
    """Filter in SQL, hydrating only the current page and preserving query links."""

    def value(key: str) -> str:
        return params.get(key, [""])[0]

    page = max(1, int(value("page") or "1"))
    filters = LearningFilter(
        goal_id=value("goal") or None,
        run=value("run") or run_id,
        part=value("part") or None,
        team=value("team") or None,
        agent=value("agent") or None,
        round=int(value("round")) if value("round") else None,
        tags=params.get("tag", []),
        verified=True if value("verified") == "1" else None,
        human=True if value("human") == "1" else None,
        pinned=True if value("pinned") == "1" else None,
        hidden=None if value("hidden") == "all" else value("hidden") == "1",
    )
    query = value("q") or None
    skip = (page - 1) * _PAGE_SIZE
    selected: list[Learning] = []
    for ns in namespaces:
        if value("ns") and value("ns") != ns:
            continue
        cache = state.readers.get(ns)
        if cache is None:
            continue
        frontier = _namespace_frontier(state.frontier, ns)
        if query is None:
            count = cache.count(frontier, filters=filters)
            if skip >= count:
                skip -= count
                continue
            ids = cache.find(
                frontier,
                query=None,
                filters=filters,
                relevance_goal=None,
                offset=skip,
                limit=_PAGE_SIZE + 1 - len(selected),
            )
        else:
            # count() intentionally has no text argument; select only IDs for skipped rows.
            ids = cache.find(
                frontier,
                query=query,
                filters=filters,
                relevance_goal=None,
                limit=skip + _PAGE_SIZE + 1 - len(selected),
            )
            if skip >= len(ids):
                skip -= len(ids)
                continue
            ids = ids[skip:]
        skip = 0
        items = {item.id: item for item in cache.learnings_by_id(frontier, ids)}
        selected.extend(items[identifier] for identifier in ids)
        if len(selected) > _PAGE_SIZE:
            break
    body = '<h1>Learnings</h1><form method="get">'
    for name in (
        "q",
        "ns",
        "goal",
        "run",
        "part",
        "team",
        "agent",
        "round",
        "tag",
        "verified",
        "human",
        "pinned",
        "hidden",
    ):
        body += f'<label>{name} <input name="{name}" value="{_e(value(name))}"></label> '
    body += "<button>Search</button></form><ul>"
    for item in selected[:_PAGE_SIZE]:
        body += (
            f'<li class="learning">{_link("learning", item.id)} '
            f"{_e(item.title)} — {_e(item.author)} {_e(_markers(item))}</li>"
        )
    body += "</ul>"
    for number, label in ((page - 1, "Previous"), (page + 1, "Next")):
        if (label == "Previous" and page > 1) or (label == "Next" and len(selected) > _PAGE_SIZE):
            query_params = {**params, "page": [str(number)]}
            body += f'<a href="/?{_e(urlencode(query_params, doseq=True))}">{label}</a> '
    return body


def _learning_page(state: _Published, namespaces: Sequence[str], item: Learning, token: str) -> str:
    """Show evidence values using the shared score display policy."""
    body = (
        f"<h1>{_e(item.title)}</h1><pre>{_e(item.text)}</pre>"
        f"<p>{_e(item.author)} {_e(_markers(item))}</p><h2>Evidence</h2><ul>"
    )
    for identifier in item.evidence:
        try:
            cache, frontier, _, _ = _candidate(state, namespaces, identifier)
            scores = cache.scores(frontier, candidate_ids=[identifier])
            values = ", ".join(f"{score.judge}: {format_value(score.value)}" for score, _ in scores)
            values = values or "-"
        except KeyError:
            values = "error: unknown candidate"
        body += f"<li>{_link('candidate', identifier)} {_e(values)}</li>"
    body += "</ul>"
    for action in ("pin", "hide"):
        body += (
            f'<form method="post" action="/learning/{_e(quote(item.id, safe=""))}/{action}">'
            f'<input type="hidden" name="token" value="{_e(token)}">'
            f"<button>{action}</button></form>"
        )
    return body


def _candidate_page(state: _Published, namespaces: Sequence[str], identifier: str) -> str:
    """Render historical lineage and the shared committed trace from the same state."""
    cache, frontier, item, seq = _candidate(state, namespaces, identifier)
    body = f"<h1>{_e(item.id)}</h1><pre>{_e(item.model_dump_json())}</pre><h2>Parents</h2>"
    body += " ".join(_link("candidate", parent) for parent in item.parent_ids) or "-"
    body += "<h2>Ancestors</h2>"
    visited = {item.id}
    pending = list(item.parent_ids)
    ancestors: list[str] = []
    while pending:
        batch = sorted(set(pending) - visited)
        if not batch:
            break
        visited.update(batch)
        ancestors.extend(batch)
        pending = []
        for ns in namespaces:
            reader = state.readers.get(ns)
            if reader is not None:
                for parent, _ in reader.candidates(
                    _namespace_frontier(state.frontier, ns), ids=batch
                ):
                    pending.extend(parent.parent_ids)
    body += " ".join(_link("candidate", parent) for parent in ancestors) or "-"
    revisions = state.plans[item.run_id].revisions[item.goal_id]
    lines = _lines(cache, frontier, item, seq, revisions)
    return body + f"<h2>Trace</h2><pre>{_e(chr(10).join(lines))}</pre>"


def make_server(
    board: ArenaBoard,
    namespaces: Sequence[str],
    *,
    author: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    run_id: str | None = None,
    refresh_s: float = 5.0,
) -> ThreadingHTTPServer:
    """
    Bind a local HTML viewer with token-protected immutable human curation.

    Parameters
    ----------
    board : ArenaBoard
        Open board. The first request publishes a frontier before any read.
    namespaces : Sequence[str]
        Configured namespaces to expose; form submissions cannot escape them.
    author : str
        Human author name, prefixed with ``human:`` on curation objects.
    host : str, default="127.0.0.1"
        Binding address. Explicit non-loopback bindings expose the viewer.
    port : int, default=8765
        TCP port, or zero to ask the operating system for an available port.
    run_id : str or None, default=None
        Default run filter on the index.
    refresh_s : float, default=5.0
        Minimum interval between periodic background refreshes. A successful
        mutation publishes its new state before redirecting.

    Returns
    -------
    ThreadingHTTPServer
        Bound server; the caller owns serve, shutdown and close. Pages have no
        JavaScript or external resources. Each request uses one pinned frontier.

    Raises
    ------
    ValueError
        If the binding, refresh interval or namespace selection is invalid.

    Examples
    --------
    >>> server = make_server(board, ['toy'], author='alice', port=0)  # doctest: +SKIP
    >>> server.server_close()  # doctest: +SKIP
    """
    names = tuple(dict.fromkeys(_validate_namespace(ns) for ns in namespaces))
    if not names or any(ns not in board._namespaces for ns in names):
        raise _core_error(ValueError, "namespace outside board")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", host):
        raise _core_error(ValueError, "invalid host")
    if not 0 <= port <= 65535:
        raise _core_error(ValueError, "port must be between 0 and 65535")
    if not math.isfinite(refresh_s) or refresh_s < 0:
        raise _core_error(ValueError, "refresh_s must be finite and non-negative")
    token = secrets.token_urlsafe(16)
    lock = Lock()
    refreshed: float | None = None
    refreshing = False
    finished = Event()
    refresh_thread: Thread | None = None
    closed = False

    def publish() -> None:
        nonlocal refreshed, refreshing
        succeeded = False
        try:
            board.frontier()
            succeeded = True
        finally:
            with lock:
                if succeeded or refreshed is not None:
                    refreshed = time.monotonic()
                refreshing = False
                finished.set()

    def background_publish() -> None:
        # Published readers remain usable; never log storage exception values.
        with suppress(Exception):
            publish()

    def ensure_frontier() -> None:
        nonlocal refreshing, finished, refresh_thread
        initial_owner = False
        waiting: Event | None = None
        with lock:
            if closed:
                raise _core_error(RuntimeError, "viewer is closed")
            if refreshed is None:
                if refreshing:
                    waiting = finished
                else:
                    refreshing = True
                    finished = Event()
                    initial_owner = True
            elif not refreshing and time.monotonic() - refreshed >= refresh_s:
                refreshing = True
                finished = Event()
                refresh_thread = Thread(target=background_publish, daemon=True)
                refresh_thread.start()
        if initial_owner:
            # Only startup waits: there is no published generation to read yet.
            publish()
        elif waiting is not None:
            waiting.wait()
            with lock:
                if refreshed is None:
                    raise _core_error(RuntimeError, "viewer frontier unavailable")

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(10)

        def parse_request(self) -> bool:
            """Check authority before the framework dispatches any parsed method."""
            if not super().parse_request():
                return False
            if not self._allowed_host():
                self.send_error(403)
                return False
            return True

        def send_error(
            self, code: int, message: str | None = None, explain: str | None = None
        ) -> None:
            """Replace parser and dispatch messages with fixed, value-free responses."""
            try:
                status = HTTPStatus(code)
            except ValueError:
                status = HTTPStatus.INTERNAL_SERVER_ERROR
            self.close_connection = True
            self._reply(status.value, status.phrase)

        def log_message(self, format: str, *args: object) -> None:
            """Keep request inputs and form data out of diagnostic output."""

        def _reply(self, status: int, body: str, *, location: str | None = None) -> None:
            payload = (
                f'<!doctype html><html lang="en"><meta charset="utf-8">'
                f'<title>Arena board</title><body><a href="/">Board</a>{body}</body></html>'
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; form-action 'self'; frame-ancestors 'none'",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            if location is not None:
                self.send_header("Location", location)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _allowed_host(self) -> bool:
            hosts = self.headers.get_all("Host", [])
            return len(hosts) == 1 and hosts[0] in {
                f"127.0.0.1:{server.server_port}",
                f"localhost:{server.server_port}",
                f"{host}:{server.server_port}",
            }

        def do_GET(self) -> None:
            """Route a read after checking its authority and capturing a frontier."""
            self._route(post=False)

        def do_POST(self) -> None:
            """Route one token-protected curation operation."""
            self._route(post=True)

        def _route(self, *, post: bool) -> None:
            try:
                parsed = urlsplit(self.path)
                params = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=100)
                fields: dict[str, list[str]] = {}
                if post:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= length <= _MAX_FORM or self.headers.get("Transfer-Encoding"):
                        self._reply(400, "Invalid form")
                        return
                    fields = parse_qs(
                        self.rfile.read(length).decode("utf-8"),
                        keep_blank_values=True,
                        max_num_fields=100,
                    )
                    supplied = fields.get("token", [""])
                    if len(supplied) != 1 or not secrets.compare_digest(
                        supplied[0].encode("utf-8"), token.encode("ascii")
                    ):
                        self._reply(403, "Forbidden")
                        return
                ensure_frontier()
                if post:
                    self._mutate(parsed.path, fields)
                    return
                with board._pin_state() as state:
                    if parsed.path == "/":
                        body = _index(state, names, params, run_id)
                        body += self._add_form()
                    elif parsed.path.startswith("/learning/"):
                        _, item = _learning(state, names, unquote(parsed.path[len("/learning/") :]))
                        body = _learning_page(state, names, item, token)
                    elif parsed.path.startswith("/candidate/"):
                        body = _candidate_page(
                            state, names, unquote(parsed.path[len("/candidate/") :])
                        )
                    else:
                        raise KeyError("route")
                self._reply(200, body)
            except KeyError:
                self._reply(404, "Not found")
            except (ValueError, UnicodeError):
                self._reply(400, "Invalid request")
            except Exception:
                self._reply(500, "Request failed")

        def _add_form(self) -> str:
            options = "".join(f'<option value="{_e(ns)}">{_e(ns)}</option>' for ns in names)
            return (
                f'<h2>Add learning</h2><form method="post" action="/learnings/add">'
                f'<input type="hidden" name="token" value="{_e(token)}">'
                f'<label>Namespace <select name="ns">{options}</select></label>'
                '<label>Title <input name="title" maxlength="120" required></label>'
                '<label>Text <textarea name="text" maxlength="1500"></textarea></label>'
                '<label>Goal <input name="goal"></label><label>Tags <input name="tags"></label>'
                "<button>Add</button></form>"
            )

        def _mutate(self, path: str, fields: dict[str, list[str]]) -> None:
            def value(key: str) -> str:
                return fields.get(key, [""])[0]

            kind: Literal["add", "pin", "hide"]
            identifier: str | None = None
            if path == "/learnings/add":
                kind = "add"
                namespace = value("ns")
                if namespace not in names:
                    raise ValueError("namespace outside server")
            else:
                pieces = path.split("/")
                if len(pieces) != 4 or pieces[1] != "learning" or pieces[3] not in ("pin", "hide"):
                    raise KeyError("route")
                identifier = unquote(pieces[2])
                kind = "pin" if pieces[3] == "pin" else "hide"
                with board._pin_state() as state:
                    namespace, _ = _learning(state, names, identifier)

            async def apply() -> str:
                nonlocal refreshed
                operation = await _apply_human_operation(
                    board,
                    namespace,
                    kind,
                    author=author,
                    learning_id=identifier,
                    title=value("title"),
                    text=value("text"),
                    goal_id=value("goal"),
                    tags=[tag.strip() for tag in value("tags").split(",") if tag.strip()],
                )
                # POST is a storage operation: wait for its publication while
                # independent GETs continue reading the last published state.
                await run_sync_daemon(board.frontier)
                with lock:
                    refreshed = time.monotonic()
                return operation.learning.id if operation.learning is not None else identifier or ""

            identifier = anyio.run(apply, backend="asyncio")
            self._reply(303, "Saved", location=f"/learning/{quote(identifier, safe='')}")

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False

        def server_close(self) -> None:
            """Stop scheduling refreshes and bound shutdown of the tracked daemon."""
            nonlocal closed
            with lock:
                closed = True
                worker = refresh_thread
            super().server_close()
            if worker is not None:
                worker.join(timeout=1.0)

        def handle_error(self, request: object, client_address: object) -> None:
            """Do not print request-bearing exception tracebacks."""

    server = Server((host, port), Handler)
    return server
