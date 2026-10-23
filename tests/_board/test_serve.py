"""Exercise the local viewer through real HTTP requests."""

import json
import re
import socket
from collections.abc import Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

import pytest
from _fakes.journals import FakeRun, add_pointer

from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.serve import make_server
from swarm_arena._store.base import Store, open_store


@dataclass
class Server:
    board: ArenaBoard
    http: ThreadingHTTPServer
    run: FakeRun
    store: Store

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.http.server_port}"


@pytest.fixture
def server(tmp_path: Path) -> Iterator[Server]:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    run.commit(
        run.plan(parts=["a"]),
        run.candidate("c-grand"),
        run.candidate("c-parent", parents=["c-grand"]),
        run.candidate("c-child", part="a", parents=["c-parent"]),
        run.score("s-1", "c-child", 0.00412345),
        *(
            run.learning(
                f"L-{i}", f"Learning {i}", evidence=["c-child"], verified=i == 1, tags=["solver"]
            )
            for i in range(120)
        ),
        run.learning(
            "L-xss",
            "<script>alert(1)</script>",
            text='<img src=x onerror="x">',
            evidence=["c-child"],
        ),
    )
    add_pointer(str(tmp_path / "arena"), run)
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["toy"])
    http = make_server(board, ["toy"], author="alice", port=0)
    thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield Server(board, http, run, open_store(str(tmp_path / "arena")))
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=60)
        assert not thread.is_alive()
        board.close()


def get(server: Server, path: str, *, timeout: float = 60) -> str:
    with urlopen(server.url + path, timeout=timeout) as response:
        return response.read().decode()


def _token(server: Server) -> str:
    match = re.search(r'name="token" value="([^"]+)"', get(server, "/"))
    assert match is not None
    return match[1]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: object, code: int, msg: str, headers: object, newurl: str
    ) -> None:
        return None


def _post(server: Server, path: str, **fields: str) -> int:
    request = Request(server.url + path, data=urlencode(fields).encode())
    try:
        with build_opener(_NoRedirect()).open(request, timeout=60) as response:
            return response.status
    except HTTPError as error:
        error.close()
        return error.code


def _objects(server: Server) -> list[str]:
    return list(server.store.list("toy/curation/"))


def test_index_paginates(server: Server) -> None:
    # Exclude the extra escaping fixture so this query has exactly 120 rows.
    first = get(server, "/?q=Learning")
    last = get(server, "/?q=Learning&page=3")
    assert first.count('class="learning"') == 50
    assert last.count('class="learning"') == 20
    assert "Next" in first
    assert "Previous" in last
    assert "q=Learning" in last


def test_search_and_filters(server: Server) -> None:
    server.run.commit(
        server.run.learning(
            "L-v", "viscosity verified", verified=True, evidence=["c-child"], tags=["viscosity"]
        ),
        server.run.learning("L-u", "viscosity unverified", evidence=["c-child"]),
    )
    body = get(
        server,
        "/?q=viscosity&verified=1&ns=toy&goal=g1&part=a&team=t1&agent=a1&round=1&tag=viscosity",
    )
    assert "viscosity verified" in body
    assert "viscosity unverified" not in body
    assert get(server, "/?ns=foreign").count('class="learning"') == 0
    assert get(server, "/?run=other").count('class="learning"') == 0


def test_learning_page_shows_evidence_scores(server: Server) -> None:
    server.run.commit(server.run.learning("L-missing", "Missing evidence", evidence=["missing"]))
    body = get(server, "/learning/L-1")
    assert 'href="/candidate/c-child"' in body
    assert "0.004123" in body
    assert "verified" in body
    assert "t1/a1" in body
    assert "unknown candidate" in get(server, "/learning/L-missing")


def test_candidate_page_shows_lineage_and_trace(server: Server) -> None:
    body = get(server, "/candidate/c-child")
    assert 'href="/candidate/c-parent"' in body
    assert 'href="/candidate/c-grand"' in body
    assert "plan     r1" in body
    assert "score    j 0.004123" in body


def test_html_is_escaped(server: Server) -> None:
    body = get(server, "/learning/L-xss")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "<script>alert(1)" not in body
    assert "&lt;img" in body
    assert "<script" not in get(server, "/?q=%3Cscript%3E")


def test_pin_requires_token(server: Server) -> None:
    before = _objects(server)
    assert _post(server, "/learning/L-1/pin") == 403
    assert _post(server, "/learning/L-1/pin", token="wrong") == 403
    assert _post(server, "/learning/L-1/pin", token="非ASCII") == 403
    assert _objects(server) == before
    assert _post(server, "/learning/L-1/pin", token=_token(server)) == 303
    assert len(_objects(server)) == len(before) + 1
    assert "Learning 1" in get(server, "/?pinned=1")


def test_add_writes_human_learning(server: Server) -> None:
    assert (
        _post(
            server,
            "/learnings/add",
            token=_token(server),
            ns="toy",
            title="Human note",
            text="Try this",
            goal="g1",
            tags="solver, human",
        )
        == 303
    )
    names = _objects(server)
    assert len(names) == 1
    obj = json.loads(server.store.read(names[0]).data)
    learning = obj["learning"]
    assert re.fullmatch(r"L-h-[0-9a-f]{16}", learning["id"])
    assert learning["author"] == "human:alice"
    assert learning["run_id"] == ""
    assert learning["goal_id"] == "g1"
    assert learning["round"] == 0
    assert learning["verified"] is False
    assert learning["tags"] == ["solver", "human"]
    assert "Human note" in get(server, "/?human=1")


def test_hidden_shown_only_with_filter(server: Server) -> None:
    assert _post(server, "/learning/L-1/hide", token=_token(server)) == 303
    assert 'href="/learning/L-1"' not in get(server, "/?q=Learning")
    assert 'href="/learning/L-1"' in get(server, "/?hidden=1")
    assert 'href="/learning/L-1"' in get(server, "/?hidden=all&q=Learning+1")
    assert "Learning 1" in get(server, "/learning/L-1")
    assert _post(server, "/learning/L-1/pin", token=_token(server)) == 303


@pytest.mark.parametrize("path", ["/learning/missing", "/candidate/missing", "/missing"])
def test_unknown_id_404(server: Server, path: str) -> None:
    with pytest.raises(HTTPError) as caught:
        get(server, path)
    caught.value.close()
    assert caught.value.code == 404
    assert _post(server, "/learning/missing/pin", token=_token(server)) == 404


def test_rejects_foreign_host(server: Server) -> None:
    before = _objects(server)
    for path, data in [
        ("/", None),
        ("/learning/L-1/pin", urlencode({"token": _token(server)}).encode()),
    ]:
        request = Request(server.url + path, data=data, headers={"Host": "evil.example"})
        with pytest.raises(HTTPError) as caught:
            urlopen(request, timeout=60)
        with caught.value as response:
            assert response.code == 403
            assert b"Learning" not in response.read()
    assert _objects(server) == before
    with urlopen(
        Request(server.url, headers={"Host": f"localhost:{server.http.server_port}"}), timeout=60
    ) as response:
        assert response.status == 200


def test_rejects_namespace_and_invalid_form_without_values(server: Server) -> None:
    token = _token(server)
    assert (
        _post(server, "/learnings/add", token=token, ns="../foreign", title="SECRET", text="x")
        == 400
    )
    assert (
        _post(server, "/learnings/add", token=token, ns="toy", title="SECRET\nVALUE", text="x")
        == 400
    )
    assert not _objects(server)


def test_refresh_is_throttled_and_synchronized(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A controlled clock avoids sleeping or racing the refresh deadline.
    monkeypatch.setattr("swarm_arena._board.serve.time", SimpleNamespace(monotonic=lambda: 100.0))
    original = server.board.frontier
    calls = []
    updated = Event()

    def frontier() -> object:
        calls.append(1)
        result = original()
        updated.set()
        return result

    monkeypatch.setattr(server.board, "frontier", frontier)
    # Configure a fresh server with the same board; no read happens until its first request.
    http = make_server(server.board, ["toy"], author="alice", port=0)
    thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    url = f"http://127.0.0.1:{http.server_port}/"

    def fetch(_: int) -> str:
        with urlopen(url, timeout=60) as response:
            return response.read().decode()

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert all("Learning" in body for body in pool.map(fetch, range(8)))
        assert len(calls) == 1
        server.run.commit(server.run.learning("L-new", "Newly published", evidence=["c-child"]))
        assert "Newly published" not in fetch(0)
        monkeypatch.setattr(
            "swarm_arena._board.serve.time", SimpleNamespace(monotonic=lambda: 106.0)
        )
        updated.clear()
        fetch(0)
        assert updated.wait(60)
        assert "Newly published" in fetch(0)
        assert len(calls) == 2
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=60)
        assert not thread.is_alive()


@pytest.mark.parametrize(
    ("host", "port", "refresh_s"),
    [
        ("127.0.0.1", -1, 5.0),
        ("127.0.0.1", 65536, 5.0),
        ("bad/host", 0, 5.0),
        ("127.0.0.1", 0, -1.0),
        ("127.0.0.1", 0, float("nan")),
    ],
)
def test_server_validates_options(server: Server, host: str, port: int, refresh_s: float) -> None:
    with pytest.raises(ValueError, match=r"host|port|refresh_s"):
        make_server(
            server.board, ["toy"], author="alice", host=host, port=port, refresh_s=refresh_s
        )


def test_run_filter_defaults_to_selected_run(server: Server) -> None:
    http = make_server(server.board, ["toy"], author="alice", port=0, run_id="other")
    thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        url = f"http://127.0.0.1:{http.server_port}/"
        with urlopen(url, timeout=60) as response:
            assert 'class="learning"' not in response.read().decode()
        with urlopen(url + "?run=r1", timeout=60) as response:
            assert 'class="learning"' in response.read().decode()
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=60)
        assert not thread.is_alive()


def test_request_keeps_captured_frontier(server: Server, monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Mapping, Sequence

    from swarm_arena._board.cache import ArenaCache
    from swarm_arena._board.trace import _lines, _SessionCalls
    from swarm_arena._journal.replay import _GatewayKey, _GatewaySpend
    from swarm_arena._model.board_types import Frontier
    from swarm_arena._model.plan import Plan
    from swarm_arena._model.results import Candidate, LLMCall

    def lines(
        cache: ArenaCache,
        frontier: Frontier,
        candidate: Candidate,
        seq: int,
        revisions: Sequence[Plan],
        calls: Mapping[str, Mapping[str, LLMCall]],
        sessions: Mapping[str, _SessionCalls | None],
        gateway_keys: Mapping[tuple[str, str], _GatewayKey],
        gateway_reconcile: Mapping[tuple[str, str], _GatewaySpend],
    ) -> list[str]:
        server.run.commit(server.run.score("s-later", "c-child", 99, judge="late-score"))
        server.board.frontier()
        return _lines(
            cache,
            frontier,
            candidate,
            seq,
            revisions,
            calls,
            sessions,
            gateway_keys,
            gateway_reconcile,
        )

    monkeypatch.setattr("swarm_arena._board.trace._lines", lines)
    assert "late-score" not in get(server, "/candidate/c-child")


def test_unknown_server_namespace_is_rejected(server: Server) -> None:
    with pytest.raises(ValueError, match="namespace outside board"):
        make_server(server.board, ["foreign"], author="alice", port=0)


@contextmanager
def _default_server(server: Server) -> Generator[Server]:
    http = make_server(server.board, ["toy"], author="alice", port=0)
    thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield replace(server, http=http)
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=60)
        assert not thread.is_alive()


@pytest.mark.parametrize(
    "method", ["HEAD", "PUT", "DELETE", "OPTIONS", "PATCH", "CONNECT", "CUSTOM"]
)
def test_every_method_checks_authority(server: Server, method: str) -> None:
    with _default_server(server) as site:
        connection = HTTPConnection("127.0.0.1", site.http.server_port, timeout=60)
        try:
            connection.request(method, "/", headers={"Host": "evil.example"})
            response = connection.getresponse()
            assert response.status == 403
            assert response.reason == "Forbidden"
            assert b"Learning" not in response.read()
        finally:
            connection.close()


@pytest.mark.parametrize("failure", ["syntax", "version", "method"])
def test_protocol_errors_are_fixed(server: Server, failure: str) -> None:
    sentinel = "private-request-sentinel-817592"
    with _default_server(server) as site:
        host = f"127.0.0.1:{site.http.server_port}"
        request = {
            "syntax": f"GET /?q={sentinel} EXTRA HTTP/1.1\r\nHost: {host}\r\n\r\n",
            "version": f"GET / HTTP/{sentinel}\r\nHost: {host}\r\n\r\n",
            "method": f"{sentinel} / HTTP/1.1\r\nHost: {host}\r\n\r\n",
        }[failure]
        with socket.create_connection(("127.0.0.1", site.http.server_port), timeout=60) as client:
            client.sendall(request.encode())
            response = bytearray()
            while data := client.recv(65536):
                response.extend(data)
        assert sentinel.encode() not in response
        assert b"Traceback" not in response
        if failure == "method":
            assert response.startswith(b"HTTP/1.0 501 Not Implemented\r\n")
        elif failure == "syntax":
            assert response.startswith(b"HTTP/1.0 400 Bad Request\r\n")


def test_loopback_startup_does_not_wait_for_reverse_dns(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = Event(), Event()

    def stalled_lookup(name: str = "") -> str:
        entered.set()
        assert release.wait(300), "resolver release deadline"
        return "localhost"

    monkeypatch.setattr(socket, "getfqdn", stalled_lookup)
    with ThreadPoolExecutor(max_workers=1) as pool:
        starting = pool.submit(make_server, server.board, ["toy"], author="alice", port=0)
        try:
            http = starting.result(timeout=30)
            assert not entered.is_set()
            assert http.server_port > 0
        finally:
            release.set()
            starting.result(timeout=60).server_close()


def test_refresh_never_blocks_published_gets(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    import swarm_arena._board.arena as arena

    clock = [100.0]
    monkeypatch.setattr(
        "swarm_arena._board.serve.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    entered, release = Event(), Event()
    original = arena.read_owner

    def read_owner(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(300)
        return original(*args, **kwargs)

    with _default_server(server) as site:
        assert "Learning" in get(site, "/")
        monkeypatch.setattr(arena, "read_owner", read_owner)
        clock[0] = 106.0
        with ThreadPoolExecutor(max_workers=2) as pool:
            refreshing = pool.submit(get, site, "/")
            try:
                assert entered.wait(60)
                assert "Learning" in refreshing.result(timeout=30)
                for path in ("/", "/learning/L-1", "/candidate/c-child"):
                    assert pool.submit(get, site, path).result(timeout=30)
                assert not release.is_set()
            finally:
                release.set()
                refreshing.result(timeout=60)


def test_response_write_does_not_block_other_gets(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _default_server(server) as site:
        get(site, "/")
        entered, release = Event(), Event()
        handler = cast(Any, site.http.RequestHandlerClass)
        original = handler._reply

        def reply(self: Any, status: int, body: str, *, location: str | None = None) -> None:
            if self.path == "/learning/L-1":
                entered.set()
                assert release.wait(300)
            original(self, status, body, location=location)

        monkeypatch.setattr(handler, "_reply", reply)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writing = pool.submit(get, site, "/learning/L-1", timeout=360)
            try:
                assert entered.wait(60)
                assert "Learning" in pool.submit(get, site, "/").result(timeout=30)
                assert not writing.done()
            finally:
                release.set()
                assert writing.result(timeout=60)


@pytest.mark.parametrize("action", ["add", "pin", "hide"])
def test_default_mutation_redirect_publishes_state(server: Server, action: str) -> None:
    with _default_server(server) as site:
        token = _token(site)
        fields = {"token": token, "ns": "toy", "title": "Immediately readable", "text": "New text"}
        path = "/learnings/add" if action == "add" else f"/learning/L-1/{action}"
        connection = HTTPConnection("127.0.0.1", site.http.server_port, timeout=60)
        try:
            connection.request("POST", path, body=urlencode(fields))
            response = connection.getresponse()
            assert response.status == 303
            location = response.getheader("Location")
            response.read()
            assert location is not None
            body = get(site, location)
            marker = {"add": "Immediately readable", "pin": "pinned", "hide": "hidden"}[action]
            if action == "add":
                assert marker in body
            else:
                flags = re.search(r"<p>(.*?)</p>", body)
                assert flags is not None
                assert marker in flags[1]
            assert len(_objects(site)) == 1
        finally:
            connection.close()


@pytest.mark.parametrize("tag", ["", "solver", "absent"])
def test_rendered_search_form_ignores_blank_optional_inputs(server: Server, tag: str) -> None:
    from html.parser import HTMLParser

    class Form(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.fields: dict[str, str] = {}
            self.in_search = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            values = dict(attrs)
            if tag == "form":
                self.in_search = values.get("method") == "get"
            if self.in_search and tag == "input" and values.get("name") is not None:
                self.fields[str(values["name"])] = values.get("value") or ""

    parser = Form()
    parser.feed(get(server, "/"))
    assert {"q", "tag", "run"} <= parser.fields.keys()
    parser.fields.update(q="Learning", tag=tag)
    submitted = get(server, "/?" + urlencode(parser.fields))
    direct = get(server, "/?" + urlencode({"q": "Learning", "tag": tag}) if tag else "/?q=Learning")
    assert submitted.count('class="learning"') == direct.count('class="learning"')
    assert submitted.count('class="learning"') == (0 if tag == "absent" else 50)


def test_run_viewer_blank_run_filter_includes_human_and_other_runs(server: Server) -> None:
    from swarm_arena._board.human_objects import HumanObject, write_human_objects
    from swarm_arena._model.canonical import utc_now
    from swarm_arena._model.results import Learning

    peer = FakeRun(server.run.run_root, "peer")
    peer.commit(peer.plan(), peer.learning("L-peer", "Peer observation", evidence=["c-child"]))
    add_pointer(server.board.config.arena_root, peer)
    write_human_objects(
        server.store,
        "toy",
        "curation",
        [
            HumanObject(
                kind="add",
                author="human:alice",
                learning=Learning(
                    id="L-human",
                    namespace="toy",
                    run_id="",
                    goal_id="",
                    round=0,
                    author="human:alice",
                    title="Human observation",
                    text="Body",
                ),
            )
        ],
        now=utc_now(),
    )
    http = make_server(server.board, ["toy"], run_id=server.run.run_id, author="alice", port=0)
    thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        base = f"http://127.0.0.1:{http.server_port}"
        with urlopen(base + "/?q=observation", timeout=60) as response:
            assert response.read().count(b'class="learning"') == 0
        with urlopen(base + "/?q=observation&run=", timeout=60) as response:
            body = response.read()
        assert b"Peer observation" in body
        assert b"Human observation" in body
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=60)
        assert not thread.is_alive()


def test_first_reader_open_publishes_namespaces_before_frontier(tmp_path: Path) -> None:
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    try:
        board.open("", "", ["toy"])
        with board._read_namespace() as reader:
            assert reader.namespaces() == ("toy",)
        http = make_server(board, ["toy"], author="alice", port=0)
        http.server_close()
    finally:
        board.close()


def test_candidate_page_lists_llm_calls(server: Server) -> None:
    from _fakes.trace_calls import _seed

    _seed(server.run)
    body = get(server, "/candidate/c-llm")
    assert "manager-call" in body
    assert "local/model" in body


def test_run_llm_page_shows_cache_hit_rate_per_agent(server: Server) -> None:
    from _fakes.trace_calls import _seed

    _seed(server.run)
    body = get(server, "/run/r1/llm")
    assert "fd-solver/tuner-1" in body
    assert "50.0%" in body
    assert ":manager" in body  # The source-only failed probe is included.
    with pytest.raises(HTTPError) as error:
        get(server, "/run/absent/llm")
    assert error.value.code == 404
    error.value.close()


@pytest.mark.parametrize("path", ["/candidate/c-llm", "/run/r1/llm"])
def test_llm_get_retains_calls_during_refresh_and_close(
    server: Server, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    from _fakes.trace_calls import _call, _record, _seed

    from swarm_arena._board.arena import _ArenaReader

    _seed(server.run)
    get(server, path)  # Publish the initial server generation.
    entered, release, closing = Event(), Event(), Event()
    method = "trace" if path.startswith("/candidate") else "llm_calls"
    original = getattr(_ArenaReader, method)
    stop = server.board._lifetime.stop

    def held(self: _ArenaReader, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(300)
        return original(self, *args, **kwargs)

    def stopping() -> None:
        stop()
        closing.set()

    monkeypatch.setattr(_ArenaReader, method, held)
    monkeypatch.setattr(server.board._lifetime, "stop", stopping)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(get, server, path, timeout=360)
        close = None
        try:
            assert entered.wait(60)
            server.run.commit(
                _record(
                    _call(
                        server.run,
                        "late-call",
                        "tool",
                        candidate_id="c-llm",
                        team="late-team",
                        agent="late-agent",
                    )
                )
            )
            server.board.frontier()
            # The old GET remains pinned while refresh publishes newer calls.
            close = pool.submit(server.board.close)
            assert closing.wait(60)
            assert not close.done()
        finally:
            release.set()
        body = pending.result(timeout=60)
        assert "late-call" not in body
        assert "late-team" not in body
        if path.startswith("/candidate"):
            assert "manager-call" in body
        else:
            assert "50.0%" in body
        assert close is not None
        close.result(timeout=60)


def test_llm_pages_escape_recorded_metadata_and_check_host(server: Server) -> None:
    from _fakes.trace_calls import _call, _record

    server.run.commit(
        _record(
            _call(
                server.run,
                "escaped",
                model="<script>model</script>",
                team="<img src=x>",
                agent='"agent',
                trace_url="javascript:<script>trace</script>",
            )
        )
    )
    body = get(server, "/run/r1/llm")
    assert "&lt;img src=x&gt;/&quot;agent" in body
    assert "<img" not in body
    request = Request(server.url + "/run/r1/llm", headers={"Host": "evil.example"})
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=60)
    assert error.value.code == 403
    error.value.close()


@pytest.mark.parametrize("records", ["candidate", "plan", "calls-only"])
def test_llm_route_enforces_served_namespace_selection(tmp_path: Path, records: str) -> None:
    from _fakes.trace_calls import _call, _record

    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    for namespace in ("public", "private"):
        run = FakeRun(str(tmp_path / "runs"), f"{namespace}-run", namespace=namespace)
        drafts = [
            _record(
                _call(
                    run, f"{namespace}-call", team=f"{namespace}-team", agent=f"{namespace}-agent"
                )
            ),
        ]
        if records != "calls-only":
            drafts.insert(0, run.plan())
        if records == "candidate":
            drafts.append(run.candidate(f"{namespace}-c"))
        run.commit(*drafts)
        add_pointer(board.config.arena_root, run)
    board.open("", "", ["public", "private"])
    http = make_server(board, ["public"], author="alice", port=0)
    thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    connection = HTTPConnection("127.0.0.1", http.server_port, timeout=60)
    try:
        for namespace, status in (("private", 404), ("public", 200)):
            connection.request("GET", f"/run/{namespace}-run/llm")
            response = connection.getresponse()
            body = response.read().decode()
            assert response.status == status
            assert "private-team/private-agent" not in body
            if status == 200:
                assert "public-team/public-agent" in body
        if records == "candidate":
            connection.request("GET", "/candidate/private-c")
            response = connection.getresponse()
            response.read()
            assert response.status == 404
    finally:
        connection.close()
        http.shutdown()
        http.server_close()
        thread.join(timeout=60)
        assert not thread.is_alive()
        board.close()


def test_candidate_page_shows_committed_gateway_cap(server: Server) -> None:
    from _fakes.gateway_events import gateway_key, reconcile

    server.run.commit(gateway_key("c-child"), reconcile("c-child"))
    server.board.frontier()
    text = get(server, "/candidate/c-child")
    assert "gateway key cand-c-child: max_budget 1.00 USD, cap soft" in text
    assert "gateway spend 1.00 USD (estimated)" in text
