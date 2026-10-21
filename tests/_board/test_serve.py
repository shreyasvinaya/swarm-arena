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
        thread.join(timeout=5)
        assert not thread.is_alive()
        board.close()


def get(server: Server, path: str) -> str:
    with urlopen(server.url + path, timeout=5) as response:
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
        with build_opener(_NoRedirect()).open(request, timeout=5) as response:
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
            urlopen(request, timeout=5)
        with caught.value as response:
            assert response.code == 403
            assert b"Learning" not in response.read()
    assert _objects(server) == before
    with urlopen(
        Request(server.url, headers={"Host": f"localhost:{server.http.server_port}"}), timeout=5
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
        with urlopen(url, timeout=5) as response:
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
        assert updated.wait(3)
        assert "Newly published" in fetch(0)
        assert len(calls) == 2
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
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
        with urlopen(url, timeout=5) as response:
            assert 'class="learning"' not in response.read().decode()
        with urlopen(url + "?run=r1", timeout=5) as response:
            assert 'class="learning"' in response.read().decode()
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_request_keeps_captured_frontier(server: Server, monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Sequence

    from swarm_arena._board.cache import ArenaCache
    from swarm_arena._board.trace import _lines
    from swarm_arena._model.board_types import Frontier
    from swarm_arena._model.plan import Plan
    from swarm_arena._model.results import Candidate

    def lines(
        cache: ArenaCache,
        frontier: Frontier,
        candidate: Candidate,
        seq: int,
        revisions: Sequence[Plan],
    ) -> list[str]:
        server.run.commit(server.run.score("s-later", "c-child", 99, judge="late-score"))
        server.board.frontier()
        return _lines(cache, frontier, candidate, seq, revisions)

    monkeypatch.setattr("swarm_arena._board.serve._lines", lines)
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
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize(
    "method", ["HEAD", "PUT", "DELETE", "OPTIONS", "PATCH", "CONNECT", "CUSTOM"]
)
def test_every_method_checks_authority(server: Server, method: str) -> None:
    with _default_server(server) as site:
        connection = HTTPConnection("127.0.0.1", site.http.server_port, timeout=5)
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
        with socket.create_connection(("127.0.0.1", site.http.server_port), timeout=5) as client:
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
        assert release.wait(5)
        return original(*args, **kwargs)

    with _default_server(server) as site:
        assert "Learning" in get(site, "/")
        monkeypatch.setattr(arena, "read_owner", read_owner)
        clock[0] = 106.0
        with ThreadPoolExecutor(max_workers=2) as pool:
            refreshing = pool.submit(get, site, "/")
            assert entered.wait(3)
            try:
                assert "Learning" in refreshing.result(timeout=0.5)
                for path in ("/", "/learning/L-1", "/candidate/c-child"):
                    assert pool.submit(get, site, path).result(timeout=0.5)
            finally:
                release.set()
                refreshing.result(timeout=5)


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
                assert release.wait(5)
            original(self, status, body, location=location)

        monkeypatch.setattr(handler, "_reply", reply)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writing = pool.submit(get, site, "/learning/L-1")
            assert entered.wait(3)
            try:
                assert "Learning" in pool.submit(get, site, "/").result(timeout=0.5)
            finally:
                release.set()
                assert writing.result(timeout=5)


@pytest.mark.parametrize("action", ["add", "pin", "hide"])
def test_default_mutation_redirect_publishes_state(server: Server, action: str) -> None:
    with _default_server(server) as site:
        token = _token(site)
        fields = {"token": token, "ns": "toy", "title": "Immediately readable", "text": "New text"}
        path = "/learnings/add" if action == "add" else f"/learning/L-1/{action}"
        connection = HTTPConnection("127.0.0.1", site.http.server_port, timeout=5)
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
