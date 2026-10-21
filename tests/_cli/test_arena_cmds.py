"""Arena commands consume committed runs and publish immutable human operations."""

import csv
import json
import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from _fakes.finished import FinishedRun, run_again

from swarm_arena._board.cache import LearningFilter
from swarm_arena._cli.main import main
from swarm_arena._scoring.human_review import CSV_HEADER


@pytest.fixture
async def arena_run(finished_run: FinishedRun) -> FinishedRun:
    identifier = "2026-10-21T00-00-00Z_123abc"
    await run_again(finished_run, run_id=identifier)
    return replace(finished_run, run_id=identifier)


def _call(run: FinishedRun, *args: str) -> int:
    return main([*args, "--config", str(run.config_path)])


def _rows(output: str) -> list[list[str]]:
    return [line.split("\t") for line in output.splitlines() if "\t" in line]


def test_leaderboard_namespace(
    finished_run: FinishedRun, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _call(finished_run, "leaderboard", finished_run.namespace) == 0
    out = capsys.readouterr().out
    assert f"== {finished_run.namespace}/" in out
    assert _rows(out)[0][0] == "1"


def test_leaderboard_run_by_judge(
    arena_run: FinishedRun, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _call(arena_run, "leaderboard", arena_run.run_id, "--by", "f") == 0
    rows = _rows(capsys.readouterr().out)
    assert rows
    assert {row[2] for row in rows} == {arena_run.run_id}


def test_leaderboard_verdict_only_import(
    arena_run: FinishedRun, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "review"
    assert _call(arena_run, "review", "export", arena_run.run_id, "--out", str(out)) == 0
    capsys.readouterr()
    path = out / "review.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))[:2]
    assert len(rows) == 2
    rows[0].update(verdict="accept", score="")
    rows[1].update(verdict="reject", score="")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    assert (
        _call(arena_run, "review", "import", arena_run.run_id, str(path), "--reviewer", "alice")
        == 0
    )
    assert "imported 2 scores as human:alice" in capsys.readouterr().out
    assert _call(arena_run, "leaderboard", arena_run.run_id, "--by", "human:alice") == 0
    ranked = _rows(capsys.readouterr().out)
    assert ranked[0][0:2] == ["1", rows[0]["id"]]
    assert ranked[0][4] == "-"


def test_board_search_and_show(
    finished_run: FinishedRun, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _call(finished_run, "board", "search", "", "--run", finished_run.run_id) == 0
    rows = _rows(capsys.readouterr().out)
    assert rows
    identifier = rows[0][0]
    assert (
        _call(finished_run, "board", "show", identifier, "--namespace", finished_run.namespace) == 0
    )
    shown = capsys.readouterr().out
    assert "evidence" in shown
    assert _call(finished_run, "leaderboard", finished_run.namespace) == 0
    candidate = _rows(capsys.readouterr().out)[0][1]
    assert _call(finished_run, "board", "show", candidate, "--run", finished_run.run_id) == 0
    shown = capsys.readouterr().out
    assert all(
        key in shown for key in ("request", "status", "checks", "scores", "parents", "files")
    )


def test_learnings_pin_hide_add_write_one_object_each(
    finished_run: FinishedRun, capsys: pytest.CaptureFixture[str]
) -> None:
    ns = finished_run.namespace
    assert _call(finished_run, "board", "search", "", "--namespace", ns) == 0
    target = _rows(capsys.readouterr().out)[0][0]
    for command in ("pin", "hide"):
        assert (
            _call(
                finished_run, "learnings", command, target, "--namespace", ns, "--author", "alice"
            )
            == 0
        )
    assert (
        _call(
            finished_run,
            "learnings",
            "add",
            "--namespace",
            ns,
            "--title",
            "Human insight",
            "--text",
            "Try this",
            "--author",
            "alice",
        )
        == 0
    )
    identifier = capsys.readouterr().out.strip()
    assert re.fullmatch(r"L-h-[0-9a-f]{16}", identifier)
    objects = list((Path(finished_run.config.arena_root) / ns / "curation").glob("*.jsonl"))
    assert len(objects) == 3
    entries = [json.loads(path.read_text()) for path in objects]
    learning = next(entry["learning"] for entry in entries if entry["kind"] == "add")
    assert (learning["run_id"], learning["round"], learning["goal_id"]) == ("", 0, "")
    assert _call(finished_run, "board", "search", "", "--namespace", ns, "--hidden") == 0
    assert target in capsys.readouterr().out


def test_learnings_pin_unknown_id_writes_nothing(
    finished_run: FinishedRun, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _call(finished_run, "learnings", "pin", "L-missing", "--namespace", finished_run.namespace)
        == 1
    )
    assert "unknown learning" in capsys.readouterr().err
    folder = Path(finished_run.config.arena_root) / finished_run.namespace / "curation"
    assert not list(folder.glob("*.jsonl"))


def test_trace_prints_chain(finished_run: FinishedRun, capsys: pytest.CaptureFixture[str]) -> None:
    assert _call(finished_run, "leaderboard", finished_run.namespace) == 0
    candidate = _rows(capsys.readouterr().out)[0][1]
    assert _call(finished_run, "trace", candidate, "--run", finished_run.run_id) == 0
    out = capsys.readouterr().out
    assert "plan     " in out
    assert "run      " in out
    assert "score    " in out


def test_reindex_cmd(finished_run: FinishedRun, capsys: pytest.CaptureFixture[str]) -> None:
    assert _call(finished_run, "reindex", "--namespace", finished_run.namespace) == 0
    assert f"{finished_run.namespace}: 1 runs" in capsys.readouterr().out


def test_arena_cmds_need_arena_board(
    finished_run: FinishedRun, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena import Ref
    from swarm_arena._board.base import Board

    monkeypatch.setattr(Ref, "build", Mock(return_value=Mock(spec=Board)))
    config = finished_run.config.model_copy(update={"board": Ref(path="swarm_arena:LocalExecutor")})
    finished_run.config_path.write_text(config.dumps())
    assert _call(finished_run, "leaderboard", finished_run.namespace) == 1
    assert "arena commands need ArenaBoard" in capsys.readouterr().err


def test_usage_requires_one_scope() -> None:
    with pytest.raises(SystemExit) as error:
        main(["board", "show", "c-1"])
    assert error.value.code == 2


def test_human_evidence_filters_and_namespace_membership(
    finished_run: FinishedRun, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _call(finished_run, "leaderboard", finished_run.namespace) == 0
    candidate = _rows(capsys.readouterr().out)[0][1]
    assert (
        _call(
            finished_run,
            "learnings",
            "add",
            "--namespace",
            finished_run.namespace,
            "--title",
            "Verified human finding",
            "--text",
            "Measured result",
            "--evidence",
            candidate,
            "--tag",
            "manual",
            "--author",
            "alice",
        )
        == 0
    )
    identifier = capsys.readouterr().out.strip()
    assert (
        _call(
            finished_run,
            "board",
            "search",
            "finding",
            "--namespace",
            finished_run.namespace,
            "--human",
            "--verified",
            "--tag",
            "manual",
        )
        == 0
    )
    found = _rows(capsys.readouterr().out)
    assert found[0][0] == identifier
    assert set(found[0][3].split(",")) == {"human", "verified"}
    assert _call(finished_run, "learnings", "pin", identifier, "--namespace", "other") == 1
    assert "unknown learning" in capsys.readouterr().err
    folder = Path(finished_run.config.arena_root) / "other" / "curation"
    assert not list(folder.glob("*.jsonl"))


def test_invalid_human_text_never_echoes_values(
    finished_run: FinishedRun, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "PRIVATE_SENTINEL_TITLE"
    assert (
        _call(
            finished_run,
            "learnings",
            "add",
            "--namespace",
            finished_run.namespace,
            "--title",
            secret + "\ninvalid",
            "--text",
            "body",
        )
        == 1
    )
    assert secret not in capsys.readouterr().err
    folder = Path(finished_run.config.arena_root) / finished_run.namespace / "curation"
    assert not list(folder.glob("*.jsonl"))


async def test_cancelled_validation_never_publishes(
    finished_run: FinishedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import anyio

    from swarm_arena._board import human_ops
    from swarm_arena._board.arena import ArenaBoard
    from swarm_arena._board.human_objects import HumanObject

    entered, release, exited = threading.Event(), threading.Event(), threading.Event()

    def stalled(*args: object, **kwargs: object) -> HumanObject:
        entered.set()
        try:
            assert release.wait(5)
            return HumanObject(kind="pin", learning_id="L-example", author="human:alice")
        finally:
            exited.set()

    monkeypatch.setattr(human_ops, "_prepare_human_operation", stalled)
    board = ArenaBoard(arena_root=finished_run.config.arena_root)
    try:
        async with anyio.create_task_group() as group:

            async def apply() -> None:
                await human_ops._apply_human_operation(
                    board, finished_run.namespace, "pin", author="alice"
                )

            group.start_soon(apply)
            assert await anyio.to_thread.run_sync(entered.wait, 5)
            group.cancel_scope.cancel()
    finally:
        release.set()
        assert await anyio.to_thread.run_sync(exited.wait, 5)
        await anyio.to_thread.run_sync(board.close)
    folder = Path(finished_run.config.arena_root) / finished_run.namespace / "curation"
    assert not list(folder.glob("*.jsonl"))


def test_namespace_leaderboard_includes_foreign_main_parts(tmp_path: Path) -> None:
    import io
    from contextlib import redirect_stdout

    from _fakes.journals import FakeRun, add_pointer

    from swarm_arena import Goal, PartSpec, Plan, Ref, RunConfig

    arena = str(tmp_path / "arena")
    run = FakeRun(
        str(tmp_path / "runs"),
        "2026-10-21T00-00-00Z_123abc",
        goal=Goal(id="main", namespace="main-ns"),
    )
    plan = Plan(
        goal=run.goal,
        teams=[],
        judges=[Ref(path="missing:Judge", name="main-judge")],
        parts=[
            PartSpec(
                name="p",
                goal=Goal(id="part", namespace="part-ns"),
                judges=[Ref(path="missing:Judge", name="part-judge")],
            )
        ],
    )
    draft = run.plan()
    draft = draft.model_copy(
        update={"payload": {**draft.payload, "plan": plan.model_dump(mode="json")}}
    )
    run.commit(
        draft,
        run.candidate("c-main"),
        run.candidate("c-part", part="p"),
        run.score("s-main", "c-main", 1, judge="main-judge"),
        run.score("s-part", "c-part", 2, judge="part-judge"),
    )
    add_pointer(arena, run, "main-ns")
    add_pointer(arena, run, "part-ns")
    config = RunConfig(
        run_root=run.run_root,
        arena_root=arena,
        board=Ref(
            path="swarm_arena.board:ArenaBoard",
            config={"arena_root": arena, "cache_dir": str(tmp_path / "cache")},
        ),
    )
    path = tmp_path / "run.toml"
    path.write_text(config.dumps())
    output = io.StringIO()
    with redirect_stdout(output):
        assert main(["leaderboard", "part-ns", "--config", str(path)]) == 0
    assert "== part-ns/main/p (part-judge," in output.getvalue()
    assert "c-part\t" in output.getvalue()
    assert "c-main\t" not in output.getvalue()


def test_show_lists_changed_input_paths() -> None:
    import io
    from contextlib import redirect_stdout

    from swarm_arena._board.arena import ArenaBoard
    from swarm_arena._cli.arena_cmds import _show
    from swarm_arena.testing import make_candidate
    from swarm_arena.tool import ToolOutput, ToolRequest

    board = Mock(spec=ArenaBoard)
    board.get_candidate.return_value = make_candidate(
        ToolOutput(status="ok"),
        id="c-edited",
        request=ToolRequest(edits={"tuning.py": "PRIVATE_FILE_CONTENT"}),
    )
    board.scores.return_value = []
    output = io.StringIO()
    with redirect_stdout(output):
        _show(board, "c-edited")
    assert "files: tuning.py" in output.getvalue()
    assert "PRIVATE_FILE_CONTENT" not in output.getvalue()


@pytest.mark.parametrize(
    ("arguments", "option"),
    [
        (["review", "export", "run"], "--top"),
        (["board", "search", "query", "--namespace", "toy"], "--limit"),
    ],
)
@pytest.mark.parametrize("value", ["PRIVATE_NUMERIC_SENTINEL", "-7", "1.5"])
def test_numeric_usage_errors_are_value_free(
    arguments: list[str], option: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main([*arguments, option, value])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert f"{option} must be a non-negative integer" in captured.err
    assert value not in captured.out + captured.err


@pytest.mark.parametrize("value", ["0", "17"])
def test_numeric_options_accept_non_negative_integers(value: str) -> None:
    from swarm_arena._cli.main import _parser

    arguments = _parser().parse_args(["review", "export", "run", "--top", value])
    assert arguments.top == int(value)
    arguments = _parser().parse_args(
        ["board", "search", "query", "--namespace", "toy", "--limit", value]
    )
    assert arguments.limit == int(value)


@pytest.mark.parametrize("namespaces", [[], ["--namespace", "toy"]])
def test_reindex_uses_resolved_arena_root_and_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], namespaces: list[str]
) -> None:
    import sqlite3
    from contextlib import closing

    from _fakes.journals import FakeRun, add_pointer

    from swarm_arena import Ref, RunConfig

    arena = tmp_path / "actual-arena"
    cache = tmp_path / "actual-cache"
    run = FakeRun(str(tmp_path / "runs"), "2026-10-21T00-00-00Z_123abc")
    run.commit(run.plan(), run.candidate("c-one"), run.score("s-one", "c-one", 1))
    run._lease.release()
    add_pointer(str(arena), run)
    config = RunConfig(
        run_root=run.run_root,
        arena_root=str(tmp_path / "unused-arena"),
        board=Ref(
            path="swarm_arena.board:ArenaBoard",
            config={"arena_root": str(arena), "cache_dir": str(cache)},
        ),
    )
    config_path = tmp_path / "run.toml"
    config_path.write_text(config.dumps())
    cfg = ["--config", str(config_path)]
    assert main(["leaderboard", "toy", *cfg]) == 0
    assert "c-one" in capsys.readouterr().out
    database = cache / "toy.sqlite"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("CREATE TABLE stale_cache (marker TEXT)")
    assert main(["reindex", *namespaces, *cfg]) == 0
    assert capsys.readouterr().out == "toy: 1 runs\n"
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT id FROM candidates").fetchall() == [("c-one",)]
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name='stale_cache'").fetchall()
            == []
        )
    assert not (tmp_path / "unused-arena").exists()


@pytest.fixture
def review_cli_run(tmp_path: Path) -> FinishedRun:
    from _fakes.journals import FakeRun, add_pointer

    from swarm_arena import Plan, RunConfig

    config = RunConfig(
        run_root=str(tmp_path / "runs"),
        arena_root=str(tmp_path / "arena"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    run = FakeRun(config.run_root, "2026-10-21T00-00-00Z_123abc")
    plan = run.plan()
    run.commit(plan, run.candidate("c-one"), run.score("s-one", "c-one", 1))
    run._lease.release()
    add_pointer(config.arena_root, run)
    path = tmp_path / "run.toml"
    path.write_text(config.dumps())
    return FinishedRun(
        config=config,
        config_path=path,
        plan=Plan.model_validate(plan.payload["plan"]),
        run_id=run.run_id,
        namespace=run.goal.namespace,
    )


def test_review_export_never_looks_up_reviewer(
    review_cli_run: FinishedRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "swarm_arena._cli.arena_cmds.getpass.getuser",
        Mock(side_effect=AssertionError("export needs no identity")),
    )
    out = tmp_path / "export"
    assert _call(review_cli_run, "review", "export", review_cli_run.run_id, "--out", str(out)) == 0
    assert (out / "review.csv").is_file()


@pytest.mark.parametrize("login", ["Alice", "PRIVATE_LOGIN/INVALID"])
def test_installed_export_accepts_every_login(
    review_cli_run: FinishedRun, tmp_path: Path, login: str
) -> None:
    import os
    import subprocess
    import sys

    done = subprocess.run(
        [
            str(Path(sys.executable).parent / "swarm-arena"),
            "review",
            "export",
            review_cli_run.run_id,
            "--out",
            str(tmp_path / "export"),
            "--config",
            str(review_cli_run.config_path),
        ],
        env=dict(os.environ, LOGNAME=login),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "export" / "review.csv").is_file()


@pytest.mark.parametrize("explicit", [False, True])
def test_review_import_uses_valid_default_or_explicit_reviewer(
    review_cli_run: FinishedRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    explicit: bool,
) -> None:
    login = (
        Mock(side_effect=AssertionError("explicit reviewer needs no login"))
        if explicit
        else Mock(return_value="alice")
    )
    monkeypatch.setattr("swarm_arena._cli.arena_cmds.getpass.getuser", login)
    csv_path = tmp_path / "review.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerow({"id": "c-one", "verdict": "accept", "score": "7"})
    reviewer = ["--reviewer", "bob"] if explicit else []
    assert (
        _call(review_cli_run, "review", "import", review_cli_run.run_id, str(csv_path), *reviewer)
        == 0
    )
    name = "bob" if explicit else "alice"
    assert f"imported 1 scores as human:{name}" in capsys.readouterr().out


def test_review_import_invalid_default_requests_reviewer_without_values(
    review_cli_run: FinishedRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from swarm_arena._board.ingest import run_store
    from swarm_arena._journal.lease import OWNER_KEY
    from swarm_arena._store.base import open_store

    sentinel = "PRIVATE_LOGIN/INVALID"
    monkeypatch.setattr("swarm_arena._cli.arena_cmds.getpass.getuser", Mock(return_value=sentinel))
    store = run_store(review_cli_run.config.run_root, review_cli_run.run_id)
    before = store.read(OWNER_KEY).data
    assert (
        _call(
            review_cli_run, "review", "import", review_cli_run.run_id, str(tmp_path / "missing.csv")
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "--reviewer" in captured.err
    assert sentinel not in captured.out + captured.err
    assert store.read(OWNER_KEY).data == before
    assert open_store(review_cli_run.config.arena_root).list("toy/reviews/") == []


def test_board_serve_smoke(finished_run: FinishedRun) -> None:
    import os
    import selectors
    import signal
    import subprocess
    import sys
    from urllib.request import urlopen

    from swarm_arena._board.arena import ArenaBoard

    board = ArenaBoard(arena_root=finished_run.config.arena_root)
    board.open("", "", [finished_run.namespace])
    try:
        board.frontier()
        with board._pin_state() as state:
            cache = state.readers[finished_run.namespace]
            learning_id = cache.find(
                state.frontier, query=None, filters=LearningFilter(), relevance_goal=None, limit=1
            )[0]
            candidate_id = cache.candidates(state.frontier)[0][0].id
    finally:
        board.close()
    stderr_path = finished_run.config_path.parent / "viewer-stderr.txt"
    with stderr_path.open("w") as stderr_log:
        env = {**os.environ, "PYTHONPATH": str(Path("tests").resolve())}
        proc = subprocess.Popen(
            [
                str(Path(sys.executable).with_name("swarm-arena")),
                "board",
                "serve",
                finished_run.namespace,
                "--config",
                str(finished_run.config_path),
                "--port",
                "0",
            ],
            stdout=subprocess.PIPE,
            stderr=stderr_log,
            text=True,
            env=env,
        )
        try:
            assert proc.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                # Includes interpreter imports, config loading and initial frontier publication.
                # stdout remains buffered so this also verifies the product flushes readiness.
                assert selector.select(timeout=60), (
                    f"viewer startup timed out (exit={proc.poll()}): {stderr_path.read_text()}"
                )
            line = proc.stdout.readline()
            assert line.startswith("Serving http://127.0.0.1:"), (line, stderr_path.read_text())
            url = line.split()[-1]
            for path in ("", f"learning/{learning_id}", f"candidate/{candidate_id}"):
                with urlopen(url + path, timeout=10) as response:
                    assert response.status == 200
            proc.send_signal(signal.SIGINT)
            proc.communicate(timeout=10)
            assert proc.returncode == 0, stderr_path.read_text()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            if proc.stdout is not None:
                proc.stdout.close()


def test_reindex_lock_diagnostic_does_not_expose_configured_path(
    finished_run: FinishedRun, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from swarm_arena._board.arena import ArenaBoard
    from swarm_arena._model.canonical import sha256_hex

    # The normal CLI cache is held by an ordinary board attachment.
    board = ArenaBoard(arena_root=finished_run.config.arena_root)
    try:
        board.open("", "", [finished_run.namespace])
        board.frontier()
        assert _call(finished_run, "reindex", "--namespace", finished_run.namespace) == 1
        output = capsys.readouterr()
        assert "in use" in output.err
        assert str(tmp_path) not in output.err
        assert sha256_hex(finished_run.config.arena_root.encode())[:12] not in output.err
        assert str(board._cache_dir()) not in output.err
    finally:
        board.close()
