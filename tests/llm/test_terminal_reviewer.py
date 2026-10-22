import asyncio
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType

import anyio
import pytest
from _fakes.planning_llm import planner_config, planner_output
from _fakes.run import all_events

from swarm_arena import Goal, Plan
from swarm_arena.llm import Approve, CollaborativePlanner, Edited, Feedback, LLMPlanner
from swarm_arena.llm._terminal_reviewer import TerminalReviewer
from swarm_arena.testing import FakeModel, FakeTurn

PLAN = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])


async def test_approve() -> None:
    out = io.StringIO()
    assert await TerminalReviewer(stdin=io.StringIO("a\n"), stdout=out).review(PLAN) == Approve()
    assert PLAN.dumps() in out.getvalue()
    assert "Approve [a], feedback [f], edit [e]: " in out.getvalue()


async def test_feedback() -> None:
    assert await TerminalReviewer(
        stdin=io.StringIO("f\nmore teams\n"), stdout=io.StringIO()
    ).review(PLAN) == Feedback("more teams")


@pytest.mark.parametrize("text", ["", "f\n"])
async def test_eof(text: str) -> None:
    with pytest.raises(RuntimeError, match="terminal input ended"):
        await TerminalReviewer(stdin=io.StringIO(text), stdout=io.StringIO()).review(PLAN)


def _editor(tmp_path: Path, body: str) -> str:
    script = tmp_path / "editor"
    script.write_text("#!/usr/bin/env python3\n" + body)
    script.chmod(0o700)
    return str(script)


async def test_edit_runs_the_editor(tmp_path: Path) -> None:
    editor = _editor(
        tmp_path,
        "import pathlib, sys\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "p.write_text(p.read_text().replace('rationale = \"\"', 'rationale = \"revised\"'))\n",
    )
    plan = PLAN.model_copy()
    plan._base_dir = tmp_path
    result = await TerminalReviewer(
        stdin=io.StringIO("e\n"), stdout=io.StringIO(), editor=editor
    ).review(plan)
    assert isinstance(result, Edited)
    assert result.plan.rationale == "revised"
    assert result.plan.base_dir == tmp_path


async def test_bad_edit_asks_again(tmp_path: Path) -> None:
    editor = _editor(
        tmp_path,
        "import pathlib, sys\n"
        'pathlib.Path(sys.argv[1]).write_text("secret-sentinel-editor-invalid")\n',
    )
    out = io.StringIO()
    assert (
        await TerminalReviewer(stdin=io.StringIO("e\na\n"), stdout=out, editor=editor).review(PLAN)
        == Approve()
    )
    assert "invalid TOML" in out.getvalue()
    assert "secret-sentinel-editor-invalid" not in out.getvalue()


async def test_editor_failure_asks_again(tmp_path: Path) -> None:
    editor = _editor(
        tmp_path, 'import sys\nsys.stderr.write("secret-sentinel-editor-failure")\nsys.exit(7)\n'
    )
    out = io.StringIO()
    assert (
        await TerminalReviewer(stdin=io.StringIO("e\na\n"), stdout=out, editor=editor).review(PLAN)
        == Approve()
    )
    assert "failed" in out.getvalue()
    assert "secret-sentinel" not in out.getvalue()


async def test_cleanup_failure_cannot_return_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena.llm import _terminal_reviewer

    class BrokenCleanup(TemporaryDirectory):
        def __exit__(
            self,
            exc: type[BaseException] | None,
            value: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            super().__exit__(exc, value, tb)
            raise OSError("secret-sentinel-cleanup")

    monkeypatch.setattr(_terminal_reviewer, "TemporaryDirectory", BrokenCleanup)
    editor = _editor(tmp_path, "pass\n")
    with pytest.raises(RuntimeError, match="terminal review failed: OSError") as caught:
        await TerminalReviewer(
            stdin=io.StringIO("e\n"), stdout=io.StringIO(), editor=editor
        ).review(PLAN)
    assert_secret_absent(caught.value, "secret-sentinel-cleanup")


async def test_terminal_cancellation_restores_descriptor() -> None:
    reader, writer = os.pipe()
    try:
        with os.fdopen(reader) as stdin:
            with anyio.move_on_after(0.05) as scope:
                await TerminalReviewer(stdin=stdin, stdout=io.StringIO()).review(PLAN)
            assert scope.cancel_called
            assert os.get_blocking(reader)
            os.write(writer, b"a\n")
            assert (
                await TerminalReviewer(stdin=stdin, stdout=io.StringIO()).review(PLAN) == Approve()
            )
    finally:
        os.close(writer)


async def test_shared_cli_prompt_keeps_ascii_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from swarm_arena._cli import run_cmds

    reader, writer = os.pipe()
    try:
        with os.fdopen(reader) as stdin:
            monkeypatch.setattr(sys, "stdin", stdin)
            os.write(writer, "\N{NO-BREAK SPACE}y\n".encode())
            assert await run_cmds._prompt() is False
    finally:
        os.close(writer)


@pytest.mark.parametrize("collaborative", [False, True])
async def test_editor_cancellation_reaps_process(tmp_path: Path, collaborative: bool) -> None:
    marker = tmp_path / "pid"
    editor = _editor(
        tmp_path,
        "import os, pathlib, signal\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\nsignal.pause()\n",
    )
    reviewer = TerminalReviewer(stdin=io.StringIO("e\n"), stdout=io.StringIO(), editor=editor)
    config = planner_config(tmp_path)
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    operation = (
        CollaborativePlanner(LLMPlanner("planner"), reviewer).plan(PLAN.goal, config, None)
        if collaborative
        else reviewer.review(PLAN)
    )
    task = asyncio.create_task(operation)
    try:
        with anyio.fail_after(10):
            while not marker.exists() or not marker.read_text():
                await anyio.lowlevel.checkpoint()
            pid = int(marker.read_text())
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        if collaborative:
            from swarm_arena._journal import llm_record

            events = all_events(config)
            assert len([e for e in events if e.type == "goal_finished"]) == 1
            sessions = await anyio.to_thread.run_sync(lambda: list(Path(config.run_root).iterdir()))
            assert not any(p.name in llm_record._bindings for p in sessions)
            assert all_events(config) == events
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
