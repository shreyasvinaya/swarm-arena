import asyncio
import io
import json
import os
import pty
import select
import signal
import sys
import termios
import time
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Any

import anyio
import pytest
from _fakes.planning_llm import planner_config, planner_output
from _fakes.run import all_events

from swarm_arena import Goal, Plan
from swarm_arena.llm import Approve, CollaborativePlanner, Edited, Feedback, LLMPlanner
from swarm_arena.llm._terminal_reviewer import TerminalReviewer
from swarm_arena.testing import FakeModel, FakeTurn

PLAN = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])


@pytest.mark.parametrize("editor_kind", ["slow", "slow_interrupt", "stubborn"])
def test_repeated_interrupt_reaps_editor_and_finishes_session(
    tmp_path: Path, editor_kind: str
) -> None:
    marker = tmp_path / "editor-pid"
    result = tmp_path / "result.json"
    editor = _editor(
        tmp_path,
        "import os, pathlib, signal, time\n"
        "def slow_exit(*args):\n    time.sleep(0.8); os._exit(0)\n"
        + (
            "signal.signal(signal.SIGINT, slow_exit)\n"
            if editor_kind == "slow_interrupt"
            else "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        )
        + (
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            if editor_kind == "stubborn"
            else "signal.signal(signal.SIGTERM, slow_exit)\n"
        )
        + f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n",
    )
    code = f"""
import asyncio, io, json, os, signal, termios
from pathlib import Path
from _fakes.planning_llm import planner_config, planner_output
from _fakes.run import all_events
from swarm_arena import Goal
from swarm_arena.llm import CollaborativePlanner, LLMPlanner
from swarm_arena.llm._terminal_reviewer import TerminalReviewer
from swarm_arena.testing import FakeModel, FakeTurn
config = planner_config(Path.cwd())
FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register('planner')
before = termios.tcgetattr(0)[3]
handler = signal.getsignal(signal.SIGINT)
async def main():
    reviewer = TerminalReviewer(stdin=io.StringIO('e\\n'), editor={editor!r})
    await CollaborativePlanner(LLMPlanner('planner'), reviewer).plan(
        Goal(id='g', namespace='n'), config, None)
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
else:
    raise AssertionError('interruption expected')
after = termios.tcgetattr(0)[3]
Path({str(result)!r}).write_text(json.dumps({{
    'tty_restored': (before & (termios.ECHO | termios.ICANON)) ==
                    (after & (termios.ECHO | termios.ICANON)),
    'handler_restored': signal.getsignal(signal.SIGINT) is handler,
    'blocking_restored': os.get_blocking(0),
    'finished': [e.payload.get('reason') for e in all_events(config)
                 if e.type == 'goal_finished'],
}}))
"""
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execve(
            sys.executable,
            [sys.executable, "-B", "-c", code],
            {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
    editor_pid = None
    reaped = False
    output = bytearray()

    def pump() -> None:
        if select.select([master], [], [], 0.02)[0]:
            with suppress(OSError):
                output.extend(os.read(master, 65536))

    try:
        deadline = time.monotonic() + 20
        while not marker.exists() or not marker.read_text():
            assert time.monotonic() < deadline, bytes(output)
            pump()
        editor_pid = int(marker.read_text())
        os.write(master, b"\x03")
        # Position the second terminal interrupt inside the one-second grace.
        time.sleep(0.3)
        os.write(master, b"\x03")
        deadline = time.monotonic() + 10
        while True:
            exited, status = os.waitpid(pid, os.WNOHANG)
            if exited:
                reaped = True
                break
            assert time.monotonic() < deadline, bytes(output)
            pump()
        assert os.waitstatus_to_exitcode(status) == 0, bytes(output)
        data = json.loads(result.read_text())
        assert data == {
            "tty_restored": True,
            "handler_restored": True,
            "blocking_restored": True,
            "finished": ["stopped"],
        }
        with pytest.raises(ProcessLookupError):
            os.kill(editor_pid, 0)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        if not reaped:
            os.waitpid(pid, 0)
        if editor_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(editor_pid, signal.SIGKILL)
        os.close(master)


@pytest.mark.parametrize(
    "exit_path", ["success", "failure", "cancel", "stop", "deadline", "fence", "stubborn"]
)
async def test_editor_restores_real_tty_on_every_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_path: str
) -> None:
    from _fakes.leases import force_take_over

    from swarm_arena._engine import run as run_module
    from swarm_arena._engine.stops import RunClock
    from swarm_arena._errors import Fenced
    from swarm_arena._store.local import LocalStore

    marker, released, terminated = (tmp_path / name for name in ("ready", "release", "term"))
    editor = _editor(
        tmp_path,
        "import os, pathlib, signal, sys, time, tty\n"
        f"ready = pathlib.Path({str(marker)!r})\n"
        f"release = pathlib.Path({str(released)!r})\n"
        f"term = pathlib.Path({str(terminated)!r})\n"
        "def terminate(*args):\n    term.write_text('TERM')\n"
        + ("    return\n" if exit_path == "stubborn" else "    sys.exit(0)\n")
        + "signal.signal(signal.SIGTERM, terminate)\ntty.setraw(0)\n"
        "ready.write_text(str(os.getpid()))\n"
        "while not release.exists(): time.sleep(0.01)\n"
        + ("sys.exit(1)\n" if exit_path == "failure" else ""),
    )
    master, slave = pty.openpty()
    original_stdin = os.dup(0)
    before = termios.tcgetattr(slave)
    expired = anyio.Event()

    def left(clock: RunClock, limit_s: float) -> float:
        return 123456.0

    class ControlledClock:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        async def sleep(self, seconds: float) -> None:
            if seconds == 123456.0:
                await expired.wait()
            else:
                await anyio.sleep(seconds)

    if exit_path == "deadline":
        monkeypatch.setattr(RunClock, "left", left)
        monkeypatch.setattr(run_module, "anyio", ControlledClock())
    reviewer = TerminalReviewer(stdin=io.StringIO("e\na\n"), stdout=io.StringIO(), editor=editor)
    config = planner_config(tmp_path)
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    session = exit_path in {"stop", "deadline", "fence"}
    replacement = None
    task: asyncio.Task[Any] | None = None
    pid = None
    try:
        os.dup2(slave, 0)
        operation = (
            CollaborativePlanner(LLMPlanner("planner"), reviewer).plan(PLAN.goal, config, None)
            if session
            else reviewer.review(PLAN)
        )
        task = asyncio.create_task(operation)
        with anyio.fail_after(40):
            while not marker.exists() or not marker.read_text():
                assert not task.done()
                await anyio.lowlevel.checkpoint()
            pid = int(marker.read_text())
            assert not termios.tcgetattr(slave)[3] & termios.ICANON
            if exit_path in {"success", "failure"}:
                released.write_text("exit")
                await task
            elif exit_path in {"cancel", "stubborn"}:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                (directory,) = await anyio.to_thread.run_sync(
                    lambda: list(Path(config.run_root).iterdir())
                )
                if exit_path == "deadline":
                    expired.set()
                elif exit_path == "stop":
                    (directory / "control" / "stop-e1").write_text("")
                else:
                    replacement = force_take_over(
                        LocalStore(directory), owner_id="replacement", host="test", ttl_s=60
                    )
                with pytest.raises(Fenced if exit_path == "fence" else RuntimeError):
                    await task
            after = termios.tcgetattr(slave)
            # The kernel may set its pending-input bookkeeping bit on restore.
            after[3] &= ~getattr(termios, "PENDIN", 0)
            before[3] &= ~getattr(termios, "PENDIN", 0)
            assert after == before
            if exit_path not in {"success", "failure"}:
                assert terminated.read_text() == "TERM"
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if pid is not None:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        if replacement is not None:
            replacement.release()
        os.dup2(original_stdin, 0)
        os.close(original_stdin)
        os.close(master)
        os.close(slave)


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


@pytest.mark.parametrize("source", ["approve", "feedback", "empty", "feedback_eof", "devnull"])
async def test_descriptor_input_and_eof(tmp_path: Path, source: str) -> None:
    path = tmp_path / "answers"
    contents = {"approve": "a\n", "feedback": "f\nmore teams\n", "empty": "", "feedback_eof": "f\n"}
    if source == "devnull":
        path = Path(os.devnull)
    else:
        path.write_text(contents[source])
    with os.fdopen(os.open(path, os.O_RDONLY)) as stdin:
        reviewer = TerminalReviewer(stdin=stdin, stdout=io.StringIO())
        with anyio.fail_after(5):
            if source in {"empty", "feedback_eof", "devnull"}:
                with pytest.raises(RuntimeError, match="terminal input ended"):
                    await reviewer.review(PLAN)
            else:
                result = await reviewer.review(PLAN)
                assert result == (Approve() if source == "approve" else Feedback("more teams"))
        assert os.get_blocking(stdin.fileno())


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


@pytest.mark.parametrize("failure", ["toml", "encoding", "exit"])
async def test_edit_retries_keep_the_last_file(tmp_path: Path, failure: str) -> None:
    once = tmp_path / "once"
    broken = b"rationale = 'human work'\n[goal\n" if failure != "encoding" else b"human work\xff"
    repaired = PLAN.model_copy(update={"rationale": "human work"}).dumps().encode()
    editor = _editor(
        tmp_path,
        "import pathlib, sys\np = pathlib.Path(sys.argv[1])\n"
        f"once = pathlib.Path({str(once)!r})\n"
        "if not once.exists():\n    once.write_text('1')\n"
        f"    p.write_bytes({broken!r})\n"
        + ("    sys.exit(1)\n" if failure == "exit" else "")
        + "else:\n"
        f"    assert p.read_bytes() == {broken!r}\n    p.write_bytes({repaired!r})\n",
    )
    result = await TerminalReviewer(
        stdin=io.StringIO("e\ne\n"), stdout=io.StringIO(), editor=editor
    ).review(PLAN)
    assert isinstance(result, Edited)
    assert result.plan.rationale == "human work"


async def test_edited_file_error_names_the_human_layer(tmp_path: Path) -> None:
    secret = "secret-sentinel-encoding"
    invalid = secret.encode() + bytes([255])
    editor = _editor(
        tmp_path, "import pathlib, sys\n" + f"pathlib.Path(sys.argv[1]).write_bytes({invalid!r})\n"
    )
    out = io.StringIO()
    assert (
        await TerminalReviewer(stdin=io.StringIO("e\na\n"), stdout=out, editor=editor).review(PLAN)
        == Approve()
    )
    assert "edited plan could not be read: UnicodeDecodeError" in out.getvalue()
    assert "LLM preparation" not in out.getvalue()
    assert secret not in out.getvalue()


async def test_missing_editor_diagnostic_contains_no_path(tmp_path: Path) -> None:
    secret = "secret-sentinel-editor-path"
    out = io.StringIO()
    result = await TerminalReviewer(
        stdin=io.StringIO("e\na\n"), stdout=out, editor=str(tmp_path / secret)
    ).review(PLAN)
    assert result == Approve()
    assert "editor could not be started: FileNotFoundError" in out.getvalue()
    assert secret not in out.getvalue()


async def test_admission_refusal_keeps_edits_for_repair(tmp_path: Path) -> None:
    once = tmp_path / "once"
    editor = _editor(
        tmp_path,
        "import pathlib, sys\np = pathlib.Path(sys.argv[1])\nt = p.read_text()\n"
        f"once = pathlib.Path({str(once)!r})\n"
        "if not once.exists():\n    once.write_text('1')\n"
        "    t = t.replace('title = \"\"', 'title = \"changed\"')\n"
        "    t = t.replace('Explore the catalog', 'human work')\n"
        "else:\n    assert 'human work' in t\n    assert 'title = \"changed\"' in t\n"
        "    t = t.replace('title = \"changed\"', 'title = \"\"')\n"
        "p.write_text(t)\n",
    )
    fake = FakeModel([FakeTurn(text=json.dumps(planner_output()))])
    fake.register("planner")
    out = io.StringIO()
    reviewer = TerminalReviewer(stdin=io.StringIO("e\ne\n"), stdout=out, editor=editor)
    result = await CollaborativePlanner(LLMPlanner("planner"), reviewer).plan(
        PLAN.goal, planner_config(tmp_path), None
    )
    assert result.rationale == "human work"
    assert result.goal.title == ""
    assert len(fake.calls) == 1
    assert "edited plan admission refused" in out.getvalue()


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
