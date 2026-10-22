"""Repeated terminal interrupts retain the writer until session cleanup ends."""

import json
import os
import pty
import select
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path


def test_double_interrupt_at_prompt_finishes_session(tmp_path: Path) -> None:
    cleanup, release, received, result = (
        tmp_path / name for name in ("cleanup", "release", "received", "result")
    )
    code = f"""
import json, os, signal, termios
from pathlib import Path
import anyio
from _fakes.planning_llm import planner_config, planner_output
from _fakes.run import all_events
from swarm_arena import Goal
from swarm_arena._budget.tranche import Tranche
from swarm_arena._cli import run_cmds
from swarm_arena._cli.main import main
from swarm_arena.llm import CollaborativePlanner, LLMPlanner
from swarm_arena.llm._terminal_reviewer import TerminalReviewer
from swarm_arena.testing import FakeModel, FakeTurn

config = planner_config(Path.cwd())
FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register('planner')
before = termios.tcgetattr(0)
handler = signal.getsignal(signal.SIGINT)
original_close = Tranche.close
async def close(self):
    Path({str(cleanup)!r}).write_text('ready')
    saved = signal.getsignal(signal.SIGINT)
    def observe(*args):
        Path({str(received)!r}).write_text('signal')
        saved(*args)
    signal.signal(signal.SIGINT, observe)
    try:
        with anyio.fail_after(10):
            while not Path({str(release)!r}).exists():
                await anyio.sleep(0.01)
    finally:
        signal.signal(signal.SIGINT, saved)
    return await original_close(self)
Tranche.close = close
async def operation(args):
    await CollaborativePlanner(LLMPlanner('planner'), TerminalReviewer()).plan(
        Goal(id='g', namespace='n'), config, None)
run_cmds._run_async = operation
status = main(['run', '--yes', 'unused', 'unused'])
after = termios.tcgetattr(0)
Path({str(result)!r}).write_text(json.dumps({{
    'tty_restored': before == after,
    'blocking_restored': os.get_blocking(0),
    'handler_restored': signal.getsignal(signal.SIGINT) is handler,
    'finished': [e.payload.get('reason') for e in all_events(config)
                 if e.type == 'goal_finished'],
}}))
raise SystemExit(status)
"""
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execve(
            sys.executable,
            [sys.executable, "-B", "-c", code],
            {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
    output = bytearray()
    reaped = False

    def pump() -> None:
        if select.select([master], [], [], 0.02)[0]:
            with suppress(OSError):
                output.extend(os.read(master, 65536))

    def wait_marker(path: Path) -> None:
        deadline = time.monotonic() + 20
        while not path.exists():
            assert time.monotonic() < deadline, bytes(output)
            pump()

    try:
        deadline = time.monotonic() + 20
        while b"Approve [a], feedback [f], edit [e]: " not in output:
            assert time.monotonic() < deadline, bytes(output)
            pump()
        os.write(master, b"\x03")
        wait_marker(cleanup)
        os.write(master, b"\x03")
        wait_marker(received)
        release.write_text("finish")
        deadline = time.monotonic() + 15
        while True:
            exited, status = os.waitpid(pid, os.WNOHANG)
            if exited:
                reaped = True
                break
            assert time.monotonic() < deadline, bytes(output)
            pump()
        assert os.waitstatus_to_exitcode(status) == 130, bytes(output)
        assert json.loads(result.read_text()) == {
            "tty_restored": True,
            "blocking_restored": True,
            "handler_restored": True,
            "finished": ["stopped"],
        }
        assert b"Traceback" not in output
    finally:
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        if not reaped:
            os.waitpid(pid, 0)
        os.close(master)
