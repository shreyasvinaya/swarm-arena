"""Public planning CLI admission, optional imports, and publication boundaries."""

import builtins
import io
import json
import os
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from _fakes.plan_cli import PlanningFiles, call, planning_files
from _fakes.planning_llm import planner_output
from _fakes.run import all_events, plan

from swarm_arena import Plan, Ref
from swarm_arena._cli.main import main
from swarm_arena._errors import _core_error, _stop_cleanup_error

SECRET = "plan-secret-sentinel-value"


@pytest.fixture
def files(tmp_path: Path) -> PlanningFiles:
    return planning_files(tmp_path)


def fake_planner(output: dict[str, Any] | None = None, *, turns: int = 1) -> Any:
    pytest.importorskip("agents")
    from swarm_arena.testing import FakeModel, FakeTurn

    fake = FakeModel(
        [
            FakeTurn(text=json.dumps(planner_output() if output is None else output))
            for _ in range(turns)
        ]
    )
    fake.register("planner")
    return fake


def test_manual_plan_prints_estimate(
    files: PlanningFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    path = files.goal.parent / "manual.toml"
    path.write_text(plan(rounds=1).dumps())
    assert call("plan", files.run, path) == 0
    assert capsys.readouterr().out.splitlines()[0] == "worst-case estimate: 0 tokens, unlimited"
    assert not Path(files.config.run_root).exists()


def test_manual_plan_invalid_exits_1_with_every_error(
    files: PlanningFiles,
    capsys: pytest.CaptureFixture[str],
) -> None:
    draft = plan(rounds=1, agents=2)
    for i, agent in enumerate(draft.teams[0].agents):
        draft.teams[0].agents[i] = agent.model_copy(update={"replicates": 1000})
    files.goal.write_text(draft.dumps())
    assert call("plan", files.run, files.goal) == 1
    err = capsys.readouterr().err
    assert "/teams/0/agents/0" in err
    assert "/teams/0/agents/1" in err


def test_manual_plan_never_imports_a_rejected_ref(
    files: PlanningFiles,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = files.goal.parent / "imported"
    monkeypatch.setenv("SWARM_ARENA_TEST_CANARY", str(canary))
    draft = plan(rounds=1, tool_ref=Ref(path="_fakes.import_canary:Tool"))
    files.goal.write_text(draft.model_copy(update={"origin": "llm"}).dumps())
    assert call("plan", files.run, files.goal) == 1
    assert "not in the planner catalog" in capsys.readouterr().err
    assert not canary.exists()


def test_help_and_manual_plan_load_no_optional_dependency(
    files: PlanningFiles, tmp_path: Path
) -> None:
    files.goal.write_text(plan(rounds=1).dumps())
    code = f"""
import sys
import swarm_arena, swarm_arena.testing
from swarm_arena._cli.main import main
for args in [['--help'], ['plan', '--help']]:
    try: main(args)
    except SystemExit as e: assert e.code == 0
assert main(['plan', {str(files.run)!r}, {str(files.goal)!r}]) == 0
print(sorted(m for m in ('agents', 'litellm', 'langfuse') if m in sys.modules))
"""
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("[]")


@pytest.mark.parametrize("mode", ["llm", "together"])
def test_plan_llm_writes_origin_llm(
    files: PlanningFiles,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("agents")
    fake = fake_planner()
    monkeypatch.setattr(sys, "stdin", io.StringIO("a\n"))
    assert (
        call(
            "plan",
            f"--{mode}",
            "--model",
            "planner",
            "--instructions",
            "keep it small",
            files.run,
            files.goal,
        )
        == 0
    )
    out = files.output(mode)
    loaded = Plan.load(out)
    assert loaded.planning_session is not None
    assert loaded.origin == mode
    assert loaded.base_dir == files.goal.parent
    assert loaded.goal.files.manifest.keys() == {"input.txt"}
    assert out.read_text().startswith(
        f"# swarm-arena planning session: {loaded.planning_session}\n"
    )
    printed = capsys.readouterr().out
    assert str(out) in printed
    assert loaded.planning_session in printed
    if mode == "together":
        assert "Approve [a]" in printed
    assert "keep it small" in fake.calls[0]["system"]
    events = all_events(files.config)
    assert len([e for e in events if e.type == "run_started"]) == 1
    assert len([e for e in events if e.type == "goal_finished"]) == 1


def test_refuses_existing_output_without_force(
    files: PlanningFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("agents")
    fake = fake_planner()
    files.output().write_text(SECRET)
    args = ("plan", "--llm", "--model", "planner", files.run, files.goal)
    assert call(*args) == 1
    assert files.output().read_text() == SECRET
    assert not fake.calls
    assert "--force" in capsys.readouterr().err
    assert not Path(files.config.run_root).exists()
    assert call(*args, "--force") == 0
    assert Plan.load(files.output()).origin == "llm"


def test_output_outside_goal_dir_is_a_usage_error(
    files: PlanningFiles,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        call(
            "plan",
            "--llm",
            "--model",
            "planner",
            "-o",
            tmp_path / "bad.toml",
            files.run,
            files.goal,
        )
        == 2
    )
    assert "goal file's directory" in capsys.readouterr().err
    assert not Path(files.config.run_root).exists()


@pytest.mark.parametrize("target", ["directory", "dangling", "symlink"])
@pytest.mark.parametrize("force", [False, True])
def test_output_target_errors_are_safe(
    files: PlanningFiles,
    target: str,
    force: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("agents")
    from swarm_arena._cli import plan_cmd

    fake = fake_planner()
    out = files.output()
    if target == "directory":
        out.mkdir()
    else:
        destination = out.parent / SECRET
        if target == "symlink":
            destination.write_text(SECRET)
        out.symlink_to(destination)

    def unexpected_runner(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid output entered the async runner")

    monkeypatch.setattr(plan_cmd.anyio, "run", unexpected_runner)
    flags = ["--force"] if force else []
    assert call("plan", "--llm", "--model", "planner", *flags, files.run, files.goal) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.err
    assert "planning session:" not in captured.out
    assert not fake.calls
    assert not Path(files.config.run_root).exists()


def test_missing_extra_has_safe_install_hint(
    files: PlanningFiles,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = builtins.__import__

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("swarm_arena.llm"):
            raise ModuleNotFoundError(SECRET, name="agents")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    assert call("plan", "--llm", "--model", "planner", files.run, files.goal) == 1
    err = capsys.readouterr().err
    assert "swarm-arena[llm]" in err
    assert SECRET not in err
    assert not files.output().exists()
    assert not Path(files.config.run_root).exists()


def test_unknown_alias_writes_nothing(
    files: PlanningFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("agents")
    fake_planner()
    assert call("plan", "--llm", "--model", "missing", files.run, files.goal) == 1
    assert "missing" in capsys.readouterr().err
    assert not files.output().exists()
    assert not Path(files.config.run_root).exists()


def test_invalid_generation_is_not_saved(
    files: PlanningFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("agents")
    output = planner_output()
    output["teams"][0]["agents"][0]["tool"]["settings"] = {"function": SECRET}
    fake_planner(output)
    assert call("plan", "--llm", "--model", "planner", files.run, files.goal) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.err
    assert "planning session:" not in captured.out
    assert not files.output().exists()


def test_together_eof_is_not_saved(
    files: PlanningFiles,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("agents")
    fake_planner()
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert call("plan", "--together", "--model", "planner", files.run, files.goal) == 1
    assert "terminal input ended" in capsys.readouterr().err
    assert not files.output("together").exists()
    assert any(e.type == "goal_finished" for e in all_events(files.config))


def test_invalid_edited_goal_is_rejected_before_approval(
    files: PlanningFiles,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("agents")
    fake_planner()
    editor = files.goal.parent / "editor"
    editor.write_text(
        f"#!{sys.executable}\nimport pathlib, sys\np=pathlib.Path(sys.argv[1])\n"
        f"p.write_text(p.read_text().replace('title = \"\"', 'title = \"{SECRET}\"'))\n"
    )
    editor.chmod(0o700)
    monkeypatch.setenv("VISUAL", str(editor))
    monkeypatch.setattr(sys, "stdin", io.StringIO("e\na\n"))
    assert call("plan", "--together", "--model", "planner", files.run, files.goal) == 0
    captured = capsys.readouterr()
    assert "edited plan admission refused" in captured.out
    assert SECRET not in captured.out + captured.err
    assert Plan.load(files.output("together")).goal.title == ""


@pytest.mark.parametrize("stage", ["initialization", "cleanup"])
def test_failed_session_never_publishes(
    files: PlanningFiles,
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("agents")
    fake_planner()
    from swarm_arena._cli import plan_cmd

    original = plan_cmd.planning_session

    @asynccontextmanager
    async def fail(*args: Any) -> AsyncGenerator[Any]:
        if stage == "initialization":
            raise _core_error(RuntimeError, "planning session stopped")
        async with original(*args) as scope:
            yield scope
        raise _stop_cleanup_error()

    monkeypatch.setattr(plan_cmd, "planning_session", fail)
    assert call("plan", "--llm", "--model", "planner", files.run, files.goal) == 1
    captured = capsys.readouterr()
    assert (
        "planning session stopped" if stage == "initialization" else "stop cleanup timed out"
    ) in captured.err
    assert "planning session:" not in captured.out
    assert not files.output().exists()


def test_success_inside_synchronous_except(files: PlanningFiles) -> None:
    pytest.importorskip("agents")
    fake_planner()

    def invoke() -> int:
        try:
            raise ValueError(SECRET)
        except ValueError:
            return main(["plan", "--llm", "--model", "planner", str(files.run), str(files.goal)])

    from concurrent.futures import ThreadPoolExecutor
    from contextvars import Context

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(Context().run, invoke).result() == 0
    assert files.output().exists()


@pytest.mark.parametrize("interrupts", [2, 3])
def test_together_prompt_repeated_sigint_finishes_owned_cleanup(
    tmp_path: Path, interrupts: int
) -> None:
    pytest.importorskip("agents")
    import pty
    import select
    import signal
    import time
    from contextlib import suppress

    release_read, release_write = os.pipe()
    os.set_inheritable(release_read, True)
    result = tmp_path / "result.json"
    code = f"""
import json, os, signal, termios
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
import anyio
from _fakes.plan_cli import planning_files
from _fakes.planning_llm import planner_output
from _fakes.run import all_events
from swarm_arena._cli import plan_cmd, run_cmds
from swarm_arena._cli.main import main
from swarm_arena.testing import FakeModel, FakeTurn
files = planning_files(Path.cwd())
FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register('planner')
wakeup = run_cmds._sigint_wakeup
@contextmanager
def observed_wakeup(on_interrupt=None):
    count = 0
    def observed():
        nonlocal count
        count += 1
        try:
            on_interrupt()
        except KeyboardInterrupt:
            if count == 3:
                print('received-third-force', flush=True)
            raise
        if count == 2:
            print('received-second', flush=True)
    with wakeup(observed if on_interrupt is not None else None):
        yield
run_cmds._sigint_wakeup = observed_wakeup
original = plan_cmd.planning_session
async def hold():
    with anyio.CancelScope(shield=True):
        print('cleanup-ready', flush=True)
        await anyio.wait_readable({release_read})
        os.read({release_read}, 1)
@asynccontextmanager
async def held_cleanup(*args):
    try:
        async with original(*args) as scope:
            try:
                yield scope
            finally:
                if {interrupts} == 2:
                    await hold()
    finally:
        # Exercise force-exit after real journal cleanup, without cancelling its tasks.
        if {interrupts} == 3:
            await hold()
plan_cmd.planning_session = held_cleanup
handler = signal.getsignal(signal.SIGINT)
before = termios.tcgetattr(0)
status = main(['plan', '--together', '--model', 'planner', str(files.run), str(files.goal)])
Path({str(result)!r}).write_text(json.dumps({{
    'status': status,
    'handler_restored': signal.getsignal(signal.SIGINT) is handler,
    'tty_restored': termios.tcgetattr(0) == before,
    'blocking_restored': os.get_blocking(0),
    'saved': files.output('together').exists(),
    'finished': [e.payload.get('reason') for e in all_events(files.config)
                 if e.type == 'goal_finished'],
}}))
"""
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", f"import os; os.login_tty({slave})\n" + code],
        cwd=tmp_path,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        pass_fds=(slave, release_read),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    pid = process.pid
    os.close(slave)
    os.close(release_read)
    output = bytearray()

    def until(marker: bytes) -> None:
        deadline = time.monotonic() + 60
        while marker not in output:
            assert time.monotonic() < deadline, bytes(output)
            if select.select([master], [], [], 0.05)[0]:
                try:
                    output.extend(os.read(master, 65536))
                except OSError:
                    pytest.fail(bytes(output).decode(errors="replace"))

    try:
        until(b"Approve [a]")
        os.write(master, b"\x03")
        until(b"cleanup-ready")
        os.write(master, b"\x03")
        until(b"received-second")
        if interrupts == 3:
            os.write(master, b"\x03")
            until(b"received-third-force")
        os.write(release_write, b"x")
        deadline = time.monotonic() + 30
        while True:
            status = process.poll()
            if status is not None:
                break
            assert time.monotonic() < deadline, bytes(output)
            if select.select([master], [], [], 0.05)[0]:
                with suppress(OSError):
                    output.extend(os.read(master, 65536))
        assert status == 0, bytes(output)
        assert json.loads(result.read_text()) == {
            "status": 130,
            "handler_restored": True,
            "tty_restored": True,
            "blocking_restored": True,
            "saved": False,
            "finished": ["stopped"],
        }, bytes(output)
        assert b"Traceback" not in output
        assert b"never retrieved" not in output
    finally:
        # Reap an exited child before signalling its process group (macOS zombies).
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0)
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        os.close(master)
        os.close(release_write)


@pytest.mark.parametrize("extra", [["--llm"], ["--model", "planner"], ["--force"]])
def test_invalid_mode_options_are_usage_errors(files: PlanningFiles, extra: list[str]) -> None:
    assert call("plan", *extra, files.run, files.goal) == 2
    assert not Path(files.config.run_root).exists()


def test_manual_estimate_prints_capped_units_and_all_warnings(
    files: PlanningFiles,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from swarm_arena._cli import plan_cmd
    from swarm_arena.tool import Estimate

    config = files.config.model_copy(
        update={
            "limits": files.config.limits.model_copy(
                update={
                    "max_usd": 10.0,
                    "default_run_reservation": Estimate(tokens=100, usd=0.123),
                }
            )
        }
    )
    files.run.write_text(config.dumps())
    files.goal.write_text(plan(rounds=1).dumps())
    original = plan_cmd.admit_revision

    def warned(*args: Any, **kwargs: Any) -> tuple[Plan, list[str]]:
        admitted, warnings = original(*args, **kwargs)
        return admitted, [*warnings, "first warning", "second warning"]

    monkeypatch.setattr(plan_cmd, "admit_revision", warned)
    assert call("plan", files.run, files.goal) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("worst-case estimate: ")
    assert "unlimited" not in lines[0]
    assert ", $" in lines[0]
    assert lines[-2:] == ["warning: first warning", "warning: second warning"]


def test_custom_output_and_publication_failure_are_safe(
    files: PlanningFiles,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("agents")
    from swarm_arena._cli import plan_cmd

    fake_planner(turns=2)
    out = files.goal.parent / "chosen.toml"
    assert call("plan", "--llm", "--model", "planner", "-o", out, files.run, files.goal) == 0
    before = out.read_bytes()
    capsys.readouterr()

    replace = plan_cmd.os.replace

    def fail(source: Any, destination: Any, **kwargs: Any) -> None:
        if destination == out:
            raise OSError(SECRET)
        replace(source, destination, **kwargs)

    monkeypatch.setattr(plan_cmd.os, "replace", fail)
    assert (
        call("plan", "--llm", "--model", "planner", "-o", out, "--force", files.run, files.goal)
        == 1
    )
    captured = capsys.readouterr()
    assert SECRET not in captured.err
    assert "plan output failed: OSError" in captured.err
    assert "plan written:" not in captured.out
    assert out.read_bytes() == before
    assert not list(out.parent.glob(".swarm-plan-*"))


@pytest.mark.parametrize(
    "text",
    ['[goal]\nid=42\nnamespace="n"', '[goal]\nid="g"\nnamespace="n"\n[unexpected]\nvalue="SECRET"'],
)
def test_invalid_goal_input_is_safe_and_writes_nothing(
    files: PlanningFiles,
    text: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("agents")
    fake_planner()
    files.goal.write_text(text.replace("SECRET", SECRET))
    assert call("plan", "--llm", "--model", "planner", files.run, files.goal) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.err
    assert captured.err
    assert not Path(files.config.run_root).exists()
    assert not files.output().exists()


@pytest.mark.parametrize("mode", ["llm", "together"])
def test_symlinked_goal_uses_the_loaded_file_directory(
    files: PlanningFiles,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("agents")
    from hashlib import sha256

    fake_planner()
    directory = files.run.parent / "linked"
    directory.mkdir()
    source = directory / "goal.toml"
    source.symlink_to(files.goal)
    (directory / "input.txt").write_bytes(b"linked directory input")
    monkeypatch.setattr(sys, "stdin", io.StringIO("a\n"))
    assert call("plan", f"--{mode}", "--model", "planner", files.run, source) == 0
    saved = directory / f"plan.{mode}.toml"
    assert saved.exists()
    assert Plan.load(saved).goal.files.manifest["input.txt"] == (
        sha256(b"linked directory input").hexdigest()
    )


def test_missing_extra_hint_precedes_optional_catalog_loading(
    files: PlanningFiles,
    tmp_path: Path,
) -> None:
    with files.run.open("a") as stream:
        stream.write(
            '\n[[planner_catalog.entries]]\nname="sdk"\nkind="manager"\n'
            'ref="swarm_arena.llm:AgentsSDKManager"\ndescription="Manager"\n'
            '[planner_catalog.entries.config]\nmodel="opus"\n'
            '[planner_catalog.entries.settable.instructions]\ntype="string"\n'
        )
    code = f"""
import importlib.abc, sys
class Missing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'swarm_arena.llm':
            raise ModuleNotFoundError({SECRET!r}, name='agents')
sys.meta_path.insert(0, Missing())
from swarm_arena._cli.main import main
assert main(['plan', '--llm', '--model', 'planner',
             {str(files.run)!r}, {str(files.goal)!r}]) == 1
"""
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "swarm-arena[llm]" in proc.stderr
    assert SECRET not in proc.stderr
    assert not Path(files.config.run_root).exists()


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("stage", ["start", "sync"])
def test_output_interrupt_abandons_publication(tmp_path: Path, force: bool, stage: str) -> None:
    pytest.importorskip("agents")
    code = f"""
import json, os, signal, threading
from pathlib import Path
from _fakes.plan_cli import planning_files
from _fakes.planning_llm import planner_output
from swarm_arena._cli import plan_cmd
from swarm_arena._cli.main import main
from swarm_arena.testing import FakeModel, FakeTurn
files = planning_files(Path.cwd())
FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register('planner')
if {force!r}:
    files.output().write_text('original')
started, release, finished = threading.Event(), threading.Event(), threading.Event()
write = plan_cmd._write_plan
fsync = os.fsync
local = threading.local()
def hold():
    started.set()
    assert release.wait(20)
def sync(fd):
    if getattr(local, 'writing', False) and {stage!r} == 'sync':
        hold()
    return fsync(fd)
plan_cmd.os.fsync = sync
def held(*args):
    try:
        local.writing = True
        if {stage!r} == 'start':
            hold()
        return write(*args)
    finally:
        finished.set()
plan_cmd._write_plan = held
def interrupt():
    assert started.wait(20)
    os.kill(os.getpid(), signal.SIGINT)
worker = threading.Thread(target=interrupt)
worker.start()
try:
    status = main(['plan', '--llm', '--model', 'planner',
                   *(['--force'] if {force!r} else []), str(files.run), str(files.goal)])
finally:
    release.set()
    worker.join(20)
assert not worker.is_alive()
assert finished.wait(20)
assert status == 130, status
assert files.output().read_text() == 'original' if {force!r} else not files.output().exists()
assert not list(files.goal.parent.glob('.swarm-plan-*'))
"""
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "interrupted" in proc.stderr
    assert "plan written:" not in proc.stdout


@pytest.mark.parametrize("force", [False, True])
def test_output_preserves_permissions(tmp_path: Path, force: bool) -> None:
    from swarm_arena._cli.plan_cmd import _write_plan

    output = tmp_path / "plan.toml"
    if force:
        output.write_text("old")
        output.chmod(0o604)
    previous = os.umask(0o027)
    try:
        _write_plan(output, "new", force)
    finally:
        os.umask(previous)
    assert output.read_text() == "new"
    assert output.stat().st_mode & 0o777 == (0o604 if force else 0o640)
    assert not list(tmp_path.glob(".swarm-plan-*"))


@pytest.mark.parametrize("force", [False, True])
def test_output_is_synced_before_publication(
    tmp_path: Path, force: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stat

    from swarm_arena._cli import plan_cmd

    output = tmp_path / "plan.toml"
    if force:
        output.write_text("old")
    synced: set[str] = set()
    fsync = os.fsync
    publish = os.replace if force else os.link

    def sync(fd: int) -> None:
        synced.add("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        fsync(fd)

    def observed(source: Any, destination: Any) -> None:
        assert synced == {"file", "directory"}
        assert Path(source).read_text() == "new"
        publish(source, destination)

    monkeypatch.setattr(plan_cmd.os, "fsync", sync)
    monkeypatch.setattr(plan_cmd.os, "replace" if force else "link", observed)
    plan_cmd._write_plan(output, "new", force)
    assert output.read_text() == "new"


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("stage", ["file", "directory"])
def test_output_sync_failure_preserves_destination(
    tmp_path: Path, force: bool, stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stat

    from swarm_arena._cli import plan_cmd

    output = tmp_path / "plan.toml"
    if force:
        output.write_text("old")
    fsync = os.fsync

    def fail(fd: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        if kind == stage:
            raise OSError("sync failed")
        fsync(fd)

    monkeypatch.setattr(plan_cmd.os, "fsync", fail)
    with pytest.raises(OSError, match="sync failed"):
        plan_cmd._write_plan(output, "new", force)
    assert output.read_text() == "old" if force else not output.exists()
    assert not list(tmp_path.glob(".swarm-plan-*"))
