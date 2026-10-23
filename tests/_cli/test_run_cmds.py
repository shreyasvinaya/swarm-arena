import os
import re
import signal
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest
from _fakes.run import plan

from swarm_arena import Plan, RunConfig, Swarm
from swarm_arena._cli import run_cmds
from swarm_arena._cli.main import main
from swarm_arena._engine.solution import SolutionDocument
from swarm_arena._errors import ConfigError, Fenced, PlanError
from swarm_arena._journal.lease import Head, Lease
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import Estimate
from swarm_arena._store.local import LocalStore


@dataclass
class Example:
    run_toml: Path
    plan_toml: Path
    plans_dir: Path
    run_root: Path
    artifact_root: Path


@pytest.fixture
def example(tmp_path: Path, run_config: RunConfig) -> Example:
    run_toml = tmp_path / "run.toml"
    run_toml.write_text(run_config.dumps())
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    plan_toml = plans_dir / "a.toml"
    plan_toml.write_text(plan(rounds=1).dumps())
    return Example(
        run_toml,
        plan_toml,
        plans_dir,
        Path(run_config.run_root),
        Path(run_config.artifact_root),
    )


@pytest.fixture
def example_two_goals(example: Example) -> Example:
    (example.plans_dir / "b.toml").write_text(plan(goal_id="b", rounds=1).dumps())
    example.plan_toml.write_text(plan(goal_id="a", rounds=1).dumps())
    (example.plans_dir / "notes.txt").write_text("ignored")
    return example


@pytest.fixture
def example_bad(example: Example) -> Example:
    example.plan_toml.write_text(plan(rounds=2).model_copy(update={"ranking": "missing"}).dumps())
    config = RunConfig.load(example.run_toml)
    example.run_toml.write_text(
        config.model_copy(
            update={
                "limits": config.limits.model_copy(update={"max_rounds": 1}),
            }
        ).dumps()
    )
    return example


@pytest.fixture
def finished_run(example: Example, capsys: pytest.CaptureFixture[str]) -> Path:
    assert main(["run", "--yes", str(example.run_toml), str(example.plan_toml)]) == 0
    capsys.readouterr()
    return next(example.run_root.iterdir())


class _Stdin:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def fileno(self) -> int:
        return 0


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ready(fd: int) -> None:
        assert fd == 0

    monkeypatch.setattr(sys, "stdin", _Stdin(tty=True))
    monkeypatch.setattr(run_cmds, "wait_readable", ready)
    monkeypatch.setattr(run_cmds, "get_blocking", lambda fd: True)
    monkeypatch.setattr(run_cmds, "set_blocking", lambda fd, blocking: None)


def test_run_yes_prints_estimate_and_run_dir(
    example: Example, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--yes", str(example.run_toml), str(example.plan_toml)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("worst-case estimate: ")
    assert re.match(r"run \S+ at \S+", out[-2])
    assert out[-1] == "done: g 1 finalists"


def test_run_without_tty_or_yes_exits_2_and_writes_nothing(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=False))
    assert main(["run", str(example.run_toml), str(example.plan_toml)]) == 2
    assert not example.run_root.exists()
    assert not example.artifact_root.exists()


def test_run_prompt_declined_exits_1(
    example: Example, monkeypatch: pytest.MonkeyPatch, terminal: None
) -> None:
    handler = signal.getsignal(signal.SIGINT)
    answer = iter(b"n\n")
    monkeypatch.setattr(run_cmds, "read", lambda fd, size: bytes([next(answer)]))
    assert main(["run", str(example.run_toml), str(example.plan_toml)]) == 1
    assert signal.getsignal(signal.SIGINT) is handler
    assert not example.run_root.exists()
    assert not example.artifact_root.exists()


def test_run_accepts_plans_dir(
    example_two_goals: Example, capsys: pytest.CaptureFixture[str]
) -> None:
    e = example_two_goals
    assert main(["run", "--yes", str(e.run_toml), str(e.plans_dir)]) == 0
    done = [
        line.split()[1]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("done: ")
    ]
    assert done == ["a", "b"]


def test_plan_error_lists_every_error(
    example_bad: Example, capsys: pytest.CaptureFixture[str]
) -> None:
    e = example_bad
    assert main(["run", "--yes", str(e.run_toml), str(e.plan_toml)]) == 1
    assert capsys.readouterr().err.count("\n- ") == 2


def test_status_stop_resume_dispatch(
    finished_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["status", str(finished_run)]) == 0
    assert capsys.readouterr().out.startswith("run ")
    assert main(["stop", str(finished_run)]) == 1


@pytest.mark.parametrize("command", ["status", "stop"])
def test_missing_run_is_error(
    tmp_path: Path, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing"
    assert main([command, str(missing)]) == 1
    assert capsys.readouterr().err
    assert not missing.exists()


@pytest.mark.parametrize("answer", ["y", "yes", "YES", "no", "", "yep"])
def test_terminal_approval(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    capsys: pytest.CaptureFixture[str],
    terminal: None,
) -> None:
    reply = iter((answer + "\n").encode())

    def respond(fd: int, size: int) -> bytes:
        assert (fd, size) == (0, 1)
        assert threading.current_thread() is threading.main_thread()
        assert not example.run_root.exists()
        assert not example.artifact_root.exists()
        return bytes([next(reply)])

    monkeypatch.setattr(run_cmds, "read", respond)
    accepted = answer.lower() in {"y", "yes"}
    assert main(["run", str(example.run_toml), str(example.plan_toml)]) == (0 if accepted else 1)
    captured = capsys.readouterr()
    assert bool(re.search(r"run \S+ at \S+", captured.out)) == accepted
    if not accepted:
        assert captured.err == "run not approved\n"
        assert not example.run_root.exists()
        assert not example.artifact_root.exists()


def test_terminal_eof_declines(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    terminal: None,
) -> None:
    monkeypatch.setattr(run_cmds, "read", lambda fd, size: b"")
    assert main(["run", str(example.run_toml), str(example.plan_toml)]) == 1
    assert "Start the run? [y/N] " in capsys.readouterr().out
    assert not example.run_root.exists()
    assert not example.artifact_root.exists()


async def test_prompt_cancellation_releases_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    reader, writer = os.pipe()
    entered = anyio.Event()
    original_wait = run_cmds.wait_readable

    async def wait(fd: int) -> None:
        entered.set()
        await original_wait(fd)

    try:
        with os.fdopen(reader, "rb") as stdin:
            monkeypatch.setattr(sys, "stdin", stdin)
            monkeypatch.setattr(run_cmds, "wait_readable", wait)
            async with anyio.create_task_group() as group:
                group.start_soon(run_cmds._prompt)
                with anyio.fail_after(10):
                    await entered.wait()
                group.cancel_scope.cancel()
            assert os.get_blocking(reader)
            # A leaked readiness registration would reject a second waiter.
            os.write(writer, b"n\n")
            assert await run_cmds._prompt() is False
            assert os.get_blocking(reader)
    finally:
        os.close(writer)


def _write_solution(run_root: str, run_id: str, goal_id: str, *, status: str = "ok") -> None:
    document = SolutionDocument.model_validate(
        {
            "run_id": run_id,
            "goal_id": goal_id,
            "status": status,
            "reason": "no passing candidates" if status == "no_result" else None,
            "finalists": [{}] if status == "ok" else [],
            "parts": [{"part": "p"}] if status == "unjoined" else [],
            "replicate_stats": [],
            "best_of_n": None,
            "plan_revisions": [],
            "manifest": {},
            "source_head": Head(epoch=1, seq=1, segment="journal/e1-1.jsonl"),
        }
    )
    LocalStore(Path(run_root) / run_id).create(
        f"goals/{goal_id}/solution.json",
        document.model_dump_json().encode(),
    )


def test_estimate_warnings_location_and_results_are_ordered_and_flushed(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = RunConfig.load(example.run_toml)
    config = config.model_copy(
        update={
            "limits": config.limits.model_copy(
                update={
                    "max_tokens": 1000,
                    "max_usd": 10,
                }
            )
        }
    )
    example.run_toml.write_text(config.dumps())
    flushed: list[str] = []
    original_print = print

    def capture_print(text: str, *, flush: bool = False) -> None:
        if flush:
            flushed.append(text)
        original_print(text, flush=flush)

    async def run(
        self: Swarm,
        plans: Sequence[Plan],
        *,
        run_id: str,
        approve: Callable[[Estimate, Sequence[str]], Awaitable[bool]],
    ) -> dict[str, list[Candidate]]:
        assert await approve(Estimate(tokens=123, usd=1.234), ["first", "second"])
        assert flushed == [f"run {run_id} at {example.run_root / run_id}"]
        assert not example.run_root.exists()
        _write_solution(self.config.run_root, run_id, plans[0].goal.id)
        return {plans[0].goal.id: []}

    monkeypatch.setattr(run_cmds, "print", capture_print, raising=False)
    monkeypatch.setattr(Swarm, "run", run)
    assert main(["run", "--yes", str(example.run_toml), str(example.plan_toml)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == [
        "worst-case estimate: 123 tokens, $1.23",
        "warning: first",
        "warning: second",
    ]
    assert out[3].startswith("run ")
    assert out[4] == "done: g 1 finalists"


@pytest.mark.parametrize("status", ["ok", "no_result", "unjoined"])
def test_committed_solution_summary(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run(
        self: Swarm,
        plans: Sequence[Plan],
        *,
        run_id: str,
        approve: Callable[[Estimate, Sequence[str]], Awaitable[bool]],
    ) -> dict[str, list[Candidate]]:
        assert await approve(Estimate(tokens=0, usd=0), [])
        _write_solution(self.config.run_root, run_id, "g", status=status)
        return {"g": []}

    monkeypatch.setattr(Swarm, "run", run)
    assert main(["run", "--yes", str(example.run_toml), str(example.plan_toml)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "worst-case estimate: unlimited tokens, unlimited"
    assert (
        out[-1]
        == {
            "ok": "done: g 1 finalists",
            "no_result": "no result: g no passing candidates",
            "unjoined": "unjoined: g 1 parts",
        }[status]
    )


def test_real_no_result_run_exits_zero(
    example: Example, capsys: pytest.CaptureFixture[str]
) -> None:
    example.plan_toml.write_text(plan(rounds=1).model_copy(update={"teams": []}).dumps())
    assert main(["run", "--yes", str(example.run_toml), str(example.plan_toml)]) == 0
    assert "no result: g " in capsys.readouterr().out


def test_explicit_plan_order_is_preserved(
    example_two_goals: Example,
    capsys: pytest.CaptureFixture[str],
) -> None:
    e = example_two_goals
    assert (
        main(["run", "--yes", str(e.run_toml), str(e.plans_dir / "b.toml"), str(e.plan_toml)]) == 0
    )
    assert [
        line.split()[1]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("done: ")
    ] == ["b", "a"]


@pytest.mark.parametrize("now", [False, True])
def test_stop_targets_live_epoch(tmp_path: Path, now: bool) -> None:
    run_dir = tmp_path / "runs" / "r"
    store = LocalStore(run_dir)
    Lease.acquire_new(store, owner_id="owner", host="host", ttl_s=60)
    assert main(["stop", *(["--now"] if now else []), str(run_dir)]) == 0
    key = "stop-now-e1" if now else "stop-e1"
    assert store.exists(f"control/{key}")


@pytest.mark.parametrize("command", ["run", "status", "stop", "resume"])
@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError])
def test_unforeseen_errors_hide_values(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    error_type: type[Exception],
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "CLI-SECRET-SENTINEL"

    async def fail(*args: object, **kwargs: object) -> None:
        raise error_type(secret)

    target = {
        "run": "_run_async",
        "status": "_status_async",
        "stop": "request_stop",
        "resume": "_resume_async",
    }[command]
    monkeypatch.setattr(run_cmds, target, fail)
    argv = (
        ["run", "--yes", str(example.run_toml), str(example.plan_toml)]
        if command == "run"
        else [command, str(example.run_root / "r")]
    )
    assert main(argv) == 1
    assert capsys.readouterr().err == f"{error_type.__name__}\n"


@pytest.mark.parametrize("error_type", [ConfigError, Fenced, PlanError])
def test_external_core_exception_classes_hide_values(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(*args: object, **kwargs: object) -> None:
        if error_type is PlanError:
            raise PlanError(["CLI-SECRET-SENTINEL"])
        raise error_type("CLI-SECRET-SENTINEL")

    monkeypatch.setattr(Swarm, "run", fail)
    assert main(["run", "--yes", str(example.run_toml), str(example.plan_toml)]) == 1
    assert capsys.readouterr().err == f"{error_type.__name__}\n"


@pytest.mark.parametrize("target", ["run_toml", "plan_toml"])
@pytest.mark.parametrize("data", [b"\xffCLI-SECRET-SENTINEL", b'x = "CLI-SECRET-SENTINEL'])
def test_loader_errors_hide_values(
    example: Example,
    target: str,
    data: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    getattr(example, target).write_bytes(data)
    assert main(["run", "--yes", str(example.run_toml), str(example.plan_toml)]) == 1
    captured = capsys.readouterr()
    assert "CLI-SECRET-SENTINEL" not in captured.err
    assert "Traceback" not in captured.err
    assert not example.run_root.exists()


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("spelling", ["absolute", "relative", ".", "..", "symlink"])
def test_resume_forwards_saved_or_override_config_and_directory_root(
    example: Example,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: bool,
    force: bool,
    spelling: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "moved-runs" / "r"
    run_dir.mkdir(parents=True)
    saved = RunConfig.load(example.run_toml)
    (run_dir / "run.toml").write_text(saved.dumps())
    replacement = saved.model_copy(update={"sync_interval_s": 7})
    override_path = tmp_path / "override.toml"
    override_path.write_text(replacement.dumps())
    calls: list[tuple[RunConfig, str, bool]] = []

    async def resume(
        self: Swarm,
        run_id: str,
        *,
        force: bool = False,
    ) -> dict[str, list[Candidate]]:
        calls.append((self.config, run_id, force))
        _write_solution(self.config.run_root, run_id, "b", status="no_result")
        _write_solution(self.config.run_root, run_id, "a", status="unjoined")
        return {"b": [], "a": []}

    monkeypatch.setattr(Swarm, "resume", resume, raising=False)
    argv = ["resume", _directory_spelling(run_dir, spelling, monkeypatch)]
    if override:
        argv.extend(["--config", str(override_path)])
    if force:
        argv.append("--force")
    assert main(argv) == 0
    expected = (replacement if override else saved).model_copy(
        update={"run_root": str(run_dir.parent)},
    )
    assert calls == [(expected, "r", force)]
    assert capsys.readouterr().out.splitlines() == [
        "no result: b no passing candidates",
        "unjoined: a 1 parts",
    ]


def _directory_spelling(run_dir: Path, spelling: str, monkeypatch: pytest.MonkeyPatch) -> str:
    if spelling == "absolute":
        return str(run_dir)
    if spelling == "relative":
        monkeypatch.chdir(run_dir.parent)
        return run_dir.name
    if spelling == ".":
        monkeypatch.chdir(run_dir)
        return "."
    if spelling == "symlink":
        link = run_dir.parent / "alias"
        link.symlink_to(run_dir, target_is_directory=True)
        return str(link)
    child = run_dir / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    return ".."


@pytest.mark.parametrize("command", ["status", "stop"])
@pytest.mark.parametrize("spelling", ["absolute", "relative", ".", "..", "symlink"])
def test_control_normalizes_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    spelling: str,
) -> None:
    run_dir = tmp_path / "runs" / "r"
    store = LocalStore(run_dir)
    Lease.acquire_new(store, owner_id="owner", host="host", ttl_s=60)
    assert main([command, _directory_spelling(run_dir, spelling, monkeypatch)]) == 0
    if command == "status":
        assert capsys.readouterr().out.startswith("run r ")
    else:
        assert store.exists("control/stop-e1")


@pytest.mark.parametrize("override", [False, True])
def test_resume_finished_run_uses_relocated_directory(
    finished_run: Path,
    example: Example,
    tmp_path: Path,
    override: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parent = tmp_path / "relocated"
    parent.mkdir()
    moved = finished_run.rename(parent / finished_run.name)
    argv = ["resume", str(moved)]
    if override:
        argv.extend(["--config", str(example.run_toml), "--force"])
    assert main(argv) == 0
    assert capsys.readouterr().out.splitlines() == ["done: g 1 finalists"]
    assert not finished_run.exists()


def test_resume_missing_saved_config_reports_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["resume", str(tmp_path / "missing")]) == 1
    err = capsys.readouterr().err
    assert "run.toml" in err
    assert "Traceback" not in err


def test_newer_solution_version_is_rejected(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run(
        self: Swarm,
        plans: Sequence[Plan],
        *,
        run_id: str,
        approve: Callable[[Estimate, Sequence[str]], Awaitable[bool]],
    ) -> dict[str, list[Candidate]]:
        assert await approve(Estimate(tokens=0, usd=0), [])
        store = LocalStore(Path(self.config.run_root) / run_id)
        store.create("goals/g/solution.json", b'{"schema_version":999,"reason":"CLI-SECRET"}')
        return {"g": []}

    monkeypatch.setattr(Swarm, "run", run)
    assert main(["run", "--yes", str(example.run_toml), str(example.plan_toml)]) == 1
    err = capsys.readouterr().err
    assert "CLI-SECRET" not in err
    assert "Traceback" not in err


def test_status_cli_exact_total_after_real_overflow_run(
    run_config: RunConfig, tmp_path: Path
) -> None:
    import subprocess

    import anyio
    from _fakes.run import all_events
    from _fakes.tools import PyTool

    from swarm_arena import Limits, Ref
    from swarm_arena._budget.amount import USD_LIMIT

    config = run_config.model_copy(
        update={
            "limits": Limits(
                max_usd=10,
                max_concurrent_goals=1,
                default_run_reservation={"usd": 1, "tokens": 1},
            ),
            "executor": Ref(
                path="swarm_arena.executor:LocalExecutor",
                config={"workspace_root": str(tmp_path / "workspaces")},
            ),
        }
    )
    plans = []
    for goal, usd in [("normal", 0.000001), ("overflow", USD_LIMIT + 1)]:
        code = (
            f"import json,os;json.dump({{'data':{{'y':1.0}},'usage':{{'usd':{usd!r}}}}},"
            "open(os.environ['SWARM_ARENA_RESULT'],'w'))"
        )
        plans.append(plan(goal_id=goal, rounds=1, tool=PyTool({"code": code, "grace_s": 0})))
    swarm = Swarm(config)
    assert sorted(anyio.run(swarm.run, plans)) == ["normal", "overflow"]
    settlements = [e for e in all_events(config) if e.type == "settle"]
    assert [e.payload["actual"]["usd"] for e in settlements] == [0.000001, USD_LIMIT]
    assert settlements[-1].payload["usage_overflow"] is True
    assert swarm.run_id is not None
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("swarm-arena")),
            "status",
            str(Path(config.run_root) / swarm.run_id),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "$1000000000.000001" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("command", ["run", "resume"])
@pytest.mark.parametrize("failure_kind", ["fence", "settlement"])
def test_ordinary_core_failure_keeps_cli_diagnostic(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    failure_kind: str,
) -> None:
    from swarm_arena._errors import _core_error, _stop_cleanup_error

    error = (
        _core_error(Fenced, "run ownership lost")
        if failure_kind == "fence"
        else _stop_cleanup_error()
    )

    async def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(run_cmds, f"_{command}_async", fail)
    argv = (
        [command, "--yes", str(example.run_toml), str(example.plan_toml)]
        if command == "run"
        else [command, str(example.run_root / "r")]
    )
    assert main(argv) == 1
    assert capsys.readouterr().err == f"{error}\n"
