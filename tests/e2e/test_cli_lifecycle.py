"""Drive the installed CLI; inspect only authoritative committed journal events."""

import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest
from _fakes.proc import wait_until
from _fakes.tools import score_value, sleepy

from swarm_arena import (
    AgentSpec,
    FunctionTool,
    Goal,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    ScriptedManager,
    TeamSpec,
)
from swarm_arena._journal.event import Event
from swarm_arena._journal.lease import read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._model.results import Candidate, Score
from swarm_arena._store.base import open_store
from swarm_arena.tool import Estimate

_CLI = str(Path(sys.executable).with_name("swarm-arena"))
_ENV = {
    **os.environ,
    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    "PYTHONUNBUFFERED": "1",
}


def _wait_for(pred: Callable[[], bool], timeout_s: float = 60) -> None:
    assert wait_until(pred, timeout_s), "lifecycle condition did not become true"


@contextmanager
def _start_driver(
    args: list[str], root: Path
) -> Generator[tuple[subprocess.Popen[str], Path], None, None]:
    log = root / f"driver-{len(list(root.glob('driver-*.log')))}.log"
    with log.open("w") as output:
        driver = subprocess.Popen(
            [_CLI, *args], cwd=root, env=_ENV, stdout=output, stderr=output, text=True
        )
        try:
            yield driver, log
        finally:
            if driver.poll() is None:
                driver.kill()
            driver.wait(timeout=10)
            # A failed assertion must not leave any sleeping tool groups behind.
            for marker in root.glob("*.pid.json"):
                pgid = json.loads(marker.read_text())["pgid"]
                with suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)


def _wait_run_dir(driver: subprocess.Popen[str], log: Path) -> Path:
    found: list[Path] = []

    def ready() -> bool:
        match = re.search(r"^run \S+ at (.+)$", log.read_text(), re.MULTILINE)
        if match is not None:
            path = Path(match[1])
            if (path / "control/owner.json").exists():
                found.append(path)
                return True
        assert driver.poll() is None, log.read_text()
        return False

    _wait_for(ready)
    return found[0]


def _cli(args: list[str], root: Path) -> str:
    result = subprocess.run(
        [_CLI, *args], cwd=root, env=_ENV, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _events(run_dir: Path) -> list[Event]:
    store = open_store(str(run_dir))
    owner = read_owner(store)
    assert owner is not None
    head = owner[0].head
    return [] if head is None else [e for _, batch in read_chain(store, head) for e in batch]


def _score_preparation_barrier(goal: Goal, candidate: Candidate) -> Score:
    """
    Hold barrier judging within its stop grace while preparation returns.

    Parameters
    ----------
    goal : Goal
        Goal passed through to the ordinary value scorer.
    candidate : Candidate
        Result carrying the preparation fixture's coordination directory.

    Returns
    -------
    Score
        The ordinary value score after the preparation callback returns.
    """
    if candidate.round == 1:
        root = Path(candidate.request.inputs["marker"]).parent
        (root / "barrier-entered").touch()
        _wait_for(lambda: (root / "barrier-release").exists())
    return score_value(goal, candidate)


def _plan(root: Path, *, preparation: bool = False) -> Plan:
    rounds = 2 if preparation else 3
    requests = [
        {
            "round": r,
            "inputs": {
                "seconds": 0.0 if preparation or r == 3 else 60.0,
                "marker": str(root / f"round-{r}.pid.json"),
            },
        }
        for r in range(1, rounds + 1)
    ]
    agents = [
        AgentSpec(
            name="worker",
            rationale="Produce a result in every ordinary round",
            tool=Ref.of(FunctionTool("_fakes.tools:sleepy")),
            manager=Ref.of(ScriptedManager(requests)),
        )
    ]
    if preparation:
        agents.append(
            AgentSpec(
                name="preparing",
                rationale="Exercise cancellation before launch",
                tool=Ref(
                    path="_fakes.tools:BlockedPreparationTool",
                    config={"root": str(root)},
                ),
                manager=Ref.of(ScriptedManager([{"round": 1, "inputs": {}}])),
            )
        )
    return Plan(
        goal=Goal(id="toy-goal", namespace="e2e"),
        teams=[TeamSpec(name="team", rationale="Exercise lifecycle transitions", agents=agents)],
        judges=[
            Ref.of(
                ProgramJudge(
                    "e2e.test_cli_lifecycle:_score_preparation_barrier"
                    if preparation
                    else "_fakes.tools:score_value"
                )
            ).model_copy(update={"name": "value"})
        ],
        ranking="value",
        rounds=[RoundSpec() for _ in range(rounds)],
    )


def _write_inputs(root: Path, run_config: RunConfig, *, preparation: bool = False) -> list[str]:
    cfg = run_config.model_copy(
        update={
            "lease_ttl_s": 3,
            "limits": run_config.limits.model_copy(
                update={
                    "max_usd": 10.0,
                    "default_run_reservation": Estimate(tokens=0, usd=0.1),
                }
            ),
            "executor": Ref(
                path="swarm_arena.executor:LocalExecutor",
                config={"workspace_root": str(root / "workspaces"), "max_concurrent": 2},
            ),
        }
    )
    run, plan = root / "run.toml", root / "plan.toml"
    run.write_text(cfg.dumps())
    plan.write_text(_plan(root, preparation=preparation).dumps())
    return ["run", "--yes", str(run), str(plan)]


def _wait_round(run_dir: Path, root: Path, round: int, epoch: int) -> None:
    def ready() -> bool:
        return (root / f"round-{round}.pid.json").exists() and any(
            e.type == "tool_start" and e.round == round and e.epoch == epoch
            for e in _events(run_dir)
        )

    _wait_for(ready)


def _assert_stopped(run_dir: Path, *, epoch: int, round: int, now: bool = False) -> None:
    events = _events(run_dir)
    [stop] = [e for e in events if e.type == "stop_requested" and e.epoch == epoch]
    assert stop.payload["now"] is now
    [finished] = [e for e in events if e.type == "goal_finished" and e.epoch == epoch]
    assert finished.payload["reason"] == ("stop --now" if now else "stop")
    closed = [e for e in events if e.type == "round_closed" and e.goal_id == "toy-goal"]
    assert [e.payload["round"] for e in closed] == list(range(1, round + 1))
    assert closed[-1].seq < finished.seq


def _stop_and_release(run_dir: Path, root: Path, *, epoch: int, round: int) -> None:
    _cli(["stop", str(run_dir)], root)
    _wait_for(
        lambda: any(e.type == "stop_requested" and e.epoch == epoch for e in _events(run_dir))
    )
    (root / f"round-{round}.pid.release").touch()


def test_sleepy_publishes_identity_and_skips_existing_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "identity.json"
    assert sleepy(0.0, str(marker)) == {"value": 1.0}
    before = marker.read_bytes()
    assert json.loads(before) == {"pid": os.getpid(), "pgid": os.getpgid(0)}

    def unexpected_sleep(seconds: float) -> None:
        pytest.fail("an existing marker must skip sleep")

    monkeypatch.setattr("time.sleep", unexpected_sleep)
    assert sleepy(600.0, str(marker)) == {"value": 1.0}
    assert marker.read_bytes() == before
    assert list(tmp_path.iterdir()) == [marker]


def test_run_stop_resume_status(tmp_path: Path, run_config: RunConfig) -> None:
    args = _write_inputs(tmp_path, run_config)
    with _start_driver(args, tmp_path) as (driver, log):
        run_dir = _wait_run_dir(driver, log)
        _wait_round(run_dir, tmp_path, 1, 1)
        assert "epoch 1  owner live" in _cli(["status", str(run_dir)], tmp_path)
        _stop_and_release(run_dir, tmp_path, epoch=1, round=1)
        assert driver.wait(120) == 0, log.read_text()
        _assert_stopped(run_dir, epoch=1, round=1)
        assert "stop stop" in _cli(["status", str(run_dir)], tmp_path)
        assert (run_dir / "control/stop-e1").exists()

    with _start_driver(["resume", str(run_dir)], tmp_path) as (driver, log):
        _wait_round(run_dir, tmp_path, 2, 2)
        assert "round 2/3" in _cli(["status", str(run_dir)], tmp_path)
        _stop_and_release(run_dir, tmp_path, epoch=2, round=2)
        assert driver.wait(120) == 0, log.read_text()
        _assert_stopped(run_dir, epoch=2, round=2)
        assert (run_dir / "control/stop-e2").exists()

    _cli(["resume", str(run_dir)], tmp_path)
    status = _cli(["status", str(run_dir)], tmp_path)
    assert "epoch 3" in status
    assert "goal toy-goal  finished  round 3/3" in status
    events = _events(run_dir)
    assert [(e.epoch, e.round) for e in events if e.type == "tool_start"] == [
        (1, 1),
        (2, 2),
        (3, 3),
    ]


def test_stop_now_during_preparation_then_resume(tmp_path: Path, run_config: RunConfig) -> None:
    args = _write_inputs(tmp_path, run_config, preparation=True)
    with _start_driver(args, tmp_path) as (driver, log):
        try:
            run_dir = _wait_run_dir(driver, log)
            _wait_for(lambda: (tmp_path / "preparation-entered.json").exists())
            _wait_for(
                lambda: any(e.type == "tool_end" and e.agent == "worker" for e in _events(run_dir))
            )
            workspace = Path(json.loads((tmp_path / "preparation-entered.json").read_text()))
            assert workspace.is_dir()
            _cli(["stop", "--now", str(run_dir)], tmp_path)
            _wait_for(lambda: (tmp_path / "barrier-entered").exists())
            assert driver.poll() is None
            events = _events(run_dir)
            [reservation] = [
                e.payload["reservation"]
                for e in events
                if e.type == "reserve" and e.payload["reservation"]["holder"].endswith("/preparing")
            ]
            [settled] = [
                e for e in events if e.type == "settle" and e.payload["id"] == reservation["id"]
            ]
            assert reservation["amount"] == {"tokens": 0, "usd": 0.1}
            assert settled.epoch == 1
            assert settled.payload["actual"] == {"tokens": 0, "usd": 0.0}
            assert settled.payload["estimated"] is False
            assert not any(
                e.type in {"tool_start", "command_start", "run_lost"} and e.agent == "preparing"
                for e in events
            )
            assert workspace.is_dir()
            (tmp_path / "preparation-release").touch()
            _wait_for(lambda: (tmp_path / "preparation-returned").exists())
            assert (workspace / "late-write").read_text() == "retained"
            assert driver.poll() is None
            (tmp_path / "barrier-release").touch()
            assert driver.wait(120) == 0, log.read_text()
            assert not (tmp_path / "late-launch").exists()
            _assert_stopped(run_dir, epoch=1, round=1, now=True)
        finally:
            (tmp_path / "preparation-release").touch()
            (tmp_path / "barrier-release").touch()

    _cli(["resume", str(run_dir)], tmp_path)
    assert "goal toy-goal  finished  round 2/2" in _cli(["status", str(run_dir)], tmp_path)
    assert not (tmp_path / "late-launch").exists()
    events = _events(run_dir)
    assert not any(e.type in {"command_start", "run_lost"} for e in events)
    assert [(e.epoch, e.round, e.agent) for e in events if e.type == "tool_start"] == [
        (1, 1, "worker"),
        (2, 2, "worker"),
    ]
