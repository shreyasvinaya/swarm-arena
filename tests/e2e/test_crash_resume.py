"""Crash the installed CLI and inspect the committed recovery audit trail."""

import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path

from _fakes.proc import require_visible_proc, wait_until

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
from swarm_arena._exec.procinfo import group_has_live_member
from swarm_arena._journal.event import Event
from swarm_arena._journal.lease import read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._store.base import open_store
from swarm_arena.board import Candidate
from swarm_arena.tool import Estimate, ToolRequest

_CLI = str(Path(sys.executable).with_name("swarm-arena"))
_ENV = {
    **os.environ,
    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    "PYTHONUNBUFFERED": "1",
}


def _wait_for(pred: Callable[[], bool], timeout_s: float = 60) -> None:
    assert wait_until(pred, timeout_s), "crash-recovery condition did not become true"


@contextmanager
def _start_driver(
    args: list[str], root: Path
) -> Generator[tuple[subprocess.Popen[str], Path], None, None]:
    log = root / "driver.log"
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
            for marker in root.glob("*.pid.json"):
                with suppress(ProcessLookupError):
                    os.killpg(json.loads(marker.read_text())["pgid"], signal.SIGKILL)


def _wait_run_dir(driver: subprocess.Popen[str], log: Path) -> Path:
    paths: list[Path] = []

    def ready() -> bool:
        match = re.search(r"^run \S+ at (.+)$", log.read_text(), re.MULTILINE)
        if match and (Path(match[1]) / "control/owner.json").exists():
            paths.append(Path(match[1]))
            return True
        assert driver.poll() is None, log.read_text()
        return False

    _wait_for(ready)
    return paths[0]


def _committed_events(run_dir: Path) -> list[Event]:
    store = open_store(str(run_dir))
    owner = read_owner(store)
    assert owner is not None
    head = owner[0].head
    return [] if head is None else [e for _, batch in read_chain(store, head) for e in batch]


def _tool_starts_for(events: list[Event], request: ToolRequest) -> int:
    ids = {
        candidate.id
        for e in events
        if e.type == "tool_end"
        if (candidate := Candidate.model_validate(e.payload["candidate"])).request.inputs
        == request.inputs
        and candidate.request.round == request.round
    }
    return sum(e.type == "tool_start" and e.payload["candidate_id"] in ids for e in events)


def test_sigkill_mid_round_then_resume(tmp_path: Path, run_config: RunConfig) -> None:
    require_visible_proc()
    marker = tmp_path / "slow.pid.json"
    fast = [
        ToolRequest(
            round=1, inputs={"seconds": 0.0, "marker": str(tmp_path / f"fast-{n}.pid.json")}
        )
        for n in (1, 2)
    ]
    plan = Plan(
        goal=Goal(id="toy-goal", namespace="e2e"),
        teams=[
            TeamSpec(
                name="team",
                rationale="Commit fast work while another process remains live",
                agents=[
                    AgentSpec(
                        name=name,
                        rationale="Exercise crash recovery",
                        tool=Ref.of(FunctionTool("_fakes.tools:sleepy")),
                        manager=Ref.of(ScriptedManager(requests)),
                    )
                    for name, requests in (
                        ("fast", [{"round": 1, "inputs": request.inputs} for request in fast]),
                        (
                            "slow",
                            [{"round": 1, "inputs": {"seconds": 60.0, "marker": str(marker)}}],
                        ),
                    )
                ],
            )
        ],
        judges=[
            Ref.of(ProgramJudge("_fakes.tools:score_value")).model_copy(update={"name": "value"})
        ],
        ranking="value",
        rounds=[RoundSpec()],
    )
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
                config={
                    "workspace_root": str(tmp_path / "workspaces"),
                    "max_concurrent": 2,
                },
            ),
        }
    )
    run, plan_file = tmp_path / "run.toml", tmp_path / "plan.toml"
    run.write_text(cfg.dumps())
    plan_file.write_text(plan.dumps())
    with _start_driver(["run", "--yes", str(run), str(plan_file)], tmp_path) as (driver, log):
        run_dir = _wait_run_dir(driver, log)
        _wait_for(
            lambda: (
                marker.exists()
                and sum(
                    e.type == "tool_end" and e.agent == "fast" for e in _committed_events(run_dir)
                )
                == 2
            )
        )
        pgid = json.loads(marker.read_text())["pgid"]
        before = _committed_events(run_dir)
        [slow_start] = [e for e in before if e.type == "tool_start" and e.agent == "slow"]
        assert slow_start.payload["started"]["pgid"] == pgid
        assert slow_start.payload["started"]["host_id"]
        assert slow_start.payload["started"]["start_time"]
        os.kill(driver.pid, signal.SIGKILL)
        assert driver.wait(10) == -signal.SIGKILL
        assert group_has_live_member(pgid)
        resumed = subprocess.run(
            [_CLI, "resume", str(run_dir), "--force"],
            cwd=tmp_path,
            env=_ENV,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        events = _committed_events(run_dir)
        assert all(_tool_starts_for(events, request) == 1 for request in fast)
        [lost] = [e for e in events if e.type == "run_lost"]
        assert lost.payload["candidate_id"] == slow_start.payload["candidate_id"]
        assert lost.payload["reservation_id"] == slow_start.payload["reservation_id"]
        assert lost.epoch == 2
        assert lost.round == 1
        assert lost.agent == "slow"
        [reservation] = [
            e.payload["reservation"]
            for e in events
            if e.type == "reserve"
            and e.payload["reservation"]["id"] == lost.payload["reservation_id"]
        ]
        assert reservation["amount"] == {"tokens": 0, "usd": 0.1}
        assert not any(
            e.type == "settle" and e.payload["id"] == lost.payload["reservation_id"] for e in events
        )
        _wait_for(lambda: not group_has_live_member(pgid), 10)
        [takeover] = [e for e in events if e.type == "takeover"]
        assert takeover.epoch == 2
        assert takeover.payload["force"] is True
        assert json.loads((run_dir / "goals/toy-goal/solution.json").read_text())["status"] in {
            "ok",
            "no_result",
        }
