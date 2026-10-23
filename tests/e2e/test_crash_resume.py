"""Crash the installed CLI and inspect the committed recovery audit trail."""

import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from decimal import Decimal
from pathlib import Path

import anyio
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
from swarm_arena.manager import AgentContext, LimitReached, RoundReport
from swarm_arena.tool import CompletedRun, Estimate, ToolOutput, ToolRequest, Usage

_CLI = str(Path(sys.executable).with_name("swarm-arena"))
_ENV = {
    **os.environ,
    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    "PYTHONUNBUFFERED": "1",
}


def _micro_usd(usd: float | None) -> int:
    assert usd is not None
    units = Decimal(str(usd)) * 1_000_000
    assert units == units.to_integral_value()
    return int(units)


class PaidSleepyTool(FunctionTool):
    """
    Report a fixed actual charge for each completed sleepy invocation.

    Examples
    --------
    >>> PaidSleepyTool("_fakes.tools:sleepy").config.function
    '_fakes.tools:sleepy'
    """

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        """
        Collect the normal process result and charge one tenth of a dollar.

        Parameters
        ----------
        request : ToolRequest
            Executed request.
        workspace : Path
            Executor-owned result workspace.
        run : CompletedRun
            Completed process metadata.

        Returns
        -------
        ToolOutput
            Normal result with explicit usage, preventing a zero-usage refund.

        Examples
        --------
        >>> callable(PaidSleepyTool("_fakes.tools:sleepy").collect)
        True
        """
        return super().collect(request, workspace, run).model_copy(update={"usage": Usage(usd=0.1)})


class RecoveryBudgetManager(ScriptedManager):
    """
    Record live recovery balances before replaying the slow agent's requests.

    Examples
    --------
    >>> RecoveryBudgetManager([]).config.requests
    []
    """

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        """
        Snapshot the live ledger when the first request's marker already exists.

        Parameters
        ----------
        ctx : AgentContext
            Context supplied by the actual CLI engine.

        Returns
        -------
        RoundReport
            Scripted request results, unless the budget limit ends the round.

        Examples
        --------
        >>> callable(RecoveryBudgetManager([]).run_round)
        True
        """
        marker = Path(self.config.requests[0].inputs["marker"])
        if await anyio.to_thread.run_sync(marker.exists):
            # Inspect the actual resumed ledger, not another replay of its journal.
            ledger = ctx._rt.ledger
            snapshot = {
                "run_spent": sum(amount._micro_usd for amount in ledger._charges.values()),
                "run_remaining": ledger._total._micro_usd,
                "goal_spent": sum(
                    ledger._charges[id]._micro_usd
                    for id, entry in ledger._entries.items()
                    if entry.reservation.goal_id == ctx.goal.id
                ),
                "goal_remaining": _micro_usd(ledger.goal_remaining(ctx.goal.id).usd),
                "agent_remaining": _micro_usd(ctx.budget.usd),
            }
            await anyio.to_thread.run_sync(
                marker.with_suffix(".budget.json").write_text, json.dumps(snapshot)
            )
        try:
            return await super().run_round(ctx)
        except LimitReached as error:
            await anyio.to_thread.run_sync(marker.with_suffix(".limit").write_text, error.limit)
            raise


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
            driver.wait(timeout=60)
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
    extra = ToolRequest(
        round=1, inputs={"seconds": 0.0, "marker": str(tmp_path / "extra.pid.json")}
    )
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
                        tool=Ref(
                            path="e2e.test_crash_resume:PaidSleepyTool",
                            config={"function": "_fakes.tools:sleepy"},
                        ),
                        manager=Ref(
                            path="e2e.test_crash_resume:RecoveryBudgetManager",
                            config={"requests": requests},
                        )
                        if name == "slow"
                        else Ref.of(ScriptedManager(requests)),
                    )
                    for name, requests in (
                        ("fast", [{"round": 1, "inputs": request.inputs} for request in fast]),
                        (
                            "slow",
                            [
                                {"round": 1, "inputs": {"seconds": 60.0, "marker": str(marker)}},
                                {"round": 1, "inputs": extra.inputs},
                            ],
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
            # One goal, one round, no service reserve: each agent receives $0.20.
            # Two fast completions plus the lost run spend $0.30. Only the slow
            # retry fits; refunding the lost $0.10 would also admit `extra`.
            "reserve_fraction": 0.0,
            "limits": run_config.limits.model_copy(
                update={
                    "max_usd": 0.4,
                    "max_tool_runs_per_round": 2,
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
        assert driver.wait(60) == -signal.SIGKILL
        assert group_has_live_member(pgid)
        # Allow the lost attempt, its retry and the extra request in the resumed
        # run counter. Keep the dollar cap fixed so only budget can refuse extra.
        resume_config = tmp_path / "resume.toml"
        resume_config.write_text(
            cfg.model_copy(
                update={"limits": cfg.limits.model_copy(update={"max_tool_runs_per_round": 3})}
            ).dumps()
        )
        resumed = subprocess.run(
            [_CLI, "resume", str(run_dir), "--config", str(resume_config), "--force"],
            cwd=tmp_path,
            env=_ENV,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        recovered = json.loads(marker.with_suffix(".budget.json").read_text())
        assert recovered == {
            "run_spent": 300_000,
            "run_remaining": 100_000,
            "goal_spent": 300_000,
            "goal_remaining": 100_000,
            "agent_remaining": 100_000,
        }
        events = _committed_events(run_dir)
        assert all(_tool_starts_for(events, request) == 1 for request in fast)
        assert _tool_starts_for(events, extra) == 0
        assert not Path(extra.inputs["marker"]).exists()
        # Budget exhaustion may close round admission before the next request
        # reaches the budget gate. Both documented refusal paths must preserve
        # the recovered balances and refuse extra work, as asserted above.
        assert marker.with_suffix(".limit").read_text() in {"budget", "stop"}
        assert sum(e.type == "tool_start" and e.epoch == 2 for e in events) == 1
        [retry] = [
            Candidate.model_validate(e.payload["candidate"])
            for e in events
            if e.type == "tool_end" and e.epoch == 2
        ]
        assert retry.output.status == "ok"
        assert retry.request.inputs == {"seconds": 60.0, "marker": str(marker)}
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
        _wait_for(lambda: not group_has_live_member(pgid), 60)
        [takeover] = [e for e in events if e.type == "takeover"]
        assert takeover.epoch == 2
        assert takeover.payload["force"] is True
        assert json.loads((run_dir / "goals/toy-goal/solution.json").read_text())["status"] in {
            "ok",
            "no_result",
        }
