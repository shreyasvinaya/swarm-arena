"""Real process death at join-retry journal boundaries must preserve outcomes."""

import json
import os
import sys
from pathlib import Path

import anyio
import pytest
from _engine.test_goal import _parts_plan
from _fakes.engine import with_executor
from _fakes.run import all_events, events
from _fakes.tools import AlwaysFailCheck

from swarm_arena import FunctionTool, Goal, Plan, Ref, RoundSpec, RunConfig, ScriptedManager, Swarm
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog
from swarm_arena._model.planning_types import History
from swarm_arena.board import Candidate
from swarm_arena.manager import AgentContext, RoundReport
from swarm_arena.planning import Planner
from swarm_arena.scoring import CheckContext, CheckResult

_MODULE = "_engine.test_join_recovery"


class RoundValueManager(ScriptedManager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        await ctx.run_tool({"value": float(10 + ctx.round)})
        return RoundReport()


class JoinManager(ScriptedManager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        await ctx.run_tool({"effort": ctx.round})
        return RoundReport()


class AfterRetryCheck(AlwaysFailCheck):
    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        passed = candidate.round >= 3
        return CheckResult(
            check=self.name, passed=passed, reasons=[] if passed else ["retry needed"]
        )


class RetryPlanner(Planner):
    def __init__(self, *, refuse: bool = False) -> None:
        super().__init__()
        self.refuse = refuse

    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        assert history
        marker = Path(config.run_root).parent / "refused-proposal"
        if self.refuse and not await anyio.to_thread.run_sync(marker.exists):
            await anyio.to_thread.run_sync(marker.write_text, "refused")
            raise ValueError("proposal refused")
        return history.revisions[-1].model_copy(update={"rounds": [RoundSpec(), RoundSpec()]})


def sum_with_effort(effort: int) -> dict[str, float]:
    return {
        "value": sum(
            float(json.loads(p.read_text())["value"]) for p in Path("parents").glob("*/data.json")
        )
    }


def retry_plan() -> Plan:
    p = _parts_plan(fail_join=True)
    for team in p.teams:
        team.agents[0] = team.agents[0].model_copy(
            update={"manager": Ref(path=f"{_MODULE}:RoundValueManager", config={"requests": []})}
        )
    assert p.converge.team
    agent = p.converge.team.agents[0].model_copy(
        update={
            "manager": Ref(path=f"{_MODULE}:JoinManager", config={"requests": []}),
            "tool": Ref.of(FunctionTool(f"{_MODULE}:sum_with_effort")),
        }
    )
    return p.model_copy(
        update={
            "parts": [
                part.model_copy(
                    update={"goal": part.goal.model_copy(update={"inputs": {"part": part.name}})}
                )
                for part in p.parts
            ],
            "checks": [Ref(path=f"{_MODULE}:AfterRetryCheck")],
            "converge": p.converge.model_copy(
                update={
                    "team": p.converge.team.model_copy(update={"agents": [agent]}),
                    "max_join_attempts": 1,
                }
            ),
        }
    )


def retry_config(root: Path) -> RunConfig:
    from swarm_arena._planning.provenance import ref_sites

    p = retry_plan()
    entries = [
        CatalogEntry(
            name=f"entry-{i}",
            kind=site.kind,
            ref=site.ref.path,
            config=site.ref.config,
            description="Approved component",
        )
        for i, site in enumerate(ref_sites(p))
    ]
    return with_executor(
        RunConfig(
            run_root=str(root / "runs"),
            artifact_root=str(root / "artifacts"),
            planner_catalog=PlannerCatalog(entries=entries),
        ),
        workspace_root=str(root / "ws"),
    )


async def crash_driver(root: Path, boundary: str) -> None:
    cfg = retry_config(root)
    real = JournalWriter._publish

    async def publish(self: JournalWriter) -> None:
        self._group_window_s = 0.001
        hit = any(
            (boundary == "join_failed" and e.type == "join_failed")
            or (boundary == "plan_rejected" and e.type == "plan_rejected")
            or (
                boundary == "plan_revision"
                and e.type == "plan_revision"
                and e.payload["plan"]["revision"] == 1
            )
            or (boundary == "round_closed" and e.type == "round_closed" and e.round == 2)
            for e in self._queue
        )
        await real(self)
        if hit:
            os._exit(91)

    JournalWriter._publish = publish
    refuse = boundary in {"plan_rejected", "refused_control"}
    plan = retry_plan()
    if refuse:
        plan = plan.model_copy(
            update={"converge": plan.converge.model_copy(update={"max_join_attempts": 2})}
        )
    await Swarm(cfg, replanner=RetryPlanner(refuse=refuse)).run(plan, run_id="join")


async def launch(root: Path, boundary: str) -> RunConfig:
    await anyio.to_thread.run_sync(lambda: root.mkdir(parents=True))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
    )
    code = (
        f"import sys,anyio; from pathlib import Path; from {_MODULE} import crash_driver; "
        "anyio.run(crash_driver,Path(sys.argv[1]),sys.argv[2])"
    )
    result = await anyio.run_process(
        [sys.executable, "-c", code, str(root), boundary], env=environment, check=False
    )
    assert result.returncode == (0 if boundary in {"control", "refused_control"} else 91), (
        result.stderr.decode()
    )
    return RunConfig.load(root / "runs" / "join" / "run.toml")


@pytest.mark.parametrize(
    "boundary", ["join_failed", "plan_rejected", "plan_revision", "round_closed"]
)
async def test_join_retry_survives_committed_process_crash(tmp_path: Path, boundary: str) -> None:
    control = await launch(
        tmp_path / "control", "refused_control" if boundary == "plan_rejected" else "control"
    )
    cfg = await launch(tmp_path / "crashed", boundary)
    assert all("preflight_complete" not in e.payload for e in events(cfg, "plan_revision"))
    result = await Swarm(cfg, replanner=RetryPlanner(refuse=boundary == "plan_rejected")).resume(
        "join", force=True
    )
    assert result["g"], [e.payload for e in events(cfg, "goal_finished")]
    assert [e.round for e in events(cfg, "round_closed")] == [1, 2]
    for kind in ("round_closed", "join_failed", "plan_rejected", "plan_revision"):
        assert len(events(cfg, kind)) == len(events(control, kind)), kind
    assert [e.payload.get("reason") for e in events(cfg, "goal_finished")] == [None]
    assert len([e for e in all_events(cfg) if e.type == "tool_start" and e.team == "join"]) == 2
    state = RunState.from_events(all_events(cfg))
    control_state = RunState.from_events(all_events(control))
    assert state.goals["g"].join_retries == (2 if boundary == "plan_rejected" else 1)
    assert (
        result["g"][0].output.data
        == control_state.candidates[control_state.goals["g"].final_order[0]].output.data
    )
