"""Revisions must pass prerequisite hooks before dispatch, before publication."""

import json

import pytest
from _engine.test_goal import _plan
from _fakes.run import all_events, events

from swarm_arena import FunctionTool, Goal, Plan, Ref, RoundSpec, RunConfig, Swarm
from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog
from swarm_arena._model.planning_types import History
from swarm_arena._planning.planner import Planner
from swarm_arena._planning.provenance import ref_sites

_SECRET = "PREFLIGHT_REVISION_SECRET"


class RevisionTool(FunctionTool):
    calls: list[float | None] = []

    def preflight(self) -> None:
        self.calls.append(self.config.timeout_s)
        if self.config.timeout_s == 7:
            raise RuntimeError(_SECRET)
        super().preflight()


class ReplaceTool(Planner):
    def __init__(self, *, override: bool = False) -> None:
        super().__init__()
        self.override = override

    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        assert history is not None
        current = history.revisions[-1]
        tool = Ref(
            path=f"{__name__}:RevisionTool",
            config={"function": "_fakes.tools:square", "timeout_s": 7},
        )
        if self.override:
            return current.model_copy(
                update={"rounds": [RoundSpec(), RoundSpec(tool_overrides={"timeout_s": 7})]}
            )
        team = current.teams[0]
        agent = team.agents[0].model_copy(update={"tool": tool})
        return current.model_copy(update={"teams": [team.model_copy(update={"agents": [agent]})]})


def _case(run_config: RunConfig, override: bool) -> tuple[RunConfig, Plan, ReplaceTool]:
    p = _plan(teams=1, rounds=2)
    tool = Ref(
        path=f"{__name__}:RevisionTool",
        config={"function": "_fakes.tools:square", "timeout_s": 60 if override else 7},
    )
    if override:
        p.teams[0].agents[0] = p.teams[0].agents[0].model_copy(update={"tool": tool})
    entries = [
        CatalogEntry(
            name=f"entry-{i}",
            kind=s.kind,
            ref=s.ref.path,
            config={
                k: v
                for k, v in s.ref.config.items()
                if not (override and s.kind == "tool" and k == "timeout_s")
            },
            description="Approved component",
            settable={"timeout_s": {"type": "number"}} if override and s.kind == "tool" else {},
        )
        for i, s in enumerate(ref_sites(p))
    ]
    entries.append(
        CatalogEntry(
            name="replacement",
            kind="tool",
            ref=tool.path,
            config={k: v for k, v in tool.config.items() if not (override and k == "timeout_s")},
            description="Approved tool",
            settable={"timeout_s": {"type": "number"}} if override else {},
        )
    )
    config = run_config.model_copy(update={"planner_catalog": PlannerCatalog(entries=entries)})
    return config, p, ReplaceTool(override=override)


@pytest.mark.parametrize("override", [False, True])
async def test_failed_revision_preflight_preserves_prior_plan(
    run_config: RunConfig, override: bool
) -> None:
    RevisionTool.calls.clear()
    cfg, p, planner = _case(run_config, override)
    result = await Swarm(cfg, replanner=planner).run(p)
    assert 7 in RevisionTool.calls, "the newly admitted configuration must be preflighted"
    assert len(events(cfg, "plan_revision")) == 1
    assert len(events(cfg, "tool_start")) == 2
    assert result["g"]
    rejection = events(cfg, "plan_rejected")[-1].payload
    assert "preflight" in str(rejection)
    assert _SECRET not in json.dumps([e.model_dump(mode="json") for e in all_events(cfg)])
    candidates = [e.payload["candidate"] for e in events(cfg, "tool_end")]
    assert all(c["request"]["round_overrides"] == {} for c in candidates)
