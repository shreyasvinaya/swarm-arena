"""Catalog and structured proposals shared by offline planner tests."""

from pathlib import Path
from typing import Any

from swarm_arena import Limits, RunConfig
from swarm_arena.config import CatalogEntry, ModelCatalog, ModelSpec, PlannerCatalog


def planner_config(root: Path) -> RunConfig:
    return RunConfig(
        run_root=str(root / "runs"),
        artifact_root=str(root / "artifacts"),
        arena_root=str(root / "arena"),
        settle_timeout_s=2,
        limits=Limits(max_tokens=1_000_000),
        models=ModelCatalog(
            {
                "planner": ModelSpec(
                    model="fake/planner",
                    roles=["planner"],
                    max_tokens=1000,
                    prices=ModelSpec.Prices(input=1, output=2),
                ),
                "opus": ModelSpec(
                    model="fake/opus",
                    roles=["manager"],
                    max_tokens=1000,
                    prices=ModelSpec.Prices(input=1, output=2),
                ),
            }
        ),
        planner_catalog=PlannerCatalog(
            entries=[
                CatalogEntry(
                    name="tool",
                    kind="tool",
                    ref="swarm_arena:FunctionTool",
                    description="Add two numbers",
                    config={"function": "_fakes.tools:add"},
                ),
                CatalogEntry(
                    name="manager",
                    kind="manager",
                    ref="_fakes.tools:NoopManager",
                    description="No calls",
                ),
                CatalogEntry(
                    name="sdk",
                    kind="manager",
                    ref="swarm_arena.llm:AgentsSDKManager",
                    description="Tune the optimizer",
                    config={"model": "opus"},
                    settable={"instructions": {"type": "string"}},
                ),
                CatalogEntry(
                    name="judge",
                    kind="judge",
                    ref="_fakes.tools:FixedJudge",
                    description="Score",
                    config={"value": 1},
                ),
                CatalogEntry(
                    name="check", kind="check", ref="_fakes.tools:AlwaysPass", description="Check"
                ),
                CatalogEntry(
                    name="policy",
                    kind="policy",
                    ref="swarm_arena:SuccessiveHalving",
                    description="Prune",
                ),
            ]
        ),
    )


def planner_output() -> dict[str, Any]:
    return {
        "rationale": "Explore the catalog",
        "keep_teams": [],
        "keep_parts": [],
        "parts": [],
        "teams": [
            {
                "name": "fit",
                "rationale": "Explore",
                "part": None,
                "agents": [
                    {
                        "name": "a",
                        "rationale": "Optimize",
                        "replicates": 1,
                        "tool": {"entry": "tool", "settings": {}},
                        "manager": {"entry": "manager", "settings": {}},
                    }
                ],
            }
        ],
        "checks": [],
        "judges": [{"name": "quality", "entry": "judge", "settings": {}}],
        "ranking": "quality",
        "policy": {"entry": "policy", "settings": {}},
        "rounds": [{"tool_overrides": {}, "fidelity": "full"}],
        "converge": None,
    }
