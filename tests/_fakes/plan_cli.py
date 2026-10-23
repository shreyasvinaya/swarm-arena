"""Core-only fixtures for file-based planning CLI tests."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
from dataclasses import dataclass
from pathlib import Path

from swarm_arena import Limits, RunConfig
from swarm_arena._cli.main import main
from swarm_arena.config import CatalogEntry, ModelCatalog, ModelSpec, PlannerCatalog
from swarm_arena.tool import Estimate


@dataclass
class PlanningFiles:
    run: Path
    goal: Path
    config: RunConfig

    def output(self, mode: str = "llm") -> Path:
        return self.goal.parent / f"plan.{mode}.toml"


def call(*args: str | Path) -> int:
    """Give synchronous CLI entry points their own thread and clean context."""

    def invoke() -> int:
        return Context().run(main, [str(arg) for arg in args])

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(invoke).result()


def planning_files(root: Path) -> PlanningFiles:
    source = root / "source"
    source.mkdir()
    (source / "input.txt").write_text("human source bytes")
    goal = source / "goal.toml"
    goal.write_text('[goal]\nid="g"\nnamespace="n"\nfiles=["input.txt"]\n')
    config = RunConfig(
        run_root=str(root / "runs"),
        artifact_root=str(root / "artifacts"),
        arena_root=str(root / "arena"),
        settle_timeout_s=2,
        limits=Limits(max_tokens=1_000_000, default_run_reservation=Estimate(tokens=0, usd=0.0)),
        models=ModelCatalog(
            {
                "planner": ModelSpec(
                    model="fake/planner",
                    roles=["planner"],
                    max_tokens=1000,
                    prices=ModelSpec.Prices(input=1, output=2),
                )
            }
        ),
        planner_catalog=PlannerCatalog(
            entries=[
                CatalogEntry(
                    name="tool",
                    kind="tool",
                    ref="swarm_arena:FunctionTool",
                    description="Add",
                    config={"function": "_fakes.tools:add"},
                ),
                CatalogEntry(
                    name="manager",
                    kind="manager",
                    ref="_fakes.tools:NoopManager",
                    description="No calls",
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
    run = root / "run.toml"
    run.write_text(config.dumps())
    return PlanningFiles(run, goal, config)


def executable_files(root: Path) -> PlanningFiles:
    files = planning_files(root)
    example = Path(__file__).resolve().parents[2] / "examples/rosenbrock/rosen.py"
    (files.goal.parent / "rosen.py").write_bytes(example.read_bytes())
    entries = [
        entry
        for entry in files.config.planner_catalog.entries
        if entry.name not in {"tool", "manager"}
    ]
    entries += [
        CatalogEntry(
            name="tool",
            kind="tool",
            ref="swarm_arena:FunctionTool",
            description="Optimize",
            config={"function": "rosen:gradient_descent"},
        ),
        CatalogEntry(
            name="manager",
            kind="manager",
            ref="swarm_arena:ScriptedManager",
            description="Run fixed inputs",
            config={
                "requests": [
                    {
                        "inputs": {
                            "lr": 0.001,
                            "steps": 10,
                            "x0": -1.5,
                            "y0": 2.0,
                        }
                    }
                ]
            },
        ),
    ]
    files.config = files.config.model_copy(
        update={"planner_catalog": PlannerCatalog(entries=entries)}
    )
    files.run.write_text(files.config.dumps())
    return files
