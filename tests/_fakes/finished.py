"""Importable finished runs for arena, CLI and viewer integration tests."""

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from swarm_arena import Plan, RunConfig, Swarm


@dataclass(frozen=True, kw_only=True)
class FinishedRun:
    config: RunConfig
    config_path: Path
    plan: Plan
    run_id: str
    namespace: str


async def run_again(
    run: FinishedRun, *, run_id: str | None = None, plan: Plan | None = None
) -> str:
    """Run the same roots with an explicit new identity and return that identity."""
    identifier = run_id or f"run-{uuid4().hex[:12]}"
    await Swarm(run.config).run(plan or run.plan, run_id=identifier)
    return identifier
