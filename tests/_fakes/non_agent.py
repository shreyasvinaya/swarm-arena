"""Deterministic component failures and external call counts for recovery tests."""

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, cast

from _fakes.curators import CountingCurator

from swarm_arena import Goal, Plan, RunConfig
from swarm_arena._errors import ProviderFailed, _core_error, _SignalCategory
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.planning_types import History
from swarm_arena._model.results import Candidate, Score
from swarm_arena.board import BoardView, Learning
from swarm_arena.manager import LimitReached
from swarm_arena.planning import Planner
from swarm_arena.scoring import Judge, JudgeContext


class FailingJudge(Judge):
    name: ClassVar[str] = "failure"

    class Config(Judge.Config):
        phase: str = "all"
        kind: str = "provider"
        counter: str = ""

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        config = cast(FailingJudge.Config, self.config)
        if config.counter:

            def count() -> None:
                with Path(config.counter).open("a") as output:
                    output.write(f"{ctx.round}:{ctx.part}:{ctx.final}\n")

            await run_sync_daemon(count)
        if (
            config.phase == "all"
            or (config.phase == "final" and ctx.final)
            or (config.phase == "barrier" and not ctx.final)
        ):
            if config.kind == "provider":
                raise _core_error(ProviderFailed, "anthropic")
            if config.kind == "limit":
                raise _core_error(LimitReached, "budget", "judge budget")
            if config.kind == "terminal":
                raise _core_error(
                    RuntimeError, "stop cleanup timed out", _category=_SignalCategory.TERMINAL
                )
            if config.kind == "spoof":
                raise ProviderFailed("NON_AGENT_SECRET")
            raise RuntimeError("NON_AGENT_SECRET")
        return [
            Score(
                candidate_id=c.id,
                key=((1 if ctx.final else -1) * float(c.output.data.get("y", 0)),),
            )
            for c in candidates
        ]


class FailingCurator(CountingCurator):
    async def curate(
        self, namespace: str, new: Sequence[Learning], view: BoardView
    ) -> list[Learning]:
        await super().curate(namespace, new, view)
        raise _core_error(ProviderFailed, "anthropic")


class FailingPlanner(Planner):
    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        def count() -> None:
            with (Path(config.run_root).parent / "planner-calls").open("a") as output:
                output.write("called\n")

        await run_sync_daemon(count)
        raise _core_error(ProviderFailed, "anthropic")


async def crash_driver(root: str, boundary: str) -> None:
    """Die after a durable receipt, before any following phase can run."""
    import os

    import pytest
    from _engine.test_goal import Y_JUDGE, _curator_plan, _parts_plan, _plan
    from _engine.test_join_recovery import retry_config, retry_plan
    from _fakes.engine import with_executor

    from swarm_arena import Limits, Ref, Swarm
    from swarm_arena._journal.lease import Head
    from swarm_arena._journal.views import ViewWriter
    from swarm_arena._journal.writer import JournalWriter
    from swarm_arena._model.catalog import PlannerCatalog

    path = Path(root)
    rounds = 2 if boundary in {"closed", "page_before", "page_after", "planner"} else 1
    final = boundary in {"final", "final_scores"}
    failed = Ref(
        path="_fakes.non_agent:FailingJudge",
        name="failed",
        config={
            "phase": "none" if boundary == "final_scores" else "final" if final else "barrier",
            "counter": str(path / "judge-calls"),
        },
    )
    plan = _plan(
        rounds=rounds,
        teams=2,
        judges=[Y_JUDGE, failed],
        ranking="y" if boundary in {"batch", "planner"} else "failed",
    )
    config = with_executor(
        RunConfig(
            run_root=str(path / "runs"),
            arena_root=str(path / "arena"),
            artifact_root=str(path / "artifacts"),
            settle_timeout_s=2,
            limits=Limits(max_agents_per_fingerprint=4),
            planner_catalog=PlannerCatalog(entries=[]),
        ),
        workspace_root=str(path / "ws"),
    )
    replanner = FailingPlanner() if rounds > 1 else None
    if boundary == "parts":
        plan = _parts_plan(rounds=1, own_judges=True)
        plan = plan.model_copy(
            update={
                "parts": [
                    part.model_copy(update={"judges": [*(part.judges or []), failed]})
                    for part in plan.parts
                ]
            }
        )
    if boundary == "curator":
        plan = _curator_plan(rounds=1)
        config = config.model_copy(
            update={
                "curator": Ref(
                    path="_fakes.non_agent:FailingCurator",
                    config={"marker": str(path / "curator-calls")},
                )
            }
        )
    if boundary == "join":
        plan, config, replanner = retry_plan(), retry_config(path), FailingPlanner()
    original = JournalWriter._publish

    async def publish(self: JournalWriter) -> None:
        self._group_window_s = 0.001
        hit = any(
            (
                boundary in {"batch", "parts"}
                and "barrier_batch" in e.payload
                and e.type == "provider_failed"
            )
            or (boundary == "ranking_retry" and e.payload.get("barrier_batch", {}).get("retry_of"))
            or (boundary == "policy" and e.payload.get("disposition") == "policy_skipped")
            or (boundary == "curator" and e.payload.get("disposition") == "curator_skipped")
            or (final and "final_batch" in e.payload)
            or (boundary == "closed" and e.type == "round_closed" and e.round == 1)
            or (boundary in {"planner", "join"} and e.type == "plan_rejected")
            for e in self._queue
        )
        await original(self)
        if hit:
            os._exit(91)

    write = ViewWriter.write

    async def write_view(
        self: ViewWriter, key: str, content: bytes, *, source: Head, create_only: bool = False
    ) -> None:
        hit = boundary in {"page_before", "page_after"} and key.endswith("round-01.md")
        if hit and boundary == "page_before":
            os._exit(91)
        await write(self, key, content, source=source, create_only=create_only)
        if hit:
            os._exit(91)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(JournalWriter, "_publish", publish)
        patch.setattr(ViewWriter, "write", write_view)
        await Swarm(config, replanner=replanner).run(plan, run_id="failure")
