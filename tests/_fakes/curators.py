"""Deterministic curator components for round-end integration tests."""

from collections.abc import Sequence

import pytest

from swarm_arena import Goal, Plan, RunConfig, ScriptedManager
from swarm_arena._board.curator import Curator
from swarm_arena._model.planning_types import Decision, RoundState
from swarm_arena._model.ref import EmptyConfig
from swarm_arena.board import BoardView, Learning
from swarm_arena.manager import AgentContext, RoundReport
from swarm_arena.planning import History, Planner, Policy
from swarm_arena.tool import RunStatus


class MergeAll(Curator):
    """Merge every new learning, leaving authoritative metadata to the core."""

    async def curate(
        self, namespace: str, new: Sequence[Learning], view: BoardView
    ) -> list[Learning]:
        if not new:
            return []
        return [
            Learning(
                id="proposal",
                namespace=namespace,
                run_id="",
                goal_id="",
                round=0,
                author="human:proposal",
                title=f"Summary of {len(new)} learnings",
                text="Combined evidence.",
                supersedes=[item.id for item in new],
            )
        ]


class PostingManager(ScriptedManager):
    """Post current evidence without depending on a later scoring barrier."""

    async def _post_learning(self, ctx: AgentContext) -> None:
        own = [
            candidate
            for candidate in ctx.my_candidates()
            if candidate.round == ctx.round
            and candidate.output.status == RunStatus.OK
            and not candidate.output.partial
            and not candidate.gated_out
        ]
        if own:
            await ctx.post_learning(
                title=f"Round{ctx.round} observation",
                text="Current evidence.",
                evidence=[own[-1].id],
            )


class CountingCurator(MergeAll):
    """Record invocation count separately from the crashable run journal."""

    class Config(EmptyConfig):
        marker: str

    async def curate(
        self, namespace: str, new: Sequence[Learning], view: BoardView
    ) -> list[Learning]:
        from pathlib import Path
        from typing import cast

        marker = Path(cast(CountingCurator.Config, self.config).marker)
        import anyio

        def record() -> None:
            with marker.open("a") as output:
                output.write("called\n")

        await anyio.to_thread.run_sync(record)
        return await super().curate(namespace, new, view)


async def crash_driver(root: str, boundary: str) -> None:
    """Exit at an actual writer boundary, leaving only its committed prefix."""
    import os
    from pathlib import Path

    from _engine.test_goal import _curator_plan
    from _fakes.engine import with_executor

    from swarm_arena import Ref, RunConfig, Swarm
    from swarm_arena._journal.writer import JournalWriter

    path = Path(root)
    config = with_executor(
        RunConfig(
            run_root=str(path / "runs"),
            arena_root=str(path / "arena"),
            artifact_root=str(path / "artifacts"),
            curator=Ref(
                path="_fakes.curators:CountingCurator", config={"marker": str(path / "calls")}
            ),
        ),
        workspace_root=str(path / "ws"),
    )
    original = JournalWriter._publish

    async def publish(self: JournalWriter) -> None:
        self._group_window_s = 0.001
        target = {"prepared": "curated", "terminal_decision": "decision"}.get(boundary, boundary)
        hit = any(event.type == target and event.round == 1 for event in self._queue)
        if hit and boundary == "prepared":
            os._exit(91)
        await original(self)
        if hit:
            os._exit(91)

    plan = _curator_plan()
    if boundary == "terminal_decision":
        plan = plan.model_copy(update={"policy": Ref(path="_fakes.curators:TerminalPolicy")})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(JournalWriter, "_publish", publish)
        await Swarm(config).run(plan, run_id="curation")


class TerminalPolicy(Policy):
    """End after a committed policy without authorizing curation."""

    def after_round(self, state: RoundState) -> Decision:
        return Decision(no_result="terminal policy")


class PostingValueManager(PostingManager):
    """Publish part evidence while the join-retry fixture advances rounds."""

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        await ctx.run_tool({"value": float(10 + ctx.round)})
        await self._post_learning(ctx)
        return RoundReport()


async def join_crash_driver(root: str, boundary: str) -> None:
    """Crash inside curation after the existing retry planner admits a revision."""
    import os
    from pathlib import Path

    from _engine.test_join_recovery import RetryPlanner, retry_config, retry_plan

    from swarm_arena import Ref, Swarm
    from swarm_arena._journal.writer import JournalWriter
    from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog

    path = Path(root)
    plan = retry_plan()
    manager = Ref(
        path="_fakes.curators:PostingValueManager", config={"requests": [], "post_learnings": True}
    )
    for team in plan.teams:
        team.agents[0] = team.agents[0].model_copy(update={"manager": manager})
    config = retry_config(path)
    assert config.planner_catalog is not None
    config = config.model_copy(
        update={
            "curator": Ref(
                path="_fakes.curators:CountingCurator", config={"marker": str(path / "calls")}
            ),
            "planner_catalog": PlannerCatalog(
                entries=[
                    *config.planner_catalog.entries,
                    CatalogEntry(
                        name="posting",
                        kind="manager",
                        ref=manager.path,
                        config=manager.config,
                        description="Post current part evidence",
                    ),
                ]
            ),
        }
    )
    original = JournalWriter._publish

    async def publish(self: JournalWriter) -> None:
        self._group_window_s = 0.001
        hit = any(event.type == boundary and event.round == 2 for event in self._queue)
        await original(self)
        if hit:
            os._exit(91)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(JournalWriter, "_publish", publish)
        await Swarm(config, replanner=RetryPlanner()).run(plan, run_id="curation")


class PaidReplanner(Planner):
    """Record planner dispatch and settle a small affordable reserve call."""

    def __init__(self) -> None:
        self.calls = 0

    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        from swarm_arena._budget.amount import Amount
        from swarm_arena._budget.llm_scope import current_llm_scope

        self.calls += 1
        scope = current_llm_scope()
        assert scope is not None
        call = await scope.tranche.reserve_call(Amount(usd=0.000003))
        await scope.tranche.settle_call(call, Amount(usd=0.000003))
        raise RuntimeError("proposal declined")


class RefusingCurator(CountingCurator):
    """Record invocation before requesting more reserve than the funded run has."""

    async def curate(
        self, namespace: str, new: Sequence[Learning], view: BoardView
    ) -> list[Learning]:
        from swarm_arena._budget.amount import Amount
        from swarm_arena._budget.llm_scope import current_llm_scope

        await super().curate(namespace, new, view)
        scope = current_llm_scope()
        assert scope is not None
        try:
            await scope.tranche.reserve_call(Amount(usd=100))
        finally:
            assert scope.tranche._ledger.remaining("g/:curator").usd > 0
            assert not scope.tranche._ledger.run_exhausted()
        raise AssertionError("unfunded work dispatched")


async def refusal_crash_driver(root: str, boundary: str) -> None:
    """Die after a real refusal control event or receipt is published."""
    import os
    from pathlib import Path

    from _engine.test_goal import _curator_plan
    from _fakes.engine import with_executor

    from swarm_arena import Limits, Ref, Swarm
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._engine.goal import GoalRunner
    from swarm_arena._journal.writer import JournalWriter

    path = Path(root)
    config = with_executor(
        RunConfig(
            run_root=str(path / "runs"),
            arena_root=str(path / "arena"),
            artifact_root=str(path / "artifacts"),
            limits=Limits(max_usd=10),
            curator=Ref(
                path="_fakes.curators:RefusingCurator", config={"marker": str(path / "calls")}
            ),
        ),
        workspace_root=str(path / "ws"),
    )
    original_publish = JournalWriter._publish
    original_close = GoalRunner._close

    async def publish(self: JournalWriter) -> None:
        hit = any(event.type == boundary for event in self._queue)
        await original_publish(self)
        if hit:
            os._exit(91)

    async def close(self: GoalRunner, tranche: Tranche) -> None:
        if boundary == "reserve_exhausted":
            # Force the valid schedule where the control event commits before
            # receipt admission. The writer still publishes the real journal.
            await self.deps.journal.commit()
        await original_close(self, tranche)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(JournalWriter, "_publish", publish)
        patch.setattr(GoalRunner, "_close", close)
        await Swarm(config, replanner=PaidReplanner()).run(_curator_plan(), run_id="curation")
