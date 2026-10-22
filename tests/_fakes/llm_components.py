"""Import-triggered LLM integration fixtures for admission and startup tests."""

from collections.abc import Sequence

import swarm_arena.llm  # noqa: F401 - register optional model and price hooks
from swarm_arena._model.base import _SafeModel
from swarm_arena._model.goal import Goal
from swarm_arena._model.results import Candidate, Score
from swarm_arena.board import BoardView, Curator, Learning
from swarm_arena.manager import AgentContext, Manager, RoundReport
from swarm_arena.scoring import Judge, JudgeContext


class AliasManager(Manager):
    """Declare one catalog model without making provider calls."""

    class Config(_SafeModel):
        model: str

    def models(self) -> tuple[str, ...]:
        """Return the configured alias."""
        return (self.Config.model_validate(self.config).model,)

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        """Complete without work."""
        return RoundReport()


class AliasJudge(Judge):
    """Declare one catalog model without making provider calls."""

    class Config(_SafeModel):
        model: str

    def models(self) -> tuple[str, ...]:
        """Return the configured alias."""
        return (self.Config.model_validate(self.config).model,)

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        """Return no scores."""
        return []


class AliasCurator(Curator):
    """Declare one startup model without making provider calls."""

    class Config(_SafeModel):
        model: str

    def models(self) -> tuple[str, ...]:
        """Return the configured alias."""
        return (self.Config.model_validate(self.config).model,)

    async def curate(
        self, namespace: str, new: Sequence[Learning], view: BoardView
    ) -> list[Learning]:
        """Return no summaries."""
        return []
