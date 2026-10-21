"""Armed extension outcomes for PR1's boundary audit."""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from pydantic import BaseModel

from swarm_arena import Goal, Score
from swarm_arena._errors import Fenced, _core_error
from swarm_arena.board import Candidate
from swarm_arena.scoring import Judge, JudgeContext

SECRET = "pr1-r2-hook-value-0fbd784"


def failure(kind: str) -> BaseException:
    if kind == "fenced":
        error: BaseException = ExceptionGroup(
            SECRET, [ValueError(SECRET), _core_error(Fenced, "run ownership lost")]
        )
    elif kind == "cancel":
        error = asyncio.CancelledError(SECRET)
    else:
        error = BaseExceptionGroup(SECRET, [asyncio.CancelledError(SECRET), ValueError(SECRET)])
    error.add_note(SECRET)
    error.__cause__ = ValueError(SECRET)
    return error


class PreparationJudge(Judge):
    armed: ClassVar[bool] = False
    stage: ClassVar[str] = "anchor_count"
    kind: ClassVar[str] = "fenced"

    def __init__(self, config: BaseModel | Mapping[str, Any] | None = None) -> None:
        if self.armed and self.stage == "constructor":
            raise failure(self.kind)
        super().__init__(config)

    def preflight(self) -> None:
        type(self).armed = True

    def anchor_count(self) -> int:
        if self.armed and self.stage == "anchor_count":
            raise failure(self.kind)
        return 0

    def models(self) -> tuple[str, ...]:
        if self.armed and self.stage == "models":
            raise failure(self.kind)
        return ()

    def providers(self) -> frozenset[str]:
        if self.armed and self.stage == "providers":
            raise failure(self.kind)
        return frozenset()

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [Score.maximize(1.0).model_copy(update={"candidate_id": c.id}) for c in candidates]


def score(goal: Goal, candidate: Candidate) -> Score:
    raise failure(PreparationJudge.kind)
