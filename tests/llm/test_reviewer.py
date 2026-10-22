from dataclasses import FrozenInstanceError
from typing import Protocol

import pytest

from swarm_arena import Goal, Plan
from swarm_arena.llm import Approve, Edited, Feedback, Reviewer


def test_decisions_are_frozen_and_reviewer_is_a_protocol() -> None:
    assert issubclass(Reviewer, Protocol)
    for decision, field, value in [
        (Approve(), "extra", 1),
        (Feedback("x"), "text", "y"),
        (Edited(Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])), "plan", None),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(decision, field, value)

    class Human:
        async def review(self, plan: Plan) -> Approve:
            return Approve()

    reviewer: Reviewer = Human()
    assert reviewer is not None
