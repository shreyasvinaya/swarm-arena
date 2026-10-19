from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict

from swarm_arena import Goal
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import RunStatus, ToolOutput, ToolRequest

# --- Task 5 ---


class Widget(Configurable):
    name: ClassVar[str] = "widget"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        size: int = 1
        label: str = ""

    @property
    def config(self) -> Config:
        return cast(Widget.Config, super().config)


class NoNameWidget(Configurable):
    pass


# --- Task 6 ---

GOAL = Goal(id="g", namespace="n")


# --- Task 7 ---


def cand(*, data: dict[str, Any] | None = None, **fields: Any) -> Candidate:
    defaults: dict[str, Any] = {
        "id": "c1",
        "run_id": "r",
        "goal_id": "g",
        "team": "t",
        "agent": "a",
        "round": 1,
        "request": ToolRequest(),
        "output": ToolOutput(status=RunStatus.OK, data=data or {}),
    }
    return Candidate(**{**defaults, **fields})
