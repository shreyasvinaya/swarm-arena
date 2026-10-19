from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict

from swarm_arena import Goal
from swarm_arena._model.ref import Configurable

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
