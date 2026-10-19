from typing import Any, ClassVar, cast
from pathlib import Path
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict

from swarm_arena import Goal
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import RunStatus, ToolOutput, ToolRequest
from swarm_arena._model.tooling import Command, CompletedRun, ToolOutput, ToolRequest
from swarm_arena._tooling.result_file import tool_output_from_result
from swarm_arena._tooling.tool import Tool

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
# --- Task 13 ---


class PyInputs(BaseModel):
    n: int = 0


class PyTool(Tool):
    name = "py"
    description = "Runs Python code."
    input_model = PyInputs

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        code: str
        timeout_s: float | None = None
        grace_s: float = 10.0

    @property
    def config(self) -> Config:
        return cast(PyTool.Config, super().config)

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        return Command(
            argv=["python", "-c", self.config.code],
            timeout_s=self.config.timeout_s,
            grace_s=self.config.grace_s,
        )

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        return tool_output_from_result(workspace, run)


class BadCollectTool(PyTool):
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        raise RuntimeError("collect broke")


class EchoTool(Tool):
    name = "echo"
    description = "Echo text."
    input_model = PyInputs

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        text: str = ""

    @property
    def config(self) -> Config:
        return cast(EchoTool.Config, super().config)

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        return Command(argv=["echo", self.config.text])

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        return ToolOutput(
            status=run.status,
            data={"text": self.config.text},
            started_at=run.started_at,
            ended_at=run.ended_at,
        )


WRITE = "import json, os; json.dump({0}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
