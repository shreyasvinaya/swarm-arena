import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict

from swarm_arena import Goal
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Configurable, RefField
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._model.tooling import (
    Command,
    CompletedRun,
    FileRef,
    RunStatus,
    ToolOutput,
    ToolRequest,
)
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._scoring.judge import Judge, JudgeContext
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


# --- Task 12 ---


def make_job(**fields: Any) -> Job:
    defaults: dict[str, Any] = {
        "id": "x-1",
        "kind": "command",
        "command": Command(argv=["true"]),
        "stage": Job.Stage(),
        "timeout_s": 5.0,
        "grace_s": 1.0,
        "resources": Resources(),
        "env": {},
        "secret_env": (),
        "secrets_file": None,
        "artifact_root": "/tmp/a",
    }
    return Job(**{**defaults, **fields})


# --- Task 11 ---


class FilesWidget(Configurable):
    name: ClassVar[str] = "files-widget"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        files: FileSet = FileSet()
        members: list[RefField] = []
        label: str = ""


# --- Task 9 ---


class HumanOnlyWidget(Configurable):
    _human_only: ClassVar[frozenset[str]] = frozenset({"command"})

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        command: list[str] = []
        timeout_s: float | None = None


# --- Task 17 ---

RELEASE = threading.Event()


class BlockingCollectTool(PyTool):
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        RELEASE.wait(30)
        return super().collect(request, workspace, run)


class GhostFileTool(PyTool):
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        ghost = FileRef(name="ghost.txt", path="out/ghost.txt", sha256="0" * 64, size=0)
        return ToolOutput(status=run.status, data={"x": 1}, files=[ghost])


# --- Task 20 ---


class AlwaysPass(Check):
    name = "always-pass"

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        return CheckResult(check=self.name, passed=True)


class FixedJudge(Judge):
    name = "fixed"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        value: float

    @property
    def config(self) -> Config:
        return cast(FixedJudge.Config, super().config)

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            Score.maximize(self.config.value).model_copy(update={"candidate_id": c.id})
            for c in candidates
        ]


class RecordingExecutor(Executor):
    def __init__(self) -> None:
        super().__init__()
        self.jobs: list[Job] = []

    async def submit(self, job: Job) -> ToolOutput:
        self.jobs.append(job)
        return ToolOutput(status=RunStatus.OK)

    async def capacity(self) -> Resources | None:
        return None

    async def aclose(self) -> None:
        return None


# --- Task 18 ---


def gradient_descent(lr: float, steps: int, x0: float = -1.5, y0: float = 2.0) -> dict[str, float]:
    """Minimize the Rosenbrock function with plain gradient descent."""
    x, y = x0, y0
    for _ in range(steps):
        dx = 2 * (x - 1) - 400 * x * (y - x * x)
        dy = 200 * (y - x * x)
        x, y = x - lr * dx, y - lr * dy
    return {"x": x, "y": y, "f": (1 - x) ** 2 + 100 * (y - x * x) ** 2}


def add(a: int, b: int = 1) -> dict[str, int]:
    """Add two numbers."""
    return {"sum": a + b}


def not_a_dict() -> list[int]:
    return [1]


def boom() -> dict[str, int]:
    raise RuntimeError("boom")


def nan_result() -> dict[str, float]:
    return {"n": float("nan")}


async def coro() -> dict[str, int]:
    return {}


def star(*xs: int) -> dict[str, int]:
    return {"sum": sum(xs)}


def unannotated(x) -> dict[str, int]:  # noqa: ANN001
    return {"x": x}


def keyword_only(*, value: int = 2) -> dict[str, int]:
    return {"value": value}


def positional_only(value: int, /) -> dict[str, int]:
    return {"value": value}


def star_keywords(**values: int) -> dict[str, int]:
    return values


def nested_input(values: dict[str, int]) -> dict[str, int]:
    return values


def secret_error(value: str) -> dict[str, int]:
    raise RuntimeError(value)


def non_json_result() -> dict[str, Any]:
    return {"value": {1, 2}}


# --- Task 21 ---


def rosen_score(goal: Goal, candidate: Candidate) -> Score:
    return Score.minimize(candidate.output.data["f"])


def small_b(goal: Goal, candidate: Candidate) -> Score:
    return Score.minimize(abs(candidate.output.data["b"]))


def raising_score(goal: Goal, candidate: Candidate) -> Score:
    raise RuntimeError("no")


def nan_score(goal: Goal, candidate: Candidate) -> Score:
    return Score.minimize(float("nan"))


def list_score(goal: Goal, candidate: Candidate) -> list[int]:
    return [1]


async def async_score(goal: Goal, candidate: Candidate) -> Score:
    return Score.maximize(1.0)


class PanelJudge(Judge):
    name = "panel"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
        panel: str = ""
        per_candidate: dict[str, str] = {}
        anchors: int = 0
        providers: list[str] = []
        models: list[str] = []

    @property
    def config(self) -> Config:
        return cast(PanelJudge.Config, super().config)

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            Score.maximize(1.0).model_copy(
                update={"candidate_id": c.id, "panel_hash": self.config.per_candidate.get(c.id, "")}
            )
            for c in candidates
        ]

    def panel_hash(self) -> str:
        return self.config.panel

    def anchor_count(self) -> int:
        return self.config.anchors

    def providers(self) -> frozenset[str]:
        return frozenset(self.config.providers)

    def models(self) -> tuple[str, ...]:
        return tuple(self.config.models)
def raises_error() -> dict[str, int]:
    raise RuntimeError("boom")


def exits() -> dict[str, int]:
    sys.exit(0)


def changes_directory(directory: str) -> dict[str, bool]:
    os.chdir(directory)
    return {"changed": True}
