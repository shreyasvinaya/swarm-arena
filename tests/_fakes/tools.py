import hashlib
import math
import os
import secrets
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

from swarm_arena import Goal
from swarm_arena._agent.context import AgentContext
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Configurable, RefField
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._model.tooling import (
    Command,
    CompletedRun,
    Estimate,
    FileRef,
    RunStatus,
    ToolOutput,
    ToolRequest,
)
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._scoring.judge import Judge, JudgeContext
from swarm_arena._tooling.result_file import tool_output_from_result
from swarm_arena._tooling.tool import Action, Tool
from swarm_arena.manager import Manager, RoundReport


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


GOAL = Goal(id="g", namespace="n")


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


class FilesWidget(Configurable):
    name: ClassVar[str] = "files-widget"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        files: FileSet = FileSet()
        members: list[RefField] = []
        label: str = ""


class HumanOnlyWidget(Configurable):
    _human_only: ClassVar[frozenset[str]] = frozenset({"command"})

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        command: list[str] = []
        timeout_s: float | None = None


RELEASE = threading.Event()


class BlockingCollectTool(PyTool):
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        RELEASE.wait(30)
        return super().collect(request, workspace, run)


class GhostFileTool(PyTool):
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        ghost = FileRef(name="ghost.txt", path="out/ghost.txt", sha256="0" * 64, size=0)
        return ToolOutput(status=run.status, data={"x": 1}, files=[ghost])


COLLECTING = threading.Event()  # set when SlowCollectTool.collect starts
STAGING = threading.Event()  # set when SlowCommandTool.command starts


class SlowCollectTool(PyTool):
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        COLLECTING.set()
        time.sleep(1.0)
        return super().collect(request, workspace, run)


class SlowCommandTool(PyTool):
    def command(self, request: ToolRequest, workspace: Path) -> Command:
        STAGING.set()
        time.sleep(0.5)
        return super().command(request, workspace)


class OkCollectTool(PyTool):  # ignores run.status, as a careless connector might
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        return ToolOutput(status=RunStatus.OK, data={"k": 1})


class UnreadFilesTool(PyTool):  # declares out/a.txt and out/b.txt without reading them
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        files = [
            FileRef(name=name, path=f"out/{name}", sha256="0" * 64, size=0)
            for name in ("a.txt", "b.txt")
        ]
        return ToolOutput(status=run.status, files=files)


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


# Conformance-kit fakes: each one breaks one component contract.


class EmptyFingerprintTool(PyTool):
    def fingerprint(self, output: ToolOutput) -> str:
        return ""


async def _no_action(context: Any, inputs: PyInputs) -> int:
    return inputs.n


_ACTION = Action(name="same", description="An action.", input_model=PyInputs, handler=_no_action)
FAILING_COMMAND = "import sys; sys.stderr.write('COMMAND-DIAG'); sys.exit(2)"


class BrokenTool(Tool):
    """Runs Python code; the contract named by ``broken`` is broken."""

    name = "broken"
    description = "Runs Python code."
    input_model = PyInputs

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        code: str
        broken: str = ""

    @property
    def config(self) -> Config:
        return cast(BrokenTool.Config, super().config)

    def _breaks(self, method: str) -> bool:
        return self.config.broken == method

    def preflight(self) -> None:
        if self._breaks("preflight"):
            raise RuntimeError("preflight refused the tool")

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        return Command(argv=["python", "-c", self.config.code])

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        if self._breaks("collect_ok"):  # ignores run.status
            return ToolOutput(status=RunStatus.OK, data={})
        output = tool_output_from_result(workspace, run)
        if self._breaks("collect_nan"):  # model_copy skips the finite-data validator
            return output.model_copy(update={"data": {"x": math.nan}})
        if self._breaks("collect_construct"):
            return ToolOutput.model_construct(status=run.status, data={"x": math.inf})
        return output

    def get_input_model(self) -> type[BaseModel]:
        if self._breaks("get_input_model"):
            return cast(Any, dict)
        if self._breaks("get_input_model_instance"):  # a model, not a model class
            return cast(Any, PyInputs())
        return super().get_input_model()

    def get_description(self) -> str:
        return cast(Any, None) if self._breaks("get_description") else super().get_description()

    def editable(self) -> Sequence[str]:
        if self._breaks("editable"):
            return cast(Any, [1])
        if self._breaks("editable_str"):
            return "problem/*"
        if self._breaks("editable_dict"):
            return cast(Any, {"problem/*": True})
        return super().editable()

    def providers(self) -> frozenset[str]:
        if self._breaks("providers"):
            return cast(Any, {"openai"})
        if self._breaks("providers_items"):
            return cast(Any, frozenset({1}))
        return super().providers()

    def extra_actions(self) -> Sequence[Action[Any]]:
        if self._breaks("extra_actions"):
            return cast(Any, [42])
        if self._breaks("extra_actions_names"):
            return (_ACTION, _ACTION)
        return super().extra_actions()

    def fingerprint(self, output: ToolOutput) -> str:
        if self._breaks("fingerprint"):
            return secrets.token_hex(8)
        if self._breaks("fingerprint_run"):  # includes logs_ref and the run times
            return hashlib.sha256(output.model_dump_json().encode()).hexdigest()
        if self._breaks("fingerprint_type"):
            return cast(Any, 123)
        return super().fingerprint(output)

    def is_deterministic(self, request: ToolRequest) -> bool:
        broken = self._breaks("is_deterministic")
        return cast(Any, 1) if broken else super().is_deterministic(request)

    def estimate(self, request: ToolRequest) -> Estimate | None:
        broken = self._breaks("estimate")
        return cast(Any, {"seconds": 1.0}) if broken else super().estimate(request)


class StrayJudge(Judge):
    name = "stray"

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [Score.maximize(1.0).model_copy(update={"candidate_id": "nope"})]


class BrokenJudge(Judge):
    """A judge that breaks one Judge contract, chosen by ``mode``."""

    name = "broken"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        mode: str

    @property
    def config(self) -> Config:
        return cast(BrokenJudge.Config, super().config)

    def preflight(self) -> None:
        if self.config.mode == "preflight":
            raise RuntimeError("preflight refused the judge")

    def rubric_hash(self) -> str:
        mode = self.config.mode
        if mode == "rubric_unstable":
            return secrets.token_hex(32)
        if mode == "rubric_per_instance":  # stable for one object, new for a second build
            return f"{id(self):x}"
        if mode == "rubric_drifts":  # the first call of each object agrees, later calls drift
            calls = vars(self).setdefault("calls", [])
            calls.append(None)
            return f"rubric-{len(calls)}"
        if mode == "rubric_type":
            return cast(Any, 123)
        if mode == "rubric_empty":
            return ""
        return super().rubric_hash()

    def panel_hash(self) -> str:
        if self.config.mode == "panel_type":
            return cast(Any, None)
        if self.config.mode == "panel_unstable":
            return secrets.token_hex(8)
        return super().panel_hash()

    def anchor_count(self) -> int:
        if self.config.mode == "anchor_count":
            return cast(Any, True)
        return -1 if self.config.mode == "anchor_negative" else super().anchor_count()

    def providers(self) -> frozenset[str]:
        if self.config.mode == "providers":
            return cast(Any, ["openai"])
        if self.config.mode == "providers_items":
            return cast(Any, frozenset({1}))
        return super().providers()

    def models(self) -> tuple[str, ...]:
        if self.config.mode == "models":
            return cast(Any, "gpt")
        if self.config.mode == "models_list":
            return cast(Any, ["gpt"])
        return super().models()

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        mode = self.config.mode
        if mode == "not_score":
            return cast(Any, [1.0])
        if mode == "nan":  # model_construct skips the Score validator
            return [Score.model_construct(key=(math.nan,), candidate_id=c.id) for c in candidates]
        if mode == "invalid_score":
            return [
                Score.model_construct(key=(1.0,), candidate_id=c.id, notes=cast(Any, 5))
                for c in candidates
            ]
        if mode in ("repairable_score", "command_then_invalid"):
            if mode == "command_then_invalid":
                await ctx.execute(Command(argv=["python", "-c", FAILING_COMMAND]))
            return [  # lax validation would quietly turn "1.0" into 1.0
                Score.model_construct(key=(1.0,), candidate_id=c.id, value=cast(Any, "1.0"))
                for c in candidates
            ]
        if mode == "drop":  # neither a score nor a judge error
            return []
        if mode == "command_fails":
            await ctx.execute(Command(argv=["python", "-c", FAILING_COMMAND]))
        one = Score.maximize(1.0)
        copies = "ab" if mode == "duplicate" else "a"
        return [one.model_copy(update={"candidate_id": c.id}) for c in candidates for _ in copies]


class BadFeedbackCheck(AlwaysPass):
    feedback = cast(Any, "everything")


class NotACheckResult(AlwaysPass):
    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        return cast(Any, {"passed": True})


class InvalidCheckResult(AlwaysPass):  # a truthy "no": lax validation would make it False
    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        return CheckResult.model_construct(check=self.name, passed=cast(Any, "no"))


class CommandCheck(Check):
    """Run ``code`` through ``ctx.execute``; ``valid=False`` then breaks the result."""

    name = "command-check"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        code: str
        valid: bool = True

    @property
    def config(self) -> Config:
        return cast(CommandCheck.Config, super().config)

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        out = await ctx.execute(Command(argv=["python", "-c", self.config.code]))
        if not self.config.valid:
            return CheckResult.model_construct(check=self.name, passed=cast(Any, "no"))
        return CheckResult(check=self.name, passed=out.status == RunStatus.OK)


class PreflightFailsCheck(AlwaysPass):
    def preflight(self) -> None:
        raise RuntimeError("preflight refused the check")


class GoalCheck(Check):
    """Pass and report the goal that the kit gave the check."""

    name = "goal-check"

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        goal = ctx.goal
        details = {"goal": f"{goal.namespace}/{goal.id}", "files": sorted(goal.files.manifest)}
        return CheckResult(check=self.name, passed=True, details=details)


def manifest_size(goal: Goal, candidate: Candidate) -> Score:
    return Score.maximize(len(goal.files.manifest))


class ResultFingerprintTool(PyTool):  # reads a result key that a failed run does not have
    def fingerprint(self, output: ToolOutput) -> str:
        return str(output.data["result"])


class NanCollectTool(PyTool):  # model_copy skips the finite-data validator, on every path
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        output = tool_output_from_result(workspace, run)
        return output.model_copy(update={"data": {"x": math.nan}})


class ResultCheck(Check):
    """Run ``code`` through ``ctx.execute``, then read ``passed`` from its data unguarded."""

    name = "result-check"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        code: str

    @property
    def config(self) -> Config:
        return cast(ResultCheck.Config, super().config)

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        out = await ctx.execute(Command(argv=["python", "-c", self.config.code]))
        return CheckResult(check=self.name, passed=out.data["passed"])


class ResultJudge(Judge):
    """Run ``code`` through ``ctx.execute``, then read ``value`` from its data unguarded."""

    name = "result-judge"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        code: str

    @property
    def config(self) -> Config:
        return cast(ResultJudge.Config, super().config)

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        out = await ctx.execute(Command(argv=["python", "-c", self.config.code]))
        value = out.data["value"]
        return [Score.maximize(value).model_copy(update={"candidate_id": c.id}) for c in candidates]


class NoopManager(Manager):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
        models: list[str] = []

    @property
    def config(self) -> Config:
        return cast(NoopManager.Config, super().config)

    def models(self) -> tuple[str, ...]:
        return tuple(self.config.models)

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        return RoundReport()


class ProviderJudge(Judge):
    name = "prov"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    def providers(self) -> frozenset[str]:
        return frozenset({"openai"})

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            Score.minimize(c.output.data["y"]).model_copy(update={"candidate_id": c.id})
            for c in candidates
        ]


class AliasedEnvTool(Tool):
    _human_only: ClassVar[frozenset[str]] = frozenset({"env"})

    class Config(EchoTool.Config):
        model_config = ConfigDict(
            extra="forbid", frozen=True, hide_input_in_errors=True, populate_by_name=True
        )
        env: list[str] = Field(
            default_factory=list,
            validation_alias=AliasChoices(
                "env_alias",
                AliasPath("env_path", "names"),
                "env_choice",
                AliasPath("env_choice_path", "names"),
            ),
        )

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        return Command(argv=["true"])

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        return ToolOutput(status=run.status)


class AdmissionTool(Tool):
    """Expose nested refs and controlled hook failures without running processes."""

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
        members: list[RefField] = []
        files: FileSet = FileSet()
        text: str = ""
        fail_at: str = ""
        deterministic: bool = False

    def __init__(self, config: BaseModel | dict[str, Any] | None = None, /) -> None:
        super().__init__(config)
        self._fail("constructor")

    @property
    def config(self) -> Config:
        return cast(AdmissionTool.Config, super().config)

    def _fail(self, site: str) -> None:
        if self.config.fail_at == site:
            raise RuntimeError(self.config.text)

    def is_deterministic(self, request: ToolRequest) -> bool:
        self._fail("determinism")
        return self.config.deterministic

    def _own_files(self) -> dict[str, str]:
        self._fail("files")
        return self.config.files.manifest

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        return Command(argv=["true"])

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        return ToolOutput(status=run.status)


class AdmissionManager(Manager):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
        instructions: str = ""
        fail: bool = False
        models: list[str] = []

    @property
    def config(self) -> Config:
        return cast(AdmissionManager.Config, super().config)

    def models(self) -> tuple[str, ...]:
        if self.config.fail:
            raise RuntimeError(self.config.instructions)
        return tuple(self.config.models)

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        return RoundReport()


def square(x: float) -> dict[str, float]:
    return {"x": x, "y": x * x}


def touch(marker: str) -> dict[str, str]:
    Path(marker).touch()
    return {"marker": marker}


class CommandRaisesTool(PyTool):
    def command(self, request: ToolRequest, workspace: Path) -> Command:
        raise RuntimeError("boom")


WRITE_THEN_SLEEP: str = WRITE.format("{'data': {'step': 1}}") + "; import time; time.sleep(600)"


def write_then_sleep(marker: Path, seconds: float = 600) -> str:
    return (
        WRITE.format("{'data': {'step': 1}}")
        + f"; from pathlib import Path; Path({str(marker)!r}).touch()"
        + f"; import time; time.sleep({seconds!r})"
    )


class ContextInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    value: Any
    defaulted: int = 7


class ContextTool(PyTool):
    input_model = ContextInputs

    def is_deterministic(self, request: ToolRequest) -> bool:
        return False


class AlwaysFailCheck(Check):
    name = "always-fail"

    class Config(FrozenModel):
        reason: str = "always fails"
        feedback: Literal["pass_fail", "reasons"] = "reasons"

    @property
    def config(self) -> Config:
        return cast(AlwaysFailCheck.Config, super().config)

    def __init__(self, config: BaseModel | Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.feedback = self.config.feedback

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        return CheckResult(check=self.name, passed=False, reasons=[self.config.reason])


def score_min_y(goal: Goal, candidate: Candidate) -> Score:
    return Score.minimize(candidate.output.data["y"])


def score_max_y(goal: Goal, candidate: Candidate) -> Score:
    return Score.maximize(candidate.output.data["y"])


class CountingTool(PyTool):
    calls: ClassVar[int] = 0

    class Config(PyTool.Config):  # ty: ignore[invalid-attribute-override]
        # These fixtures give the parent config's required code field a default.
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
        code: str = ""
        k: int = 1

    def preflight(self) -> None:
        CountingTool.calls += 1


class FailingPreflightTool(PyTool):
    class Config(PyTool.Config):  # ty: ignore[invalid-attribute-override]
        # These fixtures give the parent config's required code field a default.
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
        code: str = ""

    def preflight(self) -> None:
        raise RuntimeError("no license")
