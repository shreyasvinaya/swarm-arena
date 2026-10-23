import hashlib
import json
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
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import LLMScope, current_llm_scope
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
from swarm_arena._tooling.function_tool import FunctionTool
from swarm_arena._tooling.result_file import tool_output_from_result
from swarm_arena._tooling.tool import Action, Tool
from swarm_arena.manager import Manager, RoundReport
from swarm_arena.planning import Decision, Policy, RoundState


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


def join_sum() -> dict[str, float]:
    """Sum the assigned parents' value fields from their staged data files."""
    return {
        "value": sum(
            float(json.loads(path.read_text())["value"])
            for path in Path("parents").glob("*/data.json")
        )
    }


SCOPE_PROBE_SEEN: list[LLMScope] = []


class ScopeProbeJudge(Judge):
    name = "probe"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        scope = current_llm_scope()
        assert scope is not None
        SCOPE_PROBE_SEEN.append(scope)
        call = await scope.tranche.reserve_call(Amount(usd=0.01))
        await scope.tranche.settle_call(call, Amount(usd=0.01))
        scope.call_ids.append(f"llm-{len(SCOPE_PROBE_SEEN)}")
        return [
            Score.minimize(c.output.data["y"]).model_copy(update={"candidate_id": c.id})
            for c in candidates
        ]


class ConstantFingerprintTool(FunctionTool):
    """Fingerprint independently of request data for distinct-selection tests."""

    def fingerprint(self, output: ToolOutput) -> str:
        return f"timeout-{self.config.timeout_s}"


class RaisingFingerprintTool(FunctionTool):
    """Contain a hook failure without making otherwise useful outputs fail."""

    def fingerprint(self, output: ToolOutput) -> str:
        raise RuntimeError("fingerprint-secret-sentinel")


class AnchoredProviderJudge(ProviderJudge):
    def anchor_count(self) -> int:
        return 2


def goal_part_value(value: float) -> dict[str, float]:
    return {"value": value}


def goal_score_value(goal: Goal, candidate: Candidate) -> Score:
    return Score.minimize(candidate.output.data["value"])


def goal_score_negative_value(goal: Goal, candidate: Candidate) -> Score:
    return Score.maximize(candidate.output.data["value"])


class FailLaterRoundCheck(Check):
    name = "later"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        return CheckResult(
            check=self.name,
            passed=candidate.round == 1,
            reasons=[] if candidate.round == 1 else ["later round failed"],
        )


def line_value(a: float) -> dict[str, float]:
    """
    Return a line evaluation using the joiner's value key.

    Parameters
    ----------
    a : float
        Value to evaluate.

    Returns
    -------
    dict[str, float]
        Numeric output suitable for scoring and joining.

    Examples
    --------
    >>> line_value(3.0)
    {'value': 3.0}
    """
    return {"value": a}


def part_value(part: str, v: float) -> dict[str, float]:
    """
    Return one part's value for a later sum join.

    Parameters
    ----------
    part : str
        Part identifier retained in the recorded request.
    v : float
        Numeric value contributed by this part.

    Returns
    -------
    dict[str, float]
        Part output using the same key as the joined result.

    Examples
    --------
    >>> part_value('a', 12.0)
    {'value': 12.0}
    """
    return {"value": v}


def score_value(goal: Goal, candidate: Candidate) -> Score:
    """
    Prefer candidates with larger values, for either a part or the main goal.

    Parameters
    ----------
    goal : Goal
        Main or part goal being scored.
    candidate : Candidate
        Candidate whose output contains a numeric value.

    Returns
    -------
    Score
        Maximization score on the shared value scale.

    Examples
    --------
    >>> score_value(GOAL, cand(data={'value': 12.0})).key
    (12.0,)
    """
    return Score.maximize(candidate.output.data["value"])


class FeedbackCheck(AlwaysFailCheck):
    """Return passing feedback under a configurable visibility policy."""

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        return CheckResult(
            check=self.name,
            passed=True,
            reasons=[self.config.reason],
            details={"private": self.config.reason},
        )


class ThreeDollarTool(Tool):
    """
    Charge three dollars for nonzero inputs, ten cents for the default input.

    Preflight estimates the default empty request; actual requests have their
    own estimates and matching reported usage through the normal engine path.

    Examples
    --------
    >>> ThreeDollarTool().estimate(ToolRequest(inputs={'n': 1})).usd
    3.0
    """

    name = "paid-value"
    description = "Evaluate a value at the input's stated price."
    input_model = PyInputs

    def estimate(self, request: ToolRequest) -> Estimate:
        """
        Estimate the selected input's price.

        Parameters
        ----------
        request : ToolRequest
            Proposed invocation.

        Returns
        -------
        Estimate
            Three dollars for a nonzero input, otherwise ten cents; zero tokens.

        Examples
        --------
        >>> ThreeDollarTool().estimate(ToolRequest()).tokens
        0
        >>> ThreeDollarTool().estimate(ToolRequest()).usd
        0.1
        """
        return Estimate(tokens=0, usd=3.0 if request.inputs.get("n", 0) else 0.1)

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        """
        Execute a real local process without an external provider.

        Parameters
        ----------
        request : ToolRequest
            Validated invocation.
        workspace : Path
            Staged working directory.

        Returns
        -------
        Command
            A successful no-op command.

        Examples
        --------
        >>> ThreeDollarTool().command(ToolRequest(), Path('.')).argv
        ['true']
        """
        return Command(argv=["true"])

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        """
        Report the request value and its actual connector usage.

        Parameters
        ----------
        request : ToolRequest
            Invocation containing the numeric value.
        workspace : Path
            Completed working directory.
        run : CompletedRun
            Real process outcome and timestamps.

        Returns
        -------
        ToolOutput
            Scorable value and explicit actual usage for settlement.

        Examples
        --------
        >>> from datetime import UTC, datetime
        >>> now = datetime(2026, 1, 1, tzinfo=UTC)
        >>> run = CompletedRun(returncode=0, status=RunStatus.OK,
        ...     stdout_path=Path('stdout'), stderr_path=Path('stderr'),
        ...     started_at=now, ended_at=now)
        >>> ThreeDollarTool().collect(ToolRequest(inputs={'n': 1}), Path('.'), run).usage.usd
        3.0
        """
        from swarm_arena.tool import Usage

        return ToolOutput(
            status=run.status,
            data={"value": request.inputs["n"]},
            usage=Usage(usd=self.estimate(request).usd),
            started_at=run.started_at,
            ended_at=run.ended_at,
        )


class ReportRefusalManager(Manager):
    """
    Offer two distinct requests and record a public action's refusal identifier.

    Examples
    --------
    >>> ReportRefusalManager().models()
    ()
    """

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        """
        Run sequential paid requests until the budget refuses one.

        Parameters
        ----------
        ctx : AgentContext
            Public action context supplied by the engine.

        Returns
        -------
        RoundReport
            Summary after both requests or a recorded limit refusal.

        Examples
        --------
        >>> import anyio
        >>> from unittest.mock import AsyncMock, Mock
        >>> ctx = Mock(run_tool=AsyncMock())
        >>> anyio.run(ReportRefusalManager().run_round, ctx).summary
        'both requests admitted'
        """
        from swarm_arena.manager import LimitReached

        evidence: list[str] = []
        for n in (1, 2):
            try:
                candidate = await ctx.run_tool({"n": n})
            except LimitReached as error:
                await ctx.post_learning(
                    title="Tool request refused", text=error.limit, evidence=evidence
                )
                return RoundReport(summary="request refused")
            evidence.append(candidate.id)
        return RoundReport(summary="both requests admitted")


def sleepy(seconds: float, marker: str) -> dict[str, float]:
    """
    Publish process identity atomically and sleep only on the first invocation.

    Parameters
    ----------
    seconds : float
        Maximum first-invocation delay in seconds. A sibling ``.release`` file
        ends the wait early so lifecycle tests can release committed work.
    marker : str
        Absolute path to the PID/PGID JSON marker. An existing marker skips sleep.

    Returns
    -------
    dict[str, float]
        Judgeable value data.

    Examples
    --------
    >>> sleepy(0.0, "/tmp/sleepy-example.json")  # doctest: +SKIP
    {'value': 1.0}
    """
    path = Path(marker)
    if not path.exists():
        pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        pending.write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgid(0)}))
        pending.replace(path)
        deadline = time.monotonic() + seconds
        release = path.with_suffix(".release")
        while not release.exists():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.02))
    return {"value": 1.0}


def _wait_preparation_marker(path: Path) -> None:
    """Bound every file barrier so failed tests cannot strand a daemon forever."""
    deadline = time.monotonic() + 60
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("preparation barrier timed out")
        time.sleep(0.02)


class BlockedPreparationTool(Tool):
    """
    Block command preparation until a file releases the executor-owned daemon.

    Examples
    --------
    >>> BlockedPreparationTool({"root": "/tmp/probe"}).name
    'blocked-preparation'
    """

    name = "blocked-preparation"
    description = "Coordinate pre-launch cancellation through files."
    input_model = PyInputs

    class Config(BaseModel):
        """
        Locate coordination files for one isolated test.

        Parameters
        ----------
        root : str
            Directory containing entered, release, returned and launch markers.

        Examples
        --------
        >>> BlockedPreparationTool.Config(root="/tmp/probe").root
        '/tmp/probe'
        """

        model_config = ConfigDict(extra="forbid", frozen=True)
        root: str

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        """
        Return a launch-marker command after the preparation release signal.

        Parameters
        ----------
        request : ToolRequest
            Invocation being prepared.
        workspace : Path
            Executor-owned directory, retained while this callback is blocked.

        Returns
        -------
        Command
            A command that must never launch after cancellation.

        Examples
        --------
        >>> tool.command(ToolRequest(), workspace)  # doctest: +SKIP
        """
        root = Path(cast(BlockedPreparationTool.Config, self.config).root)
        entered = root / "preparation-entered.json"
        pending = root / ".preparation-entered.tmp"
        pending.write_text(json.dumps(str(workspace)))
        pending.replace(entered)
        _wait_preparation_marker(root / "preparation-release")
        (workspace / "late-write").write_text("retained")
        command = Command(
            argv=[
                "python",
                "-c",
                f"from pathlib import Path; Path({str(root / 'late-launch')!r}).touch()",
            ]
        )
        (root / "preparation-returned").touch()
        return command

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        """
        Supply value data if an uncancelled invocation finishes.

        Parameters
        ----------
        request : ToolRequest
            Invocation being collected.
        workspace : Path
            Executor workspace.
        run : CompletedRun
            Process outcome.

        Returns
        -------
        ToolOutput
            Judgeable output with the actual process status.

        Examples
        --------
        >>> from datetime import UTC, datetime
        >>> now = datetime(2026, 1, 1, tzinfo=UTC)
        >>> run = CompletedRun(returncode=0, status=RunStatus.OK,
        ...     stdout_path=Path('stdout'), stderr_path=Path('stderr'),
        ...     started_at=now, ended_at=now)
        >>> tool = BlockedPreparationTool({"root": "/tmp/probe"})
        >>> tool.collect(ToolRequest(), Path('.'), run).data
        {'value': 1.0}
        """
        return ToolOutput(status=run.status, data={"value": 1.0})


class PreparationBarrierPolicy(Policy):
    """
    Keep the CLI alive after settlement while abandoned preparation returns.

    Examples
    --------
    >>> PreparationBarrierPolicy({"root": "/tmp/probe"}).config.root
    '/tmp/probe'
    """

    Config = BlockedPreparationTool.Config

    def after_round(self, state: RoundState) -> Decision:
        """
        Wait for explicit release after the first round, then retain all agents.

        Parameters
        ----------
        state : RoundState
            Current round's committed candidate evidence.

        Returns
        -------
        Decision
            An empty pruning decision.

        Examples
        --------
        >>> policy.after_round(state)  # doctest: +SKIP
        """
        if state.round == 1:
            root = Path(cast(BlockedPreparationTool.Config, self.config).root)
            (root / "barrier-entered").touch()
            _wait_preparation_marker(root / "barrier-release")
        return Decision()


def wait_for_file(path: str, timeout_s: float = 60.0) -> dict[str, Any]:
    """
    Wait for a published file with a bounded, monotonic readiness poll.

    Parameters
    ----------
    path : str
        File whose existence signals readiness, such as a committed round page.
    timeout_s : float, default=60.0
        Maximum waiting time in seconds. Polls occur every 0.1 seconds.

    Returns
    -------
    dict[str, Any]
        Elapsed waiting time under the ``waited_s`` key.

    Raises
    ------
    TimeoutError
        If the file has not appeared by the deadline.

    Examples
    --------
    >>> wait_for_file(__file__)["waited_s"] >= 0  # doctest: +SKIP
    True
    """
    started = time.monotonic()
    target = Path(path)
    while not target.is_file():
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("file readiness timed out")
        time.sleep(min(0.1, remaining))
    return {"waited_s": time.monotonic() - started}


def score_one(goal: Goal, candidate: Candidate) -> Score:
    """
    Give every candidate the same maximization score.

    Parameters
    ----------
    goal : Goal
        Goal being judged.
    candidate : Candidate
        Candidate whose successful execution is sufficient for this fixture.

    Returns
    -------
    Score
        Constant maximization score of one.

    Examples
    --------
    >>> score_one(GOAL, cand(data={"waited_s": 0.0})).key
    (1.0,)
    """
    return Score.maximize(1.0)


class ProviderPyTool(PyTool):
    """Declare the provider used by scripted failure/recovery integration tests."""

    def providers(self) -> frozenset[str]:
        return frozenset({"fake"})


WORKER_SCRIPTS: dict[str, str] = {
    "partial_on_sigterm.py": """
import argparse, json, os, signal, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--ready-file')
args = parser.parse_args()
n = 0
result = Path(os.environ['SWARM_ARENA_RESULT'])
def publish():
    pending = result.with_suffix('.tmp')
    pending.write_text(json.dumps({'data': {'step': n}}))
    pending.replace(result)
def stop(signum, frame):
    publish()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
while True:
    n += 1
    publish()
    if args.ready_file:
        Path(args.ready_file).touch()
    time.sleep(0.2)
""",
    "ignore_sigterm.py": """
import argparse, os, signal, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--pid-file')
parser.add_argument('--fifo-stderr', action='store_true')
args = parser.parse_args()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if args.fifo_stderr:
    os.remove('swarm/stderr.log')
    os.mkfifo('swarm/stderr.log')
if args.pid_file:
    Path(args.pid_file).write_text(str(os.getpid()))
time.sleep(300)
""",
    "spawn_grandchild.py": """
import argparse, signal, subprocess, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--pid-file', required=True)
parser.add_argument('--exit-after', type=float, default=300)
parser.add_argument('--ignore-term', action='store_true')
args = parser.parse_args()
if args.ignore_term:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen(['sleep', '300'])
Path(args.pid_file).write_text(str(child.pid))
time.sleep(args.exit_after)
""",
    "report_env.py": """
import json, os
selected = ('SWARM_ARENA_CANDIDATE_ID', 'CUDA_VISIBLE_DEVICES', 'SA_TEST_SECRET')
with open(os.environ['SWARM_ARENA_RESULT'], 'w') as f:
    json.dump({'data': {'env': {k: os.environ.get(k) for k in selected},
        'pgid': os.getpgrp(), 'parent_pgid': os.getpgid(os.getppid())}}, f)
""",
}


def _write_module(path: Path) -> None:
    """Write a standalone annotated FunctionTool fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('def f(a: int) -> dict[str, int]:\n    return {"a": a}\n')


class EditSpyTool(PyTool):
    """Expose the live edit contents seen by command and collection hooks."""

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        (workspace / "edit_spy.json").write_text(json.dumps(request.edits))
        return super().command(request, workspace)

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        return ToolOutput(
            status=run.status,
            data={
                "command_edits": json.loads((workspace / "edit_spy.json").read_text()),
                "collect_edits": dict(request.edits),
            },
        )


class StallCollectTool(PyTool):
    """Hold cancellation collection until the driver must force-stop the worker."""

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        script = "spawn_grandchild.py"
        argv = [script, "--pid-file", request.inputs["pid_file"], "--ignore-term"]
        code = f"import sys; sys.argv = {argv!r}\n" + WORKER_SCRIPTS[script]
        return Command(argv=[sys.executable, "-c", code])

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        time.sleep(60)
        return super().collect(request, workspace, run)
