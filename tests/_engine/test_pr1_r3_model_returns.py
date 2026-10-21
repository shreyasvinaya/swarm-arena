"""Returned subclasses cannot run serializers or field access on the driver loop."""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from _engine.test_goal import _plan
from _fakes.tools import AlwaysPass, FixedJudge, cand
from pydantic import BaseModel

from swarm_arena import (
    Goal,
    Plan,
    ProgramJudge,
    Ref,
    RunConfig,
    ScriptedManager,
    ScriptJudge,
    Swarm,
)
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.planning_types import Decision, History, RoundReport
from swarm_arena._model.results import Candidate, CheckResult, Learning, Score
from swarm_arena._planning.planner import Planner
from swarm_arena._planning.policies import SuccessiveHalving
from swarm_arena.executor import Job, LocalExecutor
from swarm_arena.manager import AgentContext
from swarm_arena.scoring import CheckContext, JudgeContext, Lexicographic
from swarm_arena.tool import RunStatus, ToolOutput


def _subclass(base: type[BaseModel], calls: list[int], loop: int) -> type[BaseModel]:
    def read(self: BaseModel, name: str) -> Any:
        if name in base.model_fields or name == "base_dir":
            assert threading.get_ident() != loop, f"loop field read: {name}"
        return object.__getattribute__(self, name)

    def dump(self: BaseModel, **kwargs: Any) -> dict[str, Any]:
        calls.append(threading.get_ident())
        assert calls[-1] != loop, "loop serializer"
        return BaseModel.model_dump(self, **kwargs)

    return type("Returned" + base.__name__, (base,), {"model_dump": dump, "__getattribute__": read})


@pytest.mark.parametrize(
    "site",
    [
        "program",
        "judge",
        "script",
        "lex",
        "check",
        "output",
        "command",
        "plan",
        "initial_plan",
        "decision",
        "report",
    ],
)
async def test_public_run_detaches_returned_models(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    loop = threading.get_ident()
    calls: list[int] = []
    base = (
        Plan
        if site in {"plan", "initial_plan"}
        else Decision
        if site == "decision"
        else RoundReport
        if site == "report"
        else CheckResult
        if site == "check"
        else ToolOutput
        if site in {"output", "command"}
        else Score
    )
    returned = _subclass(base, calls, loop)
    current = _plan(teams=1, rounds=2 if site == "plan" else 1)
    replanner = None

    async def result(model: BaseModel) -> Any:
        return await run_sync_daemon(returned.model_validate, model.model_dump())

    async def judge(
        self: Any, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            await result(Score.maximize(1).model_copy(update={"candidate_id": c.id}))
            for c in candidates
        ]

    if site == "program":

        def score(goal: Goal, candidate: Candidate) -> Any:
            return returned.model_validate(Score.maximize(1).model_dump())

        monkeypatch.setattr(ProgramJudge, "_load_function", lambda self: score)
    elif site in {"judge", "lex"}:
        monkeypatch.setattr(FixedJudge, "judge", judge)
        member = FixedJudge(FixedJudge.Config(value=1))
        selected = Lexicographic([member]) if site == "lex" else member
        current = current.model_copy(
            update={"judges": [Ref.of(selected).model_copy(update={"name": "y"})]}
        )
    elif site == "script":

        async def script(self: ScriptJudge, candidate: Candidate, ctx: JudgeContext) -> Score:
            return await result(Score.maximize(1).model_copy(update={"candidate_id": candidate.id}))

        monkeypatch.setattr(ScriptJudge, "_score", script)
        current = current.model_copy(
            update={
                "judges": [Ref.of(ScriptJudge(command=["unused"])).model_copy(update={"name": "y"})]
            }
        )
    elif site == "check":

        async def check(self: AlwaysPass, candidate: Candidate, ctx: CheckContext) -> CheckResult:
            return await result(CheckResult(check="check", passed=True))

        monkeypatch.setattr(AlwaysPass, "check", check)
        current = current.model_copy(update={"checks": [Ref.of(AlwaysPass())]})
    elif site in {"output", "command"}:
        submit = LocalExecutor.submit

        async def output(self: LocalExecutor, job: Job) -> ToolOutput:
            if site == "command" and job.kind == "command":
                return await result(ToolOutput(status=RunStatus.OK, data={"value": 1}))
            value = await submit(self, job)
            return await result(value) if site == "output" else value

        monkeypatch.setattr(LocalExecutor, "submit", output)
        if site == "command":
            current = current.model_copy(
                update={
                    "judges": [
                        Ref.of(ScriptJudge(command=["unused"])).model_copy(update={"name": "y"})
                    ]
                }
            )
    elif site in {"plan", "initial_plan"}:
        proposed = await result(current)
        if site == "initial_plan":
            current = proposed
        else:

            class ReturnsPlan(Planner):
                async def plan(
                    self, goal: Goal, config: RunConfig, history: History | None
                ) -> Plan:
                    return proposed

            replanner = ReturnsPlan()
    elif site == "decision":
        proposed = await result(Decision())
        monkeypatch.setattr(SuccessiveHalving, "after_round", lambda self, state: proposed)
    else:
        run_round = ScriptedManager.run_round

        async def report(self: ScriptedManager, ctx: AgentContext) -> RoundReport:
            await run_round(self, ctx)
            return await result(RoundReport(summary="detached"))

        monkeypatch.setattr(ScriptedManager, "run_round", report)
    finished = await Swarm(run_config, replanner=replanner).run(current, run_id="detached")
    assert finished["g"]
    assert calls
    assert all(thread != loop for thread in calls)
    for path in (Path(run_config.run_root) / "detached/journal").glob("*.jsonl"):
        text = path.read_text()
        assert "loop field read" not in text
        assert "agent error" not in text
        assert "manager error" not in text
        assert "policy error" not in text
        assert "planner error" not in text


@pytest.mark.parametrize(
    "base", [Score, CheckResult, ToolOutput, Plan, Decision, RoundReport, Learning]
)
async def test_detachment_returns_exact_core_model(base: type[BaseModel]) -> None:
    loop = threading.get_ident()
    calls: list[int] = []
    kind = _subclass(base, calls, loop)
    values: dict[type[BaseModel], BaseModel] = {
        Score: Score.maximize(1),
        CheckResult: CheckResult(check="c", passed=True),
        ToolOutput: ToolOutput(status=RunStatus.OK),
        Plan: _plan(teams=1),
        Decision: Decision(),
        RoundReport: RoundReport(),
        Learning: Learning(
            id="L",
            namespace="n",
            run_id="r",
            goal_id="g",
            round=1,
            author="t/a",
            evidence=["c"],
            title="hint",
            text="text",
        ),
    }
    original = await run_sync_daemon(kind.model_validate, values[base].model_dump())
    detached = await run_sync_daemon(_detach_model, base, original)
    assert type(detached) is base
    assert detached.model_dump() == values[base].model_dump()
    assert calls
    assert all(thread != loop for thread in calls)


@pytest.mark.parametrize("site", ["program", "command", "lex"])
def test_new_sync_detachment_helpers_preserve_interrupt(site: str) -> None:
    from swarm_arena._scoring.check import _prepare_command_output
    from swarm_arena._scoring.program_judge import _score_copy

    failure = KeyboardInterrupt("serializer")

    class InterruptScore(Score):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            raise failure

    class InterruptOutput(ToolOutput):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            raise failure

    candidate = cand()
    score = InterruptScore.maximize(1).model_copy(update={"candidate_id": candidate.id})

    def invoke() -> None:
        if site == "program":
            _score_copy(lambda goal, candidate: score, _plan(teams=1).goal, candidate)
        elif site == "command":
            _prepare_command_output(InterruptOutput(status=RunStatus.OK))
        else:
            Lexicographic._prepare_member([score], [candidate], "member")

    with pytest.raises(KeyboardInterrupt) as caught:
        invoke()
    assert caught.value is failure


async def test_detachment_removes_nested_scalar_subclasses() -> None:
    class UserString(str):
        def __str__(self) -> str:
            return self

    class UserBytes(bytes):
        def __bytes__(self) -> bytes:
            return self

    class Payload(BaseModel):
        value: Any

    value = Payload(value={"label": UserString("label"), "raw": UserBytes(b"raw")})
    copied = await run_sync_daemon(_detach_model, Payload, value)
    assert type(copied.value["label"]) is str
    assert type(copied.value["raw"]) is bytes
