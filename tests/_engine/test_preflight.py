import threading
from typing import Any

import anyio
import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import (
    AlwaysPass,
    CountingTool,
    FixedJudge,
    NoopManager,
    RecordingExecutor,
)

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    Goal,
    PartSpec,
    Plan,
    Ref,
    RoundSpec,
    RunConfig,
    TeamSpec,
)
from swarm_arena._engine.preflight import run_preflight
from swarm_arena._exec.executor import Executor


def _plan(
    *,
    agents: int = 1,
    same_config: bool = False,
    config: dict[str, Any] | None = None,
    tool: str = "_fakes.tools:CountingTool",
) -> Plan:
    return Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[
            TeamSpec(
                name="t",
                rationale="Explore",
                agents=[
                    AgentSpec(
                        name=f"a{i}",
                        rationale="Explore",
                        tool=Ref(path=tool, config=config or {}),
                        manager=Ref(path="_fakes.tools:NoopManager"),
                    )
                    for i in range(agents)
                ],
            )
        ],
        judges=[Ref(path="_fakes.tools:FixedJudge", config={"value": 1})],
    )


def _executor() -> Executor:
    return RecordingExecutor()


async def test_preflight_once_per_distinct_config() -> None:
    CountingTool.calls = 0
    await run_preflight(
        [_plan(agents=3, same_config=True), _plan(config={"k": 2})], RunConfig(), _executor()
    )
    assert CountingTool.calls == 2


async def test_preflight_failures_collected() -> None:
    plan = _plan(tool="_fakes.tools:FailingPreflightTool")
    plan.checks.append(Ref(path="_fakes.tools:PreflightFailsCheck"))
    with pytest.raises(
        RuntimeError, match=r"preflight failed:\n- .*PreflightFailsCheck: RuntimeError"
    ) as caught:
        await run_preflight([plan], RunConfig(), _executor())
    assert "- _fakes.tools:FailingPreflightTool: RuntimeError" in str(caught.value)
    assert "no license" not in str(caught.value)


async def test_normalized_defaults_and_ref_names_do_not_repeat_hooks() -> None:
    CountingTool.calls = 0
    first, second = _plan(), _plan(config={"k": 1, "code": "", "grace_s": 10.0})
    agent = second.teams[0].agents[0]
    second.teams[0].agents[0] = agent.model_copy(
        update={"tool": agent.tool.model_copy(update={"name": "other-name"})}
    )
    await run_preflight([first, second], RunConfig(), _executor())
    assert CountingTool.calls == 1


async def test_effective_rounds_converge_and_later_revisions() -> None:
    CountingTool.calls = 0
    plan = _plan()
    plan.rounds.extend([RoundSpec(tool_overrides={"k": 2}), RoundSpec(tool_overrides={"k": 3})])
    plan = plan.model_copy(
        update={
            "converge": ConvergeSpec(
                mode="team",
                team=plan.teams[0].model_copy(update={"name": "join"}),
                round=RoundSpec(tool_overrides={"k": 4}),
            )
        }
    )
    await run_preflight([plan], RunConfig(), _executor())
    assert CountingTool.calls == 3
    plan = plan.model_copy(
        update={
            "revision": 1,
            "rounds": [RoundSpec(tool_overrides={"k": 5})],
            "converge": plan.converge.model_copy(update={"round": None}),
        }
    )
    await run_preflight([plan], RunConfig(), _executor())
    assert CountingTool.calls == 4


async def test_all_hook_interfaces_and_no_policy_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    for cls in (AlwaysPass, FixedJudge, NoopManager, RecordingExecutor):

        def hook(self: object, label: str = cls.__name__) -> None:
            calls.append(label)
            assert threading.current_thread().daemon

        monkeypatch.setattr(cls, "preflight", hook)
    plan = _plan(agents=3)
    plan.checks.append(Ref(path="_fakes.tools:AlwaysPass"))
    plan.parts.extend(
        [
            PartSpec(
                name="p", goal=Goal(id="p", namespace="n"), checks=plan.checks, judges=plan.judges
            )
        ]
    )
    plan = plan.model_copy(update={"policy": Ref(path="never_import_this:Policy")})
    await run_preflight([plan, plan], RunConfig(), _executor())
    assert sorted(calls) == ["AlwaysPass", "FixedJudge", "NoopManager", "RecordingExecutor"]


async def test_hook_and_constructor_errors_are_type_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: object) -> None:
        raise RuntimeError("PREFLIGHT_SECRET")

    monkeypatch.setattr(CountingTool, "preflight", fail)
    with pytest.raises(RuntimeError, match="CountingTool: RuntimeError") as caught:
        await run_preflight([_plan()], RunConfig(), _executor())
    assert_secret_absent(caught.value, "PREFLIGHT_SECRET", sanitized=True)

    def init(self: object, config: object) -> None:
        raise ValueError("PREFLIGHT_SECRET")

    monkeypatch.setattr(CountingTool, "__init__", init)
    with pytest.raises(RuntimeError, match="CountingTool: ValueError") as caught:
        await run_preflight([_plan()], RunConfig(), _executor())
    assert_secret_absent(caught.value, "PREFLIGHT_SECRET", sanitized=True)


async def test_hung_hook_is_abandonable_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()

    def block(self: object) -> None:
        assert threading.current_thread().daemon
        entered.set()
        release.wait(10)

    monkeypatch.setattr(CountingTool, "preflight", block)
    try:
        with anyio.move_on_after(0.2) as scope:
            await run_preflight([_plan()], RunConfig(), _executor())
        assert entered.is_set()
        assert scope.cancel_called
    finally:
        release.set()


async def test_cancelled_hook_does_not_continue_to_later_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, finished = threading.Event(), threading.Event()
    called: list[str] = []

    def block(self: object) -> None:
        release.wait(10)
        finished.set()

    def later(self: object) -> None:
        called.append("executor")

    monkeypatch.setattr(CountingTool, "preflight", block)
    monkeypatch.setattr(RecordingExecutor, "preflight", later)
    try:
        with anyio.move_on_after(0.1) as scope:
            await run_preflight([_plan()], RunConfig(), _executor())
        assert scope.cancel_called
    finally:
        release.set()
    assert await anyio.to_thread.run_sync(finished.wait, 1)
    await anyio.sleep(0.02)
    assert called == []


async def test_part_overrides_and_executor_failures_are_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(self: object) -> None:
        raise LookupError("EXECUTOR_SECRET")

    monkeypatch.setattr(RecordingExecutor, "preflight", fail)
    plan = _plan()
    plan.parts.append(
        PartSpec(
            name="p",
            goal=Goal(id="p", namespace="n"),
            checks=[Ref(path="_fakes.tools:PreflightFailsCheck")],
        )
    )
    with pytest.raises(RuntimeError, match="PreflightFailsCheck: RuntimeError") as caught:
        await run_preflight([plan], RunConfig(), _executor())
    assert "swarm_arena.executor:LocalExecutor: LookupError" in str(caught.value)
    assert_secret_absent(caught.value, "EXECUTOR_SECRET", sanitized=True)
