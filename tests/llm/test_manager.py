"""Stateless SDK managers, owned cleanup, and public run/resume integration."""

import asyncio
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

pytest.importorskip("agents")

from _agent.conftest import ctx_env as ctx_env
from _agent.conftest import rt_env as rt_env
from _fakes.agent_ctx import FakeAgentContext
from _fakes.engine import CtxEnv
from _fakes.llm_run import HeldOperation, open_llm_run
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import PyTool
from agents.exceptions import AgentsException
from pydantic import ValidationError

from swarm_arena import Limits, Ref, RunConfig, ScriptedManager, Swarm
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import llm_scope
from swarm_arena._errors import (
    ConfigError,
    Fenced,
    _core_category,
    _core_error,
    _is_core_control,
    _SignalCategory,
    _stop_cleanup_error,
)
from swarm_arena._model.base import FrozenModel, JsonValue
from swarm_arena.config import ModelCatalog, ModelSpec
from swarm_arena.llm import AgentsSDKManager, BudgetExceeded
from swarm_arena.manager import AgentContext, LimitReached, Manager, RoundReport
from swarm_arena.testing import FakeModel, FakeTurn
from swarm_arena.tool import Action, Usage


class Inputs(FrozenModel):
    x: float = 0


SPEC = ModelSpec(
    model="fake/opus",
    roles=["manager"],
    max_tokens=100,
    prices=ModelSpec.Prices(input=1, output=3),
)


def config(*, limits: Limits | None = None, **fields: Any) -> RunConfig:
    return RunConfig(
        models=ModelCatalog({"opus": SPEC}),
        limits=limits or Limits(),
        settle_timeout_s=fields.pop("settle_timeout_s", RunConfig().settle_timeout_s),
        **fields,
    )


def context(**fields: Any) -> AgentContext:
    return cast(AgentContext, FakeAgentContext(input_model=Inputs, **fields))


def finish(summary: str = "tried lr 0.01") -> FakeTurn:
    return FakeTurn(tool_calls=(("finish_round", {"summary": summary}),))


def register(turns: list[FakeTurn]) -> FakeModel:
    fake = FakeModel(turns)
    fake.register("opus")
    return fake


async def test_round_ends_on_finish_round(tmp_path: Path) -> None:
    fake = register([finish(), FakeTurn(text="unused")])
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            report = await AgentsSDKManager(model="opus").run_round(context())
        assert report == RoundReport(summary="tried lr 0.01")
        assert len(fake.calls) == 1
        assert scope.call_ids == [call.id for call in run.calls()]
        assert run.charged(scope) == sum(
            (Amount(tokens=c.usage.total_tokens, usd=c.cost_usd) for c in run.calls()), Amount()
        )


async def test_each_round_starts_a_fresh_conversation(tmp_path: Path) -> None:
    fake = register(
        [
            FakeTurn(text="round-1 turn text", tool_calls=(("save_notes", {"text": "remember"}),)),
            finish(),
        ]
    )
    manager = AgentsSDKManager(model="opus")
    async with open_llm_run(tmp_path, config=config()) as run:
        for number in (1, 2):
            with llm_scope(run.scope("manager", round=number)):
                await manager.run_round(context(round=number))
    assert len(fake.calls) == 4
    assert "round-1 turn text" not in json.dumps(fake.calls[2]["input"])


async def test_system_prompt_identical_across_rounds(tmp_path: Path) -> None:
    fake = register([FakeTurn(tool_calls=(("my_candidates", {}),)), finish()])
    async with open_llm_run(tmp_path, config=config()) as run:
        manager = AgentsSDKManager(model="opus", instructions="Explore.", seed_inputs={"x": 1})
        for number in (1, 2):
            with llm_scope(run.scope("manager", round=number)):
                await manager.run_round(context(round=number, notes=f"notes {number}"))
    assert fake.calls[0]["system"] == fake.calls[2]["system"]
    assert fake.calls[0]["input"] != fake.calls[2]["input"]


async def test_max_turns_from_limits(tmp_path: Path) -> None:
    fake = register([FakeTurn(tool_calls=(("my_candidates", {}),))] * 3)
    async with open_llm_run(tmp_path, config=config(limits=Limits(max_turns_per_round=2))) as run:
        with llm_scope(run.scope("manager")), pytest.raises(LimitReached) as e:
            await AgentsSDKManager(model="opus").run_round(context())
        assert e.value.limit == "max_turns_per_round"
        assert _is_core_control(e.value)
        assert len(fake.calls) == len(run.calls()) == 2


async def test_limit_reached_in_action_ends_round_with_original_limit(
    tmp_path: Path,
    ctx_env: CtxEnv,
) -> None:
    fake = register([FakeTurn(tool_calls=(("run_tool", {"inputs": {"x": 1}}),)), finish()])
    ctx = ctx_env.ctx(tool_runs_used=ctx_env.rt.limits.max_tool_runs_per_round)
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")), pytest.raises(LimitReached) as e:
            await AgentsSDKManager(model="opus").run_round(ctx)
        assert e.value.limit == "max_tool_runs_per_round"
        assert not isinstance(e.value, AgentsException)
        assert _is_core_control(e.value)
        assert len(fake.calls) == len(run.calls()) == 1


async def test_budget_exceeded_ends_round(tmp_path: Path) -> None:
    fake = register([finish()])
    async with open_llm_run(tmp_path, config=config(limits=Limits(max_usd=1e-6))) as run:
        with llm_scope(run.scope("manager")), pytest.raises(BudgetExceeded):
            await AgentsSDKManager(model="opus").run_round(context())
        assert fake.calls == []
        assert run.calls() == []


class HeldModel(FakeModel):
    """Signal that metering has admitted the call, then await cancellation."""

    def __init__(self, error: BaseException | None = None) -> None:
        super().__init__([])
        self.entered = anyio.Event()
        self.error = error

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        self.entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            if self.error is not None:
                raise self.error


async def test_wall_time_limit(tmp_path: Path) -> None:
    fake = HeldModel()
    fake.register("opus")
    async with open_llm_run(tmp_path, config=config(limits=Limits(round_timeout_s=0.05))) as run:
        with llm_scope(run.scope("manager")), pytest.raises(LimitReached) as e:
            await AgentsSDKManager(model="opus").run_round(context())
        assert e.value.limit == "round_timeout_s"


@pytest.mark.parametrize("native", [False, True])
async def test_cancel_mid_call_charges_reservation_in_full(tmp_path: Path, native: bool) -> None:
    fake = HeldModel()
    fake.register("opus")
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            if native:
                task = asyncio.create_task(AgentsSDKManager(model="opus").run_round(context()))
                try:
                    with anyio.fail_after(30):
                        await fake.entered.wait()
                    task.cancel()
                    with anyio.fail_after(30), pytest.raises(asyncio.CancelledError):
                        await task
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            else:
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(AgentsSDKManager(model="opus").run_round, context())
                    with anyio.fail_after(30):
                        await fake.entered.wait()
                    tasks.cancel_scope.cancel()
        (call,) = run.calls()
        assert call.error is not None
        assert call.error.startswith("estimated:")
        assert run.charged(scope).usd == pytest.approx(call.cost_usd)
        assert scope.call_ids == [call.id]


async def test_round_timeout_charges_the_cut_call_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena.llm import _manager

    timers: list[anyio.CancelScope] = []
    original_timer_scope = _manager._local_timer_scope

    def observed_timer(timer: anyio.CancelScope, *, relay_provider_stop: bool = False) -> Any:
        timers.append(timer)
        return original_timer_scope(timer, relay_provider_stop=relay_provider_stop)

    monkeypatch.setattr(_manager, "_local_timer_scope", observed_timer)
    fake = HeldModel()
    fake.register("opus")
    async with open_llm_run(
        tmp_path,
        config=config(limits=Limits(round_timeout_s=60)),
    ) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            task = asyncio.create_task(AgentsSDKManager(model="opus").run_round(context()))
        try:
            with anyio.fail_after(30):
                await fake.entered.wait()
            # Expire the real round deadline only after the call is admitted.
            # A 50 ms startup deadline could legitimately cut off admission itself.
            (timer,) = timers
            timer.deadline = anyio.current_time()
            with anyio.fail_after(30), pytest.raises(LimitReached) as e:
                await task
            (call,) = run.calls()
            assert e.value.limit == "round_timeout_s"
            assert timer.cancel_called
            assert call.error is not None
            assert call.error.startswith("estimated:")
            assert run.charged(scope).usd == pytest.approx(call.cost_usd)
            assert scope.call_ids == [call.id]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_alias_without_manager_role_fails_before_any_call(tmp_path: Path) -> None:
    fake = register([finish()])
    cfg = config().model_copy(
        update={"models": ModelCatalog({"opus": replace_spec(roles=["judge"])})}
    )
    async with open_llm_run(tmp_path, config=cfg) as run:
        with llm_scope(run.scope("manager")), pytest.raises(ValueError, match="LLM preparation"):
            await AgentsSDKManager(model="opus").run_round(context())
        assert fake.calls == []
        assert run.calls() == []


def replace_spec(**fields: Any) -> ModelSpec:
    return ModelSpec.model_validate({**SPEC.model_dump(), **fields})


async def test_no_output_type_and_finish_round_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents import FunctionTool

    from swarm_arena.llm import _manager

    original = _manager.Runner.run

    async def inspected(agent: Any, *args: Any, **kwargs: Any) -> Any:
        assert agent.output_type is None
        assert not agent.handoffs
        assert all(isinstance(tool, FunctionTool) for tool in agent.tools)
        assert agent.tool_use_behavior == {"stop_at_tool_names": ["finish_round"]}
        assert kwargs["run_config"].tracing_disabled
        return await original(agent, *args, **kwargs)

    monkeypatch.setattr(_manager.Runner, "run", inspected)
    fake = register([finish(), FakeTurn(text="unused")])
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")):
            await AgentsSDKManager(
                model="opus", parallel_tool_calls=False, temperature=0.3
            ).run_round(context())
    assert len(fake.calls) == 1
    assert fake.calls[0]["output_schema"] is None
    assert fake.calls[0]["model_settings"].parallel_tool_calls is False
    assert fake.calls[0]["model_settings"].temperature == 0.3
    assert fake.calls[0]["tracing"].is_disabled()


def test_spec_toml_form_builds() -> None:
    m = Ref(path="swarm_arena.llm:AgentsSDKManager", config={"model": "opus"}).build(Manager)
    assert isinstance(m, AgentsSDKManager)
    assert m.models() == ("opus",)
    line = 'manager = { path = "swarm_arena.llm:AgentsSDKManager", config = { model = "opus" } }'
    documented = Ref.model_validate(tomllib.loads(line)["manager"]).build(Manager)
    assert isinstance(documented, AgentsSDKManager)
    assert documented.models() == ("opus",)


@pytest.mark.parametrize(
    "fields",
    [
        {"model": "opus", "temperature": "SENTINEL_SECRET"},
        {"model": "opus", "seed_inputs": {"SENTINEL_SECRET": float("nan")}},
        {"model": "opus", "action_timeout_s": 0},
        {"model": "opus", "SENTINEL_SECRET": "SENTINEL_SECRET"},
    ],
)
def test_config_errors_are_value_free(fields: dict[str, Any]) -> None:
    with pytest.raises((ConfigError, ValidationError)) as caught:
        AgentsSDKManager(**fields)
    assert_secret_absent(caught.value, "SENTINEL_SECRET")


async def test_missing_scope_is_value_free() -> None:
    with pytest.raises(RuntimeError, match="scope"):
        await AgentsSDKManager(model="opus").run_round(context())


@pytest.mark.parametrize("kind", ["cleanup", "fenced", "nested"])
async def test_terminal_action_control_prevents_another_turn(tmp_path: Path, kind: str) -> None:
    error: BaseException = _stop_cleanup_error()
    if kind == "fenced":
        error = _core_error(Fenced, "run ownership lost")
    elif kind == "nested":
        error = ExceptionGroup("SECRET_GROUP", [cast(Exception, error)])
    fake = register([FakeTurn(tool_calls=(("save_notes", {"text": "x"}),)), finish()])
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")), pytest.raises((RuntimeError, Fenced)) as caught:
            await AgentsSDKManager(model="opus").run_round(context(raise_on={"save_notes": error}))
        assert _is_core_control(caught.value)
        assert not isinstance(caught.value, AgentsException)
        assert_secret_absent(caught.value, "SECRET_GROUP")
        assert len(fake.calls) == len(run.calls()) == 1


@pytest.mark.parametrize("timed_out", [False, True])
async def test_terminal_drain_cause_wins_over_round_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timed_out: bool,
) -> None:
    from swarm_arena.llm import _manager

    fake = HeldModel() if timed_out else register([finish()])
    fake.register("opus")
    original = _manager.drain

    async def terminal(scope: Any, *, wait: bool) -> list[str]:
        await original(scope, wait=wait)
        raise _stop_cleanup_error()

    monkeypatch.setattr(_manager, "drain", terminal)
    cfg = config(limits=Limits(round_timeout_s=0.05 if timed_out else 60))
    async with open_llm_run(tmp_path, config=cfg) as run:
        with llm_scope(run.scope("manager")), pytest.raises(RuntimeError) as caught:
            await AgentsSDKManager(model="opus").run_round(context())
        assert not isinstance(caught.value, LimitReached)
        assert _is_core_control(caught.value)


async def test_prompt_read_cancellation_never_starts_a_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena._store.artifacts import ArtifactStore
    from swarm_arena.tool import FileSet

    fake = register([finish()])
    held = HeldOperation()

    def read(_: ArtifactStore, digest: str) -> bytes:
        held.wait()
        return b"reference"

    monkeypatch.setattr(ArtifactStore, "get_bytes", read)
    ctx = context()
    ctx = cast(
        AgentContext,
        replace(
            cast(FakeAgentContext, ctx),
            goal=ctx.goal.model_copy(
                update={
                    "files": FileSet(manifest={"guide.md": "a" * 64}),
                    "metadata": {"prompt_files": ["*.md"]},
                }
            ),
        ),
    )
    async with open_llm_run(tmp_path, config=config()) as run:
        try:
            with llm_scope(run.scope("manager")):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(AgentsSDKManager(model="opus").run_round, ctx)
                    await held.ready()
                    tasks.cancel_scope.cancel()
            assert fake.calls == []
            assert run.calls() == []
        finally:
            await held.finish()
        assert fake.calls == []
        assert run.calls() == []


@pytest.mark.llm_live
async def test_live_manager_round(tmp_path: Path) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not configured")
    cfg = config().model_copy(
        update={
            "models": ModelCatalog(
                {
                    "opus": ModelSpec(
                        model="anthropic/claude-sonnet-4-6",
                        roles=["manager"],
                        max_tokens=256,
                    )
                }
            )
        }
    )
    async with open_llm_run(tmp_path, config=cfg) as run:
        with llm_scope(run.scope("manager")):
            report = await AgentsSDKManager(
                model="opus", instructions="Call finish_round now with summary done."
            ).run_round(context())
        assert report.summary


async def test_manager_waits_for_controlled_late_callback(tmp_path: Path) -> None:
    from _fakes.llm_components import LateFakeModel

    release = anyio.Event()
    fake = LateFakeModel([finish()], delay_s=0, release=release)
    fake.register("opus")
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            task = asyncio.create_task(AgentsSDKManager(model="opus").run_round(context()))
        try:
            with anyio.fail_after(60):
                await fake.ready.wait()
            assert not task.done()
            assert run.calls() == []
            release.set()
            with anyio.fail_after(60):
                assert (await task).summary == "tried lr 0.01"
                await fake.flush()
            (call,) = run.calls()
            assert call.error is None
            assert run.charged(scope) == Amount(tokens=call.usage.total_tokens, usd=call.cost_usd)
            assert scope.call_ids == [call.id]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await fake.aclose()


@pytest.mark.parametrize("never", [False, True])
async def test_missing_callback_is_estimated_and_late_record_writes_nothing(
    tmp_path: Path,
    never: bool,
) -> None:
    from _fakes.llm_components import LateFakeModel

    release = anyio.Event()
    fake = LateFakeModel([finish()], delay_s=None if never else 0, release=release)
    fake.register("opus")
    async with open_llm_run(tmp_path, config=config()) as run:
        # Only the pending-report wait is shortened. The live writer and
        # binding retain >=2 s for durable cleanup and healthy writes.
        scope = replace(run.scope("manager"), settle_timeout_s=0.05)
        try:
            with llm_scope(scope), anyio.fail_after(60):
                report = await AgentsSDKManager(model="opus").run_round(context())
            assert report.summary == "tried lr 0.01"
            (call,) = run.calls()
            assert call.error is not None
            assert call.error.startswith("estimated:")
            charged = run.charged(scope)
            assert charged.usd == call.cost_usd
            objects = list(run.artifacts._store.list(""))
            release.set()
            with anyio.fail_after(60):
                await fake.flush()
            assert run.calls() == [call]
            assert run.charged(scope) == charged
            assert list(run.artifacts._store.list("")) == objects
            assert scope.call_ids == [call.id]
            assert fake.recorded.is_set() == (not never)
        finally:
            await fake.aclose()


async def test_cancel_while_draining_still_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.llm_components import LateFakeModel

    from swarm_arena.llm import _manager

    fake = LateFakeModel([finish()], delay_s=None)
    fake.register("opus")
    draining = anyio.Event()
    original = _manager.drain

    async def observed(scope: Any, *, wait: bool) -> list[str]:
        if wait:
            draining.set()
        return await original(scope, wait=wait)

    monkeypatch.setattr(_manager, "drain", observed)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            task = asyncio.create_task(AgentsSDKManager(model="opus").run_round(context()))
        try:
            with anyio.fail_after(30):
                await draining.wait()
            task.cancel()
            with anyio.fail_after(30), pytest.raises(asyncio.CancelledError):
                await task
            (call,) = run.calls()
            assert call.error is not None
            assert call.error.startswith("estimated:")
            assert run.charged(scope).usd == call.cost_usd
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await fake.aclose()


async def test_concurrent_managers_keep_action_links_local(tmp_path: Path) -> None:
    register(
        [
            FakeTurn(
                tool_calls=(("run_tool", {"inputs": {"x": 1}}), ("run_tool", {"inputs": {"x": 2}}))
            ),
            finish(),
        ]
    )
    contexts = [context(), context()]
    async with open_llm_run(tmp_path, config=config()) as run:

        async def work(index: int) -> None:
            with llm_scope(run.scope("manager", agent=f"a{index}")):
                await AgentsSDKManager(model="opus").run_round(contexts[index])

        async with anyio.create_task_group() as tasks:
            for index in range(2):
                tasks.start_soon(work, index)
        for index, ctx in enumerate(contexts):
            calls = [call for call in run.calls() if call.agent == f"a{index}"]
            links = cast(FakeAgentContext, ctx).requesting_calls
            assert links == [("run_tool", calls[0].id)] * 2
            assert len(calls) == 2


def test_fresh_process_resume_installs_the_call_logger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from _fakes.llm_components import FakeSDKManager
    from _fakes.run import all_events, plan

    from swarm_arena import Swarm

    cfg = config(
        limits=Limits(max_tokens=10_000_000, max_usd=100),
        run_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        arena_root=str(tmp_path / "arena"),
    )
    config_path = tmp_path / "run.toml"
    config_path.write_text(cfg.dumps())
    current = plan(rounds=2)
    current.teams[0].agents[0] = (
        current.teams[0]
        .agents[0]
        .model_copy(
            update={
                "manager": Ref(
                    path="_fakes.llm_components:FakeSDKManager", config={"model": "opus"}
                ),
            }
        )
    )
    original = FakeSDKManager.run_round

    async def parent() -> str:
        paused = anyio.Event()

        async def pause(self: FakeSDKManager, ctx: AgentContext) -> RoundReport:
            if ctx.round == 2:
                paused.set()
                await anyio.sleep_forever()
            return await original(self, ctx)

        monkeypatch.setattr(FakeSDKManager, "run_round", pause)
        swarm = Swarm(cfg)
        task = asyncio.create_task(swarm.run(current, run_id="resume-manager"))
        waiter = asyncio.create_task(paused.wait())
        try:
            with anyio.fail_after(60):
                done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
                if task in done:
                    await task
                    pytest.fail("run finished before the second-round pause")
        finally:
            task.cancel()
            waiter.cancel()
            await asyncio.gather(task, waiter, return_exceptions=True)
        assert swarm.run_id is not None
        return swarm.run_id

    # Sync tests may inherit the AnyIO runner from imported async fixtures.
    with ThreadPoolExecutor(max_workers=1) as pool:
        run_id = cast(str, pool.submit(anyio.run, parent).result(timeout=180))
    before = [e for e in all_events(cfg) if e.type == "llm_call"]
    assert len(before) == 2
    assert {e.round for e in before} == {1}
    source = """
import sys
import swarm_arena
assert "swarm_arena.llm" not in sys.modules
import anyio
config = swarm_arena.RunConfig.load(sys.argv[1])
async def main():
    await swarm_arena.Swarm(config).resume(sys.argv[2], force=True)
anyio.run(main)
import litellm
from swarm_arena.llm._logger import CallLogger
print(any(isinstance(c, CallLogger) for c in litellm.callbacks))
"""
    proc = subprocess.run(
        [sys.executable, "-c", source, str(config_path), run_id],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("True")
    after = [e for e in all_events(cfg) if e.type == "llm_call"]
    assert len(after) == 4
    assert [e.round for e in after] == [1, 1, 2, 2]
    assert all(e.payload["call"]["error"] is None for e in after)


@pytest.mark.parametrize("kind", ["cleanup", "fence"])
@pytest.mark.parametrize("timer_first", [False, True])
@pytest.mark.parametrize("nested", [False, True])
async def test_real_action_control_beats_round_timer(
    tmp_path: Path, ctx_env: CtxEnv, kind: str, timer_first: bool, nested: bool
) -> None:
    from swarm_arena.llm import _actions

    entered, unwinding, release, finished = (anyio.Event() for _ in range(4))

    async def handler(context: AgentContext, args: Inputs) -> JsonValue:
        entered.set()
        try:
            if timer_first:
                owner = _actions._owned_actions.get()
                assert owner is not None
                # SDK startup must finish before the round timer can interrupt the action.
                owner.timer.scope.deadline = anyio.current_time()
                await anyio.sleep_forever()
        finally:
            unwinding.set()
            with anyio.CancelScope(shield=True):
                await release.wait()
            finished.set()
            cause = (
                _stop_cleanup_error()
                if kind == "cleanup"
                else _core_error(Fenced, "run ownership lost")
            )
            if nested:
                raise BaseExceptionGroup("secret-unwind", [asyncio.CancelledError(), cause])
            raise cause

    ctx = ctx_env.ctx()
    ctx.actions = (Action(name="held", description="Held", input_model=Inputs, handler=handler),)
    fake = register([FakeTurn(tool_calls=(("held", {}),)), finish()])
    async with open_llm_run(tmp_path, config=config(limits=Limits(round_timeout_s=60))) as run:
        with llm_scope(run.scope("manager")):
            task = asyncio.create_task(AgentsSDKManager(model="opus").run_round(ctx))
        try:
            with anyio.fail_after(60):
                await unwinding.wait()
                assert entered.is_set()
                assert not task.done(), "round returned while its action was unwinding"
                release.set()
                with pytest.raises(RuntimeError) as caught:
                    await task
            expected = _SignalCategory.TERMINAL if kind == "cleanup" else _SignalCategory.FENCED
            assert _core_category(caught.value) is expected
            assert finished.is_set()
            assert task.cancelling() == 0
            assert len(fake.calls) == 1
            assert_secret_absent(caught.value, "secret-unwind")
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            with anyio.fail_after(60):
                await finished.wait()


@pytest.mark.parametrize("kind", ["cleanup", "fence"])
@pytest.mark.parametrize("timer_first", [False, True])
@pytest.mark.parametrize("owner", ["task", "scope"])
async def test_real_caller_cancellation_beats_round_timer_and_action_control(
    tmp_path: Path, ctx_env: CtxEnv, kind: str, timer_first: bool, owner: str
) -> None:
    from swarm_arena.llm import _actions

    entered, unwinding, release, finished = (anyio.Event() for _ in range(4))

    async def handler(context: AgentContext, args: Inputs) -> JsonValue:
        entered.set()
        try:
            if timer_first:
                actions = _actions._owned_actions.get()
                assert actions is not None
                # Expire the real timer after entry, before caller cancellation.
                actions.timer.scope.deadline = anyio.current_time()
            await anyio.sleep_forever()
        finally:
            unwinding.set()
            with anyio.CancelScope(shield=True):
                await release.wait()
            finished.set()
            raise (
                _stop_cleanup_error()
                if kind == "cleanup"
                else _core_error(Fenced, "run ownership lost")
            )

    ctx = ctx_env.ctx()
    ctx.actions = (Action(name="held", description="Held", input_model=Inputs, handler=handler),)
    fake = register([FakeTurn(tool_calls=(("held", {}),)), finish()])
    owner_scope = anyio.CancelScope()
    cancelled = anyio.Event()
    async with open_llm_run(tmp_path, config=config(limits=Limits(round_timeout_s=60))) as run:

        async def manage() -> None:
            with llm_scope(run.scope("manager")), owner_scope:
                try:
                    await AgentsSDKManager(model="opus").run_round(ctx)
                except asyncio.CancelledError:
                    if owner == "task":
                        raise
                    cancelled.set()
                    return
                pytest.fail("caller cancellation became a report")

        task = asyncio.create_task(manage())
        try:
            with anyio.fail_after(60):
                await entered.wait()
                if timer_first:
                    await unwinding.wait()
                if owner == "task":
                    task.cancel()
                else:
                    owner_scope.cancel()
                await unwinding.wait()
                assert not task.done(), "round returned while its action was unwinding"
                release.set()
                if owner == "task":
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    await task
                    assert cancelled.is_set()
            assert finished.is_set()
            assert len(fake.calls) == 1
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            with anyio.fail_after(60):
                await finished.wait()


@pytest.mark.parametrize("swallow", [False, True])
async def test_real_action_round_expiry_is_an_ordinary_limit(
    tmp_path: Path, ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch, swallow: bool
) -> None:
    from swarm_arena.llm import _actions

    entered, unwinding, draining, release, finished = (anyio.Event() for _ in range(5))
    original_wait = _actions._OwnedActions.wait
    drained: list[bool] = []

    async def wait_for_actions(owner: _actions._OwnedActions) -> None:
        draining.set()
        await original_wait(owner)
        drained.append(finished.is_set())

    monkeypatch.setattr(_actions._OwnedActions, "wait", wait_for_actions)

    async def handler(context: AgentContext, args: Inputs) -> JsonValue:
        entered.set()
        owner = _actions._owned_actions.get()
        assert owner is not None
        # Expire the real manager timer only after dispatch and validation finish.
        owner.timer.scope.deadline = anyio.current_time()
        try:
            await anyio.sleep_forever()
        except asyncio.CancelledError:
            if swallow:
                return {"late": "success"}
            raise
        finally:
            unwinding.set()
            with anyio.CancelScope(shield=True):
                await release.wait()
            finished.set()

    ctx = ctx_env.ctx()
    ctx.actions = (Action(name="held", description="Held", input_model=Inputs, handler=handler),)
    fake = register([FakeTurn(tool_calls=(("held", {}),)), finish()])
    async with open_llm_run(tmp_path, config=config(limits=Limits(round_timeout_s=60))) as run:
        with llm_scope(run.scope("manager")):
            task = asyncio.create_task(AgentsSDKManager(model="opus").run_round(ctx))
        try:
            with anyio.fail_after(60):
                await unwinding.wait()
                await draining.wait()
                assert entered.is_set()
                assert not finished.is_set()
                assert not task.done(), "round returned while its action was unwinding"
                release.set()
                with pytest.raises(LimitReached) as caught:
                    await task
            assert _core_category(caught.value) is _SignalCategory.LIMIT
            assert caught.value.limit == "round_timeout_s"
            assert drained == [True], "manager drain returned before action unwind"
            assert finished.is_set()
            assert len(fake.calls) == 1
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("reverse", [False, True])
async def test_round_timer_preserves_fencing_across_parallel_action_unwind(
    tmp_path: Path, ctx_env: CtxEnv, reverse: bool
) -> None:
    from swarm_arena.llm import _actions

    entered: set[str] = set()
    raised: list[str] = []

    async def wait_for_round_expiry(name: str) -> None:
        entered.add(name)
        if len(entered) == 2:
            owner = _actions._owned_actions.get()
            assert owner is not None
            # Both handlers must participate in the cancellation/control race.
            owner.timer.scope.deadline = anyio.current_time()
        await anyio.sleep_forever()

    async def cleanup(context: AgentContext, args: Inputs) -> JsonValue:
        try:
            await wait_for_round_expiry("cleanup")
        finally:
            raised.append("cleanup")
            raise _stop_cleanup_error()

    async def fence(context: AgentContext, args: Inputs) -> JsonValue:
        try:
            await wait_for_round_expiry("fence")
        finally:
            raised.append("fence")
            raise _core_error(Fenced, "run ownership lost")

    ctx = ctx_env.ctx()
    ctx.actions = tuple(
        Action(name=name, description=name, input_model=Inputs, handler=handler)
        for name, handler in (("cleanup", cleanup), ("fence", fence))
    )
    names = ("fence", "cleanup") if reverse else ("cleanup", "fence")
    fake = register([FakeTurn(tool_calls=tuple((name, {}) for name in names)), finish()])
    async with open_llm_run(tmp_path, config=config(limits=Limits(round_timeout_s=60))) as run:
        with llm_scope(run.scope("manager")), anyio.fail_after(60), pytest.raises(Fenced) as caught:
            await AgentsSDKManager(model="opus").run_round(ctx)
        assert sorted(raised) == ["cleanup", "fence"]
        assert _core_category(caught.value) is _SignalCategory.FENCED
        assert len(fake.calls) == 1


async def test_final_text_is_the_summary(tmp_path: Path) -> None:
    register([FakeTurn(text="finished exploring")])
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")):
            report = await AgentsSDKManager(model="opus").run_round(context())
        assert report == RoundReport(summary="finished exploring")
        assert len(run.calls()) == 1


def test_config_object_and_keyword_fields() -> None:
    cfg = AgentsSDKManager.Config(model="opus", seed_inputs={"x": 1})
    manager = AgentsSDKManager(cfg)
    assert manager.config == cfg
    assert manager.models() == ("opus",)
    with pytest.raises(TypeError, match="config or fields"):
        AgentsSDKManager(cfg, model="opus")


async def test_ordinary_model_failure_is_sanitized_and_drained(tmp_path: Path) -> None:
    class Broken(FakeModel):
        async def get_response(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("MODEL_SECRET")

    Broken([]).register("opus")
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = replace(run.scope("manager"), settle_timeout_s=0.05)
        with llm_scope(scope), pytest.raises(RuntimeError, match="ValueError") as caught:
            await AgentsSDKManager(model="opus").run_round(context())
        assert_secret_absent(caught.value, "MODEL_SECRET")
        (call,) = run.calls()
        assert call.error is not None
        assert call.error.startswith("estimated:")
        assert run.charged(scope).usd == call.cost_usd


async def test_caller_cancellation_wins_over_terminal_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena.llm import _manager

    fake = HeldModel()
    fake.register("opus")
    original = _manager.drain

    async def terminal(scope: Any, *, wait: bool) -> list[str]:
        await original(scope, wait=wait)
        raise _stop_cleanup_error()

    monkeypatch.setattr(_manager, "drain", terminal)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            task = asyncio.create_task(AgentsSDKManager(model="opus").run_round(context()))
        try:
            with anyio.fail_after(60):
                await fake.entered.wait()
            task.cancel()
            with anyio.fail_after(60), pytest.raises(asyncio.CancelledError):
                await task
            assert len(run.calls()) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_delayed_manager_setup_keeps_engine_round_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.run import all_events, plan
    from _fakes.tools import WRITE_THEN_SLEEP

    from swarm_arena._engine.round import RoundRunner
    from swarm_arena.llm import _manager

    deadlines: list[float] = []
    manager_deadlines: list[float] = []
    provider_entries: list[float] = []
    watch, prepare = RoundRunner._watch, _manager._prepare
    timer_scope = _manager._local_timer_scope
    response = FakeModel.get_response

    async def observed_watch(self: RoundRunner, *args: Any) -> None:
        deadlines.append(args[-1])
        await watch(self, *args)

    async def held_prepare(hook: Any, *args: Any, **kwargs: Any) -> Any:
        if hook.__name__ == "prepare":
            release = anyio.Event()
            handle = asyncio.get_running_loop().call_at(deadlines[-1] - 1, release.set)
            try:
                with anyio.fail_after(60):
                    await release.wait()
            finally:
                handle.cancel()
        return await prepare(hook, *args, **kwargs)

    def observed_timer(scope: anyio.CancelScope, **kwargs: Any) -> Any:
        manager_deadlines.append(scope.deadline)
        return timer_scope(scope, **kwargs)

    async def observed_response(self: FakeModel, *args: Any, **kwargs: Any) -> Any:
        provider_entries.append(anyio.current_time() - deadlines[-1])
        return await response(self, *args, **kwargs)

    monkeypatch.setattr(RoundRunner, "_watch", observed_watch)
    monkeypatch.setattr(_manager, "_prepare", held_prepare)
    monkeypatch.setattr(_manager, "_local_timer_scope", observed_timer)
    monkeypatch.setattr(FakeModel, "get_response", observed_response)
    cfg = config(
        limits=Limits(max_tokens=10_000_000, max_usd=100, round_timeout_s=3),
        run_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        arena_root=str(tmp_path / "arena"),
    )
    current = plan(rounds=1, tool=PyTool({"code": WRITE_THEN_SLEEP, "grace_s": 3}))
    register([FakeTurn(tool_calls=(("run_tool", {"inputs": {"n": 1}}),), usage=Usage())])
    current.teams[0].agents[0] = (
        current.teams[0]
        .agents[0]
        .model_copy(
            update={
                "manager": Ref(path="swarm_arena.llm:AgentsSDKManager", config={"model": "opus"})
            }
        )
    )
    with anyio.fail_after(30):
        await Swarm(cfg).run(current, run_id="delayed-manager")
    assert provider_entries
    assert all(value < 0 for value in provider_entries)
    assert len(provider_entries) == 1
    assert manager_deadlines == deadlines
    events = all_events(cfg)
    assert len([event for event in events if event.type == "tool_start"]) == 1
    assert {event.payload["reservation"]["id"] for event in events if event.type == "reserve"} == {
        event.payload["id"] for event in events if event.type == "settle"
    }


async def test_round_timeout_settles_owned_tools_and_preserves_next_round_shares(
    tmp_path: Path,
) -> None:
    from _fakes.run import all_events, plan
    from _fakes.tools import WRITE_THEN_SLEEP, PyTool

    shares: dict[str, Any] = {}
    for kind in ("scripted", "sdk"):
        root = tmp_path / kind
        cfg = config(
            limits=Limits(max_tokens=10_000_000, max_usd=100, round_timeout_s=2),
            run_root=str(root / "runs"),
            artifact_root=str(root / "artifacts"),
            arena_root=str(root / "arena"),
        )
        current = plan(rounds=2, tool=PyTool({"code": WRITE_THEN_SLEEP, "grace_s": 3}))
        if kind == "sdk":
            # Zero paid usage makes the budget comparison exact in both dimensions.
            fake = register(
                [FakeTurn(tool_calls=(("run_tool", {"inputs": {"n": 1}}),), usage=Usage())]
            )
            manager = Ref(path="swarm_arena.llm:AgentsSDKManager", config={"model": "opus"})
        else:
            manager = Ref.of(
                ScriptedManager([{"inputs": {"n": number}, "round": number} for number in (1, 2)])
            )
        current.teams[0].agents[0] = (
            current.teams[0].agents[0].model_copy(update={"manager": manager})
        )
        with anyio.fail_after(30):
            await Swarm(cfg).run(current, run_id=f"owned-{kind}")
        events = all_events(cfg)
        starts = [event for event in events if event.type == "tool_start"]
        ends = [event for event in events if event.type == "tool_end"]
        assert len(starts) == len(ends) == 2
        assert {event.task_id for event in starts} == {event.task_id for event in ends}
        assert all(event.payload["candidate"]["output"]["status"] == "cancelled" for event in ends)
        reserved = {
            event.payload["reservation"]["id"] for event in events if event.type == "reserve"
        }
        settled = {event.payload["id"] for event in events if event.type == "settle"}
        assert reserved == settled
        for end in ends:
            status = next(
                event
                for event in events
                if event.type == "agent_status" and event.round == end.round
            )
            assert end.seq < status.seq
        shares[kind] = next(
            event.payload["shares"]
            for event in events
            if event.type == "round_started" and event.round == 2
        )
        if kind == "sdk":
            assert len(fake.calls) == 2
    assert shares["sdk"] == shares["scripted"]


async def test_parallel_overflow_keeps_both_owned_tool_completions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.run import all_events, plan
    from _fakes.tools import PyTool

    ready = tmp_path / "slow-ready"
    code = f"""
import json, os, time
from pathlib import Path
n = json.load(open("swarm/request.json"))["params"]["n"]
ready = Path({str(ready)!r})
if n == 1:
    json.dump({{"data": {{"y": 1.0}}}}, open(os.environ["SWARM_ARENA_RESULT"], "w"))
    ready.touch()
    time.sleep(600)
else:
    deadline = time.monotonic() + 60
    while not ready.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    json.dump({{"data": {{"y": 2.0}}, "usage": {{"usd": 1e300}}}},
              open(os.environ["SWARM_ARENA_RESULT"], "w"))
"""
    cfg = config(
        limits=Limits(max_tokens=10_000_000, max_usd=100),
        run_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        arena_root=str(tmp_path / "arena"),
    )
    current = plan(rounds=1, tool=PyTool({"code": code, "grace_s": 3}))
    current.teams[0].agents[0] = (
        current.teams[0]
        .agents[0]
        .model_copy(
            update={
                "manager": Ref(path="swarm_arena.llm:AgentsSDKManager", config={"model": "opus"})
            }
        )
    )
    fake = register(
        [
            FakeTurn(
                tool_calls=(("run_tool", {"inputs": {"n": 1}}), ("run_tool", {"inputs": {"n": 2}}))
            ),
            finish("must not run"),
        ]
    )
    from swarm_arena._budget.llm_scope import LLMScope
    from swarm_arena._engine.round import _RoundContext
    from swarm_arena.llm import _manager

    active_contexts: list[AgentContext] = []
    outcomes: list[_SignalCategory | None] = []
    drain_checks: list[bool] = []
    original_round, original_drain = AgentsSDKManager.run_round, _manager.drain

    async def observed_round(self: AgentsSDKManager, ctx: AgentContext) -> RoundReport:
        active_contexts.append(ctx)
        try:
            return await original_round(self, ctx)
        except BaseException as error:
            outcomes.append(_core_category(error))
            raise

    async def checked_drain(scope: LLMScope, *, wait: bool) -> list[str]:
        assert cast(_RoundContext, active_contexts[0])._running_tools == 0
        assert len([event for event in all_events(cfg) if event.type == "tool_end"]) == 2
        result = await original_drain(scope, wait=wait)
        with pytest.raises(LimitReached) as caught:
            await scope.tranche._ledger.reserve(":run", Amount(usd=1), kind="tool", goal_id="g")
        assert _core_category(caught.value) is _SignalCategory.TERMINAL
        assert caught.value.limit == "usage_overflow"
        drain_checks.append(True)
        return result

    monkeypatch.setattr(AgentsSDKManager, "run_round", observed_round)
    monkeypatch.setattr(_manager, "drain", checked_drain)
    with anyio.fail_after(30):
        await Swarm(cfg).run(current, run_id="owned-overflow")
    events = all_events(cfg)
    starts = [event for event in events if event.type == "tool_start"]
    ends = [event for event in events if event.type == "tool_end"]
    assert len(starts) == len(ends) == 2
    assert {event.task_id for event in starts} == {event.task_id for event in ends}
    assert drain_checks == [True]
    assert outcomes == [_SignalCategory.TERMINAL]
    assert any(event.payload.get("usage_overflow") for event in events if event.type == "settle")
    status = next(event for event in events if event.type == "agent_status")
    assert all(event.seq < status.seq for event in ends)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("nested", [False, True])
async def test_parallel_over_cap_keeps_admitted_candidate(tmp_path: Path, nested: bool) -> None:
    from _fakes.run import all_events, plan
    from _fakes.tools import WRITE, PyTool

    cfg = config(
        limits=Limits(max_tokens=10_000_000, max_usd=100, max_tool_runs_per_round=1),
        run_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        arena_root=str(tmp_path / "arena"),
    )
    current = plan(
        rounds=1,
        tool=(NestedActionTool if nested else PyTool)(
            {"code": WRITE.format("{'data': {'y': 1.0}}")}
        ),
    )
    current.teams[0].agents[0] = (
        current.teams[0]
        .agents[0]
        .model_copy(
            update={
                "manager": Ref(path="swarm_arena.llm:AgentsSDKManager", config={"model": "opus"})
            }
        )
    )
    fake = register(
        [
            FakeTurn(
                tool_calls=(("nested", {"x": 1}), ("nested", {"x": 2}))
                if nested
                else (("run_tool", {"inputs": {"n": 1}}), ("run_tool", {"inputs": {"n": 2}}))
            ),
            finish("must not run"),
        ]
    )
    with anyio.fail_after(60):
        await Swarm(cfg).run(current, run_id="owned-cap")
    events = all_events(cfg)
    (start,) = [event for event in events if event.type == "tool_start"]
    (end,) = [event for event in events if event.type == "tool_end"]
    assert start.task_id == end.task_id
    assert end.payload["candidate"]["output"]["data"] == {"y": 1.0}
    assert {event.payload["reservation"]["id"] for event in events if event.type == "reserve"} == {
        event.payload["id"] for event in events if event.type == "settle"
    }
    status = next(event for event in events if event.type == "agent_status")
    assert status.payload["status"] != "starved"
    assert status.payload["reason"] == "limit reached"
    assert end.seq < status.seq
    assert len(fake.calls) == 1


@pytest.mark.parametrize("summaries", [("",), ("A", "B"), ("", "B")])
async def test_first_finish_round_summary_is_preserved(
    tmp_path: Path, summaries: tuple[str, ...]
) -> None:
    fake = register(
        [FakeTurn(tool_calls=tuple(("finish_round", {"summary": value}) for value in summaries))]
    )
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")):
            report = await AgentsSDKManager(model="opus").run_round(context())
        assert report.summary == summaries[0]
        assert len(fake.calls) == 1


async def test_later_finish_round_call_returns_a_soft_tool_error() -> None:
    from agents import Agent, Runner
    from agents import RunConfig as SDKRunConfig
    from agents.agent import StopAtTools
    from agents.items import ToolCallOutputItem

    from swarm_arena.llm._actions import build_tools
    from swarm_arena.llm._hooks import _prepare
    from swarm_arena.llm._linking import CallLinks, ManagerDeps

    ctx = context()
    deps = ManagerDeps(ctx=ctx, links=CallLinks())
    tools = await _prepare(lambda: build_tools(ctx, deps, action_timeout_s=60))
    fake = FakeModel(
        [
            FakeTurn(
                tool_calls=(("finish_round", {"summary": "A"}), ("finish_round", {"summary": "B"}))
            )
        ]
    )
    result = await Runner.run(
        Agent(
            name="summary",
            model=fake,
            tools=[*tools],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["finish_round"]),
        ),
        "go",
        context=deps,
        run_config=SDKRunConfig(tracing_disabled=True),
    )
    outputs = [
        json.loads(item.output) for item in result.new_items if isinstance(item, ToolCallOutputItem)
    ]
    assert deps.summary == "A"
    assert outputs == [{"ok": True}, {"error": "round already finished"}]


@pytest.mark.parametrize("owner", ["task", "scope"])
@pytest.mark.parametrize("nested", [False, True])
async def test_caller_cancellation_gracefully_finishes_owned_tool_without_cancelling_peer(
    tmp_path: Path, ctx_env: CtxEnv, owner: str, nested: bool
) -> None:
    from _fakes.tools import PyTool

    ready, release = tmp_path / "peer-ready", tmp_path / "peer-release"
    owned_ready = tmp_path / "owned-ready"
    code = f"""
import json, os, time
from pathlib import Path
n = json.load(open("swarm/request.json"))["params"]["n"]
json.dump({{"data": {{"y": n}}}}, open(os.environ["SWARM_ARENA_RESULT"], "w"))
if n == 1:
    Path({str(owned_ready)!r}).touch()
    time.sleep(300)
else:
    Path({str(ready)!r}).touch()
    deadline = time.monotonic() + 300
    while not Path({str(release)!r}).exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
"""
    tool = (NestedActionTool if nested else PyTool)({"code": code, "grace_s": 0.2})
    ctx, peer = ctx_env.ctx(tool=tool), ctx_env.ctx(tool=tool, agent="b")
    call = ("nested", {}) if nested else ("run_tool", {"inputs": {"n": 1}})
    fake = register([FakeTurn(tool_calls=(call,)), finish()])
    owner_scope, cancelled = anyio.CancelScope(), anyio.Event()
    async with open_llm_run(tmp_path / "llm", config=config()) as run:

        async def manage() -> None:
            with llm_scope(run.scope("manager")), owner_scope:
                try:
                    await AgentsSDKManager(model="opus").run_round(ctx)
                except asyncio.CancelledError:
                    cancelled.set()
                    if owner == "task":
                        raise
                    return
                pytest.fail("caller cancellation became a report")

        task = asyncio.create_task(manage())
        peer_task = asyncio.create_task(peer.run_tool({"n": 2}))
        try:
            with anyio.fail_after(60):
                await ctx_env.env.wait_for(ready)
                await ctx_env.env.wait_for(owned_ready)
                assert len(ctx_env.env.events("tool_start")) == 2
                if owner == "task":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    owner_scope.cancel()
                    await task
                assert cancelled.is_set()
                assert not peer_task.done()
                release.touch()
                candidate = await peer_task
            assert candidate.output.status == "ok"
            assert len(ctx_env.env.events("tool_end")) == 2
            assert {
                event.payload["reservation"]["id"] for event in ctx_env.env.events("reserve")
            } == {event.payload["id"] for event in ctx_env.env.events("settle")}
            assert len(run.events("reserve")) == len(run.events("settle")) == 1
            assert len(fake.calls) == 1
        finally:
            release.touch()
            if not task.done():
                task.cancel()
            if not peer_task.done():
                ctx_env.env.inflight.cancel_all()
            await asyncio.gather(task, peer_task, return_exceptions=True)


class NestedActionTool(PyTool):
    """Exercise nested core work followed by indefinitely waiting action code."""

    class Config(PyTool.Config):  # ty: ignore[invalid-attribute-override]
        ready: str = ""

    def extra_actions(self) -> list[Action[Any]]:
        async def nested(ctx: AgentContext, args: Inputs) -> JsonValue:
            await ctx.run_tool({"n": int(args.x) if args.x else ctx.round})
            await anyio.sleep_forever()

        async def capper(ctx: AgentContext, args: Inputs) -> JsonValue:
            ready = f"{cast(NestedActionTool.Config, self.config).ready}-{ctx.round}"
            with anyio.fail_after(60):
                while not await anyio.Path(ready).exists():
                    await anyio.lowlevel.checkpoint()
            await ctx.run_tool({"n": 99})
            return None

        return [
            Action(
                name="nested", description="Run nested work.", input_model=Inputs, handler=nested
            ),
            Action(name="capper", description="Reach the cap.", input_model=Inputs, handler=capper),
        ]


@pytest.mark.parametrize("stop", ["round", "sibling", "action"])
async def test_nested_tool_stops_settle_and_preserve_next_round_budget(
    tmp_path: Path, stop: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.run import all_events, plan
    from _fakes.tools import WRITE_THEN_SLEEP

    from swarm_arena.llm import _manager

    shares: dict[str, Any] = {}
    original_drain = _manager.drain
    for kind in ("baseline", "sdk"):
        root = tmp_path / kind
        ready = root / "ready"
        code = WRITE_THEN_SLEEP
        if stop == "sibling":
            code = (
                "import json, os, time; from pathlib import Path; "
                "json.dump({'data': {'y': 1.0}}, open(os.environ['SWARM_ARENA_RESULT'], 'w')); "
                f"n = json.load(open('swarm/request.json'))['params']['n']; "
                f"Path({str(ready)!r} + '-' + str(n)).touch(); time.sleep(1)"
            )
        cfg = config(
            limits=Limits(
                max_tokens=10_000_000,
                max_usd=100,
                max_tool_runs_per_round=1 if stop == "sibling" else 20,
                round_timeout_s=2
                if stop == "round" or (stop == "action" and kind == "baseline")
                else 60,
            ),
            run_root=str(root / "runs"),
            artifact_root=str(root / "artifacts"),
            arena_root=str(root / "arena"),
        )
        current = plan(
            rounds=2, tool=NestedActionTool({"code": code, "grace_s": 0.3, "ready": str(ready)})
        )
        if kind == "sdk":
            calls: tuple[tuple[str, dict[str, Any]], ...] = (("nested", {}),)
            if stop == "sibling":
                calls += (("capper", {}),)
            turns = [FakeTurn(tool_calls=calls, usage=Usage())]
            if stop == "action":
                turns.append(
                    FakeTurn(tool_calls=(("finish_round", {"summary": "done"}),), usage=Usage())
                )
            register(turns)
            manager = Ref.of(
                AgentsSDKManager(model="opus", action_timeout_s=2 if stop == "action" else 60)
            )
        else:
            manager = Ref.of(ScriptedManager([{"inputs": {"n": n}, "round": n} for n in (1, 2)]))
        current.teams[0].agents[0] = (
            current.teams[0].agents[0].model_copy(update={"manager": manager})
        )

        async def checked_drain(scope: Any, *, wait: bool, cfg: RunConfig = cfg) -> list[str]:
            events = all_events(cfg)
            assert len([e for e in events if e.type == "tool_start"]) == len(
                [e for e in events if e.type == "tool_end"]
            )
            return await original_drain(scope, wait=wait)

        with monkeypatch.context() as patch:
            patch.setattr(_manager, "drain", checked_drain)
            with anyio.fail_after(30):
                await Swarm(cfg).run(current, run_id=f"nested-{kind}")
        events = all_events(cfg)
        starts = [e for e in events if e.type == "tool_start"]
        ends = [e for e in events if e.type == "tool_end"]
        assert len(starts) == len(ends) == 2
        assert {e.task_id for e in starts} == {e.task_id for e in ends}
        assert {e.payload["reservation"]["id"] for e in events if e.type == "reserve"} == {
            e.payload["id"] for e in events if e.type == "settle"
        }
        assert all(
            e.payload["candidate"]["output"]["status"]
            == ("ok" if stop == "sibling" else "cancelled")
            for e in ends
        )
        shares[kind] = next(
            e.payload["shares"] for e in events if e.type == "round_started" and e.round == 2
        )
    assert shares["sdk"] == shares["baseline"]


async def test_peer_overflow_during_join_preserves_finished_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._budget.amount import USD_LIMIT
    from swarm_arena.llm._actions import _OwnedActions

    joining, overflowed = anyio.Event(), anyio.Event()
    original_wait = _OwnedActions.wait

    async def held_join(self: _OwnedActions) -> None:
        joining.set()
        await overflowed.wait()
        await original_wait(self)

    monkeypatch.setattr(_OwnedActions, "wait", held_join)
    register([finish("keep this report")])
    async with open_llm_run(tmp_path, config=config()) as run:
        peer = run.scope("manager", agent="peer")
        reservation = await peer.tranche.reserve_call(Amount(usd=1))

        async def overflow_peer() -> None:
            await joining.wait()
            await run.ledger.settle_overflow(reservation, Amount(usd=USD_LIMIT))
            overflowed.set()

        with anyio.fail_after(60):
            async with anyio.create_task_group() as group:
                group.start_soon(overflow_peer)
                with llm_scope(run.scope("manager", agent="finishing")):
                    report = await AgentsSDKManager(model="opus").run_round(context())
        assert report.summary == "keep this report"
        assert any(e.payload.get("usage_overflow") for e in run.events("settle"))
        with pytest.raises(LimitReached) as caught:
            await run.ledger.reserve(":run", Amount(usd=1), kind="tool", goal_id="g")
        assert _core_category(caught.value) is _SignalCategory.TERMINAL


class ContinuingActionInputs(FrozenModel):
    then: str


class ContinuingActionTool(PyTool):
    """Continue using the context after a nested run completes under cancellation."""

    def extra_actions(self) -> list[Action[Any]]:
        async def loop(ctx: AgentContext, args: ContinuingActionInputs) -> JsonValue:
            first = await ctx.run_tool({"n": 1})
            if args.then == "run2":
                second = await ctx.run_tool({"n": 2})
                return {"ids": [first.id, second.id]}
            if args.then == "action":
                await ctx.call_action("noop", {"x": 0})
            elif args.then == "notes":
                await ctx.save_notes("after the first run")
            elif args.then == "none":
                await anyio.sleep_forever()
            return {"id": first.id}

        async def noop(ctx: AgentContext, args: Inputs) -> JsonValue:
            return {}

        return [
            Action(
                name="loop",
                description="Run tools in a loop.",
                input_model=ContinuingActionInputs,
                handler=loop,
            ),
            Action(name="noop", description="Do nothing.", input_model=Inputs, handler=noop),
        ]


@pytest.mark.parametrize(
    ("then", "round_stop"),
    [
        ("run2", False),
        ("action", False),
        ("ret_slowprep", False),
        ("none", False),
        ("notes", False),
        ("run2", True),
    ],
)
async def test_action_timeout_refusals_stay_local_unless_the_round_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, then: str, round_stop: bool
) -> None:
    from _fakes.run import all_events, plan
    from _fakes.tools import WRITE_THEN_SLEEP
    from agents.tool import default_tool_timeout_error_message

    from swarm_arena._agent.context import _ActionInflight
    from swarm_arena._agent.tool_run import _inflight

    held = HeldOperation()
    if then == "ret_slowprep":

        def estimate(self: ContinuingActionTool, request: Any) -> None:
            if isinstance(_inflight.get(), _ActionInflight):
                held.wait()

        monkeypatch.setattr(ContinuingActionTool, "estimate", estimate)

    stops: list[tuple[bool, bool]] = []
    original_stop = _ActionInflight.stop

    def observed_stop(self: _ActionInflight, scope: anyio.CancelScope, *, cancel: bool) -> None:
        stops.append((cancel, self._parent.closed))
        original_stop(self, scope, cancel=cancel)
        if then == "ret_slowprep":
            held.release.set()

    monkeypatch.setattr(_ActionInflight, "stop", observed_stop)
    reports: list[str] = []
    controls: list[_SignalCategory | None] = []
    original_round = AgentsSDKManager.run_round

    async def observed_round(self: AgentsSDKManager, ctx: AgentContext) -> RoundReport:
        try:
            report = await original_round(self, ctx)
        except BaseException as error:
            controls.append(_core_category(error))
            raise
        reports.append(report.summary)
        return report

    monkeypatch.setattr(AgentsSDKManager, "run_round", observed_round)
    cfg = config(
        limits=Limits(max_tokens=10_000_000, max_usd=100, round_timeout_s=3 if round_stop else 60),
        run_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        arena_root=str(tmp_path / "arena"),
    )
    code = WRITE_THEN_SLEEP
    if round_stop:
        # Keep core grace active across both real deadlines, without a test-side sleep.
        code = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " + code
    current = plan(
        rounds=1, tool=ContinuingActionTool({"code": code, "grace_s": 3 if round_stop else 0.3})
    )
    current.teams[0].agents[0] = (
        current.teams[0]
        .agents[0]
        .model_copy(update={"manager": Ref.of(AgentsSDKManager(model="opus", action_timeout_s=2))})
    )
    summary = "model saw the timeout"
    fake = register([FakeTurn(tool_calls=(("loop", {"then": then}),)), finish(summary)])
    try:
        with anyio.fail_after(60):
            await Swarm(cfg).run(current, run_id="action-timeout")
    finally:
        if held.entered.is_set():
            await held.finish()
    events = all_events(cfg)
    starts = [e for e in events if e.type == "tool_start"]
    ends = [e for e in events if e.type == "tool_end"]
    assert len(starts) == len(ends) == (0 if then == "ret_slowprep" else 1)
    assert {e.task_id for e in starts} == {e.task_id for e in ends}
    assert all(e.payload["candidate"]["output"]["status"] == "cancelled" for e in ends)
    assert {e.payload["reservation"]["id"] for e in events if e.type == "reserve"} == {
        e.payload["id"] for e in events if e.type == "settle"
    }
    status = next(e for e in events if e.type == "agent_status")
    assert stops[0] == (True, False)  # The action deadline closed only its own registry first.
    if then == "ret_slowprep":
        assert held.entered.is_set()
        assert held.exited.is_set()
    if round_stop:
        assert any(parent_closed for _, parent_closed in stops)
        assert reports == []
        assert controls == [_SignalCategory.LIMIT]
        assert len(fake.calls) == 1
        assert status.payload["reason"] == "limit reached"
    else:
        assert reports == [summary]
        assert controls == []
        assert len(fake.calls) == 2
        inputs = fake.calls[1]["input"]
        assert isinstance(inputs, list)
        assert inputs[-1]["type"] == "function_call_output"
        assert inputs[-1]["output"] == default_tool_timeout_error_message(
            tool_name="loop", timeout_seconds=2
        )
        assert status.payload["reason"] == ""


async def test_live_gate_stop_after_action_timeout_ends_round(
    tmp_path: Path, ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._agent.context import _ActionInflight
    from swarm_arena._engine.stops import StopState

    stops = StopState()
    ctx = ctx_env.ctx()
    ctx._gate = stops.gate_for("g")
    original_stop = _ActionInflight.stop
    observed: list[bool] = []

    def stop(self: _ActionInflight, scope: anyio.CancelScope, *, cancel: bool) -> None:
        original_stop(self, scope, cancel=cancel)
        observed.append(self._parent.closed)
        stops.stop = True

    monkeypatch.setattr(_ActionInflight, "stop", stop)

    async def handler(context: AgentContext, args: Inputs) -> dict[str, Any]:
        try:
            await anyio.sleep_forever()
        finally:
            await context.call_action("noop", {})
        return {}

    async def noop(context: AgentContext, args: Inputs) -> dict[str, Any]:
        pytest.fail("stopped gate admitted an action")

    ctx.actions = (
        Action(name="held", description="held", input_model=Inputs, handler=handler),
        Action(name="noop", description="noop", input_model=Inputs, handler=noop),
    )
    fake = register([FakeTurn(tool_calls=(("held", {}),)), finish("continued despite stop")])
    async with open_llm_run(tmp_path, config=config(limits=Limits(round_timeout_s=60))) as run:
        with (
            llm_scope(run.scope("manager")),
            anyio.fail_after(60),
            pytest.raises(LimitReached) as caught,
        ):
            await AgentsSDKManager(model="opus", action_timeout_s=0.1).run_round(ctx)
    assert caught.value.limit == "stop"
    assert _core_category(caught.value) is _SignalCategory.LIMIT
    assert observed == [False]
    assert stops.stop
    assert len(fake.calls) == 1


async def test_component_span_stays_open_through_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Generator
    from contextlib import contextmanager

    from swarm_arena.llm import _manager

    events: list[str] = []
    original = _manager._finish_accounting

    @contextmanager
    def component(scope: Any) -> Generator[None]:
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    async def accounting(scope: Any, *, interrupted: bool) -> None:
        assert events == ["enter"]
        await original(scope, interrupted=interrupted)
        events.append("accounted")

    monkeypatch.setattr(_manager, "component_span", component)
    monkeypatch.setattr(_manager, "_finish_accounting", accounting)
    register([finish()])
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")):
            await AgentsSDKManager(model="opus").run_round(context())
    assert events == ["enter", "accounted", "exit"]
