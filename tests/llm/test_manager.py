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
from _agent.conftest import ctx_env as ctx_env
from _agent.conftest import rt_env as rt_env
from _fakes.agent_ctx import FakeAgentContext
from _fakes.engine import CtxEnv
from _fakes.llm_run import HeldOperation, open_llm_run
from _fakes.secrecy import assert_secret_absent
from agents.exceptions import AgentsException
from pydantic import ValidationError

from swarm_arena import Limits, Ref, RunConfig
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import llm_scope
from swarm_arena._errors import (
    ConfigError,
    Fenced,
    _core_error,
    _is_core_control,
    _stop_cleanup_error,
)
from swarm_arena._model.base import FrozenModel
from swarm_arena.config import ModelCatalog, ModelSpec
from swarm_arena.llm import AgentsSDKManager, BudgetExceeded
from swarm_arena.manager import AgentContext, LimitReached, Manager, RoundReport
from swarm_arena.testing import FakeModel, FakeTurn


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
        settle_timeout_s=fields.pop("settle_timeout_s", 2),
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
    async with open_llm_run(tmp_path, config=config(settle_timeout_s=10)) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            if native:
                task = asyncio.create_task(AgentsSDKManager(model="opus").run_round(context()))
                try:
                    with anyio.fail_after(5):
                        await fake.entered.wait()
                    task.cancel()
                    with anyio.fail_after(2), pytest.raises(asyncio.CancelledError):
                        await task
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            else:
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(AgentsSDKManager(model="opus").run_round, context())
                    with anyio.fail_after(5):
                        await fake.entered.wait()
                    tasks.cancel_scope.cancel()
        (call,) = run.calls()
        assert call.error is not None
        assert call.error.startswith("estimated:")
        assert run.charged(scope).usd == pytest.approx(call.cost_usd)
        assert scope.call_ids == [call.id]


async def test_round_timeout_charges_the_cut_call_without_waiting(tmp_path: Path) -> None:
    fake = HeldModel()
    fake.register("opus")
    async with open_llm_run(
        tmp_path,
        config=config(limits=Limits(round_timeout_s=0.05), settle_timeout_s=10),
    ) as run:
        scope = run.scope("manager")
        with llm_scope(scope), anyio.fail_after(2), pytest.raises(LimitReached) as e:
            await AgentsSDKManager(model="opus").run_round(context())
        (call,) = run.calls()
        assert e.value.limit == "round_timeout_s"
        assert call.error is not None
        assert call.error.startswith("estimated:")
        assert run.charged(scope).usd == pytest.approx(call.cost_usd)


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
            with anyio.fail_after(5):
                await fake.ready.wait()
            assert not task.done()
            assert run.calls() == []
            release.set()
            with anyio.fail_after(5):
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
            with llm_scope(scope), anyio.fail_after(5):
                report = await AgentsSDKManager(model="opus").run_round(context())
            assert report.summary == "tried lr 0.01"
            (call,) = run.calls()
            assert call.error is not None
            assert call.error.startswith("estimated:")
            charged = run.charged(scope)
            assert charged.usd == call.cost_usd
            objects = list(run.artifacts._store.list(""))
            release.set()
            with anyio.fail_after(5):
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
    async with open_llm_run(tmp_path, config=config(settle_timeout_s=10)) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            task = asyncio.create_task(AgentsSDKManager(model="opus").run_round(context()))
        try:
            with anyio.fail_after(5):
                await draining.wait()
            task.cancel()
            with anyio.fail_after(2), pytest.raises(asyncio.CancelledError):
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
            with anyio.fail_after(15):
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
        run_id = cast(str, pool.submit(anyio.run, parent).result(timeout=30))
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
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("True")
    after = [e for e in all_events(cfg) if e.type == "llm_call"]
    assert len(after) == 4
    assert [e.round for e in after] == [1, 1, 2, 2]
    assert all(e.payload["call"]["error"] is None for e in after)


@pytest.mark.parametrize("terminal", [False, True])
async def test_round_timer_preserves_terminal_unwind_and_refuses_swallowed_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: bool,
) -> None:
    from types import SimpleNamespace

    from swarm_arena.llm import _manager

    register([finish()])

    async def interrupted_runner(*args: Any, **kwargs: Any) -> Any:
        try:
            await anyio.sleep_forever()
        except anyio.get_cancelled_exc_class():
            if terminal:
                raise _stop_cleanup_error() from None
            return SimpleNamespace(final_output="late success")

    monkeypatch.setattr(_manager.Runner, "run", interrupted_runner)
    async with open_llm_run(tmp_path, config=config(limits=Limits(round_timeout_s=0.05))) as run:
        with llm_scope(run.scope("manager")), pytest.raises(RuntimeError) as caught:
            await AgentsSDKManager(model="opus").run_round(context())
        assert _is_core_control(caught.value)
        if terminal:
            assert not isinstance(caught.value, LimitReached)
        else:
            assert isinstance(caught.value, LimitReached)
            assert caught.value.limit == "round_timeout_s"


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
            with anyio.fail_after(5):
                await fake.entered.wait()
            task.cancel()
            with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
                await task
            assert len(run.calls()) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
