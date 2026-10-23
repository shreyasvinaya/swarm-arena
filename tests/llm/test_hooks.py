"""Metering, authoritative settings and SDK control transport."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

pytest.importorskip("agents")

from _engine.conftest import env as env
from _fakes.engine import EngineEnv
from _fakes.llm_run import open_llm_run
from _fakes.secrecy import assert_secret_absent
from agents import (
    Agent,
    AgentOutputSchema,
    Model,
    ModelResponse,
    ModelSettings,
    RunContextWrapper,
    Runner,
    function_tool,
)
from agents import RunConfig as SDKRunConfig
from agents.exceptions import AgentsException
from agents.models.interface import ModelTracing
from pydantic import BaseModel

from swarm_arena import Limits, RunConfig
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import llm_scope
from swarm_arena._budget.tranche import Tranche
from swarm_arena._errors import (
    Fenced,
    _core_category,
    _core_error,
    _is_core_control,
    _rebuild_core_signal,
    _SignalCategory,
    _stop_cleanup_error,
)
from swarm_arena._journal.llm_record import drain
from swarm_arena._outcomes import _Outcome
from swarm_arena.config import CachePolicy, ModelCatalog, ModelSpec
from swarm_arena.llm._cache import bypass_response_cache
from swarm_arena.llm._hooks import (
    BudgetExceeded,
    BudgetHooks,
    MeteredModel,
    _sdk_control,
    _unwrap_control,
)
from swarm_arena.llm._linking import CallLinks, ManagerDeps, tool_calls_of
from swarm_arena.manager import AgentContext, LimitReached
from swarm_arena.testing import FakeModel, FakeTurn

pytestmark = pytest.mark.filterwarnings(
    "ignore:Items? .*TypedDict class .*ReadOnly.*:UserWarning:pydantic._internal._generate_schema"
)

SPEC = ModelSpec(
    model="fake/m",
    roles=["manager", "judge"],
    max_tokens=8000,
    prices=ModelSpec.Prices(input=1, output=3),
)


def config(**kwargs: Any) -> RunConfig:
    return RunConfig(
        models=ModelCatalog({"m": SPEC}), limits=Limits(), settle_timeout_s=2, **kwargs
    )


def metered(capture: Model, **kwargs: Any) -> MeteredModel:
    return MeteredModel(capture, alias="m", spec=SPEC, namespace="test", **kwargs)


async def invoke(
    model: Model, settings: ModelSettings | None = None, **kwargs: Any
) -> ModelResponse:
    return await model.get_response(
        "sys",
        [],
        settings or ModelSettings(),
        kwargs.pop("tools", []),
        kwargs.pop("output_schema", None),
        [],
        ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
        **kwargs,
    )


@pytest.mark.parametrize("stage", ["entry", "funded", "settings"])
@pytest.mark.parametrize("reason", ["closed", "deadline"])
async def test_slot_round_control_refuses_provider_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, reason: str
) -> None:
    from swarm_arena._agent.context import _ActionInflight
    from swarm_arena._agent.tool_run import Inflight, track_inflight
    from swarm_arena.llm import _hooks

    registry = _ActionInflight(Inflight())
    capture = FakeModel([FakeTurn(text="unused")])
    prepare = _hooks._prepare
    reserve = Tranche.reserve_call

    def expire() -> None:
        if reason == "closed":
            registry.close()
        else:
            registry._round_deadline = anyio.current_time() - 1

    async def prepared(operation: Any) -> Any:
        result = await prepare(operation)
        if stage == "settings" and operation.__name__ == "settings":
            expire()
        return result

    async def funded(self: Tranche, *args: Any, **kwargs: Any) -> str:
        result = await reserve(self, *args, **kwargs)
        if stage == "funded":
            expire()
        return result

    monkeypatch.setattr(_hooks, "_prepare", prepared)
    monkeypatch.setattr(Tranche, "reserve_call", funded)
    if stage == "entry":
        expire()
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope), track_inflight(registry), pytest.raises(LimitReached) as caught:
            await invoke(metered(capture))
        assert caught.value.limit == ("stop" if reason == "closed" else "round_timeout_s")
        assert _core_category(caught.value) is _SignalCategory.LIMIT
        assert not capture.calls
        await drain(scope, wait=False)
        assert scope.tranche.pending() == 0
        await scope.tranche.close()
        assert run.charged(scope) == Amount()


async def test_reserves_and_sends_metadata(tmp_path: Path) -> None:
    capture = FakeModel([FakeTurn(text="ok", tool_calls=(("best", {}),))])
    links = CallLinks()
    model = metered(capture, links=links)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            response = await invoke(model)
        await drain(scope, wait=True)
        md = capture.calls[0]["model_settings"].metadata
        assert (  # noqa: PT018 - required acceptance assertion
            md["reservation_id"].startswith("r-")
            and md["call_id"].startswith("llm-")
            and md["model_alias"] == "m"
        )
        assert all(type(value) is str for value in md.values())
        assert model.call_ids == scope.call_ids == [md["call_id"]]
        assert links.get(tool_calls_of(response)[0][0]) == md["call_id"]
        assert run.charged(scope)._micro_usd == sum(
            Amount(usd=call.cost_usd)._micro_usd for call in run.calls()
        )


async def test_agent_settings_are_kept(tmp_path: Path) -> None:
    capture = FakeModel([FakeTurn(text="ok")])
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")):
            await invoke(
                metered(capture), ModelSettings(parallel_tool_calls=True, tool_choice="auto")
            )
    settings = capture.calls[0]["model_settings"]
    assert settings.parallel_tool_calls is True
    assert settings.tool_choice == "auto"


@pytest.mark.parametrize(
    ("ours", "callers", "sent"),
    [(1, None, 1), (None, 100, 100), (None, None, 8000), (None, 10_000, 8000)],
)
async def test_max_tokens_cap(
    ours: int | None, callers: int | None, sent: int, tmp_path: Path
) -> None:
    capture = FakeModel([FakeTurn(text="ok")])
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            await invoke(metered(capture, max_tokens=ours), ModelSettings(max_tokens=callers))
        assert capture.calls[0]["model_settings"].max_tokens == sent
        estimate = next(iter(scope.tranche._calls.values())).estimate
        assert estimate.tokens >= sent
        assert estimate._micro_usd == (estimate.tokens - sent) * 2 + sent * 3


def test_stream_response_is_not_supported() -> None:
    with pytest.raises(NotImplementedError, match="MeteredModel does not stream"):
        metered(FakeModel([])).stream_response(
            None,
            [],
            ModelSettings(),
            [],
            None,
            [],
            ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )


async def test_budget_exceeded_when_tranche_cannot_reserve(tmp_path: Path) -> None:
    capture = FakeModel([FakeTurn(text="never")])
    cfg = config().model_copy(update={"limits": Limits(max_usd=1e-6)})
    async with open_llm_run(tmp_path, config=cfg) as run:
        scope = run.scope("manager")
        with llm_scope(scope), pytest.raises(BudgetExceeded) as e:
            await invoke(metered(capture))
        assert (  # noqa: PT018 - required acceptance assertion
            isinstance(e.value, LimitReached)
            and isinstance(e.value, AgentsException)
            and e.value.limit == "budget"
        )
        assert capture.calls == []
        assert run.calls() == [] and run.charged(scope) == Amount()  # noqa: PT018
        assert _is_core_control(_unwrap_control(e.value))


async def test_outside_scope_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="outside an LLM scope"):
        await invoke(metered(FakeModel([])))


async def test_parallel_calls_settle_their_own_reservation(tmp_path: Path) -> None:
    capture = FakeModel([FakeTurn(text="a"), FakeTurn(text="b")])
    model = metered(capture)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(invoke, model)
                tasks.start_soon(invoke, model)
        await drain(scope, wait=True)
        a, b = run.calls()
        assert a.reservation_id != b.reservation_id
        assert run.charged(scope).usd == pytest.approx(a.cost_usd + b.cost_usd)
        assert (
            run.charged(scope)._micro_usd
            == Amount(usd=a.cost_usd)._micro_usd + Amount(usd=b.cost_usd)._micro_usd
        )


async def test_estimate_counts_tool_schemas_and_output_schema(tmp_path: Path) -> None:
    @function_tool
    def example(value: str) -> str:
        """Return a value described by a tool schema."""
        return value

    class Output(BaseModel):
        value: str

    estimates = []
    async with open_llm_run(tmp_path, config=config()) as run:
        for tools, schema in [([], None), ([example], AgentOutputSchema(Output))]:
            scope = run.scope("manager")
            with llm_scope(scope):
                await invoke(
                    metered(FakeModel([FakeTurn(text="ok")])), tools=tools, output_schema=schema
                )
            estimates.append(next(iter(scope.tranche._calls.values())).estimate)
        assert estimates[1].usd > estimates[0].usd
        assert estimates[1].tokens > estimates[0].tokens


async def test_hook_uses_tranche_credit_when_the_ledger_shows_zero(tmp_path: Path) -> None:
    cfg = config().model_copy(update={"limits": Limits(max_usd=1), "reserve_fraction": 0.25})
    async with open_llm_run(tmp_path, config=cfg) as run:
        scope = run.scope("manager")
        scope = replace(
            scope, tranche=Tranche(run.ledger, ":run", "g", size=Amount(tokens=25_000, usd=0.25))
        )
        await scope.tranche.reserve_call(Amount(usd=0.001))
        assert run.ledger.remaining(":run").usd == 0
        deps = ManagerDeps(ctx=cast(AgentContext, object()), links=CallLinks())
        agent = Agent[ManagerDeps](name="test", model=metered(FakeModel([]), max_tokens=400))
        with llm_scope(scope):
            await BudgetHooks().on_llm_start(RunContextWrapper(deps), agent, "sys", [])
        await scope.tranche.close()


async def test_hook_raises_when_neither_tranche_nor_ledger_can_cover(tmp_path: Path) -> None:
    cfg = config().model_copy(update={"limits": Limits(max_usd=1e-6)})
    async with open_llm_run(tmp_path, config=cfg) as run:
        deps = ManagerDeps(ctx=cast(AgentContext, object()), links=CallLinks())
        with llm_scope(run.scope("manager")), pytest.raises(BudgetExceeded):
            await BudgetHooks().on_llm_start(
                RunContextWrapper(deps), Agent(name="test", model=metered(FakeModel([]))), "sys", []
            )


async def test_hook_ignores_unmetered_model() -> None:
    deps = ManagerDeps(ctx=cast(AgentContext, object()), links=CallLinks())
    await BudgetHooks().on_llm_start(
        RunContextWrapper(deps), Agent(name="test", model=FakeModel([])), "sys", []
    )


@pytest.mark.parametrize("bypass", [False, True])
async def test_outbound_merge_keeps_metadata_caps_cache_and_retry_authority(
    tmp_path: Path, bypass: bool
) -> None:
    capture = FakeModel([FakeTurn(text="ok")])
    policy = CachePolicy(backend="disk", disk_dir=str(tmp_path / "cache"), namespace="approved")
    caller = ModelSettings(
        max_tokens=9000,
        metadata={"reservation_id": "forged"},
        extra_args={
            "metadata": {"call_id": "forged"},
            "num_retries": 7,
            "max_retries": 8,
            "cache": {"namespace": "forged"},
            "preset_cache_key": "forged",
            "max_completion_tokens": 9000,
        },
    )
    async with open_llm_run(tmp_path, config=config(cache=policy)) as run:
        scope = run.scope("judge")
        with llm_scope(scope):
            if bypass:
                with bypass_response_cache():
                    await invoke(metered(capture), caller)
            else:
                await invoke(metered(capture), caller)
        sent = capture.calls[0]["model_settings"]
        assert sent.metadata["reservation_id"].startswith("r-")
        assert sent.max_tokens == 8000
        assert sent.extra_args["num_retries"] == sent.extra_args["max_retries"] == 0
        assert "metadata" not in sent.extra_args
        assert "preset_cache_key" not in sent.extra_args
        assert "max_completion_tokens" not in sent.extra_args
        assert ("cache" in sent.extra_args) is not bypass
        if not bypass:
            assert sent.extra_args["cache"]["namespace"] == "approved"


@pytest.mark.parametrize("kind", ["fence", "budget", "overflow", "cleanup"])
@pytest.mark.parametrize("nested", [False, True])
async def test_sdk_carrier_stops_before_another_model_turn(kind: str, nested: bool) -> None:
    error = {
        "fence": _core_error(Fenced, "run ownership lost"),
        "budget": _core_error(LimitReached, "budget", "budget used up"),
        "overflow": _core_error(
            LimitReached,
            "usage_overflow",
            "budget exhausted by usage overflow",
            _category=_SignalCategory.TERMINAL,
        ),
        "cleanup": _stop_cleanup_error(),
    }[kind]
    carried = _sdk_control(error)
    assert isinstance(carried, AgentsException)
    assert _rebuild_core_signal(carried) is None
    secret = "private-carrier-value"
    carried.args = (secret,)
    carried.__dict__.update(limit=secret, message=secret, core=RuntimeError(secret))
    carried.add_note(secret)

    @function_tool(failure_error_function=None)
    async def stop() -> str:
        """Stop the model run at a terminal boundary."""
        if nested:
            group = ExceptionGroup(secret, [ValueError(secret), carried])
            # Action adapters bridge before the SDK's generic tool wrapper.
            bridged = _sdk_control(group)
            assert bridged is not None
            raise bridged from None
        raise carried

    capture = FakeModel([FakeTurn(tool_calls=(("stop", {}),)), FakeTurn(text="must not run")])
    with pytest.raises(AgentsException) as caught:
        await Runner.run(
            Agent(name="test", model=capture, tools=[stop]),
            "go",
            run_config=SDKRunConfig(tracing_disabled=True),
        )
    assert len(capture.calls) == 1
    unwrapped = _unwrap_control(caught.value)
    outcome = _Outcome()
    if kind == "fence":
        with pytest.raises(Fenced) as fenced:
            outcome.classify(unwrapped)
        assert_secret_absent(fenced.value, secret)
    else:
        rebuilt = outcome.classify(unwrapped)
        assert _is_core_control(rebuilt)
        assert_secret_absent(rebuilt, secret)
        if kind in {"budget", "overflow"}:
            assert isinstance(rebuilt, LimitReached)
            assert rebuilt.limit == ("budget" if kind == "budget" else "usage_overflow")


@pytest.mark.parametrize("kind", ["budget", "fence", "cleanup", "subclass"])
def test_untrusted_exceptions_do_not_acquire_carrier_authority(kind: str) -> None:
    class Forged(BudgetExceeded):
        pass

    genuine = _sdk_control(_core_error(LimitReached, "budget", "budget used up"))
    forged = {
        "budget": BudgetExceeded("budget", "secret"),
        "fence": Fenced("secret"),
        "cleanup": RuntimeError("stop cleanup timed out"),
        "subclass": Forged("budget", "secret"),
    }[kind]
    forged.__dict__.update(genuine.__dict__)
    assert not _is_core_control(_unwrap_control(forged))
    assert _sdk_control(forged) is None


async def test_caller_cancellation_wins_nested_carrier() -> None:
    carried = _sdk_control(_core_error(Fenced, "run ownership lost"))
    assert carried is not None
    outcome = _Outcome()
    task = asyncio.current_task()
    assert task is not None
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            outcome.classify(_unwrap_control(ExceptionGroup("private", [carried])))
        with pytest.raises(asyncio.CancelledError):
            await anyio.lowlevel.checkpoint()
    finally:
        task.uncancel()


async def test_cancellation_abandons_call_without_waiting_for_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeModel([])
    entered, finished = anyio.Event(), anyio.Event()
    cancel_scope = anyio.CancelScope()

    async def blocked(*args: Any, **kwargs: Any) -> ModelResponse:
        entered.set()
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    monkeypatch.setattr(capture, "get_response", blocked)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = replace(run.scope("manager"), settle_timeout_s=60)
        model = metered(capture)

        async def request() -> None:
            with cancel_scope, llm_scope(scope):
                await invoke(model)
            finished.set()

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(request)
                await entered.wait()
                cancel_scope.cancel()
                await finished.wait()
            await drain(scope, wait=True)
        calls = run.calls()
        assert len(calls) == 1
        assert calls[0].status == "error"
        assert calls[0].error is not None
        assert calls[0].error.startswith("estimated:")
        assert model.call_ids == [calls[0].id]
        assert run.charged(scope) == next(iter(scope.tranche._calls.values())).estimate


async def test_non_cancellation_failure_keeps_callback_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _fakes.llm_run import record_kwargs

    from swarm_arena._journal.llm_record import record_call

    metadata: dict[str, str] = {}
    capture = FakeModel([])

    async def fail(*args: Any, **kwargs: Any) -> ModelResponse:
        metadata.update(args[2].metadata)
        raise ValueError("provider failure")

    monkeypatch.setattr(capture, "get_response", fail)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope), pytest.raises(ValueError, match="provider failure"):
            await invoke(metered(capture))
        await record_call(
            **record_kwargs(metadata, cost_usd=0.0001, status="error", error="ValueError")
        )
        await drain(scope, wait=True)
        assert len(run.calls()) == 1
        assert run.calls()[0].status == "error"
        assert run.charged(scope) == Amount(tokens=120, usd=0.0001)


async def test_schema_preparation_is_abandonable_before_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _fakes.llm_run import HeldOperation

    from tests.llm.test_upstream_facts import _DictSchema

    held = HeldOperation()
    schema = _DictSchema()
    original = schema.json_schema

    def block() -> dict[str, Any]:
        held.wait()
        return original()

    monkeypatch.setattr(schema, "json_schema", block)
    capture = FakeModel([FakeTurn(text="never")])
    cancelled, finished = anyio.CancelScope(), anyio.Event()
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")

        async def request() -> None:
            with cancelled, llm_scope(scope):
                await invoke(metered(capture), output_schema=schema)
            finished.set()

        try:
            with anyio.fail_after(5):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(request)
                    await held.ready()
                    cancelled.cancel()
                    await finished.wait()
            assert capture.calls == []
            assert run.events("reserve") == []
        finally:
            await held.finish()
        assert capture.calls == []
        assert run.events("reserve") == []


async def test_real_adapter_outbound_disables_hidden_retries_and_protects_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from agents.extensions.models.litellm_model import LitellmModel

    calls: list[dict[str, Any]] = []

    async def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ok"}}])

    monkeypatch.setattr(litellm, "acompletion", completion)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            await invoke(
                metered(LitellmModel("anthropic/claude-sonnet-4-6")),
                ModelSettings(
                    extra_args={
                        "num_retries": 9,
                        "max_retries": 9,
                        "metadata": {"call_id": "forged"},
                    }
                ),
            )
        # The stub deliberately omits a callback; settle its prepared call now.
        await drain(scope, wait=False)
        assert len(calls) == 1
        assert calls[0]["num_retries"] == calls[0]["max_retries"] == 0
        assert calls[0]["metadata"]["call_id"] == run.calls()[0].id
        assert calls[0]["max_tokens"] == SPEC.max_tokens


async def test_user_raised_cancellation_is_a_diagnostic_not_a_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeModel([])
    secret = "private-model-cancellation"

    async def fail(*args: Any, **kwargs: Any) -> ModelResponse:
        raise asyncio.CancelledError(secret)

    monkeypatch.setattr(capture, "get_response", fail)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope), pytest.raises(RuntimeError) as caught:
            await invoke(metered(capture))
        assert not _is_core_control(caught.value)
        assert_secret_absent(caught.value, secret)
        await drain(scope, wait=False)


@pytest.mark.parametrize("site", ["final", "replan", "policy"])
@pytest.mark.parametrize("kind", ["cleanup", "overflow", "fence", "budget"])
@pytest.mark.parametrize("unwrap", [False, True])
async def test_carried_control_reaches_each_goal_handler_without_another_turn(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    kind: str,
    unwrap: bool,
) -> None:
    from _engine.test_goal import _plan
    from _fakes.board import candidate
    from _fakes.planners import SequencePlanner

    from swarm_arena._engine.goal import GoalRunner

    plan = _plan(teams=1, rounds=1)
    runner = GoalRunner(env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([plan]))
    core = {
        "budget": _core_error(LimitReached, "budget", "budget used up"),
        "cleanup": _stop_cleanup_error(),
        "overflow": _core_error(
            LimitReached,
            "usage_overflow",
            "budget exhausted by usage overflow",
            _category=_SignalCategory.TERMINAL,
        ),
        "fence": _core_error(Fenced, "run ownership lost"),
    }[kind]
    carrier = _sdk_control(core)
    assert carrier is not None
    carrier.args = ("private-carrier-mutation",)
    carrier.__dict__.update(message="private-carrier-mutation", limit="budget")

    @function_tool(failure_error_function=None)
    async def stop() -> str:
        """Report a registered terminal control."""
        transported = _sdk_control(ExceptionGroup("private", [ValueError("private"), carrier]))
        assert transported is not None
        raise transported from None

    capture = FakeModel([FakeTurn(tool_calls=(("stop", {}),)), FakeTurn(text="must not run")])

    supervised = runner._supervised_step

    async def invoke_sdk(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("settle_on_stop"):
            return await supervised(*args, **kwargs)
        try:
            await Runner.run(
                Agent(name="test", model=capture, tools=[stop]),
                "go",
                run_config=SDKRunConfig(tracing_disabled=True),
            )
        except BaseException as error:
            # Component adapters unwrap after their owned accounting cleanup.
            raise _unwrap_control(error) if unwrap else error from None
        raise AssertionError("control must stop the SDK")

    monkeypatch.setattr(runner, "_supervised_step", invoke_sdk)
    operation = (
        runner._final([candidate("c", goal_id="g")])
        if site == "final"
        else runner._replan()
        if site == "replan"
        else runner._round_decision(1)
    )
    if kind == "budget":
        result = await operation
        await env.deps.journal.commit()
        if site == "final":
            assert result == []
            assert (runner.status, runner.reason) == ("no_result", "reserve_exhausted")
        elif site == "replan":
            assert result is False
            assert env.events("plan_rejected")[-1].payload["errors"] == [
                "planner error: LimitReached"
            ]
        else:
            assert (
                env.events("decision")[-1].payload["decision"]["policy"]
                == "policy error: LimitReached"
            )
    elif kind == "overflow" and site == "final":
        assert await operation == []
        assert env.deps.stops.run_budget
        assert (runner.status, runner.reason) == ("no_result", "run budget")
        outcome = await runner.run()
        assert outcome.finalists == []
        assert (outcome.status, outcome.reason) == ("no_result", "run budget")
        assert env.events("no_result")[-1].payload["reason"] == "run budget"
        assert env.events("goal_finished")
    else:
        with pytest.raises(Exception, match=r"timed out|overflow|ownership lost") as caught:
            await operation
        assert _is_core_control(caught.value)
        assert_secret_absent(caught.value, "private-carrier-mutation")
        assert not env.events("plan_rejected")
        assert not env.events("decision")
    assert len(capture.calls) == 1
    assert not env.events("judge_error")


async def test_custom_schema_is_not_reentered_on_the_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import litellm
    from agents.extensions.models.litellm_model import LitellmModel

    from tests.llm.test_upstream_facts import _DictSchema

    schema = _DictSchema()
    original = schema.json_schema
    loop_thread = threading.current_thread()
    seen: list[dict[str, Any]] = []

    def prepare_schema() -> dict[str, Any]:
        assert threading.current_thread() is not loop_thread, "schema executed on loop"
        return original()

    async def completion(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ok"}}])

    monkeypatch.setattr(schema, "json_schema", prepare_schema)
    monkeypatch.setattr(litellm, "acompletion", completion)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope):
            await invoke(metered(LitellmModel("anthropic/claude-sonnet-4-6")), output_schema=schema)
        await drain(scope, wait=False)
    assert seen[0]["response_format"]["json_schema"]["schema"] == original()


async def test_hidden_fallbacks_are_not_delegated(tmp_path: Path) -> None:
    capture = FakeModel([FakeTurn(text="ok")])
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")):
            await invoke(
                metered(capture),
                ModelSettings(
                    extra_args={
                        "fallbacks": ["fake/other"],
                        "context_window_fallback_dict": {"fake/m": "fake/other"},
                    }
                ),
            )
    sent = capture.calls[0]["model_settings"].extra_args
    assert "fallbacks" not in sent
    assert "context_window_fallback_dict" not in sent


async def test_global_fallbacks_refuse_before_reserving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    capture = FakeModel([])
    monkeypatch.setattr(litellm, "model_fallbacks", ["fake/other"])
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")), pytest.raises(ValueError, match="fallbacks"):
            await invoke(metered(capture))
        assert run.events("reserve") == []
        assert capture.calls == []


@pytest.mark.parametrize("channel", ["extra_args", "extra_body", "nested_body"])
async def test_all_provider_attempt_controls_are_stripped(tmp_path: Path, channel: str) -> None:
    from agents.retry import ModelRetrySettings

    controls = {
        "num_retries": 4,
        "max_retries": 4,
        "retry_policy": {"InternalServerErrorRetries": 4},
        "model_group_retry_policy": {"m": {"InternalServerErrorRetries": 4}},
        "retry_strategy": "constant_retry",
        "retry_after": 1,
        "fallbacks": ["fake/other"],
        "context_window_fallback_dict": {"fake/m": "fake/other"},
        "context_window_fallbacks": [{"fake/m": ["fake/other"]}],
        "content_policy_fallbacks": [{"fake/m": ["fake/other"]}],
        "max_fallbacks": 4,
        "default_fallbacks": ["fake/other"],
        "fallback_on_unsupported_params": True,
        "disable_fallbacks": False,
    }
    payload = {**controls, "seed": 42}
    caller = (
        ModelSettings(extra_args={"extra_body": payload})
        if channel == "nested_body"
        else ModelSettings(extra_args=payload)
        if channel == "extra_args"
        else ModelSettings(extra_body=payload)
    )
    caller.retry = ModelRetrySettings(max_retries=2)
    caller.metadata = {"retry_policy": "ordinary metadata"}
    capture = FakeModel([FakeTurn(text="ok")])
    async with open_llm_run(tmp_path, config=config()) as run:
        with llm_scope(run.scope("manager")):
            await invoke(metered(capture), caller)
    sent = capture.calls[0]["model_settings"]
    assert sent.extra_args["num_retries"] == sent.extra_args["max_retries"] == 0
    actual = sent.extra_args["extra_body"] if channel == "nested_body" else getattr(sent, channel)
    for key in controls:
        if channel == "extra_args" and key in {"num_retries", "max_retries"}:
            continue
        assert key not in actual, key
    assert actual["seed"] == 42
    assert payload == {**controls, "seed": 42}
    assert sent.retry == caller.retry
    assert sent.metadata["retry_policy"] == "ordinary metadata"


@pytest.mark.parametrize("direct", [False, True])
async def test_overflow_advice_and_hook_match_reservation_while_settlement_waits(
    tmp_path: Path, direct: bool
) -> None:
    from swarm_arena._budget.amount import USD_LIMIT
    from swarm_arena._errors import _core_category

    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        tranche = scope.tranche
        rid = await tranche.reserve_call(Amount(usd=1 if direct else 0.01))
        call = tranche._calls[rid]
        entry = run.ledger._entries[rid if call.batch is None else call.batch.id]
        other = Tranche(run.ledger, ":run", "g", size=Amount(usd=0.25))
        await other.reserve_call(Amount(usd=0.01))
        await entry.lock.acquire()
        task = asyncio.create_task(tranche._settle_call(rid, Amount(usd=USD_LIMIT), overflow=True))
        try:
            await anyio.wait_all_tasks_blocked()
            assert not task.done()
            assert tranche._outcome.overflow
            assert not tranche.can_cover(Amount())
            assert not other.can_cover(Amount())
            with pytest.raises(LimitReached) as refused:
                await other.reserve_call(Amount())
            assert _core_category(refused.value) is _SignalCategory.TERMINAL
            deps = ManagerDeps(ctx=cast(AgentContext, object()), links=CallLinks())
            with llm_scope(scope), pytest.raises(AgentsException) as hooked:
                await BudgetHooks().on_llm_start(
                    RunContextWrapper(deps),
                    Agent(name="test", model=metered(FakeModel([]))),
                    "sys",
                    [],
                )
            assert _core_category(_unwrap_control(hooked.value)) is _SignalCategory.TERMINAL
        finally:
            entry.lock.release()
            await task
            await other.close()


@pytest.mark.parametrize("form", ["forged", "copy", "deepcopy", "pickle", "subclass"])
def test_unissued_sdk_carriers_are_public_diagnostics(form: str) -> None:
    import copy
    import pickle

    from swarm_arena._errors import _exception_type_name
    from swarm_arena._outcomes import _classify_outcome
    from swarm_arena.llm._hooks import _SDKControl

    class Derived(_SDKControl):
        pass

    carrier = _sdk_control(_stop_cleanup_error())
    assert carrier is not None
    carrier.args = ("private-value",)
    carrier.__dict__.update(message="private-value")
    forged = {
        "forged": lambda: _SDKControl("private-value"),
        "copy": lambda: copy.copy(carrier),
        "deepcopy": lambda: copy.deepcopy(carrier),
        "pickle": lambda: pickle.loads(pickle.dumps(carrier)),
        "subclass": lambda: Derived("private-value"),
    }[form]()
    result = _classify_outcome(forged)
    assert not _is_core_control(result)
    assert isinstance(result, RuntimeError)
    assert _exception_type_name(result) == "RuntimeError"
    assert "_SDKControl" not in str(result)
    assert_secret_absent(result, "private-value")


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_shared_classifier_preserves_raw_carrier_precedence(fenced: bool, cancelled: bool) -> None:
    from swarm_arena._errors import _core_category
    from swarm_arena._outcomes import _classify_outcome

    cores = [_stop_cleanup_error(), _core_error(LimitReached, "budget", "budget used up")]
    if fenced:
        cores.append(_core_error(Fenced, "run ownership lost"))
    carriers = [_sdk_control(core) for core in cores]
    assert all(isinstance(carrier, Exception) for carrier in carriers)
    group = ExceptionGroup("private", [ExceptionGroup("nested", cast(list[Exception], carriers))])
    if cancelled or fenced:
        with pytest.raises(asyncio.CancelledError if cancelled else Fenced):
            _classify_outcome(group, cancelled=cancelled)
    else:
        result = _classify_outcome(group)
        assert _core_category(result) is _SignalCategory.TERMINAL
        assert str(result) == "stop cleanup timed out"


def test_shared_classifier_unwraps_deep_carrier_groups_iteratively() -> None:
    from swarm_arena._errors import _core_category
    from swarm_arena._outcomes import _classify_outcome

    carrier = _sdk_control(_stop_cleanup_error())
    assert carrier is not None
    error: Exception = carrier
    for _ in range(1500):
        error = ExceptionGroup("private", [error])
    result = _classify_outcome(error)
    assert _core_category(result) is _SignalCategory.TERMINAL


@pytest.mark.parametrize("drained", [False, True])
async def test_refused_call_registration_refunds_before_provider_dispatch(
    tmp_path: Path,
    drained: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena.llm import _hooks

    failure = ValueError("registration refused")

    def refuse(*args: Any, **kwargs: Any) -> str:
        raise failure

    fake = FakeModel([FakeTurn(text="unused")])
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        if drained:
            await drain(scope, wait=False)
        else:
            monkeypatch.setattr(_hooks, "open_call", refuse)
        with (
            llm_scope(scope),
            pytest.raises(ValueError, match=r"registration refused|scope is drained") as caught,
        ):
            await invoke(metered(fake))
        if not drained:
            assert caught.value is failure
        assert fake.calls == []
        assert scope.tranche.pending() == 0
        await scope.tranche.close()
        await run.journal.commit()
        assert run.calls() == []
        assert run.charged(scope) == Amount()


@pytest.mark.parametrize("stop_at", [1, 2])
async def test_admission_gate_refusal_releases_unused_reservation(
    tmp_path: Path, stop_at: int
) -> None:
    refusal = _core_error(LimitReached, "stop", "call admission closed")
    checked = 0

    def check() -> None:
        nonlocal checked
        checked += 1
        if checked == stop_at:
            raise refusal

    capture = FakeModel([FakeTurn(text="unused")])
    model = metered(capture, admission_check=check)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("judge")
        with llm_scope(scope), pytest.raises(LimitReached) as caught:
            await invoke(model)
        assert caught.value is refusal
        assert capture.calls == []
        assert model.call_ids == []
        assert run.calls() == []
        assert scope.tranche.pending() == 0
        assert run.charged(scope) == Amount()
        assert len(run.events("reserve")) == stop_at - 1


@pytest.mark.parametrize(
    "kind",
    [
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
        "APIConnectionError",
        "Timeout",
        "BadRequestError",
    ],
)
async def test_provider_failures_are_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    import httpx
    import litellm

    from swarm_arena._errors import ProviderFailed, _core_category

    error = getattr(litellm, kind)(
        **(
            {"response": httpx.Response(403, request=httpx.Request("GET", "https://test.invalid"))}
            if kind == "PermissionDeniedError"
            else {}
        ),
        message="private-sentinel credit balance",
        model="private-model",
        llm_provider="private-provider",
    )
    fake = FakeModel([])

    from swarm_arena.llm._logger import CallLogger

    logged: list[bool] = []

    async def fail(*args: Any, **kwargs: Any) -> ModelResponse:
        await CallLogger().async_log_failure_event(
            {"exception": error, "litellm_params": {"metadata": args[2].metadata}},
            None,
            None,
            None,
        )
        logged.append(True)
        raise error

    monkeypatch.setattr(fake, "get_response", fail)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope), pytest.raises(ProviderFailed) as caught:
            await invoke(metered(fake))
        assert caught.value.provider == SPEC.provider
        assert _core_category(caught.value) is _SignalCategory.PROVIDER
        assert_secret_absent(caught.value, "private-sentinel")
        assert_secret_absent(caught.value, "private-provider")
        await drain(scope, wait=False)
        assert logged == [True]
        assert len(run.calls()) == 1
        assert run.calls()[0].error == "estimated: provider spend unknown; charged in full"
        assert "private-sentinel" not in repr(run.events("llm_call")) + caplog.text

        def artifacts_are_safe() -> bool:
            return all(
                b"private-sentinel" not in p.read_bytes()
                for p in tmp_path.rglob("*")
                if p.is_file()
            )

        assert await anyio.to_thread.run_sync(artifacts_are_safe)


async def test_other_errors_are_not_provider_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    error = litellm.BadRequestError(message="invalid tools", model="m", llm_provider="fake")
    fake = FakeModel([])

    async def fail(*args: Any, **kwargs: Any) -> ModelResponse:
        raise error

    monkeypatch.setattr(fake, "get_response", fail)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        with llm_scope(scope), pytest.raises(litellm.BadRequestError) as caught:
            await invoke(metered(fake))
        assert caught.value is error
        await drain(scope, wait=False)


async def test_probe_sends_max_tokens_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena.llm import _hooks

    fake = FakeModel([FakeTurn(text="ok")])
    monkeypatch.setattr(_hooks, "build_model", lambda spec: fake)
    async with open_llm_run(
        tmp_path,
        config=config(
            cache=CachePolicy(
                backend="disk",
                disk_dir=str(tmp_path / "probe-cache"),
                roles={"manager": {"read": True, "write": True}},
            )
        ),
    ) as run:
        scope = run.scope("manager")
        assert await _hooks.probe_provider(SPEC.provider, scope)
        assert len(run.calls()) == 1
        sent = fake.calls[0]["model_settings"]
        assert sent.max_tokens == 1
        assert "cache" not in (sent.extra_args or {})
        assert run.charged(scope).usd == pytest.approx(sum(c.cost_usd for c in run.calls()))


@pytest.mark.parametrize("record", [True, False])
async def test_probe_drains_before_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: bool
) -> None:
    from _fakes.llm_components import LateFakeModel

    from swarm_arena.llm import _hooks

    release = anyio.Event()
    fake = LateFakeModel([FakeTurn(text="pong")], delay_s=0 if record else None, release=release)
    monkeypatch.setattr(_hooks, "build_model", lambda spec: fake)
    async with open_llm_run(tmp_path, config=config()) as run:
        scope = run.scope("manager")
        finished = anyio.Event()

        async def probe() -> None:
            assert await _hooks.probe_provider("fake", scope)
            finished.set()

        try:
            with anyio.fail_after(10):
                async with anyio.create_task_group() as group:
                    group.start_soon(probe)
                    await fake.ready.wait()
                    assert not finished.is_set()
                    release.set()
                    await finished.wait()
            assert set(scope.call_ids) == {c.id for c in run.calls()}
            assert run.charged(scope).usd == pytest.approx(sum(c.cost_usd for c in run.calls()))
            assert run.calls()[0].status == ("ok" if record else "error")
            assert (run.calls()[0].error is not None) is (not record)
        finally:
            release.set()
            await fake.aclose()


@pytest.mark.parametrize("kind", ["missing", "judge", "failed", "terminal", "forged"])
async def test_probe_catalog_roles_and_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import litellm

    from swarm_arena._errors import ProviderFailed
    from swarm_arena.llm import _hooks

    fake = FakeModel([FakeTurn(text="pong")])
    cfg = config()
    if kind == "judge":
        cfg = cfg.model_copy(
            update={"models": ModelCatalog({"m": SPEC.model_copy(update={"roles": ["judge"]})})}
        )
    if kind in {"failed", "terminal", "forged"}:

        async def fail(*args: Any, **kwargs: Any) -> ModelResponse:
            if kind == "terminal":
                raise _stop_cleanup_error()
            if kind == "forged":
                raise ProviderFailed("private-provider")
            raise litellm.ServiceUnavailableError(
                message="private-value", model="m", llm_provider="fake"
            )

        monkeypatch.setattr(fake, "get_response", fail)
    monkeypatch.setattr(_hooks, "build_model", lambda spec: fake)
    async with open_llm_run(tmp_path, config=cfg) as run:
        scope = run.scope("manager")
        if kind in {"terminal", "forged"}:
            with pytest.raises(RuntimeError) as caught:
                await _hooks.probe_provider("fake", scope)
            assert _core_category(caught.value) is not _SignalCategory.PROVIDER
            assert_secret_absent(caught.value, "private-provider")
        else:
            assert await _hooks.probe_provider(
                "absent" if kind == "missing" else "fake", scope
            ) is (kind != "failed")
            if kind == "judge":
                assert run.calls()[0].status == "ok"
                assert run.calls()[0].error is None
            if kind == "missing":
                assert run.calls() == []


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("insufficient_quota", True),
        ("invalid_tools", False),
        ("invalid_request_error", True),
        ("bad_request", True),
    ],
)
def test_provider_structured_code_precedes_message(code: str, expected: bool) -> None:
    import litellm

    from swarm_arena.llm._hooks import is_provider_failure

    error = litellm.BadRequestError(
        message="credit balance private-value", model="m", llm_provider="fake"
    )
    error.body = {"error": {"code": code}}
    assert is_provider_failure(error) is expected


@pytest.mark.parametrize("nested", [False, True])
async def test_provider_stop_keeps_sdk_owned_siblings_until_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    from _fakes.run import all_events, plan
    from _fakes.tools import PyTool

    from swarm_arena import Ref, Swarm
    from swarm_arena._agent.runtime import AgentRuntime
    from swarm_arena._budget.llm_scope import LLMScope
    from swarm_arena._engine.round import RoundRunner
    from swarm_arena.llm import AgentsSDKManager, _manager
    from tests.llm.test_manager import NestedActionTool, finish, register
    from tests.llm.test_manager import config as manager_config

    release = tmp_path / "release"
    from swarm_arena._exec.executor import Job
    from swarm_arena._exec.local import LocalExecutor
    from swarm_arena._model.tooling import ToolOutput

    ready = anyio.Event()
    started = 0
    original_submit = LocalExecutor.submit

    async def submit(self: LocalExecutor, job: Job) -> ToolOutput:
        async def on_started(info: Job.Started) -> None:
            nonlocal started
            if job.on_started is not None:
                await job.on_started(info)
            started += 1
            if started == 2:
                ready.set()

        return await original_submit(self, replace(job, on_started=on_started))

    monkeypatch.setattr(LocalExecutor, "submit", submit)
    code = f"""
import json, os, time
from pathlib import Path
n = json.load(open('swarm/request.json'))['params']['n']
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
json.dump({{'data': {{'y': float(n)}}}}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))
"""
    cfg = manager_config(
        limits=Limits(max_usd=100, max_tokens=10_000_000),
        run_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        arena_root=str(tmp_path / "arena"),
    )
    current = plan(
        rounds=1, tool=(NestedActionTool if nested else PyTool)({"code": code, "grace_s": 3})
    )
    current.teams[0].agents[0] = (
        current.teams[0]
        .agents[0]
        .model_copy(update={"manager": Ref.of(AgentsSDKManager(model="opus"))})
    )
    calls = (
        (("nested", {"x": 1}), ("nested", {"x": 2}))
        if nested
        else (("run_tool", {"inputs": {"n": 1}}), ("run_tool", {"inputs": {"n": 2}}))
    )
    fake = register([FakeTurn(tool_calls=calls), finish("must not run")])
    runtimes: list[AgentRuntime] = []
    original_runtime, original_drain = RoundRunner._runtime, _manager.drain

    async def runtime(self: RoundRunner, *args: Any, **kwargs: Any) -> AgentRuntime:
        rt = await original_runtime(self, *args, **kwargs)
        runtimes.append(rt)
        return rt

    drained: list[bool] = []

    async def drain_checked(scope: LLMScope, *, wait: bool) -> list[str]:
        assert len([e for e in all_events(cfg) if e.type == "tool_end"]) == 2
        drained.append(True)
        return await original_drain(scope, wait=wait)

    monkeypatch.setattr(RoundRunner, "_runtime", runtime)
    monkeypatch.setattr(_manager, "drain", drain_checked)

    async def fail_provider() -> None:
        try:
            with anyio.fail_after(15):
                await ready.wait()
            assert runtimes
            await runtimes[0].stops.fail("fake", source="judge")
        finally:
            release.touch()

    try:
        with anyio.fail_after(30):
            async with anyio.create_task_group() as group:
                group.start_soon(fail_provider)
                await Swarm(cfg).run(current, run_id="provider-owned")
    finally:
        release.touch()
    events = all_events(cfg)
    ends = [e for e in events if e.type == "tool_end"]
    assert len(ends) == 2
    assert all(e.payload["candidate"]["output"]["status"] == "ok" for e in ends)
    assert len(fake.calls) == 1
    assert drained == [True]
    assert {e.payload["reservation"]["id"] for e in events if e.type == "reserve"} == {
        e.payload["id"] for e in events if e.type == "settle"
    }
    assert next(e for e in events if e.type == "agent_status").payload["status"] == "stopped"


async def test_provider_stop_collects_scripted_manager_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.run import all_events, plan
    from _fakes.tools import ProviderPyTool

    from swarm_arena import Swarm
    from swarm_arena._agent.runtime import AgentRuntime
    from swarm_arena._engine.round import RoundRunner
    from tests.llm.test_manager import config as manager_config

    ready = tmp_path / "ready"
    code = f"""
import json, os, time
from pathlib import Path
json.dump({{'data': {{'y': 42.0}}}}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))
Path({str(ready)!r}).touch()
time.sleep(60)
"""
    cfg = manager_config(
        limits=Limits(max_usd=100, max_tokens=10_000_000),
        run_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        arena_root=str(tmp_path / "arena"),
    )
    current = plan(rounds=1, tool=ProviderPyTool({"code": code, "grace_s": 0.1}))
    runtimes: list[AgentRuntime] = []
    original = RoundRunner._runtime

    async def runtime(self: RoundRunner, *args: Any, **kwargs: Any) -> AgentRuntime:
        rt = await original(self, *args, **kwargs)
        runtimes.append(rt)
        return rt

    monkeypatch.setattr(RoundRunner, "_runtime", runtime)

    async def fail_provider() -> None:
        with anyio.fail_after(15):
            while not await anyio.Path(ready).exists():  # noqa: ASYNC110 - subprocess readiness
                await anyio.sleep(0.01)
        await runtimes[0].stops.fail("fake", source="judge")

    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(fail_provider)
            await Swarm(cfg).run(current, run_id="provider-scripted")
    events = all_events(cfg)
    ends = [e for e in events if e.type == "tool_end"]
    assert len(ends) == 1
    output = ends[0].payload["candidate"]["output"]
    assert output["status"] == "cancelled"
    assert output["data"] == {"y": 42.0}
    assert {e.payload["reservation"]["id"] for e in events if e.type == "reserve"} == {
        e.payload["id"] for e in events if e.type == "settle"
    }
    status = next(e for e in events if e.type == "agent_status")
    assert (status.payload["status"], status.payload["reason"]) == (
        "stopped",
        "provider unavailable",
    )


async def test_probe_preserves_upstream_type_without_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    from swarm_arena._errors import _exception_type_name
    from swarm_arena.llm import _hooks

    fake = FakeModel([])

    async def fail(*args: Any, **kwargs: Any) -> ModelResponse:
        raise litellm.BadRequestError(
            message="max_tokens unsupported private-probe-sentinel", model="m", llm_provider="fake"
        )

    monkeypatch.setattr(fake, "get_response", fail)
    monkeypatch.setattr(_hooks, "build_model", lambda spec: fake)
    async with open_llm_run(tmp_path, config=config()) as run:
        with pytest.raises(RuntimeError) as caught:
            await _hooks.probe_provider("fake", run.scope("manager"))
        assert _exception_type_name(caught.value) == "BadRequestError"
        assert_secret_absent(caught.value, "private-probe-sentinel")
        assert "private-probe-sentinel" not in repr(run.events("llm_call"))
