import asyncio
import json
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import Mock

import anyio
import pytest
from _fakes.engine import take_over_retrying, with_executor, with_limits
from _fakes.finished import FinishedRun, run_again
from _fakes.leases import force_take_over
from _fakes.run import WRITE, WRITE_THEN_SLEEP, all_events, events, plan, run_dir, wait_tool_started
from _fakes.tools import PyTool, RecordingExecutor

from swarm_arena import Ref, RunConfig, ScriptedManager
from swarm_arena._budget.ledger import Ledger
from swarm_arena._engine.run import RunRunner, _require_asyncio
from swarm_arena._engine.swarm import Swarm
from swarm_arena._errors import Fenced, PlanError, _core_error
from swarm_arena._journal.lease import read_owner
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.ids import IdGen
from swarm_arena._model.results import LLMCall
from swarm_arena._model.tooling import Estimate
from swarm_arena._store.local import LocalStore
from swarm_arena.manager import AgentContext, RoundReport


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("site", ["startup_models", "close_executor"])
async def test_startup_and_shutdown_preserve_terminal(site: str, nested: bool) -> None:
    from unittest.mock import AsyncMock

    from swarm_arena._engine.run import _check_startup_models, _close_executor
    from swarm_arena._errors import _core_category, _SignalCategory, _stop_cleanup_error

    def fail() -> None:
        error = _stop_cleanup_error()
        if nested:
            raise BaseExceptionGroup("engine", [BaseExceptionGroup("hook", [error])])
        raise error

    operation = (
        _check_startup_models(RunConfig(), Mock(models=fail))
        if site == "startup_models"
        else _close_executor(
            Mock(aclose=AsyncMock(side_effect=fail)),
            RunConfig(settle_timeout_s=2),
            failed=False,
        )
    )
    with pytest.raises(RuntimeError) as caught:
        await operation
    assert _core_category(caught.value) is _SignalCategory.TERMINAL


class RecordingManager(ScriptedManager):
    results: ClassVar[list[LLMCall | None]] = []
    pause_second: ClassVar[anyio.Event | None] = None
    cache_second: ClassVar[bool] = False
    order: ClassVar[list[str]] = []

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        from _fakes.llm_run import record_kwargs

        from swarm_arena._budget.llm_scope import current_llm_scope
        from swarm_arena._journal.llm_record import record_call

        if ctx.round == 2 and self.pause_second is not None:
            self.pause_second.set()
            await anyio.sleep_forever()
        scope = current_llm_scope()
        assert scope is not None
        self.order.append(f"round-{ctx.round}")
        meta = {
            "run_id": scope.run_id,
            "goal_id": scope.goal_id,
            "team": scope.team,
            "agent": scope.agent,
            "round": str(scope.round),
            "role": scope.role,
        }
        result = await record_call(
            **record_kwargs(
                meta,
                settle=False,
                litellm_call_id=f"lc-{ctx.round}" if self.cache_second else "lc-1",
                cache_hit=self.cache_second and ctx.round == 2,
                cache_identity="b" * 64,
            )
        )
        self.results.append(result)
        return await super().run_round(ctx)


def _recording_plan(manager_path: str | None = None) -> Any:
    current = plan(rounds=2)
    current.teams[0].agents[0] = (
        current.teams[0]
        .agents[0]
        .model_copy(
            update={
                "manager": Ref(
                    path=manager_path or f"{__name__}:RecordingManager",
                    config=current.teams[0].agents[0].manager.config,
                ),
            }
        )
    )
    return current


async def _pause_recording_run(config: RunConfig, current: Any) -> str:
    paused = RecordingManager.pause_second
    assert paused is not None
    swarm = Swarm(config)
    task = asyncio.create_task(swarm.run(current))
    try:
        with anyio.fail_after(15):
            await paused.wait()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert swarm.run_id is not None
    return swarm.run_id


async def test_resume_binds_committed_llm_calls(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine import run as run_module

    calls: list[RunConfig] = []
    monkeypatch.setattr(run_module, "_run_setups", [calls.append])
    monkeypatch.setattr(RecordingManager, "results", [])
    monkeypatch.setattr(RecordingManager, "order", [])
    monkeypatch.setattr(RecordingManager, "pause_second", anyio.Event())
    run_id = await _pause_recording_run(run_config, _recording_plan())
    assert len(RecordingManager.results) == 1
    assert RecordingManager.results[0] is not None
    monkeypatch.setattr(RecordingManager, "pause_second", None)
    await Swarm(run_config).resume(run_id, force=True)
    assert RecordingManager.results[1] is None
    assert len(RunState.from_events(all_events(run_config)).llm_calls) == 1
    assert calls == [run_config, run_config]


async def test_resume_points_cache_hits_to_originals_without_payloads(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._model.settings import TracingConfig

    config = run_config.model_copy(update={"tracing": TracingConfig(store_payloads=False)})
    monkeypatch.setattr(RecordingManager, "results", [])
    monkeypatch.setattr(RecordingManager, "order", [])
    monkeypatch.setattr(RecordingManager, "cache_second", True)
    monkeypatch.setattr(RecordingManager, "pause_second", anyio.Event())
    run_id = await _pause_recording_run(config, _recording_plan())
    monkeypatch.setattr(RecordingManager, "pause_second", None)
    await Swarm(config).resume(run_id, force=True)
    first, hit = RecordingManager.results
    assert first is not None
    assert hit is not None
    assert hit.cached_from_call_id == first.id
    assert first.request_ref is None
    assert hit.request_ref is None


async def test_resume_imports_plan_refs_before_run_setups(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine import run as run_module
    from swarm_arena._model.plan import Plan

    module = tmp_path / "setup_mod.py"
    module.write_text(
        "from swarm_arena._engine.run import register_run_setup\n"
        f"from {__name__} import RecordingManager\n"
        "def setup(config):\n"
        "    RecordingManager.order.append('setup')\n"
        "register_run_setup(setup)\n"
        "class Manager(RecordingManager):\n"
        "    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(run_module, "_run_setups", [])
    monkeypatch.setattr(RecordingManager, "results", [])
    monkeypatch.setattr(RecordingManager, "order", [])
    monkeypatch.setattr(RecordingManager, "pause_second", anyio.Event())
    current = _recording_plan("setup_mod:Manager")
    current = Plan._from_data(current.model_dump(), base_dir=tmp_path)
    try:
        run_id = await _pause_recording_run(run_config, current)
        assert RecordingManager.order == ["setup", "round-1"]
        sys.modules.pop("setup_mod", None)
        monkeypatch.setattr(run_module, "_run_setups", [])
        monkeypatch.setattr(RecordingManager, "order", [])
        monkeypatch.setattr(RecordingManager, "pause_second", None)
        await Swarm(run_config).resume(run_id, force=True)
        assert RecordingManager.order == ["setup", "round-2"]
    finally:
        sys.modules.pop("setup_mod", None)


async def test_setup_failure_unbinds_and_releases_services(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._engine import run as run_module
    from swarm_arena._journal.llm_record import _bindings

    def broken(_: RunConfig) -> None:
        raise ValueError("SETUP_SECRET")

    monkeypatch.setattr(run_module, "_run_setups", [broken])
    with pytest.raises(RuntimeError, match="run setup failed: ValueError") as caught:
        await Swarm(run_config).run(plan(rounds=1), run_id="setup-failure")
    assert_secret_absent(caught.value, "SETUP_SECRET")
    assert "setup-failure" not in _bindings
    owner = read_owner(LocalStore(Path(run_config.run_root) / "setup-failure"))
    assert owner is not None
    assert owner[0].expires_at <= utc_now()
    assert not events(run_config, "round_started")


async def test_completed_resume_bootstrap_does_not_construct_manager(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RecordingManager, "results", [])
    monkeypatch.setattr(RecordingManager, "order", [])
    first = await Swarm(run_config).run(_recording_plan(), run_id="completed")

    def forbidden(*_: Any, **__: Any) -> None:
        pytest.fail("completed manager was constructed during registration bootstrap")

    monkeypatch.setattr(RecordingManager, "__init__", forbidden)
    assert await Swarm(run_config).resume("completed", force=True) == first


async def test_resume_binding_keeps_the_rebased_acquired_head_clock(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine import resume as resume_module
    from swarm_arena._engine import run as run_module
    from swarm_arena._engine.stops import RunClock
    from swarm_arena._journal.event import Event
    from swarm_arena._journal.lease import Head, Lease
    from swarm_arena._journal.llm_record import _bindings
    from swarm_arena._journal.segment import encode_segment, segment_name
    from swarm_arena._store.io import _storage_io

    await Swarm(run_config).run(plan(rounds=1), run_id="rebase")
    clocks: list[RunClock] = []
    setup_clocks: list[RunClock] = []
    starts: list[float] = []
    acquire = resume_module._acquire_lease
    advanced = False

    class Clock(RunClock):
        def __init__(self, elapsed_before: float) -> None:
            super().__init__(elapsed_before)
            clocks.append(self)
            starts.append(self._start)

    def advance() -> None:
        store = LocalStore(Path(run_config.run_root) / "rebase")
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].head is not None
        doc, etag = owner
        old = doc.head
        assert old is not None
        event = Event(
            type="run_config",
            epoch=doc.epoch,
            seq=old.seq + 1,
            ts=utc_now(),
            run_id="rebase",
            context_id="rebase",
            payload={"config": run_config.model_dump(), "run_elapsed_s": 20.0},
        )
        name = segment_name(doc.epoch, event.seq)
        store.create(name, encode_segment(old.segment, [event]))
        Lease(store, doc, etag, ttl_s=60).advance(
            Head(epoch=doc.epoch, seq=event.seq, segment=name)
        )

    async def race(*args: Any, **kwargs: Any) -> Lease:
        nonlocal advanced
        if not advanced:
            assert len(clocks) == 1
            await _storage_io(advance)
            advanced = True
        return await acquire(*args, **kwargs)

    def setup(_: RunConfig) -> None:
        setup_clocks.append(_bindings["rebase"].clock)
        assert _bindings["rebase"].clock._before == 20.0

    monkeypatch.setattr(resume_module, "RunClock", Clock)
    monkeypatch.setattr(resume_module, "_acquire_lease", race)
    monkeypatch.setattr(run_module, "_run_setups", [setup])
    await Swarm(run_config).resume("rebase", force=True)
    assert len(clocks) == 1
    assert setup_clocks == clocks
    assert clocks[0]._start == starts[0]
    assert "rebase" not in _bindings


async def test_graceful_stop_teardown_expiry_is_cleanup_error(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._budget.llm_scope import current_llm_scope
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._errors import _core_category, _SignalCategory
    from swarm_arena._journal.llm_record import _bindings, drain

    config = run_config.model_copy(update={"settle_timeout_s": 2.0})
    run_id = "graceful-teardown"
    entered, exited = anyio.Event(), anyio.Event()
    run_round = ScriptedManager.run_round
    close = Tranche.close

    async def stop_after_round(self: ScriptedManager, ctx: AgentContext) -> RoundReport:
        result = await run_round(self, ctx)
        scope = current_llm_scope()
        assert scope is not None
        await drain(scope, wait=False)
        _bindings[scope.run_id].stops.stop = True
        return result

    async def held_close(self: Tranche) -> Any:
        binding = _bindings.get(run_id)
        if binding is not None and binding.closing:
            assert binding.stops.stop
            assert not binding.stops.stop_now
            entered.set()
            try:
                await anyio.sleep_forever()
            finally:
                exited.set()
        return await close(self)

    monkeypatch.setattr(ScriptedManager, "run_round", stop_after_round)
    monkeypatch.setattr(Tranche, "close", held_close)
    with (
        anyio.fail_after(20),
        pytest.raises(RuntimeError, match=r"^stop cleanup timed out$") as caught,
    ):
        await Swarm(config).run(plan(rounds=1), run_id=run_id)
    assert entered.is_set()
    assert exited.is_set()
    assert _core_category(caught.value) is _SignalCategory.TERMINAL
    assert run_id not in _bindings
    assert events(config, "goal_finished")


async def test_run_folder_layout(run_config: RunConfig) -> None:
    with anyio.fail_after(30):
        out = await Swarm(run_config).run(plan())
    root = run_dir(run_config)
    for key in [
        "run.toml",
        "manifest.json",
        "control/owner.json",
        "goals/g/plan.r0.toml",
        "goals/g/solution.json",
    ]:
        assert (root / key).exists()
    assert list(out) == ["g"]
    assert len(out["g"]) == 2
    owner = read_owner(LocalStore(root))
    assert owner is not None
    assert owner[0].expires_at < utc_now()


async def test_duplicate_goal_ids_rejected(run_config: RunConfig) -> None:
    with pytest.raises(PlanError, match="duplicate goal id"):
        await Swarm(run_config).run([plan(), plan()])
    assert not await anyio.to_thread.run_sync(Path(run_config.run_root).exists)


async def test_every_r0_committed_before_any_goal_starts(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_concurrent_goals=1)
    await Swarm(cfg).run([plan(goal_id="a"), plan(goal_id="b")])
    r0 = events(cfg, "plan_revision")
    assert {e.goal_id for e in r0} == {"a", "b"}
    assert max(e.seq for e in r0) < min(e.seq for e in events(cfg, "goal_started"))
    state = RunState.from_events(all_events(cfg))
    assert [e.type for e in state.ledger_events[:2]] == ["run_config", "run_started"]
    ledger = Ledger.replay(state, journal=cast(JournalWriter, Mock()), ids=IdGen(state.run_id, 2))
    assert not ledger.run_exhausted()


async def test_llm_plan_with_noncatalog_ref_imports_nothing(run_config: RunConfig) -> None:
    p = plan(tool_ref=Ref(path="_fakes.import_canary:Tool")).model_copy(update={"origin": "llm"})
    sys.modules.pop("_fakes.import_canary", None)
    with pytest.raises(ValueError, match="planner catalog"):
        await Swarm(run_config).run(p)
    assert "_fakes.import_canary" not in sys.modules
    assert not await anyio.to_thread.run_sync(Path(run_config.run_root).exists)


async def test_run_binds_record_and_calls_run_setups(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.llm_run import record_kwargs

    from swarm_arena._engine import run as run_module
    from swarm_arena._journal.llm_record import record_call

    calls: list[RunConfig] = []
    monkeypatch.setattr(run_module, "_run_setups", [])
    run_module.register_run_setup(calls.append)
    await Swarm(run_config).run(plan(rounds=1))
    assert calls == [run_config]
    run_id = run_dir(run_config).name
    # Explicit unbound misuse is distinct from a recognizably late callback.
    with pytest.raises(ValueError, match="not bound"):
        await record_call(**record_kwargs({"run_id": run_id}))


async def test_prebuild_never_imports_a_rejected_r0_ref(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SWARM_ARENA_TEST_CANARY", str(tmp_path / "imported"))
    sys.modules.pop("_fakes.import_canary", None)
    proposed = plan().model_copy(update={"origin": "llm"})
    proposed.teams[0].agents[0] = (
        proposed.teams[0]
        .agents[0]
        .model_copy(update={"manager": Ref(path="_fakes.import_canary:Tool")})
    )
    with pytest.raises(ValueError, match="not in the planner catalog"):
        await Swarm(run_config).run(proposed)
    assert not (tmp_path / "imported").exists()
    assert "_fakes.import_canary" not in sys.modules


async def test_estimate_over_limit_refuses(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_usd=0.5)
    with pytest.raises(RuntimeError, match="worst-case estimate"):
        await Swarm(cfg).run(plan())
    assert not await anyio.to_thread.run_sync(Path(cfg.run_root).exists)


async def test_final_slices_over_reserve_refuses(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_usd=100.0)
    with pytest.raises(RuntimeError, match="reserve cannot hold the final slices"):
        await Swarm(cfg).run(plan(converge_team=True, converge_agents=3))
    assert not await anyio.to_thread.run_sync(Path(cfg.run_root).exists)


def test_require_asyncio_outside_a_loop_raises() -> None:
    with pytest.raises(RuntimeError, match="asyncio backend only"):
        _require_asyncio()


@pytest.mark.parametrize("slots", [2, 3])
async def test_goal_queue_slots_and_agents_total(run_config: RunConfig, slots: int) -> None:
    cfg = with_limits(run_config, max_concurrent_goals=slots, max_agents_total=4)
    plans = [
        plan(goal_id=g, agents=2).model_copy(
            update={"policy": Ref(path="swarm_arena:SuccessiveHalving", config={"keep": 1.0})}
        )
        for g in ("a", "b", "c")
    ]
    await Swarm(cfg).run(plans)
    active = peak = 0
    for e in all_events(cfg):
        active += (e.type == "goal_started") - (e.type == "goal_finished")
        peak = max(peak, active)
    assert peak == 2


async def test_unused_goal_budget_returns_to_pool(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_usd=10.0, max_concurrent_goals=1)
    await Swarm(cfg).run([plan(goal_id=g, rounds=1) for g in ("a", "b")])
    shares = [next(iter(e.payload["shares"].values()))["usd"] for e in events(cfg, "round_started")]
    assert shares == [4.25, 8.5]


async def test_run_returns_empty_list_for_no_result_goal(run_config: RunConfig) -> None:
    assert (await Swarm(run_config).run(plan(check="_fakes.tools:AlwaysFailCheck")))["g"] == []


async def test_run_timeout_stops_everything(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, run_timeout_s=1.0)
    with anyio.fail_after(25):
        out = await Swarm(cfg).run(plan(tool=PyTool({"code": WRITE_THEN_SLEEP})))
    sol = json.loads((run_dir(cfg) / "goals/g/solution.json").read_text())
    assert (sol["status"], sol["reason"], out["g"]) == ("no_result", "no passing candidate", [])
    assert events(cfg, "converge_started")[0].payload["select_only"] is True


async def test_queued_goal_no_result_on_run_timeout(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_concurrent_goals=1, run_timeout_s=2.0)
    out = await Swarm(cfg).run(
        [plan(goal_id="a", tool=PyTool({"code": WRITE_THEN_SLEEP})), plan(goal_id="b")]
    )
    assert {
        e.goal_id: e.payload["reason"] for e in events(cfg, "no_result") if e.goal_id == "b"
    } == {"b": "stopped"}
    assert out["b"] == []
    assert not [e for e in events(cfg, "goal_started") if e.goal_id == "b"]
    assert json.loads((run_dir(cfg) / "goals/b/solution.json").read_text())["reason"] == "stopped"


async def test_queued_goal_no_result_on_run_budget(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_concurrent_goals=1, max_usd=10.0)
    spend = PyTool({"code": WRITE.format("{'data': {'y': 1.0}, 'usage': {'usd': 9.0}}")})
    out = await Swarm(cfg).run(
        [plan(goal_id="a", rounds=1, tool=spend), plan(goal_id="b", rounds=1)]
    )
    assert {
        e.goal_id: e.payload["reason"] for e in events(cfg, "no_result") if e.goal_id == "b"
    } == {"b": "stopped"}
    assert out["b"] == []


async def test_fenced_cancels_inflight_and_raises(run_config: RunConfig) -> None:
    runner = RunRunner(
        run_config,
        [plan(tool=PyTool({"code": WRITE_THEN_SLEEP}))],
        run_id=None,
        approve=None,
        replanner=None,
    )
    raised: list[BaseException] = []

    async def capture() -> None:
        try:
            await runner.run()
        except BaseException as exc:
            raised.append(exc)

    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(capture)
            store = await wait_tool_started(run_config)
            take_over_retrying(store)
    assert [type(e) for e in raised] == [Fenced]
    assert not events(run_config, "tool_end")


@pytest.mark.parametrize(
    "types",
    [
        {"tool_start", "tool_end", "learning"},
        {"agent_status"},
    ],
)
@pytest.mark.parametrize("agents", [1, 2])
async def test_every_event_has_context_and_task_ids(
    run_config: RunConfig, types: set[str], agents: int
) -> None:
    await Swarm(run_config).run(plan(agents=agents))
    recorded = all_events(run_config)
    if agents == 2:
        pruned = [
            e for e in recorded if e.type == "agent_status" and e.payload["status"] == "pruned"
        ]
        assert [(e.goal_id, e.team, e.agent, e.round) for e in pruned] == [("g", "t0", "a1", 1)]
        assert all(e.task_id == f"{e.goal_id}/{e.team}/{e.agent}#r{e.round}" for e in pruned)
    assert all(e.context_id == e.run_id for e in recorded)
    assert all(e.task_id for e in recorded if e.type in types)


async def test_executor_factory_spy(run_config: RunConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = RecordingExecutor()
    monkeypatch.setattr("swarm_arena._engine.run._build_executor", lambda cfg: spy)
    await Swarm(run_config).run(plan(rounds=1))
    assert len(spy.jobs) == 1


async def test_capacity_warning_reaches_approve(run_config: RunConfig) -> None:
    p = plan(teams=2)
    for team in p.teams:
        team.agents[0] = team.agents[0].model_copy(
            update={"resources": team.agents[0].resources.model_copy(update={"cpus": 1_000_000})}
        )
    seen: list[str] = []

    async def approve(estimate: Estimate, warnings: Any) -> bool:
        seen.extend(warnings)
        return False

    with pytest.raises(RuntimeError, match="not approved"):
        await Swarm(with_executor(run_config, max_concurrent=1)).run(p, approve=approve)
    assert any("Executor.capacity" in w for w in seen)


async def test_base_dir_import_rule(run_config: RunConfig, tmp_path: Path) -> None:
    module = tmp_path / "local_run_module.py"
    module.write_text('def square(x: float) -> dict[str, float]:\n    return {"y": x * x}\n')
    p = plan(
        tool_ref=Ref(
            path="swarm_arena:FunctionTool", config={"function": "local_run_module:square"}
        )
    )
    path = tmp_path / "plan.toml"
    path.write_text(p.dumps())
    from swarm_arena import Plan

    try:
        out = await Swarm(run_config).run(Plan.load(path))
        assert len(out["g"]) == 2
        assert events(run_config, "plan_revision")[0].payload["base_dir"] == str(tmp_path)
        assert sys.path.count(str(tmp_path)) == 1
    finally:
        sys.modules.pop("local_run_module", None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


@pytest.mark.parametrize("converge_team", [False, True])
async def test_one_admission_shared_by_every_goal(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, converge_team: bool
) -> None:
    from swarm_arena._engine.goal import GoalRunner
    from swarm_arena._engine.round import EngineDeps

    admissions: list[object] = []
    original = GoalRunner.__init__

    def initialize(self: GoalRunner, deps: EngineDeps, **kwargs: Any) -> None:
        admissions.append(deps.admission)
        original(self, deps, **kwargs)

    monkeypatch.setattr(GoalRunner, "__init__", initialize)
    await Swarm(run_config).run(
        [plan(goal_id=g, rounds=1, converge_team=converge_team) for g in ("a", "b", "c")]
    )
    assert len(admissions) == 3
    assert admissions[0] is not None
    assert all(a is admissions[0] for a in admissions)


async def test_effective_round_and_converge_estimates(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from swarm_arena import RoundSpec
    from swarm_arena._model.tooling import ToolRequest

    seen: list[tuple[float | None, int, dict[str, Any]]] = []

    def estimate(self: PyTool, request: ToolRequest) -> Estimate:
        assert threading.current_thread().daemon
        seen.append((self.config.timeout_s, request.round, request.round_overrides))
        return Estimate(usd=self.config.timeout_s)

    monkeypatch.setattr(PyTool, "estimate", estimate)
    p = plan(rounds=2, tool=PyTool({"code": "pass", "timeout_s": 1}), converge_team=True)
    p = p.model_copy(
        update={
            "rounds": [
                RoundSpec(tool_overrides={"timeout_s": 2}),
                RoundSpec(tool_overrides={"timeout_s": 4}),
            ],
            "converge": p.converge.model_copy(
                update={"round": RoundSpec(tool_overrides={"timeout_s": 5})}
            ),
        }
    )

    async def approve(estimate: Estimate, warnings: Any) -> bool:
        assert estimate.usd == pytest.approx((2 * 3 * 4 + 2 * 3 * 5) / 0.85, abs=1e-6)
        return False

    with pytest.raises(RuntimeError, match="not approved"):
        await Swarm(run_config).run(p, approve=approve)
    assert (2, 1, {"timeout_s": 2}) in seen
    assert (4, 2, {"timeout_s": 4}) in seen
    assert (5, 3, {"timeout_s": 5}) in seen


async def test_estimate_failure_hides_values(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._model.tooling import ToolRequest

    secret = "private-estimate-sentinel"

    def estimate(self: PyTool, request: ToolRequest) -> Estimate:
        raise ValueError(secret)

    monkeypatch.setattr(PyTool, "estimate", estimate)
    with pytest.raises(RuntimeError, match="estimate failed: ValueError") as caught:
        await Swarm(run_config).run(plan(tool=PyTool({"code": "pass"})))
    assert_secret_absent(caught.value, secret)
    assert not await anyio.to_thread.run_sync(Path(run_config.run_root).exists)


async def test_final_slice_reserve_uses_exact_microdollars(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._budget.amount import Amount

    cfg = with_limits(run_config, max_usd=0.00026)
    monkeypatch.setattr(
        "swarm_arena._engine.run._estimates",
        lambda plans, config: (Amount(), {"g": Amount(usd=0.000039)}),
    )

    async def refuse(estimate: Estimate, warnings: Any) -> bool:
        return False

    with pytest.raises(RuntimeError, match="not approved"):
        await Swarm(cfg).run(plan(), approve=refuse)


async def test_stop_latched_during_dispatch_keeps_remaining_goals_queued(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine.goal import GoalRunner
    from swarm_arena._engine.round import EngineDeps
    from swarm_arena._engine.stops import StopState
    from swarm_arena._journal.event import Event, EventDraft

    states: list[StopState] = []
    original_init = GoalRunner.__init__
    original_append = JournalWriter.append_durable

    def initialize(self: GoalRunner, deps: EngineDeps, **kwargs: Any) -> None:
        states.append(deps.stops)
        original_init(self, deps, **kwargs)

    async def append(self: JournalWriter, draft: EventDraft) -> Event:
        result = await original_append(self, draft)
        if draft.type == "goal_started" and draft.goal_id == "a":
            states[0].stop = True
        return result

    monkeypatch.setattr(GoalRunner, "__init__", initialize)
    monkeypatch.setattr(JournalWriter, "append_durable", append)
    await Swarm(run_config).run([plan(goal_id=g, rounds=1) for g in ("a", "b")])
    assert [e.goal_id for e in events(run_config, "goal_started")] == ["a"]
    assert {
        e.goal_id: e.payload["reason"] for e in events(run_config, "no_result") if e.goal_id == "b"
    } == {"b": "stopped"}


async def test_executor_capacity_failure_hides_values_and_closes(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena import Resources

    secret = "private-capacity-sentinel"

    class BrokenCapacity(RecordingExecutor):
        closed = False

        async def capacity(self) -> Resources | None:
            raise ValueError(secret)

        async def aclose(self) -> None:
            self.closed = True

    executor = BrokenCapacity()
    monkeypatch.setattr("swarm_arena._engine.run._build_executor", lambda cfg: executor)
    with pytest.raises(RuntimeError, match="executor capacity failed: ValueError") as caught:
        await Swarm(run_config).run(plan())
    assert_secret_absent(caught.value, secret)
    assert executor.closed
    assert not await anyio.to_thread.run_sync(Path(run_config.run_root).exists)


async def test_executor_shutdown_failure_hides_values(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.secrecy import assert_secret_absent

    secret = "private-shutdown-sentinel"

    class BrokenShutdown(RecordingExecutor):
        async def aclose(self) -> None:
            raise ValueError(secret)

    monkeypatch.setattr("swarm_arena._engine.run._build_executor", lambda cfg: BrokenShutdown())
    with pytest.raises(RuntimeError, match="executor shutdown failed: ValueError") as caught:
        await Swarm(run_config).run(plan(rounds=1))
    assert_secret_absent(caught.value, secret)


async def test_fenced_remains_bare_when_executor_shutdown_fails(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena import Resources

    class FencedExecutor(RecordingExecutor):
        async def capacity(self) -> Resources | None:
            raise ExceptionGroup("lost owner", [_core_error(Fenced, "run ownership lost")])

        async def aclose(self) -> None:
            raise ValueError("shutdown failed")

    monkeypatch.setattr("swarm_arena._engine.run._build_executor", lambda cfg: FencedExecutor())
    with pytest.raises(Fenced) as caught:
        await Swarm(run_config).run(plan())
    assert type(caught.value) is Fenced


async def test_default_board_is_arena(finished_run: FinishedRun) -> None:
    from swarm_arena._model.board_types import Frontier

    cfg, run = finished_run.config, finished_run.run_id
    assert (Path(cfg.arena_root) / finished_run.namespace / "runs" / f"{run}.json").exists()
    for event in events(cfg, "frontier"):
        assert run in {
            mark.run_id for mark in Frontier.model_validate(event.payload["frontier"]).marks
        }


async def test_second_run_sees_first_but_never_selects_it(finished_run: "FinishedRun") -> None:
    from swarm_arena._board.arena import ArenaBoard
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._model.board_types import Frontier
    from swarm_arena._store.base import open_store

    run2 = "second"
    result = await Swarm(finished_run.config).run(finished_run.plan, run_id=run2)
    assert result[finished_run.plan.goal.id]
    assert all(c.run_id == run2 for c in result[finished_run.plan.goal.id])
    store = open_store(finished_run.config.run_root).child(run2)
    committed = read_committed(store)
    assert committed is not None
    front = next(e for e in committed[1] if e.type == "frontier")
    frontier = Frontier.model_validate(front.payload["frontier"])
    assert {m.run_id for m in frontier.marks} == {finished_run.run_id, run2}
    board = ArenaBoard(arena_root=finished_run.config.arena_root)
    board.open(run2, finished_run.config.run_root, [finished_run.namespace])
    try:
        view = board.snapshot(finished_run.plan.goal.key(), frontier)
        assert view._visible_candidate_ids()
        assert all(
            view.get_candidate(id).run_id == finished_run.run_id
            for id in view._visible_candidate_ids()
        )
    finally:
        board.close()
    solution = json.loads(store.read(f"goals/{finished_run.plan.goal.id}/solution.json").data)
    assert solution["finalists"]
    assert all(item["candidate"]["run_id"] == run2 for item in solution["finalists"])


async def test_past_run_match_runs_again(finished_run: "FinishedRun") -> None:
    from swarm_arena._board.leaderboard import format_value
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.replay import RunState
    from swarm_arena._store.base import open_store

    states = []
    for run_id in (finished_run.run_id, await run_again(finished_run)):
        committed = read_committed(open_store(finished_run.config.run_root).child(run_id))
        assert committed is not None
        states.append(RunState.from_events(committed[1]))
    first, second = [
        next(c for c in state.candidates.values() if c.agent == "gd-1" and c.round == 1)
        for state in states
    ]
    scores = [s for s in states[0].scores.values() if s.candidate_id == first.id and s.judge == "f"]
    assert first.id != second.id
    assert second.run_id != first.run_id
    assert f"past-run match {first.id}: f={format_value(scores[-1].value)}" in second.warnings
    assert all(
        states[1].candidates[id].run_id == second.run_id
        for id in states[1].goals[finished_run.plan.goal.id].repeat_keys
    )


async def test_past_run_candidate_as_parent(finished_run: "FinishedRun") -> None:
    from swarm_arena._board.arena import ArenaBoard
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.replay import RunState
    from swarm_arena._store.artifacts import ArtifactStore
    from swarm_arena._store.base import open_store

    board = ArenaBoard(arena_root=finished_run.config.arena_root)
    board.open("", "", [finished_run.namespace])
    try:
        view = board.snapshot(finished_run.plan.goal.key(), board.frontier())
        first = view.best(finished_run.plan.goal.key(), 1, judge="f")[0]
        plan2 = finished_run.plan.model_copy(deep=True)
        spec = next(a for t in plan2.teams for a in t.agents if a.name == "gd-1")
        spec.manager.config["requests"][0]["parents"] = "best"
        run2 = await run_again(finished_run, plan=plan2)
        committed = read_committed(open_store(finished_run.config.run_root).child(run2))
        assert committed is not None
        state = RunState.from_events(committed[1])
        second = next(c for c in state.candidates.values() if c.agent == "gd-1" and c.round == 1)
        assert second.parent_ids == [first.id]
        assert second.request.parents[0].run_id == finished_run.run_id
        assert second.request.parents[0].data == first.output.data
        artifacts = ArtifactStore(open_store(finished_run.config.artifact_root))
        for file in second.request.parents[0].files:
            assert artifacts.get_bytes(file.sha256)
        latest = board.snapshot(finished_run.plan.goal.key(), board.frontier())
        assert (second.id, first.id) in latest.lineage(second.id).edges
        assert first.id not in state.goals[finished_run.plan.goal.id].final_order
    finally:
        board.close()


@pytest.mark.s3
async def test_run_with_all_roots_on_s3(s3_url: str, run_config: RunConfig) -> None:
    from swarm_arena._store.base import open_store

    cfg = run_config.model_copy(
        update={
            "run_root": f"{s3_url}/runs",
            "arena_root": f"{s3_url}/arena",
            "artifact_root": f"{s3_url}/artifacts",
        }
    )
    result = await Swarm(cfg).run(plan(rounds=1), run_id="remote")
    assert result["g"]
    assert open_store(cfg.run_root).child("remote").read("goals/g/solution.json").data


async def test_custom_board_open_and_close_on_run_and_resume(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Sequence

    from swarm_arena._board.arena import ArenaBoard
    from swarm_arena._journal.lease import read_owner
    from swarm_arena._store.base import open_store

    opened: list[str] = []
    closed: list[str] = []
    original_open, original_close = ArenaBoard.open, ArenaBoard.close

    def open_board(self: ArenaBoard, run_id: str, run_root: str, namespaces: Sequence[str]) -> None:
        owner = read_owner(open_store(run_root).child(run_id))
        assert owner is not None
        assert owner[0].head is not None
        assert namespaces == ["n"]
        opened.append(run_id)
        original_open(self, run_id, run_root, namespaces)

    def close_board(self: ArenaBoard) -> None:
        closed.append(self.config.arena_root)
        original_close(self)

    monkeypatch.setattr(ArenaBoard, "open", open_board)
    monkeypatch.setattr(ArenaBoard, "close", close_board)
    root = str(tmp_path / "custom-arena")
    cfg = run_config.model_copy(
        update={
            "board": Ref(
                path="swarm_arena.board:ArenaBoard",
                config={"arena_root": root, "cache_dir": str(tmp_path / "custom-cache")},
            )
        }
    )
    assert (await Swarm(cfg).run(plan(rounds=1), run_id="custom"))["g"]
    await Swarm(cfg).resume("custom", force=True)
    assert opened == ["custom", "custom"]
    assert closed == [root, root]
    assert (Path(root) / "n/runs/custom.json").exists()
    assert not await anyio.to_thread.run_sync(Path(cfg.arena_root).exists)


def test_finished_run_supports_sync_consumers(finished_run: FinishedRun) -> None:
    assert finished_run.config_path.read_text() == finished_run.config.dumps()
    assert len(finished_run.plan.rounds) == 2
    assert all(
        agent.manager.config["post_learnings"]
        for team in finished_run.plan.teams
        for agent in team.agents
    )
    assert (
        Path(finished_run.config.run_root) / finished_run.run_id / "control/owner.json"
    ).exists()


async def test_board_open_failure_closes_board(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Sequence

    from swarm_arena._board.arena import ArenaBoard

    closed: list[bool] = []
    original_open, original_close = ArenaBoard.open, ArenaBoard.close

    def fail_open(self: ArenaBoard, run_id: str, run_root: str, namespaces: Sequence[str]) -> None:
        original_open(self, run_id, run_root, namespaces)
        raise RuntimeError("board open failed")

    def close(self: ArenaBoard) -> None:
        closed.append(True)
        original_close(self)

    monkeypatch.setattr(ArenaBoard, "open", fail_open)
    monkeypatch.setattr(ArenaBoard, "close", close)
    with pytest.raises(RuntimeError, match="board open failed: RuntimeError"):
        await Swarm(run_config).run(plan(rounds=1))
    assert closed == [True]


async def test_custom_board_construction_errors_hide_values(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._board.base import Board

    original = Ref.build
    secret = "BOARD_CONSTRUCTOR_SENTINEL"

    def build(self: Ref, base: type[Any]) -> Any:
        if base is Board:
            raise RuntimeError(secret) from ValueError(secret)
        return original(self, base)

    monkeypatch.setattr(Ref, "build", build)
    config = run_config.model_copy(
        update={
            "board": Ref(
                path="swarm_arena.board:ArenaBoard", config={"arena_root": run_config.arena_root}
            )
        }
    )
    with pytest.raises(RuntimeError, match="construction failed") as caught:
        await Swarm(config).run(plan(rounds=1))
    assert_secret_absent(caught.value, secret)


async def test_run_end_writes_review_export(finished_run: FinishedRun) -> None:
    store = LocalStore(Path(finished_run.config.run_root) / finished_run.run_id)
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head is not None
    head = owner[0].head
    prefix = f"review/e{head.epoch}-{head.seq}/"
    assert store.read(prefix + "review.csv").data.startswith(
        b"id,goal,team,agent,scores,verdict,score,notes"
    )
    assert b"<table>" in store.read(prefix + "index.html").data


@pytest.mark.parametrize("resumed", [False, True])
async def test_review_export_failure_releases_run_services(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, resumed: bool
) -> None:
    from swarm_arena._engine import run as run_module
    from swarm_arena._errors import StorageError

    run_id = "export-cleanup"
    if resumed:
        await Swarm(run_config).run(plan(rounds=1), run_id=run_id)
    board_closed: list[int] = []
    executor_closed: list[int] = []
    original_board = run_module._close_board
    original_executor = run_module._close_executor

    async def board_close(*args: Any, **kwargs: Any) -> None:
        await original_board(*args, **kwargs)
        board_closed.append(1)

    async def executor_close(*args: Any, **kwargs: Any) -> None:
        await original_executor(*args, **kwargs)
        executor_closed.append(1)

    async def fail(*args: Any, **kwargs: Any) -> None:
        raise StorageError("review export unavailable")

    monkeypatch.setattr(run_module, "_close_board", board_close)
    monkeypatch.setattr(run_module, "_close_executor", executor_close)
    monkeypatch.setattr(run_module, "write_review_export", fail)
    operation = (
        Swarm(run_config).resume(run_id)
        if resumed
        else Swarm(run_config).run(plan(rounds=1), run_id=run_id)
    )
    with pytest.raises(StorageError, match="review export unavailable"):
        await operation
    owner = read_owner(LocalStore(Path(run_config.run_root) / run_id))
    assert owner is not None
    assert owner[0].expires_at < utc_now()
    assert board_closed == [1]
    assert executor_closed == [1]


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("interrupt", ["cancel", "fence"])
async def test_stalled_completion_export_ends_owner_before_worker_returns(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, resumed: bool, interrupt: str
) -> None:
    import asyncio
    import threading

    from swarm_arena._exec.threads import run_sync_daemon
    from swarm_arena._journal.lease import Lease

    # A whole run needs at least a three-second lease under parallel CI load. Its one-second
    # renewal interval still detects the successor inside the unchanged three-second bound.
    config = run_config.model_copy(update={"settle_timeout_s": 1.0, "lease_ttl_s": 3.0})
    run_id = "export-cancel"
    if resumed:
        await Swarm(run_config).run(plan(rounds=1), run_id=run_id)
    store = LocalStore(Path(config.run_root) / run_id)
    prefix = f"review/e{2 if resumed else 1}-"
    entered, release, returned, caller_done = (threading.Event() for _ in range(4))
    creates: list[str] = []
    workers: list[threading.Thread] = []
    renewals_after_return: list[int] = []
    failures: list[BaseException] = []
    scopes: list[anyio.CancelScope] = []
    real_create, real_renew = LocalStore.create, Lease.renew

    def create(self: LocalStore, key: str, data: bytes) -> str:
        if key.startswith(prefix):
            creates.append(key)
            if len(creates) == 1:
                workers.append(threading.current_thread())
                entered.set()
                assert release.wait(10), "worker release deadline"
            try:
                return real_create(self, key, data)
            finally:
                returned.set()
        return real_create(self, key, data)

    def renew(self: Lease) -> None:
        if caller_done.is_set():
            renewals_after_return.append(self.doc.epoch)
        real_renew(self)

    async def caller() -> None:
        try:
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                if resumed:
                    await Swarm(config).resume(run_id)
                else:
                    await Swarm(config).run(plan(rounds=1), run_id=run_id)
        except BaseException as error:
            failures.append(error)
        finally:
            caller_done.set()

    monkeypatch.setattr(LocalStore, "create", create)
    monkeypatch.setattr(Lease, "renew", renew)
    task = asyncio.create_task(caller())
    services: list[asyncio.Task[Any]] = []
    successor: Lease | None = None
    try:
        assert await run_sync_daemon(entered.wait, 15)
        services = [t for t in asyncio.all_tasks() if t.get_name() == f"resume-{run_id}"]
        started = anyio.current_time()
        if interrupt == "cancel":
            scopes[0].cancel()
        else:
            successor = force_take_over(store, owner_id="successor", host="local", ttl_s=60)
        done, _ = await asyncio.wait([task], timeout=3 * config.settle_timeout_s)
        assert done, "caller retained by stalled completion export"
        assert anyio.current_time() - started < 3 * config.settle_timeout_s
        owner = read_owner(store)
        assert owner is not None
        if interrupt == "cancel":
            assert owner[0].expires_at <= utc_now()
            assert not failures
        else:
            assert owner[0].owner_id == "successor"
            assert len(failures) == 1
            assert isinstance(failures[0], Fenced)
        assert all(t.done() for t in services)
        assert not returned.is_set()
        release.set()
        assert await run_sync_daemon(returned.wait, 5)
        for worker in workers:
            await run_sync_daemon(worker.join, 5)
            assert not worker.is_alive()
        await asyncio.gather(task, *services)
        assert len(creates) == 1, "abandoned export started another file"
        assert not renewals_after_return
        assert read_owner(store) == owner
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(task, *services, return_exceptions=True), 5)
        for worker in workers:
            await run_sync_daemon(worker.join, 5)
            assert not worker.is_alive()
        if successor is not None:
            successor.release()


async def test_driving_timeout_is_not_a_stop_command(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine.control import read_status
    from swarm_arena._engine.goal import GoalOutcome, GoalRunner
    from swarm_arena._engine.round import EngineDeps

    driving = anyio.Event()
    original = GoalRunner.run

    async def run_goal(runner: GoalRunner) -> GoalOutcome:
        driving.set()
        return await original(runner)

    async def expire(runner: RunRunner, deps: EngineDeps) -> None:
        await driving.wait()
        deps.stops.run_timeout = True

    monkeypatch.setattr(GoalRunner, "run", run_goal)
    monkeypatch.setattr(RunRunner, "_watch_clock", expire)
    config = with_limits(run_config, max_concurrent_goals=1)
    with anyio.fail_after(20):
        result = await Swarm(config).run(
            [plan(goal_id="a", rounds=1), plan(goal_id="b", rounds=1)], run_id="r"
        )
    assert driving.is_set()
    assert result["b"] == []
    assert not events(config, "stop_requested")
    assert (await read_status(config.run_root, "r")).stop == "none"
    assert {e.goal_id: e.payload["reason"] for e in events(config, "goal_finished")}[
        "b"
    ] == "stopped"


@pytest.mark.parametrize("role", ["planner", "curator"])
async def test_replanner_and_curator_models_checked_before_start(
    role: str, run_config: RunConfig
) -> None:
    if role == "curator":
        pytest.importorskip("agents")
    from _fakes.planners import SequencePlanner

    class AliasPlanner(SequencePlanner):
        def models(self) -> tuple[str, ...]:
            return ("nope",)

    current = plan(rounds=1)
    config = run_config
    replanner = AliasPlanner([current]) if role == "planner" else None
    if role == "curator":
        pytest.importorskip("litellm")
        config = config.model_copy(
            update={
                "curator": Ref(path="_fakes.llm_components:AliasCurator", config={"model": "nope"})
            }
        )
    with pytest.raises(ValueError, match="nope"):
        await Swarm(config, replanner=replanner).run(current)
    assert await anyio.to_thread.run_sync(
        lambda: not Path(config.run_root).exists() or not list(Path(config.run_root).iterdir())
    )


async def test_replanner_capabilities_checked_before_start(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.planners import SequencePlanner

    from swarm_arena._planning import validate as validation
    from swarm_arena.config import ModelCatalog, ModelSpec

    class AliasPlanner(SequencePlanner):
        def models(self) -> tuple[str, ...]:
            return ("p",)

    monkeypatch.setattr(
        validation,
        "_MODEL_CHECKS",
        [lambda uses, config: (["planner capability refused"] if uses.get("planner") else [], [])],
    )
    config = run_config.model_copy(
        update={"models": ModelCatalog({"p": ModelSpec(model="fake/p", roles=["planner"])})}
    )
    current = plan(rounds=1)
    with pytest.raises(ValueError, match="planner capability refused"):
        await Swarm(config, replanner=AliasPlanner([current])).run(current)
    assert await anyio.to_thread.run_sync(
        lambda: not Path(config.run_root).exists() or not list(Path(config.run_root).iterdir())
    )


async def test_startup_model_warning_reaches_approval(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.planners import SequencePlanner

    from swarm_arena._planning import validate as validation
    from swarm_arena.config import ModelCatalog, ModelSpec

    class AliasPlanner(SequencePlanner):
        def models(self) -> tuple[str, ...]:
            return ("p",)

    monkeypatch.setattr(
        validation,
        "_MODEL_CHECKS",
        [lambda uses, config: ([], ["two-stage planner"] if uses.get("planner") else [])],
    )
    config = with_limits(run_config, max_tokens=1_000_000).model_copy(
        update={"models": ModelCatalog({"p": ModelSpec(model="fake/p", roles=["planner"])})}
    )
    seen = []

    async def approve(estimate: Estimate, warnings: Sequence[str]) -> bool:
        seen.extend(warnings)
        return False

    current = plan(rounds=1)
    with pytest.raises(RuntimeError, match="run not approved"):
        await Swarm(config, replanner=AliasPlanner([current])).run(current, approve=approve)
    assert "two-stage planner" in seen


async def test_startup_model_hook_is_abandonable(run_config: RunConfig) -> None:
    import threading

    from _fakes.planners import SequencePlanner

    entered = threading.Event()
    release = threading.Event()
    workers: list[threading.Thread] = []

    class HeldPlanner(SequencePlanner):
        def models(self) -> tuple[str, ...]:
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(15)
            return ()

    current = plan(rounds=1)
    task = asyncio.create_task(Swarm(run_config, replanner=HeldPlanner([current])).run(current))
    try:
        with anyio.fail_after(5):
            assert await anyio.to_thread.run_sync(entered.wait, 4)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()
    assert await anyio.to_thread.run_sync(lambda: not Path(run_config.run_root).exists())


def _planning_config(
    config: RunConfig, *, tokens: int | None = 1000, usd: float | None = None
) -> RunConfig:
    from swarm_arena import Limits
    from swarm_arena.config import ModelCatalog, ModelSpec

    return config.model_copy(
        update={
            "limits": Limits(max_tokens=tokens, max_usd=usd),
            "models": ModelCatalog(
                {"p": ModelSpec(model="unpriced/planner-test", roles=["planner"], max_tokens=10)}
            ),
            "settle_timeout_s": 2,
        }
    )


def _session_planner() -> Any:
    from _fakes.planners import SequencePlanner

    from swarm_arena import Goal, Plan

    class Planner(SequencePlanner):
        def models(self) -> tuple[str, ...]:
            return ("p",)

    return Planner([Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])])


async def test_planning_session_needs_max_tokens(run_config: RunConfig) -> None:
    from swarm_arena import Goal
    from swarm_arena._engine.run import planning_session

    with pytest.raises(ValueError, match="max_tokens"):
        async with planning_session(
            _planning_config(run_config, tokens=None),
            Goal(id="g", namespace="n"),
            _session_planner(),
        ):
            pytest.fail("invalid session yielded")
    assert not await anyio.Path(run_config.run_root).exists()


async def test_planning_session_rejects_an_unpriced_model_with_max_usd(
    run_config: RunConfig,
) -> None:
    from swarm_arena import Goal
    from swarm_arena._engine.run import planning_session

    with pytest.raises(ValueError, match="no known price"):
        async with planning_session(
            _planning_config(run_config, usd=1), Goal(id="g", namespace="n"), _session_planner()
        ):
            pytest.fail("unpriced session yielded")
    assert not await anyio.Path(run_config.run_root).exists()


async def test_planning_session_runs_model_checks(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena import Goal
    from swarm_arena._engine.run import planning_session
    from swarm_arena._planning import validate

    uses: list[Any] = []

    def check(roles: Any, config: RunConfig) -> tuple[list[str], list[str]]:
        uses.append(dict(roles))
        return [], ["planner warning"]

    monkeypatch.setattr(validate, "_MODEL_CHECKS", [check])
    config = _planning_config(run_config)
    async with planning_session(config, Goal(id="g", namespace="n"), _session_planner()):
        pass
    assert uses == [{"planner": ["p"]}]
    started = next(e for e in all_events(config) if e.type == "run_started")
    assert started.payload["planning"] is True
    assert "planner warning" in started.payload["warnings"]


@pytest.mark.parametrize("interrupted", [False, True])
async def test_planning_session_releases_lease_and_unbinds(
    run_config: RunConfig, interrupted: bool
) -> None:
    from _fakes.llm_run import record_kwargs

    from swarm_arena import Goal
    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.llm_scope import current_llm_scope
    from swarm_arena._engine.run import planning_session
    from swarm_arena._journal.llm_record import open_call, record_call
    from swarm_arena.manager import LimitReached
    from swarm_arena.tool import Usage

    config = _planning_config(run_config, tokens=20).model_copy(update={"reserve_fraction": 0})
    scope = None
    try:
        async with planning_session(
            config, Goal(id="g", namespace="n"), _session_planner()
        ) as scope:
            assert current_llm_scope() is scope
            rid = await scope.tranche.reserve_call(Amount(tokens=20))
            cid = open_call(
                scope,
                reservation_id=rid,
                model_alias="p",
                model="unpriced/planner-test",
                provider="unpriced",
                estimate=Amount(tokens=20),
            )
            if interrupted:
                raise asyncio.CancelledError()
            await record_call(
                **record_kwargs(
                    {
                        "run_id": scope.run_id,
                        "goal_id": "g",
                        "team": "",
                        "agent": "",
                        "round": "0",
                        "role": "planner",
                        "model_alias": "p",
                        "reservation_id": rid,
                        "call_id": cid,
                    },
                    cost_usd=0,
                    usage=Usage(input_tokens=20),
                )
            )
    except asyncio.CancelledError:
        assert interrupted
    assert scope is not None
    assert current_llm_scope() is None
    owner = read_owner(LocalStore(Path(config.run_root) / scope.run_id))
    assert owner is not None
    assert owner[0].expires_at <= utc_now()
    with pytest.raises(ValueError, match="not bound"):
        await record_call(**record_kwargs({"run_id": scope.run_id}, settle=False))
    state = RunState.from_events(all_events(config))
    assert len(state.llm_calls) == 1
    assert list(state.llm_calls) == scope.call_ids
    assert state.run_elapsed_s > 0
    assert [e.type for e in all_events(config)][:2] == ["run_config", "run_started"]
    recovered = Ledger.replay(state, journal=Mock(), ids=IdGen(scope.run_id, 2))
    with pytest.raises(LimitReached):
        await recovered.reserve(":run", Amount(tokens=1), kind="call", goal_id="g")


@pytest.mark.parametrize("stop", ["stop", "cancel"])
async def test_planning_session_stopped_setup_never_yields(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    from _fakes.llm_run import HeldOperation

    from swarm_arena import Goal
    from swarm_arena._engine import run as run_module
    from swarm_arena._journal import llm_record

    held = HeldOperation()
    monkeypatch.setattr(run_module, "_run_setups", [lambda config: held.wait()])
    config = _planning_config(run_config)
    yielded: list[bool] = []

    async def enter() -> None:
        async with run_module.planning_session(
            config, Goal(id="g", namespace="n"), _session_planner()
        ):
            yielded.append(True)

    task = asyncio.create_task(enter())
    try:
        await held.ready()
        (session,) = await anyio.to_thread.run_sync(lambda: list(Path(config.run_root).iterdir()))
        if stop == "stop":
            (session / "control" / "stop-e1").write_text("")
        else:
            task.cancel()
        with anyio.fail_after(10):
            if stop == "stop":
                with pytest.raises(RuntimeError, match=r"^planning session stopped$"):
                    await task
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert yielded == []
        assert session.name not in llm_record._bindings
        owner = read_owner(LocalStore(session))
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
        assert RunState.from_events(all_events(config)).goals["g"].finished
        before = all_events(config)
        await held.finish()
        assert all_events(config) == before
    finally:
        await held.finish()
        if not task.done():
            task.cancel()
            with suppress(BaseException):
                await task


async def test_queued_r0_carries_planning_session(run_config: RunConfig) -> None:
    from swarm_arena._engine.run import mark_planning_session

    config = with_limits(run_config, max_concurrent_goals=1)
    plans = [
        mark_planning_session(
            plan(goal_id=name, rounds=1), Mock(run_id="r-planning", call_ids=["llm-1"])
        )
        for name in ("g", "queued")
    ]
    await Swarm(config).run(plans)
    revisions = events(config, "plan_revision")
    assert {e.goal_id for e in revisions} == {"g", "queued"}
    assert all(
        (e.payload["planning_session"], e.payload["planner_call_ids"]) == ("r-planning", ["llm-1"])
        for e in revisions
    )
    state = RunState.from_events(all_events(config))
    assert all(
        entries[0][0].planning_session == "r-planning"
        and entries[0][0].planner_call_ids == ("llm-1",)
        for entries in state.plans.values()
    )


@pytest.mark.parametrize("stop", ["deadline", "stop", "cancel"])
async def test_planning_acquisition_is_supervised_before_owner_response(
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    from threading import Event as ThreadEvent

    from _fakes.llm_run import HeldOperation

    from swarm_arena import Goal
    from swarm_arena._engine import run as run_module
    from swarm_arena._journal.lease import Lease

    held, released = HeldOperation(), ThreadEvent()
    create, release = LocalStore.create, Lease.release
    setups: list[bool] = []
    yielded: list[bool] = []

    def hold_owner(store: LocalStore, key: str, data: bytes) -> Any:
        result = create(store, key, data)
        if key == "control/owner.json":
            held.wait()
        return result

    def release_owner(lease: Lease) -> None:
        release(lease)
        released.set()

    monkeypatch.setattr(LocalStore, "create", hold_owner)
    monkeypatch.setattr(Lease, "release", release_owner)
    monkeypatch.setattr(run_module, "_run_setups", [lambda config: setups.append(True)])
    config = _planning_config(run_config)
    config = with_limits(config, run_timeout_s=0.2 if stop == "deadline" else 30)

    async def enter() -> None:
        async with run_module.planning_session(
            config, Goal(id="g", namespace="n"), _session_planner()
        ):
            yielded.append(True)

    task = asyncio.create_task(enter())
    try:
        await held.ready()
        (session,) = await anyio.to_thread.run_sync(lambda: list(Path(config.run_root).iterdir()))
        if stop == "stop":
            (session / "control/stop-e1").write_text("")
        elif stop == "cancel":
            task.cancel()
        with (
            anyio.fail_after(3),
            pytest.raises(
                asyncio.CancelledError if stop == "cancel" else RuntimeError,
                match=None if stop == "cancel" else "planning session stopped",
            ),
        ):
            await asyncio.shield(task)
        assert setups == []
        assert yielded == []
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await held.finish()
        assert await anyio.to_thread.run_sync(released.wait, 5)
    owner = read_owner(LocalStore(session))
    assert owner is not None
    assert owner[0].expires_at <= utc_now()
    assert setups == []
    assert yielded == []


async def test_provider_paused_goal_keeps_queued_goal_out_until_completion(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.tools import ProviderPyTool

    from swarm_arena._budget.llm_scope import LLMScope
    from swarm_arena._engine import stops as stops_module

    cfg = with_limits(
        run_config,
        max_agents_total=1,
        max_concurrent_goals=2,
        max_usd=10.0,
        max_tool_runs_per_round=1,
        default_run_reservation=Estimate(usd=0.1),
    )
    code = """
import json, os
n = json.load(open('swarm/request.json'))['params']['n']
json.dump({'data': {}, 'failed_provider': 'fake'} if n == 1 else {'data': {'y': float(n)}},
          open(os.environ['SWARM_ARENA_RESULT'], 'w'))
"""
    g = plan(goal_id="g", rounds=2, tool=ProviderPyTool({"code": code}))
    g.teams[0].agents[0] = (
        g.teams[0]
        .agents[0]
        .model_copy(
            update={
                "manager": Ref.of(
                    ScriptedManager([{"inputs": {"n": n}, "round": n} for n in (1, 2)])
                )
            }
        )
    )
    h = plan(goal_id="h", rounds=1)
    entered, release = anyio.Event(), anyio.Event()

    async def probe(provider: str, scope: LLMScope) -> bool:
        assert (provider, scope.goal_id, scope.round) == ("fake", "g", 2)
        entered.set()
        await release.wait()
        return True

    monkeypatch.setattr(stops_module, "_probe", probe)

    async def run() -> None:
        try:
            await Swarm(cfg).run([g, h], run_id="provider-capacity")
        finally:
            entered.set()

    with anyio.fail_after(25):
        async with anyio.create_task_group() as group:
            group.start_soon(run)
            try:
                await entered.wait()
                assert not any(e.goal_id == "h" for e in events(cfg, "goal_started")), [
                    (e.type, e.payload)
                    for e in all_events(cfg)
                    if e.type in {"agent_status", "provider_failed", "tool_end", "decision"}
                ]
                assert events(cfg, "provider_failed")[0].payload["source"] == "tool"
            finally:
                release.set()
    ev = all_events(cfg)
    assert next(e.seq for e in ev if e.type == "goal_finished" and e.goal_id == "g") < next(
        e.seq for e in ev if e.type == "goal_started" and e.goal_id == "h"
    )
    ends = [e for e in ev if e.type == "tool_end" and e.goal_id == "g"]
    assert [(e.round, e.payload["candidate"]["output"]["status"]) for e in ends] == [
        (1, "provider_failed"),
        (2, "ok"),
    ]


@pytest.mark.parametrize("failure", ["ordinary", "budget", "nonbool"])
async def test_provider_probe_failure_is_goal_local(
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from _fakes.tools import ProviderPyTool

    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.llm_scope import LLMScope
    from swarm_arena._engine import stops as stops_module

    cfg = with_limits(
        run_config, max_usd=10.0, max_concurrent_goals=2, default_run_reservation=Estimate(usd=0.1)
    )
    code = """
import json, os
json.dump({'data': {}, 'failed_provider': 'fake'}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))
"""
    g = plan(goal_id="g", rounds=3, tool=ProviderPyTool({"code": code}))
    peer = plan(goal_id="g", rounds=3).teams[0].agents[0].model_copy(update={"name": "peer"})
    g.teams[0].agents.append(peer)
    h = plan(goal_id="h", rounds=3)
    calls: list[int] = []

    async def probe(provider: str, scope: LLMScope) -> bool:
        calls.append(scope.round)
        if failure == "nonbool":
            return "not a bool"  # ty: ignore[invalid-return-type]
        if failure == "budget":
            await scope.tranche.reserve_call(Amount(usd=1_000_000))
        raise KeyError("private-probe-message")

    monkeypatch.setattr(stops_module, "_probe", probe)
    await Swarm(cfg).run([g, h], run_id="probe-failure")
    ev = all_events(cfg)
    assert calls == ([2] if failure == "budget" else [2, 3])
    assert {e.goal_id for e in ev if e.type == "goal_finished"} == {"g", "h"}
    assert max(e.round for e in ev if e.type == "round_started" and e.goal_id == "h") == 3
    assert not any(e.payload.get("transition") == "recovered" for e in ev)
    if failure == "budget":
        assert max(e.round for e in ev if e.type == "round_started" and e.goal_id == "g") == 1
        assert (
            next(e for e in ev if e.type == "goal_finished" and e.goal_id == "g").payload["reason"]
            == "goal budget"
        )
    else:
        assert any(
            e.type == "agent_status" and e.goal_id == "g" and e.agent == "peer" and e.round == 3
            for e in ev
        )
        assert ("ValueError" if failure == "nonbool" else "KeyError") in caplog.text
    assert "private-probe-message" not in caplog.text + str([e.payload for e in ev])


@pytest.mark.parametrize("destination", ["queued", "converge"])
async def test_fresh_slots_probe_outstanding_provider_latch(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    from _fakes.tools import ProviderPyTool

    from swarm_arena import ConvergeSpec
    from swarm_arena._budget.llm_scope import LLMScope
    from swarm_arena._engine import stops as stops_module

    cfg = with_limits(
        run_config, max_usd=10.0, max_concurrent_goals=1, default_run_reservation=Estimate(usd=0.1)
    )
    code = """
import json, os
json.dump({'data': {}, 'failed_provider': 'fake'}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))
"""
    g = plan(goal_id="g", rounds=1, tool=ProviderPyTool({"code": code}))
    healthy = plan(
        goal_id="h",
        rounds=1,
        tool=ProviderPyTool(
            {
                "code": code.replace(
                    "{'data': {}, 'failed_provider': 'fake'}", "{'data': {'y': 10.0}}"
                )
            }
        ),
    )
    if destination == "queued":
        plans = [g, healthy]
        expected = ("fake", "h", 1)
    else:
        # A passing main-round peer permits a converge team to run.
        peer = plan(goal_id="g", rounds=1).teams[0].agents[0].model_copy(update={"name": "peer"})
        g.teams[0].agents.append(peer)
        healthy.teams[0].agents[0] = (
            healthy.teams[0]
            .agents[0]
            .model_copy(update={"manager": Ref.of(ScriptedManager([{"inputs": {"n": 50}}]))})
        )
        g = g.model_copy(
            update={
                "converge": ConvergeSpec(
                    mode="team", team=healthy.teams[0].model_copy(update={"name": "conv"})
                )
            }
        )
        plans = [g]
        expected = ("fake", "g", 2)
    seen: list[tuple[str, str, int]] = []

    async def probe(provider: str, scope: LLMScope) -> bool:
        seen.append((provider, scope.goal_id, scope.round))
        assert not any(
            e.type == "tool_start" and e.goal_id == expected[1] and e.round == expected[2]
            for e in all_events(cfg)
        )
        return True

    monkeypatch.setattr(stops_module, "_probe", probe)
    await Swarm(cfg).run(plans, run_id="outstanding-provider")
    assert seen == [expected]
    assert any(
        e.type == "tool_end"
        and e.goal_id == expected[1]
        and e.round == expected[2]
        and e.payload["candidate"]["output"]["status"] == "ok"
        for e in all_events(cfg)
    )


async def test_provider_stop_reaches_registered_agents_in_concurrent_goals(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Callable

    from _fakes.tools import ProviderPyTool

    from swarm_arena._agent.context import AgentContext
    from swarm_arena._agent.manager import RoundReport
    from swarm_arena._engine.stops import ProviderStops

    cfg = with_limits(
        run_config, max_usd=10.0, max_concurrent_goals=2, default_run_reservation=Estimate(usd=0.1)
    )
    g = plan(goal_id="g", rounds=1, tool=ProviderPyTool({"code": WRITE}))
    h = plan(goal_id="h", rounds=1, tool=ProviderPyTool({"code": WRITE}))
    h.teams[0].agents.append(
        plan(goal_id="h", rounds=1).teams[0].agents[0].model_copy(update={"name": "healthy"})
    )
    registered, stopped = anyio.Event(), anyio.Event()
    addresses: set[str] = set()
    services: list[ProviderStops] = []
    original_register, original_run = ProviderStops.register, ScriptedManager.run_round

    def register(
        self: ProviderStops, address: str, providers: frozenset[str], cancel: Callable[[], None]
    ) -> None:
        original_register(self, address, providers, cancel)
        addresses.add(address)
        services.append(self)
        if len(addresses) == 3:
            registered.set()

    async def run_round(self: ScriptedManager, ctx: AgentContext) -> RoundReport:
        await stopped.wait()
        return await original_run(self, ctx)

    async def fail() -> None:
        await registered.wait()
        await services[0].fail("fake", source="manager")
        stopped.set()

    monkeypatch.setattr(ProviderStops, "register", register)
    monkeypatch.setattr(ScriptedManager, "run_round", run_round)
    with anyio.fail_after(20):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(fail)
            await Swarm(cfg).run([g, h], run_id="provider-cross-goal")
    ev = all_events(cfg)
    failure = next(
        e for e in ev if e.type == "provider_failed" and e.payload.get("transition") == "failed"
    )
    assert failure.payload["stopped"] == ["g/t0/a0", "h/t0/a0"]
    statuses = {(e.goal_id, e.agent): e.payload["status"] for e in ev if e.type == "agent_status"}
    assert statuses == {
        ("g", "a0"): "stopped",
        ("h", "a0"): "stopped",
        ("h", "healthy"): "finished",
    }
    assert any(e.type == "tool_end" and e.agent == "healthy" for e in ev)
    assert not any(e.type == "tool_start" and e.agent == "a0" for e in ev)


def test_library_forced_interrupt_preserves_cancelled_shutdown(tmp_path: Path) -> None:
    import os
    import select
    import signal
    import subprocess
    import time

    code = """
import asyncio
import signal
from pathlib import Path
from typing import Any
import anyio
from _fakes.run import plan
from swarm_arena import RunConfig, Swarm
from swarm_arena._budget.ledger import Ledger

root = Path.cwd()
config = RunConfig(run_root=str(root / 'runs'), artifact_root=str(root / 'artifacts'),
                   arena_root=str(root / 'arena'), settle_timeout_s=2)
settle = Ledger._settle
async def held_settle(self: Ledger, *args: Any, **kwargs: Any) -> None:
    first_interrupt = anyio.Event()
    runner_handler = signal.getsignal(signal.SIGINT)
    assert callable(runner_handler)
    def observe(signum: int, frame: Any) -> None:
        try:
            runner_handler(signum, frame)
        finally:
            first_interrupt.set()
    signal.signal(signal.SIGINT, observe)
    try:
        with anyio.CancelScope(shield=True):
            print('settlement-ready', flush=True)
            await first_interrupt.wait()
    finally:
        signal.signal(signal.SIGINT, runner_handler)
        try:
            with anyio.CancelScope(shield=True):
                print('cleanup-ready', flush=True)
                await anyio.sleep_forever()
        finally:
            with anyio.CancelScope(shield=True):
                print('settlement-resumed', flush=True)
                await settle(self, *args, **kwargs)
Ledger._settle = held_settle
handler = signal.getsignal(signal.SIGINT)
try:
    asyncio.run(Swarm(config).run(plan(rounds=1)))
except KeyboardInterrupt:
    print('interrupted', flush=True)
    print('handler-restored=' + str(signal.getsignal(signal.SIGINT) is handler), flush=True)
    raise SystemExit(130)
"""
    process = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", code],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        start_new_session=True,
    )
    stdout = process.stdout
    assert stdout is not None
    output = bytearray()

    def until(marker: bytes) -> None:
        deadline = time.monotonic() + 60
        while marker not in output:
            assert time.monotonic() < deadline, bytes(output)
            if select.select([stdout], [], [], 0.05)[0]:
                chunk = os.read(stdout.fileno(), 65536)
                assert chunk, bytes(output)
                output.extend(chunk)

    try:
        until(b"settlement-ready")
        process.send_signal(signal.SIGINT)
        until(b"cleanup-ready")
        process.send_signal(signal.SIGINT)
        try:
            rest, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail(f"library forced interrupt did not exit: {bytes(output)!r}")
        output.extend(rest)
        assert process.returncode == 130, bytes(output)
        assert b"settlement-resumed" in output
        assert b"interrupted" in output
        assert b"handler-restored=True" in output
        assert b"Traceback" not in output
        assert b"during asyncio.run() shutdown" not in output
        assert b"unhandled exception" not in output
        assert b"never retrieved" not in output
    finally:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0)
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        stdout.close()
