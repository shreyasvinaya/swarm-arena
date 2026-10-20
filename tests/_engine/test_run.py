import json
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import anyio
import pytest
from _fakes.engine import take_over_retrying, with_executor, with_limits
from _fakes.run import WRITE, WRITE_THEN_SLEEP, all_events, events, plan, run_dir, wait_tool_started
from _fakes.tools import PyTool, RecordingExecutor

from swarm_arena import Ref, RunConfig
from swarm_arena._budget.ledger import Ledger
from swarm_arena._engine.run import RunRunner, _require_asyncio
from swarm_arena._engine.swarm import Swarm
from swarm_arena._errors import Fenced, PlanError
from swarm_arena._journal.lease import read_owner
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.ids import IdGen
from swarm_arena._model.tooling import Estimate
from swarm_arena._store.local import LocalStore


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
        pytest.param(
            {"agent_status"},
            marks=pytest.mark.xfail(
                strict=True, reason="agent_status task_id lands with the Task 22 fix"
            ),
        ),
    ],
)
async def test_every_event_has_context_and_task_ids(run_config: RunConfig, types: set[str]) -> None:
    await Swarm(run_config).run(plan())
    recorded = all_events(run_config)
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


async def test_one_admission_shared_by_every_goal(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine.goal import GoalRunner
    from swarm_arena._engine.round import EngineDeps

    admissions: list[object] = []
    original = GoalRunner.__init__

    def initialize(self: GoalRunner, deps: EngineDeps, **kwargs: Any) -> None:
        admissions.append(deps.admission)
        original(self, deps, **kwargs)

    monkeypatch.setattr(GoalRunner, "__init__", initialize)
    await Swarm(run_config).run([plan(goal_id=g, rounds=1) for g in ("a", "b", "c")])
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
            raise ExceptionGroup("lost owner", [Fenced()])

        async def aclose(self) -> None:
            raise ValueError("shutdown failed")

    monkeypatch.setattr("swarm_arena._engine.run._build_executor", lambda cfg: FencedExecutor())
    with pytest.raises(Fenced) as caught:
        await Swarm(run_config).run(plan())
    assert type(caught.value) is Fenced
