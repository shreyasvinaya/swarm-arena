import json
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import anyio
import pytest
from _fakes.engine import take_over_retrying, with_executor, with_limits
from _fakes.finished import FinishedRun, run_again
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

    # The stalled phase keeps a short renewal interval (ttl / 3) so a successor's fence is seen
    # within the bound; the setup run keeps the default lease so CI load cannot fence it early.
    config = run_config.model_copy(update={"settle_timeout_s": 1.0, "lease_ttl_s": 1.5})
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
            successor = Lease.take_over(
                store, owner_id="successor", host="local", ttl_s=60, force=True
            )
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
