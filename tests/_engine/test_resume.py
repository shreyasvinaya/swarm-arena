"""Recover deliberately incomplete, committed run journals."""

import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, TypeVar, cast

import anyio
import pytest
from _fakes.engine import with_executor, with_limits

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    FunctionTool,
    Goal,
    KeepTopTeams,
    PartSpec,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    ScriptedManager,
    Swarm,
    TeamSpec,
)
from swarm_arena._agent.context import AgentContext
from swarm_arena._agent.manager import LimitReached, Manager
from swarm_arena._agent.repeats import repeat_key
from swarm_arena._agent.tool_run import Inflight
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger
from swarm_arena._engine import resume
from swarm_arena._engine.control import request_stop
from swarm_arena._engine.goal import GoalOutcome, GoalRunner, r0_revision_draft
from swarm_arena._engine.resume import resume_run
from swarm_arena._engine.round import RoundOutcome, RoundRunner
from swarm_arena._errors import (
    ConditionalWritesUnsupported,
    Fenced,
    LeaseHeld,
    PlanError,
    SoftError,
)
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._journal.event import Event, EventDraft
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.ids import IdGen
from swarm_arena._model.planning_types import RoundReport
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Candidate, Score
from swarm_arena._model.tooling import RunStatus, ToolOutput, ToolRequest
from swarm_arena._planning.validate import admit_revision
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import Store
from swarm_arena._store.local import LocalStore

_T = TypeVar("_T", bound=Configurable)


class _Recorder(Manager):
    def __init__(self, requests: dict[int, list[float | str]] | None = None) -> None:
        super().__init__()
        self.requests = requests if requests is not None else {1: [1.0], 2: [2.0], 3: [3.0]}
        self.results: list[Candidate] = []
        self.limits_hit: list[str] = []
        self.soft_errors: list[str] = []

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        for x in self.requests.get(ctx.round, []):
            try:
                if x == "search":
                    ctx.search_learnings("something")
                else:
                    self.results.append(await ctx.run_tool({"x": x}))
            except LimitReached as exc:
                self.limits_hit.append(exc.limit)
                break
            except SoftError as exc:
                self.soft_errors.append(str(exc))
        return RoundReport()


@dataclass
class Crashed:
    config: RunConfig
    run_id: str
    store: LocalStore
    old_lease: Lease
    head_seq: int
    frontier: Frontier
    recorders: dict[str, _Recorder]
    round: int = 1
    in_flight: list[str] = field(default_factory=list)
    share: Amount = field(default_factory=Amount)
    r1_amount: Amount = field(default_factory=lambda: Amount(usd=1))
    b_charged: Amount = field(default_factory=lambda: Amount(usd=1))
    live_remaining: dict[str, Amount] = field(default_factory=dict)
    command_started: Job.Started | None = None
    plan_dir: Path = field(default_factory=Path.cwd)
    cx: str = "c1"
    cy: str = "c2"
    cz: str = "c3"

    def config_with(self, executor: Executor, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
        monkeypatch.setattr("swarm_arena._engine.run._build_executor", lambda config: executor)
        return self.config


def _events(c: Crashed, type: str | None = None) -> list[Event]:
    committed = read_committed(c.store)
    assert committed is not None
    return [e for e in committed[1] if type is None or e.type == type]


def _settle_for(c: Crashed, reservation: str) -> Event | None:
    return next((e for e in _events(c, "settle") if e.payload["id"] == reservation), None)


def _goal_status(c: Crashed, goal: str) -> str:
    return RunState.from_events(_events(c)).goals[goal].status


def _final_order(c: Crashed) -> list[str]:
    return _events(c, "final_order")[-1].payload["candidate_ids"]


def _plan(
    goal: str = "g",
    *,
    agents: tuple[str, ...] = ("a", "b"),
    rounds: int = 2,
    converge: bool = False,
) -> Plan:
    def team(name: str, names: tuple[str, ...]) -> TeamSpec:
        return TeamSpec(
            name=name,
            rationale="Explore",
            agents=[
                AgentSpec(
                    name=a,
                    rationale="Explore",
                    tool=Ref.of(
                        FunctionTool("_fakes.tools:square", timeout_s=61 if name == "conv" else 60)
                    ),
                    manager=Ref.of(ScriptedManager([])).model_copy(update={"name": a}),
                )
                for a in names
            ],
        )

    return Plan(
        goal=Goal(id=goal, namespace="n"),
        teams=[team("t", agents)],
        rounds=[RoundSpec() for _ in range(rounds)],
        policy=Ref.of(KeepTopTeams(1)),
        judges=[Ref.of(ProgramJudge("_fakes.tools:score_min_y")).model_copy(update={"name": "y"})],
        ranking="y",
        converge=ConvergeSpec(
            mode="team" if converge else "select",
            team=team("conv", ("c" + goal,)) if converge else None,
        ),
    )


@pytest.fixture
def recorders(monkeypatch: pytest.MonkeyPatch) -> dict[str, _Recorder]:
    mapping: dict[str, _Recorder] = {}
    real = Ref.build

    def build(self: Ref, expected: type[_T]) -> _T:
        if expected is Manager and self.name in mapping:
            return cast(_T, mapping[self.name])
        return real(self, expected)

    monkeypatch.setattr(Ref, "build", build)
    return mapping


async def _crash(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    *,
    plans: list[Plan] | None = None,
    live: bool = False,
    completed: tuple[tuple[str, float, RunStatus], ...] = (("a", 1.0, RunStatus.OK),),
    lost: tuple[str, ...] = ("b",),
    stopped: dict[str, str] | None = None,
    queued: tuple[str, ...] = (),
    converge: bool = False,
    terminal: str | None = None,
    counters: dict[str, dict[str, int]] | None = None,
    command: bool = False,
    charge: Amount | None = None,
    budget_usd: float = 1000,
    slice_usd: float = 10,
) -> Crashed:
    charge = charge if charge is not None else Amount(usd=1)
    config = with_limits(
        with_executor(run_config, workspace_root=str(tmp_path / "ws")),
        max_usd=budget_usd,
        max_tokens=None,
    )
    plans = plans or [_plan()]
    artifacts = ArtifactStore(LocalStore(Path(config.artifact_root)))
    plans = [admit_revision(p, config, artifacts)[0] for p in plans]
    for plan in plans:
        for team in [*plan.teams, *([plan.converge.team] if plan.converge.team else [])]:
            for agent in team.agents:
                recorders.setdefault(agent.name, _Recorder())
    run_id = "resume-run"
    store = LocalStore(Path(config.run_root) / run_id)
    lease = Lease.acquire_new(store, owner_id="old", host="local", ttl_s=60)
    ids = IdGen(run_id, 1)
    frontier = Frontier(marks=[])
    c = Crashed(config, run_id, store, lease, 0, frontier, recorders, r1_amount=charge)
    async with JournalWriter(
        store,
        lease,
        run_id=run_id,
        next_seq=1,
        local_path=tmp_path / "old.jsonl",
        sync_interval_s=60,
        group_window_s=0,
    ) as journal:
        ledger = Ledger(
            config.limits,
            reserve_fraction=config.reserve_fraction,
            goal_weights=config.goal_weights,
            journal=journal,
            ids=ids,
        )
        journal.append(
            EventDraft(type="run_config", payload={"config": config.model_dump(mode="json")})
        )
        slices = {p.goal.id: Amount(usd=slice_usd) for p in plans}
        journal.append(
            EventDraft(
                type="run_started",
                payload={"final_slices": {g: a.model_dump() for g, a in slices.items()}},
            )
        )
        await ledger.hold_final_slices(slices)
        for plan in plans:
            journal.append(r0_revision_draft(plan, plan.base_dir))
            journal.append(EventDraft(type="goal_queued", goal_id=plan.goal.id))
        for plan in plans:
            g = plan.goal.id
            if g in queued:
                continue
            ledger.open_goal(g)
            journal.append(EventDraft(type="goal_started", goal_id=g))
            head = await journal.commit()
            assert head is not None
            frontier = Frontier(
                marks=[
                    Frontier.Mark(
                        run_id=run_id,
                        run_root=config.run_root,
                        epoch=1,
                        seq=head.seq,
                        segment=head.segment,
                    )
                ]
            )
            c.frontier = frontier
            shares = ledger.open_round(
                g,
                1,
                [f"{g}/t/{a.name}" for a in plan.teams[0].agents],
                plan.rounds_total(config.limits),
            )
            c.share = next(iter(shares.values()))
            journal.append(
                EventDraft(
                    type="round_started",
                    goal_id=g,
                    round=1,
                    payload={"round": 1, "shares": {h: a.model_dump() for h, a in shares.items()}},
                )
            )
            journal.append(
                EventDraft(
                    type="frontier",
                    goal_id=g,
                    round=1,
                    payload={"round": 1, "frontier": frontier.model_dump(mode="json")},
                )
            )
            for index, (agent, x, status) in enumerate(completed, 1):
                if agent not in {a.name for a in plan.teams[0].agents}:
                    continue
                reservation = await ledger.reserve(f"{g}/t/{agent}", charge, kind="tool", goal_id=g)
                id = f"c{index}" if len(plans) == 1 else f"{g}-c{index}"
                started = Job.Started(
                    kind="process",
                    job_id=id,
                    host_id="old-host:boot",
                    pid=999999,
                    pgid=999999,
                    start_time="old",
                )
                journal.append(
                    EventDraft(
                        type="tool_start",
                        goal_id=g,
                        team="t",
                        agent=agent,
                        round=1,
                        payload={
                            "candidate_id": id,
                            "reservation_id": reservation.id,
                            "started": started.model_dump(mode="json"),
                        },
                    )
                )
                spec = next(a for a in plan.teams[0].agents if a.name == agent)
                req = ToolRequest(inputs={"x": x}, round=1)
                candidate = Candidate(
                    id=id,
                    run_id=run_id,
                    goal_id=g,
                    part=plan.teams[0].part,
                    team="t",
                    agent=agent,
                    round=1,
                    request=req,
                    fidelity=plan.rounds[0].effective_fidelity(),
                    output=ToolOutput(status=status, data={"y": x * x}),
                )
                journal.append(
                    EventDraft(
                        type="tool_end",
                        goal_id=g,
                        team="t",
                        agent=agent,
                        round=1,
                        payload={
                            "candidate": candidate.model_dump(mode="json"),
                            "fingerprint": str(x),
                            "repeat_key": repeat_key(
                                goal_key=plan.goal.key(),
                                tool_ref=spec.tool,
                                request=req,
                                deterministic=bool(spec.tool.config.get("deterministic", True)),
                            ),
                        },
                    )
                )
                if converge or (stopped and g in stopped) or terminal:
                    score = Score.minimize(x * x).model_copy(
                        update={"id": "s-" + id, "candidate_id": id, "judge": "y"}
                    )
                    journal.append(
                        EventDraft(
                            type="score",
                            goal_id=g,
                            round=1,
                            payload={"score": score.model_dump(mode="json")},
                        )
                    )
            for agent in lost:
                if agent not in {a.name for a in plan.teams[0].agents}:
                    continue
                reservation = await ledger.reserve(f"{g}/t/{agent}", charge, kind="tool", goal_id=g)
                id = f"lost-{g}-{agent}"
                c.in_flight.append(id)
                started = Job.Started(
                    kind="process",
                    job_id=id,
                    host_id="old-host:boot",
                    pid=999998,
                    pgid=999998,
                    start_time="old",
                )
                journal.append(
                    EventDraft(
                        type="tool_start",
                        goal_id=g,
                        team="t",
                        agent=agent,
                        round=1,
                        payload={
                            "candidate_id": id,
                            "reservation_id": reservation.id,
                            "started": started.model_dump(mode="json"),
                        },
                    )
                )
            for agent, values in (counters or {}).items():
                journal.append(
                    EventDraft(
                        type="agent_counters",
                        goal_id=g,
                        team="t",
                        agent=agent,
                        round=1,
                        payload=values,
                    )
                )
            if command:
                c.command_started = Job.Started(
                    kind="process",
                    job_id="command1",
                    host_id="old-host:boot",
                    pid=999997,
                    pgid=999997,
                    start_time="old",
                )
                journal.append(
                    EventDraft(
                        type="command_start",
                        goal_id=g,
                        payload={
                            "job_id": "command1",
                            "started": c.command_started.model_dump(mode="json"),
                        },
                    )
                )
            if converge or (stopped and g in stopped) or terminal:
                journal.append(
                    EventDraft(type="round_closed", goal_id=g, round=1, payload={"round": 1})
                )
                journal.append(
                    EventDraft(type="converge_started", goal_id=g, payload={"select_only": False})
                )
            if (stopped and g in stopped) or terminal:
                if terminal == "no_result":
                    journal.append(
                        EventDraft(
                            type="no_result", goal_id=g, payload={"reason": "no passing candidate"}
                        )
                    )
                else:
                    journal.append(
                        EventDraft(
                            type="final_order",
                            goal_id=g,
                            payload={
                                "candidate_ids": (
                                    ["c1" if len(plans) == 1 else f"{g}-c1"]
                                    if completed and terminal != "unjoined"
                                    else []
                                ),
                                "parts": {"part": "c1"} if terminal == "unjoined" else None,
                            },
                        )
                    )
                if terminal not in {"selection", "no_result", "unjoined"}:
                    journal.append(
                        EventDraft(
                            type="goal_finished",
                            goal_id=g,
                            payload={"reason": (stopped or {}).get(g)},
                        )
                    )
                    await journal.commit()
                    await ledger.end_goal(g)
            c.live_remaining.update({h: ledger.remaining(h) for h in shares})
        head = await journal.commit()
        assert head is not None
        c.head_seq = head.seq
    if live:
        lease.renew()
    return c


@pytest.fixture
async def crashed(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> Crashed:
    return await _crash(tmp_path, run_config, recorders)


@pytest.fixture
async def crashed_live(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> Crashed:
    return await _crash(tmp_path, run_config, recorders, live=True)


async def test_live_lease_needs_force(crashed_live: Crashed) -> None:
    with pytest.raises(LeaseHeld):
        await Swarm(crashed_live.config).resume(crashed_live.run_id)


async def test_seq_continues_and_takeover_logged(crashed: Crashed) -> None:
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    ev = _events(crashed)
    assert [e.seq for e in ev] == list(range(1, len(ev) + 1))
    take = next(e for e in ev if e.type == "takeover")
    assert take.epoch == 2
    assert take.seq == crashed.head_seq + 1
    assert take.payload == {"force": True}


async def test_in_flight_runs_lost_and_charged(crashed: Crashed) -> None:
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    lost = _events(crashed, "run_lost")
    assert [e.payload["candidate_id"] for e in lost] == crashed.in_flight
    assert _settle_for(crashed, lost[0].payload["reservation_id"]) is None
    assert (lost[0].goal_id, lost[0].team, lost[0].agent, lost[0].round) == ("g", "t", "b", 1)


async def test_completed_runs_not_repeated(crashed: Crashed) -> None:
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    assert not [
        e
        for e in _events(crashed, "tool_start")
        if e.epoch == 2 and e.agent == "a" and e.round == 1
    ]
    assert crashed.recorders["a"].results[0].id == "c1"


async def test_unfinished_round_uses_recorded_frontier(
    crashed: Crashed, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Frontier | None] = []
    real = RoundRunner.run

    async def spy(self: RoundRunner, round: int) -> RoundOutcome:
        seen.append(self.frontier)
        return await real(self, round)

    monkeypatch.setattr(RoundRunner, "run", spy)
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    assert seen[0] == crashed.frontier
    assert [e.epoch for e in _events(crashed, "frontier") if e.payload["round"] == 1] == [1]
    assert len([e for e in _events(crashed, "round_started") if e.payload["round"] == 1]) == 1


async def test_takeover_retries_when_old_owner_commits_meanwhile(
    crashed_live: Crashed, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Lease.take_over.__func__
    attempts: list[int] = []

    def racing(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        attempts.append(1)
        if len(attempts) == 1:
            crashed_live.old_lease.renew()
            raise Fenced("another owner changed the lease")
        return real(cls, store, **kw)

    monkeypatch.setattr(Lease, "take_over", classmethod(racing))
    await Swarm(crashed_live.config).resume(crashed_live.run_id, force=True)
    owner = read_owner(crashed_live.store)
    assert len(attempts) == 2
    assert owner is not None
    assert owner[0].epoch == 2


@pytest.mark.parametrize("race_before_call", [False, True])
async def test_concurrent_resume_fails_cleanly(
    crashed_live: Crashed, monkeypatch: pytest.MonkeyPatch, race_before_call: bool
) -> None:
    real = Lease.take_over.__func__

    def other_first(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        other_kw = {**kw, "owner_id": "other"}
        real(cls, store, **cast(dict[str, Any], other_kw))
        if race_before_call:
            return real(cls, store, **kw)
        raise Fenced("another owner changed the lease")

    monkeypatch.setattr(Lease, "take_over", classmethod(other_first))
    with pytest.raises(Fenced, match="concurrent resume"):
        await Swarm(crashed_live.config).resume(crashed_live.run_id, force=True)
    doc = read_owner(crashed_live.store)
    assert doc is not None
    assert (doc[0].owner_id, doc[0].epoch) == ("other", 2)


async def test_completed_run_without_settle_stays_charged_in_full(crashed: Crashed) -> None:
    runner = await resume_run(crashed.config, crashed.run_id, force=True, replanner=None)
    assert runner.resumed is not None
    assert "c1" in runner.resumed.state.candidates
    assert runner.resumed.ledger.remaining("g/t/a") == crashed.share - crashed.r1_amount
    assert not [e for e in _events(crashed, "run_lost") if e.payload["candidate_id"] == "c1"]
    await runner.run()


async def test_resume_never_links_other_agents_same_round_result(crashed: Crashed) -> None:
    for r in crashed.recorders.values():
        r.requests = {1: [1.0]}
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    assert [e.agent for e in _events(crashed, "tool_start") if e.epoch == 2] == ["b"]
    assert crashed.recorders["b"].limits_hit == []
    assert crashed.recorders["b"].soft_errors == []


async def test_round_counters_survive_resume(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    c = await _crash(
        tmp_path,
        run_config,
        recorders,
        completed=(),
        lost=(),
        counters={"a": {"turns": 19, "board_reads": 10}, "b": {"turns": 0, "board_reads": 10}},
    )
    recorders["a"].requests = {1: [4.0, 5.0]}
    recorders["b"].requests = {1: ["search"]}
    await Swarm(c.config).resume(c.run_id, force=True)
    assert len([e for e in _events(c, "tool_start") if e.agent == "a" and e.epoch == 2]) == 1
    assert recorders["a"].limits_hit == ["max_turns_per_round"]
    assert any("max_board_reads_per_round" in m for m in recorders["b"].soft_errors)


async def test_counted_prestart_failure_allowance_survives(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    c = await _crash(
        tmp_path,
        run_config,
        recorders,
        completed=(),
        lost=(),
        counters={"a": {"turns": 2, "board_reads": 0, "tool_runs": 3}},
    )
    recorders["a"].requests = {1: [4.0]}
    await Swarm(c.config).resume(c.run_id, force=True)
    assert recorders["a"].limits_hit == ["max_tool_runs_per_round"]
    assert not [e for e in _events(c, "tool_start") if e.agent == "a" and e.epoch == 2]
    assert not _events(c, "run_lost")


async def test_watchdog_counters_rebuilt(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    p = _plan(agents=("a",), rounds=1)
    tool = Ref.of(FunctionTool("_fakes.resume:fail_square"))
    p = p.model_copy(
        update={
            "teams": [
                p.teams[0].model_copy(
                    update={"agents": [p.teams[0].agents[0].model_copy(update={"tool": tool})]}
                )
            ]
        }
    )
    c = await _crash(
        tmp_path,
        run_config,
        recorders,
        plans=[p],
        lost=(),
        completed=(("a", 1.0, RunStatus.ERROR), ("a", 2.0, RunStatus.ERROR)),
    )
    recorders["a"].requests = {1: [3.0, 4.0]}
    await Swarm(c.config).resume(c.run_id, force=True)
    st = [e for e in _events(c, "agent_status") if e.agent == "a" and e.round == 1]
    assert (st[-1].payload["status"], st[-1].payload["reason"]) == (
        "watchdog_stopped",
        "3 consecutive failures",
    )
    assert len(_events(c, "tool_end")) == 3


async def test_limits_changed_journaled(crashed: Crashed) -> None:
    cfg = with_limits(crashed.config, max_rounds=4)
    await Swarm(cfg).resume(crashed.run_id, force=True)
    assert _events(crashed, "limits_changed")[0].payload["new"]["max_rounds"] == 4


async def test_resume_refuses_max_agents_total_below_a_goal_peak(crashed: Crashed) -> None:
    with pytest.raises(PlanError) as err:
        await Swarm(with_limits(crashed.config, max_agents_total=1)).resume(
            crashed.run_id, force=True
        )
    assert err.value.errors == [
        "goal 'g': max_agents_total: peak 2 agents + 0 active in other goals > 1"
    ]
    assert len(_events(crashed)) == crashed.head_seq
    owner = read_owner(crashed.store)
    assert owner is not None
    assert owner[0].expires_at < utc_now()


@pytest.mark.parametrize("changes", [{"max_usd": 1}, {"max_tokens": 100}])
async def test_budget_decrease_refused_before_takeover(
    crashed: Crashed, changes: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="budget cap may only increase"):
        await resume_run(
            with_limits(crashed.config, **changes), crashed.run_id, force=True, replanner=None
        )
    assert len(_events(crashed)) == crashed.head_seq
    owner = read_owner(crashed.store)
    assert owner is not None
    assert owner[0].epoch == 1


async def test_old_stop_file_ignored(crashed: Crashed) -> None:
    crashed.store.create("control/stop-e1", b"")
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    assert _goal_status(crashed, "g") == "finished"


@pytest.mark.parametrize("terminal", ["selection", "no_result", "unjoined", "finished"])
async def test_terminal_recovery_does_not_rerun_managers(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder], terminal: str
) -> None:
    plan = _plan(rounds=1, converge=terminal == "unjoined")
    if terminal == "unjoined":
        plan = plan.model_copy(
            update={
                "parts": [PartSpec(name="part", goal=Goal(id="part-g", namespace="n"))],
                "teams": [plan.teams[0].model_copy(update={"part": "part"})],
            }
        )
    c = await _crash(tmp_path, run_config, recorders, plans=[plan], lost=(), terminal=terminal)
    results = await Swarm(c.config).resume(c.run_id, force=True)
    assert not [e for e in _events(c, "tool_start") if e.epoch == 2]
    assert len(_events(c, "goal_finished")) == 1
    assert c.store.exists("goals/g/solution.json")
    assert results["g"] == (
        []
        if terminal in {"no_result", "unjoined"}
        else [RunState.from_events(_events(c)).candidates["c1"]]
    )
    if terminal == "unjoined":
        view = json.loads(c.store.read("goals/g/solution.json").data)
        assert view["status"] == "unjoined"
        assert view["parts"][0]["candidate"]["id"] == "c1"
    if terminal == "finished":
        assert not [
            e
            for e in _events(c)
            if e.seq > c.head_seq and e.goal_id == "g" and e.type.startswith("goal_")
        ]


async def test_stopped_goal_with_rounds_left_continues(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    c = await _crash(
        tmp_path, run_config, recorders, plans=[_plan(rounds=3)], lost=(), stopped={"g": "stop"}
    )
    await Swarm(c.config).resume(c.run_id)
    ev = _events(c)
    assert [e.payload["round"] for e in ev if e.type == "round_closed"] == [1, 2, 3]
    assert len(_events(c, "converge_started")) == len(_events(c, "final_order")) == 2
    assert _goal_status(c, "g") == "finished"
    view = json.loads(c.store.read("goals/g/solution.json").data)
    assert view["source_head"]["epoch"] == 2
    assert view["reason"] is None
    gs = _events(c, "goal_started")
    assert [e.epoch for e in gs] == [1, 2]
    assert "pool" in gs[-1].payload


@pytest.mark.parametrize(("raise_cap", "rounds_after"), [(False, []), (True, [2])])
async def test_limit_stopped_goal_continues_only_with_raised_limits(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    raise_cap: bool,
    rounds_after: list[int],
) -> None:
    c = await _crash(tmp_path, run_config, recorders, lost=(), stopped={"g": "run budget"})
    cfg = with_limits(c.config, max_usd=2000) if raise_cap else c.config
    await Swarm(cfg).resume(c.run_id)
    assert [e.payload["round"] for e in _events(c, "round_started") if e.epoch == 2] == rounds_after
    if raise_cap:
        reopen = next(e for e in _events(c, "goal_started") if e.epoch == 2)
        assert reopen.seq < _events(c, "limits_changed")[0].seq


async def test_queued_goal_starts_from_journaled_r0(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    c = await _crash(tmp_path, run_config, recorders, plans=[_plan("a"), _plan("b")], queued=("b",))
    await Swarm(with_limits(c.config, max_concurrent_goals=1)).resume(c.run_id, force=True)
    assert [e.epoch for e in _events(c, "goal_started") if e.goal_id == "b"] == [2]
    assert len([e for e in _events(c, "plan_revision") if e.goal_id == "b"]) == 1
    assert c.store.exists("goals/b/solution.json")


async def test_resume_during_converge(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    c = await _crash(
        tmp_path, run_config, recorders, plans=[_plan(rounds=1)], lost=(), converge=True
    )
    await Swarm(c.config).resume(c.run_id, force=True)
    assert [e.payload["round"] for e in _events(c, "round_closed")] == [1]
    assert [e.payload["select_only"] for e in _events(c, "converge_started")] == [False, False]
    assert _goal_status(c, "g") == "finished"
    assert _final_order(c)


async def test_prepared_runner_can_be_closed_without_running(crashed: Crashed) -> None:
    runner = await resume_run(crashed.config, crashed.run_id, force=True, replanner=None)
    await runner.aclose()
    await runner.aclose()
    owner = read_owner(crashed.store)
    assert owner is not None
    assert owner[0].expires_at < utc_now()
    with pytest.raises(RuntimeError, match="closed"):
        await runner.run()


async def _append(c: Crashed, *drafts: EventDraft) -> None:
    owner = read_owner(c.store)
    assert owner is not None
    assert owner[0].head is not None
    lease = Lease(c.store, *owner, ttl_s=60)
    lease.renew()
    async with JournalWriter(
        c.store,
        lease,
        run_id=c.run_id,
        next_seq=owner[0].head.seq + 1,
        local_path=Path(c.config.run_root) / f"append-e{owner[0].epoch}.jsonl",
        sync_interval_s=60,
        group_window_s=0,
    ) as journal:
        for draft in drafts:
            journal.append(draft)
        head = await journal.commit()
        assert head is not None
        c.head_seq = head.seq


async def test_resumed_round_keeps_agent_shares(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    c = await _crash(tmp_path, run_config, recorders, lost=(), completed=())
    owner = read_owner(c.store)
    assert owner is not None
    lease = Lease(c.store, *owner, ttl_s=60)
    lease.renew()
    async with JournalWriter(
        c.store,
        lease,
        run_id=c.run_id,
        next_seq=c.head_seq + 1,
        local_path=tmp_path / "charges",
        sync_interval_s=60,
        group_window_s=0,
    ) as journal:
        ledger = Ledger.replay(
            RunState.from_events(_events(c)), journal=journal, ids=IdGen(c.run_id, 1)
        )
        await ledger.reserve("g/t/a", c.share, kind="tool", goal_id="g")
        await ledger.reserve("g/t/b", c.b_charged, kind="tool", goal_id="g")
    runner = await resume_run(c.config, c.run_id, force=True, replanner=None)
    assert runner.resumed is not None
    assert runner.resumed.ledger.remaining("g/t/a") == Amount()
    assert runner.resumed.ledger.remaining("g/t/b") == c.share - c.b_charged
    await runner.run()
    assert c.recorders["a"].limits_hit[:1] == ["budget"]
    assert len([e for e in _events(c, "round_started") if e.payload["round"] == 1]) == 1
    assert not _events(c, "run_lost")


async def test_two_continued_goals_split_the_free_pool_once(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    cfg = run_config.model_copy(
        update={"reserve_fraction": 0.2, "goal_weights": {"g1": 3.0, "g2": 1.0}}
    )
    c = await _crash(
        tmp_path,
        cfg,
        recorders,
        plans=[_plan("g1", rounds=3), _plan("g2", rounds=3)],
        completed=(),
        lost=(),
        stopped={"g1": "stop", "g2": "stop"},
        budget_usd=50,
        slice_usd=1,
    )
    runner = await resume_run(c.config, c.run_id, force=False, replanner=None)
    assert runner.resumed is not None
    assert runner.resumed.ledger.free() == Amount()
    await runner.run()
    pools = {
        e.goal_id: Amount.model_validate(e.payload["pool"])
        for e in _events(c, "goal_started")
        if e.epoch == 2
    }
    assert pools == {"g1": Amount(usd=30), "g2": Amount(usd=10)}


class _SpyExecutor(LocalExecutor):
    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace_root=str(workspace))
        self.killed: list[Job.Started] = []
        self.closed_count = 0

    async def kill_orphan(self, started: Job.Started) -> bool:
        killed = await super().kill_orphan(started)
        if killed:
            self.killed.append(started)
        return killed

    async def aclose(self) -> None:
        self.closed_count += 1
        await super().aclose()


@pytest.mark.parametrize(("same_host", "same_start"), [(True, True), (False, True), (True, False)])
async def test_orphan_killed_only_on_host_and_start_time_match(
    crashed: Crashed, monkeypatch: pytest.MonkeyPatch, same_host: bool, same_start: bool
) -> None:
    from swarm_arena._exec import procinfo

    spy = _SpyExecutor(Path(crashed.config.run_root) / "spy")
    monkeypatch.setattr(procinfo, "host_id", lambda: "old-host:boot" if same_host else "other:boot")
    monkeypatch.setattr(procinfo, "process_start_time", lambda pid: "old" if same_start else "new")
    signalled: list[int] = []
    monkeypatch.setattr(procinfo.os, "killpg", lambda pgid, sig: signalled.append(pgid))
    runner = await resume_run(
        crashed.config_with(spy, monkeypatch), crashed.run_id, force=True, replanner=None
    )
    assert signalled == ([999998] if same_host and same_start else [])
    assert runner.resumed is not None
    assert runner.resumed.orphans_killed == (crashed.in_flight if same_host and same_start else [])
    await runner.aclose()
    assert spy.closed_count == 1
    assert len(_events(crashed, "run_lost")) == 1


@pytest.mark.parametrize("same_host", [True, False])
async def test_orphan_command_job_killed_on_resume(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
    same_host: bool,
) -> None:
    from swarm_arena._exec import procinfo

    c = await _crash(tmp_path, run_config, recorders, completed=(), lost=(), command=True)
    spy = _SpyExecutor(tmp_path / "spy")
    monkeypatch.setattr(procinfo, "host_id", lambda: "old-host:boot" if same_host else "other:boot")
    monkeypatch.setattr(procinfo, "process_start_time", lambda pid: "old")
    monkeypatch.setattr(procinfo.os, "killpg", lambda pgid, sig: None)
    runner = await resume_run(c.config_with(spy, monkeypatch), c.run_id, force=True, replanner=None)
    await runner.aclose()
    assert spy.killed == ([c.command_started] if same_host else [])
    assert _events(c, "run_lost") == []
    payload = dict(_events(c, "command_end")[-1].payload)
    assert payload.pop("run_elapsed_s") >= 0
    assert payload == {"job_id": "command1", "status": "lost"}
    assert RunState.from_events(_events(c)).started_commands == {}
    runner = await resume_run(c.config, c.run_id, force=True, replanner=None)
    await runner.aclose()
    assert len(_events(c, "command_end")) == 1


async def test_recovered_candidates_are_earlier_round_entries_next_round(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    p = _plan(agents=("a", "b", "n"))
    n = (
        p.teams[0]
        .agents[2]
        .model_copy(
            update={
                "replicates": 2,
                "tool": Ref.of(FunctionTool("_fakes.tools:square", deterministic=False)),
            }
        )
    )
    p = p.model_copy(
        update={"teams": [p.teams[0].model_copy(update={"agents": [*p.teams[0].agents[:2], n]})]}
    )
    c = await _crash(
        tmp_path,
        run_config,
        recorders,
        plans=[p],
        lost=(),
        completed=(("a", 1.0, RunStatus.OK), ("a", 2.0, RunStatus.OK), ("n", 3.0, RunStatus.OK)),
    )
    recorders["a"].requests = {1: [1.0, 2.0], 2: [1.0]}
    recorders["b"].requests = {1: [1.0], 2: [2.0]}
    recorders["n"].requests = {2: [3.0]}
    await Swarm(c.config).resume(c.run_id, force=True)
    assert [(e.round, e.agent) for e in _events(c, "tool_start") if e.epoch == 2] == [
        (1, "b"),
        (2, "n"),
    ]
    assert [r.id for r in recorders["a"].results] == [c.cx, c.cy]
    assert [(r.agent, r.round) for r in recorders["b"].results] == [("b", 1), ("a", 1)]
    assert recorders["b"].results[1].id == c.cy
    assert recorders["b"].soft_errors == []
    assert [m.split(":")[0] for m in recorders["a"].soft_errors] == ["repeat refused"]
    assert [e.payload["refused"] for e in _events(c, "watchdog") if "refused" in e.payload] == [
        c.cx
    ]
    assert [r.request.replicate for r in recorders["n"].results] == [1]


async def test_continued_goal_waits_for_a_goal_slot(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    c = await _crash(
        tmp_path,
        run_config,
        recorders,
        plans=[_plan("a", agents=("a",), converge=True), _plan("b", agents=("b",), converge=True)],
        lost=(),
        stopped={"a": "goal budget"},
    )
    cfg = with_limits(c.config, max_usd=2000, max_concurrent_goals=1)
    await Swarm(cfg).resume(c.run_id, force=True)
    ev = _events(c)
    b_done = next(e.seq for e in ev if e.type == "goal_finished" and e.goal_id == "b")
    a_rounds = [
        e.seq for e in ev if e.type == "round_started" and e.goal_id == "a" and e.epoch == 2
    ]
    assert a_rounds
    assert b_done < min(a_rounds)
    assert [e.epoch for e in _events(c, "goal_started") if e.goal_id == "a"] == [1, 2]
    assert (_goal_status(c, "a"), _goal_status(c, "b")) == ("finished", "finished")


class _HoldExecutor(LocalExecutor):
    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace_root=str(workspace))
        self.entered, self.release = anyio.Event(), anyio.Event()
        self.held = False

    async def submit(self, job: Job) -> ToolOutput:
        output = await super().submit(job)
        if not self.held:
            self.held = True
            self.entered.set()
            await self.release.wait()
        return output


@dataclass
class _Gauge:
    now: int = 0
    peak: int = 0
    entered: list[str] = field(default_factory=list)

    @contextmanager
    def held(self, key: str) -> Generator[None]:
        self.entered.append(key)
        self.now += 1
        self.peak = max(self.peak, self.now)
        try:
            yield
        finally:
            self.now -= 1


@pytest.mark.parametrize(("max_goals", "max_agents"), [(1, 1), (2, 1)])
async def test_waiting_continued_goal_converges_on_early_end(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
    max_goals: int,
    max_agents: int,
) -> None:
    c = await _crash(
        tmp_path,
        run_config,
        recorders,
        plans=[_plan("a", agents=("a",), converge=True), _plan("b", agents=("b",), converge=True)],
        lost=(),
        stopped={"a": "goal budget"},
    )
    for recorder in recorders.values():
        recorder.requests = {1: [1.0], 2: [2.0]}
    hold = _HoldExecutor(tmp_path / "held")
    cfg = with_limits(
        c.config_with(hold, monkeypatch),
        max_usd=2000,
        max_concurrent_goals=max_goals,
        max_agents_total=max_agents,
    )
    goals, agents = _Gauge(), _Gauge()
    real_goal, real_round, real_close = GoalRunner.run, _Recorder.run_round, Inflight.close

    async def counted_goal(self: GoalRunner) -> GoalOutcome:
        with goals.held(self.goal_id):
            return await real_goal(self)

    async def counted_round(self: _Recorder, ctx: AgentContext) -> RoundReport:
        with agents.held(ctx.spec.name):
            return await real_round(self, ctx)

    closed = anyio.Event()

    def signalling_close(self: Inflight) -> None:
        real_close(self)
        closed.set()

    monkeypatch.setattr(GoalRunner, "run", counted_goal)
    monkeypatch.setattr(_Recorder, "run_round", counted_round)
    monkeypatch.setattr(Inflight, "close", signalling_close)
    with anyio.fail_after(20):
        async with anyio.create_task_group() as tg:
            tg.start_soon(partial(Swarm(cfg).resume, c.run_id, force=True))
            await hold.entered.wait()
            assert await request_stop(cfg.run_root, c.run_id, now=False) == 2
            await closed.wait()
            hold.release.set()
    assert agents.peak == 1
    if max_goals == 1:
        assert goals.peak == 1
    assert goals.entered == ["b", "a"]
    assert agents.entered == ["b", "cb", "ca"]
    a2 = [e for e in _events(c) if e.goal_id == "a" and e.epoch == 2]
    assert [
        e.type
        for e in a2
        if e.type in {"round_closed", "converge_started", "final_order", "goal_finished"}
    ] == ["converge_started", "final_order", "goal_finished"]
    assert next(e for e in a2 if e.type == "final_order").payload["candidate_ids"]
    assert next(e for e in a2 if e.type == "goal_finished").payload["reason"] == "stop"
    assert not [e for e in _events(c, "no_result") if e.goal_id == "a"]


async def test_nested_frozen_revision_admitted_and_replayed_after_source_deletion(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    from swarm_arena import ScriptJudge
    from swarm_arena.scoring import Lexicographic

    source = tmp_path / "judge.py"
    source.write_text("print('frozen rubric')\n")
    judge = Ref.of(
        Lexicographic([Ref.of(ScriptJudge(command=["python", "judge.py"], files=["judge.py"]))])
    ).model_copy(update={"name": "y"})
    plan = _plan().model_copy(update={"judges": [judge]})
    plan = Plan._from_data(plan.model_dump(mode="json"), base_dir=tmp_path)
    c = await _crash(tmp_path, run_config, recorders, plans=[plan], completed=(), lost=())
    artifacts = ArtifactStore(LocalStore(Path(c.config.artifact_root)))
    initial = RunState.from_events(_events(c)).plans["g"][0][0]
    revision, _ = admit_revision(
        initial.model_copy(update={"revision": 1}), c.config, artifacts, history=[initial]
    )
    nested = revision.judges[0].config["judges"][0]
    assert "name" not in nested
    frozen = nested["config"]["files"]["manifest"]
    assert frozen == {"judge.py": artifacts.put_bytes(source.read_bytes())}
    await _append(c, r0_revision_draft(revision, tmp_path))
    source.unlink()
    replayed = RunState.from_events(_events(c)).plans["g"][-1][0]
    readmitted, _ = admit_revision(replayed, c.config, artifacts, history=[initial])
    assert readmitted.judges[0].config["judges"][0]["config"]["files"]["manifest"] == frozen
    assert "name" not in readmitted.judges[0].config["judges"][0]
    runner = await resume_run(c.config, c.run_id, force=True, replanner=None)
    assert runner.resumed is not None
    assert runner.resumed.state.plans["g"][-1][0].judges == revision.judges
    await runner.aclose()


async def test_base_dir_restored(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    directory = tmp_path / "plans"
    directory.mkdir()
    (directory / "resume_local.py").write_text(
        "def square(x: float) -> dict[str, float]:\n    return {'y': x*x}\n"
    )
    plan = _plan(agents=("a",), rounds=1)
    team = plan.teams[0]
    plan = plan.model_copy(
        update={
            "teams": [
                team.model_copy(
                    update={
                        "agents": [
                            team.agents[0].model_copy(
                                update={"tool": Ref.of(FunctionTool("resume_local:square"))}
                            )
                        ]
                    }
                )
            ]
        }
    )
    plan = Plan._from_data(plan.model_dump(mode="json"), base_dir=directory)
    monkeypatch.syspath_prepend(str(directory))
    c = await _crash(tmp_path, run_config, recorders, plans=[plan], completed=(), lost=())
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(directory)])
    monkeypatch.delitem(sys.modules, "resume_local", raising=False)
    await Swarm(c.config).resume(c.run_id, force=True)
    assert sys.path.count(str(directory)) == 1
    assert _goal_status(c, "g") == "finished"


async def test_crash_in_continued_round_resumes_as_running(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = await _crash(
        tmp_path, run_config, recorders, plans=[_plan(rounds=3)], lost=(), stopped={"g": "stop"}
    )
    real = RoundRunner.run
    remaining: dict[str, Amount] = {}

    class Crash(RuntimeError):
        pass

    async def crash(self: RoundRunner, round: int) -> RoundOutcome:
        out = await real(self, round)
        if round == 2:
            await self.deps.journal.commit()
            remaining.update(
                {s.address: self.deps.ledger.remaining(s.address) for s in self.agents}
            )
            raise Crash()
        return out

    monkeypatch.setattr(RoundRunner, "run", crash)
    with pytest.raises(BaseExceptionGroup) as err:
        await Swarm(c.config).resume(c.run_id)
    assert err.value.subgroup(Crash) is not None
    monkeypatch.setattr(RoundRunner, "run", real)
    for recorder in recorders.values():
        recorder.requests = {2: [2.0]}
        recorder.results.clear()
    runner = await resume_run(c.config, c.run_id, force=True, replanner=None)
    assert runner.resumed is not None
    assert {h: runner.resumed.ledger.remaining(h) for h in remaining} == remaining
    await runner.run()
    assert len([e for e in _events(c, "round_started") if e.payload["round"] == 2]) == 1
    assert [e.epoch for e in _events(c, "frontier") if e.payload["round"] == 2] == [2]
    assert not [e for e in _events(c, "tool_start") if e.epoch == 3 and e.agent == "a"]
    assert [e.epoch for e in _events(c, "goal_started")] == [1, 2]
    assert recorders["a"].limits_hit == []
    assert _goal_status(c, "g") == "finished"


@pytest.mark.parametrize("late_score", [False, True])
async def test_committed_converge_round_restores_frontier_and_own_results(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
    late_score: bool,
) -> None:
    c = await _crash(
        tmp_path, run_config, recorders, plans=[_plan(rounds=1, converge=True)], lost=()
    )
    recorders["cg"].requests = {2: [0.5]}
    real = RoundRunner.run

    class Crash(RuntimeError):
        pass

    async def crash(self: RoundRunner, round: int) -> RoundOutcome:
        out = await real(self, round)
        if self.converge:
            await self.deps.journal.commit()
            raise Crash()
        return out

    monkeypatch.setattr(RoundRunner, "run", crash)
    with pytest.raises(BaseExceptionGroup) as err:
        await Swarm(c.config).resume(c.run_id, force=True)
    assert err.value.subgroup(Crash) is not None
    monkeypatch.setattr(RoundRunner, "run", real)
    converged = [e.payload["candidate"]["id"] for e in _events(c, "tool_end") if e.team == "conv"]
    assert converged
    if late_score:
        score = dict(_events(c, "score")[0].payload["score"])
        score.update(id="late-score", key=[-1000.0], value=1000.0)
        await _append(c, EventDraft(type="score", goal_id="g", round=0, payload={"score": score}))
    results = await Swarm(c.config).resume(c.run_id, force=True)
    assert [e.epoch for e in _events(c, "frontier") if e.payload["round"] == 2] == [2]
    assert not [e for e in _events(c, "tool_start") if e.team == "conv" and e.epoch == 3]
    assert any(candidate.id in converged for candidate in results["g"])


async def test_action_counter_producer_retains_counted_prestart_error(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swarm_arena._agent import tool_run
    from swarm_arena._agent.runtime import AgentRuntime

    c = await _crash(tmp_path, run_config, recorders, completed=(), lost=())

    async def fail_before_start(rt: AgentRuntime, *args: Any, **kwargs: Any) -> Candidate:
        await rt.journal.commit()
        raise RuntimeError("before start")

    monkeypatch.setattr(tool_run, "_run", fail_before_start)
    await Swarm(c.config).resume(c.run_id, force=True)
    counted = [
        e
        for e in _events(c, "agent_counters")
        if e.agent == "a" and e.round == 1 and "tool_runs" in e.payload
    ]
    assert counted[-1].payload["tool_runs"] == 1
    assert counted[-1].task_id
    assert not _events(c, "tool_start")


@pytest.mark.parametrize("failure_site", ["build", "orphan"])
async def test_preparation_failure_closes_and_sanitizes(
    crashed: Crashed, monkeypatch: pytest.MonkeyPatch, failure_site: str
) -> None:
    import traceback

    sentinel = "private-resume-value-732"
    spy = _SpyExecutor(Path(crashed.config.run_root) / "spy")

    def build(config: RunConfig) -> Executor:
        if failure_site == "build":
            raise ValueError(sentinel)
        return spy

    async def orphan(started: Job.Started) -> bool:
        raise ValueError(sentinel)

    monkeypatch.setattr("swarm_arena._engine.run._build_executor", build)
    monkeypatch.setattr(spy, "kill_orphan", orphan)
    with pytest.raises(RuntimeError) as err:
        await resume_run(crashed.config, crashed.run_id, force=True, replanner=None)
    assert sentinel not in "".join(traceback.format_exception(err.value))
    assert sentinel not in repr(err.value.args)
    assert spy.closed_count == (0 if failure_site == "build" else 1)
    assert len(_events(crashed, "takeover")) == 1
    owner = read_owner(crashed.store)
    assert owner is not None
    assert owner[0].expires_at < utc_now()


async def test_post_takeover_limit_recheck_is_auditable(
    crashed: Crashed, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._journal.lease import Head
    from swarm_arena._journal.segment import encode_segment, segment_name

    real = Lease.take_over.__func__

    def race(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        old = crashed.old_lease
        old.renew()
        head = old.doc.head
        assert head is not None
        plan = _plan(agents=("a", "b", "n"))
        plan = plan.model_copy(update={"revision": 1})
        draft = r0_revision_draft(plan, plan.base_dir)
        event = Event(
            **draft.model_dump(),
            epoch=1,
            seq=head.seq + 1,
            run_id=crashed.run_id,
            context_id=crashed.run_id,
            ts=utc_now(),
        )
        name = segment_name(1, event.seq)
        store.create(name, encode_segment(head.segment, [event]))
        old.advance(Head(epoch=1, seq=event.seq, segment=name))
        return real(cls, store, **kw)

    monkeypatch.setattr(Lease, "take_over", classmethod(race))
    with pytest.raises(PlanError, match="peak 3"):
        await resume_run(
            with_limits(crashed.config, max_agents_total=2),
            crashed.run_id,
            force=True,
            replanner=None,
        )
    assert [e.type for e in _events(crashed) if e.epoch == 2] == ["takeover"]
    owner = read_owner(crashed.store)
    assert owner is not None
    assert owner[0].epoch == 2
    assert owner[0].expires_at < utc_now()


async def test_id_counters_and_active_elapsed_restored(crashed: Crashed) -> None:
    await _append(
        crashed,
        EventDraft(
            type="agent_counters",
            goal_id="g",
            team="t",
            agent="a",
            round=1,
            payload={"turns": 1, "board_reads": 0, "tool_runs": 1, "run_elapsed_s": 12.0},
        ),
    )
    before = RunState.from_events(_events(crashed))
    runner = await resume_run(crashed.config, crashed.run_id, force=True, replanner=None)
    assert runner.resumed is not None
    deps = runner.resumed._services.deps
    assert deps is not None
    assert 12 <= deps.clock.elapsed() < 15
    assert int(deps.ids.next("r").rsplit("-", 1)[-1]) == before.id_counters["r"] + 1
    await runner.aclose()


async def test_finished_goal_only_repairs_view(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    c = await _crash(tmp_path, run_config, recorders, lost=(), terminal="finished")
    head = c.head_seq
    result = await Swarm(c.config).resume(c.run_id, force=True)
    assert c.store.exists("goals/g/solution.json")
    assert [candidate.id for candidate in result["g"]] == ["c1"]
    assert not [
        e for e in _events(c) if e.seq > head and e.goal_id == "g" and e.type.startswith("goal_")
    ]


async def test_never_started_no_result_remains_terminal(
    tmp_path: Path, run_config: RunConfig, recorders: dict[str, _Recorder]
) -> None:
    c = await _crash(tmp_path, run_config, recorders, queued=("g",), completed=(), lost=())
    await _append(
        c,
        EventDraft(type="no_result", goal_id="g", payload={"reason": "stopped"}),
        EventDraft(type="goal_finished", goal_id="g", payload={"reason": "stopped"}),
    )
    result = await Swarm(with_limits(c.config, max_usd=2000)).resume(c.run_id, force=True)
    assert result == {"g": []}
    assert not _events(c, "goal_started")
    assert not _events(c, "tool_start")


async def test_prepared_services_survive_preparing_task_exit(crashed: Crashed) -> None:
    from swarm_arena._engine.run import RunRunner

    runners: list[RunRunner] = []

    async def prepare() -> None:
        runners.append(await resume_run(crashed.config, crashed.run_id, force=True, replanner=None))

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(prepare)
    assert await runners[0].run()
    owner = read_owner(crashed.store)
    assert owner is not None
    assert owner[0].expires_at < utc_now()


async def test_counter_snapshots_keep_admissions_and_deduplicate_lost(crashed: Crashed) -> None:
    await _append(
        crashed,
        EventDraft(
            type="agent_counters",
            goal_id="g",
            team="t",
            agent="b",
            round=1,
            payload={"turns": 1, "board_reads": 0, "tool_runs": 3},
        ),
        EventDraft(
            type="agent_counters",
            goal_id="g",
            team="t",
            agent="b",
            round=1,
            payload={"turns": 2, "board_reads": 1},
        ),
    )
    state = RunState.from_events(_events(crashed))
    assert state.goals["g"].counters["t/b"] == {"turns": 2, "board_reads": 1, "tool_runs": 3}
    crashed.recorders["b"].requests = {1: [9.0]}
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    assert crashed.recorders["b"].limits_hit == ["max_tool_runs_per_round"]
    assert not [e for e in _events(crashed, "tool_start") if e.epoch == 2 and e.agent == "b"]


async def test_learning_allowance_restored(crashed: Crashed) -> None:
    from swarm_arena._model.results import Learning

    drafts = []
    for index in range(crashed.config.limits.max_learnings_per_agent_round):
        learning = Learning(
            id=f"L-{index}",
            run_id=crashed.run_id,
            goal_id="g",
            namespace="n",
            author="g/t/a",
            evidence=["c1"],
            round=1,
            title="A learning",
            text="Some learning",
        )
        drafts.append(
            EventDraft(
                type="learning",
                goal_id="g",
                team="t",
                agent="a",
                round=1,
                payload={"learning": learning.model_dump(mode="json")},
            )
        )
    await _append(crashed, *drafts)

    class Posting(_Recorder):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            if ctx.round == 1:
                try:
                    await ctx.post_learning("Another", "Another learning", ["c1"])
                except SoftError as error:
                    self.soft_errors.append(str(error))
            return RoundReport()

    recorder = Posting()
    crashed.recorders["a"] = recorder
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    assert recorder.soft_errors == ["max_learnings_per_agent_round reached"]


@pytest.mark.parametrize("close_mode", ["raises", "hangs"])
@pytest.mark.parametrize("fenced", [False, True])
async def test_resume_shutdown_bound_and_primary_failure(
    crashed: Crashed, monkeypatch: pytest.MonkeyPatch, close_mode: str, fenced: bool
) -> None:
    import asyncio

    from swarm_arena._engine.run import RunRunner

    release = anyio.Event()
    closing = anyio.Event()
    spy = _SpyExecutor(Path(crashed.config.run_root) / "spy")

    async def close() -> None:
        closing.set()
        if close_mode == "raises":
            raise ValueError("private-shutdown-value")
        await release.wait()

    monkeypatch.setattr(spy, "aclose", close)
    config = crashed.config_with(spy, monkeypatch).model_copy(update={"settle_timeout_s": 0.05})
    runner = await resume_run(config, crashed.run_id, force=True, replanner=None)
    assert runner.resumed is not None
    expected = {"g": [runner.resumed.state.candidates["c1"]]}

    async def drive(self: RunRunner, deps: object, epoch: int) -> dict[str, list[Candidate]]:
        if fenced:
            raise Fenced()
        return expected

    monkeypatch.setattr(RunRunner, "_drive", drive)
    task = asyncio.create_task(runner.run())
    try:
        await closing.wait()
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done, "executor shutdown exceeded its configured bound"
        if fenced:
            with pytest.raises(Fenced) as error:
                await task
            assert error.value.args == ()
        elif close_mode == "raises":
            with pytest.raises(RuntimeError) as error:
                await task
            assert "private-shutdown-value" not in str(error.value)
        else:
            assert await task == expected
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


class _CrashBoundary(BaseException):
    """Interrupt orchestration after an explicitly committed checkpoint."""


@pytest.mark.parametrize("reason", ["stop", "stop --now", "run_timeout_s", None])
async def test_pending_selection_preserves_lifecycle_reason(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
    reason: str | None,
) -> None:
    c = await _crash(tmp_path, run_config, recorders, plans=[_plan(rounds=3)], lost=())
    real_stop = GoalRunner._stop
    real_durable = JournalWriter.append_durable

    def stop_after_first_round(self: GoalRunner) -> bool:
        if self.rounds_closed >= 1:
            if reason == "stop":
                self.deps.stops.stop = True
            elif reason == "stop --now":
                self.deps.stops.stop_now = True
            elif reason == "run_timeout_s":
                self.deps.stops.run_timeout = True
        return real_stop(self)

    async def crash_before_finish(self: JournalWriter, draft: EventDraft) -> Event:
        if draft.type == "goal_finished":
            await self.commit()
            raise _CrashBoundary()
        return await real_durable(self, draft)

    with monkeypatch.context() as patch:
        patch.setattr(GoalRunner, "_stop", stop_after_first_round)
        patch.setattr(JournalWriter, "append_durable", crash_before_finish)
        with pytest.raises(BaseExceptionGroup) as error:
            await Swarm(c.config).resume(c.run_id)
        assert error.value.subgroup(_CrashBoundary) is not None
    selected = RunState.from_events(_events(c)).goals["g"]
    assert selected.selection_recorded
    assert not selected.finished
    await Swarm(c.config).resume(c.run_id)
    ended = RunState.from_events(_events(c)).goals["g"]
    assert ended.end_reason == reason
    assert ended.status == ("stopped" if reason else "finished")
    assert _events(c, "converge_started")[-1].payload["reason"] == reason
    assert _events(c, "final_order")[-1].payload["reason"] == reason
    before = len(_events(c, "round_started"))
    await Swarm(c.config).resume(c.run_id)
    later = _events(c, "round_started")[before:]
    assert [e.round for e in later] == ([2, 3] if reason in {"stop", "stop --now"} else [])
    if reason == "run_timeout_s":
        cfg = with_limits(c.config, run_timeout_s=c.config.limits.run_timeout_s + 1)
        await Swarm(cfg).resume(c.run_id)
        assert [e.round for e in _events(c, "round_started")[before:]] == [2, 3]
        assert len(_events(c, "limits_changed")) == 1


@pytest.mark.parametrize("boundary", ["policy", "closed", "revision"])
async def test_round_revision_crash_boundaries_match_uninterrupted_run(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    from _fakes.planners import AddAgentPlanner

    from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog
    from swarm_arena._planning.provenance import ref_sites

    plan = _plan(agents=("a",), rounds=2)
    entries = [
        CatalogEntry(
            name=f"entry-{i}",
            kind=s.kind,
            ref=s.ref.path,
            config=s.ref.config,
            description="Approved",
        )
        for i, s in enumerate(ref_sites(plan))
    ]
    tool = plan.teams[0].agents[0].tool
    entries.append(
        CatalogEntry(
            name="catalog-agent",
            kind="tool",
            ref=tool.path,
            config=tool.config,
            description="Approved",
        )
    )

    async def exercise(crash: bool) -> tuple[tuple[Any, ...], tuple[Event, Event]]:
        root = tmp_path / ("crash" if crash else "control")
        monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        config = run_config.model_copy(
            update={
                "run_root": str(root / "runs"),
                "arena_root": str(root / "arena"),
                "planner_catalog": PlannerCatalog(entries=entries),
            }
        )
        c = await _crash(root, config, recorders, plans=[plan], lost=())
        managers: list[tuple[str, int]] = []
        barriers: list[int] = []
        replans: list[int] = []
        manage = _Recorder.run_round
        barrier = GoalRunner._barrier
        replan = GoalRunner._replan
        durable = JournalWriter.append_durable
        armed = crash

        async def manager_spy(self: _Recorder, ctx: AgentContext) -> RoundReport:
            managers.append((ctx.spec.name, ctx.round))
            return await manage(self, ctx)

        async def barrier_spy(self: GoalRunner, candidates: Any, round: int) -> bool:
            barriers.append(round)
            return await barrier(self, candidates, round)

        async def replan_spy(self: GoalRunner) -> bool:
            nonlocal armed
            replans.append(self.rounds_closed)
            result = await replan(self)
            if armed and boundary == "revision" and result:
                armed = False
                raise _CrashBoundary()
            return result

        async def durable_spy(self: JournalWriter, draft: EventDraft) -> Event:
            nonlocal armed
            if armed and draft.type == "round_closed" and draft.round == 1:
                if boundary == "policy":
                    armed = False
                    await self.commit()
                    raise _CrashBoundary()
                if boundary == "closed":
                    armed = False
                    await durable(self, draft)
                    raise _CrashBoundary()
            return await durable(self, draft)

        with monkeypatch.context() as patch:
            patch.setattr(_Recorder, "run_round", manager_spy)
            patch.setattr(GoalRunner, "_barrier", barrier_spy)
            patch.setattr(GoalRunner, "_replan", replan_spy)
            patch.setattr(JournalWriter, "append_durable", durable_spy)
            if crash:
                with pytest.raises(BaseExceptionGroup) as error:
                    await Swarm(c.config, replanner=AddAgentPlanner("catalog-agent")).resume(
                        c.run_id
                    )
                assert error.value.subgroup(_CrashBoundary) is not None
            await Swarm(c.config, replanner=AddAgentPlanner("catalog-agent")).resume(c.run_id)
        events = _events(c)
        history = (
            sorted(managers, key=lambda item: (item[1], item[0])),
            barriers,
            replans,
            [
                (e.type, e.round)
                for e in events
                if e.type in {"round_started", "decision", "round_closed"}
            ],
            [e.payload["shares"] for e in events if e.type == "round_started"],
            sorted(
                (e.agent, e.round, e.payload["status"]) for e in events if e.type == "agent_status"
            ),
        )
        revision = _events(c, "plan_revision")[-1]
        closed = _events(c, "round_closed")[0]
        return history, (closed, revision)

    control, control_publication = await exercise(False)
    recovered, recovered_publication = await exercise(True)
    histories = [control, recovered]
    publications = [control_publication, recovered_publication]
    assert histories[1] == histories[0]
    for closed, revision in publications:
        assert closed.seq < revision.seq
        assert revision.payload["first_round"] == 2
    assert histories[1][0] == [("a", 1), ("a", 2), ("new-1", 2)]
    assert histories[1][1] == [1, 2]
    assert histories[1][2] == [1]


async def test_committed_replan_rejection_is_not_repeated_on_resume(
    tmp_path: Path,
    run_config: RunConfig,
    recorders: dict[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _fakes.planners import RaisingPlanner

    c = await _crash(tmp_path, run_config, recorders, plans=[_plan(agents=("a",))], lost=())
    replan = GoalRunner._replan
    attempts: list[int] = []
    armed = True

    async def crash_after_rejection(self: GoalRunner) -> bool:
        nonlocal armed
        attempts.append(self.rounds_closed)
        result = await replan(self)
        assert not result
        await self.deps.journal.commit()
        if armed:
            armed = False
            raise _CrashBoundary()
        return result

    monkeypatch.setattr(GoalRunner, "_replan", crash_after_rejection)
    with pytest.raises(BaseExceptionGroup) as error:
        await Swarm(c.config, replanner=RaisingPlanner()).resume(c.run_id)
    assert error.value.subgroup(_CrashBoundary) is not None
    await Swarm(c.config, replanner=RaisingPlanner()).resume(c.run_id)
    assert attempts == [1]
    assert [e.round for e in _events(c, "round_started")] == [1, 2]
    rejected = _events(c, "plan_rejected")
    assert len(rejected) == 1
    assert rejected[0].payload["first_round"] == 2


async def test_cancelling_preparation_abandons_hung_constructor_and_releases_lease(
    crashed: Crashed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import threading

    from swarm_arena._engine import run as run_module

    entered, release = threading.Event(), threading.Event()
    deadlines: list[anyio.CancelScope] = []
    real = run_module._build_executor
    config = crashed.config.model_copy(update={"settle_timeout_s": 0.05, "lease_ttl_s": 0.3})

    def blocked(config: RunConfig) -> Executor:
        entered.set()
        release.wait()
        return real(config)

    async def prepare() -> None:
        with anyio.fail_after(10) as deadline:
            deadlines.append(deadline)
            await Swarm(config).resume(crashed.run_id)

    with monkeypatch.context() as patch:
        patch.setattr(run_module, "_build_executor", blocked)
        task = asyncio.create_task(prepare())
        try:
            with anyio.fail_after(5):
                while not entered.is_set():  # noqa: ASYNC110 - daemon entry signal
                    await anyio.sleep(0.01)
            deadlines[0].deadline = anyio.current_time() + 0.01
            done, _ = await asyncio.wait([task], timeout=0.5)
            completed_without_release = bool(done)
        finally:
            release.set()
            result = await asyncio.gather(task, return_exceptions=True)
    assert completed_without_release, "caller cancellation waited for the constructor"
    assert isinstance(result[0], TimeoutError)
    owner = read_owner(crashed.store)
    assert owner is not None
    assert owner[0].expires_at <= utc_now()
    await anyio.sleep(0.35)
    assert read_owner(crashed.store) == owner
    await Swarm(crashed.config).resume(crashed.run_id)
    next_owner = read_owner(crashed.store)
    assert next_owner is not None
    assert next_owner[0].epoch == owner[0].epoch + 1


async def _run_published_crash_case(root: Path, mode: str, crash: bool) -> None:
    """Produce real journals in a subprocess, optionally dying after head publication."""
    import os
    import signal

    from swarm_arena._model.planning_types import Decision, RoundState
    from swarm_arena._model.tooling import CompletedRun, Usage
    from swarm_arena._planning.policy import Policy

    config = with_executor(
        RunConfig(
            run_root=str(root / "runs"),
            arena_root=str(root / "arena"),
            artifact_root=str(root / "artifacts"),
        ),
        workspace_root=str(root / "ws"),
    )
    config = with_limits(config, max_usd=1000, max_tokens=None)
    swarm = Swarm(config)
    store = LocalStore(Path(config.run_root) / "published")
    recorder = _Recorder()
    build, collect = Ref.build, FunctionTool.collect
    replace, append = LocalStore.replace, JournalWriter.append
    durable = JournalWriter.append_durable
    terminal_policy = mode in {"policy", "policy_before_close"}

    class TerminalPolicy(Policy):
        def after_round(self, state: RoundState) -> Decision:
            return Decision(no_result="policy terminal") if state.round == 1 else Decision()

    class StopRecorder(_Recorder):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            result = await super().run_round(ctx)
            if ctx.round == 1:
                await swarm.stop(now=mode == "stop --now")
                while True:
                    committed = read_committed(store)
                    if committed and RunState.from_events(committed[1]).stop != "none":
                        break
                    await anyio.sleep(0.01)
            return result

    if not terminal_policy:
        recorder = StopRecorder()

    def build_hook(self: Ref, expected: type[_T]) -> _T:
        if expected is Manager:
            return cast(_T, recorder)
        if expected is Policy and terminal_policy:
            return cast(_T, TerminalPolicy())
        return build(self, expected)

    def paid_collect(
        self: FunctionTool,
        request: ToolRequest,
        workspace: Path,
        run: CompletedRun,
    ) -> ToolOutput:
        output = collect(self, request, workspace, run)
        return output.model_copy(update={"usage": Usage(input_tokens=3, usd=0.000003)})

    def append_hook(self: JournalWriter, draft: EventDraft) -> Event:
        if terminal_policy and draft.type == "round_closed":
            # The committed Decision is sufficient even without an optional reason checkpoint.
            draft = draft.model_copy(
                update={"payload": {k: v for k, v in draft.payload.items() if k != "reason"}}
            )
        return append(self, draft)

    async def durable_hook(self: JournalWriter, draft: EventDraft) -> Event:
        if mode == "policy_before_close" and draft.type == "round_closed":
            # Publish the real policy checkpoint before the closure can be appended.
            await self.commit()
        return await durable(self, draft)

    def replace_hook(self: LocalStore, key: str, data: bytes, *, if_match: str) -> str:
        result = replace(self, key, data, if_match=if_match)
        if crash and key == "control/owner.json":
            head = json.loads(data).get("head")
            if head:
                lines = self.read(head["segment"]).data.splitlines()[1:]
                batch = [json.loads(line) for line in lines]
                target = (
                    "decision"
                    if mode == "policy_before_close"
                    else "round_closed"
                    if terminal_policy
                    else "converge_started"
                )
                if any(e["type"] == target for e in batch):
                    os.kill(os.getpid(), signal.SIGKILL)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Ref, "build", build_hook)
        patch.setattr(FunctionTool, "collect", paid_collect)
        patch.setattr(LocalStore, "replace", replace_hook)
        patch.setattr(JournalWriter, "append", append_hook)
        patch.setattr(JournalWriter, "append_durable", durable_hook)
        await swarm.run(_plan(agents=("a",), rounds=3), run_id="published")


async def _published_crash_case(root: Path, mode: str, *, crash: bool) -> tuple[RunConfig, Crashed]:
    import os
    import signal
    import sys

    await anyio.to_thread.run_sync(partial(root.mkdir, parents=True))
    env = dict(os.environ)
    env["XDG_CACHE_HOME"] = str(root / "cache")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).parent),
            str(Path(__file__).parents[1]),
            str(Path(__file__).parents[2] / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    code = (
        "import sys, anyio; from pathlib import Path; "
        "from test_resume import _run_published_crash_case; "
        "anyio.run(_run_published_crash_case, Path(sys.argv[1]), sys.argv[2], sys.argv[3]=='1')"
    )
    with anyio.fail_after(30):
        result = await anyio.run_process(
            [sys.executable, "-c", code, str(root), mode, str(int(crash))],
            env=env,
            check=False,
        )
    assert result.returncode == (-signal.SIGKILL if crash else 0), result.stderr.decode()
    config = RunConfig.load(root / "runs" / "published" / "run.toml")
    store = LocalStore(Path(config.run_root) / "published")
    return config, Crashed(config, "published", store, cast(Lease, None), 0, Frontier(), {})


@pytest.mark.parametrize("reason", ["stop", "stop --now"])
async def test_real_crash_at_stopped_convergence_preserves_second_resume(
    tmp_path: Path,
    recorders: dict[str, _Recorder],
    reason: str,
) -> None:
    config, c = await _published_crash_case(tmp_path / "stopped", reason, crash=True)
    before = RunState.from_events(_events(c)).goals["g"]
    assert before.converge_started
    assert not before.finished
    assert before.rounds_closed == 1
    recorders["a"] = _Recorder()
    await Swarm(config).resume(c.run_id, force=True)
    first = RunState.from_events(_events(c)).goals["g"]
    assert first.status == "stopped"
    assert first.end_reason == reason
    assert first.rounds_closed == 1
    await Swarm(config).resume(c.run_id)
    assert [e.round for e in _events(c, "round_closed")] == [1, 2, 3]
    assert _goal_status(c, "g") == "finished"


@pytest.mark.parametrize("boundary", ["before_close", "after_close"])
async def test_real_crash_after_terminal_policy_starts_no_more_paid_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    from swarm_arena._planning.policy import Policy
    from swarm_arena._scoring.judge import Judge

    _, control = await _published_crash_case(tmp_path / "control", "policy", crash=False)
    mode = "policy_before_close" if boundary == "before_close" else "policy"
    config, c = await _published_crash_case(tmp_path / "crash", mode, crash=True)
    before = RunState.from_events(_events(c)).goals["g"]
    assert before.decisions[-1][0] == 1
    assert before.decisions[-1][1].no_result == "policy terminal"
    assert before.rounds_closed == (0 if boundary == "before_close" else 1)
    assert not before.selection_recorded
    assert not before.finished
    last_seq = _events(c)[-1].seq
    build = Ref.build
    constructions: list[type[Configurable]] = []

    def forbid_components(self: Ref, expected: type[_T]) -> _T:
        if expected in {Manager, Judge, Policy}:
            constructions.append(expected)
            raise AssertionError("terminal policy recovery must not construct components")
        return build(self, expected)

    monkeypatch.setattr(Ref, "build", forbid_components)
    result = await Swarm(config).resume(c.run_id, force=True)
    assert constructions == []
    later = [e for e in _events(c) if e.seq > last_seq]
    assert not [
        e
        for e in later
        if e.type
        in {
            "round_started",
            "tool_start",
            "command_start",
            "score",
            "reserve",
            "converge_started",
            "decision",
            "plan_revision",
            "plan_rejected",
        }
    ]
    assert result == {"g": []}
    state = RunState.from_events(_events(c))
    assert state.goals["g"].status == "no_result"
    assert state.goals["g"].no_result_reason == "policy terminal"
    assert state.goals["g"].rounds_closed == 1
    assert state.goals["g"].finished
    assert [e.round for e in _events(c, "round_closed")] == [1]
    assert [e.type for e in later if e.type in {"round_closed", "no_result", "goal_finished"}] == (
        ["round_closed", "no_result", "goal_finished"]
        if boundary == "before_close"
        else ["no_result", "goal_finished"]
    )

    def without_clock(event: Event) -> dict[str, Any]:
        payload = dict(event.payload)
        assert payload.pop("run_elapsed_s") >= 0
        return payload

    assert without_clock(_events(c, "goal_finished")[-1]) == without_clock(
        _events(control, "goal_finished")[-1]
    )
    solution = json.loads(c.store.read("goals/g/solution.json").data)
    expected = json.loads(control.store.read("goals/g/solution.json").data)
    for key in ("status", "reason", "finalists", "parts"):
        assert solution[key] == expected[key]
    for event_type in ("reserve", "settle"):
        assert [without_clock(e) for e in _events(c, event_type)] == [
            without_clock(e) for e in _events(control, event_type)
        ]


async def test_resume_checks_storage_before_takeover(
    crashed: Crashed, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[RunConfig] = []
    loop_thread = threading.get_ident()

    def spy(config: RunConfig) -> None:
        assert threading.get_ident() != loop_thread
        calls.append(config)
        raise ConditionalWritesUnsupported("create did not behave conditionally")

    monkeypatch.setattr(resume, "check_storage", spy)
    before = crashed.store.read("control/owner.json").data
    with pytest.raises(ConditionalWritesUnsupported):
        await Swarm(crashed.config).resume(crashed.run_id, force=True)
    assert calls == [crashed.config]
    assert crashed.store.read("control/owner.json").data == before
    owner = read_owner(crashed.store)
    assert owner is not None
    assert owner[0].epoch == 1


async def test_resume_repairs_missing_pages_of_finished_goals(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.run import all_events

    from swarm_arena._journal.views import ViewWriter

    config = with_executor(run_config, workspace_root=str(tmp_path / "workspace"))
    original = ViewWriter.write

    async def miss_first(self: ViewWriter, key: str, content: bytes, **kwargs: Any) -> None:
        if key == "goals/g/rounds/round-01.md":
            return
        await original(self, key, content, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ViewWriter, "write", miss_first)
        await Swarm(config).run(_plan(agents=("a",), rounds=2), run_id="pages")
    store = LocalStore(Path(config.run_root) / "pages")
    config = RunConfig.load(Path(config.run_root) / "pages" / "run.toml")
    assert not store.exists("goals/g/rounds/round-01.md")
    second = store.read("goals/g/rounds/round-02.md").data
    before = all_events(config)
    await Swarm(config).resume("pages", force=True)
    first = store.read("goals/g/rounds/round-01.md").data
    assert b"round 1 of 2" in first
    assert store.read("goals/g/rounds/round-02.md").data == second
    after = all_events(config)
    paid = {"reserve", "tool_start", "score", "decision", "curated", "plan_revision"}
    assert [e for e in after if e.type in paid] == [e for e in before if e.type in paid]
    await Swarm(config).resume("pages", force=True)
    assert store.read("goals/g/rounds/round-01.md").data == first
    assert store.read("goals/g/rounds/round-02.md").data == second


async def test_prepare_resume_repairs_page_after_real_close_crash(tmp_path: Path) -> None:
    import os
    import sys

    from _fakes.run import all_events

    environment = dict(os.environ)
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
    )
    script = (
        "import sys,anyio; from _fakes.curators import crash_driver; "
        "anyio.run(crash_driver,sys.argv[1],'round_closed')"
    )
    with anyio.fail_after(60):
        process = await anyio.run_process(
            [sys.executable, "-c", script, str(tmp_path)], env=environment, check=False
        )
    assert process.returncode == 91, process.stderr.decode()
    config = RunConfig.load(tmp_path / "runs" / "curation" / "run.toml")
    store = LocalStore(tmp_path / "runs" / "curation")
    key = "goals/g/rounds/round-01.md"
    assert not store.exists(key)
    before = all_events(config)
    summaries = [e.payload["summaries"][0] for e in before if e.type == "curated"]
    assert len(summaries) == 1
    with anyio.fail_after(60):
        runner = await resume_run(config, "curation", force=True, replanner=None)
        try:
            page = store.read(key).data.decode()
            assert summaries[0] in page
            assert "New learnings (2)" in page
            assert "· 2 learnings in namespace `n`." in page
            assert "round 1 of 2" in page
        finally:
            await runner.aclose()
    assert (tmp_path / "calls").read_text().splitlines() == ["called"]
    assert [e.type for e in all_events(config) if e.seq > before[-1].seq] == ["takeover"]
