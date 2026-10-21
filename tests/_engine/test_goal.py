import json
from pathlib import Path
from typing import Any, cast

import pytest
from _fakes.engine import EngineEnv

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    FunctionTool,
    Goal,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    ScriptedManager,
    TeamSpec,
)
from swarm_arena._engine.goal import GoalRunner
from swarm_arena._engine.stops import StopState
from swarm_arena._journal.replay import GoalState

Y_JUDGE = Ref.of(ProgramJudge("_fakes.tools:score_min_y")).model_copy(update={"name": "y"})


def _plan(
    *,
    teams: int = 2,
    agents_per_team: int = 1,
    rounds: int = 2,
    judges: list[Ref] | None = None,
    ranking: str = "y",
    check: str | None = None,
    converge_team: bool = False,
    converge_agents: int = 1,
    converge_outputs: int = 1,
    converge_round: RoundSpec | None = None,
    top_n: int = 2,
    **changes: Any,
) -> Plan:
    def team(name: str, count: int, offset: int, *, final: bool = False) -> TeamSpec:
        return TeamSpec(
            name=name,
            rationale="Explore",
            agents=[
                AgentSpec(
                    name=f"a{i}",
                    rationale="Explore",
                    tool=Ref.of(FunctionTool("_fakes.tools:square")),
                    manager=Ref.of(
                        ScriptedManager(
                            [
                                {
                                    "inputs": {"x": float(offset + i * 10 + r)},
                                    **({} if final else {"round": r}),
                                }
                                for r in range(1, (converge_outputs if final else rounds) + 1)
                            ]
                        )
                    ),
                )
                for i in range(count)
            ],
        )

    return Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[team(f"t{i}", agents_per_team, 10 * i) for i in range(teams)],
        judges=judges if judges is not None else [Y_JUDGE],
        ranking=ranking,
        checks=[] if check is None else [Ref(path=check)],
        rounds=[RoundSpec() for _ in range(rounds)],
        converge=ConvergeSpec(
            mode="team" if converge_team else "select",
            team=team("conv", converge_agents, 50, final=True) if converge_team else None,
            round=converge_round,
            top_n=top_n,
        ),
    ).model_copy(update=changes)


def _raise_flag(stops: StopState, flag: str, *, goal_id: str) -> None:
    if flag == "goal_budget":
        stops.goal_budget.add(goal_id)
    elif flag == "stop_now":
        stops.stop_now = stops.stop = True
    else:
        setattr(stops, flag, True)


async def test_rounds_select_converge_and_final_order(env: EngineEnv) -> None:
    out = await GoalRunner(env.deps, goal_id="g", plan=_plan(), replanner=None).run()
    assert out.status == "finished"
    assert len(out.finalists) == 2
    assert env.events("final_order")[-1].payload["candidate_ids"] == [c.id for c in out.finalists]
    assert json.loads(env.store.read("goals/g/solution.json").data)["status"] == "ok"
    assert env.store.exists("goals/g/plan.r0.toml")


async def test_barrier_judges_only_new_passing_candidates(env: EngineEnv) -> None:
    await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1), replanner=None).run()
    c1, c2 = sorted(env.candidates(), key=lambda c: c.round)
    by_round = {
        r: [e.payload["score"]["candidate_id"] for e in env.events("score") if e.round == r]
        for r in (1, 2)
    }
    assert by_round == {1: [c1.id], 2: [c2.id]}


async def test_all_gated_out_writes_no_result(env: EngineEnv) -> None:
    out = await GoalRunner(
        env.deps, goal_id="g", plan=_plan(check="_fakes.tools:AlwaysFailCheck"), replanner=None
    ).run()
    assert (out.status, out.finalists, out.reason) == ("no_result", [], "no passing candidate")
    assert env.events("score") == []
    assert json.loads(env.store.read("goals/g/solution.json").data)["status"] == "no_result"


async def test_all_judge_errors_writes_no_result(env: EngineEnv) -> None:
    raising = Ref.of(ProgramJudge("_fakes.tools:raising_score")).model_copy(update={"name": "y"})
    out = await GoalRunner(
        env.deps, goal_id="g", plan=_plan(rounds=1, judges=[raising]), replanner=None
    ).run()
    assert (out.status, out.finalists, out.reason) == ("no_result", [], "no passing candidate")
    assert env.events("judge_error")
    assert env.events("score") == []
    assert json.loads(env.store.read("goals/g/solution.json").data)["status"] == "no_result"


@pytest.mark.parametrize(
    ("flag", "reason", "select_only"),
    [
        ("stop", "stop", False),
        ("stop_now", "stop --now", True),
        ("run_timeout", "run_timeout_s", True),
        ("run_budget", "run budget", False),
        ("goal_budget", "goal budget", False),
    ],
)
async def test_early_end_stops_rounds_then_converges(
    env: EngineEnv, flag: str, reason: str, select_only: bool
) -> None:
    env.on_commit("round_closed", lambda ev: _raise_flag(env.deps.stops, flag, goal_id="g"))
    out = await GoalRunner(
        env.deps, goal_id="g", plan=_plan(rounds=3, converge_team=True), replanner=None
    ).run()
    assert [e.payload["round"] for e in env.events("round_closed")] == [1]
    assert env.events("converge_started")[0].payload["select_only"] is select_only
    assert any(e.team == "conv" for e in env.events("tool_start")) is not select_only
    assert (out.status, out.reason) == ("stopped", reason)
    assert out.finalists


async def test_active_agents_counts_converge_team_before_it_starts(env: EngineEnv) -> None:
    plan = _plan(teams=1, converge_team=True, converge_agents=2)
    assert GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).active_agents() == 2


async def test_round_closed_is_durable_before_next_frontier(env: EngineEnv) -> None:
    await GoalRunner(env.deps, goal_id="g", plan=_plan(), replanner=None).run()
    rc1 = next(e for e in env.events("round_closed") if e.payload["round"] == 1)
    f2 = next(e for e in env.events("frontier") if e.payload["round"] == 2)
    assert rc1.seq < f2.seq
    assert env.segment_index(rc1.seq) < env.segment_index(f2.seq)


async def test_judges_run_in_llm_scopes_and_scores_get_their_call_ids(env: EngineEnv) -> None:
    from _fakes.tools import SCOPE_PROBE_SEEN

    SCOPE_PROBE_SEEN.clear()
    plan = _plan(
        teams=1,
        rounds=1,
        judges=[Ref(path="_fakes.tools:ScopeProbeJudge", name="probe")],
        ranking="probe",
    )
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert [(s.role, s.goal_id, s.round, s.force_no_cache_read) for s in SCOPE_PROBE_SEEN] == [
        ("judge", "g", 1, False),
        ("judge", "g", 0, True),
    ]
    holders = [
        e.payload["reservation"]["holder"]
        for e in env.events("reserve")
        if e.payload["reservation"]["kind"] == "tranche"
    ]
    assert holders == ["g/:judges", "g/:final"]
    assert {e.round: e.payload["score"]["llm_call_ids"] for e in env.events("score")} == {
        1: ["llm-1"],
        0: ["llm-2"],
    }


async def test_stop_reason_is_in_solution_view(env: EngineEnv) -> None:
    env.on_commit("round_closed", lambda ev: _raise_flag(env.deps.stops, "stop", goal_id="g"))
    await GoalRunner(env.deps, goal_id="g", plan=_plan(rounds=3), replanner=None).run()
    view = json.loads(env.store.read("goals/g/solution.json").data)
    assert view["reason"] == "stop"
    assert view["source_head"]["seq"] >= env.events("goal_finished")[-1].seq


async def test_converge_team_runs_one_round_paid_from_final_slice(env: EngineEnv) -> None:
    out = await GoalRunner(
        env.deps, goal_id="g", plan=_plan(rounds=1, converge_team=True), replanner=None
    ).run()
    holders = {
        e.payload["reservation"]["holder"]
        for e in env.events("reserve")
        if e.payload["reservation"]["kind"] == "tool"
    }
    assert "g/:final" in holders
    assert not any(h.startswith("g/conv/") for h in holders)
    survivors = {c.id for c in env.candidates() if c.team != "conv"}
    assert all(
        c.parent_ids and set(c.parent_ids) <= survivors
        for c in env.candidates()
        if c.team == "conv"
    )
    assert out.finalists


async def test_team_converge_final_order_holds_top_n(env: EngineEnv) -> None:
    env = env.with_limits(max_tool_runs_per_round=4)
    plan = _plan(rounds=1, converge_team=True, converge_outputs=4, top_n=2)
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert len([c for c in env.candidates() if c.team == "conv"]) == 4
    assert len(env.events("final_order")[-1].payload["candidate_ids"]) == 2
    assert len(json.loads(env.store.read("goals/g/solution.json").data)["finalists"]) == 2


async def test_team_converge_keeps_newest_fidelity_only(env: EngineEnv) -> None:
    plan = _plan(rounds=1, converge_team=True, converge_round=RoundSpec(fidelity="full"))
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert all(
        env.candidate(i).fidelity == "full"
        for i in env.events("final_order")[-1].payload["candidate_ids"]
    )


async def test_converge_outputs_receive_other_judges(env: EngineEnv) -> None:
    extra = Y_JUDGE.model_copy(update={"name": "extra"})
    plan = _plan(rounds=1, converge_team=True, judges=[Y_JUDGE, extra])
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    ids = {c.id for c in env.candidates() if c.team == "conv"}
    assert ids
    for id in ids:
        assert {
            e.payload["score"]["judge"]
            for e in env.events("score")
            if e.payload["score"]["candidate_id"] == id and e.round != 0
        } == {"y", "extra"}


@pytest.mark.parametrize(
    ("tool", "expected"), [("ConstantFingerprintTool", 1), ("RaisingFingerprintTool", 2)]
)
async def test_distinct_uses_durable_fingerprints(env: EngineEnv, tool: str, expected: int) -> None:
    from swarm_arena._journal.reader import read_chain
    from swarm_arena._journal.replay import RunState

    plan = _plan(rounds=1)
    teams = [
        t.model_copy(
            update={
                "agents": [
                    a.model_copy(
                        update={
                            "tool": Ref(
                                path=f"_fakes.tools:{tool}",
                                config={"function": "_fakes.tools:square"},
                            )
                        }
                    )
                    for a in t.agents
                ]
            }
        )
        for t in plan.teams
    ]
    runner = GoalRunner(
        env.deps, goal_id="g", plan=plan.model_copy(update={"teams": teams}), replanner=None
    )
    out = await runner.run()
    assert len(out.finalists) == expected
    assert env.deps.journal.head is not None
    state = RunState.from_events(
        e for _, events in read_chain(env.store, env.deps.journal.head) for e in events
    )
    assert runner.recorded.fingerprints == state.fingerprints
    assert "fingerprint-secret-sentinel" not in "".join(
        e.model_dump_json() for e in env.events("tool_end")
    )


async def test_policy_prunes_with_reasons_and_pruned_agents_stop(env: EngineEnv) -> None:
    from swarm_arena import KeepTopTeams

    plan = _plan(teams=3, policy=Ref.of(KeepTopTeams(1)))
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    prune = env.events("decision")[0].payload["decision"]["prune"]
    assert set(prune) == {"t1", "t2"}
    assert all(prune.values())
    assert {c.team for c in env.candidates() if c.round == 2} == {"t0"}


async def test_survivor_shares_grow(env: EngineEnv) -> None:
    from swarm_arena import KeepTopTeams
    from swarm_arena.tool import Estimate

    env = env.with_limits(max_usd=100, default_run_reservation=Estimate(usd=0.01, tokens=0))
    await GoalRunner(
        env.deps, goal_id="g", plan=_plan(teams=2, policy=Ref.of(KeepTopTeams(1))), replanner=None
    ).run()
    first, second = env.events("round_started")
    assert second.payload["shares"]["g/t0/a0"]["usd"] > first.payload["shares"]["g/t0/a0"]["usd"]


async def test_candidate_fidelity_follows_round_spec(env: EngineEnv) -> None:
    plan = _plan(teams=1).model_copy(
        update={
            "rounds": [RoundSpec(fidelity="quick"), RoundSpec(tool_overrides={"timeout_s": 60.0})]
        }
    )
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert [c.fidelity for c in sorted(env.candidates(), key=lambda c: c.round)] == [
        s.effective_fidelity() for s in plan.rounds
    ]


def _tool_plan(plan: Plan, tool: Ref) -> Plan:
    return plan.model_copy(
        update={
            "teams": [
                team.model_copy(
                    update={
                        "agents": [agent.model_copy(update={"tool": tool}) for agent in team.agents]
                    }
                )
                for team in plan.teams
            ]
        }
    )


async def test_reserve_exhausted_skips_to_select_only(env: EngineEnv) -> None:
    from _fakes.tools import WRITE, PyTool

    from swarm_arena.tool import Estimate

    e = env.with_limits(max_usd=10.0, default_run_reservation=Estimate(tokens=0, usd=2.5))
    plan = _tool_plan(
        _plan(teams=1, judges=[Y_JUDGE, Ref(path="_fakes.tools:ProviderJudge", name="extra")]),
        Ref.of(PyTool({"code": WRITE.format("{'data': {'y': 1.0}, 'usage': {'usd': 8.0}}")})),
    )
    out = await GoalRunner(e.deps, goal_id="g", plan=plan, replanner=None).run()
    assert [ev.payload["round"] for ev in e.events("reserve_exhausted")] == [1]
    assert [ev.payload["round"] for ev in e.events("round_closed")] == [1]
    assert e.events("converge_started")[0].payload["select_only"] is True
    assert len(out.finalists) == 1
    assert (out.status, out.reason) == ("stopped", "reserve_exhausted")


async def test_free_judges_make_no_judge_reservations(env: EngineEnv) -> None:
    from swarm_arena import ScriptJudge

    extra = Ref.of(
        ScriptJudge(
            command=[
                "python",
                "-c",
                "import json,os; from pathlib import Path; "
                "Path(os.environ['SWARM_ARENA_RESULT']).write_text("
                "json.dumps({'value': 1}))",
            ]
        )
    ).model_copy(update={"name": "extra"})
    await GoalRunner(
        env.deps, goal_id="g", plan=_plan(rounds=1, judges=[Y_JUDGE, extra]), replanner=None
    ).run()
    assert not [
        e for e in env.events("reserve") if e.payload["reservation"]["kind"] in {"judge", "final"}
    ]
    starts = [e for e in env.events("command_start") if e.payload["component"] == "extra"]
    assert starts
    assert {e.payload["job_id"] for e in starts} <= {
        e.payload["job_id"] for e in env.events("command_end")
    }


async def test_final_judge_reservation_on_final_slice(env: EngineEnv) -> None:
    plan = _plan(
        rounds=1, judges=[Ref(path="_fakes.tools:ProviderJudge", name="prov")], ranking="prov"
    )
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    finals = [
        e.payload["reservation"]
        for e in env.events("reserve")
        if e.payload["reservation"]["kind"] == "final"
    ]
    assert len(finals) == 1
    assert finals[0]["holder"] == "g/:final"
    assert finals[0]["amount"]["usd"] == 2.0
    assert (
        next(e for e in env.events("settle") if e.payload["id"] == finals[0]["id"]).payload[
            "estimated"
        ]
        is True
    )


async def test_provider_reservation_includes_unique_anchors(env: EngineEnv) -> None:
    plan = _plan(
        teams=1,
        judges=[Ref(path="_fakes.tools:AnchoredProviderJudge", name="prov")],
        ranking="prov",
    )
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    reservations = [
        e.payload["reservation"]
        for e in env.events("reserve")
        if e.payload["reservation"]["kind"] == "judge"
    ]
    assert [r["amount"]["usd"] for r in reservations] == [1.0, 2.0]
    assert len([e for e in env.events("score") if e.round == 2]) == 2


async def test_final_judge_reservation_exhausted_writes_no_result(env: EngineEnv) -> None:
    from _fakes.tools import WRITE, PyTool

    from swarm_arena.tool import Estimate

    e = env.with_limits(max_usd=10.0, default_run_reservation=Estimate(tokens=0, usd=2.0))
    plan = _tool_plan(
        _plan(
            teams=1,
            rounds=1,
            judges=[Ref(path="_fakes.tools:ProviderJudge", name="prov")],
            ranking="prov",
        ),
        Ref.of(PyTool({"code": WRITE.format("{'data': {'y': 1.0}, 'usage': {'usd': 8.0}}")})),
    )
    out = await GoalRunner(e.deps, goal_id="g", plan=plan, replanner=None).run()
    assert (out.status, out.reason, out.finalists) == ("no_result", "final slice used up", [])
    assert e.events("no_result")[-1].payload["reason"] == "final slice used up"


def _catalog_env(env: EngineEnv, plan: Plan, **tools: Ref) -> EngineEnv:
    from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog
    from swarm_arena._planning.provenance import ref_sites

    entries = [
        CatalogEntry(
            name=f"entry-{i}",
            kind=site.kind,
            ref=site.ref.path,
            config=site.ref.config,
            description="Approved component",
        )
        for i, site in enumerate(ref_sites(plan))
    ]
    entries += [
        CatalogEntry(
            name=name, kind="tool", ref=ref.path, config=ref.config, description="Approved tool"
        )
        for name, ref in tools.items()
    ]
    env.deps.config = env.deps.config.model_copy(
        update={"planner_catalog": PlannerCatalog(entries=entries)}
    )
    return env


def _admitted(env: EngineEnv, plan: Plan, base_dir: "Path") -> Plan:
    from swarm_arena._tooling.files import snapshot_plan

    return snapshot_plan(
        Plan._from_data(plan.model_dump(mode="json"), base_dir=base_dir), env.deps.artifacts
    )


async def test_replanner_revision_is_autonomous_and_validated(
    env: EngineEnv, tmp_path: "Path"
) -> None:
    from _fakes.planners import AddAgentPlanner

    plan = _admitted(env, _plan(teams=1), tmp_path)
    _catalog_env(env, plan, **{"catalog-agent": plan.teams[0].agents[0].tool})
    out = await GoalRunner(
        env.deps, goal_id="g", plan=plan, replanner=AddAgentPlanner("catalog-agent")
    ).run()
    rev = env.events("plan_revision")[-1].payload
    assert (rev["origin"], rev["plan"]["revision"], rev["plan"]["parent_revision"]) == (
        "autonomous",
        1,
        0,
    )
    assert env.store.exists("goals/g/plan.r1.toml")
    assert out.finalists
    assert any(c.agent == "new-1" for c in env.candidates())


async def test_replanner_runs_in_planner_scope_and_revisions_journal_call_ids(
    env: EngineEnv, tmp_path: "Path"
) -> None:
    from _fakes.planners import AddAgentPlanner

    from swarm_arena import RunConfig
    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.llm_scope import LLMScope, current_llm_scope
    from swarm_arena.planning import History, Planner

    class _ScopeProbePlanner(Planner):
        def __init__(self, inner: Planner) -> None:
            self.inner, self.seen = inner, []

        async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
            scope = current_llm_scope()
            assert scope is not None
            self.seen.append(scope)
            call = await scope.tranche.reserve_call(Amount(usd=0.01))
            await scope.tranche.settle_call(call, Amount(usd=0.01))
            scope.call_ids.append(f"llm-p{len(self.seen)}")
            return await self.inner.plan(goal, config, history)

    plan = _admitted(env, _plan(teams=1), tmp_path)
    _catalog_env(env, plan, **{"catalog-agent": plan.teams[0].agents[0].tool})
    planner = _ScopeProbePlanner(AddAgentPlanner("catalog-agent"))
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=planner).run()
    s: LLMScope = planner.seen[0]
    assert (s.role, s.goal_id, s.team, s.agent, s.round, s.candidate_id) == (
        "planner",
        "g",
        "",
        "",
        1,
        None,
    )
    revs = [e.payload for e in env.events("plan_revision")]
    assert (revs[0]["planner_call_ids"], revs[1]["planner_call_ids"]) == ([], ["llm-p1"])
    assert "g/:planner" in {
        e.payload["reservation"]["holder"]
        for e in env.events("reserve")
        if e.payload["reservation"]["kind"] == "tranche"
    }


async def test_rejected_revision_is_journaled_and_old_plan_kept(
    env: EngineEnv, tmp_path: "Path"
) -> None:
    from _fakes.planners import SequencePlanner

    plan = _admitted(env, _plan(teams=1), tmp_path)
    bad = _tool_plan(plan, Ref(path="untrusted_module:SecretTool"))
    out = await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([bad])).run()
    assert env.events("plan_rejected")[-1].payload["errors"]
    assert len(env.events("plan_revision")) == 1
    assert not env.store.exists("goals/g/plan.r1.toml")
    assert out.status == "finished"


async def test_planner_exception_is_type_only_and_skipped(env: EngineEnv) -> None:
    from swarm_arena import RunConfig
    from swarm_arena.planning import History, Planner

    class BadPlanner(Planner):
        async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
            raise RuntimeError("planner-secret-sentinel")

    out = await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1), replanner=BadPlanner()).run()
    errors = env.events("plan_rejected")[-1].payload["errors"]
    assert errors == ["planner error: RuntimeError"]
    assert "planner-secret-sentinel" not in "".join(e.model_dump_json() for e in env._events())
    assert out.status == "finished"


async def test_replanned_fingerprints_preserve_producing_effective_settings(
    env: EngineEnv, tmp_path: "Path"
) -> None:
    from _fakes.planners import SequencePlanner

    first_tool = Ref(
        path="_fakes.tools:ConstantFingerprintTool",
        config={"function": "_fakes.tools:square", "timeout_s": 30.0},
    )
    second_tool = first_tool.model_copy(update={"config": {**first_tool.config, "timeout_s": 60.0}})
    plan = _admitted(env, _tool_plan(_plan(teams=1), first_tool), tmp_path)
    _catalog_env(env, plan, changed=second_tool)
    runner = GoalRunner(
        env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([_tool_plan(plan, second_tool)])
    )
    out = await runner.run()
    assert len(env.events("plan_revision")) == 2
    assert set(runner.recorded.fingerprints.values()) == {"timeout-30.0", "timeout-60.0"}
    assert len(out.finalists) == 2


def _parts_plan(*, rounds: int = 1, own_judges: bool = False, fail_join: bool = False) -> Plan:
    from swarm_arena import KeepTopTeams, PartSpec

    value = Ref.of(ProgramJudge("_fakes.tools:goal_score_value")).model_copy(
        update={"name": "value"}
    )
    teams = []
    for part in ["a", "b"]:
        for i in (1, 2) if own_judges else (1,):
            teams.append(
                TeamSpec(
                    name=f"{part}{i}",
                    rationale="Explore",
                    part=part,
                    agents=[
                        AgentSpec(
                            name="worker",
                            rationale="Explore",
                            tool=Ref.of(FunctionTool("_fakes.tools:goal_part_value")),
                            manager=Ref.of(
                                ScriptedManager(
                                    [
                                        {"round": r, "inputs": {"value": float(i * 10 + r)}}
                                        for r in range(1, rounds + 1)
                                    ]
                                )
                            ),
                        )
                    ],
                )
            )
    parts = [
        PartSpec(
            name=part,
            goal=Goal(id=f"part-{part}", namespace="n"),
            checks=[],
            judges=[
                Ref.of(
                    ProgramJudge(
                        "_fakes.tools:goal_score_negative_value"
                        if part == "b"
                        else "_fakes.tools:goal_score_value"
                    )
                ).model_copy(update={"name": part})
            ]
            if own_judges
            else None,
            ranking=part if own_judges else None,
        )
        for part in ["a", "b"]
    ]
    joiner = AgentSpec(
        name="joiner",
        rationale="Combine",
        tool=Ref.of(FunctionTool("_fakes.tools:join_sum")),
        manager=Ref.of(ScriptedManager([{"inputs": {}}])),
    )
    return Plan(
        goal=Goal(id="g", namespace="n"),
        teams=teams,
        parts=parts,
        judges=[value],
        ranking="value",
        checks=[Ref(path="_fakes.tools:AlwaysFailCheck")] if fail_join else [],
        policy=Ref.of(KeepTopTeams(1)),
        rounds=[RoundSpec() for _ in range(rounds)],
        converge=ConvergeSpec(
            mode="team", team=TeamSpec(name="join", rationale="Combine parts", agents=[joiner])
        ),
    )


async def test_parts_join_and_main_goal_scoring(env: EngineEnv) -> None:
    out = await GoalRunner(env.deps, goal_id="g", plan=_parts_plan(), replanner=None).run()
    join = out.finalists[0]
    assert {env.candidate(p).part for p in join.parent_ids} == {"a", "b"}
    assert join.part is None
    assert join.output.data["value"] == 22.0
    assert {e.payload["score"]["judge"] for e in env.events("score") if e.round == 0} == {"value"}


async def test_parts_with_own_ranking_judges_prune_and_join(env: EngineEnv) -> None:
    out = await GoalRunner(
        env.deps, goal_id="g", plan=_parts_plan(rounds=2, own_judges=True), replanner=None
    ).run()
    pruned = env.events("decision")[0].payload["decision"]["prune"]
    assert set(pruned) == {"a2", "b1"}
    assert sorted(env.candidate(p).team for p in out.finalists[0].parent_ids) == ["a1", "b2"]


async def test_stop_now_with_parts_writes_unjoined_without_final_judge(env: EngineEnv) -> None:
    env.on_commit("round_closed", lambda ev: _raise_flag(env.deps.stops, "stop_now", goal_id="g"))
    out = await GoalRunner(env.deps, goal_id="g", plan=_parts_plan(rounds=2), replanner=None).run()
    assert set(env.events("final_order")[-1].payload["parts"]) == {"a", "b"}
    assert not [e for e in env.events("score") if e.round == 0]
    view = json.loads(env.store.read("goals/g/solution.json").data)
    assert (view["status"], view["reason"], out.status) == ("unjoined", "stop --now", "stopped")


async def test_part_with_no_active_agent_keeps_best(env: EngineEnv) -> None:
    env = env.with_limits(watchdog_idle_rounds=1)
    plan = _parts_plan(rounds=2)
    plan = plan.model_copy(
        update={
            "parts": [
                plan.parts[0],
                plan.parts[1].model_copy(
                    update={"checks": [Ref(path="_fakes.tools:FailLaterRoundCheck")]}
                ),
            ]
        }
    )
    out = await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    kept = env.events("part_kept")[-1].payload
    assert kept["part"] == "b"
    assert kept["candidate_id"] in out.finalists[0].parent_ids
    assert env.candidate(kept["candidate_id"]).round == 1


async def test_join_failure_replans_once_then_no_result(env: EngineEnv, tmp_path: Path) -> None:
    from _fakes.planners import SequencePlanner

    from swarm_arena import RunConfig
    from swarm_arena.planning import History

    plan = _admitted(env, _parts_plan(fail_join=True), tmp_path)
    revised = plan.model_copy(update={"rounds": [RoundSpec(), RoundSpec()]})
    # Both ordinary teams must have work in the newly added round.
    revised = revised.model_copy(update={"teams": _parts_plan(rounds=2).teams})
    _catalog_env(env, revised)

    class JoiningPlanner(SequencePlanner):
        def __init__(self) -> None:
            super().__init__([revised])
            self.failures: list[list[str]] = []

        async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
            assert history is not None
            self.failures.append(list(history.join_failures))
            return await super().plan(goal, config, history)

    planner = JoiningPlanner()
    out = await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=planner).run()
    assert len(planner.failures) == 1
    assert planner.failures[0]
    assert len(env.events("join_failed")) == 2
    assert [e.payload["round"] for e in env.events("round_closed")] == [1, 2]
    assert out.status == "no_result"
    assert out.reason == "; ".join(env.events("join_failed")[-1].payload["reasons"])
    assert json.loads(env.store.read("goals/g/solution.json").data)["status"] == "no_result"


def _with_part_c(plan: Plan, globs: list[str]) -> Plan:
    from swarm_arena import PartSpec
    from swarm_arena._model.goal import FileSet

    part = PartSpec(name="c", goal=Goal(id="part-c", namespace="n", files=FileSet(globs=globs)))
    team = plan.teams[0].model_copy(update={"name": "c1", "part": "c"})
    return plan.model_copy(update={"parts": [*plan.parts, part], "teams": [*plan.teams, team]})


async def _check_part_file_resolution(env: EngineEnv, tmp_path: Path, outside: bool) -> None:
    from _fakes.planners import SequencePlanner

    from swarm_arena._model.goal import FileSet

    (tmp_path / "problem").mkdir()
    (tmp_path / "problem/a.txt").write_text("input")
    env = env.with_limits(max_agents_per_fingerprint=3)
    plan = _parts_plan(rounds=2)
    plan = plan.model_copy(
        update={"goal": plan.goal.model_copy(update={"files": FileSet(globs=["problem/a.txt"])})}
    )
    plan = _admitted(env, plan, tmp_path)
    main_sha = plan.goal.files.manifest
    _catalog_env(env, plan)
    glob = "judge/hidden/*.npz" if outside else "problem/a.*"
    await GoalRunner(
        env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([_with_part_c(plan, [glob])])
    ).run()
    if outside:
        rejected = env.events("plan_rejected")
        assert rejected
        assert any(glob in error for error in rejected[-1].payload["errors"])
        assert not env.store.exists("goals/g/plan.r1.toml")
    else:
        assert len(env.events("plan_revision")) == 2, [
            e.payload for e in env.events("plan_rejected")
        ]
        revision = env.events("plan_revision")[1].payload["plan"]
        part = next(p for p in revision["parts"] if p["name"] == "c")
        assert part["goal"]["files"]["manifest"] == main_sha
        assert env.store.exists("goals/g/plan.r1.toml")


async def test_replanned_part_files_resolve_from_main_manifest(
    env: EngineEnv, tmp_path: Path
) -> None:
    await _check_part_file_resolution(env, tmp_path, False)


async def test_replanned_part_glob_outside_main_manifest_rejected(
    env: EngineEnv, tmp_path: Path
) -> None:
    await _check_part_file_resolution(env, tmp_path, True)


async def test_replanned_catalog_script_tool_is_snapshotted_and_runs(
    env: EngineEnv, tmp_path: Path
) -> None:
    from _fakes.planners import AddAgentPlanner

    from swarm_arena import ScriptTool
    from swarm_arena._model.canonical import sha256_hex

    code = (
        b"import json, os, hashlib\nfrom pathlib import Path\n"
        b"Path(os.environ['SWARM_ARENA_RESULT']).write_text(json.dumps({'data': "
        b"{'sha': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'y': 0.0}}))\n"
    )
    (tmp_path / "problem").mkdir()
    (tmp_path / "problem/run.py").write_bytes(code)
    script = Ref.of(
        ScriptTool(
            command=["python", "problem/run.py"],
            files=["problem/run.py"],
            params={"x": {"type": "number"}},
        )
    )
    plan = _admitted(env, _plan(teams=1), tmp_path)
    _catalog_env(env, plan, **{"catalog-script": script})
    await GoalRunner(
        env.deps, goal_id="g", plan=plan, replanner=AddAgentPlanner("catalog-script")
    ).run()
    candidate = next(c for c in env.candidates() if c.agent == "new-1")
    assert candidate.round == 2
    assert candidate.output.data["sha"] == sha256_hex(code)


async def test_replan_capacity_is_rechecked_against_other_goals(
    env: EngineEnv, tmp_path: Path
) -> None:
    from _fakes.planners import AddAgentPlanner

    from swarm_arena._engine.admission import RunAdmission

    env = env.with_limits(max_agents_total=2)
    env.deps.admission = RunAdmission(2)
    await env.deps.admission.update("other", 1)
    plan = _admitted(env, _plan(teams=1), tmp_path)
    _catalog_env(env, plan, **{"catalog-agent": plan.teams[0].agents[0].tool})
    await GoalRunner(
        env.deps, goal_id="g", plan=plan, replanner=AddAgentPlanner("catalog-agent")
    ).run()
    assert len(env.events("plan_revision")) == 1
    assert "max_agents_total" in env.events("plan_rejected")[-1].payload["errors"][0]
    assert {c.agent for c in env.candidates()} == {"a0"}


async def _crash_after_tool_end(env: EngineEnv, plan: Plan, *, round: int) -> "GoalState":
    import anyio

    from swarm_arena._journal.reader import read_chain
    from swarm_arena._journal.replay import RunState

    append = env.deps.journal.append_durable
    with anyio.CancelScope() as scope:
        env.on_commit("tool_end", lambda event: scope.cancel() if event.round == round else None)
        await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    cast(Any, env.deps.journal).append_durable = append
    head = env.deps.journal.head
    assert head is not None
    state = RunState.from_events(
        event for _, events in read_chain(env.store, head) for event in events
    )
    assert state.goals["g"].rounds_closed == round - 1
    return state.goals["g"]


async def test_early_break_merges_recovered_pairs_into_converge(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine.round import RoundOutcome, RoundRunner

    plan = _plan(teams=1, converge_team=True)
    state = await _crash_after_tool_end(env, plan, round=2)
    c1, cx = sorted(env.candidates(), key=lambda c: c.round)
    seen: list[tuple[bool, list[str]]] = []
    real = RoundRunner.run

    async def spy(self: RoundRunner, round: int) -> RoundOutcome:
        seen.append((self.converge, [c.id for _, c in self.repeat_entries]))
        return await real(self, round)

    monkeypatch.setattr(RoundRunner, "run", spy)
    env.deps.stops.stop = True
    await GoalRunner(
        env.deps, goal_id="g", plan=plan, replanner=None, state=state, journal_r0=False
    ).run()
    assert seen == [(True, [c1.id, cx.id])]


async def test_restarted_round_recovers_private_pairs_then_promotes_them(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine.round import RoundOutcome, RoundRunner

    plan = _plan(teams=1, rounds=3)
    first = plan.teams[0].agents[0]
    first = first.model_copy(
        update={
            "manager": Ref.of(
                ScriptedManager(
                    [
                        {"round": 1, "inputs": {"x": 1.0}},
                        {"round": 2, "inputs": {"x": 2.0}},
                        {"round": 3, "inputs": {"x": 2.0}},
                    ]
                )
            )
        }
    )
    second = first.model_copy(
        update={
            "name": "other",
            "manager": Ref.of(ScriptedManager([{"round": 3, "inputs": {"x": 2.0}}])),
        }
    )
    plan = plan.model_copy(
        update={"teams": [plan.teams[0].model_copy(update={"agents": [first, second]})]}
    )
    state = await _crash_after_tool_end(env, plan, round=2)
    c1, recovered = sorted(env.candidates(), key=lambda c: c.round)
    seen: list[tuple[int, list[str], dict[str, list[str]]]] = []
    real = RoundRunner.run

    async def spy(self: RoundRunner, round: int) -> RoundOutcome:
        seen.append(
            (
                round,
                [c.id for _, c in self.repeat_entries],
                {key: [c.id for _, c in values] for key, values in self.resumed.items()},
            )
        )
        return await real(self, round)

    monkeypatch.setattr(RoundRunner, "run", spy)
    runner = GoalRunner(
        env.deps, goal_id="g", plan=plan, replanner=None, state=state, journal_r0=False
    )
    await runner.run()
    assert seen == [(2, [c1.id], {"t0/a0": [recovered.id]}), (3, [c1.id, recovered.id], {})]
    assert len(env.candidates()) == 2
    assert [c.id for _, c in runner.repeat_entries] == [c1.id, recovered.id]
    assert any(e.payload.get("refused") == recovered.id for e in env.events("watchdog"))
    assert {e.payload["score"]["candidate_id"] for e in env.events("score") if e.round == 2} == {
        recovered.id
    }


async def test_replanner_can_extend_remaining_round_specs(env: EngineEnv, tmp_path: Path) -> None:
    from _fakes.planners import SequencePlanner

    plan = _plan(teams=1, rounds=3)
    plan = _admitted(env, plan.model_copy(update={"rounds": plan.rounds[:2]}), tmp_path)
    extended = plan.model_copy(update={"rounds": [RoundSpec(), RoundSpec(), RoundSpec()]})
    _catalog_env(env, plan)
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=SequencePlanner([extended])).run()
    assert [e.payload["round"] for e in env.events("round_closed")] == [1, 2, 3]


@pytest.mark.parametrize(
    ("tool", "expected"),
    [("ConstantFingerprintTool", [52.0, 1.0]), ("RaisingFingerprintTool", [52.0, 51.0])],
)
async def test_team_final_order_deduplicates_after_final_reranking(
    env: EngineEnv, tool: str, expected: list[float]
) -> None:
    ranking = Ref.of(ProgramJudge("_fakes.tools:score_max_y")).model_copy(update={"name": "y"})
    plan = _plan(teams=1, rounds=1, converge_team=True, converge_outputs=2, judges=[ranking])
    team = plan.converge.team
    assert team is not None
    team = team.model_copy(
        update={
            "agents": [
                a.model_copy(
                    update={
                        "tool": Ref(
                            path=f"_fakes.tools:{tool}", config={"function": "_fakes.tools:square"}
                        )
                    }
                )
                for a in team.agents
            ]
        }
    )
    plan = plan.model_copy(update={"converge": plan.converge.model_copy(update={"team": team})})
    out = await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert [c.output.data["x"] for c in out.finalists] == expected
    assert len([e for e in env.events("score") if e.round == 0]) == 3


async def test_malformed_planner_result_is_rejected_without_crashing(env: EngineEnv) -> None:
    from typing import cast

    from swarm_arena import RunConfig
    from swarm_arena.planning import History, Planner

    class BadPlanner(Planner):
        async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
            return cast(Plan, "planner-secret-sentinel")

    out = await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1), replanner=BadPlanner()).run()
    assert out.status == "finished"
    assert env.events("plan_rejected")[-1].payload["errors"] == ["planner error: TypeError"]
    assert len(env.events("plan_revision")) == 1


async def test_unfunded_converge_barrier_selects_only_prior_survivors(env: EngineEnv) -> None:
    from _fakes.tools import WRITE, PyTool

    from swarm_arena.tool import Estimate

    env = env.with_limits(max_usd=10.0, default_run_reservation=Estimate(usd=1, tokens=0))
    plan = _plan(
        teams=1,
        rounds=1,
        converge_team=True,
        judges=[Y_JUDGE, Ref(path="_fakes.tools:ProviderJudge", name="extra")],
    )
    team = plan.converge.team
    assert team is not None
    agent = team.agents[0].model_copy(
        update={
            "tool": Ref.of(
                PyTool({"code": WRITE.format("{'data': {'y': 0.0}, 'usage': {'usd': 8.5}}")})
            )
        }
    )
    plan = plan.model_copy(
        update={
            "converge": plan.converge.model_copy(
                update={"team": team.model_copy(update={"agents": [agent]})}
            )
        }
    )
    out = await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert env.events("reserve_exhausted")[-1].payload["round"] == 2
    assert (out.status, out.reason) == ("stopped", "reserve_exhausted")
    assert out.finalists
    assert all(c.team == "t0" for c in out.finalists)


def _curator_plan(*, rounds: int = 2) -> Plan:
    plan = _plan(teams=1, rounds=rounds)
    for team in plan.teams:
        team.agents[0] = team.agents[0].model_copy(
            update={
                "manager": team.agents[0].manager.model_copy(
                    update={
                        "config": {**team.agents[0].manager.config, "post_learnings": True},
                        "path": "_fakes.curators:PostingManager",
                    }
                )
            }
        )
    return plan


async def test_engine_runs_curator_at_round_end(env: EngineEnv) -> None:
    from swarm_arena._journal.replay import RunState

    env.deps.config = env.deps.config.model_copy(
        update={"curator": Ref(path="_fakes.curators:MergeAll")}
    )
    plan = _curator_plan()
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    receipts = env.events("curated")
    assert [e.round for e in receipts] == [1, 2]
    state = RunState.from_events(env._events())
    originals = [item for item in state.learnings.values() if item.author != "curator"]
    assert len(originals) == 2
    assert len({item.title for item in originals}) == 2
    for item in originals:
        assert item.verified
        assert len(item.evidence) == 1
        candidate = state.candidates[item.evidence[0]]
        assert candidate.round == item.round
        assert candidate.output.status == "ok"
        assert not candidate.output.partial
        assert not candidate.gated_out
    for event in receipts:
        decision = next(e for e in env.events("decision") if e.round == event.round)
        assert event.payload["decision_seq"] == decision.seq
        assert event.payload["revision"] == 0
        assert event.payload["errors"] == []
        for id in event.payload["summaries"]:
            record = next(e for e in env.events("learning") if e.payload["learning"]["id"] == id)
            assert env.segment_index(record.seq) == env.segment_index(event.seq)
        assert (
            decision.seq
            < event.seq
            < next(e.seq for e in env.events("round_closed") if e.round == event.round)
        )
    frontier = state.goals["g"].frontiers[2]
    view = env.deps.board.snapshot(plan.goal.key(), frontier)
    ids = [item.id for item in view.learnings("n")]
    first = state.learnings[receipts[0].payload["summaries"][0]]
    assert all(ids.index(first.id) < ids.index(id) for id in first.supersedes)


@pytest.mark.parametrize("configured", [False, True])
async def test_curator_skips_without_configuration_or_new_learnings(
    env: EngineEnv, configured: bool
) -> None:
    if configured:
        env.deps.config = env.deps.config.model_copy(
            update={"curator": Ref(path="_fakes.curators:MergeAll")}
        )
    await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1, rounds=1), replanner=None).run()
    assert env.events("curated") == []


async def _curation_checkpoint(env: EngineEnv) -> GoalRunner:
    """Commit a real policy boundary without spending on a tool subprocess."""
    from swarm_arena._engine.goal import r0_revision_draft
    from swarm_arena._journal.event import EventDraft, EventType
    from swarm_arena._model.planning_types import Decision
    from swarm_arena.board import Learning

    plan = _plan(teams=1, rounds=1)
    env.deps.config = env.deps.config.model_copy(
        update={
            "curator": Ref(path="_fakes.curators:MergeAll"),
            "settle_timeout_s": 0.02,
        }
    )
    env.deps.journal.append(r0_revision_draft(plan, plan.base_dir))
    item = Learning(
        id="L-source",
        namespace="n",
        run_id=env.deps.run_id,
        goal_id="g",
        round=1,
        author="agent",
        title="Original",
        text="Evidence",
        evidence=["c-1"],
    )
    env.deps.journal.append(
        EventDraft(
            type=EventType.LEARNING,
            goal_id="g",
            round=1,
            payload={"learning": item.model_dump(mode="json")},
        )
    )
    await env.deps.journal.append_durable(
        EventDraft(
            type=EventType.DECISION,
            goal_id="g",
            round=1,
            payload={"round": 1, "decision": Decision().model_dump(mode="json")},
        )
    )
    return GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None, journal_r0=False)


@pytest.mark.parametrize("stage", ["constructor", "models", "invoke", "serialize", "traverse"])
@pytest.mark.parametrize("stop", ["stop", "deadline"])
async def test_curator_blocking_hooks_are_abandoned_before_ids(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    stop: str,
) -> None:
    import threading
    from collections.abc import Sequence

    import anyio
    from _fakes.curators import MergeAll

    from swarm_arena.board import Learning

    runner = await _curation_checkpoint(env)
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    finished = anyio.Event()
    before = env.deps.ids.counters()
    original_init = MergeAll.__init__
    original_curate = MergeAll.curate

    def block() -> None:
        entered.set()
        try:
            assert release.wait(5)
        finally:
            exited.set()

    def init(self: MergeAll, *args: Any, **kwargs: Any) -> None:
        block()
        original_init(self, *args, **kwargs)

    def models(self: MergeAll) -> tuple[str, ...]:
        block()
        return ()

    class BlockingLearning(Learning):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            block()
            return super().model_dump(*args, **kwargs)

    class BlockingSequence(Sequence[Learning]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: Any) -> Any:
            block()
            raise IndexError

    async def curate(self: MergeAll, *args: Any, **kwargs: Any) -> Any:
        if stage == "invoke":
            entered.set()
            try:
                await anyio.sleep_forever()
            finally:
                exited.set()
        result = await original_curate(self, *args, **kwargs)
        if stage == "traverse":
            return BlockingSequence()
        if stage == "serialize":
            return [*result, BlockingLearning.model_validate(result[0].model_dump())]
        return result

    if stage == "constructor":
        monkeypatch.setattr(MergeAll, "__init__", init)
    elif stage == "models":
        monkeypatch.setattr(MergeAll, "models", models)
    else:
        monkeypatch.setattr(MergeAll, "curate", curate)

    async def work() -> None:
        try:
            await runner._curate_round(1)
        finally:
            finished.set()

    try:
        with anyio.fail_after(4):
            async with anyio.create_task_group() as group:
                group.start_soon(work)
                assert await anyio.to_thread.run_sync(entered.wait, 2)
                if stop == "stop":
                    env.deps.stops.stop = True
                else:
                    env.deps.config = env.deps.config.model_copy(
                        update={
                            "limits": env.deps.config.limits.model_copy(
                                update={"run_timeout_s": 0.001}
                            )
                        }
                    )
                await finished.wait()
                assert env.deps.ids.counters() == before
                assert env.events("curated") == []
                release.set()
                assert await anyio.to_thread.run_sync(exited.wait, 1)
                # A subsequent healthy core operation proves the abandoned daemon
                # cannot publish into its former goal after returning.
                await runner._refresh()
                assert len(env.events("learning")) == 1
                assert env.deps.ids.counters() == before
    finally:
        release.set()


@pytest.mark.parametrize("stage", ["constructor", "models", "curate", "empty", "malformed"])
async def test_curator_failed_attempt_commits_safe_completion(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    from _fakes.curators import MergeAll

    runner = await _curation_checkpoint(env)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CURATOR_SENTINEL_SECRET")

    async def curate(*args: Any, **kwargs: Any) -> Any:
        if stage == "empty":
            return []
        if stage == "malformed":
            return [object()]
        return fail()

    monkeypatch.setattr(
        MergeAll,
        {"constructor": "__init__", "models": "models"}.get(stage, "curate"),
        fail if stage in {"constructor", "models"} else curate,
    )
    before = env.deps.ids.counters()
    await runner._curate_round(1)
    await runner._curate_round(1)
    receipts = env.events("curated")
    assert len(receipts) == 1
    assert receipts[0].payload["summaries"] == []
    assert bool(receipts[0].payload["errors"]) == (stage != "empty")
    assert "CURATOR_SENTINEL_SECRET" not in repr(receipts)
    assert env.deps.ids.counters() == before


@pytest.mark.parametrize("mode", ["reported", "unknown", "overflow", "spoof"])
async def test_curator_uses_reserve_and_settles_authoritatively(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from _fakes.curators import MergeAll

    from swarm_arena._budget.amount import USD_LIMIT, Amount
    from swarm_arena._budget.ledger import Ledger
    from swarm_arena._budget.llm_scope import current_llm_scope
    from swarm_arena._journal.replay import RunState
    from swarm_arena._model.ids import IdGen
    from swarm_arena.manager import LimitReached

    runner = await _curation_checkpoint(env)
    original = MergeAll.curate

    async def curate(self: MergeAll, *args: Any, **kwargs: Any) -> Any:
        scope = current_llm_scope()
        assert scope is not None
        assert (scope.role, scope.goal_id, scope.round) == ("curator", "g", 1)
        if mode == "spoof":
            raise LimitReached("budget", "CURATOR_SENTINEL_SECRET")
        estimate = Amount(usd=0.01)
        id = await scope.tranche.reserve_call(estimate)
        if mode == "reported":
            await scope.tranche.settle_call(id, Amount(usd=0.000003))
        elif mode == "overflow":
            await scope.tranche.settle_call(id, Amount.model_construct(tokens=2, usd=USD_LIMIT * 2))
        elif mode == "unknown":
            raise RuntimeError("CURATOR_SENTINEL_SECRET")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(MergeAll, "curate", curate)
    await runner._curate_round(1)
    await env.deps.journal.commit()
    reserves = env.events("reserve")
    settlements = env.events("settle")
    assert all(e.payload["reservation"]["holder"] == "g/:curator" for e in reserves)
    assert len(reserves) == len(settlements) == (0 if mode == "spoof" else 1)
    live = env.deps.ledger
    replay = Ledger.replay(
        RunState.from_events(env._events()), journal=env.deps.journal, ids=IdGen(env.deps.run_id, 1)
    )
    assert live.free() == replay.free()
    assert live._charges == replay._charges
    assert all(entry.settled for entry in live._entries.values())
    if mode in {"reported", "unknown"}:
        assert settlements[0].payload["actual"]["usd"] == (0.000003 if mode == "reported" else 0.01)
    if mode == "overflow":
        assert live.run_exhausted()
        assert replay.run_exhausted()
        assert all(not event.payload["summaries"] for event in env.events("curated"))
    if mode == "spoof":
        assert runner.reason is None
    assert "CURATOR_SENTINEL_SECRET" not in repr(env.events("curated"))
    if mode != "overflow":
        assert len(env.events("curated")) == 1


@pytest.mark.parametrize(
    "boundary", ["decision", "prepared", "curated", "round_closed", "terminal_decision"]
)
async def test_curator_recovers_committed_phase_after_process_death(
    tmp_path: Path, boundary: str
) -> None:
    import os
    import sys

    import anyio
    from _fakes.run import all_events, events

    from swarm_arena import RunConfig, Swarm
    from swarm_arena._journal.replay import RunState

    environment = dict(os.environ)
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
    )
    script = (
        "import sys,anyio; from _fakes.curators import crash_driver; "
        "anyio.run(crash_driver,sys.argv[1],sys.argv[2])"
    )
    with anyio.fail_after(60):
        process = await anyio.run_process(
            [sys.executable, "-c", script, str(tmp_path), boundary], env=environment, check=False
        )
    assert process.returncode == 91, process.stderr.decode()
    cfg = RunConfig.load(tmp_path / "runs" / "curation" / "run.toml")
    before = all_events(cfg)
    assert sum(e.type == "curated" for e in before) == int(boundary in {"curated", "round_closed"})
    if boundary == "prepared":
        assert all(
            e.payload["learning"]["author"] != "curator" for e in before if e.type == "learning"
        )
    with anyio.fail_after(60):
        result = await Swarm(cfg).resume("curation", force=True)
    if boundary == "terminal_decision":
        assert not result["g"]
        assert events(cfg, "curated") == []
        assert not (tmp_path / "calls").exists()
        assert len(events(cfg, "tool_start")) == 1
        assert len(events(cfg, "decision")) == 1
        assert len(events(cfg, "round_closed")) == 1
        assert len(events(cfg, "score")) == sum(e.type == "score" for e in before)
        return
    assert result["g"]
    receipts = events(cfg, "curated")
    assert [e.round for e in receipts] == [1, 2]
    assert [e.round for e in events(cfg, "decision")] == [1, 2]
    assert [e.round for e in events(cfg, "round_closed")] == [1, 2]
    assert len(events(cfg, "tool_start")) == 2
    expected_calls = 3 if boundary == "prepared" else 2
    assert len((tmp_path / "calls").read_text().splitlines()) == expected_calls
    state = RunState.from_events(all_events(cfg))
    summaries = [item for item in state.learnings.values() if item.author == "curator"]
    assert len(summaries) == 2
    assert len({item.id for item in summaries}) == 2
    assert all(item.evidence for item in summaries)
    assert state.goals["g"].curated_decisions == [e.payload["decision_seq"] for e in receipts]


async def test_curator_reserve_exhaustion_completes_without_more_paid_work(
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from _fakes.curators import MergeAll
    from _fakes.engine import with_executor
    from _fakes.run import events

    from swarm_arena import Limits, Swarm
    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.llm_scope import current_llm_scope

    async def curate(*args: Any, **kwargs: Any) -> Any:
        scope = current_llm_scope()
        assert scope is not None
        await scope.tranche.reserve_call(Amount(usd=100))
        pytest.fail("unfunded work dispatched")

    monkeypatch.setattr(MergeAll, "curate", curate)
    config = with_executor(
        run_config.model_copy(
            update={"limits": Limits(max_usd=10), "curator": Ref(path="_fakes.curators:MergeAll")}
        ),
        workspace_root=str(tmp_path / "ws"),
    )
    await Swarm(config).run(_curator_plan())
    receipts = events(config, "curated")
    assert len(receipts) == 1
    assert receipts[0].payload["summaries"] == []
    assert receipts[0].payload["errors"] == ["curator limit: budget"]
    assert len(events(config, "tool_start")) == 1
    assert events(config, "goal_finished")[-1].payload["reason"] == "reserve_exhausted"


@pytest.mark.parametrize(
    "phase", ["reserve_admission", "pending_call", "settle_admission", "settle_parent"]
)
async def test_curator_cancellation_retains_settlement_finality(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    import asyncio

    import anyio
    from _fakes.curators import MergeAll

    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.ledger import Ledger
    from swarm_arena._budget.llm_scope import current_llm_scope
    from swarm_arena._journal.replay import RunState
    from swarm_arena._journal.writer import JournalWriter
    from swarm_arena._model.ids import IdGen

    runner = await _curation_checkpoint(env)
    original_wait = JournalWriter._wait_for
    admitted: list[int] = []
    dispatched: list[str] = []
    pending = anyio.Event()
    parent_scope = anyio.CancelScope()

    async def wait(writer: JournalWriter, seq: int) -> None:
        target = "reserve" if phase == "reserve_admission" else "settle"
        if (
            phase != "pending_call"
            and not admitted
            and any(event.seq == seq and event.type == target for event in writer._queue)
        ):
            admitted.append(seq)
            if phase in {"reserve_admission", "settle_parent"}:
                parent_scope.cancel()
            else:
                # Native task cancellation can interrupt a durable settlement even
                # inside an AnyIO shield. Its already-admitted charge stays final.
                task = asyncio.current_task()
                assert task is not None
                asyncio.get_running_loop().call_soon(task.cancel)
        await original_wait(writer, seq)

    async def curate(*args: Any, **kwargs: Any) -> Any:
        scope = current_llm_scope()
        assert scope is not None
        call = await scope.tranche.reserve_call(Amount(usd=0.01))
        dispatched.append(call)
        if phase == "pending_call":
            pending.set()
            await anyio.sleep_forever()
        await scope.tranche.settle_call(call, Amount(usd=0.000003))
        return []

    monkeypatch.setattr(MergeAll, "curate", curate)
    monkeypatch.setattr(JournalWriter, "_wait_for", wait)
    if phase in {"reserve_admission", "settle_parent"}:
        with parent_scope:
            await runner._curate_round(1)
        if phase == "reserve_admission":
            assert not dispatched
    elif phase == "settle_admission":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(runner._curate_round(1))
    else:
        with anyio.fail_after(3):
            async with anyio.create_task_group() as group:
                group.start_soon(runner._curate_round, 1)
                await pending.wait()
                env.deps.stops.stop = True
    await env.deps.journal.commit()
    assert env.events("curated") == []
    assert len(env.events("reserve")) == len(env.events("settle")) == 1
    settlement = env.events("settle")[0]
    assert (
        settlement.payload["actual"]["usd"]
        == {
            "reserve_admission": 0.0,
            "pending_call": 0.01,
            "settle_admission": 0.000003,
            "settle_parent": 0.000003,
        }[phase]
    )
    live = env.deps.ledger
    replay = Ledger.replay(
        RunState.from_events(env._events()), journal=env.deps.journal, ids=IdGen(env.deps.run_id, 1)
    )
    assert live._charges == replay._charges
    assert live.free() == replay.free()
    assert all(entry.settled for entry in live._entries.values())
    seq = env.deps.journal._next_seq
    await live.settle(settlement.payload["id"], Amount())
    assert env.deps.journal._next_seq == seq


async def test_curator_fencing_is_not_swallowed(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.curators import MergeAll
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._errors import Fenced, _core_error

    runner = await _curation_checkpoint(env)
    error = _core_error(Fenced, "ownership lost")
    error.args = ("CURATOR_SENTINEL_SECRET",)
    error.add_note("CURATOR_SENTINEL_SECRET")

    async def curate(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(MergeAll, "curate", curate)
    with pytest.raises(Fenced) as caught:
        await runner._curate_round(1)
    assert_secret_absent(caught.value, "CURATOR_SENTINEL_SECRET")
    assert env.events("curated") == []
    assert env.deps.ids.counters().get("L", 0) == 0


@pytest.mark.parametrize(
    "ref",
    [
        Ref(path="_fakes.curators:Missing"),
        Ref(path="_fakes.curators:MergeAll", config={"unknown": "CURATOR_SENTINEL_SECRET"}),
        Ref(path="swarm_arena:ScriptedManager", config={"requests": []}),
    ],
)
async def test_curator_real_build_failure_is_a_safe_completed_attempt(
    env: EngineEnv, ref: Ref
) -> None:
    runner = await _curation_checkpoint(env)
    env.deps.config = env.deps.config.model_copy(update={"curator": ref})
    await runner._curate_round(1)
    receipt = env.events("curated")[0]
    assert receipt.payload["summaries"] == []
    assert receipt.payload["errors"]
    assert "CURATOR_SENTINEL_SECRET" not in repr(receipt)


@pytest.mark.parametrize("hook", ["models", "curate"])
async def test_curator_hook_descriptors_run_off_loop(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    hook: str,
) -> None:
    import threading

    from _fakes.curators import MergeAll

    runner = await _curation_checkpoint(env)
    original = getattr(MergeAll, hook)

    def getter(self: MergeAll) -> Any:
        assert threading.current_thread() is not threading.main_thread()
        return original.__get__(self, type(self))

    monkeypatch.setattr(MergeAll, hook, property(getter))
    await runner._curate_round(1)
    receipt = env.events("curated")[0]
    assert receipt.payload["errors"] == []
    assert len(receipt.payload["summaries"]) == 1


async def test_curator_does_not_resurrect_a_human_hidden_new_learning(env: EngineEnv) -> None:
    from swarm_arena._board.human_objects import HumanObject, write_human_objects
    from swarm_arena._model.canonical import utc_now
    from swarm_arena._store.base import open_store

    runner = await _curation_checkpoint(env)
    write_human_objects(
        open_store(env.deps.config.arena_root),
        "n",
        "curation",
        [HumanObject(kind="hide", learning_id="L-source", author="human:reviewer")],
        now=utc_now(),
    )
    before = env.deps.ids.counters()
    await runner._curate_round(1)
    assert env.events("curated") == []
    assert len(env.events("learning")) == 1
    assert env.deps.ids.counters() == before


@pytest.mark.parametrize("boundary", ["decision", "curated", "round_closed"])
async def test_curator_recovery_preserves_join_retry_phase(tmp_path: Path, boundary: str) -> None:
    import os
    import sys

    import anyio
    from _engine.test_join_recovery import RetryPlanner
    from _fakes.run import all_events, events

    from swarm_arena import Swarm
    from swarm_arena._journal.replay import RunState

    environment = dict(os.environ)
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
    )
    script = (
        "import sys,anyio; from _fakes.curators import join_crash_driver; "
        "anyio.run(join_crash_driver,sys.argv[1],sys.argv[2])"
    )
    with anyio.fail_after(90):
        process = await anyio.run_process(
            [sys.executable, "-c", script, str(tmp_path), boundary], env=environment, check=False
        )
    assert process.returncode == 91, process.stderr.decode()
    config = RunConfig.load(tmp_path / "runs" / "curation" / "run.toml")
    before = RunState.from_events(all_events(config)).goals["g"]
    assert before.join_retries == 1
    assert before.join_phase == "rounds"
    with anyio.fail_after(90):
        result = await Swarm(config, replanner=RetryPlanner()).resume("curation", force=True)
    assert result["g"]
    state = RunState.from_events(all_events(config)).goals["g"]
    assert state.join_retries == 1
    assert state.pending_replan is None
    assert [event.round for event in events(config, "curated")] == [1, 2]
    assert len((tmp_path / "calls").read_text().splitlines()) == 2
    assert len(events(config, "join_failed")) == 1
    assert len(events(config, "plan_revision")) == 2
    assert [event.round for event in events(config, "round_closed")] == [1, 2]
    assert events(config, "goal_finished")[-1].payload["reason"] is None
