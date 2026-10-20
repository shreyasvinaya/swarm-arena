"""Exercise plan admission, revision authority, and frozen file boundaries."""

import sys
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    FunctionTool,
    Goal,
    PartSpec,
    Plan,
    Ref,
    RoundSpec,
    RunConfig,
    TeamSpec,
)
from swarm_arena._agent.manager import Manager
from swarm_arena._errors import PlanError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog
from swarm_arena._model.goal import FileSet
from swarm_arena._model.run_config import SecretsPolicy
from swarm_arena._planning import validate as validation
from swarm_arena._planning.provenance import ref_sites, stamp
from swarm_arena._planning.validate import (
    admit_revision,
    agent_fingerprint,
    judge_private_paths,
    register_model_check,
    validate_plan,
)
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.files import snapshot_plan
from swarm_arena._tooling.tool import effective_tool_ref

J_A = Ref(path="_fakes.tools:FixedJudge", config={"value": 1}, name="a")
J_B = Ref(path="_fakes.tools:FixedJudge", config={"value": 2}, name="b")
MANAGER = Ref(path="_fakes.tools:NoopManager")
CATALOG_TOOL = Ref(
    path="swarm_arena:ScriptTool",
    config={"command": ["python", "problem/run.py"], "files": ["problem/run.py"]},
)
CATALOG_JUDGE = J_B
CATALOG_CHECK = Ref(path="_fakes.tools:AlwaysPass", name="c2")
CFG_WITH_CATALOG = RunConfig(
    planner_catalog=PlannerCatalog(
        entries=[
            CatalogEntry(
                name="tool",
                kind="tool",
                ref=CATALOG_TOOL.path,
                config=CATALOG_TOOL.config,
                description="run",
            ),
            CatalogEntry(name="manager", kind="manager", ref=MANAGER.path, description="manage"),
            CatalogEntry(
                name="judge", kind="judge", ref=J_B.path, config=J_B.config, description="score"
            ),
            CatalogEntry(name="check", kind="check", ref=CATALOG_CHECK.path, description="check"),
        ]
    )
)


def _plan(**kw: Any) -> Plan:
    """Build distinct ordinary agents; malformed fixtures bypass model validators."""
    teams = []
    for ti in range(kw.get("teams", 1)):
        agents = []
        for ai in range(kw.get("agents_per_team", 1)):
            config = {
                "function": "_fakes.tools:add",
                "timeout_s": kw.get("offset", 10) + (0 if kw.get("same_tool") else ti * 10 + ai),
            }
            config.update(kw.get("tool_config", {}))
            tool = kw.get("tool", Ref(path="swarm_arena:FunctionTool", config=config))
            agents.append(
                AgentSpec(
                    name=kw.get("agent_names", [f"a{i}" for i in range(20)])[ai],
                    rationale="Explore",
                    tool=tool,
                    manager=MANAGER,
                    replicates=kw.get("replicates", 1),
                )
            )
        teams.append(TeamSpec(name=f"t{ti}", rationale="Explore", agents=agents))
    rounds = kw.get("rounds", [])
    if isinstance(rounds, int):
        rounds = [RoundSpec() for _ in range(rounds)]
    plan = Plan(
        goal=Goal(
            id="g",
            namespace="demo",
            inputs=kw.get("goal_inputs", {}),
            files=FileSet(manifest=kw.get("goal_manifest", {})),
        ),
        teams=teams,
        judges=kw.get("judges", [J_A]),
        ranking=kw.get("ranking", "a"),
        rounds=rounds,
    )
    if "rationale" in kw:
        plan = _renamed(plan, team_rationale=kw["rationale"])
    if "converge_agents" in kw or "converge_team_name" in kw:
        team = _plan(agents_per_team=kw.get("converge_agents", 1), offset=99).teams[0]
        team = team.model_copy(update={"name": kw.get("converge_team_name", "join")})
        plan = plan.model_copy(update={"converge": ConvergeSpec(mode="team", team=team)})
    if "base_dir" in kw:
        plan._base_dir = kw["base_dir"]
    return plan


def _renamed(plan: Plan, **changes: Any) -> Plan:
    plan = plan.model_copy(deep=True)
    for i, team in enumerate(plan.teams):
        updates: dict[str, Any] = {}
        if "team_names" in changes:
            updates["name"] = changes["team_names"][i]
        if i == 0:
            updates.update(
                {
                    field: changes[key]
                    for key, field in [("team", "name"), ("team_rationale", "rationale")]
                    if key in changes
                }
            )
        if i in changes.get("team_part", {}):
            updates["part"] = changes["team_part"][i]
        agents = []
        for j, agent in enumerate(team.agents):
            fields: dict[str, Any] = {}
            if "agent_names" in changes:
                fields["name"] = changes["agent_names"][j]
            if i == j == 0:
                fields.update(
                    {
                        field: changes[key]
                        for key, field in [("agent", "name"), ("agent_rationale", "rationale")]
                        if key in changes
                    }
                )
            agents.append(agent.model_copy(update=fields))
        plan.teams[i] = team.model_copy(update={**updates, "agents": agents})
    for i, part in enumerate(plan.parts):
        if "part_names" in changes:
            plan.parts[i] = part.model_copy(update={"name": changes["part_names"][i]})
        elif i == 0 and "part" in changes:
            plan.parts[i] = part.model_copy(update={"name": changes["part"]})
    return plan


def _parts_plan(**kw: Any) -> Plan:
    names = kw.get("parts", ["a"])
    assignments = kw.get("team_parts", names)
    plan = _plan(teams=len(assignments), goal_manifest=kw.get("main_manifest", {}))
    parts = [PartSpec(name=name, goal=Goal(id=name, namespace="demo")) for name in names]
    teams = [
        team.model_copy(update={"part": part})
        for team, part in zip(plan.teams, assignments, strict=True)
    ]
    join = _plan(tool_config={"timeout_s": 100}).teams[0].model_copy(update={"name": "join"})
    return plan.model_copy(
        update={
            "parts": parts,
            "teams": teams,
            "converge": kw.get("converge", ConvergeSpec(mode="team", team=join)),
        }
    )


def _with_part(plan: Plan, name: str, **part_fields: Any) -> Plan:
    agent = AgentSpec(name="new", rationale="Explore", tool=CATALOG_TOOL, manager=MANAGER)
    team = TeamSpec(name=f"team-{name}", rationale="Explore", agents=[agent], part=name)
    return plan.model_copy(
        update={
            "parts": [
                *plan.parts,
                PartSpec(name=name, goal=Goal(id=name, namespace="demo"), **part_fields),
            ],
            "teams": [*plan.teams, team],
        }
    )


def _script_plan(*, env: list[str]) -> Plan:
    return _plan(tool=Ref(path="swarm_arena:ScriptTool", config={"command": ["run"], "env": env}))


def _script_judge_plan(**kw: Any) -> Plan:
    judge = Ref(
        path="swarm_arena:ScriptJudge",
        name="a",
        config={
            "command": ["score"],
            "files": {"globs": [], "manifest": {"judge/score.py": "ab" * 32}},
        },
    )
    return _plan(judges=[judge], **kw)


def _add_agent(plan: Plan, *, path: str, config: dict[str, Any] | None = None) -> Plan:
    agent = AgentSpec(
        name="new", rationale="Explore", tool=Ref(path=path, config=config or {}), manager=MANAGER
    )
    return plan.model_copy(
        update={"teams": [*plan.teams, TeamSpec(name="new", rationale="Explore", agents=[agent])]}
    )


def _add_catalog_agent(plan: Plan, **kw: Any) -> Plan:
    return _add_agent(plan, path=CATALOG_TOOL.path, config={**CATALOG_TOOL.config, **kw})


def errs(plan: Plan, cfg: RunConfig, **kw: Any) -> list[str]:
    with pytest.raises(PlanError) as error:
        validate_plan(plan, cfg, **kw)
    return error.value.errors


@pytest.fixture(autouse=True)
def _clear_model_checks() -> Any:
    validation._MODEL_CHECKS.clear()
    yield
    validation._MODEL_CHECKS.clear()


def test_valid_plan_passes() -> None:
    assert validate_plan(_plan(), RunConfig()) == []


@pytest.mark.parametrize(
    ("plan", "needle"),
    [
        (_plan(teams=5), "max_teams"),
        (_plan(agents_per_team=5), "max_agents_per_team"),
        (_plan(teams=4, agents_per_team=4), "max_agents_per_goal"),
        (_plan(rounds=4), "max_rounds"),
    ],
)
def test_limits(plan: Plan, needle: str) -> None:
    assert any(needle in e for e in errs(plan, RunConfig()))


def test_max_agents_total_counts_other_goals() -> None:
    assert any(
        "max_agents_total" in e
        for e in errs(_plan(agents_per_team=3), RunConfig(), other_active_agents=30)
    )
    assert "max_agents_total: peak 3 agents + 30 active in other goals > 32" in errs(
        _plan(converge_agents=3), RunConfig(), other_active_agents=30
    )
    assert (
        validation._rule_max_agents_total(
            _plan(converge_agents=3), RunConfig(), (), other_active_agents=29
        )
        == []
    )

    assert validate_plan(_plan(converge_agents=3), RunConfig(), other_active_agents=29) == []


def test_bad_or_duplicate_names_rejected() -> None:
    for bad in ["a/b", "A", "x y", "t:1", "", "a\n"]:
        for where, pointer in [
            ("team", "/teams/0/name"),
            ("agent", "/teams/0/agents/0/name"),
            ("part", "/parts/0/name"),
        ]:
            assert any(
                e.startswith(pointer) and "name" in e
                for e in errs(_renamed(_parts_plan(), **{where: bad}), RunConfig())
            ), (where, bad)
    assert any(
        "duplicate team" in e
        for e in errs(_renamed(_plan(teams=2), team_names=["t", "t"]), RunConfig())
    )
    assert any(
        "duplicate agent" in e
        for e in errs(_renamed(_plan(agents_per_team=2), agent_names=["a", "a"]), RunConfig())
    )
    assert any(
        "duplicate part" in e
        for e in errs(_renamed(_parts_plan(parts=["p", "q"]), part_names=["p", "p"]), RunConfig())
    )
    assert any("duplicate team" in e for e in errs(_plan(converge_team_name="t0"), RunConfig()))


def test_invalid_component_config_names_pointer() -> None:
    assert any(
        e.startswith("/teams/0/agents/0/tool")
        for e in errs(_plan(tool_config={"timeout_s": "soon"}), RunConfig())
    )


def test_effective_config_after_overrides_is_validated() -> None:
    assert any(
        "/rounds/0" in e
        for e in errs(_plan(rounds=[RoundSpec(tool_overrides={"timeout_s": "soon"})]), RunConfig())
    )


def test_env_must_be_allowed() -> None:
    assert any(
        "ANTHROPIC_API_KEY" in e for e in errs(_script_plan(env=["ANTHROPIC_API_KEY"]), RunConfig())
    )
    assert (
        validate_plan(
            _script_plan(env=["ANTHROPIC_API_KEY"]),
            RunConfig(secrets=SecretsPolicy(allow=["ANTHROPIC_API_KEY"])),
        )
        == []
    )


@pytest.mark.parametrize(
    ("path", "config"),
    [
        ("swarm_arena:ScriptTool", {"command": ["SECRET_VALUE"]}),
        ("swarm_arena:FunctionTool", {"function": "_fakes.tools:add"}),
    ],
)
def test_catalog_env_still_requires_secrets_allow(path: str, config: dict[str, Any]) -> None:
    ref = Ref(path=path, config={**config, "env": ["API_TOKEN"]})
    r0 = _plan()
    r1 = stamp(_add_agent(r0, path=path, config=ref.config), origin="autonomous", parent=r0)
    cfg = RunConfig(
        planner_catalog=PlannerCatalog(
            entries=[
                *CFG_WITH_CATALOG.planner_catalog.entries,
                CatalogEntry(
                    name="secret-tool", kind="tool", ref=path, config=ref.config, description="run"
                ),
            ]
        )
    )
    errors = errs(r1, cfg, history=[r0])
    assert any("API_TOKEN" in e and "secrets.allow" in e for e in errors)
    assert_secret_absent(PlanError(errors), "SECRET_VALUE")
    assert (
        validate_plan(
            r1, cfg.model_copy(update={"secrets": SecretsPolicy(allow=["API_TOKEN"])}), history=[r0]
        )
        == []
    )


def test_rationale_whitespace_rejected() -> None:
    assert any(
        e.startswith("/teams/0/rationale")
        for e in errs(_renamed(_plan(), team_rationale="  "), RunConfig())
    )
    assert any(
        e.startswith("/teams/0/agents/0/rationale")
        for e in errs(_renamed(_plan(), agent_rationale="\t\n"), RunConfig())
    )


def test_part_without_team_and_unknown_part_rejected() -> None:
    assert any(
        e.startswith("/parts/1") and "part 'b' has no team" in e
        for e in errs(_parts_plan(parts=["a", "b"], team_parts=["a"]), RunConfig())
    )
    assert any(
        e.startswith("/teams/1/part") and "unknown part 'c'" in e
        for e in errs(
            _renamed(_parts_plan(parts=["a"], team_parts=["a", "a"]), team_part={1: "c"}),
            RunConfig(),
        )
    )


def test_select_converge_with_parts_rejected() -> None:
    assert any(
        e.startswith("/converge/mode") and "parts need converge mode 'team'" in e
        for e in errs(_parts_plan(converge=ConvergeSpec(mode="select")), RunConfig())
    )


def test_ranking_must_name_a_judge_and_names_unique() -> None:
    assert any(
        e.startswith("/ranking") and "'nope'" in e
        for e in errs(_plan(judges=[J_A, J_B], ranking="nope"), RunConfig())
    )
    assert any(
        e.startswith("/judges") and "duplicate" in e
        for e in errs(_plan(judges=[J_A, J_A], ranking="a"), RunConfig())
    )


def test_replicates_need_nondeterministic_tool() -> None:
    assert any(
        e.startswith("/teams/0/agents/0/replicates") and "deterministic" in e
        for e in errs(_plan(replicates=2, tool=FunctionTool("_fakes.tools:add")), RunConfig())
    )
    assert (
        validate_plan(
            _plan(replicates=2, tool=FunctionTool("_fakes.tools:add", deterministic=False)),
            RunConfig(),
        )
        == []
    )


def test_frozen_main_goal_key() -> None:
    r0 = _plan()
    r1 = stamp(_plan(goal_inputs={"x": 2}), origin="autonomous", parent=r0)
    assert any("main goal" in e for e in errs(r1, RunConfig(), history=[r0]))


def test_non_human_noncatalog_ref_rejected_and_carried_human_accepted() -> None:
    r0 = _script_judge_plan()
    ok = stamp(_add_catalog_agent(r0), origin="autonomous", parent=r0)
    bad = stamp(_add_agent(r0, path="_fakes.tools:EchoTool"), origin="autonomous", parent=r0)
    assert validate_plan(ok, CFG_WITH_CATALOG, history=[r0]) == []
    assert any("not in the planner catalog" in e for e in errs(bad, CFG_WITH_CATALOG, history=[r0]))


@pytest.mark.parametrize("admit", [False, True])
def test_noncatalog_ref_rejected_before_import(
    admit: bool, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    r0 = _script_judge_plan()
    bad = stamp(_add_agent(r0, path="_fakes.import_canary:Tool"), origin="autonomous", parent=r0)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must authorize before import")

    monkeypatch.setattr(Ref, "load", forbidden)
    if admit:
        with pytest.raises(PlanError, match="not in the planner catalog"):
            admit_revision(bad, CFG_WITH_CATALOG, artifacts, history=[r0])
    else:
        with pytest.raises(PlanError, match="not in the planner catalog"):
            validate_plan(bad, CFG_WITH_CATALOG, history=[r0])
    assert "_fakes.import_canary" not in sys.modules


@pytest.mark.parametrize("mutate", ["delete", "modify"])
def test_catalog_script_tool_valid_after_snapshot_and_in_later_revision(
    tmp_path: Path, artifacts: ArtifactStore, mutate: str
) -> None:
    source = tmp_path / "problem/run.py"
    source.parent.mkdir()
    source.write_text("print(1)")
    r0 = _script_judge_plan(base_dir=tmp_path)
    r1, warnings = admit_revision(
        stamp(_add_catalog_agent(r0), origin="autonomous", parent=r0),
        CFG_WITH_CATALOG,
        artifacts,
        history=[r0],
    )
    h = sha256_hex(source.read_bytes())
    assert warnings == []
    assert r1.teams[-1].agents[-1].tool.config["files"]["manifest"] == {"problem/run.py": h}
    if mutate == "delete":
        source.unlink()
    else:
        source.write_text("different")
    history = [Plan.model_validate_json(p.model_dump_json()) for p in (r0, r1)]
    history[0]._base_dir = tmp_path
    r2, _ = admit_revision(
        stamp(history[1], origin="autonomous", parent=history[1]),
        CFG_WITH_CATALOG,
        artifacts,
        history=history,
    )
    assert r2.teams[-1].agents[-1].tool.config["files"]["manifest"] == {"problem/run.py": h}
    assert r2.base_dir == tmp_path


def test_catalog_chain_and_lexicographic_admitted_and_roundtrip(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    source = tmp_path / "judge.py"
    source.write_text("print(1)")
    member = Ref(
        path="swarm_arena:ScriptJudge",
        config={"command": ["judge"], "files": ["judge.py"], "timeout_s": 12},
    )
    policy = Ref(
        path="swarm_arena:Chain",
        config={"policies": [{"path": "swarm_arena:Patience", "config": {"rounds": 2}}]},
    )
    judge = Ref(
        path="swarm_arena.scoring:Lexicographic",
        name="a",
        config={"judges": [member.model_dump(exclude_none=True)]},
    )
    r0 = stamp(
        _plan(base_dir=tmp_path).model_copy(update={"policy": policy, "judges": [judge]}),
        origin="llm",
        parent=None,
    )
    cfg = RunConfig(
        planner_catalog=PlannerCatalog(
            entries=[
                CatalogEntry(
                    name=f"e{i}",
                    kind=site.kind,
                    ref=site.ref.path,
                    config=site.ref.config,
                    description="component",
                )
                for i, site in enumerate(ref_sites(r0))
            ]
        )
    )
    a0, _ = admit_revision(r0, cfg, artifacts)
    assert "name" not in a0.policy.config["policies"][0]
    assert "name" not in a0.judges[0].config["judges"][0]
    assert type(a0.judges[0].config["judges"][0]["config"]["timeout_s"]) is int
    source.unlink()
    for j0 in [
        Plan.model_validate_json(a0.model_dump_json()),
        Plan.loads(a0.dumps(), base_dir=tmp_path),
    ]:
        j0._base_dir = tmp_path
        a1, _ = admit_revision(
            stamp(j0, origin="autonomous", parent=j0), cfg, artifacts, history=[j0]
        )
        assert a1.policy.config == a0.policy.config
        assert a1.judges == a0.judges
        assert snapshot_plan(a1, artifacts).model_dump() == a1.model_dump()


def test_ever_present_checks_kept() -> None:
    c1 = Ref(path="_fakes.tools:AlwaysPass", name="c1")
    r0 = _plan().model_copy(update={"checks": [c1]})
    r1 = stamp(
        r0.model_copy(update={"checks": [c1, CATALOG_CHECK]}), origin="autonomous", parent=r0
    )
    assert validate_plan(r1, CFG_WITH_CATALOG, history=[r0]) == []
    dropped = stamp(r0, origin="autonomous", parent=r1)
    assert any("check 'c2'" in e for e in errs(dropped, CFG_WITH_CATALOG, history=[r0, r1]))
    changed = stamp(
        r1.model_copy(
            update={"checks": [c1.model_copy(update={"config": {"changed": True}}), CATALOG_CHECK]}
        ),
        origin="autonomous",
        parent=r1,
    )
    assert errs(changed, CFG_WITH_CATALOG, history=[r0, r1])


def test_new_part_must_leave_scoring_unset() -> None:
    r0 = _parts_plan(parts=["a"])
    bad = stamp(_with_part(r0, "b", judges=[CATALOG_JUDGE]), origin="autonomous", parent=r0)
    assert any(
        e.startswith("/parts/1/judges") and "new part" in e
        for e in errs(bad, CFG_WITH_CATALOG, history=[r0])
    )
    ok = stamp(_with_part(r0, "b"), origin="autonomous", parent=r0)
    assert validate_plan(ok, CFG_WITH_CATALOG, history=[r0]) == []


def test_duplicate_fingerprint_capped() -> None:
    plan = _plan(agents_per_team=3, same_tool=True, instructions=["a", "b", "c"])
    assert any("max_agents_per_fingerprint" in e for e in errs(plan, RunConfig()))
    spec = plan.teams[0].agents[0]
    manager = spec.manager.build(Manager)
    assert agent_fingerprint(spec, manager) == agent_fingerprint(
        spec.model_copy(
            update={
                "rationale": "different",
                "tool": spec.tool.model_copy(update={"name": "different"}),
            }
        ),
        manager,
    )
    other = Ref(path=MANAGER.path, config={"models": ["other"]}).build(Manager)
    assert agent_fingerprint(spec, manager) != agent_fingerprint(spec, other)


def test_address_not_reused() -> None:
    r0 = _plan(agent_names=["a", "b"], agents_per_team=2)
    r1 = stamp(
        r0.model_copy(
            update={"teams": [r0.teams[0].model_copy(update={"agents": [r0.teams[0].agents[0]]})]}
        ),
        origin="autonomous",
        parent=r0,
    )
    r2 = stamp(r0, origin="autonomous", parent=r1)
    assert any("address t0/b" in e for e in errs(r2, RunConfig(), history=[r0, r1]))


def test_judge_file_in_goal_rejected() -> None:
    plan = _script_judge_plan(goal_manifest={"judge/score.py": "ab" * 32})
    assert any("judge/score.py" in e for e in errs(plan, RunConfig()))
    assert judge_private_paths(plan) == frozenset({"judge/score.py"})


def test_non_human_part_files_outside_main_manifest_rejected() -> None:
    r0 = _parts_plan(main_manifest={"problem/a.txt": "aa" * 32})
    part = r0.parts[0]
    for manifest in [
        {"judge/hidden/x.npz": "bb" * 32},
        {"problem/a.txt": "bb" * 32},
        {"Problem/a.txt": "aa" * 32},
    ]:
        r1 = stamp(
            r0.model_copy(
                update={
                    "parts": [
                        part.model_copy(
                            update={
                                "goal": part.goal.model_copy(
                                    update={"files": FileSet(manifest=manifest)}
                                )
                            }
                        )
                    ]
                }
            ),
            origin="autonomous",
            parent=r0,
        )
        assert any(next(iter(manifest)) in e for e in errs(r1, RunConfig(), history=[r0]))


def test_all_errors_listed_and_model_check_hook() -> None:
    register_model_check(lambda plan, cfg: (["model error"], ["model warning"]))
    with pytest.raises(PlanError) as err:
        validate_plan(_plan(teams=5, rationale=" "), RunConfig())
    assert len(err.value.errors) == 3
    assert "model error" in err.value.errors
    validation._MODEL_CHECKS.clear()
    register_model_check(lambda plan, cfg: ([], ["model warning"]))
    assert validate_plan(_plan(), RunConfig()) == ["model warning"]


@pytest.mark.parametrize("round_override", [False, True])
@pytest.mark.parametrize("depth", [1, 3])
def test_nested_tool_env_checked_after_overrides(depth: int, round_override: bool) -> None:
    ref = Ref(
        path="swarm_arena:FunctionTool",
        config={"function": "_fakes.tools:add", "env": ["API_TOKEN"]},
    )
    for _ in range(depth):
        ref = Ref(
            path="_fakes.tools:AdmissionTool",
            config={"members": [ref.model_dump(exclude_none=True)]},
        )
    plan = _plan(tool=ref)
    if round_override:
        plan = _plan(tool=Ref(path=ref.path), rounds=[RoundSpec(tool_overrides=ref.config)])
    assert any("API_TOKEN" in e and "/config/members/0" in e for e in errs(plan, RunConfig()))
    assert validate_plan(plan, RunConfig(secrets=SecretsPolicy(allow=["API_TOKEN"]))) == []


@pytest.mark.parametrize("site", ["constructor", "determinism", "files"])
def test_connector_failures_are_value_free_plan_errors(site: str) -> None:
    secret = "admission-secret-c719f1"
    ref = Ref(path="_fakes.tools:AdmissionTool", config={"fail_at": site, "text": secret})
    with pytest.raises(PlanError) as error:
        validate_plan(_plan(tool=ref, replicates=2), RunConfig())
    assert any(
        "/teams/0/agents/0/tool" in item and "RuntimeError" in item for item in error.value.errors
    )
    assert_secret_absent(error.value, secret)


def test_invalid_config_and_goal_key_are_value_free() -> None:
    secret = "admission-invalid-c719f1"
    plan = _plan(tool_config={"timeout_s": secret})
    plan = plan.model_copy(
        update={"goal": plan.goal.model_copy(update={"inputs": {"x": object()}})}
    )
    with pytest.raises(PlanError) as error:
        validate_plan(plan, RunConfig())
    assert any(e.startswith("/goal") for e in error.value.errors)
    assert any("/teams/0/agents/0/tool" in e for e in error.value.errors)
    assert_secret_absent(error.value, secret)


def test_manager_models_failure_and_instruction_independent_fingerprint() -> None:
    plan = _plan(agents_per_team=3, same_tool=True)
    team = plan.teams[0]
    agents = [
        a.model_copy(
            update={
                "manager": Ref(
                    path="_fakes.tools:AdmissionManager", config={"instructions": str(i)}
                )
            }
        )
        for i, a in enumerate(team.agents)
    ]
    plan = plan.model_copy(update={"teams": [team.model_copy(update={"agents": agents})]})
    assert any("max_agents_per_fingerprint" in e for e in errs(plan, RunConfig()))
    secret = "manager-secret-c719f1"
    agents[0] = agents[0].model_copy(
        update={
            "manager": Ref(
                path="_fakes.tools:AdmissionManager", config={"fail": True, "instructions": secret}
            )
        }
    )
    with pytest.raises(PlanError) as error:
        validate_plan(plan, RunConfig())
    assert_secret_absent(error.value, secret)
    assert any("/manager: RuntimeError" in e for e in error.value.errors)


def test_part_globs_resolve_frozen_main_without_disk(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    r0 = _parts_plan(main_manifest={"problem/a.txt": "aa" * 32, "problem/b.txt": "bb" * 32})
    r0._base_dir = tmp_path
    part = r0.parts[0]
    updated = part.model_copy(
        update={"goal": part.goal.model_copy(update={"files": FileSet(globs=["problem/*.txt"])})}
    )
    r1 = stamp(r0.model_copy(update={"parts": [updated]}), origin="autonomous", parent=r0)
    frozen, _ = admit_revision(r1, RunConfig(), artifacts, history=[r0])
    assert frozen.parts[0].goal.files.manifest == r0.goal.files.manifest
    assert frozen.base_dir == tmp_path
    assert r1.parts[0].goal.files.manifest == {}
    missing = updated.model_copy(
        update={
            "goal": updated.goal.model_copy(
                update={"files": FileSet(globs=["problem/*.txt", "missing/*"])}
            )
        }
    )
    bad = r1.model_copy(update={"parts": [missing]})
    with pytest.raises(PlanError) as error:
        admit_revision(bad, RunConfig(), artifacts, history=[r0])
    assert error.value.errors == ["part 'a': glob 'missing/*' matches no file of the main goal"]


@pytest.mark.parametrize("target", ["main", "part", "tool", "round", "converge"])
def test_private_files_are_compared_with_canonical_fold(target: str) -> None:
    private = "judge/Café.py"
    alias = "JUDGE/CAFE\u0301.PY"
    judge = Ref(
        path="swarm_arena:ScriptJudge",
        name="a",
        config={"command": ["score"], "files": {"globs": [], "manifest": {private: "ab" * 32}}},
    )
    plan = _parts_plan().model_copy(update={"judges": [judge]})
    files = FileSet(manifest={alias: "ab" * 32})
    if target == "main":
        plan = plan.model_copy(update={"goal": plan.goal.model_copy(update={"files": files})})
    elif target == "part":
        p = plan.parts[0]
        plan = plan.model_copy(
            update={
                "parts": [p.model_copy(update={"goal": p.goal.model_copy(update={"files": files})})]
            }
        )
    else:
        tool = Ref(path="_fakes.tools:AdmissionTool", config={"files": files.model_dump()})
        if target == "converge":
            team = plan.converge.team
            assert team is not None
            team = team.model_copy(
                update={"agents": [team.agents[0].model_copy(update={"tool": tool})]}
            )
            plan = plan.model_copy(
                update={"converge": plan.converge.model_copy(update={"team": team})}
            )
        else:
            team = plan.teams[0]
            if target == "round":
                plan = plan.model_copy(update={"rounds": [RoundSpec(tool_overrides=tool.config)]})
                tool = Ref(path=tool.path)
            plan = plan.model_copy(
                update={
                    "teams": [
                        team.model_copy(
                            update={"agents": [team.agents[0].model_copy(update={"tool": tool})]}
                        )
                    ]
                }
            )
    assert any(alias in e and "private" in e for e in errs(plan, RunConfig()))
    assert judge_private_paths(plan) == frozenset({private})


def test_tool_root_respelling_rejected_after_round_merge() -> None:
    tool = Ref(path="_fakes.tools:AdmissionTool")
    files = FileSet(manifest={"Problem/a.txt": "aa" * 32})
    plan = _plan(
        tool=tool,
        goal_manifest={"problem/a.txt": "aa" * 32},
        rounds=[RoundSpec(tool_overrides={"files": files.model_dump()})],
    )
    assert any(e.startswith("/rounds/0") for e in errs(plan, RunConfig()))


@pytest.mark.parametrize("field", ["checks", "judges", "ranking"])
def test_retained_part_scoring_is_fixed(field: str) -> None:
    r0 = _parts_plan()
    value = [] if field != "ranking" else "a"
    part = r0.parts[0].model_copy(update={field: value})
    r1 = stamp(r0.model_copy(update={"parts": [part]}), origin="autonomous", parent=r0)
    assert any(
        e.startswith(f"/parts/0/{field}") and "retained" in e
        for e in errs(r1, RunConfig(), history=[r0])
    )


def test_typed_ever_present_check_equality() -> None:
    first = Ref(
        path="swarm_arena:ScriptCheck", config={"command": ["check"], "timeout_s": 1}, name="c"
    )
    changed = first.model_copy(update={"config": {**first.config, "timeout_s": 1.0}})
    r0 = _plan().model_copy(update={"checks": [first]})
    r1 = stamp(r0.model_copy(update={"checks": [changed]}), origin="autonomous", parent=r0)
    cfg = RunConfig(
        planner_catalog=PlannerCatalog(
            entries=[
                CatalogEntry(
                    name="check",
                    kind="check",
                    ref=changed.path,
                    config=changed.config,
                    description="check",
                )
            ]
        )
    )
    assert any("check 'c'" in e and "retained" in e for e in errs(r1, cfg, history=[r0]))


def test_team_and_converge_addresses_do_not_return() -> None:
    r0 = _plan(converge_agents=1)
    r1 = stamp(
        r0.model_copy(update={"teams": [], "converge": ConvergeSpec()}), origin="human", parent=r0
    )
    r2 = stamp(r0, origin="human", parent=r1)
    errors = errs(r2, RunConfig(), history=[r0, r1])
    assert any("address t0 " in e for e in errors)
    assert any("address join " in e for e in errors)


def test_inherited_converge_override_authorized_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r0 = _plan(rounds=[RoundSpec(tool_overrides={"timeout_s": 15})])
    join = (
        _plan(tool=Ref(path="_fakes.tools:AdmissionTool"))
        .teams[0]
        .model_copy(update={"name": "join"})
    )
    r1 = stamp(
        r0.model_copy(update={"converge": ConvergeSpec(mode="team", team=join)}),
        origin="autonomous",
        parent=r0,
    )
    cfg = RunConfig(
        planner_catalog=PlannerCatalog(
            entries=[
                *CFG_WITH_CATALOG.planner_catalog.entries,
                CatalogEntry(
                    name="join", kind="tool", ref="_fakes.tools:AdmissionTool", description="join"
                ),
            ]
        )
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("imported before complete authorization")

    monkeypatch.setattr(Ref, "load", forbidden)
    assert any("/converge/round" in e for e in errs(r1, cfg, history=[r0]))


def test_function_env_cannot_be_round_overridden_even_by_human() -> None:
    plan = _plan(rounds=[RoundSpec(tool_overrides={"env": ["API_TOKEN"]})])
    assert any(
        e.startswith("/rounds/0")
        for e in errs(plan, RunConfig(secrets=SecretsPolicy(allow=["API_TOKEN"])))
    )


def test_default_nested_tool_env_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    from _fakes.tools import AdmissionTool
    from pydantic import create_model

    from swarm_arena._model.ref import RefField

    nested = Ref(
        path="swarm_arena:FunctionTool",
        config={"function": "_fakes.tools:add", "env": ["API_TOKEN"]},
    )
    cfg = create_model("Config", __base__=AdmissionTool.Config, members=(list[RefField], [nested]))
    monkeypatch.setattr(AdmissionTool, "Config", cfg)
    plan = _plan(tool=Ref(path="_fakes.tools:AdmissionTool"))
    assert any("API_TOKEN" in e for e in errs(plan, RunConfig()))


def test_admission_snapshots_effective_nested_refs(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    source = tmp_path / "problem/a.txt"
    source.parent.mkdir()
    source.write_text("original")
    nested = Ref(path="_fakes.tools:AdmissionTool", config={"files": ["problem/a.txt"]})
    tool = Ref(path="_fakes.tools:AdmissionTool")
    plan = _plan(
        tool=tool,
        base_dir=tmp_path,
        rounds=[RoundSpec(tool_overrides={"members": [nested.model_dump(exclude_none=True)]})],
    )
    admitted, _ = admit_revision(plan, RunConfig(), artifacts)
    member = admitted.rounds[0].tool_overrides["members"][0]
    assert member["config"]["files"]["manifest"] == {
        "problem/a.txt": sha256_hex(source.read_bytes())
    }
    source.unlink()
    resumed = Plan.loads(admitted.dumps(), base_dir=tmp_path)
    child, _ = admit_revision(
        stamp(resumed, origin="autonomous", parent=resumed),
        RunConfig(),
        artifacts,
        history=[resumed],
    )
    assert child.rounds == admitted.rounds


def test_admission_validates_overrides_before_snapshot_io(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    tool = Ref(path="_fakes.tools:AdmissionTool", config={"files": ["missing.py"]})
    plan = _plan(
        tool=tool,
        base_dir=tmp_path,
        rounds=[RoundSpec(tool_overrides={"deterministic": "not-bool"})],
    )
    with pytest.raises(PlanError) as error:
        admit_revision(plan, RunConfig(), artifacts)
    assert any(e.startswith("/rounds/0") for e in error.value.errors)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("converge", [False, True])
@pytest.mark.parametrize("alias_kind", ["simple", "choices", "path"])
def test_admission_snapshots_aliased_overrides_after_reload(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    converge: bool,
    alias_kind: str,
) -> None:
    from _fakes.tools import AdmissionTool
    from pydantic import AliasChoices, AliasPath, Field, create_model

    from swarm_arena._model.ref import RefField

    alias: str | AliasChoices | AliasPath = "sources"
    key = "sources"
    raw_files: Any = ["problem/run.py"]
    if alias_kind == "choices":
        alias = AliasChoices("sources", "legacy")
        key = "legacy"
    elif alias_kind == "path":
        alias = AliasPath("payload", 0)
        key = "payload"
        raw_files = [raw_files]
    schema = create_model(
        "Config",
        __base__=AdmissionTool.Config,
        files=(FileSet, Field(default_factory=FileSet, validation_alias=alias)),
        members=(list[RefField], Field(default_factory=list, validation_alias="children")),
    )
    monkeypatch.setattr(AdmissionTool, "Config", schema)
    source = tmp_path / "problem/run.py"
    source.parent.mkdir()
    source.write_text("print(1)")
    expected = {"problem/run.py": sha256_hex(source.read_bytes())}
    overrides: dict[str, Any] = {key: raw_files}
    if nested:
        overrides = {"children": [{"path": "_fakes.tools:AdmissionTool", "config": overrides}]}
    plan = _plan(
        tool=Ref(path="_fakes.tools:AdmissionTool"),
        base_dir=tmp_path,
        rounds=[RoundSpec(tool_overrides=overrides)],
    )
    if converge:
        plan = plan.model_copy(
            update={
                "converge": ConvergeSpec(
                    mode="team", team=plan.teams[0].model_copy(update={"name": "join"})
                )
            }
        )
    admitted, warnings = admit_revision(plan, RunConfig(), artifacts)
    assert warnings == []
    stored = admitted.rounds[0].tool_overrides
    files = (stored["children"][0]["config"] if nested else stored)[key]
    if alias_kind == "path":
        files = files[0]
    assert files["manifest"] == expected
    assert "files" not in stored
    assert "members" not in stored
    source.unlink()
    for restored in (
        Plan.model_validate_json(admitted.model_dump_json()),
        Plan.loads(admitted.dumps(), base_dir=tmp_path),
    ):
        restored._base_dir = tmp_path
        resumed, warnings = admit_revision(restored, RunConfig(), artifacts)
        assert warnings == []
        assert resumed.rounds == admitted.rounds
        effective = effective_tool_ref(
            resumed.teams[0].agents[0].tool, resumed.rounds[0].tool_overrides
        ).build(AdmissionTool)
        if nested:
            effective = effective.config.members[0].build(AdmissionTool)
        assert effective.config.files.snapshotted
        assert effective._own_files() == expected
        assert artifacts.get_bytes(expected["problem/run.py"]) == b"print(1)"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("converge", [False, True])
def test_admission_refuses_unsnapshotted_effective_defaults(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    converge: bool,
) -> None:
    from _fakes.tools import AdmissionTool
    from pydantic import create_model

    from swarm_arena._model.ref import RefField

    secret = "unfrozen-source-sentinel.py"
    fields: dict[str, Any] = {"files": (FileSet, FileSet(globs=[secret]))}
    if nested:
        fields = {
            "members": (
                list[RefField],
                [
                    Ref(
                        path="swarm_arena:ScriptTool",
                        config={"command": ["true"], "files": [secret]},
                    )
                ],
            )
        }
    schema = create_model("Config", __base__=AdmissionTool.Config, **fields)
    monkeypatch.setattr(AdmissionTool, "Config", schema)
    plan = _plan(tool=Ref(path="_fakes.tools:AdmissionTool"), base_dir=tmp_path)
    if converge:
        plan = _plan(base_dir=tmp_path).model_copy(
            update={
                "converge": ConvergeSpec(
                    mode="team", team=plan.teams[0].model_copy(update={"name": "join"})
                )
            }
        )
    with pytest.raises(PlanError) as error:
        admit_revision(plan, RunConfig(), artifacts)
    assert any("unsnapshotted FileSet" in item for item in error.value.errors)
    assert_secret_absent(error.value, secret)


def test_nested_diagnostic_mapping_keys_are_not_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    from _fakes.tools import AdmissionTool
    from pydantic import create_model

    schema = create_model("Config", __base__=AdmissionTool.Config, payload=(dict[str, Any], {}))
    monkeypatch.setattr(AdmissionTool, "Config", schema)
    secret = "data-key-c719f1"
    ref = Ref(
        path="_fakes.tools:AdmissionTool", config={"payload": {secret: {"path": "absent:Tool"}}}
    )
    with pytest.raises(PlanError) as error:
        validate_plan(_plan(tool=ref), RunConfig())
    assert_secret_absent(error.value, secret)


def test_tool_env_in_untyped_nested_ref_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    from _fakes.tools import AdmissionTool
    from pydantic import create_model

    schema = create_model("Config", __base__=AdmissionTool.Config, payload=(dict[str, Any], {}))
    monkeypatch.setattr(AdmissionTool, "Config", schema)
    nested = Ref(
        path="swarm_arena:FunctionTool",
        config={"function": "_fakes.tools:add", "env": ["API_TOKEN"]},
    )
    ref = Ref(
        path="_fakes.tools:AdmissionTool",
        config={"payload": {"child": nested.model_dump(exclude_none=True)}},
    )
    assert any("API_TOKEN" in e for e in errs(_plan(tool=ref), RunConfig()))


@pytest.mark.parametrize("converge", [False, True])
def test_replicates_use_effective_round_determinism(converge: bool) -> None:
    tool = Ref(path="_fakes.tools:AdmissionTool", config={"deterministic": True})
    plan = _plan(
        tool=tool, replicates=2, rounds=[RoundSpec(tool_overrides={"deterministic": False})]
    )
    if converge:
        team = plan.teams[0].model_copy(update={"name": "join"})
        plan = _plan().model_copy(
            update={
                "converge": ConvergeSpec(
                    mode="team", team=team, round=RoundSpec(tool_overrides={"deterministic": False})
                )
            }
        )
    assert validate_plan(plan, RunConfig()) == []


def test_nested_diagnostic_validation_alias_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    from _fakes.tools import AdmissionTool
    from pydantic import Field, create_model

    from swarm_arena._model.ref import RefField

    schema = create_model(
        "Config",
        __base__=AdmissionTool.Config,
        members=(list[RefField], Field(default_factory=list, validation_alias="children")),
    )
    monkeypatch.setattr(AdmissionTool, "Config", schema)
    ref = Ref(path="_fakes.tools:AdmissionTool", config={"children": [{"path": "absent:Tool"}]})
    assert any(
        e.startswith("/teams/0/agents/0/tool/config/children/0:")
        for e in errs(_plan(tool=ref), RunConfig())
    )
