"""Exercise plan admission, revision authority, and frozen file boundaries."""

import contextlib
import sys
import threading
import traceback
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from _fakes.admission import SENTINEL as BOUNDARY_SECRET
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
from swarm_arena._model.base import Origin
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog
from swarm_arena._model.goal import FileSet
from swarm_arena._model.ref import Configurable
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
from swarm_arena._store.base import open_store
from swarm_arena._tooling.files import snapshot_plan, snapshot_ref
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
    before = validation._MODEL_CHECKS[:]
    validation._MODEL_CHECKS.clear()
    yield
    validation._MODEL_CHECKS[:] = before


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


def test_non_human_noncatalog_ref_rejected_and_carried_human_accepted(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    r0 = _script_judge_plan()
    (tmp_path / "problem").mkdir()
    (tmp_path / "problem/run.py").write_text("pass")
    r0._base_dir = tmp_path
    ok = stamp(snapshot_plan(_add_catalog_agent(r0), artifacts), origin="autonomous", parent=r0)
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


def test_new_part_must_leave_scoring_unset(tmp_path: Path, artifacts: ArtifactStore) -> None:
    r0 = _parts_plan(parts=["a"])
    bad = stamp(_with_part(r0, "b", judges=[CATALOG_JUDGE]), origin="autonomous", parent=r0)
    assert any(
        e.startswith("/parts/1/judges") and "new part" in e
        for e in errs(bad, CFG_WITH_CATALOG, history=[r0])
    )
    (tmp_path / "problem").mkdir()
    (tmp_path / "problem/run.py").write_text("pass")
    r0._base_dir = tmp_path
    ok = stamp(snapshot_plan(_with_part(r0, "b"), artifacts), origin="autonomous", parent=r0)
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
    register_model_check(lambda uses, cfg: (["model error"], ["model warning"]))
    with pytest.raises(PlanError) as err:
        validate_plan(_plan(teams=5, rationale=" "), RunConfig())
    assert len(err.value.errors) == 3
    assert "model error" in err.value.errors
    validation._MODEL_CHECKS.clear()
    register_model_check(lambda uses, cfg: ([], ["model warning"]))
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
        plan = _plan(tool=ref, rounds=[RoundSpec(tool_overrides={"text": "round"})])
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
    needle = "unsnapshotted FileSet" if nested else "/snapshot: ValueError"
    assert any(needle in item for item in error.value.errors)
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


BOUNDARY_LEAF = "_fakes.admission:ProbeTool"
BOUNDARY_WRAP = "_fakes.admission:Wrapper"
BOUNDARY_MANAGER = Ref(path="_fakes.tools:NoopManager")
BOUNDARY_JUDGE = Ref(path="_fakes.tools:FixedJudge", config={"value": 1}, name="score")


def _boundary_nested(ref: Ref, depth: int) -> Ref:
    for _ in range(depth):
        ref = Ref(path=BOUNDARY_WRAP, config={"members": [ref.model_dump(exclude_none=True)]})
    return ref


def _boundary_plan(ref: Ref, depth: int, merged: bool) -> Plan:
    root = _boundary_nested(ref, depth)
    tool = Ref(path=root.path) if merged else root
    return Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[
            TeamSpec(
                name="t",
                rationale="Explore",
                agents=[
                    AgentSpec(name="a", rationale="Explore", tool=tool, manager=BOUNDARY_MANAGER)
                ],
            )
        ],
        judges=[BOUNDARY_JUDGE],
        rounds=[RoundSpec(tool_overrides=root.config)] if merged else [],
    )


def _boundary_catalog(
    plan: Plan, leaf_config: dict[str, Any], *, alias_settable: bool = False, allow_env: bool = True
) -> RunConfig:
    entries: list[CatalogEntry] = []
    seen: set[tuple[str, str]] = set()
    for site in ref_sites(plan):
        if site.ref.path in (BOUNDARY_LEAF, "_fakes.admission:OtherTool"):
            continue
        key = (site.kind, site.ref.path)
        if key in seen:
            continue
        seen.add(key)
        settings = {"members": {"type": "array"}} if site.ref.path == BOUNDARY_WRAP else {}
        fixed = {} if site.ref.path == BOUNDARY_WRAP else site.ref.config
        entries.append(
            CatalogEntry(
                name=f"e{len(entries)}",
                kind=site.kind,
                ref=site.ref.path,
                config=fixed,
                settable=settings,
                description="admission fixture",
            )
        )
    settable: dict[str, Any] = {"tune_alias": {"type": "integer", "minimum": 0, "maximum": 10}}
    if alias_settable:
        settable["command_alias"] = {"type": "string"}
        settable["command_path"] = {"type": "array", "items": {"type": "string"}}
    entries.append(
        CatalogEntry(
            name="leaf",
            kind="tool",
            ref=BOUNDARY_LEAF,
            config=leaf_config,
            settable=settable,
            description="leaf",
        )
    )
    if alias_settable:
        # Simulate a copied catalog that bypassed its loader's invariant.
        catalog = PlannerCatalog().model_copy(update={"entries": entries})
        return RunConfig(
            secrets=SecretsPolicy(allow=["REVIEW_TOKEN"] if allow_env else [])
        ).model_copy(update={"planner_catalog": catalog})
    return RunConfig(
        planner_catalog=PlannerCatalog(entries=entries),
        secrets=SecretsPolicy(allow=["REVIEW_TOKEN"] if allow_env else []),
    )


def _boundary_reject(
    plan: Plan, config: RunConfig, needle: str, *, history: list[Plan] | None = None
) -> None:
    with pytest.raises(PlanError) as caught:
        validate_plan(plan, config, history=history or [])
    assert any(needle in e for e in caught.value.errors), caught.value.errors
    assert_secret_absent(caught.value, BOUNDARY_SECRET)
    assert BOUNDARY_SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize(
    "rule", ["noncatalog", "fixed", "human_only", "alias_schema", "env", "files"]
)
def test_paired_rule_matrix(depth: int, merged: bool, rule: str) -> None:
    good_cfg: dict[str, Any] = {}
    bad_cfg: dict[str, Any] = {}
    bad_path = BOUNDARY_LEAF
    needle = "catalog"
    if rule == "noncatalog":
        bad_path = "_fakes.admission:OtherTool"
    elif rule == "fixed":
        good_cfg = {"text": "fixed"}
        bad_cfg = {"text": BOUNDARY_SECRET}
    elif rule == "human_only":
        bad_cfg = {"command": BOUNDARY_SECRET}
        if merged and depth == 0:
            needle = "override sets fixed field"
    elif rule == "alias_schema":
        good_cfg = {"tune_alias": 3}
        bad_cfg = {"tune_alias": BOUNDARY_SECRET}
        needle = "tune_alias"
    elif rule == "env":
        good_cfg = bad_cfg = {"env": ["REVIEW_TOKEN"]}
        needle = "secrets.allow"
    elif rule == "files":
        good_cfg = {"files": {"globs": [BOUNDARY_SECRET], "manifest": {BOUNDARY_SECRET: "aa" * 32}}}
        bad_cfg = {"files": [BOUNDARY_SECRET]}
        needle = "snapshott"
    good = _boundary_plan(Ref(path=BOUNDARY_LEAF, config=good_cfg), depth, merged)
    bad = _boundary_plan(Ref(path=bad_path, config=bad_cfg), depth, merged)
    if rule == "env" and merged:
        good = _boundary_plan(Ref(path=BOUNDARY_LEAF, config=good_cfg), depth, False).model_copy(
            update={"rounds": [RoundSpec(tool_overrides={"tune_alias": 3})]}
        )
        bad = good
    if rule == "files":
        assert validate_plan(good, RunConfig()) == []
        _boundary_reject(bad, RunConfig(), "snapshott")
        return
    cfg_fixed = {k: v for k, v in good_cfg.items() if k != "tune_alias"}
    cfg = _boundary_catalog(good, cfg_fixed)
    if merged and (rule == "env" or (depth == 0 and rule in ("files", "fixed"))):
        history = [good]
        good = stamp(good, origin="autonomous", parent=history[0])
        bad = stamp(bad, origin="autonomous", parent=history[0])
    else:
        history = []
        good = stamp(good, origin="llm", parent=None)
        bad = stamp(bad, origin="llm", parent=None)
    assert validate_plan(good, cfg, history=history) == []
    if rule == "env":
        cfg = cfg.model_copy(update={"secrets": SecretsPolicy()})
    _boundary_reject(bad, cfg, needle, history=history)


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("alias", ["command_alias", "command_path"])
def test_human_only_alias_cannot_be_planner_settable(depth: int, merged: bool, alias: str) -> None:
    good = stamp(_boundary_plan(Ref(path=BOUNDARY_LEAF), depth, merged), origin="llm", parent=None)
    cfg = _boundary_catalog(good, {}, alias_settable=True)
    assert validate_plan(good, cfg) == []
    value: Any = BOUNDARY_SECRET if alias == "command_alias" else [BOUNDARY_SECRET]
    bad = stamp(
        _boundary_plan(Ref(path=BOUNDARY_LEAF, config={alias: value}), depth, merged),
        origin="llm",
        parent=None,
    )
    with pytest.raises(PlanError):
        validate_plan(bad, cfg)


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("merged", [False, True])
def test_mixed_human_then_planner_provenance(depth: int, merged: bool) -> None:
    human = _boundary_plan(
        Ref(path="_fakes.admission:OtherTool", config={"text": "human"}), depth, merged
    )
    cfg = _boundary_catalog(human, {})
    assert validate_plan(human, cfg) == []
    one = stamp(human, origin="autonomous", parent=human)
    two = stamp(one, origin="autonomous", parent=one)
    assert validate_plan(one, cfg, history=[human]) == []
    assert validate_plan(two, cfg, history=[human, one]) == []
    bad = stamp(
        _boundary_plan(
            Ref(path="_fakes.admission:OtherTool", config={"text": BOUNDARY_SECRET}), depth, merged
        ),
        origin="autonomous",
        parent=two,
    )
    _boundary_reject(bad, cfg, "catalog", history=[human, one, two])


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("merged", [False, True])
def test_validator_requires_frozen_tool_files(depth: int, merged: bool) -> None:
    good = _boundary_plan(
        Ref(
            path=BOUNDARY_LEAF,
            config={
                "files": {"globs": [BOUNDARY_SECRET], "manifest": {BOUNDARY_SECRET: "aa" * 32}}
            },
        ),
        depth,
        merged,
    )
    assert validate_plan(good, RunConfig()) == []
    bad = _boundary_plan(
        Ref(path=BOUNDARY_LEAF, config={"files": [BOUNDARY_SECRET]}), depth, merged
    )
    _boundary_reject(bad, RunConfig(), "snapshott")


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("fail_at", ["constructor", "files"])
def test_secrecy_of_connector_exception_surfaces(depth: int, merged: bool, fail_at: str) -> None:
    ref = Ref(path=BOUNDARY_LEAF, config={"text": BOUNDARY_SECRET, "fail_at": fail_at})
    plan = _boundary_plan(ref, depth, merged)
    with pytest.raises(PlanError) as caught:
        validate_plan(plan, RunConfig())
    assert_secret_absent(caught.value, BOUNDARY_SECRET)
    assert BOUNDARY_SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("depth", range(4))
def test_admission_snapshots_judge_defaults(depth: int, tmp_path: Path) -> None:
    leaf = Ref(path="_fakes.admission:FileJudge", config={"value": 1}, name="score")
    for _ in range(depth):
        leaf = Ref(
            path="swarm_arena.scoring:Lexicographic",
            config={"judges": [leaf.model_dump(exclude_none=True)]},
            name="score",
        )
    plan = _boundary_plan(Ref(path=BOUNDARY_LEAF), 0, False).model_copy(update={"judges": [leaf]})
    plan._base_dir = tmp_path
    artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))
    good_leaf = Ref(
        path="_fakes.admission:FileJudge", config={"value": 1, "files": []}, name="score"
    )
    for _ in range(depth):
        good_leaf = Ref(
            path="swarm_arena.scoring:Lexicographic",
            config={"judges": [good_leaf.model_dump(exclude_none=True)]},
            name="score",
        )
    good = plan.model_copy(update={"judges": [good_leaf]})
    assert admit_revision(good, RunConfig(), artifacts)[1] == []
    _boundary_reject(plan, RunConfig(), "unsnapshotted")
    source = tmp_path / BOUNDARY_SECRET
    source.write_text("judge source")
    frozen, warnings = admit_revision(plan, RunConfig(), artifacts)
    assert warnings == []
    assert judge_private_paths(frozen) == frozenset({BOUNDARY_SECRET})
    source.unlink()
    for restored in (
        Plan.model_validate_json(frozen.model_dump_json()),
        Plan.loads(frozen.dumps(), base_dir=tmp_path),
    ):
        restored._base_dir = tmp_path
        assert admit_revision(restored, RunConfig(), artifacts)[1] == []
        assert judge_private_paths(restored) == frozenset({BOUNDARY_SECRET})


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("merged", [False, True])
def test_explicit_snapshot_pair(depth: int, merged: bool, tmp_path: Path) -> None:
    source = tmp_path / "data.txt"
    source.write_text(BOUNDARY_SECRET)
    plan = _boundary_plan(Ref(path=BOUNDARY_LEAF, config={"files": ["data.txt"]}), depth, merged)
    plan._base_dir = tmp_path
    artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))
    frozen, warnings = admit_revision(plan, RunConfig(), artifacts)
    assert warnings == []
    source.unlink()
    for resumed in [
        Plan.model_validate_json(frozen.model_dump_json()),
        Plan.loads(frozen.dumps(), base_dir=tmp_path),
    ]:
        resumed._base_dir = tmp_path
        assert admit_revision(resumed, RunConfig(), artifacts)[1] == []


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("origin", ["llm", "autonomous"])
def test_admission_refuses_human_only_alias_changes(
    depth: int, merged: bool, origin: Origin, tmp_path: Path
) -> None:
    human = _boundary_plan(Ref(path=BOUNDARY_LEAF), depth, merged)
    human._base_dir = tmp_path
    cfg = _boundary_catalog(human, {}, alias_settable=True)
    artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))
    human, _ = admit_revision(human, cfg, artifacts)
    history = [human] if origin == "autonomous" else []
    parent = human if history else None
    good = stamp(human, origin=origin, parent=parent)
    assert admit_revision(good, cfg, artifacts, history=history)[1] == []
    bad = _boundary_plan(
        Ref(path=BOUNDARY_LEAF, config={"command_alias": BOUNDARY_SECRET}), depth, merged
    )
    bad._base_dir = tmp_path
    bad = stamp(bad, origin=origin, parent=parent)
    with pytest.raises(PlanError):
        admit_revision(bad, cfg, artifacts, history=history)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_admission_snapshots_judge_refs_inside_mappings(depth: int, tmp_path: Path) -> None:
    source = tmp_path / "judge.py"
    source.write_text(BOUNDARY_SECRET)
    raw = Ref(
        path="swarm_arena:ScriptJudge",
        config={"command": ["python", "judge.py"], "files": ["judge.py"]},
    )
    frozen_leaf = raw.model_copy(
        update={
            "config": {
                **raw.config,
                "files": {"globs": ["judge.py"], "manifest": {"judge.py": "aa" * 32}},
            }
        }
    )

    def wrap(ref: Ref) -> Ref:
        for _ in range(depth):
            ref = Ref(
                path="_fakes.admission:MapJudge",
                config={"value": 1, "children": {"named": ref.model_dump(exclude_none=True)}},
                name="score",
            )
        return ref

    base = _boundary_plan(Ref(path=BOUNDARY_LEAF), 0, False)
    base._base_dir = tmp_path
    artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))
    assert (
        admit_revision(
            base.model_copy(update={"judges": [wrap(frozen_leaf)]}), RunConfig(), artifacts
        )[1]
        == []
    )
    admitted, warnings = admit_revision(
        base.model_copy(update={"judges": [wrap(raw)]}), RunConfig(), artifacts
    )
    assert warnings == []
    ref = admitted.judges[0]
    for _ in range(depth):
        _, cfg = ref._resolve_config(Configurable)
        ref = cast(Any, cfg).children["named"]
    _, cfg = ref._resolve_config(Configurable)
    assert cast(Any, cfg).files.snapshotted, (
        "admission returned a raw FileSet nested under a judge mapping"
    )
    assert cast(Any, cfg).files.manifest == {"judge.py": sha256_hex(source.read_bytes())}
    assert judge_private_paths(admitted) == frozenset({"judge.py"})
    overlap = admitted.model_copy(
        update={"goal": admitted.goal.model_copy(update={"files": cast(Any, cfg).files})}
    )
    with pytest.raises(PlanError, match="private to a check or judge"):
        admit_revision(overlap, RunConfig(), artifacts)
    source.unlink()
    for restored in (
        Plan.model_validate_json(admitted.model_dump_json()),
        Plan.loads(admitted.dumps(), base_dir=tmp_path),
    ):
        restored._base_dir = tmp_path
        frozen_again, warnings = admit_revision(restored, RunConfig(), artifacts)
        assert warnings == []
        assert frozen_again.judges == admitted.judges
        assert judge_private_paths(frozen_again) == frozenset({"judge.py"})


@pytest.mark.parametrize("merged", [False, True])
def test_determinism_failure_secrecy(merged: bool) -> None:
    p = _boundary_plan(
        Ref(path=BOUNDARY_LEAF, config={"text": BOUNDARY_SECRET, "fail_at": "determinism"}),
        0,
        merged,
    )
    p.teams[0].agents[0] = p.teams[0].agents[0].model_copy(update={"replicates": 2})
    with pytest.raises(PlanError) as caught:
        validate_plan(p, RunConfig())
    assert_secret_absent(caught.value, BOUNDARY_SECRET)
    assert BOUNDARY_SECRET not in "".join(traceback.format_exception(caught.value))


def test_omitted_judge_files_do_not_lose_private_protection(tmp_path: Path) -> None:
    (tmp_path / "judge.py").write_text(BOUNDARY_SECRET)
    judge = Ref(
        path="_fakes.admission:DefaultScriptJudge",
        config={"command": ["python", "judge.py"]},
        name="score",
    )
    plan = _boundary_plan(Ref(path=BOUNDARY_LEAF), 0, False).model_copy(
        update={
            "judges": [judge],
            "goal": Goal(id="g", namespace="n", files=FileSet(globs=["judge.py"])),
        }
    )
    plan._base_dir = tmp_path
    artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))
    explicit = judge.model_copy(update={"config": {**judge.config, "files": ["judge.py"]}})
    with pytest.raises(PlanError, match="private to a check or judge"):
        admit_revision(plan.model_copy(update={"judges": [explicit]}), RunConfig(), artifacts)
    with pytest.raises(PlanError):
        admit_revision(plan, RunConfig(), artifacts)


def test_real_script_tool_command_alias_is_human_only(tmp_path: Path) -> None:
    human = _boundary_plan(Ref(path="_fakes.admission:AliasScriptTool"), 0, False)
    human._base_dir = tmp_path
    cfg = RunConfig().model_copy(
        update={
            "planner_catalog": PlannerCatalog().model_copy(
                update={
                    "entries": [
                        CatalogEntry(
                            name="script",
                            kind="tool",
                            ref="_fakes.admission:AliasScriptTool",
                            settable={"argv": {"type": "array", "items": {"type": "string"}}},
                            description="script",
                        )
                    ]
                }
            )
        }
    )
    artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))
    human, _ = admit_revision(human, cfg, artifacts)
    assert (
        admit_revision(
            stamp(human, origin="autonomous", parent=human), cfg, artifacts, history=[human]
        )[1]
        == []
    )
    bad = human.model_copy(deep=True)
    bad.teams[0].agents[0] = (
        bad.teams[0]
        .agents[0]
        .model_copy(
            update={
                "tool": Ref(
                    path="_fakes.admission:AliasScriptTool",
                    config={"argv": ["echo", BOUNDARY_SECRET]},
                )
            }
        )
    )
    with pytest.raises(PlanError):
        admit_revision(
            stamp(bad, origin="autonomous", parent=human), cfg, artifacts, history=[human]
        )


@pytest.mark.parametrize(
    "root",
    [
        "tool",
        "manager",
        "check",
        "judge",
        "policy",
        "part-check",
        "part-judge",
        "converge-tool",
        "converge-manager",
    ],
)
@pytest.mark.parametrize("default", [False, True])
def test_frozen_files_required_at_every_component_root(
    root: str,
    default: bool,
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import create_model

    from swarm_arena._model.ref import RefField

    plan = _parts_plan()
    owner: Any = plan
    field = "policy"
    if root in ("tool", "manager"):
        owner = plan.teams[0].agents[0]
        field = root
    elif root.startswith("converge-"):
        assert plan.converge.team is not None
        owner = plan.converge.team.agents[0]
        field = root.removeprefix("converge-")
    elif root == "part-judge":
        owner = plan.parts[0]
        field = "judges"
    elif root == "part-check":
        owner = plan.parts[0]
        field = "checks"
    elif root in ("check", "judge"):
        field = root + "s"
    refs = getattr(owner, field)
    if field == "checks":
        ref = Ref(path="_fakes.tools:AlwaysPass", name="check")
    elif field == "judges":
        ref = J_A
    else:
        ref = refs
    cls = ref.load()
    assert issubclass(cls, Configurable)
    if default:
        # Non-FileSet defaults stay omitted, so their nested unfrozen files
        # must be caught by the final shared admission check.
        child = Ref(path="_fakes.admission:FileJudge", config={"value": 1})
        schema = create_model("Config", __base__=cls.Config, hidden=(RefField, child))
    else:
        schema = create_model("Config", __base__=cls.Config, files=(FileSet, FileSet()))
        ref = ref.model_copy(update={"config": {**ref.config, "files": [BOUNDARY_SECRET]}})
    monkeypatch.setattr(cls, "Config", schema)
    object.__setattr__(owner, field, [ref] if field in ("judges", "checks") else ref)
    plan._base_dir = tmp_path
    _boundary_reject(plan, RunConfig(), "unsnapshotted")
    if default:
        with pytest.raises(PlanError, match="unsnapshotted") as error:
            admit_revision(plan, RunConfig(), artifacts)
        assert_secret_absent(error.value, BOUNDARY_SECRET)
    else:
        source = tmp_path / BOUNDARY_SECRET
        source.write_text("frozen source")
        frozen, warnings = admit_revision(plan, RunConfig(), artifacts)
        assert warnings == []
        source.unlink()
        for restored in (
            Plan.model_validate_json(frozen.model_dump_json()),
            Plan.loads(frozen.dumps(), base_dir=tmp_path),
        ):
            restored._base_dir = tmp_path
            assert validate_plan(restored, RunConfig()) == []
            assert admit_revision(restored, RunConfig(), artifacts)[1] == []


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("origin", ["llm", "autonomous"])
def test_script_command_alias_protected_at_every_depth(
    depth: int,
    merged: bool,
    origin: Origin,
    tmp_path: Path,
    artifacts: ArtifactStore,
) -> None:
    path = "_fakes.admission:AliasScriptTool"
    human = _boundary_plan(Ref(path=path), depth, merged)
    human._base_dir = tmp_path
    config = _boundary_catalog(human, {})
    human, _ = admit_revision(human, config, artifacts)
    history = [human] if origin == "autonomous" else []
    parent = human if history else None
    good = stamp(human, origin=origin, parent=parent)
    assert admit_revision(good, config, artifacts, history=history)[1] == []
    # Exercise admission independently of the catalog loader, which now refuses
    # this unsafe schema before a planner can run.
    entries = [
        entry.model_copy(
            update={"settable": {"argv": {"type": "array", "items": {"type": "string"}}}}
        )
        if entry.ref == path
        else entry
        for entry in config.planner_catalog.entries
    ]
    unsafe = config.model_copy(
        update={"planner_catalog": config.planner_catalog.model_copy(update={"entries": entries})}
    )
    bad = _boundary_plan(Ref(path=path, config={"argv": ["echo", BOUNDARY_SECRET]}), depth, merged)
    bad._base_dir = tmp_path
    bad = stamp(bad, origin=origin, parent=parent)
    with pytest.raises(PlanError) as error:
        validate_plan(bad, unsafe, history=history)
    assert_secret_absent(error.value, BOUNDARY_SECRET)
    with pytest.raises(PlanError) as error:
        admit_revision(bad, unsafe, artifacts, history=history)
    assert_secret_absent(error.value, BOUNDARY_SECRET)


@pytest.mark.parametrize("source_state", ["unchanged", "modified", "deleted"])
def test_catalog_omitted_fileset_default_is_frozen_and_private(
    source_state: str, tmp_path: Path, artifacts: ArtifactStore
) -> None:
    source = tmp_path / "judge.py"
    source.write_text("private judge source")
    judge = Ref(
        path="_fakes.admission:DefaultScriptJudge",
        config={"command": ["python", "judge.py"]},
        name="score",
    )
    raw = _boundary_plan(Ref(path=BOUNDARY_LEAF), 0, False).model_copy(update={"judges": [judge]})
    raw._base_dir = tmp_path
    config = _boundary_catalog(raw, {})
    raw = stamp(raw, origin="llm", parent=None)
    frozen, warnings = admit_revision(raw, config, artifacts)
    assert warnings == []
    assert frozen.judges[0].config["files"] == {
        "globs": ["judge.py"],
        "manifest": {"judge.py": sha256_hex(source.read_bytes())},
    }
    assert judge_private_paths(frozen) == frozenset({"judge.py"})
    overlap = frozen.model_copy(
        update={
            "goal": frozen.goal.model_copy(
                update={"files": FileSet.model_validate(frozen.judges[0].config["files"])}
            )
        }
    )
    with pytest.raises(PlanError, match="private to a check or judge"):
        validate_plan(overlap, config)
    if source_state == "modified":
        source.write_text("updated human judge source")
    elif source_state == "deleted":
        source.unlink()
    for restored in (
        Plan.model_validate_json(frozen.model_dump_json()),
        Plan.loads(frozen.dumps(), base_dir=tmp_path),
    ):
        restored._base_dir = tmp_path
        assert validate_plan(restored, config) == []
        if source_state == "deleted":
            with pytest.raises(PlanError) as error:
                admit_revision(restored, config, artifacts)
            assert error.value.errors == ["/snapshot: ValueError"]
            human = stamp(restored, origin="human", parent=None)
            vouched, warnings = admit_revision(human, config, artifacts)
            assert warnings == []
            assert vouched.judges == frozen.judges
        else:
            readmitted, warnings = admit_revision(restored, config, artifacts)
            assert warnings == []
            assert readmitted.judges[0].config["files"] == {
                "globs": ["judge.py"],
                "manifest": {"judge.py": sha256_hex(source.read_bytes())},
            }
            assert judge_private_paths(readmitted) == frozenset({"judge.py"})
        revision = stamp(restored, origin="autonomous", parent=restored)
        carried, warnings = admit_revision(revision, config, artifacts, history=[restored])
        assert warnings == []
        assert carried.judges == frozen.judges
        assert judge_private_paths(carried) == frozenset({"judge.py"})


@pytest.mark.parametrize("source_exists", [False, True])
@pytest.mark.parametrize("origin", ["llm", "autonomous"])
def test_catalog_default_forged_manifest_uses_only_human_source(
    source_exists: bool, origin: Origin, tmp_path: Path, artifacts: ArtifactStore
) -> None:
    source = tmp_path / "judge.py"
    if source_exists:
        source.write_bytes(b"real human judge source")
    forged_sha = artifacts.put_bytes(BOUNDARY_SECRET.encode())
    judge = Ref(
        path="_fakes.admission:DefaultScriptJudge",
        config={"command": ["python", "judge.py"]},
        name="score",
    )
    raw = _boundary_plan(Ref(path=BOUNDARY_LEAF), 0, False).model_copy(update={"judges": [judge]})
    raw._base_dir = tmp_path
    config = _boundary_catalog(raw, {})
    forged_files = {"globs": ["judge.py"], "manifest": {"judge.py": forged_sha}}
    forged = judge.model_copy(update={"config": {**judge.config, "files": forged_files}})
    plan = stamp(raw.model_copy(update={"judges": [forged]}), origin=origin, parent=None)
    if source_exists:
        admitted, warnings = admit_revision(plan, config, artifacts)
        assert warnings == []
        manifest = admitted.judges[0].config["files"]["manifest"]
        assert manifest == {"judge.py": sha256_hex(source.read_bytes())}
        assert manifest["judge.py"] != forged_sha
        assert artifacts.get_bytes(manifest["judge.py"]) == source.read_bytes()
        assert validate_plan(admitted, config) == []
    else:
        with pytest.raises(PlanError) as error:
            admit_revision(plan, config, artifacts)
        assert error.value.errors == ["/snapshot: ValueError"]
        assert_secret_absent(error.value, BOUNDARY_SECRET)
        assert_secret_absent(error.value, forged_sha)
    assert plan.judges[0].config["files"] == forged_files


@pytest.mark.parametrize("field", ["files", "unrelated"])
@pytest.mark.parametrize("post_snapshot", [False, True])
def test_catalog_omitted_default_does_not_authorize_other_values(
    field: str,
    post_snapshot: bool,
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = Ref(
        path="_fakes.admission:DefaultScriptJudge",
        config={"command": ["python", "judge.py"]},
        name="score",
    )
    plan = _boundary_plan(Ref(path=BOUNDARY_LEAF), 0, False).model_copy(update={"judges": [judge]})
    config = _boundary_catalog(plan, {})
    value: Any = [BOUNDARY_SECRET]
    if field == "files" and post_snapshot:
        value = {"globs": [BOUNDARY_SECRET], "manifest": {BOUNDARY_SECRET: "aa" * 32}}
    changed = judge.model_copy(update={"config": {**judge.config, field: value}})
    plan = stamp(plan.model_copy(update={"judges": [changed]}), origin="llm", parent=None)
    if post_snapshot:
        with pytest.raises(PlanError, match="not in the planner catalog") as caught:
            validate_plan(plan, config)
        assert_secret_absent(caught.value, BOUNDARY_SECRET)
    else:
        deferred: list[Any] = []
        _, errors = validation.authorize_revision(plan, config, deferred=deferred)
        assert errors == []
        assert deferred
        plan._base_dir = tmp_path

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("snapshot read before default authorization")

        monkeypatch.setattr(validation, "_resolve_revision", forbidden)
        monkeypatch.setattr(validation, "snapshot_plan", forbidden)
        with pytest.raises(PlanError) as caught:
            admit_revision(plan, config, artifacts)
        assert caught.value.errors == ["/judges/0: field '<field>' is not in the planner catalog"]
        assert_secret_absent(caught.value, BOUNDARY_SECRET)


def test_snapshot_preserves_unrelated_normalized_list(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    ref = Ref(path="_fakes.admission:NormalizeTool", config={"labels": ["one", "one"]})
    plan = _boundary_plan(ref, 0, False)
    plan._base_dir = tmp_path
    assert validate_plan(plan, RunConfig()) == []
    _, config = ref._resolve_config(Configurable)
    assert cast(Any, config).labels == ["one"]
    assert snapshot_ref(ref, tmp_path, artifacts).config == ref.config
    admitted, warnings = admit_revision(plan, RunConfig(), artifacts)
    assert warnings == []
    assert admitted.teams[0].agents[0].tool.config["labels"] == ["one", "one"]


def test_snapshot_preserves_unrelated_normalized_mapping(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    metadata = {"Key": "original", "key": "changed"}
    ref = Ref(path="_fakes.admission:NormalizeTool", config={"metadata": metadata})
    plan = _boundary_plan(ref, 0, False)
    plan._base_dir = tmp_path
    _, config = ref._resolve_config(Configurable)
    assert cast(Any, config).metadata == {"key": "changed"}
    assert snapshot_ref(ref, tmp_path, artifacts).config["metadata"] == metadata
    admitted, warnings = admit_revision(plan, RunConfig(), artifacts)
    assert warnings == []
    assert admitted.teams[0].agents[0].tool.config["metadata"] == metadata


def test_snapshot_refuses_shape_change_in_reference_list(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    member = {"path": BOUNDARY_LEAF, "config": {"text": BOUNDARY_SECRET}}
    ref = Ref(path="_fakes.admission:ShrinkingRefsTool", config={"members": [member, member]})
    plan = _boundary_plan(ref, 0, False)
    plan._base_dir = tmp_path
    for action in (
        lambda: snapshot_ref(ref, tmp_path, artifacts),
        lambda: admit_revision(plan, RunConfig(), artifacts),
    ):
        with pytest.raises(PlanError) as caught:
            action()
        assert caught.value.errors == [
            "field 'members': validation changed the shape of a file or reference list"
        ]
        assert_secret_absent(caught.value, BOUNDARY_SECRET)


def test_snapshot_heterogeneous_container_preserves_unrelated_list(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.admission import ProbeTool
    from pydantic import create_model, field_validator

    from swarm_arena._model.ref import RefField

    def normalize(value: tuple[list[str], Ref]) -> tuple[list[str], Ref]:
        return list(dict.fromkeys(value[0])), value[1]

    schema = create_model(
        "Config",
        __base__=ProbeTool.Config,
        payload=(tuple[list[str], RefField], ...),
        __validators__={"normalize": field_validator("payload")(normalize)},
    )
    monkeypatch.setattr(ProbeTool, "Config", schema)
    (tmp_path / "judge.py").write_text("private")
    child = {"path": "_fakes.admission:DefaultScriptJudge", "config": {"command": ["score"]}}
    ref = Ref(path=BOUNDARY_LEAF, config={"payload": [["one", "one"], child]})
    frozen = snapshot_ref(ref, tmp_path, artifacts)
    assert frozen.config["payload"][0] == ["one", "one"]
    assert frozen.config["payload"][1]["config"]["files"]["manifest"] == {
        "judge.py": sha256_hex(b"private")
    }


def test_snapshot_refuses_shape_change_in_reference_mapping(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.admission import MapJudge
    from pydantic import create_model, field_validator

    def normalize(value: dict[str, Ref]) -> dict[str, Ref]:
        return {"normalized": next(iter(value.values()))}

    schema = create_model(
        "Config",
        __base__=MapJudge.Config,
        __validators__={"normalize": field_validator("children")(normalize)},
    )
    monkeypatch.setattr(MapJudge, "Config", schema)
    ref = Ref(
        path="_fakes.admission:MapJudge",
        config={"value": 1, "children": {BOUNDARY_SECRET: {"path": "_fakes.tools:FixedJudge"}}},
    )
    with pytest.raises(PlanError) as caught:
        snapshot_ref(ref, tmp_path, artifacts)
    assert caught.value.errors == [
        "field 'children': validation changed the shape of a file or reference mapping"
    ]
    assert_secret_absent(caught.value, BOUNDARY_SECRET)


@pytest.mark.parametrize("position", [0, 1])
def test_snapshot_union_ref_positions_survive_deleted_source(
    position: int, tmp_path: Path, artifacts: ArtifactStore
) -> None:
    source = tmp_path / "judge.py"
    source.write_text("private")
    child = {"path": "_fakes.admission:DefaultScriptJudge", "config": {"command": ["score"]}}
    payload: list[Any] = ["label", "label"]
    payload[position] = child
    ref = Ref(path="_fakes.admission:UnionRefsTool", config={"payload": payload})
    frozen_ref = snapshot_ref(ref, tmp_path, artifacts)
    expected = {"globs": ["judge.py"], "manifest": {"judge.py": sha256_hex(b"private")}}
    assert frozen_ref.config["payload"][position]["config"]["files"] == expected
    assert frozen_ref.config["payload"][1 - position] == "label"
    plan = _boundary_plan(ref, 0, False)
    plan._base_dir = tmp_path
    admitted, warnings = admit_revision(plan, RunConfig(), artifacts)
    assert warnings == []
    assert admitted.teams[0].agents[0].tool == frozen_ref
    source.unlink()
    for restored in (
        Plan.model_validate_json(admitted.model_dump_json()),
        Plan.loads(admitted.dumps(), base_dir=tmp_path),
    ):
        restored._base_dir = tmp_path
        readmitted, warnings = admit_revision(restored, RunConfig(), artifacts)
        assert warnings == []
        assert readmitted.teams[0].agents[0].tool == frozen_ref


@pytest.mark.anyio
async def test_admitted_default_tool_retains_identity_at_consumers(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    from dataclasses import fields
    from unittest.mock import MagicMock

    from swarm_arena._agent.manager import Manager
    from swarm_arena._agent.watchdog import Watchdog
    from swarm_arena._board.base import BoardView
    from swarm_arena._engine.manifest import build_manifest
    from swarm_arena._engine.round import AgentSlot, EngineDeps, RoundRunner
    from swarm_arena._model.board_types import Frontier
    from swarm_arena._model.results import AgentStatus
    from swarm_arena._model.tooling import ParentRef
    from swarm_arena._planning.catalog_check import find_entry
    from swarm_arena._tooling.staging import fixed_config, plan_tool_stage, tool_identity

    (tmp_path / "tool.py").write_text("tool source")
    ref = Ref(
        path="_fakes.admission:DefaultFilesTool",
        config={"command": ["python", "tool.py"], "editable": ["work.txt"]},
    )
    raw = _boundary_plan(ref, 0, False)
    config = _boundary_catalog(raw, {})
    entries = [
        entry.model_copy(update={"settable": {"tuning": {"type": "integer"}}})
        if entry.ref == ref.path
        else entry
        for entry in config.planner_catalog.entries
    ]
    config = config.model_copy(update={"planner_catalog": PlannerCatalog(entries=entries)})
    entry = next(entry for entry in entries if entry.ref == ref.path)
    plans: list[Plan] = []
    runtimes = []
    for tuning in (1, 2):
        variant = Ref(path=ref.path, config={**ref.config, "tuning": tuning})
        plan = stamp(_boundary_plan(variant, 0, False), origin="llm", parent=None)
        plan._base_dir = tmp_path
        admitted, warnings = admit_revision(plan, config, artifacts)
        assert warnings == []
        plans.append(admitted)
        team = admitted.teams[0]
        spec = team.agents[0]
        assert find_entry(spec.tool, "tool", config.planner_catalog) is entry
        assert "tuning" not in fixed_config(spec.tool, entry)
        slot = AgentSlot(
            address=f"{admitted.goal.id}/{team.name}/{spec.name}",
            team=team,
            spec=spec,
            manager=spec.manager.build(Manager),
            tool_ref=spec.tool,
            goal=admitted.goal,
            goal_key=admitted.goal.key(),
            part=None,
            notes="",
            watchdog=Watchdog(config.limits),
            status=AgentStatus.ACTIVE,
        )
        frontier = Frontier()
        view = MagicMock(spec=BoardView)
        view.goal = slot.goal_key
        view.frontier = frontier
        view.leaderboard.return_value = []
        view.learnings.return_value = []
        deps = MagicMock(
            spec=EngineDeps, **{field.name: MagicMock() for field in fields(EngineDeps)}
        )
        deps.config = config
        deps.run_id = "identity-run"
        deps.board.snapshot.return_value = view
        runner = RoundRunner(
            deps,
            goal_id=admitted.goal.id,
            plan=admitted,
            agents=[slot],
            round_spec=RoundSpec(),
            rounds_left=1,
        )
        runtime = await runner._runtime(slot, 1, frontier, frozenset(), {})
        assert runtime.catalog_entry is entry
        assert runtime.tool_identity == tool_identity(spec.tool, entry)
        runtimes.append(runtime)
    assert runtimes[0].tool_identity == runtimes[1].tool_identity
    manifest = build_manifest(run_id="identity-run", config=config, plans=plans, capacity=None)
    assert len(manifest["tools"]) == 1
    assert manifest["tools"][0]["identity"] == runtimes[0].tool_identity
    digest = artifacts.put_bytes(b"parent edit")
    stage, overlay, warnings = plan_tool_stage(
        goal_files={},
        tool_files=runtimes[1].tool._own_files(),
        editable=["work.txt"],
        protected=frozenset({"tool.py"}),
        private_paths=frozenset(),
        parents=[
            ParentRef(
                candidate_id="c1",
                run_id="identity-run",
                tool_identity=runtimes[0].tool_identity,
                overlay={"work.txt": digest},
            )
        ],
        receiving_identity=runtimes[1].tool_identity,
        edits={},
        request_json=None,
    )
    assert warnings == []
    assert overlay == {"work.txt": digest}
    assert stage.root["work.txt"] == digest


def _model_plan(alias: str) -> Plan:
    plan = _plan()
    plan.teams[0].agents[0] = (
        plan.teams[0]
        .agents[0]
        .model_copy(
            update={"manager": Ref(path="_fakes.tools:NoopManager", config={"models": [alias]})}
        )
    )
    return plan


def test_llm_alias_unknown_or_wrong_role_rejected() -> None:
    from swarm_arena.config import ModelCatalog, ModelSpec

    for config in [
        RunConfig(),
        RunConfig(models=ModelCatalog({"x": ModelSpec(model="fake/m", roles=["judge"])})),
    ]:
        with pytest.raises(PlanError) as error:
            validate_plan(_model_plan("x"), config)
        assert any(
            "alias 'x'" in message and "role 'manager'" in message for message in error.value.errors
        )


def test_max_tokens_required_when_llm_component_used() -> None:
    from swarm_arena import Limits
    from swarm_arena.config import ModelCatalog, ModelSpec

    cfg = RunConfig(
        limits=Limits(max_tokens=None),
        models=ModelCatalog({"m": ModelSpec(model="fake/m", roles=["manager"])}),
    )
    assert any("max_tokens" in e for e in errs(_model_plan("m"), cfg))


def test_unpriced_model_rejected_when_max_usd_set() -> None:
    from swarm_arena import Limits
    from swarm_arena.config import ModelCatalog, ModelSpec

    cfg = RunConfig(
        limits=Limits(max_usd=1),
        models=ModelCatalog({"m": ModelSpec(model="fake/unpriced", roles=["manager"])}),
    )
    assert any("no known price" in e for e in errs(_model_plan("m"), cfg))


def test_check_models_runs_registered_checks_on_a_role_map() -> None:
    from swarm_arena._planning.validate import check_models
    from swarm_arena.config import ModelCatalog, ModelSpec

    seen = []
    register_model_check(lambda uses, config: (seen.append(uses) or [], ["warning"]))
    cfg = RunConfig(
        limits=RunConfig().limits.model_copy(update={"max_tokens": 1000}),
        models=ModelCatalog({"p": ModelSpec(model="fake/p", roles=["planner"])}),
    )
    assert check_models({"planner": ["p"]}, cfg) == ([], ["warning"])
    assert seen == [{"planner": ["p"]}]


def test_fresh_process_first_validation_runs_model_checks() -> None:
    import os
    import subprocess
    import sys

    pytest.importorskip("litellm")
    plan = _model_plan("r1")
    plan.teams[0].agents[0] = (
        plan.teams[0]
        .agents[0]
        .model_copy(
            update={
                "manager": Ref(path="_fakes.llm_components:AliasManager", config={"model": "r1"})
            }
        )
    )
    script = (
        "from swarm_arena import Plan, RunConfig, validate_plan\n"
        "from swarm_arena.config import ModelCatalog, ModelSpec\n"
        f"plan = Plan.model_validate_json({plan.model_dump_json()!r})\n"
        "config = RunConfig(models=ModelCatalog({'r1': ModelSpec("
        "model='deepseek/deepseek-reasoner', roles=['manager'])}))\n"
        "validate_plan(plan, config)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30
    )
    assert proc.returncode != 0
    assert "function calling" in proc.stderr


@pytest.mark.parametrize("site", ["manager", "judge", "nested member of a Lexicographic judge"])
def test_rejected_ref_is_never_imported_or_built(
    site: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    (tmp_path / "evil_mod.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'imported')!r}).touch()\n"
        "class X:\n    def __init__(self):\n"
        f"        Path({str(tmp_path / 'built')!r}).touch()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    r0 = _plan()
    rev = r0.model_copy(deep=True)
    evil = Ref(path="evil_mod:X", name="evil")
    if site == "manager":
        rev.teams[0].agents[0] = rev.teams[0].agents[0].model_copy(update={"manager": evil})
    else:
        if site != "judge":
            evil = Ref(
                path="swarm_arena.scoring:Lexicographic",
                name="evil",
                config={"judges": [evil.model_dump(exclude_none=True)]},
            )
        rev = rev.model_copy(update={"judges": [*rev.judges, evil]})
    rev = stamp(rev, origin="autonomous", parent=r0)
    cfg = RunConfig()
    if site == "nested member of a Lexicographic judge":
        cfg = RunConfig(
            planner_catalog=PlannerCatalog(
                entries=[
                    CatalogEntry(
                        name="panel",
                        kind="judge",
                        ref=evil.path,
                        config=evil.config,
                        description="Panel",
                    )
                ]
            )
        )
    with pytest.raises(PlanError) as error:
        validate_plan(rev, cfg, history=[r0])
    assert any("not in the planner catalog" in m for m in error.value.errors)
    if site == "nested member of a Lexicographic judge":
        assert any("/config/judges/0" in m for m in error.value.errors)
    assert not (tmp_path / "imported").exists()
    assert not (tmp_path / "built").exists()
    assert "evil_mod" not in sys.modules


def test_model_hooks_and_prices_hide_exception_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena._budget import estimate
    from swarm_arena._planning.validate import check_models
    from swarm_arena.config import ModelCatalog, ModelSpec

    def fail(*args: object) -> Any:
        raise RuntimeError(BOUNDARY_SECRET)

    cfg = RunConfig(
        models=ModelCatalog({"m": ModelSpec(model="fake/m", roles=["manager"])}),
        limits=RunConfig().limits.model_copy(update={"max_tokens": 1000, "max_usd": 1}),
    )
    monkeypatch.setattr(estimate, "_price_sources", [fail])
    register_model_check(fail)
    errors, warnings = check_models({"manager": ["m"]}, cfg)
    assert warnings == []
    assert errors == ["/models: RuntimeError", "/models: RuntimeError"]
    with pytest.raises(PlanError) as error:
        validate_plan(_model_plan("m"), cfg)
    assert_secret_absent(error.value, BOUNDARY_SECRET)


def test_model_aliases_and_hook_diagnostics_are_plain_strings() -> None:
    from swarm_arena._planning.validate import check_models
    from swarm_arena.config import ModelCatalog, ModelSpec

    class Strange(str):
        def __hash__(self) -> int:
            raise AssertionError("subclass hash ran")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("subclass comparison ran")

        def __str__(self) -> str:
            raise AssertionError("subclass str ran")

    register_model_check(lambda uses, config: ([], [Strange("warning")]))
    cfg = RunConfig(
        models=ModelCatalog({"m": ModelSpec(model="fake/m", roles=["manager"])}),
        limits=RunConfig().limits.model_copy(update={"max_tokens": 1000}),
    )
    errors, warnings = check_models({"manager": [Strange("m")]}, cfg)
    assert errors == []
    assert warnings == ["warning"]
    assert type(warnings[0]) is str


@pytest.mark.parametrize("site", ["manager", "converge", "judge", "part", "nested"])
def test_model_checks_cover_admitted_roles_without_rebuilding(
    site: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena import Limits
    from swarm_arena.config import ModelCatalog, ModelSpec

    current = _model_plan("m")
    if site == "converge":
        current = current.model_copy(
            update={
                "converge": ConvergeSpec(
                    mode="team", team=current.teams[0].model_copy(update={"name": "final"})
                ),
                "teams": [],
            }
        )
    if site in {"judge", "part", "nested"}:
        pytest.importorskip("litellm")
        judge = Ref(path="_fakes.llm_components:AliasJudge", config={"model": "m"}, name="a")
        if site == "nested":
            judge = Ref(
                path="swarm_arena.scoring:Lexicographic",
                name="a",
                config={"judges": [judge.model_dump(exclude_none=True)]},
            )
        if site == "part":
            current = current.model_copy(
                update={
                    "parts": [
                        PartSpec(
                            name="p",
                            goal=Goal(id="p", namespace="demo", files=FileSet(manifest={})),
                            judges=[judge],
                            ranking="a",
                        )
                    ]
                }
            )
            current.teams[0] = current.teams[0].model_copy(update={"part": "p"})
            current = current.model_copy(
                update={
                    "converge": ConvergeSpec(
                        mode="team", team=TeamSpec(name="final", rationale="Join", agents=[])
                    )
                }
            )
        else:
            current = current.model_copy(update={"judges": [judge]})
        current.teams[0].agents[0] = (
            current.teams[0].agents[0].model_copy(update={"manager": MANAGER})
        )
    built_managers = 0
    original = Ref.build

    def build(self: Ref, base: type[Any]) -> Any:
        nonlocal built_managers
        if base is Manager:
            built_managers += 1
        return original(self, base)

    monkeypatch.setattr(Ref, "build", build)
    seen = []
    register_model_check(lambda uses, config: (seen.append(uses) or [], []))
    cfg = RunConfig(
        limits=Limits(max_tokens=10000),
        models=ModelCatalog({"m": ModelSpec(model="fake/m", roles=["manager", "judge"])}),
    )
    assert validate_plan(current, cfg) == []
    role = "manager" if site in {"manager", "converge"} else "judge"
    assert seen[0][role] == ["m"]
    assert built_managers == 1


def test_instructions_do_not_change_the_fingerprint() -> None:
    pytest.importorskip("agents")
    from swarm_arena import Limits
    from swarm_arena.config import ModelCatalog, ModelSpec

    original = _plan(agents_per_team=3, same_tool=True)
    agents = [
        spec.model_copy(
            update={
                "manager": Ref(
                    path="swarm_arena.llm:AgentsSDKManager",
                    config={"model": "opus", "instructions": instructions},
                )
            }
        )
        for spec, instructions in zip(original.teams[0].agents, ("a", "b", "c"), strict=True)
    ]
    plan = original.model_copy(
        update={"teams": [original.teams[0].model_copy(update={"agents": agents})]}
    )
    cfg = RunConfig(
        models=ModelCatalog({"opus": ModelSpec(model="fake/opus", roles=["manager"])}),
        limits=Limits(max_agents_per_fingerprint=2, max_tokens=100_000),
    )
    two_agents = plan.model_copy(
        update={"teams": [plan.teams[0].model_copy(update={"agents": agents[:2]})]}
    )
    assert validate_plan(two_agents, cfg) == []
    with pytest.raises(PlanError, match="max_agents_per_fingerprint"):
        validate_plan(plan, cfg)


async def test_draft_admission_uses_explicit_base_dir_and_keeps_provenance(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    from unittest.mock import Mock

    from swarm_arena._engine.run import mark_planning_session
    from swarm_arena._planning.validate import _admit_draft, _freeze_draft_goal

    (tmp_path / "problem").mkdir()
    (tmp_path / "problem/run.py").write_text("print('source')")
    original = _plan(tool=CATALOG_TOOL, judges=[J_B], ranking="b", base_dir=tmp_path)
    original = original.model_copy(
        update={"goal": original.goal.model_copy(update={"files": FileSet(globs=["problem/*"])})}
    )
    path = tmp_path / "plan.toml"
    path.write_text(original.dumps())
    loaded = Plan.load(path)
    assert Path.cwd() != loaded.base_dir
    current, _ = admit_revision(loaded, CFG_WITH_CATALOG, artifacts)
    frozen_goal = await _freeze_draft_goal(
        loaded.goal, base_dir=loaded.base_dir, artifacts=artifacts
    )
    proposed = mark_planning_session(current, Mock(run_id="r-p", call_ids=["llm-1"]))
    admitted = await _admit_draft(
        proposed,
        CFG_WITH_CATALOG,
        artifacts,
        frozen_goal=frozen_goal,
        base_dir=loaded.base_dir,
        history=[current],
        origin="autonomous",
    )
    assert admitted.goal.key() == current.goal.key()
    assert admitted.base_dir == loaded.base_dir
    assert admitted.planning_session == "r-p"
    assert admitted.planner_call_ids == ("llm-1",)
    assert (admitted.origin, admitted.revision, admitted.parent_revision) == ("autonomous", 1, 0)


@pytest.mark.parametrize("change", ["inputs", "files", "mutate-inputs"])
async def test_initial_draft_cannot_replace_frozen_human_goal(
    tmp_path: Path, artifacts: ArtifactStore, change: str
) -> None:
    from swarm_arena._planning.validate import _admit_draft, _freeze_draft_goal

    goal = Goal(id="g", namespace="demo", inputs={"value": "original"})
    frozen = await _freeze_draft_goal(goal, base_dir=tmp_path, artifacts=artifacts)
    if change == "mutate-inputs":
        goal.inputs["value"] = "secret-sentinel"
        altered = goal
    elif change == "inputs":
        altered = goal.model_copy(update={"inputs": {"value": "secret-sentinel"}})
    else:
        altered = goal.model_copy(update={"files": FileSet(manifest={"problem/data": "a" * 64})})
    proposed = _plan().model_copy(update={"goal": altered, "origin": "human"})
    with pytest.raises(PlanError, match="frozen main goal") as caught:
        await _admit_draft(
            proposed, CFG_WITH_CATALOG, artifacts, frozen_goal=frozen, base_dir=tmp_path
        )
    assert_secret_absent(caught.value, "secret-sentinel")


async def test_initial_draft_cannot_claim_human_catalog_authority(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    from swarm_arena._planning.validate import _admit_draft, _freeze_draft_goal

    proposed = _plan(tool=Ref(path="_fakes.import_canary:Tool"))
    sys.modules.pop("_fakes.import_canary", None)
    frozen = await _freeze_draft_goal(proposed.goal, base_dir=tmp_path, artifacts=artifacts)
    with pytest.raises(PlanError, match="catalog"):
        await _admit_draft(proposed, RunConfig(), artifacts, frozen_goal=frozen, base_dir=tmp_path)
    assert "_fakes.import_canary" not in sys.modules


async def test_abandoned_draft_snapshot_starts_no_later_artifact_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    draft_spool_closed: threading.Event,
) -> None:
    import asyncio

    import anyio
    from _fakes.llm_run import HeldOperation

    from swarm_arena._planning.validate import _freeze_draft_goal

    (tmp_path / "problem").mkdir()
    (tmp_path / "problem/a").write_text("a")
    (tmp_path / "problem/b").write_text("b")
    artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))
    held = HeldOperation()
    uploaded: list[bytes] = []
    original = artifacts.put_bytes

    def put(data: bytes) -> str:
        uploaded.append(data)
        held.wait()
        return original(data)

    monkeypatch.setattr(artifacts, "put_bytes", put)
    task = asyncio.create_task(
        _freeze_draft_goal(
            Goal(id="g", namespace="n", files=FileSet(globs=["problem/*"])),
            base_dir=tmp_path,
            artifacts=artifacts,
        )
    )
    try:
        await held.ready()
        task.cancel()
        with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
            await task
        assert await anyio.to_thread.run_sync(draft_spool_closed.wait, 5)
        await held.finish()
        assert len(uploaded) == 1
    finally:
        await held.finish()
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task


@pytest.mark.parametrize("count", [4, 24])
async def test_draft_snapshot_retains_only_one_file_at_first_upload(
    tmp_path: Path, count: int
) -> None:
    import gc
    import tracemalloc

    from swarm_arena._model.canonical import sha256_hex
    from swarm_arena._planning.validate import _freeze_draft_goal

    size = 1024 * 1024
    for index in range(count):
        (tmp_path / str(index)).write_bytes(bytes([index]) * size)

    class Sink(ArtifactStore):
        def __init__(self) -> None:
            self.live: list[int] = []

        def put_bytes(self, data: bytes) -> str:
            self.live.append(tracemalloc.get_traced_memory()[0])
            return sha256_hex(data)

    sink = Sink()
    gc.collect()
    tracemalloc.start()
    try:
        result = await _freeze_draft_goal(
            Goal(id="g", namespace="n", files=FileSet(globs=["*"])),
            base_dir=tmp_path,
            artifacts=sink,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(result.files.manifest) == len(sink.live) == count
    assert sink.live[0] < 3 * size
    assert peak < 5 * size


async def test_draft_snapshot_uploads_frozen_bytes_once_per_digest(tmp_path: Path) -> None:
    from swarm_arena._model.canonical import sha256_hex
    from swarm_arena._planning.validate import _freeze_draft_goal

    (tmp_path / "a").write_bytes(b"first")
    (tmp_path / "b").write_bytes(b"second")
    (tmp_path / "c").write_bytes(b"first")

    class Sink(ArtifactStore):
        def __init__(self) -> None:
            self.uploaded: list[bytes] = []

        def put_bytes(self, data: bytes) -> str:
            if not self.uploaded:
                (tmp_path / "b").write_bytes(b"changed after snapshot")
            self.uploaded.append(data)
            return sha256_hex(data)

    sink = Sink()
    result = await _freeze_draft_goal(
        Goal(id="g", namespace="n", files=FileSet(globs=["*"])), base_dir=tmp_path, artifacts=sink
    )
    assert sink.uploaded == [b"first", b"second"]
    assert result.files.manifest == {
        "a": sha256_hex(b"first"),
        "b": sha256_hex(b"second"),
        "c": sha256_hex(b"first"),
    }


@pytest.fixture
def draft_spool_closed(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    from swarm_arena._planning import validate

    original = validate.TemporaryFile
    closed = threading.Event()

    @contextlib.contextmanager
    def tracked() -> Generator[Any]:
        try:
            with original() as spool:
                yield spool
        finally:
            closed.set()

    monkeypatch.setattr(validate, "TemporaryFile", tracked)
    return closed


@pytest.mark.parametrize("cancel", [False, True])
async def test_draft_spool_cleanup_after_failed_or_abandoned_preparation(
    tmp_path: Path,
    draft_spool_closed: threading.Event,
    cancel: bool,
) -> None:
    import asyncio

    import anyio
    from _fakes.llm_run import HeldOperation

    from swarm_arena._planning.validate import _draft_work

    held = HeldOperation()
    artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))

    def prepare(staging: ArtifactStore) -> Goal:
        staging.put_bytes(b"snapshot")
        if cancel:
            held.wait()
            return Goal(id="g", namespace="n")
        raise ValueError("secret-sentinel")

    task = asyncio.create_task(_draft_work(prepare, artifacts))
    try:
        if cancel:
            await held.ready()
            task.cancel()
            with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
                await task
            await held.finish()
        else:
            with pytest.raises(ValueError, match="draft admission failed") as caught:
                await task
            assert_secret_absent(caught.value, "secret-sentinel")
        assert await anyio.to_thread.run_sync(draft_spool_closed.wait, 5)
        assert list(open_store(str(tmp_path / "artifacts")).list("")) == []
    finally:
        if cancel:
            await held.finish()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
