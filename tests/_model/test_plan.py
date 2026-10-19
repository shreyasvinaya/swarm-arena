import traceback
from pathlib import Path
from typing import Any

import pytest
from _fakes.tools import GOAL, Widget
from pydantic import BaseModel, ValidationError

from swarm_arena._errors import ConfigError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import (
    AgentSpec,
    ConvergeSpec,
    FlowSpec,
    PartSpec,
    Plan,
    Resources,
    RoundSpec,
    TeamSpec,
)
from swarm_arena._model.ref import Ref

SPEC_TOML = (
    "[goal]\n"
    'id = "ns-viscosity"\n'
    'namespace = "physika"\n'
    'description = "Recover rho and nu from the observed flow field."\n'
    'files = ["problem/*.phyk", "problem/obs.npz", "problem/run.py"]   # snapshotted at pl'
    "an time\n"
    "\n"
    "[[teams]]\n"
    'name = "fd-solver"\n'
    'rationale = "Finite-difference forward model, tune the inverse step."\n'
    "\n"
    "  [[teams.agents]]\n"
    '  name = "tuner-1"\n'
    '  rationale = "Explore learning rate and step size."\n'
    '  manager = { path = "swarm_arena.llm:AgentsSDKManager", config = { model = "opus" } '
    "}   # alias from run.toml [models]\n"
    "\n"
    "    [teams.agents.tool]\n"
    '    path = "swarm_arena:ScriptTool"\n'
    "    [teams.agents.tool.config]\n"
    '    command  = ["python", "problem/run.py"]    # cwd = the workspace; goal files keep'
    ' their paths; "python" = config.python or the current interpreter\n'
    '    python   = "/opt/envs/physika/bin/python"  # the script\'s own environment (option'
    "al)\n"
    '    editable = ["problem/*.phyk"]              # which files the manager may change o'
    "r add (Tool.editable())\n"
    '    params   = { lr = { type = "number", minimum = 1e-5, maximum = 1.0 }, epochs = { '
    'type = "integer", maximum = 400 } }\n'
    "    timeout_s = 1200\n"
    '    env = ["ANTHROPIC_API_KEY"]                # allowlist: for a script that calls a'
    "n LLM itself\n"
    "\n"
    "[[checks]]\n"
    'path = "swarm_arena:ScriptCheck"\n'
    'config = { command = ["python", "checks/check.py"], files = ["checks/check.py"] }\n'
    "\n"
    "[[judges]]\n"
    'name = "rel-error"\n'
    'path = "swarm_arena:ScriptJudge"\n'
    'config = { command = ["python", "judge/score.py"], files = ["judge/score.py", "judge/'
    'hidden/*.npz"], direction = "min" }\n'
)
AGENT = AgentSpec(name="a", rationale="r", tool=Ref(path="m:T"), manager=Ref(path="m:M"))
TEAMS = [TeamSpec(name="t", rationale="r", agents=[AGENT])]
JUDGES = [Ref(path="m:J", name="rel-error")]


def plan_with(**kw: Any) -> Plan:
    return Plan.model_validate({"goal": GOAL, "teams": TEAMS, "judges": JUDGES} | kw)


def test_converge_defaults() -> None:
    c = ConvergeSpec()
    assert (
        c.mode,
        c.top_n,
        c.distinct,
        c.allow_partial,
        c.parents_per_part,
        c.max_join_attempts,
    ) == ("select", 2, True, False, 1, 1)


def test_effective_fidelity() -> None:
    assert RoundSpec(fidelity="full").effective_fidelity() == "full"
    assert (
        RoundSpec(tool_overrides={"epochs": 100}).effective_fidelity()
        == sha256_hex(b'{"epochs":100}')[:12]
    )
    assert (
        RoundSpec(tool_overrides={"b": 2, "a": 1}).effective_fidelity()
        == RoundSpec(tool_overrides={"a": 1, "b": 2}).effective_fidelity()
    )
    assert RoundSpec(fidelity="").effective_fidelity() == sha256_hex(b"{}")[:12]


def test_rationale_required() -> None:
    for r in ("", "  ", "\t\n"):
        with pytest.raises(ValidationError):
            AgentSpec(name="a", rationale=r, tool=Ref(path="m:T"), manager=Ref(path="m:M"))
        with pytest.raises(ValidationError):
            TeamSpec(name="t", rationale=r, agents=[AGENT])
    assert TeamSpec(name="t", rationale=" r ", agents=[]).rationale == "r"
    assert AgentSpec.model_validate(AGENT.model_dump() | {"rationale": " r "}).rationale == "r"


def test_flows_reserved_for_v2() -> None:
    with pytest.raises(ValidationError, match="v2"):
        Plan(
            goal=GOAL,
            teams=TEAMS,
            judges=JUDGES,
            flows=[FlowSpec(sender="a", recipient="*", rationale="r")],
        )


def test_load_spec_example(tmp_path: Path) -> None:
    (tmp_path / "plan.toml").write_text(SPEC_TOML)
    plan = Plan.load(tmp_path / "plan.toml")
    assert plan.goal.files.globs == ["problem/*.phyk", "problem/obs.npz", "problem/run.py"]
    tool = plan.teams[0].agents[0].tool
    assert tool.path == "swarm_arena:ScriptTool"
    assert tool.config["params"]["lr"]["minimum"] == 1e-5
    assert plan.judges[0].name == "rel-error"
    assert plan.origin == "human"
    assert plan.base_dir == tmp_path
    assert plan.policy == Ref(path="swarm_arena:SuccessiveHalving")
    assert plan.schema_kind == "plan"
    assert plan.schema_version == 1


@pytest.mark.parametrize("origin", ["llm", "together", "autonomous"])
def test_load_keeps_non_human_origin(tmp_path: Path, origin: str) -> None:
    (tmp_path / "p.toml").write_text(f'origin = "{origin}"\n' + SPEC_TOML)
    plan = Plan.load(tmp_path / "p.toml")
    assert plan.origin == origin
    assert Plan.loads(plan.dumps(), base_dir=tmp_path).origin == origin


def test_dumps_roundtrip(tmp_path: Path) -> None:
    plan = Plan.loads(SPEC_TOML, base_dir=tmp_path)
    assert Plan.loads(plan.dumps(), base_dir=tmp_path) == plan
    assert 'origin = "human"' in plan.dumps()
    assert "base_dir" not in plan.model_dump()


def test_load_errors_name_the_file(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text("teams = [")
    with pytest.raises(ConfigError, match=r"a\.toml"):
        Plan.load(tmp_path / "a.toml")
    (tmp_path / "b.toml").write_text("teamz = 1\n" + SPEC_TOML)
    with pytest.raises(ConfigError, match=r"(?s)b\.toml.*teamz"):
        Plan.load(tmp_path / "b.toml")


def test_ranking_name() -> None:
    one = [Ref(path="m:J", name="rel-error")]
    two = [Ref(path="m:J", name="a"), Ref(path="m:J", name="b")]
    assert plan_with(judges=one).ranking_name() == "rel-error"
    assert plan_with(judges=two, ranking="a").ranking_name() == "a"
    with pytest.raises(ValueError, match="several judges"):
        plan_with(judges=two).ranking_name()
    with pytest.raises(ValueError, match="zz"):
        plan_with(judges=two, ranking="zz").ranking_name()
    with pytest.raises(ValueError, match="duplicate"):
        plan_with(judges=[two[0], two[0]], ranking="a").ranking_name()
    with pytest.raises(ValueError, match="judges"):
        plan_with(judges=[]).ranking_name()


def test_rounds_total() -> None:
    assert plan_with().rounds_total(Limits()) == 3
    assert plan_with(rounds=[RoundSpec(), RoundSpec()]).rounds_total(Limits()) == 2


def test_structure_rules() -> None:
    with pytest.raises(ValidationError, match="duplicate team"):
        plan_with(teams=TEAMS * 2)
    with pytest.raises(ValidationError, match="duplicate agent"):
        plan_with(teams=[TeamSpec(name="t", rationale="r", agents=[AGENT, AGENT])])
    with pytest.raises(ValidationError, match="duplicate part"):
        plan_with(parts=[PartSpec(name="p", goal=GOAL)] * 2)
    team = TEAMS[0].model_copy(update={"part": "p"})
    with pytest.raises(ValidationError, match="unknown part"):
        plan_with(teams=[team])
    assert plan_with(teams=[team], parts=[PartSpec(name="p", goal=GOAL)]).teams[0].part == "p"
    # Agent names are scoped to each team.
    assert (
        len(plan_with(teams=[*TEAMS, TeamSpec(name="t2", rationale="r", agents=[AGENT])]).teams)
        == 2
    )


@pytest.mark.parametrize("mode", ["team", "select"])
@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"agents": [AGENT, AGENT]}, "duplicate agent"),
        ({"name": "t"}, "duplicate team"),
        ({"part": "missing"}, "unknown part"),
        ({"rationale": "  "}, "converge"),
        ({"agents": [AGENT.model_dump() | {"rationale": "  "}]}, "converge"),
        ({"name": "a/b"}, "converge"),
    ],
)
def test_converge_team_structure(mode: str, updates: dict[str, Any], message: str) -> None:
    team = TEAMS[0].model_dump() | {"name": "converger"} | updates
    with pytest.raises(ValidationError, match=message):
        plan_with(converge={"mode": mode, "team": team})


def test_converge_team_accepts_declared_part_and_scoped_agent_name() -> None:
    team = TeamSpec(name="converger", rationale="r", part="p", agents=[AGENT])
    plan = plan_with(parts=[PartSpec(name="p", goal=GOAL)], converge=ConvergeSpec(team=team))
    assert plan.converge.team == team
    assert plan.teams == TEAMS


@pytest.mark.parametrize("name", ["a/b", "A", "", "../a", "a b"])
@pytest.mark.parametrize(
    ("model", "values"),
    [
        (PartSpec, {"goal": GOAL}),
        (TeamSpec, {"rationale": "r", "agents": []}),
        (AgentSpec, {"rationale": "r", "tool": Ref(path="m:T"), "manager": Ref(path="m:M")}),
    ],
)
def test_name_slug(
    model: type[PartSpec | TeamSpec | AgentSpec], values: dict[str, Any], name: str
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(values | {"name": name})


def test_base_dir_captured_at_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    plan = plan_with()
    monkeypatch.chdir("/")
    assert plan.base_dir == tmp_path
    assert plan.model_copy(update={"rationale": "x"}).base_dir == tmp_path


def test_relative_load_base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "p.toml").write_text(SPEC_TOML)
    plan = Plan.load("p.toml")
    monkeypatch.chdir("/")
    assert plan.base_dir == tmp_path


def test_refs_accept_component_instances() -> None:
    spec = AgentSpec.model_validate(
        {"name": "a1", "rationale": "r", "tool": Widget(), "manager": Widget()}
    )
    assert spec.tool == Ref.of(Widget())
    assert spec.manager == Ref.of(Widget())
    part = PartSpec.model_validate(
        {"name": "p", "goal": GOAL, "checks": [Widget()], "judges": [Widget]}
    )
    assert part.checks == [Ref.of(Widget())]
    assert part.judges == [Ref.of(Widget)]
    plan = plan_with(checks=[Widget], judges=[Widget()], policy=Widget())
    assert plan.checks == [Ref.of(Widget)]
    assert plan.judges == [plan.policy] == [Ref.of(Widget())]


@pytest.mark.parametrize(
    "bad", [{"cpus": 0}, {"gpus": -1}, {"memory_gib": 0}, {"custom": {"gpu": -1}}]
)
def test_resources_bounds(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Resources.model_validate(bad)
    assert Resources(cpus=0.5, gpus=0, custom={"gpu": 0}).cpus == 0.5


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (RoundSpec, {"tool_timeout_s": 0}),
        (ConvergeSpec, {"top_n": 0}),
        (ConvergeSpec, {"parents_per_part": 0}),
        (ConvergeSpec, {"max_join_attempts": -1}),
        (ConvergeSpec, {"mode": "other"}),
        (AgentSpec, AGENT.model_dump() | {"replicates": 0}),
    ],
)
def test_spec_bounds(
    model: type[RoundSpec | ConvergeSpec | AgentSpec], values: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(values)
    assert ConvergeSpec(max_join_attempts=0).max_join_attempts == 0


SECRET = "sentinel-secret-92845"


@pytest.mark.parametrize(
    "text",
    [
        f'origin = "{SECRET}"\n' + SPEC_TOML,
        f'teamz = "{SECRET}"\n' + SPEC_TOML,
        SPEC_TOML.replace('name = "fd-solver"', f'name = "{SECRET}/bad"'),
        SPEC_TOML + f'\n[goal.files.manifest]\n"{SECRET}" = "bad"\n',
        SPEC_TOML.replace(
            "[teams.agents.tool]",
            f'[teams.agents.resources.custom]\n"{SECRET}" = -1\n[teams.agents.tool]',
        ),
        f'"{SECRET}" = 1\n"{SECRET}" = 2\n',
        SPEC_TOML.replace('path = "swarm_arena:ScriptTool"', f'path = "{SECRET}"'),
    ],
)
def test_load_errors_do_not_leak_values_or_keys(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(text)
    for load in (lambda: Plan.load(path), lambda: Plan.loads(text, base_dir=tmp_path)):
        with pytest.raises(ConfigError) as exc:
            load()
        assert SECRET not in "".join(traceback.format_exception(exc.value))
        assert exc.value.__cause__ is None


def test_validation_locations_keep_only_schema_fields(tmp_path: Path) -> None:
    text = SPEC_TOML.replace(
        "[teams.agents.tool]", "[teams.agents.resources.custom]\ncpus = -1\n[teams.agents.tool]"
    )
    with pytest.raises(ConfigError, match=r"teams\.<…>"):
        Plan.loads(text, base_dir=tmp_path)


@pytest.mark.parametrize("duplicate", [False, True])
def test_ranking_errors_show_identifiers_without_config_values(duplicate: bool) -> None:
    judge = Ref(path="m:J", name="judge-id", config={"token": SECRET})
    judges = [judge, judge] if duplicate else [judge]
    identifier = "judge-id" if duplicate else "unknown-ranking"
    with pytest.raises(ValueError, match=identifier) as exc:
        plan_with(judges=judges, ranking="unknown-ranking").ranking_name()
    assert SECRET not in "".join(traceback.format_exception(exc.value))


def test_ranking_resolution_errors_do_not_leak_values(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_name(self: Ref) -> str:
        raise ValueError(SECRET)

    monkeypatch.setattr(Ref, "component_name", fail_name)
    with pytest.raises(ValueError, match="judges") as exc:
        plan_with().ranking_name()
    assert SECRET not in "".join(traceback.format_exception(exc.value))


@pytest.mark.parametrize(
    ("table", "key", "location"),
    [
        ("", "teamz", "teamz"),
        ("teams", "agentz", "teams.<…>"),
        ("teams.agents", "replicatez", "teams.<…>"),
        ("teams.agents.resources", "cpuz", "teams.<…>"),
    ],
)
def test_unknown_plan_fields_show_identifiers_without_values(
    tmp_path: Path, table: str, key: str, location: str
) -> None:
    data = plan_with().model_dump()
    target = data
    for segment in table.split(".") if table else []:
        target = target[segment]
        if isinstance(target, list):
            target = target[0]
    target[key] = SECRET
    with pytest.raises(ValidationError) as exc:
        Plan.model_validate(data)
    assert location in str(exc.value)
    assert SECRET not in "".join(traceback.format_exception(exc.value))
    assert SECRET not in str(exc.value.errors())
    text = (
        f'{key} = "{SECRET}"\n' + SPEC_TOML
        if not table
        else SPEC_TOML + (f'\n[{table}]\n{key} = "{SECRET}"\n')
    )
    if table in ("teams", "teams.agents"):
        text = SPEC_TOML.replace(f"[[{table}]]", f'[[{table}]]\n{key} = "{SECRET}"')
    path = tmp_path / "typo.toml"
    path.write_text(text)
    for load in (lambda: Plan.load(path), lambda: Plan.loads(text, base_dir=tmp_path)):
        with pytest.raises(ConfigError) as exc:
            load()
        assert location in str(exc.value)
        assert SECRET not in "".join(traceback.format_exception(exc.value))


def test_roundtrip_all_nested_specs() -> None:
    plan = plan_with(
        parts=[PartSpec(name="p", goal=GOAL, checks=[], judges=JUDGES, ranking="rel-error")],
        rounds=[RoundSpec(tool_overrides={"epochs": 10}, tool_timeout_s=5)],
        converge=ConvergeSpec(
            mode="team",
            team=TEAMS[0].model_copy(update={"name": "converger"}),
            round=RoundSpec(fidelity="full"),
        ),
        revision=2,
        parent_revision=1,
    )
    restored = Plan.loads(plan.dumps(), base_dir=plan.base_dir)
    assert restored == plan


def test_models_frozen_and_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        plan_with(typo=1)
    plan = plan_with()
    with pytest.raises(ValidationError):
        plan.rationale = "changed"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (Resources, {"custom": {SECRET: -1}}),
        (Resources, {"cpus": SECRET}),
        (RoundSpec, {"tool_timeout_s": SECRET}),
        (ConvergeSpec, {"mode": SECRET}),
        (AgentSpec, AGENT.model_dump() | {"name": SECRET + "/bad"}),
        (TeamSpec, {"name": "t", "rationale": SECRET, "agents": [{"name": SECRET + "/bad"}]}),
        (PartSpec, {"name": "p", "goal": {"id": SECRET + "/bad", "namespace": "n"}}),
        (FlowSpec, {"sender": "a", "recipient": "*", "rationale": "r", "typo": SECRET}),
        (Plan, {"goal": GOAL, "teams": TEAMS, "judges": JUDGES, "origin": SECRET}),
        (
            Plan,
            {
                "goal": {"id": "g", "namespace": "n", "files": {"manifest": {SECRET: "bad"}}},
                "teams": [],
                "judges": [],
            },
        ),
    ],
)
def test_direct_validation_errors_are_sanitized(
    model: type[BaseModel], values: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError) as exc:
        model.model_validate(values)
    assert SECRET not in "".join(traceback.format_exception(exc.value))
    assert SECRET not in str(exc.value.errors())


def test_fidelity_errors_do_not_leak_values() -> None:
    class SecretObject:
        def __repr__(self) -> str:
            return SECRET

    with pytest.raises(ValueError, match="tool_overrides") as exc:
        RoundSpec(tool_overrides={SECRET: SecretObject()}).effective_fidelity()
    assert SECRET not in "".join(traceback.format_exception(exc.value))
