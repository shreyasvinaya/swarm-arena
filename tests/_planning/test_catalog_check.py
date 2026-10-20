"""Exercise import-free catalog authorization and frozen manifest provenance."""

import builtins
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena import AgentSpec, ConvergeSpec, Goal, Plan, Ref, RoundSpec, RunConfig, TeamSpec
from swarm_arena._errors import PlanError
from swarm_arena._model.base import CatalogKind
from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog
from swarm_arena._planning.catalog_check import (
    _check_catalog_defaults,
    authorize_revision,
    catalog_form,
    check_catalog,
    find_entry,
    parent_human,
)
from swarm_arena._planning.provenance import RefSite, ref_sites, site_key, stamp

FIXED = {"command": ["python", "problem/run.py"], "editable": ["problem/*.phyk"]}
CAT = PlannerCatalog(
    entries=[
        CatalogEntry(
            name="phys",
            kind="tool",
            ref="swarm_arena:ScriptTool",
            description="d",
            config=FIXED,
            settable={"timeout_s": {"type": "number", "maximum": 1200}},
        )
    ]
)
CAT_FILES = PlannerCatalog(
    entries=[
        CatalogEntry(
            name="runner",
            kind="tool",
            ref="swarm_arena:ScriptTool",
            description="d",
            config={"command": ["python", "problem/run.py"], "files": ["problem/run.py"]},
        )
    ]
)
MEMBERS: list[dict[str, Any]] = [
    {"path": "_fakes.judges:FixedJudge", "config": {"value": 1}},
    {"path": "_fakes.judges:FixedJudge", "config": {"value": 2}},
]
CAT_NESTED = PlannerCatalog(
    entries=[
        CatalogEntry(
            name="patience",
            kind="policy",
            ref="swarm_arena:Patience",
            description="d",
            config={"rounds": 2},
        ),
        CatalogEntry(
            name="chain",
            kind="policy",
            ref="swarm_arena:Chain",
            description="d",
            config={"policies": [{"path": "swarm_arena:Patience", "config": {"rounds": 2}}]},
        ),
        *[
            CatalogEntry(
                name=f"judge-{i}", kind="judge", ref=m["path"], description="d", config=m["config"]
            )
            for i, m in enumerate(MEMBERS)
        ],
        CatalogEntry(
            name="lex",
            kind="judge",
            ref="swarm_arena.scoring:Lexicographic",
            description="d",
            config={"judges": MEMBERS},
        ),
    ]
)


def _site(
    config: dict[str, Any], *, path: str = "swarm_arena:ScriptTool", kind: CatalogKind = "tool"
) -> RefSite:
    return RefSite(pointer="/teams/0/agents/0/tool", kind=kind, ref=Ref(path=path, config=config))


def _lex_site(name: str | None) -> RefSite:
    members = [{**member, **({"name": name} if name is not None else {})} for member in MEMBERS]
    return _site({"judges": members}, path="swarm_arena.scoring:Lexicographic", kind="judge")


def _human_plan() -> Plan:
    return stamp(
        Plan(
            goal=Goal(id="g", namespace="demo"),
            teams=[
                TeamSpec(
                    name="t",
                    rationale="Explore",
                    agents=[
                        AgentSpec(
                            name="a",
                            rationale="Explore",
                            tool=Ref(path="human:Tool"),
                            manager=Ref(path="human:Manager"),
                        )
                    ],
                )
            ],
            judges=[],
        ),
        origin="human",
        parent=None,
    )


def _with_tool(plan: Plan, *, tool: Ref) -> Plan:
    team = plan.teams[0]
    agent = team.agents[0].model_copy(update={"tool": tool})
    return plan.model_copy(update={"teams": [team.model_copy(update={"agents": [agent]})]})


def _runner_ref(*, files: Any) -> Ref:
    return Ref(
        path="swarm_arena:ScriptTool",
        config={"command": ["python", "problem/run.py"], "files": files},
    )


def _cfg(catalog: PlannerCatalog) -> RunConfig:
    return RunConfig(planner_catalog=catalog)


def _files(digest: str = "ab", *, globs: list[str] | None = None) -> dict[str, Any]:
    return {
        "globs": ["problem/run.py"] if globs is None else globs,
        "manifest": {"problem/run.py": digest * 32},
    }


def test_matching_ref_passes() -> None:
    assert check_catalog(_site({**FIXED, "timeout_s": 600}), CAT, overrides={}) == []


def test_unknown_ref_rejected() -> None:
    assert check_catalog(_site({}, path="evil:Tool"), CAT, overrides={}) == [
        "/teams/0/agents/0/tool: evil:Tool is not in the planner catalog for kind 'tool'"
    ]


def test_fixed_field_change_rejected() -> None:
    assert any(
        "fixed field 'command'" in e
        for e in check_catalog(
            _site({"command": ["sh", "x"], "editable": ["problem/*.phyk"]}), CAT, overrides={}
        )
    )


def test_settable_schema_enforced() -> None:
    assert any(
        "timeout_s" in e
        for e in check_catalog(_site({**FIXED, "timeout_s": 5000}), CAT, overrides={})
    )


def test_override_of_fixed_field_rejected() -> None:
    assert any(
        "override sets fixed field 'editable'" in e
        for e in check_catalog(_site(FIXED), CAT, overrides={"editable": ["**"]})
    )


def test_extra_unknown_field_rejected() -> None:
    errors = check_catalog(_site({**FIXED, "python": "/bin/x"}), CAT, overrides={})
    assert any("field '<field>'" in e for e in errors)
    assert "python" not in str(errors)
    assert "/bin/x" not in str(errors)


def test_snapshotted_site_matches_raw_entry() -> None:
    site = _site({"command": ["python", "problem/run.py"], "files": _files()})
    assert check_catalog(site, CAT_FILES, overrides={}) == []
    assert find_entry(site.ref, "tool", CAT_FILES) is CAT_FILES.entries[0]


def test_noncatalog_module_never_imported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canary = tmp_path / "imported"
    monkeypatch.setenv("SWARM_ARENA_TEST_CANARY", str(canary))
    monkeypatch.delitem(sys.modules, "_fakes.import_canary", raising=False)
    r0 = _human_plan()
    r1 = stamp(
        _with_tool(r0, tool=Ref(path="_fakes.import_canary:Tool")), origin="autonomous", parent=r0
    )
    _, errors = authorize_revision(r1, _cfg(CAT_FILES), history=[r0])
    assert len(errors) == 1
    assert "not in the planner catalog" in errors[0]
    assert "_fakes.import_canary" not in sys.modules
    assert not canary.exists()


def test_planner_written_manifest_dropped() -> None:
    r0 = _human_plan()
    r1 = stamp(_with_tool(r0, tool=_runner_ref(files=_files())), origin="autonomous", parent=r0)
    plan, errors = authorize_revision(r1, _cfg(CAT_FILES), history=[r0])
    assert errors == []
    assert plan.teams[-1].agents[-1].tool.config["files"] == ["problem/run.py"]
    assert r1.teams[-1].agents[-1].tool.config["files"] == _files()
    assert plan.base_dir == r1.base_dir


def test_committed_manifest_kept_for_unchanged_site() -> None:
    r0 = _human_plan()
    r1 = stamp(_with_tool(r0, tool=_runner_ref(files=_files())), origin="autonomous", parent=r0)
    r2 = stamp(r1, origin="autonomous", parent=r1)
    plan, errors = authorize_revision(r2, _cfg(CAT_FILES), history=[r0, r1])
    assert errors == []
    assert plan.teams[-1].agents[-1].tool.config["files"]["manifest"] == {
        "problem/run.py": "ab" * 32
    }
    changed = _with_tool(r2, tool=_runner_ref(files=_files("cd")))
    plan, _ = authorize_revision(changed, _cfg(CAT_FILES), history=[r0, r1])
    assert plan.teams[-1].agents[-1].tool.config["files"] == ["problem/run.py"]


def test_nested_ref_defaults_match_after_snapshot() -> None:
    member = {"path": "swarm_arena:Patience", "config": {"rounds": 2}}
    assert (
        check_catalog(
            _site({"policies": [member]}, path="swarm_arena:Chain", kind="policy"),
            CAT_NESTED,
            overrides={},
        )
        == []
    )
    assert check_catalog(_lex_site(name=None), CAT_NESTED, overrides={}) == []
    assert catalog_form({"p": [{"path": "a.b:C"}]}) == catalog_form(
        {"p": [{"path": "a.b:C", "config": {}, "name": None}]}
    )
    bad = {**member, "config": {"rounds": 3}}
    assert any(
        "fixed field 'policies'" in e
        for e in check_catalog(
            _site({"policies": [bad]}, path="swarm_arena:Chain", kind="policy"),
            CAT_NESTED,
            overrides={},
        )
    )
    renamed = {**member, "name": "other"}
    assert (
        check_catalog(
            _site({"policies": [renamed]}, path="swarm_arena:Chain", kind="policy"),
            CAT_NESTED,
            overrides={},
        )
        != []
    )


@pytest.mark.parametrize(("fixed", "changed"), [(True, 1), (False, 0), (1, 1.0), (1.0, 1)])
def test_normalization_preserves_scalar_types(fixed: Any, changed: Any) -> None:
    entry = CatalogEntry(
        name="typed",
        kind="tool",
        ref="absent:Tool",
        description="d",
        config={
            "nested": [{"path": "absent:Child", "config": {"value": [fixed]}}],
            "files": ["problem/run.py"],
        },
    )
    catalog = PlannerCatalog(entries=[entry])
    config = {**entry.config, "files": _files()}
    assert check_catalog(_site(config, path=entry.ref), catalog, overrides={}) == []
    config["nested"] = [{"path": "absent:Child", "config": {"value": [changed]}}]
    assert find_entry(Ref(path=entry.ref, config=config), "tool", catalog) is None
    assert any(
        "fixed field 'nested'" in e
        for e in check_catalog(_site(config, path=entry.ref), catalog, overrides={})
    )


def test_same_path_entries_and_first_match() -> None:
    first = CAT_FILES.entries[0]
    other = first.model_copy(
        update={"name": "other", "config": {"command": ["other"], "files": ["problem/run.py"]}}
    )
    duplicate = other.model_copy(update={"name": "duplicate"})
    catalog = PlannerCatalog(entries=[first, other, duplicate])
    ref = Ref(path=first.ref, config={**other.config, "files": _files()})
    assert find_entry(ref, "tool", catalog) is other
    assert find_entry(ref, "judge", catalog) is None
    assert check_catalog(_site(ref.config), catalog, overrides={}) == []
    errors = check_catalog(
        _site({"command": ["wrong"], "files": ["problem/run.py"]}), catalog, overrides={}
    )
    assert any("fixed field 'command'" in e for e in errors)


@pytest.mark.parametrize("overrides", [{"timeout_s": 600}, {"timeout_s": 600.5}])
def test_number_override_accepts_integer_and_float(overrides: dict[str, Any]) -> None:
    assert check_catalog(_site(FIXED), CAT, overrides=overrides) == []


@pytest.mark.parametrize(
    "overrides", [{"timeout_s": 5000}, {"timeout_s": True}, {"timeout_s": "SECRET_VALUE"}]
)
def test_override_schema_errors_are_sanitized(overrides: dict[str, Any]) -> None:
    errors = check_catalog(_site(FIXED), CAT, overrides=overrides)
    assert errors
    assert all(e.startswith("/teams/0/agents/0/tool: ") for e in errors)
    assert_secret_absent(PlanError(errors), "SECRET_VALUE")


def test_config_and_override_unknown_keys_are_hidden() -> None:
    for config, overrides in [
        ({**FIXED, "SECRET_KEY": "SECRET_VALUE"}, {}),
        (FIXED, {"SECRET_KEY": "SECRET_VALUE"}),
        ({**FIXED, "command": ["SECRET_VALUE"]}, {}),
    ]:
        errors = check_catalog(_site(config), CAT, overrides=overrides)
        assert errors
        assert_secret_absent(PlanError(errors), "SECRET_KEY")
        assert_secret_absent(PlanError(errors), "SECRET_VALUE")


def test_missing_fixed_field_is_rejected() -> None:
    assert any(
        "fixed field 'command'" in e
        for e in check_catalog(_site({"editable": FIXED["editable"]}), CAT, overrides={})
    )


def test_history_walker_carries_only_unbroken_human_authority() -> None:
    r0 = _human_plan()
    r1 = stamp(r0, origin="autonomous", parent=r0)
    r2 = stamp(r1, origin="llm", parent=r1)
    assert all(parent_human([r0, r1, r2]).values())
    assert authorize_revision(r2, RunConfig(), history=[r0, r1])[1] == []
    changed = stamp(_with_tool(r1, tool=Ref(path="other:Tool")), origin="autonomous", parent=r1)
    reverted = stamp(r0, origin="autonomous", parent=changed)
    tool_key = site_key(next(s for s in ref_sites(reverted) if s.kind == "tool"))
    assert parent_human([r0, r1, changed, reverted])[tool_key] is False
    human = stamp(reverted, origin="human", parent=reverted)
    assert all(parent_human([r0, changed, reverted, human]).values())
    assert parent_human([]) == {}


def test_human_revision_is_returned_untouched() -> None:
    plan = _with_tool(_human_plan(), tool=_runner_ref(files=_files()))
    result, errors = authorize_revision(plan, RunConfig())
    assert result is plan
    assert errors == []


@pytest.mark.parametrize("converge", [False, True])
def test_new_override_cannot_repeat_fixed_field(converge: bool) -> None:
    r0 = _with_tool(_human_plan(), tool=Ref(path="swarm_arena:ScriptTool", config=FIXED))
    if converge:
        r0 = r0.model_copy(update={"converge": ConvergeSpec(mode="team", team=r0.teams[0])})
        updated = r0.model_copy(
            update={
                "converge": r0.converge.model_copy(
                    update={"round": RoundSpec(tool_overrides={"editable": FIXED["editable"]})}
                )
            }
        )
    else:
        updated = r0.model_copy(
            update={"rounds": [RoundSpec(tool_overrides={"editable": FIXED["editable"]})]}
        )
    child = stamp(updated, origin="autonomous", parent=r0)
    _, errors = authorize_revision(child, _cfg(CAT), history=[r0])
    assert any("override sets fixed field 'editable'" in e for e in errors)


def test_carried_human_override_remains_authorized() -> None:
    r0 = _human_plan().model_copy(update={"rounds": [RoundSpec(tool_overrides={"human_only": 1})]})
    child = stamp(r0, origin="autonomous", parent=r0)
    assert authorize_revision(child, RunConfig(), history=[r0])[1] == []


@pytest.mark.parametrize("path", ["swarm_arena:ScriptTool", "swarm_arena:FunctionTool"])
def test_env_is_fixed_and_cannot_be_settable_or_overridden(path: str) -> None:
    fixed = {"env": ["API_TOKEN"]}
    entry = CatalogEntry(name="env", kind="tool", ref=path, description="d", config=fixed)
    catalog = PlannerCatalog(entries=[entry])
    assert check_catalog(_site(fixed, path=path), catalog, overrides={}) == []
    assert any(
        "fixed field 'env'" in e
        for e in check_catalog(_site({"env": ["SECRET_VALUE"]}, path=path), catalog, overrides={})
    )
    errors = check_catalog(_site(fixed, path=path), catalog, overrides={"env": ["SECRET_VALUE"]})
    assert any("override sets fixed field 'env'" in e for e in errors)
    assert_secret_absent(PlanError(errors), "SECRET_VALUE")
    with pytest.raises(ValidationError, match="human-only field 'env'"):
        PlannerCatalog(
            entries=[
                entry.model_copy(
                    update={"settable": {"env": {"type": "array", "items": {"type": "string"}}}}
                )
            ]
        )


def test_authorization_does_not_import_components_or_read_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r0 = _human_plan()
    r1 = stamp(_with_tool(r0, tool=_runner_ref(files=_files())), origin="autonomous", parent=r0)
    config = _cfg(CAT_FILES)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("authorization attempted import or I/O")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(importlib, "import_module", forbidden)
        guard.setattr(Ref, "load", forbidden)
        guard.setattr(Path, "open", forbidden)
        assert authorize_revision(r1, config, history=[r0])[1] == []


def test_normalization_is_recursive_symmetric_and_nonmutating() -> None:
    raw = {"members": [{"path": "absent:Child", "config": {"data": [{"files": ["a"]}]}}]}
    snapshot = {
        "members": [
            {
                "path": "absent:Child",
                "config": {"data": [{"files": {"globs": ["a"], "manifest": {"a": "ab" * 32}}}]},
                "name": None,
            }
        ]
    }
    assert catalog_form(raw) == catalog_form(snapshot)
    assert "name" not in raw["members"][0]
    assert snapshot["members"][0]["name"] is None
    entry = CatalogEntry(
        name="nested", kind="tool", ref="absent:Tool", description="d", config={"files": _files()}
    )
    catalog = PlannerCatalog(entries=[entry])
    assert (
        check_catalog(_site({"files": ["problem/run.py"]}, path=entry.ref), catalog, overrides={})
        == []
    )
    # Extra keys and incomplete path matches are ordinary dictionaries.
    for value in [
        {"path": "absent:Child", "other": 1},
        {"path": "absent:Child!"},
        {"globs": ["a"], "manifest": {}, "other": True},
    ]:
        assert catalog_form({"value": value}) == {"value": value}


@pytest.mark.parametrize("origin", ["llm", "together", "autonomous"])
def test_no_history_means_no_carried_authority(origin: str) -> None:
    plan = Plan.model_validate({**_human_plan().model_dump(), "origin": origin})
    _, errors = authorize_revision(plan, RunConfig())
    assert len(errors) == len(ref_sites(plan))


def test_typed_change_to_human_ref_is_not_carried() -> None:
    r0 = _with_tool(_human_plan(), tool=Ref(path="_fakes.import_canary:Tool", config={"x": True}))
    r1 = stamp(
        _with_tool(r0, tool=Ref(path="_fakes.import_canary:Tool", config={"x": 1})),
        origin="autonomous",
        parent=r0,
    )
    _, errors = authorize_revision(r1, RunConfig(), history=[r0])
    assert len(errors) == 1
    assert "not in the planner catalog" in errors[0]
    assert "_fakes.import_canary" not in sys.modules


def test_nested_noncatalog_ref_is_checked_without_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "nested-import"
    monkeypatch.setenv("SWARM_ARENA_TEST_CANARY", str(marker))
    nested = {"members": [{"path": "_fakes.import_canary:Tool"}]}
    entry = CatalogEntry(
        name="container", kind="tool", ref="absent:Container", description="d", config=nested
    )
    r0 = _human_plan()
    child = stamp(
        _with_tool(r0, tool=Ref(path=entry.ref, config=nested)), origin="autonomous", parent=r0
    )
    _, errors = authorize_revision(child, _cfg(PlannerCatalog(entries=[entry])), history=[r0])
    assert len(errors) == 1
    assert errors[0].startswith("/teams/0/agents/0/tool/config/members/0: ")
    assert "not in the planner catalog" in errors[0]
    assert not marker.exists()


def test_trusted_pairs_come_from_all_history_sites_and_depths() -> None:
    r0 = _human_plan().model_copy(
        update={
            "checks": [
                Ref(
                    path="human:Check",
                    config={"nested": [{"path": "human:Nested", "config": {"files": _files()}}]},
                )
            ]
        }
    )
    r1 = stamp(r0.model_copy(update={"checks": []}), origin="autonomous", parent=r0)
    # Manifest trust deliberately applies even when copied to another component.
    r2 = stamp(_with_tool(r1, tool=_runner_ref(files=_files())), origin="autonomous", parent=r1)
    restored_history = [Plan.model_validate_json(p.model_dump_json()) for p in [r0, r1]]
    result, errors = authorize_revision(r2, _cfg(CAT_FILES), history=restored_history)
    assert errors == []
    assert result.teams[0].agents[0].tool.config["files"] == _files()


@pytest.mark.parametrize("changed", [_files(globs=["problem/*.py"]), _files("cd")])
def test_manifest_trust_requires_both_globs_and_manifest(changed: dict[str, Any]) -> None:
    r0 = _with_tool(_human_plan(), tool=_runner_ref(files=_files()))
    child = stamp(_with_tool(r0, tool=_runner_ref(files=changed)), origin="autonomous", parent=r0)
    result, _ = authorize_revision(child, _cfg(CAT_FILES), history=[r0])
    assert result.teams[0].agents[0].tool.config["files"] == changed["globs"]


def test_nested_manifests_are_stripped_and_trusted_ones_preserved() -> None:
    member = {"path": "absent:Child", "config": {"files": ["problem/run.py"]}}
    root_config = {"a/b~c": [member], "deeper": [{"files": ["problem/run.py"]}]}
    catalog = PlannerCatalog(
        entries=[
            CatalogEntry(
                name="root", kind="tool", ref="absent:Root", description="d", config=root_config
            ),
            CatalogEntry(
                name="child",
                kind="tool",
                ref="absent:Child",
                description="d",
                config=member["config"],
            ),
        ]
    )
    snapshot = {
        "a/b~c": [{"path": "absent:Child", "config": {"files": _files()}}],
        "deeper": [{"files": _files("cd")}],
    }
    r0 = _human_plan().model_copy(
        update={"policy": Ref(path="human:Policy", config={"files": _files()})}
    )
    child = stamp(
        _with_tool(r0, tool=Ref(path="absent:Root", config=snapshot)),
        origin="autonomous",
        parent=r0,
    )
    result, errors = authorize_revision(child, _cfg(catalog), history=[r0])
    assert errors == []
    assert result.teams[0].agents[0].tool.config == {
        "a/b~c": [{"path": "absent:Child", "config": {"files": _files()}}],
        "deeper": [{"files": ["problem/run.py"]}],
    }
    assert child.teams[0].agents[0].tool.config == snapshot
    Plan.model_validate_json(result.model_dump_json())


@pytest.mark.parametrize("converge", [False, True])
def test_manifest_stripping_reaches_effective_override_storage(converge: bool) -> None:
    # Catalog loading permits object-shaped free JSON only as fixed data, so this
    # override is rejected but its untrusted manifest must still be discarded.
    r0 = _with_tool(_human_plan(), tool=_runner_ref(files=_files()))
    override = RoundSpec(tool_overrides={"files": _files("cd")})
    if converge:
        child = r0.model_copy(
            update={"converge": ConvergeSpec(mode="team", team=r0.teams[0], round=override)}
        )
    else:
        child = r0.model_copy(update={"rounds": [override]})
    child = stamp(child, origin="autonomous", parent=r0)
    result, errors = authorize_revision(child, _cfg(CAT_FILES), history=[r0])
    assert errors
    round_spec = result.converge.round if converge else result.rounds[0]
    assert round_spec is not None
    assert round_spec.tool_overrides["files"] == ["problem/run.py"]
    assert override.tool_overrides["files"] == _files("cd")
    assert result.teams[0].agents[0].tool.config["files"] == _files()


def test_unknown_nested_pointer_keys_are_hidden() -> None:
    r0 = _human_plan()
    child = stamp(
        _with_tool(
            r0,
            tool=Ref(
                path="swarm_arena:ScriptTool",
                config={
                    **FIXED,
                    "SECRET_KEY": [{"path": "absent:Tool"}],
                },
            ),
        ),
        origin="autonomous",
        parent=r0,
    )
    _, errors = authorize_revision(child, _cfg(CAT), history=[r0])
    assert errors
    assert_secret_absent(PlanError(errors), "SECRET_KEY")
    assert any("/config/<field>/0: " in error for error in errors)


def test_new_agent_requires_both_tool_and_manager_entries() -> None:
    r0 = _human_plan()
    added = AgentSpec(
        name="new",
        rationale="Try another approach",
        tool=_runner_ref(files=_files()),
        manager=Ref(path="absent:Manager"),
    )
    child = stamp(
        r0.model_copy(
            update={
                "teams": [
                    r0.teams[0].model_copy(
                        update={"agents": [*r0.teams[0].agents, added]},
                    )
                ]
            }
        ),
        origin="autonomous",
        parent=r0,
    )
    _, errors = authorize_revision(child, _cfg(CAT_FILES), history=[r0])
    assert len(errors) == 1
    assert errors[0].startswith("/teams/0/agents/1/manager: ")
    manager = CatalogEntry(name="manager", kind="manager", ref="absent:Manager", description="d")
    catalog = PlannerCatalog(entries=[*CAT_FILES.entries, manager])
    result, errors = authorize_revision(child, _cfg(catalog), history=[r0])
    assert errors == []
    assert result.teams[0].agents[1].tool.config["files"] == ["problem/run.py"]


def test_settable_keys_are_removed_only_at_top_level() -> None:
    entry = CAT.entries[0].model_copy(update={"config": {**FIXED, "nested": {"timeout_s": 1}}})
    catalog = PlannerCatalog(entries=[entry])
    site = _site({**entry.config, "timeout_s": 600})
    assert check_catalog(site, catalog, overrides={}) == []
    changed = _site({**site.ref.config, "nested": {"timeout_s": 2}})
    assert any("fixed field 'nested'" in e for e in check_catalog(changed, catalog, overrides={}))


def test_settable_values_cannot_switch_between_fixed_config_entries() -> None:
    first = CAT.entries[0]
    second = first.model_copy(
        update={"name": "longer", "settable": {"timeout_s": {"type": "number", "maximum": 5000}}}
    )
    catalog = PlannerCatalog(entries=[first, second])
    site = _site({**FIXED, "timeout_s": 5000})
    assert find_entry(site.ref, "tool", catalog) is first
    assert check_catalog(site, catalog, overrides={})


@pytest.mark.parametrize("converge", [False, True])
def test_noop_override_edit_loses_human_authority(converge: bool) -> None:
    r0 = _with_tool(_human_plan(), tool=Ref(path="swarm_arena:ScriptTool", config=FIXED))
    override = RoundSpec(tool_overrides={"editable": FIXED["editable"]})
    if converge:
        r0 = r0.model_copy(
            update={"converge": ConvergeSpec(mode="team", team=r0.teams[0], round=RoundSpec())}
        )
        changed = r0.model_copy(
            update={"converge": r0.converge.model_copy(update={"round": override})}
        )
    else:
        r0 = r0.model_copy(update={"rounds": [RoundSpec()]})
        changed = r0.model_copy(update={"rounds": [override]})
    child = stamp(changed, origin="autonomous", parent=r0)
    _, errors = authorize_revision(child, _cfg(CAT), history=[r0])
    assert any("override sets fixed field 'editable'" in error for error in errors)
    flags = parent_human([r0, child])
    effective = next(s for s in ref_sites(child) if "tool_overrides@" in s.pointer)
    assert flags[site_key(effective)] is False
    later = stamp(child, origin="autonomous", parent=child)
    assert authorize_revision(later, _cfg(CAT), history=[r0, child])[1]


@pytest.mark.parametrize("nested", [False, True])
def test_component_config_remains_a_mapping_when_its_fields_resemble_fileset(nested: bool) -> None:
    entry = CatalogEntry(
        name="globber", kind="tool", ref="absent:Globber", description="d", config=_files()
    )
    r0 = _human_plan()
    child = stamp(
        _with_tool(r0, tool=Ref(path=entry.ref, config=entry.config)),
        origin="autonomous",
        parent=r0,
    )
    entries = [entry]
    if nested:
        container = CatalogEntry(
            name="container",
            kind="tool",
            ref="absent:Container",
            description="d",
            config={"member": {"path": entry.ref, "config": entry.config}},
        )
        entries.append(container)
        child = _with_tool(child, tool=Ref(path=container.ref, config=container.config))
    result, errors = authorize_revision(child, _cfg(PlannerCatalog(entries=entries)), history=[r0])
    assert errors == []
    assert result.teams[0].agents[0].tool.config == child.teams[0].agents[0].tool.config


def _with_default_decision(
    site: RefSite, catalog: PlannerCatalog, *, overrides: dict[str, Any] | None = None
) -> list[str]:
    deferred: list[tuple[RefSite, CatalogEntry, tuple[str, ...]]] = []
    errors = check_catalog(site, catalog, overrides=overrides or {}, deferred=deferred)
    if errors:
        return errors
    return _check_catalog_defaults(deferred)[0]


@pytest.mark.parametrize("path", ["DefaultScriptJudge", "AliasedDefaultScriptJudge"])
def test_frozen_catalog_allows_only_omitted_fileset_defaults(path: str) -> None:
    entry = CatalogEntry(
        name="judge",
        kind="judge",
        ref=f"_fakes.admission:{path}",
        description="d",
        config={"command": ["python", "judge.py"]},
    )
    catalog = PlannerCatalog(entries=[entry])
    frozen = {"globs": ["judge.py"], "manifest": {"judge.py": "ab" * 32}}
    site = _site({**entry.config, "files": frozen}, path=entry.ref, kind="judge")
    assert check_catalog(site, catalog, overrides={})
    assert find_entry(site.ref, "judge", catalog, raw=True) is None
    assert find_entry(site.ref, "judge", catalog) is entry
    assert _with_default_decision(site, catalog) == []
    assert find_entry(site.ref, "judge", catalog, raw=True, defer_defaults=True) is entry
    assert site.ref.config["files"] == frozen
    assert entry.config == {"command": ["python", "judge.py"]}
    for extra in (
        {"files": {"globs": ["SECRET_VALUE"], "manifest": {"SECRET_VALUE": "ab" * 32}}},
        {"files": {**frozen, "editable": ["SECRET_VALUE"]}},
        {"files": frozen, "direction": "max"},  # Non-FileSet defaults remain fixed.
        {"files": frozen, "SECRET_KEY": "SECRET_VALUE"},
        {"files": frozen, "command": ["SECRET_VALUE"]},
    ):
        changed = _site({**entry.config, **extra}, path=entry.ref, kind="judge")
        errors = _with_default_decision(changed, catalog)
        assert errors
        assert find_entry(changed.ref, "judge", catalog) is None
        assert_secret_absent(PlanError(errors), "SECRET_VALUE")
        assert_secret_absent(PlanError(errors), "SECRET_KEY")
    assert _with_default_decision(site, catalog, overrides={"files": frozen})


@pytest.mark.parametrize(
    "supplied",
    [
        {"files": ["fixed.py"]},
        {"sources": ["fixed.py"]},
        {"bundle": {"sources": ["fixed.py"]}},
    ],
)
def test_explicit_catalog_alias_is_not_an_omitted_default(supplied: dict[str, Any]) -> None:
    entry = CatalogEntry(
        name="judge",
        kind="judge",
        ref="_fakes.admission:AliasedDefaultScriptJudge",
        description="d",
        config={"command": ["python", "judge.py"], **supplied},
    )
    site = _site(
        {**entry.config, "files": {"globs": ["judge.py"], "manifest": {"judge.py": "ab" * 32}}},
        path=entry.ref,
        kind="judge",
    )
    assert _with_default_decision(site, PlannerCatalog(entries=[entry]))


def test_default_resolution_requires_matching_human_catalog_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = CatalogEntry(
        name="judge",
        kind="judge",
        ref="_fakes.admission:DefaultScriptJudge",
        description="d",
        config={"command": ["python", "judge.py"]},
    )
    catalog = PlannerCatalog(entries=[entry])
    config = {**entry.config, "files": {"globs": ["judge.py"], "manifest": {}}}
    calls: list[str] = []
    original = Ref.load

    def tracked(ref: Ref) -> type:
        calls.append(ref.path)
        assert ref.path == entry.ref
        return original(ref)

    monkeypatch.setattr(Ref, "load", tracked)
    assert check_catalog(_site(config, path=entry.ref, kind="judge"), catalog, overrides={})
    ref = Ref(path=entry.ref, config=config)
    assert find_entry(ref, "judge", catalog, raw=True) is None
    assert find_entry(ref, "judge", catalog, raw=True, defer_defaults=True) is entry
    assert calls == []
    for path, kind in [("_fakes.import_canary:Tool", "judge"), (entry.ref, "tool")]:
        site = _site(config, path=path, kind=kind)  # ty: ignore[invalid-argument-type]
        assert _with_default_decision(site, catalog)
        assert find_entry(site.ref, site.kind, catalog) is None
    assert calls == []
    assert _with_default_decision(_site(config, path=entry.ref, kind="judge"), catalog) == []
    assert find_entry(ref, "judge", catalog) is entry
    assert calls
    assert set(calls) == {entry.ref}


def test_default_resolution_failure_keeps_catalog_error_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = CatalogEntry(
        name="judge",
        kind="judge",
        ref="_fakes.admission:DefaultScriptJudge",
        description="d",
        config={"command": ["python", "judge.py"]},
    )
    site = _site({**entry.config, "files": ["judge.py"]}, path=entry.ref, kind="judge")

    def broken(ref: Ref) -> type:
        raise RuntimeError("SECRET_VALUE")

    monkeypatch.setattr(Ref, "load", broken)
    errors = _with_default_decision(site, PlannerCatalog(entries=[entry]))
    assert errors == [f"{site.pointer}: field '<field>' is not in the planner catalog"]
    assert_secret_absent(PlanError(errors), "SECRET_VALUE")


def test_frozen_default_lookup_retains_first_fixed_match() -> None:
    first = CatalogEntry(
        name="first",
        kind="judge",
        ref="_fakes.admission:DefaultScriptJudge",
        description="d",
        config={"command": ["first"]},
    )
    second = first.model_copy(update={"name": "second", "config": {"command": ["second"]}})
    duplicate = second.model_copy(update={"name": "duplicate"})
    catalog = PlannerCatalog(entries=[first, second, duplicate])
    ref = Ref(path=first.ref, config={**second.config, "files": ["judge.py"]})
    assert find_entry(ref, "judge", catalog) is second


def test_default_decision_needs_no_component_construction_or_required_settable_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _fakes.admission import DefaultScriptJudge
    from pydantic import create_model

    schema = create_model("Config", __base__=DefaultScriptJudge.Config, tuning=(int, ...))
    monkeypatch.setattr(DefaultScriptJudge, "Config", schema)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("constructed a component during the typed default check")

    monkeypatch.setattr(DefaultScriptJudge, "__init__", forbidden)
    entry = CatalogEntry(
        name="judge",
        kind="judge",
        ref="_fakes.admission:DefaultScriptJudge",
        description="d",
        config={"command": ["score"]},
        settable={"tuning": {"type": "integer"}},
    )
    site = _site({**entry.config, "tuning": 1, "files": ["judge.py"]}, path=entry.ref, kind="judge")
    assert _with_default_decision(site, PlannerCatalog(entries=[entry])) == []
