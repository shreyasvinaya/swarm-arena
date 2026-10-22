"""Exercise import-free reference traversal and human authority inheritance."""

from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena import AgentSpec, ConvergeSpec, Goal, PartSpec, Plan, Ref, RoundSpec, TeamSpec
from swarm_arena._planning.provenance import (
    diff_revision,
    is_nested_ref,
    ref_sites,
    site_key,
    stamp,
)


def _agent(name: str) -> AgentSpec:
    return AgentSpec(
        name=name,
        rationale="Explore",
        tool=Ref(path="unimportable:Tool", config={"x": 1}),
        manager=Ref(path="unimportable:Manager"),
    )


def _plan() -> Plan:
    return Plan(
        goal=Goal(id="g", namespace="demo"),
        teams=[TeamSpec(name="t1", rationale="Explore", agents=[_agent("a")])],
        checks=[Ref(path="unimportable:Check", name="valid")],
        judges=[Ref(path="unimportable:Judge", name="score")],
        rounds=[RoundSpec(tool_overrides={"x": 2})],
    )


def _add_agent(plan: Plan, team: str, name: str) -> Plan:
    return plan.model_copy(
        update={
            "teams": [
                t.model_copy(update={"agents": [*t.agents, _agent(name)]}) if t.name == team else t
                for t in plan.teams
            ]
        }
    )


def _plan_with_everything() -> Plan:
    plan = _add_agent(_plan(), "t1", "b")
    return plan.model_copy(
        update={
            "judges": [
                Ref(
                    path="unimportable:Lexicographic",
                    config={
                        "judges": [
                            {"path": "unimportable:Judge"},
                            {"path": "unimportable:OtherJudge"},
                        ]
                    },
                )
            ],
            "policy": Ref(
                path="unimportable:Chain", config={"policies": [{"path": "unimportable:Policy"}]}
            ),
            "parts": [
                PartSpec(
                    name="part",
                    goal=Goal(id="p", namespace="demo"),
                    checks=[Ref(path="unimportable:Check")],
                    judges=[Ref(path="unimportable:Judge")],
                )
            ],
            "converge": ConvergeSpec(
                mode="team",
                team=TeamSpec(name="join", rationale="Combine", agents=[_agent("a")]),
                round=RoundSpec(tool_overrides={"x": 3}),
            ),
        }
    )


def test_ref_sites_cover_every_depth() -> None:
    ptrs = {s.pointer: s.kind for s in ref_sites(_plan_with_everything())}
    assert (  # noqa: PT018
        ptrs["/teams/0/agents/1/tool"] == "tool" and ptrs["/teams/0/agents/1/manager"] == "manager"
    )
    assert ptrs["/judges/0/config/judges/1"] == "judge"
    assert ptrs["/policy/config/policies/0"] == "policy"
    assert (  # noqa: PT018
        ptrs["/parts/0/checks/0"] == "check" and ptrs["/converge/team/agents/0/tool"] == "tool"
    )
    assert "/rounds/0/tool_overrides@/teams/0/agents/1/tool" in ptrs
    assert ptrs["/parts/0/judges/0"] == "judge"
    assert ptrs["/converge/team/agents/0/manager"] == "manager"
    assert ptrs["/converge/round/tool_overrides@/converge/team/agents/0/tool"] == "tool"
    assert len(ptrs) == 17


def test_diff_marks_unchanged_human_sites_carried() -> None:
    r0 = stamp(_plan(), origin="human", parent=None)
    r1 = stamp(_add_agent(r0, "t1", "new"), origin="autonomous", parent=r0)
    human = {site_key(s): True for s in ref_sites(r0)}
    carried = {s.pointer: s.carried_human for s in diff_revision(r0, r1, parent_human=human)}
    assert (  # noqa: PT018
        carried["/judges/0"] is True and carried["/teams/0/agents/1/tool"] is False
    )


def test_changed_override_is_not_carried() -> None:
    r0 = _plan_with_everything()
    r1 = r0.model_copy(
        update={
            "rounds": [RoundSpec(tool_overrides={"x": 4})],
            "converge": r0.converge.model_copy(
                update={"round": RoundSpec(tool_overrides={"x": 4})}
            ),
        }
    )
    sites = diff_revision(r0, r1, parent_human={site_key(s): True for s in ref_sites(r0)})
    assert all(s.carried_human is ("tool_overrides@" not in s.pointer) for s in sites)


def test_override_sites_are_effective_refs_without_mutation() -> None:
    plan = _plan_with_everything()
    before = plan.model_dump()
    sites = {s.pointer: s for s in ref_sites(plan)}
    main = sites["/rounds/0/tool_overrides@/teams/0/agents/1/tool"]
    converge = sites["/converge/round/tool_overrides@/converge/team/agents/0/tool"]
    assert main.ref.config == {"x": 2}
    assert converge.ref.config == {"x": 3}
    assert main.ref.path == plan.teams[0].agents[1].tool.path
    assert plan.model_dump() == before


def test_stamp_sets_revision_and_parent() -> None:
    r0 = stamp(
        _plan().model_copy(update={"origin": "human", "revision": 9}), origin="llm", parent=None
    )
    r1 = stamp(r0, origin="autonomous", parent=r0)
    assert (r0.origin, r0.revision, r0.parent_revision) == ("llm", 0, None)
    assert (r1.origin, r1.revision, r1.parent_revision) == ("autonomous", 1, 0)


def test_stamp_keeps_base_dir_and_does_not_mutate(tmp_path: Path) -> None:
    original = Plan._from_data(_plan().model_dump(), base_dir=tmp_path)
    child = stamp(original, origin="together", parent=original.model_copy(update={"revision": 7}))
    assert child.base_dir == original.base_dir == tmp_path
    assert (child.origin, child.revision, child.parent_revision) == ("together", 8, 7)
    assert (original.origin, original.revision, original.parent_revision) == ("human", 0, None)


@pytest.mark.parametrize(("left", "right"), [(True, 1), (False, 0), (1, 1.0), (0, 0.0)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("location", ["base", "nested", "round", "converge"])
def test_provenance_distinguishes_json_types(
    left: Any,
    right: Any,
    reverse: bool,
    location: str,
) -> None:
    if reverse:
        left, right = right, left

    def changed(value: Any) -> Plan:
        plan = _plan_with_everything()
        config = {"deep": [{"x": value}]}
        if location == "round":
            return plan.model_copy(update={"rounds": [RoundSpec(tool_overrides=config)]})
        if location == "converge":
            return plan.model_copy(
                update={
                    "converge": plan.converge.model_copy(
                        update={"round": RoundSpec(tool_overrides=config)}
                    )
                }
            )
        if location == "nested":
            config = {"wrapped": [{"path": "unimportable:Judge", "config": config}]}
        return plan.model_copy(update={"judges": [Ref(path="unimportable:Judge", config=config)]})

    parent, child = changed(left), changed(right)
    diffs = diff_revision(
        parent, child, parent_human={site_key(s): True for s in ref_sites(parent)}
    )
    prefix = {
        "base": "/judges",
        "nested": "/judges",
        "round": "/rounds",
        "converge": "/converge/round",
    }[location]
    targets = [s for s in diffs if s.pointer.startswith(prefix)]
    assert targets
    assert all(not s.carried_human for s in targets)


def test_missing_false_and_discontinued_authority_fail_closed() -> None:
    plan = _plan()
    assert not any(s.carried_human for s in diff_revision(None, plan, parent_human={}))
    assert not any(s.carried_human for s in diff_revision(plan, plan, parent_human={}))
    human = {site_key(s): True for s in ref_sites(plan)}
    first = diff_revision(plan, plan, parent_human=human)
    second = diff_revision(plan, plan, parent_human={site_key(s): s.carried_human for s in first})
    assert all(s.carried_human for s in second)
    false = {site_key(s): False for s in first}
    assert not any(s.carried_human for s in diff_revision(plan, plan, parent_human=false))
    removed = plan.model_copy(update={"checks": []})
    assert not next(
        s for s in diff_revision(removed, plan, parent_human=human) if s.pointer == "/checks/0"
    ).carried_human


def test_site_keys_stable_across_reordering_scopes_and_roundtrip() -> None:
    parent = _plan_with_everything()
    extra = parent.teams[0].model_copy(update={"name": "t2"})
    parent = parent.model_copy(
        update={
            "teams": [*parent.teams, extra],
            "checks": [
                Ref(path="unimportable:Check", name="one"),
                Ref(path="unimportable:Check", name="two"),
            ],
        }
    )
    child = parent.model_copy(
        update={
            "teams": [
                t.model_copy(update={"agents": list(reversed(t.agents))})
                for t in reversed(parent.teams)
            ],
            "checks": list(reversed(parent.checks)),
        }
    )
    human = {site_key(s): True for s in ref_sites(parent)}
    assert all(s.carried_human for s in diff_revision(parent, child, parent_human=human))
    restored = Plan.model_validate_json(parent.model_dump_json())
    assert [site_key(s) for s in ref_sites(parent)] == [site_key(s) for s in ref_sites(restored)]
    assert len(human) == len(ref_sites(parent))


def test_named_path_and_occurrence_keys_never_collide() -> None:
    refs = [
        Ref(path="unimportable:Judge", name=name)
        for name in [
            "a",
            "a",
            "a#2",
            "a/b",
            "a:b",
            "unimportable:Judge",
            None,
        ]
    ]
    plan = _plan().model_copy(update={"judges": refs, "checks": refs})
    sites = ref_sites(plan)
    assert len({site_key(s) for s in sites}) == len(sites)
    parent = plan.model_copy(update={"judges": [refs[0], refs[0], refs[2]]})
    child = plan.model_copy(update={"judges": [refs[2].model_copy(update={"config": {"x": 1}})]})
    diffs = diff_revision(
        parent, child, parent_human={site_key(s): True for s in ref_sites(parent)}
    )
    assert not next(s for s in diffs if s.pointer == "/judges/0").carried_human


def test_nested_paths_escape_pointer_tokens_without_collisions() -> None:
    leaf = {"path": "unimportable:Judge"}
    config = {
        "a/b": leaf,
        "a": {"b": leaf},
        "a~1b": leaf,
        "": [leaf],
        "deeper": [{"box": [{"path": "unimportable:Judge", "config": {"child": leaf}}]}],
    }
    plan = _plan().model_copy(update={"judges": [Ref(path="unimportable:Judge", config=config)]})
    sites = {s.pointer: s for s in ref_sites(plan)}
    for pointer in [
        "/judges/0/config/a~1b",
        "/judges/0/config/a/b",
        "/judges/0/config/a~01b",
        "/judges/0/config//0",
        "/judges/0/config/deeper/0/box/0/config/child",
    ]:
        assert pointer in sites
    assert len({site_key(s) for s in sites.values()}) == len(sites)
    root = site_key(sites["/judges/0"])
    assert site_key(sites["/judges/0/config/a~1b"]) == root + "/config/a~1b"


@pytest.mark.parametrize(
    "path",
    [
        "demo:Thing",
        "demo.sub:Outer.Inner",
        "_demo:_Thing",
        "aé:bλ",
        "review_a\u0301:Policy",
        "demo:Poli\u0301cy",
        "demo.su\u0301b:Outer.Inne\u0301r",
    ],
)
def test_nested_ref_accepts_full_ref_path(path: str) -> None:
    Ref(path=path)
    assert is_nested_ref({"path": path})
    assert is_nested_ref({"path": path, "config": {}, "name": "alias"})


@pytest.mark.parametrize("path", ["review_a\u0301:Policy", "demo:Poli\u0301cy"])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_combining_mark_nested_sites_and_revision_diff(path: str, depth: int) -> None:
    def plan(value: int) -> Plan:
        nested = Ref(path=path, config={"fixed": value}).model_dump(exclude_none=True)
        for _ in range(depth):
            nested = {"box": [nested]}
        return Plan(
            goal=Goal(id="g", namespace="demo"),
            teams=[],
            judges=[],
            policy=Ref(path="demo:Chain", config={"policies": [nested]}),
        )

    parent = stamp(plan(1), origin="human", parent=None)
    pointer = "/policy/config/policies/0" + "/box/0" * depth
    sites = {site.pointer: site for site in ref_sites(parent)}
    assert set(sites) == {"/policy", pointer}
    assert sites[pointer].ref.path == path
    assert sites[pointer].kind == "policy"
    human = {site_key(site): True for site in sites.values()}
    for value in (1, 2):
        child = stamp(plan(value), origin="autonomous", parent=parent)
        diffs = {site.pointer: site for site in diff_revision(parent, child, parent_human=human)}
        assert set(diffs) == set(sites)
        assert diffs[pointer].ref.config == {"fixed": value}
        assert diffs[pointer].carried_human is (value == 1)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "demo:Thing",
        {},
        {"path": 1},
        {"path": "demo"},
        {"path": "demo:Thing:Other"},
        {"path": "demo:Thing\n"},
        {"path": "demo:Thing/other"},
        {"path": "demo:Thing", "extra": True},
        {"path": "0demo:Thing"},
        {"path": "demo.0thing:Thing"},
    ],
)
def test_nested_ref_rejects_nonrefs(value: Any) -> None:
    assert not is_nested_ref(value)


def test_nonref_mapping_still_traverses_nested_values() -> None:
    config = {"box": {"path": "demo:Thing", "extra": [{"path": "demo:Nested"}]}}
    sites = ref_sites(
        _plan().model_copy(update={"judges": [Ref(path="demo:Judge", config=config)]})
    )
    pointers = {s.pointer for s in sites}
    assert "/judges/0/config/box" not in pointers
    assert "/judges/0/config/box/extra/0" in pointers


@pytest.mark.parametrize("module_name", ["provenance_canary", "provenance_canary_a\u0301"])
def test_ref_operations_never_import_even_changed_human_canary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
) -> None:
    marker = tmp_path / "imported"
    (tmp_path / f"{module_name}.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('imported')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    parent = _plan().model_copy(
        update={
            "judges": [
                Ref(
                    path=f"{module_name}:Judge",
                    config={
                        "nested": {"path": f"{module_name}:Judg\u0301e", "config": {"x": True}}
                    },
                )
            ]
        }
    )
    child = parent.model_copy(
        update={
            "judges": [
                Ref(
                    path=f"{module_name}:Judge",
                    config={"nested": {"path": f"{module_name}:Judg\u0301e", "config": {"x": 1}}},
                )
            ]
        }
    )

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("provenance must not resolve or import components")

    with monkeypatch.context() as patch:
        for attr in ["component_name", "load", "build", "_resolve_config"]:
            patch.setattr(Ref, attr, unexpected)
        patch.setattr("swarm_arena._model.ref.load_object", unexpected)
        patch.setattr("importlib.import_module", unexpected)
        human = {site_key(s): True for s in ref_sites(parent)}
        diffs = diff_revision(parent, child, parent_human=human)
        assert not next(s for s in diffs if s.pointer == "/judges/0").carried_human
        nested = next(s for s in diffs if s.pointer == "/judges/0/config/nested")
        assert not nested.carried_human
    assert not marker.exists()


def test_malformed_nested_ref_diagnostics_hide_values() -> None:
    secret = "f7cfa2a9-825a-4d74-secret"
    plan = _plan().model_copy(
        update={
            "judges": [
                Ref(
                    path="demo:Judge",
                    config={
                        "nested": {"path": "demo:Judge", "config": secret},
                    },
                )
            ]
        }
    )
    with pytest.raises(ValidationError) as caught:
        ref_sites(plan)
    assert_secret_absent(caught.value, secret, sanitized=True)


def test_detach_and_stamp_keep_planner_metadata(tmp_path: Path) -> None:
    from unittest.mock import Mock

    from swarm_arena._engine.run import mark_planning_session
    from swarm_arena._planning.provenance import _detach_plan

    source = Plan.loads(_plan().dumps(), base_dir=tmp_path)
    marked = mark_planning_session(source, Mock(run_id="r-p", call_ids=["llm-1", "llm-2"]))
    for result in (
        _detach_plan(marked),
        stamp(marked, origin="llm", parent=None),
        marked.model_copy(deep=True),
    ):
        assert result.base_dir == tmp_path
        assert result.planning_session == "r-p"
        assert result.planner_call_ids == ("llm-1", "llm-2")


@pytest.mark.parametrize(
    ("field", "bad"),
    [("_planning_session", "../secret-sentinel"), ("_planner_call_ids", [object()])],
)
def test_detach_validates_private_planner_metadata(field: str, bad: Any) -> None:
    from swarm_arena._planning.provenance import _detach_plan

    plan = _plan()
    setattr(plan, field, bad)
    with pytest.raises(ValueError, match=r"invalid planner|invalid planning") as caught:
        _detach_plan(plan)
    assert_secret_absent(caught.value, "secret-sentinel")
