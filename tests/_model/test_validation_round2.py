"""Pin restored identifiers against b4ed078 and the Round 2 secrecy rulings."""

import json
import traceback
from pathlib import Path
from typing import Any, ClassVar

import pytest
from _fakes.tools import GOAL, cand
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
)

from swarm_arena._errors import ConfigError
from swarm_arena._model.base import _SafeModel
from swarm_arena._model.catalog import PlannerCatalog
from swarm_arena._model.plan import Plan
from swarm_arena._model.ref import Ref, _ConfigInputModel
from swarm_arena._model.results import Candidate
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.schema import compile_object_schema
from swarm_arena._model.tooling import ToolOutput
from swarm_arena._scoring.script_check import _CheckResultFile
from swarm_arena._tooling.result_file import ToolResultFile, _validation_reason

# Captured by importing src/ from a temporary b4ed078 worktree, then removing it.
# RunConfig's old custom code becomes the built-in code; Candidate's old flattened
# message becomes structured. Ref's data-key segments are deliberately redacted.
NESTED_SNAPSHOTS = [
    (
        Plan,
        {
            "goal": GOAL,
            "teams": [
                {
                    "name": "t",
                    "rationale": "r",
                    "agents": [
                        {
                            "name": "a",
                            "rationale": "r",
                            "manager": {"path": "m:M"},
                            "tool": {"path": "m:T"},
                            "replicatez": "snapshot_secret",
                        }
                    ],
                }
            ],
            "judges": [],
        },
        "1 validation error for Plan\nteams.0.agents.0.replicatez\n"
        "  extra_forbidden [type=extra_forbidden]",
    ),
    (
        RunConfig,
        {"secrets": {"allow": ["ok", 13]}},
        "1 validation error for RunConfig\nsecrets.allow.1\n  string_type [type=string_type]",
    ),
    (
        Ref,
        {"path": "m:T", "config": {123: "snapshot_secret"}},
        "1 validation error for Ref\nconfig.<…>.<…>\n  string_type [type=string_type]",
    ),
    (
        Candidate,
        {
            **cand().model_dump(),
            "output": {"status": "ok", "usage": {"input_tokens": "snapshot_secret"}},
        },
        "1 validation error for Candidate\noutput.usage.input_tokens\n"
        "  int_parsing [type=int_parsing]",
    ),
]


@pytest.mark.parametrize(("model", "data", "expected"), NESTED_SNAPSHOTS)
def test_nested_path_snapshot(model: type[BaseModel], data: dict[str, Any], expected: str) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    assert str(caught.value) == expected
    assert "snapshot_secret" not in caught.value.json()


@pytest.mark.parametrize("entry", ["constructor", "python", "json", "strings"])
def test_outer_model_owns_nested_unknown_policy(entry: str) -> None:
    class Human(_SafeModel):
        _unknown_fields: ClassVar = "show"
        ref: Ref

    class Runtime(_ConfigInputModel):
        human: Human

    def validate(model: type[BaseModel], data: dict[str, Any]) -> BaseModel:
        if entry == "constructor":
            return model(**data)
        if entry == "json":
            return model.model_validate_json(json.dumps(data))
        if entry == "strings":
            return model.model_validate_strings(data)
        return model.model_validate(data)

    data = {"ref": {"path": "m:T", "confg": "outer_secret"}}
    for model, payload, expected in (
        (Human, data, ("ref", "confg")),
        (Runtime, {"human": data}, ("human", "ref", "<field>")),
        (Ref, data["ref"], ("<field>",)),
    ):
        with pytest.raises(ValidationError) as caught:
            validate(model, payload)
        assert caught.value.errors()[0]["loc"] == expected
        assert "outer_secret" not in caught.value.json()
    # An error must reset the context, including across different entry points.
    assert Human(ref={"path": "m:T"}).ref.path == "m:T"


def test_empty_locations_and_loader_format(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as caught:
        Ref.model_validate_json('{"secret":')
    assert caught.value.errors()[0]["loc"] == ()
    text = 'judges = []\n[goal]\nid = "g"\nnamespace = "n"\n'
    text += '[[teams]]\nname = "t"\nrationale = "r"\nagents = []\n' * 2
    with pytest.raises(ConfigError, match=r"^plan: Value error, duplicate team names$"):
        Plan.loads(text, base_dir=tmp_path)
    path = tmp_path / "run.toml"
    path.write_text('[cache]\nbackend = "s3"\n')
    with pytest.raises(ConfigError) as loaded:
        RunConfig.load(path)
    assert str(loaded.value) == f"{path}: cache: s3 backend requires s3_path"
    path.write_text("schema_version = 99\n")
    with pytest.raises(ConfigError) as loaded:
        RunConfig.load(path)
    assert (
        str(loaded.value)
        == f"{path}: run_config: schema_version is newer than this swarm-arena supports"
    )


def test_core_schema_refs_match_unwrapped_twins_and_reuse() -> None:
    class Leaf(_SafeModel):
        count: int

    safe = create_model("Twin", __base__=_SafeModel, first=(Leaf, ...), second=(Leaf, ...))
    plain_leaf = create_model("Leaf", __config__=ConfigDict(extra="forbid"), count=(int, ...))
    plain = create_model(
        "Twin",
        __config__=ConfigDict(extra="forbid"),
        first=(plain_leaf, ...),
        second=(plain_leaf, ...),
    )
    assert safe.model_json_schema() == plain.model_json_schema()
    schema = safe.model_json_schema()
    assert schema["properties"]["first"] == {"$ref": "#/$defs/Leaf"}
    assert schema["properties"]["second"] == {"$ref": "#/$defs/Leaf"}


def test_recursive_model_keeps_definitions_and_nested_locations() -> None:
    class Node(_SafeModel):
        count: int
        children: list["Node"] = []

    schema = Node.model_json_schema()
    assert schema["$ref"] == "#/$defs/Node"
    assert schema["$defs"]["Node"]["properties"]["children"]["items"]["$ref"] == "#/$defs/Node"
    assert Node(count=1, children=[{"count": 2}]).children[0].count == 2
    with pytest.raises(ValidationError) as caught:
        Node(count=1, children=[{"count": "recursive_secret"}])
    assert caught.value.errors()[0]["loc"] == ("children", 0, "count")
    assert "recursive_secret" not in caught.value.json()


@pytest.mark.parametrize(
    "alias",
    [AliasPath("payload", 0, "count"), AliasChoices("amount", AliasPath("payload", 0, "count"))],
)
def test_alias_paths_retain_all_schema_segments(alias: AliasPath | AliasChoices) -> None:
    class Inner(_SafeModel):
        count: int = Field(validation_alias=alias)

    class Outer(_SafeModel):
        items: tuple[Inner, ...]

    with pytest.raises(ValidationError) as caught:
        Outer(items=[{"payload": [{"count": "alias_secret"}]}])
    assert caught.value.errors()[0]["loc"] == ("items", 0, "payload", 0, "count")


def test_catalog_error_names_custom_human_only_field(monkeypatch: pytest.MonkeyPatch) -> None:
    class Component:
        _human_only = frozenset({"private_option"})

    monkeypatch.setattr("swarm_arena._model.catalog.load_object", lambda _: Component)
    entries = [
        {
            "name": "entry-id",
            "kind": "tool",
            "ref": "m:T",
            "description": "secret",
            "settable": {"private_option": {"type": "string"}},
        }
    ]
    with pytest.raises(ValidationError) as caught:
        PlannerCatalog(entries=entries)
    assert caught.value.errors()[0]["msg"] == (
        "catalog entry 'entry-id': human-only field 'private_option' cannot be settable"
    )
    assert "secret" not in caught.value.json()


def test_schema_compiler_is_safe_and_namespaces_are_explicit() -> None:
    model = compile_object_schema("Params", {"count": {"type": "integer"}})
    assert issubclass(model, _SafeModel)
    assert model.model_config["protected_namespaces"] == ("model_validate", "model_dump")
    with pytest.raises(ValidationError) as caught:
        model.model_validate_json('{"count":"compiled_secret"}')
    assert "compiled_secret" not in caught.value.json()


@pytest.mark.parametrize(("model", "field"), [(ToolOutput, "data"), (_CheckResultFile, "details")])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
def test_finite_json_fixed_reason_and_secrecy(model: type[BaseModel], field: str, bad: Any) -> None:
    data = {
        field: {"json_secret": bad},
        **({"status": "ok"} if model is ToolOutput else {"passed": True}),
    }
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    assert caught.value.errors()[0]["msg"] == f"{field} must be finite JSON"
    assert "json_secret" not in caught.value.json()
    assert "json_secret" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ({"usage": {"input_tokens": "result_secret"}}, "usage.input_tokens: int_parsing"),
        ({"files": [13]}, "files.0: string_type"),
        ({"usgae": "result_secret"}, "<field>: extra_forbidden"),
    ],
)
def test_result_reason_uses_core_locations(data: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValidationError) as caught:
        ToolResultFile.model_validate({"data": {}, **data})
    assert _validation_reason(caught.value) == "swarm/result.json: " + reason


def test_union_branch_fields_remain_identifiers() -> None:
    class Left(_SafeModel):
        count: int

    class Right(_SafeModel):
        label: str

    class Either(_SafeModel):
        value: Left | Right

    with pytest.raises(ValidationError) as caught:
        Either(value={"count": "union_secret"})
    assert caught.value.errors()[0]["loc"] == ("value", "<…>", "count")
    assert "union_secret" not in caught.value.json()


def test_config_alias_takes_precedence_over_colliding_field_name() -> None:
    from swarm_arena._model.ref import _validate_config

    class Config(BaseModel):
        credentials: list[int] = Field(alias="items")
        lookup: dict[int, int] = Field(alias="credentials")

    with pytest.raises(ConfigError) as caught:
        _validate_config(Config, {"items": [], "credentials": {12345: "alias_secret"}}, where="m:T")
    assert str(caught.value) == "m:T: credentials.<…>: int_parsing"


@pytest.mark.parametrize("loc_by_alias", [False, True])
@pytest.mark.parametrize("by_alias", [False, True])
def test_alias_collision_never_exposes_integer_data_keys(
    loc_by_alias: bool, by_alias: bool
) -> None:
    class Config(_SafeModel):
        model_config = ConfigDict(loc_by_alias=loc_by_alias)
        credentials: dict[int, int] = Field(alias="items")
        lookup: list[int] = Field(alias="credentials")

    data = (
        {"items": {739201: "alias_secret"}, "credentials": []}
        if by_alias
        else {"credentials": {739201: "alias_secret"}, "lookup": []}
    )
    with pytest.raises(ValidationError) as caught:
        Config.model_validate(data, by_alias=by_alias, by_name=not by_alias)
    assert caught.value.errors()[0]["loc"] == (
        "items" if by_alias and loc_by_alias else "credentials",
        "<…>",
    )
    assert "739201" not in caught.value.json()
    assert "alias_secret" not in caught.value.json()


def test_configured_validation_by_name_redacts_colliding_mapping_keys() -> None:
    class Config(_SafeModel):
        model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)
        credentials: dict[int, int] = Field(alias="items")
        lookup: list[int] = Field(alias="credentials")

    with pytest.raises(ValidationError) as caught:
        Config.model_validate({"credentials": {739201: "alias_secret"}, "lookup": []})
    assert caught.value.errors()[0]["loc"] == ("credentials", "<…>")
    assert "739201" not in caught.value.json()
