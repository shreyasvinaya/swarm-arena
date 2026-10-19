from typing import Any, cast

import pytest
from pydantic import ValidationError

from swarm_arena._model.schema import compile_object_schema

SPEC_PARAMS = {
    "lr": {"type": "number", "minimum": 1e-5, "maximum": 1.0},
    "epochs": {"type": "integer", "maximum": 400},
}  # spec §4.12


def test_spec_params_example() -> None:
    M = compile_object_schema("Params", SPEC_PARAMS)
    assert M.model_validate({"lr": 1, "epochs": 400}).model_dump() == {"lr": 1.0, "epochs": 400}
    for bad in (
        {"lr": 0.1, "epochs": 401},
        {"lr": 0.0, "epochs": 1},
        {"epochs": 1},
        {"lr": 0.1, "epochs": "5"},
        {"lr": 0.1, "epochs": True},
        {"lr": 0.1, "epochs": 1, "zz": 1},
    ):
        with pytest.raises(ValidationError):
            M.model_validate(bad)


def test_enum_default_and_array_items() -> None:
    M = compile_object_schema(
        "P",
        {
            "mode": {"enum": ["a", "b"], "default": "a"},
            "xs": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "default": [],
            },
        },
    )
    assert M().model_dump() == {"mode": "a", "xs": []}
    assert M.model_validate({"xs": [1, 2]}).model_dump()["xs"] == [1, 2]
    with pytest.raises(ValidationError):
        M.model_validate({"mode": "c"})
    with pytest.raises(ValidationError):
        M.model_validate({"xs": [-1]})


@pytest.mark.parametrize(
    ("prop", "match"),
    [
        ({"type": "string", "format": "email"}, "unsupported JSON-schema keyword 'format'"),
        ({"type": "integer", "default": "x"}, "default"),
        ({"type": "string", "minimum": 1}, "minimum"),
        ({"description": "no type"}, "type or enum"),
        ({"type": "object"}, "type 'object'"),
    ],
)
def test_rejected_schemas(prop: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compile_object_schema("P", {"p": prop})


def test_bad_property_name() -> None:
    with pytest.raises(ValueError, match="_secret"):
        compile_object_schema("P", {"_secret": {"type": "string"}})


def test_string_constraints_boolean_and_untyped_array() -> None:
    M = compile_object_schema(
        "P",
        {
            "text": {
                "type": "string",
                "minLength": 2,
                "maxLength": 4,
                "pattern": "^[a-z]+$",
                "description": "Lowercase text.",
            },
            "flag": {"type": "boolean"},
            "xs": {"type": "array"},
        },
    )
    assert M.model_validate({"text": "ok", "flag": True, "xs": [1, "x"]}).model_dump()["xs"] == [
        1,
        "x",
    ]
    assert M.model_json_schema()["properties"]["text"]["description"] == "Lowercase text."
    for bad in ("a", "abcde", "AB", 12):
        with pytest.raises(ValidationError):
            M.model_validate({"text": bad, "flag": True, "xs": []})
    with pytest.raises(ValidationError):
        M.model_validate({"text": "ok", "flag": 1, "xs": []})
    with pytest.raises(ValidationError):
        M.model_validate({"text": "ok", "flag": True, "xs": ()})
    model = M.model_validate({"text": "ok", "flag": True, "xs": []})
    with pytest.raises(ValidationError):
        cast(Any, model).text = "no"


def test_exclusive_bounds_and_nested_items() -> None:
    M = compile_object_schema(
        "P",
        {
            "xs": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "exclusiveMaximum": 2,
                    },
                },
            }
        },
    )
    assert M.model_validate({"xs": [[1]]}).model_dump() == {"xs": [[1.0]]}
    for bad in ([[0]], [[2]], [["1"]], [[True]]):
        with pytest.raises(ValidationError):
            M.model_validate({"xs": bad})


def test_defaults_are_validated_and_copied() -> None:
    with pytest.raises(ValueError, match=r"P.xs: default .* does not match the schema"):
        compile_object_schema(
            "P", {"xs": {"type": "array", "items": {"type": "integer"}, "default": ["1"]}}
        )
    M = compile_object_schema("P", {"xs": {"type": "array", "default": []}})
    first = M()
    cast(Any, first).xs.append(1)
    assert M().model_dump() == {"xs": []}


@pytest.mark.parametrize(
    "prop",
    [
        {"type": "array", "items": {"type": "string", "format": "email"}},
        {"type": "integer", "minLength": 1},
        {"type": "boolean", "maximum": 1},
    ],
)
def test_rejects_nested_and_inapplicable_keywords(prop: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"P\.p"):
        compile_object_schema("P", {"p": prop})


def test_typed_enum_preserves_strict_type_and_constraints() -> None:
    M = compile_object_schema("P", {"n": {"type": "integer", "enum": [1, 2], "minimum": 2}})
    assert M.model_validate({"n": 2}).model_dump() == {"n": 2}
    for bad in (1, True, "2", 2.0, 3):
        with pytest.raises(ValidationError):
            M.model_validate({"n": bad})


def test_array_item_description() -> None:
    M = compile_object_schema(
        "P", {"xs": {"type": "array", "items": {"type": "string", "description": "An item."}}}
    )
    assert M.model_json_schema()["properties"]["xs"]["items"]["description"] == "An item."
