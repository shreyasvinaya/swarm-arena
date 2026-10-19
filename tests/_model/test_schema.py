import re
import traceback
from typing import Any, cast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import InitErrorDetails

from swarm_arena._model.schema import _SchemaError, compile_object_schema

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
        ({"type": "object"}, "unsupported type"),
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
    with pytest.raises(ValueError, match=r"P\.xs: default does not match the schema"):
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


@pytest.mark.parametrize(
    ("kind", "choice"),
    [("integer", True), ("boolean", 1), ("integer", 1.0), ("number", True)],
)
@pytest.mark.parametrize("with_default", [False, True])
def test_typed_enum_rejects_members_of_the_wrong_type(
    kind: str, choice: Any, *, with_default: bool
) -> None:
    prop = {"type": kind, "enum": [choice]}
    if with_default:
        prop["default"] = choice
    with pytest.raises(ValueError, match=r"P\.p: enum member does not match the declared type"):
        compile_object_schema("P", {"p": prop})


@pytest.mark.parametrize(
    ("kind", "choice", "other"),
    [
        (None, True, 1),
        (None, 1, True),
        (None, 1, 1.0),
        (None, 1.0, 1),
        ("boolean", True, 1),
        ("integer", 1, True),
        ("integer", 1, 1.0),
        ("number", 1.0, True),
    ],
)
def test_enum_membership_distinguishes_equal_values_of_different_types(
    kind: str | None, choice: Any, other: Any
) -> None:
    prop = {"enum": [choice]}
    if kind is not None:
        prop["type"] = kind
    M = compile_object_schema("P", {"p": prop})
    result = M.model_validate({"p": choice}).model_dump()["p"]
    assert type(result) is type(choice)
    with pytest.raises(ValidationError):
        M.model_validate({"p": other})
    with pytest.raises(ValueError, match=r"P\.p: default does not match the schema"):
        compile_object_schema("P", {"p": {**prop, "default": other}})


@pytest.mark.parametrize("choice", [True, 1, 1.0])
def test_mixed_enum_preserves_the_input_type(choice: Any) -> None:
    M = compile_object_schema("P", {"p": {"enum": [True, 1, 1.0], "default": choice}})
    assert type(M.model_validate({"p": choice}).model_dump()["p"]) is type(choice)
    assert type(M().model_dump()["p"]) is type(choice)


def test_invalid_enum_member_errors_hide_secret_values_and_exception_chain() -> None:
    secret = "synthetic-secret-enum-member-4827"
    with pytest.raises(ValueError, match=r"P\.p: enum member") as exc_info:
        compile_object_schema("P", {"p": {"type": "integer", "enum": [secret]}})
    assert secret not in str(exc_info.value)
    assert secret not in "".join(traceback.format_exception(exc_info.value))
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__


@pytest.mark.parametrize(
    ("kind", "choice", "expected_type"),
    [("integer", 1, int), ("boolean", True, bool), ("number", 1.0, float), ("number", 1, float)],
)
def test_typed_enum_keeps_declared_output_type_for_values_and_defaults(
    kind: str, choice: Any, expected_type: type
) -> None:
    M = compile_object_schema("P", {"p": {"type": kind, "enum": [choice], "default": choice}})
    assert type(M.model_validate({"p": choice}).model_dump()["p"]) is expected_type
    assert type(M().model_dump()["p"]) is expected_type
    assert M.model_json_schema()["properties"]["p"]["const"] == choice


@pytest.mark.parametrize("use_default", [False, True])
@pytest.mark.parametrize(
    ("kind", "choice", "value", "expected_type"),
    [
        ("integer", 1, 1, int),
        ("boolean", True, True, bool),
        ("boolean", False, False, bool),
        ("number", 1, 1, float),
        ("number", 1, 1.0, float),
        ("number", 1.0, 1, float),
        ("number", 1.5, 1.5, float),
    ],
)
def test_typed_enum_values_and_defaults_roundtrip(
    kind: str, choice: Any, value: Any, expected_type: type, *, use_default: bool
) -> None:
    prop = {"type": kind, "enum": [choice]}
    if use_default:
        prop["default"] = value
    M = compile_object_schema("P", {"p": prop})
    model = M() if use_default else M.model_validate({"p": value})
    dumped = model.model_dump()
    assert dumped == {"p": value}
    assert type(dumped["p"]) is expected_type
    restored = M.model_validate(dumped).model_dump()
    assert restored == dumped
    assert type(restored["p"]) is expected_type


def test_recursive_number_enum_roundtrip() -> None:
    M = compile_object_schema(
        "P",
        {
            "xs": {
                "type": "array",
                "items": {"type": "number", "enum": [1, 2.5]},
                "default": [1, 2.5],
            }
        },
    )
    for model in (M(), M.model_validate({"xs": [1.0, 2.5]})):
        dumped = model.model_dump()
        assert dumped == {"xs": [1.0, 2.5]}
        assert all(type(value) is float for value in dumped["xs"])
        assert M.model_validate(dumped).model_dump() == dumped


@pytest.mark.parametrize("choice", [1, 1.0])
@pytest.mark.parametrize("bad", [True, False, "1", 2])
def test_number_enum_rejects_wrong_types_and_nonmembers(choice: int | float, bad: Any) -> None:
    prop = {"type": "number", "enum": [choice]}
    M = compile_object_schema("P", {"p": prop})
    with pytest.raises(ValidationError):
        M.model_validate({"p": bad})
    with pytest.raises(ValueError, match=r"P\.p: default does not match the schema"):
        compile_object_schema("P", {"p": {**prop, "default": bad}})


def test_recursive_enum_membership_is_strict() -> None:
    M = compile_object_schema("P", {"xs": {"type": "array", "items": {"enum": [1]}}})
    with pytest.raises(ValidationError):
        M.model_validate({"xs": [True]})


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("constraint", ["maxLength", "pattern", "enum"])
def test_invalid_default_errors_hide_secret_values_and_exception_chain(
    constraint: str, *, nested: bool
) -> None:
    secret = "synthetic-secret-default-8395"
    allowed = "synthetic-secret-enum-member-2694"
    prop: dict[str, Any] = {"type": "string", "default": secret}
    prop[constraint] = {"maxLength": 1, "pattern": "^safe$", "enum": [allowed]}[constraint]
    schema = {"type": "array", "items": prop} if nested else prop
    with pytest.raises(ValueError, match=r"P\.p(?:\.items)?: default") as exc_info:
        compile_object_schema("P", {"p": schema})
    rendered = "".join(traceback.format_exception(exc_info.value))
    for sensitive in (secret, allowed):
        assert sensitive not in str(exc_info.value)
        assert sensitive not in rendered
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__


@pytest.mark.parametrize("keyword", ["type", "pattern", "minimum", "default", "enum"])
def test_compile_errors_hide_values_and_preserve_identifiers(keyword: str) -> None:
    secret = "SECRET_SENTINEL_schema_compile"
    schemas = {
        "type": {"type": secret},
        "pattern": {"type": "string", "pattern": "(" + secret},
        "minimum": {"type": "number", "minimum": secret},
        "default": {"type": "integer", "default": secret},
        "enum": {"type": "integer", "enum": [secret]},
    }
    with pytest.raises(ValueError, match=r"P\.lr") as caught:
        compile_object_schema("P", {"lr": schemas[keyword]})
    error = caught.value
    assert "P.lr" in str(error)
    assert keyword in str(error)
    for diagnostic in (
        str(error),
        repr(error),
        repr(error.args),
        repr(getattr(error, "__notes__", ())),
        "".join(traceback.format_exception(error)),
    ):
        assert secret not in diagnostic
    assert error.__cause__ is None
    if isinstance(error, ValidationError):
        assert secret not in repr(error.errors())
        assert secret not in error.json()


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize(
    "malformed",
    [
        ["SECRET_SENTINEL_subschema"],
        "SECRET_SENTINEL_subschema",
        73918462,
        [73918462],
        None,
        True,
        [],
        ("SECRET_SENTINEL_subschema",),
    ],
)
def test_non_mapping_subschemas_are_rejected_without_values(malformed: Any, depth: int) -> None:
    schema = malformed
    for _ in range(depth):
        schema = {"type": "array", "items": schema}
    with pytest.raises(_SchemaError, match="invalid schema") as caught:
        compile_object_schema("P", {"xs": schema})
    error = caught.value
    assert str(error) == "P.xs" + ".items" * depth + ": invalid schema"
    for diagnostic in (
        str(error),
        repr(error),
        repr(error.args),
        repr(getattr(error, "__notes__", ())),
        "".join(traceback.format_exception(error)),
    ):
        assert "SECRET_SENTINEL_subschema" not in diagnostic
        assert "73918462" not in diagnostic
    assert error.__cause__ is None


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "prop",
    [
        "model_validate",
        "model_validate_json",
        "model_dump",
        "model_dump_json",
        "model_config",
        "model_fields",
        "model_post_init",
        "model_validate_custom",
        "model_dump_custom",
        "schema",
        "schema_json",
        "dict",
        "copy",
        "json",
        "parse_obj",
        "construct",
        "validate",
        "mro",
        "register",
        "Config",
    ],
)
def test_reserved_property_names_are_rejected_at_compile_time(prop: str) -> None:
    with pytest.raises(_SchemaError, match="invalid property name") as caught:
        compile_object_schema("P", {prop: {"type": "string", "default": "secret-default"}})
    assert str(caught.value) == f"P.{prop}: invalid property name"
    assert caught.value.__cause__ is None


def test_nonreserved_property_names_remain_usable() -> None:
    values = {
        "modelvalue": 1,
        "schema_version": 2,
        "dictionary": 3,
        "copy_count": 4,
        "config": 5,
        "model_name": 6,
        "model_id": 7,
        "model_type": 8,
        "model_path": 9,
        "model_custom": 10,
    }
    model = compile_object_schema("P", {key: {"type": "integer"} for key in values})
    assert model.model_validate(values).model_dump() == values


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("namespace", ["custom_", re.compile("^custom_name$")])
def test_reserved_names_follow_generated_model_config(
    monkeypatch: pytest.MonkeyPatch, namespace: str | re.Pattern[str]
) -> None:
    monkeypatch.setattr(
        BaseModel, "model_config", {**BaseModel.model_config, "protected_namespaces": (namespace,)}
    )
    with pytest.raises(_SchemaError, match=r"^P\.custom_name: invalid property name$"):
        compile_object_schema("P", {"custom_name": {"type": "string"}})
    model = compile_object_schema("P", {"model_dump_custom": {"type": "integer"}})
    assert model.model_validate({"model_dump_custom": 1}).model_dump() == {"model_dump_custom": 1}
    assert model.model_config["protected_namespaces"] == (namespace,)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    ("kind", "keyword", "bad"),
    [
        (kind, keyword, bad)
        for kind in ("integer", "number")
        for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
        for bad in (
            True,
            False,
            None,
            "SECRET_keyword",
            ["SECRET_keyword"],
            {"v": "SECRET_keyword"},
        )
    ]
    + [
        ("string", keyword, bad)
        for keyword in ("minLength", "maxLength")
        for bad in (True, False, -1, 1.5, None, "SECRET_keyword", [], {})
    ]
    + [
        ("string", keyword, bad)
        for keyword in ("description", "pattern")
        for bad in (True, 5, 1.5, None, ["SECRET_keyword"], {"v": "SECRET_keyword"})
    ]
    + [("string", "pattern", re.compile("SECRET_keyword"))]
    + [("string", "enum", bad) for bad in (True, 5, None, "SECRET_keyword", (), {}, [])]
    + [(None, "type", bad) for bad in (True, 5, None, ["SECRET_keyword"], {})],
)
def test_wrong_typed_keywords_have_exact_value_free_errors(
    kind: str | None, keyword: str, bad: Any, *, nested: bool
) -> None:
    prop = {"type": kind, keyword: bad}
    schema = {"type": "array", "items": prop} if nested else prop
    with pytest.raises(_SchemaError, match="invalid") as caught:
        compile_object_schema("P", {"p": schema})
    assert str(caught.value) == "P.p" + (".items" if nested else "") + f": invalid {keyword}"
    error = caught.value
    for diagnostic in (
        str(error),
        repr(error),
        repr(error.args),
        repr(getattr(error, "__notes__", ())),
        "".join(traceback.format_exception(error)),
    ):
        assert "SECRET_keyword" not in diagnostic
    assert error.__cause__ is None


def test_keyword_boundary_values_produce_valid_schema() -> None:
    model = compile_object_schema(
        "P",
        {
            "text": {
                "type": "string",
                "minLength": 0,
                "maxLength": 0,
                "description": "",
                "pattern": "",
            },
            "n": {
                "type": "number",
                "minimum": 0,
                "maximum": 2.0,
                "exclusiveMinimum": -1.0,
                "exclusiveMaximum": 3,
            },
            "flag": {"enum": [True, False], "default": False},
        },
    )
    assert model.model_validate({"text": "", "n": 1}).model_dump() == {
        "text": "",
        "n": 1.0,
        "flag": False,
    }
    properties = model.model_json_schema()["properties"]
    assert properties["text"]["description"] == ""
    assert properties["text"]["minLength"] == properties["text"]["maxLength"] == 0
    assert type(properties["n"]["minimum"]) is int
    assert type(properties["n"]["maximum"]) is float


def test_large_bad_default_has_sorted_unique_bounded_reasons() -> None:
    with pytest.raises(_SchemaError, match="default") as caught:
        compile_object_schema(
            "P",
            {
                "xs": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 2},
                    "default": ["SECRET_default"] * 199_998 + [0, 3],
                }
            },
        )
    assert str(caught.value) == (
        "P.xs: default does not match the schema (greater_than_equal, int_type, less_than_equal)"
    )
    assert "SECRET_default" not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("count", [5, 6, 7])
def test_default_reasons_show_at_most_five_types(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    secret = "SECRET_default"
    codes = [
        "tuple_type",
        "string_type",
        "list_type",
        "int_type",
        "float_type",
        "dict_type",
        "bool_type",
    ][:count]

    def reject_default(self: TypeAdapter[Any], value: Any, **kwargs: Any) -> Any:
        errors: list[InitErrorDetails] = [{"type": code, "input": value} for code in codes * 2]
        raise ValidationError.from_exception_data("default", errors)

    monkeypatch.setattr(TypeAdapter, "validate_python", reject_default)
    with pytest.raises(_SchemaError, match="default") as caught:
        compile_object_schema("P", {"p": {"type": "integer", "default": secret}})
    expected = sorted(set(codes))[:5] + (["…"] if count > 5 else [])
    assert str(caught.value) == f"P.p: default does not match the schema ({', '.join(expected)})"
    assert "SECRET_default" not in "".join(traceback.format_exception(caught.value))
