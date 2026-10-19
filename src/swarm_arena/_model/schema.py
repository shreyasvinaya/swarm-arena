"""Compile the supported JSON-schema property subset into pydantic models."""

import re
from collections.abc import Mapping
from functools import partial
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    create_model,
)
from pydantic.fields import FieldInfo

_BOUNDS = {
    "minimum": "ge",
    "maximum": "le",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
}
_STRINGS = {"minLength": "min_length", "maxLength": "max_length", "pattern": "pattern"}
_KEYWORDS = {"type", "enum", "items", "default", "description", *_BOUNDS, *_STRINGS}
_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}


def _validate_enum(
    value: Any,
    handler: ValidatorFunctionWrapHandler,
    *,
    choices: tuple[Any, ...],
    adapter: TypeAdapter[Any] | None,
) -> Any:
    """Check membership by type and value, preserving the declared output type."""
    validated = adapter.validate_python(value, strict=True) if adapter is not None else value
    if not any(type(validated) is type(choice) and validated == choice for choice in choices):
        raise ValueError("value does not match an enum member")
    handler(validated)
    # Literal can substitute an equal member of another type; keep our validated value.
    return validated


def _compile_field(path: str, schema: Mapping[str, Any]) -> tuple[Any, FieldInfo]:
    """Compile a property or array item and validate its declared default."""
    for keyword in schema:
        if keyword not in _KEYWORDS:
            raise ValueError(f"{path}: unsupported JSON-schema keyword {keyword!r}")
    kind = schema.get("type")
    if kind is None and "enum" not in schema:
        raise ValueError(f"{path}: expected type or enum")
    annotation: Any
    if kind == "array":
        item_type = (
            _compile_field(f"{path}.items", schema["items"])[0] if "items" in schema else Any
        )
        annotation = list[item_type]
    elif kind is None:
        annotation = Any
    elif kind in _TYPES:
        annotation = _TYPES[kind]
    else:
        raise ValueError(f"{path}: unsupported type {kind!r}")
    if "items" in schema and kind != "array":
        raise ValueError(f"{path}: items requires type 'array'")
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{path}: enum must be a non-empty list")
        adapter = TypeAdapter(annotation, config=ConfigDict(strict=True)) if kind else None
        if adapter is not None:
            try:
                choices = [adapter.validate_python(choice, strict=True) for choice in choices]
            except ValidationError:
                raise ValueError(f"{path}: enum member does not match the declared type") from None
        # Enum values are runtime schema data rather than static type arguments.
        literal = cast(Any, Literal)[tuple(choices)]
        annotation = Annotated[
            literal,
            WrapValidator(partial(_validate_enum, choices=tuple(choices), adapter=adapter)),
        ]
    constraints: dict[str, Any] = {}
    for keyword, constraint in _BOUNDS.items():
        if keyword in schema:
            if kind not in ("integer", "number"):
                raise ValueError(f"{path}: {keyword} requires type 'integer' or 'number'")
            constraints[constraint] = schema[keyword]
    for keyword, constraint in _STRINGS.items():
        if keyword in schema:
            if kind != "string":
                raise ValueError(f"{path}: {keyword} requires type 'string'")
            constraints[constraint] = schema[keyword]
    annotation = Annotated[annotation, Field(description=schema.get("description"), **constraints)]
    default = schema.get("default", ...)
    if "default" in schema:
        try:
            default = TypeAdapter(annotation, config=ConfigDict(strict=True)).validate_python(
                default
            )
        except ValidationError as err:
            reasons = ", ".join(
                error["type"]
                for error in err.errors(
                    include_input=False, include_url=False, include_context=False
                )
            )
            raise ValueError(f"{path}: default does not match the schema ({reasons})") from None
    field = cast(FieldInfo, Field(default=default))
    return annotation, field


def compile_object_schema(
    name: str, properties: Mapping[str, Mapping[str, Any]]
) -> type[BaseModel]:
    """
    Compile supported JSON-schema properties into a strict frozen model.

    Parameters
    ----------
    name : str
        Generated model name, also used in schema error messages.
    properties : Mapping[str, Mapping[str, Any]]
        Property names and schemas. Supported types are string, integer,
        number, boolean, and array, with recursive items, enum, numeric bounds,
        string length and pattern constraints, default, and description.
        Properties without a default are required; extra fields are forbidden.
        Enum membership compares both type and value after declared strict type
        validation of members and inputs. Number enums accept equivalent integer
        and float values, excluding booleans; their outputs and defaults are
        floats, so dumped values can be validated again.

    Returns
    -------
    type[BaseModel]
        Strict, frozen pydantic model with validated defaults.

    Raises
    ------
    ValueError
        If a property name, keyword, type, constraint, or default is invalid.

    Examples
    --------
    >>> Params = compile_object_schema("Params", {"epochs": {"type": "integer", "maximum": 400}})
    >>> Params(epochs=400).model_dump()
    {'epochs': 400}
    """
    fields: dict[str, Any] = {}
    for prop, schema in properties.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prop):
            raise ValueError(f"{name}.{prop}: invalid property name")
        fields[prop] = _compile_field(f"{name}.{prop}", schema)
    return create_model(
        name, __config__=ConfigDict(extra="forbid", strict=True, frozen=True), **fields
    )
