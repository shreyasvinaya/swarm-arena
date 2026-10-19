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

from swarm_arena._model.base import _SafeModel

_BOUNDS = {
    "minimum": "ge",
    "maximum": "le",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
}
_STRINGS = {"minLength": "min_length", "maxLength": "max_length", "pattern": "pattern"}
_KEYWORDS = {"type", "enum", "items", "default", "description", *_BOUNDS, *_STRINGS}
_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}


class _SchemaError(ValueError):
    """Identify compiler diagnostics containing only schema identifiers and fixed reasons."""


def _check_keyword_types(path: str, schema: Mapping[str, Any]) -> None:
    """Refuse malformed keyword values before pydantic can coerce them."""
    for keyword, value in schema.items():
        if keyword in _BOUNDS:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif keyword in ("minLength", "maxLength"):
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        elif keyword in ("type", "description", "pattern"):
            valid = isinstance(value, str)
        elif keyword == "enum":
            valid = isinstance(value, list) and bool(value)
        else:
            # items is compiled recursively; default is checked against its annotation.
            continue
        if not valid:
            raise _SchemaError(f"{path}: invalid {keyword}")


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


def _compile_field_impl(path: str, schema: Mapping[str, Any]) -> tuple[Any, FieldInfo]:
    """Compile a property or array item and validate its declared default."""
    if not isinstance(schema, Mapping):
        raise _SchemaError(f"{path}: invalid schema")
    for keyword in schema:
        if keyword not in _KEYWORDS:
            raise _SchemaError(f"{path}: unsupported JSON-schema keyword {keyword!r}")
    _check_keyword_types(path, schema)
    kind = schema.get("type")
    if kind is None and "enum" not in schema:
        raise _SchemaError(f"{path}: expected type or enum")
    annotation: Any
    if kind == "array":
        item_type = (
            _compile_field(f"{path}.items", schema["items"])[0] if "items" in schema else Any
        )
        annotation = list[item_type]
    elif kind is None:
        annotation = Any
    elif isinstance(kind, str) and kind in _TYPES:
        annotation = _TYPES[kind]
    else:
        raise _SchemaError(f"{path}: unsupported type")
    if "items" in schema and kind != "array":
        raise _SchemaError(f"{path}: items requires type 'array'")
    if "enum" in schema:
        choices = schema["enum"]
        adapter = TypeAdapter(annotation, config=ConfigDict(strict=True)) if kind else None
        if adapter is not None:
            try:
                choices = [adapter.validate_python(choice, strict=True) for choice in choices]
            except ValidationError:
                raise _SchemaError(
                    f"{path}: enum member does not match the declared type"
                ) from None
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
                raise _SchemaError(f"{path}: {keyword} requires type 'integer' or 'number'")
            constraints[constraint] = schema[keyword]
    for keyword, constraint in _STRINGS.items():
        if keyword in schema:
            if kind != "string":
                raise _SchemaError(f"{path}: {keyword} requires type 'string'")
            constraints[constraint] = schema[keyword]
    annotation = Annotated[annotation, Field(description=schema.get("description"), **constraints)]
    adapter = TypeAdapter(annotation, config=ConfigDict(strict=True))
    default = schema.get("default", ...)
    if "default" in schema:
        try:
            default = adapter.validate_python(default)
        except ValidationError as err:
            error_types = sorted(
                {
                    error["type"]
                    for error in err.errors(
                        include_input=False, include_url=False, include_context=False
                    )
                }
            )
            reasons = ", ".join(error_types[:5] + (["…"] if len(error_types) > 5 else []))
            raise _SchemaError(f"{path}: default does not match the schema ({reasons})") from None
    field = cast(FieldInfo, Field(default=default))
    return annotation, field


def _compile_field(path: str, schema: Mapping[str, Any]) -> tuple[Any, FieldInfo]:
    """Preserve safe diagnostics and replace native schema errors with keyword identifiers."""
    try:
        return _compile_field_impl(path, schema)
    except _SchemaError:
        raise
    except Exception:
        keywords = (
            ", ".join(key for key in schema if isinstance(key, str))
            if isinstance(schema, Mapping)
            else ""
        )
        suffix = f" (keywords: {keywords})" if keywords else ""
        raise _SchemaError(f"{path}: invalid schema{suffix}") from None


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
        Every property and items schema must be a mapping. Property names must
        start with an ASCII letter and contain only ASCII letters, digits, or
        underscores; BaseModel member names, names matching the generated model's
        configured protected namespaces, and Config are reserved. Other model_
        names, such as model_name, are allowed. Numeric bounds must be integers
        or floats, length bounds must be non-negative integers, and neither
        accepts booleans. Descriptions and patterns must be strings.

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
    >>> Named = compile_object_schema("Named", {"model_name": {"type": "string"}})
    >>> Named(model_name="small").model_dump()
    {'model_name': 'small'}
    """
    config = ConfigDict(
        strict=True,
        protected_namespaces=("model_validate", "model_dump"),
    )
    protected_namespaces = config["protected_namespaces"]
    fields: dict[str, Any] = {}
    for prop, schema in properties.items():
        if (
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prop)
            or any(
                prop.startswith(namespace)
                if isinstance(namespace, str)
                else namespace.match(prop) is not None
                for namespace in protected_namespaces
            )
            or prop == "Config"
            or hasattr(BaseModel, prop)
        ):
            raise _SchemaError(f"{name}.{prop}: invalid property name")
        fields[prop] = _compile_field(f"{name}.{prop}", schema)
    return create_model(name, __base__=_SafeModel, __config__=config, **fields)
