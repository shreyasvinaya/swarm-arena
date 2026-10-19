"""Define frozen models, stored-format validation, and shared type aliases."""

import re
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, Literal, LiteralString, Self, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    ValidationError,
    model_validator,
)
from pydantic import JsonValue as JsonValue
from pydantic.config import ExtraValues
from pydantic_core import CoreSchema, ErrorType, InitErrorDetails, PydanticCustomError, core_schema

from swarm_arena._model.migrate import upgrade

Origin = Literal["human", "llm", "together", "autonomous"]
LLMRole = Literal["manager", "judge", "planner", "curator"]
CallRole = Literal["manager", "judge", "planner", "curator", "tool"]
CatalogKind = Literal["tool", "manager", "check", "judge", "policy"]


# One allowlist for all models: custom codes and messages cannot carry user data.
_VALIDATION_ERROR_TYPES = frozenset(get_args(ErrorType))
# This template admits only catalog entry/property identifiers, never values.
_CATALOG_HUMAN_ONLY_REASON = re.compile(
    r"catalog entry '[a-z0-9][a-z0-9._-]*': "
    r"human-only field '[A-Za-z][A-Za-z0-9_]*' cannot be settable"
)
_NEWER_VERSION_REASON = re.compile(
    r"Value error, [a-z_]+ schema_version [0-9]+ "
    r"is newer than this swarm-arena supports \([0-9]+\)"
)
_VALIDATION_BY_ALIAS: ContextVar[bool | None] = ContextVar(
    "swarm_arena_validation_by_alias", default=None
)
_VALIDATION_ACTIVE: ContextVar[bool] = ContextVar("swarm_arena_validation_active", default=False)
_SAFE_MESSAGES = frozenset(
    {
        "Value error, Plan.flows is reserved for v2 messaging and must be empty",
        "Value error, duplicate part names",
        "Value error, duplicate team names",
        "Value error, duplicate agent names within a team",
        "Value error, team.part: unknown part",
        "key needs at least one value",
        "score key values must be finite",
        "overlay keys must be normalized workspace paths",
        "overlay values must be 64 lowercase hex characters",
        "title must be nonempty, single-line, and at most 120 characters",
        "text must be at most 1500 characters",
        "evidence needs at least one id unless the author is human",
        "schema_version must be a positive integer",
        "schema_version is newer than this swarm-arena supports",
        "model has no schema_kind",
        "s3 backend requires s3_path",
        "redis backend requires redis_url_env",
        "roles must be unique",
        "invalid settable schema",
        "duplicate catalog entry names",
        "cannot load catalog component class",
        "data must be finite JSON",
        "details must be finite JSON",
    }
)


def _schema_definitions(schema: dict[str, Any]) -> dict[str, Any]:
    """Index core-schema references without following repeated or recursive objects."""
    refs: dict[str, Any] = {}
    pending: list[Any] = [schema]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if not isinstance(node, (dict, list, tuple)) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, dict):
            if isinstance(node.get("ref"), str):
                refs[node["ref"]] = node
            pending.extend(node.values())
        else:
            pending.extend(node)
    return refs


def _location_parts(
    schema: dict[str, Any],
    location: tuple[str | int, ...],
    refs: dict[str, Any],
    *,
    show_unknown: bool,
) -> tuple[str | int, ...]:
    """Walk schema nodes, retaining identifiers and redacting each data segment."""
    if not location:
        return ()
    # Wrappers do not consume a location; guard pathological reference-only loops.
    seen: set[int] = set()
    loc_by_alias = True
    validate_by_alias = True
    while id(schema) not in seen:
        seen.add(id(schema))
        kind = schema.get("type")
        if kind == "definition-ref":
            schema = refs.get(schema["schema_ref"], {})
        elif kind == "model":
            config = schema.get("config", {})
            loc_by_alias = config.get("loc_by_alias", True)
            validate_by_alias = config.get("validate_by_alias", True)
            schema = schema.get("schema", {})
        elif kind == "definitions" or "schema" in schema:
            schema = schema.get("schema", {})
        elif kind == "json-or-python":
            schema = schema["python_schema"]
        elif kind == "lax-or-strict":
            schema = schema["strict_schema"]
        else:
            break
    first, *tail = location
    rest = tuple(tail)
    kind = schema.get("type")
    if kind in {"model-fields", "typed-dict"}:
        aliases: list[tuple[tuple[str | int, ...], dict[str, Any]]] = []
        names: list[tuple[tuple[str | int, ...], dict[str, Any]]] = []
        for name, field in schema["fields"].items():
            names.append(((name,), field))
            alias = field.get("validation_alias")
            paths = [[alias]] if isinstance(alias, str) else alias or []
            if paths and not isinstance(paths[0], list):
                paths = [paths]
            aliases.extend((tuple(path), field) for path in paths)
        # Pydantic reports the validation alias on collisions, even when it is
        # another field's name. Resolve all aliases before looking at names.
        by_alias = _VALIDATION_BY_ALIAS.get()
        use_alias = validate_by_alias if by_alias is None else by_alias
        paths = [*aliases, *names] if loc_by_alias and use_alias else names
        for path, field in paths:
            if location[: len(path)] == path:
                return path + _location_parts(
                    field["schema"], location[len(path) :], refs, show_unknown=show_unknown
                )
        return ((first if show_unknown else "<field>"),) + ("<…>",) * len(rest)
    if kind in {"list", "set", "frozenset", "generator", "tuple"} and isinstance(first, int):
        items = schema.get("items_schema", {})
        if kind == "tuple":
            index = schema.get("variadic_item_index")
            items = (
                items[first]
                if 0 <= first < len(items)
                else (items[index] if index is not None else {})
            )
        return (first, *_location_parts(items, rest, refs, show_unknown=show_unknown))
    if kind == "dict":
        if rest and rest[0] == "[key]":
            return ("<…>",) * len(location)
        return (
            "<…>",
            *_location_parts(
                schema.get("values_schema", {}), rest, refs, show_unknown=show_unknown
            ),
        )
    if kind == "tagged-union":
        return (
            "<…>",
            *_location_parts(
                schema["choices"].get(first, {}), rest, refs, show_unknown=show_unknown
            ),
        )
    if kind == "union":
        for choice in schema["choices"]:
            branch, label = choice if isinstance(choice, tuple) else (choice, None)
            node = branch
            while "schema" in node and node.get("type") != "model":
                node = node["schema"]
            if branch.get("type") == "function-wrap":
                label = label or f"function-wrap[{branch['function']['function'].__name__}()]"
            label = label or getattr(node.get("cls"), "__name__", node.get("type"))
            if first == label:
                return ("<…>", *_location_parts(branch, rest, refs, show_unknown=show_unknown))
    return ("<…>",) * len(location)


def _safe_location(
    model: type[BaseModel], location: tuple[str | int, ...], *, show_unknown: bool = False
) -> tuple[str | int, ...]:
    """Retain schema fields, aliases and indices, with data keys always redacted."""
    schema = cast(Any, model.__pydantic_core_schema__)
    return _location_parts(schema, location, _schema_definitions(schema), show_unknown=show_unknown)


def _sanitize_validation_error(
    model: type[BaseModel], error: Exception, *, unknown_fields: Literal["show", "hide"] = "hide"
) -> ValidationError:
    """
    Rebuild a validation failure without rejected values or custom error context.

    Parameters
    ----------
    model : type[BaseModel]
        Schema whose fields, aliases and sequence indices are safe identifiers.
    error : Exception
        Validation failure; other exception types become a single ``invalid`` error.
    unknown_fields : {'show', 'hide'}, default='hide'
        Whether unknown keys at model levels are human typo identifiers.

    Returns
    -------
    ValidationError
        Error containing only schema locations, built-in codes (or ``invalid``),
        fixed messages, and ``input=None`` without context. Raise it ``from None``.

    Examples
    --------
    >>> class Example(BaseModel):
    ...     count: int
    >>> try:
    ...     Example(count="private")
    ... except ValidationError as error:
    ...     safe = _sanitize_validation_error(Example, error)
    >>> safe.errors()[0]
    {'type': 'int_parsing', 'loc': ('count',), 'msg': 'int_parsing', 'input': None}
    """
    details: list[InitErrorDetails] = []
    if isinstance(error, ValidationError):
        for item in error.errors(include_input=False, include_url=False, include_context=False):
            code = item["type"] if item["type"] in _VALIDATION_ERROR_TYPES else "invalid"
            message = item["msg"]
            # Only complete fixed messages survive; never forward validator text.
            if message.removeprefix("Value error, ") in _SAFE_MESSAGES:
                message = message.removeprefix("Value error, ")
            elif _CATALOG_HUMAN_ONLY_REASON.fullmatch(message):
                pass
            elif _NEWER_VERSION_REASON.fullmatch(message):
                message = "schema_version is newer than this swarm-arena supports"
            elif message not in _SAFE_MESSAGES:
                message = code
            details.append(
                InitErrorDetails(
                    type=PydanticCustomError(
                        cast(LiteralString, code), cast(LiteralString, message)
                    ),
                    loc=_safe_location(
                        model,
                        item["loc"],
                        show_unknown=unknown_fields == "show" and code == "extra_forbidden",
                    ),
                    input=None,
                )
            )
    else:
        details.append(
            InitErrorDetails(
                type=PydanticCustomError(
                    "invalid", cast(LiteralString, f"invalid: {type(error).__name__}")
                ),
                loc=(),
                input=None,
            )
        )
    return ValidationError.from_exception_data(model.__name__, details, hide_input=True)


def _validation_wrapper(
    model: type[BaseModel], *, by_alias: bool | None = None
) -> Callable[[Any, Callable[[Any], Any]], Any]:
    """Build the one outer validation boundary, including subclass after validators."""

    def validate(data: Any, handler: Callable[[Any], Any]) -> Any:
        """Suppress input-bearing exception chains and rebuild safe error details."""
        if _VALIDATION_ACTIVE.get():
            return handler(data)
        token = _VALIDATION_ACTIVE.set(True)
        alias_token = _VALIDATION_BY_ALIAS.set(by_alias)
        try:
            return handler(data)
        except Exception as error:
            raise _sanitize_validation_error(
                model, error, unknown_fields=getattr(model, "_unknown_fields", "hide")
            ) from None
        finally:
            _VALIDATION_BY_ALIAS.reset(alias_token)
            _VALIDATION_ACTIVE.reset(token)

    validate.__name__ = model.__name__
    return validate


class _SafeModel(BaseModel):
    """
    Freeze fields and sanitize every validation entry point through one boundary.

    Attributes
    ----------
    _unknown_fields : ClassVar[Literal['show', 'hide']]
        Unknown-key policy. Human plan/settings classes select ``show``; component,
        job, catalog and result classes inherit ``hide``. The outermost model owns
        this policy throughout the tree. Fields, aliases and indices remain;
        segments under free-form mappings or Any become ``<…>``. JSON parse
        errors and frozen assignments use the same core.

    Examples
    --------
    >>> class Example(_SafeModel):
    ...     count: int
    >>> Example.model_validate_json('{"count": 2}').count
    2
    >>> try:
    ...     Example(count="private")
    ... except ValidationError as error:
    ...     print(error)
    1 validation error for Example
    count
      int_parsing [type=int_parsing]
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    _unknown_fields: ClassVar[Literal["show", "hide"]] = "hide"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[BaseModel], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Wrap the completed schema so subclass after validators stay inside the boundary."""
        inner = dict(handler(source))
        ref = cast(str | None, inner.pop("ref", None))
        return core_schema.no_info_wrap_validator_function(
            _validation_wrapper(cls), cast(CoreSchema, inner), ref=ref
        )

    if not TYPE_CHECKING:
        # ty treats any declared __setattr__ override as mutable. This wrapper
        # preserves Pydantic's frozen behavior, including its static contract.
        def __setattr__(self, name: str, value: Any) -> None:
            """Sanitize frozen-assignment errors, which bypass schema validation."""
            try:
                super().__setattr__(name, value)
            except ValidationError as error:
                raise _sanitize_validation_error(type(self), error) from None

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """
        Validate Python data with the caller's alias mode and safe diagnostics.

        Parameters
        ----------
        obj : Any
            Python input to validate.
        strict : bool or None, optional
            Override strict validation.
        extra : {'allow', 'ignore', 'forbid'} or None, optional
            Override handling of extra fields.
        from_attributes : bool or None, optional
            Read object attributes when validating.
        context : Any, optional
            Context passed to validators.
        by_alias : bool or None, optional
            Accept validation aliases.
        by_name : bool or None, optional
            Accept model field names.

        Returns
        -------
        Self
            Validated model.

        Raises
        ------
        ValidationError
            If validation fails; diagnostics omit rejected values.

        Examples
        --------
        >>> _SafeModel.model_validate({}).model_dump()
        {}
        """
        validate = super().model_validate
        return _validation_wrapper(cls, by_alias=by_alias)(
            obj,
            lambda data: validate(
                data,
                strict=strict,
                extra=extra,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            ),
        )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """
        Validate JSON with safe diagnostics even when parsing fails before validators.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            JSON document to validate.
        strict : bool or None, optional
            Override strict validation.
        extra : {'allow', 'ignore', 'forbid'} or None, optional
            Override handling of extra fields.
        context : Any, optional
            Context passed to validators.
        by_alias : bool or None, optional
            Accept validation aliases.
        by_name : bool or None, optional
            Accept model field names.

        Returns
        -------
        Self
            Validated model.

        Raises
        ------
        ValidationError
            If parsing or validation fails; diagnostics omit rejected values.

        Examples
        --------
        >>> _SafeModel.model_validate_json('{}').model_dump()
        {}
        """
        validate = super().model_validate_json
        return _validation_wrapper(cls, by_alias=by_alias)(
            json_data,
            lambda data: validate(
                data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            ),
        )

    @classmethod
    def model_validate_strings(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """
        Validate string mappings without retaining input in diagnostic details.

        Parameters
        ----------
        obj : Any
            Input composed of string keys and values.
        strict : bool or None, optional
            Override strict validation.
        extra : {'allow', 'ignore', 'forbid'} or None, optional
            Override handling of extra fields.
        context : Any, optional
            Context passed to validators.
        by_alias : bool or None, optional
            Accept validation aliases.
        by_name : bool or None, optional
            Accept model field names.

        Returns
        -------
        Self
            Validated model.

        Examples
        --------
        >>> _SafeModel.model_validate_strings({}).model_dump()
        {}
        """
        validate = super().model_validate_strings
        return _validation_wrapper(cls, by_alias=by_alias)(
            obj,
            lambda data: validate(
                data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            ),
        )


class FrozenModel(_SafeModel):
    """
    Validate immutable model fields and reject unknown fields with safe diagnostics.

    Examples
    --------
    >>> class Example(FrozenModel):
    ...     x: int
    >>> Example(x=1).model_copy(update={"x": 2}).x
    2
    """


class StoredModel(FrozenModel):
    """
    Upgrade stored data before validating its fields.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version, upgraded to the current version when loaded.

    Attributes
    ----------
    schema_kind : ClassVar[str]
        Registry kind that every concrete stored model must declare.

    Examples
    --------
    >>> class Example(StoredModel):
    ...     schema_kind: ClassVar[str] = "plan"
    >>> Example.model_validate({}).schema_version
    1
    """

    schema_version: int = 1
    schema_kind: ClassVar[str] = ""

    @model_validator(mode="before")
    @classmethod
    def _upgrade(cls, data: object) -> object:
        """Upgrade dictionary inputs through the model's registered schema kind."""
        if isinstance(data, dict):
            if not cls.schema_kind:
                raise ValueError("model has no schema_kind")
            return upgrade(cls.schema_kind, data)
        return data
