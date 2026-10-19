"""Define frozen models, stored-format validation, and shared type aliases."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, LiteralString, Self, cast, get_args

from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    ValidationError,
    model_validator,
)
from pydantic import JsonValue as JsonValue
from pydantic.config import ExtraValues
from pydantic_core import CoreSchema, ErrorType, InitErrorDetails, PydanticCustomError, core_schema

from swarm_arena._model.migrate import CURRENT, upgrade

Origin = Literal["human", "llm", "together", "autonomous"]
LLMRole = Literal["manager", "judge", "planner", "curator"]
CallRole = Literal["manager", "judge", "planner", "curator", "tool"]
CatalogKind = Literal["tool", "manager", "check", "judge", "policy"]


# One allowlist for all models: custom codes and messages cannot carry user data.
_VALIDATION_ERROR_TYPES = frozenset(get_args(ErrorType))
_HUMAN_ONLY_REASONS: dict[str, LiteralString] = {
    field: f"human-only field '{field}' cannot be settable"
    for field in ("command", "python", "files", "env", "editable")
}
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
        "human-only field cannot be settable",
        *_HUMAN_ONLY_REASONS.values(),
    }
)


def _safe_location(
    model: type[BaseModel], location: tuple[str | int, ...], *, show_unknown: bool = False
) -> tuple[str | int, ...]:
    """Keep a top-level schema name or alias and collapse deeper segments."""
    names = set() if model.__pydantic_root_model__ else set(model.model_fields)
    for field in () if model.__pydantic_root_model__ else model.model_fields.values():
        if field.alias is not None:
            names.add(field.alias)
        alias = field.validation_alias
        choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
        for choice in choices:
            if isinstance(choice, str):
                names.add(choice)
            elif isinstance(choice, AliasPath) and isinstance(choice.path[0], str):
                names.add(choice.path[0])
    first = location[0] if location else None
    name = first if isinstance(first, str) and (first in names or show_unknown) else "<field>"
    return (name, "<…>") if len(location) > 1 else (name,)


def _sanitize_validation_error(
    model: type[BaseModel], error: Exception, *, unknown_fields: Literal["show", "hide"] = "hide"
) -> ValidationError:
    """
    Rebuild a validation failure without rejected values or custom error context.

    Parameters
    ----------
    model : type[BaseModel]
        Schema whose top-level fields and aliases are safe identifiers.
    error : Exception
        Validation failure; other exception types become a single ``invalid`` error.
    unknown_fields : {'show', 'hide'}, default='hide'
        Whether a top-level unknown field name is a human typo diagnostic.

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
            InitErrorDetails(type=PydanticCustomError("invalid", "invalid"), loc=(), input=None)
        )
    return ValidationError.from_exception_data(model.__name__, details, hide_input=True)


def _validation_wrapper(
    model: type[BaseModel],
) -> Callable[[Any, Callable[[Any], Any]], Any]:
    """Build the one outer validation boundary, including subclass after validators."""

    def validate(data: Any, handler: Callable[[Any], Any]) -> Any:
        """Suppress input-bearing exception chains and rebuild safe error details."""
        try:
            return handler(data)
        except Exception as error:
            raise _sanitize_validation_error(
                model, error, unknown_fields=getattr(model, "_unknown_fields", "hide")
            ) from None

    return validate


class _SafeModel(BaseModel):
    """
    Freeze fields and sanitize every validation entry point through one boundary.

    Attributes
    ----------
    _unknown_fields : ClassVar[Literal['show', 'hide']]
        Unknown-key policy. Human plan/settings classes select ``show``; component,
        job, catalog and result classes inherit ``hide``. Deeper locations always
        collapse to ``<…>``. JSON parse errors and frozen assignments use the same core.

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
        return core_schema.no_info_wrap_validator_function(
            _validation_wrapper(cls), handler(source)
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
        return _validation_wrapper(cls)(
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
        return _validation_wrapper(cls)(
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
            try:
                return upgrade(cls.schema_kind, data)
            except ValueError:
                version = data.get("schema_version", 1)
                if (
                    type(version) is int
                    and cls.schema_kind in CURRENT
                    and version > CURRENT[cls.schema_kind]
                ):
                    raise ValueError(
                        "schema_version is newer than this swarm-arena supports"
                    ) from None
                raise
        return data
