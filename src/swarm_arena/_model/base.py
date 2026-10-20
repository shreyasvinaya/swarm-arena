"""Define frozen models, stored-format validation, and shared type aliases."""

from collections.abc import Mapping
from threading import get_ident
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic import JsonValue as JsonValue
from pydantic.config import ExtraValues
from pydantic_core import CoreSchema, core_schema

from swarm_arena._model.compat import _standard_init
from swarm_arena._model.data import check_utf8
from swarm_arena._model.diagnostics import _restore_validation_error
from swarm_arena._model.diagnostics import validation_summary as validation_summary
from swarm_arena._model.locations import _safe_location as _safe_location
from swarm_arena._model.migrate import upgrade
from swarm_arena._model.safe import (
    _ERROR_CAP_REASON as _ERROR_CAP_REASON,
)
from swarm_arena._model.safe import (
    _VALIDATION_ACTIVE as _VALIDATION_ACTIVE,
)
from swarm_arena._model.safe import (
    _minimal_validation_error as _minimal_validation_error,
)
from swarm_arena._model.safe import (
    _sanitize_validation_error as _sanitize_validation_error,
)
from swarm_arena._model.safe import (
    _validation_wrapper,
)
from swarm_arena._model.safe import (
    _ValidationOwner as _ValidationOwner,
)

Origin = Literal["human", "llm", "together", "autonomous"]
LLMRole = Literal["manager", "judge", "planner", "curator"]
CallRole = Literal["manager", "judge", "planner", "curator", "tool"]
CatalogKind = Literal["tool", "manager", "check", "judge", "policy"]


class _SafeModel(BaseModel):
    """
    Freeze fields and sanitize every validation entry point through one boundary.

    Attributes
    ----------
    _unknown_fields : ClassVar[Literal['show', 'hide']]
        Unknown-key policy. Human plan/settings classes select ``show``; component,
        job, catalog and result classes inherit ``hide``. The outermost model owns
        this policy for nested fields in its originating thread. Explicit
        constructor and ``model_validate*`` calls inside validators establish
        their own boundary. Fields, aliases and indices remain; segments under
        free-form mappings or Any become ``<…>``. JSON parse errors and frozen
        assignments use the same core. Sanitizer failures yield a minimal
        ``invalid`` error, and Pydantic developer errors propagate unchanged.

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

    model_config = ConfigDict(
        frozen=True, extra="forbid", hide_input_in_errors=True, allow_inf_nan=False
    )
    _unknown_fields: ClassVar[Literal["show", "hide"]] = "hide"

    @field_validator("*")
    @classmethod
    def _utf8_fields(cls, value: Any) -> Any:
        """Refuse unencodable text before any JSON-bound field can be serialized."""
        return check_utf8(value)

    @_standard_init
    def __init__(self, /, **data: Any) -> None:
        """
        Validate an explicit constructor call with this model's own policy.

        Parameters
        ----------
        **data : Any
            Field inputs to validate.

        Examples
        --------
        >>> _SafeModel().model_dump()
        {}
        """
        owner = _VALIDATION_ACTIVE.get()
        if owner is None or owner.thread_id != get_ident():
            try:
                BaseModel.__init__(self, **data)
            except ValidationError as error:
                # Pydantic's constructor rebuilds even a sanitized exception as
                # its native class. Restore our transport without re-walking locs.
                raise _restore_validation_error(
                    error.title, error.errors(include_url=False)
                ) from None
            return
        initialize = super().__init__
        _validation_wrapper(type(self), explicit=True)(data, lambda values: initialize(**values))

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
        return _validation_wrapper(cls, by_alias=by_alias, explicit=True)(
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
        return _validation_wrapper(cls, by_alias=by_alias, explicit=True)(
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
        return _validation_wrapper(cls, by_alias=by_alias, explicit=True)(
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
        if isinstance(data, Mapping):
            if not cls.schema_kind:
                raise ValueError("model has no schema_kind")
            return upgrade(cls.schema_kind, dict(data))
        return data
