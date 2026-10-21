"""Sanitize model failures and own explicit or nested validation boundaries."""

import re
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import get_ident
from typing import Any, Final, Literal, LiteralString, cast, get_args

from pydantic import BaseModel, ValidationError
from pydantic.errors import PydanticErrorMixin
from pydantic_core import ErrorType, InitErrorDetails, PydanticCustomError

from swarm_arena._errors import _exception_type_name
from swarm_arena._model.diagnostics import _SanitizedValidationError
from swarm_arena._model.locations import (
    _VALIDATION_BY_ALIAS,
    _safe_location,
    _schema_index,
    _sequence_paths,
)
from swarm_arena._outcomes import _classify_outcome, _propagate_control

# One allowlist for all models: custom codes and messages cannot carry user data.
_VALIDATION_ERROR_TYPES: Final = frozenset(get_args(ErrorType))
# This template admits only catalog entry/property identifiers, never values.
_CATALOG_HUMAN_ONLY_REASON: Final = re.compile(
    r"catalog entry '[a-z0-9][a-z0-9._-]*': "
    r"human-only field '[A-Za-z][A-Za-z0-9_]*' cannot be settable"
)
_NEWER_VERSION_REASON: Final = re.compile(
    r"Value error, [a-z_]+ schema_version [0-9]+ "
    r"is newer than this swarm-arena supports \([0-9]+\)"
)
_ERROR_CAP_REASON: Final = re.compile(r"and ([1-9][0-9]*) more errors")


@dataclass(slots=True)
class _ValidationOwner:
    """Own nested field validation only while active in the originating thread."""

    thread_id: int | None
    # Successful validation never needs to allocate or initialize this error-only slot.
    sanitized: set[tuple[tuple[str | int, ...], str, str]] = field(init=False, repr=False)


_VALIDATION_ACTIVE: Final[ContextVar[_ValidationOwner | None]] = ContextVar(
    "swarm_arena_validation_active", default=None
)
_SAFE_MESSAGES: Final = frozenset(
    {
        "amount exceeds the supported USD range",
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
        "run_id must be a single relative key component",
        "run_root must be absolute or s3://",
        "kind requires its matching payload only",
        "s3 backend requires s3_path",
        "redis backend requires redis_url_env",
        "roles must be unique",
        "invalid settable schema",
        "duplicate catalog entry names",
        "cannot load catalog component class",
        "data must be finite JSON",
        "details must be finite JSON",
        "text is not valid UTF-8",
        "TOML data cannot contain null",
        "JSON keys must be strings",
        "JSON nesting exceeds 100 levels",
        "cannot reference a lambda",
        "cannot reference a local object",
        "cannot reference an object defined in __main__",
        "component has no importable class",
    }
)


def _explicit_location(
    model: type[BaseModel],
    location: tuple[str | int, ...],
    code: str,
    message: str,
    known: set[tuple[tuple[str | int, ...], str, str]],
) -> tuple[str | int, ...] | None:
    """Preserve a sanitized explicit-call suffix after Pydantic adds field prefixes."""
    for offset in range(len(location) + 1):
        if (location[offset:], code, message) in known:
            return _safe_location(model, location[:offset]) + location[offset:]
    return None


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
        fixed messages, and ``input=None`` without context. At most 20 original
        details survive, followed by ``and N more errors`` when truncated.
        Counts include errors omitted by explicit nested validation boundaries.
        Raise it ``from None``.

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
    try:
        details: list[InitErrorDetails] = []
        if isinstance(error, ValidationError):
            sequence_paths = _sequence_paths(_schema_index(model))
            owner = _VALIDATION_ACTIVE.get()
            known = (
                getattr(owner, "sanitized", set())
                if owner is not None and owner.thread_id == get_ident()
                else set()
            )
            types: dict[tuple[str, str], PydanticCustomError] = {}
            items = cast(
                list[dict[str, Any]],
                error.errors(include_input=False, include_url=False, include_context=False),
            )
            omitted = 0
            if known:
                original_items = []
                for item in items:
                    # Only a trailer registered by an explicit safe boundary
                    # represents omitted errors; validator text is untrusted.
                    if (
                        item["type"] == "invalid"
                        and ((), "invalid", item["msg"]) in known
                        and (match := _ERROR_CAP_REASON.fullmatch(item["msg"]))
                    ):
                        omitted += int(match[1])
                    else:
                        original_items.append(item)
                items = original_items
            omitted += max(0, len(items) - 20)
            items = items[:20]
            previous: tuple[str, str] | None = None
            safe_type: PydanticCustomError | None = None
            for item in items:
                raw_code: str = item["type"]
                message: str = item["msg"]
                key = (raw_code, message)
                if key != previous:
                    safe_type = types.get(key)
                if safe_type is None:
                    code = raw_code if raw_code in _VALIDATION_ERROR_TYPES else "invalid"
                    # Only complete fixed messages survive; never forward validator text.
                    if message.removeprefix("Value error, ") in _SAFE_MESSAGES:
                        message = message.removeprefix("Value error, ")
                    elif _CATALOG_HUMAN_ONLY_REASON.fullmatch(message):
                        pass
                    elif _NEWER_VERSION_REASON.fullmatch(message):
                        message = "schema_version is newer than this swarm-arena supports"
                    elif message not in _SAFE_MESSAGES:
                        message = code
                    safe_type = PydanticCustomError(
                        cast(LiteralString, code), cast(LiteralString, message)
                    )
                    types[key] = safe_type
                previous = key
                item["type"] = safe_type
                location: tuple[str | int, ...] = item["loc"]
                if (
                    not known
                    and location
                    and isinstance(location[-1], int)
                    and location[:-1] in sequence_paths
                ):
                    continue
                explicit_location = (
                    _explicit_location(model, location, raw_code, key[1], known) if known else None
                )
                # A cached schema proof allows sequence-item errors to reuse the
                # original location tuple; integer mapping keys never qualify.
                if explicit_location is not None:
                    location = explicit_location
                elif not (
                    location and isinstance(location[-1], int) and location[:-1] in sequence_paths
                ):
                    location = _safe_location(
                        model,
                        location,
                        show_unknown=unknown_fields == "show" and raw_code == "extra_forbidden",
                    )
                item["loc"] = location
            details = cast(list[InitErrorDetails], items)
            if omitted:
                details.append(
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "invalid", cast(LiteralString, f"and {omitted} more errors")
                        ),
                        loc=(),
                        input=None,
                    )
                )
        else:
            details.append(
                InitErrorDetails(
                    type=PydanticCustomError(
                        "invalid", cast(LiteralString, f"invalid: {_exception_type_name(error)}")
                    ),
                    loc=(),
                    input=None,
                )
            )
        return _SanitizedValidationError.from_exception_data(
            model.__name__, details, hide_input=True
        )
    except KeyboardInterrupt:
        raise
    except BaseException as failure:
        _propagate_control(failure)
    return _minimal_validation_error(model)


def _minimal_validation_error(model: type[BaseModel]) -> ValidationError:
    """Build the value-free fallback without schema walking or error inspection."""
    return _SanitizedValidationError.from_exception_data(
        model.__name__,
        [{"type": PydanticCustomError("invalid", "invalid"), "loc": (), "input": None}],
        hide_input=True,
    )


def _validation_wrapper(
    model: type[BaseModel], *, by_alias: bool | None = None, explicit: bool = False
) -> Callable[[Any, Callable[[Any], Any]], Any]:
    """Build the one outer validation boundary, including subclass after validators."""

    def validate(data: Any, handler: Callable[[Any], Any]) -> Any:
        """Suppress input-bearing exception chains and rebuild safe error details."""
        current = _VALIDATION_ACTIVE.get()
        if not explicit and current is not None and current.thread_id == get_ident():
            return handler(data)
        owner = _ValidationOwner(get_ident())
        token = _VALIDATION_ACTIVE.set(owner)
        alias_token = (
            _VALIDATION_BY_ALIAS.set(by_alias)
            if _VALIDATION_BY_ALIAS.get() is not by_alias
            else None
        )
        try:
            return handler(data)
        except KeyboardInterrupt:
            raise
        except BaseException as error:
            if isinstance(error, PydanticErrorMixin):
                raise
            try:
                error = _classify_outcome(error)
                safe = _sanitize_validation_error(
                    model, error, unknown_fields=getattr(model, "_unknown_fields", "hide")
                )
            except RecursionError:
                safe = None
            except KeyboardInterrupt:
                raise
            except BaseException as error:
                try:  # noqa: SIM105 - a context manager itself needs spare stack frames
                    _propagate_control(error)
                except RecursionError:
                    pass
                safe = None
        finally:
            owner.thread_id = None
            if alias_token is not None:
                _VALIDATION_BY_ALIAS.reset(alias_token)
            _VALIDATION_ACTIVE.reset(token)
        # No raw error is active if stack exhaustion also defeats the fallback.
        if safe is None:
            safe = _minimal_validation_error(model)
        if current is not None and current.thread_id == get_ident():
            if not hasattr(current, "sanitized"):
                current.sanitized = set()
            current.sanitized.update(
                (item["loc"], item["type"], item["msg"])
                for item in safe.errors(
                    include_input=False, include_url=False, include_context=False
                )
            )
        raise safe from None

    validate.__name__ = model.__name__
    return validate
