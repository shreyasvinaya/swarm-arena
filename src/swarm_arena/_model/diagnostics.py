"""Transport and render validation errors that already contain sanitized details."""

from collections.abc import Callable
from typing import LiteralString, cast

from pydantic import ValidationError as _PydanticValidationError
from pydantic_core import ErrorDetails, InitErrorDetails, PydanticCustomError


class ValidationError(_PydanticValidationError):
    """Preserve context-free custom error codes across pickle and deep copy."""

    def __reduce__(
        self,
    ) -> tuple[Callable[..., _PydanticValidationError], tuple[str, list[ErrorDetails]]]:
        """Rebuild safe codes as custom errors rather than context-requiring built-ins."""
        return _restore_validation_error, (self.title, self.errors(include_url=False))


_SanitizedValidationError = ValidationError


def _restore_validation_error(title: str, errors: list[ErrorDetails]) -> _PydanticValidationError:
    """Restore only the value-free details emitted by the safe validation boundary."""
    details = [
        InitErrorDetails(
            type=PydanticCustomError(
                cast(LiteralString, item["type"]), cast(LiteralString, item["msg"])
            ),
            loc=item["loc"],
            input=None,
        )
        for item in errors
    ]
    return ValidationError.from_exception_data(title, details, hide_input=True)


def validation_summary(
    error: _PydanticValidationError, *, root: str = "", codes_only: bool = False
) -> str:
    """
    Render locations and messages from an already sanitized validation error.

    Parameters
    ----------
    error : ValidationError
        Error produced by the shared model boundary or sanitizer. Raw Pydantic
        errors must be sanitized before calling this renderer.
    root : str, default=''
        Identifier to use for an empty error location.
    codes_only : bool, default=False
        Show error codes instead of messages, retaining an omitted-error count.

    Returns
    -------
    str
        Semicolon-separated field diagnostics without inputs or contexts.

    Examples
    --------
    >>> from swarm_arena._model.base import FrozenModel
    >>> class Count(FrozenModel):
    ...     count: int
    >>> try:
    ...     Count(count="bad")
    ... except ValidationError as error:
    ...     print(validation_summary(error))
    count: int_parsing
    """
    reasons = []
    for item in error.errors(include_input=False, include_url=False, include_context=False):
        location = ".".join(str(part) for part in item["loc"]) or root
        message = item["msg"]
        if codes_only and not (
            not item["loc"]
            and item["type"] == "invalid"
            and message.startswith("and ")
            and message.endswith(" more errors")
        ):
            message = item["type"]
        reasons.append((location + ": " if location else "") + message)
    return "; ".join(reasons)
