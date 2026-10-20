"""Shared assertions for the public surfaces of sanitized exceptions."""

import traceback

from pydantic import ValidationError


def assert_secret_absent(error: BaseException, secret: str, *, sanitized: bool = False) -> None:
    """
    Assert that visible exception data and chains omit a secret.

    Inspect each node's data without traceback frames, whose source code may
    contain sentinel literals. Follow explicit causes, otherwise unsuppressed
    contexts, and inspect exception group children. Suppressed contexts are
    allowed under ruling 144.

    Parameters
    ----------
    error : BaseException
        Failure whose public diagnostics are inspected.
    secret : str
        Sentinel that must be absent.
    sanitized : bool, default=False
        Also require value-free Pydantic details and suppressed original contexts.

    Examples
    --------
    >>> assert_secret_absent(ValueError("invalid"), "private")
    """
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        surfaces = [
            str(current),
            repr(current),
            repr(current.args),
            repr(getattr(current, "__notes__", ())),
            "".join(traceback.format_exception_only(current)),
        ]
        if isinstance(current, ValidationError):
            surfaces.extend([repr(current.errors()), current.json()])
            if sanitized:
                assert all(item["input"] is None and "ctx" not in item for item in current.errors())
        assert all(secret not in surface for surface in surfaces)
        if sanitized:
            assert current.__cause__ is None
            assert current.__context__ is None or current.__suppress_context__
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif not current.__suppress_context__ and current.__context__ is not None:
            pending.append(current.__context__)
