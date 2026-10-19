"""Shared assertions for the public surfaces of sanitized exceptions."""

import traceback

from pydantic import ValidationError


def assert_secret_absent(error: BaseException, secret: str) -> None:
    """Assert that visible exception surfaces and explicit causes omit a secret."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        surfaces = [
            str(current),
            repr(current),
            repr(current.args),
            repr(getattr(current, "__notes__", ())),
            "".join(traceback.format_exception(current)),
        ]
        if isinstance(current, ValidationError):
            surfaces.extend([repr(current.errors()), current.json()])
        assert all(secret not in surface for surface in surfaces)
        current = current.__cause__
