"""Detach strings accepted by public context APIs before core operations."""


def _context_string(value: object, field: str) -> str:
    """Return an exact string without subclass hooks or value-bearing errors."""
    # type() reads the real type; isinstance() would consult a user __class__ hook.
    if not issubclass(type(value), str):
        raise TypeError(f"{field} must be str")
    return str.__str__(value)
