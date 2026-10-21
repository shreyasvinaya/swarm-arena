"""Detach strings accepted by public context APIs before core operations."""


def _context_string(value: object, field: str) -> str:
    """Return an exact string without subclass hooks or value-bearing errors."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    return str.__str__(value)
