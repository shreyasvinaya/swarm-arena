"""Validate portable JSON data and content-addressed file manifests."""

import math
from typing import Annotated, Any, Final

from pydantic import AfterValidator, Field

from swarm_arena._model.canonical import canonical_json
from swarm_arena._tooling.paths import check_key

MAX_JSON_DEPTH: Final = 100
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Manifest = dict[Annotated[str, AfterValidator(check_key)], Sha256]


def check_utf8(value: Any) -> Any:
    """
    Require strict UTF-8 for strings and container keys without changing values.

    Parameters
    ----------
    value : Any
        Field value, possibly a nested mapping or sequence.

    Returns
    -------
    Any
        Unchanged input.

    Raises
    ------
    ValueError
        If a string contains a lone surrogate.

    Examples
    --------
    >>> check_utf8({"text": ["λ"]})
    {'text': ['λ']}
    """
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8", errors="strict")
            except UnicodeError:
                raise ValueError("text is not valid UTF-8") from None
        elif isinstance(item, (dict, list, tuple)) and id(item) not in seen:
            seen.add(id(item))
            if isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            else:
                pending.extend(item)
    return value


def finite_json(value: Any, *, allow_none: bool = True) -> Any:
    """
    Require finite JSON with string keys, strict UTF-8 and bounded nesting.

    Parameters
    ----------
    value : Any
        JSON value to validate without coercion.
    allow_none : bool, default=True
        Permit JSON null; disable for data stored in TOML.

    Returns
    -------
    Any
        Unchanged input.

    Raises
    ------
    ValueError
        If a value is not portable JSON or is nested beyond 100 containers.
        Diagnostics contain only a fixed reason, never data keys or values.

    Examples
    --------
    >>> finite_json({"values": [1, 2.5, None]})
    {'values': [1, 2.5, None]}
    """
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise ValueError("JSON nesting exceeds 100 levels")
        if item is None:
            if not allow_none:
                raise ValueError("TOML data cannot contain null")
        elif isinstance(item, str):
            check_utf8(item)
        elif type(item) in (bool, int):
            pass
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("data must be finite JSON")
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("JSON keys must be strings")
                check_utf8(key)
                pending.append((child, depth + 1))
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        else:
            raise ValueError("data must be finite JSON")
    try:
        canonical_json(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("data must be finite JSON") from None
    return value


def _toml_json(value: Any) -> Any:
    """Validate free-form TOML data without dropping explicit nulls."""
    return finite_json(value, allow_none=False)


FiniteJsonObject = Annotated[dict[str, Any], AfterValidator(finite_json)]
TomlJsonObject = Annotated[dict[str, Any], AfterValidator(_toml_json)]


def json_equal(left: Any, right: Any) -> bool:
    """
    Compare validated JSON values without conflating scalar types.

    Parameters
    ----------
    left : Any
        Validated JSON value.
    right : Any
        Validated JSON value to compare.

    Returns
    -------
    bool
        Whether types and values agree recursively, ignoring mapping order.

    Examples
    --------
    >>> json_equal({"x": [True, 1.0]}, {"x": [1, 1.0]})
    False
    >>> json_equal({"a": 1, "b": 2}, {"b": 2, "a": 1})
    True
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            json_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right
