"""Read TOML and write its supported values in deterministic insertion order."""

import json
import math
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from swarm_arena._errors import ConfigError, _core_error


def read_toml(path: Path) -> dict[str, Any]:
    """
    Read a UTF-8 TOML file with its path in parse errors.

    Parameters
    ----------
    path : Path
        File to read and parse.

    Returns
    -------
    dict[str, Any]
        Parsed TOML document.

    Raises
    ------
    ConfigError
        If TOML parsing fails.

    Examples
    --------
    >>> read_toml(Path("plan.toml"))  # doctest: +SKIP
    {'revision': 0}
    """
    try:
        text = path.read_text("utf-8")
    except (OSError, UnicodeError):
        raise _core_error(ConfigError, f"{path}: cannot read valid UTF-8 TOML") from None
    return loads_toml(text, where=str(path))


def loads_toml(text: str, *, where: str) -> dict[str, Any]:
    """
    Parse TOML without exposing snippets, preserving safe line and column numbers.

    Parameters
    ----------
    text : str
        TOML source text.
    where : str
        File path or format identifier for diagnostics.

    Returns
    -------
    dict[str, Any]
        Parsed TOML fields.

    Raises
    ------
    ConfigError
        If source encoding or syntax is invalid.

    Examples
    --------
    >>> loads_toml('count = 2', where="config")
    {'count': 2}
    """
    try:
        text.encode("utf-8", errors="strict")
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        match = re.search(r"\(at line ([0-9]+), column ([0-9]+)\)$", str(error))
        if match:
            line, column = match.groups()
        else:
            # Python versions that report end-of-document still have a safe position.
            line, column = str(text.count("\n") + 1), str(len(text.rsplit("\n", 1)[-1]) + 1)
        raise _core_error(
            ConfigError, f"{where}: invalid TOML at line {line}, column {column}"
        ) from None
    except UnicodeError:
        raise _core_error(ConfigError, f"{where}: cannot read valid UTF-8 TOML") from None


def _quote(value: str) -> str:
    """Quote a string with TOML-compatible JSON escapes."""
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def _key(key: str) -> str:
    """Keep bare keys and quote all other keys, including dotted ones."""
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else _quote(key)


def _inline(value: Any) -> str:
    """Render a value without emitting a table header."""
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TOML floats must be finite")
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_inline(item) for item in value) + "]"
    if isinstance(value, Mapping):
        items = (
            f"{_key(key)} = {_inline(item)}" for key, item in value.items() if item is not None
        )
        return "{" + ", ".join(items) + "}"
    raise TypeError(f"unsupported TOML value type: {type(value).__name__}")


def _is_array_of_tables(value: Any) -> bool:
    """Identify a non-empty list containing only dictionaries."""
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def _table(
    data: Mapping[str, Any], path: tuple[str, ...], header: str | None, blocks: list[str]
) -> None:
    """Append scalar entries, then subtables, then arrays of tables."""
    lines = [] if header is None else [header]
    tables: list[tuple[str, Mapping[str, Any]]] = []
    arrays: list[tuple[str, list[dict[str, Any]]]] = []
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and value:
            tables.append((key, value))
        elif _is_array_of_tables(value):
            arrays.append((key, value))
        else:
            lines.append(f"{_key(key)} = {_inline(value)}")
    if lines:
        blocks.append("\n".join(lines))
    for key, table in tables:
        child = (*path, key)
        name = ".".join(_key(part) for part in child)
        _table(table, child, f"[{name}]", blocks)
    for key, rows in arrays:
        child = (*path, key)
        name = ".".join(_key(part) for part in child)
        for row in rows:
            _table(row, child, f"[[{name}]]", blocks)


def dumps_toml(data: Mapping[str, Any]) -> str:
    """
    Serialize a mapping as deterministic TOML without null entries.

    Parameters
    ----------
    data : Mapping[str, Any]
        Insertion-ordered string keys and supported values: strings, booleans,
        integers, finite floats, lists, and nested mappings. Mapping entries
        whose value is ``None`` are dropped.

    Returns
    -------
    str
        TOML with scalar values before tables, then arrays of tables, and a
        blank line between table headers. Non-empty output ends in a newline.

    Raises
    ------
    ValueError
        If a float is not finite.
    TypeError
        If a value has an unsupported type.

    Examples
    --------
    >>> import tomllib
    >>> tomllib.loads(dumps_toml({"name": "é", "tool": {"enabled": True}}))
    {'name': 'é', 'tool': {'enabled': True}}
    """
    blocks: list[str] = []
    _table(data, (), None, blocks)
    return "\n\n".join(blocks) + "\n" if blocks else ""
