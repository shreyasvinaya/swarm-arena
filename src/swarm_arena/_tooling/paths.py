"""Validate workspace paths and match the shared editable-glob dialect."""

import fnmatch
import re
from collections.abc import Sequence
from functools import cache
from pathlib import PurePosixPath
from typing import Final

RESERVED: Final = frozenset({"swarm", "parents", "candidate"})


def check_relpath(path: str) -> str:
    """Validate and normalize a relative POSIX workspace path.

    Parameters
    ----------
    path : str
        Path relative to the workspace root.

    Returns
    -------
    str
        Path with redundant separators and dot segments removed.

    Raises
    ------
    ValueError
        If the path is empty, absolute, contains a parent segment or NUL,
        or uses a reserved top-level name.

    Examples
    --------
    >>> check_relpath("./problem//run.py")
    'problem/run.py'
    """
    parts = PurePosixPath(path).parts
    reason: str | None = None
    if not parts:
        reason = "empty"
    elif path.startswith("/"):
        reason = "absolute"
    elif ".." in parts:
        reason = "contains '..'"
    elif "\x00" in path:
        reason = "contains NUL"
    elif parts[0] in RESERVED:
        reason = f"uses the reserved name {parts[0]!r}"
    if reason is not None:
        raise ValueError(f"path {path!r}: {reason}")
    return PurePosixPath(path).as_posix()


def check_key(path: str) -> str:
    """Require a workspace dictionary key to use its normal form.

    Parameters
    ----------
    path : str
        Relative path used as a dictionary key.

    Returns
    -------
    str
        Validated, unchanged path.

    Raises
    ------
    ValueError
        If the path is invalid or differs from its normalized spelling.

    Examples
    --------
    >>> check_key("problem/run.py")
    'problem/run.py'
    """
    norm = check_relpath(path)
    if norm != path:
        raise ValueError(f"path {path!r}: not in normal form (use {norm!r})")
    return norm


def check_glob(pattern: str) -> str:
    """Validate a relative POSIX glob without changing its spelling.

    Parameters
    ----------
    pattern : str
        Workspace-relative glob. Reserved names are permitted in patterns.

    Returns
    -------
    str
        Validated, unchanged pattern.

    Raises
    ------
    ValueError
        If the glob is empty, absolute, contains a parent segment or NUL.

    Examples
    --------
    >>> check_glob("problem/**/*.phyk")
    'problem/**/*.phyk'
    """
    reason: str | None = None
    if not pattern:
        reason = "empty"
    elif pattern.startswith("/"):
        reason = "absolute"
    elif ".." in PurePosixPath(pattern).parts:
        reason = "contains '..'"
    elif "\x00" in pattern:
        reason = "contains NUL"
    if reason is not None:
        raise ValueError(f"glob {pattern!r}: {reason}")
    return pattern


def _translate_segment(segment: str) -> str:
    """Translate nonrecursive wildcards without allowing a slash match."""
    translated: list[str] = []
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            translated.append("[^/]*")
        elif char == "?":
            translated.append("[^/]")
        elif char == "[":
            end = index + 1
            if end < len(segment) and segment[end] == "!":
                end += 1
            if end < len(segment) and segment[end] == "]":
                end += 1
            while end < len(segment) and segment[end] != "]":
                end += 1
            if end == len(segment):
                translated.append(r"\[")
            else:
                # Delegate ranges, negation and malformed ranges to fnmatch.
                # Strip its end anchor; fullmatch anchors the complete path.
                bracket = fnmatch.translate(segment[index : end + 1])
                bracket = bracket.removesuffix(r"\Z").removesuffix(r"\z")
                translated.append(f"(?!/){bracket}")
                index = end
        else:
            translated.append(re.escape(char))
        index += 1
    return "".join(translated)


@cache
def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile each validated glob once, including recursive segments."""
    parts: list[str] = []
    for part in PurePosixPath(check_glob(pattern)).parts:
        if part != "**" or not parts or parts[-1] != "**":
            parts.append(part)
    translated: list[str] = []
    for index, part in enumerate(parts):
        trailing_recursive = part == "**" and index == len(parts) - 1
        if index and parts[index - 1] != "**" and not trailing_recursive:
            translated.append("/")
        if part == "**":
            if trailing_recursive:
                translated.append("(?:/.*)?" if index else ".*")
            else:
                translated.append("(?:.*/)?")
        else:
            translated.append(_translate_segment(part))
    return re.compile("".join(translated), re.DOTALL)


def matches(path: str, globs: Sequence[str]) -> bool:
    """Match a complete path against any glob in the shared dialect.

    Parameters
    ----------
    path : str
        Normalized, relative POSIX path to match.
    globs : Sequence[str]
        Relative patterns. Stars, question marks and bracket expressions stay
        within a segment. A whole ``**`` segment spans zero or more segments.
        Wildcards include dotfiles; matching is case sensitive.

    Returns
    -------
    bool
        Whether at least one pattern matches the entire path.

    Raises
    ------
    ValueError
        If a pattern examined during matching is invalid.

    Examples
    --------
    >>> matches("problem/a.phyk", ["problem/**/*.phyk"])
    True
    >>> matches("problem/sub/a.phyk", ["problem/*.phyk"])
    False
    """
    return any(_compile_glob(pattern).fullmatch(path) is not None for pattern in globs)
