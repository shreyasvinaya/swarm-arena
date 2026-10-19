"""Validate workspace paths and match the shared editable-glob dialect."""

import fnmatch
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import PurePosixPath
from typing import Final, TypeVar

RESERVED: Final = frozenset({"swarm", "parents", "candidate"})
_T = TypeVar("_T")


def fold_key(path: str) -> str:
    """Fold a path for case- and Unicode-insensitive security comparisons.

    Parameters
    ----------
    path : str
        Path spelling to compare without changing its stored representation.

    Returns
    -------
    str
        NFC-normalized path with Unicode case folding applied.

    Examples
    --------
    >>> fold_key("problem/RUN.py")
    'problem/run.py'
    """
    return unicodedata.normalize("NFC", path).casefold()


def fold_conflicts(
    files: Iterable[str], *, directories: Iterable[str] = ()
) -> list[tuple[str, str]]:
    """Find folded layout conflicts, including implicit ancestor directories.

    Parameters
    ----------
    files : Iterable[str]
        Normal-form file paths. Repeated files are conflicts; callers allowing
        exact-path replacement must merge those entries first.
    directories : Iterable[str], optional
        Explicit directories, including empty generated directories.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of original input paths whose nodes disagree in spelling or
        file/directory kind, or repeat a file. Shared same-spelling directories
        are allowed. Paths need not pass ``check_key``: sink paths can include
        generated reserved trees.

    Examples
    --------
    >>> fold_conflicts(["p/a.txt", "P/b.txt"])
    [('p/a.txt', 'P/b.txt')]
    >>> fold_conflicts(["p", "p/a.txt"])
    [('p', 'p/a.txt')]
    >>> fold_conflicts(["p/a.txt", "p/b.txt"])
    []
    """
    nodes: dict[str, tuple[str, bool, str]] = {}
    conflicts: dict[tuple[str, str], None] = {}
    for paths, is_directory in ((files, False), (directories, True)):
        for rel in paths:
            path = PurePosixPath(rel)
            for component in (path, *path.parents):
                if component == PurePosixPath("."):
                    continue
                spelling = component.as_posix()
                directory = is_directory or component != path
                key = fold_key(spelling)
                previous = nodes.get(key)
                if previous is None:
                    nodes[key] = (spelling, directory, rel)
                    continue
                if previous[0] != spelling or not (previous[1] and directory):
                    conflicts[previous[2], rel] = None
    return list(conflicts)


def fold_merge(base: Mapping[str, _T], overlay: Mapping[str, _T]) -> dict[str, _T]:
    """Merge file maps by folded key, keeping each winner's spelling.

    Parameters
    ----------
    base : Mapping[str, _T]
        Lower-priority file entries.
    overlay : Mapping[str, _T]
        Higher-priority entries. These replace folded matches in ``base``.

    Returns
    -------
    dict[str, _T]
        A new mapping with a valid folded layout.

    Raises
    ------
    ValueError
        If either input or the merged layout has conflicting paths.

    Examples
    --------
    >>> fold_merge({"p/a": "old"}, {"P/A": "new"})
    {'P/A': 'new'}
    """
    conflicts = fold_conflicts(base) + fold_conflicts(overlay)
    if conflicts:
        first, second = conflicts[0]
        raise ValueError(f"conflicting workspace path: {first} and {second}")
    entries = {fold_key(path): (path, value) for path, value in base.items()}
    entries.update({fold_key(path): (path, value) for path, value in overlay.items()})
    merged = dict(entries.values())
    conflicts = fold_conflicts(merged)
    if conflicts:
        first, second = conflicts[0]
        raise ValueError(f"conflicting workspace path: {first} and {second}")
    return merged


def output_paths(
    declared: Sequence[str], *, effective_files: Iterable[str], directories: Iterable[str] = ()
) -> list[str]:
    """Normalize outputs and reject conflicts within their destination layout.

    Parameters
    ----------
    declared : Sequence[str]
        Output declarations, including any redundant separators or dot segments.
    effective_files : Iterable[str]
        Original staged file paths. Exact-spelling output replacements are
        allowed; differently spelled aliases are refused.
    directories : Iterable[str], optional
        Explicit staged directories.

    Returns
    -------
    list[str]
        Normalized output paths in declaration order.

    Raises
    ------
    ValueError
        If a path is invalid or the combined layout conflicts. Diagnostics
        name paths only, never file contents.

    Examples
    --------
    >>> output_paths(["./p/a.txt"], effective_files=["p/a.txt"])
    ['p/a.txt']
    """
    paths = [check_relpath(path) for path in declared]
    conflicts = fold_conflicts(paths)
    if conflicts:
        first, second = conflicts[0]
        originals = [
            original
            for original, rel in zip(declared, paths, strict=True)
            if rel in (first, second)
        ]
        raise ValueError(
            "duplicate output path or conflicting output paths: " + " and ".join(originals)
        )
    conflicts = fold_conflicts(dict.fromkeys([*effective_files, *paths]), directories=directories)
    if conflicts:
        first, second = conflicts[0]
        raise ValueError(f"conflicting output path with staged layout: {first} and {second}")
    return paths


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
    elif fold_key(parts[0]) in RESERVED:
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
