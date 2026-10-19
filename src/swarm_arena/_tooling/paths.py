"""Validate workspace paths and match the shared editable-glob dialect."""

import fnmatch
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import PurePosixPath
from typing import Final, TypeVar

RESERVED: Final = frozenset({"swarm", "parents", "candidate"})
PARENT_DATA: Final = "data.json"
_T = TypeVar("_T")


def fold_key(path: str) -> str:
    r"""Fold a path for case- and Unicode-insensitive security comparisons.

    This is canonical caseless matching (Unicode D145): decompose, case fold,
    then compose. Case folding can return text that is not normalized, so
    folding NFC text alone splits names that APFS treats as one file, such as
    a long s (U+017F) with a combining caron and an s with caron (U+0161).
    Equal keys mean equal names under any mix of case and composed or
    decomposed spelling.

    Parameters
    ----------
    path : str
        Path spelling to compare without changing its stored representation.

    Returns
    -------
    str
        ``NFC(casefold(NFD(path)))``, an internal comparison key.

    Examples
    --------
    >>> fold_key("problem/RUN.py")
    'problem/run.py'
    >>> alias = "\N{LATIN SMALL LETTER LONG S}\N{COMBINING CARON}olver.py"
    >>> fold_key(alias) == fold_key("\N{LATIN SMALL LETTER S WITH CARON}olver.py")
    True
    >>> decomposed = "Cafe\N{COMBINING ACUTE ACCENT}"
    >>> fold_key(decomposed) == fold_key("caf\N{LATIN SMALL LETTER E WITH ACUTE}")
    True
    """
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", path).casefold())


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


def fold_merge(
    base: Mapping[str, _T], overlay: Mapping[str, _T], *, prefix: str = ""
) -> dict[str, _T]:
    """Merge file maps by folded key; a replacement keeps the base spelling.

    This is the one spelling rule for every stage: a later layer replaces a
    file's content, never its name. Every later stage starts from the same
    goal manifest, so the goal's spelling is the one all stages agree on.

    Parameters
    ----------
    base : Mapping[str, _T]
        Lower-priority file entries. Their spellings are kept.
    overlay : Mapping[str, _T]
        Higher-priority entries. Each replaces the value of the ``base`` entry
        with the same folded key; other entries keep their own spelling.
    prefix : str, optional
        Workspace prefix of the tree, such as ``"parents/c1/"``. Errors name
        paths with it, so they show which parent or tree failed.

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
    >>> fold_merge({"p/a": "old"}, {"p/A": "new", "p/b": "add"})
    {'p/a': 'new', 'p/b': 'add'}
    >>> fold_merge({"p/a": "x"}, {"P/b": "y"}, prefix="parents/c1/")
    Traceback (most recent call last):
    ...
    ValueError: conflicting workspace path: parents/c1/p/a and parents/c1/P/b
    """

    def check(files: Iterable[str]) -> None:
        conflicts = fold_conflicts(files)
        if conflicts:
            first, second = conflicts[0]
            raise ValueError(f"conflicting workspace path: {prefix}{first} and {prefix}{second}")

    check(base)
    check(overlay)
    entries = {fold_key(path): (path, value) for path, value in base.items()}
    for path, value in overlay.items():
        key = fold_key(path)
        entries[key] = (entries[key][0] if key in entries else path, value)
    merged = dict(entries.values())
    check(merged)
    return merged


def parent_tree(
    overlay: Mapping[str, _T], outputs: Mapping[str, _T], *, prefix: str = ""
) -> dict[str, _T]:
    """Lay out a parent's read-only tree: output files over its overlay.

    The core writes the parent's structured output to ``data.json`` at the top
    of the tree, so an entry whose whole path folds to ``data.json`` is
    skipped. Producers refuse paths below a folded ``data.json`` directory
    with ``parent_data_conflicts``, the same rule the sink applies.

    Parameters
    ----------
    overlay : Mapping[str, _T]
        The parent's effective overlay.
    outputs : Mapping[str, _T]
        The parent's output files. At a shared folded key an output replaces
        the overlay's content, and the overlay's spelling is kept.
    prefix : str, optional
        Workspace prefix for errors, such as ``"parents/c1/"``.

    Returns
    -------
    dict[str, _T]
        Tree-relative paths to values, without the generated ``data.json``.

    Raises
    ------
    ValueError
        If the overlay and outputs do not form a valid folded layout.

    Examples
    --------
    >>> parent_tree({"p/a": "old", "Data.json": "x"}, {"p/a": "out"})
    {'p/a': 'out'}
    """
    merged = fold_merge(overlay, outputs, prefix=prefix)
    return {path: value for path, value in merged.items() if fold_key(path) != PARENT_DATA}


def parent_data_conflicts(paths: Iterable[str]) -> list[str]:
    """Return paths that a later parent tree cannot hold beside ``data.json``.

    A candidate's overlay and output files are staged under ``parents/<id>/``
    in later runs, next to the generated ``data.json`` file. A path below a
    top-level directory that folds to ``data.json`` would conflict with that
    file there, so producers refuse it in the run that made it. A path that
    folds to ``data.json`` itself is skipped by ``parent_tree`` instead.

    Parameters
    ----------
    paths : Iterable[str]
        Tree-relative overlay, edit, or output paths in normal form.

    Returns
    -------
    list[str]
        Conflicting input paths, in input order.

    Examples
    --------
    >>> parent_data_conflicts(["Data.json/x", "data.json", "p/data.json/x"])
    ['Data.json/x']
    """
    kept = [path for path in paths if fold_key(path) != PARENT_DATA]
    return [path for first, path in fold_conflicts([PARENT_DATA, *kept]) if first == PARENT_DATA]


def output_paths(
    declared: Sequence[str], *, effective_files: Iterable[str], directories: Iterable[str] = ()
) -> list[str]:
    """Normalize outputs and reject conflicts within their destination layout.

    Outputs are also checked against the later parent tree they join, so an
    output below a folded top-level ``data.json`` directory fails the run that
    made it rather than every later stage that stages the candidate.

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
    >>> output_paths(["Data.json/x"], effective_files=[])
    Traceback (most recent call last):
    ...
    ValueError: conflicting output path with parent data.json: Data.json/x
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
    if reserved := parent_data_conflicts(paths):
        raise ValueError(f"conflicting output path with parent {PARENT_DATA}: {reserved[0]}")
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
