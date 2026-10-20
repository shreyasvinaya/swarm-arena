"""Find editable Python-module names for construction-time import warnings."""

import fnmatch
import re
import string
from collections import deque
from collections.abc import Collection
from pathlib import PurePosixPath

from swarm_arena._tooling.paths import fold_key, matches


def _can_create_python(pattern: str, directory: str, protected: Collection[str]) -> bool:
    """Intersect a glob's final segment with Python filenames in one directory.

    The finite search tracks the glob position, the ``.py`` suffix automaton,
    and prefixes of protected names. Full-path matching checks the witness
    against the shared glob implementation before reporting a warning.
    """
    prefix = "" if directory == "." else directory + "/"
    leaf = PurePosixPath(pattern).name
    tokens = re.findall(r"\*|\?|\[!?\]?[^]]*\]|.", leaf)
    forbidden = {
        fold_key(PurePosixPath(path).name)
        for path in protected
        if str(PurePosixPath(path).parent) == directory
    }
    # Range boundaries and their neighbors represent Unicode character classes;
    # ASCII includes every character needed for the literal .py suffix.
    alphabet = set(string.ascii_letters + string.digits + "_.-λ") | set(leaf)
    for char in leaf:
        for point in (ord(char) - 1, ord(char) + 1):
            if 0 <= point <= 0x10FFFF:
                alphabet.add(chr(point))
    alphabet = {char for char in alphabet if char.isprintable() and char != "/"}
    pending: deque[tuple[int, int, str | None, str]] = deque([(0, 0, "", "")])
    seen: set[tuple[int, int, str | None]] = set()
    while pending:
        index, suffix, locked, name = pending.popleft()
        state = (index, suffix, locked)
        if state in seen:
            continue
        seen.add(state)
        if index == len(tokens):
            path = prefix + name
            if (
                suffix == 3
                and fold_key(path) not in {fold_key(p) for p in protected}
                and matches(path, [pattern])
            ):
                return True
            continue
        token = tokens[index]
        if token == "*":
            pending.append((index + 1, suffix, locked, name))
        for char in sorted(alphabet):
            if not fnmatch.fnmatchcase(char, token):
                continue
            text = ".py"[:suffix] + char
            next_suffix = max(length for length in range(4) if text.endswith(".py"[:length]))
            next_locked = None if locked is None else fold_key(locked + char)
            if next_locked is not None and not any(p.startswith(next_locked) for p in forbidden):
                next_locked = None
            pending.append(
                (index if token == "*" else index + 1, next_suffix, next_locked, name + char)
            )
    return False
