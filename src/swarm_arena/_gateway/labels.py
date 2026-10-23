"""Normalize untrusted gateway labels and bound profile diagnostics."""

import re
from collections.abc import Sequence


def _observation_label(value: object, limit: int, *, fallback: str = "unknown") -> str:
    """Normalize untrusted labels to the recorder's bounded printable domain."""
    if type(value) is str and 0 < len(value) <= limit and value.isprintable() and value.strip():
        return value
    return fallback


def _profile_failures(failures: Sequence[str]) -> tuple[str, ...]:
    """Keep ten printable diagnostics and one idempotent omission count."""
    bounded = tuple(_observation_label(value, 512, fallback="<unknown>") for value in failures[:10])
    if len(failures) <= 10:
        return bounded
    if (
        len(failures) == 11
        and len(failures[-1]) <= 512
        and re.fullmatch(r"and [1-9][0-9]* more", failures[-1])
    ):
        return (*bounded, failures[-1])
    return (*bounded, f"and {len(failures) - 10} more")
