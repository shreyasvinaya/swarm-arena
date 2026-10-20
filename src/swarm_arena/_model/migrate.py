"""Register and apply sequential stored-format migrations."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Final

MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

CURRENT: Final[Mapping[str, int]] = MappingProxyType(
    {
        "candidate": 1,
        "learning": 1,
        "llm_call": 1,
        "plan": 1,
        "run_config": 1,
        "job_started": 1,
        "request": 1,
        "candidate_request": 1,
        "candidate_output": 1,
        "tool_result": 1,
        "check_result": 1,
        "judge_result": 1,
    }
)
_migration_functions: Final[dict[tuple[str, int], MigrationFn]] = {}
_MIGRATIONS: Final[Mapping[tuple[str, int], MigrationFn]] = MappingProxyType(_migration_functions)


def migration(kind: str, from_version: int) -> Callable[[MigrationFn], MigrationFn]:
    """
    Register a function that upgrades one stored-format version.

    Parameters
    ----------
    kind : str
        Stored-format registry kind.
    from_version : int
        Version consumed by the function; the next version is produced.

    Returns
    -------
    Callable[[MigrationFn], MigrationFn]
        Decorator that registers and returns the original migration function.

    Raises
    ------
    ValueError
        If a migration for the kind and version is already registered.

    Examples
    --------
    >>> callable(migration("plan", 1))
    True
    """

    def register(fn: MigrationFn) -> MigrationFn:
        """Register a step without replacing an existing migration."""
        key = (kind, from_version)
        if key in _MIGRATIONS:
            raise ValueError(f"migration for {kind} {from_version} already registered")
        _migration_functions[key] = fn
        return fn

    return register


def upgrade(kind: str, raw: dict[str, Any]) -> dict[str, Any]:
    """
    Copy stored data and apply migrations up to the supported version.

    Parameters
    ----------
    kind : str
        Stored-format registry kind.
    raw : dict[str, Any]
        Stored fields; ``schema_version`` must be a positive integer, excluding
        booleans. An absent version means version one.

    Returns
    -------
    dict[str, Any]
        Upgraded fields with an explicit current ``schema_version``.

    Raises
    ------
    ValueError
        If the kind is unknown, the stored version is invalid, or the stored
        version is newer than supported.
    RuntimeError
        If a required migration step is missing.

    Examples
    --------
    >>> upgrade("plan", {"x": 1})
    {'x': 1, 'schema_version': 1}
    """
    if kind not in CURRENT:
        raise ValueError(f"unknown stored-format kind {kind!r}")
    current = CURRENT[kind]
    data = raw.copy()
    version = data.get("schema_version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("schema_version must be a positive integer")
    if version > current:
        raise ValueError(
            f"{kind} schema_version {version} is newer than this swarm-arena supports ({current})"
        )
    while version < current:
        step = _MIGRATIONS.get((kind, version))
        if step is None:
            raise RuntimeError(f"no migration for {kind} {version}")
        data = step(data)
        version += 1
        data["schema_version"] = version
    data["schema_version"] = current
    return data
