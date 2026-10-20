"""Retain only schema-proven identifiers in validation-error locations."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Final, cast

from pydantic import BaseModel

from swarm_arena._model.compat import _alias_paths, _core_schema, _union_branch

_VALIDATION_BY_ALIAS: Final[ContextVar[bool | None]] = ContextVar(
    "swarm_arena_validation_by_alias", default=None
)


def _schema_definitions(schema: dict[str, Any]) -> dict[str, Any]:
    """Index core-schema references without following repeated or recursive objects."""
    refs: dict[str, Any] = {}
    pending: list[Any] = [schema]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if not isinstance(node, (dict, list, tuple)) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, dict):
            if isinstance(node.get("ref"), str):
                refs[node["ref"]] = node
            pending.extend(node.values())
        else:
            pending.extend(node)
    return refs


@dataclass
class _SchemaIndex:
    """Retain schema-only routes and references, never rejected input data."""

    schema: dict[str, Any]
    refs: dict[str, Any]
    sequence_paths: dict[bool | None, frozenset[tuple[str | int, ...]]]


def _schema_index(model: type[BaseModel]) -> _SchemaIndex:
    """Cache references on each class, invalidating on core-schema identity changes."""
    schema = _core_schema(model)
    cached = model.__dict__.get("__swarm_arena_location_index__")
    if cached is None or cached.schema is not schema:
        cached = _SchemaIndex(schema, _schema_definitions(schema), {})
        cast(Any, model).__swarm_arena_location_index__ = cached
    return cached


def _common_location(paths: list[tuple[str | int, ...]]) -> tuple[str | int, ...]:
    """Keep only segments all matching schema paths establish as identifiers."""
    return tuple(
        parts[0] if all(part == parts[0] for part in parts) else "<…>"
        for parts in zip(*paths, strict=True)
    )


def _location_parts(
    schema: dict[str, Any],
    location: tuple[str | int, ...],
    refs: dict[str, Any],
    *,
    show_unknown: bool,
) -> tuple[str | int, ...]:
    """Walk schema nodes, retaining identifiers and redacting each data segment."""
    if not location:
        return ()
    # Wrappers do not consume a location; guard pathological reference-only loops.
    seen: set[int] = set()
    loc_by_alias = True
    validate_by_alias = True
    while id(schema) not in seen:
        seen.add(id(schema))
        kind = schema.get("type")
        if kind == "definition-ref":
            schema = refs.get(schema["schema_ref"], {})
        elif kind == "model":
            config = schema.get("config", {})
            loc_by_alias = config.get("loc_by_alias", True)
            validate_by_alias = config.get("validate_by_alias", True)
            schema = schema.get("schema", {})
        elif kind == "definitions" or "schema" in schema:
            schema = schema.get("schema", {})
        elif kind == "json-or-python":
            schema = schema["python_schema"]
        elif kind == "lax-or-strict":
            schema = schema["strict_schema"]
        else:
            break
    first, *tail = location
    rest = tuple(tail)
    kind = schema.get("type")
    if kind in {"model-fields", "typed-dict"}:
        aliases: list[tuple[tuple[str | int, ...], dict[str, Any]]] = []
        names: list[tuple[tuple[str | int, ...], dict[str, Any]]] = []
        for name, field in schema["fields"].items():
            names.append(((name,), field))
            alias = field.get("validation_alias")
            aliases.extend((path, field) for path in _alias_paths(alias))
        # Alias/name collisions can describe different schema nodes. Retain a
        # segment only when all matching paths establish it as an identifier.
        by_alias = _VALIDATION_BY_ALIAS.get()
        use_alias = validate_by_alias if by_alias is None else by_alias
        paths = [*aliases, *names] if loc_by_alias and use_alias else names
        matches = [
            path
            + _location_parts(
                field["schema"], location[len(path) :], refs, show_unknown=show_unknown
            )
            for path, field in paths
            if location[: len(path)] == path
        ]
        if matches:
            return _common_location(matches)
        return ((first if show_unknown else "<field>"),) + ("<…>",) * len(rest)
    if kind in {"list", "set", "frozenset", "generator", "tuple"} and isinstance(first, int):
        items = schema.get("items_schema", {})
        if kind == "tuple":
            index = schema.get("variadic_item_index")
            items = (
                items[first]
                if 0 <= first < len(items)
                else (items[index] if index is not None else {})
            )
        return (first, *_location_parts(items, rest, refs, show_unknown=show_unknown))
    if kind == "dict":
        if rest and rest[0] == "[key]":
            return ("<…>",) * len(location)
        return (
            "<…>",
            *_location_parts(
                schema.get("values_schema", {}), rest, refs, show_unknown=show_unknown
            ),
        )
    if kind == "tagged-union":
        return (
            "<…>",
            *_location_parts(
                schema["choices"].get(first, {}), rest, refs, show_unknown=show_unknown
            ),
        )
    if kind == "union":
        matches = []
        for choice in schema["choices"]:
            branch, label = _union_branch(choice)
            if first == label:
                matches.append(
                    ("<…>", *_location_parts(branch, rest, refs, show_unknown=show_unknown))
                )
        if matches:
            return _common_location(matches)
    return ("<…>",) * len(location)


def _safe_location(
    model: type[BaseModel], location: tuple[str | int, ...], *, show_unknown: bool = False
) -> tuple[str | int, ...]:
    """Retain schema fields, aliases and indices, with data keys always redacted."""
    try:
        index = _schema_index(model)
        return _location_parts(index.schema, location, index.refs, show_unknown=show_unknown)
    except Exception:
        return ("<…>",) * len(location)


def _sequence_paths(index: _SchemaIndex) -> frozenset[tuple[str | int, ...]]:
    """Prove and cache direct sequence routes for large repeated item errors."""
    mode = _VALIDATION_BY_ALIAS.get()
    if mode in index.sequence_paths:
        return index.sequence_paths[mode]
    node = index.schema
    seen: set[int] = set()
    while id(node) not in seen:
        seen.add(id(node))
        if node.get("type") == "definition-ref":
            node = index.refs.get(node["schema_ref"], {})
        elif "schema" in node:
            node = node["schema"]
        else:
            break
    paths: set[tuple[str | int, ...]] = set()
    if node.get("type") in {"model-fields", "typed-dict"}:
        for name, field in node["fields"].items():
            candidates = [(name,)]
            alias = field.get("validation_alias")
            candidates.extend(_alias_paths(alias))
            for path in candidates:
                probe = (*path, 0)
                if _location_parts(index.schema, probe, index.refs, show_unknown=False) == probe:
                    paths.add(path)
    result = frozenset(paths)
    index.sequence_paths[mode] = result
    return result
