"""Isolate Pydantic implementation assumptions pinned by compatibility tests."""

from collections.abc import Callable
from typing import Any, TypeVar, cast

from pydantic import BaseModel

_F = TypeVar("_F", bound=Callable[..., Any])


def _standard_init(function: _F) -> _F:
    """Keep nested field construction on the core schema, like BaseModel.__init__."""
    cast(Any, function).__pydantic_base_init__ = True
    return function


def _core_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Read the implementation schema only at this compatibility seam."""
    return cast(Any, model.__pydantic_core_schema__)


def _alias_paths(alias: Any) -> list[tuple[str | int, ...]]:
    """Normalize Pydantic's alias, AliasPath and AliasChoices core-schema shapes."""
    paths = [[alias]] if isinstance(alias, str) else alias or []
    if paths and not isinstance(paths[0], list):
        paths = [paths]
    return [tuple(path) for path in paths]


def _union_branch(choice: Any) -> tuple[dict[str, Any], str | None]:
    """Match Pydantic-core's inferred labels for wrapped union branches."""
    branch, label = choice if isinstance(choice, tuple) else (choice, None)
    node = branch
    while "schema" in node and node.get("type") != "model":
        node = node["schema"]
    if branch.get("type") == "function-wrap":
        fn = branch["function"]["function"]
        label = label or f"function-wrap[{getattr(fn, '__name__', None) or type(fn).__name__}()]"
    return branch, label or getattr(node.get("cls"), "__name__", node.get("type"))
