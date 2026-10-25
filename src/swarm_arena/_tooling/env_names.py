"""Discover effective credential names from typed configuration without command hooks."""

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from swarm_arena._model.ref import Configurable, Ref
from swarm_arena._model.tooling import Command
from swarm_arena._planning.provenance import is_nested_ref
from swarm_arena._tooling.tool import Tool, effective_tool_ref


def _env_names(component: Configurable) -> list[tuple[str, str]]:
    """Inspect typed nested defaults as well as explicitly written references."""
    result: list[tuple[str, str]] = []

    def visit(value: Any, pointer: str, *, tool_config: bool = False) -> None:
        if isinstance(value, Ref):
            cls, cfg = value._resolve_config(Configurable)
            visit(cfg, pointer, tool_config=issubclass(cls, Tool))
        elif isinstance(value, BaseModel):
            if tool_config:
                names = getattr(value, "env", ())
                if not isinstance(names, (list, tuple)) or any(
                    not isinstance(name, str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
                    for name in names
                ):
                    raise ValueError("invalid env names")
                result.extend((pointer + "/env", str.__str__(name)) for name in names)
            for field in type(value).model_fields:
                visit(getattr(value, field), pointer + "/config/" + field)
        elif is_nested_ref(value):
            visit(Ref.model_validate(value), pointer)
        elif isinstance(value, dict):
            for child in value.values():
                # Arbitrary mapping keys are not safe diagnostic identifiers.
                visit(child, pointer + "/<key>")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, pointer + f"/{index}")

    visit(component.config, "", tool_config=isinstance(component, Tool))
    return result


def effective_secret_env(
    component: Tool | Ref | Command, *, overrides: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Read unique credential names from validated effective configuration.

    Parameters
    ----------
    component : Tool or Ref or Command
        Already effective tool, tool reference, or driver-known command.
    overrides : Mapping[str, Any] or None, default=None
        Round overrides applied through the shared effective-tool-ref helper.

    Returns
    -------
    tuple[str, ...]
        Sorted unique names; worker-only command hooks are never invoked.

    Examples
    --------
    >>> effective_secret_env(Command(argv=["python"], env=["OPENAI_API_KEY"]))
    ('OPENAI_API_KEY',)
    """
    if isinstance(component, Command):
        return tuple(sorted({str.__str__(name) for name in component.env}))
    if isinstance(component, Ref) or overrides is not None:
        ref = component if isinstance(component, Ref) else Ref.of(component)
        component = effective_tool_ref(ref, dict(overrides or {})).build(Tool)
    return tuple(sorted({name for _, name in _env_names(component)}))
