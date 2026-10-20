"""Prepare effective component settings, snapshots, and matching goals."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeVar

from pydantic import AliasChoices, AliasPath

from swarm_arena._model.goal import Goal
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate
from swarm_arena._scoring.check import Check
from swarm_arena._scoring.judge import Judge
from swarm_arena._tooling.files import snapshot_fileset, snapshot_ref
from swarm_arena._tooling.tool import Tool
from swarm_arena.testing._state import _artifacts

_DEFAULT_GOAL: Final = Goal(id="check", namespace="testing")
_C = TypeVar("_C", Tool, Check, Judge)


def _build(component: _C | Ref, base: type[_C], goal: Goal | None) -> tuple[Ref, _C, Goal]:
    """Snapshot files relative to the working directory, then build and preflight."""
    artifacts, base_dir = _artifacts(), Path.cwd()
    ref = component if isinstance(component, Ref) else Ref.of(component)
    ref = snapshot_ref(ref, base_dir, artifacts)
    built = ref.build(base)
    built.preflight()
    goal = _DEFAULT_GOAL if goal is None else goal
    files = snapshot_fileset(goal.files, base_dir, artifacts)
    return ref, built, goal.model_copy(update={"files": files})


def _effective_tool_ref(tool: Tool | Ref, overrides: Mapping[str, Any]) -> Ref:
    """Merge round settings before snapshot validation, preserving human-only fields."""
    ref = tool if isinstance(tool, Ref) else Ref.of(tool)
    if not overrides:
        return ref
    cls = ref.load()
    if not issubclass(cls, Tool):
        raise TypeError(f"{ref.path} is not a subclass of Tool")
    for name in getattr(cls, "_human_only", frozenset()):
        keys = {name}
        field = cls.Config.model_fields.get(name)
        if field is not None:
            alias = field.validation_alias
            if isinstance(alias, str):
                keys.add(alias)
            elif isinstance(alias, AliasPath):
                keys.add(alias.path[0])
            elif isinstance(alias, AliasChoices):
                keys.update(path[0] for path in alias.convert_to_aliases())
        if keys.intersection(overrides):
            raise ValueError(f"round_overrides: human-only field {name} cannot be overridden")
    # snapshot_ref validates this merged config through cls.Config before reading files.
    return ref.model_copy(update={"config": {**ref.config, **overrides}})


def _check_goal(goal: Goal, candidates: Sequence[Candidate]) -> None:
    """Reject a mismatched goal before building or running a scoring component."""
    for candidate in candidates:
        if candidate.goal_id != goal.id:
            raise ValueError(
                f"candidate {candidate.id} goal_id {candidate.goal_id} "
                f"does not match goal {goal.id}"
            )
