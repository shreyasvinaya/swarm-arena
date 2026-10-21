from collections.abc import Callable
from dataclasses import MISSING, asdict, fields
from typing import Any, cast
from unittest.mock import Mock

import pytest

from swarm_arena._agent.runtime import AgentCounters, AgentRuntime
from swarm_arena._model.ref import Ref


def _values() -> dict[str, Any]:
    return {
        f.name: Mock(name=f.name)
        for f in fields(AgentRuntime)
        if f.default is MISSING and f.default_factory is MISSING
    }


def test_counters_start_at_zero_and_are_independent() -> None:
    counters = AgentCounters()
    assert asdict(counters) == {
        "turns": 0,
        "tool_runs": 0,
        "board_reads": 0,
        "learnings": 0,
        "refused_repeats": 0,
        "consecutive_failures": 0,
    }
    counters.turns += 1
    assert AgentCounters().turns == 0


def test_runtime_is_keyword_only_and_requires_holder_and_execution_ref() -> None:
    values = _values()
    base = Ref(path="swarm_arena:ScriptTool", config={"command": ["true"]})
    execution = base.model_copy(update={"config": {**base.config, "timeout_s": 60.0}})
    values.update(address="g/t/a", holder="g/:final", tool_ref=base, execution_ref=execution)
    runtime = AgentRuntime(**values)
    assert runtime.address == "g/t/a"
    assert runtime.holder == "g/:final"
    assert runtime.tool_ref is base
    assert runtime.execution_ref is execution
    assert runtime.private_paths == frozenset()
    assert all(f.kw_only for f in fields(AgentRuntime))
    for required in ("holder", "execution_ref"):
        incomplete = {k: v for k, v in values.items() if k != required}
        with pytest.raises(TypeError, match=required):
            AgentRuntime(**incomplete)
    with pytest.raises(TypeError):
        cast(Callable[..., AgentRuntime], AgentRuntime)(*values.values())


def test_runtime_accepts_private_paths() -> None:
    runtime = AgentRuntime(**_values(), private_paths=frozenset({"judge/secret.txt"}))
    assert runtime.private_paths == frozenset({"judge/secret.txt"})


def test_runtime_rankings_default_is_independent_and_accepts_own_group() -> None:
    first = AgentRuntime(**_values())
    second = AgentRuntime(**_values())
    assert first.rankings == {}
    assert first.rankings is not second.rankings
    part = AgentRuntime(**_values(), rankings={"part": "quality"})
    assert part.rankings == {"part": "quality"}


def test_runtime_main_goal_required_and_seed_and_start_defaults() -> None:
    values = _values()
    values.pop("goal_id", None)
    with pytest.raises(TypeError, match="goal_id"):
        AgentRuntime(**values)
    values["goal_id"] = "g"
    runtime = AgentRuntime(**values)
    assert runtime.goal_id == "g"
    assert runtime.seed_parents == ()
    assert runtime.started_runs == 0
    restored = AgentRuntime(**values, started_runs=4, seed_parents=("c2", "c1"))
    assert restored.started_runs == 4
    assert restored.seed_parents == ("c2", "c1")


def test_registry_documents_tranche_outcomes_and_overflow_admission() -> None:
    from pathlib import Path

    roadmap = Path("docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md").read_text()
    budget = roadmap.split("#### R4.13.6 ", 1)[1].split("#### R4.13.7 ", 1)[0]
    assert (
        "async def settle_call(self, reservation_id: str, actual: Amount) -> TrancheOutcome"
        in budget
    )
    assert "async def close(self) -> TrancheOutcome" in budget
    limits = roadmap.split("class LimitReached(RuntimeError):", 1)[1].split("class Manager", 1)[0]
    assert '"usage_overflow"' in limits
