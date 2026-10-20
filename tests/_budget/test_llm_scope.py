from dataclasses import FrozenInstanceError
from typing import cast

import anyio
import anyio.lowlevel
import pytest

from swarm_arena._budget.llm_scope import (
    LLMScope,
    current_llm_scope,
    current_requesting_call,
    llm_scope,
    requesting_call,
)
from swarm_arena._budget.tranche import Tranche
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.base import CallRole
from swarm_arena._model.catalog import ModelCatalog
from swarm_arena._model.settings import TracingConfig
from swarm_arena._store.artifacts import ArtifactStore


def _scope(role: CallRole) -> LLMScope:
    return LLMScope(
        role=role,
        run_id="r",
        goal_id="g",
        team="t",
        agent="a",
        round=1,
        candidate_id=None,
        tranche=cast(Tranche, object()),
        journal=cast(JournalWriter, object()),
        artifacts=cast(ArtifactStore, object()),
        models=ModelCatalog({}),
        tracing=TracingConfig(),
        cache=None,
    )


def test_scope_nests_and_restores() -> None:
    assert current_llm_scope() is None
    with llm_scope(_scope("manager")):
        with llm_scope(_scope("judge")):
            scope = current_llm_scope()
            assert scope is not None
            assert scope.role == "judge"
        scope = current_llm_scope()
        assert scope is not None
        assert scope.role == "manager"
    assert current_llm_scope() is None


async def test_requesting_call_is_per_task() -> None:
    seen: dict[str, str | None] = {}

    async def work(cid: str) -> None:
        with requesting_call(cid):
            await anyio.sleep(0.01)
            seen[cid] = current_requesting_call()

    async with anyio.create_task_group() as tg:
        tg.start_soon(work, "llm-1")
        tg.start_soon(work, "llm-2")
    assert seen == {"llm-1": "llm-1", "llm-2": "llm-2"}
    assert current_requesting_call() is None


def test_restore_after_exception_and_nested_none() -> None:
    def fail() -> None:
        with requesting_call(None), llm_scope(_scope("judge")):
            assert current_requesting_call() is None
            raise RuntimeError

    with requesting_call("outer"):
        with pytest.raises(RuntimeError):
            fail()
        assert current_requesting_call() == "outer"
        assert current_llm_scope() is None
    assert current_requesting_call() is None


def test_scope_frozen_and_call_ids_are_independent() -> None:
    a, b = _scope("manager"), _scope("judge")
    a.call_ids.append("call-1")
    assert b.call_ids == []
    assert not a.force_no_cache_read
    with pytest.raises(FrozenInstanceError):
        a.__setattr__("role", "judge")


async def test_scope_inheritance_and_task_isolation() -> None:
    outer = _scope("manager")
    seen: dict[str, LLMScope | None] = {}

    async def work(role: CallRole) -> None:
        assert current_llm_scope() is outer
        with llm_scope(_scope(role)):
            await anyio.lowlevel.checkpoint()
            seen[role] = current_llm_scope()
        assert current_llm_scope() is outer

    with llm_scope(outer):
        async with anyio.create_task_group() as tg:
            tg.start_soon(work, "judge")
            tg.start_soon(work, "planner")
    assert seen["judge"] is not None
    assert seen["judge"].role == "judge"
    assert seen["planner"] is not None
    assert seen["planner"].role == "planner"
    assert current_llm_scope() is None
