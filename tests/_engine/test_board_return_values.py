"""Connector values are normalized before core work resumes on the loop."""

import threading
from typing import Any

import pytest
from _engine.test_goal import _curation_checkpoint, _plan
from _engine.test_round import _round, _scripted
from _fakes.engine import EngineEnv
from _fakes.secrecy import assert_secret_absent

from swarm_arena._board.arena import ArenaView
from swarm_arena._engine.goal import GoalRunner, _page_namespace_count
from swarm_arena._model.board_types import Frontier


class Count(int):
    def __add__(self, other: object) -> int:
        raise OSError("private-count-marker")


@pytest.mark.parametrize("value", [Count(3), -1, True, "3"])
async def test_page_count_is_validated_in_worker(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    threads: list[int] = []

    class View:
        @property
        def count_learnings(self) -> Any:
            threads.append(threading.get_ident())
            return lambda namespace: value

    monkeypatch.setattr(env.deps.board, "snapshot", lambda *args: View())
    operation = _page_namespace_count(env.deps, _plan(teams=1, rounds=1), Frontier())
    if isinstance(value, Count):
        count = await operation
        assert type(count) is int
        assert count + 1 == 4
    else:
        with pytest.raises(RuntimeError, match="round page board read failed"):
            await operation
    assert threads
    assert threading.get_ident() not in threads


@pytest.mark.parametrize("method", ["learnings", "_visible_candidate_ids"])
async def test_runtime_preparation_reads_connector_values_off_loop(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    threads: list[int] = []
    original = getattr(ArenaView, method)

    def read(view: ArenaView, *args: Any, **kwargs: Any) -> Any:
        threads.append(threading.get_ident())
        return original(view, *args, **kwargs)

    monkeypatch.setattr(ArenaView, method, read)
    outcome = await _round(env, [_scripted("a", [])]).run(1)
    assert outcome.reasons == {"t/a": ""}
    assert threads
    assert threading.get_ident() not in threads


async def test_curator_frontier_is_detached_before_snapshot(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = await _curation_checkpoint(env)
    original_frontier = env.deps.board.frontier
    original_snapshot = env.deps.board.snapshot
    types: list[type[Frontier]] = []

    class ReturnedFrontier(Frontier):
        pass

    def frontier() -> Frontier:
        return ReturnedFrontier.model_validate(original_frontier().model_dump())

    def snapshot(goal: Any, value: Frontier) -> Any:
        types.append(type(value))
        return original_snapshot(goal, value)

    monkeypatch.setattr(env.deps.board, "frontier", frontier)
    monkeypatch.setattr(env.deps.board, "snapshot", snapshot)
    await runner._curate_round(1)
    assert types == [Frontier]


async def test_page_existence_result_cannot_execute_on_loop(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "private-existence-marker"
    original = env.store.exists
    threads: list[int] = []

    class Result:
        def __bool__(self) -> bool:
            threads.append(threading.get_ident())
            raise OSError(secret)

    def exists(key: str) -> Any:
        return Result() if key.endswith("round-01.md") else original(key)

    monkeypatch.setattr(env.store, "exists", exists)
    runner = GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1, rounds=1), replanner=None)
    with pytest.raises(Exception, match="round page storage failed: TypeError") as caught:
        await runner.run()
    assert_secret_absent(caught.value, secret)
    assert threading.get_ident() not in threads
