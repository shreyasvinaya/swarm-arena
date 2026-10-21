"""Round closure publishes its body once without revisiting previous pages."""

from typing import Any

import pytest
from _engine.test_goal import _plan
from _fakes.engine import EngineEnv

from swarm_arena._engine import goal as goal_module
from swarm_arena._engine.goal import GoalRunner
from swarm_arena._journal.views import read_source_head


async def test_engine_keeps_the_first_body_line(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"# Round heading\n\nRound body.\n"

    def render(*args: Any, **kwargs: Any) -> bytes:
        return body

    monkeypatch.setattr(goal_module, "round_page", render)
    await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1, rounds=1), replanner=None).run()
    page = env.store.read("goals/g/rounds/round-01.md").data
    assert read_source_head("md", page) is not None
    assert page.partition(b"\n")[2] == body
    assert page.count(b"<!-- swarm-arena source_head:") == 1


async def test_live_closure_probes_only_its_own_page(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = env.store.exists
    probes: list[str] = []

    def exists(key: str) -> bool:
        if "/rounds/round-" in key:
            probes.append(key)
        return original(key)

    monkeypatch.setattr(env.store, "exists", exists)
    await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1, rounds=3), replanner=None).run()
    assert probes == [f"goals/g/rounds/round-{round:02d}.md" for round in (1, 2, 3)]
