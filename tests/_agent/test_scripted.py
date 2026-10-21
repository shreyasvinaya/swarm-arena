from typing import Any
from unittest.mock import AsyncMock, Mock, call

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import GOAL
from pydantic import ValidationError

from swarm_arena import FunctionTool, Limits, ProgramJudge, Ref, RunConfig, Score, ScriptedManager
from swarm_arena._errors import ConfigError
from swarm_arena.manager import LimitReached, Manager, RoundReport
from swarm_arena.testing import check_manager
from swarm_arena.testing._harness import OfflineHarness

SQUARE = FunctionTool("_fakes.tools:square")


async def _run(manager: ScriptedManager, rounds: int) -> OfflineHarness:
    harness = OfflineHarness()
    await harness.run_rounds(
        manager,
        goal=GOAL,
        tool=SQUARE,
        rounds=rounds,
        limits=Limits(),
        config=RunConfig(),
        judges=[Ref.of(ProgramJudge("_fakes.tools:score_min_y"))],
    )
    return harness


async def test_runs_requests_in_order_and_reports() -> None:
    reports = await check_manager(
        ScriptedManager(requests=[{"inputs": {"x": 1.0}}, {"inputs": {"x": 2.0}}]),
        goal=GOAL,
        tool=FunctionTool("_fakes.tools:square"),
    )
    assert reports == [RoundReport(summary="ran 2 of 2 requests")]


async def test_round_filter_and_best_parent() -> None:
    manager = ScriptedManager(
        [
            {"inputs": {"x": 3.0}, "round": 1},
            {"inputs": {"x": 1.0}, "parents": "best", "round": 2},
        ]
    )
    harness = await _run(manager, rounds=2)
    assert [c.round for c in harness.candidates] == [1, 2]
    assert harness.candidates[1].parent_ids == [harness.candidates[0].id]
    assert [c.output.data["y"] for c in harness.candidates] == [9.0, 1.0]
    assert harness.candidates[1].read_ids == [harness.candidates[0].id]


async def test_best_parent_with_empty_board_runs_without_parents() -> None:
    harness = await _run(ScriptedManager([{"inputs": {"x": 1.0}, "parents": "best"}]), rounds=1)
    assert harness.candidates[0].parent_ids == []


async def test_soft_error_skips_request() -> None:
    reports = await check_manager(
        ScriptedManager([{"inputs": {"x": "bad"}}, {"inputs": {"x": 1.0}}]),
        goal=GOAL,
        tool=SQUARE,
    )
    assert reports[0].summary == "ran 1 of 2 requests (1 soft error)"


async def test_multiple_soft_errors_and_round_filter_summary() -> None:
    reports = await check_manager(
        ScriptedManager(
            [
                {"inputs": {"x": "bad"}},
                {"inputs": {"x": "worse"}},
                {"inputs": {"x": 1.0}, "round": 2},
            ]
        ),
        goal=GOAL,
        tool=SQUARE,
        rounds=2,
    )
    assert [r.summary for r in reports] == [
        "ran 0 of 2 requests (2 soft errors)",
        "ran 1 of 3 requests (2 soft errors)",
    ]


async def test_post_learnings_cites_best_own_candidate() -> None:
    manager = ScriptedManager(
        ScriptedManager.Config(
            requests=[
                ScriptedManager.Request(inputs={"x": 3.0}, round=1),
                ScriptedManager.Request(inputs={"x": 1.0}, round=2),
            ],
            post_learnings=True,
        )
    )
    harness = await _run(manager, rounds=2)
    first = harness.candidates[0]
    assert [(item.round, item.title, item.text, item.evidence) for item in harness.learnings] == [
        (2, f"Best so far: {first.id}", "score 9", [first.id]),
    ]


@pytest.mark.parametrize(
    "data",
    [
        {"inputs": {}, "parents": 12},
        {"inputs": {}, "edits": {"dynamic-secret-key": object()}},
        {"inputs": {}, "round": "config-secret-sentinel"},
        {"unknown": "config-secret-sentinel"},
    ],
)
def test_request_config_errors_hide_values(data: dict[str, Any]) -> None:
    with pytest.raises(ConfigError) as caught:
        ScriptedManager([data])
    assert_secret_absent(caught.value, "config-secret-sentinel", sanitized=True)
    assert_secret_absent(caught.value, "dynamic-secret-key", sanitized=True)


def test_request_model_validation_hides_values() -> None:
    secret = "config-secret-sentinel"
    with pytest.raises(ValidationError) as caught:
        ScriptedManager.Request.model_validate({"round": secret})
    assert_secret_absent(caught.value, secret, sanitized=True)


def test_config_roundtrips_through_ref() -> None:
    manager = ScriptedManager(
        ScriptedManager.Config(
            requests=[ScriptedManager.Request(inputs={"x": 1.0}, parents="best")],
            post_learnings=True,
        )
    )
    rebuilt = Ref.of(manager).build(Manager)
    assert isinstance(rebuilt, ScriptedManager)
    assert rebuilt.config == manager.config
    assert "round" not in Ref.of(manager).config["requests"][0]


async def test_limit_reached_propagates_from_scripted_manager() -> None:
    limit = LimitReached("budget", "budget used up")
    ctx = Mock(round=1, run_tool=AsyncMock(side_effect=limit))
    with pytest.raises(LimitReached) as caught:
        await ScriptedManager([{"inputs": {}}, {"inputs": {}}]).run_round(ctx)
    assert caught.value is limit
    assert ctx.run_tool.await_count == 1


async def test_learning_selects_greatest_own_key_and_falls_back_to_key_value() -> None:
    ctx = Mock(round=2, post_learning=AsyncMock())
    ctx.my_candidates.return_value = [Mock(id="first"), Mock(id="second"), Mock(id="unscored")]
    ctx.scores_of.side_effect = [
        [Score(candidate_id="first", key=(-9.0,), value=9.0)],
        [Score(candidate_id="second", key=(-5.0,)), Score(candidate_id="second", key=(-1.0,))],
        [],
    ]
    manager = ScriptedManager(ScriptedManager.Config(requests=[], post_learnings=True))
    await manager.run_round(ctx)
    ctx.post_learning.assert_awaited_once_with(
        title="Best so far: second",
        text="score -1",
        evidence=["second"],
    )


async def test_explicit_parents_and_edits_are_forwarded_in_order() -> None:
    ctx = Mock(round=3, run_tool=AsyncMock())
    manager = ScriptedManager(
        [
            {"inputs": {"x": 1.0}, "parents": ["parent"], "edits": {"model.txt": "new"}},
            {"inputs": {"x": 2.0}},
            {"inputs": {"x": 3.0}, "round": 1},
        ]
    )
    report = await manager.run_round(ctx)
    assert report.summary == "ran 2 of 2 requests"
    assert ctx.run_tool.await_args_list == [
        call({"x": 1.0}, parents=["parent"], edits={"model.txt": "new"}),
        call({"x": 2.0}, parents=[], edits={}),
    ]
