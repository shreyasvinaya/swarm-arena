"""Compare in-memory fixture read results with the real context contract."""

from typing import Any, cast
from unittest.mock import Mock

import pytest
from _fakes.agent_ctx import FakeAgentContext
from _fakes.board import learning

from swarm_arena import Goal, Limits, ScriptTool, TeamSpec
from swarm_arena._errors import SoftError
from swarm_arena.board import BoardView, Row
from swarm_arena.manager import AgentContext
from swarm_arena.scoring import CheckResult


def _row(id: str, producer_part: str | None = None) -> Row:
    return Row(
        rank=1,
        candidate_id=id,
        run_id="peer",
        goal_id="g",
        team="team",
        agent="agent",
        part=producer_part,
        fidelity="full",
        key=(1.0,),
        value=1,
        judge="j",
        rubric_hash="r",
        panel_hash="p",
        built_on=[],
        partial=False,
    )


def _real(fake: FakeAgentContext, *, part: str | None = None) -> tuple[AgentContext, Mock]:
    view = Mock(spec=BoardView)
    rt = Mock(
        goal=fake.goal,
        goal_id=fake.goal.id,
        goal_key=fake.goal.key(),
        address="g/team/agent",
        team=TeamSpec(name="team", rationale="Explore", agents=[fake.spec]),
        round=fake.round,
        round_spec=fake.round_spec,
        digest="",
        notes="",
        resumed=(),
        checks=[],
        tool=ScriptTool(command=["echo"]),
        own=[],
        limits=Limits(),
        watchdog=Mock(limit=None),
        counters=Mock(turns=0, board_reads=0, learnings=0),
        part=part,
        rankings={part or "": "j"},
        sibling_views={},
        view=view,
        repeats=Mock(run_id="run-1"),
    )
    rt.spec = fake.spec
    rt.repeats.candidates_of.return_value = fake.candidates
    rt.view.get_candidate.side_effect = lambda id: next(c for c in fake.candidates if c.id == id)
    return AgentContext(rt), rt


def test_main_group_selector_parity() -> None:
    fake = FakeAgentContext(rows=[_row("main")])
    real, rt = _real(fake)
    rt.view.leaderboard.return_value = fake.rows
    assert fake.best(1) == real.best(1) == fake.rows
    assert fake.best(1, part="") == real.best(1, part="") == fake.rows
    assert fake.best(0, part="") == real.best(0, part="") == []


@pytest.mark.parametrize("own_part", [None, "current"])
def test_fixed_group_membership_ignores_producer_part(own_part: str | None) -> None:
    # The foreign row belongs to current's capability despite its recorded label.
    main_rows, part_rows = [_row("main")], [_row("foreign", "producer-name")]
    fake = FakeAgentContext(
        part=own_part,
        rows=part_rows if own_part else main_rows,
        sibling_rows={"": main_rows, "current": part_rows, "empty": []},
    )
    real, rt = _real(fake, part=own_part)
    rt.view.leaderboard.return_value = fake.rows
    for name, rows in fake.sibling_rows.items():
        view = Mock(spec=BoardView)
        view.goal = Goal(id=name or "main", namespace="n").key()
        view.leaderboard.return_value = rows
        rt.sibling_views[name] = view
        rt.rankings[name] = "j"
    expected = part_rows if own_part else main_rows
    assert fake.best(1) == real.best(1) == expected
    for name, rows in fake.sibling_rows.items():
        assert fake.best(1, part=name) == real.best(1, part=name) == rows
    assert fake.best(1, part="current")[0].part == "producer-name"
    assert fake.best(1, part="current")[0] is not part_rows[0]


@pytest.mark.parametrize("selector", ["missing", "producer-name"])
def test_undeclared_group_rejection_parity(selector: str) -> None:
    fake = FakeAgentContext(rows=[_row("foreign", "producer-name")])
    real, _ = _real(fake)
    for ctx in (fake, real):
        with pytest.raises(SoftError, match="part outside board snapshot"):
            ctx.best(1, part=selector)


@pytest.mark.parametrize("policy", [None, "pass_fail", "unknown", "reasons"])
def test_candidate_feedback_projection_parity(policy: str | None) -> None:
    fake = FakeAgentContext()
    original = fake.add_candidate("c-1").model_copy(
        update={
            "checks": [
                CheckResult(
                    check="private",
                    passed=False,
                    reasons=["PRIVATE_REASON"],
                    details={"secret": ["PRIVATE_DETAIL"]},
                )
            ]
        }
    )
    fake.candidates[:] = [original]
    real, _ = _real(fake)
    if policy is not None:
        feedback = {"private": policy}
        fake.feedback_by_candidate[original.id] = feedback
        real._feedback_for = lambda candidate: feedback
    for actual, expected in [
        (fake.my_candidates()[0], real.my_candidates()[0]),
        (fake.get_candidate(original.id), real.get_candidate(original.id)),
    ]:
        assert actual == expected
        assert actual is not original
        assert actual.checks[0].reasons == (["PRIVATE_REASON"] if policy == "reasons" else [])
        assert actual.checks[0].details == (
            {"secret": ["PRIVATE_DETAIL"]} if policy == "reasons" else {}
        )
        actual.request.inputs["changed"] = True
        actual.checks[0].reasons.append("mutated")
        if policy == "reasons":
            cast(list[str], actual.checks[0].details["secret"]).append("mutated")
        assert original.request.inputs == {}
        assert original.checks[0].reasons == ["PRIVATE_REASON"]
        assert original.checks[0].details == {"secret": ["PRIVATE_DETAIL"]}


def test_candidate_policy_belongs_to_producer_parity() -> None:
    fake = FakeAgentContext()
    local = fake.add_candidate("local").model_copy(
        update={
            "checks": [
                CheckResult(check="check", passed=False, reasons=["allowed"], details={"n": 1})
            ]
        }
    )
    foreign = local.model_copy(update={"id": "foreign", "run_id": "peer"})
    fake.candidates[:] = [local, foreign]
    fake.feedback_by_candidate[local.id] = {"check": "reasons"}
    real, _ = _real(fake)
    real._feedback_for = lambda candidate: fake.feedback_by_candidate.get(candidate.id, {})
    assert fake.my_candidates() == real.my_candidates()
    assert fake.get_candidate("local").checks[0].reasons == ["allowed"]
    assert fake.get_candidate("foreign").checks[0].reasons == []
    assert fake.get_candidate("foreign").checks[0].details == {}


@pytest.mark.anyio
@pytest.mark.parametrize("query", ["", "hint", "SECOND"])
@pytest.mark.parametrize("scope", ["goal", "namespace"])
@pytest.mark.parametrize("limit", [0, 1, 10])
async def test_newest_own_search_and_snapshot_order_parity(
    query: str, scope: str, limit: int
) -> None:
    fake = FakeAgentContext()
    first = await fake.post_learning("First", "hint", ["c-1"], tags=["step"])
    second = await fake.post_learning("Second", "hint", ["c-1"], tags=["step"])
    snapshot = [
        learning("snapshot", title="Older hint", tags=["step"]),
        learning("other-goal", goal_id="other", title="Second hint", tags=["step"]),
        learning("hidden", title="Second hint", hidden=True, tags=["step"]),
        learning("untagged", title="Second hint"),
        learning("other-namespace", namespace="other", title="Second hint", tags=["step"]),
    ]
    fake.learnings[:0] = snapshot
    real, rt = _real(fake)
    rt.own = [first, second]
    visible = [
        item
        for item in snapshot
        if not item.hidden
        and item.namespace == fake.goal.namespace
        and (scope == "namespace" or item.goal_id == fake.goal.id)
        and "step" in item.tags
        and (query.casefold() in item.title.casefold() or query.casefold() in item.text.casefold())
    ][:limit]
    rt.view.learnings.return_value = visible
    rt.view.search_learnings.return_value = visible
    args: dict[str, Any] = {"tags": ["step"], "scope": scope, "limit": limit}
    actual, expected = fake.search_learnings(query, **args), real.search_learnings(query, **args)
    assert actual == expected
    if limit:
        assert actual[0].id == second.id
        assert actual[0] is not second
    # Only the real context is an accounting oracle.
    assert rt.counters.board_reads == rt.counters.turns == 1
    reader = rt.view.learnings if not query else rt.view.search_learnings
    assert reader.call_args.kwargs["goal"] == (fake.goal.key() if scope == "goal" else None)


@pytest.mark.anyio
async def test_duplicate_own_snapshot_search_parity() -> None:
    fake = FakeAgentContext()
    first = await fake.post_learning("First", "hint", ["c-1"])
    second = await fake.post_learning("Second", "hint", ["c-1"])
    real, rt = _real(fake)
    rt.own = [first, second]
    rt.view.learnings.return_value = [first, second]
    assert fake.search_learnings("") == real.search_learnings("") == [second, first]
    assert fake.learnings == [first, second]


def test_invalid_search_scope_rejection_parity() -> None:
    fake = FakeAgentContext()
    real, rt = _real(fake)
    for ctx in (fake, real):
        with pytest.raises(SoftError, match=r"^invalid learning scope$"):
            ctx.search_learnings("", scope=cast(Any, "invalid"))
    assert rt.counters.board_reads == 1
