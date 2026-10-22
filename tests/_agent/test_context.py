from collections.abc import Awaitable
from dataclasses import replace
from doctest import SKIP, DocTestParser, DocTestRunner
from threading import get_ident
from typing import Any, cast

import anyio
import pytest
from _fakes.engine import CtxEnv
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import AlwaysFailCheck

from swarm_arena import FunctionTool, Limits, Score
from swarm_arena._agent.context import AgentContext
from swarm_arena._agent.repeats import RepeatIndex
from swarm_arena._board.base import BoardView
from swarm_arena._board.learnings import LEARNING_GUIDE
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import requesting_call
from swarm_arena._errors import (
    SoftError,
    _core_category,
    _core_error,
    _is_core_control,
    _is_core_signal,
    _SignalCategory,
)
from swarm_arena._journal.a2a import agent_task_id
from swarm_arena._journal.event import EventDraft
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._model.base import FrozenModel, JsonValue
from swarm_arena.board import Candidate, Learning, Row
from swarm_arena.executor import Job
from swarm_arena.manager import LimitReached
from swarm_arena.tool import Action, Estimate, ToolOutput, ToolRequest

SECRET = "secret-context-value-9x"


async def _gather(*calls: Awaitable[Any]) -> list[Any]:
    results: list[Any] = [None] * len(calls)

    async def run(index: int, call: Awaitable[Any]) -> None:
        try:
            results[index] = await call
        except Exception as error:
            results[index] = error

    async with anyio.create_task_group() as group:
        for index, call in enumerate(calls):
            group.start_soon(run, index, call)
    return results


async def test_parallel_run_tool_respects_last_slot(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(tool_runs_used=2)
    results = await _gather(ctx.run_tool({"x": 1.0}), ctx.run_tool({"x": 2.0}))
    assert sum(isinstance(r, Candidate) for r in results) == 1
    assert [r.limit for r in results if isinstance(r, LimitReached)] == ["max_tool_runs_per_round"]


async def test_parallel_post_learning_respects_rate_limit(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(learnings_used=2)
    c = await ctx.run_tool({"x": 1.0})
    results = await _gather(*(ctx.post_learning(f"t{i}", "x", [c.id]) for i in range(2)))
    assert sum(isinstance(r, Learning) for r in results) == 1
    assert [str(r) for r in results if isinstance(r, SoftError)] == [
        "max_learnings_per_agent_round reached"
    ]
    assert ctx_env.counters.learnings == 3
    assert len(ctx_env.events("learning")) == 1
    assert sum(isinstance(item, Learning) for item in ctx_env.rt.own) == 1
    assert isinstance(await ctx.run_tool({"x": 2.0}), Candidate)


async def test_soft_errors_do_not_use_a_run(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    for bad in [
        ctx.run_tool({"x": "nope"}),
        ctx.run_tool({"x": 1.0}, parents=["c-unknown"]),
        ctx.run_tool({"x": 1.0}, edits={"secret.py": "x"}),
        ctx.run_tool({"x": 1.0}, edits={"problem/a.phyk": "x" * 262_145}),
    ]:
        with pytest.raises(SoftError):
            await bad
    assert ctx_env.counters.tool_runs == 0


async def test_refused_repeats_trip_watchdog(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    await ctx.run_tool({"x": 1.0})
    for _ in range(3):
        with pytest.raises(SoftError, match="repeat refused"):
            await ctx.run_tool({"x": 1.0})
    with pytest.raises(LimitReached) as err:
        await ctx.run_tool({"x": 1.0})
    assert err.value.limit == "max_refused_repeats"
    assert ctx_env.counters.tool_runs == 1
    assert len([e for e in ctx_env.events("watchdog") if "refused" in e.payload]) == 4


async def test_turn_limit(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(limits=Limits(max_turns_per_round=2, max_board_reads_per_round=10))
    ctx.search_learnings("a")
    ctx.search_learnings("b")
    with pytest.raises(LimitReached) as err:
        ctx.search_learnings("c")
    assert err.value.limit == "max_turns_per_round"


async def test_board_reads_capped(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    for i in range(10):
        ctx.search_learnings(str(i))
    with pytest.raises(SoftError, match="max_board_reads_per_round"):
        ctx.read_learning("L-x")


async def test_learning_rules(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    c = await ctx.run_tool({"x": 1.0})
    for title, text, evidence in [
        ("", "x", [c.id]),
        ("x" * 121, "x", [c.id]),
        ("x", "x" * 1501, [c.id]),
        ("x", "x", []),
        ("x", "x", ["c-unknown"]),
    ]:
        with pytest.raises(SoftError):
            await ctx.post_learning(title, text, evidence)
    assert ctx_env.counters.learnings == 0


@pytest.mark.parametrize("verified", [False, True])
async def test_post_learning_sets_verified_from_evidence(ctx_env: CtxEnv, verified: bool) -> None:
    c = _candidate().model_copy(update={"output": ToolOutput(status="ok", partial=not verified)})
    ctx = ctx_env.ctx(own=[c])
    posted = await ctx.post_learning("Measured result", "Converged in 8 iterations.", [c.id])
    assert posted.verified is verified
    assert ctx_env.events("learning")[-1].payload["learning"]["verified"] is verified
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (1, 0)


async def test_post_learning_duplicate_is_soft_error(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    view = await _snapshot(ctx_env, candidates=(_candidate(),), learnings=(_learning(),))
    ctx = ctx_env.ctx(view=view)
    allocate = Mock(wraps=ctx._rt.ids.next)
    monkeypatch.setattr(ctx._rt.ids, "next", allocate)
    before = len(ctx_env.events("learning"))
    with pytest.raises(SoftError, match="near-duplicate of L-1"):
        await ctx.post_learning("baseline", "Measured again.", ["c-old"])
    assert ctx_env.counters.learnings == 0
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (1, 0)
    assert len(ctx_env.events("learning")) == before
    allocate.assert_not_called()
    assert ctx._rt.own == []
    assert (
        await ctx.post_learning("baseline", "8 iterations.", ["c-old"], refines=["L-1"])
    ).verified


@pytest.mark.parametrize("attack", ["none", "mutate", "impostor"])
@pytest.mark.parametrize("failure", ["title", "text", "duplicate"])
async def test_post_learning_guide_survives_action_boundary_safely(
    ctx_env: CtxEnv, attack: str, failure: str
) -> None:
    import traceback

    ctx = ctx_env.ctx(own=[_candidate(), _learning(title=SECRET)])
    originals: list[SoftError] = []

    async def handler(context: AgentContext, args: EmptyArgs) -> JsonValue:
        try:
            await context.post_learning(
                SECRET + ("\n" if failure == "title" else ""),
                SECRET * 100 if failure == "text" else "body",
                ["c-old"],
            )
        except SoftError as error:
            originals.append(error)
            if attack == "none":
                raise
            unsafe = SoftError(SECRET) if attack == "impostor" else error
            unsafe.args = (SECRET,)
            unsafe.add_note(SECRET)
            raise unsafe from RuntimeError(SECRET)
        return None

    ctx.actions = (Action(name="post", description="Post", input_model=EmptyArgs, handler=handler),)
    with pytest.raises(SoftError) as caught:
        await ctx.call_action("post", {})
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value is not originals[0]
    if attack == "impostor":
        assert str(caught.value) == "action failed: SoftError"
    elif failure == "duplicate":
        assert str(caught.value).startswith("near-duplicate of L-1")
    else:
        assert str(caught.value).endswith(LEARNING_GUIDE)
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads, ctx_env.counters.learnings) == (
        1,
        0,
        0,
    )
    assert ctx_env.events("learning") == []


async def test_notes_cap_and_carry(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    with pytest.raises(SoftError):
        await ctx.save_notes("x" * 16_385)
    await ctx.save_notes("lr 0.01 diverged")
    assert ctx_env.events("notes_saved")[-1].payload == {"text": "lr 0.01 diverged"}


async def test_other_agents_round_items_hidden(ctx_env: CtxEnv) -> None:
    a, b = ctx_env.ctx(agent="a"), ctx_env.ctx(agent="b")
    c = await a.run_tool({"x": 1.0})
    with pytest.raises(SoftError, match="unknown candidate"):
        b.get_candidate(c.id)
    assert a.get_candidate(c.id).id == c.id


async def test_read_ids_recorded(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(
        own=[
            Learning(
                id="L-1",
                namespace="n",
                run_id="engine-run",
                goal_id="g",
                round=1,
                author="human:test",
                title="hint",
                text="body",
            )
        ]
    )
    ctx.read_learning("L-1")
    c = await ctx.run_tool({"x": 1.0})
    assert c.read_ids == ["L-1"]
    assert ctx_env.events("tool_end")[-1].payload["candidate"]["read_ids"] == ["L-1"]
    assert ctx_env.rt.board.get_candidate(c.id).read_ids == ["L-1"]


async def test_extra_action_counts_as_board_read(ctx_env: CtxEnv) -> None:
    ctx = await ctx_env.script_ctx()
    assert await ctx.call_action("list_files", {}) == {
        "files": ["problem/a.phyk", "problem/run.py"]
    }  # M1a Task 19 shape
    assert ctx_env.counters.board_reads == 1
    with pytest.raises(SoftError):
        await ctx.call_action("nope", {})


async def test_read_file_action_reads_only_the_agent_view(ctx_env: CtxEnv) -> None:
    ctx = await ctx_env.script_ctx()
    assert await ctx.call_action("read_file", {"path": "problem/a.phyk"}) == {
        "path": "problem/a.phyk",
        "content": "a = 1\n",
    }
    assert ctx_env.counters.board_reads == 1
    with pytest.raises(SoftError, match=r"no such file: secret\.py"):
        await ctx.call_action("read_file", {"path": "secret.py"})
    outside = ctx_env.artifacts.put_bytes(b"not a file of this agent's view")
    with pytest.raises(SoftError, match="file not in your view"):
        ctx._read_artifact(outside)


async def test_pass_fail_feedback_hides_reasons(ctx_env: CtxEnv) -> None:
    hidden = AlwaysFailCheck({"reason": "secret threshold 0.3", "feedback": "pass_fail"})
    ctx = ctx_env.ctx(checks=[("hidden", hidden)])
    c = await ctx.run_tool({"x": 1.0})
    assert (c.checks[0].passed, c.checks[0].reasons) == (False, [])
    assert ctx.get_candidate(c.id).checks[0].reasons == []
    assert ctx.my_candidates()[0].checks[0].reasons == []
    assert ctx_env.events("tool_end")[-1].payload["candidate"]["checks"][0]["reasons"] == [
        "secret threshold 0.3"
    ]


async def test_gate_blocks_new_actions(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(gate=lambda: LimitReached("stop", "the run is stopping"))
    with pytest.raises(LimitReached) as err:
        await ctx.run_tool({"x": 1.0})
    assert err.value.limit == "stop"


@pytest.mark.parametrize("source", ["gate", "watchdog"])
@pytest.mark.parametrize("limit", [*Limits.model_fields, "budget", "stop"])
async def test_context_accepts_only_ordinary_limit_authority(
    ctx_env: CtxEnv, source: str, limit: str
) -> None:
    ctx_env.ctx()
    blocked = LimitReached(limit, "stopping")
    if source == "watchdog":
        ctx_env.rt.watchdog.limit = limit
        ctx_env.rt.watchdog.reason = "stopping"
    ctx = AgentContext(ctx_env.rt, gate=lambda: blocked if source == "gate" else None)
    with pytest.raises(LimitReached, match=r"^stopping$") as error:
        ctx.search_learnings("x")
    assert error.value is not blocked
    assert error.value.limit == limit
    assert _core_category(error.value) is _SignalCategory.LIMIT
    assert _is_core_control(error.value)
    assert ctx_env.counters.turns == 0
    assert ctx_env.events("agent_counters") == []


@pytest.mark.parametrize("source", ["gate", "watchdog"])
@pytest.mark.parametrize("limit", ["usage_overflow", SECRET, ""])
async def test_context_rejects_unknown_limits_without_terminal_authority(
    ctx_env: CtxEnv, source: str, limit: str
) -> None:
    ctx_env.ctx()
    if source == "watchdog":
        ctx_env.rt.watchdog.limit = limit
        ctx_env.rt.watchdog.reason = SECRET
    ctx = AgentContext(
        ctx_env.rt, gate=lambda: LimitReached(limit, SECRET) if source == "gate" else None
    )
    with pytest.raises(ValueError, match=r"^invalid agent limit$") as error:
        await ctx.save_notes("x")
    assert _is_core_signal(error.value)
    assert _core_category(error.value) is _SignalCategory.DIAGNOSTIC
    assert not _is_core_control(error.value)
    assert_secret_absent(error.value, SECRET, sanitized=True)
    assert ctx_env.counters.turns == 0
    assert ctx_env.events("agent_counters") == []


async def test_actions_journal_counters(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    ctx.search_learnings("a")
    ctx.search_learnings("b")
    last = ctx_env.events("agent_counters")[-1]
    assert last.payload == {"turns": 2, "board_reads": 2}
    assert last.task_id == agent_task_id(ctx_env.address, 1)


async def test_parallel_identical_nondeterministic_requests_get_distinct_replicates(
    ctx_env: CtxEnv,
) -> None:
    ctx = ctx_env.ctx(tool=FunctionTool("_fakes.tools:square", deterministic=False), replicates=2)
    results = await _gather(ctx.run_tool({"x": 1.0}), ctx.run_tool({"x": 1.0}))
    assert {c.request.replicate for c in results} == {0, 1}


async def test_budget_is_an_estimate(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    assert isinstance(ctx.budget, Estimate)
    assert ctx.budget.usd == ctx_env.rt.ledger.remaining(ctx_env.rt.holder).usd


def _candidate(id: str = "c-old", *, agent: str = "a", round: int = 0) -> Candidate:
    return Candidate(
        id=id,
        run_id="engine-run",
        goal_id="g",
        team="t",
        agent=agent,
        round=round,
        request=ToolRequest(inputs={"x": 1.0}),
        output=ToolOutput(status="ok"),
    )


def _learning(id: str = "L-1", *, title: str = "baseline", **fields: Any) -> Learning:
    return Learning(
        id=id,
        namespace="n",
        run_id="engine-run",
        goal_id="g",
        round=0,
        author="human:test",
        title=title,
        text="hint",
        **fields,
    )


async def _snapshot(
    ctx_env: CtxEnv,
    *,
    candidates: tuple[Candidate, ...] = (),
    learnings: tuple[Learning, ...] = (),
    scores: tuple[Score, ...] = (),
) -> BoardView:
    rt = ctx_env.env.rt
    for candidate in candidates:
        event = rt.journal.append(
            EventDraft(
                type="tool_end",
                goal_id="g",
                payload={"candidate": candidate.model_dump(mode="json")},
            )
        )
        rt.board.put_candidate(candidate, epoch=event.epoch, seq=event.seq)
    for learning in learnings:
        event = rt.journal.append(
            EventDraft(
                type="learning", goal_id="g", payload={"learning": learning.model_dump(mode="json")}
            )
        )
        rt.board.put_learning(learning, epoch=event.epoch, seq=event.seq)
    for score in scores:
        event = rt.journal.append(
            EventDraft(type="score", goal_id="g", payload={"score": score.model_dump(mode="json")})
        )
        rt.board.put_score(score, epoch=event.epoch, seq=event.seq)
    await rt.journal.commit()
    frontier = await anyio.to_thread.run_sync(rt.board.frontier)
    return await anyio.to_thread.run_sync(rt.board.snapshot, rt.goal_key, frontier)


async def test_parallel_run_tool_respects_last_budget_share(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = ctx_env.ctx()
    rt = ctx_env.rt
    remainder = rt.ledger.remaining(rt.holder)
    await rt.ledger.reserve(rt.holder, Amount(usd=remainder.usd - 1), kind="tool", goal_id="g")
    release = anyio.Event()
    submit = rt.executor.submit

    async def held_submit(job: Job) -> ToolOutput:
        await release.wait()
        return await submit(job)

    monkeypatch.setattr(rt.executor, "submit", held_submit)

    async def run(x: float) -> Candidate | LimitReached:
        try:
            return await ctx.run_tool({"x": x})
        except LimitReached as error:
            release.set()
            return error

    with anyio.fail_after(10):
        results = await _gather(run(1.0), run(2.0))
    assert sum(isinstance(result, Candidate) for result in results) == 1
    assert [result.limit for result in results if isinstance(result, LimitReached)] == ["budget"]
    assert rt.ledger.remaining(rt.holder).usd >= 0


@pytest.mark.parametrize("part", [None, "p"])
async def test_best_uses_own_group_and_records_rows(ctx_env: CtxEnv, part: str | None) -> None:
    rt = ctx_env.env.rt
    if part is not None:
        from swarm_arena import Goal, PartSpec, Plan
        from swarm_arena._engine.goal import r0_revision_draft

        subgoal = Goal(id="sub", namespace="n")
        plan = Plan(
            goal=rt.goal, teams=[], judges=[], revision=1, parts=[PartSpec(name=part, goal=subgoal)]
        )
        await rt.journal.append_durable(r0_revision_draft(plan, plan.base_dir))
        ctx_env.env.rt = replace(rt, goal=subgoal, goal_key=subgoal.key(), part=part)
        rt = ctx_env.env.rt
    candidate = _candidate(round=1).model_copy(update={"part": part})
    score = Score(id="s-1", candidate_id=candidate.id, judge="rank", key=(2.0,))
    view = await _snapshot(ctx_env, candidates=(candidate,), scores=(score,))
    ctx = ctx_env.ctx(view=view, part=part, rankings={part or "": "rank"})
    rows = ctx.best(1, part=part)
    assert len(rows) == 1
    assert isinstance(rows[0], Row)
    assert rows[0].candidate_id == candidate.id
    assert ctx.best(0) == []
    assert ctx.best(-1) == []
    with pytest.raises(SoftError, match="part outside board snapshot"):
        ctx.best(1, part="other")
    c = await ctx.run_tool({"x": 2.0})
    assert c.read_ids == [candidate.id]
    assert ctx_env.counters.board_reads == 4


async def test_best_without_ranking_is_soft_error(ctx_env: CtxEnv) -> None:
    with pytest.raises(SoftError, match="missing ranking"):
        ctx_env.ctx().best(1)


async def test_scores_and_read_provenance_are_snapshot_scoped(ctx_env: CtxEnv) -> None:
    c = _candidate()
    score = Score(id="s-1", candidate_id=c.id, judge="rank", key=(2.0,))
    view = await _snapshot(ctx_env, candidates=(c,), learnings=(_learning(),), scores=(score,))
    ctx = ctx_env.ctx(view=view, rankings={"": "rank"})
    assert ctx.search_learnings("baseline")[0].id == "L-1"
    ctx.read_learning("L-1")
    ctx.get_candidate(c.id)
    ctx.best(1)
    assert ctx.scores_of(c.id) == [score]
    new = await ctx.run_tool({"x": 2.0})
    assert ctx.scores_of(new.id) == []
    assert new.read_ids == ["L-1", c.id]
    with pytest.raises(SoftError, match="unknown candidate"):
        ctx.scores_of("c-hidden")


async def test_read_snapshot_is_taken_before_tool_awaits(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = await _snapshot(ctx_env, learnings=(_learning(), _learning("L-2")))
    ctx = ctx_env.ctx(view=view)
    ctx.read_learning("L-1")
    entered, release = anyio.Event(), anyio.Event()
    submit = ctx_env.rt.executor.submit

    async def held(job: Job) -> ToolOutput:
        entered.set()
        await release.wait()
        return await submit(job)

    monkeypatch.setattr(ctx_env.rt.executor, "submit", held)
    result: list[Candidate] = []

    async def run() -> None:
        result.append(await ctx.run_tool({"x": 1.0}))

    async with anyio.create_task_group() as group:
        group.start_soon(run)
        with anyio.fail_after(10):
            await entered.wait()
        ctx.read_learning("L-2")
        release.set()
    assert result[0].read_ids == ["L-1"]
    assert ctx_env.events("tool_end")[-1].payload["candidate"]["read_ids"] == ["L-1"]
    assert ctx_env.rt.repeats.candidates_of("t/a")[0].read_ids == ["L-1"]


async def test_recovered_and_linked_results_keep_original_read_ids(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(own=[_learning()])
    ctx.read_learning("L-1")
    original = await ctx.run_tool({"x": 1.0})
    rt = ctx_env.rt
    recovered = AgentContext(replace(rt, resumed=(original,), own=[original, _learning("L-2")]))
    recovered.read_learning("L-2")
    assert (await recovered.run_tool({"x": 1.0})).read_ids == ["L-1"]
    earlier = original.model_copy(update={"round": 0})
    linked = ctx_env.ctx(
        agent="b",
        own=[_learning("L-2")],
        repeats=RepeatIndex(rt.repeats.run_id, [(rt.repeats.key(rt, original.request), earlier)]),
    )
    linked.read_learning("L-2")
    result = await linked.run_tool({"x": 1.0})
    assert result.id == original.id
    assert result.read_ids == ["L-1"]
    assert ctx_env.counters.tool_runs == 0
    assert len(ctx_env.events("tool_end")) == 1


async def test_my_candidates_deduplicate_prior_and_current_and_copy(ctx_env: CtxEnv) -> None:
    old, current = _candidate(), _candidate("c-current", round=1)
    ctx = ctx_env.ctx(
        own=[current], repeats=RepeatIndex("engine-run", [("a", old), ("b", current)])
    )
    assert [c.id for c in ctx.my_candidates()] == [old.id, current.id]
    ctx.my_candidates()[0].read_ids.append("changed")
    assert old.read_ids == []
    assert ctx_env.counters.turns == 0


async def test_notes_utf8_limit_and_validation_secrecy(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    await ctx.save_notes("é" * 8192)
    assert ctx.notes == "é" * 8192
    for text in ["é" * 8193, SECRET + "\ud800"]:
        with pytest.raises(SoftError) as error:
            await ctx.save_notes(text)
        assert_secret_absent(error.value, SECRET)
    assert len(ctx_env.events("notes_saved")) == 1


@pytest.mark.parametrize(
    "fields",
    [
        {"title": SECRET + "\n"},
        {"title": SECRET + "\ud800"},
        {"text": SECRET * 100},
        {"tags": [SECRET + "\ud800"]},
    ],
)
async def test_learning_validation_secrecy(ctx_env: CtxEnv, fields: dict[str, Any]) -> None:
    c = _candidate()
    ctx = ctx_env.ctx(own=[c])
    args: dict[str, Any] = {"title": "title", "text": "text", "evidence": [c.id], **fields}
    with pytest.raises(SoftError) as error:
        await ctx.post_learning(**args)
    assert_secret_absent(error.value, SECRET, sanitized=True)
    assert ctx_env.counters.learnings == 0
    assert ctx_env.events("learning") == []


async def test_learning_provenance_visibility_and_refines(ctx_env: CtxEnv) -> None:
    c = _candidate()
    a = ctx_env.ctx(own=[c, _learning()])
    b = ctx_env.ctx(agent="b")
    with requesting_call("call-1"):
        learning = await a.post_learning(
            "  é baseline  ",
            "works",
            [c.id],
            tags=["fit"],
            refines=["L-1"],
            duplicate_reason="allowed",
        )
    assert learning.title == "é baseline"
    assert learning.author == "t/a"
    assert learning.posted_by_call == "call-1"
    assert learning.id.startswith("L-")
    assert a.read_learning(learning.id) == learning
    assert a.search_learnings("é", tags=["fit"]) == [learning]
    assert b.search_learnings("é") == []
    with pytest.raises(SoftError, match="unknown learning"):
        b.read_learning(learning.id)
    with pytest.raises(SoftError, match="unknown learning"):
        await a.post_learning("next", "text", [c.id], refines=["L-hidden"])
    assert ctx_env.rt.board.get_learning(learning.id) == learning


async def test_all_actions_obey_gate_and_watchdog(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(gate=lambda: LimitReached("stop", "stopping"))
    for call in [
        lambda: ctx.get_candidate("c"),
        lambda: ctx.read_learning("L"),
        lambda: ctx.search_learnings("x"),
        lambda: ctx.best(1),
        lambda: ctx.scores_of("c"),
    ]:
        with pytest.raises(LimitReached, match="stopping"):
            call()
    for call in [
        ctx.run_tool({}),
        ctx.post_learning("t", "x", []),
        ctx.save_notes("x"),
        ctx.call_action("nope", {}),
    ]:
        with pytest.raises(LimitReached, match="stopping"):
            await call
    assert ctx_env.counters.turns == 0
    assert ctx_env.events("agent_counters") == []
    ctx = ctx_env.ctx()
    ctx_env.rt.watchdog.limit = "max_consecutive_failures"
    ctx_env.rt.watchdog.reason = "failed"
    with pytest.raises(LimitReached) as error:
        ctx.search_learnings("x")
    assert error.value.limit == "max_consecutive_failures"


async def test_worker_nested_reads_use_loop_and_charge_once(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = await _snapshot(ctx_env, candidates=(_candidate(),))
    owner = get_ident()
    get = view.get_candidate
    threads: list[int] = []

    def lookup(id: str) -> Candidate:
        threads.append(get_ident())
        return get(id)

    monkeypatch.setattr(view, "get_candidate", lookup)
    ctx = ctx_env.ctx(view=view)

    def worker(context: AgentContext) -> JsonValue:
        assert get_ident() != owner
        context.get_candidate("c-old")
        context.get_candidate("c-old")
        context.search_learnings("x")
        return "done"

    async def handler(context: AgentContext, args: EmptyArgs) -> JsonValue:
        return await anyio.to_thread.run_sync(worker, context)

    ctx.actions = (
        Action(
            name="nested",
            description="Read twice",
            input_model=EmptyArgs,
            handler=handler,
            counts_as="board_read",
        ),
    )
    assert await ctx.call_action("nested", {}) == "done"
    assert threads == [owner, owner]
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (1, 1)
    assert len(ctx_env.events("agent_counters")) == 1
    ctx.get_candidate("c-old")
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (2, 2)


async def test_simultaneous_contexts_and_cancellation_keep_charges_independent(
    ctx_env: CtxEnv,
) -> None:
    a = ctx_env.ctx()
    counters_a = ctx_env.counters
    b = ctx_env.ctx(agent="b")
    counters_b = ctx_env.counters
    entered, release = anyio.Event(), anyio.Event()

    async def handler(context: AgentContext, args: EmptyArgs) -> JsonValue:
        await anyio.to_thread.run_sync(context.search_learnings, "inside")
        b.search_learnings("other context")
        entered.set()
        await release.wait()
        return None

    a.actions = (
        Action(
            name="wait",
            description="Wait",
            input_model=EmptyArgs,
            handler=handler,
            counts_as="board_read",
        ),
    )
    scope = anyio.CancelScope()

    async def run() -> None:
        with scope:
            await a.call_action("wait", {})
        a.search_learnings("after cancellation")

    async with anyio.create_task_group() as group:
        group.start_soon(run)
        with anyio.fail_after(10):
            await entered.wait()
        b.search_learnings("parallel")
        scope.cancel()
    assert (counters_a.turns, counters_a.board_reads) == (2, 2)
    assert (counters_b.turns, counters_b.board_reads) == (2, 2)
    assert len(ctx_env.events("agent_counters")) == 4


async def test_action_validation_and_failure_hide_values(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()

    async def handler(context: AgentContext, args: EmptyArgs) -> JsonValue:
        raise ValueError(SECRET)

    ctx.actions = (Action(name="fail", description="Fail", input_model=EmptyArgs, handler=handler),)
    with pytest.raises(SoftError) as error:
        await ctx.call_action("fail", {SECRET: SECRET})
    assert_secret_absent(error.value, SECRET, sanitized=True)
    with pytest.raises(SoftError, match="action failed: ValueError") as error:
        await ctx.call_action("fail", {})
    assert_secret_absent(error.value, SECRET, sanitized=True)
    ctx.search_learnings("after exception")
    assert (ctx_env.counters.turns, ctx_env.counters.board_reads) == (3, 1)


async def test_script_candidate_files_authorized_and_fetched_in_worker(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = await ctx_env.script_ctx()
    rt = ctx_env.rt
    digest = ctx_env.artifacts.put_bytes(b"a = 2\n")
    c = _candidate().model_copy(update={"overlay": {"problem/a.phyk": digest}})
    rt.own.append(c)
    owner = get_ident()
    getter = rt.artifacts.get_bytes
    reads: list[int] = []

    def get_bytes(sha256: str) -> bytes:
        reads.append(get_ident())
        return getter(sha256)

    monkeypatch.setattr(rt.artifacts, "get_bytes", get_bytes)
    assert await ctx.call_action("read_file", {"path": "problem/a.phyk", "candidate": c.id}) == {
        "path": "problem/a.phyk",
        "content": "a = 2\n",
    }
    assert reads
    assert all(thread != owner for thread in reads)
    assert (rt.counters.turns, rt.counters.board_reads) == (1, 1)
    assert len(ctx_env.events("agent_counters")) == 1
    assert (await ctx.run_tool({})).read_ids == [c.id]


async def test_edit_byte_limit_uses_editable_script(ctx_env: CtxEnv) -> None:
    ctx = await ctx_env.script_ctx()
    with pytest.raises(SoftError):
        await ctx.run_tool({}, edits={"problem/a.phyk": "x" * 262_145})
    assert ctx_env.counters.tool_runs == 0
    assert ctx_env.events("reserve") == []


async def test_counters_and_learning_replay(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(own=[_candidate()])
    ctx.search_learnings("x")
    learning = await ctx.post_learning("title", "body", ["c-old"])
    await ctx.save_notes("carry")
    await ctx_env.rt.journal.commit()
    events = read_committed(ctx_env.env.run_store)
    assert events is not None
    state = RunState.from_events(events[1])
    assert state.learnings[learning.id] == learning
    assert state.goals["g"].counters["t/a"] == {"turns": 3, "board_reads": 1}
    assert state.goals["g"].notes["t/a"] == "carry"
    counter_events = [e for e in events[1] if e.type == "agent_counters"]
    assert counter_events[-1].payload == {"turns": 3, "board_reads": 1}


class EmptyArgs(FrozenModel):
    pass


async def test_properties_and_my_candidates_are_free(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(digest="summary", notes="carried", resumed=(_candidate(),))
    assert ctx.goal == ctx_env.rt.goal
    assert ctx.spec == ctx_env.rt.spec
    assert ctx.round == 1
    assert ctx.round_spec == ctx_env.rt.round_spec
    assert ctx.digest == "summary"
    assert ctx.notes == "carried"
    assert ctx.resumed is True
    assert ctx.tool_description == ctx_env.rt.tool.get_description()
    assert (
        ctx.input_model.model_json_schema() == ctx_env.rt.tool.get_input_model().model_json_schema()
    )
    assert ctx.editable == ()
    assert ctx.actions == ()
    assert ctx.my_candidates() == []
    assert ctx.budget.tokens is None
    assert ctx_env.counters.turns == 0
    assert ctx_env.events("agent_counters") == []


@pytest.mark.parametrize("prior_own", [False, True])
async def test_snapshot_file_access_does_not_admit_new_other_agent_files(
    ctx_env: CtxEnv, prior_own: bool
) -> None:
    ctx = await ctx_env.script_ctx()
    rt = ctx_env.rt
    digest = ctx_env.artifacts.put_bytes(b"earlier\n")
    earlier = _candidate(agent="a" if prior_own else "b", round=1).model_copy(
        update={"part": rt.part, "overlay": {"problem/a.phyk": digest}}
    )
    event = rt.journal.append(
        EventDraft(
            type="tool_end", goal_id="g", payload={"candidate": earlier.model_dump(mode="json")}
        )
    )
    rt.board.put_candidate(earlier, epoch=event.epoch, seq=event.seq)
    await rt.journal.commit()
    frontier = await anyio.to_thread.run_sync(rt.board.frontier)
    view = await anyio.to_thread.run_sync(rt.board.snapshot, rt.goal_key, frontier)
    ctx = AgentContext(
        replace(
            rt,
            view=view,
            repeats=RepeatIndex("engine-run", [("earlier", earlier)]),
        )
    )
    if prior_own:
        own = ctx.my_candidates()
        assert [c.id for c in own] == [earlier.id]
        digest = own[0].overlay["problem/a.phyk"]
    hidden_digest = ctx_env.artifacts.put_bytes(b"later\n")
    later = earlier.model_copy(
        update={
            "id": "c-later",
            "agent": "b",
            "round": 1,
            "overlay": {"problem/a.phyk": hidden_digest},
        }
    )
    event = rt.journal.append(
        EventDraft(
            type="tool_end", goal_id="g", payload={"candidate": later.model_dump(mode="json")}
        )
    )
    rt.board.put_candidate(later, epoch=event.epoch, seq=event.seq)
    await rt.journal.commit()
    assert await anyio.to_thread.run_sync(ctx._read_artifact, digest) == b"earlier\n"
    with pytest.raises(SoftError, match=r"^file not in your view$"):
        await anyio.to_thread.run_sync(ctx._read_artifact, hidden_digest)
    assert (rt.counters.turns, rt.counters.board_reads) == (0, 0)
    assert ctx_env.events("agent_counters") == []
    assert await ctx.call_action(
        "read_file", {"path": "problem/a.phyk", "candidate": earlier.id}
    ) == {"path": "problem/a.phyk", "content": "earlier\n"}
    with pytest.raises(SoftError, match="unknown candidate"):
        await ctx.call_action("read_file", {"path": "problem/a.phyk", "candidate": later.id})
    with pytest.raises(SoftError, match="file not in your view"):
        await anyio.to_thread.run_sync(ctx._read_artifact, hidden_digest)
    assert (rt.counters.turns, rt.counters.board_reads) == (2, 2)


@pytest.mark.parametrize(
    "method",
    [
        "budget",
        "scores_of",
        "search_learnings",
        "read_learning",
        "get_candidate",
        "best",
        "post_learning",
        "save_notes",
        "call_action",
    ],
)
def test_memory_examples_are_self_contained_and_run(method: str) -> None:
    doc = getattr(AgentContext, method).__doc__
    assert doc is not None
    example = DocTestParser().get_doctest(doc, {"AgentContext": AgentContext}, method, None, 0)
    assert example.examples
    assert not any(item.options.get(SKIP) for item in example.examples)
    runner = DocTestRunner()
    result = runner.run(example)
    assert result.failed == 0
    assert result.attempted > 1


async def test_tool_metadata_failure_is_sanitized(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> str:
        raise ValueError(SECRET)

    monkeypatch.setattr(ctx_env.env.rt.tool, "get_description", broken)
    with pytest.raises(SoftError, match="invalid tool metadata: ValueError") as error:
        ctx_env.ctx()
    assert_secret_absent(error.value, SECRET, sanitized=True)


async def test_part_context_routes_learnings_events_and_read_ids(ctx_env: CtxEnv) -> None:
    from swarm_arena import Goal, PartSpec, Plan

    goal = Goal(id="sub", namespace="n")
    await ctx_env.rt.journal.append_durable(
        EventDraft(
            type="plan_revision",
            goal_id="g",
            payload={
                "plan": Plan(
                    goal=ctx_env.rt.goal,
                    teams=[ctx_env.rt.team.model_copy(update={"part": "part"})],
                    judges=[],
                    parts=[PartSpec(name="part", goal=goal)],
                    revision=1,
                ).model_dump(mode="json"),
                "origin": "human",
                "first_round": 1,
            },
        )
    )
    ctx = ctx_env.ctx(
        goal=goal,
        goal_key=goal.key(),
        part="part",
        view=ctx_env.env.board.snapshot(goal.key(), ctx_env.env.board.frontier()),
    )
    assert ctx.goal.id == "sub"
    evidence = await ctx.run_tool({"x": 0})
    learning = await ctx.post_learning("part hint", "body", [evidence.id])
    assert learning.goal_id == "g"
    assert ctx.search_learnings("part hint") == [learning]
    result = await ctx.run_tool({"x": 1})
    assert result.goal_id == "g"
    assert result.read_ids == [learning.id]
    await ctx_env.rt.journal.commit()
    committed = read_committed(ctx_env.env.run_store)
    assert committed is not None
    events = [e for e in committed[1] if e.agent == "a"]
    assert all(e.goal_id == "g" for e in events)
    state = RunState.from_events(committed[1])
    assert state.candidates[result.id].read_ids == [learning.id]


async def test_seed_parents_are_read_only_to_manager(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    first = await ctx.run_tool({"x": 1})
    seeded = AgentContext(replace(ctx_env.rt, seed_parents=(first.id,)), gate=lambda: None)
    assert seeded.seed_parents == (first.id,)
    with pytest.raises(AttributeError):
        cast(Any, seeded).seed_parents = ()
    second = await seeded.run_tool({"x": 2}, parents=(first.id, first.id))
    assert second.parent_ids == [first.id]


async def test_new_context_does_not_inherit_another_runtimes_starts(ctx_env: CtxEnv) -> None:
    from swarm_arena._agent.tool_run import execute_tool_run

    await execute_tool_run(ctx_env.env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert ctx_env.env.rt.started_runs == 1
    ctx_env.ctx(agent="b")
    assert ctx_env.rt.started_runs == 0


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("operation", ["notes", "board", "paid"])
@pytest.mark.parametrize("available", [False, True])
async def test_escaped_action_lifetime_is_revoked(
    ctx_env: CtxEnv, cancelled: bool, operation: str, available: bool
) -> None:
    import asyncio

    from swarm_arena._budget.ledger import Ledger
    from swarm_arena.tool import Usage

    class PaidExecutor:
        async def submit(self, job: Job) -> ToolOutput:
            assert job.on_started is not None
            await job.on_started(Job.Started(kind="process", job_id=job.id))
            return ToolOutput(status="ok", usage=Usage(usd=0.000003))

    ctx = ctx_env.ctx(
        limits=Limits(max_turns_per_round=2 if available else 1), executor=PaidExecutor()
    )
    rt = ctx_env.rt
    release, entered = anyio.Event(), anyio.Event()
    children: list[asyncio.Task[None]] = []
    results: list[Any] = []

    async def child() -> None:
        await release.wait()
        for _ in range(2):
            try:
                if operation == "notes":
                    results.append(await ctx.save_notes("escaped"))
                elif operation == "board":
                    results.append(ctx.search_learnings("escaped"))
                else:
                    results.append(await ctx.run_tool({"x": float(len(results))}))
            except Exception as error:
                results.append(error)

    async def handler(context: AgentContext, args: EmptyArgs) -> JsonValue:
        children.append(asyncio.create_task(child()))
        entered.set()
        if cancelled:
            await anyio.sleep_forever()
        return None

    ctx.actions = (
        Action(name="spawn", description="Spawn", input_model=EmptyArgs, handler=handler),
    )
    parent = asyncio.create_task(ctx.call_action("spawn", {}))
    try:
        with anyio.fail_after(10):
            await entered.wait()
        if cancelled:
            parent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await parent
        else:
            await parent
        release.set()
        await children[0]
    finally:
        release.set()
        for task in [parent, *children]:
            if not task.done():
                task.cancel()
        await asyncio.gather(parent, *children, return_exceptions=True)
    assert sum(isinstance(result, LimitReached) for result in results) == (1 if available else 2)
    assert all(
        result.limit == "max_turns_per_round"
        for result in results
        if isinstance(result, LimitReached)
    )
    assert rt.counters.turns == (2 if available else 1)
    assert rt.counters.board_reads == int(available and operation == "board")
    assert rt.counters.tool_runs == int(available and operation == "paid")
    assert len(ctx_env.events("notes_saved")) == int(available and operation == "notes")
    assert len(ctx_env.events("reserve")) == int(available and operation == "paid")
    assert [event.payload["actual"] for event in ctx_env.events("settle")] == (
        [Amount(usd=0.000003).model_dump()] if available and operation == "paid" else []
    )
    await rt.journal.commit()
    committed = read_committed(ctx_env.env.run_store)
    assert committed is not None
    state = RunState.from_events(committed[1])
    assert state.goals["g"].counters["t/a"] == {
        "turns": rt.counters.turns,
        "board_reads": rt.counters.board_reads,
        **({"tool_runs": rt.counters.tool_runs} if available and operation == "paid" else {}),
    }
    assert Ledger.replay(state, journal=rt.journal, ids=rt.ids).remaining(
        rt.holder
    ) == rt.ledger.remaining(rt.holder)


@pytest.mark.parametrize("kind", [ValueError, SoftError, LimitReached])
async def test_action_connector_control_errors_are_sanitized(
    ctx_env: CtxEnv, kind: type[Exception]
) -> None:
    from uuid import uuid4

    secret = str(uuid4())
    ctx = ctx_env.ctx()

    async def handler(context: AgentContext, args: EmptyArgs) -> JsonValue:
        error = LimitReached("budget", secret) if kind is LimitReached else kind(secret)
        error.add_note(secret)
        raise error from RuntimeError(secret)

    ctx.actions = (Action(name="fail", description="Fail", input_model=EmptyArgs, handler=handler),)
    with pytest.raises(SoftError, match=f"action failed: {kind.__name__}") as caught:
        await ctx.call_action("fail", {})
    assert_secret_absent(caught.value, secret, sanitized=True)
    assert caught.value.__cause__ is None
    assert not getattr(caught.value, "__notes__", ())


@pytest.mark.parametrize("control", ["soft", "limit", "gate", "watchdog", "fenced", "cleanup"])
async def test_action_preserves_core_control_signals(ctx_env: CtxEnv, control: str) -> None:
    from swarm_arena._errors import Fenced, _is_core_control, _stop_cleanup_error

    blocked = False
    ctx = ctx_env.ctx(gate=lambda: LimitReached("stop", "stopping") if blocked else None)

    async def handler(context: AgentContext, args: EmptyArgs) -> JsonValue:
        nonlocal blocked
        if control == "soft":
            return context.get_candidate("c-absent").id
        if control == "fenced":
            raise _core_error(Fenced, "run ownership lost")
        if control == "cleanup":
            raise _stop_cleanup_error()
        if control == "gate":
            blocked = True
        else:
            ctx_env.rt.watchdog.limit = "max_consecutive_failures"
            ctx_env.rt.watchdog.reason = "agent stopped"
        context.search_learnings("nested")
        return None

    ctx.actions = (Action(name="core", description="Core", input_model=EmptyArgs, handler=handler),)
    expected = {
        "fenced": Fenced,
        "soft": SoftError,
        "cleanup": RuntimeError,
    }.get(control, LimitReached)
    with pytest.raises(expected) as caught:
        await ctx.call_action("core", {})
    if control == "soft":
        assert str(caught.value) == "unknown candidate: c-absent"
    elif control == "cleanup":
        assert _is_core_control(caught.value)
        assert str(caught.value) == "stop cleanup timed out"
    elif isinstance(caught.value, LimitReached):
        assert caught.value.limit == ("stop" if control == "gate" else "max_consecutive_failures")


@pytest.mark.parametrize("kind", [SoftError, LimitReached])
@pytest.mark.parametrize("attack", ["copy", "mutate"])
async def test_core_signal_payload_cannot_be_transferred_or_mutated(
    ctx_env: CtxEnv, kind: type[Exception], attack: str
) -> None:
    import traceback
    from uuid import uuid4

    secret = str(uuid4())
    ctx = ctx_env.ctx()
    originals: list[Exception] = []

    async def handler(context: AgentContext, args: EmptyArgs) -> JsonValue:
        try:
            if kind is LimitReached and attack == "mutate":
                # Obtain a genuine public gate control without a private marker import.
                limited = ctx_env.ctx(gate=lambda: LimitReached("stop", "the run is stopping"))
                limited.get_candidate("c-absent")
            else:
                context.get_candidate("c-absent")
        except (SoftError, LimitReached) as core:
            originals.append(core)
            if attack == "copy":
                error = (
                    LimitReached("budget", secret) if kind is LimitReached else SoftError(secret)
                )
                error.__dict__.update(core.__dict__)
            else:
                error = core
                error.args = (secret,)
                if isinstance(error, LimitReached):
                    error.limit = secret
            error.add_note(secret)
            raise error from RuntimeError(secret)
        return None

    ctx.actions = (
        Action(name="attack", description="Attack", input_model=EmptyArgs, handler=handler),
    )
    with pytest.raises((SoftError, LimitReached)) as caught:
        await ctx.call_action("attack", {})
    assert_secret_absent(caught.value, secret, sanitized=True)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert secret not in str(getattr(caught.value, "limit", ""))
    assert not getattr(caught.value, "__notes__", ())
    assert caught.value.__cause__ is None
    assert caught.value is not originals[0]
    if attack == "copy":
        assert type(caught.value) is SoftError
        assert str(caught.value) == f"action failed: {kind.__name__}"
    elif kind is SoftError:
        assert str(caught.value) == "unknown candidate: c-absent"
    else:
        assert isinstance(caught.value, LimitReached)
        assert caught.value.limit == "stop"
        assert str(caught.value) == "the run is stopping"


@pytest.mark.parametrize("own_part", [None, "p"])
async def test_best_reads_declared_fixed_siblings_and_preserves_permissions(
    ctx_env: CtxEnv, own_part: str | None
) -> None:
    from swarm_arena import Goal, PartSpec, Plan
    from swarm_arena._engine.goal import r0_revision_draft

    rt = ctx_env.env.rt
    part_goal = Goal(id="sub", namespace="n", inputs={"part": 1})
    plan = Plan(
        goal=rt.goal, teams=[], judges=[], revision=1, parts=[PartSpec(name="p", goal=part_goal)]
    )
    await rt.journal.append_durable(r0_revision_draft(plan, plan.base_dir))
    main = _candidate("main", round=1)
    digest = rt.artifacts.put_bytes(b"sibling-file")
    part = _candidate("part", round=1).model_copy(
        update={"part": "p", "overlay": {"sibling.txt": digest}}
    )
    view = await _snapshot(
        ctx_env,
        candidates=(main, part),
        scores=(
            Score(id="main-score", candidate_id="main", judge="main-rank", key=(1.0,)),
            Score(id="part-score", candidate_id="part", judge="part-rank", key=(3.0,)),
            Score(id="part-same-judge", candidate_id="part", judge="main-rank", key=(100.0,)),
        ),
    )
    from swarm_arena._board.arena import ArenaView

    assert isinstance(view, ArenaView)
    siblings = {"": view._sibling_view(None), "p": view._sibling_view("p")}
    goal = part_goal if own_part else rt.goal
    ctx = ctx_env.ctx(
        goal=goal,
        goal_key=goal.key(),
        part=own_part,
        view=siblings[own_part or ""],
        sibling_views=siblings,
        rankings={"": "main-rank", "p": "part-rank"},
    )
    assert [row.candidate_id for row in ctx.best(1)] == (["part"] if own_part else ["main"])
    assert [row.candidate_id for row in ctx.best(1, part="p")] == ["part"]
    assert [row.candidate_id for row in ctx.best(1, part="")] == ["main"]
    with pytest.raises(SoftError, match="outside"):
        ctx.best(1, part="not-declared")
    assert ctx_env.counters.board_reads == 4
    assert set(ctx._read_ids) == {"main", "part"}
    if own_part:
        with pytest.raises(SoftError, match="unknown candidate"):
            ctx.get_candidate("main")
    # A later committed score cannot change any selected sibling's fixed ranking.
    await _snapshot(
        ctx_env, scores=(Score(id="later", candidate_id="part", judge="part-rank", key=(9.0,)),)
    )
    assert ctx.best(1, part="p")[0].key == (3.0,)


@pytest.mark.parametrize("provider_first", [True, False])
async def test_provider_stop_cannot_become_action_timeout(
    ctx_env: CtxEnv, provider_first: bool
) -> None:
    from swarm_arena._agent.context import _ActionInflight, _local_timer_scope
    from swarm_arena._agent.tool_run import Inflight
    from swarm_arena._errors import ProviderFailed

    ctx = ctx_env.ctx()
    owned = _ActionInflight(Inflight())
    outer_scope, manager_scope, action_scope = (anyio.CancelScope() for _ in range(3))
    with (
        _local_timer_scope(outer_scope) as outer,
        _local_timer_scope(manager_scope, relay_provider_stop=True) as manager,
        _local_timer_scope(action_scope) as action,
    ):
        if provider_first:
            outer.stop_provider(_core_error(ProviderFailed, "openai"))
        owned.stop(action_scope, cancel=True)
        if not provider_first:
            outer.stop_provider(_core_error(ProviderFailed, "openai"))
        assert manager_scope.cancel_called
        assert not outer_scope.cancel_called
        assert manager.failure is not None
        assert _core_category(manager.failure) is _SignalCategory.PROVIDER
        assert not owned.is_timeout_refusal(
            _core_error(LimitReached, "stop", "action stopped"), action, ctx
        )
    assert outer.provider_child is None


@pytest.mark.parametrize("control", ["fenced", "terminal"])
async def test_provider_timer_preserves_stronger_control(control: str) -> None:
    from swarm_arena._agent.context import _local_timer_scope
    from swarm_arena._errors import Fenced, ProviderFailed

    error = (
        _core_error(Fenced, "run ownership lost")
        if control == "fenced"
        else _core_error(RuntimeError, "terminal", _category=_SignalCategory.TERMINAL)
    )
    with _local_timer_scope(anyio.CancelScope()) as timer:
        timer.stop_provider(_core_error(ProviderFailed, "openai"))
        timer.record(error)
        timer.stop_provider(_core_error(ProviderFailed, "openai"))
        assert timer.failure is not None
        assert _core_category(timer.failure) is (
            _SignalCategory.FENCED if control == "fenced" else _SignalCategory.TERMINAL
        )


@pytest.mark.parametrize("closed_registry", ["parent", "action"])
@pytest.mark.parametrize("limit", ["budget", "run_timeout_s", "stop"])
async def test_run_tool_preserves_live_gate_after_registry_closes(
    ctx_env: CtxEnv, closed_registry: str, limit: str
) -> None:
    from swarm_arena._agent.context import _ActionInflight
    from swarm_arena._agent.tool_run import Inflight, track_inflight
    from swarm_arena._engine.stops import StopState

    state = StopState()
    parent = Inflight()
    owned = _ActionInflight(parent)
    stopped = anyio.Event()
    ctx = ctx_env.ctx(gate=state.gate_for(ctx_env.rt.goal_id))

    async def stop() -> None:
        if limit == "budget":
            state.goal_budget.add(ctx_env.rt.goal_id)
        elif limit == "run_timeout_s":
            state.run_timeout = True
        else:
            state.stop = True
        (parent if closed_registry == "parent" else owned).close()
        stopped.set()

    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(stop)
            await stopped.wait()
            with track_inflight(owned), pytest.raises(LimitReached) as caught:
                await ctx.run_tool({"x": 1.0})
    assert caught.value.limit == limit
    assert _is_core_signal(caught.value)
    assert ctx_env.counters.turns == ctx_env.counters.tool_runs == 0
    assert not ctx_env.events("reserve")
    assert not ctx_env.events("tool_start")
