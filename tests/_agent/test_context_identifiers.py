from typing import Any

import pytest
from _fakes.engine import CtxEnv
from _fakes.pr1_r5_ids import HostileString, NotString, hostile

from swarm_arena import Limits
from swarm_arena._errors import SoftError


async def test_agent_read_identifiers_are_plain_strings(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(limits=Limits(max_turns_per_round=30, max_board_reads_per_round=30))
    candidate = await ctx.run_tool({"x": 1.0})
    learning = await ctx.post_learning("one", "body", [candidate.id], tags=["tag"])
    assert ctx.get_candidate(hostile(candidate.id)).id == candidate.id
    assert ctx.scores_of(hostile(candidate.id)) == []
    assert ctx.read_learning(hostile(learning.id)).id == learning.id
    assert ctx.search_learnings("body", tags=[hostile("tag")], scope=hostile("goal")) == [learning]  # ty: ignore[invalid-argument-type]
    assert all(type(key) is str for key in ctx._read_ids)
    with pytest.raises(SoftError, match="own part"):
        ctx.best(1, part=hostile("other"))
    with pytest.raises(SoftError, match=r"^unknown candidate: absent$"):
        ctx.get_candidate(hostile("absent"))
    with pytest.raises(SoftError, match=r"^unknown learning: absent$"):
        ctx.read_learning(hostile("absent"))


async def test_agent_action_and_collection_identifiers(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.script_ctx()
    assert await ctx.call_action(hostile("list_files"), {}) == {
        "files": ["problem/a.phyk", "problem/run.py"]
    }
    ctx = ctx_env.ctx()
    parent = await ctx.run_tool({"x": 1.0})
    candidate = await ctx.run_tool({"x": 2.0}, parents=[hostile(parent.id), hostile(parent.id)])
    assert [ref.candidate_id for ref in candidate.request.parents] == [parent.id]
    first = await ctx.post_learning("first", "body", [hostile(candidate.id)], tags=[hostile("tag")])
    second = await ctx.post_learning(
        "second", "body", [hostile(candidate.id)], refines=[hostile(first.id)]
    )
    assert first.tags == ["tag"]
    assert second.refines == [first.id]
    assert all(type(value) is str for value in [*first.evidence, *first.tags, *second.refines])


async def test_agent_edit_names_are_normalized_at_entry(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.script_ctx()
    name = HostileString("problem/a.phyk")
    edits: dict[str, str] = {name: "changed"}
    name.armed = True
    # Admission reaches the ordinary tool validation; the filename hook must not run.
    result = await ctx.run_tool({}, edits=edits)
    assert result.request.edits == {"problem/a.phyk": "changed"}


@pytest.mark.parametrize(
    ("method", "field"),
    [
        ("get_candidate", "id"),
        ("read_learning", "id"),
        ("scores_of", "candidate_id"),
        ("best", "part"),
        ("search_learnings", "tags"),
        ("search_learnings", "scope"),
        ("run_tool", "parents"),
        ("run_tool", "edits path"),
        ("post_learning", "evidence"),
        ("post_learning", "refines"),
        ("post_learning", "tags"),
        ("call_action", "name"),
    ],
)
async def test_agent_rejects_non_string_identifiers_before_admission(
    ctx_env: CtxEnv, method: str, field: str
) -> None:
    ctx = ctx_env.ctx()
    bad: Any = NotString()

    async def invoke() -> None:
        if method in {"get_candidate", "read_learning", "scores_of"}:
            getattr(ctx, method)(bad)
        elif method == "best":
            ctx.best(1, part=bad)
        elif method == "search_learnings":
            search_kwargs: dict[str, Any] = {field: [bad] if field == "tags" else bad}
            ctx.search_learnings("query", **search_kwargs)
        elif method == "run_tool":
            run_kwargs: dict[str, Any] = (
                {"parents": [bad]} if field == "parents" else {"edits": {bad: "x"}}
            )
            await ctx.run_tool({}, **run_kwargs)
        elif method == "post_learning":
            kwargs: dict[str, Any] = {"evidence": ["absent"], field: [bad]}
            await ctx.post_learning("title", "body", **kwargs)
        else:
            await ctx.call_action(bad, {})

    with pytest.raises(TypeError, match=f"^{field} must be str$"):
        await invoke()
    assert ctx_env.counters.turns == 0
    assert ctx_env.counters.board_reads == 0
