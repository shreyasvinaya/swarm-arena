from dataclasses import replace
from typing import Any

import pytest
from _fakes.engine import CtxEnv
from _fakes.pr1_r5_ids import HostileString, NotString, hostile
from _fakes.tools import AlwaysFailCheck

from swarm_arena import Limits, ScriptTool
from swarm_arena._agent.context import _prepare_metadata
from swarm_arena._errors import SoftError, _core_category, _is_core_signal, _SignalCategory
from swarm_arena.manager import LimitReached


@pytest.mark.parametrize("source", ["gate", "watchdog"])
@pytest.mark.parametrize("valid", [True, False])
async def test_context_limit_strings_are_normalized_before_validation(
    ctx_env: CtxEnv, source: str, valid: bool
) -> None:
    limit = hostile("stop" if valid else "secret-unknown-limit")
    ctx = ctx_env.ctx(gate=lambda: LimitReached(limit, "stopping") if source == "gate" else None)
    if source == "watchdog":
        ctx_env.rt.watchdog.limit = limit
        ctx_env.rt.watchdog.reason = "stopping"
    with pytest.raises(LimitReached if valid else ValueError) as error:
        ctx.search_learnings("x")
    assert _is_core_signal(error.value)
    if valid:
        assert isinstance(error.value, LimitReached)
        assert type(error.value.limit) is str
        assert error.value.limit == "stop"
        assert _core_category(error.value) is _SignalCategory.LIMIT
    else:
        assert str(error.value) == "invalid agent limit"
        assert _core_category(error.value) is _SignalCategory.DIAGNOSTIC
    assert ctx_env.counters.turns == 0


@pytest.mark.parametrize("source", ["gate", "watchdog"])
async def test_context_rejects_non_string_limits_without_value_hooks(
    ctx_env: CtxEnv, source: str
) -> None:
    limit: Any = NotString()
    ctx = ctx_env.ctx(gate=lambda: LimitReached(limit, "stopping") if source == "gate" else None)
    if source == "watchdog":
        ctx_env.rt.watchdog.limit = limit
    with pytest.raises(ValueError, match=r"^invalid agent limit$") as error:
        ctx.search_learnings("x")
    assert _is_core_signal(error.value)
    assert _core_category(error.value) is _SignalCategory.DIAGNOSTIC
    assert ctx_env.counters.turns == 0


async def test_agent_read_identifiers_are_plain_strings(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(limits=Limits(max_turns_per_round=30, max_board_reads_per_round=30))
    candidate = await ctx.run_tool({"x": 1.0})
    learning = await ctx.post_learning("one", "body", [candidate.id], tags=["tag"])
    assert ctx.get_candidate(hostile(candidate.id)).id == candidate.id
    assert ctx.scores_of(hostile(candidate.id)) == []
    assert ctx.read_learning(hostile(learning.id)).id == learning.id
    assert ctx.search_learnings("body", tags=[hostile("tag")], scope=hostile("goal")) == [learning]  # ty: ignore[invalid-argument-type]
    assert all(type(key) is str for key in ctx._read_ids)
    with pytest.raises(SoftError, match="part outside board snapshot"):
        ctx.best(1, part=hostile("other"))
    with pytest.raises(SoftError, match=r"^unknown candidate: absent$"):
        ctx.get_candidate(hostile("absent"))
    with pytest.raises(SoftError, match=r"^unknown learning: absent$"):
        ctx.read_learning(hostile("absent"))


async def test_agent_action_and_collection_identifiers(ctx_env: CtxEnv) -> None:
    ctx = await ctx_env.script_ctx()
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
    ctx = await ctx_env.script_ctx()
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


async def test_bound_metadata_lookup_strings_are_detached(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = ctx_env.env.rt.tool
    monkeypatch.setattr(tool, "get_description", lambda: hostile("description"))
    monkeypatch.setattr(tool, "editable", lambda: (hostile("file.py"),))
    check = AlwaysFailCheck()
    # Feedback is connector metadata; no execution or scoring is needed here.
    monkeypatch.setattr(check, "feedback", hostile("reasons"))
    metadata = _prepare_metadata(tool, [(hostile("judge"), check)])
    assert type(metadata.description) is str
    assert all(type(path) is str for path in metadata.editable)
    assert all(type(name) is str and type(mode) is str for name, mode in metadata.feedback.items())
    assert metadata.feedback["judge"] == "reasons"


@pytest.mark.parametrize("field", ["name", "description", "counts_as"])
async def test_invalid_action_metadata_is_value_free(
    ctx_env: CtxEnv, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    original = ScriptTool(command=["echo"]).extra_actions()[0]
    action = replace(original, **{field: NotString()})
    monkeypatch.setattr(ctx_env.env.rt.tool, "extra_actions", lambda: (action,))
    with pytest.raises(SoftError, match="invalid tool metadata: TypeError"):
        ctx_env.ctx()
