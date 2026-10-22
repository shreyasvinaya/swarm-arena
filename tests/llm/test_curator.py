"""Evidence-backed curation, invocation linkage and owned cleanup."""

import asyncio
import json
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import anyio
import pytest
from _engine.conftest import env as env
from _fakes.engine import EngineEnv
from _fakes.llm_components import LateFakeModel
from _fakes.llm_run import open_llm_run

from swarm_arena import Limits, Ref, RunConfig
from swarm_arena._board.digest import UNTRUSTED_CLOSE, UNTRUSTED_OPEN
from swarm_arena._budget.llm_scope import llm_scope
from swarm_arena.board import BoardView, Curator, Learning
from swarm_arena.config import ModelCatalog, ModelSpec
from swarm_arena.llm import BudgetExceeded, LLMCurator
from swarm_arena.testing import FakeModel, FakeTurn


def learning(index: int, **changes: Any) -> Learning:
    return Learning.model_validate(
        {
            "id": f"L{index}",
            "namespace": "physika",
            "run_id": "r",
            "goal_id": "g",
            "round": 1,
            "author": "agent",
            "title": f"Observation {index}",
            "text": "Evidence",
            "evidence": [f"c-{index}"],
            "tags": [f"tag-{index}"],
            **changes,
        }
    )


def configuration() -> RunConfig:
    return RunConfig(
        limits=Limits(max_tokens=100000),
        models=ModelCatalog(
            {
                "cur": ModelSpec(
                    model="fake/cur",
                    roles=["curator"],
                    max_tokens=1000,
                    prices=ModelSpec.Prices(input=1, output=2),
                )
            }
        ),
        settle_timeout_s=2,
    )


def turn(groups: list[dict[str, Any]]) -> FakeTurn:
    return FakeTurn(text=json.dumps({"groups": groups}))


def group(members: list[Any], **changes: Any) -> dict[str, Any]:
    return {"members": members, "title": "Summary", "text": "Combined evidence.", **changes}


def view() -> Mock:
    result = Mock(spec=BoardView)
    result.learnings.return_value = []
    return result


def test_configuration_and_public_ref() -> None:
    curator = LLMCurator(model="cur")
    assert curator.models() == ("cur",)
    assert isinstance(curator.config, LLMCurator.Config)
    assert curator.config.max_group == 8
    assert LLMCurator(LLMCurator.Config(model="cur")).models() == ("cur",)
    assert Ref.of(curator).build(Curator).models() == ("cur",)


@pytest.mark.parametrize(
    "fields", [{"max_group": 1}, {"max_group": True}, {"model": []}, {"unknown": "SECRET"}]
)
def test_config_errors_are_value_free(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r".") as caught:
        LLMCurator(**{"model": "cur", **fields})
    assert "SECRET" not in "".join(traceback.format_exception(caught.value))


async def test_merges_a_group_into_one_summary(tmp_path: Path) -> None:
    fake = FakeModel([turn([group([0, 2])])])
    fake.register("cur")
    items = [
        learning(1),
        learning(2),
        learning(3, evidence=["c-3", "c-1"], tags=["tag-3", "tag-1"], verified=True),
    ]
    async with open_llm_run(tmp_path, config=configuration()) as run:
        scope = run.scope("curator")
        with llm_scope(scope):
            (summary,) = await LLMCurator(model="cur").curate("physika", items, view())
        assert summary.supersedes == ["L1", "L3"]
        assert summary.evidence == ["c-1", "c-3"]
        assert summary.tags == ["tag-1", "tag-3"]
        assert summary.author == "curator"
        assert summary.id == ""
        assert summary.verified
        assert not summary.hidden
        assert not summary.pinned
        assert (summary.namespace, summary.run_id, summary.goal_id, summary.round) == (
            "physika",
            "r",
            "g",
            1,
        )
        assert summary.posted_by_call == scope.call_ids[-1]
        assert scope.tranche.pending() == 0
        assert run.charged(scope).usd == pytest.approx(sum(c.cost_usd for c in run.calls()))
    from swarm_arena.llm._curator import CURATOR_SCHEMA

    assert fake.calls[0]["output_schema"].json_schema() == CURATOR_SCHEMA


@pytest.mark.parametrize("count", [0, 1])
async def test_fewer_than_two_learnings_makes_no_call(count: int) -> None:
    fake = FakeModel([])
    fake.register("cur")
    assert await LLMCurator(model="cur").curate("physika", [learning(1)] * count, view()) == []
    assert fake.calls == []


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        ([group([0])], []),
        ([group(list(range(9)))], []),
        ([group([0, 7])], []),
        ([group([0, 0])], []),
        ([group([-1, 0])], []),
        ([group([0, 1]), group([1, 2])], [["L1", "L2"]]),
        ([group([0, 7]), group([0, 2])], [["L1", "L3"]]),
    ],
)
async def test_invalid_groups_are_dropped(
    groups: list[dict[str, Any]], expected: list[list[str]], tmp_path: Path
) -> None:
    FakeModel([turn(groups)]).register("cur")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        with llm_scope(run.scope("curator")):
            result = await LLMCurator(model="cur").curate(
                "physika", [learning(i) for i in range(1, 4)], view()
            )
    assert [s.supersedes for s in result] == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"title": "a" * 121},
        {"title": " " * 120 + "a"},
        {"title": ""},
        {"title": "  "},
        {"title": "a\nb"},
        {"title": "a\u2028b"},
        {"title": "a\x00"},
        {"text": "x" * 1501},
    ],
)
async def test_summary_over_limits_is_skipped(changes: dict[str, str], tmp_path: Path) -> None:
    FakeModel([turn([group([0, 1], **changes)])]).register("cur")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        with llm_scope(run.scope("curator")):
            assert (
                await LLMCurator(model="cur").curate("physika", [learning(1), learning(2)], view())
                == []
            )


@pytest.mark.parametrize(
    "raw",
    [
        '{"groups":[{"members":[true,1],"title":"SECRET","text":"x"}]}',
        '{"groups":[{"members":[[0],1],"title":"SECRET","text":"x"}]}',
        '{"groups":{}}',
        '{"groups":[],"extra":{"secret":NaN}}',
        "SECRET",
    ],
)
async def test_malformed_nested_output_is_value_free_and_drained(raw: str, tmp_path: Path) -> None:
    FakeModel([FakeTurn(text=raw)]).register("cur")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        scope = run.scope("curator")
        with llm_scope(scope), pytest.raises(ValueError, match=r".") as caught:
            await LLMCurator(model="cur").curate("physika", [learning(1), learning(2)], view())
        assert "SECRET" not in "".join(traceback.format_exception(caught.value))
        assert scope.tranche.pending() == 0
        assert scope.call_ids


async def test_evidence_free_group_does_not_discard_later_valid_group(tmp_path: Path) -> None:
    FakeModel([turn([group([0, 1]), group([1, 2])])]).register("cur")
    items = [
        learning(0, author="human:a", evidence=[]),
        learning(1, author="human:a", evidence=[]),
        learning(2),
    ]
    async with open_llm_run(tmp_path, config=configuration()) as run:
        with llm_scope(run.scope("curator")):
            (summary,) = await LLMCurator(model="cur").curate("physika", items, view())
    assert summary.supersedes == ["L1", "L2"]
    assert summary.evidence == ["c-2"]


async def test_learning_text_is_marked_untrusted(tmp_path: Path) -> None:
    fake = FakeModel([turn([])])
    fake.register("cur")
    fixed = view()
    async with open_llm_run(tmp_path, config=configuration()) as run:
        with llm_scope(run.scope("curator")):
            await LLMCurator(model="cur").curate(
                "physika", [learning(1, text=UNTRUSTED_CLOSE), learning(2)], fixed
            )
    prompt = fake.calls[0]["input"]
    assert prompt.count(UNTRUSTED_OPEN) == prompt.count(UNTRUSTED_CLOSE) == 1
    assert prompt.endswith(UNTRUSTED_CLOSE)
    fixed.learnings.assert_not_called()


async def test_model_needs_curator_role(tmp_path: Path) -> None:
    fake = FakeModel([])
    fake.register("cur")
    config = configuration()
    config.models.root["cur"] = config.models.root["cur"].model_copy(update={"roles": ["judge"]})
    async with open_llm_run(tmp_path, config=config) as run:
        with llm_scope(run.scope("curator")), pytest.raises(ValueError, match="curator"):
            await LLMCurator(model="cur").curate("physika", [learning(1), learning(2)], view())
    assert fake.calls == []


@pytest.mark.parametrize("delay_s", [0.05, None])
async def test_curator_drains_before_it_returns(delay_s: float | None, tmp_path: Path) -> None:
    fake = LateFakeModel([turn([group([0, 1])])], delay_s=delay_s)
    fake.register("cur")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        scope = run.scope("curator")
        try:
            with llm_scope(scope):
                (summary,) = await LLMCurator(model="cur").curate(
                    "physika", [learning(1), learning(2)], view()
                )
            assert scope.call_ids
            assert set(scope.call_ids) <= {c.id for c in run.calls()}
            assert summary.posted_by_call == scope.call_ids[-1]
            assert delay_s is not None or (run.calls()[0].error or "").startswith("estimated:")
            assert scope.tranche.pending() == 0
            assert run.charged(scope).usd == pytest.approx(sum(c.cost_usd for c in run.calls()))
        finally:
            await fake.aclose()


async def test_two_invocations_share_parent_without_draining_each_other(tmp_path: Path) -> None:
    fake = FakeModel([turn([group([0, 1])]), turn([group([0, 1])])])
    fake.register("cur")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        scope = run.scope("curator")
        with llm_scope(scope):
            for _ in range(2):
                await LLMCurator(model="cur").curate("physika", [learning(1), learning(2)], view())
        assert len(scope.call_ids) == len(run.calls()) == 2


async def test_refused_reservation_propagates_unchanged(tmp_path: Path) -> None:
    fake = FakeModel([])
    fake.register("cur")
    config = configuration().model_copy(update={"limits": Limits(max_tokens=1)})
    async with open_llm_run(tmp_path, config=config) as run:
        with llm_scope(run.scope("curator")), pytest.raises(BudgetExceeded):
            await LLMCurator(model="cur").curate("physika", [learning(1), learning(2)], view())
        assert run.calls() == []
    assert fake.calls == []


async def test_cancellation_drains_and_late_callback_cannot_publish(tmp_path: Path) -> None:
    release = anyio.Event()
    fake = LateFakeModel([turn([group([0, 1])])], delay_s=0, release=release)
    fake.register("cur")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        scope = run.scope("curator")
        task: asyncio.Task[list[Learning]] | None = None
        try:
            with llm_scope(scope):
                task = asyncio.create_task(
                    LLMCurator(model="cur").curate("physika", [learning(1), learning(2)], view())
                )
                with anyio.fail_after(10):
                    await fake.ready.wait()
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
            assert scope.tranche.pending() == 0
            assert len(run.calls()) == 1
            assert (run.calls()[0].error or "").startswith("estimated:")
            release.set()
            await fake.flush()
            assert len(run.calls()) == 1
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await fake.aclose()


async def test_engine_publishes_drained_call_link_atomically(
    env: EngineEnv,
) -> None:
    from _engine.test_goal import _curation_checkpoint

    from swarm_arena._journal.event import EventDraft, EventType
    from swarm_arena._journal.llm_record import _bound_run
    from swarm_arena._journal.replay import RunState

    fake = LateFakeModel([turn([group([0, 1])])], delay_s=0.05)
    fake.register("cur")
    runner = await _curation_checkpoint(env)
    second = learning(2, namespace="n", run_id=env.deps.run_id)
    await env.deps.journal.append_durable(
        EventDraft(
            type=EventType.LEARNING,
            goal_id="g",
            round=1,
            payload={"learning": second.model_dump(mode="json")},
        )
    )
    env.deps.config = env.deps.config.model_copy(
        update={
            "curator": Ref.of(LLMCurator(model="cur")),
            "models": configuration().models,
            "limits": configuration().limits,
        }
    )
    try:
        async with _bound_run(
            env.deps.run_id,
            journal=env.deps.journal,
            ids=env.deps.ids,
            artifacts=env.deps.artifacts,
            config=env.deps.config,
            stops=env.deps.stops,
            clock=env.deps.clock,
            state=RunState.from_events(env._events()),
        ):
            await runner._curate_round(1)
            await runner._curate_round(1)
        (receipt,) = env.events("curated")
        (call,) = env.events("llm_call")
        summary_event = next(
            e for e in env.events("learning") if e.payload["learning"]["author"] == "curator"
        )
        summary = Learning.model_validate(summary_event.payload["learning"])
        assert summary.posted_by_call == call.payload["call"]["id"]
        assert call.seq < summary_event.seq < receipt.seq
        assert receipt.payload["summaries"] == [summary.id]
        assert env.segment_index(summary_event.seq) == env.segment_index(receipt.seq)
        assert summary.supersedes == ["L-source", "L2"]
        assert summary.evidence == ["c-1", "c-2"]
        assert all(
            e.payload["reservation"]["holder"] == "g/:curator" for e in env.events("reserve")
        )
        state = RunState.from_events(env._events())
        assert state.learnings[summary.id].posted_by_call == summary.posted_by_call
    finally:
        await fake.aclose()


async def test_two_stage_links_the_formatting_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena.llm import _structured

    first = FakeModel([FakeTurn(text="Combine the evidence")])
    formatter = FakeModel([turn([group([0, 1])])])
    first.register("cur")
    formatter.register("fmt")
    config = configuration()
    config.models.root["fmt"] = config.models.root["cur"].model_copy(update={"model": "fake/fmt"})
    monkeypatch.setattr(_structured, "supports", lambda spec, feature: spec.model == "fake/fmt")
    async with open_llm_run(tmp_path, config=config) as run:
        scope = run.scope("curator")
        with llm_scope(scope):
            (summary,) = await LLMCurator(model="cur").curate(
                "physika", [learning(1), learning(2)], view()
            )
        assert len(scope.call_ids) == 2
        assert summary.posted_by_call == scope.call_ids[-1]
        assert run.calls()[-1].model_alias == "fmt"
        assert run.charged(scope).usd == pytest.approx(sum(c.cost_usd for c in run.calls()))


async def test_core_waits_for_custom_curator_callback_before_linking(tmp_path: Path) -> None:
    from swarm_arena._board.curator import _prepare_curator
    from swarm_arena._budget.llm_scope import current_llm_scope
    from swarm_arena.llm._curator import CURATOR_SCHEMA
    from swarm_arena.llm._structured import structured_call

    fake = LateFakeModel([turn([])], delay_s=0.05)
    fake.register("cur")

    class Custom(Curator):
        async def curate(
            self, namespace: str, new: Sequence[Learning], view: BoardView
        ) -> list[Learning]:
            scope = current_llm_scope()
            assert scope is not None
            _, ids = await structured_call(
                alias="cur",
                role="curator",
                models=scope.models,
                namespace=namespace,
                system="merge",
                user="data",
                schema=CURATOR_SCHEMA,
                name="curator",
                no_cache_read=False,
                temperature=None,
            )
            return [learning(3, supersedes=[item.id for item in new], posted_by_call=ids[-1])]

    async with open_llm_run(tmp_path, config=configuration()) as run:
        scope = run.scope("curator")
        try:
            with llm_scope(scope):
                (summary,), errors = await _prepare_curator(
                    Custom(),
                    namespace="physika",
                    run_id=run.run_id,
                    goal_id="g",
                    round=1,
                    new=[learning(1), learning(2)],
                    view=view(),
                )
            assert not errors
            assert summary.posted_by_call == scope.call_ids[-1]
            assert fake.recorded.is_set()
            assert scope.tranche.pending() == 0
        finally:
            await fake.aclose()


@pytest.mark.parametrize(
    ("size", "max_group", "accepted"), [(8, 8, True), (9, 8, False), (9, 9, True), (3, 2, False)]
)
async def test_max_group_bounds_valid_indices(
    size: int, max_group: int, accepted: bool, tmp_path: Path
) -> None:
    FakeModel([turn([group(list(range(size)), title="t" * 120, text="x" * 1500)])]).register("cur")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        with llm_scope(run.scope("curator")):
            result = await LLMCurator(model="cur", max_group=max_group).curate(
                "physika", [learning(i) for i in range(size)], view()
            )
    assert len(result) == int(accepted)


async def test_identifier_subclasses_do_not_run_comparison_hooks(tmp_path: Path) -> None:
    class Identifier(str):
        def __hash__(self) -> int:
            raise AssertionError("untrusted hash")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("untrusted comparison")

        def __str__(self) -> str:
            raise AssertionError("untrusted string conversion")

    FakeModel([turn([group([0, 1])])]).register("cur")
    items = [learning(1).model_copy(update={"id": Identifier("L1")}), learning(2)]
    async with open_llm_run(tmp_path, config=configuration()) as run:
        with llm_scope(run.scope("curator")):
            (summary,) = await LLMCurator(model=Identifier("cur")).curate(
                Identifier("physika"), items, view()
            )
    assert summary.supersedes == ["L1", "L2"]
    assert type(summary.namespace) is str
