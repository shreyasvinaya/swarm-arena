from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import Mock

import pytest
from _engine.conftest import env as env
from _fakes.curators import MergeAll
from _fakes.engine import EngineEnv

from swarm_arena import Ref
from swarm_arena._board.curator import Curator, _prepare_curator, _stamp_summaries
from swarm_arena._errors import Fenced, _core_error
from swarm_arena._model.ids import IdGen
from swarm_arena.board import BoardView, Learning
from swarm_arena.manager import LimitReached


def learning(id: str = "L1", **changes: Any) -> Learning:
    return Learning.model_validate(
        {
            "id": id,
            "namespace": "toy",
            "run_id": "r1",
            "goal_id": "g1",
            "round": 1,
            "author": "agent",
            "title": "Observation",
            "text": "Evidence",
            "evidence": ["c-1"],
            **changes,
        }
    )


class Returning(Curator):
    def __init__(self, result: Any) -> None:
        super().__init__()
        self.result = result

    async def curate(
        self, namespace: str, new: Sequence[Learning], view: BoardView
    ) -> list[Learning]:
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


async def run(
    curator: Curator,
    new: Sequence[Learning],
    *,
    visible: Sequence[Learning] = (),
    ids: IdGen | None = None,
) -> tuple[list[Learning], list[str]]:
    view = Mock(spec=BoardView)
    view.learnings.return_value = list(visible)
    summaries, errors = await _prepare_curator(
        curator,
        namespace="toy",
        run_id="r1",
        goal_id="g1",
        round=1,
        new=new,
        view=view,
    )
    return _stamp_summaries(summaries, ids or IdGen("r1", 1)), errors


async def test_summary_supersedes_and_unions_evidence() -> None:
    l1 = learning(verified=True)
    l2 = learning("L2", evidence=["c-2", "c-1"])
    summaries, errors = await run(MergeAll(), [l1, l2])
    assert errors == []
    assert summaries[0].supersedes == ["L1", "L2"]
    assert summaries[0].evidence == ["c-1", "c-2"]
    assert summaries[0].author == "curator"
    assert summaries[0].id.startswith("L-")
    assert summaries[0].verified
    assert not summaries[0].hidden
    assert not summaries[0].pinned


@pytest.mark.parametrize("supersedes", [[], ["absent"], ["hidden"]])
async def test_unknown_supersedes_dropped_with_error(supersedes: list[str]) -> None:
    summaries, errors = await run(
        Returning([learning(supersedes=supersedes)]), [], visible=[learning("hidden", hidden=True)]
    )
    assert summaries == []
    assert errors


async def test_evidence_free_summary_is_dropped() -> None:
    summaries, errors = await run(MergeAll(), [learning(author="human:a", evidence=[])])
    assert summaries == []
    assert errors == ["summary has no evidence"]


async def test_curator_exception_is_skipped() -> None:
    assert await run(Returning(RuntimeError("SECRET")), []) == ([], ["curator error: RuntimeError"])


def test_curator_default_models_empty() -> None:
    assert MergeAll().models() == ()
    assert isinstance(Ref(path="_fakes.curators:MergeAll").build(Curator), MergeAll)
    assert MergeAll().config.model_dump() == {}


@pytest.mark.parametrize(
    "result", [None, "SECRET", [object()], [learning().model_copy(update={"title": "SECRET\n"})]]
)
async def test_malformed_batch_allocates_no_ids(result: Any) -> None:
    ids = IdGen("r1", 1)
    summaries, errors = await run(Returning(result), [learning()], ids=ids)
    assert not summaries
    assert errors
    assert "SECRET" not in repr(errors)
    assert ids.counters() == {}


async def test_core_metadata_overwrites_untrusted_fields() -> None:
    proposal = learning(
        supersedes=["L1"],
        author="human:fake",
        hidden=True,
        pinned=True,
        verified=False,
        evidence=["fake"],
        posted_by_call="fake",
    )
    summaries, errors = await run(Returning([proposal]), [learning(verified=True)])
    assert not errors
    assert summaries[0].evidence == ["c-1"]
    assert summaries[0].posted_by_call is None
    assert (
        summaries[0].namespace,
        summaries[0].run_id,
        summaries[0].goal_id,
        summaries[0].round,
    ) == ("toy", "r1", "g1", 1)


async def test_visible_previous_learning_can_be_superseded() -> None:
    summaries, errors = await run(
        Returning([learning(supersedes=["old"])]), [], visible=[learning("old")]
    )
    assert not errors
    assert summaries[0].supersedes == ["old"]


async def test_core_signal_is_rebuilt_and_spoof_is_contained() -> None:
    error = _core_error(LimitReached, "budget", "budget used up")
    error.args = ("SECRET",)
    error.add_note("SECRET")
    with pytest.raises(LimitReached, match="budget used up") as caught:
        await run(Returning(error), [])
    assert caught.value is not error
    assert "SECRET" not in repr(caught.value)
    assert await run(Returning(LimitReached("budget", "SECRET")), []) == (
        [],
        ["curator error: LimitReached"],
    )
    with pytest.raises(Fenced):
        await run(Returning(_core_error(Fenced, "run ownership lost")), [])
    assert await run(Returning(Fenced("SECRET")), []) == ([], ["curator error: Fenced"])


@pytest.mark.parametrize(
    "changes",
    [
        {"title": "SECRET\n"},
        {"text": "SECRET" * 300},
        {"tags": [object()]},
        {"supersedes": ["L1", 1]},
    ],
)
async def test_production_learning_validation_precedes_ids(changes: dict[str, Any]) -> None:
    ids = IdGen("r1", 1)
    proposal = learning(supersedes=["L1"]).model_copy(update=changes)
    summaries, errors = await run(Returning([proposal]), [learning()], ids=ids)
    assert not summaries
    assert errors
    assert ids.counters() == {}
    assert "SECRET" not in repr(errors)


async def test_full_batch_is_detached_before_any_id_is_allocated() -> None:
    ids = IdGen("r1", 1)
    valid = learning(supersedes=["L1"])
    broken = valid.model_copy(update={"title": "SECRET\n"})
    assert await run(Returning([valid, broken]), [learning()], ids=ids) == (
        [],
        ["curator error: ValidationError"],
    )
    assert ids.counters() == {}


@pytest.mark.parametrize("source", [learning(hidden=True), learning(namespace="foreign")])
async def test_hidden_or_foreign_new_learning_cannot_be_superseded(source: Learning) -> None:
    summaries, errors = await run(Returning([learning(supersedes=["L1"])]), [source])
    assert summaries == []
    assert errors == ["invalid summary supersedes"]


async def test_serializer_cannot_intercept_authoritative_metadata_assignment() -> None:
    class IgnoringUpdate(dict[str, Any]):
        def update(self, *args: Any, **kwargs: Any) -> None:
            pass

    class HostileLearning(Learning):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return IgnoringUpdate(super().model_dump(*args, **kwargs))

    raw = learning(
        supersedes=["L1"],
        author="human:fake",
        evidence=["fake"],
        pinned=True,
        hidden=True,
        verified=False,
    )
    proposal = HostileLearning.model_validate(raw.model_dump())
    summaries, errors = await run(Returning([proposal]), [learning(verified=True)])
    assert errors == []
    assert summaries[0].author == "curator"
    assert summaries[0].evidence == ["c-1"]
    assert summaries[0].verified
    assert not summaries[0].hidden
    assert not summaries[0].pinned


@pytest.mark.parametrize(
    ("link", "expected"), [(None, "llm-7"), ("llm-3", "llm-7"), ("llm-6", "llm-6")]
)
async def test_hook_stamps_empty_curated_ids(link: str | None, expected: str) -> None:
    from types import SimpleNamespace

    from swarm_arena._budget.llm_scope import LLMScope, llm_scope

    scope = SimpleNamespace(run_id="unbound", call_ids=[])

    class Linked(Returning):
        async def curate(
            self, namespace: str, new: Sequence[Learning], view: BoardView
        ) -> list[Learning]:
            scope.call_ids.extend(["llm-6", "llm-7"])
            return await super().curate(namespace, new, view)

    with llm_scope(cast(LLMScope, scope)):
        summaries, errors = await run(
            Linked([learning(supersedes=["L1"], posted_by_call=link)]), [learning()]
        )
    assert errors == []
    assert summaries[0].id.startswith("L-")
    assert summaries[0].posted_by_call == expected


async def test_terminal_cleanup_failure_is_not_a_curator_error() -> None:
    from swarm_arena._errors import _stop_cleanup_error

    with pytest.raises(RuntimeError, match="stop cleanup timed out"):
        await run(Returning(_stop_cleanup_error()), [])


@pytest.mark.parametrize("stage", ["hook", "close"])
async def test_engine_never_publishes_after_terminal_curator_failure(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from _engine.test_goal import _curation_checkpoint

    from swarm_arena._engine.goal import GoalRunner
    from swarm_arena._errors import _stop_cleanup_error

    runner = await _curation_checkpoint(env)

    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise _stop_cleanup_error()

    if stage == "hook":
        monkeypatch.setattr(MergeAll, "curate", fail)
    else:
        monkeypatch.setattr(GoalRunner, "_close", fail)
    with pytest.raises(RuntimeError, match="stop cleanup timed out"):
        await runner._curate_round(1)
    await env.deps.journal.commit()
    assert env.events("curated") == []
    assert all(event.payload["learning"]["author"] != "curator" for event in env.events("learning"))


async def test_engine_terminal_overflow_is_not_reserve_exhaustion(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _engine.test_goal import _curation_checkpoint

    from swarm_arena._errors import _SignalCategory

    runner = await _curation_checkpoint(env)

    async def overflow(*args: Any, **kwargs: Any) -> Any:
        raise _core_error(
            LimitReached, "usage_overflow", "usage overflow", _category=_SignalCategory.TERMINAL
        )

    monkeypatch.setattr(MergeAll, "curate", overflow)
    with pytest.raises(LimitReached) as caught:
        await runner._curate_round(1)
    assert caught.value.limit == "usage_overflow"
    await env.deps.journal.commit()
    assert env.events("curated") == env.events("reserve_exhausted") == []


async def test_engine_curator_drain_has_a_bound(
    env: EngineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio
    from _engine.test_goal import _curation_checkpoint

    from swarm_arena._engine import goal as goal_module

    runner = await _curation_checkpoint(env)
    env.deps.config = env.deps.config.model_copy(update={"settle_timeout_s": 0.05})
    entered = anyio.Event()
    exited = anyio.Event()

    async def held_drain(*args: Any, **kwargs: Any) -> None:
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            exited.set()

    monkeypatch.setattr(goal_module, "drain", held_drain)
    with anyio.fail_after(30), pytest.raises(RuntimeError, match="stop cleanup timed out"):
        await runner._curate_round(1)
    assert entered.is_set()
    assert exited.is_set()
    await env.deps.journal.commit()
    assert env.events("curated") == []
    assert all(event.payload["learning"]["author"] != "curator" for event in env.events("learning"))


async def test_prepare_curator_reraises_provider_failed() -> None:
    from swarm_arena._errors import ProviderFailed

    with pytest.raises(ProviderFailed):
        await run(Returning(_core_error(ProviderFailed, "anthropic")), [])
