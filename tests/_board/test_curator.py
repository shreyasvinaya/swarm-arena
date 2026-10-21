from collections.abc import Sequence
from typing import Any
from unittest.mock import Mock

import pytest
from _fakes.curators import MergeAll

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
        await run(Returning(Fenced()), [])


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
