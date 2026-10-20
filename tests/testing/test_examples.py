"""Keep the public judge example executable and meaningful."""

import doctest
import sys
from collections.abc import Sequence
from typing import Any

import pytest

import swarm_arena.testing as kit
from swarm_arena import Goal, Ref, Score
from swarm_arena.board import Candidate
from swarm_arena.scoring import Judge


def test_check_judge_example_scores_a_nonempty_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    original = kit.check_judge
    batches: list[list[Candidate]] = []
    modules: list[str] = []

    async def recording_judge(
        judge: Judge | Ref, goal: Goal, candidates: Sequence[Candidate], **kwargs: Any
    ) -> list[Score]:
        batches.append(list(candidates))
        if isinstance(judge, Judge):
            modules.append(judge.config.model_dump()["function"].split(":")[0])
        return await original(judge, goal, candidates, **kwargs)

    monkeypatch.setattr(kit, "check_judge", recording_judge)
    for _ in range(2):
        example = doctest.DocTestParser().get_doctest(
            original.__doc__ or "", {}, "check_judge", None, 0
        )
        assert not any(item.options.get(doctest.SKIP) for item in example.examples)
        runner = doctest.DocTestRunner()
        result = runner.run(example)
        assert result.failed == 0
    assert batches
    assert all(batches), "the example must score a nonempty batch"
    assert all(name not in sys.modules for name in modules), "remove the temporary scoring module"
