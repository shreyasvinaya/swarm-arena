"""Curator routing identifiers obey the shared context string boundary."""

from unittest.mock import Mock

from _board.test_curator import Returning, learning
from _fakes.pr1_r5_ids import hostile

from swarm_arena._board.curator import _prepare_curator
from swarm_arena.board import BoardView


async def test_curator_routing_uses_exact_strings() -> None:
    view = Mock(spec=BoardView)
    view.learnings.return_value = []
    summaries, errors = await _prepare_curator(
        Returning([learning(supersedes=["L1"])]),
        namespace=hostile("toy"),
        run_id=hostile("r1"),
        goal_id=hostile("g1"),
        round=1,
        new=[learning()],
        view=view,
    )
    assert errors == []
    assert len(summaries) == 1
    assert all(
        type(value) is str
        for value in [
            view.learnings.call_args.args[0],
            summaries[0].namespace,
            summaries[0].run_id,
            summaries[0].goal_id,
        ]
    )
