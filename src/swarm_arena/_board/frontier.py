"""Build canonical board frontiers from committed journal heads."""

from collections.abc import Sequence

from swarm_arena._board.human_objects import human_object_order
from swarm_arena._board.ingest import _validate_marks
from swarm_arena._journal.lease import Head
from swarm_arena._model.board_types import Frontier


def build_frontier(
    board_reads: Sequence[tuple[str, str, Head]],
    reviews: Sequence[str],
    curation: Sequence[str],
) -> Frontier:
    """
    Build a frontier with sorted run marks and exact human-object names.

    Parameters
    ----------
    board_reads : Sequence[tuple[str, str, Head]]
        Run identifier, run root, and committed head for each read.
    reviews : Sequence[str]
        Exact included review object names.
    curation : Sequence[str]
        Exact included curation object names.

    Returns
    -------
    Frontier
        Independent canonical lists with identical marks and names deduplicated.

    Raises
    ------
    ChainError
        If two marks for one run disagree or a mark's metadata is invalid.

    Examples
    --------
    >>> build_frontier([], ["b", "a"], []).reviews
    ['a', 'b']
    """
    return Frontier(
        marks=_validate_marks(
            [
                Frontier.Mark(
                    run_id=run_id,
                    run_root=root,
                    epoch=head.epoch,
                    seq=head.seq,
                    segment=head.segment,
                )
                for run_id, root, head in sorted(
                    board_reads,
                    key=lambda item: (
                        item[0],
                        item[1],
                        item[2].epoch,
                        item[2].seq,
                        item[2].segment,
                    ),
                )
            ]
        ),
        reviews=sorted(set(reviews), key=human_object_order),
        curation=sorted(set(curation), key=human_object_order),
    )
