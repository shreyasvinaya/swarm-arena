"""Storage admission and receipt recovery use the board's arena root."""

import csv
import io
from pathlib import Path
from unittest.mock import Mock

import anyio
import pytest
from _fakes.run import all_events, plan

from swarm_arena import Ref, RunConfig, Swarm
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._engine import preflight
from swarm_arena._errors import ConfigError
from swarm_arena.scoring import HumanReview


@pytest.mark.parametrize("board_remote", [False, True])
def test_storage_preflight_uses_effective_arena(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, board_remote: bool
) -> None:
    remote = "s3://private-arena-location/board"
    local = run_config.arena_root
    config = run_config.model_copy(
        update={
            "arena_root": local if board_remote else remote,
            "board": Ref(
                path="swarm_arena.board:ArenaBoard",
                config={
                    "arena_root": remote if board_remote else local,
                },
            ),
        }
    )
    opener = Mock()
    monkeypatch.setattr(preflight, "open_store", opener)
    if board_remote:
        with pytest.raises(ConfigError, match="local run_root") as caught:
            preflight.check_storage(config)
        assert remote not in str(caught.value)
    else:
        preflight.check_storage(config)
    opener.assert_not_called()


async def test_resume_repairs_receipt_in_configured_board_arena(
    tmp_path: Path, run_config: RunConfig
) -> None:
    arena = tmp_path / "shared-arena"
    config = run_config.model_copy(
        update={
            "board": Ref(
                path="swarm_arena.board:ArenaBoard",
                config={"arena_root": str(arena)},
            )
        }
    )
    current = plan(rounds=1)
    await Swarm(config).run(current, run_id="review-root")
    out = await HumanReview.export("review-root", config=config, out_dir=tmp_path / "review")
    path = out / "review.csv"
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    assert rows
    rows[0].update(verdict="accept", score="7")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    await HumanReview("alice").import_csv("review-root", path, config=config)
    receipt = next(event for event in all_events(config) if event.type == "review_imported")
    originals = {key: (arena / key).read_bytes() for key in receipt.payload["objects"]}
    for key in originals:
        (arena / key).unlink()
    await Swarm(config).resume("review-root", force=True)
    assert {key: (arena / key).read_bytes() for key in originals} == originals
    assert not await anyio.to_thread.run_sync(Path(config.arena_root).exists)
    board = ArenaBoard(arena_root=str(arena), cache_dir=str(tmp_path / "reader"))
    board.open("", "", ["n"])
    try:
        board.frontier()
        assert board.leaderboard(current.goal.key(), judge="human:alice")
    finally:
        board.close()
