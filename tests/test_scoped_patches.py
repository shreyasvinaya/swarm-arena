"""Crash-driver probes must restore library functions when called in-process."""

from functools import partial
from pathlib import Path
from typing import Any

import pytest
from _engine.test_clock_checkpoints import clock_driver
from _engine.test_extension_boundaries import _worker
from _engine.test_join_recovery import crash_driver as join_driver
from _fakes import curators
from _fakes.tools import ProviderJudge
from e2e.test_many_goals_parts import cross_namespace_crash

from swarm_arena import RunConfig, Swarm
from swarm_arena._engine import round as round_module
from swarm_arena._engine.goal import GoalRunner
from swarm_arena._journal.lease import Lease
from swarm_arena._journal.writer import JournalWriter


@pytest.mark.parametrize(
    "driver", ["clock", "extension", "join", "curator", "curator_join", "refusal", "parts"]
)
async def test_helpers_restore_library_patches_after_failure(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, driver: str
) -> None:
    targets = [
        (JournalWriter, "_publish"),
        (Lease, "renew"),
        (GoalRunner, "_close"),
        (ProviderJudge, "__init__"),
        (ProviderJudge, "judge"),
        (round_module, "anyio"),
    ]
    originals = [getattr(owner, name) for owner, name in targets]
    expected = {
        "clock": {0, 3, 4},
        "extension": {1, 5},
        "join": {0},
        "curator": {0},
        "curator_join": {0},
        "refusal": {0, 2},
        "parts": {0},
    }[driver]

    def fail(self: Swarm, *args: Any, **kwargs: Any) -> None:
        for index in expected:
            owner, name = targets[index]
            assert getattr(owner, name) is not originals[index]
        raise RuntimeError("driver failed")

    config_path = tmp_path / "run.toml"
    config_path.write_text(run_config.dumps())
    helpers = {
        "clock": partial(clock_driver, tmp_path, "round"),
        "extension": partial(_worker, "output-serializer", tmp_path),
        "join": partial(join_driver, tmp_path, "control"),
        "curator": partial(curators.crash_driver, str(tmp_path), "control"),
        "curator_join": partial(curators.join_crash_driver, str(tmp_path), "control"),
        "refusal": partial(curators.refusal_crash_driver, str(tmp_path), "control"),
        "parts": partial(cross_namespace_crash, str(config_path)),
    }
    monkeypatch.setattr(Swarm, "__init__", fail)
    with pytest.raises(RuntimeError, match="driver failed"):
        await helpers[driver]()
    for (owner, name), original in zip(targets, originals, strict=True):
        assert getattr(owner, name) is original, (driver, name)
