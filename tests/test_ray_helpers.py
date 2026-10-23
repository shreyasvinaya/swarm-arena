"""Keep shared journal decoding and Ray acceptance boundaries observable."""

import json
from pathlib import Path

import pytest
from _fakes.dependencies import ray_dependency
from _fakes.journals import committed_events
from _fakes.ray_executor import tool_job
from _fakes.ray_transport import _without_driver_files


def test_committed_events_follow_only_the_published_chain(tmp_path: Path) -> None:
    (tmp_path / "control").mkdir()
    (tmp_path / "control/owner.json").write_text(json.dumps({"head": {"segment": "second"}}))
    for name, link, rows in [
        ("first", None, [{"seq": 1}, {"seq": 2}]),
        ("second", "first", [{"seq": 3}]),
        ("orphan", "second", [{"seq": 4}]),
    ]:
        (tmp_path / name).write_text("\n".join(json.dumps(row) for row in [{"link": link}, *rows]))
    assert committed_events(tmp_path) == [{"seq": 1}, {"seq": 2}, {"seq": 3}]


def test_denial_keeps_the_production_worker_error_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ray_dependency()
    from swarm_arena.ray import _task
    from swarm_arena.ray._files import MIB, unframe

    roots = (tmp_path / "driver-source", tmp_path / "plan-directory")
    artifacts = tmp_path / "driver-artifacts"
    for root in (*roots, artifacts):
        root.mkdir()
        (root / "input.txt").write_text("driver-only")
    payload = _task.job_payload(
        tool_job("", artifact_root=str(artifacts)), _task.TaskSettings(cap_bytes=MIB)
    )

    def failing(frame: bytes) -> bytes:
        for root in (*roots, artifacts):
            with pytest.raises(PermissionError, match="test filesystem boundary"):
                (root / "input.txt").read_bytes()
        raise ValueError("private-worker-detail")

    monkeypatch.setattr(_task, "_run_frame", failing)
    output = unframe(_without_driver_files(payload, driver_roots=roots))
    result = json.loads(output.head)
    assert result["status"] == "error"
    assert result["data"] == {"swarm_arena_error": "worker failed: ValueError"}
    assert output.blobs == {}
