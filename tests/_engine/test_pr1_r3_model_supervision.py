from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
from _engine.test_pr1_fm_outcomes import config

from swarm_arena import Swarm
from swarm_arena.tool import Estimate

CHECKOUT = Path(__file__).parents[2]


@pytest.mark.anyio
@pytest.mark.parametrize("cause", ["stop", "takeover"])
async def test_program_score_serialization_is_supervised(tmp_path: Path, cause: str) -> None:
    process = subprocess.Popen(  # noqa: ASYNC220 - child isolates a deliberately blocked loop
        [sys.executable, "-B", "-m", "_fakes.pr1_r3_models", str(tmp_path)],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(CHECKOUT / "tests"), str(CHECKOUT / "src")]),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with anyio.fail_after(10):
            while not (tmp_path / "dump-entered").exists():
                assert process.poll() is None, process.communicate()
                await anyio.sleep(0.01)
        thread_ids = json.loads((tmp_path / "dump-entered").read_text())
        before = (tmp_path / "heartbeat").read_text()
        swarm = Swarm(config(tmp_path))
        swarm.run_id = "r"
        if cause == "stop":
            await swarm.stop(now=True)
        else:
            cfg = config(tmp_path)
            cfg = cfg.model_copy(
                update={
                    "limits": cfg.limits.model_copy(
                        update={"default_run_reservation": Estimate(tokens=7, usd=0.000007)}
                    )
                }
            )
            with anyio.fail_after(10):
                await Swarm(cfg).resume("r", force=True)
        # Stop files are polled every 2 s and the hook stays blocked until `release`,
        # so a 10 s bound still proves the run ends before the serializer returns.
        with anyio.move_on_after(10):
            while not (  # noqa: ASYNC110 - observes a separate driver process
                (tmp_path / "done.json").exists() if cause == "stop" else process.poll() is not None
            ):
                await anyio.sleep(0.01)
        stopped = (
            (tmp_path / "done.json").exists() if cause == "stop" else process.poll() is not None
        )
        # The fake serializer releases itself after 12 s; completion must come first.
        serializer_returned = (tmp_path / "dump-exited").exists()
        after = (tmp_path / "heartbeat").read_text()
        evidence = {
            "cause": cause,
            "owner_epoch": json.loads((tmp_path / "runs/r/control/owner.json").read_text())[
                "epoch"
            ],
            **thread_ids,
            "heartbeat_before": before,
            "heartbeat_after": after,
            "finished_before_release": stopped,
            "stop_file_exists": (tmp_path / "runs/r/control/stop-now-e1").exists(),
            "serializer_returned_first": serializer_returned,
        }
        (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
        assert stopped, evidence
        assert not serializer_returned, evidence
        assert thread_ids["hook_thread"] != thread_ids["loop_thread"], evidence
        if cause == "takeover":
            assert evidence["owner_epoch"] == 2
        else:
            assert evidence["stop_file_exists"]
    finally:
        (tmp_path / "release").write_text("release")
        try:
            out, err = await anyio.to_thread.run_sync(lambda: process.communicate(timeout=8))
            (tmp_path / "driver-output.txt").write_text(out + err)
            if cause == "takeover":
                assert "Fenced" in err
            else:
                assert process.returncode == 0, out + err
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
