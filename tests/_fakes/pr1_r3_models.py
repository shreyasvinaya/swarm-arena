from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import anyio
from _engine.test_pr1_fm_outcomes import config
from _fakes.run import plan

from swarm_arena import Goal, ProgramJudge, Ref, Score, Swarm
from swarm_arena.board import Candidate
from swarm_arena.tool import Estimate

ROOT: Path | None = None
LOOP_THREAD: int | None = None


class BlockingScore(Score):
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        assert ROOT is not None
        if not (ROOT / "dump-exited").exists():
            (ROOT / "dump-entered").write_text(
                json.dumps({"loop_thread": LOOP_THREAD, "hook_thread": threading.get_ident()})
            )
            released = threading.Event()

            def release_on_file() -> None:
                assert ROOT is not None
                deadline = time.monotonic() + 60  # backstop only; the test writes `release`
                while not (ROOT / "release").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                released.set()

            watcher = threading.Thread(target=release_on_file, daemon=True)
            watcher.start()
            released.wait()
            watcher.join()
            (ROOT / "dump-exited").write_text("done")
        return super().model_dump(**kwargs)


def score(goal: Goal, candidate: Candidate) -> Score:
    return Score.maximize(1.0) if ROOT is None else BlockingScore.maximize(1.0)


async def main() -> None:
    from _fakes import pr1_r3_models as module

    root = Path(sys.argv[1])
    module.ROOT = root
    module.LOOP_THREAD = threading.get_ident()
    cfg = config(root)
    cfg = cfg.model_copy(
        update={
            "settle_timeout_s": 0.1,
            "lease_ttl_s": 3.0,
            "limits": cfg.limits.model_copy(
                update={"default_run_reservation": Estimate(tokens=7, usd=0.000007)}
            ),
        }
    )
    p = plan(rounds=1).model_copy(
        update={
            "judges": [
                Ref.of(ProgramJudge("_fakes.pr1_r3_models:score")).model_copy(update={"name": "y"})
            ]
        }
    )

    async def heartbeat() -> None:
        count = 0
        while True:
            (root / "heartbeat").write_text(str(count))
            count += 1
            await anyio.sleep(0.02)

    async with anyio.create_task_group() as group:
        group.start_soon(heartbeat)
        result = await Swarm(cfg).run(p, run_id="r")
        (root / "done.json").write_text(json.dumps({"result": [c.id for c in result["g"]]}))
        group.cancel_scope.cancel()


if __name__ == "__main__":
    anyio.run(main)
