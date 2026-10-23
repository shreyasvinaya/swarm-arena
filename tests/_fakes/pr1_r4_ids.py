"""Real driver for blocked returned-score scalar operations."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import anyio
from _engine.test_pr1_fm_outcomes import config
from _fakes.run import plan

from swarm_arena import Goal, Ref, Score, Swarm
from swarm_arena.board import Candidate
from swarm_arena.scoring import Judge, JudgeContext
from swarm_arena.tool import Estimate

ROOT: Path | None = None
LOOP: int = 0
MODE: str = "eq"
NONFINITE: bool = False


def block() -> None:
    assert ROOT is not None
    root = ROOT
    (root / "entered.json").write_text(json.dumps({"loop": LOOP, "hook": threading.get_ident()}))
    released = threading.Event()

    def release() -> None:
        deadline = time.monotonic() + 300  # backstop only; the test writes `release`
        while not (root / "release").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        released.set()

    watcher = threading.Thread(target=release, daemon=True)
    watcher.start()
    assert released.wait(360), "release watcher did not finish"
    watcher.join(60)
    assert not watcher.is_alive()


class Identifier(str):
    def __eq__(self, other: object) -> bool:
        if MODE == "eq" and threading.get_ident() == LOOP:
            block()
        return str.__eq__(self, other)

    def __hash__(self) -> int:
        if MODE == "hash":
            block()
        return str.__hash__(self)


class Identifiers(Judge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if ctx.final:
            return [Score(candidate_id=c.id, key=(1.0,)) for c in candidates]
        assert ROOT is not None
        scores = [
            Score.maximize(1).model_copy(
                update={
                    "candidate_id": Identifier(c.id),
                    "key": (float("nan"),) if NONFINITE else (1.0,),
                }
            )
            for c in candidates
        ]
        (ROOT / "returned").write_text("ready")
        return scores


class Waits(Judge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        assert ROOT is not None
        (ROOT / "next-judge").write_text("ready")
        await anyio.sleep_forever()
        return []


async def main() -> None:
    from _fakes import pr1_r4_ids as module

    root = Path(sys.argv[1])
    module.ROOT, module.LOOP = root, threading.get_ident()
    module.MODE, module.NONFINITE = sys.argv[2], sys.argv[3] == "nonfinite"
    cfg = config(root)
    cfg = cfg.model_copy(
        update={
            # This experiment deliberately holds the judge/scalar through stop.
            "settle_timeout_s": 2.0,
            # Keep the owner live while entry/exit events supervise the public
            # stop request; whole-run CI leases must be at least three seconds.
            "lease_ttl_s": 3.0,
            "limits": cfg.limits.model_copy(
                update={"default_run_reservation": Estimate(tokens=7, usd=0.000007)}
            ),
        }
    )
    current = plan(rounds=1).model_copy(
        update={
            "judges": [
                Ref.of(module.Identifiers()).model_copy(update={"name": "y"}),
                Ref.of(module.Waits()).model_copy(update={"name": "waiting"}),
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
        result = await Swarm(cfg).run(current, run_id="r")
        (root / "done.json").write_text(json.dumps(list(result)))
        group.cancel_scope.cancel()


if __name__ == "__main__":
    anyio.run(main)
