from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio
import pytest
from _engine.test_barrier_recovery import MeteredJudge, _calls, _charges, _config, _history, _plan

from swarm_arena import Goal, Plan, Score, Swarm
from swarm_arena.board import Candidate
from swarm_arena.scoring import JudgeContext


class ChangedFinalJudge(MeteredJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        scores = await super().judge(goal, candidates, ctx)
        if ctx.final and goal.inputs["scenario"] == "reverse":
            return [s.model_copy(update={"key": (-s.key[0],)}) for s in scores]
        if ctx.final and goal.inputs["scenario"] == "error":
            raise ValueError("final-private-value")
        return scores


def make_plan(root: Path, scenario: str, top: int) -> Plan:
    p = _plan(root, 2, 1, False)
    return p.model_copy(
        update={
            "goal": p.goal.model_copy(update={"inputs": {**p.goal.inputs, "scenario": scenario}}),
            "judges": [
                p.judges[0].model_copy(update={"path": "_fakes.pr1_r3_final:ChangedFinalJudge"}),
                p.judges[1],
            ],
            "converge": p.converge.model_copy(update={"top_n": top}),
        }
    )


async def main() -> None:
    mode, root_arg, scenario, top_arg, cutoff_arg = sys.argv[1:]
    root, top, cutoff = Path(root_arg), int(top_arg), int(cutoff_arg)
    await anyio.Path(root).mkdir(parents=True, exist_ok=True)
    original = os.replace

    def replace(src: Any, dst: Any, **kwargs: Any) -> None:
        original(src, dst, **kwargs)
        if not str(dst).endswith("/control/owner.json"):
            return
        events = _history(root)
        head = events[-1].seq if events else 0
        with (root / "heads.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"head": head, "types": [[e.seq, e.type] for e in events]}) + "\n"
            )
        if mode == "crash" and head >= cutoff:
            os._exit(91)

    patch = pytest.MonkeyPatch()
    patch.setattr(os, "replace", replace)
    cfg = _config(root)
    result = (
        await Swarm(cfg).resume("paid", force=True)
        if mode == "resume"
        else await Swarm(cfg).run(make_plan(root, scenario, top), run_id="paid")
    )
    events = _history(root)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "calls": _calls(root),
                "charges": _charges(events),
                "final": [c.output.data for c in result["g"]],
                "final_ids": [c.id for c in result["g"]],
                "orders": [e.payload for e in events if e.type == "final_order"],
                "reasons": [e.payload for e in events if e.type in {"goal_finished", "no_result"}],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    anyio.run(main)
