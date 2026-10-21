"""Reuse a paid final batch after process death at its committed score/settle head."""

import os
import sys
from collections.abc import Sequence
from pathlib import Path

import anyio
import pytest

from swarm_arena import Goal, Plan, Score, Swarm
from swarm_arena._engine.control import read_status
from swarm_arena._journal.reader import read_committed
from swarm_arena._store.local import LocalStore
from swarm_arena.board import Candidate
from swarm_arena.scoring import JudgeContext
from tests._engine.test_barrier_recovery import (
    MeteredJudge,
    _calls,
    _charges,
    _config,
    _history,
    _plan,
)


class FinalFailureJudge(MeteredJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        scores = await super().judge(goal, candidates, ctx)
        if ctx.final:
            raise ValueError("final provider failed")
        return scores


class FinalFlipJudge(MeteredJudge):
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        scores = await super().judge(goal, candidates, ctx)
        return (
            [score.model_copy(update={"key": tuple(-v for v in score.key)}) for score in scores]
            if ctx.final
            else scores
        )


def final_plan(root: Path, failed: bool, flip: bool = False) -> Plan:
    current = _plan(root, 2, 1, False)
    if failed or flip:
        current = current.model_copy(
            update={
                "judges": [
                    current.judges[0].model_copy(
                        update={
                            "path": "_engine.test_pr1_r2_final_recovery:"
                            + ("FinalFailureJudge" if failed else "FinalFlipJudge")
                        }
                    ),
                    *current.judges[1:],
                ]
            }
        )
    return current


async def final_crash_driver(root: Path, failed: bool, flip: bool = False) -> None:
    await anyio.Path(root).mkdir(parents=True, exist_ok=True)
    original = os.replace

    def replace(
        src: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        dst: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        original(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        if not str(dst).endswith("/control/owner.json"):
            return
        history = read_committed(LocalStore(root / "runs" / "paid"))
        if history is None:
            return
        events = history[1]
        reservations = {
            e.payload["reservation"]["id"]
            for e in events
            if e.type == "reserve" and e.payload["reservation"]["kind"] == "final"
        }
        final_score = any(e.type in {"score", "judge_error"} and e.round == 0 for e in events)
        settled = any(e.type == "settle" and e.payload["id"] in reservations for e in events)
        selected = any(e.type == "final_order" for e in events)
        if final_score and settled and not selected:
            os._exit(91)

    patch = pytest.MonkeyPatch()
    patch.setattr(os, "replace", replace)
    try:
        await Swarm(_config(root)).run(final_plan(root, failed, flip), run_id="paid")
    finally:
        patch.undo()


@pytest.mark.parametrize(("failed", "flip"), [(False, False), (True, False), (False, True)])
async def test_resume_reuses_committed_final_scores_and_order(
    tmp_path: Path, failed: bool, flip: bool
) -> None:
    control, crashed = tmp_path / "control", tmp_path / "crashed"
    control.mkdir()
    expected = await Swarm(_config(control)).run(final_plan(control, failed, flip), run_id="paid")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
    )
    code = (
        "import anyio,sys; from pathlib import Path; "
        "from _engine.test_pr1_r2_final_recovery import final_crash_driver; "
        "anyio.run(final_crash_driver, Path(sys.argv[1]), "
        "sys.argv[2] == 'True', sys.argv[3] == 'True')"
    )
    with anyio.fail_after(30):
        process = await anyio.run_process(
            [sys.executable, "-c", code, str(crashed), str(failed), str(flip)],
            env=environment,
            check=False,
        )
    assert process.returncode == 91, process.stderr.decode()
    before = _history(crashed)
    assert any(e.type == ("judge_error" if failed else "score") and e.round == 0 for e in before)
    assert not any(e.type == "final_order" for e in before)
    if flip:
        assert [c.output.data["n"] for c in expected["g"]] == [5, 6]
    before_calls = _calls(crashed)
    assert sum(c["final"] for c in before_calls) == 1
    completed = [e for e in before if "final_batch" in e.payload]
    assert len(completed) == 1
    record = completed[0].payload["final_batch"]
    lifecycle = [e for e in before if e.type == "goal_started"][-1]
    assert record["lifecycle"] == [lifecycle.epoch, lifecycle.seq]
    assert record["revision"] == 0
    assert record["members"] == sorted(next(c["ids"] for c in before_calls if c["final"]))
    assert record["order"] == [c.id for c in expected["g"]]
    assert completed[0].seq == max(
        e.seq for e in before if e.round == 0 and e.type in {"score", "judge_error"}
    )
    recovered = await Swarm(_config(crashed)).resume("paid", force=True)
    calls = _calls(crashed)
    assert calls == before_calls, "a committed final judge batch must not be dispatched again"
    assert [(c["judge"], c["round"], c["final"]) for c in calls] == [
        (c["judge"], c["round"], c["final"]) for c in _calls(control)
    ]
    oracle = 18 + sum(7 * len(c["ids"]) for c in _calls(control))
    assert _charges(_history(crashed)) == _charges(_history(control)) == (oracle, oracle)
    status = await read_status(_config(crashed).run_root, "paid")
    assert (status.spent.tokens, status.spent.micro_usd) == (oracle, oracle)
    assert [c.id for c in recovered["g"]] == [c.id for c in expected["g"]]
    assert (crashed / "tools.log").read_text().splitlines() == ["3", "4", "5", "6"]
    final_order = [e.payload["candidate_ids"] for e in _history(crashed) if e.type == "final_order"]
    assert final_order == ([] if failed else [[c.id for c in expected["g"]]])
    if failed:
        assert [e.payload["reason"] for e in _history(crashed) if e.type == "no_result"] == [
            "provider_failed"
        ]
