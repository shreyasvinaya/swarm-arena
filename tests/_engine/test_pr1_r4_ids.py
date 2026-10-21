"""Scalar fields retained outside detached scores are also core-owned."""

import json
import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import anyio
import pytest
from _engine.test_pr1_fm_outcomes import config
from _fakes.tools import GOAL, FixedJudge, RecordingExecutor, cand

from swarm_arena import Goal, Score, Swarm
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._scoring.runner import _prepare_judge_result
from swarm_arena.board import Candidate
from swarm_arena.scoring import JudgeContext
from swarm_arena.testing import check_judge
from swarm_arena.testing._contracts import _check_scores


@pytest.mark.parametrize(
    ("mode", "key"), [("eq", "finite"), ("eq", "nonfinite"), ("hash", "finite")]
)
async def test_returned_identifier_keeps_stop_heartbeat_and_renewal(
    tmp_path: Path, mode: str, key: str
) -> None:
    checkout = Path(__file__).parents[2]
    process = subprocess.Popen(  # noqa: ASYNC220 - isolates deliberately blocked event loop
        [sys.executable, "-B", "-m", "_fakes.pr1_r4_ids", str(tmp_path), mode, key],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(checkout / "tests"), str(checkout / "src")]),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    evidence = {}
    try:
        with anyio.fail_after(15):
            while not ((tmp_path / "entered.json").exists() or (tmp_path / "next-judge").exists()):
                assert process.poll() is None, process.communicate()
                await anyio.sleep(0.01)
        owner = tmp_path / "runs/r/control/owner.json"
        before = json.loads(owner.read_text())["expires_at"]
        heartbeat = (tmp_path / "heartbeat").read_text()
        # A 3 s lease renews about every second; allow several renewal periods.
        with anyio.move_on_after(5):
            while (  # noqa: ASYNC110 - observes another process
                json.loads(owner.read_text())["expires_at"] == before
                or (tmp_path / "heartbeat").read_text() == heartbeat
            ):
                await anyio.sleep(0.01)
        evidence = {
            "renewed": json.loads(owner.read_text())["expires_at"] != before,
            "heartbeat": (tmp_path / "heartbeat").read_text() != heartbeat,
        }
        swarm = Swarm(config(tmp_path))
        swarm.run_id = "r"
        await swarm.stop(now=True)
        # Stop files are polled every 2 s; the hook stays blocked until `release`.
        with anyio.move_on_after(10):
            while not (tmp_path / "done.json").exists():  # noqa: ASYNC110 - observes another process
                await anyio.sleep(0.01)
        evidence["done_before_release"] = (tmp_path / "done.json").exists()
        evidence["stop_requested"] = (tmp_path / "runs/r/control/stop-now-e1").exists()
        assert all(evidence.values()), evidence
        if mode == "eq":
            assert not (tmp_path / "entered.json").exists()
            assert (tmp_path / "next-judge").exists()
        else:
            entered = json.loads((tmp_path / "entered.json").read_text())
            assert entered["hook"] != entered["loop"]
    finally:
        (tmp_path / "evidence.json").write_text(json.dumps(evidence))
        (tmp_path / "release").write_text("release")
        try:
            stdout, stderr = await anyio.to_thread.run_sync(lambda: process.communicate(timeout=8))
            (tmp_path / "driver-output.txt").write_text(stdout + stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    assert process.returncode == 0, stderr


class Identifier(str):
    __hash__ = str.__hash__


async def test_preparation_returns_only_canonical_batch_ids() -> None:
    good, bad = cand(id="good"), cand(id="bad")
    judge = FixedJudge(FixedJudge.Config(value=1))
    scores = [
        Score(candidate_id="good", key=(1.0,)).model_copy(
            update={"candidate_id": Identifier("good")}
        ),
        Score(candidate_id="good", key=(2.0,)),
        Score(candidate_id="foreign", key=(3.0,)),
        Score(candidate_id="bad", key=(4.0,)).model_copy(
            update={"candidate_id": Identifier("bad"), "key": ()}
        ),
    ]
    prepared, seen, nonfinite = await run_sync_daemon(
        _prepare_judge_result, scores, {good.id, bad.id}, set(), judge, "fixed", "batch"
    )
    assert [s.key for s in prepared] == [(1.0,)]
    assert seen == {"good", "bad"}
    assert nonfinite == {"bad"}
    assert all(type(key) is str for key in seen | nonfinite)
    ctx = JudgeContext(goal=GOAL, executor=RecordingExecutor(), artifact_root="/unused")
    valid, scored = await run_sync_daemon(_check_scores, [scores[0], scores[3]], [good, bad], ctx)
    assert [s.candidate_id for s in valid] == ["good"]
    assert scored == {"good", "bad"}
    assert all(type(key) is str for key in scored | ctx.errors.keys())


@pytest.mark.parametrize("nonfinite", [False, True])
async def test_public_conformance_keeps_scalar_hooks_off_loop(
    monkeypatch: pytest.MonkeyPatch, nonfinite: bool
) -> None:
    loop = threading.get_ident()
    calls: list[int] = []

    class Tracked(str):
        def __eq__(self, other: object) -> bool:
            calls.append(threading.get_ident())
            return str.__eq__(self, other)

        def __hash__(self) -> int:
            calls.append(threading.get_ident())
            return str.__hash__(self)

    async def judge(
        self: FixedJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            Score.maximize(1).model_copy(
                update={"candidate_id": Tracked(c.id), "key": () if nonfinite else (1.0,)}
            )
            for c in candidates
        ]

    monkeypatch.setattr(FixedJudge, "judge", judge)
    if nonfinite:
        with pytest.warns(UserWarning, match="score key must be nonempty and finite"):
            assert (
                await check_judge(
                    FixedJudge(FixedJudge.Config(value=1)),
                    GOAL,
                    [cand()],
                    executor=RecordingExecutor(),
                )
                == []
            )
    else:
        [score] = await check_judge(
            FixedJudge(FixedJudge.Config(value=1)), GOAL, [cand()], executor=RecordingExecutor()
        )
        assert type(score.candidate_id) is str
    assert calls
    assert loop not in calls
