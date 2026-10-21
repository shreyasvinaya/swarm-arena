"""Active driver time survives crashes within rounds and terminal phases."""

import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio
import pytest
from _engine.test_goal import _plan
from _fakes.engine import with_executor
from _fakes.run import all_events
from _fakes.tools import ProviderJudge

from swarm_arena import Goal, Ref, RunConfig, Score, ScriptedManager, Swarm
from swarm_arena._budget.llm_scope import current_llm_scope
from swarm_arena._engine.stops import RunClock
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena.board import Candidate
from swarm_arena.manager import AgentContext, RoundReport
from swarm_arena.scoring import JudgeContext

_MODULE = "_engine.test_clock_checkpoints"


class SlowManager(ScriptedManager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        for x in (1.0, 2.0, 3.0):
            await anyio.sleep(0.15)
            await ctx.run_tool({"x": x})
        return RoundReport()


async def clock_driver(root: Path, phase: str) -> None:
    cfg = with_executor(
        RunConfig(
            run_root=str(root / "runs"),
            arena_root=str(root / "arena"),
            artifact_root=str(root / "artifacts"),
        ),
        workspace_root=str(root / "ws"),
    )
    p = _plan(
        teams=1,
        rounds=1,
        converge_team=phase == "converge",
        judges=[Ref(path="_fakes.tools:ProviderJudge", name="provider")],
        ranking="provider",
    )
    if phase == "round":
        p.teams[0].agents[0] = (
            p.teams[0]
            .agents[0]
            .model_copy(
                update={"manager": Ref(path=f"{_MODULE}:SlowManager", config={"requests": []})}
            )
        )
    publish, construct, judge = JournalWriter._publish, ProviderJudge.__init__, ProviderJudge.judge
    completions = 0

    def init(self: ProviderJudge, *args: Any, **kwargs: Any) -> None:
        scope = current_llm_scope()
        if phase == "final" and scope and scope.force_no_cache_read:
            time.sleep(0.3)
        construct(self, *args, **kwargs)

    async def score(
        self: ProviderJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if phase == "converge" and ctx.round == 2:
            await anyio.sleep(0.3)
        return await judge(self, goal, candidates, ctx)

    async def commit(self: JournalWriter) -> None:
        nonlocal completions
        self._group_window_s = 0.001
        completions += sum(e.type == "tool_end" for e in self._queue)
        hit = (phase == "round" and completions >= 3) or any(
            (phase == "converge" and e.type == "score" and e.round == 2)
            or (
                phase == "final"
                and e.type == "reserve"
                and e.payload["reservation"]["kind"] == "final"
            )
            for e in self._queue
        )
        await publish(self)
        if hit:
            os._exit(91)

    JournalWriter._publish = commit
    ProviderJudge.__init__ = init
    ProviderJudge.judge = score
    await Swarm(cfg).run(p, run_id="clock")


@pytest.mark.parametrize("phase", ["round", "converge", "final"])
async def test_clock_checkpoint_survives_process_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from swarm_arena._engine import resume as resume_module

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
    )
    code = (
        f"import sys,anyio; from pathlib import Path; from {_MODULE} import clock_driver; "
        "anyio.run(clock_driver,Path(sys.argv[1]),sys.argv[2])"
    )
    child = await anyio.run_process(
        [sys.executable, "-c", code, str(tmp_path), phase], env=environment, check=False
    )
    assert child.returncode == 91, child.stderr.decode()
    cfg = RunConfig.load(tmp_path / "runs" / "clock" / "run.toml")
    before = all_events(cfg)
    checkpoint = RunState.from_events(before).run_elapsed_s
    start = next(e for e in before if e.type == "run_started")
    committed_work = (before[-1].ts - start.ts).total_seconds()
    assert checkpoint >= committed_work - 0.05, (checkpoint, committed_work)
    await anyio.sleep(0.2)
    seeds: list[float] = []

    class CapturedClock(RunClock):
        def __init__(self, elapsed_before: float) -> None:
            seeds.append(elapsed_before)
            super().__init__(elapsed_before)

    monkeypatch.setattr(resume_module, "RunClock", CapturedClock)
    await Swarm(cfg).resume("clock", force=True)
    assert seeds == [checkpoint]
    state = RunState.from_events(all_events(cfg))
    assert state.run_elapsed_s >= checkpoint
    assert state.goals["g"].finished
