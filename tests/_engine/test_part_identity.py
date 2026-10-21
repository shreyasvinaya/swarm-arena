"""Part names cannot merge evidence from different problem identities."""

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import anyio
import pytest
from _engine.test_goal import _plan
from _fakes.engine import with_executor
from _fakes.run import all_events

from swarm_arena import (
    FunctionTool,
    Goal,
    PartSpec,
    Plan,
    ProgramJudge,
    Ref,
    RunConfig,
    Score,
    ScriptedManager,
    Swarm,
)
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog
from swarm_arena._model.planning_types import History
from swarm_arena._planning.provenance import ref_sites
from swarm_arena._store.local import LocalStore
from swarm_arena.board import Candidate
from swarm_arena.manager import AgentContext, RoundReport
from swarm_arena.planning import Planner
from swarm_arena.scoring import JudgeContext
from swarm_arena.tool import FileSet

_MODULE = "_engine.test_part_identity"


class PartManager(ScriptedManager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        value = ctx.goal.inputs.get("partition", 2 if "b.txt" in ctx.goal.files.manifest else 1)
        await ctx.run_tool({"x": float(value)})
        return RoundReport()


class ReplacePart(Planner):
    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        assert history
        p = history.revisions[-1]
        part = p.parts[0]
        changed = (
            part.goal.model_copy(update={"inputs": {"partition": 2}})
            if part.goal.inputs
            else part.goal.model_copy(update={"files": FileSet(globs=["b.txt"])})
        )
        return p.model_copy(update={"parts": [part.model_copy(update={"goal": changed})]})


def parent_partition() -> dict[str, float]:
    parent = json.loads(next(Path("parents").glob("*/data.json")).read_text())
    return {"parent_partition": parent["x"], "y": 0.0}


def _anchor_probe(root: Path, patch: pytest.MonkeyPatch) -> None:
    original = ProgramJudge.judge

    async def judge(
        self: ProgramJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        if ctx.part == "a" and ctx.round == 2:
            await anyio.to_thread.run_sync(
                (root / "barrier-inputs.json").write_text,
                json.dumps([c.request.inputs["x"] for c in candidates]),
            )
        return await original(self, goal, candidates, ctx)

    patch.setattr(ProgramJudge, "anchor_count", lambda self: 1)
    patch.setattr(ProgramJudge, "judge", judge)


async def part_driver(root: Path, kind: str, crash: bool) -> None:
    """Run the part scenario with writer and judge probes scoped to this call."""
    p = _plan(teams=1, rounds=2, converge_team=True)
    if kind == "manifest":
        for file in ["a.txt", "b.txt"]:
            await anyio.to_thread.run_sync((root / file).write_text, file)
        p = p.model_copy(
            update={"goal": p.goal.model_copy(update={"files": FileSet(globs=["a.txt", "b.txt"])})}
        )
    part = PartSpec(
        name="a",
        goal=Goal(
            id="part",
            namespace="n",
            inputs={"partition": 1} if kind == "inputs" else {},
            files=FileSet(globs=["a.txt"]) if kind == "manifest" else FileSet(),
        ),
    )
    team = p.teams[0]
    team = team.model_copy(
        update={
            "part": "a",
            "agents": [
                team.agents[0].model_copy(
                    update={"manager": Ref(path=f"{_MODULE}:PartManager", config={"requests": []})}
                )
            ],
        }
    )
    assert p.converge.team
    conv = p.converge.team
    join = conv.agents[0].model_copy(
        update={
            "tool": Ref.of(FunctionTool(f"{_MODULE}:parent_partition")),
            "manager": Ref.of(ScriptedManager([{"inputs": {}}])),
        }
    )
    p = p.model_copy(
        update={
            "parts": [part],
            "teams": [team],
            "converge": p.converge.model_copy(
                update={"team": conv.model_copy(update={"agents": [join]})}
            ),
        }
    )
    p = Plan._from_data(p.model_dump(mode="json"), base_dir=root)
    entries = [
        CatalogEntry(
            name=f"entry-{i}",
            kind=s.kind,
            ref=s.ref.path,
            config=s.ref.config,
            description="Approved component",
        )
        for i, s in enumerate(ref_sites(p))
    ]
    cfg = with_executor(
        RunConfig(
            run_root=str(root / "runs"),
            arena_root=str(root / "arena"),
            artifact_root=str(root / "artifacts"),
            planner_catalog=PlannerCatalog(entries=entries),
        ),
        workspace_root=str(root / "ws"),
    )
    publish = JournalWriter._publish

    async def commit(self: JournalWriter) -> None:
        self._group_window_s = 0.001
        hit = any(
            e.type == "plan_revision" and e.payload["plan"]["revision"] == 1 for e in self._queue
        )
        await publish(self)
        if crash and hit:
            os._exit(91)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(JournalWriter, "_publish", commit)
        _anchor_probe(root, patch)
        await Swarm(cfg, replanner=ReplacePart()).run(p, run_id="part")


async def test_part_driver_restores_writer_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_publish = JournalWriter._publish

    async def fail(self: Swarm, plan: Plan, *, run_id: str) -> None:
        assert JournalWriter._publish is not original_publish
        raise RuntimeError("driver failed")

    monkeypatch.setattr(Swarm, "run", fail)
    with pytest.raises(RuntimeError, match="driver failed"):
        await part_driver(tmp_path, "inputs", False)
    assert JournalWriter._publish is original_publish


@pytest.mark.parametrize("kind", ["inputs", "manifest"])
@pytest.mark.parametrize("resumed", [False, True])
async def test_current_part_identity_controls_join_and_board(
    tmp_path: Path, kind: str, resumed: bool
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
    )
    code = (
        f"import sys,anyio; from pathlib import Path; from {_MODULE} import part_driver; "
        "anyio.run(part_driver,Path(sys.argv[1]),sys.argv[2],sys.argv[3]=='1')"
    )
    child = await anyio.run_process(
        [sys.executable, "-c", code, str(tmp_path), kind, str(int(resumed))],
        env=environment,
        check=False,
    )
    assert child.returncode == (91 if resumed else 0), child.stderr.decode()
    cfg = RunConfig.load(tmp_path / "runs" / "part" / "run.toml")
    if resumed:
        with pytest.MonkeyPatch.context() as patch:
            _anchor_probe(tmp_path, patch)
            await Swarm(cfg, replanner=ReplacePart()).resume("part", force=True)
    assert json.loads(
        await anyio.to_thread.run_sync((tmp_path / "barrier-inputs.json").read_text)
    ) == [2.0]
    state = RunState.from_events(all_events(cfg))
    assert len(state.plans["g"]) == 2
    goal = state.goals["g"]
    assert goal.final_order
    winner = state.candidates[goal.final_order[0]]
    assert winner.output.data["parent_partition"] == 2
    old, new = sorted(
        (c for c in state.candidates.values() if c.part == "a"), key=lambda c: c.round
    )
    assert winner.parent_ids == [new.id]
    store = LocalStore(Path(cfg.run_root) / "part")
    committed = read_committed(store)
    assert committed
    board = ArenaBoard(arena_root=cfg.arena_root, cache_dir=str(tmp_path / "cache"))
    board.open("part", cfg.run_root, ["n"])
    key = state.plans["g"][-1][0].parts[0].goal.key()
    view = board.snapshot(key, board.frontier())
    assert [c.id for c in view.best(key, 5, judge="y")] == [new.id]
    with pytest.raises(KeyError):
        view.get_candidate(old.id)
    board.close()
