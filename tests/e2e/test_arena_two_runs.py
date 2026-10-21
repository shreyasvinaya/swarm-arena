"""Two live swarms exchange committed learnings without sharing mutable files."""

from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path

import anyio
import pytest
from _fakes import tools
from _fakes.curators import PostingManager

from swarm_arena import (
    AgentSpec,
    FunctionTool,
    Goal,
    KeepTopTeams,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    ScriptedManager,
    Swarm,
    TeamSpec,
)
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.reindex import recorded_frontiers
from swarm_arena._errors import AlreadyExists, PreconditionFailed
from swarm_arena._journal.event import Event
from swarm_arena._journal.lease import read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.results import Candidate, Learning
from swarm_arena._store.local import LocalStore
from swarm_arena.tool import RunStatus

WRITER: ContextVar[str] = ContextVar("WRITER", default="")
A = "2026-10-18T13-00-00Z_aaaaaa"
B = "2026-10-18T13-00-00Z_bbbbbb"


def _plan(goal_id: str, own_plan: Path, other_page: Path) -> Plan:
    return Plan(
        goal=Goal(id=goal_id, namespace="toy"),
        rounds=[RoundSpec(), RoundSpec(), RoundSpec()],
        policy=Ref.of(KeepTopTeams(1)),
        teams=[
            TeamSpec(
                name="team",
                rationale="Wait for the peer's committed round page.",
                agents=[
                    AgentSpec(
                        name="agent",
                        rationale="Observe committed file readiness.",
                        tool=Ref.of(FunctionTool("_fakes.tools:wait_for_file")),
                        manager=Ref.of(
                            PostingManager(
                                ScriptedManager.Config(
                                    requests=[
                                        {"inputs": {"path": str(own_plan)}, "round": 1},
                                        {"inputs": {"path": str(other_page)}, "round": 2},
                                        {
                                            "inputs": {"path": str(own_plan), "timeout_s": 30.0},
                                            "round": 3,
                                        },
                                    ],
                                    post_learnings=True,
                                )
                            )
                        ),
                    )
                ],
            )
        ],
        judges=[Ref.of(ProgramJudge("_fakes.tools:score_one"))],
    )


async def test_two_swarms_share_learnings_within_one_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[tuple[str, str, str, bool]] = []
    create, replace = LocalStore.create, LocalStore.replace

    def tracked_create(store: LocalStore, key: str, data: bytes) -> str:
        try:
            result = create(store, key, data)
        except (AlreadyExists, PreconditionFailed):
            writes.append((WRITER.get(), "create", f"{store.url}/{key}", False))
            raise
        writes.append((WRITER.get(), "create", f"{store.url}/{key}", True))
        return result

    def tracked_replace(store: LocalStore, key: str, data: bytes, *, if_match: str) -> str:
        try:
            result = replace(store, key, data, if_match=if_match)
        except (AlreadyExists, PreconditionFailed):
            writes.append((WRITER.get(), "replace", f"{store.url}/{key}", False))
            raise
        writes.append((WRITER.get(), "replace", f"{store.url}/{key}", True))
        return result

    monkeypatch.setattr(LocalStore, "create", tracked_create)
    monkeypatch.setattr(LocalStore, "replace", tracked_replace)
    arena_root = str(tmp_path / "arena")
    artifact_root = str(tmp_path / "artifacts")
    run_root_a, run_root_b = str(tmp_path / "runs-a"), str(tmp_path / "runs-b")
    config_a = RunConfig(run_root=run_root_a, arena_root=arena_root, artifact_root=artifact_root)
    config_b = RunConfig(run_root=run_root_b, arena_root=arena_root, artifact_root=artifact_root)
    plan_a = _plan(
        "a",
        Path(run_root_a) / A / "goals/a/plan.r0.toml",
        Path(run_root_b) / B / "goals/b/rounds/round-01.md",
    )
    plan_b = _plan(
        "b",
        Path(run_root_b) / B / "goals/b/plan.r0.toml",
        Path(run_root_a) / A / "goals/a/rounds/round-01.md",
    )
    configs, plans = {A: config_a, B: config_b}, {A: plan_a, B: plan_b}

    async def run_as(name: str, swarm: Swarm, plan: Plan) -> None:
        token = WRITER.set(name)
        try:
            await swarm.run(plan, run_id=name)
        finally:
            WRITER.reset(token)

    # Failure cancels the sibling task; Swarm owns and closes its subprocess executor.
    with anyio.fail_after(120):
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_as, A, Swarm(config_a), plan_a)
            tg.start_soon(run_as, B, Swarm(config_b), plan_b)

    def events_of(run_id: str) -> list[Event]:
        store = LocalStore(Path(configs[run_id].run_root) / run_id)
        owner = read_owner(store)
        assert owner is not None
        assert owner[0].head is not None
        return [event for _, events in read_chain(store, owner[0].head) for event in events]

    events = {run_id: await anyio.to_thread.run_sync(events_of, run_id) for run_id in (A, B)}

    def goal_key(run_id: str) -> GoalKey:
        return plans[run_id].goal.key()

    def round1_learning_id(run_id: str) -> str:
        [learning] = [
            Learning.model_validate(event.payload["learning"])
            for event in events[run_id]
            if event.type == "learning" and event.payload["learning"]["round"] == 1
        ]
        return learning.id

    for run_events in events.values():
        candidates = [
            Candidate.model_validate(event.payload["candidate"])
            for event in run_events
            if event.type == "tool_end"
        ]
        assert [candidate.round for candidate in candidates] == [1, 2, 3]
        assert all(candidate.output.status is RunStatus.OK for candidate in candidates)
        assert all(candidate.output.data["waited_s"] >= 0 for candidate in candidates)
        assert [event.payload["round"] for event in run_events if event.type == "round_closed"] == [
            1,
            2,
            3,
        ]

    board = ArenaBoard(arena_root=arena_root, cache_dir=str(tmp_path / "reader-cache"))
    try:
        await anyio.to_thread.run_sync(board.open, "", "", ["toy"])
        await anyio.to_thread.run_sync(board.frontier)
        for me, other in ((A, B), (B, A)):
            _, _, f3 = next(x for x in recorded_frontiers(events[me]) if x[1] == 3)
            snapshot = await anyio.to_thread.run_sync(board.snapshot, goal_key(me), f3)
            assert round1_learning_id(other) in [x.id for x in snapshot.learnings("toy", limit=100)]
    finally:
        await anyio.to_thread.run_sync(board.close)

    assert {run for run, _, _, _ in writes} == {A, B}
    tried_by: dict[str, set[str]] = defaultdict(set)
    for run, _, path, _ in writes:
        tried_by[path].add(run)
    shared = sorted(
        path
        for path, runs in tried_by.items()
        if len(runs) > 1 and not path.startswith(artifact_root)
    )
    assert shared == []
    assert sorted(path for _, _, path, ok in writes if ok and path.startswith(arena_root)) == [
        f"{arena_root}/toy/runs/{A}.json",
        f"{arena_root}/toy/runs/{B}.json",
    ]
    replaced = {path for _, op, path, ok in writes if ok and op == "replace"}
    assert replaced == {
        f"{run_root_a}/{A}/control/owner.json",
        f"{run_root_b}/{B}/control/owner.json",
    }


def test_wait_for_file_polls_until_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ready"
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds
        path.write_text("committed")

    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tools.time, "sleep", sleep)
    assert tools.wait_for_file(str(path)) == {"waited_s": 0.1}
    assert sleeps == [0.1]
    assert tools.wait_for_file(str(path), timeout_s=0.0) == {"waited_s": 0.0}


def test_wait_for_file_timeout_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tools.time, "sleep", sleep)
    with pytest.raises(TimeoutError, match="file readiness timed out"):
        tools.wait_for_file(str(tmp_path / "absent"), timeout_s=0.2)
    assert clock[0] == 0.2


def test_score_one_ignores_candidate_value() -> None:
    score = tools.score_one(Goal(id="a", namespace="toy"), tools.cand(data={"waited_s": 12.0}))
    assert score.value == 1.0
    assert score.key == (1.0,)
