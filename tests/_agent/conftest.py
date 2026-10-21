"""Shared real-service fixtures for agent round tests."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from _fakes.engine import CtxEnv, RtEnv
from _fakes.tools import RecordingExecutor

from swarm_arena import (
    AgentSpec,
    FunctionTool,
    Goal,
    Limits,
    Plan,
    Ref,
    RoundSpec,
    RunConfig,
    TeamSpec,
)
from swarm_arena._agent.repeats import RepeatIndex
from swarm_arena._agent.runtime import AgentCounters, AgentRuntime
from swarm_arena._agent.tool_run import Inflight, track_inflight
from swarm_arena._agent.watchdog import Watchdog
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger
from swarm_arena._budget.tranche import Tranche
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.lease import Lease
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.ids import IdGen
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.local import LocalStore
from swarm_arena._tooling.staging import tool_identity


@pytest.fixture
async def rt_env(
    tmp_path: Path, run_config: RunConfig, artifacts: ArtifactStore, request: pytest.FixtureRequest
) -> AsyncIterator[RtEnv]:
    run_id = "engine-run"
    store = LocalStore(Path(run_config.run_root) / run_id)
    limits = Limits(**getattr(request, "param", {"max_usd": 100}))
    config = run_config.model_copy(update={"limits": limits})
    lease = Lease.acquire_new(store, owner_id="engine", host="local", ttl_s=60)
    ids = IdGen(run_id, 1)
    goal = Goal(id="g", namespace="n")
    tool = FunctionTool("_fakes.tools:square")
    ref = Ref.of(tool)
    spec = AgentSpec(
        name="a", rationale="Explore", tool=ref, manager=Ref(path="_fakes.tools:NoopManager")
    )
    team = TeamSpec(name="t", rationale="Explore", agents=[spec])
    async with JournalWriter(
        store,
        lease,
        run_id=run_id,
        next_seq=1,
        local_path=tmp_path / "journal.jsonl",
        sync_interval_s=60,
        group_window_s=0,
    ) as journal:
        ledger = Ledger(limits, reserve_fraction=0.15, goal_weights=None, journal=journal, ids=ids)
        await journal.append_durable(
            EventDraft(
                type=EventType.RUN_CONFIG, payload={"config": config.model_dump(mode="json")}
            )
        )
        await journal.append_durable(
            EventDraft(
                type=EventType.PLAN_REVISION,
                goal_id="g",
                payload={
                    "plan": Plan(goal=goal, teams=[team], judges=[]).model_dump(mode="json"),
                    "origin": "human",
                    "preflight_complete": True,
                    "first_round": 1,
                },
            )
        )
        await ledger.hold_final_slices({"g": Amount(usd=5)})
        await journal.append_durable(
            EventDraft(
                type=EventType.RUN_STARTED, payload={"final_slices": {"g": {"tokens": 0, "usd": 5}}}
            )
        )
        ledger.open_goal("g")
        await journal.append_durable(EventDraft(type=EventType.GOAL_STARTED, goal_id="g"))
        grants = ledger.open_round("g", 1, ["g/t/a", "g/t/b"], rounds_left=1)
        await journal.append_durable(
            EventDraft(
                type=EventType.ROUND_STARTED,
                goal_id="g",
                round=1,
                payload={"round": 1, "shares": {h: a.model_dump() for h, a in grants.items()}},
            )
        )
        board = ArenaBoard(arena_root=config.arena_root, cache_dir=str(tmp_path / "cache"))
        board.open(run_id, config.run_root, ["n"])
        executor = LocalExecutor(workspace_root=str(tmp_path / "ws"))
        round_spec = RoundSpec()
        rt = AgentRuntime(
            address="g/t/a",
            holder="g/t/a",
            goal_id="g",
            goal=goal,
            goal_key=goal.key(),
            part=None,
            team=team,
            spec=spec,
            round=1,
            round_spec=round_spec,
            fidelity=round_spec.effective_fidelity(),
            tool=tool,
            tool_ref=ref,
            execution_ref=ref,
            tool_identity=tool_identity(ref, None),
            catalog_entry=None,
            checks=[],
            view=board.snapshot(goal.key(), board.frontier()),
            own=[],
            digest="",
            notes="",
            ledger=ledger,
            tranche=Tranche(ledger, "g/t/a", "g", size=Amount(usd=0.25)),
            limits=limits,
            counters=AgentCounters(),
            repeats=RepeatIndex(run_id),
            watchdog=Watchdog(limits),
            journal=journal,
            board=board,
            executor=executor,
            artifacts=artifacts,
            ids=ids,
            config=config,
            resumed=(),
        )
        registry = Inflight()
        with track_inflight(registry):
            try:
                yield RtEnv(rt, store, board, tmp_path / "marker", RecordingExecutor(), registry)
            finally:
                board.close()
                await executor.aclose()


@pytest.fixture
async def ctx_env(rt_env: RtEnv) -> CtxEnv:
    return CtxEnv(rt_env)
