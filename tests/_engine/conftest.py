"""Real lease, journal, board, budget, and executor for round tests."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from _fakes.engine import EngineEnv, with_executor

from swarm_arena import RunConfig
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger
from swarm_arena._engine.round import EngineDeps
from swarm_arena._engine.stops import RunClock, StopState
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.lease import Lease
from swarm_arena._journal.views import ViewWriter
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.ids import IdGen
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.local import LocalStore


@pytest.fixture
async def env(
    tmp_path: Path, run_config: RunConfig, artifacts: ArtifactStore, local_executor: LocalExecutor
) -> AsyncIterator[EngineEnv]:
    run_id = "engine-run"
    config = with_executor(run_config, **local_executor.config.model_dump(exclude_none=True))
    store = LocalStore(Path(config.run_root) / run_id)
    lease = Lease.acquire_new(store, owner_id="engine", host="local", ttl_s=60)
    ids = IdGen(run_id, lease.doc.epoch)
    async with JournalWriter(
        store,
        lease,
        run_id=run_id,
        next_seq=1,
        local_path=tmp_path / "journal.jsonl",
        sync_interval_s=60,
    ) as journal:
        ledger = Ledger(
            config.limits,
            reserve_fraction=config.reserve_fraction,
            goal_weights=config.goal_weights,
            journal=journal,
            ids=ids,
        )
        journal.append(
            EventDraft(
                type=EventType.RUN_CONFIG, payload={"config": config.model_dump(mode="json")}
            )
        )
        await ledger.hold_final_slices({"g": Amount()})
        journal.append(
            EventDraft(
                type=EventType.RUN_STARTED, payload={"final_slices": {"g": Amount().model_dump()}}
            )
        )
        ledger.open_goal("g")
        await journal.append_durable(EventDraft(type=EventType.GOAL_STARTED, goal_id="g"))
        board = ArenaBoard(arena_root=config.arena_root, cache_dir=str(tmp_path / "cache"))
        board.open(run_id, config.run_root, ["n"])
        deps = EngineDeps(
            run_id=run_id,
            config=config,
            store=store,
            journal=journal,
            ledger=ledger,
            board=board,
            executor=local_executor,
            artifacts=artifacts,
            ids=ids,
            views=ViewWriter(store, journal),
            stops=StopState(),
            clock=RunClock(0.0),
        )
        try:
            yield EngineEnv(deps, store)
        finally:
            board.close()
            await local_executor.aclose()
