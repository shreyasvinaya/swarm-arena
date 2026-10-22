"""Real owned journal and ledger services for core-only LLM tests."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Any
from uuid import uuid4

import anyio

from swarm_arena import Limits, RunConfig
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger
from swarm_arena._budget.llm_scope import LLMScope
from swarm_arena._budget.tranche import Tranche
from swarm_arena._engine.run import _setup_run, _supervise_writer
from swarm_arena._engine.stops import RunClock, StopState
from swarm_arena._journal.event import Event, EventDraft
from swarm_arena._journal.lease import Lease
from swarm_arena._journal.llm_record import _bound_run
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.base import CallRole
from swarm_arena._model.ids import IdGen
from swarm_arena._model.results import LLMCall
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.local import LocalStore


@dataclass
class HeldOperation:
    """Block a synchronous worker until its test explicitly releases and joins it."""

    entered: ThreadEvent = field(default_factory=ThreadEvent)
    release: ThreadEvent = field(default_factory=ThreadEvent)
    exited: ThreadEvent = field(default_factory=ThreadEvent)

    def wait(self) -> None:
        self.entered.set()
        try:
            assert self.release.wait(30), "test did not release held operation"
        finally:
            self.exited.set()

    async def ready(self) -> None:
        with anyio.fail_after(5):
            assert await anyio.to_thread.run_sync(self.entered.wait, 5)

    async def finish(self) -> None:
        self.release.set()
        with anyio.fail_after(5):
            assert await anyio.to_thread.run_sync(self.exited.wait, 5)


@dataclass
class LLMRun:
    """Expose the recorder's real services and committed accounting oracles."""

    run_id: str
    config: RunConfig
    store: LocalStore
    journal: JournalWriter
    artifacts: ArtifactStore
    ledger: Ledger
    ids: IdGen
    stops: StopState
    clock: RunClock
    scopes: list[LLMScope] = field(default_factory=list)

    def scope(
        self,
        role: CallRole,
        *,
        goal_id: str = "g",
        team: str = "t",
        agent: str = "a",
        round: int = 1,
        force_no_cache_read: bool = False,
    ) -> LLMScope:
        # Direct reservations make charged() a committed ledger-event oracle.
        tranche = Tranche(self.ledger, ":run", goal_id, size=Amount())
        scope = LLMScope(
            role=role,
            run_id=self.run_id,
            goal_id=goal_id,
            team=team,
            agent=agent,
            round=round,
            candidate_id=None,
            tranche=tranche,
            journal=self.journal,
            artifacts=self.artifacts,
            models=self.config.models,
            tracing=self.config.tracing,
            cache=self.config.cache,
            limits=self.config.limits,
            settle_timeout_s=self.config.settle_timeout_s,
            force_no_cache_read=force_no_cache_read,
        )
        self.scopes.append(scope)
        return scope

    def metadata(self, scope: LLMScope, *, reservation_id: str, call_id: str) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "goal_id": scope.goal_id,
            "team": scope.team,
            "agent": scope.agent,
            "round": str(scope.round),
            "role": scope.role,
            "reservation_id": reservation_id,
            "call_id": call_id,
            "model_alias": "opus",
        }

    def events(self, type: str) -> list[Event]:
        committed = read_committed(self.store)
        assert committed is not None
        return [event for event in committed[1] if event.type == type]

    def calls(self) -> list[LLMCall]:
        return [LLMCall.model_validate(event.payload["call"]) for event in self.events("llm_call")]

    def state(self) -> RunState:
        committed = read_committed(self.store)
        assert committed is not None
        return RunState.from_events(committed[1])

    def charged(self, scope: LLMScope) -> Amount:
        ids = set(scope.tranche._calls) | {batch.id for batch in scope.tranche._batches}
        charges = {
            event.payload["reservation"]["id"]: Amount.model_validate(
                event.payload["reservation"]["amount"]
            )
            for event in self.events("reserve")
            if event.payload["reservation"]["id"] in ids
        }
        for event in self.events("settle"):
            if event.payload["id"] in charges and not event.payload["estimated"]:
                charges[event.payload["id"]] = Amount.model_validate(event.payload["actual"])
        return sum(charges.values(), Amount())


@asynccontextmanager
async def open_llm_run(
    tmp_path: Path, *, config: RunConfig | None = None
) -> AsyncGenerator[LLMRun]:
    config = config or RunConfig(limits=Limits(max_tokens=None, max_usd=None), settle_timeout_s=2)
    run_id = f"llm-test-{uuid4().hex}"
    store = LocalStore(tmp_path / run_id)
    artifacts = ArtifactStore(LocalStore(tmp_path / "artifacts"))
    lease = Lease.acquire_new(store, owner_id="test", host="local", ttl_s=config.lease_ttl_s)
    ids = IdGen(run_id, lease.doc.epoch)
    stops, clock = StopState(), RunClock(0)
    async with (
        JournalWriter(
            store,
            lease,
            run_id=run_id,
            next_seq=1,
            local_path=tmp_path / f"{run_id}.jsonl",
            sync_interval_s=60,
            group_window_s=0.001,
            settle_timeout_s=config.settle_timeout_s,
        ) as journal,
        _supervise_writer(journal),
    ):
        ledger = Ledger(
            config.limits,
            reserve_fraction=config.reserve_fraction,
            goal_weights=None,
            journal=journal,
            ids=ids,
        )
        journal.append(EventDraft(type="run_config", payload={"config": config.model_dump()}))
        journal.append(EventDraft(type="run_started", payload={"final_slices": {}}))
        await ledger.hold_final_slices({})
        await journal.commit()
        async with _bound_run(
            run_id,
            journal=journal,
            ids=ids,
            artifacts=artifacts,
            config=config,
            stops=stops,
            clock=clock,
            state=RunState(),
        ):
            await _setup_run((), config)
            yield LLMRun(run_id, config, store, journal, artifacts, ledger, ids, stops, clock)


def record_kwargs(metadata: dict[str, str], **overrides: Any) -> dict[str, Any]:
    """Make a normal callback without importing any optional model adapter."""
    from swarm_arena.tool import Usage

    return {
        "metadata": metadata,
        "model": "anthropic/claude",
        "provider": "anthropic",
        "usage": Usage(input_tokens=100, output_tokens=20),
        "cost_usd": 0.002,
        "cache_hit": False,
        "latency_ms": 12.0,
        "status": "ok",
        "error": None,
        "request": {"messages": [{"role": "user", "content": "hello"}]},
        "response": {"content": "world"},
        "litellm_call_id": None,
        "tool_call_ids": [("call_1", "run_tool")],
        "settle": True,
        **overrides,
    }
