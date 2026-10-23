"""Capability-bearing in-memory adapter and committed reconciliation fixtures."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from swarm_arena import RunConfig
from swarm_arena._gateway.adapter import ProfileReport, SpendRow, VirtualKey
from swarm_arena._gateway.keys import KeyRecord
from swarm_arena._journal.event import Event, EventDraft
from swarm_arena._journal.lease import Lease
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.settings import GatewayConfig
from swarm_arena._model.tooling import Usage
from swarm_arena._store.local import LocalStore

T0 = datetime(2026, 1, 1, tzinfo=UTC)


class CapableGateway:
    """Explicit test capability; never stand in for LiteLLM's strict profile."""

    terminal_accounting = False

    def __init__(self) -> None:
        self.profile = ProfileReport(
            verified=True, failures=(), request_timeout_s=300, batch_write_interval_s=10
        )
        self.closure: datetime | None = T0
        self.rows: list[SpendRow] = []
        self.blocked: list[str] = []
        self.deleted: list[str] = []
        self.windows: list[tuple[str, datetime, datetime]] = []
        self.errors: dict[str, Exception] = {}

    def _check(self, operation: str) -> None:
        if operation in self.errors:
            raise self.errors[operation]

    async def preflight(self) -> ProfileReport:
        self._check("preflight")
        return self.profile

    async def create_key(
        self,
        *,
        alias: str,
        max_budget_usd: float,
        duration_s: float,
        metadata: Mapping[str, Any],
        default_estimated_output_tokens: int,
    ) -> tuple[VirtualKey, str]:
        raise AssertionError("reconcile never creates a key")

    async def block_key(self, key: VirtualKey) -> None:
        self._check("block")
        self.blocked.append(key.key_id)

    async def key_matches(self, key: VirtualKey, *, run_id: str) -> bool | None:
        self._check("identity")
        return True

    async def closed_at(self, key: VirtualKey) -> datetime | None:
        self._check("closed_at")
        return self.closure

    async def spend_rows(
        self, key: VirtualKey, *, start: datetime, end: datetime
    ) -> list[SpendRow]:
        self._check("read")
        self.windows.append((key.key_id, start, end))
        return [row for row in self.rows if row.key_id == key.key_id]

    async def delete_key(self, key: VirtualKey) -> None:
        self._check("delete")
        self.deleted.append(key.key_id)


def key_record(run_id: str, *, key_id: str = "hash-1", candidate_id: str = "c-1") -> KeyRecord:
    return KeyRecord(
        candidate_id=candidate_id,
        key_id=key_id,
        alias=f"cand-{candidate_id}",
        reservation_id=f"res-{candidate_id}",
        max_budget_usd=1,
        cap="soft",
        run_id=run_id,
        owner_epoch=1,
        goal_id="g",
        team="t",
        agent="a",
        round=1,
        created_at=T0 - timedelta(days=2),
    )


def created(record: KeyRecord) -> EventDraft:
    return EventDraft(
        type="gateway_key", payload={"action": "created", **record.model_dump(mode="json")}
    )


def spend(request_id: str, cost: float, *, key_id: str = "hash-1") -> SpendRow:
    return SpendRow(
        key_id=key_id,
        request_id=request_id,
        litellm_call_id="same-client-id",
        model="openai/gpt-6",
        usage=Usage(input_tokens=5, output_tokens=1),
        cost_usd=cost,
        ts=T0,
        metadata={"owner_epoch": "SECRET", "candidate_id": "untrusted"},
    )


def config_at(path: Path) -> RunConfig:
    return RunConfig(
        run_root=str(path / "runs"),
        artifact_root=str(path / "artifacts"),
        llm_gateway=GatewayConfig(url="http://gateway.invalid", require_hard_cap=False),
    )


def committed(config: RunConfig, run_id: str = "r-reconcile") -> list[Event]:
    snapshot = read_committed(LocalStore(Path(config.run_root) / run_id))
    assert snapshot is not None
    return snapshot[1]


async def seed_run(
    config: RunConfig, *, run_id: str = "r-reconcile", with_key: bool = True
) -> None:
    store = LocalStore(Path(config.run_root) / run_id)
    lease = Lease.acquire_new(store, owner_id="old", host="local", ttl_s=config.lease_ttl_s)
    async with JournalWriter(
        store,
        lease,
        run_id=run_id,
        next_seq=1,
        local_path=Path(config.run_root) / "seed.jsonl",
        sync_interval_s=60,
        group_window_s=0,
    ) as writer:
        writer.append(EventDraft(type="run_config", payload={"config": config.model_dump()}))
        if with_key:
            writer.append(created(key_record(run_id)))
        await writer.commit()
