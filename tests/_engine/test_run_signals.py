from pathlib import Path

import anyio
import pytest
from _fakes import run as run_fakes

from swarm_arena import RunConfig
from swarm_arena._errors import ChainError
from swarm_arena._journal.event import Event, EventDraft
from swarm_arena._journal.lease import Lease, OwnerDoc
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("startup", ["missing_root", "empty_root", "missing_owner", "empty_head"])
async def test_wait_tool_started_waits_through_startup(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, startup: str
) -> None:
    root = Path(run_config.run_root)
    if startup != "missing_root":
        await anyio.to_thread.run_sync(root.mkdir)
    store = LocalStore(root / "r")
    lease = None
    if startup in {"missing_owner", "empty_head"}:
        await anyio.to_thread.run_sync(lambda: (root / "r" / "control").mkdir(parents=True))
    if startup == "empty_head":
        lease = Lease.acquire_new(store, owner_id="o", host="h", ttl_s=60)
    sampled = anyio.Event()
    returned = anyio.Event()
    original_dir = run_fakes.run_dir
    original_read = run_fakes.read_committed

    def sample_dir(config: RunConfig) -> Path:
        try:
            return original_dir(config)
        except (FileNotFoundError, StopIteration):
            anyio.from_thread.run_sync(sampled.set)
            raise

    def sample_read(store: LocalStore) -> tuple[OwnerDoc, list[Event]] | None:
        result = original_read(store)
        anyio.from_thread.run_sync(sampled.set)
        return result

    monkeypatch.setattr(run_fakes, "run_dir", sample_dir)
    monkeypatch.setattr(run_fakes, "read_committed", sample_read)

    async def wait() -> None:
        result = await run_fakes.wait_tool_started(run_config)
        assert result.url == store.url
        returned.set()

    with anyio.fail_after(20):
        async with anyio.create_task_group() as tg:
            tg.start_soon(wait)
            await sampled.wait()
            assert not returned.is_set()
            if lease is None:
                lease = await anyio.to_thread.run_sync(
                    lambda: Lease.acquire_new(store, owner_id="o", host="h", ttl_s=60)
                )
            async with JournalWriter(
                store,
                lease,
                run_id="r",
                next_seq=1,
                local_path=tmp_path / "local.jsonl",
                sync_interval_s=3600,
            ) as journal:
                await journal.append_durable(EventDraft(type="tool_start"))
                await returned.wait()


async def test_wait_tool_started_requires_committed_start(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalStore(Path(run_config.run_root) / "r")
    lease = Lease.acquire_new(store, owner_id="o", host="h", ttl_s=60)
    sampled = anyio.Event()
    returned = anyio.Event()
    original_read = run_fakes.read_committed

    def sample_read(store: LocalStore) -> tuple[OwnerDoc, list[Event]] | None:
        result = original_read(store)
        anyio.from_thread.run_sync(sampled.set)
        return result

    monkeypatch.setattr(run_fakes, "read_committed", sample_read)

    async def wait() -> None:
        await run_fakes.wait_tool_started(run_config)
        returned.set()

    with anyio.fail_after(20):
        async with JournalWriter(
            store,
            lease,
            run_id="r",
            next_seq=1,
            local_path=tmp_path / "local.jsonl",
            sync_interval_s=3600,
            group_window_s=3600,
        ) as journal:
            journal.append(EventDraft(type="run_started"))
            await journal.commit()
            journal.append(EventDraft(type="tool_start"))
            async with anyio.create_task_group() as tg:
                tg.start_soon(wait)
                await sampled.wait()
                await anyio.wait_all_tasks_blocked()
                assert not returned.is_set()
                committed = await anyio.to_thread.run_sync(run_fakes.all_events, run_config)
                assert [e.type for e in committed] == ["run_started"]
                await journal.commit()
                await returned.wait()


async def test_wait_tool_started_does_not_hide_malformed_journal(run_config: RunConfig) -> None:
    store = LocalStore(Path(run_config.run_root) / "r")
    await anyio.to_thread.run_sync(store.create, "control/owner.json", b"malformed")
    with pytest.raises(ChainError, match="invalid owner document"):
        await run_fakes.wait_tool_started(run_config)
