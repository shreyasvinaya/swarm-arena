"""Per-tool key ownership and informational recording, without network I/O."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.llm_run import LLMRun, open_llm_run
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import make_job

from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Reservation
from swarm_arena._budget.llm_scope import llm_scope
from swarm_arena._gateway.adapter import GatewayError, ProfileReport, SpendRow, VirtualKey
from swarm_arena._gateway.keys import (
    KeyManager,
    gateway_call_id,
    gateway_job,
    key_alias,
    key_metadata,
    needs_key,
    unreplaced_provider_keys,
)
from swarm_arena._journal.event import Event, EventDraft
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.settings import GatewayConfig
from swarm_arena._tooling.llm_calls import PROVIDER_KEYS
from swarm_arena.tool import Usage

SECRET = "sk-virtual-SENTINEL"


def _all_bytes(path: Path) -> bytes:
    return b"".join(item.read_bytes() for item in path.rglob("*") if item.is_file())


class MemoryGateway:
    terminal_accounting = False

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.blocked: list[str] = []
        self.inspected: list[str] = []
        self.rows: list[SpendRow] = []
        self.windows: list[tuple[datetime, datetime]] = []
        self.read_entered = anyio.Event()
        self.read_release: anyio.Event | None = None
        self.read_error: BaseException | None = None
        self.block_error: BaseException | None = None
        self.confirmed = True
        self.on_create: Any = None

    async def preflight(self) -> ProfileReport:
        raise NotImplementedError

    async def create_key(self, **kwargs: Any) -> tuple[VirtualKey, str]:
        self.created.append(kwargs)
        if self.on_create is not None:
            self.on_create()
        return VirtualKey(key_id=f"hash-{len(self.created)}", alias=kwargs["alias"]), SECRET

    async def block_key(self, key: VirtualKey) -> None:
        self.blocked.append(key.key_id)
        if self.block_error is not None:
            raise self.block_error

    async def key_matches(self, key: VirtualKey, *, run_id: str) -> bool | None:
        for index, record in enumerate(self.created, 1):
            if key.key_id == f"hash-{index}":
                return record["alias"] == key.alias and record["metadata"]["run_id"] == run_id
        return None

    async def closed_at(self, key: VirtualKey) -> datetime | None:
        self.inspected.append(key.key_id)
        return datetime.now(UTC) if self.confirmed else None

    async def spend_rows(
        self, key: VirtualKey, *, start: datetime, end: datetime
    ) -> list[SpendRow]:
        self.windows.append((start, end))
        self.read_entered.set()
        if self.read_release is not None:
            await self.read_release.wait()
        if self.read_error is not None:
            raise self.read_error
        return [row for row in self.rows if row.key_id == key.key_id]

    async def delete_key(self, key: VirtualKey) -> None:
        raise NotImplementedError


def _manager(gateway: MemoryGateway, run: LLMRun, **kw: Any) -> KeyManager:
    return KeyManager(
        gateway,
        GatewayConfig(url="http://gateway.invalid"),
        cap="soft",
        journal=run.journal,
        run_id=run.run_id,
        settle_timeout_s=kw.pop("settle_timeout_s", 1),
        **kw,
    )


async def _open(manager: KeyManager, run: LLMRun, **kw: Any) -> Any:
    return await manager.open(
        candidate_id=kw.pop("candidate_id", "c-a-e1-0001"),
        reservation=Reservation(id="res", holder=":run", kind="tool", amount=Amount(usd=1)),
        run_id=kw.pop("run_id", run.run_id),
        goal_id="g",
        team="t",
        agent="a",
        round=1,
        **kw,
    )


def _row(key_id: str = "hash-1", request_id: str = "request-1", **kw: Any) -> SpendRow:
    return SpendRow(
        key_id=key_id,
        request_id=request_id,
        litellm_call_id=kw.pop("litellm_call_id", "upstream"),
        model="openai/gpt-6",
        cost_usd=0.02,
        usage=Usage(input_tokens=5, output_tokens=1),
        ts=datetime.now(UTC),
        metadata=kw.pop("metadata", {}),
        **kw,
    )


def test_needs_key_rules() -> None:
    assert needs_key(frozenset({"openai"}), ())
    assert needs_key(frozenset(), ["OPENAI_API_KEY"])
    assert not needs_key(frozenset(), ["GEMINI_API_KEY"])
    assert not needs_key(frozenset(), [])


def test_unreplaced_provider_keys() -> None:
    assert unreplaced_provider_keys(["MY_VAR", "OPENAI_API_KEY", "GEMINI_API_KEY"]) == [
        "GEMINI_API_KEY"
    ]


def test_key_alias() -> None:
    assert key_alias("c-3f9a1c2b5d6e7f80-e1-0193") == "cand-c-3f9a1c2b5d6e7f80-e1-0193"


def test_key_metadata_tags_carry_candidate() -> None:
    meta = key_metadata(candidate_id="c-a-e1-0001", run_id="r1", goal_id="g", team="t", agent="a")
    assert "candidate:c-a-e1-0001" in meta["tags"]
    assert meta["spend_logs_metadata"] == {
        "candidate_id": "c-a-e1-0001",
        "run_id": "r1",
        "goal_id": "g",
        "team": "t",
        "agent": "a",
    }


def test_gateway_job_env() -> None:
    original = make_job(
        secret_env=("OPENAI_API_KEY", "GEMINI_API_KEY", "MY_VAR"),
        env={"KEEP": "yes"},
        secret_values={"CUSTOM": "value"},
    )
    job = gateway_job(original, url="http://gw:4000/", secret="sk-v")
    assert job.env == {
        "KEEP": "yes",
        "OPENAI_BASE_URL": "http://gw:4000/v1",
        "ANTHROPIC_BASE_URL": "http://gw:4000",
    }
    assert job.secret_values == {
        "CUSTOM": "value",
        "OPENAI_API_KEY": "sk-v",
        "ANTHROPIC_API_KEY": "sk-v",
    }
    assert job.secret_env == ("MY_VAR",)
    assert job.drop_env == tuple(sorted(PROVIDER_KEYS))
    assert original.env == {"KEEP": "yes"}
    assert "sk-v" not in repr(job)


def test_gateway_identity_uses_primary_request_and_key() -> None:
    assert gateway_call_id("k", "r") == "gw:" + sha256_hex(canonical_json(["k", "r"]))
    assert (
        len({gateway_call_id("k", "r"), gateway_call_id("k2", "r"), gateway_call_id("k", "r2")})
        == 3
    )


async def test_open_journals_record_without_secret(tmp_path: Path) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        before = datetime.now(UTC)
        record, secret = await _open(manager, run)
        await run.journal.commit()
        assert record.alias == "cand-c-a-e1-0001"
        assert record.max_budget_usd == 1.0
        assert before <= record.created_at <= datetime.now(UTC)
        assert record.run_id == run.run_id
        assert record.owner_epoch == run.journal.epoch
        assert manager.records(goal_id="g") == [record]
        assert manager.records(goal_id="other") == []
        assert gw.created[0]["duration_s"] == 86400
        assert gw.created[0]["default_estimated_output_tokens"] == 4096
        assert secret == SECRET
        assert secret.encode() not in await anyio.to_thread.run_sync(_all_bytes, tmp_path)
        assert run.events("gateway_key")[0].payload["action"] == "created"


async def test_rows_recorded_once_and_never_fund_ambient_scope(tmp_path: Path) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        gw.rows = [
            _row(request_id="one", litellm_call_id=None),
            _row(request_id="two", litellm_call_id="same"),
            _row(
                request_id="three",
                litellm_call_id="same",
                metadata={"owner_epoch": "999", "candidate_id": "foreign"},
            ),
        ]
        scope = run.scope("manager")
        with llm_scope(scope):
            calls = await manager.read([record], wait_first=False)
        assert len(calls) == 3
        assert not scope.call_ids
        assert [(call.role, call.candidate_id, call.litellm_call_id) for call in calls] == [
            ("tool", record.candidate_id, gateway_call_id(record.key_id, row.request_id))
            for row in gw.rows
        ]
        assert all(call.cost_usd == 0.02 and call.provider == "openai" for call in calls)
        assert await manager.read([record], wait_first=False) == []
        assert run.charged(scope) == Amount()
        assert not run.events("settle")
        start, end = gw.windows[0]
        assert start == record.created_at.replace(microsecond=0) - timedelta(minutes=15, seconds=1)
        assert end >= record.created_at


async def test_read_waits_for_first_rows(tmp_path: Path) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run, settle_timeout_s=3)
        record, _ = await _open(manager, run)

        async def produce() -> None:
            await gw.read_entered.wait()
            gw.rows = [_row()]

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(produce)
            calls = await manager.read([record], wait_first=True)
        assert len(calls) == 1
        assert len(gw.windows) >= 2


async def test_read_gives_up_after_settle_timeout(tmp_path: Path) -> None:
    gw = MemoryGateway()
    gw.read_release = anyio.Event()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run, settle_timeout_s=0.05)
        record, _ = await _open(manager, run)
        with anyio.fail_after(5):
            assert await manager.read([record], wait_first=True) == []
        assert gw.read_entered.is_set()
        assert not run.events("settle")


async def test_revoke_blocks_without_journal(tmp_path: Path) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        before = run.journal._next_seq
        await manager.revoke(record)
        assert gw.blocked == [record.key_id]
        assert not gw.inspected
        assert run.journal._next_seq == before


async def test_close_only_journals_confirmed_closure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        gw.confirmed = False
        assert (await manager.close(record)).closed_at is None
        gw.block_error = GatewayError(SECRET)
        assert (await manager.close(record)).closed_at is None
        assert SECRET not in caplog.text
        gw.block_error = None
        gw.confirmed = True
        closed = await manager.close(record)
        assert closed.closed_at is not None
        assert manager.records() == [closed]
        await run.journal.commit()
        assert [event.payload["action"] for event in run.events("gateway_key")] == [
            "created",
            "closed",
        ]


async def test_open_append_failure_blocks_and_does_not_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)

        def fail(draft: EventDraft) -> Event:
            raise ValueError(SECRET)

        with monkeypatch.context() as patch:
            patch.setattr(run.journal, "append", fail)
            with pytest.raises(GatewayError) as caught:
                await _open(manager, run)
        assert_secret_absent(caught.value, SECRET)
        assert gw.blocked == ["hash-1"]
        assert not gw.inspected
        assert manager.records() == []


async def test_cancel_at_create_handoff_blocks_without_publication(tmp_path: Path) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        before = run.journal._next_seq
        with anyio.CancelScope() as cancel:
            gw.on_create = cancel.cancel
            await _open(manager, run)
            pytest.fail("cancelled creation returned")
        assert gw.blocked == ["hash-1"]
        assert not manager.records()
        assert run.journal._next_seq == before


async def test_run_mismatch_refuses_before_creation(tmp_path: Path) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        with pytest.raises(ValueError, match="binding mismatch"):
            await _open(_manager(gw, run), run, run_id="other")
        assert not gw.created


async def test_restore_from_events_and_legacy_creation_time(tmp_path: Path) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        closed = await manager.close(record)
        await run.journal.commit()
        events = run.events("gateway_key")
        restored = _manager(gw, run)
        restored.restore(events)
        assert restored.records()[0].closed_at == closed.closed_at
        legacy = events[0].model_copy(
            update={
                "payload": {
                    key: value
                    for key, value in events[0].payload.items()
                    if key not in {"created_at", "run_id", "owner_epoch"}
                }
            }
        )
        restored.restore([legacy])
        assert restored.records()[0].created_at == legacy.ts
        with pytest.raises(ValueError, match="binding mismatch"):
            restored.restore([legacy.model_copy(update={"run_id": "foreign"})])


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("signal", ["fenced", "terminal", "lookalike", "limit_lookalike"])
@pytest.mark.parametrize("operation", ["read", "close", "revoke", "open"])
async def test_control_authority_at_key_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool, signal: str, operation: str
) -> None:
    from swarm_arena._agent.manager import LimitReached
    from swarm_arena._errors import Fenced, _core_error, _is_core_control, _SignalCategory

    if signal == "fenced":
        error: BaseException = _core_error(Fenced, "safe")
    elif signal == "terminal":
        error = _core_error(RuntimeError, "safe", _category=_SignalCategory.TERMINAL)
    elif signal == "limit_lookalike":
        error = LimitReached("budget", SECRET)
    else:
        error = Fenced(SECRET)
    raised = BaseExceptionGroup(SECRET, [error]) if nested else error
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)

        async def invoke() -> Any:
            if operation == "read":
                gw.read_error = raised
                return await manager.read([record], wait_first=False)
            if operation == "open":

                def fail(draft: EventDraft) -> Event:
                    raise raised

                with monkeypatch.context() as patch:
                    patch.setattr(run.journal, "append", fail)
                    return await _open(manager, run, candidate_id="c-two")
            gw.block_error = raised
            return await getattr(manager, operation)(record)

        if signal in {"fenced", "terminal"}:
            with pytest.raises(Exception, match=r"safe|ownership|primary fenced") as caught:
                await invoke()
            assert _is_core_control(caught.value)
            assert_secret_absent(caught.value, SECRET)
        elif operation in {"revoke", "open"}:
            with pytest.raises(GatewayError) as caught:
                await invoke()
            assert_secret_absent(caught.value, SECRET)
        else:
            result = await invoke()
            assert result == ([] if operation == "read" else record)
        if operation == "open":
            assert gw.blocked == ["hash-2"]
            assert manager.records() == [record]


async def test_failed_or_incomplete_read_keeps_charge_and_warns_safely(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        reservation = await run.ledger.reserve(
            ":run", Amount(usd=1), kind="tool", goal_id="g", keep_charged=True
        )
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        before = run.ledger.remaining(":run")
        gw.read_error = GatewayError(SECRET)
        assert await manager.read([record], wait_first=False) == []
        assert run.ledger.remaining(":run") == before
        assert not run.events("settle")
        assert run.state().reservations[reservation.id].reservation["keep_charged"]
        assert "gateway spend read failed" in caplog.text
        assert SECRET not in caplog.text


@pytest.mark.parametrize("mode", ["empty", "rows", "duplicate", "failed", "timeout"])
async def test_read_with_status_distinguishes_failure_from_empty(tmp_path: Path, mode: str) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run, settle_timeout_s=0.05)
        record, _ = await _open(manager, run)
        if mode in {"rows", "duplicate"}:
            gw.rows = [_row()]
        if mode == "duplicate":
            assert len(await manager.read([record], wait_first=False)) == 1
        if mode == "failed":
            gw.read_error = GatewayError(SECRET)
        if mode == "timeout":
            gw.read_release = anyio.Event()
        with anyio.fail_after(30):
            calls, status = await manager.read_with_status([record], wait_first=mode == "timeout")
        assert status == {"failed": "failed", "timeout": "incomplete"}.get(mode, "ok")
        assert len(calls) == (1 if mode == "rows" else 0)
        assert SECRET not in status
        assert await manager.read_with_status([], wait_first=False) == ([], "ok")


async def test_read_with_status_reports_partial_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._gateway import keys

    gw = MemoryGateway()
    gw.rows = [_row(request_id="bad"), _row(request_id="good")]
    record_call = keys.record_call

    async def sometimes_fail(**kwargs: Any) -> Any:
        if kwargs["litellm_call_id"] == gateway_call_id("hash-1", "bad"):
            raise ValueError(SECRET)
        return await record_call(**kwargs)

    monkeypatch.setattr(keys, "record_call", sometimes_fail)
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        calls, status = await manager.read_with_status([record], wait_first=False)
        assert status == "incomplete"
        assert [call.litellm_call_id for call in calls] == [gateway_call_id("hash-1", "good")]
        # The existing method retains its list-only, best-effort contract.
        assert await manager.read([record], wait_first=False) == []


async def test_restore_accepts_shared_gateway_key_actions(tmp_path: Path) -> None:
    from typing import get_args

    from swarm_arena._gateway.reports import GatewayKeyAction

    async with open_llm_run(tmp_path) as run:
        manager = _manager(MemoryGateway(), run)
        record, _ = await _open(manager, run)
        await run.journal.commit()
        created = run.events("gateway_key")[0]
        observations = [
            created.model_copy(update={"payload": {"action": action, "error": "gateway failed"}})
            for action in get_args(GatewayKeyAction)
            if action not in {"created", "closed"}
        ]
        assert {e.payload["action"] for e in observations} >= {"read_failed", "close_failed"}
        manager.restore([created, *observations])
        assert manager.records() == [record]
        invalid = created.model_copy(update={"payload": {"action": "unknown"}})
        with pytest.raises(ValueError, match="invalid gateway key action"):
            manager.restore([created, invalid])
        assert manager.records() == [record]


async def test_caller_cancel_interrupts_read_without_recording(tmp_path: Path) -> None:
    import asyncio

    gw = MemoryGateway()
    gw.read_release = anyio.Event()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        task = asyncio.create_task(manager.read([record], wait_first=True))
        try:
            with anyio.fail_after(5):
                await gw.read_entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert not run.calls()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_expired_open_cleanup_budget_never_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._gateway import keys

    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        monkeypatch.setattr(keys, "_CLEANUP_TIMEOUT_S", 0)
        with anyio.CancelScope() as cancelled:
            gw.on_create = cancelled.cancel
            await _open(manager, run)
        assert not gw.blocked
        assert not manager.records()
        assert not run.events("gateway_key")


async def test_open_cleanup_failure_preserves_primary_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._errors import Fenced, _core_category, _core_error, _SignalCategory

    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        gw.block_error = _core_error(
            RuntimeError, "cleanup terminal", _category=_SignalCategory.TERMINAL
        )

        def fail(draft: EventDraft) -> Event:
            raise _core_error(Fenced, "primary fenced")

        with monkeypatch.context() as patch:
            patch.setattr(run.journal, "append", fail)
            with pytest.raises(Exception, match=r"safe|ownership|primary fenced") as caught:
                await _open(manager, run)
        assert _core_category(caught.value) == _SignalCategory.FENCED
        assert not manager.records()


async def test_funded_call_collision_keeps_separate_tool_namespace(tmp_path: Path) -> None:
    from _fakes.llm_run import record_kwargs

    from swarm_arena._journal.llm_record import open_call, record_call

    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        scope = run.scope("manager")
        estimate = Amount(tokens=9000, usd=0.1)
        rid = await scope.tranche.reserve_call(estimate)
        cid = open_call(
            scope,
            reservation_id=rid,
            estimate=estimate,
            model_alias="opus",
            model="anthropic/claude",
            provider="anthropic",
        )
        meta = run.metadata(scope, reservation_id=rid, call_id=cid)
        canonical = gateway_call_id(record.key_id, "request-1")
        gw.rows = [_row()]
        with llm_scope(scope):
            funded = await record_call(**record_kwargs(meta, litellm_call_id=canonical))
            observed = await manager.read([record], wait_first=False)
        assert funded is not None
        assert len(observed) == 1
        assert scope.call_ids == [funded.id]
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)
        assert observed[0].reservation_id is None


async def test_restore_epoch_two_uses_current_owner_and_replayed_dedup(tmp_path: Path) -> None:
    from _fakes.leases import force_take_over
    from _fakes.llm_run import record_kwargs

    from swarm_arena._engine.stops import RunClock, StopState
    from swarm_arena._journal.llm_record import _bound_run, record_call
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.writer import JournalWriter
    from swarm_arena._model.ids import IdGen

    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        gw.rows = [_row(request_id="old")]
        assert len(await manager.read([record], wait_first=False)) == 1
        await run.journal.commit()
        state = run.state()
    committed = read_committed(run.store)
    assert committed is not None
    owner, events = committed
    assert owner.head is not None
    successor = force_take_over(run.store, owner_id="successor", host="local", ttl_s=60)
    async with (
        JournalWriter(
            run.store,
            successor,
            run_id=run.run_id,
            next_seq=owner.head.seq + 1,
            local_path=tmp_path / "epoch2.jsonl",
            sync_interval_s=60,
            group_window_s=0.001,
        ) as journal,
        _bound_run(
            run.run_id,
            journal=journal,
            ids=IdGen(run.run_id, journal.epoch),
            artifacts=run.artifacts,
            config=run.config,
            stops=StopState(),
            clock=RunClock(0),
            state=state,
        ),
    ):
        restored = KeyManager(
            gw,
            GatewayConfig(url="http://gateway.invalid"),
            cap="soft",
            journal=journal,
            run_id=run.run_id,
            settle_timeout_s=1,
        )
        restored.restore(events)
        assert restored.records()[0].owner_epoch == 1
        assert journal.epoch == 2
        gw.rows.append(_row(request_id="late"))
        calls = await restored.read(restored.records(), wait_first=False)
        assert len(calls) == 1
        assert calls[0].candidate_id == record.candidate_id
        assert calls[0].litellm_call_id == gateway_call_id(record.key_id, "late")
        assert await restored.read(restored.records(), wait_first=False) == []
        stale = await record_call(
            **record_kwargs(
                {
                    "run_id": run.run_id,
                    "owner_epoch": "1",
                    "goal_id": "g",
                    "role": "tool",
                    "candidate_id": record.candidate_id,
                },
                settle=False,
                _external_observation=True,
                request=None,
                response=None,
                litellm_call_id="stale",
            )
        )
        assert stale is None
        await journal.commit()
    all_events = read_committed(run.store)
    assert all_events is not None
    assert [event.epoch for event in all_events[1] if event.type == "llm_call"] == [1, 2]


async def test_compensation_block_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._gateway import keys

    entered = anyio.Event()
    ended = anyio.Event()
    gw = MemoryGateway()

    async def block(key: VirtualKey) -> None:
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            ended.set()

    monkeypatch.setattr(gw, "block_key", block)
    monkeypatch.setattr(keys, "_CLEANUP_TIMEOUT_S", 0.1)
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        with anyio.fail_after(5):
            with anyio.CancelScope() as cancelled:
                gw.on_create = cancelled.cancel
                await _open(manager, run)
        assert entered.is_set()
        assert ended.is_set()
        assert not manager.records()
        assert not run.events("gateway_key")


async def test_empty_records_and_duplicate_rows_end_polling(tmp_path: Path) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        assert await manager.read([], wait_first=True) == []
        assert not gw.windows
        record, _ = await _open(manager, run)
        gw.rows = [_row()]
        assert len(await manager.read([record], wait_first=False)) == 1
        before = len(gw.windows)
        assert await manager.read([record], wait_first=True) == []
        assert len(gw.windows) == before + 1


async def test_read_after_fencing_refuses_before_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._errors import Fenced, _core_error

    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)

        def fail() -> None:
            raise _core_error(Fenced, "owner gone")

        with monkeypatch.context() as patch:
            patch.setattr(run.journal, "_check_open", fail)
            with pytest.raises(Fenced, match="owner gone"):
                await manager.read([record], wait_first=False)
            await manager.revoke(record)
        assert not gw.windows
        assert gw.blocked == [record.key_id]


@pytest.mark.parametrize("boundary", [False, True])
async def test_read_timer_does_not_bound_healthy_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: bool
) -> None:
    from swarm_arena._gateway import keys

    entered, release = anyio.Event(), anyio.Event()
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run, settle_timeout_s=0.1)
        record, _ = await _open(manager, run)
        gw.rows = [_row()]
        original = keys.record_call

        async def held(**kwargs: Any) -> Any:
            entered.set()
            await release.wait()
            return await original(**kwargs)

        monkeypatch.setattr(keys, "record_call", held)

        async def release_after_report_deadline() -> None:
            await entered.wait()
            with anyio.move_on_after(0.2):
                await anyio.sleep_forever()
            release.set()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(release_after_report_deadline)
            with anyio.fail_after(30):
                token = keys._read_deadline.set(anyio.current_time() + 0.1)
                try:
                    calls = await manager.read([record], wait_first=not boundary)
                finally:
                    keys._read_deadline.reset(token)
        assert len(calls) == 1


def test_gateway_job_preserves_existing_drops() -> None:
    job = gateway_job(make_job(drop_env=("CUSTOM",)), url="http://gw", secret="sk-v")
    assert job.drop_env == tuple(sorted(PROVIDER_KEYS | {"CUSTOM"}))


@pytest.mark.parametrize(
    ("model", "expected_model", "expected_provider"),
    [
        ("openai/" + "x" * 300, "unknown", "unknown"),
        ("openai/gpt\n6", "unknown", "unknown"),
        ("x" * 65 + "/model", "x" * 65 + "/model", "unknown"),
        ("", "unknown", "unknown"),
        ("   ", "unknown", "unknown"),
        (None, "unknown", "unknown"),
    ],
)
async def test_poison_model_does_not_starve_later_rows(
    tmp_path: Path, model: Any, expected_model: str, expected_provider: str
) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        gw.rows = [
            _row(request_id="good"),
            _row(request_id="poison").model_copy(update={"model": model}),
            _row(request_id="late"),
        ]
        calls = await manager.read([record], wait_first=False)
        assert [call.litellm_call_id for call in calls] == [
            gateway_call_id(record.key_id, name) for name in ("good", "poison", "late")
        ]
        assert (calls[1].model, calls[1].provider) == (expected_model, expected_provider)
        gw.rows.append(_row(request_id="next"))
        following = await manager.read([record], wait_first=False)
        assert [call.litellm_call_id for call in following] == [
            gateway_call_id(record.key_id, "next")
        ]


async def test_row_recording_error_warns_once_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from swarm_arena._gateway import keys

    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        gw.rows = [_row(request_id=name) for name in ("good", "poison", "late")]
        original = keys.record_call

        async def reject_poison(**kwargs: Any) -> Any:
            if kwargs["litellm_call_id"] == gateway_call_id(record.key_id, "poison"):
                raise ValueError(SECRET * 10000)
            return await original(**kwargs)

        monkeypatch.setattr(keys, "record_call", reject_poison)
        calls = await manager.read([record], wait_first=False)
        assert [call.litellm_call_id for call in calls] == [
            gateway_call_id(record.key_id, name) for name in ("good", "late")
        ]
        assert [entry.getMessage() for entry in caplog.records] == [
            "gateway spend row recording failed"
        ]
        caplog.clear()
        gw.rows.append(_row(request_id="next"))
        following = await manager.read([record], wait_first=False)
        assert [call.litellm_call_id for call in following] == [
            gateway_call_id(record.key_id, "next")
        ]
        assert [entry.getMessage() for entry in caplog.records] == [
            "gateway spend row recording failed"
        ]
        assert SECRET not in caplog.text


@pytest.mark.parametrize("skew_s", [-3, 3])
async def test_spend_window_includes_gateway_clock_skew(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skew_s: int
) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        gw.rows = [_row().model_copy(update={"ts": record.created_at + timedelta(seconds=skew_s)})]
        original = gw.spend_rows

        async def dated(key: VirtualKey, *, start: datetime, end: datetime) -> list[SpendRow]:
            rows = await original(key, start=start, end=end)
            return [row for row in rows if start <= row.ts <= end]

        monkeypatch.setattr(gw, "spend_rows", dated)
        before = datetime.now(UTC)
        assert len(await manager.read([record], wait_first=False)) == 1
        start, end = gw.windows[0]
        assert start == record.created_at.replace(microsecond=0) - timedelta(minutes=15, seconds=1)
        assert end >= before + timedelta(minutes=15)


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("restored", [False, True])
async def test_open_refuses_owned_candidate_before_create(
    tmp_path: Path, closed: bool, restored: bool
) -> None:
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run, candidate_id=SECRET)
        if closed:
            await manager.close(record)
        if restored:
            await run.journal.commit()
            manager = _manager(gw, run)
            manager.restore(run.events("gateway_key"))
        with pytest.raises(ValueError, match="candidate already has a gateway key") as caught:
            await _open(manager, run, candidate_id=SECRET)
        assert_secret_absent(caught.value, SECRET)
        assert len(gw.created) == 1


@pytest.mark.parametrize("blocked", [False, True])
def test_fake_gateway_refuses_duplicate_alias(blocked: bool) -> None:
    from _fakes.gateway import FakeGateway

    gateway = FakeGateway()
    body = {"key_alias": "cand-one", "metadata": {}, "default_estimated_output_tokens": 4096}
    authorization = f"Bearer {gateway.master_key}"
    status, created, _ = gateway._route("POST", "/key/generate", body, authorization)
    assert status == 200
    if blocked:
        assert (
            gateway._route("POST", "/key/block", {"key": created["token_id"]}, authorization)[0]
            == 200
        )
    status, error, _ = gateway._route("POST", "/key/generate", body, authorization)
    assert status == 400
    assert error == {"error": "key_alias already exists"}
    assert len(gateway.keys) == 1


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("signal", ["fenced", "terminal"])
async def test_row_recording_preserves_registered_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signal: str, nested: bool
) -> None:
    from swarm_arena._errors import Fenced, _core_error, _is_core_control, _SignalCategory
    from swarm_arena._gateway import keys

    error = (
        _core_error(Fenced, "safe")
        if signal == "fenced"
        else _core_error(RuntimeError, "safe", _category=_SignalCategory.TERMINAL)
    )
    raised = BaseExceptionGroup(SECRET, [error]) if nested else error
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        gw.rows = [_row()]

        async def fail(**kwargs: Any) -> Any:
            raise raised

        monkeypatch.setattr(keys, "record_call", fail)
        with pytest.raises(Exception, match=r"safe|ownership") as caught:
            await manager.read([record], wait_first=False)
        assert _is_core_control(caught.value)
        assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("signal", ["keyboard", "system", "generator", "cancellation"])
async def test_row_recording_preserves_base_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    signal: str,
) -> None:
    from swarm_arena._gateway import keys

    error_type = {
        "keyboard": KeyboardInterrupt,
        "system": SystemExit,
        "generator": GeneratorExit,
        "cancellation": anyio.get_cancelled_exc_class(),
    }[signal]
    error = error_type("row interrupted")
    attempted: list[str] = []
    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        gw.rows = [_row(request_id=name) for name in ("interrupted", "late")]

        async def interrupt(**kwargs: Any) -> Any:
            attempted.append(kwargs["litellm_call_id"])
            raise error

        monkeypatch.setattr(keys, "record_call", interrupt)
        with pytest.raises(error_type) as caught:
            await manager.read([record], wait_first=False)
        assert caught.value is error
        assert attempted == [gateway_call_id(record.key_id, "interrupted")]
        assert "gateway spend row recording failed" not in caplog.messages


async def test_row_recording_preserves_caller_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._gateway import keys

    gw = MemoryGateway()
    async with open_llm_run(tmp_path) as run:
        manager = _manager(gw, run)
        record, _ = await _open(manager, run)
        gw.rows = [_row()]
        with anyio.CancelScope() as cancelled:

            async def fail(**kwargs: Any) -> Any:
                cancelled.cancel()
                raise ValueError(SECRET)

            monkeypatch.setattr(keys, "record_call", fail)
            await manager.read([record], wait_first=False)
            pytest.fail("row error swallowed caller cancellation")
        assert cancelled.cancelled_caught


def test_fake_gateway_deleted_alias_can_be_reused() -> None:
    from _fakes.gateway import FakeGateway

    gateway = FakeGateway()
    body = {"key_alias": "cand-one", "metadata": {}, "default_estimated_output_tokens": 4096}
    authorization = f"Bearer {gateway.master_key}"
    _, created, _ = gateway._route("POST", "/key/generate", body, authorization)
    status, _, _ = gateway._route(
        "POST", "/key/delete", {"keys": [created["token_id"]]}, authorization
    )
    assert status == 200
    assert gateway._route("POST", "/key/generate", body, authorization)[0] == 200


async def test_restore_accepts_engine_elapsed_metadata(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        manager = _manager(MemoryGateway(), run)
        record, _ = await _open(manager, run)
        await run.journal.commit()
        created = run.events("gateway_key")[0]
        stamped = created.model_copy(
            update={
                "payload": {**created.payload, "run_elapsed_s": 2.5},
            }
        )
        manager.restore([stamped])
        assert manager.records() == [record]
