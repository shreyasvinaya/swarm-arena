import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.cleanup import observe_task, settle_tasks
from _fakes.llm_run import HeldOperation, LLMRun, open_llm_run, record_kwargs
from _fakes.secrecy import assert_secret_absent

from swarm_arena import RunConfig
from swarm_arena._budget.amount import USD_LIMIT, Amount
from swarm_arena._budget.llm_scope import LLMScope, llm_scope
from swarm_arena._errors import Fenced, StorageError
from swarm_arena._journal.llm_record import (
    abandon_call,
    bind_run,
    drain,
    open_call,
    pending_scope,
    record_call,
    unbind_run,
)
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.results import LLMCall
from swarm_arena._model.settings import TracingConfig
from swarm_arena.tool import Usage

ESTIMATE = Amount(tokens=9_000, usd=0.01)


async def test_maintenance_binding_records_and_deduplicates_external_observations(
    tmp_path: Path,
) -> None:
    from swarm_arena._journal import llm_record

    async with open_llm_run(tmp_path) as run:
        meta = {
            "run_id": run.run_id,
            "owner_epoch": str(run.journal.epoch),
            "role": "tool",
            "goal_id": "g",
            "candidate_id": "c",
        }
        kwargs = record_kwargs(
            meta,
            request=None,
            response=None,
            settle=False,
            _external_observation=True,
            litellm_call_id="gw:earlier",
        )
        assert await record_call(**kwargs) is not None
        await run.journal.commit()
        state = run.state()
        unbind_run(run.run_id, journal=run.journal)
        run.clock._rebase(run.config.limits.run_timeout_s + 100)
        async with llm_record._bound_run(
            run.run_id,
            journal=run.journal,
            ids=run.ids,
            artifacts=run.artifacts,
            config=run.config,
            stops=run.stops,
            clock=None,
            state=state,
            _maintenance_lease=run.journal._lease,
        ):
            binding = llm_record._bindings[run.run_id]
            assert binding.clock is None
            assert await record_call(**kwargs) is None
            kwargs["litellm_call_id"] = "gw:late"
            call = await record_call(**kwargs)
            assert call is not None
            assert call.role == "tool"
            assert call.reservation_id is None
            assert binding.scopes == {}
            assert binding.pending == {}
            await run.journal.commit()
            assert len(run.calls()) == 2
            assert run.events("settle") == []
        assert run.run_id not in llm_record._bindings
        assert await record_call(**kwargs) is None


@pytest.mark.parametrize("invalid", ["funded", "settle", "reservation", "open_call"])
async def test_maintenance_binding_refuses_funding(tmp_path: Path, invalid: str) -> None:
    from swarm_arena._journal import llm_record

    async with open_llm_run(tmp_path) as run:
        state = run.state()
        unbind_run(run.run_id, journal=run.journal)
        async with llm_record._bound_run(
            run.run_id,
            journal=run.journal,
            ids=run.ids,
            artifacts=run.artifacts,
            config=run.config,
            stops=run.stops,
            clock=None,
            state=state,
            _maintenance_lease=run.journal._lease,
        ):
            meta = {
                "run_id": run.run_id,
                "owner_epoch": str(run.journal.epoch),
                "role": "tool",
                "goal_id": "g",
            }
            if invalid == "reservation":
                meta["reservation_id"] = "SECRET"
            kwargs = record_kwargs(
                meta,
                settle=invalid == "settle",
                _external_observation=invalid != "funded",
                litellm_call_id="gw:one",
            )

            async def attempt() -> None:
                if invalid == "open_call":
                    open_call(
                        run.scope("judge"),
                        reservation_id="SECRET",
                        estimate=ESTIMATE,
                        model_alias="opus",
                        model="local",
                        provider="local",
                    )
                else:
                    await record_call(**kwargs)

            with pytest.raises(ValueError, match="maintenance binding") as caught:
                await attempt()
            assert_secret_absent(caught.value, "SECRET")
            await run.journal.commit()
            assert run.calls() == []
            assert run.events("settle") == []


@pytest.mark.parametrize("invalid", ["missing", "unentered", "expired", "different"])
async def test_maintenance_binding_requires_held_writer_lease(
    tmp_path: Path,
    invalid: str,
) -> None:
    from datetime import timedelta

    from swarm_arena._journal import llm_record
    from swarm_arena._journal.lease import Lease
    from swarm_arena._model.canonical import utc_now

    async with open_llm_run(tmp_path) as run:
        unbind_run(run.run_id, journal=run.journal)
        lease = run.journal._lease
        doc = lease.doc
        supplied = lease
        if invalid == "missing":
            supplied = None
        elif invalid == "unentered":
            run.journal._entered = False
        elif invalid == "expired":
            lease.doc = doc.model_copy(update={"expires_at": utc_now() - timedelta(seconds=1)})
        else:
            supplied = Lease(run.store, lease.doc, lease.etag, ttl_s=lease.ttl_s)
        try:
            with pytest.raises((RuntimeError, ValueError), match=r"maintenance binding|not active"):
                async with llm_record._bound_run(
                    run.run_id,
                    journal=run.journal,
                    ids=run.ids,
                    artifacts=run.artifacts,
                    config=run.config,
                    stops=run.stops,
                    clock=None,
                    state=run.state(),
                    _maintenance_lease=supplied,
                ):
                    pytest.fail("invalid maintenance binding entered")
            assert run.run_id not in llm_record._bindings
        finally:
            run.journal._entered = True
            lease.doc = doc


async def test_graceful_report_timeout_survives_delayed_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from threading import Timer

    from swarm_arena._journal import llm_record

    held = HeldOperation()
    release: Timer | None = None
    try:
        async with open_llm_run(tmp_path) as run:
            # Only the deliberately missing report gets a short deadline.
            scope, meta = await _open(run, replace(run.scope("manager"), settle_timeout_s=2))
            watch_owner = llm_record._watch_owner
            report_watchers: set[asyncio.Task[Any]] = set()

            async def observed_watcher(
                binding: Any, owner: anyio.CancelScope, interrupt_reports: bool = False
            ) -> None:
                task = asyncio.current_task()
                assert task is not None
                if interrupt_reports:
                    report_watchers.add(task)
                try:
                    await watch_owner(binding, owner, interrupt_reports)
                finally:
                    report_watchers.discard(task)

            class DelayedPolling:
                def __getattr__(self, name: str) -> Any:
                    return getattr(anyio, name)

                async def sleep(self, delay: float) -> None:
                    nonlocal release
                    if asyncio.current_task() in report_watchers and release is None:
                        # Hold the report watcher's first poll across expiry,
                        # before the report timer can deliver its cancellation.
                        release = Timer(scope.settle_timeout_s + 0.1, held.release.set)
                        release.start()
                        held.wait()
                        return
                    await anyio.sleep(delay)

            monkeypatch.setattr(llm_record, "_watch_owner", observed_watcher)
            monkeypatch.setattr(llm_record, "anyio", DelayedPolling())
            run.stops.stop = True
            assert await drain(scope, wait=True) == [meta["call_id"]]
            assert held.entered.is_set()
            assert held.exited.is_set()
            assert llm_record._bindings[run.run_id].terminal is None
            assert not run.journal._settlement_expired
            call = run.calls()[0]
            assert call.error is not None
            assert call.error.startswith("estimated:")
            assert run.charged(scope) == ESTIMATE
    finally:
        held.release.set()
        if release is not None:
            release.cancel()
            release.join(60)
            assert not release.is_alive()


async def test_stop_now_interrupts_drain_report_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._journal import llm_record

    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        entered, returned = anyio.Event(), anyio.Event()
        watch_owner, await_reports = llm_record._watch_owner, llm_record.await_reports

        async def observed_watcher(
            binding: Any, owner: anyio.CancelScope, interrupt_reports: bool = False
        ) -> None:
            if interrupt_reports:
                entered.set()
            await watch_owner(binding, owner, interrupt_reports)

        async def observed_reports(scope: LLMScope) -> None:
            await await_reports(scope)
            # A report-timer expiry cancels this frame before reaching here.
            returned.set()

        monkeypatch.setattr(llm_record, "_watch_owner", observed_watcher)
        monkeypatch.setattr(llm_record, "await_reports", observed_reports)
        task = asyncio.create_task(drain(scope, wait=True))
        try:
            with anyio.fail_after(60):
                await entered.wait()
                run.stops.stop_now = True
                assert await observe_task(task) == [meta["call_id"]]
            assert returned.is_set()
            assert llm_record._bindings[run.run_id].terminal is None
            call = run.calls()[0]
            assert call.error is not None
            assert call.error.startswith("estimated:")
            assert run.charged(scope) == ESTIMATE
        finally:
            task.cancel()
            await settle_tasks(task)


async def test_await_reports_requires_finite_deadline(tmp_path: Path) -> None:
    from swarm_arena._errors import _is_core_signal
    from swarm_arena._journal.llm_record import await_reports

    async with open_llm_run(tmp_path) as run:
        # An empty scope makes a missing guard return immediately, not hang.
        scope = run.scope("curator", agent="PRIVATE_UNBOUNDED_REPORT")
        with (
            anyio.CancelScope(shield=True),
            pytest.raises(
                RuntimeError, match=r"^report wait requires a finite deadline$"
            ) as caught,
        ):
            await await_reports(scope)
        assert _is_core_signal(caught.value)
        assert_secret_absent(caught.value, "PRIVATE_UNBOUNDED_REPORT")


async def test_await_reports_preserves_pending_caller_cancellation(tmp_path: Path) -> None:
    from swarm_arena._journal.llm_record import await_reports

    async with open_llm_run(tmp_path) as run:
        scope, _ = await _open(run)
        with anyio.move_on_after(0) as cancelled:
            await await_reports(scope)
        assert cancelled.cancelled_caught
        await drain(scope, wait=False)


async def test_empty_drained_scope_cannot_admit_another_call(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("planner", team="", agent="", round=0)
        assert await drain(scope, wait=False) == []
        with pytest.raises(ValueError, match="drained"):
            open_call(
                scope,
                reservation_id="late",
                estimate=Amount(),
                model_alias="opus",
                model="anthropic/claude",
                provider="anthropic",
            )


async def test_late_unbound_gateway_report_is_ignored() -> None:
    assert (
        await record_call(
            **record_kwargs({"run_id": "expired"}, litellm_call_id="gateway-row", settle=False)
        )
        is None
    )


async def test_planner_fixture_records_without_immediate_settlement(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run, run.scope("planner", team="", agent="", round=0))
        call = await record_call(**record_kwargs(meta, settle=False))
        await run.journal.commit()
        assert call is not None
        assert call.role == "planner"
        assert call.round == 0
        assert run.charged(scope) == ESTIMATE
        assert run.events("settle") == []
        assert await drain(scope, wait=False) == []
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)


async def _open(
    run: LLMRun, scope: LLMScope | None = None, *, estimate: Amount = ESTIMATE
) -> tuple[LLMScope, dict[str, str]]:
    scope = scope or run.scope("manager")
    rid = await scope.tranche.reserve_call(estimate)
    cid = open_call(
        scope,
        reservation_id=rid,
        estimate=estimate,
        model_alias="opus",
        model="anthropic/claude",
        provider="anthropic",
    )
    return scope, run.metadata(scope, reservation_id=rid, call_id=cid)


async def test_record_call_journals_one_llm_call_event(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        with llm_scope(scope):
            call = await record_call(**record_kwargs(meta, litellm_call_id="lc-1"))
        assert call is not None
        assert call.id == meta["call_id"]
        assert call.id.startswith("llm-")
        assert [e.payload["call"]["id"] for e in run.events("llm_call")] == [call.id]
        assert call.tool_calls == [{"id": "call_1", "name": "run_tool"}]
        assert call.model_alias == "opus"
        assert scope.call_ids == [call.id]


async def test_payloads_stored_by_sha256(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        _, meta = await _open(run)
        kwargs = record_kwargs(meta)
        call = await record_call(**kwargs)
        assert call is not None
        for value, ref in [
            (kwargs["request"], call.request_ref),
            (kwargs["response"], call.response_ref),
        ]:
            assert ref == sha256_hex(canonical_json(value))
            assert run.artifacts.get_bytes(ref) == canonical_json(value)


async def test_store_payloads_false_keeps_refs_none(tmp_path: Path) -> None:
    config = RunConfig(tracing=TracingConfig(store_payloads=False))
    async with open_llm_run(tmp_path, config=config) as run:
        _, meta = await _open(run)
        call = await record_call(**record_kwargs(meta))
        assert call is not None
        assert (call.request_ref, call.response_ref) == (None, None)


async def test_settles_by_reservation_id_outside_any_scope(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        await record_call(**record_kwargs(meta))
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)


async def test_cache_hit_costs_zero_and_points_to_original(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        first = await record_call(**record_kwargs(meta, cache_identity="b" * 64))
        _, meta = await _open(run, scope)
        second = await record_call(
            **record_kwargs(meta, cache_hit=True, cost_usd=0.5, cache_identity="b" * 64)
        )
        assert first is not None
        assert second is not None
        assert (second.cost_usd, second.cached_from_call_id) == (0.0, first.id)
        assert second.usage == first.usage
        assert run.charged(scope) == Amount(tokens=120, usd=first.cost_usd)


async def test_known_litellm_call_id_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        _, meta = await _open(run)
        kwargs = record_kwargs(meta, litellm_call_id="lc-1")
        await record_call(**kwargs)

        def forbidden(_: bytes) -> str:
            pytest.fail("known duplicate uploaded a payload")

        monkeypatch.setattr(run.artifacts, "put_bytes", forbidden)
        assert await record_call(**kwargs) is None
        assert len(run.events("llm_call")) == 1


async def test_drain_charges_unrecorded_call_in_full(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        # Only the deliberately missing report gets a short deadline.
        scope, meta = await _open(run, replace(run.scope("manager"), settle_timeout_s=2))
        assert await drain(scope, wait=True) == [meta["call_id"]]
        await run.journal.commit()
        (placeholder,) = run.calls()
        assert placeholder.id == meta["call_id"]
        assert scope.call_ids == [placeholder.id]
        assert placeholder.status == "error"
        assert placeholder.error is not None
        assert placeholder.error.startswith("estimated:")
        assert placeholder.cost_usd == 0.01
        assert run.charged(scope) == ESTIMATE


async def test_late_record_after_drain_writes_nothing(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        await drain(scope, wait=False)
        assert await record_call(**record_kwargs(meta)) is None
        assert len(run.calls()) == 1
        with pytest.raises(ValueError, match="drained"):
            open_call(
                scope,
                reservation_id="late",
                estimate=ESTIMATE,
                model_alias="m",
                model="m",
                provider="p",
            )


async def test_drain_without_wait_charges_at_once(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        # No callback releases the pending report; its backstop outlives the observation.
        scope, _ = await _open(run, replace(run.scope("manager"), settle_timeout_s=300))
        with anyio.fail_after(30):
            await drain(scope, wait=False)
        assert len(run.calls()) == 1


async def test_drain_overtaking_a_suspended_record_writes_one_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        # Only the deliberately missing report gets a short deadline.
        scope, meta = await _open(run, replace(run.scope("manager"), settle_timeout_s=2))
        held = HeldOperation()
        original = run.artifacts.put_bytes
        uploads: list[bytes] = []
        results: list[LLMCall | None] = []

        def upload(data: bytes) -> str:
            uploads.append(data)
            held.wait()
            return original(data)

        async def callback() -> None:
            results.append(await record_call(**record_kwargs(meta)))

        monkeypatch.setattr(run.artifacts, "put_bytes", upload)
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(callback)
                await held.ready()
                assert await drain(scope, wait=True) == [meta["call_id"]]
            assert results == [None]
            assert len(run.events("llm_call")) == 1
            assert scope.call_ids == [meta["call_id"]]
            assert run.charged(scope) == ESTIMATE
        finally:
            await held.finish()
        assert len(uploads) == 1


async def test_concurrent_duplicate_callbacks_record_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        held = HeldOperation()
        original = run.artifacts.put_bytes
        results: list[LLMCall | None] = []

        def upload(data: bytes) -> str:
            held.wait()
            return original(data)

        async def callback() -> None:
            results.append(await record_call(**record_kwargs(meta, litellm_call_id="same")))

        monkeypatch.setattr(run.artifacts, "put_bytes", upload)
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(callback)
                group.start_soon(callback)
                await held.ready()
                held.release.set()
        finally:
            await held.finish()
        assert len(run.events("llm_call")) == 1
        assert sum(result is not None for result in results) == 1
        assert scope.call_ids == [meta["call_id"]]


async def test_drain_does_not_wait_for_an_abandoned_call(tmp_path: Path) -> None:
    from swarm_arena._journal.llm_record import _bindings

    async with open_llm_run(tmp_path) as run:
        # An abandoned report remains unavailable for the entire observation window.
        scope, meta = await _open(run, replace(run.scope("manager"), settle_timeout_s=300))
        abandon_call(scope, meta["reservation_id"])
        entry = _bindings[run.run_id].pending[meta["reservation_id"]]
        entry.event = anyio.Event()
        with anyio.fail_after(30):
            placeholders = await drain(scope, wait=True)
        assert placeholders == [meta["call_id"]]
        assert len(run.calls()) == 1


async def test_settle_above_the_estimate_charges_the_actual(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run, estimate=Amount(tokens=100, usd=0.001))
        await record_call(
            **record_kwargs(meta, usage=Usage(input_tokens=400, output_tokens=100), cost_usd=0.005)
        )
        assert run.charged(scope) == Amount(tokens=500, usd=0.005)


async def test_charge_reservation_settles_the_estimate(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        call = await record_call(
            **record_kwargs(
                meta,
                status="error",
                error="PROVIDER_SECRET",
                usage=Usage(),
                cost_usd=0.0,
                charge_reservation=True,
            )
        )
        assert call is not None
        assert (call.usage, call.cost_usd) == (Usage(input_tokens=9_000), 0.01)
        assert call.error == "estimated: provider spend unknown; charged in full"
        assert run.charged(scope) == ESTIMATE


async def test_cache_hit_and_unknown_spend_flags_are_rejected(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        with pytest.raises(ValueError, match="LLM record preparation failed") as caught:
            await record_call(
                **record_kwargs(
                    meta,
                    cache_hit=True,
                    charge_reservation=True,
                    status="error",
                    error="CONTRADICTORY_FLAGS_SECRET",
                    usage=Usage(),
                    cost_usd=0,
                )
            )
        assert_secret_absent(caught.value, "CONTRADICTORY_FLAGS_SECRET")
        await run.journal.commit()
        assert run.calls() == []
        assert run.charged(scope) == ESTIMATE
        with pytest.raises(ValueError, match="LLM record preparation failed"):
            await drain(scope, wait=False)
        (placeholder,) = run.calls()
        assert placeholder.id == meta["call_id"]
        assert placeholder.cache_hit is False
        assert run.charged(scope) == ESTIMATE
        assert list(run.state().llm_calls) == [meta["call_id"]]


async def test_settle_error_after_claim_wakes_drain_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        settle = scope.tranche.settle_call

        async def broken(*_: Any) -> Any:
            raise StorageError("SETTLE_SECRET")

        monkeypatch.setattr(scope.tranche, "settle_call", broken)
        with pytest.raises(StorageError) as caught:
            await record_call(**record_kwargs(meta))
        assert_secret_absent(caught.value, "SETTLE_SECRET")
        monkeypatch.setattr(scope.tranche, "settle_call", settle)
        with anyio.fail_after(60), pytest.raises(StorageError):
            await drain(scope, wait=True)
        assert len(run.events("llm_call")) == 1
        assert scope.call_ids == [meta["call_id"]]
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)


async def test_cancel_during_settle_wakes_drain_and_settles_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        entered, release = anyio.Event(), anyio.Event()
        settle = scope.tranche.settle_call
        cancel = anyio.CancelScope()

        async def delayed(*args: Any) -> Any:
            entered.set()
            await release.wait()
            return await settle(*args)

        async def callback() -> None:
            with cancel:
                await record_call(**record_kwargs(meta))

        monkeypatch.setattr(scope.tranche, "settle_call", delayed)
        async with anyio.create_task_group() as group:
            group.start_soon(callback)
            await entered.wait()
            cancel.cancel()
            release.set()
        with anyio.fail_after(60):
            assert await drain(scope, wait=True) == []
        assert len(run.events("llm_call")) == 1
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)


async def test_unbound_run_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not bound"):
        await record_call(**record_kwargs({"run_id": "nope"}))


async def test_round_metadata_is_read_as_int(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        _, meta = await _open(run)
        call = await record_call(**record_kwargs(meta))
        assert call is not None
        assert call.round == 1
        assert isinstance(meta["round"], str)


async def test_pending_scope_finds_the_scope_by_reservation_id(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        assert pending_scope(meta) is scope
        assert pending_scope({"run_id": run.run_id, "reservation_id": "r-unknown"}) is None


async def test_journals_request_sha256_without_payloads(tmp_path: Path) -> None:
    async with open_llm_run(
        tmp_path, config=RunConfig(tracing=TracingConfig(store_payloads=False))
    ) as run:
        _, meta = await _open(run)
        kwargs = record_kwargs(meta)
        await record_call(**kwargs)
        assert run.events("llm_call")[0].payload["request_sha256"] == sha256_hex(
            canonical_json(kwargs["request"])
        )


@pytest.mark.parametrize(
    "change", [{"latency_ms": float("nan")}, {"cost_usd": float("inf")}, {"status": "SECRET"}]
)
async def test_invalid_callback_leaves_recoverable_reservation(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        with pytest.raises(ValueError, match=r"LLM .* failed") as caught:
            await record_call(**record_kwargs(meta, **change))
        assert_secret_absent(caught.value, "SECRET")
        with pytest.raises(ValueError, match=r"LLM .* failed"):
            await drain(scope, wait=False)
        assert len(run.calls()) == 1
        assert run.calls()[0].id == meta["call_id"]
        assert run.charged(scope) == ESTIMATE


async def test_append_failure_can_be_recovered_without_losing_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        append = run.journal.append

        def fail(_: Any) -> Any:
            raise ValueError("APPEND_SECRET")

        with monkeypatch.context() as patch:
            patch.setattr(run.journal, "append", fail)
            with pytest.raises(ValueError, match=r"LLM .* failed") as caught:
                await record_call(**record_kwargs(meta))
        assert_secret_absent(caught.value, "APPEND_SECRET")
        assert run.journal.append == append
        with pytest.raises(ValueError, match=r"LLM .* failed"):
            await drain(scope, wait=False)
        assert len(run.calls()) == 1
        assert run.charged(scope) == ESTIMATE


@pytest.mark.parametrize("reservation", [None, "UNMATCHED_RESERVATION_SECRET", "other"])
async def test_preallocated_call_requires_matching_reservation(
    tmp_path: Path, reservation: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, original = await _open(run)
        other_scope, other = await _open(run)
        meta = dict(original)
        if reservation is None:
            del meta["reservation_id"]
        else:
            meta["reservation_id"] = (
                other["reservation_id"] if reservation == "other" else reservation
            )

        def forbidden(_: bytes) -> str:
            pytest.fail("malformed identity uploaded a payload")

        with monkeypatch.context() as patch:
            patch.setattr(run.artifacts, "put_bytes", forbidden)
            with pytest.raises(ValueError, match="matching reservation") as caught:
                await record_call(**record_kwargs(meta))
        assert_secret_absent(caught.value, "UNMATCHED_RESERVATION_SECRET")
        assert run.calls() == []
        assert await drain(scope, wait=False) == [original["call_id"]]
        assert scope.tranche.pending() == 0
        assert other_scope.tranche.pending() == 1
        assert run.charged(scope) == ESTIMATE
        assert await drain(other_scope, wait=False) == [other["call_id"]]
        assert list(run.state().llm_calls) == [original["call_id"], other["call_id"]]


async def test_drain_does_not_duplicate_an_already_journaled_id(tmp_path: Path) -> None:
    from swarm_arena._journal.llm_record import _bindings

    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        await record_call(**record_kwargs(meta, settle=False))
        # Defend the journal identity even if pending bookkeeping is inconsistent.
        pending = _bindings[run.run_id].pending[meta["reservation_id"]]
        pending.journaled = False
        try:
            with pytest.raises(ValueError, match="already journaled"):
                await drain(scope, wait=False)
            await run.journal.commit()
            assert list(run.state().llm_calls) == [meta["call_id"]]
        finally:
            pending.journaled = True


@pytest.mark.parametrize("batched", [False, True])
async def test_shared_tranche_closes_after_all_scopes_drain(
    tmp_path: Path, batched: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._budget.tranche import Tranche

    async with open_llm_run(tmp_path) as run:
        first = run.scope("judge", team="", agent="")
        if batched:
            first = replace(
                first,
                tranche=Tranche(run.ledger, ":run", "g", size=Amount(tokens=25_000, usd=0.25)),
            )
        second = replace(first, round=2, call_ids=[])
        closed_pending: list[int] = []
        close = first.tranche.close

        async def tracked_close() -> Any:
            closed_pending.append(first.tranche.pending())
            return await close()

        monkeypatch.setattr(first.tranche, "close", tracked_close)
        for scope in (first, second):
            _, meta = await _open(run, scope)
            await record_call(**record_kwargs(meta, settle=False))
        await run.journal.commit()
        assert first.tranche.pending() == 2
    assert run.charged(first) == Amount(tokens=240, usd=0.004)
    assert sum(Amount(usd=call.cost_usd)._micro_usd for call in run.calls()) == 4000
    assert first.tranche.pending() == 0
    assert closed_pending == [0]


@pytest.mark.parametrize("batched", [False, True])
async def test_scope_drain_leaves_other_shared_scope_live(tmp_path: Path, batched: bool) -> None:
    from swarm_arena._budget.tranche import Tranche

    async with open_llm_run(tmp_path) as run:
        first = run.scope("judge", team="", agent="")
        if batched:
            first = replace(
                first,
                tranche=Tranche(run.ledger, ":run", "g", size=Amount(tokens=25_000, usd=0.25)),
            )
        second = replace(first, round=2, call_ids=[])
        _, first_meta = await _open(run, first)
        _, second_meta = await _open(run, second)
        await record_call(**record_kwargs(first_meta, settle=False))
        assert await drain(first, wait=False) == []
        assert first.tranche.pending() == 1
        assert pending_scope(second_meta) is second
        assert all(
            event.payload["id"] != second_meta["reservation_id"] for event in run.events("settle")
        )
        assert await record_call(**record_kwargs(second_meta)) is not None
    assert run.charged(first) == Amount(tokens=240, usd=0.004)


async def test_money_is_normalized_once_in_record_and_settlement(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("manager")
        for amount in (0.0000006, 0.0000015, 0.0000025):
            _, meta = await _open(run, scope)
            call = await record_call(**record_kwargs(meta, cost_usd=amount))
            assert call is not None
            assert call.cost_usd == Amount(usd=amount).usd
        assert run.charged(scope)._micro_usd == 5
        assert sum(Amount(usd=c.cost_usd)._micro_usd for c in run.calls()) == 5


async def test_overflow_is_terminal_even_with_unlimited_budget(tmp_path: Path) -> None:
    from swarm_arena.manager import LimitReached

    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        with pytest.raises(LimitReached) as caught:
            await record_call(**record_kwargs(meta, cost_usd=USD_LIMIT + 1))
        assert caught.value.limit == "usage_overflow"
        assert run.calls()[0].error == "usage_overflow: charged at the supported USD limit"
        assert run.calls()[0].cost_usd == USD_LIMIT
        assert run.events("settle")[0].payload["usage_overflow"] is True
        with pytest.raises(LimitReached) as later:
            await scope.tranche.reserve_call(Amount(usd=0.001))
        assert later.value.limit == "usage_overflow"
        # A handled terminal signal is not raised repeatedly by final cleanup.
        await drain(scope, wait=False)


async def test_cache_identity_is_replayed_and_never_guessed_from_request(tmp_path: Path) -> None:
    async with open_llm_run(
        tmp_path, config=RunConfig(tracing=TracingConfig(store_payloads=False))
    ) as run:
        scope = run.scope("manager")
        originals: list[LLMCall] = []
        for identity in ("a" * 64, "b" * 64, "c" * 64):
            _, meta = await _open(run, scope)
            call = await record_call(**record_kwargs(meta, cache_identity=identity))
            assert call is not None
            originals.append(call)
        state = run.state()
        unbind_run(run.run_id)
        bind_run(
            run.run_id,
            journal=run.journal,
            ids=run.ids,
            artifacts=run.artifacts,
            tracing=run.config.tracing,
            committed=list(state.llm_calls.values()),
            request_sha256=state.llm_request_sha256,
            cache_identity=state.llm_cache_identity,
            stops=run.stops,
            clock=run.clock,
            limits=run.config.limits,
            settle_timeout_s=run.config.settle_timeout_s,
        )
        try:
            # Identities include the effective key, namespace, and generating response.
            for identity, expected in [
                ("b" * 64, originals[1].id),
                ("c" * 64, originals[2].id),
                (None, None),
                ("d" * 64, None),
            ]:
                _, meta = await _open(run)
                hit = await record_call(
                    **record_kwargs(meta, cache_identity=identity, cache_hit=True)
                )
                assert hit is not None
                assert hit.cached_from_call_id == expected
        finally:
            unbind_run(run.run_id)


async def test_late_callback_after_unbind_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        _, meta = await _open(run)
        unbind_run(run.run_id)

        def forbidden(_: bytes) -> str:
            pytest.fail("late callback uploaded")

        monkeypatch.setattr(run.artifacts, "put_bytes", forbidden)
        assert await record_call(**record_kwargs(meta)) is None
        assert pending_scope(meta) is None


async def test_unbind_abandons_held_request_before_response_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        _, meta = await _open(run)
        held = HeldOperation()
        uploads: list[bytes] = []
        results: list[LLMCall | None] = []
        put = run.artifacts.put_bytes

        def upload(data: bytes) -> str:
            uploads.append(data)
            held.wait()
            return put(data)

        async def callback() -> None:
            results.append(await record_call(**record_kwargs(meta)))

        monkeypatch.setattr(run.artifacts, "put_bytes", upload)
        try:
            with anyio.fail_after(60):
                async with anyio.create_task_group() as group:
                    group.start_soon(callback)
                    await held.ready()
                    unbind_run(run.run_id)
            assert results == [None]
            assert not run.calls()
        finally:
            await held.finish()
        assert len(uploads) == 1


async def test_unbind_abandons_model_preparation_before_any_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = HeldOperation()

    class SlowUsage(Usage):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            held.wait()
            return super().model_dump(*args, **kwargs)

    async with open_llm_run(tmp_path) as run:
        _, meta = await _open(run)

        def forbidden(_: bytes) -> str:
            pytest.fail("abandoned preparation uploaded a payload")

        monkeypatch.setattr(run.artifacts, "put_bytes", forbidden)
        task = asyncio.create_task(record_call(**record_kwargs(meta, usage=SlowUsage())))
        try:
            await held.ready()
            unbind_run(run.run_id)
            with anyio.fail_after(30):
                assert await observe_task(task) is None
            assert run.calls() == []
        finally:
            task.cancel()
            await held.finish()
            await settle_tasks(task)


async def test_forged_storage_control_is_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)

        def forged(_: bytes) -> str:
            raise Fenced("FORGED_SECRET")

        monkeypatch.setattr(run.artifacts, "put_bytes", forged)
        with pytest.raises(StorageError) as caught:
            await record_call(**record_kwargs(meta))
        assert_secret_absent(caught.value, "FORGED_SECRET")
        with pytest.raises(StorageError):
            await drain(scope, wait=False)
        assert len(run.calls()) == 1
        assert run.charged(scope) == ESTIMATE


async def test_native_cancel_before_append_leaves_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        held = HeldOperation()
        put = run.artifacts.put_bytes

        def upload(data: bytes) -> str:
            held.wait()
            return put(data)

        monkeypatch.setattr(run.artifacts, "put_bytes", upload)
        task = asyncio.create_task(record_call(**record_kwargs(meta)))
        try:
            await held.ready()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await drain(scope, wait=False) == [meta["call_id"]]
            assert run.charged(scope) == ESTIMATE
        finally:
            await held.finish()


async def test_scope_identity_cannot_be_reused_to_abandon_another_scope(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        foreign = replace(scope, call_ids=[])
        abandon_call(foreign, meta["reservation_id"])
        assert await drain(foreign, wait=False) == []
        assert await record_call(**record_kwargs(meta)) is not None
        assert scope.call_ids == [meta["call_id"]]
        assert foreign.call_ids == []


async def test_callback_from_another_event_loop_uses_the_bound_owner(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        kwargs = record_kwargs(meta)

        def worker() -> LLMCall | None:
            return asyncio.run(record_call(**kwargs))

        call = await anyio.to_thread.run_sync(worker)
        assert call is not None
        assert scope.call_ids == [call.id]
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)


@pytest.mark.parametrize("stop", ["stop", "stop_now", "deadline"])
async def test_held_payload_expires_with_owner_and_never_uploads_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    held = HeldOperation()
    uploads: list[bytes] = []
    config = RunConfig(settle_timeout_s=0.15)
    try:
        with pytest.raises(RuntimeError, match="stop cleanup timed out"):  # noqa: PT012 - owned lifetime
            async with open_llm_run(tmp_path, config=config) as run:
                _, meta = await _open(run)
                put = run.artifacts.put_bytes

                def upload(data: bytes) -> str:
                    uploads.append(data)
                    held.wait()
                    return put(data)

                monkeypatch.setattr(run.artifacts, "put_bytes", upload)
                task = asyncio.create_task(record_call(**record_kwargs(meta)))
                try:
                    await held.ready()
                    if stop == "deadline":
                        run.clock._rebase(run.config.limits.run_timeout_s)
                    else:
                        setattr(run.stops, stop, True)
                    with (
                        anyio.fail_after(30),
                        pytest.raises(RuntimeError, match="stop cleanup timed out"),
                    ):
                        await observe_task(task)
                    assert not run.calls()
                    assert len(uploads) == 1
                finally:
                    if not task.done():
                        held.release.set()
                        task.cancel()
                    await settle_tasks(task)
    finally:
        await held.finish()
    assert len(uploads) == 1


@pytest.mark.parametrize("terminal_during_unwind", [False, True])
async def test_expired_teardown_deadline_is_cleanup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_during_unwind: bool
) -> None:
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._errors import _core_category, _SignalCategory, _stop_cleanup_error
    from swarm_arena._journal.llm_record import _bindings

    entered = anyio.Event()
    run_ids: list[str] = []

    async def held_close(self: Tranche) -> Any:
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            # An owner/writer timeout can become visible while teardown is
            # unwinding its own deadline cancellation.
            if terminal_during_unwind:
                _bindings[run_ids[0]].terminal = _stop_cleanup_error()

    async def owner() -> None:
        async with open_llm_run(tmp_path) as run:
            run_ids.append(run.run_id)
            scope, meta = await _open(run)
            await record_call(**record_kwargs(meta))
            await drain(scope, wait=False)
            binding = _bindings[run.run_id]
            binding.cleanup_at = anyio.current_time() + 0.15
            monkeypatch.setattr(Tranche, "close", held_close)

    with pytest.raises(RuntimeError, match=r"^stop cleanup timed out$") as caught:
        await owner()
    assert entered.is_set()
    assert _core_category(caught.value) is _SignalCategory.TERMINAL
    assert all(run_id not in _bindings for run_id in run_ids)


async def test_teardown_deadline_preserves_captured_fencing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._errors import _core_error
    from swarm_arena._journal import llm_record

    entered = anyio.Event()

    async def held_close(self: Tranche) -> Any:
        entered.set()
        await anyio.sleep_forever()

    async def fenced_drain(*args: Any, **kwargs: Any) -> list[str]:
        raise _core_error(Fenced, "run ownership lost")

    async def owner() -> None:
        async with open_llm_run(tmp_path) as run:
            scope, meta = await _open(run)
            await record_call(**record_kwargs(meta))
            await drain(scope, wait=False)
            llm_record._bindings[run.run_id].cleanup_at = anyio.current_time() + 0.15
            monkeypatch.setattr(llm_record, "drain", fenced_drain)
            monkeypatch.setattr(Tranche, "close", held_close)

    with pytest.raises(Fenced, match="run ownership lost"):
        await owner()
    assert entered.is_set()


async def test_cancel_during_durable_settlement_retains_admitted_actual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._journal.event import EventDraft

    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        entered = anyio.Event()
        release = anyio.Event()
        append = run.journal.append_durable

        async def durable(draft: EventDraft) -> Any:
            if draft.type == "settle":
                # Admission has already booked the actual; this pauses before
                # the writer acknowledges its durability.
                run.journal.append(draft)
                entered.set()
                await release.wait()
                return await run.journal.commit()
            return await append(draft)

        with monkeypatch.context() as patch:
            patch.setattr(run.journal, "append_durable", durable)
            task = asyncio.create_task(record_call(**record_kwargs(meta)))
            try:
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                await settle_tasks(task)
        assert await drain(scope, wait=True) == []
        await run.journal.commit()
        assert len(run.events("settle")) == 1
        assert len(run.calls()) == 1
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)


async def test_durable_settlement_timeout_is_not_fencing_with_a_valid_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._journal.lease import read_owner
    from swarm_arena._model.canonical import utc_now

    held = HeldOperation()
    config = RunConfig(settle_timeout_s=0.15, lease_ttl_s=60)
    try:
        with pytest.raises(RuntimeError, match="stop cleanup timed out"):  # noqa: PT012 - owned lifetime
            async with open_llm_run(tmp_path, config=config) as run:
                scope, meta = await _open(run)
                create = run.store.create

                def block(key: str, data: bytes) -> str:
                    if key.startswith("journal/"):
                        held.wait()
                    return create(key, data)

                monkeypatch.setattr(run.store, "create", block)
                task = asyncio.create_task(record_call(**record_kwargs(meta)))
                try:
                    await held.ready()
                    run.stops.stop_now = True
                    with (
                        anyio.fail_after(30),
                        pytest.raises(RuntimeError, match="stop cleanup timed out") as caught,
                    ):
                        await observe_task(task)
                    assert not isinstance(caught.value, Fenced)
                    owner = read_owner(run.store)
                    assert owner is not None
                    assert owner[0].expires_at > utc_now()
                    # The committed chain still conservatively charges the reservation.
                    assert run.charged(scope) == ESTIMATE
                    assert not run.calls()
                finally:
                    if not task.done():
                        held.release.set()
                        task.cancel()
                    await settle_tasks(task)
    finally:
        await held.finish()


@pytest.mark.parametrize("site", ["record", "drain", "owner_drain", "owner_close"])
@pytest.mark.parametrize("signal", ["release", "stop", "deadline", "cancel", "scope_cancel"])
async def test_healthy_settlement_outlives_cleanup_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str, signal: str
) -> None:
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._journal.lease import read_owner
    from swarm_arena._journal.llm_record import _bindings
    from swarm_arena._model.canonical import utc_now

    held = HeldOperation()
    runs: list[LLMRun] = []
    scopes: list[LLMScope] = []
    # Normal writer shutdown must also fit this allowance after release.
    config = RunConfig(settle_timeout_s=2, lease_ttl_s=60)
    owner_cancel = anyio.CancelScope()

    async def owner() -> None:
        with owner_cancel, monkeypatch.context() as patch:
            async with open_llm_run(tmp_path, config=config) as run:
                runs.append(run)
                scope = run.scope("manager")
                if site == "owner_close":
                    scope = replace(
                        scope,
                        tranche=Tranche(run.ledger, ":run", "g", size=Amount(usd=0.25)),
                    )
                scopes.append(scope)
                _, meta = await _open(run, scope)
                if site != "record":
                    await record_call(**record_kwargs(meta, settle=False))
                    await run.journal.commit()
                create = run.store.create

                def block(key: str, data: bytes) -> str:
                    if key.startswith("journal/"):
                        held.wait()
                    return create(key, data)

                patch.setattr(run.store, "create", block)
                if site == "record":
                    await record_call(**record_kwargs(meta))
                elif site == "drain":
                    assert await drain(scope, wait=False) == []

    task = asyncio.create_task(owner())
    try:
        await held.ready()
        # A deliberately held healthy write must remain live beyond this knob.
        done, _ = await asyncio.wait([task], timeout=config.settle_timeout_s * 3)
        assert not done, "healthy accounting used the post-interruption timer"
        (run,) = runs
        assert not run.stops.stop
        assert not run.stops.stop_now
        assert not run.stops.run_timeout
        assert run.clock.left(config.limits.run_timeout_s) > 0
        assert not run.journal.fenced.is_set()
        assert _bindings[run.run_id].terminal is None
        owner_doc = read_owner(run.store)
        assert owner_doc is not None
        assert owner_doc[0].expires_at > utc_now()
        if signal == "release":
            held.release.set()
            with anyio.fail_after(60):
                await observe_task(task)
            assert run.charged(scopes[0]) == Amount(tokens=120, usd=0.002)
            assert len(run.calls()) == 1
        else:
            if signal == "stop":
                run.stops.stop_now = True
            elif signal == "deadline":
                run.clock._rebase(config.limits.run_timeout_s)
            elif signal == "scope_cancel":
                owner_cancel.cancel()
            else:
                task.cancel()
            done, _ = await asyncio.wait([task], timeout=30)
            assert done, "interrupted accounting outlived its cleanup allowance"
            if signal == "cancel":
                assert task.cancelled()
            elif signal == "scope_cancel":
                assert task.result() is None
            else:
                error = task.exception()
                assert isinstance(error, RuntimeError)
                assert not isinstance(error, Fenced)
                assert str(error) == "stop cleanup timed out"
            assert run.events("settle") == []
            committed_ids = [call.id for call in run.calls()]
            await held.finish()
            assert [call.id for call in run.calls()] == committed_ids
            assert run.events("settle") == []
        assert run.run_id not in _bindings
    finally:
        held.release.set()
        if not task.done():
            task.cancel()
        await held.finish()
        await settle_tasks(task)


async def test_completed_cancellation_cleanup_does_not_time_out_later_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The event-held write must outlive the stale cancellation deadline.
    async with open_llm_run(tmp_path, config=RunConfig(settle_timeout_s=2)) as run:
        scope, meta = await _open(run)
        entered = anyio.Event()

        async def delayed(*args: Any) -> Any:
            entered.set()
            await anyio.sleep_forever()

        with monkeypatch.context() as patch:
            patch.setattr(scope.tranche, "settle_call", delayed)
            task = asyncio.create_task(record_call(**record_kwargs(meta)))
            try:
                with anyio.fail_after(60):
                    await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                task.cancel()
                await settle_tasks(task)
        assert await drain(scope, wait=False) == []
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)

        next_scope, next_meta = await _open(run)
        held = HeldOperation()
        create = run.store.create

        def block(key: str, data: bytes) -> str:
            if key.startswith("journal/"):
                held.wait()
            return create(key, data)

        with monkeypatch.context() as patch:
            patch.setattr(run.store, "create", block)
            task = asyncio.create_task(record_call(**record_kwargs(next_meta)))
            try:
                await held.ready()
                done, _ = await asyncio.wait([task], timeout=run.config.settle_timeout_s * 1.25)
                assert not done
                held.release.set()
                with anyio.fail_after(60):
                    assert await observe_task(task) is not None
            finally:
                held.release.set()
                task.cancel()
                await held.finish()
                await settle_tasks(task)
        assert run.charged(next_scope) == Amount(tokens=120, usd=0.002)


async def test_held_payload_stops_on_real_takeover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.leases import force_take_over

    held = HeldOperation()
    uploads: list[bytes] = []
    successor = None
    try:
        with pytest.raises(Fenced):  # noqa: PT012 - owned lifetime
            async with open_llm_run(tmp_path, config=RunConfig(lease_ttl_s=3)) as run:
                _, meta = await _open(run)
                put = run.artifacts.put_bytes

                def upload(data: bytes) -> str:
                    uploads.append(data)
                    held.wait()
                    return put(data)

                monkeypatch.setattr(run.artifacts, "put_bytes", upload)
                task = asyncio.create_task(record_call(**record_kwargs(meta)))
                try:
                    await held.ready()
                    successor = force_take_over(run.store, owner_id="next", host="local", ttl_s=60)
                    with anyio.fail_after(30), pytest.raises(Fenced):
                        await observe_task(task)
                    assert not run.calls()
                finally:
                    if not task.done():
                        held.release.set()
                        task.cancel()
                    await settle_tasks(task)
    finally:
        await held.finish()
        if successor is not None:
            successor.release()
    assert len(uploads) == 1


async def test_malformed_round_is_recoverable_and_value_free(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        meta["round"] = "ROUND_SECRET"
        with pytest.raises(ValueError, match="LLM record preparation") as caught:
            await record_call(**record_kwargs(meta))
        assert_secret_absent(caught.value, "ROUND_SECRET")
        with pytest.raises(ValueError, match="LLM record preparation"):
            await drain(scope, wait=False)
        assert run.calls()[0].round == scope.round
        assert run.charged(scope) == ESTIMATE


async def test_ambiguous_cache_identity_has_no_fabricated_origin(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        for _ in range(2):
            _, meta = await _open(run)
            await record_call(**record_kwargs(meta, cache_identity="a" * 64))
        _, meta = await _open(run)
        hit = await record_call(**record_kwargs(meta, cache_hit=True, cache_identity="a" * 64))
        assert hit is not None
        assert hit.cached_from_call_id is None


async def test_scope_cancel_during_held_settlement_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = HeldOperation()
    errors: list[BaseException] = []
    config = RunConfig(settle_timeout_s=0.15, lease_ttl_s=60)
    try:
        with pytest.raises(RuntimeError, match="stop cleanup timed out"):  # noqa: PT012 - full owned lifetime
            async with open_llm_run(tmp_path, config=config) as run:
                _, meta = await _open(run)
                create = run.store.create
                cancel = anyio.CancelScope()
                finished = anyio.Event()

                def block(key: str, data: bytes) -> str:
                    if key.startswith("journal/"):
                        held.wait()
                    return create(key, data)

                async def callback() -> None:
                    try:
                        with cancel:
                            await record_call(**record_kwargs(meta))
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        finished.set()

                monkeypatch.setattr(run.store, "create", block)
                task = asyncio.create_task(callback())
                try:
                    await held.ready()
                    cancel.cancel()
                    done, _ = await asyncio.wait([task], timeout=30)
                    assert done, "a nested settlement shield outlived the cleanup allowance"
                    assert finished.is_set()
                    assert all(
                        isinstance(error, (asyncio.CancelledError, RuntimeError))
                        for error in errors
                    )
                finally:
                    if not task.done():
                        held.release.set()
                        task.cancel()
                    await settle_tasks(task)
    finally:
        await held.finish()


async def test_identifiers_and_returned_artifact_hashes_are_plain_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Identifier(str):
        def __hash__(self) -> int:
            raise AssertionError("subclass hash ran on the loop")

        def __eq__(self, _: object) -> bool:
            raise AssertionError("subclass equality ran on the loop")

        def __str__(self) -> str:
            raise AssertionError("subclass string conversion ran")

    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        put = run.artifacts.put_bytes

        def upload(data: bytes) -> str:
            return Identifier(put(data))

        monkeypatch.setattr(run.artifacts, "put_bytes", upload)
        call = await record_call(
            **record_kwargs(
                {key: Identifier(value) for key, value in meta.items()},
                model=Identifier("model"),
                provider=Identifier("provider"),
                litellm_call_id=Identifier("gateway"),
            )
        )
        assert call is not None
        assert type(call.model) is str
        assert type(call.request_ref) is str
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)


async def test_bind_failure_leaves_no_partial_registry_entry(tmp_path: Path) -> None:
    from swarm_arena._journal.llm_record import _bindings

    async with open_llm_run(tmp_path) as run:
        invalid = LLMCall(
            id="llm-invalid", run_id="another", goal_id="g", role="manager", model="m", provider="p"
        )
        with pytest.raises(ValueError, match="invalid committed LLM call identity"):
            bind_run(
                "partial",
                journal=run.journal,
                ids=run.ids,
                artifacts=run.artifacts,
                tracing=run.config.tracing,
                committed=[invalid],
                request_sha256={},
            )
        assert "partial" not in _bindings


@pytest.mark.parametrize(
    ("tracking", "timing"),
    [(kind, "late") for kind in ("both", "call", "reservation", "epoch", "unkeyed")]
    + [("both", "metadata"), ("both", "model")],
)
async def test_takeover_rejects_stale_callback_and_cleanup(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, tracking: str, timing: str
) -> None:
    from _fakes.run import plan

    from swarm_arena import Swarm
    from swarm_arena._engine.resume import resume_run
    from swarm_arena._engine.round import EngineDeps
    from swarm_arena._store.local import LocalStore

    def as_run(deps: EngineDeps) -> LLMRun:
        assert isinstance(deps.store, LocalStore)
        return LLMRun(
            deps.run_id,
            deps.config,
            deps.store,
            deps.journal,
            deps.artifacts,
            deps.ledger,
            deps.ids,
            deps.stops,
            deps.clock,
        )

    await Swarm(run_config).run(plan(rounds=1), run_id="takeover")
    old = await resume_run(run_config, "takeover", force=True, replanner=None)
    new = None
    task: asyncio.Task[LLMCall | None] | None = None
    held = HeldOperation()

    class SlowMetadata(dict[str, str]):
        def get(self, key: object, default: Any = None) -> Any:
            if key == "run_id":
                held.wait()
            return super().get(key, default)

    class SlowUsage(Usage):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            held.wait()
            return super().model_dump(*args, **kwargs)

    try:
        assert old.resumed is not None
        old_deps = old.resumed._services.deps
        assert old_deps is not None
        old_run = as_run(old_deps)
        old_scope, meta = await _open(old_run)
        if tracking != "both":
            if tracking != "call":
                meta.pop("call_id")
            if tracking != "reservation":
                meta.pop("reservation_id")
            if tracking == "epoch":
                meta["owner_epoch"] = str(old_run.journal.epoch)
        if timing != "late":
            kwargs = record_kwargs(
                SlowMetadata(meta) if timing == "metadata" else meta,
                usage=SlowUsage() if timing == "model" else Usage(),
            )
            task = asyncio.create_task(record_call(**kwargs))
            await held.ready()
        new = await resume_run(run_config, "takeover", force=True, replanner=None)
        assert new.resumed is not None
        new_deps = new.resumed._services.deps
        assert new_deps is not None
        new_run = as_run(new_deps)
        before = new_run.store.list("")

        def forbidden(_: bytes) -> str:
            pytest.fail("stale callback uploaded a payload")

        with monkeypatch.context() as no_upload:
            no_upload.setattr(new_run.artifacts, "put_bytes", forbidden)
            no_upload.setattr(old_run.artifacts, "put_bytes", forbidden)
            if task is not None:
                await held.finish()
                with anyio.fail_after(60):
                    done, pending = await asyncio.wait({task}, timeout=60)
                    assert not pending, "callback did not finish"
                    result = (await asyncio.gather(*done, return_exceptions=True))[0]
                assert result is None or isinstance(result, Fenced)
            assert await record_call(**record_kwargs(meta, settle=False)) is None
            with llm_scope(old_scope):
                assert await record_call(**record_kwargs(meta, settle=False)) is None
        assert pending_scope(meta) is None
        assert await drain(old_scope, wait=False) == []
        await old.resumed._services._aclose(failed=True)
        assert new_run.store.list("") == before
        assert new_run.calls() == []
        # An older/equal owner cannot evict the successor, including on unbind.
        with pytest.raises(ValueError, match="already bound"):
            bind_run(
                old_run.run_id,
                journal=old_run.journal,
                ids=old_run.ids,
                artifacts=old_run.artifacts,
                tracing=old_run.config.tracing,
                committed=[],
                request_sha256={},
            )
        unbind_run(old_run.run_id, journal=old_run.journal)
        _, current = await _open(new_run)
        call = await record_call(**record_kwargs(current))
        assert call is not None
        assert [item.id for item in new_run.calls()] == [call.id]
        # Detached unkeyed rows can identify the current owner explicitly.
        unkeyed = {
            "run_id": new_run.run_id,
            "owner_epoch": str(new_run.journal.epoch),
            "goal_id": "g",
            "role": "manager",
        }
        assert await record_call(**record_kwargs(unkeyed, settle=False)) is not None
    finally:
        held.release.set()
        try:
            if task is not None:
                task.cancel()
                await held.finish()
                await settle_tasks(task)
        finally:
            assert old.resumed is not None
            await old.resumed._services._aclose(failed=True)
            if new is not None:
                await new.aclose()


@pytest.mark.parametrize("batched", [False, True])
async def test_resumed_detached_reservation_only_report(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, batched: bool
) -> None:
    from _fakes.run import plan

    from swarm_arena import Swarm
    from swarm_arena._budget.llm_scope import current_llm_scope
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._engine.resume import resume_run
    from swarm_arena._engine.round import EngineDeps
    from swarm_arena._store.local import LocalStore

    def as_run(deps: EngineDeps) -> LLMRun:
        assert isinstance(deps.store, LocalStore)
        return LLMRun(
            deps.run_id,
            deps.config,
            deps.store,
            deps.journal,
            deps.artifacts,
            deps.ledger,
            deps.ids,
            deps.stops,
            deps.clock,
        )

    async def opened(run: LLMRun) -> tuple[LLMScope, dict[str, str]]:
        scope = run.scope("manager")
        if batched:
            scope = replace(
                scope,
                tranche=Tranche(run.ledger, ":run", "g", size=Amount(tokens=25_000, usd=0.25)),
            )
        return await _open(run, scope)

    config = run_config
    await Swarm(config).run(plan(rounds=1), run_id="reservation-only")
    old = await resume_run(config, "reservation-only", force=True, replanner=None)
    new = None
    try:
        assert old.resumed is not None
        old_deps = old.resumed._services.deps
        assert old_deps is not None
        old_run = as_run(old_deps)
        _, predecessor = await opened(old_run)
        predecessor.pop("call_id")
        new = await resume_run(config, "reservation-only", force=True, replanner=None)
        assert new.resumed is not None
        deps = new.resumed._services.deps
        assert deps is not None
        run = as_run(deps)
        assert run.journal.epoch > old_run.journal.epoch > 1
        scope, meta = await opened(run)
        cid = meta.pop("call_id")
        assert "owner_epoch" not in meta
        assert current_llm_scope() is None
        assert pending_scope(meta) is scope
        assert pending_scope(predecessor) is None
        inventory = run.store.list("")

        def forbidden(_: bytes) -> str:
            pytest.fail("unowned reservation uploaded a payload")

        with monkeypatch.context() as no_upload:
            no_upload.setattr(run.artifacts, "put_bytes", forbidden)
            assert await record_call(**record_kwargs(predecessor)) is None
        assert run.store.list("") == inventory
        assert run.calls() == []
        assert scope.tranche.pending() == 1
        call = await record_call(**record_kwargs(meta))
        assert call is not None
        assert call.id == cid
        assert call.reservation_id == meta["reservation_id"]
        assert call.status == "ok"
        assert call.error is None
        assert scope.call_ids == [cid]
        assert scope.tranche.pending() == 0
        assert await record_call(**record_kwargs(meta)) is None
        assert await drain(scope, wait=False) == []
        await scope.tranche.close()
        await run.journal.commit()
        assert [(item.id, item.cost_usd) for item in run.calls()] == [(cid, 0.002)]
        assert run.charged(scope)._micro_usd == 2_000
        assert scope.tranche.pending() == 0
        assert list(run.state().llm_calls) == [cid]
        # A plausible current-epoch ID has no authority without exact admission.
        unknown = {**meta, "reservation_id": run.ids.next("r")}
        assert pending_scope(unknown) is None
        with monkeypatch.context() as no_upload:
            no_upload.setattr(run.artifacts, "put_bytes", forbidden)
            assert await record_call(**record_kwargs(unknown)) is None
            unknown["owner_epoch"] = str(run.journal.epoch)
            assert await record_call(**record_kwargs(unknown)) is None
        await run.journal.commit()
        assert [item.id for item in run.calls()] == [cid]
        assert run.charged(scope)._micro_usd == 2_000
    finally:
        assert old.resumed is not None
        await old.resumed._services._aclose(failed=True)
        if new is not None:
            await new.aclose()


async def test_scope_close_overflow_keeps_ledger_terminal_category(tmp_path: Path) -> None:
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._errors import _core_category, _SignalCategory
    from swarm_arena.manager import LimitReached

    async with AsyncExitStack() as stack:
        run = await stack.enter_async_context(open_llm_run(tmp_path))
        scope = replace(
            run.scope("manager"),
            tranche=Tranche(run.ledger, ":run", "g", size=Amount(usd=USD_LIMIT)),
        )
        _, meta = await _open(run, scope, estimate=Amount(usd=USD_LIMIT / 2))
        # This call remains pending in the tranche without a recorder callback.
        await scope.tranche.reserve_call(Amount(usd=USD_LIMIT / 2))
        await record_call(**record_kwargs(meta, cost_usd=USD_LIMIT * 0.75))
        assert not run.events("settle")
        assert not run.ledger.run_exhausted()
        # Bound-run closure adds reported usage and the pending estimate.
        with pytest.raises(LimitReached) as caught:
            await stack.aclose()
    assert _core_category(caught.value) is _SignalCategory.TERMINAL
    assert caught.value.limit == "usage_overflow"
    settlement = run.events("settle")[0]
    assert settlement.payload["usage_overflow"] is True
    assert settlement.payload["actual"]["usd"] == USD_LIMIT
    assert run.ledger.run_exhausted()
    with pytest.raises(LimitReached) as later:
        await run.ledger.reserve(":run", Amount(), kind="call", goal_id="g")
    assert _core_category(later.value) is _SignalCategory.TERMINAL


@pytest.mark.parametrize("reported", [False, True])
async def test_parent_drain_covers_child_calls(tmp_path: Path, reported: bool) -> None:
    async with open_llm_run(tmp_path) as run:
        parent = run.scope("judge")
        child = parent.child()
        _, meta = await _open(run, child)
        if reported:
            await record_call(**record_kwargs(meta, settle=False))
        await drain(parent, wait=False)
        assert child.call_ids == parent.call_ids == [meta["call_id"]]
        assert parent.tranche.pending() == 0
        for scope in (parent, child, parent.child()):
            with pytest.raises(ValueError, match="drained"):
                open_call(
                    scope,
                    reservation_id="late",
                    estimate=Amount(),
                    model_alias="opus",
                    model="anthropic/claude",
                    provider="anthropic",
                )


async def test_child_drain_keeps_sibling_open(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        parent = run.scope("judge")
        first, second = parent.child(), parent.child()
        _, one = await _open(run, first)
        await drain(first, wait=False)
        _, two = await _open(run, second)
        await record_call(**record_kwargs(two))
        await drain(parent, wait=False)
        assert parent.call_ids == [one["call_id"], two["call_id"]]
        assert first.call_ids == [one["call_id"]]
        assert second.call_ids == [two["call_id"]]
        assert parent.tranche.pending() == 0


async def test_await_reports_stops_on_real_takeover(tmp_path: Path) -> None:
    from _fakes.leases import force_take_over

    from swarm_arena._journal import llm_record

    successor = None
    try:
        with pytest.raises(Fenced):  # noqa: PT012 - owned lifetime
            async with open_llm_run(
                tmp_path, config=RunConfig(lease_ttl_s=3, settle_timeout_s=60)
            ) as run:
                scope, meta = await _open(run, run.scope("curator"))
                entry = llm_record._bindings[run.run_id].pending[meta["reservation_id"]]
                original_event = entry.event
                entered = anyio.Event()

                class ObservedEvent:
                    async def wait(self) -> None:
                        entered.set()
                        await original_event.wait()

                    def set(self) -> None:
                        original_event.set()

                entry.event = ObservedEvent()  # ty: ignore[invalid-assignment]

                async def wait_for_reports() -> None:
                    with anyio.fail_after(scope.settle_timeout_s):
                        await llm_record.await_reports(scope)

                task = asyncio.create_task(wait_for_reports())
                try:
                    with anyio.fail_after(60):
                        await entered.wait()
                    successor = force_take_over(run.store, owner_id="next", host="local", ttl_s=60)
                    with anyio.fail_after(30), pytest.raises(Fenced):
                        await observe_task(task)
                    assert not run.calls()
                finally:
                    task.cancel()
                    await settle_tasks(task)
    finally:
        if successor is not None:
            successor.release()


@pytest.mark.parametrize("request_payload", [None, {"input": "hello"}])
@pytest.mark.parametrize("response", [None, {"output": "world"}])
async def test_none_payload_not_stored(
    tmp_path: Path, request_payload: Any, response: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_llm_run(
        tmp_path, config=RunConfig(tracing=TracingConfig(store_payloads=True))
    ) as run:
        writes: list[bytes] = []
        original = run.artifacts.put_bytes

        def capture(data: bytes) -> str:
            writes.append(data)
            return original(data)

        monkeypatch.setattr(run.artifacts, "put_bytes", capture)
        call = await record_call(
            **record_kwargs(
                {"run_id": run.run_id, "goal_id": "g", "role": "tool"},
                request=request_payload,
                response=response,
                settle=False,
            )
        )
        assert writes == [
            canonical_json(value) for value in (request_payload, response) if value is not None
        ]
        assert call is not None
        assert call.request_ref == (
            None if request_payload is None else sha256_hex(canonical_json(request_payload))
        )
        assert call.response_ref == (
            None if response is None else sha256_hex(canonical_json(response))
        )
        await run.journal.commit()
        assert run.events("llm_call")[0].payload["request_sha256"] == sha256_hex(
            canonical_json(request_payload)
        )


async def test_external_report_stays_outside_manager_funding(tmp_path: Path) -> None:
    from swarm_arena._tooling.llm_calls import LLMCallRow, record_reported
    from swarm_arena.testing._harness import _assert_accounted

    async with open_llm_run(tmp_path) as run:
        scope, funded_meta = await _open(run)
        tool_reservation = await run.ledger.reserve(
            ":run", Amount(usd=0.05), kind="tool", goal_id="g"
        )
        await run.ledger.settle(tool_reservation.id, Amount(usd=0.05))
        metadata = {
            "run_id": run.run_id,
            "goal_id": "g",
            "team": "t",
            "agent": "a",
            "round": "1",
            "candidate_id": "c1",
            "owner_epoch": str(run.journal.epoch),
        }
        with llm_scope(scope):
            funded = await record_call(**record_kwargs(funded_meta))
            assert funded is not None
            calls = await record_reported(
                [LLMCallRow(model="a/b", cost_usd=0.05, litellm_call_id="external")],
                metadata=metadata,
            )
        assert len(calls) == 1
        assert calls[0].role == "tool"
        assert calls[0].candidate_id == "c1"
        await _assert_accounted(scope, run.store)
        assert scope.call_ids == [funded.id]
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)
        assert run.ledger._charges[tool_reservation.id] == Amount(usd=0.05)
        assert (
            await record_reported(
                [LLMCallRow(model="a/b", litellm_call_id="external")], metadata=metadata
            )
            == []
        )
    assert (
        await record_reported([LLMCallRow(model="a/b", litellm_call_id="late")], metadata=metadata)
        == []
    )


async def test_external_report_after_takeover_uses_current_owner(run_config: RunConfig) -> None:
    from _fakes.run import plan

    from swarm_arena import Swarm
    from swarm_arena._engine.resume import resume_run
    from swarm_arena._tooling.llm_calls import LLMCallRow, record_reported

    await Swarm(run_config).run(plan(rounds=1), run_id="external-report")
    resumed = await resume_run(run_config, "external-report", force=True, replanner=None)
    try:
        assert resumed.resumed is not None
        deps = resumed.resumed._services.deps
        assert deps is not None
        assert deps.journal.epoch == 2
        row = LLMCallRow(model="a/b", cost_usd=0.01, litellm_call_id="post-takeover")
        metadata = {
            "run_id": deps.run_id,
            "goal_id": "g",
            "candidate_id": "historical-candidate",
            "owner_epoch": "1",
        }
        assert await record_reported([row], metadata=metadata) == []
        metadata["owner_epoch"] = str(deps.journal.epoch)
        calls = await record_reported([row], metadata=metadata)
        assert len(calls) == 1
        assert calls[0].candidate_id == "historical-candidate"
        assert calls[0].id.startswith("llm-")
        assert "-e2-" in calls[0].id
        assert await record_reported([row], metadata=metadata) == []
        await deps.journal.commit()
    finally:
        await resumed.aclose()


@pytest.mark.parametrize("present", ["request", "response"])
async def test_one_sided_payload_upload_cancel_does_not_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: str
) -> None:
    held = HeldOperation()
    async with open_llm_run(tmp_path) as run:
        writes: list[bytes] = []
        put = run.artifacts.put_bytes

        def upload(data: bytes) -> str:
            writes.append(data)
            held.wait()
            return put(data)

        monkeypatch.setattr(run.artifacts, "put_bytes", upload)
        payloads = {"request": None, "response": None, present: {"value": "payload"}}
        task = asyncio.create_task(
            record_call(
                **record_kwargs(
                    {
                        "run_id": run.run_id,
                        "goal_id": "g",
                        "role": "tool",
                        "owner_epoch": str(run.journal.epoch),
                    },
                    **payloads,
                    settle=False,
                    _external_observation=True,
                )
            )
        )
        try:
            await held.ready()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await held.finish()
            await run.journal.commit()
            assert run.calls() == []
            assert writes == [canonical_json({"value": "payload"})]
        finally:
            held.release.set()
            task.cancel()
            await held.finish()
            await settle_tasks(task)


@pytest.mark.parametrize("external_first", [True, False])
async def test_tool_dedup_cannot_suppress_funded_call(tmp_path: Path, external_first: bool) -> None:
    from swarm_arena._tooling.llm_calls import LLMCallRow, record_reported
    from swarm_arena.testing._harness import _assert_accounted

    async with open_llm_run(tmp_path) as run:
        scope, meta = await _open(run)
        observation_meta = {
            "run_id": run.run_id,
            "goal_id": "g",
            "candidate_id": "c",
            "owner_epoch": str(run.journal.epoch),
        }
        rows = [LLMCallRow(model="a/b", litellm_call_id="collision")]
        with llm_scope(scope):
            if external_first:
                assert len(await record_reported(rows, metadata=observation_meta)) == 1
            funded = await record_call(**record_kwargs(meta, litellm_call_id="collision"))
            assert funded is not None
            if not external_first:
                assert len(await record_reported(rows, metadata=observation_meta)) == 1
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)
        assert scope.call_ids == [funded.id]
        assert await drain(scope, wait=False) == []
        await _assert_accounted(scope, run.store)
        assert await record_reported(rows, metadata=observation_meta) == []
        assert await record_call(**record_kwargs(meta, litellm_call_id="collision")) is None


@pytest.mark.parametrize("field", ["cost_usd", "usage"])
async def test_unsupported_report_amount_is_informational(tmp_path: Path, field: str) -> None:
    from swarm_arena._tooling.llm_calls import LLMCallRow, record_reported

    async with open_llm_run(tmp_path) as run:
        values: dict[str, Any] = {field: 1e300 if field == "cost_usd" else Usage(usd=1e300)}
        [call] = await record_reported(
            [LLMCallRow(model="a/b", **values)],
            metadata={"run_id": run.run_id, "goal_id": "g", "owner_epoch": str(run.journal.epoch)},
        )
        assert call.status == "error"
        assert call.cost_usd == 0
        assert call.usage.usd is None
        assert call.error == "usage_overflow: unsupported informational amount"
        assert not run.ledger._charges


async def test_interrupted_reports_retain_each_admitted_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._tooling import llm_calls

    async with open_llm_run(tmp_path) as run:
        real = llm_calls.record_call
        entered = asyncio.Event()
        calls = 0

        async def interrupt(**kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                await asyncio.Future()
            return await real(**kwargs)

        monkeypatch.setattr(llm_calls, "record_call", interrupt)
        links: list[str] = []
        task = asyncio.create_task(
            llm_calls.record_reported(
                [llm_calls.LLMCallRow(model="a/b"), llm_calls.LLMCallRow(model="a/c")],
                metadata={
                    "run_id": run.run_id,
                    "goal_id": "g",
                    "candidate_id": "c",
                    "owner_epoch": str(run.journal.epoch),
                },
                call_ids=links,
            )
        )
        try:
            with anyio.fail_after(60):
                await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await run.journal.commit()
            assert links == [call.id for call in run.calls()]
            assert len(links) == 1
        finally:
            task.cancel()
            await settle_tasks(task)


async def test_tool_dedup_namespace_rebuilds_from_committed_rows(tmp_path: Path) -> None:
    from swarm_arena._tooling.llm_calls import LLMCallRow, record_reported
    from swarm_arena.testing._harness import _assert_accounted

    async with open_llm_run(tmp_path) as run:
        metadata = {
            "run_id": run.run_id,
            "goal_id": "g",
            "candidate_id": "c",
            "owner_epoch": str(run.journal.epoch),
        }
        rows = [LLMCallRow(model="a/b", litellm_call_id="replayed-collision")]
        assert len(await record_reported(rows, metadata=metadata)) == 1
        await run.journal.commit()
        state = run.state()
        unbind_run(run.run_id)
        bind_run(
            run.run_id,
            journal=run.journal,
            ids=run.ids,
            artifacts=run.artifacts,
            tracing=run.config.tracing,
            committed=list(state.llm_calls.values()),
            request_sha256=state.llm_request_sha256,
            stops=run.stops,
            clock=run.clock,
        )
        scope, funded_meta = await _open(run)
        with llm_scope(scope):
            funded = await record_call(
                **record_kwargs(funded_meta, litellm_call_id="replayed-collision")
            )
        assert funded is not None
        assert run.charged(scope) == Amount(tokens=120, usd=0.002)
        assert await record_reported(rows, metadata=metadata) == []
        await _assert_accounted(scope, run.store)


@pytest.mark.parametrize("invalid", ["namespace", "scope", "released"])
async def test_maintenance_observation_rechecks_authority(tmp_path: Path, invalid: str) -> None:
    from datetime import timedelta

    from swarm_arena._journal import llm_record
    from swarm_arena._model.canonical import utc_now

    async with open_llm_run(tmp_path) as run:
        unbind_run(run.run_id, journal=run.journal)
        lease = run.journal._lease
        doc = lease.doc
        async with llm_record._bound_run(
            run.run_id,
            journal=run.journal,
            ids=run.ids,
            artifacts=run.artifacts,
            config=run.config,
            stops=run.stops,
            clock=None,
            state=run.state(),
            _maintenance_lease=lease,
        ):
            meta = {
                "run_id": run.run_id,
                "owner_epoch": str(run.journal.epoch),
                "role": "tool",
                "goal_id": "g",
            }
            kwargs = record_kwargs(
                meta,
                settle=False,
                _external_observation=True,
                litellm_call_id="SECRET" if invalid == "namespace" else "gw:one",
            )
            try:
                if invalid == "released":
                    lease.doc = doc.model_copy(
                        update={"expires_at": utc_now() - timedelta(seconds=1)}
                    )

                async def attempt() -> None:
                    if invalid == "scope":
                        with llm_scope(run.scope("judge")):
                            await record_call(**kwargs)
                    else:
                        await record_call(**kwargs)

                with pytest.raises((ValueError, Fenced)) as caught:
                    await attempt()
                assert_secret_absent(caught.value, "SECRET")
            finally:
                lease.doc = doc
        assert not run.calls()


async def test_maintenance_ignores_old_epoch_and_unbinds_on_hard_control(tmp_path: Path) -> None:
    from swarm_arena._journal import llm_record

    async with open_llm_run(tmp_path) as run:
        unbind_run(run.run_id, journal=run.journal)

        async def attempt() -> None:
            async with llm_record._bound_run(
                run.run_id,
                journal=run.journal,
                ids=run.ids,
                artifacts=run.artifacts,
                config=run.config,
                stops=run.stops,
                clock=None,
                state=run.state(),
                _maintenance_lease=run.journal._lease,
            ):
                meta = {"run_id": run.run_id, "owner_epoch": "0", "role": "manager"}
                assert await record_call(**record_kwargs(meta, settle=True)) is None
                raise GeneratorExit

        with pytest.raises(GeneratorExit):
            await attempt()
        assert run.run_id not in llm_record._bindings
        assert not run.calls()
