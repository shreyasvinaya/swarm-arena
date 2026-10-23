"""Reconcile real owned journals using explicit offline gateway capabilities."""

from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from _fakes.llm_run import open_llm_run
from _fakes.reconcile import (
    T0,
    CapableGateway,
    committed,
    config_at,
    created,
    key_record,
    seed_run,
    spend,
)
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import ConditionalWritesUnsupported, Fenced, _core_error
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._gateway import reconcile
from swarm_arena._gateway.adapter import GatewayError, ProfileReport
from swarm_arena._gateway.keys import gateway_call_id
from swarm_arena._journal.event import EventDraft
from swarm_arena._journal.lease import OWNER_KEY, Lease, read_owner
from swarm_arena._journal.replay import RunState
from swarm_arena._model.canonical import canonical_json, utc_now
from swarm_arena._model.settings import GatewayConfig
from swarm_arena._store.base import Store
from swarm_arena._store.local import LocalStore


def executor(dead: bool | None = True) -> Executor:
    return cast(Executor, Mock(spec=Executor, confirm_dead=AsyncMock(return_value=dead)))


@pytest.mark.parametrize("terminal", [False, True])
async def test_all_conditions_and_replay(tmp_path: Path, terminal: bool) -> None:
    adapter = CapableGateway()
    adapter.terminal_accounting = terminal
    adapter.rows = [spend("one", 0.01), spend("two", 0.02)]
    async with open_llm_run(tmp_path) as run:
        run.journal.append(created(key_record(run.run_id)))
        await run.journal.commit()
        first = await reconcile.reconcile_keys(
            run.events("gateway_key"),
            run_id=run.run_id,
            journal=run.journal,
            adapter=adapter,
            executor=executor(),
            gateway=GatewayConfig(url="http://gw"),
            profile=adapter.profile,
            now=lambda: T0,
        )
        assert first.keys[0].label == "estimated"
        assert adapter.rows
        assert not adapter.windows
        assert not adapter.deleted
        await run.journal.commit()
        second = await reconcile.reconcile_keys(
            run.events("gateway_key"),
            run_id=run.run_id,
            journal=run.journal,
            adapter=adapter,
            executor=executor(),
            gateway=GatewayConfig(url="http://gw"),
            profile=adapter.profile,
            now=lambda: T0 + timedelta(seconds=610),
        )
        await run.journal.commit()
        key = second.keys[0]
        assert key.observed_usd == second.spend_usd() == 0.03
        assert key.label == ("observed" if terminal else "observed (completeness unverified)")
        assert key.deleted
        assert key.reasons == []
        assert adapter.deleted == ["hash-1"]
        assert len([e for e in run.events("gateway_key") if e.payload["action"] == "closed"]) == 1
        assert run.state().gateway_reconcile[("c-1", "hash-1")] == key
        assert {call.litellm_call_id for call in run.calls()} == {
            gateway_call_id("hash-1", "one"),
            gateway_call_id("hash-1", "two"),
        }
        assert {call.candidate_id for call in run.calls()} == {"c-1"}
        assert not run.events("settle")


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("alive", "tool not confirmed dead"),
        ("unknown", "tool not confirmed dead"),
        ("not_closed", "key closure not confirmed by the gateway"),
        ("too_soon", "request_timeout_s has not passed since closure"),
        ("no_interval", "gateway batch write interval unknown"),
        ("no_duration_limit", "gateway maximum request duration not verified"),
        ("duration_too_long", "gateway maximum request duration not verified"),
        ("no_profile", "gateway maximum request duration not verified"),
    ],
)
async def test_failed_condition_is_estimated(tmp_path: Path, setup: str, reason: str) -> None:
    adapter = CapableGateway()
    profile: ProfileReport | None = adapter.profile
    dead = None if setup == "unknown" else setup != "alive"
    clock = T0 + timedelta(seconds=609 if setup == "too_soon" else 610)
    if setup == "not_closed":
        adapter.closure = None
    if setup == "no_profile":
        profile = None
    elif setup in {"no_interval", "no_duration_limit", "duration_too_long"}:
        profile = ProfileReport(
            verified=False,
            failures=("unverified",),
            request_timeout_s=None
            if setup == "no_duration_limit"
            else (900 if setup == "duration_too_long" else 300),
            batch_write_interval_s=None if setup == "no_interval" else 10,
        )
    async with open_llm_run(tmp_path) as run:
        run.journal.append(created(key_record(run.run_id)))
        started = Job.Started(kind="dispatch", job_id="c-1", task_name="c-1", ray_job_id="old")
        run.journal.append(
            EventDraft(
                type="tool_dispatch",
                payload={
                    "candidate_id": "c-1",
                    "reservation_id": "res-c-1",
                    "started": started.model_dump(mode="json"),
                },
            )
        )
        await run.journal.commit()
        probe = executor(dead)
        report = await reconcile.reconcile_keys(
            [*run.events("gateway_key"), *run.events("tool_dispatch")],
            run_id=run.run_id,
            journal=run.journal,
            adapter=adapter,
            executor=probe,
            gateway=GatewayConfig(url="http://gw"),
            profile=profile,
            now=lambda: clock,
        )
        row = report.keys[0]
        assert row.label == "estimated"
        assert row.observed_usd is None
        assert not row.deleted
        assert reason in row.reasons
        assert report.spend_usd() == 1
        assert adapter.deleted == []
        assert adapter.windows == []
        cast(Any, probe.confirm_dead).assert_awaited_once_with(started)


@pytest.mark.parametrize(
    "costs", [[0.1, 0.2], [600_000_000.0, 600_000_000.0], [0.0000005, 0.0000015]]
)
async def test_exact_deduplicated_totals_survive_deleted_key_retention(
    tmp_path: Path, costs: list[float]
) -> None:
    from swarm_arena._budget.amount import _micro_units

    config = config_at(tmp_path)
    await seed_run(config)
    adapter = CapableGateway()
    adapter.rows = [spend(str(i), cost) for i, cost in enumerate(costs)]
    adapter.rows += adapter.rows[:1]
    expected = sum(_micro_units(cost) for cost in costs) / 1_000_000
    first = await reconcile.reconcile_run(config, "r-reconcile", adapter, executor())
    assert first.spend_usd() == expected
    before = committed(config)
    adapter.rows = [spend("late", 0.02)]
    second = await reconcile.reconcile_run(config, "r-reconcile", adapter, executor())
    assert second.spend_usd() == (sum(_micro_units(cost) for cost in costs) + 20_000) / 1_000_000
    adapter.rows = []
    third = await reconcile.reconcile_run(config, "r-reconcile", adapter, executor())
    assert third == second
    assert adapter.deleted == ["hash-1"]
    calls = [e for e in committed(config) if e.type == "llm_call"]
    assert len(calls) == len(costs) + 1
    assert len([e for e in before if e.type == "llm_call"]) == len(costs)
    assert not [e for e in committed(config) if e.type == "settle"]
    assert "-e3-" in calls[-1].payload["call"]["id"]


async def test_conflicting_observation_stays_estimated(tmp_path: Path) -> None:
    config = config_at(tmp_path)
    await seed_run(config)
    adapter = CapableGateway()
    adapter.rows = [spend("one", 0.1), spend("one", 0.2)]
    report = await reconcile.reconcile_run(config, "r-reconcile", adapter, executor())
    assert report.keys[0].label == "estimated"
    assert not adapter.deleted
    assert not [e for e in committed(config) if e.type == "llm_call"]
    assert "conflicting gateway observations" in report.keys[0].reasons


async def test_reconcile_run_fails_with_live_owner_writes_nothing(tmp_path: Path) -> None:
    config = config_at(tmp_path)
    store = LocalStore(Path(config.run_root) / "r-reconcile")
    lease = Lease.acquire_new(store, owner_id="live", host="h", ttl_s=60)
    before = store.read(OWNER_KEY)
    try:
        with pytest.raises(RuntimeError, match="has a live owner"):
            await reconcile.reconcile_run(config, "r-reconcile", CapableGateway(), executor())
        assert store.read(OWNER_KEY) == before
        assert store.list("journal/") == []
    finally:
        lease.release()


async def test_reconcile_checks_storage_before_takeover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_at(tmp_path)
    await seed_run(config)
    store = LocalStore(Path(config.run_root) / "r-reconcile")
    before = store.read(OWNER_KEY)

    def refuse(config: Any) -> None:
        raise ConditionalWritesUnsupported("conditional writes not honoured")

    monkeypatch.setattr(reconcile, "check_storage", refuse)
    with pytest.raises(ConditionalWritesUnsupported):
        await reconcile.reconcile_run(config, "r-reconcile", CapableGateway(), executor())
    assert store.read(OWNER_KEY) == before
    assert not [e for e in committed(config) if e.type in {"takeover", "reconcile"}]


async def test_reconcile_run_without_head_reports_no_keys(tmp_path: Path) -> None:
    config = config_at(tmp_path)
    store = LocalStore(Path(config.run_root) / "r-reconcile")
    lease = Lease.acquire_new(store, owner_id="old", host="h", ttl_s=60)
    lease.release()
    report = await reconcile.reconcile_run(config, "r-reconcile", CapableGateway(), executor())
    assert report.run_id == "r-reconcile"
    assert report.keys == []
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].epoch == 2
    assert owner[0].expires_at < utc_now()
    assert [e.payload for e in committed(config) if e.type == "takeover"] == [{"force": False}]
    assert RunState.from_events(committed(config)).run_id == "r-reconcile"


@pytest.mark.parametrize("race", ["renew", "other_fenced", "other_first"])
async def test_takeover_races_preserve_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race: str
) -> None:
    config = config_at(tmp_path)
    await seed_run(config)
    store = LocalStore(Path(config.run_root) / "r-reconcile")
    before = store.list("journal/")
    real = Lease.take_over.__func__
    calls: list[int] = []

    def racing(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        calls.append(1)
        if len(calls) == 1:
            if race == "renew":
                doc, etag = cast(Any, read_owner(store))
                store.replace(
                    OWNER_KEY,
                    canonical_json(
                        doc.model_copy(update={"version": doc.version + 1}).model_dump(mode="json")
                    ),
                    if_match=etag,
                )
            else:
                other: dict[str, Any] = {**kw, "owner_id": "other"}
                real(cls, store, **other)
            if race != "other_first":
                raise _core_error(Fenced, "another owner changed the lease")
        return real(cls, store, **kw)

    monkeypatch.setattr(Lease, "take_over", classmethod(racing))
    if race == "renew":
        await reconcile.reconcile_run(config, "r-reconcile", CapableGateway(), executor())
        assert len(calls) == 2
    else:
        with pytest.raises(Fenced, match="another owner took the run"):
            await reconcile.reconcile_run(config, "r-reconcile", CapableGateway(), executor())
        assert store.list("journal/") == before
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].epoch == 2
    if race != "renew":
        assert owner[0].owner_id == "other"


async def test_gateway_failure_is_value_free(tmp_path: Path) -> None:
    config = config_at(tmp_path)
    await seed_run(config)
    adapter = CapableGateway()
    adapter.errors["read"] = GatewayError("SECRET")
    report = await reconcile.reconcile_run(config, "r-reconcile", adapter, executor())
    assert report.keys[0].label == "estimated"
    assert not report.keys[0].deleted
    assert "SECRET" not in repr(committed(config))
    assert "SECRET" not in report.model_dump_json()


async def test_missing_owner_has_safe_cli_diagnostic(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        await reconcile.take_run(
            config_at(tmp_path), "r-reconcile", owner_id="reconciler", force=False
        )
    assert "has no control/owner.json" in str(caught.value)
    assert_secret_absent(caught.value, str(tmp_path))


async def test_pinned_proxy_profile_keeps_full_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.gateway import FakeGateway

    from swarm_arena._gateway.litellm_proxy import LiteLLMProxy

    wire = FakeGateway()
    proxy = LiteLLMProxy(url="http://gateway.invalid", master_key="secret", request_timeout_s=600)

    def request(operation: Any, method: str, path: str, **kwargs: Any) -> Any:
        status, body, _ = wire._route(
            method, path, kwargs.get("body") or {}, f"Bearer {wire.master_key}"
        )
        assert status == 200
        return body

    monkeypatch.setattr(proxy, "_request", request)
    adapter = CapableGateway()
    adapter.profile = await proxy.preflight()
    assert adapter.profile.verified is False
    assert adapter.profile.request_timeout_s is None
    config = config_at(tmp_path)
    await seed_run(config)
    report = await reconcile.reconcile_run(config, "r-reconcile", adapter, executor())
    assert report.keys[0].label == "estimated"
    assert report.spend_usd() == 1
    assert not adapter.deleted
    assert not adapter.windows


@pytest.mark.parametrize("site", ["preflight", "read"])
async def test_cancelled_reconcile_releases_and_unbinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    import anyio

    from swarm_arena._journal import llm_record

    config = config_at(tmp_path)
    await seed_run(config)
    adapter = CapableGateway()
    entered = anyio.Event()

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await anyio.sleep_forever()

    monkeypatch.setattr(adapter, "preflight" if site == "preflight" else "spend_rows", held)
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(reconcile.reconcile_run, config, "r-reconcile", adapter, executor())
            await entered.wait()
            tasks.cancel_scope.cancel()
    owner = read_owner(LocalStore(Path(config.run_root) / "r-reconcile"))
    assert owner is not None
    assert owner[0].expires_at < utc_now()
    assert "r-reconcile" not in llm_record._bindings
    assert not adapter.deleted


async def test_writer_entry_failure_releases_takeover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._journal.writer import JournalWriter

    config = config_at(tmp_path)
    await seed_run(config)

    async def failed(self: JournalWriter) -> JournalWriter:
        raise _core_error(RuntimeError, "writer unavailable")

    monkeypatch.setattr(JournalWriter, "__aenter__", failed)
    with pytest.raises(RuntimeError, match="writer unavailable"):
        await reconcile.reconcile_run(config, "r-reconcile", CapableGateway(), executor())
    owner = read_owner(LocalStore(Path(config.run_root) / "r-reconcile"))
    assert owner is not None
    assert owner[0].expires_at < utc_now()


@pytest.mark.parametrize("kind", ["foreign", "terminal_group", "hard_group", "cancel"])
async def test_external_boundary_preserves_controls(kind: str) -> None:
    import asyncio

    import anyio

    from swarm_arena._errors import _core_category, _SignalCategory, _stop_cleanup_error

    error: BaseException
    if kind == "foreign":
        error = ValueError("PRIVATE")
    elif kind == "terminal_group":
        error = BaseExceptionGroup("PRIVATE", [asyncio.CancelledError(), _stop_cleanup_error()])
    elif kind == "hard_group":
        error = BaseExceptionGroup("PRIVATE", [GeneratorExit(), ValueError("PRIVATE")])
    else:
        error = asyncio.CancelledError()

    async def fail() -> None:
        raise error

    expected = {
        "foreign": GatewayError,
        "terminal_group": RuntimeError,
        "hard_group": BaseExceptionGroup,
        "cancel": GatewayError,
    }[kind]
    with pytest.raises(expected) as caught:
        await reconcile._external(fail, anyio.current_time() + 1)
    if kind in {"foreign", "cancel"}:
        assert_secret_absent(caught.value, "PRIVATE")
    elif kind == "terminal_group":
        assert _core_category(caught.value) is _SignalCategory.TERMINAL
    else:
        assert caught.value is error


@pytest.mark.parametrize("stop", [True, False])
async def test_reconcile_external_wait_obeys_stop_and_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: bool
) -> None:
    import anyio

    from swarm_arena._journal import llm_record

    config = config_at(tmp_path)
    await seed_run(config)
    original = reconcile.take_run
    authority: dict[str, Any] = {}

    async def capture(*args: Any, **kwargs: Any) -> Any:
        result = await original(*args, **kwargs)
        authority["writer"] = result[2]
        authority["stops"] = kwargs["_stops"]
        return result

    async def held(*args: Any, **kwargs: Any) -> Any:
        if stop:
            authority["stops"].stop_now = True
        else:
            authority["writer"]._fail(_core_error(Fenced, "owner lost"))
        await anyio.sleep_forever()

    adapter = CapableGateway()
    monkeypatch.setattr(reconcile, "take_run", capture)
    monkeypatch.setattr(adapter, "spend_rows", held)
    with anyio.fail_after(5), pytest.raises(RuntimeError if stop else Fenced):
        await reconcile.reconcile_run(config, "r-reconcile", adapter, executor())
    assert "r-reconcile" not in llm_record._bindings
    assert not adapter.deleted
    assert not [e for e in committed(config) if e.type == "reconcile"]


async def test_same_upstream_id_on_distinct_keys_has_distinct_totals(tmp_path: Path) -> None:
    adapter = CapableGateway()
    adapter.rows = [spend("same", 0.1), spend("same", 0.2, key_id="hash-2")]
    async with open_llm_run(tmp_path) as run:
        run.journal.append(created(key_record(run.run_id)))
        run.journal.append(created(key_record(run.run_id, key_id="hash-2", candidate_id="c-2")))
        await run.journal.commit()
        report = await reconcile.reconcile_keys(
            run.events("gateway_key"),
            run_id=run.run_id,
            journal=run.journal,
            adapter=adapter,
            executor=executor(),
            gateway=GatewayConfig(url="http://gw"),
            profile=adapter.profile,
            now=lambda: T0 + timedelta(seconds=610),
        )
        assert report.spend_usd() == 0.3
        assert [row.observed_usd for row in report.keys] == [0.1, 0.2]
        await run.journal.commit()
        assert len(run.calls()) == 2
        assert len({call.litellm_call_id for call in run.calls()}) == 2
