"""Assembled gateway exit checks with real tools and loopback-only HTTP."""

import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.gateway import FakeGateway
from _fakes.journals import committed_events
from _fakes.tools import FixedJudge

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    FunctionTool,
    Goal,
    Limits,
    Plan,
    Ref,
    RoundSpec,
    RunConfig,
    ScriptedManager,
    Swarm,
    TeamSpec,
)
from swarm_arena._board.trace import trace
from swarm_arena._cli.main import main
from swarm_arena.board import ArenaBoard, Candidate
from swarm_arena.config import GatewayConfig, SecretsPolicy
from swarm_arena.tool import Estimate, RunStatus


def _config(
    gateway: FakeGateway,
    root: Path,
    *,
    require_hard_cap: bool | None = False,
    request_timeout_s: float = 600,
) -> RunConfig:
    return RunConfig(
        run_root=str(root / "runs"),
        arena_root=str(root / "arena"),
        artifact_root=str(root / "artifacts"),
        settle_timeout_s=60.0,
        secrets=SecretsPolicy(allow=["OPENAI_API_KEY"]),
        llm_gateway=GatewayConfig(
            url=gateway.url,
            require_hard_cap=require_hard_cap,
            request_timeout_s=request_timeout_s,
        ),
        limits=Limits(
            max_usd=5.0,
            max_tool_runs_per_round=2,
            default_run_reservation=Estimate(tokens=100_000, usd=0.5),
        ),
    )


def _team(name: str, *, env_only: bool = False) -> TeamSpec:
    keyed = name != "plain"
    tool = FunctionTool(
        "_fakes.tools:llm_tool" if keyed else "_fakes.tools:plain_tool",
        providers=["openai"] if keyed and not env_only else [],
        env=["OPENAI_API_KEY"] if keyed and env_only else [],
    )
    requests = [
        {"inputs": {"prompt": f"{name}-{index}"} if keyed else {"x": float(index)}}
        for index in range(2)
    ]
    return TeamSpec(
        name=name,
        rationale="Exercise real tool execution",
        agents=[
            AgentSpec(
                name="a",
                rationale="Make two deterministic requests",
                tool=Ref.of(tool),
                manager=Ref.of(ScriptedManager(requests)),
            )
        ],
    )


def _plan(*, env_only: bool = False, converge: bool = False) -> Plan:
    return Plan(
        goal=Goal(id="g", namespace="gateway-exit"),
        teams=[_team("llm", env_only=env_only), _team("plain")],
        judges=[Ref.of(FixedJudge({"value": 1}))],
        ranking="fixed",
        rounds=[RoundSpec()],
        converge=ConvergeSpec(mode="team", team=_team("final"), max_join_attempts=0)
        if converge
        else ConvergeSpec(),
    )


def _run_dir(root: Path) -> Path:
    paths = list((root / "runs").iterdir())
    assert len(paths) == 1
    return paths[0]


def _events(root: Path, kind: str | None = None) -> list[dict[str, Any]]:
    return [e for e in committed_events(_run_dir(root)) if kind is None or e["type"] == kind]


def _candidates(root: Path) -> list[Candidate]:
    return [Candidate.model_validate(e["payload"]["candidate"]) for e in _events(root, "tool_end")]


def _last_reconcile(root: Path) -> dict[str, Any]:
    return _events(root, "reconcile")[-1]["payload"]


def _calls(gateway: FakeGateway, path: str) -> list[dict[str, Any]]:
    return [body for _, route, body in gateway.requests if route == path]


async def _run_swarm(
    gateway: FakeGateway,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    require_hard_cap: bool | None = False,
    env_only: bool = False,
    converge: bool = False,
) -> list[Candidate]:
    monkeypatch.setenv("LITELLM_MASTER_KEY", gateway.master_key)
    config = _config(gateway, root, require_hard_cap=require_hard_cap)
    if converge:
        # Two final tool runs need 1 USD, more than the normal 0.75 USD reserve.
        config = config.model_copy(update={"reserve_fraction": 0.25})
    with anyio.fail_after(120):
        result = await Swarm(config).run(_plan(env_only=env_only, converge=converge))
    assert result["g"]
    return _candidates(root)


def _assert_full_charge(root: Path) -> None:
    creations = [
        e["payload"] for e in _events(root, "gateway_key") if e["payload"]["action"] == "created"
    ]
    assert creations
    reserves = {
        e["payload"]["reservation"]["id"]: e["payload"]["reservation"]
        for e in _events(root, "reserve")
    }
    settles = {e["payload"]["id"]: e["payload"] for e in _events(root, "settle")}
    for key in creations:
        identifier = key["reservation_id"]
        assert reserves[identifier]["keep_charged"] is True
        assert settles[identifier]["keep_charged"] is True
        assert settles[identifier]["charged"] == reserves[identifier]["amount"]
        assert settles[identifier]["estimated"] is True


async def _reconcile_cli(config: RunConfig, root: Path) -> None:
    path = root / "reconcile.toml"
    path.write_text(config.dumps())
    with anyio.fail_after(60):
        assert (
            await anyio.to_thread.run_sync(
                main, ["reconcile", str(_run_dir(root)), "--config", str(path)]
            )
            == 0
        )


async def test_estimate_fits_max_usd(
    fake_gateway: FakeGateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _run_swarm(fake_gateway, tmp_path, monkeypatch)
    usd = _events(tmp_path, "run_started")[0]["payload"]["estimate"]["usd"]
    assert usd == pytest.approx(2 * 1 * 2 * 0.5 / 0.85)
    assert usd < 5.0


@pytest.mark.parametrize("env_only", [False, True])
async def test_key_per_llm_tool_run_only(
    fake_gateway: FakeGateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_only: bool
) -> None:
    candidates = await _run_swarm(fake_gateway, tmp_path, monkeypatch, env_only=env_only)
    llm = [c for c in candidates if c.team == "llm"]
    plain = [c for c in candidates if c.team == "plain"]
    assert len(llm) == len(plain) == 2
    assert all(c.output.status == RunStatus.OK for c in candidates)
    assert sorted(k["alias"] for k in fake_gateway.keys.values()) == sorted(
        f"cand-{c.id}" for c in llm
    )
    assert all(c.llm_calls_source == "gateway" and c.llm_call_ids for c in llm)
    assert all(c.llm_calls_source == "none" and not c.llm_call_ids for c in plain)
    assert [c.output.data for c in llm] == [{"reply": "ok"}, {"reply": "ok"}]
    assert sorted(c.output.data["y"] for c in plain) == [0.0, 1.0]
    assert len(_calls(fake_gateway, "/v1/chat/completions")) == 2
    assert all(k["blocked"] and not k["deleted"] for k in fake_gateway.keys.values())


async def test_full_reservation_held(
    fake_gateway: FakeGateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _run_swarm(fake_gateway, tmp_path, monkeypatch)
    _assert_full_charge(tmp_path)


async def test_reconcile_labels(
    fake_gateway: FakeGateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta

    from swarm_arena._gateway import litellm_proxy, reconcile
    from swarm_arena._gateway.adapter import ProfileReport
    from swarm_arena._model.canonical import utc_now

    await _run_swarm(fake_gateway, tmp_path, monkeypatch)
    assert {k["label"] for k in _last_reconcile(tmp_path)["keys"]} == {"estimated"}
    original_preflight = litellm_proxy.LiteLLMProxy.preflight
    profiles: list[ProfileReport] = []

    async def observed_preflight(self: Any) -> ProfileReport:
        profile = await original_preflight(self)
        profiles.append(profile)
        return profile

    real_reconcile = reconcile.reconcile_keys
    now = utc_now() + timedelta(days=1)
    closures = [
        datetime.fromisoformat(e["payload"]["closed_at"])
        for e in _events(tmp_path, "gateway_key")
        if e["payload"]["action"] == "closed"
    ]
    assert len(closures) == 2
    assert all((now - closure).total_seconds() > 600 for closure in closures)

    async def controlled(*args: Any, **kwargs: Any) -> Any:
        return await real_reconcile(*args, **kwargs, now=lambda: now)

    monkeypatch.setattr(litellm_proxy.LiteLLMProxy, "preflight", observed_preflight)
    monkeypatch.setattr(reconcile, "reconcile_keys", controlled)
    before = len(_calls(fake_gateway, "/model/info"))
    await _reconcile_cli(_config(fake_gateway, tmp_path), tmp_path)
    assert len(_calls(fake_gateway, "/model/info")) == before + 1
    assert len(profiles) == 1
    assert profiles[0].verified is False
    assert profiles[0].failures
    assert profiles[0].request_timeout_s is None
    assert profiles[0].batch_write_interval_s is None
    keys = _last_reconcile(tmp_path)["keys"]
    assert len(keys) == 2
    assert {k["label"] for k in keys} == {"estimated"}
    assert all(k["observed_usd"] is None and not k["deleted"] for k in keys)
    assert not _calls(fake_gateway, "/key/delete")
    _assert_full_charge(tmp_path)


async def test_reconcile_preflight_timeout_stays_estimated(
    fake_gateway: FakeGateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from swarm_arena._gateway import reconcile
    from swarm_arena._gateway.adapter import GatewayError

    await _run_swarm(fake_gateway, tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    fake_gateway.entered["/health/liveliness"] = entered
    fake_gateway.hold["/health/liveliness"] = release
    original = reconcile._external
    expired = []

    async def bounded(operation: Any, deadline: float) -> Any:
        if getattr(operation, "__name__", None) != "preflight":
            return await original(operation, deadline)

        async def expire(scope: anyio.CancelScope) -> None:
            assert await anyio.to_thread.run_sync(entered.wait, 60)
            scope.cancel()

        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(expire, scope)
                return await original(operation, deadline)
        expired.append(True)
        raise GatewayError("reconcile time allowance exhausted")

    monkeypatch.setattr(reconcile, "_external", bounded)
    before = len(fake_gateway.requests)
    try:
        await _reconcile_cli(_config(fake_gateway, tmp_path), tmp_path)
        assert expired == [True]
        routes = [route for _, route, _ in fake_gateway.requests[before:]]
        assert "/health/liveliness" in routes
        assert "/model/info" not in routes
        assert "/spend/logs/v2" not in routes
        keys = _last_reconcile(tmp_path)["keys"]
        assert len(keys) == 2
        assert {k["label"] for k in keys} == {"estimated"}
        assert all(k["observed_usd"] is None and not k["deleted"] for k in keys)
        assert not _calls(fake_gateway, "/key/delete")
        _assert_full_charge(tmp_path)
    finally:
        release.set()


@pytest.mark.parametrize("strict", [False, True])
async def test_soft_cap_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    with FakeGateway(strict=strict) as gateway:
        candidates = await _run_swarm(gateway, tmp_path, monkeypatch)
    manifest = json.loads((_run_dir(tmp_path) / "manifest.json").read_text())
    assert manifest["llm_gateway"]["cap"] == "soft"
    assert manifest["llm_gateway"]["verified"] is False
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["gateway-exit"])
    try:
        await anyio.to_thread.run_sync(board.frontier)
        candidate = next(c for c in candidates if c.team == "llm")
        text = "\n".join(trace(board, candidate.id))
        assert "cap soft" in text
        assert "gateway spend 0.50 USD (estimated)" in text
    finally:
        board.close()


@pytest.mark.parametrize("strict", [False, True])
async def test_hard_cap_required_fails_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    with (
        FakeGateway(strict=strict) as gateway,
        pytest.raises(ValueError, match="strict profile not verified"),
    ):
        await _run_swarm(gateway, tmp_path, monkeypatch, require_hard_cap=None)
    assert not _calls(gateway, "/key/generate")


async def test_converge_uses_the_gateway(
    fake_gateway: FakeGateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = await _run_swarm(fake_gateway, tmp_path, monkeypatch, converge=True)
    final = [c for c in candidates if c.team == "final"]
    assert len(final) == 2
    assert len(candidates) == 6
    assert all(c.output.status == RunStatus.OK for c in candidates)
    assert all(c.llm_calls_source == "gateway" and c.llm_call_ids for c in final)
    assert {k["alias"] for k in fake_gateway.keys.values()} == {
        f"cand-{c.id}" for c in candidates if c.team != "plain"
    }
    assert len(_last_reconcile(tmp_path)["keys"]) == 4
    _assert_full_charge(tmp_path)


def _observed_calls(root: Path) -> list[dict[str, Any]]:
    return [e["payload"]["call"] for e in _events(root, "llm_call")]


def _total_micro(root: Path) -> int:
    from decimal import Decimal

    return sum(int(Decimal(str(c["cost_usd"])) * 1_000_000) for c in _observed_calls(root))


@pytest.mark.parametrize("upstream_id", [None, "same-client-id"])
async def test_resume_late_rows_exact_totals_and_old_reader_isolation(
    fake_gateway: FakeGateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream_id: str | None,
) -> None:
    from swarm_arena._gateway.keys import gateway_call_id

    fake_gateway.row_overrides = {"litellm_call_id": upstream_id, "spend": 0.1}
    candidates = await _run_swarm(fake_gateway, tmp_path, monkeypatch)
    initial = _observed_calls(tmp_path)
    assert len(initial) == 2
    assert _total_micro(tmp_path) == 200_000
    candidate = next(c for c in candidates if c.team == "llm")
    run_id = candidate.run_id
    board = ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    board.open("", "", ["gateway-exit"])
    try:
        await anyio.to_thread.run_sync(board.frontier)
        with board._read_namespace() as old:
            before = old.trace(["gateway-exit"], candidate.id)
            # Make the already recorded amounts match subsequent reads, then add
            # pending rows. Visibility advances only at the explicit release.
            for _, row in fake_gateway._spend:
                row["spend"] = 0.1
                row["litellm_call_id"] = upstream_id
            fake_gateway.row_overrides.clear()
            for key_id in fake_gateway.keys:
                fake_gateway.add_spend(
                    key_id,
                    model="chat-model",
                    cost_usd=0.2,
                    tokens=(10, 2),
                    call_id=upstream_id,
                    visible_after_s=float("inf"),
                )
            assert len(_observed_calls(tmp_path)) == 2
            fake_gateway.reveal_spend()
            config = _config(fake_gateway, tmp_path)
            with anyio.fail_after(120):
                assert await Swarm(config).resume(run_id)
            assert len(_candidates(tmp_path)) == 4
            calls = _observed_calls(tmp_path)
            assert len(calls) == 4
            assert _total_micro(tmp_path) == 600_000
            assert {c["litellm_call_id"] for c in calls} == {
                gateway_call_id(row["api_key"], row["request_id"]) for _, row in fake_gateway._spend
            }
            assert {
                e["epoch"]
                for e in _events(tmp_path, "llm_call")
                if e["payload"]["call"] not in initial
            } == {2}
            assert trace(board, candidate.id) == before
            await anyio.to_thread.run_sync(board.frontier)
            assert old.trace(["gateway-exit"], candidate.id) == before
            after = trace(board, candidate.id)
            assert after != before
            late = [c for c in calls if c["candidate_id"] == candidate.id and c not in initial]
            assert len(late) == 1
            assert any(late[0]["id"] in line for line in after)
            assert late[0]["id"] not in candidate.llm_call_ids
        settlements = _events(tmp_path, "settle")
        for _ in range(2):
            await _reconcile_cli(config, tmp_path)
            assert _total_micro(tmp_path) == 600_000
            assert len(_observed_calls(tmp_path)) == 4
            assert {k["label"] for k in _last_reconcile(tmp_path)["keys"]} == {"estimated"}
            assert _events(tmp_path, "settle") == settlements
        _assert_full_charge(tmp_path)
    finally:
        board.close()


@pytest.mark.parametrize("failed_closure", [False, True])
async def test_resume_runs_remaining_round_with_new_keys(
    fake_gateway: FakeGateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_closure: bool,
) -> None:
    from swarm_arena._engine.goal import GoalRunner
    from swarm_arena._gateway.keys import gateway_call_id

    monkeypatch.setenv("LITELLM_MASTER_KEY", fake_gateway.master_key)
    config = _config(fake_gateway, tmp_path)
    if failed_closure:
        fake_gateway.fail["/key/block"] = 500
    swarm = Swarm(config)
    plan = _plan()
    teams = []
    for team in plan.teams:
        agent = team.agents[0]
        manager = ScriptedManager(
            [
                {
                    "round": round_number,
                    "inputs": {"prompt": f"round-{round_number}-{index}"}
                    if team.name == "llm"
                    else {"x": float(round_number * 10 + index)},
                }
                for round_number in (1, 2)
                for index in range(2)
            ]
        )
        teams.append(
            team.model_copy(
                update={"agents": [agent.model_copy(update={"manager": Ref.of(manager)})]}
            )
        )
    plan = plan.model_copy(update={"teams": teams, "rounds": [RoundSpec(), RoundSpec()]})
    close = GoalRunner._close_round
    stopped = anyio.Event()

    async def stop_after_round(self: GoalRunner, round: int, *, replan: bool) -> None:
        await close(self, round, replan=replan)
        if round != 1:
            return
        previous = self.deps.stops.on_change

        def notify() -> None:
            if previous is not None:
                previous()
            if self.deps.stops.stop:
                stopped.set()

        self.deps.stops.on_change = notify
        try:
            await swarm.stop()
            with anyio.fail_after(60):
                await stopped.wait()
        finally:
            self.deps.stops.on_change = previous

    with monkeypatch.context() as patch:
        patch.setattr(GoalRunner, "_close_round", stop_after_round)
        with anyio.fail_after(120):
            assert await swarm.run(plan)
    assert stopped.is_set()
    assert len(_candidates(tmp_path)) == 4
    assert len(fake_gateway.keys) == 2
    old_keys = set(fake_gateway.keys)
    if failed_closure:
        assert not [
            e for e in _events(tmp_path, "gateway_key") if e["payload"]["action"] == "closed"
        ]
        assert all(not key["blocked"] for key in fake_gateway.keys.values())
    fake_gateway.fail.clear()
    late_ids = set()
    for key_id in old_keys:
        fake_gateway.add_spend(
            key_id,
            model="chat-model",
            cost_usd=0.2,
            tokens=(10, 2),
            call_id=None,
            visible_after_s=float("inf"),
        )
        late_ids.add(gateway_call_id(key_id, fake_gateway._spend[-1][1]["request_id"]))
    assert not late_ids.intersection(c["litellm_call_id"] for c in _observed_calls(tmp_path))
    fake_gateway.reveal_spend()
    assert swarm.run_id is not None
    with anyio.fail_after(120):
        assert await Swarm(config).resume(swarm.run_id)
    candidates = _candidates(tmp_path)
    assert len(candidates) == 8
    assert all(c.output.status == RunStatus.OK for c in candidates)
    assert len(fake_gateway.keys) == 4
    assert {e["epoch"] for e in _events(tmp_path, "tool_start") if e["round"] == 2} == {2}
    assert {k["alias"] for k in fake_gateway.keys.values()} == {
        f"cand-{c.id}" for c in candidates if c.team == "llm"
    }
    assert all(
        c.llm_calls_source == "gateway" and c.llm_call_ids for c in candidates if c.team == "llm"
    )
    events = _events(tmp_path)
    first_work = next(
        index
        for index, event in enumerate(events)
        if event["epoch"] == 2 and event["type"] == "tool_start"
    )
    late_reads = [
        (index, event)
        for index, event in enumerate(events)
        if event["type"] == "llm_call" and event["payload"]["call"]["litellm_call_id"] in late_ids
    ]
    assert len(late_reads) == len(old_keys) == 2
    assert all(event["epoch"] == 2 and index < first_work for index, event in late_reads)
    if failed_closure:
        closed = [
            (index, event)
            for index, event in enumerate(events)
            if event["type"] == "gateway_key"
            and event["payload"]["action"] == "closed"
            and event["payload"]["key_id"] in old_keys
        ]
        assert {event["payload"]["key_id"] for _, event in closed} == old_keys
        assert all(event["epoch"] == 2 and index < first_work for index, event in closed)
    _assert_full_charge(tmp_path)


@pytest.mark.parametrize("terminal", [False, True])
async def test_capability_stub_observed_labels_and_repeated_totals(
    fake_gateway: FakeGateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    from datetime import datetime, timedelta

    from _fakes.reconcile import CapableGateway

    from swarm_arena._gateway import litellm_proxy, reconcile
    from swarm_arena._gateway.adapter import SpendRow, VirtualKey
    from swarm_arena._model.canonical import utc_now
    from swarm_arena.tool import Usage

    # Leave the original keys unclosed so the identity gate protects both
    # blocking and deletion, rather than only the latter.
    fake_gateway.fail["/key/block"] = 500
    await _run_swarm(fake_gateway, tmp_path, monkeypatch)
    fake_gateway.fail.clear()
    assert all(not key["blocked"] for key in fake_gateway.keys.values())
    adapter = CapableGateway()
    adapter.terminal_accounting = terminal
    verifier = litellm_proxy.LiteLLMProxy(
        url=fake_gateway.url, master_key=fake_gateway.master_key, request_timeout_s=600
    )
    identity_checks: list[tuple[str, bool | None]] = []

    async def key_matches(key: VirtualKey, *, run_id: str) -> bool | None:
        result = await verifier.key_matches(key, run_id=run_id)
        identity_checks.append((key.key_id, result))
        return result

    monkeypatch.setattr(adapter, "key_matches", key_matches)
    from swarm_arena._engine.preflight import gateway_preflight

    cap, profile = await gateway_preflight(
        _config(fake_gateway, tmp_path, require_hard_cap=True), adapter
    )
    assert cap == "hard"
    assert profile.verified
    # Re-deliver identities already committed by the initial tool read, before
    # deletion can suppress subsequent reads. These rows must count only once.
    for _, row in fake_gateway._spend:
        adapter.rows.append(
            SpendRow(
                key_id=row["api_key"],
                request_id=row["request_id"],
                litellm_call_id=row["litellm_call_id"],
                model=row["model"],
                usage=Usage(
                    input_tokens=row["prompt_tokens"], output_tokens=row["completion_tokens"]
                ),
                cost_usd=row["spend"],
                ts=datetime.fromisoformat(row["startTime"]),
                metadata={},
            )
        )
    assert len(adapter.rows) == len(_observed_calls(tmp_path)) == 2
    for key_id in fake_gateway.keys:
        for index, cost in enumerate((0.1, 0.2)):
            adapter.rows.append(
                SpendRow(
                    key_id=key_id,
                    request_id=f"stub-{index}",
                    litellm_call_id=None,
                    model="chat-model",
                    usage=Usage(input_tokens=1),
                    cost_usd=cost,
                    ts=utc_now(),
                    metadata={},
                )
            )
    real_reconcile = reconcile.reconcile_keys
    now = utc_now() + timedelta(days=1)

    async def controlled(*args: Any, **kwargs: Any) -> Any:
        return await real_reconcile(*args, **kwargs, now=lambda: now)

    monkeypatch.setattr(reconcile, "reconcile_keys", controlled)
    monkeypatch.setattr(litellm_proxy, "adapter_for", lambda config, environ: adapter)
    config = _config(fake_gateway, tmp_path)
    settlements = _events(tmp_path, "settle")
    expected = "observed" if terminal else "observed (completeness unverified)"
    # The real /key/info route must bind both metadata.run_id and key_alias.
    key_ids = list(fake_gateway.keys)
    assert len(key_ids) == 2
    first, second = (fake_gateway.keys[key_id] for key_id in key_ids)
    old_run_id, old_alias = first["metadata"]["run_id"], second["alias"]
    first["metadata"]["run_id"] = "unrelated-run"
    second["alias"] = "cand-unrelated"
    await _reconcile_cli(config, tmp_path)
    refused = _last_reconcile(tmp_path)["keys"]
    assert {key["key_id"] for key in refused} == set(key_ids)
    assert {key["label"] for key in refused} == {"estimated"}
    assert all(key["observed_usd"] is None and not key["deleted"] for key in refused)
    assert set(identity_checks) == {(key_id, False) for key_id in key_ids}
    assert not adapter.blocked
    assert not adapter.deleted
    assert not adapter.windows
    assert _total_micro(tmp_path) == 20_000
    first["metadata"]["run_id"], second["alias"] = old_run_id, old_alias
    identity_checks.clear()
    await _reconcile_cli(config, tmp_path)
    keys = _last_reconcile(tmp_path)["keys"]
    assert len(keys) == 2
    assert {k["label"] for k in keys} == {expected}
    assert all(k["observed_usd"] == 0.31 and k["deleted"] for k in keys)
    assert _last_reconcile(tmp_path)["total_usd"] == 0.62
    assert _last_reconcile(tmp_path)["total_label"] == expected
    assert not _last_reconcile(tmp_path)["total_reasons"]
    assert _total_micro(tmp_path) == 620_000
    assert set(identity_checks) == {(key_id, True) for key_id in key_ids}
    assert set(adapter.blocked) == set(key_ids)
    assert set(adapter.deleted) == set(fake_gateway.keys)
    assert len(adapter.windows) == len(identity_checks) == 2
    windows = list(adapter.windows)
    info_reads = len(_calls(fake_gateway, "/key/info"))
    # Retention after deletion cannot erase committed rows. Deleted keys must
    # not cause another identity lookup, spend read, block or delete attempt.
    adapter.rows.clear()
    await _reconcile_cli(config, tmp_path)
    repeated = _last_reconcile(tmp_path)["keys"]
    assert {k["key_id"] for k in repeated} == {k["key_id"] for k in keys}
    assert {k["label"] for k in repeated} == {expected}
    assert all(k["observed_usd"] == 0.31 and k["deleted"] for k in repeated)
    assert adapter.windows == windows
    assert len(identity_checks) == len(adapter.blocked) == len(adapter.deleted) == 2
    assert len(_calls(fake_gateway, "/key/info")) == info_reads
    assert _last_reconcile(tmp_path)["total_usd"] == 0.62
    assert _last_reconcile(tmp_path)["total_label"] == expected
    assert _total_micro(tmp_path) == 620_000
    assert len(_observed_calls(tmp_path)) == 6
    assert _events(tmp_path, "settle") == settlements


@pytest.mark.parametrize("path", ["/key/generate", "/key/block", "/spend/logs/v2"])
@pytest.mark.parametrize("failure", ["http", "timeout"])
async def test_gateway_failures_never_stop_the_run(
    fake_gateway: FakeGateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    failure: str,
) -> None:
    import threading

    from swarm_arena._gateway import litellm_proxy

    entered, release = threading.Event(), threading.Event()
    fake_gateway.entered[path] = entered
    if failure == "http":
        fake_gateway.fail[path] = 500
    else:
        fake_gateway.hold[path] = release
        original = litellm_proxy.LiteLLMProxy._run

        async def bounded(self: Any, method: str, route: str, work: Any) -> Any:
            if route != path:
                return await original(self, method, route, work)
            entered.clear()

            async def expire(scope: anyio.CancelScope) -> None:
                assert await anyio.to_thread.run_sync(entered.wait, 60)
                scope.cancel()

            # Exercise cancellation of a real held HTTP operation, then expose
            # the adapter's timeout outcome only after the request has arrived.
            with anyio.CancelScope() as scope:
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(expire, scope)
                    return await original(self, method, route, work)
            raise TimeoutError

        monkeypatch.setattr(litellm_proxy.LiteLLMProxy, "_run", bounded)
    try:
        candidates = await _run_swarm(fake_gateway, tmp_path, monkeypatch)
        assert entered.is_set()
        llm = [c for c in candidates if c.team == "llm"]
        plain = [c for c in candidates if c.team == "plain"]
        assert len(llm) == len(plain) == 2
        assert all(c.output.status == RunStatus.OK for c in plain)
        if path == "/key/generate":
            assert all(c.output.status == RunStatus.ERROR for c in llm)
            assert not _calls(fake_gateway, "/v1/chat/completions")
            assert {e["team"] for e in _events(tmp_path, "tool_start")} == {"plain"}
            promotions = [
                e["payload"]["reservation"]
                for e in _events(tmp_path, "reserve")
                if e["payload"].get("action") == "keep_charged"
            ]
            assert len(promotions) == 2
            settles = {e["payload"]["id"]: e["payload"] for e in _events(tmp_path, "settle")}
            assert all(settles[r["id"]]["charged"] == r["amount"] for r in promotions)
        else:
            assert all(c.output.status == RunStatus.OK for c in llm)
            assert len(_calls(fake_gateway, "/v1/chat/completions")) == 2
            _assert_full_charge(tmp_path)
        assert all(c.warnings for c in llm)
        assert _events(tmp_path, "final_order")
        assert (_run_dir(tmp_path) / "goals/g/solution.json").is_file()
        assert _events(tmp_path, "goal_finished")
        assert fake_gateway.master_key not in repr(_events(tmp_path))
    finally:
        release.set()


async def test_cancel_during_key_open_never_dispatches(
    fake_gateway: FakeGateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    entered, release, blocked = threading.Event(), threading.Event(), threading.Event()
    fake_gateway.entered["/key/generate"] = entered
    fake_gateway.entered["/key/block"] = blocked
    fake_gateway.hold["/key/generate"] = release
    monkeypatch.setenv("LITELLM_MASTER_KEY", fake_gateway.master_key)
    try:
        with anyio.fail_after(120):
            with anyio.CancelScope() as scope:
                async with anyio.create_task_group() as tasks:

                    async def cancel() -> None:
                        assert await anyio.to_thread.run_sync(entered.wait, 60)
                        scope.cancel()

                    tasks.start_soon(cancel)
                    await Swarm(_config(fake_gateway, tmp_path)).run(_plan())
            assert scope.cancelled_caught
            release.set()
            assert await anyio.to_thread.run_sync(blocked.wait, 60)
        assert len(fake_gateway.keys) == 1
        assert all(k["blocked"] for k in fake_gateway.keys.values())
        assert not _calls(fake_gateway, "/v1/chat/completions")
        assert not [e for e in _events(tmp_path, "tool_start") if e["team"] == "llm"]
        assert not _events(tmp_path, "gateway_key")
        promotions = [
            e["payload"]["reservation"]
            for e in _events(tmp_path, "reserve")
            if e["payload"].get("action") == "keep_charged"
        ]
        assert len(promotions) == 1
        from typing import cast

        from swarm_arena._budget.amount import Amount
        from swarm_arena._budget.ledger import Ledger
        from swarm_arena._journal.event import Event
        from swarm_arena._journal.replay import RunState
        from swarm_arena._journal.writer import JournalWriter
        from swarm_arena._model.ids import IdGen

        state = RunState.from_events([Event.model_validate(e) for e in _events(tmp_path)])
        ledger = Ledger.replay(
            state, journal=cast(JournalWriter, object()), ids=IdGen(_run_dir(tmp_path).name, 2)
        )
        assert ledger._charges[promotions[0]["id"]] == Amount.model_validate(
            promotions[0]["amount"]
        )
        assert ledger._charges[promotions[0]["id"]].usd == 0.5
    finally:
        release.set()


async def test_virtual_key_never_written(
    fake_gateway: FakeGateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tempfile

    from swarm_arena._engine.run import _local_journal_path
    from swarm_arena._exec.executor import Job
    from swarm_arena.executor import LocalExecutor
    from swarm_arena.tool import ToolOutput

    submitted: list[Job] = []
    original_submit = LocalExecutor.submit

    async def inspect_job(self: LocalExecutor, job: Job) -> ToolOutput:
        submitted.append(job)
        return await original_submit(self, job)

    monkeypatch.setattr(LocalExecutor, "submit", inspect_job)

    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    candidates = await _run_swarm(fake_gateway, tmp_path, monkeypatch, env_only=True)
    config = _config(fake_gateway, tmp_path)
    await _reconcile_cli(config, tmp_path)
    mirrors = [_local_journal_path(candidates[0].run_id, epoch) for epoch in (1, 2)]
    assert all(p.is_file() and p.is_relative_to(temp) for p in mirrors)
    assert "gateway_key" in mirrors[0].read_text()
    assert "reconcile" in mirrors[1].read_text()
    files = await anyio.to_thread.run_sync(lambda: [p for p in tmp_path.rglob("*") if p.is_file()])
    assert any(p.name == "manifest.json" for p in files)
    assert any(p.is_relative_to(tmp_path / "artifacts") for p in files)
    captured = capsys.readouterr()
    values = [fake_gateway.master_key, *(k["secret"] for k in fake_gateway.keys.values())]
    assert len(values) == 3
    assert len(submitted) == 4
    keyed = [job for job in submitted if job.secret_values]
    assert len(keyed) == 2
    assert {job.secret_values["OPENAI_API_KEY"] for job in keyed} == set(values[1:])
    assert all(
        job.secret_values
        == dict.fromkeys(
            ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"), job.secret_values["OPENAI_API_KEY"]
        )
        for job in keyed
    )
    for secret in values:
        assert all(secret not in repr(job) and secret not in repr(job.env) for job in submitted)
        assert secret not in caplog.text + captured.out + captured.err + repr(candidates)
        for path in files:
            assert secret.encode() not in path.read_bytes(), path
