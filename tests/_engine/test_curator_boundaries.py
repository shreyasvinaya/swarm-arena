"""Curator integration with shared cancellation and reservation boundaries."""

from pathlib import Path
from typing import Any

import pytest
from _engine.test_goal import _curation_checkpoint, _curator_plan
from _fakes.engine import EngineEnv

from swarm_arena import Limits, Ref, RunConfig


@pytest.fixture
def run_config(run_config: RunConfig) -> RunConfig:
    """Fund the real engine fixture so refund checks observe spendable money."""
    return run_config.model_copy(update={"limits": Limits(max_usd=10)})


@pytest.mark.parametrize("kind", ["ordinary", "cancel", "group"])
async def test_curator_cancellation_shaped_errors_are_safe_public_failures(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import asyncio
    import traceback
    from uuid import uuid4

    import anyio
    from _fakes.curators import MergeAll
    from _fakes.engine import with_executor
    from _fakes.run import all_events, events
    from _fakes.secrecy import assert_secret_absent

    import swarm_arena._engine.goal as goal_module
    from swarm_arena import Swarm
    from swarm_arena._journal.replay import RunState

    secret = str(uuid4())
    invoked: list[str] = []
    handled: list[Exception] = []
    original_error = goal_module._curator_error

    def capture_error(error: Exception) -> str:
        handled.append(error)
        return original_error(error)

    monkeypatch.setattr(goal_module, "_curator_error", capture_error)

    async def curate(*args: Any, **kwargs: Any) -> Any:
        invoked.append(kind)
        error: BaseException
        if kind == "group":
            error = BaseExceptionGroup(
                secret, [asyncio.CancelledError(secret), RuntimeError(secret)]
            )
        elif kind == "cancel":
            error = asyncio.CancelledError(secret)
        else:
            error = RuntimeError(secret)
        error.add_note(secret)
        raise error from ValueError(secret)

    monkeypatch.setattr(MergeAll, "curate", curate)
    config = with_executor(
        run_config.model_copy(update={"curator": Ref(path="_fakes.curators:MergeAll")}),
        workspace_root=str(tmp_path / "ws"),
    )
    caught: BaseException | None = None
    result: Any = None
    try:
        with anyio.fail_after(30):
            result = await Swarm(config).run(_curator_plan(rounds=1))
    except BaseException as error:
        caught = error
    if caught is not None:
        assert_secret_absent(caught, secret)
        assert secret not in "".join(traceback.format_exception(caught))
    committed = all_events(config)
    assert secret not in repr(result)
    assert secret not in "".join(event.model_dump_json() for event in committed)
    assert invoked == [kind]
    assert caught is None
    if kind != "ordinary":
        assert len(handled) == 1
        assert_secret_absent(handled[0], secret)
        assert secret not in "".join(traceback.format_exception(handled[0]))
    assert result["g"]
    receipts = events(config, "curated")
    assert len(receipts) == 1
    error_type = "RuntimeError" if kind == "ordinary" else "_CoreRuntimeError"
    assert receipts[0].payload["errors"] == [f"curator error: {error_type}"]
    assert receipts[0].payload["summaries"] == []
    assert len(events(config, "round_closed")) == 1
    assert len(events(config, "goal_finished")) == 1
    assert events(config, "goal_finished")[0].payload["reason"] is None
    assert RunState.from_events(committed).goals["g"].status == "finished"


@pytest.mark.parametrize(
    ("estimate_micros", "reserved_micros"), [(10_000, 250_000), (400_000, 400_000)]
)
async def test_curator_native_cancel_at_reserve_admission_refunds_exactly(
    env: EngineEnv,
    monkeypatch: pytest.MonkeyPatch,
    estimate_micros: int,
    reserved_micros: int,
) -> None:
    import asyncio

    import anyio
    from _fakes.curators import MergeAll

    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.ledger import Ledger
    from swarm_arena._budget.llm_scope import current_llm_scope
    from swarm_arena._journal.replay import RunState
    from swarm_arena._journal.writer import JournalWriter
    from swarm_arena._model.ids import IdGen

    runner = await _curation_checkpoint(env)
    live = env.deps.ledger
    before = live.remaining("g/:curator")._micro_usd
    assert before > reserved_micros
    original_wait = JournalWriter._wait_for
    admitted: list[int] = []
    dispatched: list[str] = []

    async def wait(writer: JournalWriter, seq: int) -> None:
        if not admitted and any(e.seq == seq and e.type == "reserve" for e in writer._queue):
            admitted.append(seq)
            assert live.remaining("g/:curator")._micro_usd == before - reserved_micros
            task = asyncio.current_task()
            assert task is not None
            asyncio.get_running_loop().call_soon(task.cancel)
        await original_wait(writer, seq)

    async def curate(*args: Any, **kwargs: Any) -> Any:
        scope = current_llm_scope()
        assert scope is not None
        call = await scope.tranche.reserve_call(Amount(usd=estimate_micros / 1_000_000))
        dispatched.append(call)
        return []

    monkeypatch.setattr(MergeAll, "curate", curate)
    monkeypatch.setattr(JournalWriter, "_wait_for", wait)
    with anyio.fail_after(10):
        await runner._curate_round(1)
    await env.deps.journal.commit()
    assert len(admitted) == 1
    assert not dispatched
    reserves, settlements = env.events("reserve"), env.events("settle")
    assert len(reserves) == 1
    assert len(settlements) == 1
    reservation = reserves[0].payload["reservation"]
    assert reservation["holder"] == "g/:curator"
    assert reservation["kind"] == ("tranche" if estimate_micros == 10_000 else "call")
    assert Amount.model_validate(reservation["amount"])._micro_usd == reserved_micros
    assert settlements[0].payload["id"] == reservation["id"]
    actual = Amount.model_validate(settlements[0].payload["actual"])
    assert (actual.tokens, actual._micro_usd) == (0, 0)
    assert all(entry.settled for entry in live._entries.values())
    assert sum(amount._micro_usd for amount in live._charges.values()) == 0
    assert live.remaining("g/:curator")._micro_usd == before
    replay = Ledger.replay(
        RunState.from_events(env._events()), journal=env.deps.journal, ids=IdGen(env.deps.run_id, 1)
    )
    assert replay.remaining("g/:curator")._micro_usd == before
    assert all(entry.settled for entry in replay._entries.values())
    assert sum(amount._micro_usd for amount in replay._charges.values()) == 0
    receipt = env.events("curated")
    assert len(receipt) == 1
    assert receipt[0].payload["errors"] == ["curator error: _CoreRuntimeError"]
