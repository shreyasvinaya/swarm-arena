"""Exercise cancellation of admitted settlements through the public run engine."""

import asyncio
from contextlib import suppress
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from _fakes.run import plan
from _fakes.tools import PyTool

from swarm_arena import Limits, Ref, RunConfig, Swarm
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger, Reservation
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.ids import IdGen
from swarm_arena._store.local import LocalStore
from swarm_arena.manager import AgentContext, Manager, RoundReport


class RecoverCancelManager(Manager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        with suppress(asyncio.CancelledError):
            await asyncio.create_task(ctx.run_tool({"n": 1}))
        return RoundReport(summary="Continue after cancelling one action")


async def test_native_settlement_cancellation_in_swarm_matches_committed_replay(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = run_config.model_copy(
        update={
            "limits": Limits(max_usd=10, default_run_reservation={"usd": 1, "tokens": 1}),
            "executor": Ref(
                path="swarm_arena.executor:LocalExecutor",
                config={"workspace_root": str(tmp_path / "workspaces")},
            ),
        }
    )
    real_wait = JournalWriter._wait_for
    injected: list[int] = []

    async def wait(writer: JournalWriter, seq: int) -> None:
        if not injected and any(e.seq == seq and e.type == "settle" for e in writer._queue):
            injected.append(seq)
            task = asyncio.current_task()
            assert task is not None
            asyncio.get_running_loop().call_soon(task.cancel)
        await real_wait(writer, seq)

    real_reserve = Ledger.reserve
    seen: list[Ledger] = []

    async def reserve(ledger: Ledger, *args: Any, **kwargs: Any) -> Reservation:
        if not seen:
            seen.append(ledger)
        return await real_reserve(ledger, *args, **kwargs)

    monkeypatch.setattr(JournalWriter, "_wait_for", wait)
    monkeypatch.setattr(Ledger, "reserve", reserve)
    code = (
        "import json,os;json.dump({'data': {'y': 1.0},'usage':{'usd':0.000003}},"
        "open(os.environ['SWARM_ARENA_RESULT'],'w'))"
    )
    p = plan(rounds=1, tool=PyTool({"code": code, "grace_s": 0}))
    agent = p.teams[0].agents[0].model_copy(update={"manager": Ref.of(RecoverCancelManager())})
    p = p.model_copy(update={"teams": [p.teams[0].model_copy(update={"agents": [agent]})]})
    swarm = Swarm(config)
    assert sorted(await swarm.run(p)) == ["g"]
    assert swarm.run_id is not None
    committed = read_committed(LocalStore(Path(config.run_root) / swarm.run_id))
    assert committed is not None
    events = committed[1]
    assert len(injected) == 1
    assert sum(e.type == "reserve" for e in events) == 1
    settlements = [e for e in events if e.type == "settle"]
    assert len(settlements) == 1
    oracle = int(Decimal(str(settlements[0].payload["actual"]["usd"])) * 1_000_000)
    assert oracle == 3
    live = seen[0]
    replay = Ledger.replay(
        RunState.from_events(events), journal=live._journal, ids=IdGen(swarm.run_id, 1)
    )
    assert sum(v._micro_usd for v in live._charges.values()) == oracle
    assert sum(v._micro_usd for v in replay._charges.values()) == oracle
    assert live.free() == replay.free()
    assert live.free()._micro_usd == 8_499_997
    assert all(e.settled for e in live._entries.values())
    # The writer is closed: any accidental second append raises instead of hiding a duplicate.
    before = live._journal._next_seq
    await live.settle(settlements[0].payload["id"], Amount())
    assert live._journal._next_seq == before
