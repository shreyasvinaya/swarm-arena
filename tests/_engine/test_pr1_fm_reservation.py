"""Exactly-once pre-dispatch refunds through Swarm and a native action task."""

import asyncio
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes import pr1_fm as hooks

from swarm_arena import Ref, Swarm
from swarm_arena._agent import tool_run
from swarm_arena._journal.writer import JournalWriter
from tests._engine.test_pr1_fm_outcomes import config, events, plan


def micro(value: float | None) -> int:
    assert value is not None
    return int(Decimal(str(value)) * 1_000_000)


@pytest.mark.parametrize("phase", ["command", "put_edits"])
async def test_native_cancel_refunds_undispatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    hooks.ENTERED.clear()
    hooks.RELEASE.clear()
    hooks.EXITED.clear()
    hooks.BUDGETS.clear()
    original = tool_run.put_edits
    calls = 0

    def put_edits(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            hooks.ENTERED.set()
            try:
                hooks.RELEASE.wait()
            finally:
                hooks.EXITED.set()
        return original(*args)

    if phase == "put_edits":
        monkeypatch.setattr(tool_run, "put_edits", put_edits)
    p = plan("manager", "group")
    agent = (
        p.teams[0]
        .agents[0]
        .model_copy(
            update={
                "manager": Ref(path="_fakes.pr1_fm_cancel:CancelPreparation"),
                "tool": p.teams[0]
                .agents[0]
                .tool.model_copy(
                    update={
                        "path": "_fakes.pr1_fm_cancel:PreparingTool"
                        if phase == "command"
                        else "_fakes.pr1_fm:PaidTool"
                    }
                ),
            }
        )
    )
    p = p.model_copy(update={"teams": [p.teams[0].model_copy(update={"agents": [agent]})]})
    try:
        result = await Swarm(config(tmp_path)).run(p, run_id="r")
    finally:
        hooks.RELEASE.set()
    journal = events(tmp_path)
    reserves = [e["payload"]["reservation"] for e in journal if e["type"] == "reserve"]
    settlements = [e["payload"] for e in journal if e["type"] == "settle"]
    starts = [e for e in journal if e["type"] == "tool_start"]
    assert result["g"]
    assert len(reserves) == 2
    assert len(starts) == 1
    assert starts[0]["payload"]["reservation_id"] == reserves[1]["id"]
    assert len(settlements) == 2
    assert [s["id"] for s in settlements] == [r["id"] for r in reserves]
    assert settlements[0]["actual"] == {"tokens": 0, "usd": 0.0}
    assert settlements[0]["estimated"] is False
    actual = {s["id"]: s["actual"] for s in settlements}
    assert sum(micro(actual.get(r["id"], r["amount"])["usd"]) for r in reserves) == 4
    assert sum(actual.get(r["id"], r["amount"])["tokens"] for r in reserves) == 4
    assert len(hooks.BUDGETS) == 2
    assert micro(hooks.BUDGETS[0]) == micro(hooks.BUDGETS[1])


async def test_refund_durability_wait_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = JournalWriter._wait_for
    waits: list[float] = []

    async def wait_for(self: JournalWriter, seq: int) -> None:
        if any(
            e.seq == seq and e.type == "settle" and e.payload["actual"]["tokens"] == 0
            for e in self._queue
        ):
            start = time.monotonic()
            try:
                await anyio.sleep_forever()
            finally:
                waits.append(time.monotonic() - start)
        else:
            await original(self, seq)

    monkeypatch.setattr(JournalWriter, "_wait_for", wait_for)
    async with asyncio.timeout(10):
        await test_native_cancel_refunds_undispatched(tmp_path, monkeypatch, "command")
    assert len(waits) == 1
    assert waits[0] < config(tmp_path).settle_timeout_s + 1
