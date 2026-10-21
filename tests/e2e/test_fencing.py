"""Fence a public Swarm run and inspect its committed ownership boundary."""

import contextlib
import json
import os
import signal
from pathlib import Path

import anyio
from _fakes.proc import require_visible_proc, wait_until

from swarm_arena import (
    AgentSpec,
    FunctionTool,
    Goal,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    ScriptedManager,
    Swarm,
    TeamSpec,
)
from swarm_arena._errors import Fenced
from swarm_arena._exec.procinfo import group_has_live_member
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._store.base import Store, open_store


def _plan(marker: Path) -> Plan:
    return Plan(
        goal=Goal(id="toy-goal", namespace="e2e"),
        teams=[
            TeamSpec(
                name="team",
                rationale="Keep a command alive across takeover",
                agents=[
                    AgentSpec(
                        name="worker",
                        rationale="Wait until fenced",
                        tool=Ref.of(FunctionTool("_fakes.tools:sleepy")),
                        manager=Ref.of(
                            ScriptedManager(
                                [
                                    {
                                        "round": 1,
                                        "inputs": {"seconds": 600.0, "marker": str(marker)},
                                    }
                                ]
                            )
                        ),
                    )
                ],
            )
        ],
        judges=[
            Ref.of(ProgramJudge("_fakes.tools:score_value")).model_copy(update={"name": "value"})
        ],
        ranking="value",
        rounds=[RoundSpec()],
    )


async def _run_capture(swarm: Swarm, plan: Plan, raised: list[BaseException]) -> None:
    try:
        await swarm.run(plan)
    except BaseException as error:
        raised.append(error)


async def _wait_tool_started(config: RunConfig, swarm: Swarm, marker: Path) -> tuple[Store, int]:
    def ready() -> tuple[Store, int] | None:
        if swarm.run_id is None:
            return None
        store = open_store(config.run_root).child(swarm.run_id)
        owner = read_owner(store)
        if owner is None or owner[0].head is None or not marker.exists():
            return None
        starts = [
            e
            for _, batch in read_chain(store, owner[0].head)
            for e in batch
            if e.type == "tool_start" and e.agent == "worker" and e.round == 1
        ]
        if not starts:
            return None
        [started] = starts
        data = json.loads(marker.read_text())
        assert type(data["pid"]) is int
        assert type(data["pgid"]) is int
        assert data["pid"] > 0
        assert data["pgid"] > 0
        assert started.payload["started"]["pgid"] == data["pgid"]
        assert group_has_live_member(data["pgid"])
        return store, data["pgid"]

    with anyio.fail_after(30):
        while True:
            result = await anyio.to_thread.run_sync(ready)
            if result is not None:
                return result
            await anyio.sleep(0.02)


def _take_over_retrying(store: Store) -> Lease:
    owner = read_owner(store)
    assert owner is not None
    epoch = owner[0].epoch
    for _ in range(100):
        try:
            return Lease.take_over(
                store, owner_id="intruder", host="local", ttl_s=3, force=True, expect_epoch=epoch
            )
        except Fenced:
            current = read_owner(store)
            if current is None or current[0].epoch != epoch:
                raise
    raise AssertionError("takeover did not make progress")


async def test_forced_takeover_fences_old_owner(run_config: RunConfig, tmp_path: Path) -> None:
    require_visible_proc()
    run_config = run_config.model_copy(
        update={
            "lease_ttl_s": 3,
            "executor": Ref(
                path="swarm_arena.executor:LocalExecutor",
                config={
                    "workspace_root": str(tmp_path / "workspaces"),
                },
            ),
        }
    )
    swarm = Swarm(run_config)
    marker = tmp_path / "pid.json"
    raised: list[BaseException] = []
    new: Lease | None = None
    try:
        with anyio.fail_after(60):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_run_capture, swarm, _plan(marker), raised)
                store, pgid = await _wait_tool_started(run_config, swarm, marker)
                new = await anyio.to_thread.run_sync(_take_over_retrying, store)
        assert [type(error) for error in raised] == [Fenced]
        owner = read_owner(store)
        assert owner is not None
        assert new is not None
        doc, _ = owner
        assert doc.epoch == 2
        assert doc.head == new.doc.head
        assert doc.head is not None
        assert all(name.startswith("journal/e1-") for name, _ in read_chain(store, doc.head))
        assert await anyio.to_thread.run_sync(
            lambda: wait_until(lambda: not group_has_live_member(pgid), 10)
        )
    finally:
        if new is not None:
            await anyio.to_thread.run_sync(new.release)
        if marker.exists():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(json.loads(marker.read_text())["pgid"], signal.SIGKILL)
