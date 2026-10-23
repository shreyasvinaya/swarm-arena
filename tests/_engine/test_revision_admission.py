"""Public runs abandon blocked revision validators without publishing late plans."""

import threading
from functools import partial
from typing import Any, Self

import anyio
import pytest
from _engine.test_goal import _plan
from _fakes.run import events
from pydantic import model_validator

from swarm_arena import FunctionTool, Goal, Plan, Ref, RunConfig, Swarm
from swarm_arena._engine import goal as goal_module
from swarm_arena._engine import run as run_module
from swarm_arena._journal.writer import JournalWriter
from swarm_arena.config import CatalogEntry, PlannerCatalog
from swarm_arena.planning import History, Planner

_ENTERED, _RELEASE, _RETURNED = threading.Event(), threading.Event(), threading.Event()


class BlockingRevisionTool(FunctionTool):
    class Config(FunctionTool.Config):  # ty: ignore[invalid-attribute-override]
        block: bool = False

        @model_validator(mode="after")
        def wait_for_release(self) -> Self:
            if self.block:
                _ENTERED.set()
                _RELEASE.wait()
                _RETURNED.set()
            return self


class ReplaceWithBlockingTool(Planner):
    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        assert history is not None
        current = history.revisions[-1]
        team = current.teams[0]
        agent = team.agents[0].model_copy(
            update={
                "tool": Ref(
                    path=f"{__name__}:BlockingRevisionTool",
                    config={"function": "_fakes.tools:square", "block": True},
                )
            }
        )
        return current.model_copy(update={"teams": [team.model_copy(update={"agents": [agent]})]})


@pytest.mark.parametrize("signal", ["stop_now", "deadline"])
async def test_public_run_abandons_blocked_revision_admission(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, signal: str
) -> None:
    for event in (_ENTERED, _RELEASE, _RETURNED):
        event.clear()
    cfg = run_config.model_copy(
        update={
            "settle_timeout_s": 60.0,
            "limits": run_config.limits.model_copy(update={"run_timeout_s": 180.0}),
            "planner_catalog": PlannerCatalog(
                entries=[
                    CatalogEntry(
                        name="replacement",
                        kind="tool",
                        ref=f"{__name__}:BlockingRevisionTool",
                        config={"function": "_fakes.tools:square", "block": True},
                        description="Approved replacement tool",
                    )
                ]
            ),
        }
    )
    done, stopped = anyio.Event(), anyio.Event()
    admission_returned = threading.Event()
    real_publish = JournalWriter._publish
    real_watch = run_module.RunRunner._watch_clock
    real_admit = goal_module.admit_revision

    def admit(*args: Any, **kwargs: Any) -> tuple[Plan, list[str]]:
        try:
            return real_admit(*args, **kwargs)
        finally:
            admission_returned.set()

    async def watch(self: run_module.RunRunner, deps: Any) -> None:
        if signal == "deadline":
            assert await anyio.to_thread.run_sync(_ENTERED.wait, 60)
            deps.clock._rebase(cfg.limits.run_timeout_s)
        await real_watch(self, deps)

    async def publish(self: JournalWriter) -> None:
        self._group_window_s = 0.001
        stopping = any(e.type == "stop_requested" for e in self._queue)
        await real_publish(self)
        if stopping:
            stopped.set()

    monkeypatch.setattr(run_module.RunRunner, "_watch_clock", watch)
    monkeypatch.setattr(goal_module, "admit_revision", admit)
    monkeypatch.setattr(JournalWriter, "_publish", publish)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    swarm = Swarm(cfg, replanner=ReplaceWithBlockingTool())

    async def drive() -> None:
        await swarm.run(_plan(teams=1, rounds=2))
        done.set()

    async with anyio.create_task_group() as group:
        group.start_soon(drive)
        try:
            assert await anyio.to_thread.run_sync(_ENTERED.wait, 60), "validator was not reached"
            if signal == "stop_now":
                await swarm.stop(now=True)
                with anyio.fail_after(60):
                    await stopped.wait()
            # This only observes completion; it never cancels the driver on success.
            with anyio.move_on_after(30):
                await done.wait()
            assert done.is_set(), "the run must finish without releasing the validator"
            assert not _RELEASE.is_set()
            assert events(cfg, "goal_finished")[-1].payload["reason"] == (
                "stop --now" if signal == "stop_now" else "run_timeout_s"
            )
        finally:
            _RELEASE.set()
            group.cancel_scope.cancel()
    assert await anyio.to_thread.run_sync(_RETURNED.wait, 60)
    assert await anyio.to_thread.run_sync(admission_returned.wait, 60)
    assert len(events(cfg, "plan_revision")) == 1
    assert len(events(cfg, "tool_start")) == 1
    assert events(cfg, "plan_rejected")[-1].payload["errors"] == ["revision preparation stopped"]
