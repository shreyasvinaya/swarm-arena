"""Real engine services and small helpers shared by agent and engine tests."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anyio
from _fakes.leases import force_take_over
from _fakes.tools import RecordingExecutor

from swarm_arena import ScriptTool
from swarm_arena._agent.context import AgentContext
from swarm_arena._agent.repeats import RepeatIndex
from swarm_arena._agent.runtime import AgentCounters, AgentRuntime
from swarm_arena._agent.tool_run import Inflight
from swarm_arena._agent.watchdog import Watchdog
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._budget.amount import Amount
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._model.goal import FileSet
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate
from swarm_arena._scoring.check import Check
from swarm_arena._store.base import Store
from swarm_arena._store.local import LocalStore
from swarm_arena._tooling.staging import tool_identity
from swarm_arena._tooling.tool import Tool, effective_tool_ref

if TYPE_CHECKING:
    from swarm_arena._engine.round import EngineDeps
    from swarm_arena._model.run_config import RunConfig


def take_over_retrying(store: Store, *, owner_id: str = "intruder") -> Lease:
    """Retry a racing owner renewal, but never take over a newer epoch."""
    owner = read_owner(store)
    assert owner is not None
    epoch = owner[0].epoch
    return force_take_over(
        store, owner_id=owner_id, host="local", ttl_s=60, expect_epoch=epoch, attempts=100
    )


@dataclass
class RtEnv:
    rt: AgentRuntime
    run_store: LocalStore
    board: ArenaBoard
    marker: Path
    spy: RecordingExecutor
    inflight: Inflight
    result: Candidate | None = None
    _reservation_ids: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Associate reservations by calling task, including runs that never start."""
        pending: dict[int, str] = {}
        append = self.rt.journal.append_durable

        async def wrapped(draft: EventDraft) -> Event:
            event = await append(draft)
            task = anyio.get_current_task().id
            if draft.type == EventType.RESERVE and draft.payload["reservation"]["kind"] == "tool":
                pending[task] = draft.payload["reservation"]["id"]
            elif draft.type == EventType.TOOL_END:
                self._reservation_ids[draft.payload["candidate"]["id"]] = pending.pop(task)
            return event

        cast(Any, self.rt.journal).append_durable = wrapped

    def with_tool(self, tool: str | Tool) -> AgentRuntime:
        ref = Ref(path=tool) if isinstance(tool, str) else Ref.of(tool)
        execution = effective_tool_ref(ref, self.rt.round_spec.tool_overrides)
        return replace(
            self.rt,
            tool=execution.build(Tool),
            tool_ref=ref,
            execution_ref=execution,
            tool_identity=tool_identity(ref, None),
        )

    def rt_with_spy_executor(self, **rt_fields: Any) -> AgentRuntime:
        return replace(self.rt, executor=self.spy, **rt_fields)

    def rt_with_checks(self, checks: Sequence[tuple[str, Check]]) -> AgentRuntime:
        return replace(self.rt, checks=checks)

    def rt_for_agent(self, agent: str, *, entries: Sequence[tuple[str, Candidate]]) -> AgentRuntime:
        return replace(
            self.rt,
            address=f"{self.rt.goal_id}/t/{agent}",
            holder=f"{self.rt.goal_id}/t/{agent}",
            spec=self.rt.spec.model_copy(update={"name": agent}),
            own=[],
            counters=AgentCounters(),
            started_runs=0,
            watchdog=Watchdog(self.rt.limits),
            repeats=RepeatIndex(self.rt.repeats.run_id, entries),
        )

    def events(self, type: str) -> list[Event]:
        committed = read_committed(self.run_store)
        assert committed is not None
        return [event for event in committed[1] if event.type == type]

    def types_after_start(self) -> list[str]:
        types = self.committed_types()
        return types[types.index("round_started") + 1 :]

    def committed_types(self) -> list[str]:
        committed = read_committed(self.run_store)
        assert committed is not None
        return [event.type.value for event in committed[1]]

    def _reservation_id(self, candidate: Candidate | str) -> str:
        if isinstance(candidate, str):
            return candidate
        return self._reservation_ids[candidate.id]

    def reservation_state(self, c: Candidate | str) -> str:
        committed = read_committed(self.run_store)
        assert committed is not None
        return RunState.from_events(committed[1]).reservations[self._reservation_id(c)].state

    def charged(self, c: Candidate | str) -> Amount:
        committed = read_committed(self.run_store)
        assert committed is not None
        entry = RunState.from_events(committed[1]).reservations[self._reservation_id(c)]
        if entry.state == "settled" and not entry.reservation.get("keep_charged", False):
            return Amount.model_validate(entry.actual)
        return Amount.model_validate(entry.reservation["amount"])

    def with_script_tool(self, *, editable: Sequence[str], **rt_fields: Any) -> AgentRuntime:
        files = {
            "problem/a.phyk": self.rt.artifacts.put_bytes(b"a = 1\n"),
            "problem/run.py": self.rt.artifacts.put_bytes(
                b"import json, os; json.dump({'data': {}}, "
                b"open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
            ),
        }
        goal = self.rt.goal.model_copy(update={"files": FileSet(manifest=files)})
        rt = self.with_tool(ScriptTool(command=["python", "problem/run.py"], editable=editable))
        return replace(rt, goal=goal, goal_key=goal.key(), **rt_fields)

    def fence_on_append(self, type: EventType) -> None:
        append = self.rt.journal.append_durable

        async def wrapped(draft: EventDraft) -> Event:
            if draft.type == type:
                await anyio.to_thread.run_sync(partial(take_over_retrying, self.run_store))
            return await append(draft)

        cast(Any, self.rt.journal).append_durable = wrapped

    def after_append(self, type: EventType, fn: Callable[[], None]) -> None:
        append = self.rt.journal.append_durable

        async def wrapped(draft: EventDraft) -> Event:
            event = await append(draft)
            if draft.type == type:
                fn()
            return event

        cast(Any, self.rt.journal).append_durable = wrapped

    async def wait_for(self, path: Path) -> None:
        with anyio.fail_after(60):
            while not await anyio.to_thread.run_sync(path.exists):  # noqa: ASYNC110 - external file signal
                await anyio.sleep(0.01)

    async def wait_started(self) -> None:
        with anyio.fail_after(60):
            while not await anyio.to_thread.run_sync(self.events, "tool_start"):  # noqa: ASYNC110
                await anyio.sleep(0.01)

    def cancel_inflight(self) -> None:
        self.inflight.cancel_all()


class CtxEnv:
    """Build independent contexts over shared real journal and budget services."""

    def __init__(self, env: RtEnv) -> None:
        self.env = env
        self.rt = env.rt
        self.counters = env.rt.counters
        self.artifacts = env.rt.artifacts
        self.address = env.rt.address
        self._events: list[Event] = []
        append = env.rt.journal.append

        def record(draft: EventDraft) -> Event:
            event = append(draft)
            self._events.append(event)
            return event

        cast(Any, env.rt.journal).append = record

    def ctx(self, **rt_fields: Any) -> AgentContext:
        gate = rt_fields.pop("gate", lambda: None)
        agent = rt_fields.pop("agent", "a")
        tool_runs = rt_fields.pop("tool_runs_used", 0)
        learnings = rt_fields.pop("learnings_used", 0)
        replicates = rt_fields.pop("replicates", 1)
        base = self.env.rt
        if "tool" in rt_fields:
            base = self.env.with_tool(rt_fields.pop("tool"))
        limits = rt_fields.get("limits", base.limits)
        self.rt = replace(
            base,
            **{
                "address": f"{base.goal_id}/t/{agent}",
                "holder": f"{base.goal_id}/t/{agent}",
                "spec": base.spec.model_copy(update={"name": agent, "replicates": replicates}),
                "counters": AgentCounters(tool_runs=tool_runs, learnings=learnings),
                "started_runs": 0,
                "watchdog": Watchdog(limits),
                "own": [],
                "repeats": RepeatIndex(base.repeats.run_id),
                **rt_fields,
            },
        )
        self.counters, self.address = self.rt.counters, self.rt.address
        return AgentContext(self.rt, gate=gate)

    async def script_ctx(self) -> AgentContext:
        files = {
            "problem/a.phyk": self.artifacts.put_bytes(b"a = 1\n"),
            "problem/run.py": self.artifacts.put_bytes(b"print(1)\n"),
        }
        goal = self.env.rt.goal.model_copy(update={"files": FileSet(manifest=files)})
        from swarm_arena import PartSpec, Plan
        from swarm_arena._engine.goal import r0_revision_draft

        plan = Plan(
            goal=self.env.rt.goal,
            teams=[],
            judges=[],
            revision=1,
            parts=[PartSpec(name="script", goal=goal)],
        )
        await self.env.rt.journal.append_durable(r0_revision_draft(plan, plan.base_dir))
        view = self.env.board.snapshot(goal.key(), self.env.board.frontier())
        return self.ctx(
            tool=ScriptTool(command=["python", "problem/run.py"], editable=["problem/a.phyk"]),
            goal=goal,
            goal_key=goal.key(),
            part="script",
            view=view,
        )

    def events(self, type: str) -> list[Event]:
        return [event for event in self._events if event.type == type]


def with_limits(cfg: "RunConfig", **limits: Any) -> "RunConfig":
    return cfg.model_copy(update={"limits": cfg.limits.model_copy(update=limits)})


def with_executor(cfg: "RunConfig", **fields: Any) -> "RunConfig":
    return cfg.model_copy(
        update={
            "executor": cfg.executor.model_copy(
                update={"config": {**cfg.executor.config, **fields}}
            )
        }
    )


@dataclass
class EngineEnv:
    deps: "EngineDeps"
    store: LocalStore

    def _events(self) -> list[Event]:
        return [
            Event.model_validate_json(line)
            for line in self.deps.journal._local_path.read_bytes().splitlines()
        ]

    def events(self, type: str) -> list[Event]:
        return [event for event in self._events() if event.type == type]

    def event_types(self) -> list[str]:
        return [event.type.value for event in self._events()]

    def is_durable(self, type: str) -> bool:
        first = self.events(type)[0]
        head = self.deps.journal.head
        return head is not None and first.seq <= head.seq

    def candidate(self, id: str) -> Candidate:
        return self.deps.board.get_candidate(id)

    def candidates(self) -> list[Candidate]:
        return [Candidate.model_validate(ev.payload["candidate"]) for ev in self.events("tool_end")]

    def segment_index(self, seq: int) -> int:
        from swarm_arena._journal.reader import read_chain

        head = self.deps.journal.head
        assert head is not None
        return next(
            i
            for i, (_, events) in enumerate(read_chain(self.store, head))
            if any(event.seq == seq for event in events)
        )

    def on_commit(self, type: str, fn: Callable[[Event], None]) -> None:
        append = self.deps.journal.append_durable

        async def wrapped(draft: EventDraft) -> Event:
            event = await append(draft)
            if event.type == type:
                fn(event)
            return event

        cast(Any, self.deps.journal).append_durable = wrapped

    async def wait_for(self, path: Path) -> None:
        with anyio.fail_after(60):
            while not await anyio.to_thread.run_sync(path.exists):  # noqa: ASYNC110 - external file signal
                await anyio.sleep(0.01)

    def _configured(self, config: "RunConfig") -> "EngineEnv":
        from swarm_arena._budget.ledger import Ledger

        events = [
            event.model_copy(update={"payload": {"config": config.model_dump(mode="json")}})
            if event.type == EventType.RUN_CONFIG
            else event
            for event in self._events()
        ]
        ledger = Ledger.replay(
            RunState.from_events(events), journal=self.deps.journal, ids=self.deps.ids
        )
        return replace(self, deps=replace(self.deps, config=config, ledger=ledger))

    def with_limits(self, **limits: Any) -> "EngineEnv":
        return self._configured(with_limits(self.deps.config, **limits))

    def with_executor(self, **fields: Any) -> "EngineEnv":
        from swarm_arena._exec.local import LocalExecutor

        env = self._configured(with_executor(self.deps.config, **fields))
        return replace(
            env, deps=replace(env.deps, executor=env.deps.config.executor.build(LocalExecutor))
        )
