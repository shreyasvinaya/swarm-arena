import asyncio
import sys
from pathlib import Path

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import GOAL, NoopManager

from swarm_arena import (
    FunctionTool,
    Limits,
    ProgramJudge,
    Ref,
    RunConfig,
    ScriptedManager,
    ScriptJudge,
)
from swarm_arena._errors import Fenced, _core_error
from swarm_arena._journal.event import Event
from swarm_arena._journal.reader import read_committed
from swarm_arena.executor import Job, LocalExecutor
from swarm_arena.manager import AgentContext, RoundReport
from swarm_arena.testing._harness import OfflineHarness
from swarm_arena.tool import ToolOutput

SQUARE = FunctionTool("_fakes.tools:square")


async def test_manager_error_closes_and_judges_round_before_continuing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "manager-round-isolation-sentinel"
    failure = RuntimeError(secret)
    seen: list[int] = []
    scored: list[str] = []
    harness = OfflineHarness()

    async def run_round(self: NoopManager, ctx: AgentContext) -> RoundReport:
        seen.append(ctx.round)
        if ctx.round == 3:
            scored.extend(score.candidate_id for score in ctx.scores_of(harness.candidates[1].id))
        rows = ctx.best(1)
        await ctx.run_tool({"x": float(ctx.round)}, parents=[rows[0].candidate_id] if rows else [])
        if ctx.round == 2:
            raise failure
        return RoundReport(summary="done")

    monkeypatch.setattr(NoopManager, "run_round", run_round)
    reports = await harness.run_rounds(
        NoopManager(),
        goal=GOAL,
        tool=SQUARE,
        rounds=3,
        limits=Limits(),
        config=RunConfig(),
        judges=[Ref.of(ProgramJudge("_fakes.tools:score_min_y"))],
    )
    assert seen == [1, 2, 3]
    assert reports == [RoundReport(summary="done"), RoundReport(), RoundReport(summary="done")]
    assert harness.manager_errors == [failure]
    assert [event.round for event in _events(harness, "score")] == [1, 2, 3]
    assert [event.round for event in _events(harness, "round_closed")] == [1, 2, 3]
    assert scored == [harness.candidates[1].id]
    assert [c.read_ids for c in harness.candidates] == [
        [],
        [harness.candidates[0].id],
        [harness.candidates[1].id, harness.candidates[0].id],
    ]
    (error,) = _events(harness, "agent_status")
    assert (error.round, error.team, error.agent, error.task_id) == (
        2,
        "team",
        "agent",
        "g/team/agent#r2",
    )
    assert error.payload["status"] == "finished"
    assert error.payload["reason"] == "manager error: RuntimeError"
    assert error.payload["a2a"]["id"] == error.task_id
    assert error.seq < _events(harness, "score")[1].seq < _events(harness, "round_closed")[1].seq
    assert _events(harness, "round_closed")[1].seq < _events(harness, "frontier")[2].seq
    committed = read_committed(harness.store)
    assert committed is not None
    assert [event.payload["candidate"]["read_ids"] for event in _events(harness, "tool_end")] == [
        candidate.read_ids for candidate in harness.candidates
    ]
    for event in committed[1]:
        assert_secret_absent(RuntimeError(event.model_dump_json()), secret)


async def test_harness_retains_each_failure_and_resets_them_for_a_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures = [RuntimeError("first"), ValueError("second")]

    async def run_round(self: NoopManager, ctx: AgentContext) -> RoundReport:
        raise failures[ctx.round - 1]

    monkeypatch.setattr(NoopManager, "run_round", run_round)
    harness = OfflineHarness()
    reports = await harness.run_rounds(
        NoopManager(),
        goal=GOAL,
        tool=SQUARE,
        rounds=2,
        limits=Limits(),
        config=RunConfig(),
    )
    assert reports == [RoundReport(), RoundReport()]
    assert harness.manager_errors == failures
    assert [event.payload["reason"] for event in _events(harness, "agent_status")] == [
        "manager error: RuntimeError",
        "manager error: ValueError",
    ]
    await harness.run_rounds(
        ScriptedManager([]),
        goal=GOAL,
        tool=SQUARE,
        rounds=1,
        limits=Limits(),
        config=RunConfig(),
    )
    assert harness.manager_errors == []
    assert _events(harness, "agent_status") == []


async def test_manager_control_flow_propagates_without_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = _core_error(Fenced, "run ownership lost")
    seen: list[int] = []

    async def run_round(self: NoopManager, ctx: AgentContext) -> RoundReport:
        seen.append(ctx.round)
        raise failure

    monkeypatch.setattr(NoopManager, "run_round", run_round)
    harness = OfflineHarness()
    with pytest.raises(type(failure)) as caught:
        await harness.run_rounds(
            NoopManager(),
            goal=GOAL,
            tool=SQUARE,
            rounds=3,
            limits=Limits(),
            config=RunConfig(),
        )
    assert caught.value is not failure
    assert caught.value.args == ("run ownership lost",)
    assert seen == [1]
    assert _events(harness, "agent_status") == []
    assert _events(harness, "round_closed") == []


def _events(harness: OfflineHarness, type: str) -> list[Event]:
    committed = read_committed(harness.store)
    assert committed is not None
    return [event for event in committed[1] if event.type == type]


def _script_judge() -> ScriptJudge:
    return ScriptJudge(
        command=[
            sys.executable,
            "-c",
            "import json, os; json.dump({'value': 4}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))",
        ],
        files=[],
        direction="min",
    )


async def test_harness_judges_each_round_before_the_next_snapshot() -> None:
    harness = OfflineHarness()
    manager = ScriptedManager(
        [
            {"inputs": {"x": 2.0}, "round": 1},
            {"inputs": {"x": 3.0}, "round": 2},
        ]
    )
    await harness.run_rounds(
        manager,
        goal=GOAL,
        tool=SQUARE,
        rounds=2,
        limits=Limits(),
        config=RunConfig(),
        judges=[Ref.of(ProgramJudge("_fakes.tools:score_min_y"))],
    )
    scores = _events(harness, "score")
    assert [s.payload["score"]["candidate_id"] for s in scores] == [
        c.id for c in harness.candidates
    ]
    assert scores[0].seq < _events(harness, "frontier")[1].seq
    assert [c.output.data["y"] for c in harness.candidates] == [4.0, 9.0]


async def test_harness_judge_commands_journal_command_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs: list[Job] = []
    submit = LocalExecutor.submit

    async def record(self: LocalExecutor, job: Job) -> ToolOutput:
        jobs.append(job)
        return await submit(self, job)

    monkeypatch.setattr(LocalExecutor, "submit", record)
    harness = OfflineHarness()
    await harness.run_rounds(
        ScriptedManager([{"inputs": {"x": 2.0}}]),
        goal=GOAL,
        tool=SQUARE,
        rounds=1,
        limits=Limits(tool_timeout_s=7),
        config=RunConfig(),
        judges=[Ref.of(_script_judge())],
    )
    start, end = _events(harness, "command_start")[-1], _events(harness, "command_end")[-1]
    assert start.payload["job_id"] == end.payload["job_id"]
    assert start.seq < end.seq
    assert start.payload["component"] == "script-judge"
    assert start.seq < end.seq < _events(harness, "score")[0].seq

    command_jobs = [job for job in jobs if job.kind == "command"]
    assert len(command_jobs) == 1
    assert command_jobs[0].id == start.payload["job_id"]
    assert "-e1-" in command_jobs[0].id
    assert command_jobs[0].timeout_s == 7
    assert command_jobs[0].on_started is not None
    assert "-e1-" in _events(harness, "score")[0].payload["score"]["id"]


async def test_no_judges_has_an_empty_ranking() -> None:
    harness = OfflineHarness()
    await harness.run_rounds(
        ScriptedManager([{"inputs": {"x": 2.0}, "parents": "best"}]),
        goal=GOAL,
        tool=SQUARE,
        rounds=1,
        limits=Limits(),
        config=RunConfig(),
    )
    assert len(harness.candidates) == 1
    assert harness.candidates[0].parent_ids == []
    assert _events(harness, "score") == []


async def test_multiple_judges_rank_with_the_first() -> None:
    harness = OfflineHarness()
    await harness.run_rounds(
        ScriptedManager(
            [
                {"inputs": {"x": 3.0}, "round": 1},
                {"inputs": {"x": 1.0}, "round": 1},
                {"inputs": {"x": 2.0}, "parents": "best", "round": 2},
            ]
        ),
        goal=GOAL,
        tool=SQUARE,
        rounds=2,
        limits=Limits(),
        config=RunConfig(),
        judges=[
            Ref.of(ProgramJudge("_fakes.tools:score_min_y")).model_copy(update={"name": "quality"}),
            Ref.of(ProgramJudge("_fakes.tools:score_max_y")).model_copy(update={"name": "other"}),
        ],
    )
    assert harness.candidates[2].parent_ids == [harness.candidates[1].id]
    assert len(_events(harness, "score")) == 6


async def test_judge_errors_are_journaled_once_per_candidate() -> None:
    harness = OfflineHarness()
    await harness.run_rounds(
        ScriptedManager([{"inputs": {"x": 2.0}}, {"inputs": {"x": 3.0}}]),
        goal=GOAL,
        tool=SQUARE,
        rounds=1,
        limits=Limits(),
        config=RunConfig(),
        judges=[Ref.of(ProgramJudge("_fakes.tools:raising_score"))],
    )
    errors = _events(harness, "judge_error")
    assert [e.payload["candidate_id"] for e in errors] == [c.id for c in harness.candidates]
    assert _events(harness, "score") == []


async def test_harness_budget_cap_and_round_grants_are_real() -> None:
    harness = OfflineHarness()
    reports = await harness.run_rounds(
        ScriptedManager([{"inputs": {"x": 2.0}}]),
        goal=GOAL,
        tool=SQUARE,
        rounds=1,
        limits=Limits(max_usd=1),
        config=RunConfig(),
    )
    assert reports == [RoundReport()]
    assert harness.candidates == []
    started = _events(harness, "round_started")[0]
    assert started.payload["shares"] == {"g/team/agent": {"tokens": 0, "usd": 0.85}}
    assert _events(harness, "tool_start") == []


async def test_harness_carries_notes_and_snapshots_learnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, str, list[str]]] = []

    async def run_round(self: NoopManager, ctx: AgentContext) -> RoundReport:
        observed.append((ctx.round, ctx.notes, [item.id for item in ctx.search_learnings("")]))
        if ctx.round == 1:
            candidate = await ctx.run_tool({"x": 1.0})
            await ctx.post_learning("Baseline", "Try this first", [candidate.id])
            await ctx.save_notes("Use the baseline")
        return RoundReport(summary="done")

    monkeypatch.setattr(NoopManager, "run_round", run_round)
    harness = OfflineHarness()
    await harness.run_rounds(
        NoopManager(),
        goal=GOAL,
        tool=SQUARE,
        rounds=2,
        limits=Limits(),
        config=RunConfig(),
    )
    assert observed == [(1, "", []), (2, "Use the baseline", [harness.learnings[0].id])]
    assert _events(harness, "learning")[0].seq < _events(harness, "frontier")[1].seq


async def test_harness_uses_a_temp_arena_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    h = OfflineHarness()
    await h.run_rounds(
        ScriptedManager([{"inputs": {"x": 1.0}}]),
        goal=GOAL,
        tool=SQUARE,
        rounds=1,
        limits=Limits(),
        config=RunConfig(
            arena_root=str(tmp_path / "real-arena"), board=Ref(path="unavailable:Board")
        ),
    )
    assert all("preflight_complete" not in e.payload for e in _events(h, "plan_revision"))
    root = Path(h.store.url).parent.parent
    assert (root / "arena" / GOAL.namespace / "runs" / "local.json").exists()
    assert (root / "cache" / f"{GOAL.namespace}.sqlite").exists()
    assert not (tmp_path / "real-arena").exists()
    assert not (tmp_path / "xdg").exists()


@pytest.mark.parametrize("kind", ["group", "cancel"])
async def test_nonstandard_manager_reason_uses_originating_type(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    secret = "harness-manager-secret"
    failure = (
        ExceptionGroup(secret, [ValueError(secret)])
        if kind == "group"
        else asyncio.CancelledError(secret)
    )

    async def fail(self: NoopManager, ctx: AgentContext) -> RoundReport:
        raise failure

    monkeypatch.setattr(NoopManager, "run_round", fail)
    harness = OfflineHarness()
    await harness.run_rounds(
        NoopManager(),
        goal=GOAL,
        tool=SQUARE,
        rounds=1,
        limits=Limits(),
        config=RunConfig(),
        judges=[],
    )
    [event] = _events(harness, "agent_status")
    assert event.payload["reason"] == f"manager error: {type(failure).__name__}"
    assert_secret_absent(RuntimeError(event.model_dump_json()), secret)


async def test_harness_cancellation_abandons_board_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import anyio

    from swarm_arena._board.arena import ArenaBoard
    from swarm_arena._board.frontier import Frontier

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = ArenaBoard.frontier

    def frontier(board: ArenaBoard) -> Frontier:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(30)
        return original(board)

    monkeypatch.setattr(ArenaBoard, "frontier", frontier)
    cancelled = anyio.Event()
    scope = anyio.CancelScope()

    async def run() -> None:
        with scope:
            await OfflineHarness().run_rounds(
                ScriptedManager([]),
                goal=GOAL,
                tool=SQUARE,
                rounds=1,
                limits=Limits(),
                config=RunConfig(),
            )
        cancelled.set()

    task = asyncio.create_task(run())
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 15)
        scope.cancel()
        with anyio.move_on_after(3):
            await cancelled.wait()
        assert cancelled.is_set(), "harness retained its board storage worker"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        for worker in workers:
            if worker.name.startswith("swarm-arena-daemon-"):
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()


async def test_harness_honors_run_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    import anyio

    async def wait(self: NoopManager, ctx: AgentContext) -> RoundReport:
        await anyio.sleep_forever()

    monkeypatch.setattr(NoopManager, "run_round", wait)
    harness = OfflineHarness()
    with anyio.fail_after(10), pytest.raises(RuntimeError, match=r"^offline harness stopped$"):
        await harness.run_rounds(
            NoopManager(),
            goal=GOAL,
            tool=SQUARE,
            rounds=1,
            limits=Limits(run_timeout_s=1),
            config=RunConfig(),
        )


def test_harness_documents_run_deadline() -> None:
    doc = OfflineHarness.run_rounds.__doc__
    assert doc is not None
    assert "Raises" in doc
    assert "Notes" in doc
    assert "run_timeout_s" in doc
    assert "offline harness stopped" in doc
