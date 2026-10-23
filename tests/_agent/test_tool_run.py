import asyncio
import dataclasses
import os
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.engine import RtEnv
from _fakes.tools import WRITE, CommandRaisesTool, PyTool, write_then_sleep

from swarm_arena import FunctionTool, RoundSpec, ScriptCheck
from swarm_arena._agent.runtime import AgentRuntime
from swarm_arena._agent.tool_run import execute_tool_run
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import requesting_call
from swarm_arena._errors import Fenced, SoftError, _core_error
from swarm_arena._journal.event import EventType
from swarm_arena._journal.lease import read_owner
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.plan import Resources
from swarm_arena._model.tooling import RunStatus, ToolOutput
from swarm_arena._tooling.tool import effective_tool_timeout, tool_grace
from swarm_arena.executor import Executor, Job
from swarm_arena.manager import LimitReached


@pytest.mark.parametrize("kind", ["terminal", "limit", "provider"])
@pytest.mark.parametrize("nested", [False, True])
def test_request_hook_control_disposition(kind: str, nested: bool) -> None:
    from swarm_arena._agent.tool_run import _request_hook
    from swarm_arena._errors import (
        ProviderFailed,
        _core_category,
        _core_error,
        _core_signal,
        _SignalCategory,
        _stop_cleanup_error,
    )
    from swarm_arena.manager import LimitReached

    def fail() -> None:
        error: BaseException = (
            _stop_cleanup_error()
            if kind == "terminal"
            else _core_signal(LimitReached("stop", "the round is stopping"))
            if kind == "limit"
            else _core_error(ProviderFailed, "provider")
        )
        if nested:
            raise BaseExceptionGroup("request", [BaseExceptionGroup("hook", [error])])
        raise error

    with pytest.raises(RuntimeError if kind == "terminal" else SoftError) as caught:
        _request_hook(fail)
    assert _core_category(caught.value) is (
        _SignalCategory.TERMINAL if kind == "terminal" else _SignalCategory.DIAGNOSTIC
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "site", ["request", "stage_settings", "stage", "estimate", "dispatch", "usage", "fingerprint"]
)
async def test_tool_conversion_preserves_terminal(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, site: str, nested: bool
) -> None:
    from unittest.mock import AsyncMock, Mock

    from swarm_arena._agent import tool_run as module
    from swarm_arena._errors import _core_category, _SignalCategory, _stop_cleanup_error

    reached: list[str] = []

    def fail(*args: Any, **kwargs: Any) -> Any:
        reached.append(site)
        error = _stop_cleanup_error()
        if nested:
            raise BaseExceptionGroup("tool", [BaseExceptionGroup("boundary", [error])])
        raise error

    executor = Mock(spec=Executor, submit=AsyncMock(return_value=ToolOutput(status=RunStatus.OK)))
    rt = dataclasses.replace(rt_env.rt, executor=executor)
    if site == "request":
        monkeypatch.setattr(module, "_validate_config", fail)
    elif site == "stage_settings":
        monkeypatch.setattr(rt.tool, "_own_files", fail)
    elif site == "stage":
        monkeypatch.setattr(module, "plan_tool_stage", fail)
    elif site == "estimate":
        monkeypatch.setattr(rt.tool, "estimate", fail)
    elif site == "dispatch":
        executor.submit.side_effect = fail
    elif site == "usage":

        def amount(*args: Any, **kwargs: Any) -> Amount:
            if "tokens" in kwargs:
                fail()
            return Amount(*args, **kwargs)

        monkeypatch.setattr(module, "Amount", Mock(wraps=Amount, side_effect=amount))
    else:
        monkeypatch.setattr(rt.tool, "fingerprint", fail)

    with pytest.raises(RuntimeError) as caught:
        await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert reached == [site]
    assert _core_category(caught.value) is _SignalCategory.TERMINAL
    assert not rt_env.events("tool_end")


class _QueuedExecutor(Executor):
    def __init__(self, delegate: Executor) -> None:
        super().__init__()
        self.delegate = delegate
        self.entered = anyio.Event()
        self.release = anyio.Event()

    async def submit(self, job: Job) -> ToolOutput:
        self.entered.set()
        await self.release.wait()
        return await self.delegate.submit(job)

    async def capacity(self) -> Resources | None:
        return await self.delegate.capacity()

    async def aclose(self) -> None:
        await self.delegate.aclose()


class _CommitSpy(_QueuedExecutor):
    def __init__(self, env: RtEnv) -> None:
        super().__init__(env.rt.executor)
        self.env = env

    async def submit(self, job: Job) -> ToolOutput:
        owner = read_owner(self.env.run_store)
        assert owner is not None
        assert owner[0].head is not None
        assert owner[0].head.seq >= self.env.events("reserve")[-1].seq
        return await self.delegate.submit(job)


async def _run_capturing(
    rt: AgentRuntime, inputs: dict[str, Any], errors: list[LimitReached]
) -> None:
    try:
        await execute_tool_run(rt, inputs, parents=(), replicate=None, edits=None)
    except LimitReached as error:
        errors.append(error)


async def _run_and_store(rt: AgentRuntime, env: RtEnv) -> None:
    env.result = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)


async def test_happy_path_event_order_and_logs(rt_env: RtEnv) -> None:
    cand = await execute_tool_run(rt_env.rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert rt_env.types_after_start() == [
        "agent_counters",
        "reserve",
        "tool_start",
        "tool_end",
        "settle",
    ]
    assert cand.output.status == RunStatus.OK
    assert cand.tool_identity
    assert cand.fidelity == rt_env.rt.fidelity
    assert cand.output.logs_ref == f"goals/g/logs/{cand.id}"
    assert rt_env.run_store.exists(f"goals/g/logs/{cand.id}.out")
    assert rt_env.run_store.exists(f"goals/g/logs/{cand.id}.err")
    assert rt_env.board.get_candidate(cand.id) == cand


async def test_checks_run_before_any_reserve(rt_env: RtEnv) -> None:
    with pytest.raises(SoftError):
        await execute_tool_run(rt_env.rt, {"x": "nope"}, parents=(), replicate=None, edits=None)
    with pytest.raises(SoftError, match="c-unknown"):
        await execute_tool_run(
            rt_env.rt, {"x": 1.0}, parents=["c-unknown"], replicate=None, edits=None
        )
    assert "reserve" not in rt_env.committed_types()
    assert rt_env.rt.counters.tool_runs == 0


async def test_link_counts_no_run_and_reserves_nothing(rt_env: RtEnv) -> None:
    first = await execute_tool_run(rt_env.rt, {"x": 2.0}, parents=(), replicate=None, edits=None)
    rt_b = rt_env.rt_for_agent(
        "b", entries=rt_env.rt.repeats.added
    )  # agent a's entry, as from an earlier round
    assert await execute_tool_run(rt_b, {"x": 2.0}, parents=(), replicate=None, edits=None) == first
    assert rt_b.counters.tool_runs == 0
    assert rt_b.started_runs == 0
    assert rt_env.committed_types().count("reserve") == 1


async def test_tool_end_journals_repeat_key(rt_env: RtEnv) -> None:
    cand = await execute_tool_run(rt_env.rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    key = rt_env.rt.repeats.key(rt_env.rt, cand.request)
    assert rt_env.events("tool_end")[-1].payload["repeat_key"] == key
    assert rt_env.rt.repeats.added == [(key, cand)]


async def test_dispatch_after_reserve_commit(rt_env: RtEnv) -> None:
    rt = dataclasses.replace(rt_env.rt, executor=_CommitSpy(rt_env))
    await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)


async def test_tool_never_runs_when_start_commit_fails(rt_env: RtEnv) -> None:
    rt_env.fence_on_append(
        EventType.TOOL_START
    )  # the reserve commit succeeds, then the lease is taken over
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:touch"))
    with pytest.raises(Fenced):
        await execute_tool_run(
            rt, {"marker": str(rt_env.marker)}, parents=(), replicate=None, edits=None
        )
    committed = rt_env.committed_types()
    assert "reserve" in committed
    assert "tool_start" not in committed
    assert not rt_env.marker.exists()


async def test_fenced_at_tool_end_leaves_reservation_unsettled(rt_env: RtEnv) -> None:
    rt_env.fence_on_append(EventType.TOOL_END)
    with pytest.raises(Fenced):
        await execute_tool_run(rt_env.rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    committed = rt_env.committed_types()
    assert "reserve" in committed
    assert "tool_start" in committed
    assert "tool_end" not in committed
    assert "settle" not in committed


async def test_closed_admission_refuses_dispatch(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:touch"))
    rt_env.after_append(
        EventType.RESERVE, rt_env.inflight.close
    )  # closed right after the reserve commit
    with pytest.raises(LimitReached) as err:
        await execute_tool_run(
            rt, {"marker": str(rt_env.marker)}, parents=(), replicate=None, edits=None
        )
    assert err.value.limit == "stop"
    assert not rt_env.marker.exists()
    assert "tool_start" not in rt_env.committed_types()
    res = rt_env.events("reserve")[-1].payload["reservation"]["id"]
    assert rt_env.reservation_state(res) == "settled"
    assert rt_env.charged(res) == Amount()


@pytest.mark.parametrize("escalation", ["stop", "cancel"])
async def test_budget_close_after_reservation_preserves_only_admitted_work(
    rt_env: RtEnv, escalation: str
) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:touch"))

    def close() -> None:
        rt_env.inflight.close(budget=True)
        if escalation == "stop":
            rt_env.inflight.close()
            rt_env.inflight.close(budget=True)  # a budget latch cannot reopen hard closure
        elif escalation == "cancel":
            rt_env.inflight.cancel_all()

    rt_env.after_append(EventType.RESERVE, close)
    with pytest.raises(LimitReached):
        await execute_tool_run(
            rt, {"marker": str(rt_env.marker)}, parents=(), replicate=None, edits=None
        )
    assert not rt_env.marker.exists()
    assert "tool_start" not in rt_env.committed_types()
    res = rt_env.events("reserve")[-1].payload["reservation"]["id"]
    assert rt_env.reservation_state(res) == "settled"
    assert rt_env.charged(res) == Amount()
    with pytest.raises(LimitReached):
        await execute_tool_run(rt, {"marker": "new"}, parents=(), replicate=None, edits=None)
    assert len(rt_env.events("reserve")) == 1


async def test_admission_closed_while_queued_refuses_start(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:touch"))
    queued = _QueuedExecutor(
        rt.executor
    )  # test-local: submit sets `entered`, awaits `release`, then delegates
    rt = dataclasses.replace(rt, executor=queued)
    errors: list[
        LimitReached
    ] = []  # _run_capturing appends the LimitReached that execute_tool_run raises
    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_capturing, rt, {"marker": str(rt_env.marker)}, errors)
        with anyio.fail_after(10):
            await queued.entered.wait()
        rt_env.inflight.close()
        queued.release.set()
    err = errors[0]
    assert err.limit == "stop"
    assert not rt_env.marker.exists()
    assert "tool_start" not in rt_env.committed_types()
    res = rt_env.events("reserve")[-1].payload["reservation"]["id"]
    assert rt_env.reservation_state(res) == "settled"
    assert rt_env.charged(res) == Amount()


async def test_replicates_one_refuses_second_identical_request(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(
        FunctionTool("_fakes.tools:square", deterministic=False)
    )  # spec.replicates == 1
    await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    with pytest.raises(SoftError, match="replicates used up"):
        await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert rt.counters.tool_runs == 1
    assert rt_env.committed_types().count("tool_start") == 1


async def test_edits_stored_before_dispatch(rt_env: RtEnv) -> None:
    rt = rt_env.with_script_tool(editable=["problem/*.phyk"])
    cand = await execute_tool_run(
        rt, {}, parents=(), replicate=None, edits={"problem/a.phyk": "a = 2\n"}
    )
    sha = sha256_hex(b"a = 2\n")
    assert cand.output.status == RunStatus.OK
    assert cand.overlay == {"problem/a.phyk": sha}
    assert rt.artifacts.get_bytes(sha) == b"a = 2\n"


async def test_judge_file_edit_refused(rt_env: RtEnv) -> None:
    with pytest.warns(UserWarning, match="Python import shadowing"):
        rt = rt_env.with_script_tool(editable=["**"], private_paths=frozenset({"judge/score.py"}))
    with pytest.raises(SoftError):
        await execute_tool_run(rt, {}, parents=(), replicate=None, edits={"judge/score.py": "x"})
    assert rt.counters.tool_runs == 0


async def test_executor_exception_becomes_error_candidate(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(
        CommandRaisesTool({"code": ""})
    )  # command() raises RuntimeError("boom") inside submit
    cand = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    assert cand.output.status == RunStatus.ERROR
    assert cand.warnings == ["tool error: RuntimeError"]
    assert rt_env.reservation_state(cand) == "settled"
    assert cand.output.logs_ref is None
    assert not rt_env.run_store.exists(f"goals/g/logs/{cand.id}.out")


async def test_round_cancel_keeps_partial_output(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(
        PyTool({"code": write_then_sleep(rt_env.marker)})
    )  # result.json, then marker, then 600 s
    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                _run_and_store, rt, rt_env
            )  # stores the returned candidate in rt_env.result
            await rt_env.wait_for(rt_env.marker)
            rt_env.cancel_inflight()
    cand = rt_env.result
    assert cand is not None
    assert cand.output.status == RunStatus.CANCELLED
    assert cand.output.partial is True
    assert cand.output.data == {"step": 1}
    assert rt_env.reservation_state(cand) == "settled"
    assert "tool_end" in rt_env.committed_types()


async def test_check_commands_journal_command_start_and_end(rt_env: RtEnv) -> None:
    rt = rt_env.rt_with_checks([("sc", ScriptCheck(command=["python", "-c", "print(1)"]))])
    cand = await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    start, end = rt_env.events("command_start")[-1], rt_env.events("command_end")[-1]
    assert (start.payload["candidate_id"], start.payload["component"]) == (cand.id, "sc")
    assert end.payload["job_id"] == start.payload["job_id"]
    assert start.seq < end.seq


async def test_providers_without_usage_settle_estimated(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", providers=["openai"]))
    cand = await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert rt_env.reservation_state(cand) == "estimated"


async def test_reported_usage_settles_actual(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(PyTool({"code": WRITE.format("{'data': {}, 'usage': {'usd': 0.2}}")}))
    cand = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    assert rt_env.charged(cand) == Amount(usd=0.2)


async def test_requested_by_call_and_swarm_env(rt_env: RtEnv) -> None:
    with requesting_call("llm-x"):
        cand = await execute_tool_run(
            rt_env.rt_with_spy_executor(), {"x": 1.0}, parents=(), replicate=None, edits=None
        )
    assert cand.requested_by_call == "llm-x"
    assert {
        "SWARM_ARENA_RUN_ID",
        "SWARM_ARENA_GOAL_ID",
        "SWARM_ARENA_CANDIDATE_ID",
        "SWARM_ARENA_TEAM",
        "SWARM_ARENA_AGENT",
        "SWARM_ARENA_ROUND",
    } <= set(rt_env.spy.jobs[-1].env)


async def test_job_uses_m1a_timeout_and_partial_hook(rt_env: RtEnv) -> None:
    rt = rt_env.rt_with_spy_executor(round_spec=RoundSpec(tool_timeout_s=30))
    await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    job = rt_env.spy.jobs[-1]
    assert (
        job.timeout_s == effective_tool_timeout(rt.tool, round_timeout_s=30, limit_s=3600.0) == 30
    )
    assert job.grace_s == tool_grace(rt.tool)
    assert job.on_started is not None
    assert job.on_partial is not None


class _OutputExecutor(_QueuedExecutor):
    def __init__(self, env: RtEnv, output: Any, *, starts: bool = True) -> None:
        super().__init__(env.rt.executor)
        self.output = output
        self.starts = starts

    async def submit(self, job: Job) -> ToolOutput:
        if self.starts:
            assert job.on_started is not None
            await job.on_started(Job.Started(kind="process", job_id=job.id))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@pytest.mark.parametrize("kind", ["limit", "provider"])
@pytest.mark.parametrize("nested", [False, True])
async def test_started_tool_check_control_disposition(
    rt_env: RtEnv, kind: str, nested: bool
) -> None:
    from unittest.mock import AsyncMock, Mock

    from swarm_arena._errors import ProviderFailed, _core_signal
    from swarm_arena._scoring.check import Check

    error: BaseException = (
        _core_signal(LimitReached("stop", "the round is stopping"))
        if kind == "limit"
        else _core_error(ProviderFailed, "provider")
    )
    if nested:
        error = BaseExceptionGroup("check", [BaseExceptionGroup("hook", [error])])
    check = Mock(spec=Check, check=AsyncMock(side_effect=error))
    rt = dataclasses.replace(
        rt_env.rt,
        executor=_OutputExecutor(rt_env, ToolOutput(status=RunStatus.OK)),
        checks=[("check", check)],
    )
    candidate = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert len(rt_env.events("tool_start")) == len(rt_env.events("tool_end")) == 1
    assert candidate.output.status is RunStatus.OK
    assert len(candidate.checks) == 1
    assert not candidate.checks[0].passed
    assert candidate.checks[0].reasons == [
        f"check error: {'LimitReached' if kind == 'limit' else 'ProviderFailed'}"
    ]
    assert rt_env.reservation_state(candidate) == "settled"
    assert rt.board.get_candidate(candidate.id) == candidate
    assert candidate in rt.own


async def test_final_request_matches_stage_job_candidate_and_repeat_key(rt_env: RtEnv) -> None:
    from _fakes.tools import ContextTool

    from swarm_arena._model.base import JsonValue
    from swarm_arena._model.canonical import canonical_json
    from swarm_arena._tooling.result_file import request_json
    from swarm_arena._tooling.staging import tool_identity
    from swarm_arena._tooling.tool import Tool, effective_tool_ref

    parent = await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    code = (
        "import json, os; "
        "wire=json.load(open('swarm/request.json')); "
        "json.dump({'data': {'wire': wire, 'effective': True}}, "
        "open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
    )
    rt = rt_env.with_tool(ContextTool({"code": WRITE.format("{'data': {'effective': False}}")}))
    round_spec = RoundSpec(tool_overrides={"code": code}, tool_timeout_s=30)
    execution = effective_tool_ref(rt.tool_ref, round_spec.tool_overrides)
    rt = dataclasses.replace(
        rt,
        round=2,
        round_spec=round_spec,
        fidelity=round_spec.effective_fidelity(),
        execution_ref=execution,
        tool=execution.build(Tool),
        spec=rt.spec.model_copy(update={"replicates": 3}),
    )
    jobs: list[Job] = []
    delegate = rt.executor

    class Inspect(_QueuedExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            jobs.append(job)
            return await self.delegate.submit(job)

    rt = dataclasses.replace(rt, executor=Inspect(delegate))
    values: dict[str, JsonValue] = {"null": None, "bool": True, "int": 1, "float": 1.0}
    cand = await execute_tool_run(
        rt, {"value": values}, parents=[parent.id], replicate=2, edits=None
    )
    job = jobs[0]
    assert cand.output.status == RunStatus.OK
    assert cand.output.data["effective"] is True
    assert job.request is cand.request
    assert canonical_json(cand.request.inputs) == canonical_json({"value": values, "defaulted": 7})
    assert canonical_json(cand.output.data["wire"]) == canonical_json(request_json(cand.request))
    assert job.stage.request_json == request_json(cand.request)
    assert cand.request.round == 2
    assert cand.request.replicate == 2
    assert cand.request.round_overrides == round_spec.tool_overrides
    assert cand.parent_ids == [parent.id]
    assert cand.request.parents[0].candidate_id == parent.id
    assert job.tool == execution
    assert cand.tool_identity == tool_identity(rt.tool_ref, None)
    assert cand.tool_identity != tool_identity(execution, None)
    assert rt.repeats.added[-1] == (rt.repeats.key(rt, cand.request), cand)
    assert job.env["SWARM_ARENA_ROUND"] == "2"
    assert job.secret_env == ()
    assert job.resources == rt.spec.resources


async def test_stage_refusal_before_count_and_reserve_releases_replicate(
    rt_env: RtEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swarm_arena._agent.tool_run as pipeline
    from swarm_arena._errors import _core_signal

    original = pipeline.plan_tool_stage
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", deterministic=False))

    def refuse(**kwargs: Any) -> Any:
        raise _core_signal(SoftError("stage refused"))

    monkeypatch.setattr(pipeline, "plan_tool_stage", refuse)
    with pytest.raises(SoftError, match="stage refused"):
        await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert rt.counters.tool_runs == 0
    assert rt_env.events("reserve") == []
    monkeypatch.setattr(pipeline, "plan_tool_stage", original)
    cand = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert cand.request.replicate == 0


@pytest.mark.parametrize("kind", ["duplicates", "root", "layout", "size", "script"])
async def test_invalid_stage_never_counts_or_reserves(rt_env: RtEnv, kind: str) -> None:
    from swarm_arena._model.goal import FileSet

    with pytest.warns(UserWarning, match="Python import shadowing"):
        rt = rt_env.with_script_tool(editable=["problem/*"])
    parents: tuple[str, ...] = ()
    edits = {"problem/a.phyk": "a = 2\n"}
    if kind == "duplicates":
        parents = ("c-unknown", "c-unknown")
    elif kind == "root":
        rt = dataclasses.replace(
            rt,
            goal=rt.goal.model_copy(
                update={
                    "files": FileSet(
                        manifest={
                            "problem/a.phyk": "a" * 64,
                            "Problem/b.phyk": "b" * 64,
                        }
                    )
                }
            ),
        )
    elif kind == "layout":
        edits = {"problem/a.phyk/child": "bad"}
    elif kind == "size":
        rt = dataclasses.replace(rt, limits=rt.limits.model_copy(update={"max_edit_bytes": 1}))
    else:
        edits = {"problem/run.py": "bad"}
    with pytest.raises(SoftError):
        await execute_tool_run(rt, {}, parents=parents, replicate=None, edits=edits)
    assert rt.counters.tool_runs == 0
    assert rt_env.events("reserve") == []


async def test_repeat_refusal_and_resumed_return(rt_env: RtEnv) -> None:
    cand = await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    for _ in range(rt_env.rt.limits.max_refused_repeats):
        with pytest.raises(SoftError, match=cand.id):
            await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    with pytest.raises(LimitReached) as error:
        await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert error.value.limit == "max_refused_repeats"
    await rt_env.rt.journal.commit()
    assert rt_env.events("watchdog")[-1].payload["refused"] == cand.id
    resumed = dataclasses.replace(rt_env.rt, resumed=(cand,))
    assert await execute_tool_run(resumed, {"x": 1}, parents=(), replicate=None, edits=None) == cand
    assert len(rt_env.events("reserve")) == 1


async def test_parallel_requests_share_last_run_slot(rt_env: RtEnv) -> None:
    rt = dataclasses.replace(
        rt_env.rt, limits=rt_env.rt.limits.model_copy(update={"max_tool_runs_per_round": 1})
    )
    errors: list[LimitReached] = []
    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_capturing, rt, {"x": 1}, errors)
        tg.start_soon(_run_capturing, rt, {"x": 2}, errors)
    assert len(errors) == 1
    assert errors[0].limit == "max_tool_runs_per_round"
    assert rt.counters.tool_runs == 1
    assert len(rt_env.events("tool_end")) == 1


@pytest.mark.parametrize("where", ["constructor", "command"])
async def test_cancelled_preparation_settles_zero_and_never_launches_late(
    rt_env: RtEnv,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
) -> None:
    import threading
    from pathlib import Path

    from swarm_arena._model.tooling import Command, ToolRequest

    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    rt = rt_env.with_tool(PyTool({"code": f"open({str(rt_env.marker)!r}, 'w').close()"}))
    rt = dataclasses.replace(rt, checks=[("sc", ScriptCheck(command=["python", "-c", "print(1)"]))])
    monkeypatch.setattr(rt.tool, "providers", lambda: frozenset({"openai"}))
    original_init, original_command = PyTool.__init__, PyTool.command
    workspaces: list[Path] = []
    workers: list[threading.Thread] = []

    def block() -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(10)

    def init(self: PyTool, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if where == "constructor":
            block()

    def command(self: PyTool, request: ToolRequest, workspace: Path) -> Command:
        workspaces.append(workspace)
        if where == "command":
            block()
        (workspace / "late_write").write_text("safe")
        done.set()
        return original_command(self, request, workspace)

    monkeypatch.setattr(PyTool, "__init__", init)
    monkeypatch.setattr(PyTool, "command", command)
    try:
        with anyio.fail_after(15):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_run_and_store, rt, rt_env)
                assert await anyio.to_thread.run_sync(entered.wait, 5)
                rt_env.cancel_inflight()
            cand = rt_env.result
            assert cand is not None
            assert cand.output.status == RunStatus.CANCELLED
            assert rt_env.charged(cand) == Amount()
            assert rt_env.reservation_state(cand) == "settled"
            before = rt_env.committed_types()
            assert "tool_start" not in before
            assert "command_start" not in before
            assert not rt_env.inflight._scopes
            release.set()
            assert await anyio.to_thread.run_sync(done.wait, 5)
            with anyio.fail_after(5):
                await anyio.to_thread.run_sync(workers[0].join, 5)
            assert not workers[0].is_alive()
            assert rt_env.committed_types() == before
            assert not rt_env.marker.exists()
            assert workspaces[0].joinpath("late_write").exists()
    finally:
        release.set()
        for worker in workers:
            await anyio.to_thread.run_sync(worker.join, 5)
            assert not worker.is_alive()


async def test_outer_cancellation_keeps_unsettled_reservation(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(PyTool({"code": write_then_sleep(rt_env.marker)}))
    scopes: list[anyio.CancelScope] = []

    async def run() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)

    with anyio.fail_after(20):
        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            await rt_env.wait_for(rt_env.marker)
            scopes[0].cancel()
    assert rt_env.events("tool_start")
    assert rt_env.events("tool_end") == []
    assert rt_env.events("settle") == []
    assert not rt_env.inflight._scopes


@pytest.mark.parametrize("native", [False, True], ids=["anyio", "native"])
async def test_cancelled_reserve_admission_settles_undispatched_tool_at_zero(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    rt = rt_env.rt_with_spy_executor()
    before = rt.ledger.remaining(rt.holder)
    wait_for = rt.journal._wait_for
    injected = False

    async def cancel_at_wait(seq: int) -> None:
        nonlocal injected
        if not injected and any(
            event.type == "reserve" and event.seq == seq for event in rt.journal._queue
        ):
            injected = True
            if native:
                caller = asyncio.current_task()
                assert caller is not None
                caller.cancel()
            else:
                scope.cancel()
        await wait_for(seq)

    monkeypatch.setattr(rt.journal, "_wait_for", cancel_at_wait)
    with anyio.fail_after(5):
        with anyio.CancelScope() as scope:
            with pytest.raises(asyncio.CancelledError):
                await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
        await rt.journal.commit()
    reserves = rt_env.events("reserve")
    settlements = rt_env.events("settle")
    assert len(reserves) == len(settlements) == 1
    assert not rt_env.events("tool_start")
    assert not rt_env.events("tool_end")
    assert not rt_env.spy.jobs
    assert settlements[0].payload["id"] == reserves[0].payload["reservation"]["id"]
    assert Amount.model_validate(settlements[0].payload["actual"]) == Amount()
    assert settlements[0].payload["estimated"] is False
    assert all(entry.settled for entry in rt.ledger._entries.values())
    assert rt.ledger.remaining(rt.holder) == before


@pytest.mark.parametrize("failure", ["raise", "invalid", "nonfinite", "logs"])
async def test_connector_failures_are_safe_settled_error_candidates(
    rt_env: RtEnv, failure: str
) -> None:
    from _fakes.secrecy import assert_secret_absent

    secret = "sentinel-private-tool-value"
    outputs: dict[str, Any] = {
        "raise": RuntimeError(secret),
        "invalid": {"secret": secret},
        "nonfinite": ToolOutput.model_construct(status=RunStatus.OK, data={secret: float("nan")}),
        "logs": ToolOutput(status=RunStatus.OK, logs_ref=secret),
    }
    rt = dataclasses.replace(rt_env.rt, executor=_OutputExecutor(rt_env, outputs[failure]))
    cand = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert cand.output.status == RunStatus.ERROR
    assert rt_env.reservation_state(cand) == "settled"
    assert rt_env.charged(cand) == Amount()
    assert secret not in str(cand.warnings)
    assert cand.output.logs_ref is None
    assert secret not in str(rt_env.events("tool_end")[-1].payload)
    # Input validation also suppresses all public exception surfaces.
    with pytest.raises(SoftError) as caught:
        await execute_tool_run(rt_env.rt, {"x": secret}, parents=(), replicate=None, edits=None)
    assert_secret_absent(caught.value, secret)


async def test_output_lax_revalidation_and_log_order(rt_env: RtEnv) -> None:
    from swarm_arena._model.tooling import format_logs_ref

    out_sha = rt_env.rt.artifacts.put_bytes(b"stdout\n")
    err_sha = rt_env.rt.artifacts.put_bytes(b"stderr\n")
    output = ToolOutput.model_construct(
        status="ok", data={"v": 1}, logs_ref=format_logs_ref(out_sha, err_sha)
    )
    rt = dataclasses.replace(rt_env.rt, executor=_OutputExecutor(rt_env, output))
    cand = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert cand.output.status == RunStatus.OK
    assert cand.output.logs_ref == f"goals/g/logs/{cand.id}"
    assert rt_env.run_store.read(cand.output.logs_ref + ".out").data == b"stdout\n"
    assert rt_env.run_store.read(cand.output.logs_ref + ".err").data == b"stderr\n"


async def test_check_fencing_propagates_without_completion_or_settlement(rt_env: RtEnv) -> None:
    rt_env.fence_on_append(EventType.COMMAND_START)
    rt = rt_env.rt_with_checks([("sc", ScriptCheck(command=["python", "-c", "print(1)"]))])
    with pytest.raises(Fenced):
        await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert rt_env.events("tool_end") == []
    assert rt_env.events("settle") == []


async def test_each_check_runs_once_with_its_component(rt_env: RtEnv) -> None:
    check = ScriptCheck(command=["python", "-c", WRITE.format("{'passed': True}")])
    rt = rt_env.rt_with_checks([("first", check), ("second", check)])
    cand = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert [result.check for result in cand.checks] == ["first", "second"]
    assert [event.payload["component"] for event in rt_env.events("command_start")] == [
        "first",
        "second",
    ]
    assert len(rt_env.events("command_end")) == 2


async def test_artifact_write_failure_is_settled_error_candidate(
    rt_env: RtEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swarm_arena._agent.tool_run as pipeline

    def fail(*args: Any) -> None:
        raise OSError("private-artifact-value")

    monkeypatch.setattr(pipeline, "put_edits", fail)
    cand = await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert cand.output.status == RunStatus.ERROR
    assert cand.warnings == ["tool error: OSError"]
    assert rt_env.charged(cand) == Amount()
    assert rt_env.events("tool_start") == []


async def test_invalid_edit_precedes_replicate_validation(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", deterministic=False))
    await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    with pytest.raises(SoftError, match="not editable"):
        await execute_tool_run(rt, {"x": 1}, parents=(), replicate=1, edits={"new.txt": "text"})
    await rt.journal.commit()
    assert rt_env.events("watchdog") == []
    assert rt.counters.tool_runs == 1


async def test_bad_logs_preserve_reported_usage(rt_env: RtEnv) -> None:
    from swarm_arena._model.tooling import Usage

    output = ToolOutput(status=RunStatus.OK, usage=Usage(usd=0.2), logs_ref="bad")
    rt = dataclasses.replace(rt_env.rt, executor=_OutputExecutor(rt_env, output))
    cand = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert cand.output.status == RunStatus.ERROR
    assert cand.output.logs_ref is None
    assert rt_env.charged(cand) == Amount(usd=0.2)


@pytest.mark.parametrize(
    "hook", ["get_input_model", "_own_files", "is_deterministic", "estimate", "providers"]
)
async def test_hook_failures_do_not_expose_values(
    rt_env: RtEnv,
    monkeypatch: pytest.MonkeyPatch,
    hook: str,
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._errors import ConfigError

    secret = "private-hook-sentinel"

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ConfigError(secret)

    monkeypatch.setattr(rt_env.rt.tool, hook, fail)
    if hook in {"estimate", "providers"}:
        cand = await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
        assert cand.output.status == RunStatus.ERROR
        assert cand.warnings == ["tool error: ConfigError"]
        assert rt_env.charged(cand) == Amount()
        assert secret not in str(rt_env.events("tool_end")[-1].payload)
    else:
        with pytest.raises(SoftError) as caught:
            await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
        assert_secret_absent(caught.value, secret)
        assert rt_env.events("reserve") == []


async def test_past_run_parent_and_overlay_warning_are_retained(
    rt_env: RtEnv, tmp_path: Path
) -> None:
    from _fakes.tools import cand

    parent = cand(
        id="c-past",
        run_id="older-run",
        tool_identity="another-tool",
        overlay={"notes.txt": rt_env.rt.artifacts.put_bytes(b"notes")},
    )
    from _fakes.board import records
    from _fakes.journals import FakeRun, add_pointer

    past = FakeRun(str(tmp_path / "past"), "older-run", goal=rt_env.rt.goal)
    parent = parent.model_copy(update={"goal_id": rt_env.rt.goal_id})
    past.commit(past.plan(), *records(parent))
    add_pointer(rt_env.rt.config.arena_root, past)
    rt = dataclasses.replace(
        rt_env.rt, view=rt_env.board.snapshot(rt_env.rt.goal_key, rt_env.board.frontier())
    )
    candidate = await execute_tool_run(
        rt, {"x": 1}, parents=[parent.id], replicate=None, edits=None
    )
    assert candidate.output.status == RunStatus.OK
    assert candidate.parent_ids == [parent.id]
    assert candidate.request.parents[0].run_id == "older-run"
    assert candidate.overlay == {}
    assert candidate.warnings == [
        "base parent c-past not overlaid: made by another tool; paths: notes.txt"
    ]


async def test_collect_exception_still_completes_and_settles(rt_env: RtEnv) -> None:
    from _fakes.tools import BadCollectTool

    rt = rt_env.with_tool(BadCollectTool({"code": "pass"}))
    candidate = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    assert candidate.output.status == RunStatus.ERROR
    assert rt_env.reservation_state(candidate) == "settled"
    assert rt_env.types_after_start() == [
        "agent_counters",
        "reserve",
        "tool_start",
        "tool_end",
        "settle",
    ]


async def test_nondeterministic_run_limit_releases_pending_index(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", deterministic=False))
    rt.counters.tool_runs = rt.limits.max_tool_runs_per_round
    with pytest.raises(LimitReached):
        await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert not rt.repeats._pending
    rt.counters.tool_runs = 0
    candidate = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert candidate.request.replicate == 0


async def test_unstarted_provider_result_settles_zero(rt_env: RtEnv) -> None:
    from swarm_arena._model.tooling import Usage

    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", providers=["openai"]))
    rt = dataclasses.replace(
        rt,
        executor=_OutputExecutor(
            rt_env,
            ToolOutput(status=RunStatus.ERROR, usage=Usage(usd=0.2)),
            starts=False,
        ),
    )
    candidate = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert rt_env.reservation_state(candidate) == "settled"
    assert rt_env.charged(candidate) == Amount()
    assert rt_env.events("tool_start") == []


@pytest.mark.parametrize(
    "rt_env",
    [{"max_usd": 100}, {"max_tokens": 1_000_000}, {}],
    indirect=True,
    ids=["capped", "tokens-only", "uncapped"],
)
async def test_tool_usage_overflow_is_terminal(rt_env: RtEnv) -> None:
    from swarm_arena._budget.amount import USD_LIMIT
    from swarm_arena._budget.ledger import Ledger
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.replay import RunState

    rt = rt_env.with_tool(
        PyTool(
            {
                "code": WRITE.format(
                    "{'data': {}, 'usage': {'usd': 1000000001.0, "
                    "'input_tokens': 7, 'output_tokens': 3}}"
                )
            }
        )
    )
    candidate = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    assert candidate.output.status == RunStatus.ERROR
    assert candidate.output.usage.total_tokens == 10
    assert candidate.warnings == ["tool error: usage exceeds supported range"]
    assert rt_env.charged(candidate) == Amount(tokens=10, usd=USD_LIMIT)
    assert rt_env.events("settle")[-1].payload["usage_overflow"] is True
    assert rt_env.types_after_start()[-2:] == ["tool_end", "settle"]
    committed = read_committed(rt_env.run_store)
    assert committed is not None
    replayed = Ledger.replay(RunState.from_events(committed[1]), journal=rt.journal, ids=rt.ids)
    for ledger in (rt.ledger, replayed):
        assert ledger.run_exhausted()
        assert ledger.goal_exhausted("g")
        with pytest.raises(LimitReached) as caught:
            await ledger.reserve(rt.holder, Amount(), kind="tool", goal_id="g")
        assert caught.value.limit == "usage_overflow"


@pytest.mark.parametrize("fail_call", [2, 3])
async def test_replicate_admission_does_not_reenter_determinism_hook(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, fail_call: int
) -> None:
    from swarm_arena._model.tooling import ToolRequest

    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", deterministic=False))
    calls = 0
    secret = "PRIVATE_REPLICATE_HOOK"

    def deterministic(request: ToolRequest) -> bool:
        nonlocal calls
        calls += 1
        if calls >= fail_call:
            raise RuntimeError(secret)
        return False

    monkeypatch.setattr(rt.tool, "is_deterministic", deterministic)
    candidate = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert candidate.output.status == RunStatus.OK
    assert calls == 1
    assert not rt.repeats._pending
    assert len(rt_env.events("reserve")) == 1


async def test_release_hook_cannot_replace_success(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._model.tooling import ToolRequest

    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", deterministic=False))
    secret = "PRIVATE_RELEASE_HOOK"

    def deterministic(request: ToolRequest) -> bool:
        if rt.own:
            raise RuntimeError(secret)
        return False

    monkeypatch.setattr(rt.tool, "is_deterministic", deterministic)
    try:
        result = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    except Exception as error:
        assert_secret_absent(error, secret)
        raise
    assert result.output.status == RunStatus.OK
    assert rt_env.board.get_candidate(result.id) == result
    assert not rt.repeats._pending
    assert rt_env.reservation_state(result) == "settled"


@pytest.mark.parametrize("failure", ["budget", "fence", "cancel"])
async def test_release_keeps_primary_control_outcome(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._model.tooling import ToolRequest

    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", deterministic=False))
    secret = "PRIVATE_CLEANUP_HOOK"
    calls = 0

    def deterministic(request: ToolRequest) -> bool:
        nonlocal calls
        calls += 1
        if calls > 3:
            raise RuntimeError(secret)
        return False

    monkeypatch.setattr(rt.tool, "is_deterministic", deterministic)
    if failure == "budget":
        await rt.ledger.reserve(rt.holder, rt.ledger.remaining(rt.holder), kind="tool", goal_id="g")
        with pytest.raises(LimitReached) as caught:
            await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
        assert caught.value.limit == "budget"
        assert_secret_absent(caught.value, secret)
    elif failure == "fence":
        rt_env.fence_on_append(EventType.TOOL_START)
        with pytest.raises(Fenced) as fenced:
            await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
        assert_secret_absent(fenced.value, secret)
    else:
        queued = _QueuedExecutor(rt.executor)
        rt = dataclasses.replace(rt, executor=queued)
        async with anyio.create_task_group() as group:
            group.start_soon(_run_capturing, rt, {"x": 1}, [])
            with anyio.fail_after(10):
                await queued.entered.wait()
            group.cancel_scope.cancel()
    assert not rt.repeats._pending


async def test_part_runtime_routes_main_goal_and_retains_part_context(rt_env: RtEnv) -> None:
    from swarm_arena._agent.repeats import repeat_key
    from swarm_arena._model.goal import FileSet, Goal

    sha = rt_env.rt.artifacts.put_bytes(b"part content")
    goal = Goal(id="sub", namespace="n", files=FileSet(manifest={"part.txt": sha}))
    from swarm_arena import PartSpec, Plan
    from swarm_arena._engine.goal import r0_revision_draft

    plan = Plan(
        goal=rt_env.rt.goal,
        teams=[],
        judges=[],
        revision=1,
        parts=[PartSpec(name="part", goal=goal)],
    )
    await rt_env.rt.journal.append_durable(r0_revision_draft(plan, plan.base_dir))
    rt = dataclasses.replace(
        rt_env.rt,
        goal=goal,
        goal_key=goal.key(),
        part="part",
        checks=[
            (
                "part_check",
                ScriptCheck(
                    command=[
                        "python",
                        "-c",
                        "assert open('candidate/files/part.txt').read() == 'part content'; "
                        + WRITE.format("{'passed': True}"),
                    ]
                ),
            )
        ],
        view=rt_env.board.snapshot(goal.key(), rt_env.board.frontier()),
    )
    # A real command verifies staged part files and the main-goal environment.
    code = "import os; assert open('part.txt').read() == 'part content'; " + WRITE.format(
        "{'data': {'goal': os.environ['SWARM_ARENA_GOAL_ID']}}"
    )
    tool_rt = rt_env.with_tool(PyTool({"code": code}))
    rt = dataclasses.replace(
        rt,
        tool=tool_rt.tool,
        tool_ref=tool_rt.tool_ref,
        execution_ref=tool_rt.execution_ref,
        tool_identity=tool_rt.tool_identity,
    )
    candidate = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    assert candidate.output.status == RunStatus.OK
    assert candidate.goal_id == "g"
    assert candidate.part == "part"
    assert candidate.output.data == {"goal": "g"}
    assert len(candidate.checks) == 1
    assert candidate.checks[0].passed
    assert candidate.output.logs_ref == f"goals/g/logs/{candidate.id}"
    assert (
        rt_env.board.snapshot(goal.key(), rt_env.board.frontier()).get_candidate(candidate.id)
        == candidate
    )
    for event in rt_env.events("tool_start") + rt_env.events("tool_end"):
        assert event.goal_id == "g"
    assert rt.repeats.added[0][0] == repeat_key(
        goal_key=goal.key(), tool_ref=rt.tool_ref, request=candidate.request, deterministic=True
    )


async def test_started_runs_increment_only_after_durable_start_and_restore(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._journal.event import Event, EventDraft

    rt = rt_env.rt
    assert rt.started_runs == 0
    original = rt.journal.append_durable
    observed: list[int] = []

    async def append(draft: EventDraft) -> Event:
        if draft.type == EventType.TOOL_START:
            observed.append(rt.started_runs)
        event = await original(draft)
        if draft.type == EventType.TOOL_START:
            observed.append(rt.started_runs)
        return event

    monkeypatch.setattr(rt.journal, "append_durable", append)
    await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert observed == [0, 0]
    assert rt.started_runs == 1
    # The resume runner restores all current-round starts, including completed runs.
    restored = sum(
        e.goal_id == "g" and e.team == "t" and e.agent == "a" and e.round == 1
        for e in rt_env.events("tool_start")
    )
    resumed = dataclasses.replace(rt, started_runs=restored)
    await execute_tool_run(resumed, {"x": 2}, parents=(), replicate=None, edits=None)
    assert resumed.started_runs == 2


@pytest.mark.parametrize("failure", ["prepare", "fence", "queued", "budget"])
async def test_unstarted_runs_do_not_increment_started_count(rt_env: RtEnv, failure: str) -> None:
    rt = rt_env.rt
    if failure == "prepare":
        rt = rt_env.with_tool(CommandRaisesTool({"code": ""}))
        await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    elif failure == "fence":
        rt_env.fence_on_append(EventType.TOOL_START)
        with pytest.raises(Fenced):
            await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    elif failure == "budget":
        await rt.ledger.reserve(rt.holder, rt.ledger.remaining(rt.holder), kind="tool", goal_id="g")
        with pytest.raises(LimitReached):
            await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    else:
        queued = _QueuedExecutor(rt.executor)
        rt = dataclasses.replace(rt, executor=queued)
        errors: list[LimitReached] = []
        async with anyio.create_task_group() as group:
            group.start_soon(_run_capturing, rt, {"x": 1}, errors)
            with anyio.fail_after(10):
                await queued.entered.wait()
            assert rt.started_runs == 0
            rt_env.inflight.close()
            queued.release.set()
        assert errors[0].limit == "stop"
    assert rt.counters.tool_runs == 1
    assert rt.started_runs == 0


async def test_inflight_tracks_active_job_grace_until_collection(rt_env: RtEnv) -> None:
    from swarm_arena._agent.tool_run import Inflight

    registry = Inflight()
    assert registry.max_grace_s() == 0.0
    with anyio.CancelScope() as first, anyio.CancelScope() as second:
        await anyio.lowlevel.checkpoint()
        registry.add(first, grace_s=2.0)
        registry.add(second, grace_s=5.0)
        assert registry.max_grace_s() == 5.0
        registry.close()
        assert registry.max_grace_s() == 5.0
        registry.discard(second)
        assert registry.max_grace_s() == 2.0
        registry.discard(first)
        registry.discard(first)
        assert registry.max_grace_s() == 0.0
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", grace_s=3.0))
    queued = _QueuedExecutor(rt.executor)
    rt = dataclasses.replace(rt, executor=queued)
    async with anyio.create_task_group() as group:
        group.start_soon(_run_capturing, rt, {"x": 1}, [])
        with anyio.fail_after(10):
            await queued.entered.wait()
        assert rt_env.inflight.max_grace_s() == tool_grace(rt.tool)
        rt_env.cancel_inflight()
    assert rt_env.inflight.max_grace_s() == 0.0


async def test_seed_parents_prepend_and_deduplicate_stably(rt_env: RtEnv) -> None:
    first = await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    second = await execute_tool_run(rt_env.rt, {"x": 2}, parents=(), replicate=None, edits=None)
    rt = dataclasses.replace(rt_env.rt, seed_parents=(second.id, first.id, second.id))
    result = await execute_tool_run(
        rt,
        {"x": 3},
        parents=(first.id, second.id, first.id),
        replicate=None,
        edits=None,
        read_ids=(first.id,),
    )
    assert result.parent_ids == [second.id, first.id]
    assert [p.candidate_id for p in result.request.parents] == result.parent_ids
    assert result.read_ids == [first.id]
    assert rt_env.events("tool_end")[-1].payload["candidate"]["read_ids"] == [first.id]
    assert rt.seed_parents == (second.id, first.id, second.id)


async def test_inflight_grace_remains_during_cancel_collection(rt_env: RtEnv) -> None:
    class CollectingExecutor(_QueuedExecutor):
        def __init__(self, delegate: Executor) -> None:
            super().__init__(delegate)
            self.collecting = anyio.Event()

        async def submit(self, job: Job) -> ToolOutput:
            self.entered.set()
            try:
                await anyio.sleep_forever()
            except anyio.get_cancelled_exc_class():
                with anyio.CancelScope(shield=True):
                    self.collecting.set()
                    await self.release.wait()
                return ToolOutput(status=RunStatus.CANCELLED)
            raise AssertionError("cancelled submission returned without collection")

    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", grace_s=3.0))
    executor = CollectingExecutor(rt.executor)
    rt = dataclasses.replace(rt, executor=executor)
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(_run_capturing, rt, {"x": 1}, [])
            await executor.entered.wait()
            rt_env.inflight.close()
            rt_env.inflight.cancel_all()
            await executor.collecting.wait()
            assert rt_env.inflight.max_grace_s() == 3.0
            executor.release.set()
    assert rt_env.inflight.max_grace_s() == 0.0
    assert rt_env.events("tool_end")[-1].payload["candidate"]["output"]["status"] == "cancelled"


@pytest.mark.parametrize("mode", ["same", "raise", "invalid", "surrogate"])
async def test_durable_fingerprint_uses_effective_tool_and_contains_failures(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    runtime = rt_env.rt
    seen: list[ToolOutput] = []

    def fingerprint(output: ToolOutput) -> Any:
        seen.append(output)
        if mode == "raise":
            raise RuntimeError("fingerprint-secret-sentinel")
        if mode == "invalid":
            return 3
        if mode == "surrogate":
            return "\ud800"
        return "same"

    monkeypatch.setattr(runtime.tool, "fingerprint", fingerprint)
    candidate = await execute_tool_run(runtime, {"x": 2.0}, parents=(), replicate=None, edits=None)
    event = rt_env.events("tool_end")[-1]
    assert seen == [candidate.output]
    assert event.payload["fingerprint"] == ("same" if mode == "same" else None)
    assert "fingerprint-secret-sentinel" not in event.model_dump_json()
    if mode != "same":
        assert any("fingerprint error:" in warning for warning in candidate.warnings)


@pytest.mark.parametrize("status", [RunStatus.ERROR, RunStatus.LOST])
async def test_unusable_output_has_no_fingerprint(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, status: RunStatus
) -> None:
    async def submit(job: Job) -> ToolOutput:
        return ToolOutput(status=status, partial=True, data={"y": 1.0})

    def fingerprint(output: ToolOutput) -> str:
        raise AssertionError("unusable output reached fingerprint hook")

    monkeypatch.setattr(rt_env.rt.executor, "submit", submit)
    monkeypatch.setattr(rt_env.rt.tool, "fingerprint", fingerprint)
    candidate = await execute_tool_run(
        rt_env.rt, {"x": 2.0}, parents=(), replicate=None, edits=None
    )
    assert rt_env.events("tool_end")[-1].payload["fingerprint"] is None
    assert not candidate.warnings


async def test_tool_start_and_end_share_task_id(rt_env: RtEnv) -> None:
    candidate = await execute_tool_run(
        rt_env.rt, {"x": 1.0}, parents=(), replicate=None, edits=None
    )
    started = rt_env.events("tool_start")[-1]
    ended = rt_env.events("tool_end")[-1]
    assert started.task_id == ended.task_id == candidate.id


@pytest.mark.parametrize("phase", ["command", "started"])
async def test_connector_limit_is_a_safe_error_candidate(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from pathlib import Path
    from uuid import uuid4

    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._model.tooling import Command, ToolRequest

    secret = str(uuid4())
    error = LimitReached("budget", secret)
    error.add_note(secret)
    error.__cause__ = RuntimeError(secret)
    if phase == "command":
        rt = rt_env.with_tool(PyTool({"code": "pass"}))

        def command(self: PyTool, request: ToolRequest, workspace: Path) -> Command:
            raise error

        monkeypatch.setattr(PyTool, "command", command)
        monkeypatch.setattr(rt.tool, "providers", lambda: frozenset({"openai"}))
        inputs: dict[str, Any] = {}
    else:
        rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", providers=["openai"]))
        rt = dataclasses.replace(rt, executor=_OutputExecutor(rt_env, error))
        inputs = {"x": 1}
    try:
        candidate = await execute_tool_run(rt, inputs, parents=(), replicate=None, edits=None)
    except Exception as escaped:
        assert_secret_absent(escaped, secret)
        raise
    assert candidate.output.status == RunStatus.ERROR
    assert candidate.warnings == ["tool error: LimitReached"]
    reservation = rt_env.events("reserve")[-1].payload["reservation"]
    assert rt_env.charged(candidate) == (
        Amount.model_validate(reservation["amount"]) if phase == "started" else Amount()
    )
    assert rt_env.reservation_state(candidate) == ("estimated" if phase == "started" else "settled")
    assert len(rt_env.events("tool_start")) == int(phase == "started")
    assert len(rt_env.events("tool_end")) == len(rt_env.events("settle")) == 1
    assert secret not in rt_env.events("tool_end")[-1].model_dump_json()


@pytest.mark.parametrize("bad", ["truth", "integer", "string", "none"])
async def test_determinism_return_is_contained(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    from uuid import uuid4

    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._model.tooling import ToolRequest

    secret = str(uuid4())
    calls = 0

    class BadTruth:
        def __bool__(self) -> bool:
            error = RuntimeError(secret)
            error.add_note(secret)
            raise error from ValueError(secret)

    def deterministic(request: ToolRequest) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"truth": BadTruth(), "integer": 1, "string": "yes", "none": None}[bad]
        return False

    monkeypatch.setattr(rt_env.rt.tool, "is_deterministic", deterministic)
    with pytest.raises(SoftError) as caught:
        await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert_secret_absent(caught.value, secret, sanitized=True)
    assert isinstance(caught.value, SoftError)
    assert str(caught.value) == "invalid tool request: TypeError"
    assert not rt_env.events("reserve")
    assert rt_env.rt.counters.tool_runs == 0
    assert calls == 1
    assert not rt_env.rt.repeats._pending


async def test_determinism_fence_keeps_control_meaning(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._model.tooling import ToolRequest

    calls = 0

    def deterministic(request: ToolRequest) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _core_error(Fenced, "run ownership lost")
        return False

    monkeypatch.setattr(rt_env.rt.tool, "is_deterministic", deterministic)
    with pytest.raises(Fenced):
        await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert not rt_env.events("reserve")
    assert calls == 1
    assert not rt_env.rt.repeats._pending


async def test_fingerprint_mutation_cannot_change_candidate_or_accounting(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from uuid import uuid4

    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.replay import RunState
    from swarm_arena._model.tooling import Usage

    secret = str(uuid4())
    original = ToolOutput(status=RunStatus.OK, data={"nested": {"value": 1}}, usage=Usage(usd=0.2))
    rt = dataclasses.replace(rt_env.rt, executor=_OutputExecutor(rt_env, original))

    def fingerprint(output: ToolOutput) -> str:
        output.data["nested"]["value"] = float("nan")
        object.__setattr__(output.usage, "usd", 999)
        error = RuntimeError(secret)
        error.add_note(secret)
        raise error from ValueError(secret)

    monkeypatch.setattr(rt.tool, "fingerprint", fingerprint)
    candidate = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert candidate.output.data == {"nested": {"value": 1}}
    assert candidate.output.usage.usd == 0.2
    assert rt_env.charged(candidate) == Amount(usd=0.2)
    assert rt_env.events("tool_end")[-1].payload["fingerprint"] is None
    assert candidate.warnings == ["fingerprint error: RuntimeError"]
    committed = read_committed(rt_env.run_store)
    assert committed is not None
    assert RunState.from_events(committed[1]).candidates[candidate.id] == candidate
    assert secret not in rt_env.events("tool_end")[-1].model_dump_json()


@pytest.mark.parametrize("mode", ["timeout", "cancel", "fence"])
async def test_fingerprint_abandonment_discards_late_mutation(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import threading

    import swarm_arena._agent.tool_run as pipeline
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.replay import RunState
    from swarm_arena._model.results import Candidate

    entered, worker_done, done = anyio.Event(), anyio.Event(), anyio.Event()
    release = threading.Event()
    token = anyio.lowlevel.current_token()
    scopes: list[anyio.CancelScope] = []
    candidates: list[Candidate] = []
    fenced: list[Fenced] = []
    monkeypatch.setattr(
        pipeline, "FINGERPRINT_TIMEOUT_S", 0.1 if mode == "timeout" else 30, raising=False
    )

    def fingerprint(output: ToolOutput) -> str:
        anyio.from_thread.run_sync(entered.set, token=token)
        release.wait()
        output.data["late"] = {"value": float("nan")}
        anyio.from_thread.run_sync(worker_done.set, token=token)
        return "too late"

    async def run() -> None:
        try:
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                candidates.append(
                    await execute_tool_run(
                        rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None
                    )
                )
        except Fenced as error:
            fenced.append(error)
        finally:
            done.set()

    monkeypatch.setattr(rt_env.rt.tool, "fingerprint", fingerprint)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(run)
        try:
            with anyio.fail_after(5):
                await entered.wait()
            if mode != "timeout":
                if mode == "fence":
                    rt_env.rt.journal.fenced.set()
                scopes[0].cancel()
            with anyio.move_on_after(1):
                await done.wait()
            assert done.is_set(), "fingerprint waiter did not abandon blocked hook"
            if mode == "fence":
                assert fenced
                assert not rt_env.events("tool_end")
                assert not rt_env.events("settle")
            else:
                assert len(candidates) == 1
                candidate = candidates[0]
                assert candidate.output.data == {"x": 1, "y": 1}
                assert rt_env.reservation_state(candidate) == "settled"
                assert rt_env.events("tool_end")[-1].payload["fingerprint"] is None
                assert candidate.warnings == [
                    "fingerprint error: "
                    + ("TimeoutError" if mode == "timeout" else "CancelledError")
                ]
            before = read_committed(rt_env.run_store)
            release.set()
            with anyio.fail_after(5):
                await worker_done.wait()
            await anyio.lowlevel.checkpoint()
            assert read_committed(rt_env.run_store) == before
            if candidates:
                assert before is not None
                assert RunState.from_events(before[1]).candidates[candidates[0].id] == candidates[0]
                assert rt_env.board.get_candidate(candidates[0].id) == candidates[0]
        finally:
            release.set()


def _fingerprint_round(rt_env: RtEnv) -> Any:
    from swarm_arena import Plan, Ref, ScriptedManager
    from swarm_arena._engine.round import AgentSlot, EngineDeps, RoundRunner
    from swarm_arena._engine.stops import RunClock, StopState
    from swarm_arena._journal.views import ViewWriter
    from swarm_arena._model.results import AgentStatus

    rt = rt_env.rt
    deps = EngineDeps(
        run_id=rt.repeats.run_id,
        config=rt.config,
        store=rt_env.run_store,
        journal=rt.journal,
        ledger=rt.ledger,
        board=rt.board,
        executor=rt.executor,
        artifacts=rt.artifacts,
        ids=rt.ids,
        views=ViewWriter(rt_env.run_store, rt.journal),
        stops=StopState(),
        clock=RunClock(0),
    )
    slot = AgentSlot(
        address=rt.address,
        team=rt.team,
        spec=rt.spec,
        manager=ScriptedManager([{"inputs": {"x": 1}}]),
        tool_ref=rt.tool_ref,
        goal=rt.goal,
        goal_key=rt.goal_key,
        part=None,
        notes="",
        watchdog=rt.watchdog,
        status=AgentStatus.ACTIVE,
    )
    return RoundRunner(
        deps,
        goal_id="g",
        plan=Plan(
            goal=rt.goal,
            teams=[rt.team],
            judges=[
                Ref(
                    path="swarm_arena:ProgramJudge", config={"function": "_fakes.tools:score_min_y"}
                )
            ],
        ),
        agents=[slot],
        round_spec=rt.round_spec,
        rounds_left=1,
    )


@pytest.mark.parametrize("mode", ["stop", "deadline"])
async def test_round_finishes_with_blocked_fingerprint(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import threading

    from swarm_arena._engine import round as round_module
    from swarm_arena._journal.reader import read_committed
    from swarm_arena._journal.replay import RunState

    runner = _fingerprint_round(rt_env)
    entered, watch_done, done = anyio.Event(), anyio.Event(), anyio.Event()
    release = threading.Event()
    token = anyio.lowlevel.current_token()
    results: list[Any] = []
    original_watch = runner._watch
    monkeypatch.setattr(round_module, "BEFORE_KILL_TIMEOUT_S", 0.0)

    def fingerprint(self: FunctionTool, output: ToolOutput) -> str:
        anyio.from_thread.run_sync(entered.set, token=token)
        release.wait()
        output.data["late"] = float("nan")
        return "late"

    async def watch(*args: Any) -> None:
        try:
            await original_watch(*args)
        finally:
            watch_done.set()

    async def run() -> None:
        try:
            results.append(await runner.run(2))
        finally:
            done.set()

    monkeypatch.setattr(FunctionTool, "fingerprint", fingerprint)
    monkeypatch.setattr(runner, "_watch", watch)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(run)
        try:
            with anyio.fail_after(5):
                await entered.wait()
            if mode == "stop":
                runner.deps.stops.stop_now = True
            else:
                runner.deps.clock._before = runner.deps.config.limits.run_timeout_s
            with anyio.fail_after(8):
                await watch_done.wait()
            with anyio.move_on_after(1):
                await done.wait()
            assert done.is_set(), "round still waiting for fingerprint after stop/deadline"
            candidate = results[0].candidates[0]
            assert rt_env.events("tool_end")[-1].payload["fingerprint"] is None
            assert len(rt_env.events("settle")) == 1
            assert candidate.output.data == {"x": 1, "y": 1}
            committed = read_committed(rt_env.run_store)
            assert committed is not None
            assert RunState.from_events(committed[1]).candidates[candidate.id] == candidate
        finally:
            release.set()


@pytest.mark.parametrize("wrapped", [False, True], ids=["direct", "extra-action"])
@pytest.mark.parametrize(
    ("case", "limit"),
    [
        ("budget", "budget"),
        ("usage_overflow", "usage_overflow"),
        ("tool_limit", "max_tool_runs_per_round"),
        ("repeat_limit", "max_refused_repeats"),
        ("closed_admission", "stop"),
        ("queued_start", "stop"),
    ],
)
async def test_core_tool_limits_survive_extra_actions_and_round_classification(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, case: str, limit: str, wrapped: bool
) -> None:
    from swarm_arena._agent.context import AgentContext
    from swarm_arena._agent.manager import Manager
    from swarm_arena._agent.watchdog import Watchdog
    from swarm_arena._budget.amount import USD_LIMIT
    from swarm_arena._model.base import FrozenModel, JsonValue
    from swarm_arena._model.planning_types import RoundReport
    from swarm_arena.tool import Action, Estimate

    class Args(FrozenModel):
        pass

    rt = rt_env.rt
    if case == "budget":
        monkeypatch.setattr(rt.tool, "estimate", lambda request: Estimate(usd=1000))
    elif case == "usage_overflow":
        res = await rt.ledger.reserve(rt.holder, Amount(), kind="tool", goal_id="g")
        await rt.ledger.settle_overflow(res.id, Amount(usd=USD_LIMIT))
    elif case == "tool_limit":
        rt.counters.tool_runs = rt.limits.max_tool_runs_per_round
    elif case == "repeat_limit":
        rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", deterministic=False))
        rt = dataclasses.replace(
            rt, watchdog=Watchdog(rt.limits.model_copy(update={"max_refused_repeats": 1}))
        )
        await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
        with pytest.raises(SoftError):
            await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    elif case == "closed_admission":
        rt_env.inflight.close()
    else:

        class QueuedStart(_OutputExecutor):
            async def submit(self, job: Job) -> ToolOutput:
                rt_env.inflight.close()
                return await super().submit(job)

        rt = dataclasses.replace(rt, executor=QueuedStart(rt_env, ToolOutput(status=RunStatus.OK)))
    ctx = AgentContext(rt)

    async def handler(context: AgentContext, args: Args) -> JsonValue:
        await context.run_tool({"x": 1})
        return None

    ctx.actions = (Action(name="run", description="Run", input_model=Args, handler=handler),)
    call = ctx.call_action("run", {}) if wrapped else ctx.run_tool({"x": 1})
    with pytest.raises(LimitReached) as caught:
        await call
    assert caught.value.limit == limit
    if case == "closed_admission":
        assert not rt_env.events("reserve")
        assert not rt_env.events("tool_start")
    if case == "queued_start":
        assert len(rt_env.events("reserve")) == len(rt_env.events("settle")) == 1
        assert not rt_env.events("tool_start")
        reservation = rt_env.events("reserve")[-1].payload["reservation"]["id"]
        assert rt_env.charged(reservation) == Amount()

    class RaisesControl(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            raise caught.value

    runner = _fingerprint_round(rt_env)
    runner.agents[0].manager = RaisesControl()
    runner.converge = True
    result = await runner.run(2)
    assert result.reasons["t/a"] == ("run budget" if case == "usage_overflow" else "limit reached")
    if case == "usage_overflow":
        assert runner.deps.stops.run_budget


async def test_stage_soft_error_keeps_safe_diagnostic_through_extra_action(rt_env: RtEnv) -> None:
    from swarm_arena._agent.context import AgentContext
    from swarm_arena._model.base import FrozenModel, JsonValue
    from swarm_arena.tool import Action

    class Args(FrozenModel):
        pass

    ctx = AgentContext(rt_env.with_script_tool(editable=["problem/*.phyk"]))

    async def handler(context: AgentContext, args: Args) -> JsonValue:
        await context.run_tool({}, edits={"problem/run.py": "pass"})
        return None

    ctx.actions = (Action(name="run", description="Run", input_model=Args, handler=handler),)
    with pytest.raises(SoftError, match=r"problem/run\.py") as caught:
        await ctx.call_action("run", {})
    assert "action failed" not in str(caught.value)
    assert not rt_env.events("reserve")


@pytest.mark.parametrize("mode", ["foreign", "queued-mutated", "queued-copy"])
async def test_only_current_admission_refusal_can_refund_and_propagate(
    rt_env: RtEnv, mode: str
) -> None:
    from uuid import uuid4

    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._agent.tool_run import Inflight

    secret = str(uuid4())
    registry = Inflight()
    registry.close()
    with anyio.CancelScope() as scope:
        await anyio.lowlevel.checkpoint()
        with pytest.raises(LimitReached) as prior:
            registry.add(scope)

    class ReplayedSignal(_OutputExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            if mode == "foreign":
                assert job.on_started is not None
                await job.on_started(Job.Started(kind="process", job_id=job.id))
                raise prior.value
            rt_env.inflight.close()
            try:
                return await super().submit(job)
            except LimitReached as core:
                error = core
                if mode == "queued-copy":
                    error = LimitReached("stop", secret)
                    error.__dict__.update(core.__dict__)
                error.args = (secret,)
                error.limit = secret
                error.add_note(secret)
                raise error from ValueError(secret)

    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", providers=["openai"]))
    rt = dataclasses.replace(rt, executor=ReplayedSignal(rt_env, ToolOutput(status=RunStatus.OK)))
    if mode == "queued-mutated":
        with pytest.raises(LimitReached) as caught:
            await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
        assert_secret_absent(caught.value, secret, sanitized=True)
        assert caught.value.limit == "stop"
        assert str(caught.value) == "the round is stopping"
        assert not rt_env.events("tool_end")
        assert rt_env.events("settle")[-1].payload["actual"] == Amount().model_dump()
    else:
        candidate = await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
        assert candidate.warnings == ["tool error: LimitReached"]
        assert candidate.output.status == RunStatus.ERROR
        reserved = rt_env.events("reserve")[-1].payload["reservation"]["amount"]
        assert rt_env.charged(candidate) == (
            Amount.model_validate(reserved) if mode == "foreign" else Amount()
        )
        assert secret not in rt_env.events("tool_end")[-1].model_dump_json()


async def test_round_rejects_transferred_core_signal(rt_env: RtEnv) -> None:
    from uuid import uuid4

    from swarm_arena._agent.context import AgentContext
    from swarm_arena._agent.manager import Manager
    from swarm_arena._model.planning_types import RoundReport

    secret = str(uuid4())

    class ForgingManager(Manager):
        async def run_round(self, ctx: AgentContext) -> RoundReport:
            try:
                ctx.get_candidate("c-absent")
            except SoftError as core:
                error = LimitReached("stop", secret)
                error.__dict__.update(core.__dict__)
                error.add_note(secret)
                raise error from ValueError(secret)
            raise AssertionError("missing candidate unexpectedly visible")

    runner = _fingerprint_round(rt_env)
    runner.agents[0].manager = ForgingManager()
    result = await runner.run(2)
    assert result.reasons["t/a"] == "manager error: LimitReached"
    await rt_env.rt.journal.commit()
    assert secret not in str(rt_env.events("agent_status")[-1].payload)


@pytest.mark.parametrize("fail_on", [1, 2], ids=["settings", "plan"])
@pytest.mark.parametrize("registered", [True, False], ids=["core", "connector"])
@pytest.mark.parametrize("kind", [SoftError, LimitReached])
async def test_staging_containers_are_detached_or_errors_rebuilt(
    rt_env: RtEnv,
    monkeypatch: pytest.MonkeyPatch,
    fail_on: int,
    registered: bool,
    kind: type[SoftError] | type[LimitReached],
) -> None:
    import traceback
    from collections.abc import Iterator
    from uuid import uuid4

    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._agent.context import AgentContext

    ctx = AgentContext(rt_env.rt)
    secret = str(uuid4())
    if registered:
        if kind is SoftError:
            with pytest.raises(SoftError) as missing:
                ctx.get_candidate("c-absent")
            original: SoftError | LimitReached = missing.value
        else:
            rt_env.rt.counters.tool_runs = rt_env.rt.limits.max_tool_runs_per_round
            with pytest.raises(LimitReached) as exhausted:
                await ctx.run_tool({"x": 1})
            original = exhausted.value
            rt_env.rt.counters.tool_runs = 0
        safe_args = original.args
        safe_limit = original.limit if isinstance(original, LimitReached) else None
    else:
        original = SoftError(secret) if kind is SoftError else LimitReached(secret, secret)
        safe_args = (f"invalid tool stage: {kind.__name__}",)
        safe_limit = None
    original.args = (secret,)
    original.add_note(secret)
    original.__cause__ = RuntimeError(secret)
    if isinstance(original, LimitReached):
        original.limit = secret
    iterations = 0

    class Globs(list[str]):
        def __iter__(self) -> Iterator[str]:
            nonlocal iterations
            iterations += 1
            if iterations == fail_on:
                raise original
            return super().__iter__()

    monkeypatch.setattr(rt_env.rt.tool, "editable", lambda: Globs(["**"]))
    if fail_on == 2:
        candidate = await ctx.run_tool({"x": 1}, edits={"p/a.txt": "x"})
        assert candidate.output.status == RunStatus.OK
        assert iterations == 1
        return
    with pytest.raises((SoftError, LimitReached)) as caught:
        await ctx.run_tool({"x": 1}, edits={"p/a.txt": "x"})
    error = caught.value
    assert iterations == fail_on
    assert_secret_absent(error, secret, sanitized=True)
    surfaces = {
        "str": str(error),
        "repr": repr(error),
        "args": repr(error.args),
        "notes": repr(getattr(error, "__notes__", ())),
        "cause": repr(error.__cause__),
        "traceback": "".join(traceback.format_exception(error)),
    }
    assert {name: secret in value for name, value in surfaces.items()} == dict.fromkeys(
        surfaces, False
    )
    assert error is not original
    assert type(error) is (kind if registered else SoftError)
    assert error.args == safe_args
    assert getattr(error, "limit", None) == safe_limit
    assert rt_env.rt.counters.tool_runs == 0
    assert not rt_env.events("reserve")


@pytest.mark.parametrize("fail_on", [1, 2], ids=["settings", "plan"])
async def test_staging_container_fencing_propagates(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, fail_on: int
) -> None:
    from collections.abc import Iterator

    from swarm_arena._agent.context import AgentContext

    ctx = AgentContext(rt_env.rt)
    fenced = _core_error(Fenced, "run ownership lost")
    iterations = 0

    class Globs(list[str]):
        def __iter__(self) -> Iterator[str]:
            nonlocal iterations
            iterations += 1
            if iterations == fail_on:
                raise fenced
            return super().__iter__()

    monkeypatch.setattr(rt_env.rt.tool, "editable", lambda: Globs(["**"]))
    if fail_on == 2:
        candidate = await ctx.run_tool({"x": 1}, edits={"p/a.txt": "x"})
        assert candidate.output.status == RunStatus.OK
        assert iterations == 1
        return
    with pytest.raises(Fenced) as caught:
        await ctx.run_tool({"x": 1}, edits={"p/a.txt": "x"})
    assert caught.value is not fenced
    assert caught.value.args == ("run ownership lost",)
    assert iterations == fail_on
    assert rt_env.rt.counters.tool_runs == 0
    assert not rt_env.events("reserve")


async def test_staging_refusal_keeps_safe_diagnostic_directly(rt_env: RtEnv) -> None:
    from uuid import uuid4

    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._agent.context import AgentContext

    secret = str(uuid4())
    rt = rt_env.with_script_tool(editable=["problem/*.phyk"])
    with pytest.raises(SoftError) as caught:
        await AgentContext(rt).run_tool({}, edits={"problem/run.py": secret})
    assert str(caught.value) == "problem/run.py: protected"
    assert_secret_absent(caught.value, secret, sanitized=True)
    assert rt.counters.tool_runs == 0
    assert not rt_env.events("reserve")


async def test_unmarked_stage_refusal_is_sanitized_and_releases_replicate(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    import traceback
    from uuid import uuid4

    from _fakes.secrecy import assert_secret_absent

    import swarm_arena._agent.tool_run as pipeline
    from swarm_arena._agent.context import AgentContext

    original = pipeline.plan_tool_stage
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", deterministic=False))
    ctx = AgentContext(rt)
    secret = str(uuid4())
    unmarked = SoftError(secret)
    unmarked.add_note(secret)
    unmarked.__cause__ = RuntimeError(secret)

    def refuse(**kwargs: Any) -> Any:
        raise unmarked

    monkeypatch.setattr(pipeline, "plan_tool_stage", refuse)
    with pytest.raises(SoftError) as caught:
        await ctx.run_tool({"x": 1})
    assert str(caught.value) == "invalid tool stage: SoftError"
    assert caught.value is not unmarked
    assert_secret_absent(caught.value, secret, sanitized=True)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert rt.counters.tool_runs == 0
    assert not rt_env.events("reserve")
    monkeypatch.setattr(pipeline, "plan_tool_stage", original)
    candidate = await ctx.run_tool({"x": 1})
    assert candidate.request.replicate == 0


@pytest.mark.parametrize("hook", ["effective_tool_ref", "find_entry", "tool_identity"])
async def test_runtime_config_and_identity_evaluation_use_daemon(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, hook: str
) -> None:
    import threading

    from swarm_arena._engine import round as round_module

    runner = _fingerprint_round(rt_env)
    observed: list[threading.Thread] = []
    original = getattr(round_module, hook)

    def wrapped(*args: Any) -> Any:
        observed.append(threading.current_thread())
        return original(*args)

    monkeypatch.setattr(round_module, hook, wrapped)
    result = await runner.run(2)
    assert len(result.candidates) == 1
    assert observed
    assert all(thread is not threading.main_thread() and thread.daemon for thread in observed)


@pytest.mark.parametrize("scored", [False, True])
async def test_past_run_hint_in_warnings(rt_env: RtEnv, tmp_path: "Path", scored: bool) -> None:
    from _fakes.journals import FakeRun, add_pointer

    from swarm_arena._agent.repeats import RepeatIndex, repeat_key
    from swarm_arena._board.arena import past_run_repeat_entries
    from swarm_arena.tool import ToolRequest

    rt = rt_env.rt
    past = FakeRun(str(tmp_path / "past"), "past", goal=rt.goal)
    key = repeat_key(
        goal_key=rt.goal_key,
        tool_ref=rt.tool_ref,
        request=ToolRequest(inputs={"x": 1.0}),
        deterministic=True,
    )
    past.commit(past.plan(ranking="rank"), past.candidate("past-c", repeat_key=key))
    if scored:
        past.commit(
            past.score("old", "past-c", 8, judge="rank"),
            past.score("new", "past-c", 0.25, judge="rank"),
        )
    add_pointer(rt.config.arena_root, past)
    view = rt.board.snapshot(rt.goal_key, rt.board.frontier())
    rt = dataclasses.replace(
        rt,
        view=view,
        rankings={"": "rank"},
        repeats=RepeatIndex(rt.repeats.run_id, past_run_repeat_entries(view, rt.repeats.run_id)),
    )
    result = await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert result.run_id == rt.repeats.run_id
    assert result.warnings == [f"past-run match past-c: {'rank=0.25' if scored else 'unscored'}"]
    assert rt.repeats.added == [(key, result)]


async def test_blocked_log_publication_is_abandoned_on_round_stop(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from swarm_arena._store.local import LocalStore

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = LocalStore.create
    writes: list[str] = []

    def create(store: LocalStore, key: str, data: bytes) -> str:
        if "/logs/" in key:
            writes.append(key)
            if key.endswith(".out"):
                workers.append(threading.current_thread())
                entered.set()
                assert release.wait(30)
        return original(store, key, data)

    monkeypatch.setattr(LocalStore, "create", create)
    task = asyncio.create_task(
        execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    )
    try:
        assert await anyio.to_thread.run_sync(entered.wait, 15)
        rt_env.cancel_inflight()
        done, _ = await asyncio.wait([task], timeout=3)
        assert done, "log publication ignored the round's cancellation"
        candidate = task.result()
        assert candidate.output.logs_ref is None
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for worker in workers:
            if worker.name.startswith("swarm-arena-daemon-"):
                await anyio.to_thread.run_sync(worker.join, 5)
                assert not worker.is_alive()
    assert len(writes) == 1, "abandoned log write published another file"


@pytest.mark.parametrize("failure", ["estimate", "providers", "put_edits"])
async def test_predispatch_error_retains_candidate_after_completion_cancel(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from swarm_arena._agent import tool_run
    from swarm_arena._errors import ConfigError

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ConfigError("pre-dispatch failure")

    monkeypatch.setattr(tool_run if failure == "put_edits" else rt_env.rt.tool, failure, fail)
    real = tool_run.run_sync_daemon
    observed: list[int] = []

    async def wrapped(fn: Any, *args: Any, **kwargs: Any) -> Any:
        if fn is tool_run._prepare_output:
            observed.append(len(rt_env.inflight._scopes))
            rt_env.inflight.cancel_all()
        return await real(fn, *args, **kwargs)

    monkeypatch.setattr(tool_run, "run_sync_daemon", wrapped)
    candidate = await execute_tool_run(rt_env.rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert candidate.output.status == RunStatus.ERROR
    assert observed == [1]  # only the bounded output scope remains registered
    assert not rt_env.events("tool_start")
    assert len(rt_env.events("tool_end")) == 1
    reserves = rt_env.events("reserve")
    assert len(reserves) == 1
    reservation = reserves[0].payload["reservation"]["id"]
    settles = rt_env.events("settle")
    assert len(settles) == 1
    assert settles[0].payload["id"] == reservation
    assert rt_env.reservation_state(reservation) == "settled"
    assert rt_env.charged(reservation) == Amount()
    assert not rt_env.inflight._scopes


async def test_reported_rows_recorded_as_tool_calls(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import threading

    from swarm_arena._engine.stops import RunClock, StopState
    from swarm_arena._journal.llm_record import _bound_run
    from swarm_arena._journal.replay import RunState
    from swarm_arena._tooling.llm_calls import LLMCallRow

    loop_thread = threading.get_ident()
    parsed: list[bool] = []
    validate = LLMCallRow.model_validate

    def validate_off_loop(cls: type[LLMCallRow], value: Any, **kwargs: Any) -> LLMCallRow:
        assert threading.get_ident() != loop_thread
        parsed.append(True)
        return validate(value, **kwargs)

    monkeypatch.setattr(LLMCallRow, "model_validate", classmethod(validate_off_loop))
    report = "\n".join(
        json.dumps(row)
        for row in [
            {"model": "a/b", "cost_usd": 0.01, "owner_epoch": "999", "candidate_id": "forged"},
            {"model": "a/c", "cost_usd": 0.02},
        ]
    )
    result = json.dumps({"data": {"ok": True}, "usage": {"usd": 0.04}})
    code = (
        "from pathlib import Path; import os; "
        f"Path('swarm/llm_calls.jsonl').write_text({report!r}); "
        f"Path(os.environ['SWARM_ARENA_RESULT']).write_text({result!r})"
    )
    rt = rt_env.with_tool(PyTool({"code": code}))
    async with _bound_run(
        rt.repeats.run_id,
        journal=rt.journal,
        ids=rt.ids,
        artifacts=rt.artifacts,
        config=rt.config,
        stops=StopState(),
        clock=RunClock(0),
        state=RunState(),
    ):
        candidate = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    calls = [event.payload["call"] for event in rt_env.events("llm_call")]
    assert candidate.output.data == {"ok": True}
    assert [call["role"] for call in calls] == ["tool", "tool"]
    assert [call["candidate_id"] for call in calls] == [candidate.id, candidate.id]
    assert candidate.llm_calls_source == "reported"
    assert len(candidate.llm_call_ids) == 2
    assert set(candidate.llm_call_ids) == {call["id"] for call in calls}
    assert rt_env.charged(candidate).usd == 0.04
    assert parsed == [True, True]


@pytest.mark.parametrize("kind", ["oversize", "hash_limit", "symlink", "malformed"])
async def test_bad_optional_report_warns_preserving_candidate(rt_env: RtEnv, kind: str) -> None:
    scripts = {
        "oversize": "p.open('wb').truncate(17*1024*1024)",
        "hash_limit": "p.open('wb').truncate(1024*1024*1024+1)",
        "symlink": "p.symlink_to('missing')",
        "malformed": "p.write_text('SECRET malformed input')",
    }
    code = (
        "from pathlib import Path; import os; p=Path('swarm/llm_calls.jsonl'); "
        + scripts[kind]
        + "; Path(os.environ['SWARM_ARENA_RESULT']).write_text('{\"data\":{\"ok\":true}}')"
    )
    rt = rt_env.with_tool(PyTool({"code": code}))
    candidate = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    assert candidate.output.status == RunStatus.OK
    assert candidate.output.data == {"ok": True}
    assert len(candidate.warnings) == 1
    assert "swarm/llm_calls.jsonl" in candidate.warnings[0]
    assert "SECRET" not in candidate.warnings[0]
    assert not rt_env.events("llm_call")
    if kind == "oversize":
        assert "over the 16 MiB limit; not collected" in candidate.warnings[0]


@pytest.mark.parametrize("provider", [True, False])
async def test_tool_with_providers_or_env_and_no_rows_is_unknown(
    rt_env: RtEnv, provider: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = FunctionTool(
        "_fakes.tools:square",
        providers=["openai"] if provider else [],
        env=[] if provider else ["OPENAI_API_KEY"],
    )
    rt = rt_env.with_tool(tool)
    received: list[Job] = []
    submit = rt.executor.submit

    async def capture(job: Job) -> ToolOutput:
        received.append(job)
        return await submit(job)

    monkeypatch.setattr(rt.executor, "submit", capture)
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel")
    candidate = await execute_tool_run(rt, {"x": 2}, parents=(), replicate=None, edits=None)
    assert candidate.llm_calls_source == "unknown"
    assert received[0].secret_env == (() if provider else ("OPENAI_API_KEY",))


@pytest.mark.parametrize("site", ["read", "upload"])
async def test_report_backend_failure_propagates(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._errors import StorageError
    from swarm_arena.tool import FileRef

    if site == "upload":
        rt = rt_env.with_tool(
            PyTool(
                {
                    "code": "from pathlib import Path; "
                    "Path('swarm/llm_calls.jsonl').write_bytes(b'REPORT')"
                }
            )
        )
        put = rt.artifacts.put_bytes

        def fail_upload(data: bytes) -> str:
            if data == b"REPORT":
                raise RuntimeError("SECRET")
            return put(data)

        # The LocalExecutor opens its own ArtifactStore for the same backend.
        from swarm_arena._store.artifacts import ArtifactStore

        original = ArtifactStore.put_bytes

        def upload(self: ArtifactStore, data: bytes) -> str:
            if data == b"REPORT":
                return fail_upload(data)
            return original(self, data)

        monkeypatch.setattr(ArtifactStore, "put_bytes", upload)
    else:
        from unittest.mock import AsyncMock, Mock

        ref = FileRef(name="llm_calls.jsonl", path="llm_calls.jsonl", sha256="0" * 64, size=1)
        executor = Mock(
            spec=Executor,
            submit=AsyncMock(return_value=ToolOutput(status=RunStatus.OK, llm_calls_file=ref)),
        )
        rt = dataclasses.replace(rt_env.rt, executor=executor)

        def fail_read(digest: str) -> bytes:
            raise RuntimeError("SECRET")

        from swarm_arena._store.base import Store

        store = Mock(spec=Store)
        store.read.side_effect = fail_read
        monkeypatch.setattr(rt.artifacts, "_store", store)
    with pytest.raises(StorageError) as caught:
        await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert_secret_absent(caught.value, "SECRET")
    assert not rt_env.events("tool_end")


async def test_interrupted_recording_preserves_candidate_links(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine.stops import RunClock, StopState
    from swarm_arena._journal.llm_record import _bound_run
    from swarm_arena._journal.replay import RunState
    from swarm_arena._tooling import llm_calls

    rt = rt_env.with_tool(
        PyTool(
            {
                "code": "from pathlib import Path; "
                'Path(\'swarm/llm_calls.jsonl\').write_text(\'{"model":"a/b"}\\n{"model":"a/c"}\')'
            }
        )
    )
    entered = asyncio.Event()
    real = llm_calls.record_call
    count = 0

    async def hold(**kwargs: Any) -> Any:
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
            await asyncio.Future()
        return await real(**kwargs)

    monkeypatch.setattr(llm_calls, "record_call", hold)
    async with _bound_run(
        rt.repeats.run_id,
        journal=rt.journal,
        ids=rt.ids,
        artifacts=rt.artifacts,
        config=rt.config,
        stops=StopState(),
        clock=RunClock(0),
        state=RunState(),
    ):
        task = asyncio.create_task(execute_tool_run(rt, {}, parents=(), replicate=None, edits=None))
        try:
            with anyio.fail_after(5):
                await entered.wait()
                rt_env.cancel_inflight()
                candidate = await task
            calls = [event.payload["call"] for event in rt_env.events("llm_call")]
            assert len(calls) == 1
            assert candidate.llm_call_ids == [calls[0]["id"]]
            assert candidate.llm_calls_source == "reported"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("claimed_size", [1, 3])
async def test_report_fetch_rejects_mismatched_size(rt_env: RtEnv, claimed_size: int) -> None:
    from swarm_arena._errors import StorageError
    from swarm_arena.tool import FileRef

    digest = rt_env.rt.artifacts.put_bytes(b"{}")
    ref = FileRef(name="llm_calls.jsonl", path="llm_calls.jsonl", sha256=digest, size=claimed_size)
    rt = dataclasses.replace(
        rt_env.rt,
        executor=_OutputExecutor(rt_env, ToolOutput(status=RunStatus.OK, llm_calls_file=ref)),
    )
    with pytest.raises(StorageError):
        await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)


async def test_report_fetch_refuses_large_local_blob_before_read(
    rt_env: RtEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._errors import StorageError
    from swarm_arena.tool import FileRef

    artifacts = rt_env.rt.artifacts
    ref = FileRef(name="llm_calls.jsonl", path="llm_calls.jsonl", sha256="0" * 64, size=10)
    from swarm_arena._store.local import LocalStore

    assert isinstance(artifacts._store, LocalStore)
    path = artifacts._store._root / ref.sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(17 * 1024 * 1024)
    reads: list[int] = []
    real = os.read

    def read(fd: int, size: int) -> bytes:
        reads.append(size)
        return real(fd, size)

    original_read_bytes = Path.read_bytes

    def read_bytes(file: Path) -> bytes:
        data = original_read_bytes(file)
        if file == path:
            reads.append(len(data))
        return data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(os, "read", read)
    rt = dataclasses.replace(
        rt_env.rt,
        executor=_OutputExecutor(rt_env, ToolOutput(status=RunStatus.OK, llm_calls_file=ref)),
    )
    with pytest.raises(StorageError):
        await execute_tool_run(rt, {"x": 1}, parents=(), replicate=None, edits=None)
    assert reads == []


class _DispatchSpy(_QueuedExecutor):
    def __init__(self, env: RtEnv, *, close_first: str = "") -> None:
        super().__init__(env.rt.executor)
        self.env = env
        self.close_first = close_first
        self.committed_at_submit: list[str] | None = None
        self.head_at_submit: Any = None

    async def submit(self, job: Job) -> ToolOutput:
        if self.close_first:
            self.env.inflight.close(budget=self.close_first == "budget")
        assert job.on_started is not None
        await job.on_started(
            Job.Started(kind="dispatch", job_id=job.id, task_name=job.id, ray_job_id="driver-1")
        )
        self.committed_at_submit = self.env.committed_types()
        owner = read_owner(self.env.run_store)
        assert owner is not None
        self.head_at_submit = owner[0].head
        return ToolOutput(status=RunStatus.OK, data={"y": 1.0})


@pytest.mark.parametrize("close_first", ["", "budget"])
async def test_dispatch_start_is_durable_before_submit(rt_env: RtEnv, close_first: str) -> None:
    fake = _DispatchSpy(rt_env, close_first=close_first)
    rt = dataclasses.replace(rt_env.rt, executor=fake)
    candidate = await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert fake.committed_at_submit is not None
    assert fake.committed_at_submit[-1] == "tool_dispatch"
    intent = rt_env.events("tool_dispatch")[-1]
    assert fake.head_at_submit.seq == intent.seq
    assert intent.task_id == candidate.id
    assert intent.payload["started"]["task_name"] == candidate.id
    assert intent.payload["started"]["ray_job_id"] == "driver-1"
    assert rt.started_runs == 1


async def test_closed_admission_refuses_dispatch_before_intent(rt_env: RtEnv) -> None:
    fake = _DispatchSpy(rt_env, close_first="hard")
    rt = dataclasses.replace(rt_env.rt, executor=fake)
    with pytest.raises(LimitReached) as err:
        await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert err.value.limit == "stop"
    assert fake.committed_at_submit is None
    reservation = rt_env.events("reserve")[-1].payload["reservation"]["id"]
    assert "tool_dispatch" not in rt_env.committed_types()
    assert rt_env.charged(reservation) == Amount()
    assert rt.started_runs == 0
