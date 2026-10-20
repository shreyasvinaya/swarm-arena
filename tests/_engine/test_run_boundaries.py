"""Regression coverage for approval, convergence estimates and bounded shutdown."""

import asyncio
import contextlib
import traceback
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.engine import take_over_retrying, with_limits
from _fakes.run import WRITE, WRITE_THEN_SLEEP, all_events, events, plan, wait_tool_started
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import PyTool, RecordingExecutor

from swarm_arena import Plan, Ref, Resources, RoundSpec, RunConfig
from swarm_arena._budget.amount import Amount
from swarm_arena._engine import run as run_module
from swarm_arena._engine.stops import StopState
from swarm_arena._engine.swarm import Swarm
from swarm_arena._errors import Fenced
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.lease import read_owner
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.tooling import CompletedRun, Estimate, ToolOutput, ToolRequest, Usage

SECRET = "c672c04b-19af-4b0b-9a06-9d047ecf811f"


def _secret_error() -> ValueError:
    error = ValueError(SECRET)
    error.add_note(SECRET)
    error.__cause__ = RuntimeError(SECRET)
    return error


class BrokenConstructor(RecordingExecutor):
    def __init__(self, config: Any = None) -> None:
        raise _secret_error()


class CapacityPayload:
    @property
    def cpus(self) -> float:
        raise _secret_error()


class MalformedCapacity(RecordingExecutor):
    def __init__(self, config: Any = None) -> None:
        super().__init__()

    async def capacity(self) -> Any:
        return CapacityPayload()


class UnsafeTruth:
    def __bool__(self) -> bool:
        raise _secret_error()


class RoundCostTool(PyTool):
    def estimate(self, request: ToolRequest) -> Estimate:
        return Estimate(usd=1.0 if request.round == 2 else 0.01)

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        output = super().collect(request, workspace, run)
        return output.model_copy(update={"usage": Usage(usd=self.estimate(request).usd)})


def _round_cost_plan(tool: PyTool | None = None) -> Plan:
    p = plan(rounds=3, converge_team=True, tool=tool)
    assert p.converge.team is not None
    agent = p.converge.team.agents[0].model_copy(
        update={
            "tool": Ref(
                path=f"{__name__}:RoundCostTool",
                config={"code": WRITE.format("{'data': {'y': 1.0}}")},
            )
        }
    )
    return p.model_copy(
        update={
            "converge": p.converge.model_copy(
                update={"team": p.converge.team.model_copy(update={"agents": [agent]})}
            )
        }
    )


async def test_early_converge_slice_refused_before_publication(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_usd=20)
    with pytest.raises(RuntimeError, match="reserve cannot hold the final slices"):
        await Swarm(cfg).run(_round_cost_plan())
    assert not await anyio.to_thread.run_sync(Path(cfg.run_root).exists)
    assert not await anyio.to_thread.run_sync(Path(cfg.artifact_root).exists)


async def test_early_stop_reserves_integer_bound_for_actual_convergence(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = with_limits(run_config, max_usd=100)
    release = tmp_path / "release"
    observed = anyio.Event()

    class SignalingStops(StopState):
        def __setattr__(self, name: str, value: object) -> None:
            super().__setattr__(name, value)
            if name == "stop" and value:
                observed.set()

    monkeypatch.setattr(run_module, "StopState", SignalingStops)
    monkeypatch.setattr(
        run_module, "watch_stop_files", partial(run_module.watch_stop_files, interval_s=0.01)
    )
    held = PyTool(
        {
            "code": f"import pathlib, time\nwhile not pathlib.Path({str(release)!r}).exists():"
            " time.sleep(0.01)\n" + WRITE.format("{'data': {'y': 1.0}}"),
            "grace_s": 0.01,
        }
    )
    swarm = Swarm(cfg)
    with anyio.fail_after(20):
        async with anyio.create_task_group() as tg:
            tg.start_soon(swarm.run, _round_cost_plan(held))
            try:
                await wait_tool_started(cfg)
                await swarm.stop()
                await observed.wait()
            finally:
                release.write_text("go")
    history = all_events(cfg)
    start = next(e for e in history if e.type == "run_started")
    assert Amount.model_validate(start.payload["final_slices"]["g"])._micro_usd == 6_000_000
    assert [
        e.round for e in history if e.type == "round_started" and e.payload.get("converge")
    ] == [2]
    reservations = [
        e.payload["reservation"]["amount"]
        for e in history
        if e.type == "reserve" and e.payload["reservation"]["holder"] == "g/:final"
    ]
    assert [Amount.model_validate(a)._micro_usd for a in reservations] == [1_000_000]
    assert [
        e.payload["candidate"]["output"]["usage"]["usd"]
        for e in history
        if e.type == "tool_end" and e.team == "conv"
    ] == [1.0]


@pytest.mark.parametrize("explicit_round", [False, True])
def test_estimate_covers_every_reachable_converge_request(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, explicit_round: bool
) -> None:
    p = _round_cost_plan()
    spec = RoundSpec(tool_overrides={"timeout_s": 7.0})
    if explicit_round:
        p = p.model_copy(update={"converge": p.converge.model_copy(update={"round": spec})})
    else:
        p = p.model_copy(update={"rounds": [*p.rounds[:-1], spec]})
    seen: list[tuple[int, dict[str, Any], dict[str, Any], float | None]] = []

    def estimate(self: RoundCostTool, request: ToolRequest) -> Estimate:
        seen.append((request.round, request.round_overrides, request.inputs, self.config.timeout_s))
        return Estimate(
            tokens=10 if request.round == 1 else 0, usd=1 if request.round == 2 else 0.01
        )

    monkeypatch.setattr(RoundCostTool, "estimate", estimate)
    _, slices = run_module._estimates([p], run_config)
    assert {r for r, _, _, _ in seen} == {1, 2, 3, 4}
    assert all(
        overrides == {"timeout_s": 7.0} and inputs == {} and timeout == 7.0
        for _, overrides, inputs, timeout in seen
    )
    assert slices["g"].tokens == 60
    assert slices["g"]._micro_usd == 6_000_000


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("constructor", "executor construction failed: ValueError"),
        ("capacity", "executor capacity failed: CapacityPayload"),
        ("capacity_warning", "executor capacity failed: ValueError"),
        ("approval", "approval failed: ValueError"),
        ("approval_truth", "approval failed: UnsafeTruth"),
        ("approval_integer", "approval failed: int"),
    ],
)
async def test_startup_errors_are_type_only_and_write_nothing(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, case: str, expected: str
) -> None:
    cfg = run_config
    if case in {"constructor", "capacity"}:
        cls = BrokenConstructor if case == "constructor" else MalformedCapacity
        cfg = cfg.model_copy(update={"executor": Ref(path=f"{__name__}:{cls.__name__}")})
    if case == "capacity_warning":

        def warning(plans: Sequence[Plan], capacity: Resources | None) -> list[str]:
            raise _secret_error()

        monkeypatch.setattr(run_module, "_capacity_warning", warning)

    async def approve(estimate: Estimate, warnings: Sequence[str]) -> Any:
        if case == "approval":
            raise _secret_error()
        if case == "approval_truth":
            return UnsafeTruth()
        return 1

    with pytest.raises((ValueError, RuntimeError)) as caught:
        await Swarm(cfg).run(
            plan(rounds=1), approve=approve if case.startswith("approval") else None
        )
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert isinstance(caught.value, RuntimeError)
    assert str(caught.value) == expected
    assert not await anyio.to_thread.run_sync(Path(cfg.run_root).exists)
    assert not await anyio.to_thread.run_sync(Path(cfg.artifact_root).exists)


@pytest.mark.parametrize("boundary", ["constructor", "capacity", "approval"])
@pytest.mark.parametrize("signal", ["fenced", "grouped_fenced", "cancel"])
async def test_startup_preserves_core_signals(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, boundary: str, signal: str
) -> None:
    error = (
        Fenced()
        if signal == "fenced"
        else ExceptionGroup("ownership", [Fenced()])
        if signal == "grouped_fenced"
        else asyncio.CancelledError()
    )

    class SignalingExecutor(RecordingExecutor):
        async def capacity(self) -> Resources | None:
            if boundary == "capacity":
                raise error
            return None

    def build(config: RunConfig) -> RecordingExecutor:
        if boundary == "constructor":
            raise error
        return SignalingExecutor()

    async def approve(estimate: Estimate, warnings: Sequence[str]) -> bool:
        raise error

    monkeypatch.setattr(run_module, "_build_executor", build)
    expected = asyncio.CancelledError if signal == "cancel" else Fenced
    with pytest.raises(expected) as caught:
        await Swarm(run_config).run(
            plan(rounds=1), approve=approve if boundary == "approval" else None
        )
    assert type(caught.value) is expected
    assert not await anyio.to_thread.run_sync(Path(run_config.run_root).exists)


@pytest.mark.parametrize("shutdown", ["raises", "hangs"])
async def test_shutdown_preserves_approval_refusal(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, shutdown: str
) -> None:
    class BadClose(RecordingExecutor):
        async def aclose(self) -> None:
            if shutdown == "raises":
                raise _secret_error()
            await anyio.sleep_forever()

    async def refuse(estimate: Estimate, warnings: Sequence[str]) -> bool:
        return False

    cfg = run_config.model_copy(update={"settle_timeout_s": 0.05})
    monkeypatch.setattr(run_module, "_build_executor", lambda cfg: BadClose())
    task = asyncio.create_task(Swarm(cfg).run(plan(rounds=1), approve=refuse))
    try:
        done, _ = await asyncio.wait({task}, timeout=5)
        assert done, "executor shutdown exceeded settlement bound"
        with pytest.raises(RuntimeError, match=r"^run not approved$"):
            task.result()
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.parametrize("takeover", [False, True])
async def test_hung_shutdown_returns_original_result_or_real_fence(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, takeover: bool
) -> None:
    cfg = run_config.model_copy(update={"settle_timeout_s": 0.05})
    entered = anyio.Event()
    writers: list[JournalWriter] = []
    scopes: list[anyio.CancelScope] = []
    original_init = JournalWriter.__init__

    class HungClose(LocalExecutor):
        async def aclose(self) -> None:
            entered.set()
            await anyio.sleep_forever()

    def capture(self: JournalWriter, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        writers.append(self)

    executor = HungClose(workspace_root=str(tmp_path / "workspaces"))
    monkeypatch.setattr(run_module, "_build_executor", lambda cfg: executor)
    monkeypatch.setattr(JournalWriter, "__init__", capture)
    p = plan(
        rounds=1, tool=PyTool({"code": WRITE_THEN_SLEEP, "grace_s": 0.01}) if takeover else None
    )

    async def run() -> dict[str, list[Any]]:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            return await Swarm(cfg).run(p)
        raise AssertionError("fencing was replaced with cancellation")

    task = asyncio.create_task(run())
    try:
        with anyio.fail_after(20):
            if takeover:
                store = await wait_tool_started(cfg)
                take_over_retrying(store)
                with pytest.raises(Fenced):
                    await writers[0].append_durable(
                        EventDraft(type=EventType.STOP_REQUESTED, payload={"now": True})
                    )
            await entered.wait()
        if takeover:
            scopes[0].cancel()
        started = anyio.current_time()
        done, _ = await asyncio.wait({task}, timeout=0.5)
        assert done, "hung aclose prevented return after ten settlement intervals"
        assert anyio.current_time() - started < 0.5
        if takeover:
            with pytest.raises(Fenced) as caught:
                task.result()
            assert type(caught.value) is Fenced
            owner = read_owner(store)
            assert owner is not None
            assert owner[0].epoch == 2
            assert not events(cfg, "tool_end")
        else:
            result = task.result()
            assert len(result["g"]) == 1
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Fenced):
                await task
        await LocalExecutor.aclose(executor)


async def test_shutdown_timeout_keeps_result_when_close_unwinding_raises(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RaisingOnTimeout(RecordingExecutor):
        async def aclose(self) -> None:
            try:
                await anyio.sleep_forever()
            finally:
                raise _secret_error()

    cfg = run_config.model_copy(update={"settle_timeout_s": 0.05})
    monkeypatch.setattr(run_module, "_build_executor", lambda cfg: RaisingOnTimeout())
    with anyio.fail_after(10):
        result = await Swarm(cfg).run(plan(rounds=1))
    assert result == {"g": []}
