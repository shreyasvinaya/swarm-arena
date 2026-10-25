"""Budget-stopped runs finish recovery without replaying paid work."""

import json
import os
import sys
from pathlib import Path

import anyio
import pytest
from _engine.test_barrier_recovery import MeteredTool, _calls, _config, _history, _plan

from swarm_arena import ConvergeSpec, Plan, Ref, RunConfig, ScriptedManager, Swarm
from swarm_arena._budget.amount import USD_LIMIT
from swarm_arena._engine.stops import StopState
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena.tool import CompletedRun, ToolOutput, ToolRequest, Usage

_MODULE = "_engine.test_budget_resume"


class BudgetTool(MeteredTool):
    """Exhaust the ordinary pool with authoritative tool usage."""

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        output = super().collect(request, workspace, run)
        usd = USD_LIMIT + 1 if request.inputs.get("overflow") else 0.000250
        return output.model_copy(update={"usage": Usage(usd=usd)})


def _budget_plan(root: Path, overflow: bool) -> Plan:
    proposed = _plan(root, 1, 0, False)
    agent = proposed.teams[0].agents[0]
    converge = proposed.teams[0].model_copy(
        update={
            "name": "converge",
            "agents": [
                agent.model_copy(
                    update={
                        "manager": Ref.of(
                            ScriptedManager(
                                [{"inputs": {"n": 9, "marker": str(root / "tools.log")}}]
                            )
                        )
                    }
                )
            ],
        }
    )
    ordinary = agent.model_copy(
        update={
            "tool": Ref(path=f"{_MODULE}:BudgetTool", config={"function": f"{_MODULE}:paid_value"}),
            "manager": Ref.of(
                ScriptedManager(
                    [
                        {
                            "round": r,
                            "inputs": {
                                "n": r * 3,
                                "marker": str(root / "tools.log"),
                                "overflow": overflow,
                            },
                        }
                        for r in (1, 2)
                    ]
                )
            ),
        }
    )
    return proposed.model_copy(
        update={
            "teams": [proposed.teams[0].model_copy(update={"agents": [ordinary]})],
            "converge": ConvergeSpec(mode="team", team=converge),
        }
    )


def paid_value(n: int, marker: str, overflow: bool = False) -> dict[str, int]:
    from _engine.test_barrier_recovery import paid_value as record_value

    return record_value(n, marker)


def _budget_config(root: Path) -> RunConfig:
    config = _config(root)
    return config.model_copy(
        update={
            "lease_ttl_s": 3,
            "limits": config.limits.model_copy(update={"max_usd": 0.000500, "max_tokens": None}),
        }
    )


def _environment() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
        ),
    }


async def _crash_at_converge(root: Path) -> None:
    publish = JournalWriter._publish

    async def commit(journal: JournalWriter) -> None:
        hit = any(event.type == "converge_started" for event in journal._queue)
        await publish(journal)
        if hit:
            os._exit(91)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(JournalWriter, "_publish", commit)
        await Swarm(_budget_config(root)).run(_budget_plan(root, False), run_id="paid")


async def _resume_cli(root: Path) -> None:
    with anyio.fail_after(60):
        child = await anyio.run_process(
            [
                str(Path(sys.executable).with_name("swarm-arena")),
                "resume",
                str(root / "runs/paid"),
                "--force",
            ],
            env=_environment(),
            check=False,
        )
    output = child.stdout.decode() + child.stderr.decode()
    assert "resume initialization stopped" not in output, output
    assert child.returncode == 0, output


@pytest.mark.parametrize("overflow", [False, True], ids=["budget", "usage-overflow"])
async def test_resume_completed_budget_stop_does_not_repeat_paid_work(
    tmp_path: Path, overflow: bool
) -> None:
    await Swarm(_budget_config(tmp_path)).run(_budget_plan(tmp_path, overflow), run_id="paid")
    before = _history(tmp_path)
    state = RunState.from_events(before)
    assert state.goals["g"].finished
    assert any(e.type == "round_closed" for e in before)
    assert any(e.type == "converge_started" and e.payload["reason"] == "run budget" for e in before)
    tools = (tmp_path / "tools.log").read_text()
    judge_log = tmp_path / "judge.log"
    judges = judge_log.read_bytes() if judge_log.exists() else b""
    await _resume_cli(tmp_path)
    assert (tmp_path / "tools.log").read_text() == tools
    assert (judge_log.read_bytes() if judge_log.exists() else b"") == judges
    after = _history(tmp_path)
    assert not [
        e
        for e in after
        if e.seq > before[-1].seq and e.type in {"reserve", "tool_start", "llm_call"}
    ]
    assert RunState.from_events(after).goals["g"].finished


async def test_resume_budget_stop_crashed_at_converge_finishes_from_reserve(tmp_path: Path) -> None:
    code = (
        f"import anyio,sys; from pathlib import Path; from {_MODULE} import _crash_at_converge; "
        "anyio.run(_crash_at_converge, Path(sys.argv[1]))"
    )
    with anyio.fail_after(60):
        child = await anyio.run_process(
            [sys.executable, "-c", code, str(tmp_path)], env=_environment(), check=False
        )
    assert child.returncode == 91, child.stderr.decode()
    before = _history(tmp_path)
    assert before[-1].type == "converge_started"
    assert before[-1].payload["reason"] == "run budget"
    assert not RunState.from_events(before).goals["g"].finished
    assert (tmp_path / "tools.log").read_text().splitlines() == ["3"]
    judges = _calls(tmp_path)
    assert judges
    assert not any(call["final"] for call in judges)
    await _resume_cli(tmp_path)
    after = _history(tmp_path)
    added = [e for e in after if e.seq > before[-1].seq]
    assert (tmp_path / "tools.log").read_text().splitlines() == ["3", "9"]
    recovered_judges = _calls(tmp_path)
    assert recovered_judges[: len(judges)] == judges
    assert [(call["round"], call["final"]) for call in recovered_judges[len(judges) :]] == [
        (2, False),
        (0, True),
    ]
    assert [e.team for e in added if e.type == "tool_start"] == ["converge"]
    assert any(e.type == "reserve" and e.payload["reservation"]["kind"] == "final" for e in added)
    assert RunState.from_events(after).goals["g"].finished
    solution = json.loads((tmp_path / "runs/paid/goals/g/solution.json").read_text())
    assert solution["status"] == "ok"


async def test_stop_after_replayed_budget_still_cancels_resume_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _budget_config(tmp_path)
    await Swarm(config).run(_budget_plan(tmp_path, False), run_id="paid")
    before = _history(tmp_path)
    tools = (tmp_path / "tools.log").read_bytes()
    judges = _calls(tmp_path)
    budget_replayed = anyio.Event()
    set_flag = StopState.__setattr__

    def stop_after_budget(stops: StopState, name: str, value: object) -> None:
        was_budget = getattr(stops, "run_budget", False)
        set_flag(stops, name, value)
        if name == "run_budget" and value is True and not was_budget:
            assert stops.on_change is not None
            stops.stop = True
            budget_replayed.set()

    monkeypatch.setattr(StopState, "__setattr__", stop_after_budget)
    failures: list[RuntimeError] = []

    async def resume() -> None:
        try:
            await Swarm(config).resume("paid", force=True)
        except RuntimeError as error:
            failures.append(error)

    with anyio.fail_after(60):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(resume)
            await budget_replayed.wait()

    assert [str(error) for error in failures] == ["resume initialization stopped"]
    added = [event for event in _history(tmp_path) if event.seq > before[-1].seq]
    assert [event.type for event in added] == ["takeover"]
    assert (tmp_path / "tools.log").read_bytes() == tools
    assert _calls(tmp_path) == judges
