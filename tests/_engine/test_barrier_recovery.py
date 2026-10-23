"""Reuse paid barrier outcomes after a real exit at their committed score head."""

import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic import BaseModel, ConfigDict

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    FunctionTool,
    Goal,
    Limits,
    Plan,
    Ref,
    RoundSpec,
    RunConfig,
    Score,
    ScriptedManager,
    Swarm,
    TeamSpec,
)
from swarm_arena._engine.control import read_status
from swarm_arena._journal.event import Event
from swarm_arena._journal.reader import read_chain, read_committed
from swarm_arena._store.local import LocalStore
from swarm_arena.board import Candidate
from swarm_arena.scoring import Judge, JudgeContext
from swarm_arena.tool import CompletedRun, Estimate, ToolOutput, ToolRequest, Usage

_MODULE = "_engine.test_barrier_recovery"


def paid_value(n: int, marker: str) -> dict[str, int]:
    with Path(marker).open("a") as stream:
        stream.write(f"{n}\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"n": n}


class MeteredTool(FunctionTool):
    def estimate(self, request: ToolRequest) -> Estimate:
        return Estimate(tokens=7, usd=0.000007)

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        result = super().collect(request, workspace, run)
        n = int(request.inputs["n"])
        return result.model_copy(update={"usage": Usage(input_tokens=n, usd=n / 1_000_000)})


class MeteredJudge(Judge):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
        label: str = "primary"
        anchors: int = 0
        fail_first: bool = False

    def providers(self) -> frozenset[str]:
        return frozenset({"offline-paid"})

    def anchor_count(self) -> int:
        assert isinstance(self.config, self.Config)
        return self.config.anchors

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        assert isinstance(self.config, self.Config)
        call = {
            "judge": self.config.label,
            "round": ctx.round,
            "final": ctx.final,
            "ids": [c.id for c in candidates],
            "anchors": sorted(ctx.anchors),
        }
        await anyio.to_thread.run_sync(_append_call, Path(goal.inputs["judge_log"]), call)
        if self.config.fail_first and ctx.round == 1:
            raise ValueError("offline paid judge failure")
        return [Score(candidate_id=c.id, key=(float(c.output.data["n"]),)) for c in candidates]


def _append_call(path: Path, call: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(call) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _config(root: Path) -> RunConfig:
    return RunConfig(
        run_root=str(root / "runs"),
        artifact_root=str(root / "artifacts"),
        arena_root=str(root / "arena"),
        reserve_fraction=0.5,
        executor=Ref(
            path="swarm_arena.executor:LocalExecutor",
            config={
                "workspace_root": str(root / "workspaces"),
                "max_concurrent": 2,
            },
        ),
        limits=Limits(
            max_rounds=2,
            max_tool_runs_per_round=2,
            max_usd=0.000500,
            max_tokens=500,
            default_run_reservation={"usd": 0.000007, "tokens": 7},
        ),
    )


def _plan(root: Path, judges: int, anchors: int, fail_first: bool) -> Plan:
    return Plan(
        goal=Goal(id="g", namespace="paid", inputs={"judge_log": str(root / "judge.log")}),
        teams=[
            TeamSpec(
                name="t",
                rationale="Solve",
                agents=[
                    AgentSpec(
                        name="a",
                        rationale="Solve",
                        tool=Ref(
                            path=f"{_MODULE}:MeteredTool",
                            config={"function": f"{_MODULE}:paid_value"},
                        ),
                        manager=Ref.of(
                            ScriptedManager(
                                [
                                    {
                                        "round": r,
                                        "inputs": {
                                            "n": r * 2 + offset,
                                            "marker": str(root / "tools.log"),
                                        },
                                    }
                                    for r in (1, 2)
                                    for offset in (1, 2)
                                ]
                            )
                        ),
                    )
                ],
            )
        ],
        judges=[
            Ref(
                path=f"{_MODULE}:MeteredJudge",
                name=name,
                config={"label": name, "anchors": anchors, "fail_first": fail_first},
            )
            for name in ("primary", "secondary")[:judges]
        ],
        ranking="primary",
        rounds=[RoundSpec(), RoundSpec()],
        converge=ConvergeSpec(top_n=2),
    )


async def crash_driver(
    root: Path, judges: int, anchors: int, fail_first: bool, cutoff: int
) -> None:
    await anyio.to_thread.run_sync(lambda: root.mkdir(parents=True, exist_ok=True))
    replace = os.replace

    def exit_after_cas(
        src: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        dst: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        if str(dst).endswith("/control/owner.json"):
            head = json.loads(Path(os.fsdecode(dst)).read_text())["head"]
            if head is not None and head["seq"] >= cutoff:
                os._exit(91)

    patch = pytest.MonkeyPatch()
    patch.setattr(os, "replace", exit_after_cas)
    try:
        await Swarm(_config(root)).run(_plan(root, judges, anchors, fail_first), run_id="paid")
    finally:
        patch.undo()


def _history(root: Path) -> list[Event]:
    committed = read_committed(LocalStore(root / "runs" / "paid"))
    assert committed is not None
    return committed[1]


def _calls(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "judge.log").read_text().splitlines()]


def _charges(events: list[Event]) -> tuple[int, int]:
    charges: dict[str, tuple[int, int]] = {}
    for event in events:
        if event.type == "reserve":
            reservation = event.payload["reservation"]
            amount = reservation["amount"]
            charges[reservation["id"]] = (
                amount["tokens"],
                int(Decimal(str(amount["usd"])) * 1_000_000),
            )
        elif event.type == "settle":
            amount = event.payload["actual"]
            charges[event.payload["id"]] = (
                amount["tokens"],
                int(Decimal(str(amount["usd"])) * 1_000_000),
            )
    return sum(x[0] for x in charges.values()), sum(x[1] for x in charges.values())


@pytest.mark.parametrize(
    ("judges", "anchors", "fail_first"), [(1, 0, False), (2, 1, False), (1, 0, True)]
)
async def test_resume_reuses_committed_paid_barrier_batches(
    tmp_path: Path, judges: int, anchors: int, fail_first: bool
) -> None:
    control, crashed = tmp_path / "control", tmp_path / "crashed"
    control.mkdir()
    expected = await Swarm(_config(control)).run(
        _plan(control, judges, anchors, fail_first), run_id="paid"
    )
    committed = read_committed(LocalStore(control / "runs" / "paid"))
    assert committed is not None
    assert committed[0].head is not None
    target_type = "judge_error" if fail_first else "score"
    cutoff = next(
        batch[-1].seq
        for _, batch in read_chain(LocalStore(control / "runs" / "paid"), committed[0].head)
        if any(e.type == target_type and e.round == 1 for e in batch)
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
    )
    code = (
        f"import anyio,sys; from pathlib import Path; from {_MODULE} import crash_driver; "
        "anyio.run(crash_driver, Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), "
        "sys.argv[4] == 'True', int(sys.argv[5]))"
    )
    # A whole child swarm run up to its crash point: a generous safety bound
    # (CI-SWEEP), not the behaviour under test.
    with anyio.fail_after(180):
        process = await anyio.run_process(
            [
                sys.executable,
                "-c",
                code,
                str(crashed),
                str(judges),
                str(anchors),
                str(fail_first),
                str(cutoff),
            ],
            env=environment,
            check=False,
        )
    assert process.returncode == 91, process.stderr.decode()
    before = _history(crashed)
    assert sum(e.type == target_type for e in before) == 2
    assert not [e for e in before if e.type in {"decision", "round_closed"}]
    recovered = await Swarm(_config(crashed)).resume("paid", force=True)
    expected_calls, actual_calls = _calls(control), _calls(crashed)
    assert [(c["judge"], c["round"], c["final"]) for c in actual_calls] == [
        (c["judge"], c["round"], c["final"]) for c in expected_calls
    ]
    counts = Counter((c["judge"], c["round"], c["final"], tuple(c["ids"])) for c in actual_calls)
    assert set(counts.values()) == {1}
    assert any(c["final"] for c in actual_calls)
    if anchors and not fail_first:
        assert any(c["round"] == 2 and c["anchors"] for c in actual_calls)
    history = _history(crashed)
    reservations = [
        e.payload["reservation"]
        for e in history
        if e.type == "reserve" and e.payload["reservation"]["kind"] in {"judge", "final"}
    ]
    assert len(reservations) == len(expected_calls)
    oracle = 18 + sum(7 * len(c["ids"]) for c in expected_calls)
    assert _charges(history) == _charges(_history(control)) == (oracle, oracle)
    status = await read_status(_config(crashed).run_root, "paid")
    assert (status.spent.tokens, status.spent.micro_usd) == (oracle, oracle)
    assert (crashed / "tools.log").read_text().splitlines() == ["3", "4", "5", "6"]
    assert [c.output.data for c in recovered["g"]] == [c.output.data for c in expected["g"]]
