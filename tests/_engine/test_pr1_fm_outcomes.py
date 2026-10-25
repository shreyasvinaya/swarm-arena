"""Public API and CLI regressions for nonstandard extension outcomes."""

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pytest
from _fakes import pr1_fm as hooks

from swarm_arena import (
    AgentSpec,
    Goal,
    Limits,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    Swarm,
    TeamSpec,
)
from swarm_arena._errors import Fenced
from swarm_arena.executor import LocalExecutor


def config(root: Path) -> RunConfig:
    return RunConfig(
        run_root=str(root / "runs"),
        artifact_root=str(root / "artifacts"),
        arena_root=str(root / "arena"),
        executor=Ref.of(LocalExecutor(workspace_root=str(root / "ws"))),
        limits=Limits(max_tokens=1000, max_usd=0.001, run_timeout_s=120, round_timeout_s=60),
    )


def plan(where: str, kind: str) -> Plan:
    tool = Ref(
        path="_fakes.pr1_fm:PaidTool",
        config={
            "command": [
                sys.executable,
                "-c",
                "import json,os; json.dump({'data': {'n': 4}}, "
                "open(os.environ['SWARM_ARENA_RESULT'], 'w'))",
            ],
            "grace_s": 0,
        },
    )
    return Plan(
        goal=Goal(id="g", namespace="fm", inputs={"where": where, "kind": kind}),
        teams=[
            TeamSpec(
                name="t",
                rationale="probe",
                agents=[
                    AgentSpec(
                        name=name,
                        rationale="probe",
                        tool=tool,
                        manager=Ref(path=f"_fakes.pr1_fm:{manager}"),
                    )
                    for name, manager in [("bad", "Broken"), ("peer", "Peer")]
                ],
            )
        ],
        judges=[Ref.of(ProgramJudge("_fakes.pr1_fm:score"))],
        rounds=[RoundSpec()],
        checks=[Ref(path="_fakes.pr1_fm:BrokenCheck")] if where == "check" else [],
    )


def events(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in sorted((root / "runs/r/journal").glob("*.jsonl"))
        for line in path.read_text().splitlines()[1:]
    ]


@pytest.mark.parametrize("where", ["manager", "action", "check", "command", "collect", "estimate"])
@pytest.mark.parametrize("kind", ["group", "cancel"])
async def test_extension_failure_is_local(tmp_path: Path, where: str, kind: str) -> None:
    hooks.PEER_READY = None
    result = None
    failure_text = ""
    try:
        result = await Swarm(config(tmp_path)).run(plan(where, kind), run_id="r")
    except BaseException as error:
        failure_text = "".join(traceback.format_exception(error))
    journal = events(tmp_path)
    assert hooks.MARK not in failure_text + repr(result) + json.dumps(journal)
    assert result is not None
    assert result["g"]
    assert any(e["type"] == "tool_end" and e["agent"] == "peer" for e in journal)
    assert any(e["type"] == "goal_finished" for e in journal)
    if where == "check":
        bad = [
            e["payload"]["candidate"]
            for e in journal
            if e["type"] == "tool_end" and e["agent"] == "bad"
        ]
        assert len(bad) == 1
        assert bad[0]["checks"][0]["passed"] is False
    if where == "action":
        assert any("action contained" in json.dumps(e) for e in journal)


@pytest.mark.parametrize("kind", ["group", "cancel"])
def test_cli_manager_failure_is_local(tmp_path: Path, kind: str) -> None:
    (tmp_path / "run.toml").write_text(config(tmp_path).dumps())
    (tmp_path / "plan.toml").write_text(plan("manager", kind).dumps())
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))
    process = subprocess.run(
        [
            str(Path(sys.executable).with_name("swarm-arena")),
            "run",
            "--yes",
            str(tmp_path / "run.toml"),
            str(tmp_path / "plan.toml"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    output = process.stdout + process.stderr
    assert hooks.MARK not in output
    assert process.returncode == 0, output
    assert "Traceback" not in output
    journals = list((tmp_path / "runs").glob("*/journal/*.jsonl"))
    text = "".join(path.read_text() for path in journals)
    assert hooks.MARK not in text
    assert '"goal_finished"' in text
    journal = [json.loads(line) for path in journals for line in path.read_text().splitlines()[1:]]
    assert any(e["type"] == "tool_end" and e["agent"] == "peer" for e in journal)


async def test_caller_cancellation_rejects_hook_replacement(tmp_path: Path) -> None:
    await test_extension_failure_is_local(tmp_path, "action", "replace_cancel")


@pytest.mark.parametrize("where", ["manager", "action", "check", "command", "collect", "estimate"])
async def test_nested_fencing_reaches_public_caller(tmp_path: Path, where: str) -> None:
    hooks.PEER_READY = None
    with pytest.raises(Fenced) as caught:
        await Swarm(config(tmp_path)).run(plan(where, "fenced_group"), run_id="r")
    assert hooks.MARK not in "".join(traceback.format_exception(caught.value))
    assert hooks.MARK not in repr(caught.value.args)
    assert caught.value.__cause__ is None
    assert hooks.MARK not in json.dumps(events(tmp_path))
