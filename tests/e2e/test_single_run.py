from pathlib import Path

import pytest
from _fakes.tools import gradient_descent, rosen_score, small_b

from swarm_arena import FunctionTool, Goal, ProgramJudge, ScriptCheck, ScriptJudge, ScriptTool
from swarm_arena.executor import LocalExecutor
from swarm_arena.scoring import Lexicographic
from swarm_arena.testing import check_check, check_judge, check_tool, make_candidate
from swarm_arena.tool import FileSet, RunStatus, ToolRequest

DATA = Path(__file__).parent / "data" / "single_run"
CHECK = {"command": ["python", "checks/check.py"], "files": ["checks/check.py"]}


@pytest.fixture
def executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalExecutor:
    monkeypatch.chdir(DATA)
    return LocalExecutor(workspace_root=str(tmp_path / "ws"))


async def test_function_tool_run_checked_and_judged(executor: LocalExecutor) -> None:
    goal = Goal(id="rosenbrock", namespace="e2e")
    tool = FunctionTool(gradient_descent)
    out = await check_tool(tool, {"lr": 0.001, "steps": 2000}, goal=goal, executor=executor)
    assert out.status is RunStatus.OK
    assert set(out.data) == {"x", "y", "f"}
    assert out.data["f"] < 12.5
    cand = make_candidate(
        out,
        request=ToolRequest(inputs={"lr": 0.001, "steps": 2000}),
        tool=tool,
        goal_id="rosenbrock",
    )
    result = await check_check(ScriptCheck(**CHECK), cand, goal=goal, executor=executor)
    assert (result.passed, result.check) == (True, "script-check")
    [score] = await check_judge(ProgramJudge(rosen_score), goal, [cand], executor=executor)
    assert (score.candidate_id, score.value, score.key, score.judge) == (
        cand.id,
        out.data["f"],
        (-out.data["f"],),
        "program",
    )


async def test_script_tool_edit_overlay_check_and_lexicographic_judge(
    executor: LocalExecutor,
) -> None:
    goal = Goal(id="fit", namespace="e2e", files=FileSet(globs=["problem/*"]))
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*.txt"])
    first = await check_tool(
        tool, {}, goal=goal, edits={"problem/model_a.txt": "2.0"}, executor=executor
    )
    assert (first.status, first.data["a"], first.data["b"]) == (RunStatus.OK, 2.0, 0.0)
    parent = make_candidate(
        first,
        request=ToolRequest(edits={"problem/model_a.txt": "2.0"}),
        tool=tool,
        goal_id="fit",
    )
    out = await check_tool(
        tool,
        {},
        goal=goal,
        parents=[parent],
        edits={"problem/model_b.txt": "1.0"},
        executor=executor,
    )
    # The base overlay supplies a, and this run's edit supplies b.
    assert (out.status, out.data["a"], out.data["b"]) == (RunStatus.OK, 2.0, 1.0)
    # Guard against judge data leaking into tool workspaces.
    assert "judge/hidden/points.csv" not in out.data["files_seen"]
    assert "judge/score.py" not in out.data["files_seen"]
    assert "problem/run.py" in out.data["files_seen"]
    child = make_candidate(out, tool=tool, goal_id="fit", parent_ids=[parent.id])
    baseline = make_candidate(first, tool=tool, goal_id="fit")
    for c in (child, baseline):
        assert (await check_check(ScriptCheck(**CHECK), c, goal=goal, executor=executor)).passed
    judge = Lexicographic(
        [
            ScriptJudge(
                command=["python", "judge/score.py"],
                files=["judge/score.py", "judge/hidden/points.csv"],
                direction="min",
            ),
            ProgramJudge(small_b),
        ]
    )
    scores = {
        s.candidate_id: s
        for s in await check_judge(judge, goal, [child, baseline], executor=executor)
    }
    # MSE 0 versus 1, then |b|; larger keys rank higher.
    assert scores[child.id].key == (-0.0, -1.0)
    assert scores[baseline.id].key == (-1.0, -0.0)
    assert max(scores.values(), key=lambda s: s.key).candidate_id == child.id
    assert {s.judge for s in scores.values()} == {"lexicographic"}
