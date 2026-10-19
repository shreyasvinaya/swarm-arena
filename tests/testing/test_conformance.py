import traceback
from pathlib import Path
from typing import cast

import pytest
from _fakes.tools import (
    GOAL,
    WRITE,
    BadFeedbackCheck,
    BrokenJudge,
    BrokenTool,
    EmptyFingerprintTool,
    NotACheckResult,
    PyTool,
    RecordingExecutor,
    StrayJudge,
    add,
    nan_score,
    rosen_score,
)
from pydantic import ValidationError

from swarm_arena import FunctionTool, Goal, ProgramJudge, Ref, ScriptCheck, ScriptTool
from swarm_arena._errors import SoftError
from swarm_arena._model.ids import run_tag
from swarm_arena.executor import Job, LocalExecutor
from swarm_arena.scoring import Check
from swarm_arena.testing import check_check, check_judge, check_tool, make_candidate
from swarm_arena.testing._conformance import _artifact_root
from swarm_arena.tool import FileRef, FileSet, RunStatus, ToolOutput, ToolRequest

SECRET = "SECRET_SENTINEL_KIT"
RUN_PY = """\
import json, os
data = {name: open(f"problem/{name}.txt").read() for name in ("a", "b")}
json.dump({"data": data}, open(os.environ["SWARM_ARENA_RESULT"], "w"))
"""


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "problem").mkdir()
    (tmp_path / "problem/run.py").write_text(RUN_PY)
    (tmp_path / "problem/a.txt").write_text("1")
    (tmp_path / "problem/b.txt").write_text("1")
    return tmp_path


class FixedExecutor(RecordingExecutor):
    """Record each job and return a fixed object, which may break the Executor contract."""

    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output

    async def submit(self, job: Job) -> ToolOutput:
        self.jobs.append(job)
        return cast(ToolOutput, self.output)


async def test_function_tool(local_executor: LocalExecutor) -> None:
    out = await check_tool(FunctionTool(add), {"a": 2}, executor=local_executor)
    assert (out.status, out.data) == (RunStatus.OK, {"sum": 3})
    ref = Ref(path="swarm_arena:FunctionTool", config={"function": "_fakes.tools:add"})
    assert (await check_tool(ref, {"a": 1}, executor=local_executor)).data == {"sum": 2}
    with pytest.raises(ValidationError):
        await check_tool(FunctionTool(add), {"a": "x"}, executor=local_executor)


async def test_script_tool_edit_and_same_identity_overlay(
    ws: Path, local_executor: LocalExecutor
) -> None:
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*.txt"])
    goal = Goal(id="g", namespace="ns", files=FileSet(globs=["problem/*"]))
    edited = ToolRequest(edits={"problem/a.txt": "2"})
    parent = make_candidate(ToolOutput(status=RunStatus.OK), request=edited, tool=tool)
    out = await check_tool(
        tool, {}, goal=goal, parents=[parent], edits={"problem/b.txt": "3"}, executor=local_executor
    )
    assert out.data == {"a": "2", "b": "3"}


async def test_parent_of_tool_with_own_files_is_overlaid(
    ws: Path, local_executor: LocalExecutor
) -> None:
    (ws / "tool").mkdir()
    (ws / "tool/helper.txt").write_text("help")
    tool = ScriptTool(
        command=["python", "problem/run.py"], files=["tool/helper.txt"], editable=["problem/*.txt"]
    )
    goal = Goal(id="g", namespace="ns", files=FileSet(globs=["problem/*"]))
    edited = ToolRequest(edits={"problem/a.txt": "2"})
    parent = make_candidate(ToolOutput(status=RunStatus.OK), request=edited, tool=tool)
    out = await check_tool(tool, {}, goal=goal, parents=[parent], executor=local_executor)
    assert out.data == {"a": "2", "b": "1"}


async def test_other_tool_parent_warns(ws: Path, local_executor: LocalExecutor) -> None:
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*.txt"])
    other = ScriptTool(command=["python", "problem/run.py", "--other"], editable=["problem/*.txt"])
    edited = ToolRequest(edits={"problem/a.txt": "2"})
    parent = make_candidate(ToolOutput(status=RunStatus.OK), request=edited, tool=other)
    with pytest.warns(UserWarning, match="not overlaid: made by another tool"):
        out = await check_tool(
            tool,
            {},
            goal=Goal(id="g", namespace="ns", files=FileSet(globs=["problem/*"])),
            parents=[parent],
            executor=local_executor,
        )
    assert out.data["a"] == "1"


async def test_protected_edit_refused(ws: Path, local_executor: LocalExecutor) -> None:
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*"])
    with pytest.raises(SoftError, match=r"problem/run\.py: protected"):
        await check_tool(
            tool,
            {},
            goal=Goal(id="g", namespace="ns", files=FileSet(globs=["problem/*"])),
            edits={"problem/run.py": "x"},
            executor=local_executor,
        )


def surfaces(exc: BaseException) -> list[str]:
    found = [
        str(exc),
        repr(exc),
        repr(exc.args),
        repr(getattr(exc, "__notes__", [])),
        repr(exc.__cause__),
        "".join(traceback.format_exception(exc)),
    ]
    if isinstance(exc, ValidationError):
        found += [repr(exc.errors()), exc.json()]
    return found


async def test_input_and_edit_errors_hide_values(ws: Path, local_executor: LocalExecutor) -> None:
    with pytest.raises(ValidationError) as invalid:
        await check_tool(FunctionTool(add), {"a": SECRET}, executor=local_executor)
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*"])
    goal = Goal(id="g", namespace="ns", files=FileSet(globs=["problem/*"]))
    edits = {"problem/run.py": SECRET}
    with pytest.raises(SoftError, match="protected") as refused:
        await check_tool(tool, {}, goal=goal, edits=edits, executor=local_executor)
    texts = [*surfaces(invalid.value), *surfaces(refused.value)]
    assert all(SECRET not in text for text in texts)


async def test_conformance_violation(local_executor: LocalExecutor) -> None:
    with pytest.raises(AssertionError, match="fingerprint"):
        await check_tool(
            EmptyFingerprintTool(PyTool.Config(code=WRITE.format("{'data': {}}"))),
            {},
            executor=local_executor,
        )


@pytest.mark.parametrize(
    "method",
    [
        "get_input_model",
        "get_description",
        "editable",
        "providers",
        "fingerprint",
        "is_deterministic",
        "estimate",
    ],
)
async def test_each_tool_contract_is_asserted(method: str, local_executor: LocalExecutor) -> None:
    tool = BrokenTool(BrokenTool.Config(code=WRITE.format("{'data': {}}"), broken=method))
    with pytest.raises(AssertionError, match=rf"Tool\.{method}\(\)"):
        await check_tool(tool, {}, executor=local_executor)


async def test_job_is_staged_like_the_core() -> None:
    executor = FixedExecutor(ToolOutput(status=RunStatus.OK, data={"x": 1}))
    tool = PyTool(PyTool.Config(code="pass", timeout_s=120.0, grace_s=3.0))
    out = await check_tool(tool, {"n": 4}, executor=executor, timeout_s=30.0)
    [job] = executor.jobs
    assert out.data == {"x": 1}
    assert (job.kind, job.tool, job.timeout_s, job.grace_s) == ("tool", Ref.of(tool), 30.0, 3.0)
    assert job.id.startswith(f"c-{run_tag('local')}-e0-")
    assert job.artifact_root == _artifact_root()
    assert job.request is not None
    assert job.request.inputs == {"n": 4}
    assert job.stage.request_json is not None
    assert job.stage.request_json["params"] == {"n": 4}
    assert job.env["SWARM_ARENA_CANDIDATE_ID"] == job.id
    assert (job.env["SWARM_ARENA_RUN_ID"], job.env["SWARM_ARENA_GOAL_ID"]) == ("local", "check")


@pytest.mark.parametrize(
    ("output", "match"),
    [
        (None, r"Executor\.submit\(\) must return a ToolOutput"),
        (
            ToolOutput(
                status=RunStatus.OK,
                files=[FileRef(name="ghost.txt", path="out/ghost.txt", sha256="0" * 64, size=0)],
            ),
            "out/ghost.txt is not in the artifact store",
        ),
    ],
)
async def test_executor_contract_is_asserted(output: object, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        await check_tool(PyTool(PyTool.Config(code="pass")), {}, executor=FixedExecutor(output))


async def test_check_check(ws: Path, local_executor: LocalExecutor) -> None:
    (ws / "checks").mkdir()
    (ws / "checks/ok.py").write_text(WRITE.format("{'passed': True}"))
    check = ScriptCheck(command=["python", "checks/ok.py"], files=["checks/ok.py"])
    result = await check_check(
        check, make_candidate(ToolOutput(status=RunStatus.OK)), executor=local_executor
    )
    assert (result.passed, result.check) == (True, "script-check")


async def test_default_executor_is_closed_and_ref_name_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[LocalExecutor] = []

    async def aclose(self: LocalExecutor) -> None:
        closed.append(self)

    monkeypatch.setattr(LocalExecutor, "aclose", aclose)
    ref = Ref(path="_fakes.tools:AlwaysPass", name="gate")
    result = await check_check(ref, make_candidate(ToolOutput(status=RunStatus.OK)))
    assert (result.check, result.passed, len(closed)) == ("gate", True, 1)


@pytest.mark.parametrize(
    ("check", "match"),
    [
        (BadFeedbackCheck(), r"Check\.feedback must be"),
        (NotACheckResult(), r"Check\.check\(\) must return a CheckResult"),
    ],
)
async def test_check_contract_is_asserted(check: Check, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        await check_check(check, make_candidate(ToolOutput(status=RunStatus.OK)))


async def test_check_judge_stamps_and_warns(local_executor: LocalExecutor) -> None:
    c1 = make_candidate(ToolOutput(status=RunStatus.OK, data={"f": 2.0}))
    c2 = make_candidate(ToolOutput(status=RunStatus.OK, data={"f": 1.0}))
    scores = await check_judge(ProgramJudge(rosen_score), GOAL, [c1, c2], executor=local_executor)
    assert [s.candidate_id for s in scores] == [c1.id, c2.id]
    assert {s.judge for s in scores} == {"program"}
    assert all(s.id.startswith(f"s-{run_tag('local')}-e0-") and s.rubric_hash for s in scores)
    assert len({s.batch_id for s in scores}) == 1
    assert scores[0].batch_id.startswith("b-")
    with pytest.warns(UserWarning, match=f"judge error for {c1.id}"):
        assert await check_judge(ProgramJudge(nan_score), GOAL, [c1], executor=local_executor) == []


async def test_check_judge_rejects_unknown_candidate(local_executor: LocalExecutor) -> None:
    candidate = make_candidate(ToolOutput(status=RunStatus.OK))
    with pytest.raises(AssertionError, match="unknown candidate"):
        await check_judge(StrayJudge(), GOAL, [candidate], executor=local_executor)


@pytest.mark.parametrize(
    ("mode", "match"),
    [("duplicate", "more than one score for candidate"), ("not_score", "list of Score")],
)
async def test_judge_contract_is_asserted(mode: str, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        await check_judge(
            BrokenJudge({"mode": mode}), GOAL, [make_candidate(ToolOutput(status=RunStatus.OK))]
        )


async def test_non_finite_score_is_a_judge_error() -> None:
    c = make_candidate(ToolOutput(status=RunStatus.OK))
    with pytest.warns(UserWarning, match=f"judge error for {c.id}: score key must be"):
        assert await check_judge(BrokenJudge({"mode": "nan"}), GOAL, [c]) == []
