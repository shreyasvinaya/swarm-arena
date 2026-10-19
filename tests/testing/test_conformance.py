import math
import subprocess
import sys
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import (
    FAILING_COMMAND,
    GOAL,
    WRITE,
    AlwaysPass,
    BadCollectTool,
    BadFeedbackCheck,
    BrokenJudge,
    BrokenTool,
    CommandCheck,
    EchoTool,
    EmptyFingerprintTool,
    FixedJudge,
    GoalCheck,
    InvalidCheckResult,
    NotACheckResult,
    OkCollectTool,
    PanelJudge,
    PreflightFailsCheck,
    PyTool,
    RecordingExecutor,
    StrayJudge,
    add,
    manifest_size,
    nan_score,
    raising_score,
    rosen_score,
)
from pydantic import ValidationError

from swarm_arena import FunctionTool, Goal, Limits, ProgramJudge, Ref, ScriptCheck, ScriptTool
from swarm_arena._errors import ConfigError, SoftError
from swarm_arena._model.ids import run_tag
from swarm_arena.board import Candidate
from swarm_arena.executor import Job, LocalExecutor
from swarm_arena.scoring import Check, CheckContext, CheckResult, Judge, Lexicographic
from swarm_arena.testing import check_check, check_judge, check_tool, make_candidate
from swarm_arena.testing._conformance import _artifact_root
from swarm_arena.tool import FileRef, FileSet, RunStatus, ToolOutput, ToolRequest

SECRET = "SECRET_SENTINEL_KIT"
RUN_PY = """\
import json, os
data = {name: open(f"problem/{name}.txt").read() for name in ("a", "b")}
json.dump({"data": data}, open(os.environ["SWARM_ARENA_RESULT"], "w"))
"""
EMPTY_RESULT = WRITE.format("{'data': {}}")
PROBLEM = FileSet(globs=["problem/*"])
VALID_OUTPUT = r"Tool\.collect\(\) must return a valid ToolOutput \(data must be finite JSON\)"


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


def broken(mode: str, code: str = EMPTY_RESULT) -> BrokenTool:
    return BrokenTool(BrokenTool.Config(code=code, broken=mode))


def ok() -> ToolOutput:
    return ToolOutput(status=RunStatus.OK)


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
    goal = Goal(id="g", namespace="ns", files=PROBLEM)
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
    goal = Goal(id="g", namespace="ns", files=PROBLEM)
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
            goal=Goal(id="g", namespace="ns", files=PROBLEM),
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
            goal=Goal(id="g", namespace="ns", files=PROBLEM),
            edits={"problem/run.py": "x"},
            executor=local_executor,
        )


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        ({"problem/c.md": "x"}, r"problem/c\.md: not editable"),
        ({"problem/a.txt": "x" * (Limits().max_edit_bytes + 1)}, "> max_edit_bytes"),
    ],
)
async def test_edit_rules_are_the_core_rules(ws: Path, edit: dict[str, str], match: str) -> None:
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*.txt"])
    executor = RecordingExecutor()
    goal = Goal(id="g", namespace="ns", files=PROBLEM)
    with pytest.raises(SoftError, match=match):
        await check_tool(tool, {}, goal=goal, edits=edit, executor=executor)
    assert executor.jobs == []


async def test_respelled_tool_file_is_a_config_error() -> None:
    goal = Goal(id="g", namespace="ns", files=FileSet(manifest={"p/run.py": "1" * 64}))
    files = FileSet(manifest={"P/RUN.py": "2" * 64})  # snapshotted already: no file is read
    tool = ScriptTool(command=["python", "p/run.py"], files=files)
    executor = RecordingExecutor()
    with pytest.raises(ConfigError, match=r"tool files do not fit the goal files: .*P/RUN\.py"):
        await check_tool(tool, {}, goal=goal, executor=executor)
    # The human's config error comes before any refusal of the agent's edits.
    with pytest.raises(ConfigError, match="tool files do not fit the goal files"):
        await check_tool(tool, {}, goal=goal, edits={"p/x.txt": "x"}, executor=executor)
    assert executor.jobs == []


async def test_input_and_edit_errors_hide_values(ws: Path, local_executor: LocalExecutor) -> None:
    with pytest.raises(ValidationError) as invalid:
        await check_tool(FunctionTool(add), {"a": SECRET}, executor=local_executor)
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*"])
    goal = Goal(id="g", namespace="ns", files=PROBLEM)
    edits = {"problem/run.py": SECRET}
    with pytest.raises(SoftError, match="protected") as refused:
        await check_tool(tool, {}, goal=goal, edits=edits, executor=local_executor)
    assert_secret_absent(invalid.value, SECRET)
    assert_secret_absent(refused.value, SECRET)


async def test_connector_input_model_errors_hide_values() -> None:
    # PyInputs is a plain BaseModel without hide_input_in_errors, as a connector may write it.
    executor = RecordingExecutor()
    with pytest.raises(ValidationError) as invalid:
        await check_tool(PyTool(PyTool.Config(code="pass")), {"n": SECRET}, executor=executor)
    assert_secret_absent(invalid.value, SECRET)
    assert [(e["loc"], e["type"]) for e in invalid.value.errors()] == [(("n",), "int_parsing")]
    assert invalid.value.__cause__ is None
    assert invalid.value.__suppress_context__
    assert executor.jobs == []


async def test_conformance_violation(local_executor: LocalExecutor) -> None:
    with pytest.raises(AssertionError, match="fingerprint"):
        await check_tool(
            EmptyFingerprintTool(PyTool.Config(code=EMPTY_RESULT)), {}, executor=local_executor
        )


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("get_input_model", r"Tool\.get_input_model\(\) must return"),
        ("get_input_model_instance", r"Tool\.get_input_model\(\) must return"),
        ("get_description", r"Tool\.get_description\(\) must return"),
        ("editable", r"Tool\.editable\(\) must return"),
        ("editable_str", r"Tool\.editable\(\) must return"),
        ("editable_dict", r"Tool\.editable\(\) must return"),
        ("providers", r"Tool\.providers\(\) must return"),
        ("providers_items", r"Tool\.providers\(\) must return"),
        ("extra_actions", r"Tool\.extra_actions\(\) must return"),
        ("extra_actions_names", r"Tool\.extra_actions\(\) must return .* unique names"),
        ("fingerprint", r"Tool\.fingerprint\(\) must return the same"),
        ("fingerprint_type", r"Tool\.fingerprint\(\) must return a nonempty str"),
        ("fingerprint_run", r"Tool\.fingerprint\(\) must not depend on"),
        ("is_deterministic", r"Tool\.is_deterministic\(\) must return"),
        ("estimate", r"Tool\.estimate\(\) must return"),
        ("collect_nan", r"Tool\.collect\(\) must return a valid ToolOutput \(data must be finite"),
        ("collect_construct", r"Tool\.collect\(\) must return a valid ToolOutput"),
        ("collect_ok", r"Tool\.collect\(\) must not report ok for a failed run"),
    ],
)
async def test_each_tool_contract_is_asserted(
    mode: str, match: str, local_executor: LocalExecutor
) -> None:
    with pytest.raises(AssertionError, match=match):
        await check_tool(broken(mode), {}, executor=local_executor)


async def test_collect_probe_accepts_error_status_and_raising(
    local_executor: LocalExecutor,
) -> None:
    with pytest.raises(AssertionError, match="must not report ok for a failed run"):
        await check_tool(
            OkCollectTool(PyTool.Config(code=EMPTY_RESULT)), {}, executor=local_executor
        )
    with pytest.warns(UserWarning, match="RuntimeError: collect broke"):  # collect raises
        out = await check_tool(
            BadCollectTool(PyTool.Config(code=EMPTY_RESULT)), {}, executor=local_executor
        )
    assert out.status == RunStatus.ERROR
    out = await check_tool(PyTool(PyTool.Config(code=EMPTY_RESULT)), {}, executor=local_executor)
    assert out.status == RunStatus.OK
    # EchoTool reports run.status as it is: the probe's failed run must stay failed.
    out = await check_tool(EchoTool(EchoTool.Config(text="hi")), {}, executor=local_executor)
    assert (out.status, out.data) == (RunStatus.OK, {"text": "hi"})


async def test_failed_run_shows_the_end_of_its_stderr_log(local_executor: LocalExecutor) -> None:
    code = "import sys; sys.stderr.write('x' * 5000 + 'SOLVER-DIAG'); sys.exit(3)"
    with pytest.warns(UserWarning, match=r"ended with status error") as record:
        out = await check_tool(PyTool(PyTool.Config(code=code)), {}, executor=local_executor)
    assert out.status == RunStatus.ERROR
    [message] = [str(w.message) for w in record if "ended with status error" in str(w.message)]
    assert "SOLVER-DIAG" in message
    assert "x" * 2100 not in message
    assert "x" * 1000 in message


async def test_failed_check_command_shows_its_stderr_log(
    ws: Path, local_executor: LocalExecutor
) -> None:
    (ws / "checks").mkdir()
    (ws / "checks/crash.py").write_text("raise SystemExit('CHECK-DIAG')")
    check = ScriptCheck(command=["python", "checks/crash.py"], files=["checks/crash.py"])
    with pytest.warns(UserWarning, match=r"(?s)ended with status error.*CHECK-DIAG"):
        result = await check_check(check, make_candidate(ok()), executor=local_executor)
    assert not result.passed


async def test_failed_judge_command_shows_its_stderr_log(local_executor: LocalExecutor) -> None:
    c = make_candidate(ok())
    judge = BrokenJudge({"mode": "command_fails"})
    with pytest.warns(UserWarning, match=r"(?s)ended with status error.*COMMAND-DIAG"):
        [score] = await check_judge(judge, GOAL, [c], executor=local_executor)
    assert score.candidate_id == c.id


async def test_log_tail_that_is_not_utf8_is_decoded_safely(local_executor: LocalExecutor) -> None:
    # 2-byte characters then one invalid byte: the 2 KB tail starts in a character, too.
    code = "import sys; sys.stderr.buffer.write('é'.encode() * 1100 + b'\\xff'); sys.exit(1)"
    with pytest.warns(UserWarning, match="ended with status error") as record:
        await check_tool(PyTool(PyTool.Config(code=code)), {}, executor=local_executor)
    [message] = [str(w.message) for w in record]
    assert message.endswith(chr(0xFFFD))
    assert ":\n" + chr(0xFFFD) + "é" * 1023 + chr(0xFFFD) in message


@pytest.mark.parametrize("status", [s for s in RunStatus if s != RunStatus.ERROR])
async def test_only_status_error_gives_the_failed_run_warning(status: RunStatus) -> None:
    executor = FixedExecutor(ToolOutput(status=status))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = await check_tool(PyTool(PyTool.Config(code="pass")), {}, executor=executor)
    assert out.status == status


def notes_of(error: BaseException) -> str:
    return "".join(traceback.format_exception(error))


async def test_contract_failure_keeps_the_failed_run_tail(local_executor: LocalExecutor) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the tail goes on the AssertionError, not a warning
        with pytest.raises(AssertionError, match=r"Tool\.estimate\(\)") as tool:
            await check_tool(broken("estimate", FAILING_COMMAND), {}, executor=local_executor)
        check = CommandCheck({"code": FAILING_COMMAND, "valid": False})
        with pytest.raises(AssertionError, match="valid CheckResult") as checked:
            await check_check(check, make_candidate(ok()), executor=local_executor)
        judge = BrokenJudge({"mode": "command_then_invalid"})
        with pytest.raises(AssertionError, match="valid Scores") as judged:
            await check_judge(judge, GOAL, [make_candidate(ok())], executor=local_executor)
    for caught in (tool, checked, judged):
        text = notes_of(caught.value)
        assert "ended with status error" in text
        assert "COMMAND-DIAG" in text


async def test_note_check_matches_the_exact_note_line(local_executor: LocalExecutor) -> None:
    # The tool prints the executor's note text inside its own line: that is not the note.
    code = (
        "import sys; sys.stderr.write('echo: swarm-arena: collect failed: Tool.collect() "
        "did not return a ToolOutput\\n'); sys.exit(1)"
    )
    with pytest.warns(UserWarning, match="ended with status error"):
        out = await check_tool(PyTool(PyTool.Config(code=code)), {}, executor=local_executor)
    assert out.status == RunStatus.ERROR


async def test_output_files_in_the_store_pass(ws: Path, local_executor: LocalExecutor) -> None:
    (ws / "problem/out.py").write_text(
        "import pathlib\n"
        "pathlib.Path('out').mkdir()\n"
        "pathlib.Path('out/x.txt').write_text('made')\n"
        + WRITE.format("{'data': {}, 'files': ['out/x.txt']}")
    )
    tool = ScriptTool(command=["python", "problem/out.py"])
    goal = Goal(id="g", namespace="ns", files=FileSet(globs=["problem/out.py"]))
    out = await check_tool(tool, {}, goal=goal, executor=local_executor)
    assert out.status == RunStatus.OK
    assert [(f.path, f.size) for f in out.files] == [("out/x.txt", 4)]


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
    await check_tool(tool, {}, goal=Goal(id="other", namespace="ns"), executor=executor)
    assert executor.jobs[1].env["SWARM_ARENA_GOAL_ID"] == "other"


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
        (  # the kit's own re-validation: an executor other than LocalExecutor
            ToolOutput(status=RunStatus.OK).model_copy(update={"data": {"x": math.nan}}),
            VALID_OUTPUT,
        ),
        (ToolOutput.model_construct(data={}), VALID_OUTPUT),  # no status at all
        (ToolOutput.model_construct(status=RunStatus.ERROR, logs_ref=5), VALID_OUTPUT),
    ],
)
async def test_executor_contract_is_asserted(output: object, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        await check_tool(PyTool(PyTool.Config(code="pass")), {}, executor=FixedExecutor(output))


async def test_custom_executor_with_microsecond_times_passes() -> None:
    # The core re-validates a ToolOutput in lax mode; timestamp precision is not a contract.
    when = datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=UTC)
    out = ToolOutput(status=RunStatus.OK, data={"x": 1})
    out = out.model_copy(update={"started_at": when, "ended_at": when})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checked = await check_tool(
            PyTool(PyTool.Config(code="pass")), {}, executor=FixedExecutor(out)
        )
    assert (checked.data, checked.started_at, checked.ended_at) == ({"x": 1}, when, when)


async def test_preflight_runs_before_any_job() -> None:
    executor = RecordingExecutor()
    with pytest.raises(RuntimeError, match="preflight refused the tool"):
        await check_tool(broken("preflight"), {}, executor=executor)
    with pytest.raises(RuntimeError, match="preflight refused the check"):
        await check_check(PreflightFailsCheck(), make_candidate(ok()), executor=executor)
    with pytest.raises(RuntimeError, match="preflight refused the judge"):
        await check_judge(BrokenJudge({"mode": "preflight"}), GOAL, [], executor=executor)
    assert executor.jobs == []


async def test_check_check(ws: Path, local_executor: LocalExecutor) -> None:
    (ws / "checks").mkdir()
    (ws / "checks/ok.py").write_text(WRITE.format("{'passed': True}"))
    check = ScriptCheck(command=["python", "checks/ok.py"], files=["checks/ok.py"])
    result = await check_check(check, make_candidate(ok()), executor=local_executor)
    assert (result.passed, result.check) == (True, "script-check")


async def test_check_gets_the_default_or_snapshotted_goal(ws: Path) -> None:
    default = await check_check(GoalCheck(), make_candidate(ok()))
    assert default.details == {"goal": "testing/check", "files": []}
    goal = Goal(id="g", namespace="ns", files=PROBLEM)
    given = await check_check(GoalCheck(), make_candidate(ok()), goal=goal)
    files = ["problem/a.txt", "problem/b.txt", "problem/run.py"]
    assert given.details == {"goal": "ns/g", "files": files}


async def test_default_executor_is_closed_and_ref_name_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[LocalExecutor] = []

    async def aclose(self: LocalExecutor) -> None:
        closed.append(self)

    monkeypatch.setattr(LocalExecutor, "aclose", aclose)
    ref = Ref(path="_fakes.tools:AlwaysPass", name="gate")
    result = await check_check(ref, make_candidate(ok()))
    assert (result.check, result.passed, len(closed)) == ("gate", True, 1)


@pytest.mark.parametrize(
    ("check", "match"),
    [
        (BadFeedbackCheck(), r"Check\.feedback must be"),
        (NotACheckResult(), r"Check\.check\(\) must return a CheckResult"),
        (InvalidCheckResult(), r"Check\.check\(\) must return a valid CheckResult"),
    ],
)
async def test_check_contract_is_asserted(check: Check, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        await check_check(check, make_candidate(ok()))


async def test_check_result_is_validated_in_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    async def one(self: Check, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        # Lax mode makes 1 into True, and 1 == True, so only strict mode refuses it.
        return CheckResult.model_construct(check="always-pass", passed=cast(Any, 1))

    monkeypatch.setattr(AlwaysPass, "check", one)
    with pytest.raises(AssertionError, match=r"Check\.check\(\) must return a valid CheckResult"):
        await check_check(AlwaysPass(), make_candidate(ok()))


@pytest.mark.parametrize("check", [AlwaysPass(), GoalCheck()])
async def test_good_checks_pass_strict_validation(check: Check) -> None:
    result = await check_check(check, make_candidate(ok()))
    assert result.passed is True


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


async def test_judge_name_comes_from_the_ref_and_goal_files_are_snapshotted(ws: Path) -> None:
    candidate = make_candidate(ok())
    ref = Ref(path="_fakes.tools:FixedJudge", config={"value": 1.0}, name="x")
    [score] = await check_judge(ref, GOAL, [candidate])
    assert score.judge == "x"
    goal = Goal(id="g", namespace="ns", files=PROBLEM)
    [score] = await check_judge(ProgramJudge(manifest_size), goal, [candidate])
    assert score.key == (3.0,)


async def test_check_judge_rejects_unknown_candidate(local_executor: LocalExecutor) -> None:
    candidate = make_candidate(ok())
    with pytest.raises(AssertionError, match="unknown candidate"):
        await check_judge(StrayJudge(), GOAL, [candidate], executor=local_executor)


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("duplicate", "more than one score for candidate"),
        ("not_score", "list of Score"),
        ("invalid_score", r"Judge\.judge\(\) must return valid Scores"),
        ("repairable_score", r"Judge\.judge\(\) must return valid Scores"),
        ("rubric_unstable", r"Judge\.rubric_hash\(\) must return the same nonempty str"),
        ("rubric_per_instance", r"Judge\.rubric_hash\(\) must return the same nonempty str"),
        ("rubric_drifts", r"Judge\.rubric_hash\(\) must return the same nonempty str"),
        ("rubric_type", r"Judge\.rubric_hash\(\) must return the same nonempty str"),
        ("rubric_empty", r"Judge\.rubric_hash\(\) must return the same nonempty str"),
        ("panel_type", r"Judge\.panel_hash\(\) must return the same str"),
        ("panel_unstable", r"Judge\.panel_hash\(\) must return the same str"),
        ("anchor_count", r"Judge\.anchor_count\(\) must return an int >= 0"),
        ("anchor_negative", r"Judge\.anchor_count\(\) must return an int >= 0"),
        ("providers", r"Judge\.providers\(\) must return a frozenset of str"),
        ("providers_items", r"Judge\.providers\(\) must return a frozenset of str"),
        ("models", r"Judge\.models\(\) must return a tuple of str"),
        ("models_list", r"Judge\.models\(\) must return a tuple of str"),
    ],
)
async def test_judge_contract_is_asserted(mode: str, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        await check_judge(BrokenJudge({"mode": mode}), GOAL, [make_candidate(ok())])


@pytest.mark.parametrize(
    "judge",
    [
        ProgramJudge(rosen_score),
        FixedJudge({"value": 2.0}),
        PanelJudge({"panel": "p", "anchors": 1, "providers": ["openai"], "models": ["m"]}),
        Lexicographic([ProgramJudge(rosen_score), FixedJudge({"value": 1.0})]),
    ],
)
async def test_good_judges_pass_strict_validation(judge: Judge) -> None:
    c = make_candidate(ToolOutput(status=RunStatus.OK, data={"f": 0.5}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        [score] = await check_judge(judge, GOAL, [c])
    assert score.candidate_id == c.id


async def test_non_finite_score_is_a_judge_error() -> None:
    c = make_candidate(ok())
    with pytest.warns(UserWarning, match=f"judge error for {c.id}: score key must be") as record:
        assert await check_judge(BrokenJudge({"mode": "nan"}), GOAL, [c]) == []
    assert [str(w.message) for w in record] == [
        f"judge error for {c.id}: score key must be nonempty and finite"
    ]


async def test_judge_error_without_a_score_gives_one_warning() -> None:
    c = make_candidate(ok())
    with pytest.warns(UserWarning, match=f"judge error for {c.id}") as record:
        assert await check_judge(ProgramJudge(raising_score), GOAL, [c]) == []
    assert [str(w.message).split(":")[0] for w in record] == [f"judge error for {c.id}"]


async def test_candidate_without_score_or_error_warns() -> None:
    c = make_candidate(ok())
    with pytest.warns(UserWarning, match=f"judge gave no score and no judge error for {c.id}"):
        assert await check_judge(BrokenJudge({"mode": "drop"}), GOAL, [c]) == []


def test_artifact_root_is_one_per_process_and_removed_at_exit() -> None:
    code = (  # the root holds a store tree before exit, so an empty-only cleanup fails
        "import os\n"
        "from swarm_arena.testing._conformance import _artifact_root, _artifacts\n"
        "root = _artifact_root()\n"
        "assert root == _artifact_root() and os.path.isdir(root)\n"
        "assert os.path.basename(root).startswith('swarm-arena-testing-')\n"
        "_artifacts().put_bytes(b'kept until exit')\n"
        "assert os.listdir(root)\n"
        "print(root)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    root = Path(done.stdout.strip())
    assert root.name.startswith("swarm-arena-testing-")
    assert not root.exists()
