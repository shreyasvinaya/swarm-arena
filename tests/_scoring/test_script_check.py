import traceback
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.tools import GOAL, RecordingExecutor, cand
from pydantic import ValidationError

from swarm_arena._exec.executor import Job
from swarm_arena._model.catalog import PlannerCatalog
from swarm_arena._model.goal import FileSet
from swarm_arena._model.results import CheckResult
from swarm_arena._model.tooling import FileRef, RunStatus, ToolOutput
from swarm_arena._scoring.check import CheckContext
from swarm_arena._scoring.script_check import ScriptCheck, _CheckResultFile
from swarm_arena._scoring.script_judge import ScriptJudge, _JudgeResultFile
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.files import snapshot_fileset

WRITE = "import json, os; json.dump({}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
SECRET = "SENTINEL_PRIVATE_VALUE_22"


def script_check(tmp_path: Path, artifacts: ArtifactStore, body: str, **cfg: Any) -> ScriptCheck:
    path = tmp_path / "checks/check.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(body)
    files = snapshot_fileset(FileSet(globs=["checks/check.py"]), tmp_path, artifacts)
    return ScriptCheck(command=["python", "checks/check.py"], files=files, **cfg)


async def test_pass_and_reasons(
    tmp_path: Path, artifacts: ArtifactStore, ctx: CheckContext
) -> None:
    chk = await anyio.to_thread.run_sync(
        script_check,
        tmp_path,
        artifacts,
        WRITE.format("{'passed': False, 'reasons': ['x too big']}"),
    )
    assert await chk.check(cand(), ctx) == CheckResult(
        check="script-check", passed=False, reasons=["x too big"]
    )


@pytest.mark.parametrize(
    ("body", "cfg", "reason"),
    [
        ("raise SystemExit(2)", {}, "check script failed: error"),
        ("pass", {}, "check script failed: error"),
        (WRITE.format("{'passed': 'yes'}"), {}, "passed"),
        (WRITE.format("{'passed': 1}"), {}, "passed"),
        (WRITE.format("{'passed': True, 'reasons': 'x'}"), {}, "reasons"),
        (WRITE.format("{'passed': True, 'reason': ['x']}"), {}, "extra_forbidden"),
        (
            "import time; time.sleep(60)",
            {"timeout_s": 1.0, "grace_s": 0.5},
            "check script failed: timeout",
        ),
    ],
)
async def test_crash_fails_closed(
    tmp_path: Path,
    artifacts: ArtifactStore,
    ctx: CheckContext,
    body: str,
    cfg: dict[str, Any],
    reason: str,
) -> None:
    chk = await anyio.to_thread.run_sync(partial(script_check, tmp_path, artifacts, body, **cfg))
    result = await chk.check(cand(), ctx)
    assert result.passed is False
    assert reason in " ".join(result.reasons)


async def test_sees_own_files_and_candidate_layout(
    tmp_path: Path, artifacts: ArtifactStore, ctx: CheckContext
) -> None:
    def prepare() -> tuple[ScriptCheck, str, str, str]:
        base = artifacts.put_bytes(b"base")
        overlay = artifacts.put_bytes(b"edited")
        output = artifacts.put_bytes(b"output")
        body = (
            "from pathlib import Path; import json; "
            "assert Path('checks/check.py').is_file(); "
            "assert json.load(open('candidate/output.json'))['status'] == 'ok'; "
            "assert Path('candidate/files/problem/a.txt').read_text() == 'edited'; "
            "assert Path('candidate/files/out.txt').read_text() == 'output'; "
            + WRITE.format("{'passed': True, 'details': {'n': 2}}")
        )
        return script_check(tmp_path, artifacts, body), base, overlay, output

    chk, base, overlay, output = await anyio.to_thread.run_sync(prepare)
    ctx.goal = GOAL.model_copy(update={"files": FileSet(manifest={"problem/a.txt": base})})
    candidate = cand(
        overlay={"problem/a.txt": overlay},
        output=ToolOutput(
            status=RunStatus.OK,
            files=[FileRef(name="out.txt", path="out.txt", sha256=output, size=6)],
        ),
    )
    result = await chk.check(candidate, ctx)
    assert result.passed is True
    assert result.details == {"n": 2}


def test_feedback_and_preflight() -> None:
    assert ScriptCheck(command=["x"], feedback="pass_fail").feedback == "pass_fail"
    for cls in (ScriptCheck, ScriptJudge):
        cls(command=["x"]).preflight()
        with pytest.raises(RuntimeError, match="not snapshotted"):
            cls(command=["x"], files=["a/*"]).preflight()
        config: Any = cls.Config(command=["x"])
        assert cls(config).config is config
        with pytest.raises(TypeError, match="config or fields"):
            cls(config, command=["y"])


@pytest.mark.parametrize(
    ("ref", "kind"), [("swarm_arena:ScriptCheck", "check"), ("swarm_arena:ScriptJudge", "judge")]
)
@pytest.mark.parametrize("field", ["command", "python", "files", "env"])
def test_catalog_refuses_settable_script_fields(ref: str, kind: str, field: str) -> None:
    entry = {"name": "s", "kind": kind, "ref": ref, "description": "d"}
    with pytest.raises(ValidationError, match=f"human-only field '{field}'"):
        PlannerCatalog.model_validate(
            {"entries": [{**entry, "settable": {field: {"type": "string"}}}]}
        )
    PlannerCatalog.model_validate(
        {"entries": [{**entry, "settable": {"timeout_s": {"type": "number"}}}]}
    )


def assert_secret_absent(error: BaseException) -> None:
    surfaces = [
        str(error),
        repr(error),
        repr(error.args),
        repr(getattr(error, "__notes__", [])),
        "".join(traceback.format_exception(error)),
    ]
    if isinstance(error, ValidationError):
        surfaces += [repr(error.errors()), error.json()]
    assert all(SECRET not in value for value in surfaces)
    if error.__cause__ is not None:
        assert_secret_absent(error.__cause__)


@pytest.mark.parametrize("cls", [ScriptCheck, ScriptJudge])
@pytest.mark.parametrize(
    "fields",
    [
        {"command": []},
        {"command": [SECRET, {}]},
        {"env": [SECRET + "!"]},
        {"timeout_s": SECRET},
        {"grace_s": -1, "python": SECRET},
        {"files": {"manifest": {SECRET: "bad"}}},
        {SECRET: SECRET},
        {"resources": {SECRET: SECRET}},
    ],
)
def test_config_errors_hide_secrets(
    cls: type[ScriptCheck] | type[ScriptJudge], fields: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError) as caught:
        cls(**{"command": ["x"], **fields})
    assert_secret_absent(caught.value)


@pytest.mark.parametrize(
    "model", [ScriptCheck.Config, ScriptJudge.Config, _CheckResultFile, _JudgeResultFile]
)
def test_malformed_json_hides_secret(model: Any) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate_json(SECRET)
    assert_secret_absent(caught.value)
    assert model.model_config["hide_input_in_errors"] is True


async def test_invalid_check_result_hides_secret() -> None:
    class Results(RecordingExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            return ToolOutput(status=RunStatus.OK, data={"passed": True, SECRET: SECRET})

    context = CheckContext(goal=GOAL, executor=Results(), artifact_root="/a")
    result = await ScriptCheck(command=["x"]).check(cand(), context)
    assert result.passed is False
    assert SECRET not in repr(result)
