from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.tools import GOAL, WRITE, RecordingExecutor, cand

from swarm_arena._exec.executor import Job
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Resources
from swarm_arena._model.tooling import RunStatus, ToolOutput, ToolRequest
from swarm_arena._scoring.check import CheckContext
from swarm_arena._scoring.judge import JudgeContext
from swarm_arena._scoring.script_check import ScriptCheck
from swarm_arena._scoring.script_judge import ScriptJudge
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.files import snapshot_fileset


def sj(tmp_path: Path, artifacts: ArtifactStore, body: str, **cfg: Any) -> ScriptJudge:
    path = tmp_path / "judge/score.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(body)
    globs = cfg.pop("globs", ["judge/score.py"])
    files = snapshot_fileset(FileSet(globs=globs), tmp_path, artifacts)
    return ScriptJudge(command=["python", "judge/score.py"], files=files, **cfg)


@pytest.mark.parametrize(("direction", "key"), [("min", (-0.25,)), ("max", (0.25,))])
async def test_value_with_direction(
    tmp_path: Path,
    artifacts: ArtifactStore,
    jctx: JudgeContext,
    direction: str,
    key: tuple[float, ...],
) -> None:
    judge = await anyio.to_thread.run_sync(
        partial(
            sj,
            tmp_path,
            artifacts,
            WRITE.format("{'value': 0.25, 'verdict': 'ok', 'notes': 'n'}"),
            direction=direction,
        )
    )
    [s] = await judge.judge(GOAL, [cand(id="c1")], jctx)
    assert (s.candidate_id, s.key, s.value) == ("c1", key, 0.25)
    assert (s.verdict, s.notes) == ("ok", "n")


async def test_key_list(tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext) -> None:
    body = WRITE.format("{'key': [1, 2], 'value': 0.25, 'verdict': 'ok', 'notes': 'n'}")
    judge = await anyio.to_thread.run_sync(sj, tmp_path, artifacts, body)
    [s] = await judge.judge(GOAL, [cand(id="c1")], jctx)
    assert (s.key, s.value, s.verdict, s.notes) == ((1.0, 2.0), 0.25, "ok", "n")


async def test_nonfinite_value_is_judge_error(
    tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext
) -> None:
    body = "import os; open(os.environ['SWARM_ARENA_RESULT'], 'w').write('{\"value\": NaN}')"
    judge = await anyio.to_thread.run_sync(sj, tmp_path, artifacts, body)
    assert await judge.judge(GOAL, [cand(id="c1")], jctx) == []
    assert jctx.errors["c1"] == "judge script failed: error"


async def test_one_workspace_per_candidate(
    tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext
) -> None:
    body = (
        "import json; from pathlib import Path; "
        "assert not Path('marker').exists(); Path('marker').write_text('mine'); "
        "cid = json.load(open('candidate/request.json'))['params']['cid']; "
        + WRITE.format("{'value': 1.0, 'notes': cid}")
    )
    judge = await anyio.to_thread.run_sync(sj, tmp_path, artifacts, body)
    cands = [cand(id=f"c{i}", request=ToolRequest(inputs={"cid": f"c{i}"})) for i in range(3)]
    scores = await judge.judge(GOAL, cands, jctx)
    assert [(s.candidate_id, s.notes) for s in scores] == [("c0", "c0"), ("c1", "c1"), ("c2", "c2")]


async def test_hidden_file_staged_for_the_judge(
    tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext
) -> None:
    def prepare() -> ScriptJudge:
        path = tmp_path / "judge/hidden/points.csv"
        path.parent.mkdir(parents=True)
        path.write_text("7.5")
        body = "value = float(open('judge/hidden/points.csv').read()); " + WRITE.format(
            "{'value': value}"
        )
        return sj(tmp_path, artifacts, body, globs=["judge/score.py", "judge/hidden/points.csv"])

    judge = await anyio.to_thread.run_sync(prepare)
    [score] = await judge.judge(GOAL, [cand()], jctx)
    assert score.value == 7.5


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("raise SystemExit(1)", "judge script failed: error"),
        (WRITE.format("{'other': 1}"), "invalid judge result: <field>: extra_forbidden"),
    ],
)
async def test_failed_script_or_bad_result_is_judge_error(
    tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext, body: str, error: str
) -> None:
    judge = await anyio.to_thread.run_sync(sj, tmp_path, artifacts, body)
    assert await judge.judge(GOAL, [cand(id="c1")], jctx) == []
    assert jctx.errors["c1"] == error


RESULTS = {
    "c0": {"value": 1.0},
    "c1": {"value": None},
    "c2": {"key": None},
    "c3": {"value": "0.5"},
    "c4": {"key": [1, "a"]},
    "c5": {"value": True},
    "c6": {"key": []},
}


async def test_malformed_results_are_per_candidate_errors(
    tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext
) -> None:
    body = (
        "import json, os; cid = json.load(open('candidate/request.json'))['params']['cid']; "
        f"json.dump({RESULTS!r}[cid], open(os.environ['SWARM_ARENA_RESULT'], 'w'))"
    )
    judge = await anyio.to_thread.run_sync(sj, tmp_path, artifacts, body)
    cands = [cand(id=c, request=ToolRequest(inputs={"cid": c})) for c in RESULTS]
    scores = await judge.judge(GOAL, cands, jctx)
    assert [s.candidate_id for s in scores] == ["c0"]
    assert jctx.errors["c1"] == jctx.errors["c2"] == "result needs key or value"
    assert all("invalid judge result" in jctx.errors[c] for c in ("c3", "c4", "c5"))
    assert jctx.errors["c6"] == "invalid judge result: key: must be nonempty and finite"


async def test_execute_errors_propagate() -> None:
    class Refusing(RecordingExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            raise RuntimeError("fenced")

    def ctx_args() -> dict[str, Any]:
        return {"goal": GOAL, "executor": Refusing(), "artifact_root": "/a"}

    with pytest.raises(RuntimeError, match="fenced"):
        await ScriptCheck(command=["x"]).check(cand(), CheckContext(**ctx_args()))
    with pytest.raises(RuntimeError, match="ExceptionGroup"):
        await ScriptJudge(command=["x"]).judge(
            GOAL, [cand(id="c1"), cand(id="c2")], JudgeContext(**ctx_args())
        )


async def test_executor_error_cancels_sibling() -> None:
    started = anyio.Event()
    cancelled = anyio.Event()

    class Refusing(RecordingExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            if (
                job.stage.candidate is not None
                and job.stage.candidate.request["params"]["cid"] == "bad"
            ):
                await started.wait()
                raise RuntimeError("fenced")
            started.set()
            try:
                await anyio.sleep_forever()
                raise AssertionError("unreachable")
            finally:
                cancelled.set()

    context = JudgeContext(goal=GOAL, executor=Refusing(), artifact_root="/a")
    cands = [cand(id=cid, request=ToolRequest(inputs={"cid": cid})) for cid in ("bad", "waiting")]
    with anyio.fail_after(2), pytest.raises(RuntimeError, match="ExceptionGroup"):
        await ScriptJudge(command=["x"]).judge(GOAL, cands, context)
    assert cancelled.is_set()
    assert context.errors == {}


@pytest.mark.parametrize(
    "data",
    [
        {"value": "ad25cd43a1711e358861c29d157cf8df"},
        {"key": [], "notes": "ad25cd43a1711e358861c29d157cf8df"},
        {"key": [float("inf")], "notes": "ad25cd43a1711e358861c29d157cf8df"},
    ],
)
async def test_result_errors_hide_secrets(data: dict[str, Any]) -> None:
    class Results(RecordingExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            return ToolOutput.model_construct(status=RunStatus.OK, data=data)

    context = JudgeContext(goal=GOAL, executor=Results(), artifact_root="/a")
    assert await ScriptJudge(command=["x"]).judge(GOAL, [cand()], context) == []
    assert "invalid judge result" in context.errors["c1"]
    assert "ad25cd43a1711e358861c29d157cf8df" not in repr(context.errors)


async def test_resources_pass_through() -> None:
    executor = RecordingExecutor()
    context = JudgeContext(goal=GOAL, executor=executor, artifact_root="/a")
    resources = Resources(cpus=2)
    await ScriptJudge(command=["x"], resources=resources).judge(GOAL, [cand()], context)
    assert executor.jobs[0].resources == resources


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["key", "value"])
async def test_nonfinite_result_is_per_candidate_error(number: float, field: str) -> None:
    class Results(RecordingExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            assert job.stage.candidate is not None
            cid = job.stage.candidate.request["params"]["cid"]
            data = {"key": [1.0], "value": 0.25}
            if cid == "bad":
                data[field] = [number] if field == "key" else number
            # A custom executor bypassing validation must still fail closed.
            return ToolOutput.model_construct(status=RunStatus.OK, data=data)

    context = JudgeContext(goal=GOAL, executor=Results(), artifact_root="/a")
    candidates = [cand(id=cid, request=ToolRequest(inputs={"cid": cid})) for cid in ("bad", "good")]
    scores = await ScriptJudge(command=["x"]).judge(GOAL, candidates, context)
    assert [score.candidate_id for score in scores] == ["good"]
    expected = (
        "invalid judge result: key: must be nonempty and finite"
        if field == "key"
        else "invalid judge result: value: finite_number"
    )
    assert context.errors == {"bad": expected}


async def test_nonfinite_candidate_output_does_not_cancel_batch(
    tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.tools import PyTool, make_job

    from swarm_arena._model.ref import Ref
    from swarm_arena._model.tooling import CompletedRun

    def collect(
        self: PyTool, request: ToolRequest, workspace: Path, run: CompletedRun
    ) -> ToolOutput:
        return ToolOutput(status=RunStatus.OK, data={"nested": [{"loss": float("nan")}]})

    monkeypatch.setattr(PyTool, "collect", collect)
    output = await jctx.executor.submit(
        make_job(
            id="nonfinite-collect",
            kind="tool",
            command=None,
            request=ToolRequest(),
            tool=Ref.of(PyTool({"code": "pass"})),
            artifact_root=jctx.artifact_root,
        )
    )
    body = (
        "import json; output = json.load(open('candidate/output.json')); "
        "assert output['status'] == 'ok'; " + WRITE.format("{'value': 1.0}")
    )
    judge = await anyio.to_thread.run_sync(sj, tmp_path, artifacts, body)
    scores = await judge.judge(GOAL, [cand(id="bad", output=output), cand(id="good")], jctx)
    assert output.status is RunStatus.ERROR
    assert [score.candidate_id for score in scores] == ["good"]
    assert jctx.errors == {"bad": "judge script failed: error"}


@pytest.mark.parametrize(
    "settings", [{"timeout_s": 2}, {"grace_s": 20}, {"resources": {"cpus": 2}}]
)
def test_rubric_excludes_execution_settings(settings: dict[str, Any]) -> None:
    assert (
        ScriptJudge(command=["judge"]).rubric_hash()
        == ScriptJudge(command=["judge"], **settings).rubric_hash()
    )


@pytest.mark.parametrize("version", [None, 1, 99, 0, True, "1"])
async def test_judge_result_schema_version(jctx: JudgeContext, version: Any) -> None:
    payload = {"value": 1.0}
    if version is not None:
        payload["schema_version"] = version

    class ResultExecutor(RecordingExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            return ToolOutput(status=RunStatus.OK, data=payload)

    jctx.executor = ResultExecutor()
    result = await ScriptJudge(command=["judge"]).judge(GOAL, [cand()], jctx)
    assert bool(result) is (version is None or (type(version) is int and version == 1))
    if version == 99:
        assert "schema_version" in repr(jctx.errors)
        assert "newer" in repr(jctx.errors)
