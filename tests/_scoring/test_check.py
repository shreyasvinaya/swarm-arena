from typing import cast

import anyio
import pytest
from _fakes.tools import GOAL, AlwaysPass, RecordingExecutor, cand

from swarm_arena._exec.executor import Executor, Job
from swarm_arena._model.goal import FileSet
from swarm_arena._model.ids import IdGen, run_tag
from swarm_arena._model.plan import Resources
from swarm_arena._model.tooling import Command, FileRef, RunStatus, ToolOutput
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._scoring.judge import JudgeContext
from swarm_arena._store.artifacts import ArtifactStore

H1 = "1" * 64


def test_check_defaults() -> None:
    assert AlwaysPass().feedback == "reasons"
    assert AlwaysPass().preflight() is None


async def test_check_contract() -> None:
    ctx = CheckContext(goal=GOAL, executor=RecordingExecutor(), artifact_root="/a")
    assert (await AlwaysPass().check(cand(), ctx)).passed
    with pytest.raises(TypeError, match="abstract"):
        Check()  # ty: ignore[call-non-callable]


async def test_execute_builds_a_command_job() -> None:
    ex = RecordingExecutor()
    ctx = CheckContext(
        goal=GOAL,
        executor=ex,
        artifact_root="/a",
        run_id="r",
        round=2,
        secrets_file="/s.env",
        ids=IdGen("r", 1),
    )
    c = cand(id="c9")
    await ctx.execute(
        Command(argv=["python", "checks/check.py"], env=["K"], grace_s=2.0),
        files={"checks/check.py": H1},
        candidate=c,
    )
    job = ex.jobs[0]
    assert (
        job.kind,
        job.id,
        job.timeout_s,
        job.grace_s,
        job.secret_env,
        job.secrets_file,
        job.resources,
    ) == ("command", f"x-{run_tag('r')}-e1-0001", 3600.0, 2.0, ("K",), "/s.env", Resources())
    assert job.stage.root == {"checks/check.py": H1}
    assert job.stage.candidate is not None
    assert job.env["SWARM_ARENA_CANDIDATE_ID"] == "c9"
    assert job.env["SWARM_ARENA_ROUND"] == "2"


@pytest.mark.parametrize(("configured", "expected"), [(None, 60.0), (30.0, 30.0), (99_999.0, 60.0)])
async def test_execute_timeout_follows_and_is_capped_by_limit(
    configured: float | None, expected: float
) -> None:
    ex = RecordingExecutor()
    ctx = CheckContext(goal=GOAL, executor=ex, artifact_root="/a", tool_timeout_s=60.0)
    await ctx.execute(Command(argv=["x"], timeout_s=configured))
    assert ex.jobs[0].timeout_s == expected


async def test_execute_passes_on_started_to_job() -> None:
    async def cb(s: Job.Started) -> None:
        return None

    ex = RecordingExecutor()
    await CheckContext(goal=GOAL, executor=ex, artifact_root="/a", on_started=cb).execute(
        Command(argv=["x"])
    )
    await CheckContext(goal=GOAL, executor=ex, artifact_root="/a").execute(Command(argv=["x"]))
    await JudgeContext(goal=GOAL, executor=ex, artifact_root="/a", on_started=cb).execute(
        Command(argv=["x"])
    )
    assert [j.on_started for j in ex.jobs] == [cb, None, cb]


async def test_execute_without_candidate_and_default_ids() -> None:
    ex = RecordingExecutor()
    ctx = CheckContext(goal=GOAL, executor=ex, artifact_root="/a", part="p")
    command = Command(argv=["x"])
    resources = Resources(cpus=0.5)
    out = await ctx.execute(command, resources=resources)
    await ctx.execute(command, files={"checks/x.txt": H1})
    assert out.status == RunStatus.OK
    a, b = ex.jobs
    assert a.command is command
    assert a.resources is resources
    assert a.stage == Job.Stage()
    assert b.stage.root == {"checks/x.txt": H1}
    assert a.id == f"x-{run_tag('local')}-e0-0001"
    assert b.id == f"x-{run_tag('local')}-e0-0002"
    assert ctx.ids.counters() == {"x": 2}
    assert ctx.part == "p"
    assert a.env == {
        "SWARM_ARENA_RUN_ID": "local",
        "SWARM_ARENA_GOAL_ID": "g",
        "SWARM_ARENA_CANDIDATE_ID": "",
        "SWARM_ARENA_TEAM": "",
        "SWARM_ARENA_AGENT": "",
        "SWARM_ARENA_ROUND": "0",
    }
    assert await ex.capacity() is None
    assert await ex.aclose() is None


async def test_execute_preserves_executor_errors() -> None:
    class Refusing(RecordingExecutor):
        async def submit(self, job: Job) -> ToolOutput:
            raise RuntimeError("fenced")

    ctx = CheckContext(goal=GOAL, executor=Refusing(), artifact_root="/a")
    with pytest.raises(RuntimeError, match=r"^fenced$"):
        await ctx.execute(Command(argv=["x"]))


async def test_execute_stages_effective_and_output_files() -> None:
    goal = GOAL.model_copy(update={"files": FileSet(manifest={"base.txt": H1})})
    c = cand(
        data={"x": 1},
        overlay={"base.txt": "2" * 64},
        output=ToolOutput(
            status=RunStatus.OK,
            data={"x": 1},
            files=[FileRef(name="out", path="out.txt", sha256="3" * 64, size=3)],
        ),
    )
    ex = RecordingExecutor()
    await CheckContext(goal=goal, executor=ex, artifact_root="/a").execute(
        Command(argv=["x"]), files={"checks/x.txt": H1}, candidate=c
    )
    stage = ex.jobs[0].stage
    assert stage.candidate is not None
    assert stage.candidate.files == {"base.txt": "2" * 64, "out.txt": "3" * 64}
    assert stage.candidate.output["data"] == {"x": 1}
    assert stage.root == {"checks/x.txt": H1}


async def test_execute_on_local_executor(
    request: pytest.FixtureRequest, artifacts: ArtifactStore, artifact_root: str
) -> None:
    # Task 20 is dispatched before Task 17; run this automatically after integration.
    pytest.importorskip("swarm_arena._exec.local", reason="Task 17 LocalExecutor not integrated")
    local_executor = cast(Executor, request.getfixturevalue("local_executor"))
    own_hash = await anyio.to_thread.run_sync(artifacts.put_bytes, b"own")
    output_hash = await anyio.to_thread.run_sync(artifacts.put_bytes, b"output")
    c = cand(
        output=ToolOutput(
            status=RunStatus.OK,
            data={"x": 1},
            files=[FileRef(name="out", path="out.txt", sha256=output_hash, size=6)],
        )
    )
    code = (
        "import json, os; from pathlib import Path; "
        "data = {'own': Path('checks/x.txt').read_text(), "
        "'output': json.loads(Path('candidate/output.json').read_text())['data'], "
        "'files': sorted(str(p.relative_to('candidate/files')) "
        "for p in Path('candidate/files').rglob('*') if p.is_file())}; "
        "Path(os.environ['SWARM_ARENA_RESULT']).write_text(json.dumps(data))"
    )
    out = await CheckContext(
        goal=GOAL, executor=local_executor, artifact_root=artifact_root
    ).execute(Command(argv=["python", "-c", code]), files={"checks/x.txt": own_hash}, candidate=c)
    assert out.status == RunStatus.OK
    assert out.data == {"own": "own", "output": {"x": 1}, "files": ["out.txt"]}
