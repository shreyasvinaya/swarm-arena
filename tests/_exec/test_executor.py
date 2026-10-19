import pytest
from _fakes.tools import make_job
from pydantic import ValidationError

from swarm_arena import Resources
from swarm_arena._exec.executor import Executor, Job
from swarm_arena.tool import ParentRef, RunStatus, ToolOutput


def test_kind_rules() -> None:
    with pytest.raises(ValueError, match="kind 'tool' needs tool and request"):
        make_job(kind="tool", command=None)
    with pytest.raises(ValueError, match="kind 'command' needs command"):
        make_job(command=None)


def test_secret_values_hidden_from_repr() -> None:
    assert "sk-123" not in repr(make_job(secret_values={"OPENAI_API_KEY": "sk-123"}))


def test_stage_paths_checked() -> None:
    for root in ({"swarm/x": "a" * 64}, {"./a": "a" * 64}):
        with pytest.raises(ValidationError):
            Job.Stage(root=root)
    for files in ({"../x": "a" * 64}, {"/abs": "a" * 64}):
        with pytest.raises(ValidationError):
            Job.CandidateFiles(request={}, output={}, files=files)
    assert (
        Job.Stage(candidate=Job.CandidateFiles(request={}, output={}, files={})).candidate
        is not None
    )


def test_started_roundtrip() -> None:
    s = Job.Started(
        kind="process",
        job_id="c-1",
        host_id="h:1",
        pid=10,
        pgid=10,
        start_time="Sat Oct 18 13:52:01 2026",
    )
    assert Job.Started.model_validate_json(s.model_dump_json()) == s
    assert s.schema_version == 1


async def test_executor_defaults() -> None:
    class Minimal(Executor):
        async def submit(self, job: Job) -> ToolOutput:
            return ToolOutput(status=RunStatus.OK)

        async def capacity(self) -> Resources | None:
            return None

        async def aclose(self) -> None:
            return None

    m, started = Minimal(), Job.Started(kind="process", job_id="c-1")
    assert m.preflight() is None
    assert await m.confirm_dead(started) is None
    assert await m.kill_orphan(started) is False
    assert (await m.submit(make_job())).status == RunStatus.OK
    assert await m.capacity() is None
    assert await m.aclose() is None
    assert m.config.model_dump() == {}


def test_tool_job_requires_both_tool_and_request() -> None:
    from _fakes.tools import EchoTool

    from swarm_arena import Ref
    from swarm_arena.tool import ToolRequest

    tool = Ref.of(EchoTool)
    request = ToolRequest()
    for fields in ({"tool": tool}, {"request": request}):
        with pytest.raises(ValueError, match="kind 'tool' needs tool and request"):
            make_job(kind="tool", command=None, **fields)
    job = make_job(kind="tool", command=None, tool=tool, request=request)
    assert job.tool == tool
    assert job.request is request


def test_job_is_frozen_and_defaults_are_independent() -> None:
    from dataclasses import FrozenInstanceError

    first, second = make_job(), make_job()
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'id'"):
        first.id = "changed"  # ty: ignore[invalid-assignment]
    assert first.secret_values == second.secret_values == {}
    assert first.secret_values is not second.secret_values
    assert first.env is not second.env
    stage = Job.Stage()
    stage.root["problem/a"] = "a" * 64
    stage.parents.append(ParentRef(candidate_id="c-1", run_id="r", tool_identity="t"))
    assert Job.Stage().root == {}
    assert Job.Stage().parents == []


async def test_callbacks_are_optional_hidden_and_retained() -> None:
    started_seen: list[Job.Started] = []
    partial_seen: list[ToolOutput] = []

    async def sentinel_started(started: Job.Started) -> None:
        started_seen.append(started)

    async def sentinel_partial(output: ToolOutput) -> None:
        partial_seen.append(output)

    assert make_job().on_started is None
    assert make_job().on_partial is None
    job = make_job(on_started=sentinel_started, on_partial=sentinel_partial)
    assert job.on_started is sentinel_started
    assert job.on_partial is sentinel_partial
    assert "sentinel_started" not in repr(job)
    assert "sentinel_partial" not in repr(job)
    started = Job.Started(kind="process", job_id=job.id)
    output = ToolOutput(status=RunStatus.OK)
    await job.on_started(started)
    await job.on_partial(output)
    assert started_seen == [started]
    assert partial_seen == [output]


@pytest.mark.parametrize(
    "path", ["", "a/../b", "a/./b", "a//b", "a/", "a\x00b", "parents/x", "candidate/x"]
)
def test_stage_and_candidate_keys_must_be_in_normal_form(path: str) -> None:
    with pytest.raises(ValidationError):
        Job.Stage(root={path: "a" * 64})
    with pytest.raises(ValidationError):
        Job.CandidateFiles(request={}, output={}, files={path: "a" * 64})


def test_stage_roundtrip_and_frozen_nested_models() -> None:
    candidate = Job.CandidateFiles(request={"schema_version": 1}, output={}, files={"a": "a" * 64})
    stage = Job.Stage(
        root={"problem/input": "b" * 64}, candidate=candidate, request_json={"id": "x"}
    )
    assert Job.Stage.model_validate_json(stage.model_dump_json()) == stage
    for model, field in (
        (stage, "root"),
        (candidate, "files"),
        (Job.Started(kind="dispatch", job_id="x-1", task_name="task"), "job_id"),
    ):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(model, field, None)
    assert Job.Stage.__pydantic_complete__


def test_started_uses_versioned_stored_format() -> None:
    started = Job.Started.model_validate({"kind": "dispatch", "job_id": "x-1", "task_name": "task"})
    assert started.schema_kind == "job_started"
    assert started.schema_version == 1
    assert started.pid is None
    assert started.pgid is None
    assert started.host_id is None
    for version in (0, True, 2):
        with pytest.raises(ValidationError):
            Job.Started.model_validate(
                {"kind": "process", "job_id": "x-1", "schema_version": version}
            )


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (Job.Stage, {"root": {"sentinel-secret": ["sentinel-secret"]}}),
        (Job.Stage, {"root": {"./sentinel-secret": "a" * 64}}),
        (Job.Stage, {"sentinel-secret": "sentinel-secret"}),
        (Job.CandidateFiles, {"request": {}, "output": {}, "files": {"sentinel-secret": {}}}),
        (Job.CandidateFiles, {"request": ["sentinel-secret"], "output": {}, "files": {}}),
        (Job.Started, {"kind": "sentinel-secret", "job_id": "x-1"}),
        (Job.Started, {"kind": "process", "job_id": "x-1", "pid": "sentinel-secret"}),
        (Job.Started, {"kind": "process", "job_id": "x-1", "schema_version": "sentinel-secret"}),
    ],
)
def test_validation_errors_hide_values_and_dynamic_keys(
    model: type[Job.Stage] | type[Job.CandidateFiles] | type[Job.Started], data: dict[str, object]
) -> None:
    import traceback

    with pytest.raises(ValidationError) as exc:
        model.model_validate(data)
    assert "sentinel-secret" not in str(exc.value)
    assert "sentinel-secret" not in "".join(traceback.format_exception(exc.value))
    assert "sentinel-secret" not in repr(exc.value.errors(include_input=False, include_url=False))


def test_executor_requires_all_abstract_operations() -> None:
    with pytest.raises(TypeError, match=r"abstract.*aclose.*capacity.*submit"):
        Executor()  # ty: ignore[call-non-callable]


@pytest.mark.parametrize(
    ("model", "data", "location"),
    [
        (Job.Stage, {"sentinel-secret": "sentinel-value"}, ("<field>",)),
        (
            Job.CandidateFiles,
            {"request": {}, "output": {}, "files": {}, "sentinel-secret": "sentinel-value"},
            ("<field>",),
        ),
        (
            Job.Started,
            {"kind": "process", "job_id": "x-1", "sentinel-secret": "sentinel-value"},
            ("<field>",),
        ),
        (
            Job.Stage,
            {"candidate": {"request": {}, "output": {}, "files": {}, "sentinel-secret": 1}},
            ("candidate", "<field>"),
        ),
        (
            Job.Stage,
            {
                "parents": [
                    {"candidate_id": "c", "run_id": "r", "tool_identity": "t", "sentinel-secret": 1}
                ]
            },
            ("parents", 0, "<field>"),
        ),
    ],
)
def test_runtime_unknown_fields_are_hidden(
    model: type[Job.Stage] | type[Job.CandidateFiles] | type[Job.Started],
    data: dict[str, object],
    location: tuple[str | int, ...],
) -> None:
    import traceback

    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    errors = caught.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == location
    assert errors[0]["type"] == "extra_forbidden"
    for rendered in (
        str(caught.value),
        repr(caught.value.args),
        repr(errors),
        "".join(traceback.format_exception(caught.value)),
    ):
        assert "sentinel-secret" not in rendered
        assert "sentinel-value" not in rendered
