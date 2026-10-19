import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from threading import get_ident
from typing import Any

import pytest
from _fakes.tools import cand
from pydantic import BaseModel, ValidationError

from swarm_arena import Goal
from swarm_arena._errors import ConfigError, SoftError
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.materialize import materialize
from swarm_arena._tooling.script_tool import (
    FileViewContext,
    ListFilesArgs,
    ReadFileArgs,
    ScriptTool,
    script_command,
)
from swarm_arena._tooling.staging import plan_tool_stage, put_edits, validate_edits
from swarm_arena.board import Candidate
from swarm_arena.config import PlannerCatalog
from swarm_arena.tool import Command, CompletedRun, FileSet, ParentRef, RunStatus, ToolRequest

HA, HA2, HR, HN = (digit * 64 for digit in ("a", "b", "c", "d"))
SECRET = "SECRET_SENTINEL_script_tool"


def _assert_secret_safe(error: BaseException, secret: str) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for diagnostic in (
            str(current),
            repr(current),
            repr(current.args),
            "".join(traceback.format_exception(current)),
        ):
            assert secret not in diagnostic
        pending.extend(
            chained for chained in (current.__cause__, current.__context__) if chained is not None
        )


def test_constructor_timeout_error_is_secret_safe() -> None:
    secret = "SECRET_SENTINEL_constructor_timeout"
    with pytest.raises(ValueError, match="timeout_s") as caught:
        ScriptTool(command=["x"], timeout_s=secret)
    _assert_secret_safe(caught.value, secret)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("entrypoint", ["constructor", "model_validate", "model_validate_json"])
def test_config_env_error_is_secret_safe(entrypoint: str) -> None:
    secret = "SECRET_SENTINEL_config_env"
    _check_model_error(
        ScriptTool.Config, {"command": ["x"], "env": [{"token": secret}]}, entrypoint, "env", secret
    )


def _check_model_error(
    model: type[BaseModel], data: dict[str, Any], entrypoint: str, field: str, secret: str
) -> None:
    def validate() -> BaseModel:
        if entrypoint == "constructor":
            return model(**data)
        if entrypoint == "model_validate_json":
            return model.model_validate_json(json.dumps(data))
        return model.model_validate(data)

    with pytest.raises(ValidationError, match=field) as caught:
        validate()
    _assert_secret_safe(caught.value, secret)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("entrypoint", ["constructor", "model_validate", "model_validate_json"])
@pytest.mark.parametrize("kind", ["integer", "array", "enum", "extra"])
def test_parameter_input_error_is_secret_safe(entrypoint: str, kind: str) -> None:
    secret = "SECRET_SENTINEL_parameter_input"
    schemas: dict[str, dict[str, Any]] = {
        "integer": {"type": "integer"},
        "array": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "enum": {"enum": [secret]},
        "extra": {"type": "integer"},
    }
    values: dict[str, Any] = {"integer": secret, "array": [[secret]], "enum": "wrong", "extra": 1}
    model = ScriptTool(command=["x"], params={"count": schemas[kind]}).get_input_model()
    data = {"count": values[kind]}
    if kind == "extra":
        data[secret] = secret
    _check_model_error(model, data, entrypoint, "<field>" if kind == "extra" else "count", secret)


@pytest.mark.parametrize("nested", [False, True])
def test_schema_type_error_is_secret_safe(nested: bool) -> None:
    secret = "SECRET_SENTINEL_schema_type"
    schema: dict[str, Any] = {"type": secret}
    if nested:
        schema = {"type": "array", "items": schema}
    tool = ScriptTool(command=["x"], params={"count": schema})
    with pytest.raises(ValueError, match=r"script_params\.count.*unsupported type") as caught:
        tool.get_input_model()
    _assert_secret_safe(caught.value, secret)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_config_nested_location_is_secret_safe() -> None:
    secret = "SECRET_SENTINEL_config_location"
    with pytest.raises(ValidationError, match="params") as caught:
        ScriptTool.Config(command=["x"], params={secret: secret})  # ty: ignore[invalid-argument-type]
    _assert_secret_safe(caught.value, secret)
    assert caught.value.errors()[0]["loc"] == ("params", "<…>")


class FakeView:
    def __init__(
        self,
        *,
        goal_manifest: dict[str, str],
        candidates: dict[str, Candidate],
        blobs: dict[str, bytes],
    ) -> None:
        self._goal = Goal(id="g", namespace="n", files=FileSet(manifest=goal_manifest))
        self.candidates = candidates
        self.blobs = blobs
        self.reads: list[str] = []
        self.event_loop_thread = get_ident()

    @property
    def goal(self) -> Goal:
        return self._goal

    def get_candidate(self, id: str) -> Candidate:
        if id not in self.candidates:
            raise SoftError(f"no such candidate: {id}")
        return self.candidates[id]

    def _read_artifact(self, sha256: str) -> bytes:
        assert get_ident() != self.event_loop_thread
        self.reads.append(sha256)
        return self.blobs[sha256]


def test_config_forms() -> None:
    a = ScriptTool(command=["python", "run.py"])
    assert a.config == ScriptTool(ScriptTool.Config(command=["python", "run.py"])).config
    with pytest.raises(TypeError, match="config or fields"):
        ScriptTool(ScriptTool.Config(command=["x"]), timeout_s=1)
    with pytest.raises(ConfigError, match="<field>: extra_forbidden"):
        ScriptTool(command=["x"], timeout=10)
    with pytest.raises(ValidationError, match="frozen"):
        a.config.timeout_s = 1  # ty: ignore[invalid-assignment]


def test_methods_follow_config() -> None:
    t = ScriptTool(
        command=["python", "problem/run.py"],
        editable=["problem/*.phyk"],
        deterministic=False,
        providers=["anthropic"],
        params={
            "lr": {"type": "number", "minimum": 1e-5, "maximum": 1.0},
            "epochs": {"type": "integer", "maximum": 400},
        },
        description="Tune it.",
    )
    assert tuple(t.editable()) == ("problem/*.phyk",)
    assert not t.is_deterministic(ToolRequest())
    assert t.providers() == frozenset({"anthropic"})
    assert t.get_description() == "Tune it."
    with pytest.raises(ValidationError):
        t.get_input_model().model_validate({"lr": 0.1, "epochs": 401})
    assert t.get_input_model().model_validate({"lr": 0.1, "epochs": 400}).model_dump() == {
        "lr": 0.1,
        "epochs": 400,
    }
    assert t.get_input_model() is t.get_input_model()
    assert t.get_input_model().__name__ == "script_params"
    default = ScriptTool(command=["x"])
    assert default.get_description() == "Runs your script on your files."
    assert default.is_deterministic(ToolRequest())
    assert default.providers() == frozenset()
    assert tuple(default.editable()) == ()
    assert t.command(ToolRequest(), Path("/ws")) == Command(
        argv=["python", "problem/run.py"], grace_s=10.0
    )


def test_placeholder_and_instance_models() -> None:
    t = ScriptTool(command=["x"], params={"n": {"type": "integer"}})
    placeholder = type(t).input_model
    assert placeholder.model_fields == {}
    assert placeholder.model_config["frozen"] is True
    with pytest.raises(ValidationError):
        placeholder.model_validate({"n": 1})
    with pytest.raises(ValidationError):
        t.get_input_model().model_validate({"n": "1"})
    with pytest.raises(ValidationError):
        t.get_input_model().model_validate({"n": 1, "extra": 2})
    assert ScriptTool(command=["x"]).get_input_model()().model_dump() == {}


def test_command_forwards_config() -> None:
    t = ScriptTool(
        command=["python", "./problem/run.py", "--flag"],
        python="/opt/env/bin/python",
        env=["API_KEY"],
        timeout_s=30,
        grace_s=2,
    )
    expected = Command(
        argv=["python", "./problem/run.py", "--flag"],
        python="/opt/env/bin/python",
        env=["API_KEY"],
        timeout_s=30,
        grace_s=2,
    )
    assert t.command(ToolRequest(), Path("/ws")) == expected
    assert script_command(t.config) == expected


@pytest.mark.parametrize("spelling", ["problem/run.py", "./problem/run.py", "problem//run.py"])
def test_editable_glob_never_unprotects_script(spelling: str) -> None:
    t = ScriptTool(
        command=["python", spelling],
        editable=["problem/*"],
        files=FileSet(globs=["tool/*"], manifest={"tool/helper.py": "1" * 64}),
    )
    root = {"problem/run.py", "problem/a.txt", "tool/helper.py"}
    protected = t._protected(root)
    assert protected == frozenset({"problem/run.py", "tool/helper.py"})
    with pytest.raises(SoftError, match=r"problem/run\.py: protected"):
        validate_edits(
            {"problem/run.py": "evil"},
            editable=t.editable(),
            protected=protected,
            max_edit_bytes=262_144,
        )
    parent = ParentRef(
        candidate_id="c1", run_id="r", tool_identity="T", overlay={"problem/run.py": "2" * 64}
    )
    stage, overlay, warnings = plan_tool_stage(
        goal_files={"problem/run.py": "3" * 64, "problem/a.txt": "4" * 64},
        tool_files=t._own_files(),
        editable=t.editable(),
        protected=protected,
        private_paths=set(),
        parents=[parent],
        receiving_identity="T",
        edits={},
        request_json=None,
    )
    assert stage.root["problem/run.py"] == "3" * 64
    assert overlay == {}
    assert "problem/run.py" in warnings[0]


def test_interpreter_and_own_files_are_protected() -> None:
    t = ScriptTool(
        command=["./bin//runner", "problem/run.py", "/outside", "../escape", "swarm/x", ""],
        python="./env//python",
        files=FileSet(manifest={"tool/helper.py": HA}),
    )
    assert t._own_files() == {"tool/helper.py": HA}
    assert t._protected({"bin/runner", "env/python", "problem/run.py", "data.txt"}) == frozenset(
        {"bin/runner", "env/python", "problem/run.py", "tool/helper.py"}
    )
    assert ScriptTool(command=["python", "missing.py"])._protected(set()) == frozenset()


def test_protected_script_survives_materialization(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    original = artifacts.put_bytes(b"original script")
    replacement = artifacts.put_bytes(b"replaced script")
    data = artifacts.put_bytes(b"original data")
    t = ScriptTool(command=["python", "./problem//run.py"], editable=["problem/*"])
    goal_files = {"problem/run.py": original, "problem/a.txt": data}
    edits = {"problem/a.txt": "edited data"}
    artifacts.put_bytes(b"edited data")
    protected = t._protected(set(goal_files))
    validate_edits(edits, editable=t.editable(), protected=protected, max_edit_bytes=262_144)
    stage, overlay, warnings = plan_tool_stage(
        goal_files=goal_files,
        tool_files=t._own_files(),
        editable=t.editable(),
        protected=protected,
        private_paths=set(),
        parents=[
            ParentRef(
                candidate_id="c1",
                run_id="r",
                tool_identity="T",
                overlay={"problem/run.py": replacement},
            )
        ],
        receiving_identity="T",
        edits=edits,
        request_json=None,
    )
    workspace = tmp_path / "workspace"
    materialize(stage, workspace, artifacts)
    assert (workspace / "problem/run.py").read_bytes() == b"original script"
    assert (workspace / "problem/a.txt").read_text() == "edited data"
    assert (workspace / "parents/c1/problem/run.py").read_bytes() == b"replaced script"
    assert set(overlay) == {"problem/a.txt"}
    assert len(warnings) == 1
    assert "problem/run.py" in warnings[0]


def test_folded_script_edit_is_refused_and_original_unchanged(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*"])
    original = b"original script"
    goal_files = {"problem/run.py": artifacts.put_bytes(original)}
    protected = tool._protected(set(goal_files))
    workspace = tmp_path / "workspace"

    def stage_edits(edits: dict[str, str]) -> None:
        validate_edits(edits, editable=tool.editable(), protected=protected, max_edit_bytes=262_144)
        put_edits(edits, artifacts)
        stage, _, _ = plan_tool_stage(
            goal_files=goal_files,
            tool_files=tool._own_files(),
            editable=tool.editable(),
            protected=protected,
            private_paths=set(),
            parents=[],
            receiving_identity="T",
            edits=edits,
            request_json=None,
        )
        materialize(stage, workspace, artifacts)

    stage_edits({})
    with pytest.raises(SoftError, match=r"problem/RUN\.py: protected"):
        stage_edits({"problem/RUN.py": "secret-edit-sentinel"})
    assert (workspace / "problem/run.py").read_bytes() == original
    assert [path.name for path in (workspace / "problem").iterdir()] == ["run.py"]


@pytest.mark.parametrize(
    ("stored", "argument", "interpreter"),
    [
        ("problem/run.py", "problem/RUN.py", False),
        ("problem/cafe\u0301.py", "problem/caf\u00e9.py", False),
        ("problem/caf\u00e9.py", "problem/cafe\u0301.py", False),
        ("env/python", "env/PYTHON", True),
    ],
)
def test_folded_command_path_protects_root_spelling(
    stored: str, argument: str, interpreter: bool
) -> None:
    tool = ScriptTool(
        command=["python"] if interpreter else ["python", argument],
        python=argument if interpreter else None,
        editable=["**/*"],
    )
    protected = tool._protected({stored, "data.txt"})
    assert protected == frozenset({stored})
    with pytest.raises(SoftError, match="protected") as exc:
        validate_edits(
            {stored: "secret-edit-sentinel"},
            editable=tool.editable(),
            protected=protected,
            max_edit_bytes=100,
        )
    assert str(exc.value) == f"{stored}: protected"
    assert "secret-edit-sentinel" not in str(exc.value)
    stage, overlay, warnings = plan_tool_stage(
        goal_files={stored: HR},
        tool_files={},
        editable=tool.editable(),
        protected=protected,
        private_paths=set(),
        parents=[ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay={stored: HR})],
        receiving_identity="T",
        edits={},
        request_json=None,
    )
    assert stage.root == {stored: HR}
    assert overlay == {}
    assert warnings == [f"base parent c1 not overlaid: paths not editable here; paths: {stored}"]


@pytest.mark.parametrize("status", [RunStatus.OK, RunStatus.TIMEOUT, RunStatus.ERROR])
def test_collect_uses_result_rules(tmp_path: Path, status: RunStatus) -> None:
    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/result.json").write_text('{"data": {"loss": 0.25}, "usage": {"usd": 0.1}}')
    now = datetime(2026, 10, 18, tzinfo=UTC)
    run = CompletedRun(
        returncode=0 if status == RunStatus.OK else 1,
        status=status,
        stdout_path=tmp_path / "swarm/stdout.log",
        stderr_path=tmp_path / "swarm/stderr.log",
        started_at=now,
        ended_at=now,
    )
    output = ScriptTool(command=["x"]).collect(ToolRequest(), tmp_path, run)
    assert output.status == status
    assert output.data == ({} if status == RunStatus.ERROR else {"loss": 0.25})
    assert output.partial == (status == RunStatus.TIMEOUT)
    assert output.usage.usd == 0.1
    assert output.started_at == now
    assert output.ended_at == now


def test_extra_actions() -> None:
    names = [
        (a.name, a.counts_as, a.input_model) for a in ScriptTool(command=["x"]).extra_actions()
    ]
    assert names == [
        ("list_files", "board_read", ListFilesArgs),
        ("read_file", "board_read", ReadFileArgs),
    ]


async def test_list_and_read_files() -> None:
    view = FakeView(
        goal_manifest={"problem/a.txt": HA, "problem/run.py": HR},
        candidates={"c1": cand(overlay={"problem/a.txt": HA2, "problem/new.txt": HN})},
        blobs={HA: b"a1", HA2: b"a2", HR: b"run", HN: b"\xff\xfe" + SECRET.encode()},
    )
    context: FileViewContext = view
    list_files, read_file = ScriptTool(command=["x"]).extra_actions()
    assert await list_files.handler(context, ListFilesArgs()) == {
        "files": ["problem/a.txt", "problem/run.py"]
    }
    assert await list_files.handler(context, ListFilesArgs(candidate="c1")) == {
        "files": ["problem/a.txt", "problem/new.txt", "problem/run.py"]
    }
    assert view.reads == []
    assert await read_file.handler(context, ReadFileArgs(path="problem/a.txt", candidate="c1")) == {
        "path": "problem/a.txt",
        "content": "a2",
    }
    assert await read_file.handler(context, ReadFileArgs(path="problem/a.txt")) == {
        "path": "problem/a.txt",
        "content": "a1",
    }
    with pytest.raises(SoftError, match="no such file: nope"):
        await read_file.handler(context, ReadFileArgs(path="nope"))
    with pytest.raises(SoftError, match="not UTF-8 text") as caught:
        await read_file.handler(context, ReadFileArgs(path="problem/new.txt", candidate="c1"))
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert view.reads == [HA2, HA, HN]


async def test_file_actions_respect_candidate_view() -> None:
    view = FakeView(goal_manifest={}, candidates={}, blobs={HA: b"private data"})
    list_files, read_file = ScriptTool(command=["x"]).extra_actions()
    assert await list_files.handler(view, ListFilesArgs()) == {"files": []}
    with pytest.raises(SoftError, match="no such candidate: outside-view"):
        await list_files.handler(view, ListFilesArgs(candidate="outside-view"))
    with pytest.raises(SoftError, match="no such candidate: outside-view"):
        await read_file.handler(view, ReadFileArgs(path="private.txt", candidate="outside-view"))
    for path in ("../private.txt", "/private.txt", "swarm/result.json", HA):
        with pytest.raises(SoftError, match="no such file"):
            await read_file.handler(view, ReadFileArgs(path=path))
    assert view.reads == []


def test_catalog_refuses_settable_script_fields() -> None:
    entry = {"name": "s", "kind": "tool", "ref": "swarm_arena:ScriptTool", "description": "d"}
    for field in ("command", "python", "files", "env", "editable"):
        with pytest.raises(ValidationError, match=f"human-only field '{field}'"):
            PlannerCatalog.model_validate(
                {"entries": [{**entry, "settable": {field: {"type": "string"}}}]}
            )
    catalog = PlannerCatalog.model_validate(
        {"entries": [{**entry, "settable": {"timeout_s": {"type": "number"}}}]}
    )
    assert catalog.entries[0].settable == {"timeout_s": {"type": "number"}}


def test_init_does_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("constructor performed I/O")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "glob", forbidden)
        patch.setattr(Path, "exists", forbidden)
        patch.setattr("builtins.open", forbidden)
        t = ScriptTool(command=["python", "missing.py"], files=["nowhere/*"])
        assert not t.config.files.snapshotted
