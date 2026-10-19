import traceback
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from _fakes import tools as fake_tools
from _fakes.tools import add, gradient_descent, make_job
from pydantic import BeforeValidator, ValidationError
from pydantic_core import PydanticCustomError

import swarm_arena
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._model.ref import Ref
from swarm_arena._model.tooling import Command, RunStatus, ToolRequest
from swarm_arena._tooling.function_tool import FunctionTool, function_input_model, import_root
from swarm_arena._tooling.result_file import request_json

SECRET = "SECRET_SENTINEL_TASK18"


def function_job(tmp_path: Path, tool: FunctionTool, req: ToolRequest) -> Job:
    return make_job(
        id="c-function",
        kind="tool",
        command=None,
        tool=Ref.of(tool),
        request=req,
        stage=Job.Stage(request_json=request_json(req)),
        artifact_root=str(tmp_path / "artifacts"),
        timeout_s=30.0,
    )


def assert_secret_absent(exc: BaseException) -> None:
    surfaces = [
        str(exc),
        repr(exc),
        repr(exc.args),
        repr(getattr(exc, "__notes__", [])),
        repr(exc.__cause__),
        "".join(traceback.format_exception(exc)),
    ]
    if isinstance(exc, ValidationError):
        surfaces.extend([repr(exc.errors()), exc.json()])
    assert all(SECRET not in value for value in surfaces)


def test_init_forms_and_flags() -> None:
    assert (
        FunctionTool("_fakes.tools:add").config
        == FunctionTool(add).config
        == FunctionTool(FunctionTool.Config(function="_fakes.tools:add")).config
    )
    with pytest.raises(TypeError, match="lambda"):
        FunctionTool(lambda: {})
    t = FunctionTool("_fakes.tools:add", deterministic=False, providers=["openai"])
    assert not t.is_deterministic(ToolRequest())
    assert t.providers() == frozenset({"openai"})
    assert FunctionTool(add).is_deterministic(ToolRequest())
    assert FunctionTool(add).providers() == frozenset()
    with pytest.raises(TypeError, match="config or fields"):
        FunctionTool(FunctionTool.Config(function="_fakes.tools:add"), deterministic=False)


def test_init_imports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: str) -> object:
        raise AssertionError("constructor imported referenced code")

    with monkeypatch.context() as patch:
        patch.setattr("swarm_arena._tooling.function_tool.load_object", fail)
        t = FunctionTool("no_such_mod:f")
    with pytest.raises(ValueError, match="cannot import no_such_mod:f"):
        t.get_input_model()


def test_input_model_and_description() -> None:
    m = FunctionTool(add).get_input_model()
    assert m.model_validate({"a": 2}).model_dump() == {"a": 2, "b": 1}
    assert FunctionTool(add).get_description() == "Add two numbers."
    with pytest.raises(ValidationError):
        m.model_validate({"a": 2, "c": 1})
    assert FunctionTool(fake_tools.not_a_dict).get_description() == FunctionTool.description
    assert FunctionTool.input_model.model_config["frozen"]
    assert FunctionTool.input_model.model_fields == {}
    assert FunctionTool(add).get_input_model().model_config["hide_input_in_errors"]
    FunctionTool(add).preflight()


@pytest.mark.parametrize(
    "fn",
    [
        "_fakes.tools:coro",
        "_fakes.tools:star",
        "_fakes.tools:unannotated",
        "_fakes.tools:positional_only",
        "_fakes.tools:star_keywords",
        "_fakes.tools:GOAL",
    ],
)
def test_unsupported_signatures(fn: str) -> None:
    with pytest.raises(TypeError, match="FunctionTool"):
        FunctionTool(fn).preflight()


def test_keyword_only_and_resolved_annotations() -> None:
    model = function_input_model(fake_tools.keyword_only)
    assert model().model_dump() == {"value": 2}
    assert model.model_validate({"value": "3"}).model_dump() == {"value": 3}


def test_command_argv() -> None:
    cmd = FunctionTool(add).command(ToolRequest(), Path("/ws"))
    tests_dir = str(Path(fake_tools.__file__).resolve().parents[1])
    assert cmd == Command(
        argv=[
            "python",
            "-P",
            "-m",
            "swarm_arena.run_function",
            "--import-root",
            tests_dir,
            "_fakes.tools:add",
        ]
    )
    assert import_root("swarm_arena._model.canonical") == str(
        Path(swarm_arena.__file__).resolve().parents[1]
    )
    assert import_root("swarm_arena") == str(Path(swarm_arena.__file__).resolve().parents[1])
    timed = FunctionTool(add, timeout_s=3, grace_s=0.5).command(ToolRequest(), Path("/ws"))
    assert (timed.timeout_s, timed.grace_s) == (3, 0.5)


@pytest.mark.parametrize(
    "fields", [{"providers": [SECRET, {}]}, {"deterministic": SECRET}, {"unknown": SECRET}]
)
def test_config_errors_hide_values(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="FunctionTool") as caught:
        FunctionTool(add, **fields)
    assert_secret_absent(caught.value)
    with pytest.raises(ValidationError) as direct:
        FunctionTool.Config(function="_fakes.tools:add", **fields)
    assert_secret_absent(direct.value)


def test_input_errors_hide_values_and_mapping_keys() -> None:
    model = function_input_model(fake_tools.nested_input)
    with pytest.raises(ValidationError) as caught:
        model.model_validate({"values": {SECRET: SECRET}})
    assert_secret_absent(caught.value)
    assert caught.value.errors()[0]["loc"] == ("values", "<…>")
    with pytest.raises(ValidationError) as extra:
        model.model_validate({"values": {}, SECRET: SECRET})
    assert_secret_absent(extra.value)


def test_import_and_annotation_failures_hide_values(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: str) -> object:
        raise RuntimeError(SECRET)

    with monkeypatch.context() as patch:
        patch.setattr("swarm_arena._tooling.function_tool.load_object", fail)
        with pytest.raises(ValueError, match="cannot import") as caught:
            FunctionTool(add).get_input_model()
    assert_secret_absent(caught.value)
    monkeypatch.setattr(add, "__annotations__", {"a": SECRET})
    with pytest.raises(TypeError, match="FunctionTool") as annotation:
        function_input_model(add)
    assert_secret_absent(annotation.value)


async def test_runs_on_local_executor(local_executor: LocalExecutor, tmp_path: Path) -> None:
    req = ToolRequest(inputs={"lr": 0.001, "steps": 2000})
    out = await local_executor.submit(function_job(tmp_path, FunctionTool(gradient_descent), req))
    assert out.status is RunStatus.OK
    assert set(out.data) == {"x", "y", "f"}
    assert out.data["f"] < 12.5


@pytest.mark.parametrize("custom", [False, True])
def test_custom_and_literal_validation_errors_hide_values(
    monkeypatch: pytest.MonkeyPatch, custom: bool
) -> None:
    def fail(value: Any) -> Any:
        raise PydanticCustomError(SECRET, SECRET)

    annotation = (
        Annotated[int, BeforeValidator(fail)] if custom else Literal["SECRET_SENTINEL_TASK18"]
    )
    monkeypatch.setattr(add, "__annotations__", {"a": annotation, "b": int})
    model = function_input_model(add)
    with pytest.raises(ValidationError) as caught:
        model.model_validate({"a": 1 if custom else "wrong"})
    assert_secret_absent(caught.value)


def test_import_root_rejects_modules_without_a_file() -> None:
    with pytest.raises(ValueError, match="cannot determine import root for sys"):
        import_root("sys")


def test_direct_model_json_errors_hide_input_text() -> None:
    model = FunctionTool(add).get_input_model()
    for model_cls in (model, FunctionTool.Config):
        with pytest.raises(ValidationError) as caught:
            model_cls.model_validate_json(f"{{{SECRET}")
        assert SECRET not in str(caught.value)
        assert SECRET not in "".join(traceback.format_exception(caught.value))
