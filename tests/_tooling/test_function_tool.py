import importlib
import importlib.machinery
import os
import py_compile
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from _fakes import tools as fake_tools
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import add, gradient_descent, make_job
from pydantic import BaseModel, BeforeValidator, ValidationError
from pydantic_core import PydanticCustomError

import swarm_arena
from swarm_arena._exec.executor import Job
from swarm_arena._exec.local import LocalExecutor
from swarm_arena._model.ref import Ref
from swarm_arena._model.tooling import Command, RunStatus, ToolRequest
from swarm_arena._tooling.function_tool import FunctionTool, function_input_model, import_root
from swarm_arena._tooling.result_file import request_json

SECRET = "b7c9cdbd45ac6961fae12985afab51de"


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
        patch.setattr("swarm_arena._model.ref.importlib.import_module", fail)
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
        "_fakes.tools:RecordingExecutor",
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
    with pytest.raises(ValidationError) as caught:
        FunctionTool(add, **fields)
    assert_secret_absent(caught.value, SECRET)
    with pytest.raises(ValidationError) as direct:
        FunctionTool.Config(function="_fakes.tools:add", **fields)
    assert_secret_absent(direct.value, SECRET)


def test_input_errors_hide_values_and_mapping_keys() -> None:
    model = function_input_model(fake_tools.nested_input)
    with pytest.raises(ValidationError) as caught:
        model.model_validate({"values": {SECRET: SECRET}})
    assert_secret_absent(caught.value, SECRET)
    assert caught.value.errors()[0]["loc"] == ("values", "<…>")
    with pytest.raises(ValidationError) as extra:
        model.model_validate({"values": {}, SECRET: SECRET})
    assert_secret_absent(extra.value, SECRET)


def test_import_and_annotation_failures_hide_values(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: str) -> object:
        raise RuntimeError(SECRET)

    with monkeypatch.context() as patch:
        patch.setattr("swarm_arena._model.ref.importlib.import_module", fail)
        with pytest.raises(ValueError, match="cannot import") as caught:
            FunctionTool("sentinel_import:solve").get_input_model()
    assert_secret_absent(caught.value, SECRET)
    monkeypatch.setattr(add, "__annotations__", {"a": SECRET})
    with pytest.raises(TypeError, match="FunctionTool") as annotation:
        function_input_model(add)
    assert_secret_absent(annotation.value, SECRET)


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
        Annotated[int, BeforeValidator(fail)]
        if custom
        else Literal["b7c9cdbd45ac6961fae12985afab51de"]
    )
    monkeypatch.setattr(add, "__annotations__", {"a": annotation, "b": int})
    model = function_input_model(add)
    with pytest.raises(ValidationError) as caught:
        model.model_validate({"a": 1 if custom else "wrong"})
    assert_secret_absent(caught.value, SECRET)


def test_import_root_rejects_modules_without_a_file() -> None:
    with pytest.raises(ValueError, match="cannot determine import root for sys"):
        import_root("sys")


@pytest.mark.parametrize(
    "model_cls",
    [FunctionTool(add).get_input_model(), FunctionTool.Config, FunctionTool.input_model],
)
def test_direct_model_json_errors_hide_input_text(model_cls: type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as caught:
        model_cls.model_validate_json(f"{{{SECRET}")
    assert caught.value.errors()[0]["type"] == "json_invalid"
    assert model_cls.model_config["hide_input_in_errors"]
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize(
    "function", ["_fakes.tools:add", add, FunctionTool.Config(function="_fakes.tools:add")]
)
def test_duplicate_function_keyword_is_rejected(
    function: str | Callable[..., Any] | FunctionTool.Config,
) -> None:
    with pytest.raises(TypeError, match="function") as caught:
        FunctionTool(function, function=SECRET)
    assert_secret_absent(caught.value, SECRET)


def test_command_uses_configured_module_for_reexport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "task18_reexport.py").write_text("from _fakes.tools import add\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    tool = FunctionTool("task18_reexport:add")
    try:
        tool.preflight()
        assert sys.modules["task18_reexport"].add.__module__ == "_fakes.tools"
        assert import_root("_fakes.tools") != str(tmp_path)
        command = tool.command(ToolRequest(), tmp_path)
        assert command.argv[-3:] == ["--import-root", str(tmp_path), "task18_reexport:add"]
    finally:
        sys.modules.pop("task18_reexport", None)


def test_preflight_rejects_stdlib_module_alias() -> None:
    assert os.path.__spec__ is not None
    real_name = os.path.__spec__.name
    expected = (
        f"FunctionTool os.path:basename: os.path is an alias of {real_name}; "
        "use the defining module"
    )
    with pytest.raises(TypeError, match="is an alias of") as caught:
        FunctionTool("os.path:basename").preflight()
    assert str(caught.value) == expected
    assert_secret_absent(caught.value, SECRET)
    assert import_root("os.path") == str(Path(os.path.__file__).resolve().parent)


def test_preflight_rejects_package_module_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "task18_alias"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import sys\nimport task18_defining\nsys.modules[__name__] = task18_defining\n"
    )
    (tmp_path / "task18_defining.py").write_text(
        f'PRIVATE = "{SECRET}"\ndef solve() -> dict[str, int]:\n    return {{"answer": 42}}\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(TypeError, match="is an alias of") as caught:
            FunctionTool("task18_alias:solve").preflight()
        assert str(caught.value) == (
            "FunctionTool task18_alias:solve: task18_alias is an alias of "
            "task18_defining; use the defining module"
        )
        assert_secret_absent(caught.value, SECRET)
        assert import_root("task18_alias") == str(tmp_path)
        FunctionTool("task18_defining:solve").preflight()
    finally:
        sys.modules.pop("task18_alias", None)
        sys.modules.pop("task18_defining", None)


@pytest.mark.parametrize("exception", ["RuntimeError", "SystemExit"])
def test_preflight_import_failure_hides_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: str
) -> None:
    (tmp_path / "task18_import_error.py").write_text(f'raise {exception}("{SECRET}")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="cannot import task18_import_error:solve") as caught:
        FunctionTool("task18_import_error:solve").preflight()
    assert str(caught.value) == "cannot import task18_import_error:solve"
    assert_secret_absent(caught.value, SECRET)


def _solve(x: int) -> dict[str, int]:
    """Return the input as data."""
    return {"x": x}


class _NotAModule:  # sys.modules may hold any object; the stdlib does (typing.io on 3.11)
    solve = staticmethod(_solve)


def _specless_module() -> types.ModuleType:
    module = types.ModuleType("task18_specless")  # a real module, but made without a spec
    setattr(module, "solve", _solve)  # noqa: B010 (a module attribute the checker cannot see)
    return module


@pytest.mark.parametrize(
    ("name", "entry"), [("task18_not_a_module", _NotAModule), ("task18_specless", None)]
)
def test_sys_modules_entry_without_a_spec_is_refused(
    monkeypatch: pytest.MonkeyPatch, name: str, entry: object
) -> None:
    monkeypatch.setitem(sys.modules, name, _specless_module() if entry is None else entry)
    tool = FunctionTool(f"{name}:solve")
    expected = f"FunctionTool {name}:solve: {name} has no import spec; use the defining module"
    for call in (tool.preflight, tool.get_description, lambda: tool.command(ToolRequest(), Path())):
        with pytest.raises(TypeError) as caught:
            call()
        assert str(caught.value) == expected
        assert_secret_absent(caught.value, SECRET)


def test_import_root_finds_a_sourceless_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "task18_sourceless"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("def solve(x: int) -> dict[str, int]:\n    return {'x': x}\n")
    py_compile.compile(str(source), cfile=str(package / "__init__.pyc"), doraise=True)
    source.unlink()  # only __init__.pyc is left, as in a sourceless install
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module("task18_sourceless")
        assert Path(module.__file__ or "").name == "__init__.pyc"
        assert import_root("task18_sourceless") == str(tmp_path.resolve())
        command = FunctionTool("task18_sourceless:solve").command(ToolRequest(), tmp_path)
        assert command.argv[-3:] == [
            "--import-root",
            str(tmp_path.resolve()),
            "task18_sourceless:solve",
        ]
    finally:
        sys.modules.pop("task18_sourceless", None)


def test_import_root_finds_an_extension_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # a compiled __init__ (mypyc builds) is a package by its spec, not its file name
    package = tmp_path / "task18_ext"
    package.mkdir()
    init = package / "__init__.cpython-311-darwin.so"
    init.touch()
    module = types.ModuleType("task18_ext")
    module.__spec__ = importlib.machinery.ModuleSpec(
        "task18_ext", None, origin=str(init), is_package=True
    )
    module.__file__ = str(init)
    monkeypatch.setitem(sys.modules, "task18_ext", module)
    assert import_root("task18_ext") == str(tmp_path.resolve())


@pytest.mark.parametrize(
    "fields",
    [
        {"timeout_s": 0},
        {"timeout_s": -1},
        {"timeout_s": float("nan")},
        {"grace_s": -1},
        {"grace_s": float("inf")},
        {"env": ["INVALID=KEY"]},
    ],
)
def test_function_limits_and_env_are_validated_at_construction(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FunctionTool(add, **fields)


def test_function_env_allowlist_and_human_only() -> None:
    tool = FunctionTool(add, env=["API_KEY"])
    assert tool.command(ToolRequest(), Path(".")).env == ["API_KEY"]
    assert "env" in tool._human_only


@pytest.mark.parametrize("status", [RunStatus.OK, RunStatus.TIMEOUT, RunStatus.CANCELLED])
def test_function_collect_is_data_only(tmp_path: Path, status: RunStatus) -> None:
    from datetime import UTC, datetime

    from swarm_arena.tool import CompletedRun

    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/result.json").write_text('{"data":{"answer":3},"files":["missing.txt"]}')
    now = datetime.now(UTC)
    run = CompletedRun(
        returncode=0,
        status=status,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        started_at=now,
        ended_at=now,
    )
    output = FunctionTool(add).collect(ToolRequest(), tmp_path, run)
    assert output.data == {"answer": 3}
    assert output.files == []
    assert output.status == status


async def test_function_environment_reaches_subprocess(
    tmp_path: Path, local_executor: LocalExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "environment_function.py").write_text(
        "import os\ndef solve() -> dict[str, bool]:\n"
        '    return {"allowed": os.environ.get("ALLOW_KEY") == "yes", '
        '"blocked": "BLOCK_KEY" in os.environ}\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ALLOW_KEY", "yes")
    monkeypatch.setenv("BLOCK_KEY", "hidden")
    output = await local_executor.submit(
        function_job(
            tmp_path, FunctionTool("environment_function:solve", env=["ALLOW_KEY"]), ToolRequest()
        )
    )
    assert output.status == RunStatus.OK
    assert output.data == {"allowed": True, "blocked": False}


@pytest.mark.parametrize("status", [RunStatus.OK, RunStatus.TIMEOUT, RunStatus.CANCELLED])
def test_function_checkpoint_artifacts_are_never_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: RunStatus
) -> None:
    from datetime import UTC, datetime

    from swarm_arena._tooling import result_file
    from swarm_arena.tool import CompletedRun

    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/result.json").write_text('{"data":{"answer":7},"files":["large.dat"]}')

    def fail(*args: Any, **kwargs: Any) -> None:
        pytest.fail("FunctionTool inspected an artifact")

    monkeypatch.setattr(result_file, "_hash_regular_file", fail)
    now = datetime.now(UTC)
    completed = CompletedRun(
        status=status,
        returncode=0,
        started_at=now,
        ended_at=now,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
    )
    output = FunctionTool(add).collect(ToolRequest(), tmp_path, completed)
    assert output.data == {"answer": 7}
    assert not output.files


def test_function_environment_is_not_planner_settable() -> None:
    from swarm_arena.config import PlannerCatalog

    with pytest.raises(ValidationError, match="human-only field 'env'"):
        PlannerCatalog.model_validate(
            {
                "entries": [
                    {
                        "name": "function",
                        "kind": "tool",
                        "ref": "swarm_arena:FunctionTool",
                        "description": "Function",
                        "settable": {"env": {"type": "array", "items": {"type": "string"}}},
                    }
                ]
            }
        )


def test_import_root_uses_shared_import_error_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: str) -> object:
        raise RuntimeError(SECRET)

    monkeypatch.setattr("swarm_arena._model.ref.importlib.import_module", fail)
    with pytest.raises(ValueError, match="cannot import root_failure:__name__") as caught:
        import_root("root_failure")
    assert_secret_absent(caught.value, SECRET)
