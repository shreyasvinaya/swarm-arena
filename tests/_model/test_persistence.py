"""Protect stored model versions and portable exception diagnostics."""

import copy
import pickle
from collections import UserDict
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import Field, ValidationError

from swarm_arena import AgentSpec, Plan, Ref, RunConfig
from swarm_arena._errors import ConfigError
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.ref import Configurable

SECRET = "56214f94-ae2e-498d-b87f-7104dbe8b0d9"


@pytest.mark.parametrize("model", [Plan, RunConfig])
@pytest.mark.parametrize("kind", ["encoding", "io", "syntax"])
def test_toml_read_errors_are_safe_and_located(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: type[Plan] | type[RunConfig], kind: str
) -> None:
    path = tmp_path / "settings.toml"
    if kind == "encoding":
        path.write_bytes(SECRET.encode() + b"\xff")
    elif kind == "io":

        def fail(*args: Any, **kwargs: Any) -> str:
            raise OSError(SECRET)

        monkeypatch.setattr(Path, "read_text", fail)
    else:
        path.write_text(f'good = "{SECRET}"\nbad = @\n')
    with pytest.raises(ConfigError) as caught:
        model.load(path)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    if kind == "syntax":
        assert "line 2, column 7" in str(caught.value)


@pytest.mark.parametrize("model", [Plan, RunConfig])
def test_toml_text_errors_preserve_safe_coordinates(model: type[Plan] | type[RunConfig]) -> None:
    def action() -> None:
        if model is Plan:
            Plan.loads(f'good = "{SECRET}"\nbad = @\n', base_dir=Path.cwd())
        else:
            RunConfig.loads(f'good = "{SECRET}"\nbad = @\n')

    with pytest.raises(ConfigError, match="line 2, column 7") as caught:
        action()
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize("mapping", [MappingProxyType, UserDict])
@pytest.mark.parametrize("version", [99, -3, True])
def test_mapping_inputs_cannot_skip_stored_version_checks(mapping: Any, version: Any) -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        RunConfig.model_validate(mapping({"schema_version": version}))


@pytest.mark.parametrize("name", ["", "x\ny", "human:alice", "x\x00y", "x\x7fy"])
def test_reference_names_are_printable_identifiers(name: str) -> None:
    with pytest.raises(ValidationError):
        Ref(path="builtins:str", name=name)
    assert Ref(path="builtins:str", name="Score A: quality").name == "Score A: quality"


@pytest.mark.parametrize("kind", ["local", "main", "lambda"])
@pytest.mark.parametrize("outer", [False, True])
def test_nonimportable_components_name_the_field_and_reason(kind: str, outer: bool) -> None:
    class Local(Configurable):
        pass

    component = Local
    if kind != "local":
        component = type("Component", (Configurable,), {"__module__": "__main__"})
    if kind == "lambda":
        component.__qualname__ = "<lambda>"
    data: dict[str, Any] = {
        "name": "a",
        "rationale": "r",
        "tool": component,
        "manager": Ref(path="m:M"),
    }

    def action() -> None:
        if outer:
            Plan(
                goal={"id": "g", "namespace": "n"},
                teams=[{"name": "t", "rationale": "r", "agents": [data]}],
                judges=[],
            )
        else:
            AgentSpec(**data)

    with pytest.raises(ValidationError) as caught:
        action()
    assert caught.value.errors()[0]["loc"] == (
        ("teams", 0, "agents", 0, "tool") if outer else ("tool",)
    )
    assert {"local": "local object", "main": "__main__", "lambda": "lambda"}[kind] in str(
        caught.value
    )


@pytest.mark.parametrize("transport", [pickle.loads, copy.deepcopy])
@pytest.mark.parametrize("case", ["bound", "pattern", "literal", "json", "custom"])
def test_sanitized_errors_survive_pickle_and_deepcopy(transport: Any, case: str) -> None:
    class Model(FrozenModel):
        count: int = Field(ge=0)
        name: str = Field(pattern="^[a-z]+$")

    from swarm_arena._model.safe import _sanitize_validation_error

    def action() -> None:
        if case == "json":
            Model.model_validate_json(SECRET)
        elif case == "custom":
            raise _sanitize_validation_error(Model, TypeError(SECRET))
        elif case == "literal":
            from swarm_arena._model.tooling import ToolOutput

            ToolOutput(status=SECRET)
        else:
            Model(count=-1 if case == "bound" else 0, name=SECRET)

    with pytest.raises(ValidationError) as caught:
        action()
    error = caught.value
    restored = transport(pickle.dumps(error)) if transport is pickle.loads else transport(error)
    assert restored.errors() == error.errors()
    assert str(restored) == str(error)
    assert_secret_absent(restored, SECRET, sanitized=True)


def test_plan_directory_resolution_errors_are_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "plan.toml"
    path.write_text('teams=[]\njudges=[]\n[goal]\nid="g"\nnamespace="n"')

    def fail(*args: Any, **kwargs: Any) -> Path:
        raise OSError(SECRET)

    monkeypatch.setattr(Path, "resolve", fail)
    with pytest.raises(ConfigError) as caught:
        Plan.load(path)
    assert_secret_absent(caught.value, SECRET, sanitized=True)
