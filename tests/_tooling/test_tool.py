import doctest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
from _fakes.tools import BadCollectTool, EchoTool, PyInputs, PyTool
from pydantic import ValidationError

from swarm_arena._model.base import JsonValue
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._tooling import tool as tool_module
from swarm_arena._tooling.tool import Action, Tool, cap_timeout, effective_tool_timeout, tool_grace
from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest


def test_tool_metadata_examples_execute() -> None:
    examples = doctest.DocTestFinder().find(Tool, globs=vars(tool_module))
    methods = (
        "preflight",
        "is_deterministic",
        "editable",
        "estimate",
        "providers",
        "fingerprint",
        "extra_actions",
        "get_input_model",
        "get_description",
    )
    for method in methods:
        calls = [
            example
            for test in examples
            for example in test.examples
            if f".{method}(" in example.source
        ]
        assert calls, f"No executable example for {method}"
        assert all(not example.options.get(doctest.SKIP) for example in calls), method
    runner = doctest.DocTestRunner()
    for test in examples:
        runner.run(test)
    assert runner.summarize().failed == 0


# tests/_tooling/test_tool.py
def test_tool_defaults() -> None:
    t = PyTool(PyTool.Config(code="pass"))
    assert (
        t.is_deterministic(ToolRequest()),
        tuple(t.editable()),
        t.estimate(ToolRequest()),
        t.providers(),
        tuple(t.extra_actions()),
        t.preflight(),
    ) == (True, (), None, frozenset(), (), None)
    assert t.get_input_model() is PyInputs
    assert t.get_description() == "Runs Python code."
    assert t._own_files() == {}
    assert t._protected({"a"}) == frozenset()


def test_default_fingerprint() -> None:
    t = PyTool(PyTool.Config(code="pass"))
    assert t.fingerprint(ToolOutput(status=RunStatus.OK, data={"b": 1, "a": 2})) == sha256_hex(
        b'{"a":2,"b":1}'
    )


def test_abstract_methods_required() -> None:
    class Half(Tool):
        name = "h"
        description = "h"
        input_model = PyInputs

    with pytest.raises(TypeError, match="abstract"):
        cast(Any, Half)()


@pytest.mark.parametrize(
    ("config_s", "round_s", "expected"),
    [
        (None, None, 3600.0),
        (1200.0, None, 1200.0),
        (1200.0, 2000.0, 2000.0),
        (None, 5000.0, 3600.0),
        (9000.0, None, 3600.0),
    ],
)
def test_effective_tool_timeout(
    config_s: float | None, round_s: float | None, expected: float
) -> None:
    tool = PyTool(PyTool.Config(code="pass", timeout_s=config_s))
    assert (
        effective_tool_timeout(tool, round_timeout_s=round_s, limit_s=3600.0) == expected
    )  # spec §6.1
    assert (
        effective_tool_timeout(EchoTool(), round_timeout_s=None, limit_s=60.0) == 60.0
    )  # no timeout_s field


def test_tool_grace() -> None:
    assert tool_grace(PyTool(PyTool.Config(code="pass", grace_s=2.5))) == 2.5
    assert tool_grace(EchoTool()) == 10.0


async def test_action_is_typed_frozen_and_keyword_only() -> None:
    async def handler(context: Any, inputs: PyInputs) -> JsonValue:
        return {"n": inputs.n, "context": context}

    action: Action[PyInputs] = Action(
        name="read", description="Read.", input_model=PyInputs, handler=handler
    )
    assert action.counts_as == "none"
    assert await action.handler("ctx", action.input_model(n=4)) == {"n": 4, "context": "ctx"}
    assert (
        Action(
            name="read",
            description="Read.",
            input_model=PyInputs,
            handler=handler,
            counts_as="board_read",
        ).counts_as
        == "board_read"
    )
    with pytest.raises(FrozenInstanceError, match="cannot assign"):
        cast(Any, action).name = "other"
    with pytest.raises(TypeError, match="positional"):
        cast(Any, Action)("read", "Read.", PyInputs, handler)


@pytest.mark.parametrize(
    ("configured", "expected"), [(None, 60.0), (12.0, 12.0), (70.0, 60.0), (0.0, 0.0)]
)
def test_cap_timeout(configured: float | None, expected: float) -> None:
    assert cap_timeout(configured, 60.0) == expected


def test_fakes_contract(tmp_path: Path) -> None:
    from _tooling.test_result_file import run, write_result

    tool = PyTool(PyTool.Config(code="pass", timeout_s=1.0, grace_s=2.0))
    command = tool.command(ToolRequest(), tmp_path)
    assert (command.argv, command.timeout_s, command.grace_s) == (
        ["python", "-c", "pass"],
        1.0,
        2.0,
    )
    write_result(tmp_path, {"data": {"n": 1}})
    completed = run(tmp_path, RunStatus.OK, 0)
    assert tool.collect(ToolRequest(), tmp_path, completed).data == {"n": 1}
    with pytest.raises(RuntimeError, match="collect broke"):
        BadCollectTool(tool.config).collect(ToolRequest(), tmp_path, completed)
    assert EchoTool().command(ToolRequest(), tmp_path).argv == ["echo", ""]
    assert EchoTool().collect(ToolRequest(), tmp_path, completed).data == {"text": ""}
    for cls in (PyTool.Config, EchoTool.Config):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            cls.model_validate({"unknown": "value"})


@pytest.mark.parametrize("overrides", [{}, {"timeout_s": 12}])
def test_effective_tool_ref_preserves_raw_config(overrides: dict[str, Any]) -> None:
    from swarm_arena import Ref
    from swarm_arena._tooling.tool import effective_tool_ref

    ref = Ref(
        path="swarm_arena:ScriptTool",
        config={"command": ["run"], "params": {"n": {"type": "integer", "default": 1}}},
    )
    effective = effective_tool_ref(ref, overrides)
    assert effective.config == {**ref.config, **overrides}
    assert "env" not in effective.config
    if overrides:
        assert type(effective.config["timeout_s"]) is int
    effective.config["command"].append("other")
    assert ref.config["command"] == ["run"]
    assert overrides == ({} if not overrides else {"timeout_s": 12})


@pytest.mark.parametrize(
    "overrides",
    [{"timeout_s": "SECRET_VALUE"}, {"unknown": "SECRET_VALUE"}, {"params": {"bad": float("nan")}}],
)
def test_effective_tool_ref_validates_before_snapshots(overrides: dict[str, Any]) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena import Ref
    from swarm_arena._tooling.tool import effective_tool_ref

    ref = Ref(path="swarm_arena:ScriptTool", config={"command": ["run"], "files": ["missing.py"]})
    with pytest.raises(ValueError, match=r"(invalid|forbidden|parsing|finite JSON)") as error:
        effective_tool_ref(ref, overrides)
    assert_secret_absent(error.value, "SECRET_VALUE")
    assert "missing.py" not in str(error.value)


@pytest.mark.parametrize("alias", ["env_alias", "env_path", "env_choice", "env_choice_path", "env"])
def test_effective_tool_ref_refuses_all_human_aliases(alias: str) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena import Ref
    from swarm_arena._tooling.tool import effective_tool_ref

    secret = "SECRET_VALUE"
    with pytest.raises(ValueError, match="human-only field env") as error:
        effective_tool_ref(Ref(path="_fakes.tools:AliasedEnvTool"), {alias: secret})
    assert_secret_absent(error.value, "SECRET_VALUE")


def test_effective_tool_ref_validates_empty_overrides_and_role() -> None:
    from swarm_arena import Ref
    from swarm_arena._tooling.tool import effective_tool_ref

    with pytest.raises(ValueError, match="timeout_s"):
        effective_tool_ref(Ref(path="swarm_arena:ScriptTool", config={"timeout_s": "invalid"}), {})
    with pytest.raises(TypeError, match="Tool"):
        effective_tool_ref(Ref(path="_fakes.tools:Widget"), {})
