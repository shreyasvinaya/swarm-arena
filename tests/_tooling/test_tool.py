from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
from _fakes.tools import BadCollectTool, EchoTool, PyInputs, PyTool
from pydantic import ValidationError

from swarm_arena._model.base import JsonValue
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._tooling.tool import Action, Tool, cap_timeout, effective_tool_timeout, tool_grace
from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest


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
