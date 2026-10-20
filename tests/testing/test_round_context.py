"""Exercise effective configuration and request context through the public kit."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, cast

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import RecordingExecutor
from pydantic import BaseModel, ConfigDict, Field, create_model

from swarm_arena import Ref, ScriptTool
from swarm_arena.executor import LocalExecutor
from swarm_arena.testing import check_tool, make_candidate
from swarm_arena.tool import Command, CompletedRun, RunStatus, Tool, ToolOutput, ToolRequest

SECRET = "b6822868-6b26-456a-ad7b-285a93b0c5cb"


class ContextTool(Tool):
    name = "context"
    description = "Record effective settings and every request surface."
    _human_only: ClassVar[frozenset[str]] = frozenset({"script"})

    class Config(BaseModel):
        model_config = ConfigDict(
            extra="forbid", frozen=True, hide_input_in_errors=True, populate_by_name=True
        )
        effort: int = Field(default=1, ge=1)
        label: str = "retained"
        timeout_s: float = 20.0
        grace_s: float = 0.1
        script: str = Field(default="trusted", validation_alias="script_path")

    @property
    def config(self) -> Config:
        return cast(ContextTool.Config, super().config)

    def get_input_model(self) -> type[BaseModel]:
        return create_model(
            "Inputs", n=(int, Field(default=self.config.effort, le=self.config.effort))
        )

    def editable(self) -> Sequence[str]:
        return ["problem/*.txt"] if self.config.effort >= 5 else ()

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        code = """
import json, os, pathlib, sys
request = json.loads(pathlib.Path('swarm/request.json').read_text())
edit = pathlib.Path('problem/a.txt')
data = {'staged': request, 'env_round': os.environ['SWARM_ARENA_ROUND'],
        'command': json.loads(sys.argv[1]), 'edit': edit.read_text() if edit.exists() else "absent"}
pathlib.Path(os.environ['SWARM_ARENA_RESULT']).write_text(json.dumps({'data': data}))
"""
        return Command(
            argv=[sys.executable, "-c", code, json.dumps(self._context(request))],
        )

    def _context(self, request: ToolRequest) -> dict[str, Any]:
        return {
            "round": request.round,
            "round_overrides": request.round_overrides,
            "replicate": request.replicate,
            "inputs": request.inputs,
            "effort": self.config.effort,
            "label": self.config.label,
        }

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        if run.status != RunStatus.OK:
            return ToolOutput(status=run.status)
        data = json.loads((workspace / "swarm/result.json").read_text())["data"]
        data["collect"] = self._context(request)
        return ToolOutput(status=run.status, data=data)


@pytest.mark.parametrize("as_ref", [False, True])
async def test_round_context_and_effective_config_reach_every_surface(
    as_ref: bool, local_executor: LocalExecutor
) -> None:
    tool = ContextTool({"label": "configured"})
    overrides = {"effort": 5, "timeout_s": 7.0}
    out = await check_tool(
        Ref.of(tool) if as_ref else tool,
        {"n": 4},
        round=3,
        round_overrides=MappingProxyType(overrides),
        replicate=2,
        edits={"problem/a.txt": "edited"},
        executor=local_executor,
    )
    expected = {"round": 3, "round_overrides": overrides, "replicate": 2}
    assert out.data["staged"] == {"schema_version": 1, "params": {"n": 4}, **expected}
    assert out.data["env_round"] == "3"
    assert (
        out.data["command"]
        == out.data["collect"]
        == {**expected, "inputs": {"n": 4}, "effort": 5, "label": "configured"}
    )
    assert out.data["edit"] == "edited"
    assert tool.config.effort == 1
    assert overrides == {"effort": 5, "timeout_s": 7.0}


async def test_default_context_is_unchanged(local_executor: LocalExecutor) -> None:
    out = await check_tool(ContextTool(), {}, executor=local_executor)
    assert out.data["staged"] == {
        "schema_version": 1,
        "params": {"n": 1},
        "round": 0,
        "round_overrides": {},
        "replicate": 0,
    }
    assert out.data["env_round"] == "0"


@pytest.mark.parametrize(
    ("override_timeout", "cap", "expected"), [(4.0, 60.0, 4.0), (90.0, 6.0, 6.0)]
)
async def test_effective_timeout_keeps_kit_cap(
    override_timeout: float, cap: float, expected: float
) -> None:
    executor = RecordingExecutor()
    await check_tool(
        ContextTool(),
        {},
        round_overrides={"timeout_s": override_timeout},
        timeout_s=cap,
        executor=executor,
    )
    assert executor.jobs[0].timeout_s == expected


@pytest.mark.parametrize("overrides", [{"timeout_s": SECRET}, {SECRET: SECRET}])
async def test_invalid_override_is_sanitized_before_work(overrides: dict[str, Any]) -> None:
    executor = RecordingExecutor()
    # Snapshotting would fail first if the effective config were validated too late.
    tool = Ref(path="swarm_arena:ScriptTool", config={"command": ["echo"], "files": ["missing/*"]})
    with pytest.raises(ValueError, match=r"float_parsing|extra_forbidden") as caught:
        await check_tool(tool, {}, round_overrides=overrides, executor=executor)
    assert_secret_absent(caught.value, SECRET)
    assert "matched no file" not in str(caught.value)
    assert executor.jobs == []


@pytest.mark.parametrize("field", ["command", "python", "files", "env", "editable"])
async def test_script_human_only_overrides_are_refused(field: str) -> None:
    executor = RecordingExecutor()
    with pytest.raises(ValueError, match="human-only") as caught:
        await check_tool(
            ScriptTool(command=["echo"]), {}, round_overrides={field: SECRET}, executor=executor
        )
    assert_secret_absent(caught.value, SECRET)
    assert executor.jobs == []


@pytest.mark.parametrize("field", ["script", "script_path"])
async def test_custom_human_only_field_and_alias_are_refused(field: str) -> None:
    with pytest.raises(ValueError, match="human-only") as caught:
        await check_tool(ContextTool(), {}, round_overrides={field: SECRET})
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize(
    ("parent_base", "parent_effort", "accepted"), [(1, 5, True), (1, 6, True), (2, 5, False)]
)
async def test_overlay_identity_uses_snapshotted_base_config(
    parent_base: int, parent_effort: int, accepted: bool, local_executor: LocalExecutor
) -> None:
    tool = ContextTool()
    request = ToolRequest(
        round=3,
        round_overrides={"effort": parent_effort},
        replicate=2,
        edits={"problem/a.txt": "inherited"},
    )
    parent = make_candidate(
        ToolOutput(status=RunStatus.OK), tool=ContextTool({"effort": parent_base}), request=request
    )
    equivalent = make_candidate(
        ToolOutput(status=RunStatus.OK), tool=ContextTool({"effort": parent_base})
    )
    assert parent.tool_identity == equivalent.tool_identity
    assert parent.round == request.round
    if accepted:
        out = await check_tool(
            tool,
            {},
            parents=[parent],
            round=4,
            round_overrides={"effort": 5},
            executor=local_executor,
        )
    else:
        with pytest.warns(UserWarning, match="not overlaid"):
            out = await check_tool(
                tool,
                {},
                parents=[parent],
                round=4,
                round_overrides={"effort": 5},
                executor=local_executor,
            )
    assert out.data["edit"] == ("inherited" if accepted else "absent")


async def test_custom_config_rejects_invalid_scalar_override() -> None:
    executor = RecordingExecutor()
    with pytest.raises(ValueError, match="effort: int_parsing") as caught:
        await check_tool(ContextTool(), {}, round_overrides={"effort": SECRET}, executor=executor)
    assert_secret_absent(caught.value, SECRET)
    assert executor.jobs == []


async def test_effective_editability_still_refuses_base_overlay(
    local_executor: LocalExecutor,
) -> None:
    tool = ContextTool()
    parent = make_candidate(
        ToolOutput(status=RunStatus.OK),
        tool=tool,
        request=ToolRequest(round_overrides={"effort": 5}, edits={"problem/a.txt": "inherited"}),
    )
    with pytest.warns(UserWarning, match="not overlaid"):
        out = await check_tool(
            tool, {}, parents=[parent], round_overrides={"effort": 1}, executor=local_executor
        )
    assert out.data["edit"] == "absent"


async def test_effective_protection_survives_base_identity(
    monkeypatch: pytest.MonkeyPatch, local_executor: LocalExecutor
) -> None:
    from collections.abc import Set

    def protected(self: ContextTool, root: Set[str]) -> frozenset[str]:
        return frozenset({"problem/a.txt"}) if self.config.effort == 5 else frozenset()

    monkeypatch.setattr(ContextTool, "_protected", protected)
    tool = ContextTool()
    parent = make_candidate(
        ToolOutput(status=RunStatus.OK),
        tool=tool,
        request=ToolRequest(round_overrides={"effort": 6}, edits={"problem/a.txt": "parent"}),
    )
    with pytest.warns(UserWarning, match="not overlaid"):
        out = await check_tool(
            tool, {}, parents=[parent], round_overrides={"effort": 5}, executor=local_executor
        )
    assert out.data["edit"] == "absent"
    with pytest.raises(ValueError, match="protected"):
        make_candidate(
            out,
            tool=tool,
            request=ToolRequest(round_overrides={"effort": 5}, edits={"problem/a.txt": "new"}),
        )
