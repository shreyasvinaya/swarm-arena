from pathlib import Path
from typing import Any, ClassVar, cast

import pydantic
from _fakes.tools import FixedJudge
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

from swarm_arena import ScriptJudge, ScriptTool
from swarm_arena._model.goal import FileSet
from swarm_arena._model.ref import RefField
from swarm_arena._model.results import ToolOutput
from swarm_arena._model.tooling import Command, CompletedRun, ToolRequest
from swarm_arena._tooling.tool import Tool

SENTINEL = "d4dc419b-23f0-4efa-aadb-c96e836e0b25"


class ProbeTool(Tool):
    _human_only: ClassVar[frozenset[str]] = frozenset({"command", "env"})

    class Config(BaseModel):
        model_config = ConfigDict(
            extra="forbid", frozen=True, hide_input_in_errors=True, populate_by_name=True
        )
        command: str = Field(
            default="safe",
            validation_alias=AliasChoices("command_alias", AliasPath("command_path", 0), "command"),
        )
        text: str = "fixed"
        tuning: int = Field(default=1, validation_alias=AliasChoices("tune_alias", "tuning"))
        env: list[str] = []
        files: FileSet = FileSet()
        fail_at: str = ""

    def _fail(self, where: str) -> None:
        cfg = cast(ProbeTool.Config, self.config)
        if cfg.fail_at == where:
            cause = RuntimeError(cfg.text)
            error = ValueError(cfg.text)
            error.add_note(cfg.text)
            raise error from cause

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._fail("constructor")

    def _own_files(self) -> dict[str, str]:
        self._fail("files")
        return cast(ProbeTool.Config, self.config).files.manifest

    def is_deterministic(self, request: ToolRequest) -> bool:
        self._fail("determinism")
        return False

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        return Command(argv=["true"])

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        return ToolOutput(status=run.status)


class OtherTool(ProbeTool):
    pass


class Wrapper(ProbeTool):
    _human_only: ClassVar[frozenset[str]] = frozenset()

    class Config(ProbeTool.Config):  # ty: ignore[invalid-attribute-override]
        members: list[RefField] = []


class FileJudge(FixedJudge):
    class Config(FixedJudge.Config):  # ty: ignore[invalid-attribute-override]
        files: FileSet = Field(default_factory=lambda: FileSet(globs=[SENTINEL]))


class MapJudge(FixedJudge):
    class Config(FixedJudge.Config):  # ty: ignore[invalid-attribute-override]
        children: dict[str, RefField] = {}


class DefaultScriptJudge(ScriptJudge):
    class Config(ScriptJudge.Config):  # ty: ignore[invalid-attribute-override]
        files: FileSet = Field(default_factory=lambda: FileSet(globs=["judge.py"]))


class AliasScriptTool(ScriptTool):
    class Config(ScriptTool.Config):  # ty: ignore[invalid-attribute-override]
        command: list[str] = Field(default_factory=lambda: ["true"], validation_alias="argv")


class AliasedDefaultScriptJudge(DefaultScriptJudge):
    class Config(DefaultScriptJudge.Config):  # ty: ignore[invalid-attribute-override]
        files: FileSet = Field(
            default_factory=lambda: FileSet(globs=["judge.py"]),
            validation_alias=AliasChoices("sources", AliasPath("bundle", "sources"), "files"),
        )


class NormalizeTool(ProbeTool):
    class Config(ProbeTool.Config):  # ty: ignore[invalid-attribute-override]
        labels: list[str] = []
        metadata: dict[str, str] = {}

        @pydantic.field_validator("labels")
        @classmethod
        def unique(cls, value: list[str]) -> list[str]:
            return list(dict.fromkeys(value))

        @pydantic.field_validator("metadata")
        @classmethod
        def normalize_keys(cls, value: dict[str, str]) -> dict[str, str]:
            return {key.lower(): item for key, item in value.items()}


class ShrinkingRefsTool(ProbeTool):
    class Config(ProbeTool.Config):  # ty: ignore[invalid-attribute-override]
        members: list[RefField] = []

        @pydantic.field_validator("members")
        @classmethod
        def first_only(cls, value: list[Any]) -> list[Any]:
            return value[:1]
