import doctest
import importlib
import importlib.metadata
import inspect
import subprocess
import sys
import types

import pytest
from pydantic import BaseModel

import swarm_arena
from swarm_arena._facade import publish

PUBLIC: dict[str, list[str]] = {
    m: []
    for m in (
        "swarm_arena",
        "swarm_arena.tool",
        "swarm_arena.scoring",
        "swarm_arena.manager",
        "swarm_arena.planning",
        "swarm_arena.board",
        "swarm_arena.executor",
        "swarm_arena.config",
        "swarm_arena.protocol",
        "swarm_arena.testing",
    )
}
PUBLIC["swarm_arena"] += ["Chain", "KeepTopTeams", "Patience", "SuccessiveHalving"]
PUBLIC["swarm_arena"] += ["Ref"]
PUBLIC["swarm_arena"] += ["Goal"]
PUBLIC["swarm_arena"] += [
    "Plan",
    "PartSpec",
    "TeamSpec",
    "AgentSpec",
    "RoundSpec",
    "ConvergeSpec",
    "Limits",
    "Resources",
]
PUBLIC["swarm_arena.planning"] += [
    "Decision",
    "FlowSpec",
    "History",
    "Planner",
    "Policy",
    "RoundState",
]
PUBLIC["swarm_arena.board"] += ["Board", "BoardView"]
PUBLIC["swarm_arena.board"] += ["Frontier", "GoalKey", "LineageGraph", "Row"]
PUBLIC["swarm_arena.tool"] += [
    "Command",
    "CompletedRun",
    "ToolRequest",
    "ParentRef",
    "ToolOutput",
    "FileRef",
    "FileSet",
    "Usage",
    "Estimate",
    "RunStatus",
]

PUBLIC["swarm_arena"] += ["Score"]
PUBLIC["swarm_arena.scoring"] += ["CheckResult"]
PUBLIC["swarm_arena.board"] += ["Candidate", "Learning", "LLMCall"]
PUBLIC["swarm_arena.manager"] += ["AgentStatus"]
PUBLIC["swarm_arena.protocol"] += [
    "Message",
    "Part",
    "Artifact",
    "Task",
    "TaskStatus",
    "TaskState",
    "Role",
]
PUBLIC["swarm_arena.tool"] += ["Tool", "Action"]
PUBLIC["swarm_arena.executor"] += ["Executor", "Job"]
PUBLIC["swarm_arena.executor"] += ["LocalExecutor"]
PUBLIC["swarm_arena"] += ["RunConfig"]
PUBLIC["swarm_arena"] += ["ScriptTool"]
PUBLIC["swarm_arena.config"] += [
    "CachePolicy",
    "CatalogEntry",
    "GatewayConfig",
    "ModelCatalog",
    "ModelSpec",
    "PlannerCatalog",
    "SecretsPolicy",
    "TracingConfig",
]


PUBLIC["swarm_arena.scoring"] += ["Check", "CheckContext", "Judge", "JudgeContext"]


PUBLIC["swarm_arena"] += ["FunctionTool"]


PUBLIC["swarm_arena"] += ["ProgramJudge"]
PUBLIC["swarm_arena.scoring"] += ["Lexicographic"]


@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_all_snapshot(module: str) -> None:
    names = importlib.import_module(module).__all__
    assert sorted(names) == sorted(PUBLIC[module])
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_names_report_public_module(module: str) -> None:
    mod = importlib.import_module(module)
    for name in mod.__all__:
        obj = getattr(mod, name)
        if isinstance(obj, type) or inspect.isfunction(obj):
            assert obj.__module__ == module, name
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            assert obj.__pydantic_complete__, name


def test_version_is_package_metadata() -> None:
    assert swarm_arena.__version__ == importlib.metadata.version("swarm-arena")
    assert "__version__" not in swarm_arena.__all__


def test_doctest_finder_collects_method_examples() -> None:
    assert "swarm_arena.Ref.of" in {t.name for t in doctest.DocTestFinder().find(swarm_arena)}


def test_publish_lets_doctest_find_method_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    class Demo:
        def m(self) -> int:
            """
            Return one.

            >>> Demo().m()
            1
            """
            return 1

        @classmethod
        def c(cls) -> int:
            """
            Return two.

            >>> Demo.c()
            2
            """
            return 2

        @staticmethod
        def s() -> int:
            """
            Return three.

            >>> Demo.s()
            3
            """
            return 3

        @property
        def p(self) -> int:
            """
            Four.

            >>> Demo().p
            4
            """
            return 4

    module = types.ModuleType("demo_facade")
    vars(module)["Demo"] = Demo
    monkeypatch.setitem(sys.modules, "demo_facade", module)
    publish("demo_facade", ["Demo"])
    found = {t.name for t in doctest.DocTestFinder().find(sys.modules["demo_facade"])}
    assert {
        "demo_facade.Demo.m",
        "demo_facade.Demo.c",
        "demo_facade.Demo.s",
        "demo_facade.Demo.p",
    } <= found


CORE = sorted(set(PUBLIC) - {"swarm_arena", "swarm_arena.testing"})


def test_root_import_publishes_core_facades() -> None:  # __module__ never depends on import order
    code = f"import sys, swarm_arena; print(sorted(set({CORE!r}) - set(sys.modules)))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


PUBLIC["swarm_arena"] += ["ScriptCheck", "ScriptJudge"]

PUBLIC["swarm_arena.testing"] += ["check_tool", "check_check", "check_judge", "make_candidate"]
PUBLIC["swarm_arena.testing"] += ["read_logs", "read_output_file"]
