import doctest
import importlib
import importlib.metadata
import inspect
import subprocess
import sys
import types

import pytest
from _fakes.dependencies import ray_dependency
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
PUBLIC["swarm_arena"] += ["ScriptedManager"]
PUBLIC["swarm_arena"] += ["Swarm"]
PUBLIC["swarm_arena.testing"] += ["check_manager"]
PUBLIC["swarm_arena.testing"] += ["FakeModel", "FakeTurn"]
PUBLIC["swarm_arena"] += ["Ref"]
PUBLIC["swarm_arena"] += ["validate_plan"]
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
PUBLIC["swarm_arena.board"] += ["ArenaBoard"]
PUBLIC["swarm_arena.board"] += ["Curator"]
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
PUBLIC["swarm_arena.manager"] += [
    "AgentContext",
    "AgentStatus",
    "LimitReached",
    "Manager",
    "RoundReport",
]
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


try:
    importlib.metadata.version("openai-agents")
    importlib.metadata.version("litellm")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    PUBLIC["swarm_arena.llm"] = [
        "AgentsSDKManager",
        "BudgetExceeded",
        "LLMCurator",
        "LLMJudge",
        "LLMPlanner",
        "CollaborativePlanner",
        "Reviewer",
        "Approve",
        "Feedback",
        "Edited",
    ]


try:
    importlib.metadata.version("ray")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    PUBLIC["swarm_arena.ray"] = ["RayExecutor"]


@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_all_snapshot(module: str) -> None:
    names = importlib.import_module(module).__all__
    assert sorted(names) == sorted(PUBLIC[module])
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("module", ["swarm_arena", "swarm_arena.llm", "swarm_arena.testing"])
def test_facade_has_no_extra_public_names(module: str) -> None:
    if module == "swarm_arena.llm":
        pytest.importorskip("agents")
    facade = importlib.import_module(module)
    children = {
        name
        for name, value in vars(facade).items()
        if isinstance(value, types.ModuleType) and value.__name__.startswith(f"{module}.")
    }
    for names in (vars(facade), dir(facade)):
        assert {name for name in names if not name.startswith("_")} <= set(
            facade.__all__
        ) | children


@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_names_report_public_module(module: str) -> None:
    mod = importlib.import_module(module)
    for name in mod.__all__:
        if module == "swarm_arena.testing" and name in {"FakeModel", "FakeTurn"}:
            try:
                importlib.metadata.version("openai-agents")
            except importlib.metadata.PackageNotFoundError:
                continue
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


CORE = sorted(
    set(PUBLIC) - {"swarm_arena", "swarm_arena.testing", "swarm_arena.llm", "swarm_arena.ray"}
)


@pytest.mark.parametrize("pydantic_model", [False, True])
def test_publish_invalidates_original_class_source_location(
    pydantic_model: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Plain:
        class Nested:
            pass

    class Model(BaseModel):
        class Nested(BaseModel):
            pass

    demo = Model if pydantic_model else Plain
    # Python 3.13 adds this metadata; also exercise invalidation on older versions.
    for cls in (demo, demo.Nested):
        type.__setattr__(cls, "__firstlineno__", 1)
    module = types.ModuleType("source_location_facade")
    module.__file__ = __file__
    vars(module)["Demo"] = demo
    monkeypatch.setitem(sys.modules, module.__name__, module)
    publish(module.__name__, ["Demo"])
    for cls in (demo, demo.Nested):
        assert cls.__module__ == module.__name__
        assert "__firstlineno__" not in vars(cls)
        try:
            source, _ = inspect.getsourcelines(cls)
        except OSError:
            pass  # Retargeted classes need not have inspectable source.
        else:
            assert source[0].lstrip().startswith(f"class {cls.__name__}")


def test_root_import_publishes_core_facades() -> None:  # __module__ never depends on import order
    code = f"import sys, swarm_arena; print(sorted(set({CORE!r}) - set(sys.modules)))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


PUBLIC["swarm_arena"] += ["ScriptCheck", "ScriptJudge"]

PUBLIC["swarm_arena.testing"] += ["check_tool", "check_check", "check_judge", "make_candidate"]
PUBLIC["swarm_arena.testing"] += ["read_logs", "read_output_file"]

PUBLIC["swarm_arena.scoring"] += ["HumanReview"]


@pytest.mark.parametrize("missing", ["agents", "litellm"])
def test_llm_facade_names_the_extra_when_missing(missing: str) -> None:
    code = f"import sys; sys.modules[{missing!r}] = None; import swarm_arena.llm"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert (
        'uv add "swarm-arena[llm] @ git+https://github.com/shreyasvinaya/swarm-arena"'
        in proc.stderr
    )
    assert "ModuleNotFoundError" in proc.stderr


def test_llm_facade_publishes_its_names() -> None:
    llm = pytest.importorskip("swarm_arena.llm")
    assert all(getattr(llm, name).__module__ == "swarm_arena.llm" for name in llm.__all__)


@pytest.mark.parametrize(("dist", "module"), [("openai-agents", "agents"), ("litellm", "litellm")])
def test_llm_facade_checks_distribution_metadata(dist: str, module: str) -> None:
    code = f"""
import importlib.metadata

def version(name):
    if name == {dist!r}:
        raise importlib.metadata.PackageNotFoundError(name)
    return "0"

importlib.metadata.version = version
try:
    import swarm_arena.llm
except ModuleNotFoundError as error:
    assert error.name == {module!r}
    assert (
        'uv add "swarm-arena[llm] @ git+https://github.com/shreyasvinaya/swarm-arena"'
        in str(error)
    )
else:
    raise AssertionError("missing extra imported")
"""
    subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=30
    )


@pytest.mark.parametrize(
    ("error_type", "name"),
    [("ModuleNotFoundError", "transitive_dependency"), ("ImportError", "agents")],
)
def test_llm_facade_preserves_other_import_errors(error_type: str, name: str) -> None:
    pytest.importorskip("swarm_arena.llm")
    code = f"""
import builtins
original = builtins.__import__
error = {error_type}("upstream failure", name={name!r})

def importing(module, *args, **kwargs):
    if module == "agents":
        raise error
    return original(module, *args, **kwargs)

builtins.__import__ = importing
try:
    import swarm_arena.llm
except ImportError as caught:
    assert caught is error
else:
    raise AssertionError("broken upstream imported")
"""
    subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=30
    )


def test_ray_all() -> None:
    module = ray_dependency()
    assert module.__all__ == ["RayExecutor"]


def test_ray_facade_publishes_methods() -> None:
    module = ray_dependency()
    cls = module.RayExecutor
    assert {
        cls.__module__,
        cls.Config.__module__,
        cls.submit.__module__,
        cls.confirm_dead.__module__,
    } == {"swarm_arena.ray"}
