"""Shared public-surface snapshots and dependency-aware import helpers."""

import importlib
import importlib.metadata
from types import ModuleType
from typing import Any

import pytest

PUBLIC_ALL: dict[str, list[str]] = {
    "swarm_arena": [
        "Goal",
        "Plan",
        "RunConfig",
        "PartSpec",
        "TeamSpec",
        "AgentSpec",
        "RoundSpec",
        "ConvergeSpec",
        "Limits",
        "Resources",
        "Ref",
        "Swarm",
        "validate_plan",
        "ScriptTool",
        "ScriptCheck",
        "ScriptJudge",
        "FunctionTool",
        "ProgramJudge",
        "Score",
        "ScriptedManager",
        "KeepTopTeams",
        "SuccessiveHalving",
        "Patience",
        "Chain",
    ],
    "swarm_arena.tool": [
        "Tool",
        "Command",
        "CompletedRun",
        "ToolRequest",
        "ParentRef",
        "ToolOutput",
        "FileRef",
        "FileSet",
        "Usage",
        "Estimate",
        "Action",
        "RunStatus",
    ],
    "swarm_arena.scoring": [
        "Check",
        "CheckContext",
        "CheckResult",
        "Judge",
        "JudgeContext",
        "Lexicographic",
        "HumanReview",
    ],
    "swarm_arena.manager": [
        "Manager",
        "AgentContext",
        "RoundReport",
        "LimitReached",
        "AgentStatus",
    ],
    "swarm_arena.planning": [
        "Planner",
        "History",
        "Policy",
        "RoundState",
        "Decision",
        "FlowSpec",
    ],
    "swarm_arena.board": [
        "Board",
        "BoardView",
        "ArenaBoard",
        "Frontier",
        "Learning",
        "Candidate",
        "LLMCall",
        "GoalKey",
        "Row",
        "LineageGraph",
        "Curator",
    ],
    "swarm_arena.executor": [
        "Executor",
        "LocalExecutor",
        "Job",
    ],
    "swarm_arena.config": [
        "ModelCatalog",
        "ModelSpec",
        "CachePolicy",
        "PlannerCatalog",
        "CatalogEntry",
        "SecretsPolicy",
        "TracingConfig",
        "GatewayConfig",
    ],
    "swarm_arena.llm": [
        "AgentsSDKManager",
        "LLMPlanner",
        "CollaborativePlanner",
        "Reviewer",
        "Approve",
        "Feedback",
        "Edited",
        "LLMJudge",
        "LLMCurator",
        "BudgetExceeded",
    ],
    "swarm_arena.ray": [
        "RayExecutor",
    ],
    "swarm_arena.protocol": [
        "Message",
        "Part",
        "Artifact",
        "Task",
        "TaskStatus",
        "TaskState",
        "Role",
    ],
    "swarm_arena.testing": [
        "check_tool",
        "check_check",
        "check_judge",
        "check_manager",
        "make_candidate",
        "read_output_file",
        "read_logs",
        "FakeModel",
        "FakeTurn",
    ],
}

PUBLIC_SUBMODULES: frozenset[str] = frozenset(
    {
        "tool",
        "scoring",
        "manager",
        "planning",
        "board",
        "executor",
        "config",
        "protocol",
        "testing",
        "llm",
        "ray",
    }
)
ENTRY_MODULES: frozenset[str] = frozenset({"run_function"})
EXTRA_DEPS: dict[str, tuple[str, ...]] = {
    "swarm_arena.llm": ("agents", "litellm"),
    "swarm_arena.ray": ("ray",),
}
LAZY_DEPS: dict[tuple[str, str], tuple[str, ...]] = {
    ("swarm_arena.testing", "FakeModel"): EXTRA_DEPS["swarm_arena.llm"],
    ("swarm_arena.testing", "FakeTurn"): EXTRA_DEPS["swarm_arena.llm"],
}
DISTRIBUTIONS: dict[str, str] = {
    "agents": "openai-agents",
    "litellm": "litellm",
    "ray": "ray",
}


def deps_present(module: str, name: str | None = None) -> bool:
    """
    Check distribution metadata without importing optional packages.

    Parameters
    ----------
    module : str
        Public facade module.
    name : str or None, optional
        Export whose lazy dependencies must also be available.

    Returns
    -------
    bool
        Whether every required distribution is installed.

    Examples
    --------
    >>> deps_present("swarm_arena.tool", "Tool")
    True
    """
    deps = EXTRA_DEPS.get(module, ())
    if name is not None:
        deps += LAZY_DEPS.get((module, name), ())
    try:
        for dep in deps:
            importlib.metadata.version(DISTRIBUTIONS[dep])
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def import_public(module: str) -> ModuleType:
    """
    Import a facade or skip when its optional distributions are absent.

    Parameters
    ----------
    module : str
        Public facade module.

    Returns
    -------
    ModuleType
        Imported facade. Import failures from installed packages propagate.

    Examples
    --------
    >>> import_public("swarm_arena").__name__
    'swarm_arena'
    """
    if not deps_present(module):
        extra = module.rsplit(".", 1)[-1]
        pytest.skip(f"needs the {extra} extra")
    return importlib.import_module(module)


def public_object(module: str, name: str) -> Any:
    """
    Get an export or skip when its optional distributions are absent.

    Parameters
    ----------
    module : str
        Public facade module.
    name : str
        Export to retrieve, including lazy testing exports.

    Returns
    -------
    Any
        Public object. Import failures from installed packages propagate.

    Examples
    --------
    >>> public_object("swarm_arena", "Goal").__name__
    'Goal'
    """
    mod = import_public(module)
    if not deps_present(module, name):
        pytest.skip("needs the llm extra")
    return getattr(mod, name)


def resolve(dotted: str) -> Any:
    """
    Import the longest module prefix and traverse its remaining attributes.

    Parameters
    ----------
    dotted : str
        Dotted module, class, function, or member path.

    Returns
    -------
    Any
        Resolved object. Missing attributes and internal import errors propagate.

    Examples
    --------
    >>> resolve("swarm_arena.Ref.of").__name__
    'of'
    """
    parts = dotted.split(".")
    for index in range(len(parts), 0, -1):
        prefix = ".".join(parts[:index])
        try:
            obj = importlib.import_module(prefix)
        except ModuleNotFoundError as error:
            if error.name is None or not (
                prefix == error.name or prefix.startswith(error.name + ".")
            ):
                raise
            if index == 1:
                raise
        else:
            break
    for attr in parts[index:]:
        obj = getattr(obj, attr)
    return obj
