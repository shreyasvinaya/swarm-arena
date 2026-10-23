"""Optional LLM integrations, installed with ``swarm-arena[llm]``."""

import importlib.metadata as _metadata

from swarm_arena._model.ref import _LLM_EXTRA_HINT as _MSG

_EXTRA_DEPS = ("agents", "litellm")

for _dist, _module in (("openai-agents", "agents"), ("litellm", "litellm")):
    try:
        _metadata.version(_dist)
    except _metadata.PackageNotFoundError:
        raise ModuleNotFoundError(_MSG, name=_module) from None

try:
    from swarm_arena.llm import _catalog as _catalog  # noqa: I001
    from swarm_arena.llm import _logger as _logger

    # isort: split
    import agents as _agents  # noqa: F401
except ModuleNotFoundError as _exc:
    if _exc.name in _EXTRA_DEPS:
        raise ModuleNotFoundError(_MSG, name=_exc.name) from None
    raise

from swarm_arena._facade import publish as _publish
from swarm_arena.llm._curator import LLMCurator
from swarm_arena.llm._hooks import BudgetExceeded
from swarm_arena.llm._judge import LLMJudge
from swarm_arena.llm._manager import AgentsSDKManager
from swarm_arena.llm._planner import CollaborativePlanner, LLMPlanner
from swarm_arena.llm._reviewer import Approve, Edited, Feedback, Reviewer

__all__: list[str] = [
    "AgentsSDKManager",
    "Approve",
    "BudgetExceeded",
    "CollaborativePlanner",
    "Edited",
    "Feedback",
    "LLMCurator",
    "LLMJudge",
    "LLMPlanner",
    "Reviewer",
]

_publish(__name__, __all__)
