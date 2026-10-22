"""Optional LLM integrations, installed with ``swarm-arena[llm]``."""

import importlib.metadata

_MSG = "swarm_arena.llm needs the 'llm' extra: pip install 'swarm-arena[llm]'"
_EXTRA_DEPS = ("agents", "litellm")

for _dist, _module in (("openai-agents", "agents"), ("litellm", "litellm")):
    try:
        importlib.metadata.version(_dist)
    except importlib.metadata.PackageNotFoundError as _exc:
        raise ModuleNotFoundError(_MSG, name=_module) from _exc

try:
    from swarm_arena.llm import _catalog as _catalog  # noqa: I001
    from swarm_arena.llm import _logger as _logger

    # isort: split
    import agents as _agents  # noqa: F401
except ModuleNotFoundError as _exc:
    if _exc.name in _EXTRA_DEPS:
        raise ModuleNotFoundError(_MSG, name=_exc.name) from _exc
    raise

from swarm_arena._facade import publish as _publish
from swarm_arena.llm._hooks import BudgetExceeded
from swarm_arena.llm._manager import AgentsSDKManager

__all__: list[str] = ["AgentsSDKManager", "BudgetExceeded"]

_publish(__name__, __all__)
