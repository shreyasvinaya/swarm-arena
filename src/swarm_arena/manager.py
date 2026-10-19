"""Manager connector types."""

from swarm_arena._facade import publish as _publish
from swarm_arena._model.results import AgentStatus

__all__: list[str] = ["AgentStatus"]

_publish(__name__, __all__)
