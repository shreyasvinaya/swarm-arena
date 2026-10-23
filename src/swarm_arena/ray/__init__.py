"""Optional Ray integration, installed with ``swarm-arena[ray]``."""

from swarm_arena._facade import publish as _publish
from swarm_arena.ray import _imports as _imports

__all__: list[str] = []

_publish(__name__, __all__)
