"""Optional Ray integration, installed with ``swarm-arena[ray]``."""

from swarm_arena._facade import publish as _publish
from swarm_arena.ray import _imports as _imports
from swarm_arena.ray._executor import RayExecutor

__all__ = ["RayExecutor"]

_publish(__name__, __all__)
