"""Board records: candidates, learnings, and LLM calls."""

from swarm_arena._facade import publish as _publish
from swarm_arena._model.goal import GoalKey

__all__: list[str] = ["GoalKey"]

_publish(__name__, __all__)
