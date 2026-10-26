"""A2A protocol types."""

from swarm_arena._facade import publish as _publish
from swarm_arena._model.protocol import Artifact, Message, Part, Role, Task, TaskState, TaskStatus

__all__: list[str] = [
    "Message",
    "Part",
    "Artifact",
    "Task",
    "TaskStatus",
    "TaskState",
    "Role",
]

_publish(__name__, __all__)
