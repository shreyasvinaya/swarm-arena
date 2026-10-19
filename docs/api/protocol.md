# swarm_arena.protocol

```{eval-rst}
.. automodule:: swarm_arena.protocol
   :members:
```

The core records mirror A2A v1.0.1 and require no A2A SDK. Enum values use
proto names such as `TASK_STATE_COMPLETED`. Records are frozen, slotted, and
accept keyword arguments. Their nested metadata mappings follow the declared
`Mapping` interface; freezing a record does not recursively freeze its contents.

```python
from datetime import UTC, datetime
from swarm_arena.protocol import Artifact, Part, Task, TaskState, TaskStatus

result = Artifact(artifact_id="answer", parts=(Part(data={"score": 0.25}),))
status = TaskStatus(state=TaskState.COMPLETED, timestamp=datetime.now(UTC))
task = Task(id="t1", context_id="run1", status=status, artifacts=(result,))
```

A `Part` requires exactly one of `text`, `raw`, `url`, or `data`. Passing
`data=None` creates JSON null content; omitting `data` leaves that content unset.
Messages and artifacts require at least one part. Status timestamps must be
aware and are normalized to UTC with millisecond precision.
