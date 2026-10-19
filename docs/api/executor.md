# swarm_arena.executor

`Job` describes a tool invocation or command with its staged workspace, resource
request, timeout, and environment. A tool job requires both a `Ref` and a
`ToolRequest`; a command job requires a `Command`.

The driver maps workspace paths to artifact hashes. Root and candidate-file keys
must already be normalized relative paths, with no reserved top-level names:

```python
from swarm_arena.executor import Job

stage = Job.Stage(root={"problem/input.txt": "a" * 64})
started = Job.Started(kind="process", job_id="c-1")
assert Job.Started.model_validate_json(started.model_dump_json()) == started
```

Subclass `Executor` to implement `submit`, `capacity`, and `aclose`. Construction
only validates configuration; readiness checks belong in `preflight`. The default
`confirm_dead` returns `None` (unknown) and `kill_orphan` returns `False`.

Executors await `Job.on_started`, when supplied, before releasing the job to run.
If submission is cancelled, the executor stops the job in a shielded sequence:
termination signal, grace period, partial collection, and forced termination.
`Job.on_partial` is the only channel for delivering that cancel-path partial
output to the caller, and runs inside the shielded stop sequence. These are
executor implementation obligations; the ABC does not launch or stop processes.

Callbacks are local control hooks and must be omitted from serialized payloads.
`secret_values` overrides other environment sources and is excluded from repr,
as are both callbacks. Secret values must never be journaled or logged.

```{eval-rst}
.. automodule:: swarm_arena.executor
   :members:
```
