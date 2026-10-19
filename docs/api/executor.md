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

Validation errors from `Job.Stage`, `Job.CandidateFiles`, and `Job.Started`
hide rejected values and unknown or dynamic keys, including nested locations.
Declared field names and list indices remain visible. For example, an unknown
field in a staged candidate is reported at `candidate.<key>`. The typo diagnostics
used for human-authored Plan and Limits settings do not apply to these runtime
records. Component configuration validation retains its separate rule: only
declared top-level field names or aliases are shown, with deeper locations redacted.

## LocalExecutor

`LocalExecutor` runs each job on this POSIX host in a fresh workspace,
`<workspace_root>/<job id>-<random>`, and in its own process group. The command
starts only after `Job.on_started` returns. On a timeout or a cancel, the group
gets SIGTERM, then `grace_s`, then `collect` for the partial output, then SIGKILL.
A blocked `collect` cannot delay the kill. The cancel-path output goes to
`Job.on_partial`. Logs live at `swarm/stdout.log` and `swarm/stderr.log` and are
uploaded to `artifact_root`, with `logs_ref = "sha256:<stdout>:<stderr>"`.

```python
from swarm_arena.executor import LocalExecutor

executor = LocalExecutor(max_concurrent=2, workspace_root="/tmp/my-workspaces")
executor.preflight()  # RuntimeError on a non-POSIX system
# output = await executor.submit(job)
```

Install swarm-arena in the interpreter's own environment (a venv or its
site-packages). The launcher runs as `python -I`, which reads neither
`PYTHONPATH` nor the user site. This is not strong isolation: the tool runs as
the same user, with resource limits but no sandbox.

```{eval-rst}
.. automodule:: swarm_arena.executor
   :members:
```
