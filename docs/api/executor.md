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
share the component-configuration rule: keep schema field names, aliases and
list/tuple indices, hide unknown model keys as `<field>`, and replace each
free-form mapping or `Any` segment with `<…>`. For example, an unknown field in
a staged candidate is reported at `candidate.<field>`, and in a parent at
`parents.0.<field>`. The outermost validated model owns the unknown-key policy
for its whole tree; human-authored Plan, RunConfig and Limits settings show
nested typo identifiers. Rejected values and validator context never appear.

## LocalExecutor

`LocalExecutor` runs each job on this POSIX host in a fresh workspace,
`<workspace_root>/<id>-<random>`, and in its own process group. In the name, each
character of the job id outside `[A-Za-z0-9._-]` becomes `_`, and the id is cut
to 64 characters. The default `workspace_root` is per user:
`tempfile.gettempdir()/swarm-arena-<uid>`. The root and every parent directory
must be owned by you or root, and not writable by group or others unless sticky
(like `/tmp`). The command starts only after `Job.on_started` returns.

On a timeout or a cancel, the group gets SIGTERM, then `grace_s`, then `collect`
for the partial output, then SIGKILL. The kept output gets the stop status and
`partial=True`. A blocked `collect` delays the SIGKILL by at most
`BEFORE_KILL_TIMEOUT_S` (30 s) after the grace period: at that deadline the
supervisor kills the group, and the collect is abandoned and not tried again.
Collect runs in a daemon thread, so an abandoned one never keeps the interpreter
from exiting. It may still write into the workspace after `submit` returned.

The cancel-path output goes to `Job.on_partial`, once the start handshake was
reached. This includes a cancel that comes after the tool exited on its own: if
its collect had not finished, it runs once more, bounded by
`BEFORE_KILL_TIMEOUT_S`. Logs live at `swarm/stdout.log` and `swarm/stderr.log`.
The last `max_log_bytes` (8 MiB) of each are uploaded to `artifact_root`, with
`logs_ref = "sha256:<stdout>:<stderr>"`, and a `log truncated` note in the
stderr log. If the declared output files of a run are more than
`max_output_bytes` (256 MiB) in total, the run gets status `error` and no files.

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
