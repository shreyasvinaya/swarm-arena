# Running on Ray

## Install

Install the optional runtime with uv:

```sh
uv add "swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena"
```

The extra requires `ray[default]>=2.59,<3`. Ray 2.59 provides CPython
3.11–3.13 wheels on Linux x86_64/aarch64 and macOS arm64 (macOS 12 or newer).
Intel macOS has no wheel for this version. macOS CI checks the core package;
the dedicated Linux Ray jobs check the optional runtime.

Set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before starting a Ray driver with `uv run`
to prevent Ray from automatically packaging the driver's uv environment.
Cluster authentication and worker dependency provisioning are separate setup steps.

## Running the driver as a Ray Job

`swarm-arena submit` runs `swarm-arena run --yes --` inside the cluster as a
Ray Job. The Ray executor connects from that driver with
`ray.init(address="auto")`; Ray Client (`ray://`) is not used. Check the
estimate locally before approving unattended execution:

```sh
swarm-arena plan run.toml plan.toml
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
swarm-arena submit --address http://<head>:8265 --working-dir . --yes run.toml plan.toml
```

Replace `<head>` with your head node's hostname. With `--address`, any
different nonempty value in `RAY_ADDRESS` or `RAY_API_SERVER_ADDRESS` is refused before
contacting Ray; unset the conflicting variable. Without `--address`, Ray's
environment fallback applies (`RAY_API_SERVER_ADDRESS` before `RAY_ADDRESS`).
Success prints the submission ID and exits zero;
Ray submission errors exit one. Without `--yes`, the command submits nothing
and exits two, even in a terminal. The ID is Ray's `submission_id`, not its
deprecated `job_id` argument.

Client construction, uploads and submission share one 300-second deadline.
Every submission gets a client-generated `swarm-arena-<uuid4 hex>` ID. On a
timeout or interrupt, the CLI exits nonzero and prints that ID with
`check ray job status <id> before retrying`. The request may already have
created a job, and an in-flight request can still finish after the CLI stops
waiting. Check its status before submitting again to avoid starting another
unattended driver. An abandoned client constructor cannot start a new
submission after returning.

The worker image or repeatable `--pip SPEC` options must provide
`swarm-arena[ray]` and the connector package, including the `swarm-arena`
console script. For example, before the first release:

```sh
swarm-arena submit --address http://<head>:8265 --working-dir . \
  --pip 'swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena' \
  --pip deepretro-swarm --yes run.toml plan.toml
```

Paths in the driver command are shell-quoted. Keep the run configuration,
plans and connector code accessible from the job's working directory.
When `--working-dir` is supplied, it must be an existing local directory;
the run and plan paths must exist and resolve inside it, including through
symlinks. Supply paths relative to the submitting shell's current directory
or absolute paths. The command rewrites them relative to the uploaded
directory: `--working-dir proj proj/run.toml proj/plan.toml` runs
`swarm-arena run --yes -- run.toml plan.toml` inside `proj`. Without
`--working-dir`, paths retain their meaning in the cluster's driver directory.
The `--` terminator also preserves dash-prefixed filenames as paths.
Only an explicitly supplied `working_dir` and `pip` list enter `runtime_env`;
omitting both sends no runtime environment. Choose the upload directory so it
contains no secrets, and keep credentials out of dependency specifications.
Ray uses `.gitignore` and `.rayignore` rules to exclude files from a
`working_dir` upload. Keep the secrets file outside the uploaded tree;
exclude any other private files with those ignore rules before submitting.
Dependency URLs with userinfo (for example, a username or token before `@`
in the URL authority) are refused with a fixed diagnostic. Provision package
authentication outside `--pip` specifications.
Secrets never go in `runtime_env`, including `env_vars`: provision them
separately in the worker environment or through `RunConfig.secrets.file` on
the worker. The command does not copy credentials from the submitting shell.

Authentication also needs external provisioning. Same-user local
`ray.init(address="auto")` can use token authentication through Ray's default
token file. Explicit-address drivers and HTTP Jobs clients need
`RAY_AUTH_MODE=token` and an available token (for example, externally supplied
`RAY_AUTH_TOKEN` or a provisioned token file). Provision authentication for
the submitting client and the in-cluster driver separately; local discovery
does not establish remote access. Keep tokens out of job metadata and uploaded
files. Set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` in the launching environment
before Ray initialization to retain explicit packaging control.

## The Ray executor

Select the executor in `run.toml`:

```toml
schedule_timeout_s = 600.0

[executor]
path = "swarm_arena.ray:RayExecutor"

[executor.config]
address = "auto"
poll_s = 0.2
cancel_margin_s = 10.0
object_store_cap_mib = 256
# secret_resource = "llm_keys"
```

`RayExecutor()` validates settings without I/O. Its first operation connects to
an existing cluster in a worker thread. `address` chooses the cluster;
`ray://` addresses are refused. If `address` is not `"auto"`, it must match the
connected runtime's GCS address; a mismatch fails with a value-free error.
This check prevents Ray's process-wide existing connection from silently
overriding an explicit address. `poll_s` controls worker subprocess polling.
`cancel_margin_s` adds time after the job's grace period for soft cancellation
and bounds recovery operations, including connection and lock waits. Both
intervals must be finite and positive. Initialization is single-flight per
executor: if a caller stops waiting, the next caller rejoins the same attempt
under its own allowance and can adopt a successful connection. A failed attempt
can be retried by a later caller, after the previous attempt has finished.
The abandoned caller cannot dispatch work after its lifetime ends. `aclose()`
leaves the shared Ray connection open and never calls `ray.shutdown()`.

Ray may log `SIGTERM handler is not set because current thread is not the main
thread.` during this initialization. The warning is retained: initialization
stays off the event loop so it cannot block lease renewal or cancellation.

Each job gets one task named with its job ID. When provided, `on_started`
receives the dispatch intent and original Ray driver job ID before submission.
A failed callback prevents submission. Payload serialization and `ray.put`
finish in an abandonable preparation worker before that intent; the event loop
submits the resulting object reference. Tasks are not resubmitted after a Ray
error. Out-of-memory failures become `oom`; other Ray failures become `error`
with type-only diagnostics. Core artifact-storage failures propagate to the
owner rather than becoming ordinary candidate errors.

Resources map directly to logical Ray requests: `cpus` to `num_cpus`, `gpus`
to `num_gpus`, `memory_gib` to integer bytes, and `custom` to named resources.
With `secret_resource = "llm_keys"`, jobs needing node/file secrets also request
one `llm_keys` resource. Names in `drop_env` and names already supplied by
explicit driver environment or secret overlays do not require that resource.
Secret values never enter task options or `runtime_env`. Driver-supplied
secrets travel only inside the task payload; Ray may spill that payload from
its object store to local disk. Provision and protect cluster storage accordingly.

For an `s3://` artifact root, workers read and write artifacts by SHA-256 against
the shared backend. For a local root, the driver sends staged blobs through
Ray's object store and stores hash-checked result blobs in its own artifact
store. Each upload is supervised separately; cancelled publication cannot start
the next upload. Input and output blobs each have a **256 MiB default cap**.
`object_store_cap_mib` accepts integers from 1 through 1024; use S3 for larger
transport. Log blobs count toward the output cap. Workers need access to their
connector packages and to any FunctionTool module roots resolved by the driver.
A local test with shared paths does not establish that those paths exist on a
remote worker node.

`schedule_timeout_s` measures continuous resource infeasibility while a task is
pending or not yet visible in state. Every second, the executor compares its
request with each alive node's **total** capacity. If no single node can meet
it for the whole allowance, the job returns an error and is cancelled. The
clock resets when a fitting node joins or becomes visible. A feasible request
waiting for busy resources keeps its queue position; this schedule limit does
not expire it. Queue time still counts toward the enclosing round timeout.
Unavailable resource/state evidence is handled conservatively. Pending jobs
share one resource snapshot per executor per second, and a stalled resource RPC
is rejoined rather than duplicated. The scheduling watch stops polling once
the task is no longer pending. Outside a run, the executor uses 600 seconds.

Cancellation is shielded: request `ray.cancel(ref)`, wait up to
`grace_s + cancel_margin_s`, then request `ray.cancel(ref, force=True)` even if
the soft step failed. The soft wait, force request and partial callback share
that single deadline; the latter steps get only its remaining time. Pending
work is cancelled before it can later start. A soft-cancelled worker may return
a collected partial; only `Job.on_partial` receives it. As with LocalExecutor,
that result retains data and drops file, log and report references, and starts
no new artifact upload. A timed-out or failed partial wait gives an empty
`cancelled` result. As with LocalExecutor, pending caller cancellation wins over
any callback failure, including a registered control; a successful callback
receives the data-only partial exactly once. A schedule timeout has no partial
callback. Hard process controls synchronously request force cancellation and
then propagate unchanged, without awaited cleanup.

Ray preserves its assigned `CUDA_VISIBLE_DEVICES` in the tool subprocess.
Ray resources are logical scheduling quantities, so subprocess resource limits
still apply where the operating system supports them; a logical memory request
is not an isolation boundary.

Recovery uses the connected cluster's `RayContext.dashboard_url` and the
recorded original driver's job ID. `confirm_dead` requires identity-matched
terminal-task and dead-worker evidence. `kill_orphan` waits for Ray's owner
fate sharing and never kills another task itself. Missing, ambiguous, dropped,
unavailable or late state stays unknown. **Recovery confirmation needs a
plain-HTTP dashboard reachable at `RayContext.dashboard_url`. HTTPS, proxied or
`RAY_OVERRIDE_DASHBOARD_URL` dashboards make lost runs stay unconfirmed; their
spend stays estimated.** No environment discovery fallback is used for state
lookups.
