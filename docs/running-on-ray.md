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
and before importing Ray. Ray's uv runtime-env hook otherwise packages the
driver's project and runs workers through uv, which can create a different
environment on the worker. Disable it in the launching shell, rather than in
per-task `runtime_env`; provision the worker dependencies explicitly.
Cluster authentication and worker dependency provisioning are separate setup steps.

## Worker environments and nodes

Every worker needs `swarm-arena[ray]`, the connector package, and the tool's
own environment. Bake these into the worker image or install them on each VM.
The tool's configured Python interpreter runs as a **subprocess** of the Ray
worker; `runtime_env.py_executable` is not used. Its interpreter path and
dependencies must exist on every eligible node.

A `FunctionTool` module beside the plan file, or available on the driver's
`sys.path`, contributes an import-root **path**, not a code upload. That exact
path must exist on every eligible worker: on one host, on a shared filesystem,
or at the same location in each image. Otherwise install the module in the
worker image and make its driver-resolved import root available there too.
Uploading a Ray Job's `working_dir` does not by itself establish that an
arbitrary absolute driver import root exists on every node.

With dependencies and authentication provisioned, start a head and join VMs:

```sh
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
uv run ray start --head --include-dashboard=true --disable-usage-stats
# On each worker VM (replace <head> with the head node's hostname):
uv run ray start --address=<head>:6379 --disable-usage-stats
```

The dashboard defaults to localhost. For submission from a laptop, keep an SSH
forward open in a separate terminal (replace `<head>` with your SSH host):

```sh
ssh -N -L 8265:127.0.0.1:8265 <head>
```

Submit to `http://127.0.0.1:8265` through that tunnel. A driver running as a
Ray Job on the head can keep the dashboard's localhost default; its recovery
queries reach the State API locally. To expose the dashboard directly instead,
use `--dashboard-host=0.0.0.0` **only with token authentication and a firewall
or VPN restricting access**. This exposes the Jobs and State APIs, including
job submission; do not expose them to an untrusted network.

Ray 2.59 enables token authentication by default for local clusters. Local
same-user discovery can read Ray's token file; a remote driver needs externally
provisioned `RAY_AUTH_TOKEN` and `RAY_AUTH_MODE=token`. Provision matching
authentication for joining nodes and HTTP Jobs clients as well. Keep those
values out of uploaded files and `runtime_env`.

Set tool secrets in the environment of the `ray start` process on each
eligible node, before starting it; restart the node after changing that
environment. Alternatively, provide `secrets.file` at the configured path on
each node. Restrict secret-using jobs
to nodes carrying a named resource:

```sh
uv run ray start --address=<head>:6379 --resources='{"llm_keys": 1}' --disable-usage-stats
```

Select `RayExecutor(secret_resource="llm_keys")`, or set
`executor.config.secret_resource = "llm_keys"` in the configuration below.
This is a scheduling convention: provision the keys on those nodes separately.

## Running the driver as a Ray Job

`swarm-arena submit` runs `swarm-arena run --yes --` inside the cluster as a
Ray Job. The Ray executor connects from that driver with
`ray.init(address="auto")`; Ray Client (`ray://`) is not used. Check the
estimate locally before approving unattended execution:

```sh
swarm-arena plan run.toml plan.toml
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
swarm-arena submit --address http://127.0.0.1:8265 --working-dir . --yes run.toml plan.toml
```

The address is the local end of the SSH forward above, or the head's own
localhost when submitting there. With `--address`, any
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
swarm-arena submit --address http://127.0.0.1:8265 --working-dir . \
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
`ray://` addresses are refused. RayExecutor records the configured address string
when it initializes the process-wide connection. Later executors reuse that
connection with the same string; a different explicit address fails with a
value-free error. Use one address string per process: two spellings of the same
cluster (such as `localhost:<port>` and `127.0.0.1:<port>`) are deliberately
treated as different. The first connection accepts either spelling without
comparing it with Ray's normalized GCS address.

If another caller already initialized Ray, an explicit address fails with
`Ray is already initialized in this process`. `address="auto"` reuses any
existing connection, including one initialized by another caller; this does
not claim ownership of that connection. These checks prevent Ray from silently
ignoring an explicit address. Initialization attempts across executors are
serialized in worker threads. `poll_s` controls worker subprocess polling.
`cancel_margin_s` adds time after the job's grace period for soft cancellation
and bounds recovery operations, including connection and lock waits. Both
intervals must be finite and positive. Initialization is single-flight per
executor: if a caller stops waiting, the next caller rejoins the same attempt
under its own allowance and can adopt a successful connection. A failed attempt
can be retried by a later caller, after the previous attempt has finished.
The abandoned caller cannot dispatch work after its lifetime ends. `aclose()`
leaves the shared Ray connection open and never calls `ray.shutdown()`.

### Initialization warning

Ray may log `SIGTERM handler is not set because current thread is not the main
thread.` during this initialization. The warning is retained: initialization
stays off the event loop so it cannot block lease renewal or cancellation.

### Dispatch, resources and cancellation

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
Secret values never enter task options or `runtime_env`. See the
[secret transport boundary](#gateway-secret-transport) for driver-supplied
credentials in the task payload.

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
callback. Hard process controls raised in the awaiting task synchronously request
force cancellation and then propagate unchanged, without awaited cleanup. See
[stop modes](limits-and-safety.md#stop-modes-and-resume) and
[hook-containment limits](limits-and-safety.md#local-isolation-and-secrets) for
library-mode Ctrl-C and worker-thread outcomes.

Ray preserves its assigned `CUDA_VISIBLE_DEVICES` in the tool subprocess.
Ray resources are logical scheduling quantities, so subprocess resource limits
still apply where the operating system supports them; a logical memory request
is not an isolation boundary.

### Workspace cleanup

Normal worker exits remove the temporary workspace. If preparation or collection
is still running when the task returns, cleanup waits for worker exit through
`atexit`. A hard exit, such as force cancellation or OOM, can bypass that cleanup
and leave a `swarm-worker-*` directory on the worker. Remove retained directories
only after confirming the worker and its tool processes have stopped.

### Recovery

Recovery uses the connected cluster's `RayContext.dashboard_url` and the
recorded original driver's job ID. `confirm_dead` requires identity-matched
terminal-task and dead-worker evidence. `kill_orphan` waits for Ray's owner
fate sharing and never kills another task itself. Missing, ambiguous, dropped,
unavailable or late state stays unknown. **Recovery confirmation needs a
plain-HTTP dashboard reachable at `RayContext.dashboard_url`. HTTPS, proxied or
`RAY_OVERRIDE_DASHBOARD_URL` dashboards make lost runs stay unconfirmed; their
spend stays estimated.** No environment discovery fallback is used for state
lookups. State requests use the same Ray token-header loader as the Jobs SDK,
within the existing recovery deadline. Token-authenticated clusters can confirm
lost runs when the driver has the matching token and identity-matched evidence
is available. Tokens stay in request headers and are never recorded in events
or errors. If the authentication helper or token is unavailable, rejected State
requests leave recovery unknown; this does not relax the dashboard limits above.

On resume, the committed `tool_dispatch` event supplies the task name and the
original driver's Ray job ID. The resumed driver queries the connected
cluster's State API for that identity and makes one bounded orphan-recovery
attempt per unfinished dispatch. Ray's owner fate sharing cleans up tasks of
the dead driver; resume does not kill an arbitrary matching task. Missing
rows do not prove that no task ran. Recovery records a lost run with
`confirmed_dead=false` when it cannot confirm both task termination and worker
death. Keep the dashboard **and its State API** enabled and reachable from
the driver; a disabled dashboard also leaves recovery unconfirmed.

## What the integration tests establish

The Ray conformance tests compare the Rosenbrock function with local execution
and exercise snapshotted goal/tool inputs, inherited parent overlays, live
edits, declared output files and both logs through the object store. A local
test fixture imports connector code, then denies worker reads of the driver's
artifact root and source/plan directories, and driver reads of worker temporary
workspaces. The tool script also rejects driver-source paths in its request
and environment. Results are read with the public kit
readers after execution. The shipped Rosenbrock plan checks dispatch events,
two finalists and committed candidate logs. Soft cancellation checks the
data-only partial contract, without new artifact publication.

These are local-cluster tests with explicit Python filesystem-access denial.
They do not establish multi-node deployment, remote token provisioning,
worker-image installation or real S3 acceptance. Verify those paths in your
deployment environment.

## The LLM gateway on the cluster

The cluster owner runs LiteLLM Proxy with Postgres, model/provider credentials,
and a master key. Provision `LITELLM_MASTER_KEY` in the **driver's environment**
for key management; never put its value in `run.toml`, the plan, a Ray upload,
or `runtime_env`. Workers need network access to the proxy URL. A loopback URL
works only when the proxy is on the same host as the calling process; use the
cluster-reachable proxy address for distributed workers.

**The pinned LiteLLM 1.107.0rc2 profile provides soft caps and estimated spend
only.** It cannot verify a per-key output-token ceiling, a maximum total request
duration, or required virtual-key use, and its settings introspection is
unavailable. `default_estimated_output_tokens` is an estimate, not an output
limit. With `limits.max_usd`, hard caps are required by default, so explicitly
accept soft caps to run on this version:

```toml
[llm_gateway]
url = "http://litellm.internal:4000"
master_key_env = "LITELLM_MASTER_KEY"
require_hard_cap = false
key_duration_s = 86400
request_timeout_s = 600.0
default_estimated_output_tokens = 4096

[limits]
max_usd = 5.0
max_tool_runs_per_round = 2

[limits.default_run_reservation]
tokens = 100000
usd = 0.5
```

`require_hard_cap = true` fails preflight before key creation on this pinned
version; leaving it unset with `max_usd` has the same effect. The manifest and
candidate trace explicitly show `cap soft`. The gateway is optional and its
client uses the Python standard library; tool-internal calls do not require
the `llm` extra.

A tool that calls an LLM must declare `providers()` (or set
`FunctionTool(..., providers=["openai"])`). Each tool run with declared providers,
or a driver-visible `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` allowlist, gets one
virtual key named `cand-<candidate id>`. Plain tools get no key. For example:

```python
from swarm_arena import FunctionTool

llm_tool = FunctionTool("my_connector.tools:solve", providers=["openai"])
plain_tool = FunctionTool("my_connector.tools:score")
```

A keyed subprocess receives `OPENAI_BASE_URL=<url>/v1`,
`ANTHROPIC_BASE_URL=<url>`, and both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` set
to its virtual key. It must use those base URLs. The base URLs travel in
`Job.env`; bearer values travel in `Job.secret_values`. A keyed job drops
node/file provider credentials before applying these overlays. A dynamic
`Command.env` allowlist alone is invisible to driver key selection: with a
gateway it never grants the real OpenAI or Anthropic key. Declare providers
instead. Other provider credentials are not replaced: keyed jobs do not get
them; an unkeyed job that allowlists one can receive the node value and bypass
the gateway. Keep provider declarations accurate and review allowlists.

### Gateway reservations

A keyed run needs a USD reservation **above zero**, supplied by
`Tool.estimate(request)` or `limits.default_run_reservation.usd`. Its key's
`max_budget` equals that reservation. The engine keeps the **entire reservation
charged**, including failed or cancelled key-open attempts; gateway observations
and reconciliation never refund it or create a second charge. Failures before
the reservation is promoted to `keep_charged` retain ordinary undispatched-refund
behavior. Once promoted, the full reservation remains charged, including
cancellation while that promotion becomes durable, even if key creation has not
started. Tighter, realistic `Tool.estimate()` values and accurate
`Tool.providers()` declarations avoid
reserving unnecessarily large amounts or keying tools that do not use an LLM.
A soft proxy cap does not make `max_usd` a hard bound on actual provider spend.

### Gateway secret transport

The core keeps gateway keys out of journal records, manifests and artifacts.
Tools must also keep credentials out of their own results and logs.
`Job.secret_values` travels inside the Ray task payload, whose object-store
contents can spill to **worker-local disk**. Secrets that must never travel
through that payload belong in the worker environment or `secrets.file`
(subject to that file's own storage policy). Gateway virtual-key transport
itself does not offer a never-on-disk guarantee. Protect worker and object-store
storage, and never put secrets in Ray `runtime_env`.

Configure LiteLLM Proxy's own `cache` and `langfuse` callback settings for
caching and tracing **tool-internal** calls, with their credentials provisioned
on the proxy. The driver's response cache and Langfuse setup do not configure
an arbitrary subprocess or the external proxy. Apply an outbound network rule
that blocks workers from provider hosts while allowing the proxy; otherwise a
tool with a hard-coded provider URL can go around it. The proxy needs its own
provider egress.

### Reconcile gateway spend

Run completion makes a bounded reconciliation attempt. To retry later, use
the addressed run directory and its original gateway configuration. Replace
`example-id` with your run's ID:

```sh
swarm-arena reconcile runs/example-id
# Optional explicit configuration, retaining the original gateway URL:
swarm-arena reconcile runs/example-id --config run.toml
```

Run directories and run roots are **trusted operator data**: anyone who can
write them can make resume/reconcile run tools and send the master key named
by `master_key_env` to the gateway URL in the recorded configuration. Keep run
roots writable only by the operator; on shared storage pass `--config` from a
trusted copy (the continuity check then refuses a changed gateway URL). That
override does not authenticate the journal or recorded plan. Before mutating a
key, reconcile checks the addressed history and the key hash, creation run ID
and alias returned by `/key/info`. Missing or mismatched identity keeps the key
estimated without mutation; these checks do not make writable run storage trusted.

Reconcile takes the run lease and refuses a live owner. It reports each key's
reserved amount and confidence label without changing the ledger. The pinned
adapter remains **estimated**, even after waiting or lowering
`request_timeout_s`; it cannot prove the duration/completeness conditions and
does not delete keys through an observed transition. Recorded calls still
provide informational spend. Other capability-bearing adapters can report
`observed (completeness unverified)`, or `observed` with terminal accounting,
only after confirming tool death, key closure and the required elapsed window.
Deletion also waits for durable observations and closure evidence. Deleted keys
reuse committed observations without another gateway read, so rows arriving
after deletion are not discovered. Unsupported key totals remain estimated;
an aggregate may remain estimated or be unavailable instead of losing precision.
See [After the run: reconcile](limits-and-safety.md#after-the-run-reconcile) for
ownership, timing and accounting limits, including keys created but never journaled
and the `reconcile time allowance exhausted` reason.
