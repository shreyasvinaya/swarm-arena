# Concepts

The current release runs local swarms with manual plans, scripted and LLM managers,
checks, judges, and pruning policies. This page explains how they fit together;
see the [public API](api/swarm_arena.md) for their interfaces.

## Goal

A `Goal` is one hard problem. `id` and `namespace` are lowercase slugs: the
namespace is the problem family (for example `physika`), and the id is the
problem in it. `inputs` holds the structured problem data, and `files` holds
its files as a `FileSet`: globs that are snapshotted into a `path → sha256`
manifest, so that later changes to the source files cannot change the goal.
`title`, `description`, and `metadata` describe the goal for people and
managers.

`goal.key()` gives the goal's `GoalKey`: the namespace, the id, and the
SHA-256 of the canonical inputs and of the file manifest. Two goals with the
same key are the same problem, across runs. The files must be snapshotted
first.

```python
from swarm_arena import Goal

goal = Goal(id="ns-viscosity", namespace="physika", inputs={"grid": 64})
assert goal.key().namespace == "physika"
```

## Tool

A `Tool` says **what** to run; the executor decides **where** it runs. A run
takes a `ToolRequest` (inputs, parents, edits) and gives a `ToolOutput`
(status, data, files, usage). The tool's `command()` and `collect()` run next
to the program, in the job's workspace, and the core never imports the program.
Two built-in tools need no subclass:

- `ScriptTool` runs your own script on your own files, and lets a manager edit
  the files that its `editable` globs name. See
  [Use your own files and script](own-files.md).
- `FunctionTool` runs a module-level Python function in a subprocess. Its
  inputs come from the function's signature, and it must return a JSON object.
  It is data-only: file edits and parent file overlays are not supported.

To write your own tool, see [Writing a connector](writing-a-connector.md).

**Import shadowing:** protecting a Python script does not protect its imports.
An editable glob that permits new `.py` files beside the script lets an agent
create `json.py`, for example, and replace a module the script imports.
The same applies to package `__init__.py` files and other directories on the
program's import path. Prefer globs that exclude Python code, such as
`editable=["problem/*.physk"]`, and keep trusted helper modules in the tool's
own files. See [Tool workspace and protection](own-files.md#tool-workspace-and-protection).

**Tool identity** is the SHA-256 of the snapshotted base tool's import path and
validated config, with defaults filled in (fields that its matched planner
catalog entry may set are left out). Round overrides do not enter this base
identity: they affect execution and the round's fidelity. The offline kit has
no catalog entry, so it excludes no catalog-settable fields. The same base tool
written in TOML leaving out a default and in Python setting it has one identity.
The first parent's overlay (its
edited files) is staged for a new run only if that parent was made by a tool
with the same nonempty identity and the new tool may edit every path in it; otherwise
the run gets a soft warning, and the parent stays readable under
`parents/<id>/`.

## Check

A `Check` runs right after each tool run, so the manager sees the result at
once. A failed check marks the candidate **gated out**: the candidate is kept,
but it is never judged or selected. `feedback` selects what managers see:
`"pass_fail"` shows only passed or failed, and `"reasons"` (the default) also
shows the reasons. `ScriptCheck` runs a script and fails closed: a crash, a
timeout, or a missing result is a failed check.

## Judge

A `Judge` scores a batch of candidates. A `Score.key` is a tuple that is
compared in order, and **a larger key is better**: `Score.maximize(x)` gives
`(x,)` and `Score.minimize(x)` gives `(-x,)`. Every key value must be finite;
a NaN or infinite key is rejected as a judge error for that candidate, and the
other candidates are still judged.

**Judge identity** is the judge's name, its `rubric_hash()` (its import path
and config), and its panel hash. Scores from different judge identities are
never compared. Built-in judges:

- `ProgramJudge` scores each candidate with a trusted Python function.
- `ScriptJudge` scores each candidate with a script and private files, for
  example hidden test data.
- `Lexicographic` combines ordered judges: the first judge's key decides, and
  the next judge breaks ties. Only candidates that every member scored get a
  composite score. Each member must use one key length per batch, even as the sole member.

```python
from swarm_arena import ProgramJudge
from swarm_arena.scoring import Lexicographic

ranking = Lexicographic([ProgramJudge("my_judges:error"), ProgramJudge("my_judges:cost")])
```

## Executor

An `Executor` runs jobs: a tool job (a tool reference and a request) or a
command job (from a check or judge). `LocalExecutor` runs each job on this
host:

- **Workspaces.** Each job gets a fresh directory under the workspace root,
  staged with the goal's files, the tool's own files, the parents, and the
  request. It is removed after the job unless `keep_workspaces` is set.
- **Start handshake.** The executor records the job's start before allowing
  its command to run.
- **Process groups.** Each job runs in its own process group. On a timeout or
  a cancel, the whole group gets SIGTERM, then `grace_s`, then `collect()` for
  the partial output, then SIGKILL, so child processes in the group stop too.
- **Environment allowlist.** The command gets a fresh environment: a few base
  names (such as `PATH` and `HOME`), the names in `Command.env`,
  the `SWARM_ARENA_*` identity variables, and `SWARM_ARENA_RESULT`. Nothing
  else leaks in from the driver.

Set `Command(env=["API_KEY"], ...)` in a connector to request a secret by
variable name. `ScriptTool`, `ScriptCheck`, and `ScriptJudge` expose the same
allowlist as their `env` configuration, as does `FunctionTool`. Tool-requested
names must be in `RunConfig.secrets.allow`; `env` is human-only, including its
validation aliases, and cannot be changed by round overrides.
Keep values in the executor's environment
or a configured secrets file, never in the command arguments or plan.

Logs and output files are uploaded by their SHA-256. Each log keeps at most its
last 8 MiB (`max_log_bytes`), and the output files of one run can hold at most
256 MiB together (`max_output_bytes`); a run that crosses it gets status
`error`. A `LocalExecutor` applies resource limits, but it is not a sandbox:
the tool runs as the same user.

## Artifact storage

Artifacts are stored and verified by the SHA-256 digest of their bytes.

The public kit's `swarm_arena.testing.read_output_file` and `read_logs` read and
verify output/log artifact bytes. See the runnable
[four-file kit example](own-files.md#try-it-offline).

## Managers

A `Manager` chooses an agent's actions during one round through its
`AgentContext`: run a tool, read candidates or learnings, publish a learning,
save notes, or call a tool action. It returns a `RoundReport`. `ScriptedManager`
executes configured requests without a model. A tool exception becomes an error
candidate; an ordinary manager exception ends only that agent's round.
Registered ownership and terminal cleanup failures retain their core stop behavior.

`AgentsSDKManager` uses the optional Agents SDK integration (`uv add
"swarm-arena[llm]"`) to choose these actions. Its `model` is an alias in the
run configuration's model catalog and must permit the `manager` role:

```python
from swarm_arena import Ref
from swarm_arena.manager import Manager

manager = Ref(
    path="swarm_arena.llm:AgentsSDKManager",
    config={"model": "opus", "instructions": "Explore promising learning rates."},
).build(Manager)
assert manager.models() == ("opus",)
```

The corresponding manager line in an agent's TOML table is:

```toml
manager = { path = "swarm_arena.llm:AgentsSDKManager", config = { model = "opus" } }
```

Each round starts a fresh conversation. Stable system instructions include the
learning guide, manager instructions, tool description, goal and selected frozen
reference files. Changing facts—round, fidelity, remaining budget, private notes,
resume history, seed inputs and the fixed board digest—are supplied separately.
Use `save_notes` to retain private state between rounds. Instructions do not
change an agent's tool-and-model fingerprint.

The model calls `finish_round(summary=...)` to end the round immediately; ordinary
final text can also end it. The first accepted finish summary is retained, including
an empty string; later finish calls return a soft tool error. The manager has no structured output schema or hosted
tools. Core limits bound model turns, round duration, action work and spending.
`action_timeout_s` defaults to 60 seconds for actions other than `run_tool`, whose
timing belongs to the core. `parallel_tool_calls` defaults to true, with each action
linked to its originating LLM call. Usage accounting drains before the manager
returns; interrupted calls are conservatively charged without waiting for a late
provider report. Actions run in the manager's task group, so SDK cancellation
cannot discard an admitted tool run. Every exit waits for owned actions before
accounting drains; the core's tool cancellation and grace rules govern in-flight
work, including `run_tool` calls nested inside extra actions. A round stop closes
each action's tool admission and lets admitted runs finish before cancelling the
remaining handler work. An action timeout closes admission and starts graceful
cancellation of its nested runs, waits for their completion and settlement, then
cancels the rest of the handler. If that action deadline prevents the handler
from starting more work, the model receives the action-timeout message and can
continue the round. A concurrent run, round or sibling stop still ends the round,
including a live admission stop that the engine watcher has not yet applied to
the tool registry.
Registered terminal cleanup and ownership loss take precedence over the ordinary
round-time limit. Overflow from an owned action ends this manager; an otherwise
completed report is preserved when a peer overflows the shared budget.
Genuine caller cancellation retains highest precedence. See the [LLM API](api/llm.md).

An agent sees the board at the round's recorded frontier plus its own new work.
`best(part=...)` reads the agent's own group or an explicitly declared sibling
group. Candidate and learning reads automatically populate `read_ids` provenance for subsequent tool runs. Converge
agents get read-only `seed_parents`, prepended to their explicit parents. Check
feedback permissions still apply to those parents. See the
[manager API](api/manager.md) and [scoring guide](scoring-and-review.md).

## Policy

A `Policy` uses `RoundState` to choose which teams or agents to prune after a
round. Its `Decision` retains reasons in the journal. `KeepTopTeams`,
`SuccessiveHalving`, `Patience`, and `Chain` compare results within one group,
judge identity, and newest fidelity. Guards protect starved agents and the
group's last live team. Hard limits and watchdogs can stop work during a round.
See the [exact pruning rules](scoring-and-review.md#pruning-policies) and
[planning API](api/planning.md).

## Board

`Board` is the indexing interface for candidates, scores, learnings, decisions,
and lineage; `BoardView` is a fixed-frontier read view. The engine defaults to
`ArenaBoard`, which combines this run's committed journal with permitted
cross-run arena data. The journal remains the durable authority, and run state
is reconstructed on resume. Managers access permitted views through
`AgentContext`, with read caps and a UTF-8-byte-bounded digest. See the
[arena guide](arena.md) and [board API](api/board.md).

## Swarm

`Swarm(RunConfig(...))` admits, estimates, and runs one or more `Plan` objects
on AnyIO's asyncio backend. It schedules goals within shared agent and budget
limits, runs round barriers and policies, then converges and judges finalists.
`await swarm.run(plans)` returns goal IDs mapped to finalists in final order,
or empty lists for no-result goals. Each goal also has a committed
`solution.json` view with `ok`, `no_result`, or `unjoined` status.

`await swarm.stop()` drains ordinary work and permits convergence;
`await swarm.stop(now=True)` cancels jobs with grace and partial collection,
then selects without a converge team. `await swarm.resume(run_id)` takes a new
ownership epoch and replays committed state. A stopped goal with rounds left
continues without requiring higher limits, but only receives its share of
remaining unallocated work budget, possibly zero. Limit-stopped goals require
changed limits. Later solution views show the latest lifecycle; earlier final
orders and stop reasons remain in the journal. See
[Limits and safety](limits-and-safety.md#stop-modes-and-resume) before takeover
or changing limits, and [Planning modes](planning-modes.md) for runnable plans.

## Models

Install the optional integration with `uv add 'swarm-arena[llm]'`. The human's
`RunConfig` declares the provider models available to LLM components:

```toml
[limits]
max_tokens = 1000000

[models.opus]
model = "anthropic/claude-opus-5-5"
max_tokens = 8000
roles = ["manager", "judge", "planner"]
prompt_cache = { ttl = "5m" }

[models.gpt]
model = "openai/gpt-6"
roles = ["manager", "judge"]

[models.flash]
model = "gemini/gemini-3-flash"
roles = ["judge", "curator"]
prices = { input = 0.3, output = 2.5 }
```

Components declare **aliases only**, such as `opus` or `flash`. Every used alias
must exist in the catalog and allow its caller's role. A run using any model
alias requires `limits.max_tokens`; each model's `max_tokens` remains its output
limit. With `limits.max_usd` set, every used alias must also have a known price.
Unused catalog entries do not require prices or capabilities.

Plan admission authorizes raw component references before importing or building
them. It then validates aliases from the built managers and judges, including
nested judges, part scoring, and convergence. Replanners and curators receive
the same checks before run publication and before resume takes ownership.
A refused resume leaves the epoch, lease and journal unchanged. Startup hooks
run in cancellable daemon workers on both entry paths. Hook-raised interrupts,
cancellations and exception groups become value-free startup failures; genuine
caller cancellation still cancels preparation. Managers require function calling.
Judges, planners, and curators without response-schema support produce a
**two-stage** warning: a free-text answer needs a subsequent formatting call.
On resume, these warnings are retained in the journal's takeover event.
Offline `fake/` models support both capabilities and must be registered before
model construction.

Importing `swarm_arena.llm` registers capability and price checks and sets
`LITELLM_LOCAL_MODEL_COST_MAP=True` before loading LiteLLM. Fallback prices use
its bundled map, so importing does not fetch a changing remote price table.
Explicit catalog `prices` are USD per million tokens and override the provider's
reported response cost. Otherwise an authoritative response cost wins, followed
by the bundled-map calculation. An unpriced model without a dollar cap can
report zero cost when no authoritative response cost is available.

`Usage.input_tokens` excludes cache reads and writes. Explicit prices charge
ordinary input, output, cache reads, and cache writes separately; a missing
cache-read price defaults to input and a missing cache-write price defaults to
twice input. An explicit zero cache price stays zero. Provider fallback pricing
retains the reported five-minute and one-hour write subsets. Unsplit writes
use the largest of those tariffs and twice input, even if the requested TTL
was five minutes. For Sonnet 4.6, 100 five-minute writes and 900 one-hour writes
cost 5,775 micro-dollars, using rates of $3.75 and $6 per million respectively.
All token-price products are added as exact decimal rationals and rounded once
to whole micro-dollars, with ties to even.

Per-call reservation estimates use `ceil(prompt_chars / 3)` input tokens and
the output allowance. Include the complete serialized input, tools, and schema
in `prompt_chars`. Estimated input costs `max(2 * input, cache_write)` per token.
Character-to-token conversion is a heuristic; actual supported costs above the
estimate are charged in full. A two-stage final judge also needs funding for
its formatter's full input and output. An output-only judge allowance does not
establish that both complete calls fit.

## Caching

Provider prompt caching is enabled by default. Stable instructions, tools, the
frozen goal and its reference files precede changing round facts.

| Provider | Prompt-cache behavior |
| --- | --- |
| Claude on `anthropic/`, `bedrock/`, or `vertex_ai/` | `[{"location": "message", "role": "system"}, {"location": "message", "index": -1}]` injection points. TTL defaults to `5m`; `prompt_cache = { ttl = "1h" }` adds an explicit ephemeral one-hour control to each point. |
| OpenAI, DeepSeek, Fireworks/Kimi | Automatic caching of the stable prefix; no injected markers. |
| Gemini | Implicit caching; no injected markers. Explicit context caching is a later option. |

Set `prompt_cache = { enabled = false }` for a model to disable its injected
prompt-cache points. Provider minimum prompt lengths still apply. Prompt-cache
reads and writes retain their separate usage and pricing, including the 5m/1h
write split described above.

Response caching is optional: install `swarm-arena[cache]` as well as the `llm`
extra (`uv add "swarm-arena[llm,cache]"`). Configure `RunConfig.cache` in TOML:

```toml
[cache]
backend = "s3"                       # "disk" | "s3" | "redis"
s3_path = "s3://bucket/swarm-arena/llm-cache"
# disk_dir = ".cache/llm"             # for backend = "disk"
# redis_url_env = "REDIS_URL"          # for backend = "redis"; full redis:// or rediss:// URL
# namespace = "physika"               # optional override; defaults to each goal's namespace
ttl_s = 2592000                       # 30 days

[cache.roles]
judge   = { read = true,  write = true  }
planner = { read = false, write = true  }
manager = { read = false, write = false }
curator = { read = true,  write = true  }
```

The response cache uses `default_off`: each allowed call opts in with independent
read and write switches and its effective namespace. Anchors and the final judge
never read cached responses; their results may still be written when judge writes
are enabled. Streamed and thinking calls never read or write the response cache.
Thinking controls include provider-native Gemini configs and budget fields,
Bedrock reasoning configs and Fireworks template controls, including inside
provider request envelopes. Explicit thinking controls, including Ollama `think`,
bypass response caching.
Response caching fails closed: every supplied SDK field and provider parameter
must be in a pinned allowlist of known non-thinking, non-streaming controls.
This includes `stop` as a string or list of strings and `logit_bias` as a mapping
from token IDs to finite numbers. Invalid shapes disable response caching.
Unknown provider parameters disable both response-cache reads and writes,
including unknown keys nested in provider request envelopes. The request still
reaches the provider with those parameters. Schema properties and string-valued
tracking metadata are payload data, not provider control parameters.
Provider recovery probes bypass both directions. These restrictions take precedence
over caller settings, including preset cache keys and provider extra arguments.

Cache keys include request parameters such as tools, schemas, temperature, seed and
reasoning settings. Tracking metadata is excluded, so reservation and call IDs do
not invalidate cache hits. Each per-call settings object carries string-valued
run, goal, team, agent, round, role, reservation, call and model-alias metadata.
The output limit is capped by the model catalog, including caller aliases such
as `max_completion_tokens`, `max_output_tokens` and native `maxOutputTokens`.
Provider generation envelopes retain the cap when they replace generated settings.
An explicit temperature is sent
with `drop_params=True`, allowing providers that reject that parameter to omit it.

A response-cache hit costs zero. Proven origin links distinguish namespace,
effective cache key and the generating response, including replacements at the
same key. When that evidence is unavailable or ambiguous, the origin stays unset.
The recorder persists the opaque identity for replay even when payload storage
is disabled.

## Provider recovery preparation

When a provider latch is outstanding, fresh-round preparation resolves slots
concurrently before probing and selecting active agents. Healthy peers wait for
the slowest slot to resolve or reach the effective round deadline. After this
bounded preparation, the round has its own execution deadline, so a stuck slot
can make preparation plus execution take about twice the effective round timeout.
See [provider failures](limits-and-safety.md#provider-failures).
