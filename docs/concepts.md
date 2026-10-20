# Concepts

The current release runs local swarms with manual plans, scripted managers,
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

## Manager

A `Manager` chooses an agent's actions during one round through its
`AgentContext`: run a tool, read candidates or learnings, publish a learning,
save notes, or call a tool action. It returns a `RoundReport`. `ScriptedManager`
executes configured requests without a model. A tool exception becomes an error
candidate; a manager exception ends only that agent's round.

An agent sees the board at the round's recorded frontier plus its own new work.
`best(part=...)` reads only the agent's own group. Candidate and learning reads
automatically populate `read_ids` provenance for subsequent tool runs. Converge
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
and lineage; `BoardView` is a fixed-frontier read view. The engine's in-memory
run board indexes this run's committed journal and is reconstructed on resume.
The journal is the durable authority. There is no cross-run arena in this
release; that is M2. Managers access permitted views through `AgentContext`,
with read caps and a UTF-8-byte-bounded digest. See the [board API](api/board.md).

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
