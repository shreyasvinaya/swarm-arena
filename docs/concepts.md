# Concepts

The current release supports local tools, checks, judges, and their data models.
This page explains how they fit together.

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

To write your own tool, see [Writing a connector](writing-a-connector.md).

**Import shadowing:** protecting a Python script does not protect its imports.
An editable glob that permits new `.py` files beside the script lets an agent
create `json.py`, for example, and replace a module the script imports.
The same applies to package `__init__.py` files and other directories on the
program's import path. Prefer globs that exclude Python code, such as
`editable=["problem/*.physk"]`, and keep trusted helper modules in the tool's
own files. See [Tool workspace and protection](own-files.md#tool-workspace-and-protection).

**Tool identity** is the SHA-256 of the tool's import path and its validated
config, with the defaults filled in (fields that a planner catalog may set are
left out). The same tool written two ways, in TOML that leaves out a default
and in Python that sets it, has one identity. The first parent's overlay (its
edited files) is staged for a new run only if that parent was made by a tool
with the same identity and the new tool may edit every path in it; otherwise
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
allowlist as their `env` configuration. Keep values in the executor's environment
or a configured secrets file, never in the command arguments or plan.

Logs and output files are uploaded by their SHA-256. Each log keeps at most its
last 8 MiB (`max_log_bytes`), and the output files of one run can hold at most
256 MiB together (`max_output_bytes`); a run that crosses it gets status
`error`. A `LocalExecutor` applies resource limits, but it is not a sandbox:
the tool runs as the same user.

## Artifact storage

Artifacts are stored and verified by the SHA-256 digest of their bytes.

Manager execution, plan orchestration, and full swarm runs are planned for a
later release.
