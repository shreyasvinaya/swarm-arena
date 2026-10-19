# Use your own files and script

Written in M1a.

## File snapshots

A goal or component can declare a `FileSet` with globs relative to the plan file:

```toml
[goal]
id = "example"
namespace = "demo"
files = ["problem/*.txt", "problem/run.py"]
```

Snapshotting stores each selected regular file by SHA-256 and records a sorted
`path → sha256` manifest while keeping the original globs. Folded aliases of
the same device/inode are recorded once using the directory listing’s spelling,
including ancestor names. Distinct files whose names collide after NFC/casefold
comparison are refused with both paths named; the manifest must also have a
portable file/directory layout. Every glob must match
at least one regular file. Symlinks below the plan directory and reserved
top-level paths (`swarm`, `parents`, `candidate`) are rejected, including case
variants such as `SWARM` and `Parents`. Those names are
allowed as basenames below an ordinary directory, such as `problem/candidate`.

Once a file set has a manifest, snapshotting reuses it without reading the source
files again. Source edits or deletion therefore cannot change an existing
snapshot. A file set with no globs needs no source files.

The internal snapshot helpers cover the main goal, part goals, tools, managers,
checks, judges, policy, and convergence team. Component configs are validated
before traversing explicit `FileSet` and nested `Ref` fields, including lists of
refs. Omitted defaults stay omitted, unrelated settings keep their original
representation, and the returned plan retains its original base directory.

## Workspace staging

A tool workspace starts with goal and tool files. Tool files replace goal files
at a shared NFC/casefold key and keep the tool file’s spelling. The first
parent's full overlay is applied only when its nonempty tool identity matches the receiving tool and
every overlay path is editable, unprotected, and outside check or judge private
files. Otherwise the whole overlay is refused with a warning identifying the
parent and the refused paths; an empty overlay needs no warning. Live edits are
applied last. For example, an
editable `problem/*.txt` file can inherit its base parent's contents and then be
replaced by the current request's text. A protected `problem/run.py` remains
protected even if its command path was spelled `./problem/run.py`. Protection
and private-path comparisons normalize Unicode to NFC and then casefold, so
an edit to `problem/RUN.py` is refused as protected too. A command spelled
`problem/RUN.py` likewise protects a root file stored as `problem/run.py`.
Editable glob matching
remains case sensitive.

Identity hashes the import path and validated configuration with defaults
expanded, excluding catalog-settable fields. Dictionary path keys must already
be in normal form: aliases such as `./problem/a.txt` or `problem//a.txt` are
refused, including normalization collisions. Edit values are literal UTF-8 text;
the reserved `sha256:` followed by exactly 64 lowercase hex digits is a stored
reference and cannot be submitted as a live edit.

`validate_edits` soft-refuses differently spelled edits sharing an NFC/casefold
key, including ancestor directory aliases and file/directory conflicts. For
example, `P/b.txt` conflicts with root file `p/a.txt`, and file `P` conflicts
with directory `p/`. Pass the existing workspace paths as `root` to also refuse
aliases of existing files; an exact-spelling edit still replaces that file.
Stage planning rechecks edit collisions against goal files, tool files, and the accepted base
overlay before creating the stage. Refusals identify the conflicting paths
without including edit contents.

Every parent, including the base parent, has read-only overlay and output files
under `parents/<id>/`. Output files win at a shared NFC/casefold key, retaining
the output’s spelling, both over parent overlays and over effective candidate
files. `data.json` always holds the parent's structured output; aliases such as
`DATA.json` and `Data.json` are skipped at that level too. Parent ids must be
single safe path components and must be distinct after NFC normalization and case folding.
Checks and judges instead receive their own root files and read-only candidate
files under `candidate/files/`, with request and output metadata beside that
directory. The directory exists even for a candidate with no files.

Materialization revalidates every path before writing, even for model objects
constructed without validation. Before any write, it rejects destination names
or ancestor directories with different spellings but the same NFC/casefold key,
including `a.txt`/`A.txt` and composed/decomposed Unicode names, on every platform.
It also rejects symlinks and file/directory conflicts, reads verified artifact
bytes directly, and publishes through pinned directory descriptors.
Root artifacts are writable; parent and candidate
artifacts are read-only. The workspace's ancestors are trusted executor-owned
directories. Output collection uses the shared regular-file reader, refuses
symlink components and special files. The sink, staging helpers, snapshotter,
and output collectors use one folded-layout checker for files, implicit ancestor
directories, and file/directory conflicts.

`collect_output_files` requires the original `stage` as a keyword-only argument
and checks declarations against its complete layout before uploading any output.
Result parsing uses the same output helper for declarations against each other;
it has no original stage. The executor then supplies `job.stage` for the full
check. Declarations such as `out/a.txt` and `OUT/b.txt`, or output `P/RUN.py`
against staged `p/run.py`, therefore fail the producing run with both paths in
the log. Exact-spelling replacements remain allowed. The original stage is used
even if the process renames or removes a staged file.
These file-handling protections do not provide strong process isolation.

## ScriptTool

`ScriptTool` configures an external command without importing the script or
reading its files during construction:

```python
from swarm_arena import ScriptTool

script = ScriptTool(
    command=["python", "problem/run.py"],
    files=["tool/helper.py"],
    editable=["problem/*.txt"],
    params={"epochs": {"type": "integer", "maximum": 400}},
    timeout_s=120,
)
inputs = script.get_input_model()(epochs=20)
```

Use `get_input_model()` for the instance's parameter schema; the class-level
`input_model` is an empty placeholder. Configuration accepts either a
`ScriptTool.Config` positional argument or keyword fields, and rejects unknown
fields. Invalid settings raise a sanitized `pydantic.ValidationError`; building
through `Ref.build()` instead raises `ConfigError`. Unknown config keys appear as
`<field>`, and nested error locations as `<…>`. Commands must have at least one
argument, `env` contains variable names such as `API_KEY` (matching
`[A-Za-z_][A-Za-z0-9_]*`), `timeout_s` is positive when set, and `grace_s` is
nonnegative. These constraints are checked when the config is constructed.
Snapshots are prepared separately before staging.

Configuration, parameter, and file-action input errors hide rejected values,
including malformed JSON in `model_validate_json()` and the resulting
`ValidationError.errors()` / `.json()` details. For example,
`ScriptTool(command=["x"], timeout_s=-1)` reports the `timeout_s` field and the
`greater_than` error code without the rejected timeout. Invalid parameter
schemas raise `ConfigError` retaining the property path and keyword identifiers,
such as `script_params.lr` and `minimum`, while omitting configured values.

The command runs in the staged workspace and reads its parameters from
`swarm/request.json`. It writes a tool envelope such as
`{"data": {"loss": 0.25}}` to `swarm/result.json`. The shared result rules validate
the envelope, keep valid partial data after timeout or cancellation, and mark a
successful process with a missing or invalid result as an error. `python`, `env`,
`timeout_s`, and `grace_s` are passed through to the command settings.

Every command argument or configured interpreter that names a root file is
protected using its canonical relative path: `problem/run.py`,
`./problem/run.py`, and `problem//run.py` protect the same file. All tool-owned
manifest paths are protected too. Command and interpreter matching compares
NFC/casefold keys and retains the root's stored spelling, including composed
and decomposed Unicode names. These restrictions apply even under a broad
editable glob, both to live edits and to a base parent's overlay.

The tool provides two read-only actions, each counted as a board read:

- `list_files(candidate=None)` returns sorted goal-file paths. Supplying a
  candidate selects its goal files plus effective overlay.
- `read_file(path, candidate=None)` returns that path and its UTF-8 contents from
  the same view. Missing paths and non-UTF-8 contents produce soft errors.

Candidate lookup uses the agent context, so candidates outside its view cannot
be read. Artifact reads run on a worker thread. Neither action exposes arbitrary
filesystem paths or artifact hashes as a way to select additional files.

<!-- Task 22: standalone workspace-contract guide. -->

## Configure files, a tool, a check, and a judge

The design's §4.12 example below connects a physika problem to three scripts.
Its `.phyk` extension is illustrative; globs can select `.physk` files or any
other regular files. No Python subclass is needed. This is a configuration
excerpt: the LLM manager and model alias belong to later milestones, while
M1a supplies the script components and local execution.

```toml
[goal]
id = "ns-viscosity"
namespace = "physika"
description = "Recover rho and nu from the observed flow field."
files = ["problem/*.phyk", "problem/obs.npz", "problem/run.py"]   # snapshotted at plan time

[[teams]]
name = "fd-solver"
rationale = "Finite-difference forward model, tune the inverse step."

  [[teams.agents]]
  name = "tuner-1"
  rationale = "Explore learning rate and step size."
  manager = { path = "swarm_arena.llm:AgentsSDKManager", config = { model = "opus" } }   # alias from run.toml [models]

    [teams.agents.tool]
    path = "swarm_arena:ScriptTool"
    [teams.agents.tool.config]
    command  = ["python", "problem/run.py"]    # cwd = the workspace; goal files keep their paths; "python" = config.python or the current interpreter
    python   = "/opt/envs/physika/bin/python"  # the script's own environment (optional)
    editable = ["problem/*.phyk"]              # which files the manager may change or add (Tool.editable())
    params   = { lr = { type = "number", minimum = 1e-5, maximum = 1.0 }, epochs = { type = "integer", maximum = 400 } }
    timeout_s = 1200
    env = ["ANTHROPIC_API_KEY"]                # allowlist: for a script that calls an LLM itself

[[checks]]
path = "swarm_arena:ScriptCheck"
config = { command = ["python", "checks/check.py"], files = ["checks/check.py"] }

[[judges]]
name = "rel-error"
path = "swarm_arena:ScriptJudge"
config = { command = ["python", "judge/score.py"], files = ["judge/score.py", "judge/hidden/*.npz"], direction = "min" }
```

`ScriptCheck` and `ScriptJudge` accept a positional `Config` or keyword fields,
just like `ScriptTool`. Their configs reject unknown fields and sanitize
validation errors. Commands must be nonempty, timeouts positive, grace periods
nonnegative, and environment entries valid variable names. Both accept
`resources=Resources(...)` for executor scheduling. `preflight()` refuses
unresolved file globs: snapshot the files before executing a check or judge.
An empty file set needs no snapshot. Construction does no file I/O.

```python
from swarm_arena import ScriptCheck, ScriptJudge

check = ScriptCheck(
    command=["python", "checks/check.py"], files=["checks/check.py"], feedback="pass_fail"
)
judge = ScriptJudge(
    command=["python", "judge/score.py"],
    files=["judge/score.py", "judge/hidden/*.npz"],
    direction="min",
)
```

The planner catalog refuses to make `command`, `python`, `files`, or `env`
settable for either component. For example, `timeout_s` can be settable while
the judge's private input files remain human-controlled.

### Tool workspace and protection

A tool starts in a fresh workspace with this layout:

```text
<workspace>/                         # script's working directory
  problem/run.py                     # goal files keep their relative paths
  problem/ns_solver.phyk
  tool/helpers.py                    # optional tool-owned files
  parents/c1/data.json                # parent's structured output
  parents/c1/problem/ns_solver.phyk   # read-only full parent overlay
  parents/c1/out/plot.png             # read-only parent output file
  swarm/request.json                 # core writes this
  swarm/result.json                  # script writes this
  swarm/stdout.log
  swarm/stderr.log
```

Staging starts with goal files and then tool-owned files; a tool-owned file wins
at the same exact path. Next comes the first (base) parent's full overlay, then
the current request's edits. The overlay is accepted only if its nonempty tool
identity matches the receiving tool and every path is editable and unprotected.
An overlay with any refused path is rejected as a whole. The run still proceeds
with a soft warning naming the parent and refused paths. For example, a parent
from another tool remains available under `parents/c1/` but does not replace
`problem/ns_solver.phyk` at the root. A matching parent's editable change can
be inherited and then replaced by the current edit.

Only the paths named in `command` and `python`, and the tool's own `files`, are protected. A helper module your script imports (for example `problem/helpers.py`) can be edited when an `editable` glob matches it. To keep it fixed, list it in the tool's `files` or narrow `editable`.

Check and judge private paths and the core's reserved paths are also excluded
from edits and overlays. Protected names use NFC normalization and casefold;
editable globs remain case sensitive. For example, `./problem/RUN.py` in a
command protects the root's `problem/run.py`, even with `editable=["problem/*"]`.
All workspace paths must be relative, without `..` or symlinks. Top-level
`swarm/`, `parents/`, and `candidate/` are reserved, including folded variants.
For example, `out/plot.png` is a valid output path; `../plot.png` and
`candidate/plot.png` are not. Distinct spellings with the same folded key are
rejected rather than allowed to overwrite each other.

### Request parameters and parents

`swarm/request.json` uses the shape produced by `request_json`:

```json
{
  "schema_version": 1,
  "params": {"lr": 0.01, "epochs": 100},
  "round": 2,
  "round_overrides": {},
  "replicate": 0
}
```

For example, a Python driver reads
`json.load(open("swarm/request.json"))["params"]["epochs"]`.
Inputs appear under `params`; edits have already been applied to workspace
files. The first parent is the base for overlay inheritance. Every parent,
including the first, has a read-only directory under `parents/<id>/` containing
its `data.json`, full overlay copy, and declared output files. If an output file
shares an overlay path, the output wins in that parent directory. `data.json`
is reserved there for structured output. For example,
`json.load(open("parents/c1/data.json"))["loss"]` reads the parent's loss.

### Tool results, exit codes, and partial output

A tool script writes an envelope to `swarm/result.json`. The `data` object is
required. `usage`, `files`, `failed_provider`, and `schema_version` are optional;
`schema_version` defaults to 1.

```json
{
  "schema_version": 1,
  "data": {"loss": 0.25},
  "usage": {"input_tokens": 100, "output_tokens": 20, "usd": 0.01},
  "files": ["out/plot.png"]
}
```

Declared output files are validated and uploaded by hash. Exit code 0 with a
valid result means `ok`; a missing or invalid result, or an ordinary nonzero
exit, means `error`. Diagnostic logs are retained. A valid
`{"data": {}, "failed_provider": "openai"}` gives status `provider_failed`
with that provider at any exit code, except after a timeout, cancellation, or
OOM, which takes precedence.

On timeout or cancellation the process group gets SIGTERM, a `grace_s` wait,
partial collection, and SIGKILL. A valid result written before collection is
kept with `partial=true` and status `timeout` or `cancelled`; without one, data
is empty. Write partial results atomically so collection never sees half a JSON
document:

```python
import json
import os
from pathlib import Path

result = Path(os.environ["SWARM_ARENA_RESULT"])
temporary = result.with_suffix(".tmp")
temporary.write_text(json.dumps({"data": {"loss": 0.25}}))
temporary.replace(result)
```

### Environment and edits

`SWARM_ARENA_RUN_ID`, `SWARM_ARENA_GOAL_ID`, `SWARM_ARENA_CANDIDATE_ID`,
`SWARM_ARENA_TEAM`, `SWARM_ARENA_AGENT`, and `SWARM_ARENA_ROUND` identify the
invocation. Check/judge jobs have empty team and agent values.
`SWARM_ARENA_RESULT` is the absolute result-file path. The leading command
`"python"` becomes the configured `python` interpreter or the current
interpreter. Programs start directly, never through `uv run`.

```python
import os

candidate_id = os.environ["SWARM_ARENA_CANDIDATE_ID"]
result_path = os.environ["SWARM_ARENA_RESULT"]
```

`env=["ANTHROPIC_API_KEY"]` allows that variable into the process; it names the
variable and does not embed its value in the plan. The executor resolves the
allowlist from its configured secrets file or environment.

Edits are UTF-8 text replacements, including additions that match `editable`.
Each edit is limited by `limits.max_edit_bytes`, default **262,144 bytes**.
Protected, private, reserved, and folded-colliding paths are refused. The exact
`sha256:` plus 64 lowercase hexadecimal digits form is reserved for stored
references and cannot be submitted as literal edit text.

```python
# A manager supplies this mapping to its tool-run request.
edits = {"problem/ns_solver.phyk": "# revised solver source\n"}
```

### Check and judge workspaces and result formats

Checks and judges stage their own files at the root and one candidate under
`candidate/`. A judge runs once per candidate, concurrently subject to executor
capacity, with a separate workspace for each run. Private files are not staged
into tool workspaces.

```text
<check-or-judge workspace>/
  judge/score.py                      # or checks/check.py
  judge/hidden/points.csv             # private judge input
  candidate/request.json             # request shape above plus parents: [ids]
  candidate/output.json              # status, partial, data, usage, files, ...
  candidate/files/                    # goal + overlay + output files
  swarm/result.json                  # check or judge writes raw result here
```

`candidate/files/` exists even when empty. Its files and the candidate metadata
are read-only. Outputs win at overlapping effective-file paths. For example,
`json.load(open("candidate/output.json"))["data"]` reads the candidate's output,
and `open("candidate/files/problem/ns_solver.phyk").read()` reads its effective
solver file.

Checks write a raw object, without the tool's `data` envelope:

```python
import json
import os

json.dump(
    {"passed": False, "reasons": ["x too big"], "details": {"x": 12}},
    open(os.environ["SWARM_ARENA_RESULT"], "w"),
)
```

`passed` must be a boolean; `reasons` is a list of strings (default `[]`), and
`details` is an object (default `{}`). Unknown fields are rejected. A crash,
nonzero exit, timeout, missing result, or invalid result fails closed. Invalid
results produce sanitized diagnostics. `feedback="pass_fail"` selects the
feedback policy; the default is `"reasons"`.

Judges likewise write a raw object:

```json
{"value": 0.25, "verdict": "ok", "notes": "relative error"}
```

For `direction="min"`, that value yields key `(-0.25,)`; for `"max"` (the
default), it yields `(0.25,)`. Alternatively,
`{"key": [1, 2], "value": 0.25}` supplies the ranking key directly, unaffected
by direction; larger keys rank higher lexicographically. `verdict` is optional,
`notes` defaults to `""`, and extra fields are ignored. Keys must be nonempty
and finite. Numeric strings and booleans are rejected, and JSON containing
NaN or Infinity fails result collection. Neither a non-null key nor a non-null
value means `result needs key or value`.

A malformed result or script failure records a judge error for that candidate
and leaves the others eligible for scores. Returned scores preserve input
candidate order. Executor exceptions, including lifecycle fencing, propagate
from checks and judges; a judge cancels sibling runs and raises the first
exception directly. Cancellation is not converted to a candidate error.

### Isolation boundary

Local execution is **not strong isolation**. Programs run as the same OS user,
with fresh workspaces, process-group cleanup, an environment allowlist, and
resource limits. Read-only candidate files prevent accidental edits but do not
sandbox a hostile script. For example, use this local workflow with a trusted
`problem/run.py`; a script must be trusted even when its private inputs are
kept out of agents' staged workspaces.
