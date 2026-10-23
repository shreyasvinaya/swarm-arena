# Use your own files and script

Run tools, checks, and judges locally with your existing scripts. Start with
[Try it offline](#try-it-offline); no Python subclass or model provider is needed.

## File snapshots

A goal or component can declare a `FileSet` with globs relative to the plan file:

```toml
[goal]
id = "example"
namespace = "demo"
files = ["problem/*.txt", "problem/run.py"]
```

Snapshotting stores each selected regular file by SHA-256 and records a sorted
`path → sha256` manifest while keeping the original globs. Each snapshotted file
(goal, tool, check, or judge) may be at most 1 GiB; a larger file is refused by
its size, unread, with an error that names its path. A glob spelling that
only a case-insensitive lookup resolves (an alias of the same device/inode) is
recorded once with the directory listing's spelling, including ancestor names.
Names that both appear in a directory listing are separate entries, even as
hard links to one file, and two such names that collide after canonical caseless
folding are refused with both paths named. Each directory is listed once per
snapshot; the manifest must also have a portable file/directory layout. Every
glob must match at least one regular file. Symlinks below the plan directory and reserved
top-level paths (`swarm`, `parents`, `candidate`) are rejected, including case
variants such as `SWARM` and `Parents`. Those names are
allowed as basenames below an ordinary directory, such as `problem/candidate`.

Once a file set has a manifest, snapshotting reuses it without reading the source
files again. Source edits or deletion therefore cannot change an existing
snapshot. A file set with no globs needs no source files.

In the offline conformance kit, file globs are resolved from the current working
directory. Run the driver from the directory containing your declared files.

(workspace-staging)=
## Workspace staging

A tool workspace starts with goal and tool files. A tool file replaces a goal
file only at the exact same spelling. A tool file that respells a goal file
(the same folded key, another spelling, such as `P/RUN.py` for `p/run.py`) is a
configuration error, raised before any run and naming both paths, so one
workspace works on case-sensitive and case-insensitive filesystems. The first
parent's full overlay is applied only when its nonempty tool identity matches the receiving tool and
every overlay path is editable, unprotected, and outside check or judge private
files, and every path fits the receiving root as an exact-spelling replacement
or a new path (no respelling, no ancestor or file/directory conflict, nothing
below a top-level `data.json/`). Otherwise the whole overlay is refused with a
warning identifying the parent and the refused paths; an empty overlay needs no warning. Live edits are
applied last. For example, an
editable `problem/*.txt` file can inherit its base parent's contents and then be
replaced by the current request's text. A protected `problem/run.py` remains
protected even if its command path was spelled `./problem/run.py`. Protection
and private-path comparisons use canonical caseless folding (NFD, casefold,
then NFC; Unicode D145), so
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

Edit validation soft-refuses differently spelled edits sharing a folded key
(a respelling), including ancestor directory aliases and file/directory conflicts. For
example, `P/b.txt` conflicts with root file `p/a.txt`, and file `P` conflicts
with directory `p/`. An edit below a top-level directory that folds to
`data.json`, such as `Data.json/notes.txt`, is refused too, because later runs
stage the candidate under `parents/<id>/`, where `data.json` holds the parent's
output. An edit to the file `data.json` itself is allowed and is skipped in
parent trees. Aliases of existing workspace files are also refused; an
exact-spelling edit still replaces that file.
Stage planning rechecks edit collisions against goal files, tool files, and the accepted base
overlay before creating the stage. Refusals identify the conflicting paths
without including edit contents.

Every parent, including the base parent, has read-only overlay and output files
under `parents/<id>/`. An output file replaces an overlay or effective file
only at the exact same path. An output that respells a staged path fails the
producing run. `data.json` always holds the parent's structured output; aliases such as
`DATA.json` and `Data.json` are skipped at that level too. Parent ids must be
single safe path components and must be distinct after canonical caseless folding.
Checks and judges instead receive their own root files and read-only candidate
files under `candidate/files/`, with request and output metadata beside that
directory. The directory exists even for a candidate with no files.

Materialization revalidates every path before writing, even for model objects
constructed without validation. Before any write, it rejects destination names
or ancestor directories with different spellings but the same folded key,
including `a.txt`/`A.txt` and composed/decomposed Unicode names, on every platform.
No layer merge accepts a respelling. Merge and layout errors name full
workspace paths such as `parents/c2/p/a.txt` or `candidate/files/p/a.txt`.
It also rejects symlinks and file/directory conflicts and verifies artifact
contents before staging them.
Root artifacts are writable; parent and candidate
artifacts are read-only. The workspace's ancestors are trusted executor-owned
directories. Output collection refuses symlink components and special files.
The same collision rules apply to files and their ancestor directories when
snapshotting, staging, and collecting outputs.

Output collection checks declarations against each other and against the
complete original workspace layout before uploading any output. Declarations
such as `out/a.txt` and `OUT/b.txt`, or output `P/RUN.py`
against staged `p/run.py`, therefore fail the producing run with both paths in
the log. An output below a top-level `data.json/` directory (any case) also
fails the producing run. Exact-spelling replacements remain allowed. The original stage is used
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
`<field>`; nested locations retain schema field names, aliases and list/tuple
indices, while each free-form mapping or `Any` segment becomes `<…>`. Commands must have at least one
argument, `env` contains variable names such as `API_KEY` (matching
`[A-Za-z_][A-Za-z0-9_]*`), `timeout_s` is positive when set, and `grace_s` is
nonnegative. These constraints are checked when the config is constructed.
Each top-level property schema must be a table when the config is constructed.
Parameter names, keywords, and nested `items` schemas are checked on the first
`get_input_model()` call.
Snapshots are prepared separately before staging.

Configuration, parameter, and file-action input errors hide rejected values,
including malformed JSON in `model_validate_json()` and the resulting
`ValidationError.errors()` / `.json()` details. For example,
`ScriptTool(command=["x"], timeout_s=-1)` reports the `timeout_s` field and the
`greater_than` error code without the rejected timeout.

(scripttool-parameters)=
### ScriptTool parameters

Parameter names must match `[A-Za-z][A-Za-z0-9_]*`. Names that shadow `pydantic.BaseModel`
members (for example, `schema`, `dict`, `copy`, or `model_fields`), `Config`, and names
matching the generated model's configured protected namespaces are reserved to prevent clashes
with Pydantic model APIs and configuration. The project's explicit `protected_namespaces`
setting reserves names starting with `model_validate` or `model_dump`; other names such as
`model_name`, `model_id`, `model_type`, and `model_path` are allowed. Every property schema and
every recursive `items` sub-schema must be a table (a mapping in Python); lists, scalar values,
and JSON Schema boolean sub-schemas are unsupported. Numeric bounds must be integers or finite
floats; on integer properties, float bounds must be integral and within the int64 range. Length
bounds must be nonnegative integers, and neither numeric nor length bounds accept booleans;
`description` and `pattern` must be strings. Non-finite numbers are refused in enum members and
defaults, and `number` inputs must be finite. An array without `items` is untyped;
its elements do not undergo numeric validation. An `enum` cannot be combined with numeric bounds
or `pattern`. Patterns use the Rust regex dialect, with no lookaround or backreferences; `\d`
is Unicode-aware. After the config accepts the top-level property tables, invalid
parameter names, keywords, or nested schemas raise `ConfigError` on the first
`get_input_model()` call, with the property path and keyword identifiers while hiding
configured values.

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
folded (canonical caseless) keys and retains the root's stored spelling, including composed
and decomposed Unicode names. These restrictions apply to live edits and to a
base parent's overlay. They protect the named files, but do not prevent
[import shadowing](#tool-workspace-and-protection) through newly added modules.

The tool provides two read-only actions, each counted as a board read:

- `list_files(candidate=None)` returns sorted goal-file paths. Supplying a
  candidate selects its goal files plus effective overlay.
- `read_file(path, candidate=None)` returns that path and its UTF-8 contents from
  the same view. Missing paths and non-UTF-8 contents produce soft errors.

Candidate lookup uses the agent context, so candidates outside its view cannot
be read. Artifact reads run on a worker thread. Neither action exposes arbitrary
filesystem paths or artifact hashes as a way to select additional files.

## Try it offline

Install this checkout in your project environment (Python 3.11+, POSIX):

```sh
uv init --python 3.11 script-demo
cd script-demo
uv add --editable /absolute/path/to/agent-swarm
```

Save the following four files, creating their parent directories as needed.
Run `uv run python try_scripts.py` from `script-demo`. The driver writes the
initial problem file, runs the tool with an edit, checks the candidate's effective
file, and scores its output. The conformance kit snapshots file globs relative
to this current directory and manages a local executor.

### `problem/run.py`

The tool reads parameters and an editable problem file, then writes a result:

```python
import json
import os
from pathlib import Path

params = json.loads(Path("swarm/request.json").read_text())["params"]
value = float(Path("problem/value.physk").read_text())
loss = (value * params["scale"]) ** 2
Path(os.environ["SWARM_ARENA_RESULT"]).write_text(json.dumps({"data": {"loss": loss}}))
```

### `checks/check.py`

The check sees the candidate's edited file and its recorded output:

```python
import json
import os
from pathlib import Path

output = json.loads(Path("candidate/output.json").read_text())
request = json.loads(Path("candidate/request.json").read_text())
value = float(Path("candidate/files/problem/value.physk").read_text())
expected = (value * request["params"]["scale"]) ** 2
passed = output["data"]["loss"] == expected
Path(os.environ["SWARM_ARENA_RESULT"]).write_text(
    json.dumps({"passed": passed, "reasons": [] if passed else ["loss mismatch"]})
)
```

### `judge/score.py`

The judge returns a loss; its configuration below makes smaller values better:

```python
import json
import os
from pathlib import Path

output = json.loads(Path("candidate/output.json").read_text())
Path(os.environ["SWARM_ARENA_RESULT"]).write_text(json.dumps({"value": output["data"]["loss"]}))
```

### `try_scripts.py`

All imports use the public API:

```python
from pathlib import Path

import anyio

from swarm_arena import Goal, ScriptCheck, ScriptJudge, ScriptTool
from swarm_arena.testing import check_check, check_judge, check_tool, make_candidate
from swarm_arena.tool import RunStatus, ToolRequest


async def main() -> None:
    """
    Run, check, and score the local script example.

    Examples
    --------
    Save the companion scripts, then run ``uv run python try_scripts.py``.
    """
    Path("problem/value.physk").write_text("1.0")
    goal = Goal(id="script-demo", namespace="demo", files=["problem/run.py", "problem/*.physk"])
    tool = ScriptTool(
        command=["python", "problem/run.py"],
        editable=["problem/*.physk"],
        params={"scale": {"type": "number", "minimum": 0.0}},
    )
    params = {"scale": 1.0}
    edits = {"problem/value.physk": "0.5"}
    out = await check_tool(tool, params, goal=goal, edits=edits)
    assert out.status is RunStatus.OK
    candidate = make_candidate(
        out, request=ToolRequest(inputs=params, edits=edits), tool=tool, goal_id=goal.id
    )
    check = ScriptCheck(command=["python", "checks/check.py"], files=["checks/check.py"])
    result = await check_check(check, candidate, goal=goal)
    assert result.passed
    judge = ScriptJudge(
        command=["python", "judge/score.py"], files=["judge/score.py"], direction="min"
    )
    [score] = await check_judge(judge, goal, [candidate])
    assert score.key == (-0.25,)
    print(f"{out.status.value}: loss={out.data['loss']}, check={result.passed}, key={score.key}")


if __name__ == "__main__":
    anyio.run(main)
```

Expected output:

```text
ok: loss=0.25, check=True, key=(-0.25,)
```

## Configure files, a tool, a check, and a judge

The configuration example below connects a physika problem to three scripts.
Its `.phyk` extension is illustrative; globs can select `.physk` files or any
other regular files. No Python subclass is needed. This is a configuration
excerpt: the LLM manager and model alias belong to later milestones, while
the current release supplies the script components and local execution.

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

See [Workspace staging](#workspace-staging) for the staging order, exact-spelling
replacement rule, respelling refusal, and overlay acceptance rules. For example,
a parent from another tool remains available under `parents/c1/` but does not replace
`problem/ns_solver.phyk` at the root. A matching parent's editable change can
be inherited and then replaced by the current edit.

Only the paths named in `command` and `python`, and the tool's own `files`, are protected. A helper module your script imports (for example `problem/helpers.py`) can be edited when an `editable` glob matches it. To keep it fixed, list it in the tool's `files` or narrow `editable`.

**Import shadowing:** even when `problem/run.py` is protected, an editable glob
such as `problem/*` lets an agent create `problem/json.py` and replace the
`json` module that script imports. Newly created package `__init__.py` files
can do the same. Protecting existing helpers does not prevent these additions.
The risk applies anywhere on the program's import path, including the working
directory for `python -m` or `python -c`. Prefer globs that exclude code, such
as `editable=["problem/*.physk"]`, and keep trusted scripts and helpers outside
the editable set. `ScriptTool` warns at construction when editable globs can
create Python modules beside a protected script, including protected `.py`
manifest files and the working directory for `python -m run`. It also warns
when rebuilt after snapshotting. The warning leaves execution and edit
permissions unchanged.

Check and judge private paths and the core's reserved paths are also excluded
from edits and overlays. Protected names use canonical caseless folding;
editable globs remain case sensitive. For example, `./problem/RUN.py` in a
command protects the root's `problem/run.py`. Use `editable=["problem/*.physk"]`
to permit problem edits without allowing new Python modules beside that script.
All workspace paths must be relative strict UTF-8 (no lone surrogates), without `..` or symlinks. Top-level
`swarm/`, `parents/`, and `candidate/` are reserved, including folded variants.
For example, `out/plot.png` is a valid output path; `../plot.png` and
`candidate/plot.png` are not. Distinct spellings with the same folded key are
rejected rather than allowed to overwrite each other.

### Request parameters and parents

`swarm/request.json` contains the request parameters and run settings:

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
has the exact path of an overlay file, the output wins in that parent directory.
`data.json` is reserved there for structured output, as a file name and as a
directory name: edits and outputs below a top-level `data.json/` are refused
in the run that makes them. For example,
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
kept with `partial=true` and status `timeout` or `cancelled`, even if its data
is empty. If its declared files are then over `max_output_bytes` or fail to
upload, they are dropped and the status becomes `error`; `partial` stays true.
For a stopped run, a valid collect output is partial if it marks itself partial
or contains data or files. A valid but empty collect output that does not itself
mark partial gives `partial=false`, including a `ScriptTool` or `FunctionTool` with no
valid `result.json`. Failed, timed-out, or invalid collection gives empty data,
`partial=false`, and a note. Write partial results atomically so collection never
sees half a JSON document:

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
variable and does not embed its value in the plan. For custom connectors, use
the public `Command.env` field:

```python
from swarm_arena.tool import Command

command = Command(argv=["python", "problem/run.py"], env=["ANTHROPIC_API_KEY"])
```

The executor resolves each allowlisted name from its environment, falling back
to its configured secrets file when the name is absent. Keep secret values out
of command arguments, request parameters, and plans. A script receives only its
allowlisted names plus the base environment and invocation variables; it can
read a value with `os.environ["ANTHROPIC_API_KEY"]`. The script must also avoid
printing secrets to its own logs.

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
into tool workspaces. If a check or judge runs candidate code, see the
[Isolation boundary](#isolation-boundary) warning below before using private data.

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
are read-only. An output replaces an effective file only at the exact same
path; an output that respells one fails the producing run. For example,
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
`notes` defaults to `""`, and unknown fields are rejected. Keys must be nonempty
and finite. Numeric strings and booleans are rejected, and JSON containing
NaN or Infinity fails result collection. Neither a non-null key nor a non-null
value means `result needs key or value`.

A malformed result or script failure records a judge error for that candidate
and leaves the others eligible for scores. Returned scores preserve input
candidate order. Executor exceptions, including lifecycle fencing, propagate
from checks and judges; a judge cancels sibling runs and raises the first
exception directly. Cancellation is not converted to a candidate error.

(isolation-boundary)=
### Isolation boundary

Local execution is **not strong isolation**. Programs run as the same OS user,
with fresh workspaces, process-group cleanup, an environment allowlist, and
resource limits. Read-only candidate files prevent accidental edits but do not
sandbox a hostile script. For example, use this local workflow with a trusted
`problem/run.py`; a script must be trusted even when its private inputs are
kept out of agents' staged workspaces.

A check or judge that runs candidate code gives that code read access to its
private files, write access to the check or judge's own files and
`swarm/result.json`, and access to its allowlisted environment. For example,
`candidate/files/problem/solve.py` could read or overwrite
`judge/hidden/points.csv` or forge the result file. Read hidden data into memory
first and run candidate code in a separate process and directory, passing only
the inputs it needs and keeping result writing in the trusted judge. These
precautions reduce accidental exposure; a separate process or directory under
the same OS user is still not a security sandbox for hostile code.

### Result format versions and bounds

Check and judge result objects may include `"schema_version": 1`; omitting it
also means version 1. Both reject newer versions and unknown fields, including
misspelled names. The result JSON limit is 1 MiB and 100 nesting levels; declare
larger data as output artifacts. At most 4096 output files may be declared,
including empty files, within `LocalExecutor.Config.max_output_bytes`.
Workspace paths have at most 1024 UTF-8 bytes, 255 bytes per component and 32
components. Code points unassigned in the running Python are refused.
The same limits apply to `FileSet.globs` and `ScriptTool.editable`, including
wildcard components; glob syntax counts towards the byte limits. For example,
`FileSet(globs=["problem/**/*.txt"])` selects nested text files. Snapshotting
supports the full 1024-byte relative-path boundary with literal and recursive
globs, even below a long source directory.

`FunctionTool` is data-only, including timeout and cancellation checkpoints.
Use `ScriptTool` or a custom `Tool` when a function's files must become output
artifacts. Its `env=["ANTHROPIC_API_KEY"]` setting explicitly allows named
variables into the subprocess and remains human-controlled.

## Run it

The bring-your-own-script example in `examples/own_script/`
runs a complete two-round line fit with `ScriptedManager`. From the repository
root, run:

```sh
uv run swarm-arena run --yes examples/own_script/run.toml examples/own_script/plan.toml
```

The completion line is `done: line-fit 1 finalists`. The final slope is `2.0`,
the intercept is `1.0`, and both public and held-out MSE are `0.0`.
Run artifacts live under `.swarm-arena/` relative to your working directory;
the final document is `runs/<run-id>/goals/line-fit/solution.json` beneath it.

The first agent edits the slope in round 1. The second agent uses
`parents = "best"` in round 2 and edits the intercept. Their byte-identical
base tool configs allow the slope overlay to reach the second workspace.
`KeepTopTeams(k=1)` retains the single team and both agents through round 2.
The defaulted `scale` parameter is optional in each scripted request; the tool
reads it from `params` in `swarm/request.json` and scales both coefficients.

```{literalinclude} ../examples/own_script/plan.toml
:language: toml
:caption: plan.toml
```

```{literalinclude} ../examples/own_script/run.toml
:language: toml
:caption: run.toml
```

Only `problem/model_*.txt` is editable. These files contain one finite numeric
TOML assignment each, allowing whitespace and comments without executing code.

```{literalinclude} ../examples/own_script/problem/model_a.txt
:language: toml
:caption: problem/model_a.txt
```

```{literalinclude} ../examples/own_script/problem/model_b.txt
:language: toml
:caption: problem/model_b.txt
```

`problem/points.txt` is public training data: whitespace-separated `x y` pairs,
one per line, with no header. It is snapshotted by the goal glob and protected
from edits by excluding it from the editable glob. The protected tool script
computes public `data.fit_error` using these points.

```{literalinclude} ../examples/own_script/problem/points.txt
:language: text
:caption: problem/points.txt (public)
```

```{literalinclude} ../examples/own_script/problem/run.py
:language: python
:caption: problem/run.py
```

The check reads `candidate/output.json` under `data` and writes
`passed`/`reasons`. The judge reads the coefficients from the same document,
computes MSE independently on its private CSV, and writes `value`. All three
scripts write to `SWARM_ARENA_RESULT` using the existing result formats.

```{literalinclude} ../examples/own_script/checks/check.py
:language: python
:caption: checks/check.py
```

```{literalinclude} ../examples/own_script/judge/score.py
:language: python
:caption: judge/score.py
```

The private fixture `judge/hidden/points.csv` contains an `x,y` header and
held-out points distinct from the public dataset. Only the judge receives
that file and its script; neither is staged into tool workspaces. This is a
staging guarantee for trusted scripts, subject to the
[isolation boundary](#isolation-boundary) above.

Engine candidates publish run-relative log paths under
`goals/line-fit/logs/`. Read those paths beneath the run folder. The conformance
kit's public `read_logs` helper reads content-addressed kit results in the same
Python process, rather than these engine paths.

The example's README also shows the Python API. Its end-to-end test retains
tool workspaces, verifies both overlay hashes and parent lineage, checks that
private fixture bytes never occur in a tool workspace, and verifies the final
solution:

```sh
uv run pytest tests/e2e/test_own_script.py -v
```

(reporting-llm-calls)=
## Reporting LLM calls: `swarm/llm_calls.jsonl`

A cooperating tool can write one UTF-8 JSON object per line in
`swarm/llm_calls.jsonl`, relative to its workspace. Only tool jobs collect this
file; command jobs, checks and judges do not inspect or upload it. For example:

```json
{"model":"openai/gpt-6","usage":{"input_tokens":100,"output_tokens":20},"cost_usd":0.002,"latency_ms":120,"status":"ok","litellm_call_id":"my-call-1"}
{"schema_version":1,"model":"local-model","status":"error","error":"provider failed"}
```

Only `model` is required. The supported fields are:

| Field | Default and meaning |
| --- | --- |
| `schema_version` | `1`; optional, newer versions are refused per line. |
| `model` | Required printable model identifier, at most 256 characters. |
| `provider` | Printable identifier, at most 64 characters; defaults to the prefix before the first `/` in `model`, or `unknown` when there is no prefix. |
| `usage` | Empty usage: nonnegative `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `cache_write_5m_tokens`, `cache_write_1h_tokens`; optional nonnegative `usd`. Cache-write subsets are not counted twice. |
| `cost_usd` | `0`; finite nonnegative cost for this observation. |
| `latency_ms` | `null`; otherwise finite and nonnegative. |
| `status` | `ok`, or `error`. |
| `error` | `null`; the recorded event uses a fixed diagnostic rather than the supplied text. |
| `litellm_call_id` | `null`; otherwise printable and at most 128 characters. The `gw:` prefix is reserved for core gateway observations; such tool-reported rows are skipped as invalid. Deduplicates tool-origin observations within the run, independently of funded manager/judge/planner/curator calls. |
| `request`, `response` | `null`; finite JSON payloads. Missing payloads produce no artifact. Present payloads are stored only when `tracing.store_payloads` is enabled. |

Unknown fields are ignored. Run, candidate, role and ownership metadata always
come from the core. Rows are informational `LLMCall` events with role `tool`;
they do not reserve or settle money and do not join a manager's funded call
list. Continue reporting the tool's aggregate usage in `swarm/result.json`:
that usage drives ordinary tool settlement. Reported rows alone do not establish
complete spend or a hard spending cap. An informational cost or `usage.usd`
above the supported USD range is recorded with `status="error"`, zero
`cost_usd`, absent `usage.usd`, and the fixed diagnostic
`usage_overflow: unsupported informational amount`. It never claims a charge.
Each admitted row is linked immediately, so an interrupted report retains its
already recorded call IDs on the candidate.

Collection parses and uploads at most **16 MiB** and reads at most **10,000
nonempty rows**, counting invalid rows too. Parsing and recording stream one
line at a time. Each line is at most **64 KiB**, excluding its newline; a
longer line is skipped before JSON decoding. Blank lines within that limit are
ignored. Numbers are strict: booleans and numeric strings are not numbers,
and token counts require integers. Invalid JSON, unsupported schemas,
negative counts, nonfinite numbers and invalid payloads or identifiers are
skipped with value-free candidate warnings. At most 20 individual
bad-line warnings are retained, followed by one omitted-warning count and an
optional row-limit warning.

A regular report over 16 MiB and at most **1 GiB** is fully hashed but is not
uploaded or fetched by the driver; its size produces a warning. Above 1 GiB,
collection refuses the file before reading it and returns no reference. Files
with symlink components, nonregular files, unreadable files and files observed
to change during their pinned read also warn without discarding the valid tool
result. An absent file is normal. Core storage failures and ownership loss
still propagate. Invalid private collection diagnostics add one fixed warning
without discarding valid output; diagnostics do not affect output equality.
The driver verifies the declared report size and digest before parsing. Local
reads are bounded before allocation; S3 checks metadata and bounds its body
read. A custom byte-only store can be checked only after its read returns.
`ToolOutput.llm_calls_file`, when present, uses
`path="llm_calls.jsonl"` with its real SHA-256 and size; the reserved `swarm/`
prefix describes the workspace location, not the artifact reference path.

The core supplies `SWARM_ARENA_RUN_ID`, `SWARM_ARENA_GOAL_ID`,
`SWARM_ARENA_CANDIDATE_ID`, `SWARM_ARENA_TEAM`, `SWARM_ARENA_AGENT` and
`SWARM_ARENA_ROUND`. For tool-owned Langfuse traces, use `SWARM_ARENA_RUN_ID`
as `session_id` and `SWARM_ARENA_CANDIDATE_ID` as `metadata.candidate_id`.
`SWARM_ARENA_RESULT` gives the absolute result-file path. These identity
variables help correlate traces; they do not make those traces visible to the
core recorder.

`Candidate.llm_calls_source` describes the available evidence:

- `gateway`: the core keyed this run through its gateway; gateway integration
  takes precedence over tool reports and skips their ingestion.
- `reported`: at least one valid row was supplied, including a row deduplicated
  against a prior report.
- `unknown`: no valid rows, but declared providers or recognized provider
  credential names indicate possible LLM use.
- `none`: no valid rows and neither declaration indicates LLM use.
- `langfuse`: reserved by the model; the core never sets it in v1 because it
  cannot inspect a tool's own Langfuse traces.

Declare providers on LLM tools. Without a gateway, plan validation warns for
agents whose tools declare providers. Credential discovery uses the validated
effective configuration, including `ScriptTool` and `FunctionTool` env
allowlists. A `Command.env` created only inside a worker's `Tool.command()`
cannot be discovered by the driver before dispatch.
