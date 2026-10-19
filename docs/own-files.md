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
`path → sha256` manifest while keeping the original globs. Every glob must match
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

A tool workspace starts with goal and tool files. The first parent's full overlay
is applied only when its nonempty tool identity matches the receiving tool and
every overlay path is editable, unprotected, and outside check or judge private
files. Otherwise the whole overlay is refused with a warning identifying the
parent and the refused paths; an empty overlay needs no warning. Live edits are
applied last. For example, an
editable `problem/*.txt` file can inherit its base parent's contents and then be
replaced by the current request's text. A protected `problem/run.py` remains
protected even if its command path was spelled `./problem/run.py`. Protection
and private-path comparisons normalize Unicode to NFC and then casefold, so
an edit to `problem/RUN.py` is refused as protected too. Editable glob matching
remains case sensitive.

Identity hashes the import path and validated configuration with defaults
expanded, excluding catalog-settable fields. Dictionary path keys must already
be in normal form: aliases such as `./problem/a.txt` or `problem//a.txt` are
refused, including normalization collisions. Edit values are literal UTF-8 text;
the reserved `sha256:` followed by exactly 64 lowercase hex digits is a stored
reference and cannot be submitted as a live edit.

Every parent, including the base parent, has read-only overlay and output files
under `parents/<id>/`. Output files win at a shared path; `data.json` always holds
the parent's structured output. Parent ids must be single safe path components
and must be distinct after NFC normalization and case folding.
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
symlink components and special files, and rejects duplicate normalized paths.
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
fields. Snapshots are prepared separately before staging.

The command runs in the staged workspace and reads its parameters from
`swarm/request.json`. It writes a tool envelope such as
`{"data": {"loss": 0.25}}` to `swarm/result.json`. The shared result rules validate
the envelope, keep valid partial data after timeout or cancellation, and mark a
successful process with a missing or invalid result as an error. `python`, `env`,
`timeout_s`, and `grace_s` are passed through to the command settings.

Every command argument or configured interpreter that names a root file is
protected using its canonical relative path: `problem/run.py`,
`./problem/run.py`, and `problem//run.py` protect the same file. All tool-owned
manifest paths are protected too. These restrictions apply even under a broad
editable glob, both to live edits and to a base parent's overlay.

The tool provides two read-only actions, each counted as a board read:

- `list_files(candidate=None)` returns sorted goal-file paths. Supplying a
  candidate selects its goal files plus effective overlay.
- `read_file(path, candidate=None)` returns that path and its UTF-8 contents from
  the same view. Missing paths and non-UTF-8 contents produce soft errors.

Candidate lookup uses the agent context, so candidates outside its view cannot
be read. Artifact reads run on a worker thread. Neither action exposes arbitrary
filesystem paths or artifact hashes as a way to select additional files.
