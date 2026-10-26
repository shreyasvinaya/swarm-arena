# swarm_arena.tool

```{eval-rst}
.. automodule:: swarm_arena.tool
   :members:
```

## Implement a connector

Subclass `Tool`, declare `name`, `description`, and `input_model`, and implement
`command(request, workspace)` and `collect(request, workspace, run)`. Construction
validates the nested `Config` without performing I/O. Put readiness checks in
`preflight()`. Defaults provide deterministic behavior, no editable files or extra
actions, no provider dependencies, and a SHA-256 fingerprint of canonical result data.

`Action[Inputs]` couples a Pydantic input model with an asynchronous handler taking
the agent context and an instance of `Inputs`. Actions are frozen and keyword-only;
`counts_as="board_read"` charges a board read, while the default `"none"` does not.

## Script request and result files

The staged request contains `schema_version`, `params` (the tool request's inputs),
`round`, `round_overrides`, and `replicate`. A script writes its result to the path
in `SWARM_ARENA_RESULT`, normally `swarm/result.json`:

<!-- docs-test: file write_result.py -->
```python
import json
import os

with open(os.environ["SWARM_ARENA_RESULT"], "w", encoding="utf-8") as result:
    json.dump({"data": {"score": 1.5}, "usage": {"output_tokens": 7}}, result)
```

The envelope requires an object under `data`. Optional fields are `schema_version`
(default 1), `usage`, `files` (workspace-relative output paths), and
`failed_provider` (a nonempty provider name). Unknown fields and non-finite numbers,
including overflowing numeric literals, invalidate the result. Declared files must
exist, avoid reserved workspace names, and be regular files without symlinked
components. The collector hashes their bytes locally.

Every workspace path, such as a declared file, an edit key, or a manifest key, is
relative, in normal form, and strict UTF-8. A name with a lone surrogate, such as
the `surrogateescape` spelling of a non-ASCII name, is refused as
`not valid UTF-8`: it names the same file on disk as the real name, but compares
as another path. So a file whose name on disk is not valid UTF-8 cannot be
snapshotted or declared. A path has at most 1024 UTF-8 bytes, at most 255
bytes per component, and at most 32 components. Characters unassigned in the
running Python's Unicode tables are refused, including newer case pairs that
some filesystems would otherwise fold differently.

Every read of a tool-written file is bounded. Its size is checked before it is
read, so a larger file, even a sparse one, is refused unread:

- `swarm/result.json`: at most 1 MiB, with at most 100 levels of JSON nesting.
  The byte cap also bounds memory used by decoded JSON objects.
- At most 4096 output files may be declared, including empty files. Hashing
  and artifact uploads read files in chunks of at most 1 MiB.
- `LocalExecutor` uploads the last `max_log_bytes` (default 8 MiB) of each log,
  and at most `max_output_bytes` (default 256 MiB) of declared output files per
  run (`LocalExecutor.Config`). A run whose outputs cross that cap gets status
  `error` and no files.

Snapshots of human-authored goal, tool, check, and judge files are bounded too:
at most 1 GiB per file. Snapshot selection supports these path limits even
when the source root makes the full absolute name longer than the operating
system's path limit. Both literal globs and recursive patterns such as `**/*`
can select a file at the 1024-byte relative-path boundary.

`FileSet.globs` and `ScriptTool.editable` apply the same strict UTF-8, byte,
component-count and assigned-character rules to patterns. Glob syntax counts
towards the byte limits; wildcards such as `*`, `?` and `[ab]` remain supported.
For example, `FileSet(globs=["problem/**/*.txt"])` selects text files recursively,
and `ScriptTool(command=["python", "run.py"], editable=["problem/*.txt"])`
permits text edits in one directory.

A successful process requires a valid result; otherwise collection returns `error`
and appends a sanitized reason to stderr. A valid result after timeout or
cancellation, even `{"data": {}}`, keeps data and artifacts with `partial=True`.
Without a valid result, a stopped run keeps empty data with `partial=False`, and
the reason, such as `swarm/result.json is missing`, is appended to stderr. Error
and out-of-memory runs retain only valid reported usage. For successful or error runs,
`failed_provider` produces `provider_failed` with usage but no data or artifacts;
timeout, cancellation, and out-of-memory status take precedence over this field.
Check and judge commands use the raw JSON object rather than this tool envelope.

Timeout selection uses the round override, then the tool's configured timeout,
then the hard limit, always capped at the hard limit. Shutdown grace defaults to
ten seconds. A connector's returned `Command` can tighten these limits: execution
uses the smaller non-`None` timeout from the Command and job, and the smaller
shutdown grace. A `None` timeout imposes no additional bound. Local collection
runs once per job in a daemon thread and has a 30-second bound, including after
normal process exit. Cancellation during collection waits for the original
collect until its existing deadline. `SystemExit` in `command` or `collect`
becomes a job failure; KeyboardInterrupt and cancellation propagate.

Result reads and diagnostic appends refuse special files such as
FIFOs so collection cannot stall waiting for another process to open them.

## Python function tools

`swarm_arena.FunctionTool` returns structured JSON data on success, timeout and
cancellation. Files written in its workspace are never uploaded, even when a
checkpoint result declares them. Use `swarm_arena.ScriptTool` or a custom `Tool`
for output artifacts.

```python
from swarm_arena import FunctionTool

tool = FunctionTool("my_package.tools:solve", env=["ANTHROPIC_API_KEY"], timeout_s=30)
assert tool.config.env == ["ANTHROPIC_API_KEY"]
```

Environment names use the same explicit allowlist as `Command.env`; their
values come from the executor's environment or configured secrets source.
The environment allowlist is human-controlled and cannot be planner-settable.
Timeouts must be positive, grace periods nonnegative, and both finite.
Built-in component constructors report invalid settings as a sanitized Pydantic
`ValidationError`; reference configuration validation reports `ConfigError`.

`ScriptTool` emits a `UserWarning` at construction when an editable glob can
create a Python module beside a protected Python script, including `.py` files
in its snapshotted `files` manifest. The warning also covers `python -m run`
and its working directory, before and after snapshotting. Such modules can
shadow imports; the warning does not change which edits are accepted. Restrict
edits to data formats such as `problem/*.physk`.
