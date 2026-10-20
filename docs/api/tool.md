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
snapshotted or declared.

Every read of a tool-written file is bounded. Its size is checked before it is
read, so a larger file, even a sparse one, is refused unread:

- `swarm/result.json`: at most 16 MiB (`MAX_RESULT_JSON_BYTES`).
- The files one result declares: at most 256 MiB together
  (`MAX_DECLARED_FILES_BYTES`). Their sizes are summed before any is read, and
  they are hashed one at a time.
- `LocalExecutor` uploads the last `max_log_bytes` (default 8 MiB) of each log,
  and at most `max_output_bytes` (default 256 MiB) of declared output files per
  run (`LocalExecutor.Config`). A run whose outputs cross that cap gets status
  `error` and no files.

Snapshots of human-authored goal, tool, check, and judge files are bounded too:
at most 1 GiB per file (`MAX_SNAPSHOT_FILE_BYTES`).

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
ten seconds. Result reads and diagnostic appends refuse special files such as
FIFOs so collection cannot stall waiting for another process to open them.
