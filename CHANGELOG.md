# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Consolidate model validation diagnostics: rejected inputs and validator context
  are removed from `errors()` and `json()` as well as text and tracebacks.
  Schema fields, aliases and list/tuple indices remain visible at every depth;
  free-form mapping and `Any` segments use `<…>`. The outermost validated model
  owns the unknown-key policy for nested fields; explicit constructor and
  `model_validate*` calls inside validators use the called model's own policy,
  and contexts copied to another thread establish their own boundary. For
  ambiguous alias or union paths, any segment on which the matching schema
  paths disagree is hidden, string identifiers included. The
  policy is: human plan/settings models show nested typo names,
  while runtime records and component configs use `<field>` for unknown keys.
  Empty locations stay empty, restoring `plan`/`run_config` loader fallbacks and
  removing false nested locations. Custom codes become `invalid`; wrapped built-in
  causes previously labelled `arena_config` now use the built-in code (such as
  `float_parsing`, `bool_parsing`, or `string_type`). Non-validation
  exceptions report `invalid: <TypeName>`. Result errors expose structured locations
  and built-in codes, and result-file reasons use those same locations and markers.
  Fixed safe reasons remain, including the value-free newer-schema reason; catalog
  human-only violations identify both the entry and the field. Compiled parameter
  models also sanitize all validation entry points and retain JSON-schema `$defs`
  and recursive references. Previously unsanitized standalone `Ref`, `Goal`,
  `FileSet`, and other frozen models now use fixed messages and built-in codes,
  redact data keys and hide unknown model keys as `<field>`. Pydantic developer
  errors retain their original type and fix-it hint at every entry point. If
  sanitization itself fails, diagnostics fall back to an empty location with
  type and message `invalid`, without exposing the original failure. Diagnostic
  details are capped at 20 in Pydantic order, plus `and N more errors` when truncated.
- `ToolOutput.data` and script-check `details` must be finite JSON, with fixed
  `data must be finite JSON` / `details must be finite JSON` validation messages;
  invalid connector output is reported as an execution error.

### Fixed

- Workspace paths must be strict UTF-8. A path with a lone surrogate, such as
  the `surrogateescape` spelling of a non-ASCII name, is refused everywhere a
  path is checked (`not valid UTF-8`): it names the same file on disk as the
  real name but folds to another key, so an edit could replace a protected script.
- A tool root whose goal manifest (or tool files) conflict by themselves reports
  `goal files conflict: …` (or `tool files conflict: …`); the respell hint is
  given only when a tool path respells a goal path. A declared output must use
  its on-disk spelling (`declared output <p> is spelled <q> on disk`), as
  snapshots do, so a run that passes on APFS also passes on a case-sensitive disk.
- Every read of a tool-written file is bounded, and a file's size is checked
  before it is read, so a sparse file is never read whole: `swarm/result.json`
  is at most 16 MiB, the files one result declares at most 256 MiB together
  (summed before any is hashed; hashed one at a time), and the executor's
  upload reads stay within `max_output_bytes`.
- A stopped run's output has `partial=True` whenever its collect returned a
  valid `ToolOutput`, even an empty one, also when its files are dropped later
  by `max_output_bytes` or a failed upload; a failed, timed-out or invalid
  collect gives `partial=False` and a note. `tool_output_from_result` notes why
  a stopped run kept no result.
- `LocalExecutor` removes a workspace whose tool locked the workspace root, and
  its fallback remover changes a directory's mode only when an open, unlink or
  rmdir in it is refused.

### Added

- End-to-end offline test of one tool run, checked and judged through the public API.
- Extend the offline exit test to reject invalid candidates and crashed checks,
  upload output files, and verify candidate overlays and parent file staging.
  Document installation from a clone before running the README example.
- `ScriptTool`: bring your own files and script, with `list_files` and `read_file` actions.
  ScriptTool parameters and catalog `settable` schemas use a strict, frozen subset
  compiler with validated defaults, bounded value-free diagnostics, explicit
  protected namespaces, and non-clashing names such as `model_name`. Numeric
  inputs, bounds, enum members and defaults must be finite; integer float bounds
  must be integral int64 values. Enum cannot be combined with bounds or pattern.
  Patterns use Rust regex syntax (no lookaround or backreferences; Unicode-aware
  `\d`).
- `RunConfig` and the `swarm_arena.config` models, with load-time catalog rules.

- Project scaffold: package, `swarm-arena --version`, CI, and docs skeleton.
- `Ref`: components referenced by import path, with validated `Config`.
- `Goal`, `GoalKey`, and the tool-run data types of `swarm_arena.tool`.
- Result types (`Candidate`, `Score`, `CheckResult`, `Learning`, `LLMCall`, `AgentStatus`) and the A2A v1.0.1 types of `swarm_arena.protocol`.
- Internal write-once local storage and SHA-256-addressed artifacts with verified retrieval.
- `Tool` ABC and `Action`; the `swarm/result.json` contract.
- `Plan` and its specs with TOML load and dump, and `Limits` with the §8.1 defaults.
- `Executor` ABC and `Job` (`swarm_arena.executor`).
- `LocalExecutor`: process-group supervision, launcher handshake, env allowlist, rlimits, and logs by hash, with capped log and output uploads (`max_log_bytes`, `max_output_bytes`).
- `Check` and `Judge` ABCs with `CheckContext.execute` and score stamping.
- `FunctionTool`: a module-level Python function run in a subprocess.
- `ProgramJudge` and `Lexicographic` judges.

- `ScriptCheck` and `ScriptJudge`, and the "Use your own files and script" docs page.
- Conformance kit `swarm_arena.testing` and the "Writing a connector" and "Concepts" docs pages.
