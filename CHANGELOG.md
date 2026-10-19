# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `ScriptTool` parameter schemas allow non-clashing names such as `model_name`,
  refuse reserved model names and malformed sub-schemas or keyword values,
  and bound default-mismatch diagnostics without exposing configured values.
- Consolidate model validation diagnostics: rejected inputs and validator context
  are removed from `errors()` and `json()` as well as text and tracebacks.
  Schema fields, aliases and list/tuple indices remain visible at every depth;
  free-form mapping and `Any` segments use `<…>`. The outermost validated model
  owns the unknown-key policy: human plan/settings models show nested typo names,
  while runtime records and component configs use `<field>` for unknown keys.
  Empty locations stay empty, restoring `plan`/`run_config` loader fallbacks and
  removing false nested locations. Custom codes become `invalid`; non-validation
  exceptions report `invalid: <TypeName>`. Result errors expose structured locations
  and built-in codes, and result-file reasons use those same locations and markers.
  Fixed safe reasons remain, including the value-free newer-schema reason; catalog
  human-only violations identify both the entry and the field. Compiled parameter
  models also sanitize all validation entry points and retain JSON-schema `$defs`
  and recursive references.
- `ToolOutput.data` and script-check `details` must be finite JSON, with fixed
  `data must be finite JSON` / `details must be finite JSON` validation messages;
  invalid connector output is reported as an execution error.

### Added

- End-to-end offline test of one tool run, checked and judged through the public API.
- Extend the offline exit test to reject invalid candidates and crashed checks,
  upload output files, and verify candidate overlays and parent file staging.
  Document installation from a clone before running the README example.
- `ScriptTool`: bring your own files and script, with `list_files` and `read_file` actions.
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
