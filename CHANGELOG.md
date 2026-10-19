# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Consolidate model validation diagnostics: rejected inputs and validator context
  are removed from `errors()` and `json()` as well as text and tracebacks. Human
  plan/settings models show top-level typo names; other models use `<field>`.
  All deeper locations collapse to `<…>`, custom error codes become `invalid`,
  and result errors expose structured locations and built-in codes. Fixed safe
  validation reasons remain; catalog human-only violations use a fixed message.

### Added

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
