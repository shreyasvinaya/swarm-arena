# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `Swarm.run` approves estimates before publishing snapshots, runs queued goals
  with shared budgets and capacity, and returns durable finalists; `Swarm.stop`
  requests graceful or immediate stops, with journaled stop modes and fencing.

- Run preflight checks distinct effective component configurations; versioned run
  manifests record provenance, and epoch-scoped stop requests and committed-journal
  status provide run control without importing user components for status.
- Goal execution supports round barriers, pruning, validated replanning, part
  joins, and final judging, with durable result fingerprints and explicit
  no-result solutions.

- `ScriptedManager` runs ordered baseline requests without a model;
  `testing.check_manager` exercises managers offline through real core actions.

- `AgentContext`, the manager action set (`swarm_arena.manager`).

#### Tools and local execution

- `ScriptTool` runs your own scripts with typed parameters, editable file globs,
  and `list_files` / `read_file` actions for inspecting candidate files.
- `FunctionTool` runs an importable, annotated Python function in a subprocess
  and returns its JSON object as result data.
- `Tool` and `Action` support custom connectors; `Command`, `CompletedRun`,
  `ToolRequest`, `ParentRef`, `ToolOutput`, `RunStatus`, `Usage`, and `Estimate`
  describe invocations, results, and resource use.
- `Executor` and `Job` support execution backends. `LocalExecutor` runs jobs in
  fresh workspaces with process-group cleanup, timeouts, cancellation, resource
  limits, and an environment allowlist. Local execution runs as the same OS user
  and does not provide a security sandbox.
- Tool scripts exchange parameters and finite JSON results through
  `swarm/request.json` and `swarm/result.json`. Results can declare output files,
  usage, and provider failures. A stopped run retains partial output when a valid
  collection marks itself partial or contains data or files.
- Logs retain their last 8 MiB by default; declared output files are limited to
  256 MiB per run by default. Configure these bounds with
  `LocalExecutor.Config.max_log_bytes` and `max_output_bytes`.

#### Goals, files, and configuration

- `validate_plan` checks limits, component configurations, planner provenance,
  frozen goals and scoring, stable addresses, duplicate agents, private files,
  and requested environment names before admitting a revision.

- `Goal` describes a problem and `GoalKey` identifies its inputs and file
  snapshot by SHA-256. `FileSet` snapshots globs and `FileRef` describes an
  output artifact. Each source file is limited to 1 GiB.
- File edits support base-parent overlays and read-only parent views. Workspace
  paths must be relative strict UTF-8, with no symlinks, reserved names, or
  collisions after Unicode normalization and case folding. Scripts and
  human-controlled tool files are protected from edits.
- `Ref` selects components by public import path and validates their configuration.
  `Plan`, `PartSpec`, `TeamSpec`, `AgentSpec`, `RoundSpec`, `ConvergeSpec`, and
  `FlowSpec` describe plans; `Plan` and `RunConfig` load and save TOML.
  Plan orchestration and full swarm runs are not yet available.
- `Limits` and `Resources` describe budgets and execution requirements.
  `ModelCatalog`, `ModelSpec`, `PlannerCatalog`, and `CatalogEntry` describe model
  aliases and permitted planner choices. `CachePolicy`, `GatewayConfig`,
  `SecretsPolicy`, and `TracingConfig` configure run services.
- Configuration diagnostics retain field names and error codes while hiding
  rejected values, defaults, and data-derived mapping keys, including in
  Pydantic error details and tracebacks. Validation reports at most 20 details
  followed by an omitted-error count.

#### Checks, scoring, and records

- `Board` and `BoardView` define journal-backed indexing and fixed snapshot reads;
  an in-memory run board provides ranked results, namespace learning search,
  lineage, deterministic frontiers, and digests with a strict UTF-8 byte cap.

- `Policy`, `KeepTopTeams`, `SuccessiveHalving`, `Patience`, and `Chain` provide
  configurable per-group pruning with starved-agent and last-team protection.
- `Planner` ABC (`swarm_arena.planning`).
- `Manager` ABC, `LimitReached`, and `RoundReport` (`swarm_arena.manager`).

- `Decision`, `RoundState`, and `History` describe policy decisions and planning
  evidence, with ranking isolated by group, judge identity, and newest fidelity.
  `Frontier`, `Row`, and `LineageGraph` describe board snapshots and result ancestry.

- `ScriptCheck` gates candidates with an external script and fails closed on
  invalid results. `ScriptJudge` scores candidates using its own files;
  `ProgramJudge` uses a trusted Python function. `Lexicographic` combines
  ordered judges with fixed-length member keys.
- `Check`, `Judge`, `CheckContext`, and `JudgeContext` support custom evaluators.
  `CheckResult` records a gate decision; `Score` supports minimization and
  maximization with finite ranking keys and judge identity.
- `Candidate`, `Learning`, `LLMCall`, and `AgentStatus` record results, evidence,
  usage, and agent state. Stored records carry schema versions and refuse newer
  unsupported formats.
- `Message`, `Part`, `Artifact`, `Task`, `TaskStatus`, `TaskState`, and `Role`
  provide A2A protocol records without requiring an A2A SDK.

#### Offline development and documentation

- `swarm_arena.testing.check_tool`, `check_check`, `check_judge`, and
  `make_candidate` exercise connectors locally without a manager or provider.
- `swarm_arena.testing.read_output_file` and `read_logs` return the verified
  bytes of a run's declared output files and of its stdout and stderr logs.
- Runnable function and script examples, connector authoring guidance, and a
  public API reference cover local execution, file edits, checks, and scoring.
- `swarm-arena --version` reports the installed package version.
