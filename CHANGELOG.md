# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- Candidate traces show the committed round-applicable plan, tool usage, checks,
  journal and human scores, and citing learnings without importing producer plugins.
- Arena snapshots resolve owning groups before foreign key matches, keeping
  part candidates, declared siblings, and goal learning scope consistent.
- Arena lineage batches ancestor IDs inside one frontier read scope per namespace,
  avoiding frontier-set rebuilds for each visited candidate.
- Arena views reuse committed replay checkpoints and memoize per-run plan
  resolution, avoiding history replay on unchanged reads and sibling lookups.
- Runs now default to the shared arena board, with fixed sibling rankings and
  scored past-run repeat hints; the offline manager harness uses an isolated arena.

- Arena cache reads release temporary membership and ID selections on outermost
  exit, including failures, while preserving nested-read membership.
- `ArenaBoard` provides synchronized multi-run frontiers, fixed goal and sibling views, current producing-key rankings, hidden-safe learning counts, and lazy XDG-aware local caches; empty human objects retain durable coverage.

- Hide temporary and lock basenames consistently in local and paginated S3 listings.

- Preserve valid raw key characters when opening S3 child stores.

- Preserve unknown S3 write outcomes by disabling automatic conditional-PUT
  retries in s3fs, the SDK, and the underlying HTTP transport.
- Explicit candidate and item ID filters share indexed TEMP sets with frontier
  membership, supporting large selections without SQLite binding limits.

- Learning search and count execute filters, relevance ordering, and pagination
  in SQLite, avoiding full-corpus Python model hydration on selective reads.

- Fixed-frontier cache reads share indexed connection-local membership tables,
  including large human-object frontiers beyond SQLite parameter limits.

- Arena cache sequence indexing keeps junction and newest-segment queries bounded
  as committed history grows, including existing version-1 cache files.

- Open arena caches retain committed replay checkpoints across refreshes, fold
  only missing events, and discard uncommitted state after rollback.
- Learning search tokenizes literal text safely and ranks by frontier-local FTS5
  relevance, curation flags, goal identity, verification, matching tags, and time.
- Arena leaderboard helpers rank one recorded judge identity per selected goal
  group, preserve commit-order ties, and resolve offline judges without plugins.

- Arena cache reads use exact recorded frontiers, immutable producing keys, and
  named human review and curation objects for reproducible snapshot membership.

- Local arena caches incrementally index committed run chains, preserve producing
  part identities across replans, and validate historical frontier coverage.

- Conditional S3 storage with uncached listings, per-root storage probes, and
  validation before preflight and resume takeover.

- Write-once arena run pointers and validated human review/curation objects,
  with versioned loading, deterministic JSONL bytes, and fixed application order.

- Offline Rosenbrock swarm example with a Getting started guide and README quickstart.

- `Swarm.resume` continues committed runs with auditable lease takeover,
  conservative lost-run accounting, restored round state, and repaired result views.

- CLI `run`, `status`, `stop`, and `resume` commands support terminal approval,
  sorted plan directories, committed status, epoch-scoped stops, and saved or
  override resume configuration. Use `swarm-arena run --yes run.toml plans/`
  for unattended runs or `swarm-arena resume runs/<run-id> --force` for takeover.

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

- Bring-your-own-script example fits a line across rounds with inherited edits,
  protected public training points, and private judge data.
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

- Planning, limits and safety, scoring and review, and concepts guides cover
  manual swarm execution, catalog and snapshot trust, exact pruning rules,
  committed budgets, epoch-scoped stops, and stopped-goal continuation.

- `swarm_arena.testing.check_tool`, `check_check`, `check_judge`, and
  `make_candidate` exercise connectors locally without a manager or provider.
- `swarm_arena.testing.read_output_file` and `read_logs` return the verified
  bytes of a run's declared output files and of its stdout and stderr logs.
- Runnable function and script examples, connector authoring guidance, and a
  public API reference cover local execution, file edits, checks, and scoring.
- `swarm-arena --version` reports the installed package version.
