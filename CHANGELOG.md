# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- CollaborativePlanner supports bounded human approval, accumulated feedback and
  catalog-admitted edits in one planning session, with a cancellable terminal
  reviewer and supervised editor cleanup.

- LLMPlanner generates catalog-authorized drafts, freezes human goal identity,
  and preserves human components during autonomous replanning; standalone planning
  sessions meter and journal calls with replayable provenance links in saved plans.

- LLMJudge scores fixed rubrics with mixed-provider panels, repeated anchors,
  survivor identities and metered structured-output fallback; full-call input
  funding is checked before provider dispatch and accounting drains before return.

- Offline manager and judge conformance checks run in isolated, bound LLM scopes with
  registered run setups, verify self-drain and requesting-call provenance,
  force parallel manager tool calls, and accept a model catalog through
  `check_judge(..., config=...)`.

- AgentsSDKManager runs a fresh metered Agents SDK conversation each round,
  exposes authorized context actions and finish_round, honors core limits, and
  drains usage accounting on success, failure, timeout and cancellation.

- Manager actions as SDK function tools, with validated schemas, per-call
  provenance, bounded feedback-redacted results, untrusted board content,
  recoverable soft errors, and terminal controls that stop further model turns.
- OfflineHarness and check_manager classify user-created limit exceptions as
  manager errors while registered limits, including nested groups, end only
  the affected round.

- Metered LLM calls reserve against committed tranche credit, protect per-call
  tracking and output caps, link requested tools to their call, and preserve
  registered budget, fencing and cleanup controls across SDK boundaries.

- LiteLLM call logging with per-run catalog prices, exact cache-lifetime usage,
  conservative failed-call charging, proven response-cache origins, and idempotent
  logger/cache installation under the owning run lifetime.

- Lazy `FakeModel` and `FakeTurn` helpers replay isolated offline LLM
  conversations, capture SDK inputs, and record exact catalog-priced usage
  through the real journal and reservation settlement path.
- Per-call LLM settings with tracking metadata, Claude prompt-cache points,
  per-role response-cache controls, complete probe bypass and replayable cache origins.

- Manager prompts keep system instructions and frozen reference files ahead of round facts;
  shared untrusted-content wrapping removes embedded markers before digest byte capping.
- Model alias, role, capability and price validation for plans, replanners and
  curators; exact cache-lifetime-aware LLM costs, conservative call estimates,
  and LiteLLM or registered offline model construction.

- Optional `llm`, `cache`, and `langfuse` dependencies, an LLM import guard,
  bundled LiteLLM prices, and offline upstream compatibility checks.

- Core LLM call recording with replayable deduplication and cache provenance,
  exact reservation accounting, bounded missing-report cleanup, and supervised
  fresh/resumed run bindings without optional provider dependencies.
- Local `board serve` viewer with search, filters, pagination, evidence and lineage
  pages, and token-protected pin, hide and add curation.
- Offline arena commands export/import human reviews, rank runs and namespaces,
  search and inspect board evidence, curate learnings, trace provenance and reindex caches.
- Human review exports spreadsheet-safe CSV, canonical candidate JSON and escaped
  HTML, imports validated scores under an ended run's lease, and repairs committed
  review objects on resume.
- Round digests show each group's ranking identity and human verdicts, plus
  namespace-wide learning titles and counts under a strict UTF-8 byte cap.
- Immutable per-round Markdown pages show the top ten candidates, new learnings,
  and prune reasons, with historical frontier counts and missing-page resume repair.
- Candidate traces show the committed round-applicable plan, tool usage, checks,
  journal and human scores, and citing learnings without importing producer plugins.
- Learning posts vet visible evidence and refinements before ID allocation,
  require justification for near-duplicates, derive verification from complete
  passing evidence, and include value-free writing guidance on invalid content.
- Optional configurable curators merge round learnings with supervised normalization,
  reserve accounting, and atomic completion receipts for crash-safe recovery.
- Runs default to `ArenaBoard`, with fixed sibling rankings, scored past-run repeat
  hints, and an isolated arena in the offline manager harness. Configured-board
  lifecycle calls preserve safe diagnostics, responsive cancellation and bounded shutdown.
- `ArenaBoard` provides synchronized multi-run frontiers, fixed goal and sibling
  views across group namespaces, current producing-key rankings, hidden-safe
  learning counts, and lazy XDG-aware SQLite/FTS5 caches. Published reads remain
  available during storage refresh; call `frontier()` before reading committed data.
- Arena caches incrementally index committed chains and exact named human objects,
  retain producing identities and frontier-aware collision provenance, and rank
  literal-text learning searches with frontier-local relevance. Reindex rebuilds
  the same logical memberships and refuses to delete caches in use.
- Conditional S3 storage provides uncached listings, raw child-key preservation,
  single-attempt conditional writes with explicit unknown outcomes, per-root
  storage probes, and validation before preflight and resume takeover.
- Write-once arena run pointers and validated human review/curation objects use
  versioned loading, deterministic JSONL bytes, and fixed application order.

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
  the shared arena provides ranked results, namespace learning search,
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
