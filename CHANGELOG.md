# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-27

### Added

- Typed goals, candidates, scores, learnings, protocol records and TOML plans and
  run configurations. Content-addressed file snapshots and catalog-admitted
  component references preserve provenance; stored models carry schema versions
  and refuse newer formats. Configuration diagnostics identify fields without
  exposing rejected values or secrets.
- `ScriptTool` and `FunctionTool` run user scripts and importable Python functions,
  with typed parameters, editable files, parent overlays and protected reference
  data. The workspace contract uses `swarm/request.json` and `swarm/result.json`,
  declared output files, bounded logs and output sizes, provider declarations,
  and validated informational tool LLM calls.
- `LocalExecutor` provides fresh workspaces, environment allowlists, resource
  limits, timeout and cancellation handling, process-group cleanup and partial
  output collection. It runs as the same OS user and is not a security sandbox.
- Script and function judges, script checks, lexicographic scoring and custom
  evaluator interfaces support finite ranking keys, fixed judge identities and
  fail-closed validation. Scores support both minimization and maximization.
- The swarm engine executes queued goals, teams and agents through rounds,
  pruning policies, validated replanning, part joins, convergence and final
  judging. `ScriptedManager` supplies an offline baseline. Approval precedes run
  publication; graceful and immediate stops, fenced ownership and conservative
  resume preserve durable finalists, no-result outcomes and completed work.
- The run journal, leases and epoch-scoped controls provide committed snapshots,
  replay, auditable takeover, repaired result views and crash-safe completion
  receipts. Immutable round pages retain rankings, learnings and prune reasons.
- The budget ledger reserves and settles tool and LLM usage against committed
  credit, tracks deduplicated call provenance and cache origins, drains accounting
  on shutdown and conservatively charges uncertain or lost work. Provider pauses
  retain capacity and recover through uncached probes before fresh rounds.
- `ArenaBoard` is the default board, with fixed goal and sibling views, multi-run
  frontiers, ranking, lineage, learning search and bounded digests. Lazy SQLite
  and FTS5 caches index committed evidence and support safe reindexing. Local and
  optional conditional S3 storage support validated roots and conservative
  unknown-write outcomes; S3 checks live SDK conditional-write capabilities.
- Human review exports spreadsheet-safe CSV, canonical candidate JSON and escaped
  HTML, then imports validated reviewer scores under an ended run's lease.
  Learning curation records evidence, refinements and provenance. The local board
  viewer supports search and evidence pages with token-protected curation writes;
  sharing review files also requires access to the run, arena and artifact stores.
- The CLI validates and estimates plans, runs, stops and resumes swarms, displays
  committed status and version, exports/imports reviews, ranks and traces results,
  serves the board and reindexes caches. Planning supports manual, LLM-generated
  and terminal-reviewed proposals with saved planning-session provenance.
- Optional `llm`, `cache` and `langfuse` integrations provide metered Agents SDK
  managers, rubric judges, catalog-authorized planners and curators, bounded human
  approval, model pricing, response and prompt-cache accounting, and opt-in trace
  links. Provider failures retain survivor identities and replayable completion
  receipts. The supported LiteLLM floor remains `>=1.107.0rc2,<1.108` because
  1.107.0 final is unavailable; the owner must choose whether to ship with that
  release candidate or wait before publishing 0.1.0.
- `RayExecutor`, the `ray` extra and `swarm-arena submit` support resource-aware
  cluster execution and Ray Jobs, checked file transfer, bounded recovery and
  two-step cancellation with data-only partial results. Per-tool LLM gateway keys,
  profile preflight and `swarm-arena reconcile` record cap and spend evidence.
  The pinned LiteLLM profile supports soft caps and estimated reservations only;
  requiring a hard cap fails preflight, and incomplete death evidence stays unknown.
- The offline conformance kit checks tools, checks, judges and managers; builds
  candidates; reads verified output and log bytes; and replays deterministic
  `FakeModel`/`FakeTurn` conversations. Missing optional integrations give explicit
  install guidance while core imports remain independent of extras.
- Runnable documentation covers an offline hello-world run/view/review quickstart,
  Rosenbrock examples, user scripts, connectors, concepts, planning, scoring,
  limits and safety, Ray, the arena, versioning and the public API. Python 3.11–3.13
  on POSIX, typed wheel metadata, Read the Docs configuration and release artifact
  checks support the initial package release.

[Unreleased]: https://github.com/shreyasvinaya/swarm-arena/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shreyasvinaya/swarm-arena/releases/tag/v0.1.0
