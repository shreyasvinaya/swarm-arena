# swarm-arena v1 roadmap

- **Spec (source of truth):** [`docs/superpowers/specs/2026-10-16-swarm-arena-design.md`](../specs/2026-10-16-swarm-arena-design.md), APPROVED r8. Section numbers (§) below refer to it.
- **Date:** 2026-10-18
- **Owner:** Shreyas Vinaya
- **API facts:** `tasks/research/api-*.md`. Facts that the research notes do not cover are marked **[verify]** and are the first task of the milestone that needs them.

## R0. How to use this file

- This roadmap is **binding** for every milestone plan (`docs/superpowers/plans/<date>-swarm-arena-<milestone>.md`). A milestone plan expands one milestone of R2 into TDD tasks. It must use the module paths of R3 and the signatures of R4.
- **References.** `§n` points to the spec. `Rn` points to a section of this roadmap (for example R2.0, R4.13.2).
- **Tags.** A name or value with no tag is copied from the spec. **[plan]** marks a decision this roadmap makes where the spec leaves it open. **[verify]** marks a third-party fact to confirm before use.
- **Changes.** A milestone that needs to change R3 or R4 edits this file in the same commit as the code, and says why in the commit message. A change to a name in spec §9 is not allowed.
- **Public surface rule [plan].** Public modules (R3.1, "facade") hold only imports and `__all__`. All definitions live in private modules (leading underscore). Docs and `__module__` show the public path: each facade sets `obj.__module__ = "swarm_arena.<facade>"` for the names it exports, so autodoc and pickling use the public path.

## R1. Architecture summary

One **driver** process per run loads a human-only `RunConfig` and one `Plan` per goal. It admits every plan revision through `admit_revision` (R4.13.12): a non-human revision first passes the non-importing planner-catalog check, then goal and script files are snapshotted into the content-addressed `artifact_root`, then `validate_plan` runs. It estimates the worst case and runs preflight. Goals run concurrently under one **budget ledger** (run pool, reserve, per-goal final slices, per-round and per-agent shares, committed reservations and LLM tranches). In each round, every active `Manager` acts only through its `AgentContext`. `ctx.run_tool` reserves budget, stages a workspace, and submits a `Job` to an `Executor`: `LocalExecutor` (a `ProcessSupervisor` with a launcher handshake and process-group kill) or `RayExecutor` (a generic task, dispatch intent, two-step cancel). Checks run after each tool run. At the barrier, judges score (with anchors), the `Policy` prunes with reasons, and an optional replanner writes a new revision. Then converge (select or a converge team, with parts joined) and one final-judge batch produce `solution.json`. **All durable state is one run journal**: write-once segments chained back from the head in `control/owner.json`, which is also the run lease. Both are written with conditional writes on a `Store` (local filesystem or S3). Plan files, round pages, and `solution.json` are **views** of committed events, and each records its source head. The **arena** is the set of run journals, plus write-once run pointers and review and curation objects under `arena_root`. `ArenaBoard` reads them by recorded **frontier** into a local SQLite/FTS5 cache. Agents get a capped digest and search on demand. The optional `llm` extra runs every built-in LLM call through the Agents SDK's `LitellmModel`, with one LiteLLM `CustomLogger` that writes `LLMCall` records and settles reservations. The core never imports an optional dependency.

## R2. Milestones

Order: **M1a → M1b → M2 → M3 → M4 → M5**. Each milestone starts from the merged previous one and ends with all its exit criteria green on `main`. Each one is built on branch `m<id>-<slug>` (for example `m1a-foundations`) and merged with a PR.

### R2.0 Common gate (part of every milestone's exit criteria)

```bash
uv sync --all-extras --all-groups
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest                                    # default markers only: no ray, s3, *_live
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

CI (`.github/workflows/ci.yml`) runs the same steps in three jobs [plan]:
1. `core`: Python 3.11, 3.12, 3.13. `uv sync --all-groups` with **no extras**. Lint and pytest, **no ty** (ty checks every import, and the extra-only modules import optional packages). This proves the core imports and passes with no optional dependency. Test modules that need an extra are not collected in this job (`collect_ignore`, R3.2).
2. `extras`: Python 3.11 and 3.13. `uv sync --all-extras --all-groups`. ty, then pytest, including the `s3` marker against moto (from M2) and the `ray` marker against a local Ray cluster (from M4).
3. `docs`: Python 3.13. `uv sync --all-extras --group docs` (autodoc imports `swarm_arena.llm` and `swarm_arena.ray`), then the Sphinx build with `-W`.

Every milestone also updates: the `__all__` snapshot test (`tests/test_public_api.py`) for the names it adds, `CHANGELOG.md` (`Unreleased`), `README.md` where user-facing, and the docs pages named in its deliverables.

### M1a Foundations

- **Scope:** §4.1, §4.2 (non-LLM parts), §4.3, §4.4 (without `LLMJudge`, `HumanReview`), §4.8 (`Executor`, `LocalExecutor`, `ProcessSupervisor`, launcher), §4.12, §9 rules 2 and 3, §11 project setup, §7.3 item 2b environment variables only.
- **Deliverables:**
  1. Scaffold. `uv init --lib --name swarm-arena` (src layout, `py.typed`, `uv_build`). Core dependencies only. `dev` and `docs` dependency groups. `[tool.ruff]`, `[tool.ty]`, and `[tool.pytest.ini_options]` in `pyproject.toml` (R5.3). Console script `swarm-arena = "swarm_arena._cli.main:main"` declared, with a stub that prints the version. `LICENSE` (MIT, Shreyas Vinaya), `README.md` stub (pitch, the tree picture of §1, "under construction"), `CHANGELOG.md` (Keep a Changelog format), `CONTRIBUTING.md` (uv, ruff, ty, TDD, numpydoc, the no-AI-attribution commit rule), `.github/workflows/ci.yml` (R2.0).
  2. Sphinx skeleton: `docs/conf.py` (extensions `sphinx.ext.autodoc`, `numpydoc`, `myst_parser`, theme `sphinx_rtd_theme`, `exclude_patterns = ["_build", "superpowers"]`), `docs/index.md`, and stub pages for all 12 pages of §11, each saying "written in M<n>". API reference pages under `docs/api/` with one `automodule` page per public module that exists.
  3. Data models (`_model/`): `Ref` with loading and `Config` validation, `Goal`, `FileSet`, `GoalKey`, `Plan` and its specs, `FlowSpec` (rejects a non-empty list), `Limits`, `RunConfig`, all `swarm_arena.config` models (structure and load-time rules only: script fields never settable, alias and role shape), `ToolRequest`, `ParentRef`, `ToolOutput`, `FileRef`, `Usage`, `Estimate`, `RunStatus`, `CompletedRun`, `Command`, `Candidate`, `CheckResult`, `Score`, `Learning` (with title and text limits), `LLMCall`, `AgentStatus`, and the A2A types with the `RunStatus → TaskState` map. `schema_version` on every stored model, a migration registry that refuses newer versions, TOML loading (`tomllib`), a deterministic TOML writer, canonical JSON and sha256 helpers, `IdGen` and `new_run_id` (R4.13.5), and the JSON-schema subset compiler (R4.13.0).
  4. Storage, first part: the `Store` ABC with its full signature set (R4.13.2), `LocalStore` with `create`, `read`, `exists`, `list`, `child` (`replace` raises `NotImplementedError` until M1b), `open_store` for local paths, and `ArtifactStore`. Why here: §4.2 and §4.12 snapshot files by sha256 into `artifact_root`, and staging needs it.
  5. Tools: `Tool` ABC, `Action`, `FunctionTool` with `python -m swarm_arena.run_function`, `ScriptTool`, file snapshot (globs to manifest), path rules, reserved names, staging order, the overlay rule with soft warnings, protected paths, `max_edit_bytes`, `materialize`, and the output-file collection.
  6. Execution: `Executor` ABC, `Job`, `ProcessSupervisor` (`start_new_session=True`, SIGTERM to the group, `grace_s`, `before_kill` collect, SIGKILL to the group, all in a shielded `finally`, stdout and stderr to files, rlimits), launcher `python -I -m swarm_arena._launch` (`-I`: the cwd, `PYTHON*` variables, and the user site never reach the launcher's `sys.path`; `os.environ`, and so the exec'd command's env, is unchanged; blocks on a pipe for "go", then `execve`), `LocalExecutor` (resource pool, a leading `"python"` replaced by `Command.python` or `sys.executable` [plan], environment allowlist resolved from the node environment and `RunConfig.secrets.file`, then `Job.secret_values` (R4.7), the `SWARM_ARENA_*` variables of §7.3 item 2b, logs uploaded to `artifact_root` for transport with `logs_ref = "sha256:<out>:<err>"`; the run-folder copies of §7.1 are written by the core in M1b, R4.2), and `procinfo` (host id, process start time, group liveness).
  7. Scoring: `Check`, `Judge`, `CheckContext.execute`, `JudgeContext.execute`, score stamping (ids from `IdGen`, `rubric_hash`, `panel_hash`), rejection of non-finite keys as a judge error, `ProgramJudge`, `Lexicographic`, `ScriptCheck`, `ScriptJudge` (one workspace per candidate, `candidate/` layout).
  8. `swarm_arena.testing`: `check_tool`, `check_check`, `check_judge`, `make_candidate`. Ids come from a local `IdGen(run_id="local", epoch=0)`.
  9. Facades with the M1a names (R3.1), `tests/test_public_api.py`, `tests/test_docstrings.py`, `tests/test_import_isolation.py`.
  10. Docs: "Concepts" (draft: Goal, Tool, Check, Judge, Executor), "Use your own files and script" (the full workspace contract of §4.12), "Writing a connector" (Tool, Check, Judge, conformance kit).
- **Exit criteria:** the common gate, plus
  ```bash
  uv run pytest tests/e2e/test_single_run.py -v
  ```
  This test uses only public imports. It runs a `FunctionTool` (Rosenbrock `gradient_descent`) and a `ScriptTool` (with an edit and a same-identity base parent overlay; the parent is made with `make_candidate(..., tool=tool)`, R4.12) through `swarm_arena.testing.check_tool` on `LocalExecutor`. It checks each output with a `ScriptCheck`, and scores it with a `ScriptJudge` (hidden file) and a `ProgramJudge` combined by `Lexicographic`. It runs offline. Also green: `tests/_exec/test_supervisor.py::test_grandchildren_killed_on_cancel`, `::test_partial_collect_after_timeout`, and `tests/_exec/test_local.py::test_tool_never_runs_when_start_commit_fails`.
- **Dependencies:** none.
- **Must NOT:** journal, lease, ledger, `Swarm`, policies, `validate_plan`, CLI subcommands, any network access, any optional extra (no `[project.optional-dependencies]` entries yet), a `Board` ABC.

### M1b Engine

- **Scope:** §4.5 (without `AgentsSDKManager`), §4.6 (`Planner` ABC, `History`, provenance, `validate_plan` without model capability checks), §4.7, §4.9 (`Board`, `BoardView`, `Frontier`, `Row`, `LineageGraph` ABCs and types only), §4.10, §4.11, §5, §6.1, §6.2, §6.4, §6.5, §7.1, §8.1, §8.2, §8.3, §12.
- **Deliverables:**
  1. `Store.replace` on `LocalStore` (lock file, version check, atomic rename) and the error set (`AlreadyExists`, `PreconditionFailed`, `NotFound`, `WriteOutcomeUnknown`, `ConditionalWritesUnsupported`). `open_store` refuses a URL scheme with no conditional writes.
  2. Run journal: `Event` and event types (R4.14), segment encoding with the link line, `OwnerDoc` and `Lease` (acquire with conditional create, renew every `lease_ttl_s / 3`, take over with one conditional replace, release, fence on an ETag mismatch), `JournalWriter` (one writer task, one commit at a time, write-through with a 200 ms group commit, the unknown-result re-read, sha256 check on segment-name reuse), `read_chain` with the chain check, `ViewWriter` (conditional create, same-bytes match, replace only for a later source head on the committed chain), and `RunState` replay.
  3. Budget: `Amount`, `Ledger` (run pool, `reserve_fraction`, final slices, `goal_weights`, return of unused goal budget, per-round and per-agent shares, durable reservations and settlements, idempotent settle, `settle_timeout_s` → `estimated`, conservative replay), `Tranche`, worst-case estimate (§5 step 2), and the `llm_scope` seam (R4.13.7) that M3 uses.
  4. Agents: `Manager`, `AgentContext` (every action of §4.5, soft errors as `ValueError`, `LimitReached`, the board-read cap, the learning rate limit, notes with `max_notes_bytes`, parallel-safe counters), the repeat rules (§6.2), the watchdog (§8.2), extra-action dispatch, `ScriptedManager`, and `testing.check_manager`.
  5. Policies: `Policy`, `RoundState`, `Decision`, `KeepTopTeams`, `SuccessiveHalving`, `Patience`, `Chain`, `NO_RESULT`, per-group ranking, never prune a part's last team, never prune a starved agent for lack of results. Each group ranks under its own ranking judge's identity (R4.5). Tests `tests/_model/test_planning_types.py::test_round_state_ranks_each_group_by_its_own_identity` and `tests/_engine/test_goal.py::test_parts_with_own_ranking_judges_prune_and_join`.
  6. `validate_plan` with every non-LLM rule of §4.6: limits, the catalog and provenance diff (§4.1), `Config` validation, rationales, `secrets.allow`, effective tool config after overrides, parts with no team, `max_agents_total`, the frozen main `GoalKey`, ever-present plan-level checks, the duplicate-fingerprint rule, stable addresses (§4.2: an agent or team address that an earlier revision had and a later one dropped never comes back), and judge-private files (§4.4, §4.12): in every revision, a goal, part-goal, or tool file path that is also a check or judge file path is rejected, and in a non-human revision every part-goal file must be a `(path, sha256)` entry of the main goal's snapshotted manifest. Tests include `tests/_planning/test_validate.py::test_address_not_reused`, `::test_judge_file_in_goal_rejected`, and `::test_non_human_part_files_outside_main_manifest_rejected`. The catalog check runs before any import (`authorize_revision`, `admit_revision`, R4.13.12) and compares in `catalog_form`. Tests also include `tests/_planning/test_catalog_check.py::test_noncatalog_module_never_imported`, `::test_snapshotted_site_matches_raw_entry`, `::test_planner_written_manifest_dropped`, `::test_nested_ref_defaults_match_after_snapshot`, `::test_committed_manifest_kept_for_unchanged_site`, `tests/_planning/test_validate.py::test_catalog_script_tool_valid_after_snapshot_and_in_later_revision` (it deletes or changes the source file before r2, and replays history before r2), and `::test_catalog_chain_and_lexicographic_admitted_and_roundtrip`.
  7. Board ABCs and types, the internal `RunBoard` (this run only, in memory, substring search; fed by `put_*` with event positions and rebuilt from `RunState` on resume, R4.6), the frontier (marks for this run only), and digest v1 (leaderboard top `k`, pinned and newest titles, counts, `digest_tokens` cap (UTF-8 bytes, R4.13.14)).
  8. Engine: `Swarm.run`, `Swarm.resume`, `Swarm.stop`. The estimate step (§5 step 2: the worst-case estimate and its warnings journaled in `run_started`, a warning when peak resource demand is more than `Executor.capacity()`, and the `approve` callback, R4.1). The import rule for `Plan.base_dir` (R4.1). Tool logs written once to `goals/<goal-id>/logs/<candidate-id>.{out,err}` (R4.2). The goal queue with goal slots and `max_agents_total`, rounds, the three clocks, mid-round stops (§6.4, without the provider rules, which come in M3), barrier judging with anchors, policy, the replanner hook with core-assigned provenance (exercised with a test-only `Planner`), parts (assignment, ranking groups, keep the best earlier candidate, join, `max_join_attempts`), converge `select` and `team`, the final judge, `solution.json` (replicate statistics, manifest), `NO_RESULT`, both stop modes (stop files per epoch, R4.13.13), resume (takeover, replay, `lost` runs, orphan kill, after the host and start-time checks, for tool runs and command jobs, `limits_changed`), `manifest.json`, preflight once per distinct tool config, and A2A ids on every event.
  9. CLI: `swarm-arena run [--yes]`, `status`, `stop [--now]`, `resume [--config] [--force]`. `run` prints the worst-case estimate and its warnings, then asks for approval (§5 step 2, manual mode) unless `--yes` is given. With no TTY and no `--yes`, it exits with code 2 and writes nothing.
  10. Examples `examples/rosenbrock/` and `examples/own_script/` (stdlib only), each with a `run.toml` and a `plan.toml`.
  11. Docs: "Getting started" (Rosenbrock), "Planning modes" (manual), "Limits and safety", "Scoring and review" (checks, judges, policies, final judge). README offline quickstart (draft).
- **Exit criteria:** the common gate, plus
  ```bash
  uv run pytest tests/e2e -v        # all offline e2e tests below
  uv run swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml
  uv run swarm-arena run --yes examples/own_script/run.toml examples/own_script/plan.toml
  ```
  Required e2e tests: `test_rosenbrock.py` (2 teams, `SuccessiveHalving`, select top 2, `solution.json` valid), `test_own_script.py` (edits, base-parent overlay across agents, hidden judge data never in a tool workspace), `test_many_goals_parts.py` (2 goals with shared budget, one goal with 2 parts and a join), `test_cli_lifecycle.py` (run in a subprocess, then `stop`, then `resume` to completion, `status` at each step; the resumed owner ignores the earlier epoch's stop file, and a second `stop` on the resumed run succeeds), `test_crash_resume.py` (SIGKILL the driver mid-round, then `resume`: committed runs are not repeated, uncommitted runs are `lost`, reservations are charged in full, the orphan group is killed), and `test_fencing.py` (a forced takeover fences the old owner, and its next conditional write fails).
- **Dependencies:** M1a.
- **Must NOT:** S3, SQLite or FTS5, arena pointers, review or curation objects, `ArenaBoard`, `Curator`, any LLM code, Ray, the gateway.

### M2 Arena & board

- **Scope:** §4.4 `HumanReview`, §4.9, §6.3, §6.3.1, §6.6, §7 (S3 storage), §7.1 "what readers read", §7.2, §7.3 item 4 (non-LLM steps).
- **Deliverables:**
  1. **[verify] task 1:** s3fs or aiobotocore conditional writes. Check `If-None-Match: *` for create and `If-Match: <etag>` for replace, and the error codes on a mismatch (412 or 409), against the s3fs version that matches the locked `fsspec`. Also check whether moto supports them; if it does not, the `s3` tests use a MinIO service container. Record the result in `tasks/research/api-s3-conditional.md`.
  2. `S3Store` (extra `s3 = ["s3fs>=<floor>", "aiobotocore>=<floor>"]` [plan]; floors from task 1: the oldest releases whose botocore `PutObject` model has `IfNoneMatch` and `IfMatch`). `S3Store` refuses to write when the installed botocore lacks either parameter. Storage that fails the conditional-write probe is refused: `check_storage` (R4.13.13) probes each distinct `s3://` root at preflight, and again in `resume`, `review import` (after its CSV pre-check), and `reconcile`, before they take over `owner.json` (they can run with another host or endpoint than the driver).
  3. `ArenaBoard` (becomes the default when `RunConfig.board` is `None`, and `RunBoard` is deleted): run pointers written at run start, the per-run authority read of `owner.json`, incremental chain download into the SQLite cache keyed by `(run, epoch, seq)`, multi-run frontiers committed at each goal-round start, its own run indexed from committed chain segments like any other run (its `put_*` rows are hints that `snapshot` never returns, R4.6), `snapshot` filtered exactly by frontier, review and curation objects applied in the fixed order (name timestamp, then object name), arena human scores taken from `reviews/` only, and an item id that the cache already holds from another run is skipped with a warning (R4.0).
  4. Search: an FTS5 table over learning title, text, and tags. `search_learnings` orders by bm25, then by the relevance order of §6.3.
  5. Learnings rules: evidence must be in the agent's view, the duplicate check (an FTS5 near-match above a threshold [plan]: bm25 top hit plus title token Jaccard ≥ 0.6) that requires `refines` or `duplicate_reason`, `verified`, hidden learnings never shown to agents, and the "from other agents, unverified" markers in the digest.
  6. `Curator` ABC and the engine hook at round end (paid from the reserve). No LLM curator.
  7. Digest v2: the full relevance order over the whole namespace, the leaderboard with the judge identity, and the token cap (counted as UTF-8 bytes, an upper bound on tokens, R4.13.14 [plan]).
  8. Leaderboard (one judge identity per group: the group's own ranking judge, or the `--by` judge for every group; the newest identity when there is no current run, `min_improvement` for first place only, ties to the earliest) and round pages (`goals/<goal-id>/rounds/round-NN.md` views).
  9. Human review: `HumanReview.export` (`review.csv`, one JSON file per candidate, a static `index.html`) and `HumanReview.import_csv` (takes the run lease, fails while a live owner holds it, writes the run copy and one `reviews/` object). The end-of-run engine hook (§5 step 8, §6.6) writes the export of every goal (top 20) to `<run_root>/<run-id>/review/` [plan], through the `Store` with conditional creates, after the final commit.
  10. Preflight: refuse a local `run_root` with an `s3://` `arena_root`.
  11. CLI: `review export|import`, `leaderboard`, `board serve|search|show`, `learnings pin|hide|add` (human learning ids `L-h-<16 hex>`, R4.0), `trace` (plan revision, tool run, checks, scores, learnings), `reindex`.
  12. Board viewer (`board serve`): stdlib `http.server` on the SQLite cache, with search, filters, pages, a learning page, a candidate page with lineage, and the pin, hide, and add actions writing curation objects.
  13. Docs: "The arena", "Scoring and review" (human review part).
- **Exit criteria:** the common gate, plus
  ```bash
  uv run pytest -m s3 -v                          # S3Store against moto or MinIO (per task 1)
  uv run pytest tests/e2e/test_arena_two_runs.py tests/e2e/test_reindex.py tests/e2e/test_review_roundtrip.py -v
  ```
  `test_arena_two_runs.py`: two concurrent swarms in one namespace see each other's learnings with a delay of at most one round, and no file is written by two writers. `test_reindex.py`: `reindex` on a fresh cache rebuilds byte-identical snapshot membership from each recorded frontier, also after a run pointer is deleted. `test_review_roundtrip.py`: export, edit the CSV, import, then `leaderboard --by human:alice` re-ranks. A CLI smoke test starts `board serve` on a free port and fetches `/`, `/learning/<id>`, and `/candidate/<id>`.
- **Dependencies:** M1b.
- **Must NOT:** LLM components (`LLMCurator` is M3), Ray, the gateway. The journal and event formats change only by adding event types or optional fields (a format change needs a `schema_version` bump and a migration).

### M3 LLM layer (extras `llm`, `cache`, `langfuse`)

- **Scope:** §4.5 `AgentsSDKManager`, §4.4 `LLMJudge` and panels, §4.6 (`LLMPlanner`, `CollaborativePlanner`, the autonomous replanner, the planner-catalog boundary for LLM output, core-assigned provenance), §4.13, §6.3 `LLMCurator`, §6.4 provider rules, §7.3 items 1 and 3, §7.3 item 4 (LLM steps), §9 rule 4 (fake model).
- **Deliverables:**
  1. **[verify] task 1:** re-check the facts in `api-agents-sdk-litellm.md` and `api-litellm-caching.md` against the locked versions: the `CustomLogger` hook names and their `kwargs` and `response_obj` fields, `litellm.utils.supports_prompt_caching`, `supports_function_calling`, `supports_response_schema`, `extra_args` pass-through, and `ModelSettings.metadata`.
  2. Extras: `llm = ["openai-agents[litellm]>=0.24,<0.25", "litellm>=1.105.1,<1.106"]`, `cache = ["litellm[caching,proxy]>=1.105.1,<1.106"]`, `langfuse = ["langfuse>=5,<6", "openinference-instrumentation-openai-agents"]`. ruff `TID251` bans these imports outside `swarm_arena/llm/` (R5.3).
  3. Model catalog: aliases only, roles, capability validation through the `validate_plan` hook, the price requirement when `max_usd` is set, `max_tokens` required when any LLM component is used, `LITELLM_LOCAL_MODEL_COST_MAP=True` set by `swarm_arena.llm` before it imports `litellm`, catalog `prices` overrides, and the estimate step extended with manager tokens times price and the LLM judge amounts: `LLMJudge.models()` lists one alias per panel member, so `judge_amount` and `final_slice_estimate` (R4.13.6) price the final judge and converge-team managers. `swarm_arena.llm` calls `register_price_source` at import (R4.13.6).
  4. `LitellmModel` wiring: one `ModelSettings` per call (`max_tokens`, `include_usage=True`, `metadata` with `reservation_id`), the Claude-only `cache_control_injection_points` with the TTL from the catalog, and the prompt layout (stable part first).
  5. `CallLogger` (LiteLLM `CustomLogger`), which only maps LiteLLM's `kwargs` and `response_obj` to the core `record_call(..., settle=True)` (`_journal/llm_record.py`, R4.13.19), and `record_call` itself: every success or failure becomes an `LLMCall` event, payloads are stored by sha256 when `tracing.store_payloads` is true, the reservation settles by `reservation_id`, and a cache hit costs 0 and records `cached_from_call_id`.
  6. Response cache: `Cache(type=…, mode="default_off")` from `CachePolicy`, per-call `use-cache`, `no-cache`, `no-store`, and `namespace` in `extra_args`, anchors and the final judge always `no-cache`, no response cache for streamed or thinking calls, and a dict JSON schema for judges.
  7. `AgentsSDKManager`: one `@function_tool` per action, `context=deps`, `finish_round(summary)`, `max_turns`, the `RunHooks` pre-check against the tranche, `BudgetExceeded`, `fail_after` around `Runner.run`, `run_tool` with `timeout=None`, other actions with `timeout_behavior="error_as_result"`, untrusted-content markers, and `tool_call_id` → `LLMCall.id`: the action wrapper enters the call-id seam (R4.13.7) before it calls `ctx.run_tool` or `ctx.post_learning`, so M1b code sets `requested_by_call` and `posted_by_call`.
  8. `LLMJudge` (rubric with anchors, batches, panels with mixed providers, `aggregate`, `panel_hash`, drop of a failed member, two-stage structured-output fallback), `LLMPlanner` (catalog entries shown with their `settable` schemas, output schema without provenance fields), `CollaborativePlanner` with `Reviewer`, `Approve`, `Feedback`, `Edited`, the terminal reviewer (`llm/_terminal_reviewer.py`), both planners stamping `type(self).origin` before they return, an `Edited` plan included (R4.5), the autonomous replanner (join-failure replans), and `LLMCurator`.
  9. Provider rules of §6.4: `provider_failed` stops matching agents (the provider is `ToolOutput.failed_provider`, or every provider in `Tool.providers()` when it is `None`, R4.2), ranking-judge pause, panel drop, planner or curator skip, and a probe call at the next round start.
  10. Langfuse export: `OpenAIAgentsInstrumentor`, `propagate_attributes`, `LLMCall.trace_url`. Nothing is sent until it is configured.
  11. `trace` and the board viewer gain LLM steps and per-agent cache hit rates.
  12. CLI: `swarm-arena plan` (manual: validate and print the estimate; `--llm` and `--together`: create a plan file that keeps the stamped `origin`, `llm` or `together`). `_cli` imports `swarm_arena.llm` only inside those two handlers (R4.13.18). `tests/_cli/test_plan_cmd.py` checks that `swarm-arena --help` and manual `swarm-arena plan` load no optional dependency (a `sys.modules` check; the `core` CI job also runs them with no extras).
  13. `swarm_arena.testing.FakeModel` and `FakeTurn`. `check_manager` runs with `parallel_tool_calls=True`.
  14. Docs: "Planning modes" (LLM, together, autonomous), "Concepts" (models, caching), "Scoring and review" (LLM judges), "Limits and safety" (tokens, USD, tranches).
- **Exit criteria:** the common gate, plus
  ```bash
  uv run pytest tests/llm tests/e2e/test_rosenbrock_llm.py tests/e2e/test_autonomous_replan.py tests/e2e/test_llm_plan_file.py tests/e2e/test_llm_final_slice.py -v
  ```
  All offline, with `FakeModel`. `test_rosenbrock_llm.py`: the Rosenbrock plan with `AgentsSDKManager` and an `LLMJudge` panel. Every LLM call has an `LLMCall` event, the ledger equals the sum of `LLMCall.cost_usd`, and `requested_by_call` is set. `test_autonomous_replan.py`: a fake replanner output that adds a non-catalog ref (`_fakes.import_canary:Tool`) is rejected and that module is never imported (`'_fakes.import_canary' not in sys.modules`), one that adds a part with `files = ["judge/hidden/*.npz"]` is rejected, and one that keeps the human r0 `ScriptJudge` is accepted. `test_llm_plan_file.py`: `swarm-arena plan --llm` writes `origin = "llm"`, and `swarm-arena run --yes` rejects that file after a non-catalog ref (`_fakes.import_canary:Tool`) is put in it, and the canary's import marker file is never created. `test_llm_final_slice.py`: a capped run (`max_tokens = 1_000_000`, `max_usd = 5.0`, `default_run_reservation = {tokens = 0, usd = 0.0}`) with an `LLMJudge` as ranking judge and an `AgentsSDKManager` converge team ends with an ok solution and no `reserve_exhausted`. `tests/test_import_isolation.py` still shows that `import swarm_arena` loads no `agents`, `litellm`, or `langfuse`. Manual check only (not in CI): `uv run pytest -m llm_live`.
- **Dependencies:** M1b (actions, ledger, tranches, `llm_scope`), M2 (digest, search, `Curator`).
- **Must NOT:** the gateway, Ray, tool-internal LLM accounting beyond the `ToolOutput.usage` that M1b already settles. No change to the public signatures of M1a to M2, except one additive keyword: `check_judge(..., config: RunConfig | None = None)` (R4.12; an `LLMJudge` needs a model catalog and an `LLMScope` to run offline, §9 rule 4).

### M4 Cluster

- **Scope:** §4.8 `RayExecutor`, §7.3 item 2 (a gateway, b `llm_calls.jsonl` and Langfuse ids, c `unknown`), §8.1 gateway tool runs, §11 secrets and Ray mechanics, §9 rule 4 completion.
- **Deliverables:**
  1. **[verify] task 1:** the LiteLLM Proxy 1.105.x setting and endpoint names for the strict profile (budget reservations on, fail closed on stale spend, max request duration, batch write interval, virtual keys required, per-key `max_budget` and `default_estimated_output_tokens`) and for `/key/generate`, `/key/block`, `/key/info`, `/key/delete`, `/spend/logs/v2`, `/model/info`, and the config read endpoint. Record them in `tasks/research/api-litellm-gateway.md`.
  2. Extra `ray = ["ray[default]>=2.59,<3"]`. `RayExecutor`: `ray.init(address="auto")` through `anyio.to_thread.run_sync`, the generic task (`max_retries=0`, `max_calls=1`, tool in the worker's process group, `p.wait(timeout=0.2)` poll, soft cancel with SIGTERM, `grace_s`, and partial collect), the dispatch-intent event with the job id as the task name, the cancel sequence in a shielded scope, artifacts through S3 by sha256 (or through the object store with a 256 MiB cap when `artifact_root` is local), resource mapping, the `schedule_timeout_s` failure, and `confirm_dead` and `kill_orphan` through `ray.util.state`.
  3. `swarm-arena submit` (wraps `JobSubmissionClient.submit_job`).
  4. Cluster secrets: never in `runtime_env`; secret values read on the worker from its environment or `secrets.file`; a custom resource (for example `{"llm_keys": 1}`) documented for tools that need keys.
  5. Gateway client (`_gateway/`, stdlib `urllib`): the strict-profile preflight and `require_hard_cap`, one virtual key per tool run that may call an LLM, environment replacement of provider keys (base URLs in `Job.env`, the virtual key in `Job.secret_values`, R4.7), a `validate_plan` warning for a provider key that the gateway does not replace, the full reservation kept for the whole run, spend read back after `settle_timeout_s` and again at round end, run end, and resume, recorded with `record_call(..., settle=False)` (idempotent by LiteLLM call id, R4.13.19), `terminal_accounting = False`, and the hard or soft cap label in the manifest and on the board.
  6. `swarm-arena reconcile` (takes the run lease; the three conditions of §7.3; `observed (completeness unverified)` labels; the key deleted after its spend is recorded). The driver also runs it at run end, best-effort.
  7. Cooperating tools: collection of `swarm/llm_calls.jsonl` (rows recorded with `record_call(..., settle=False)`), `llm_calls_source`, and the `validate_plan` warning when a tool declares `providers()` and there is no gateway.
  8. Conformance kit completion: `check_tool(..., executor=RayExecutor(...))`, connector CI instructions.
  9. Docs: "Running on Ray" (driver as a Ray Job, token auth, images, VMs as worker nodes, the gateway, the network rule for provider hosts; secret values in `Job.secret_values` (the gateway virtual keys) travel in the task payload, which Ray's object store can spill to worker-local disk; secrets that must never touch disk come from the worker's environment or `secrets.file`).
- **Exit criteria:** the common gate, plus
  ```bash
  uv run pytest -m "ray and not s3" -v             # local Ray cluster started by a session fixture
  # A command-line -m replaces the addopts expression. The one ray + s3 test needs an S3 endpoint that the Ray
  # workers can reach, so it runs apart, with -m "ray and s3".
  uv run pytest tests/_gateway tests/e2e/test_gateway_run.py -v   # against the stdlib fake gateway in tests/_fakes
  ```
  Required: `test_ray_cancel.py` (a soft cancel gives a partial output; a force cancel kills grandchildren), `test_ray_dispatch_crash.py` (driver killed after the dispatch intent, then `resume` finds the task by name, and the run is `lost`), `tests/e2e/test_rosenbrock_ray.py`, and `test_gateway_run.py` (key per LLM tool run only, full reservation held, reconcile labels). Manual check only: `uv run pytest -m gateway_live` against a real LiteLLM Proxy.
- **Dependencies:** M2 (S3 artifacts, lease for reconcile), M3 (`record_call` in the core module `_journal/llm_record.py`, so the gateway client needs no `llm` extra).
- **Must NOT:** add public names outside `swarm_arena.ray` (the `GatewayConfig` fields are already in R4.8), except two fields: `ToolOutput.llm_calls_file` (the reported-calls file must travel from the executor or Ray worker to the driver, §4.3) and `Job.drop_env` (`build_env` runs on the executor side and must not resolve real provider keys for a keyed run, §7.3 item 2a step 2); change the LLM layer beyond the shared recording path; add container or SSH executors.

### M5 Release 0.1.0

- **Scope:** §9 (all rules), §11 docs pages, repo files, CI, releases.
- **Deliverables:** all 12 docs pages of §11 complete, with runnable examples; the API reference through autodoc and numpydoc for every public module; README with the pitch, the picture, and the offline quickstart; `CHANGELOG.md` entry for 0.1.0; the "Versioning policy" page (§9 rules 1 to 6); `.readthedocs.yaml` (from `api-anyio-packaging.md` §B, but it installs all extras with the `docs` group, as the `docs` CI job does; the RTD key for uv extras is [verify], with the fallback `build.jobs` running `uv sync --all-extras --group docs`); `.github/workflows/release.yml` (build with `uv build`, publish with PyPI trusted publishing, `environment: pypi`, `id-token: write`); `.github/workflows/upstream.yml` (weekly cron, `uv pip install` with an override file that lifts the upper bounds of `openai-agents` and `litellm`, then `pytest tests/llm`); `__all__` snapshot tests that equal §9 exactly for every public module; `RELEASING.md` (release checklist); version `0.1.0` in `pyproject.toml`.
- **Exit criteria:** the common gate, plus
  ```bash
  uv run pytest tests/test_public_api.py tests/test_docstrings.py -v
  uv build
  uv run --with twine twine check dist/*
  uv venv /tmp/sa-wheel && uv pip install --python /tmp/sa-wheel dist/swarm_arena-0.1.0-py3-none-any.whl
  /tmp/sa-wheel/bin/python -c "import swarm_arena; print(swarm_arena.__version__)"
  /tmp/sa-wheel/bin/swarm-arena --help
  ```
  Then the README quickstart runs offline in that venv. The upload to PyPI (a tag push) needs the owner's go-ahead, because it reaches an audience and cannot be undone.
- **Dependencies:** M1a to M4.
- **Must NOT:** add features or public names. Only docs, packaging, tests, and fixes.

### R2.7 Scope adjustments and why

1. **Store ABC and `LocalStore` create/read move to M1a** (the task gave all storage to M1b). §4.2 and §4.12 snapshot files into `artifact_root` by sha256, and staging reads them. `replace` and the lease stay in M1b.
2. **Local secrets resolution moves to M1a.** `LocalExecutor` cannot run an env-allowlisted tool (§4.12 example) without it. M4 keeps the cluster rules (§11).
3. **The `SWARM_ARENA_*` tool environment variables (§7.3 item 2b) move to M1a.** They are part of the workspace contract (`SWARM_ARENA_RESULT`). Collecting `llm_calls.jsonl` stays in M4.
4. **The catalog and provenance rules of `validate_plan`, the replanner hook, and provenance stamping move to M1b.** They are non-LLM rules (§4.1, §4.6) and part of the round loop (§5). They are tested with a test-only `Planner`. M3 adds the LLM planners on top and re-tests the boundary end to end.
5. **`Curator` ABC and its engine hook are in M2**, with curation order. M3 adds only `LLMCurator`.
6. **`swarm-arena plan` is in M3.** Its main use is LLM and together planning. Manual validation is also offered there.
7. **`IdGen` is in M1a** (`_model/ids.py`). M1a score stamping and command-job ids (`CheckContext.execute`) need it.

## R3. Module map

Rules [plan]:
- One responsibility per file. A file that grows past about 400 lines is split by responsibility in the milestone that grows it.
- Tests mirror `src/swarm_arena/`: the module `src/swarm_arena/<path>/<name>.py` is tested in `tests/<path>/test_<name>.py`, with a leading underscore dropped from the file name only (`llm/_judge.py` → `tests/llm/test_judge.py`; `_model/ref.py` → `tests/_model/test_ref.py`). Test directories have no `__init__.py`. pytest runs with `--import-mode=importlib`. `pythonpath = ["tests"]` (R5.3) puts `tests/` on `sys.path`, so `tests/ray/` can be a namespace package for `import ray`. With Ray installed, the real package wins. Without it, `import ray` and `find_spec("ray")` succeed on that directory. So extras are found by distribution (`importlib.metadata.version`), and a test that needs Ray skips with `pytest.importorskip("swarm_arena.ray")`, never `importorskip("ray")`.
- "M" is the milestone that creates the file. "+Mn" marks a later milestone that extends it.

### R3.1 `src/swarm_arena/`

| Path | Responsibility | M | Test file |
|---|---|---|---|
| `__init__.py` | Facade: top-level `__all__` (§9) and `__version__` | M1a +M1b | `tests/test_public_api.py` |
| `py.typed` | PEP 561 marker | M1a | — |
| `tool.py` | Facade `swarm_arena.tool` | M1a | `tests/test_public_api.py` |
| `scoring.py` | Facade `swarm_arena.scoring` | M1a +M2 | `tests/test_public_api.py` |
| `manager.py` | Facade `swarm_arena.manager` | M1a +M1b | `tests/test_public_api.py` |
| `planning.py` | Facade `swarm_arena.planning` | M1a +M1b | `tests/test_public_api.py` |
| `board.py` | Facade `swarm_arena.board` | M1a +M1b +M2 | `tests/test_public_api.py` |
| `executor.py` | Facade `swarm_arena.executor` | M1a | `tests/test_public_api.py` |
| `config.py` | Facade `swarm_arena.config` | M1a | `tests/test_public_api.py` |
| `protocol.py` | Facade `swarm_arena.protocol` | M1a | `tests/test_public_api.py` |
| `run_function.py` | Entry `python -m swarm_arena.run_function`: import the function, read `swarm/request.json`, call it, write `swarm/result.json`. Name fixed by §4.1; no `__all__` | M1a | `tests/test_run_function.py` |
| `_launch.py` | Entry `python -I -m swarm_arena._launch`: block on the "go" pipe, then `os.execve` the tool command; exit 0 with no exec if the pipe closes first | M1a | `tests/test_launch.py` |
| `_errors.py` | Private exception classes (R4.13.1) | M1a +M1b | `tests/test_errors.py` |
| `_version.py` | `__version__` from `importlib.metadata` | M1a | `tests/test_public_api.py` |
| `_facade.py` | `publish()`: sets `__module__` of exported names, their nested classes, and their methods to their facade (R0), so doctest collects method examples | M1a | `tests/test_public_api.py` |
| `_model/__init__.py` | Empty | M1a | — |
| `_model/base.py` | `FrozenModel`, `StoredModel`, `JsonValue`, the `Origin` and role literals | M1a | `tests/_model/test_base.py` |
| `_model/compat.py` | Isolated Pydantic implementation assumptions | M1a | `tests/_model/test_compat.py` |
| `_model/data.py` | Finite JSON and depth bounds, `Sha256`, manifest validation; shared type-aware `json_equal` used by catalog matching and provenance | M1a | `tests/_model/test_data_validation.py` |
| `_model/diagnostics.py` | Transportable sanitized validation errors and `validation_summary` | M1a | `tests/_model/test_validation_safety.py` |
| `_model/safe.py` | Shared model validation and error sanitization boundaries | M1a | `tests/_model/test_validation_boundaries.py`, `tests/_model/test_validation_safety.py` |
| `_model/locations.py` | Schema-aware identifier-only validation locations | M1a | `tests/_model/test_validation_locations.py` |
| `_model/canonical.py` | Canonical JSON, sha256 helpers, UTC time format | M1a | `tests/_model/test_canonical.py` |
| `_model/migrate.py` | `schema_version` migration registry; refuses newer versions | M1a +M2 | `tests/_model/test_migrate.py` |
| `_model/toml_io.py` | TOML read (`tomllib`) and a deterministic TOML writer for plan and run views | M1a | `tests/_model/test_toml_io.py` |
| `_model/schema.py` | JSON-schema subset → pydantic model compiler (`params`, `settable`) | M1a | `tests/_model/test_schema.py` |
| `_model/ref.py` | `Ref`, `Configurable`, import-path loading, `Config` validation, instance → `Ref` | M1a | `tests/_model/test_ref.py` |
| `_model/goal.py` | `Goal`, `FileSet`, `GoalKey` | M1a | `tests/_model/test_goal.py` |
| `_model/plan.py` | `Plan`, `PartSpec`, `TeamSpec`, `AgentSpec`, `RoundSpec`, `ConvergeSpec`, `Resources`, `FlowSpec`; `Plan.load` and `Plan.dumps` | M1a +M3 | `tests/_model/test_plan.py` |
| `_model/limits.py` | `Limits` with §8.1 defaults | M1a | `tests/_model/test_limits.py` |
| `_model/settings.py` | `SecretsPolicy`, `TracingConfig`, `GatewayConfig`, `CachePolicy` | M1a | `tests/_model/test_settings.py` |
| `_model/catalog.py` | `ModelSpec`, `ModelCatalog`, `CatalogEntry`, `PlannerCatalog` (load-time rules) | M1a | `tests/_model/test_catalog.py` |
| `_model/run_config.py` | `RunConfig`, `RunConfig.load` and `dumps` | M1a | `tests/_model/test_run_config.py` |
| `_model/tooling.py` | `RunStatus`, `Usage`, `Estimate`, `FileRef`, `ParentRef`, `ToolRequest`, `ToolOutput`, `Command`, `CompletedRun`; `format_logs_ref`, `parse_logs_ref` | M1a | `tests/_model/test_tooling.py` |
| `_model/results.py` | `Candidate`, `CheckResult`, `Score`, `Learning`, `LLMCall`, `AgentStatus` | M1a | `tests/_model/test_results.py` |
| `_model/protocol.py` | A2A v1.0.1 frozen dataclasses and the `RunStatus → TaskState` map | M1a | `tests/_model/test_protocol.py` |
| `_model/ids.py` | `IdGen`, `run_tag`, `human_learning_id`, `new_run_id` (R4.13.5) | M1a | `tests/_model/test_ids.py` |
| `_model/board_types.py` | `Frontier`, `Row`, `LineageGraph` | M1b | `tests/_model/test_board_types.py` |
| `_model/planning_types.py` | `RoundState`, `Decision`, `History`, `RoundReport` | M1b | `tests/_model/test_planning_types.py` |
| `_store/__init__.py` | Empty | M1a | — |
| `_store/base.py` | `Store` ABC, `Blob`, `open_store` | M1a +M1b +M2 | `tests/_store/test_base.py` |
| `_store/local.py` | `LocalStore`: `O_EXCL` create, lock file + atomic rename replace | M1a +M1b | `tests/_store/test_local.py` |
| `_store/artifacts.py` | `ArtifactStore`: content-addressed put, get, fetch | M1a | `tests/_store/test_artifacts.py` |
| `_store/s3.py` | `S3Store` on s3fs with conditional writes | M2 | `tests/_store/test_s3.py` (marker `s3`) |
| `_tooling/__init__.py` | Empty | M1a | — |
| `_tooling/tool.py` | `Tool` ABC, `Action`, shared effective tool configuration | M1a +M1b | `tests/_tooling/test_tool.py` |
| `_tooling/function_tool.py` | `FunctionTool`; input model from the function signature | M1a | `tests/_tooling/test_function_tool.py` |
| `_tooling/script_tool.py` | `ScriptTool`; `list_files` and `read_file` actions | M1a | `tests/_tooling/test_script_tool.py` |
| `_tooling/shadowing.py` | Editable-glob Python import-shadowing witnesses | M1a | `tests/_tooling/test_script_tool.py` |
| `_tooling/files.py` | Glob snapshot of `FileSet`s, of one `Ref` (`snapshot_ref`), and of a whole plan into `artifact_root` | M1a | `tests/_tooling/test_files.py` |
| `_tooling/paths.py` | Path rules, reserved names, `editable` glob matching | M1a | `tests/_tooling/test_paths.py` |
| `_tooling/staging.py` | Staging order, overlay rule, protected paths, `tool_identity`, `fixed_config`, script-workspace `candidate/` plan | M1a | `tests/_tooling/test_staging.py` |
| `_tooling/materialize.py` | Write a `Job.Stage` into a directory; collect output files | M1a | `tests/_tooling/test_materialize.py` |
| `_tooling/result_file.py` | Parse and validate `swarm/result.json`; write `swarm/request.json`; `NOTE_PREFIX`, `MAX_DECLARED_FILES`, collection byte-budget context | M1a | `tests/_tooling/test_result_file.py` |
| `_tooling/llm_calls.py` | Collect `swarm/llm_calls.jsonl`; set `llm_calls_source` | M4 | `tests/_tooling/test_llm_calls.py` |
| `_exec/__init__.py` | Empty | M1a | — |
| `_exec/executor.py` | `Executor` ABC, `Job`, `Job.Stage`, `Job.Started` | M1a | `tests/_exec/test_executor.py` |
| `_exec/supervisor.py` | `ProcessSupervisor`, `RLimits`; `launch_argv` (M4: the launcher argv, shared with the Ray task) | M1a +M4 | `tests/_exec/test_supervisor.py` |
| `_exec/procinfo.py` | `host_id`, `process_start_time`, `group_has_live_member`, `kill_orphan` | M1a | `tests/_exec/test_procinfo.py` |
| `_exec/env.py` | Tool environment from `Job.env`, `Job.secret_env` (allowlist resolved from the node and the secrets file), and `Job.secret_values` (values win). Names in `Job.drop_env` are never taken from the node or the secrets file (M4). It does not know the gateway: `_gateway/keys.py` supplies the values | M1a +M4 | `tests/_exec/test_env.py` |
| `_exec/pool.py` | `ResourcePool`: caps concurrent runs by `Resources` | M1a | `tests/_exec/test_pool.py` |
| `_exec/threads.py` | `run_sync_daemon` and the daemon-call holder: run blocking user code on a daemon thread that never blocks interpreter exit | M1a | `tests/_exec/test_threads.py` |
| `_exec/local.py` | `LocalExecutor` | M1a | `tests/_exec/test_local.py` |
| `_exec/local_stage.py` | Staging, command preparation, effective limits and environment | M1a | `tests/_exec/test_local.py`, `tests/_exec/test_preparation.py` |
| `_exec/local_workspace.py` | Workspace creation, ownership checks and cleanup | M1a | `tests/_exec/test_local.py`, `tests/_exec/test_preparation.py` |
| `_exec/local_run.py` | Supervision, daemon collection, bounded output reads and uploads | M1a | `tests/_exec/test_local.py`, `tests/_exec/test_output_bounds.py` |
| `_scoring/__init__.py` | Empty | M1a | — |
| `_scoring/check.py` | `Check` ABC, `CheckContext` | M1a | `tests/_scoring/test_check.py` |
| `_scoring/judge.py` | `Judge` ABC, `JudgeContext`, identity hashes, score stamping, finite-key rule | M1a | `tests/_scoring/test_judge.py` |
| `_scoring/program_judge.py` | `ProgramJudge` | M1a | `tests/_scoring/test_program_judge.py` |
| `_scoring/lexicographic.py` | `Lexicographic` | M1a | `tests/_scoring/test_lexicographic.py` |
| `_scoring/script_check.py` | `ScriptCheck` | M1a | `tests/_scoring/test_script_check.py` |
| `_scoring/script_judge.py` | `ScriptJudge` | M1a | `tests/_scoring/test_script_judge.py` |
| `_scoring/ranking.py` | The only owner of ranking groups (a part, or the main goal) and of ranking: one judge identity per group, newest fidelity, complete only. `RoundState.groups()` and `ranked()` delegate here | M1b +M2 | `tests/_scoring/test_ranking.py` |
| `_scoring/runner.py` | Checks after a tool run; barrier batches with anchors; final judge | M1b +M3 | `tests/_scoring/test_runner.py` |
| `_scoring/human_review.py` | `HumanReview` export and import | M2 | `tests/_scoring/test_human_review.py` |
| `_journal/__init__.py` | Empty | M1b | — |
| `_journal/event.py` | `Event`, `EventDraft`, `EventType`, payload models | M1b +M2 | `tests/_journal/test_event.py` |
| `_journal/segment.py` | Segment names, encode and decode with the link line | M1b | `tests/_journal/test_segment.py` |
| `_journal/lease.py` | `Head`, `OwnerDoc`, `Lease` (acquire, renew, take over, release, fence) | M1b | `tests/_journal/test_lease.py` |
| `_journal/writer.py` | `JournalWriter` (one writer task, group commit, unknown results) | M1b | `tests/_journal/test_writer.py` |
| `_journal/reader.py` | `read_chain` with the chain check | M1b | `tests/_journal/test_reader.py` |
| `_journal/views.py` | `ViewWriter`, source-head embedding | M1b | `tests/_journal/test_views.py` |
| `_journal/replay.py` | `RunState`: pure fold of committed events | M1b +M2 | `tests/_journal/test_replay.py` |
| `_journal/a2a.py` | A2A `Task` and `Artifact` records for tool runs and agent rounds | M1b | `tests/_journal/test_a2a.py` |
| `_journal/llm_record.py` | `bind_run`, `record_call` (R4.13.19). Core module: no optional import | M3 +M4 | `tests/_journal/test_llm_record.py` |
| `_budget/__init__.py` | Empty | M1b | — |
| `_budget/amount.py` | `Amount` arithmetic and fit checks | M1b | `tests/_budget/test_amount.py` |
| `_budget/split.py` | Goal weights, final slices, round and agent shares, returns | M1b | `tests/_budget/test_split.py` |
| `_budget/ledger.py` | `Ledger`, `Reservation` | M1b +M4 | `tests/_budget/test_ledger.py` |
| `_budget/tranche.py` | `Tranche` for LLM spend | M1b +M3 | `tests/_budget/test_tranche.py` |
| `_budget/estimate.py` | Worst-case estimate (§5 step 2), `judge_amount`, `final_slice_estimate` | M1b +M3 | `tests/_budget/test_estimate.py` |
| `_budget/llm_scope.py` | `LLMScope` context variable seam with `call_ids`, and the call-id seam `requesting_call` (R4.13.7) | M1b | `tests/_budget/test_llm_scope.py` |
| `_agent/__init__.py` | Empty | M1b | — |
| `_agent/manager.py` | `Manager` ABC, `LimitReached` | M1b | `tests/_agent/test_manager.py` |
| `_agent/runtime.py` | `AgentRuntime`, `AgentCounters`, own items | M1b | `tests/_agent/test_runtime.py` |
| `_agent/context.py` | `AgentContext` (public actions) | M1b +M2 | `tests/_agent/test_context.py` |
| `_agent/tool_run.py` | The `run_tool` pipeline | M1b +M2 +M4 | `tests/_agent/test_tool_run.py` |
| `_agent/repeats.py` | Repeat key and rules (§6.2) | M1b | `tests/_agent/test_repeats.py` |
| `_agent/watchdog.py` | Watchdog (§8.2) | M1b | `tests/_agent/test_watchdog.py` |
| `_agent/scripted.py` | `ScriptedManager` | M1b +M2 | `tests/_agent/test_scripted.py` |
| `_planning/__init__.py` | Empty | M1b | — |
| `_planning/planner.py` | `Planner` ABC, provenance stamping | M1b | `tests/_planning/test_planner.py` |
| `_planning/provenance.py` | Revision diff, `RefSite`, carried-over human items | M1b | `tests/_planning/test_provenance.py` |
| `_planning/catalog_check.py` | Catalog resolution, settable validation, override rules, `catalog_form`, `authorize_revision` (no import, no I/O) | M1b | `tests/_planning/test_catalog_check.py` |
| `_planning/validate.py` | `admit_revision`, `validate_plan`, model-check hook | M1b +M3 +M4 | `tests/_planning/test_validate.py` |
| `_planning/policy.py` | `Policy` ABC | M1b | `tests/_planning/test_policy.py` |
| `_planning/policies.py` | `KeepTopTeams`, `SuccessiveHalving`, `Patience`, `Chain` | M1b | `tests/_planning/test_policies.py` |
| `_board/__init__.py` | Empty | M1b | — |
| `_board/base.py` | `Board`, `BoardView` ABCs | M1b +M2 | `tests/_board/test_base.py` |
| `_board/run_board.py` | `RunBoard`: this run only, in memory (deleted in M2) | M1b | `tests/_board/test_run_board.py` |
| `_board/frontier.py` | Build frontiers (this run's mark in M1b, every run read in M2). It does not commit: `_engine/round.py` journals them | M1b +M2 | `tests/_board/test_frontier.py` |
| `_board/digest.py` | Digest builder with the token cap | M1b +M2 | `tests/_board/test_digest.py` |
| `_board/curator.py` | `Curator` ABC and the round-end hook | M2 | `tests/_board/test_curator.py` |
| `_board/arena.py` | `ArenaBoard` | M2 | `tests/_board/test_arena.py` |
| `_board/cache.py` | SQLite schema, FTS5, queries filtered by frontier | M2 | `tests/_board/test_cache.py` |
| `_board/ingest.py` | Incremental chain download into the cache | M2 | `tests/_board/test_ingest.py` |
| `_board/pointers.py` | `runs/<run-id>.json` pointers | M2 | `tests/_board/test_pointers.py` |
| `_board/human_objects.py` | Review and curation objects and their fixed order | M2 | `tests/_board/test_human_objects.py` |
| `_board/learnings.py` | Posting rules: evidence, duplicate check, `refines`, `verified` | M2 | `tests/_board/test_learnings.py` |
| `_board/leaderboard.py` | `Row` building, `min_improvement` | M2 | `tests/_board/test_leaderboard.py` |
| `_board/pages.py` | Round pages | M2 | `tests/_board/test_pages.py` |
| `_board/serve.py` | Board viewer (stdlib HTTP server) | M2 +M3 | `tests/_board/test_serve.py` |
| `_board/trace.py` | `trace` chain | M2 +M3 | `tests/_board/test_trace.py` |
| `_board/reindex.py` | Cache rebuild | M2 | `tests/_board/test_reindex.py` |
| `_engine/__init__.py` | Empty | M1b | — |
| `_engine/swarm.py` | `Swarm` facade | M1b | `tests/_engine/test_swarm.py` |
| `_engine/run.py` | Run level: estimate and approval, goal queue, slots, run clock, stop control, end of run; review export hook (+M2); `bind_run` for `record_call` (+M3) | M1b +M2 +M3 | `tests/_engine/test_run.py` |
| `_engine/goal.py` | One goal: rounds, barrier, policy, replanner, converge call | M1b +M2 | `tests/_engine/test_goal.py` |
| `_engine/admission.py` | Run-wide peak-agent reservations and atomic revision publication | M1b | `tests/_engine/test_admission.py` |
| `_engine/round.py` | One round: frontier commit, snapshot, managers in parallel, round timeout | M1b | `tests/_engine/test_round.py` |
| `_engine/stops.py` | Mid-round stop rules (§6.4); `ProviderStops`, the run's one provider-failure registry (M3, R4.13.10) | M1b +M3 | `tests/_engine/test_stops.py` |
| `_engine/parts.py` | Part assignment, kept best of a part, join, re-join attempts (groups come from `_scoring/ranking.py`) | M1b | `tests/_engine/test_parts.py` |
| `_engine/converge.py` | Select and team converge, final judge call | M1b | `tests/_engine/test_converge.py` |
| `_engine/solution.py` | `solution.json` view, `NO_RESULT` | M1b | `tests/_engine/test_solution.py` |
| `_engine/preflight.py` | Preflight per distinct tool config; storage checks, `check_storage` (+M2, R4.13.13); gateway (+M4) | M1b +M2 +M4 | `tests/_engine/test_preflight.py` |
| `_engine/manifest.py` | `manifest.json` | M1b | `tests/_engine/test_manifest.py` |
| `_engine/control.py` | Stop files (one pair per epoch, R4.13.13), status report | M1b | `tests/_engine/test_control.py` |
| `_engine/resume.py` | Takeover, replay, orphans, `lost` runs, `limits_changed` | M1b +M2 +M4 | `tests/_engine/test_resume.py` |
| `_cli/__init__.py` | Empty | M1a | — |
| `_cli/main.py` | `argparse` root, dispatch, `anyio.run(..., backend="asyncio")` | M1a +M1b +M2 | `tests/_cli/test_main.py` |
| `_cli/run_cmds.py` | `run`, `status`, `stop`, `resume` | M1b | `tests/_cli/test_run_cmds.py` |
| `_cli/arena_cmds.py` | `review`, `leaderboard`, `board`, `learnings`, `trace`, `reindex` | M2 | `tests/_cli/test_arena_cmds.py` |
| `_cli/plan_cmd.py` | `plan`; imports `swarm_arena.llm` only inside the `--llm` and `--together` handlers | M3 | `tests/_cli/test_plan_cmd.py` |
| `_cli/cluster_cmds.py` | `submit`, `reconcile` | M4 | `tests/_cli/test_cluster_cmds.py` |
| `_gateway/__init__.py` | Empty | M4 | — |
| `_gateway/adapter.py` | `GatewayAdapter` protocol, `VirtualKey`, `SpendRow`, `ProfileReport` | M4 | `tests/_gateway/test_adapter.py` |
| `_gateway/litellm_proxy.py` | LiteLLM Proxy adapter (stdlib `urllib`), `terminal_accounting = False` | M4 | `tests/_gateway/test_litellm_proxy.py` |
| `_gateway/profile.py` | Strict-profile preflight | M4 | `tests/_gateway/test_profile.py` |
| `_gateway/keys.py` | Key per tool run; the gateway values for the `Job` (base URLs for `Job.env`, the key for `Job.secret_values`, the provider key names to drop from `Job.secret_env` and to set as `Job.drop_env`); spend read-back | M4 | `tests/_gateway/test_keys.py` |
| `_gateway/reconcile.py` | Post-run reconcile | M4 | `tests/_gateway/test_reconcile.py` |
| `llm/__init__.py` | Facade `swarm_arena.llm`; clear `ImportError` without the extra | M3 | `tests/test_public_api.py` |
| `llm/_catalog.py` | Capability checks, prices, `build_model` | M3 | `tests/llm/test_catalog.py` |
| `llm/_settings.py` | `ModelSettings` per call: caching, metadata, cache args | M3 | `tests/llm/test_settings.py` |
| `llm/_logger.py` | `CallLogger`: maps LiteLLM `kwargs` and `response_obj` to the core `record_call` | M3 | `tests/llm/test_logger.py` |
| `llm/_cache.py` | Response-cache setup from `CachePolicy` | M3 | `tests/llm/test_cache.py` |
| `llm/_prompts.py` | Prompt layout and untrusted markers | M3 | `tests/llm/test_prompts.py` |
| `llm/_structured.py` | Structured output and the two-stage fallback | M3 | `tests/llm/test_structured.py` |
| `llm/_hooks.py` | `RunHooks` pre-check, `BudgetExceeded` | M3 | `tests/llm/test_hooks.py` |
| `llm/_actions.py` | `@function_tool` wrappers for `AgentContext` actions | M3 | `tests/llm/test_actions.py` |
| `llm/_linking.py` | `tool_call_id` → `LLMCall.id` | M3 | `tests/llm/test_linking.py` |
| `llm/_manager.py` | `AgentsSDKManager` | M3 | `tests/llm/test_manager.py` |
| `llm/_judge.py` | `LLMJudge` | M3 | `tests/llm/test_judge.py` |
| `llm/_reviewer.py` | `Reviewer`, `Approve`, `Feedback`, `Edited` | M3 | `tests/llm/test_reviewer.py` |
| `llm/_terminal_reviewer.py` | Terminal `Reviewer` used by `swarm-arena plan --together` | M3 | `tests/llm/test_terminal_reviewer.py` |
| `llm/_planner.py` | `LLMPlanner`, `CollaborativePlanner` | M3 | `tests/llm/test_planner.py` |
| `llm/_curator.py` | `LLMCurator` | M3 | `tests/llm/test_curator.py` |
| `llm/_langfuse.py` | Langfuse export | M3 | `tests/llm/test_langfuse.py` |
| `ray/__init__.py` | Facade `swarm_arena.ray`; clear `ImportError` without the extra | M4 | `tests/test_public_api.py` |
| `ray/_executor.py` | `RayExecutor` | M4 | `tests/ray/test_executor.py` (marker `ray`) |
| `ray/_task.py` | Generic remote task | M4 | `tests/ray/test_task.py` (marker `ray`) |
| `ray/_state.py` | Task lookup by name, `confirm_dead` | M4 | `tests/ray/test_state.py` (marker `ray`) |
| `ray/_files.py` | Object-store file transport with the 256 MiB cap | M4 | `tests/ray/test_files.py` (marker `ray`) |
| `ray/_submit.py` | `submit` through `JobSubmissionClient` | M4 | `tests/ray/test_submit.py` |
| `testing/__init__.py` | Facade `swarm_arena.testing` | M1a +M1b +M3 | `tests/test_public_api.py` |
| `testing/_conformance.py` | `check_tool`, `check_check`, `check_judge` (M1a), `check_manager` (M1b), parallel calls and `check_judge(..., config=)` (M3) | M1a +M1b +M3 | `tests/testing/test_conformance.py` |
| `testing/_state.py` | Process-local artifact-store ownership and shared identifier source | M1a | `tests/testing/test_artifacts.py`, `tests/testing/test_candidates.py` |
| `testing/_components.py` | Component snapshots, preflight, and goal matching | M1a | `tests/testing/test_conformance.py`, `tests/testing/test_round_context.py` |
| `testing/_execution.py` | Watched execution and non-ok stderr diagnostics | M1a | `tests/testing/test_conformance.py`, `tests/testing/test_artifacts.py` |
| `testing/_tool.py` | Tool job preparation and collection contract probes | M1a | `tests/testing/test_conformance.py`, `tests/testing/test_round_context.py` |
| `testing/_contracts.py` | Result validation and shared return-type predicates | M1a | `tests/testing/test_conformance.py` |
| `testing/_candidates.py` | `make_candidate` | M1a | `tests/testing/test_candidates.py` |
| `testing/_artifacts.py` | `read_output_file`, `read_logs` | M1a | `tests/testing/test_artifacts.py` |
| `testing/_harness.py` | Offline `AgentContext` harness for `check_manager` | M1b | `tests/testing/test_harness.py` |
| `testing/_fake_model.py` | `FakeModel`, `FakeTurn` (imports `agents` lazily) | M3 | `tests/testing/test_fake_model.py` |

### R3.2 Other test files (not 1:1 with a module)

| Path | Responsibility | M |
|---|---|---|
| `conftest.py` (repo root) | Doctest collection: `collect_ignore` for `src/swarm_arena/llm`, `ray`, `_store/s3.py`, and `testing/_fake_model.py` when the extra is missing (test paths: `tests/conftest.py`) | M1a |
| `tests/conftest.py` | `anyio_backend` fixture fixed to `"asyncio"`; temp stores and roots. `collect_ignore` for every test path that imports an extra, applied when that extra's distribution is missing (`importlib.metadata.version`, never `find_spec`, R3), because `-m` deselection happens only after import: `llm`: `llm/`, `testing/test_fake_model.py`, `e2e/test_rosenbrock_llm.py`, `e2e/test_autonomous_replan.py`, `e2e/test_llm_plan_file.py`, `e2e/test_llm_final_slice.py`; `ray`: `ray/` (with `test_submit.py`), `e2e/test_rosenbrock_ray.py`, `e2e/test_ray_cancel.py`, `e2e/test_ray_dispatch_crash.py`; `s3`: `_store/test_s3.py`. A milestone that adds such a file adds it here. `test_public_api.py`, `test_docstrings.py`, and `test_compat.py` skip the `llm` and `ray` modules and the lazy `swarm_arena.testing` names through `tests/_surface.py` `deps_present` (`importlib.metadata`; never `importorskip` or `find_spec`), M5 | M1a (+M2 +M3 +M4 +M5) |
| `tests/_fakes/engine.py` | Test-side env types and helpers for agent and engine tests (`RtEnv`, `CtxEnv`, `EngineEnv`, the config helpers `with_limits` and `with_executor`, `take_over_retrying`); importable as `_fakes.engine`, never collected | M1b (+M2) |
| `tests/_agent/conftest.py` | Agent fixtures `rt_env` and `ctx_env`; M2 changes their board to `ArenaBoard` | M1b (+M2) |
| `tests/_engine/conftest.py` | Engine fixture `env` (an `EngineEnv` on a real lease, journal, ledger, board, and executor); M2 Task 9 changes its board to `ArenaBoard` | M1b (+M2) |
| `tests/_fakes/planners.py` | Test-only `Planner` subclasses for replanner tests | M1b |
| `tests/_fakes/import_canary.py` | A module that, on import, writes the file named by env `SWARM_ARENA_TEST_CANARY` (when set) and defines `Tool`; proves that a rejected non-catalog ref is never imported | M1b |
| `tests/_fakes/gateway.py` | Stdlib fake LiteLLM Proxy (strict-profile endpoints, keys, spend logs) | M4 |
| `tests/_fakes/tools.py` | Small module-level tool functions and scripts used across tests; refs name them `_fakes.tools:<name>` (importable through `pythonpath = ["tests"]`, R5.3) | M1a |
| `tests/test_public_api.py` | One `__all__` snapshot test per public module | M1a (exact §9 in M5) |
| `tests/test_docstrings.py` | numpydoc validation and a required `Examples` section for every public name and its public methods | M1a |
| `tests/test_import_isolation.py` | `import swarm_arena` (and each core facade) loads no optional dependency | M1a |
| `tests/e2e/test_single_run.py` | M1a exit test | M1a |
| `tests/e2e/test_rosenbrock.py`, `test_own_script.py`, `test_many_goals_parts.py`, `test_cli_lifecycle.py`, `test_crash_resume.py`, `test_fencing.py` | M1b exit tests | M1b |
| `tests/e2e/test_arena_two_runs.py`, `test_reindex.py`, `test_review_roundtrip.py` | M2 exit tests | M2 |
| `tests/e2e/test_rosenbrock_llm.py`, `test_autonomous_replan.py`, `test_llm_plan_file.py`, `test_llm_final_slice.py` | M3 exit tests | M3 |
| `tests/e2e/test_rosenbrock_ray.py`, `test_ray_cancel.py`, `test_ray_dispatch_crash.py`, `test_gateway_run.py` | M4 exit tests | M4 |

### R3.3 Other repo files

| Path | M |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` (`3.11`), `LICENSE`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.gitignore`, `.github/workflows/ci.yml` | M1a |
| `docs/conf.py`, `docs/index.md`, `docs/<page>.md` (12 pages of §11), `docs/api/<module>.md` | M1a (filled per milestone, complete in M5) |
| `examples/rosenbrock/{rosen.py,plan.toml,run.toml,README.md}` (refs `rosen:<name>`, importable through `Plan.base_dir`, R4.1) | M1b (+M3: `plan_llm.toml`) |
| `examples/own_script/{problem/model_a.txt,problem/model_b.txt,problem/points.txt,problem/run.py,checks/check.py,judge/score.py,judge/hidden/points.csv,plan.toml,run.toml,README.md}` | M1b |
| `.readthedocs.yaml`, `.github/workflows/release.yml`, `.github/workflows/upstream.yml`, `upstream-overrides.txt`, `RELEASING.md` | M5 |

## R4. Interface registry (binding)

Signatures only; bodies are `...`. The comment above each block names the private module (R3.1). "M" is the milestone that creates the name. A milestone plan may add private helpers, but it may not change a signature here without editing this file (R0).

### R4.0 Conventions

- `FrozenModel`: pydantic v2 `BaseModel` with `ConfigDict(frozen=True, extra="forbid")`. Frozen models are never changed in place; use `model_copy(update=...)`. `StoredModel(FrozenModel)` adds `schema_version: int = 1`. Every stored format (R4.15) is a `StoredModel` [plan].
- Field types copy the spec. Where the spec says `list`, the field is `list`; where it says `dict`, the field is `dict[str, Any]` unless typed here.
- Type aliases (in `_model/base.py`) [plan]: `Origin = Literal["human", "llm", "together", "autonomous"]`; `LLMRole = Literal["manager", "judge", "planner", "curator"]`; `CallRole = Literal["manager", "judge", "planner", "curator", "tool"]`; `CatalogKind = Literal["tool", "manager", "check", "judge", "policy"]`; `JsonValue = pydantic.JsonValue`.
- Times are timezone-aware UTC `datetime`, serialized as ISO-8601 with milliseconds and `Z` [plan].
- **Component contract (§4.1).** Every component class (tool, manager, check, judge, policy, executor, board, curator) has `Config: ClassVar[type[BaseModel]]` (default: an empty model) and is built as `cls(cls.Config.model_validate(config))`. The instance exposes the validated model as `obj.config`. Built-ins also accept the spec's shorthand arguments (for example `KeepTopTeams(2)`), but always accept a single `Config` instance positionally. `__init__` does no I/O (§9 rule 3). Typing pattern (M1a, checked with ty 0.0.89): component classes inherit `_model/ref.py:Configurable`, whose `Config = EmptyConfig` is a plain class attribute with no annotation (ty rejects a nested `class Config` that overrides an annotated base attribute). `obj.config` is a read-only property; built-ins override it with their own `Config` as the return type. The `Config: ClassVar[type[BaseModel]]` lines in R4 describe this contract. Fields typed `Ref` are `RefField` (`Annotated[Ref, BeforeValidator(coerce_ref)]`): instances are converted at runtime; statically typed callers pass `Ref.of(obj)`.
- **Ids [plan]:** `<prefix>-<run tag>-e<epoch>-<n:04d>`, for example `c-3f9a1c2b5d6e7f80-e1-0193`. The run tag is `sha256(run_id)[:16]` (64 bits), so two runs share ids only on a 64-bit tag clash (about 3e-10 for 100,000 runs), and ids are unique across epochs. `ArenaCache.ingest` (M2) skips an item whose id it already holds from another run and records a warning that names both runs. So `Board.get_candidate`, `get_learning`, lineage, `parents`, `evidence`, `refines`, `supersedes`, `trace`, and `board show` resolve one item, and gateway key aliases `cand-<candidate id>` are unique on a shared gateway. The counter per prefix continues across epochs. Prefixes: `c` candidate, `L` learning, `s` score, `r` reservation, `llm` LLM call, `x` command job, `k` gateway key alias suffix. Scores from `review import` take ids from the run's `IdGen` under the import's own epoch. Human learnings (`learnings add`, the board viewer) have no run or epoch: `L-h-<16 random hex>`. Run id: `YYYY-MM-DDTHH-MM-SSZ_<6 hex>`. Outside a run (`swarm_arena.testing`), the run id is `"local"`.
- **Addresses:** agent `"<goal>/<team>/<agent>"`; team `"<goal>/<team>"`. `Decision.prune` keys are relative to the goal: `"<team>"` or `"<team>/<agent>"` [plan]. Ledger holders (R4.13.6) [plan]: agent address, `"<goal>/:judges"`, `"<goal>/:planner"`, `"<goal>/:curator"`, `"<goal>/:final"`, `":run"`.
- **Errors visible to users** are subclasses of built-in exceptions, so the public surface stays as in §9 [plan]: plan and config errors are `ValueError`; soft action errors are `ValueError`; storage errors are `RuntimeError`; `LimitReached` is public.

### R4.1 `swarm_arena` (top level)

```python
# _model/ref.py — M1a
class Ref(FrozenModel):
    path: str                                   # "pkg.module:QualName"; checked format
    config: dict[str, Any] = {}
    name: str | None = None                     # default: the class's `name`
    @classmethod
    def of(cls, component: object) -> "Ref": ...          # instance or class → Ref; rejects lambdas, local and __main__ objects
    def load(self) -> type: ...                            # imports and caches
    def build(self, base: type[T]) -> T: ...               # cls(cls.Config.model_validate(config)); TypeError if not a subclass of base
    def component_name(self) -> str: ...                   # self.name or the class's `name`
# Fields typed `Ref` accept a component instance and convert it with Ref.of [plan].

# _model/plan.py — M1a
class Resources(FrozenModel):
    cpus: float = 1.0
    gpus: float = 0.0
    memory_gib: float | None = None
    custom: dict[str, float] = {}

# _model/goal.py — M1a
class Goal(FrozenModel):
    id: str                                     # [plan] slug: [a-z0-9][a-z0-9._-]*
    namespace: str                              # same slug rule
    title: str = ""
    description: str = ""
    inputs: dict[str, Any] = {}
    files: FileSet = FileSet()
    metadata: dict[str, Any] = {}
    def key(self) -> GoalKey: ...               # ValueError if files are not snapshotted

# _model/plan.py — M1a
class PartSpec(FrozenModel):
    name: str
    goal: Goal
    # Every main goal and part group in a run must have a distinct snapshotted GoalKey;
    # validate_plan rejects collisions within a plan, and run startup checks all plans before paid work.
    checks: list[Ref] | None = None
    judges: list[Ref] | None = None
    ranking: str | None = None

class AgentSpec(FrozenModel):
    name: str
    rationale: str                              # required, non-empty
    tool: Ref
    manager: Ref
    replicates: int = 1
    resources: Resources = Resources()

class TeamSpec(FrozenModel):
    name: str
    rationale: str                              # required, non-empty
    part: str | None = None
    agents: list[AgentSpec]

class RoundSpec(FrozenModel):
    tool_overrides: dict[str, Any] = {}
    tool_timeout_s: float | None = None
    fidelity: str | None = None
    def effective_fidelity(self) -> str: ...    # fidelity or sha256(canonical(tool_overrides))[:12] [plan]

class ConvergeSpec(FrozenModel):
    mode: Literal["select", "team"] = "select"
    top_n: int = 2
    distinct: bool = True
    allow_partial: bool = False
    min_improvement: float | None = None
    team: TeamSpec | None = None
    round: RoundSpec | None = None
    parents_per_part: int = 1
    max_join_attempts: int = 1

class Plan(StoredModel):
    goal: Goal
    parts: list[PartSpec] = []
    teams: list[TeamSpec]
    checks: list[Ref] = []
    judges: list[Ref]
    ranking: str = ""                           # [plan] "" means "the only judge"; ValueError if there are several
    policy: Ref = Ref(path="swarm_arena:SuccessiveHalving")   # [plan] default
    rounds: list[RoundSpec] = []
    converge: ConvergeSpec = ConvergeSpec()
    sharing: Literal["round"] = "round"
    flows: list[FlowSpec] = []                  # non-empty → ValueError (reserved for v2)
    rationale: str = ""
    origin: Origin = "human"                    # set by the core (§4.6); "human" for a plan a person builds in Python
    revision: int = 0
    parent_revision: int | None = None
    @classmethod
    def load(cls, path: str | Path) -> "Plan": ...         # human path; base dir = the file's directory
        # [plan] keeps a non-human origin found in the file (for example from `swarm-arena plan --llm`); no origin → "human".
        # The human path trusts "human" but never raises trust: an "llm", "together", or "autonomous" file stays so.
    @classmethod
    def loads(cls, text: str, *, base_dir: Path) -> "Plan": ...   # same origin rule as load
    def dumps(self) -> str: ...                             # deterministic TOML, origin included
    @property
    def base_dir(self) -> Path: ...                         # private attr; default Path.cwd() for Python-built plans [plan]
    # Import rule [plan]: before the first Ref.load of a plan, the core appends its base_dir to sys.path (once), so refs
    # to modules next to the plan file load (examples/rosenbrock/rosen.py as "rosen:gradient_descent"). base_dir is
    # journaled with r0 (plan_revision payload), and resume restores it.
    @property
    def planning_session(self) -> str | None: ...           # M3; private attr set by mark_planning_session (R4.13.13): the run id
    @property
    def planner_call_ids(self) -> tuple[str, ...]: ...      # M3; private attr set by mark_planning_session: its planner LLMCall ids
        # [plan] M3: dumps() writes "# swarm-arena planning session: <run id>" as its first line when planning_session is
        # set, and load/loads read it back (§7.3: trace shows the planner call that created the agent).
    def ranking_name(self) -> str: ...
    def rounds_total(self, limits: Limits) -> int: ...      # len(rounds) or limits.max_rounds

# _model/limits.py — M1a (defaults in R5.2)
class Limits(FrozenModel):
    max_teams: int = 4
    max_agents_per_team: int = 4
    max_agents_per_goal: int = 12
    max_concurrent_goals: int = 4
    max_agents_total: int = 32
    max_agents_per_fingerprint: int = 2
    max_rounds: int = 3
    max_turns_per_round: int = 20
    max_tool_runs_per_round: int = 3
    max_tokens: int | None = None               # required when any LLM component is used (checked by validate_plan)
    max_usd: float | None = None
    default_run_reservation: Estimate = Estimate(tokens=100_000, usd=1.0)   # [plan] values
    llm_tranche_usd: float = 0.25
    llm_tranche_tokens: int = 25_000
    tool_timeout_s: float = 3600.0
    round_timeout_s: float | None = None        # None → 4 × tool_timeout_s
    run_timeout_s: float = 86_400.0
    max_learnings_per_agent_round: int = 3
    max_board_reads_per_round: int = 10
    digest_tokens: int = 1500
    max_edit_bytes: int = 262_144
    max_consecutive_failures: int = 3
    max_refused_repeats: int = 3
    watchdog_idle_rounds: int = 2
    max_notes_bytes: int = 16_384               # [plan] the §4.5 notes cap
    def effective_round_timeout_s(self) -> float: ...

# _model/run_config.py — M1a
class RunConfig(StoredModel):
    limits: Limits = Limits()
    reserve_fraction: float = 0.15
    goal_weights: dict[str, float] | None = None
    executor: Ref = Ref(path="swarm_arena.executor:LocalExecutor")
    board: Ref | None = None                    # None → internal RunBoard (M1b), ArenaBoard on arena_root (M2+)
    curator: Ref | None = None
    run_root: str = ".swarm-arena/runs"         # [plan] defaults for the three roots
    arena_root: str = ".swarm-arena/arena"
    artifact_root: str = ".swarm-arena/artifacts"
    secrets: SecretsPolicy = SecretsPolicy()
    schedule_timeout_s: float = 600.0           # [plan]
    sync_interval_s: float = 60.0
    lease_ttl_s: float = 60.0
    settle_timeout_s: float = 60.0
    tracing: TracingConfig = TracingConfig()
    llm_gateway: GatewayConfig | None = None
    models: ModelCatalog = ModelCatalog({})
    planner_catalog: PlannerCatalog = PlannerCatalog()
    cache: CachePolicy | None = None            # [plan] spec types it CachePolicy; None → no response cache (R5.4)
    @classmethod
    def load(cls, path: str | Path) -> "RunConfig": ...
    def dumps(self) -> str: ...                 # secret names only

# _engine/swarm.py — M1b
class Swarm:
    def __init__(self, config: RunConfig, *, replanner: Planner | None = None) -> None: ...
    run_id: str | None                          # set when run() or resume() starts
    async def run(self, plans: Plan | Sequence[Plan], *, run_id: str | None = None,
                  approve: Callable[[Estimate, Sequence[str]], Awaitable[bool]] | None = None) -> dict[str, list[Candidate]]: ...
        # goal id → final finalists in final-judge order; [] for NO_RESULT. Raises RuntimeError on a non-asyncio backend.
        # [plan] approve(estimate, warnings) runs after validation and the worst-case estimate (§5 step 2; warnings include
        # peak demand over Executor.capacity()) before persistent run/artifact writes; False → RuntimeError, no run folder.
        # Admission stages snapshots in temporary storage, discarded on refusal; approved bytes are promoted without
        # rereading sources. Preflight and manifest preparation complete before promotion or lease acquisition.
        # None → the estimate is only journaled (autonomous mode logs it). The CLI passes a terminal prompt (manual mode).
        # Each plan keeps its own origin: r0 is stamped with it and never raised to "human" (R4.5).
    async def resume(self, run_id: str, *, force: bool = False) -> dict[str, list[Candidate]]: ...
        # self.config may raise limits (journaled as limits_changed)
    async def stop(self, *, now: bool = False) -> None: ...   # creates control/stop-e<epoch> or control/stop-now-e<epoch> for this owner's epoch (R4.13.13)

# _planning/validate.py — M1b
def validate_plan(plan: Plan, config: RunConfig, *, history: Sequence[Plan] = (),
                  other_active_agents: int = 0) -> list[str]: ...
    # history: earlier revisions of this goal, r0 first. Returns warnings.
    # Raises ValueError (private PlanError, R4.13.1) listing every error. No I/O beyond imports and the M3
    # model-capability hook. A non-human revision runs authorize_revision first (R4.13.12). Its errors are raised before
    # any Ref.load or Config validation, so a ref outside the planner catalog is never imported. The core calls
    # validate_plan through admit_revision, which snapshots first (R4.13.12). [plan] A non-human revision resolves its
    # part-goal FileSets against the main goal's manifest, never from disk.
    # [plan] M3: the catalog and provenance rule runs on unbuilt Refs first; Ref.build (and so the model-capability
    # hook) runs only on human-provenance or accepted sites. A rejected ref is never imported.

# Built-in components re-exported at top level (signatures in R4.2, R4.3, R4.4, R4.5):
# ScriptTool, FunctionTool (tool); ScriptCheck, ScriptJudge, ProgramJudge, Score (scoring);
# ScriptedManager (manager); KeepTopTeams, SuccessiveHalving, Patience, Chain (planning).
```

### R4.2 `swarm_arena.tool` (+ `ScriptTool`, `FunctionTool` at top level)

```python
# _model/tooling.py — M1a
class RunStatus(StrEnum):
    OK = "ok"; ERROR = "error"; TIMEOUT = "timeout"; OOM = "oom"
    BUDGET_EXCEEDED = "budget_exceeded"; CANCELLED = "cancelled"
    PROVIDER_FAILED = "provider_failed"; LOST = "lost"

class Usage(FrozenModel):                       # [plan] field set
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    usd: float | None = None
    @property
    def total_tokens(self) -> int: ...
    def __add__(self, other: "Usage") -> "Usage": ...

class Estimate(FrozenModel):                    # also used for budget amounts shown to users [plan]
    seconds: float | None = None
    tokens: int | None = None
    usd: float | None = None

class FileRef(FrozenModel):
    name: str; path: str; sha256: str; size: int

class FileSet(FrozenModel):                     # _model/goal.py
    globs: list[str] = []                       # a TOML list of globs validates into this field [plan]
    manifest: dict[str, str] = {}               # path → sha256, set by the snapshot
    @property
    def snapshotted(self) -> bool: ...

class ParentRef(FrozenModel):                   # [plan] fields
    candidate_id: str                           # one path component ^[A-Za-z0-9][A-Za-z0-9._-]*$ (staged as parents/<id>/)
    run_id: str
    tool_identity: str                          # R4.13.9; copied from Candidate.tool_identity (also for past-run parents)
    data: dict[str, Any] = {}
    overlay: dict[str, str] = {}                # the parent's full effective overlay; keys in normal form (check_key)
    files: list[FileRef] = []

class ToolRequest(FrozenModel):
    inputs: dict[str, Any] = {}
    parents: list[ParentRef] = []
    replicate: int = 0
    round: int = 0                              # [plan] needed for swarm/request.json
    round_overrides: dict[str, Any] = {}
    edits: dict[str, str] = {}                  # path → content in memory; journaled as path → "sha256:<hex>" [plan]; only an
        # exact "sha256:<64 hex>" value is kept as a stored reference; live content is always hashed; the exact form is
        # reserved: validate_edits, edit_hashes, and put_edits refuse it as content (like the reserved top-level names)

class ToolOutput(FrozenModel):
    status: RunStatus
    partial: bool = False
    data: dict[str, Any] = {}
    files: list[FileRef] = []
    usage: Usage = Usage()
    logs_ref: str | None = None                 # [plan] from an executor, or outside a run: "sha256:<stdout sha256>:<stderr sha256>"
        # in artifact_root (transport). In a run, the core writes "<run_root>/<run-id>/goals/<goal-id>/logs/<id>.out" and
        # ".err" once (create, §7.1; <id> is the candidate or command-job id) and sets logs_ref to "goals/<goal-id>/logs/<id>".
    started_at: datetime | None = None
    ended_at: datetime | None = None
    failed_provider: str | None = None          # [plan] with status provider_failed: the provider that failed (result.json
        # key "failed_provider"). None → every provider in Tool.providers() counts as failed (§6.4).
    llm_calls_file: FileRef | None = None       # [plan] M4: swarm/llm_calls.jsonl in artifact_root, set where the tool runs
        # (a Ray worker too) so it reaches the driver with the output (§4.3, §7.3 item 2b); None if absent

# Private shared codec beside ToolOutput; executor and conformance kit use the same format.
def format_logs_ref(stdout_sha: str, stderr_sha: str) -> str: ...
def parse_logs_ref(ref: str) -> tuple[str, str]: ...
# sha256:<stdout SHA-256>:<stderr SHA-256>; malformed references raise value-free ValueError.

class Command(FrozenModel):
    argv: list[str]
    cwd: str | None = None                      # relative to the workspace; None → the workspace
    env: list[str] = []                         # allowlist of names; values come from the node
    timeout_s: float | None = None              # the core writes the effective timeout (§6.1)
    python: str | None = None                   # replaces argv[0] == "python"; None → sys.executable [plan]
    grace_s: float = 10.0                       # [plan] default

class CompletedRun(FrozenModel):
    returncode: int | None
    status: RunStatus                           # ok, error, timeout, cancelled, or oom
    stdout_path: Path
    stderr_path: Path
    started_at: datetime
    ended_at: datetime

# _tooling/tool.py — M1a
InputT = TypeVar("InputT", bound=BaseModel)

@dataclass(frozen=True, kw_only=True)
class Action(Generic[InputT]):                   # [plan] fields
    name: str
    description: str
    input_model: type[InputT]
    handler: Callable[[Any, InputT], Awaitable[JsonValue]]
    # first argument: the agent's AgentContext; M1b may narrow Any to AgentContext
    counts_as: Literal["board_read", "none"] = "none"

class Tool(Configurable, ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    Config: ClassVar[type[BaseModel]]
    def preflight(self) -> None: ...
    def is_deterministic(self, request: ToolRequest) -> bool: ...          # default True
    def editable(self) -> Sequence[str]: ...                                # default ()
    def estimate(self, request: ToolRequest) -> Estimate | None: ...        # default None
    def providers(self) -> frozenset[str]: ...                              # default frozenset()
    def fingerprint(self, output: ToolOutput) -> str: ...                   # default sha256(canonical(output.data))
    def extra_actions(self) -> Sequence[Action[Any]]: ...                        # default ()
    def get_input_model(self) -> type[BaseModel]: ...                       # [plan] default type(self).input_model
    def get_description(self) -> str: ...                                   # [plan] default type(self).description
    @abstractmethod
    def command(self, request: ToolRequest, workspace: Path) -> Command: ...
    @abstractmethod
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput: ...
    def _own_files(self) -> dict[str, str]: ...                            # default {}
    def _protected(self, root: AbstractSet[str]) -> frozenset[str]: ...     # default frozenset()
    # _own_files(): the tool's own root files; the driver's protected set is
    # tool._protected(root) | private_paths (R4.13.9).
    # get_input_model and get_description exist because FunctionTool and ScriptTool derive them per instance.

# _tooling/function_tool.py — M1a
class FunctionTool(Tool):
    name = "function"
    class Config(BaseModel):
        function: str                           # "module:qualname", module level only
        deterministic: bool = True
        providers: list[str] = []
        timeout_s: float | None = None
        grace_s: float = 10.0
        env: list[EnvName] = []                 # list[str] with environment-name syntax
    def __init__(self, function: "str | Callable[..., Any] | FunctionTool.Config", /, **fields: Any) -> None: ...
    # command(): `python -P -m swarm_arena.run_function --import-root <dir> <function>`
    # (`-P` suppresses only the implicit cwd/script-directory `sys.path` entry; explicit `PYTHONPATH`
    # entries still apply; the launcher bootstrap itself uses `-I`) [plan], with `<dir>` the `sys.path`
    # entry the module was loaded from, computed by `command()` next to the tool, so the subprocess
    # imports what the driver imported; no `Job.env` entry is needed.
    # Inputs are validated by a model built from the function's annotated parameters,
    # so the driver imports the function (a recorded exception to §4.3, R5.4); the return value must be a JSON object → data.

    # env is human-only and forwarded to Command.env. Results are data-only on every status;
    # files written by the function are never uploaded, including after timeout or cancellation.

# _tooling/script_tool.py — M1a
class ScriptTool(Tool):
    name = "script"
    class Config(BaseModel):
        command: list[str]
        python: str | None = None
        files: FileSet = FileSet()              # the tool's own protected files besides goal files [plan]
        editable: list[str] = []
        params: dict[str, dict[str, Any]] = {}  # JSON-schema subset per param (R4.13.0)
        timeout_s: float | None = None
        grace_s: float = 10.0
        env: list[str] = []
        deterministic: bool = True
        providers: list[str] = []
        description: str = ""                   # [plan] shown to the manager
    def __init__(self, config: "ScriptTool.Config | None" = None, /, **fields: Any) -> None: ...
    # extra_actions(): list_files(*, candidate=None), read_file(path, *, candidate=None); counts_as="board_read"
```

**Executor transport of `ToolRequest.edits`** [plan] M4: `Tool.command` and `Tool.collect` get the same request on every executor, with `edits` as path → content. `LocalExecutor` passes the in-memory request, whose live contents `validate_edits` checked. A request that crosses a JSON transport (the Ray task payload, `job_payload` and `read_payload`, R4.13.16) carries each edit as its persisted reference `"sha256:<hex>"` by the field serializer above; the serializer and the reserved-form rule do not change. Before `prepare_job`, and so before `Tool.command`, the Ray worker resolves every reference through `ArtifactStore.get_bytes`, decodes it as strict UTF-8, and rebuilds the live `ToolRequest` (`_exec/job_steps.live_request`, M4 Task 2, called by `ray/_task.run_in_worker`, M4 Task 5); the partial and final `Tool.collect` get that rebuilt request too. The references resolve in a run, because `put_edits` stores each edit before `plan_tool_stage` puts it in `Job.Stage.root` under the same hash (R4.13.9, R4.13.10), so the object-store blobs and a shared `artifact_root` both hold it; a reference that does not resolve ends the job as `error` before the tool starts. Test: M4 Task 5 `test_edit_contents_same_on_ray_as_local` (contents `"hello"` and `"sha256:hello"`, both transports).

### R4.3 `swarm_arena.scoring` (+ `Score`, `ProgramJudge`, `ScriptCheck`, `ScriptJudge` at top level)

```python
# _model/results.py — M1a
class CheckResult(FrozenModel):
    check: str
    passed: bool
    reasons: list[str] = []
    details: dict[str, Any] = {}

class Score(FrozenModel):                       # judges return it; the core stamps the empty fields
    id: str = ""
    candidate_id: str = ""
    judge: str = ""
    rubric_hash: str = ""
    panel_hash: str = ""
    key: tuple[float, ...]                      # every value finite, else ValueError
    value: float | None = None                  # [plan] the human-facing number (minimize(x) stores x)
    verdict: str | None = None
    notes: str = ""
    batch_id: str = ""
    llm_call_ids: list[str] = []
    @classmethod
    def minimize(cls, x: float, *, verdict: str | None = None, notes: str = "") -> "Score": ...   # key=(-x,), value=x
    @classmethod
    def maximize(cls, x: float, *, verdict: str | None = None, notes: str = "") -> "Score": ...   # key=(x,), value=x
    @property
    def identity(self) -> tuple[str, str, str]: ...      # (judge, rubric_hash, panel_hash)

# _scoring/check.py — M1a
class CheckContext:                             # built by the core and by swarm_arena.testing
    def __init__(self, *, goal: Goal, executor: Executor, artifact_root: str, run_id: str = "local",
                 round: int = 0, part: str | None = None,
                 on_started: Callable[[Job.Started], Awaitable[None]] | None = None,
                 secrets_file: str | None = None, ids: IdGen | None = None,
                 tool_timeout_s: float = 3600.0) -> None: ...
    goal: Goal; run_id: str; round: int; part: str | None
    secrets_file: str | None; ids: IdGen; tool_timeout_s: float  # ids=None -> IdGen(run_id, 0)
    on_started: Callable[[Job.Started], Awaitable[None]] | None   # execute() puts it on the command Job; the core sets it
                                                                    # in a run (R4.14 command_start)
    async def execute(self, command: Command, *, files: Mapping[str, str] | None = None,
                      candidate: Candidate | None = None, resources: Resources | None = None) -> ToolOutput: ...
    # files: path → sha256 staged at the workspace root; candidate: staged under candidate/ (§4.12);
    # ToolOutput.data is the parsed swarm/result.json.
    # execute uses min(command.timeout_s or tool_timeout_s, tool_timeout_s), computed by cap_timeout;
    # ctx.ids is the only id source for command jobs and score stamping (M1b's scoring runner calls
    # stamp_scores(..., ids=ctx.ids) and turns ctx.errors into judge_error events).

class Check(Configurable, ABC):
    name: ClassVar[str]
    feedback: Literal["pass_fail", "reasons"] = "reasons"
    Config: ClassVar[type[BaseModel]]
    def preflight(self) -> None: ...
    @abstractmethod
    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult: ...

# _scoring/judge.py — M1a
class JudgeContext(CheckContext):
    def __init__(self, *, goal: Goal, executor: Executor, artifact_root: str, run_id: str = "local",
                 round: int = 0, part: str | None = None, fidelity: str = "", batch_id: str = "",
                 anchors: frozenset[str] = frozenset(), final: bool = False, read_cache: bool = True,
                 on_started: Callable[[Job.Started], Awaitable[None]] | None = None,
                 secrets_file: str | None = None, ids: IdGen | None = None,
                 tool_timeout_s: float = 3600.0) -> None: ...
    fidelity: str; batch_id: str; anchors: frozenset[str]; final: bool; read_cache: bool
    secrets_file: str | None; ids: IdGen; tool_timeout_s: float  # ids=None -> IdGen(run_id, 0)
    errors: dict[str, str]
    def judge_error(self, candidate_id: str, reason: str) -> None: ...  # first reason per candidate wins
    on_started: Callable[[Job.Started], Awaitable[None]] | None   # execute() puts it on the command Job; the core sets it
                                                                    # in a run (R4.14 command_start)

class Judge(Configurable, ABC):
    name: ClassVar[str]
    Config: ClassVar[type[BaseModel]]
    def preflight(self) -> None: ...
    def rubric_hash(self) -> str: ...           # [plan] default sha256(canonical(ref path + config))
    def panel_hash(self) -> str: ...            # default ""; a Score's own non-empty panel_hash wins
    def anchor_count(self) -> int: ...          # [plan] default 0; LLMJudge returns its anchors k
    def providers(self) -> frozenset[str]: ...  # [plan] default frozenset()
    def models(self) -> tuple[str, ...]: ...    # [plan] catalog aliases used; default ()
    @abstractmethod
    async def judge(self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext) -> list[Score]: ...
def is_finite_score(score: Score) -> bool: ...   # the finite-key check of M1b judge-error rule 3 (scores built with
                                                 # model_construct/model_copy skip the Score validator)

def stamp_scores(scores: Sequence[Score], *, judge: Judge, judge_name: str, ids: IdGen,
                 batch_id: str) -> list[Score]: ...
    # Always set id=ids.next("s"), judge=judge_name, and rubric_hash=judge.rubric_hash(); use the score's
    # own nonempty panel_hash/batch_id, else the judge's panel hash / supplied batch id; keep candidate_id.

# _scoring/program_judge.py — M1a
class ProgramJudge(Judge):
    name = "program"
    class Config(BaseModel):
        function: str                           # "module:qualname" of (Goal, Candidate) -> Score
    def __init__(self, function: "str | Callable[[Goal, Candidate], Score] | ProgramJudge.Config", /) -> None: ...
    # runs in the driver process (trusted human code); an exception or bad key → judge error for that candidate

# _scoring/lexicographic.py — M1a
class Lexicographic(Judge):
    name = "lexicographic"
    class Config(BaseModel):
        judges: list[Ref]
    def __init__(self, judges: "Sequence[Judge | Ref] | Lexicographic.Config", /) -> None: ...
    # [plan] runs every member on every candidate; key = concatenation of member keys in order; rubric_hash and
    # panel_hash combine the members' identities; a composite score's panel_hash combines each member score's effective
    # panel hash (its own, else member.panel_hash()), so a panel that dropped a member stays a separate identity (§4.9,
    # §6.4); anchor_count() is the members' max; providers() and models() are the members' union.

# _scoring/script_check.py — M1a
class ScriptCheck(Check):
    name = "script-check"
    class Config(BaseModel):
        command: list[str]
        python: str | None = None
        files: FileSet = FileSet()
        env: list[str] = []
        feedback: Literal["pass_fail", "reasons"] = "reasons"
        timeout_s: float | None = None
        grace_s: float = 10.0
        resources: Resources = Resources()
    def __init__(self, config: "ScriptCheck.Config | None" = None, /, **fields: Any) -> None: ...

# _scoring/script_judge.py — M1a
class ScriptJudge(Judge):
    name = "script-judge"
    class Config(BaseModel):
        command: list[str]
        python: str | None = None
        files: FileSet = FileSet()
        env: list[str] = []
        direction: Literal["min", "max"] = "max"    # [plan] default; used for {"value": x}
        timeout_s: float | None = None
        grace_s: float = 10.0
        resources: Resources = Resources()
    def __init__(self, config: "ScriptJudge.Config | None" = None, /, **fields: Any) -> None: ...

# _scoring/human_review.py — M2
class HumanReview:                              # [plan] not a Judge subclass: it never runs inside a round
    def __init__(self, reviewer: str) -> None: ...
    judge_name: str                             # "human:<reviewer>"
    async def export(self, run_id: str, *, config: RunConfig, out_dir: Path, top: int = 20,
                     goal: str | None = None) -> Path: ...
    async def import_csv(self, run_id: str, csv_path: Path, *, config: RunConfig) -> list[Score]: ...
```

### R4.4 `swarm_arena.manager` (+ `ScriptedManager` at top level)

```python
# _model/results.py — M1a
class AgentStatus(StrEnum):
    ACTIVE = "active"; FINISHED = "finished"; STARVED = "starved"; PRUNED = "pruned"
    STOPPED = "stopped"; WATCHDOG_STOPPED = "watchdog_stopped"

# _model/planning_types.py — M1b
class RoundReport(FrozenModel):
    summary: str = ""

# _agent/manager.py — M1b
class LimitReached(RuntimeError):
    def __init__(self, limit: str, message: str) -> None: ...
    limit: str                                  # the Limits field name, "budget", "usage_overflow", or "stop" [plan]
        # "stop": a stop request blocks new actions. "budget": the agent share, the goal pool, or the run cap; the
        # message says which. "usage_overflow": terminal run-wide exhaustion from over-range reported usage.

class Manager(ABC):
    Config: ClassVar[type[BaseModel]]
    def preflight(self) -> None: ...
    def models(self) -> tuple[str, ...]: ...    # [plan] catalog aliases used; default (); used by the fingerprint rule
    @abstractmethod
    async def run_round(self, ctx: AgentContext) -> RoundReport: ...

# _agent/context.py — M1b. Built only by the core; every action checks limits first.
class AgentContext:
    goal: Goal                                  # the agent's goal, or its part's sub-goal
    spec: AgentSpec
    round: int
    round_spec: RoundSpec
    digest: str
    notes: str
    budget: Estimate                            # what is left for this agent
    resumed: bool                               # [plan] true when the round restarted after resume
    seed_parents: tuple[str, ...]                # read-only assigned parent ids; prepended to explicit parents, stable first occurrence wins
    tool_description: str                       # [plan] Tool.get_description()
    input_model: type[BaseModel]                # [plan] Tool.get_input_model()
    editable: tuple[str, ...]                   # [plan] Tool.editable()
    actions: tuple[Action, ...]                 # [plan] Tool.extra_actions()
    async def run_tool(self, inputs: Mapping[str, Any], *, parents: Sequence[str] = (),
                       replicate: int | None = None, edits: Mapping[str, str] | None = None) -> Candidate: ...
    def my_candidates(self) -> list[Candidate]: ...
    def scores_of(self, candidate_id: str) -> list[Score]: ...                  # [plan]
    def search_learnings(self, query: str, *, tags: Sequence[str] = (),
                         scope: Literal["goal", "namespace"] = "goal", limit: int = 10) -> list[Learning]: ...
    def read_learning(self, id: str) -> Learning: ...
    def get_candidate(self, id: str) -> Candidate: ...
    def _read_artifact(self, sha256: str) -> bytes: ...  # [M1a plan, private]
    # Bytes of a file in the agent's view only; ScriptTool's list_files/read_file handlers
    # use it through _tooling/script_tool.py:FileViewContext.
    def best(self, k: int, *, part: str | None = None) -> list[Row]: ...
    # M1b: own group only (None or this agent's part); the round runner fills its ranking name.
    # [M2] Cross-part leaderboard reads arrive with multi-goal board views.
    async def post_learning(self, title: str, text: str, evidence: Sequence[str], *, tags: Sequence[str] = (),
                            refines: Sequence[str] = (), duplicate_reason: str | None = None) -> Learning: ...
    async def save_notes(self, text: str) -> None: ...
    async def call_action(self, name: str, args: Mapping[str, Any]) -> JsonValue: ...   # [plan] extra actions
    # Soft errors raise ValueError (private SoftError) with a message for the LLM. LimitReached ends the round.
    # run_tool and post_learning read the call-id seam (R4.13.7) into requested_by_call and posted_by_call.

# _agent/scripted.py — M1b
class ScriptedManager(Manager):
    class Request(BaseModel):                   # [plan]
        inputs: dict[str, Any] = {}
        parents: list[str] | Literal["best"] = []      # "best": the top row of ctx.best(1), if any
        edits: dict[str, str] = {}
        round: int | None = None                        # None → every round
    class Config(BaseModel):
        requests: list["ScriptedManager.Request"]
        post_learnings: bool = False            # [plan] post one learning per round citing its best candidate
    def __init__(self, requests: "Sequence[Mapping[str, Any] | ScriptedManager.Request] | ScriptedManager.Config") -> None: ...
    # positional or keyword: the spec form ScriptedManager(requests=[...]) (§4.5) works and is tested
```

### R4.5 `swarm_arena.planning` (+ the four policies at top level)

```python
# _model/plan.py — M1a
class FlowSpec(FrozenModel):
    sender: str
    recipient: str                              # an agent, a team, or "*"
    rationale: str

# _model/planning_types.py — M1b
class Decision(FrozenModel):
    prune: dict[str, str] = {}                  # "<team>" or "<team>/<agent>" → reason
    no_result: str | None = None
    policy: str = ""
    def merge(self, other: "Decision") -> "Decision": ...

class RoundState(FrozenModel):                  # [plan] fields
    goal_id: str
    round: int
    rounds_total: int
    plan: Plan                                  # the current revision
    candidates: list[Candidate]                 # this run, this goal, all rounds so far
    scores: list[Score]                         # this run, this goal
    status: dict[str, AgentStatus]              # "<team>/<agent>" → status this round
    ranking_identity: dict[str | None, tuple[str, str, str]]   # [plan] group (part; None = main goal) -> the identity of
        # the newest score that the group's ranking judge (part_scoring) gave a candidate of that group in this run;
        # a group with no such score is absent
    def groups(self) -> dict[str | None, list[str]]: ...                     # part (None = main goal) → team names; delegates to _scoring/ranking.py
    def ranked(self, group: str | None) -> list[tuple[Candidate, Score]]: ...  # complete, gated-in, newest fidelity; delegates to _scoring/ranking.py;
        # under ranking_identity[group]; absent -> []
    def team_best(self, team: str) -> Score | None: ...                      # under the identity of the team's group
    def team_best_by_round(self, team: str) -> list[Score | None]: ...       # under the identity of the team's group

class History(FrozenModel):                     # [plan] fields
    goal_id: str
    revisions: list[Plan]
    scores: list[Score]
    learnings: list[Learning]
    decisions: list[tuple[int, Decision]]       # (round, decision): the prune log
    budget_left: Estimate
    rounds_done: int
    join_failures: list[str] = []

# _planning/planner.py — M1b
class Planner(ABC):
    origin: ClassVar[Origin] = "llm"            # [plan] the provenance the core stamps; never "human" for a subclass
    def models(self) -> tuple[str, ...]: ...    # [plan] default ()
    @abstractmethod
    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan: ...
    # Stamping [plan]: origin, revision, and parent_revision in a planner's raw output are always overwritten by the core
    # stamp() (R4.13.12). LLMPlanner and CollaborativePlanner (core code in swarm_arena.llm) stamp type(self).origin as
    # revision 0 before they return, a reviewer's Edited plan included, so a Python caller that passes the result to
    # Swarm.run, and the file that `swarm-arena plan --llm/--together` writes, keep "llm" or "together", and the
    # planner-catalog boundary applies to them (§4.1, §4.6). The replanner hook re-stamps "autonomous" with revision and
    # parent_revision. Swarm.run and Plan.load never raise a non-human origin.

# _planning/policy.py — M1b
class Policy(ABC):
    Config: ClassVar[type[BaseModel]]
    @abstractmethod
    def after_round(self, state: RoundState) -> Decision: ...

# _planning/policies.py — M1b
class KeepTopTeams(Policy):
    class Config(BaseModel): k: int
    def __init__(self, k: "int | KeepTopTeams.Config", /) -> None: ...
class SuccessiveHalving(Policy):
    class Config(BaseModel): keep: float = 0.5; min_per_team: int = 1
    def __init__(self, keep: "float | SuccessiveHalving.Config" = 0.5, min_per_team: int = 1) -> None: ...
class Patience(Policy):
    class Config(BaseModel): rounds: int = 2
    def __init__(self, rounds: "int | Patience.Config" = 2) -> None: ...
# Positional or keyword, so the spec forms SuccessiveHalving(keep=0.5, min_per_team=1) and Patience(rounds=2) (§4.7)
# work; tests use them.
class Chain(Policy):
    class Config(BaseModel): policies: list[Ref]
    def __init__(self, *policies: "Policy | Ref | Chain.Config") -> None: ...
# All policies rank per group, never prune a part's last team, and never prune a starved agent for lack of results.
```

### R4.6 `swarm_arena.board`

```python
# _model/goal.py — M1a
class GoalKey(FrozenModel):
    namespace: str
    id: str
    inputs_sha256: str                          # sha256(canonical(inputs))
    files_sha256: str                           # sha256(canonical(sorted manifest))
    @classmethod
    def of(cls, goal: Goal) -> "GoalKey": ...
    # One key identifies exactly one main-goal or part group per run. Admission rejects part/part,
    # main/part and cross-plan collisions with identifier-only PlanError naming the goal/part groups.

# _model/results.py — M1a
class Candidate(StoredModel):
    id: str
    run_id: str
    goal_id: str
    part: str | None = None                     # [plan] stored so cross-run ranking groups need no plan join
    team: str
    agent: str
    round: int
    fidelity: str = ""
    tool_identity: str = ""                     # [plan] R4.13.9 identity of the tool that made it; set by the core when the
                                                # candidate is made and journaled with it; "" never matches, so it is never overlaid
    request: ToolRequest
    output: ToolOutput
    checks: list[CheckResult] = []
    parent_ids: list[str] = []
    overlay: dict[str, str] = {}
    read_ids: list[str] = []
    requested_by_call: str | None = None
    llm_call_ids: list[str] = []
    llm_calls_source: Literal["gateway", "reported", "langfuse", "none", "unknown"] = "none"
    origin: Literal["agent", "human"] = "agent"
    warnings: list[str] = []                    # [plan] soft staging warnings (base parent not overlaid) and the past-run
                                                # hint entry "past-run match <id>: <judge>=<value>" (M2, §6.2)
    @property
    def gated_out(self) -> bool: ...            # derived, never stored

class Learning(StoredModel):
    id: str
    namespace: str
    run_id: str
    goal_id: str
    round: int
    author: str                                 # "team/agent" or "human:<name>"
    title: str                                  # 1..120 chars, one line
    text: str                                   # ≤ 1500 chars
    evidence: list[str] = []                    # ≥ 1 unless author starts with "human:"
    tags: list[str] = []
    verified: bool = False
    pinned: bool = False
    hidden: bool = False
    refines: list[str] = []
    supersedes: list[str] = []
    posted_by_call: str | None = None
    @property
    def from_human(self) -> bool: ...

class LLMCall(StoredModel):
    id: str
    run_id: str
    goal_id: str
    team: str = ""
    agent: str = ""
    round: int = 0
    role: CallRole
    candidate_id: str | None = None
    model_alias: str | None = None
    model: str
    provider: str
    request_ref: str | None = None              # sha256 in artifact_root; None when store_payloads is false
    response_ref: str | None = None
    tool_calls: list[dict[str, str]] = []       # [plan] [{"id", "name"}]
    usage: Usage = Usage()
    cache_hit: bool = False
    cached_from_call_id: str | None = None
    reservation_id: str | None = None
    cost_usd: float = 0.0
    latency_ms: float | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    trace_url: str | None = None
    parent_call_id: str | None = None
    litellm_call_id: str | None = None          # [plan] dedupe key for gateway rows

# _model/board_types.py — M1b
class Frontier(StoredModel):
    class Mark(FrozenModel):
        run_id: str; run_root: str; epoch: int; seq: int; segment: str
    marks: list["Frontier.Mark"] = []
    reviews: list[str] = []                     # exact object names
    curation: list[str] = []
    def digest(self) -> str: ...

class Row(FrozenModel):
    rank: int
    candidate_id: str
    run_id: str
    goal_id: str
    part: str | None
    team: str
    agent: str
    fidelity: str
    key: tuple[float, ...]
    value: float | None
    judge: str
    rubric_hash: str
    panel_hash: str
    built_on: list[str]
    partial: bool

class LineageGraph(FrozenModel):
    root: str
    nodes: dict[str, Candidate]
    edges: list[tuple[str, str]]                # (child, parent)
    def parents(self, candidate_id: str) -> list[str]: ...
    def ancestors(self, candidate_id: str) -> list[str]: ...

# _board/base.py — M1b. Sync methods. Threading rule [plan]: read methods (of Board and BoardView) and put_* run on the
# event-loop thread (they are local and short; AgentContext reads are sync); open, close, frontier, and snapshot (storage
# I/O) run through anyio.to_thread.run_sync. A board that shares state across these threads locks it (ArenaCache, R4.13.14).
class BoardView(ABC):                           # one goal's round-start snapshot
    goal: GoalKey
    frontier: Frontier
    def _visible_candidate_ids(self) -> frozenset[str]: ...  # private seam: fixed snapshot goal/part membership, no I/O or charges; M2 views implement it too
    # abstract read methods with signatures identical to Board's below, limited to the snapshot:
    # best, learnings, search_learnings, get_learning, get_candidate, scores, lineage, leaderboard.
    # A namespace or goal argument that differs from the view's raises ValueError [plan].

class Board(ABC):
    Config: ClassVar[type[BaseModel]]
    def open(self, run_id: str, run_root: str, namespaces: Sequence[str]) -> None: ...   # [plan] optional; ArenaBoard writes pointers
    def close(self) -> None: ...                                                        # [plan] optional
    # put_* index items the core has already journaled [plan]; they never write the journal themselves (R5.4 item 3).
    # epoch and seq are the position of the journal event that recorded the item.
    @abstractmethod
    def put_candidate(self, candidate: Candidate, *, epoch: int, seq: int) -> None: ...
    @abstractmethod
    def put_score(self, score: Score, *, epoch: int, seq: int) -> None: ...
    @abstractmethod
    def put_learning(self, learning: Learning, *, epoch: int, seq: int) -> None: ...
    @abstractmethod
    def put_decision(self, goal_id: str, round: int, decision: Decision, *, epoch: int, seq: int) -> None: ...
    # Snapshot membership [plan]: snapshot(goal, frontier) holds an item of run R only if its event is on the chain named
    # by R's mark (§7.2), so other agents' items of the same round stay hidden until the barrier, and the same frontier
    # gives the same view after resume. RunBoard (one process; fed by put_*, and on resume rebuilt from RunState's
    # committed events and positions) uses seq <= mark.seq, which is exact for one owner. ArenaBoard indexes every run,
    # its own included, only from committed chain segments; its put_* rows are hints that snapshot never returns.
    @abstractmethod
    def frontier(self) -> Frontier: ...
    @abstractmethod
    def snapshot(self, goal: GoalKey, frontier: Frontier) -> BoardView: ...
    @abstractmethod
    def best(self, goal: GoalKey, k: int, *, judge: str, run_id: str | None = None,
             part: str | None = None, fidelity: str | None = None) -> list[Candidate]: ...
    @abstractmethod
    def learnings(self, namespace: str, *, tags: Sequence[str] = (), goal: GoalKey | None = None,
                  limit: int = 50) -> list[Learning]: ...
    @abstractmethod
    def search_learnings(self, namespace: str, query: str, *, tags: Sequence[str] = (),
                         goal: GoalKey | None = None, limit: int = 10) -> list[Learning]: ...
    @abstractmethod
    def get_learning(self, id: str) -> Learning: ...
    @abstractmethod
    def get_candidate(self, id: str) -> Candidate: ...
    @abstractmethod
    def scores(self, candidate_id: str) -> list[Score]: ...          # [plan] needed by read_learning and pages
    @abstractmethod
    def lineage(self, candidate_id: str) -> LineageGraph: ...
    @abstractmethod
    def leaderboard(self, goal: GoalKey, *, judge: str) -> list[Row]: ...

# _board/curator.py — M2
class Curator(ABC):
    Config: ClassVar[type[BaseModel]]
    def models(self) -> tuple[str, ...]: ...    # [plan] default ()
    @abstractmethod
    async def curate(self, namespace: str, new: Sequence[Learning], view: BoardView) -> list[Learning]: ...

# _board/arena.py — M2
class ArenaBoard(Board):
    class Config(BaseModel):
        arena_root: str
        cache_dir: str | None = None            # None → ~/.cache/swarm-arena/<sha256(arena_root)[:12]> [plan]
    def __init__(self, config: "ArenaBoard.Config | None" = None, /, **fields: Any) -> None: ...
```

### R4.7 `swarm_arena.executor`

```python
# _exec/executor.py — M1a
@dataclass(frozen=True, kw_only=True)
class Job:
    class Stage(FrozenModel):                   # what the workspace holds; computed in the driver (R4.13.9)
        root: dict[str, str] = {}               # path → sha256 at the workspace root; keys in normal form
        parents: list[ParentRef] = []           # staged read-only under parents/<id>/
        candidate: "Job.CandidateFiles | None" = None
        request_json: dict[str, Any] | None = None   # written to swarm/request.json
    class CandidateFiles(FrozenModel):          # the candidate/ dir of a script check or judge
        request: dict[str, Any]
        output: dict[str, Any]
        files: dict[str, str]                   # effective files + output files, path → sha256; keys in normal form
    class Started(StoredModel):                 # the committed tool-run start (§4.8, §11)
        kind: Literal["process", "dispatch"]
        job_id: str
        host_id: str | None = None
        pid: int | None = None
        pgid: int | None = None
        start_time: str | None = None
        task_name: str | None = None
    id: str                                     # candidate id or a command-job id "x-<run tag>-e<epoch>-<n>" (R4.0)
    kind: Literal["tool", "command"]
    tool: Ref | None = None                     # kind="tool"
    command: Command | None = None              # kind="command" (ctx.execute)
    request: ToolRequest | None = None
    stage: Job.Stage
    timeout_s: float | None
    grace_s: float
    resources: Resources
    env: Mapping[str, str]                      # non-secret values (SWARM_ARENA_*, gateway URLs)
    secret_env: tuple[str, ...]                 # names resolved on the node
    secrets_file: str | None
    artifact_root: str
    secret_values: Mapping[str, str] = field(default_factory=dict, repr=False)   # [plan] secret values the driver makes
        # (gateway virtual keys as OPENAI_API_KEY and ANTHROPIC_API_KEY, M4); they win over env and secret_env. Never
        # journaled, logged, or put in Ray runtime_env; sent to Ray only inside the task payload (R4.13.16). On Ray the
        # payload can sit in the object store, which can spill to worker-local disk (documented in M4 item 9).
    drop_env: tuple[str, ...] = ()              # [plan] M4: names build_env never takes from the node or the secrets file.
        # gateway_job sets every PROVIDER_KEYS name on a keyed run. With a gateway, every unkeyed tool job gets
        # REPLACED_KEYS. So no tool run gets a real OpenAI or Anthropic key from the node (spec 7.3 item 2a).
    on_started: Callable[["Job.Started"], Awaitable[None]] | None = None   # never serialized; None -> the executor starts
        # the job with no callback (LocalExecutor and RayExecutor). In a run, the core sets it for every job:
        # tool_start / tool_dispatch for tools, command_start for command jobs
    on_partial: Callable[[ToolOutput], Awaitable[None]] | None = None   # never serialized; the only channel by which an executor hands a cancel-path partial output to the caller (M1b's `execute_tool_run` sets it)

class Executor(ABC):
    Config: ClassVar[type[BaseModel]]
    def preflight(self) -> None: ...
    @abstractmethod
    async def submit(self, job: Job) -> ToolOutput: ...      # cancelling the await stops the job (grace + partial collect)
    @abstractmethod
    async def capacity(self) -> Resources | None: ...
    @abstractmethod
    async def aclose(self) -> None: ...
    async def confirm_dead(self, started: Job.Started) -> bool | None: ...   # [plan] default None = cannot tell
    async def kill_orphan(self, started: Job.Started) -> bool: ...          # [plan] default False

# _exec/local.py — M1a
class LocalExecutor(Executor):
    class Config(BaseModel):
        max_concurrent: int | None = None       # None → os.cpu_count()
        workspace_root: str | None = None       # None → tempfile.gettempdir()/swarm-arena-<uid> [plan, per user]
        keep_workspaces: bool = False
        rlimits: bool = True
        max_log_bytes: int = 8 * 2**20          # [plan] per log; the last max_log_bytes are uploaded, with a note
        max_output_bytes: int = 256 * 2**20     # [plan] all declared outputs of one run (the §11 Ray cap); over → error
    def __init__(self, config: "LocalExecutor.Config | None" = None, /, **fields: Any) -> None: ...
    # preflight() refuses a non-POSIX platform [plan].
    # [plan, M1a] The executor runs with `Job.timeout_s` and `Job.grace_s` and writes them into `Command.timeout_s` and `Command.grace_s`; the driver computes them with `effective_tool_timeout` and `tool_grace` (`_tooling/tool.py`).
    # [plan, M4] confirm_dead(kind "process") on this host: True if not group_has_live_member(pgid), or if process_start_time(pid)
    # differs from started.start_time. False if the group is alive with the recorded start time or an unknown one. None on
    # another host or for kind "dispatch".
```

### R4.8 `swarm_arena.config`

```python
# _model/catalog.py — M1a
class ModelSpec(FrozenModel):
    class PromptCache(FrozenModel):
        enabled: bool = True
        ttl: Literal["5m", "1h"] = "5m"
    class Prices(FrozenModel):                  # USD per 1M tokens
        input: float
        output: float
        cache_read: float | None = None
        cache_write: float | None = None
    model: str                                  # LiteLLM "provider/model"
    max_tokens: int = 4096                      # [plan] default
    roles: list[LLMRole]
    prompt_cache: "ModelSpec.PromptCache" = PromptCache()
    prices: "ModelSpec.Prices | None" = None
    @property
    def provider(self) -> str: ...              # text before the first "/"

class ModelCatalog(RootModel[dict[str, ModelSpec]]):
    def resolve(self, alias: str, role: LLMRole) -> ModelSpec: ...   # ValueError for an unknown alias or role

class CatalogEntry(FrozenModel):
    name: str
    kind: CatalogKind                           # [plan] where the entry may be used
    ref: str                                    # import path
    description: str
    config: dict[str, Any] = {}
    settable: dict[str, Any] = {}               # JSON-schema subset of an object; {} = nothing settable
    def settable_model(self) -> type[BaseModel]: ...

class PlannerCatalog(FrozenModel):
    entries: list[CatalogEntry] = []            # load fails for a script component with a human-only field in settable
    def get(self, name: str) -> CatalogEntry: ...
    def match(self, ref: Ref, kind: CatalogKind) -> CatalogEntry | None: ...

# _model/settings.py — M1a
class SecretsPolicy(FrozenModel):
    file: str | None = None                     # KEY=VALUE lines, a path present on each node [plan format]
    allow: list[str] = []

class TracingConfig(FrozenModel):
    store_payloads: bool = True
    langfuse: bool = False                      # [plan] Langfuse reads its own LANGFUSE_* env vars

class GatewayConfig(FrozenModel):
    url: str
    master_key_env: str = "LITELLM_MASTER_KEY"  # [plan] name only, never the value
    require_hard_cap: bool | None = None        # None → True when limits.max_usd is set
    request_timeout_s: float = 600.0            # [plan] default
    key_duration_s: float = 86_400.0            # [plan]
    default_estimated_output_tokens: int = 4096 # [plan]
    def hard_cap_required(self, limits: Limits) -> bool: ...

class CachePolicy(FrozenModel):
    class RoleSwitch(FrozenModel):
        read: bool
        write: bool
    backend: Literal["disk", "s3", "redis"]
    s3_path: str | None = None
    disk_dir: str | None = None                 # [plan]
    redis_url_env: str | None = None            # [plan] env var name
    namespace: str | None = None
    ttl_s: int = 2_592_000
    roles: dict[LLMRole, "CachePolicy.RoleSwitch"] = {
        "judge": RoleSwitch(read=True, write=True), "planner": RoleSwitch(read=False, write=True),
        "manager": RoleSwitch(read=False, write=False), "curator": RoleSwitch(read=True, write=True)}   # [plan] = spec example
```

### R4.9 `swarm_arena.llm` (extra `llm`) — M3

```python
class BudgetExceeded(AgentsException, LimitReached): ...      # [plan] also a LimitReached, so the core ends the round

class AgentsSDKManager(Manager):
    class Config(BaseModel):
        model: str                              # catalog alias, role "manager"
        instructions: str = ""
        seed_inputs: dict[str, Any] = {}
        temperature: float | None = None
        action_timeout_s: float = 60.0          # [plan] SDK timeout for actions other than run_tool
        parallel_tool_calls: bool = True
    def __init__(self, config: "AgentsSDKManager.Config | None" = None, /, **fields: Any) -> None: ...

class LLMJudge(Judge):
    name = "llm"
    class Config(BaseModel):
        rubric: str                             # includes the scale anchors
        models: list[str]                       # aliases, role "judge"; a panel when len > 1
        aggregate: Literal["mean", "median", "min", "max"] = "mean"   # [plan] set
        scale_min: float = 0.0
        scale_max: float = 10.0
        anchors: int = 2                        # k leaders re-scored per batch
        batch_size: int = 8
    def __init__(self, rubric: "str | LLMJudge.Config", models: Sequence[str] = (), aggregate: str = "mean", /) -> None: ...

class LLMPlanner(Planner):
    origin = "llm"
    def __init__(self, model: str, *, instructions: str = "") -> None: ...

class Approve: ...                              # frozen dataclasses
class Feedback: text: str
class Edited: plan: Plan
class Reviewer(Protocol):
    async def review(self, plan: Plan) -> Approve | Feedback | Edited: ...

class CollaborativePlanner(Planner):
    origin = "together"
    def __init__(self, planner: LLMPlanner, reviewer: Reviewer, *, max_iterations: int = 5) -> None: ...

class LLMCurator(Curator):
    class Config(BaseModel):
        model: str                              # alias, role "curator"
        max_group: int = 8                      # [plan]
    def __init__(self, config: "LLMCurator.Config | None" = None, /, **fields: Any) -> None: ...
```

### R4.10 `swarm_arena.ray` (extra `ray`) — M4

```python
class RayExecutor(Executor):
    class Config(BaseModel):
        address: str = "auto"
        poll_s: float = 0.2
        cancel_margin_s: float = 10.0           # [plan] wait = grace_s + margin before force cancel
        object_store_cap_mib: int = 256
        secret_resource: str | None = None      # [plan] e.g. "llm_keys", added for jobs with secret_env
    def __init__(self, config: "RayExecutor.Config | None" = None, /, **fields: Any) -> None: ...
```

### R4.11 `swarm_arena.protocol` — M1a

Frozen dataclasses (`slots=True, kw_only=True`) mirroring A2A v1.0.1 field for field, as in `tasks/research/api-a2a.md` "Recommendation": `Role`, `TaskState` (`StrEnum`, values are the proto names, all 9 states), `Part` (exactly one of `text`, `raw`, `url`, `data`), `Message`, `Artifact`, `TaskStatus`, `Task`. Helper (private, `_model/protocol.py`): `def task_state_for(status: RunStatus) -> tuple[TaskState, dict[str, str]]` (`cancelled` → `CANCELED`; `budget_exceeded`, `timeout`, `lost` → `FAILED` with `metadata["swarm_arena.reason"]`; [plan] `error`, `oom`, `provider_failed` → `FAILED` with the reason; `ok` → `COMPLETED`).

### R4.12 `swarm_arena.testing`

```python
# testing/_conformance.py
async def check_tool(tool: Tool | Ref, inputs: Mapping[str, Any], *, goal: Goal | None = None,
                     parents: Sequence[Candidate] = (), edits: Mapping[str, str] | None = None,
                     round: int = 0, round_overrides: Mapping[str, Any] | None = None, replicate: int = 0,
                     executor: Executor | None = None, timeout_s: float = 60.0) -> ToolOutput: ...      # M1a
    # each parent becomes a ParentRef with its Candidate.tool_identity, so the overlay rule runs as in a swarm;
    # round overrides replace top-level tool config settings, validated before snapshots/input/staging/timeout;
    # identity hashes the snapshotted base ref (entry=None), without overrides; effective tools enforce protection;
    # human-only settings and their validation aliases cannot be overridden;
    # command/collect and swarm/request.json share round, round_overrides, replicate; SWARM_ARENA_ROUND shares round;
    # the smaller of the effective tool timeout and the kit timeout_s cap wins;
    # any Executor works, so M4 runs it with RayExecutor and changes no code here
async def check_check(check: Check | Ref, candidate: Candidate, *, goal: Goal | None = None,
                      executor: Executor | None = None) -> CheckResult: ...                            # M1a
async def check_judge(judge: Judge | Ref, goal: Goal, candidates: Sequence[Candidate], *,
                      executor: Executor | None = None, config: RunConfig | None = None) -> list[Score]: ...   # M1a (+M3 config)
    # [plan] M3 adds config (additive; §9 rule 4): an LLMJudge needs a model catalog and an LLMScope to run offline.
    # None → RunConfig defaults.
async def check_manager(manager: Manager | Ref, *, goal: Goal, tool: Tool | Ref, rounds: int = 1,
                        limits: Limits | None = None, config: RunConfig | None = None) -> list[RoundReport]: ...  # M1b (+M3 parallel calls)
# testing/_candidates.py — M1a
def make_candidate(output: ToolOutput, *, request: ToolRequest | None = None, tool: Tool | Ref | None = None,
                   parents: Sequence[Candidate] = (), goal: Goal | None = None,
                   **fields: Any) -> Candidate: ...   # tool → sets tool_identity (no catalog entry), for overlay tests
    # request.round_overrides select effective behavior; identity hashes the snapshotted base ref (entry=None);
    # request.round supplies the default round;
    # accepted base overlays chain, goal supplies default goal_id, and parents populate request.parents and parent_ids
# testing/_artifacts.py — M1a
def read_output_file(output: ToolOutput, path: str) -> bytes: ...
def read_logs(output: ToolOutput) -> tuple[bytes, bytes]: ...   # stdout, stderr; verified stored bytes
# testing/_fake_model.py — M3 (lazy import of `agents`)
@dataclass(frozen=True, kw_only=True)
class FakeTurn:
    text: str | None = None
    tool_calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    usage: Usage = Usage(input_tokens=100, output_tokens=20)
class FakeModel:                                # implements agents.models.interface.Model
    def __init__(self, turns: Sequence[FakeTurn], *, provider: str = "fake") -> None: ...
    def register(self, name: str) -> None: ...  # makes ModelSpec(model="fake/<name>") resolve to this model
```

`testing.__all__` [plan] = `check_tool`, `check_check`, `check_judge`, `check_manager`, `make_candidate`, `read_output_file`, `read_logs`, `FakeModel`, `FakeTurn`. `FakeModel` calls emit `LLMCall` records through the same core `record_call` function as the LiteLLM logger (R4.13.19), so the ledger and journal path is the same below the LiteLLM boundary.

### R4.13 Internal seams (private, binding across milestones)

A type named in a signature below but not defined in this file (for example `GoalOutcome`, `StatusReport`, `JudgeOutcome`, `GoalState`, `ReservationState`, `TaskInfo`, `ReconcileReport`, `MigrationFn`) is used inside one milestone only. That milestone's plan defines it.

#### R4.13.0 Model helpers — M1a

```python
# _model/canonical.py
def canonical_json(value: JsonValue) -> bytes: ...   # sort_keys, separators=(",", ":"), ensure_ascii=False, allow_nan=False
def sha256_hex(data: bytes) -> str: ...
def utc_now() -> datetime: ...
# _model/migrate.py
CURRENT: dict[str, int]                              # stored-format kind → current schema_version (R4.15)
def migration(kind: str, from_version: int) -> Callable[[MigrationFn], MigrationFn]: ...
def upgrade(kind: str, raw: dict[str, Any]) -> dict[str, Any]: ...   # ValueError for a newer version
# _model/toml_io.py
def read_toml(path: Path) -> dict[str, Any]: ...
def dumps_toml(data: Mapping[str, Any]) -> str: ...  # deterministic; round-trips through tomllib
# _model/schema.py — keywords: type, enum, minimum, maximum, exclusiveMinimum, exclusiveMaximum,
#   minLength, maxLength, pattern, items, default, description; others → ValueError at load
def compile_object_schema(name: str, properties: Mapping[str, Mapping[str, Any]]) -> type[BaseModel]: ...
```

#### R4.13.1 Errors (`_errors.py`) — M1a (+M1b)

```python
class PlanError(ValueError):            errors: list[str]
class ConfigError(ValueError): ...
class SoftError(ValueError): ...                     # soft action errors (§4.5)
class StorageError(RuntimeError): ...
class NotFound(StorageError): ...
class AlreadyExists(StorageError): ...
class PreconditionFailed(StorageError): ...          # ETag or version mismatch
class WriteOutcomeUnknown(StorageError): ...         # timeout or lost response
class ConditionalWritesUnsupported(StorageError): ...
class ChainError(StorageError): ...                  # M1b: chain check failed
class Fenced(RuntimeError): ...                      # M1b: this owner lost the run
class LeaseHeld(RuntimeError): ...                   # M1b: a live owner holds the lease
```

#### R4.13.2 Store and artifacts — M1a (`replace` M1b, `S3Store` M2)

```python
class Blob(NamedTuple):
    data: bytes
    etag: str

class Store(ABC):                                    # sync; async callers use anyio.to_thread.run_sync
    url: str
    @abstractmethod
    def read(self, key: str) -> Blob: ...            # NotFound
    @abstractmethod
    def create(self, key: str, data: bytes) -> str: ...                    # conditional create → etag; AlreadyExists
    @abstractmethod
    def replace(self, key: str, data: bytes, *, if_match: str) -> str: ... # PreconditionFailed, NotFound
    @abstractmethod
    def exists(self, key: str) -> bool: ...
    @abstractmethod
    def list(self, prefix: str) -> list[str]: ...    # only for runs/, reviews/, curation/; never for journal/
    def child(self, prefix: str) -> "Store": ...
    def probe_conditional_writes(self) -> None: ...  # ConditionalWritesUnsupported
# There is no unconditional put [plan]: every key is either write-once (create) or replaced with if_match.

class LocalStore(Store):
    def __init__(self, root: Path) -> None: ...
    # create: O_EXCL on a temp name, then os.link to the key (fails if it exists); etag = sha256 of bytes.
    # replace: fcntl.flock on "<key>.lock", compare the etag, write a temp file, os.replace.
class S3Store(Store):                                # M2, extra s3
    def __init__(self, url: str, **storage_options: Any) -> None: ...
def open_store(url: str) -> Store: ...               # local path → LocalStore; "s3://" → S3Store; else ConditionalWritesUnsupported

class ArtifactStore:
    def __init__(self, store: Store) -> None: ...
    def put_bytes(self, data: bytes) -> str: ...     # sha256; AlreadyExists with the same hash = success
    def put_file(self, path: Path) -> FileRef: ...
    def get_bytes(self, sha256: str) -> bytes: ...   # verifies the hash
    def fetch(self, sha256: str, dest: Path, *, read_only: bool = True) -> None: ...
```

#### R4.13.3 Journal, lease, segments — M1b

```python
class Head(FrozenModel):
    epoch: int; seq: int; segment: str               # segment = "journal/e<epoch>-<last-seq>.jsonl"

class OwnerDoc(StoredModel):                         # control/owner.json
    version: int                                     # [plan] +1 on every replace (the local "version number in the file")
    epoch: int
    owner_id: str
    host: str
    expires_at: datetime
    head: Head | None = None

class EventDraft(FrozenModel):
    type: EventType
    goal_id: str = ""; team: str = ""; agent: str = ""; round: int = 0; task_id: str = ""
    payload: dict[str, Any] = {}

class Event(StoredModel):
    epoch: int; seq: int; ts: datetime; run_id: str; context_id: str; task_id: str = ""
    goal_id: str = ""; team: str = ""; agent: str = ""; round: int = 0
    type: EventType
    payload: dict[str, Any] = {}

def segment_name(epoch: int, last_seq: int) -> str: ...
def encode_segment(link: str | None, events: Sequence[Event]) -> bytes: ...
    # line 1: {"schema_version": 1, "link": <previous segment name or null>}; then one Event per line
def decode_segment(data: bytes) -> tuple[str | None, list[Event]]: ...

class Lease:                                         # sync; the writer task calls it through anyio.to_thread
    doc: OwnerDoc
    etag: str
    @classmethod
    def acquire_new(cls, store: Store, *, owner_id: str, host: str, ttl_s: float) -> "Lease": ...   # create
    @classmethod
    def take_over(cls, store: Store, *, owner_id: str, host: str, ttl_s: float, force: bool,
                  expect_epoch: int | None = None) -> "Lease": ...
        # one replace: epoch + 1, new owner, head kept; LeaseHeld if not expired and not force;
        # Fenced when the read epoch differs from expect_epoch (a concurrent takeover)
    def advance(self, head: Head) -> None: ...       # replace; also renews; Fenced on PreconditionFailed
    def renew(self) -> None: ...
    def release(self) -> None: ...                   # expires_at in the past; epoch and head kept
def read_owner(store: Store) -> tuple[OwnerDoc, str] | None: ...      # the authority read

class JournalWriter:
    def __init__(self, store: Store, lease: Lease, *, run_id: str, next_seq: int, local_path: Path,
                 sync_interval_s: float, group_window_s: float = 0.2) -> None: ...
    async def __aenter__(self) -> "JournalWriter": ...   # starts the writer task and the renewer
    async def __aexit__(self, *exc: object) -> None: ... # final commit, then release
    def append(self, draft: EventDraft) -> Event: ...    # ordinary event: durable at the next commit
    async def append_durable(self, draft: EventDraft) -> Event: ...   # write-through: returns after the head advance covers it
    async def commit(self) -> Head | None: ...           # round boundary and run end
    @property
    def head(self) -> Head | None: ...
    @property
    def committed(self) -> Sequence[str]: ...            # segment names on the chain, oldest first
    fenced: anyio.Event                                  # set on Fenced or on lease expiry; the run stops

def read_chain(store: Store, head: Head, *, stop_after: str | None = None) -> list[tuple[str, list[Event]]]: ...
    # follows links back from head.segment; stops at stop_after (exclusive) or the first segment;
    # returns oldest first; ChainError if a seq gap or a head mismatch is found. Consecutive seqs inside each segment;
    # each link must name the previous seq (stop_after included); each decoded segment's last seq must equal the seq its
    # name says (head included); ChainError otherwise
```

#### R4.13.4 Views and replay — M1b

```python
class ViewWriter:
    def __init__(self, store: Store, writer: JournalWriter) -> None: ...
    async def write(self, key: str, content: bytes, *, source: Head) -> None: ...
        # create; on AlreadyExists: same sha256 → done; else replace only if the existing source head is an
        # earlier segment on writer.committed
def embed_source_head(fmt: Literal["json", "toml", "md"], content: bytes, source: Head) -> bytes: ...
    # [plan] json: top-level "source_head"; toml: first line "# swarm-arena source_head: e<epoch>/<seq>/<segment>";
    # md: first line "<!-- swarm-arena source_head: ... -->"
def read_source_head(fmt: Literal["json", "toml", "md"], content: bytes) -> Head | None: ...

class RunState:                                      # pure fold of committed events; used by resume, status, views
    @classmethod
    def from_events(cls, events: Iterable[Event]) -> "RunState": ...
    run_id: str
    manifest: dict[str, Any]                        # run_started.manifest, independent of the event payload
    run_elapsed_s: float                            # maximum journaled active elapsed time
    epoch: int                                      # latest committed ownership epoch
    lost: set[str]                                  # candidate ids recorded by run_lost
    config: RunConfig                                # [plan] the journaled run_config with limits from the last
                                                     # limits_changed.new
    plans: dict[str, list[tuple[Plan, Origin]]]      # goal → revisions; origin from the journal only
    goals: dict[str, GoalState]                      # status, round, agents, decisions, frontier per round; [plan]
                                                     # GoalState.status "stopped" when goal_finished.reason is a stop reason (R4.14);
                                                     # a goal_started after a goal_finished (a continuation) sets status
                                                     # running and starts a new lifecycle; converge-team round_started
                                                     # preserves ordinary round/rounds_closed and finalization state,
                                                     # recording its own converge_round and frontier (converge=true)
    candidates: dict[str, Candidate]
    fingerprints: dict[str, str | None]             # tool_end fingerprint from the producing effective tool; null is always distinct
    scores: dict[str, Score]
    learnings: dict[str, Learning]
    positions: dict[str, tuple[int, int]]            # [plan] item id → (epoch, seq) of its event; feeds RunBoard on resume (R4.6)
    reservations: dict[str, ReservationState]        # open / settled / estimated
    ledger_events: list[Event]                       # [plan] committed run_config, limits_changed, run_started, goal_started,
                                                     # round_started, reserve, settle, goal_finished in seq order;
                                                     # Ledger.replay re-applies them
    started: dict[str, Job.Started]                  # tool runs with a start and no completion
    started_commands: dict[str, Job.Started]         # [plan] command jobs with a command_start and no command_end; resume
                                                     # kills their orphans
    id_counters: dict[str, int]
    stop: Literal["none", "stop", "stop-now"]        # this epoch only: a takeover event resets it to "none"
    llm_request_sha256: dict[str, str]               # [plan] M3: LLM call id → request sha256, from llm_call events (R4.13.19)
```

`GoalState.no_result_reason` retains `no_result.reason` independently of
`end_reason` (`goal_finished.reason`) and resets on continuation. The existing
`final_order` and `final_parts` fields retain `final_order.candidate_ids` and
`final_order.parts`; a null parts mapping denotes a main-goal or joined result.
`GoalState.kept_parts` folds `part_kept.{part,candidate_id}`. `join_attempts`
folds `join_failed.attempt`; `join_failures` accumulates its reasons in order.
These three fields reset on a continued lifecycle or a changed partition,
not on an ordinary round. Candidates, repeat keys, and fingerprints survive
continuation. `plan_rejected` never adds a plan revision.

Optional payload keys added in M1b fix rounds (journal v1 is unreleased, so no
schema_version bump): `plan_revision.preflight_complete: bool` is true only after
the shared component preflight succeeded for that revision (startup revisions carry
true); `plan_revision.preflight_only: bool` marks a recovery checkpoint that
completes preflight for an existing revision without adding a revision or moving its
first round; `plan_rejected.rejected_revision: int` refuses, on recovery, an accepted
revision that never completed preflight (a nonzero revision leaves the effective
history and dispatch resumes from the previous admitted revision; refusing revision 0
ends the goal without dispatch). `plan_revision.join_retry` and
`plan_rejected.join_retry` count completed retry proposals (one-based, independent of
`join_failed.attempt`; accepted and refused proposals both use the allowance).
`GoalState.join_phase` (`converge`, `replan`, `rounds`) and `join_retries` replay the
owed join-retry transition; `preflight_revisions` replays preflight completion.
Every engine-run event may carry `run_elapsed_s: float`, the accumulated active-driver
checkpoint; replay folds the maximum over all events and excludes downtime.
`RunState.candidate_goal_keys` maps each candidate to the GoalKey of the part that
produced it (derived from revision and `tool_end` order, no new stored field); part
rankings, snapshots and join parents use only candidates of the current part key.

#### R4.13.5 Ids — M1a (R2.7 item 7)

```python
# _model/ids.py
class IdGen:
    def __init__(self, run_id: str, epoch: int, counters: Mapping[str, int] | None = None) -> None: ...
    def next(self, prefix: Literal["c", "L", "s", "r", "llm", "x", "k"]) -> str: ...   # "<prefix>-<run tag>-e<epoch>-<n:04d>"
    def counters(self) -> dict[str, int]: ...
def run_tag(run_id: str) -> str: ...                 # sha256(run_id)[:16]
def human_learning_id() -> str: ...                  # "L-h-<16 random hex>"
def new_run_id(now: datetime) -> str: ...
```

#### R4.13.6 Budget — M1b (+M3 LLM prices, +M4 gateway)

```python
USD_LIMIT = 1_000_000_000                            # _budget/amount.py; inclusive signed USD bound

class Amount(FrozenModel):
    tokens: int = 0
    usd: float = 0.0
    # Construction rejects abs(usd) > USD_LIMIT before normalization.
    # Integer micro-dollar arithmetic rejects out-of-range results before float conversion;
    # supported balances and split remainders round-trip exactly. Limits.max_usd <= USD_LIMIT.
    def fits_in(self, other: "Amount", *, cap_tokens: bool, cap_usd: bool) -> bool: ...
    # + __add__, __sub__; an unset Limits cap means "unlimited" for that unit

ReservationKind = Literal["tool", "tranche", "call", "judge", "planner", "curator", "final"]
class Reservation(FrozenModel):
    id: str; holder: str; kind: ReservationKind; amount: Amount; goal_id: str = ""
    keep_charged: bool = False                       # gateway tool runs (M4)

class Ledger:
    @property
    def limits(self) -> Limits: ...                  # current limits, read-only
    # Holder pools [plan]: agent -> its round share; <goal>/:judges, :planner, :curator -> the reserve left after the
    # final slices, then the goal's pool; <goal>/:final -> the goal's final slice, then the reserve left after the final
    # slices, then the goal's pool (converge and the final judge run from the reserve, §6.4, §6.5); ":run" -> the reserve.
    def __init__(self, limits: Limits, *, reserve_fraction: float, goal_weights: Mapping[str, float] | None,
                 journal: JournalWriter, ids: IdGen) -> None: ...
    @classmethod
    def replay(cls, state: RunState, *, journal: JournalWriter, ids: IdGen) -> "Ledger": ...
        # [plan] history only: builds from the payload of the first run_config event in state.ledger_events (the
        # original limits; never from state.config, whose limits already hold every raise), then re-applies
        # state.ledger_events in seq order exactly as live: final slices, goal_started (pool on a continuation),
        # round_started shares, reserve by the live funding rule, settle, goal_finished -> end_goal always,
        # limits_changed -> change_limits; every reservation with no committed settle counts as fully spent
    async def hold_final_slices(self, slices: Mapping[str, Amount]) -> None: ...   # ValueError if the reserve cannot hold them
    def open_goal(self, goal_id: str, *, pool: Amount | None = None) -> None: ...
        # [plan] pool: re-open an ended (continued) goal with exactly that amount left, taken from free(); ValueError if
        # it does not fit
    def free(self) -> Amount: ...                    # [plan] work pool no goal holds (returns and raises that found no goal
                                                     # not ended, minus re-opens)
    def change_limits(self, limits: Limits) -> None: ...   # [plan] a raised cap: added work pool split by goal_weights over
                                                           # goals not ended (none -> free), added reserve to the reserve
    def open_round(self, goal_id: str, round: int, holders: Sequence[str], rounds_left: int) -> dict[str, Amount]: ...
        # Return the exact retained grants; the caller journals this mapping as round_started.shares.
        # Replay restores those grants exactly. Never journal remaining(holder): its read-time view clips
        # the grant left by the current global run balance, which a later refund or cap raise can increase.
    def remaining(self, holder: str) -> Amount: ...
    def goal_remaining(self, goal_id: str) -> Estimate: ...
        # Spendable goal work, excluding the final slice and shared service reserve; uncapped units are None.
        # This is History.budget_left for replanning future rounds.
    async def reserve(self, holder: str, amount: Amount, *, kind: ReservationKind, goal_id: str,
                      keep_charged: bool = False,
                      on_admit: Callable[[Reservation], None] | None = None) -> Reservation: ...
        # LimitReached("budget"); durable before return. on_admit runs synchronously after booking and before
        # the durability wait, allowing cancellation cleanup to capture ownership. It does not authorize dispatch.
        # A raising callback unwinds the booking exactly, writes no event and propagates unchanged; its id is unknown.
    async def settle(self, reservation_id: str, actual: Amount, *, estimated: bool = False) -> None: ...  # idempotent, durable
    async def settle_overflow(self, reservation_id: str, actual: Amount) -> None: ...
        # Charge reported tokens and exactly USD_LIMIT; settle carries usage_overflow=True. Replay restores
        # permanent run/goal exhaustion, even with unlimited caps. Every later reserve raises
        # LimitReached("usage_overflow"); later settlements are no-ops retaining their existing charges.
        # Failed durable appends retain exhaustion; the same overflow reservation may retry its write.
    async def end_goal(self, goal_id: str) -> None: ...   # unused -> goals not ended by goal_weights; none -> free()
    def run_exhausted(self) -> bool: ...             # [plan] charges, incl. open and estimated reservations, reach the work pool
    def goal_exhausted(self, goal_id: str) -> bool: ...   # [plan] the same for the goal pool

@dataclass(frozen=True, kw_only=True)
class TrancheOutcome:
    overflow: bool = False
    # Callers may ignore this additive outcome when relying on durable settle.usage_overflow
    # and ledger.run_exhausted(); LimitReached still propagates through normal admission.

class Tranche:                                       # one per agent and per goal judges / planner / curator holder
    # [plan] reserve_call is serialized; settle_call after close is a no-op
    def __init__(self, ledger: Ledger, holder: str, goal_id: str, *, size: Amount) -> None: ...
    async def reserve_call(self, estimate: Amount) -> str: ...      # reservation id; a new committed tranche when used up;
                                                                     # an own reservation when estimate > size
        # [plan] a new tranche reserves min(size, ledger.remaining(holder)); LimitReached only when that is below the
        # call estimate
    def can_cover(self, estimate: Amount) -> bool: ...   # [plan] M3: True when reserve_call(estimate) would succeed now;
                                                         # no side effects
    async def settle_call(self, reservation_id: str, actual: Amount) -> TrancheOutcome: ...
    def pending(self) -> int: ...                    # [plan] calls reserved and not yet settled (M1b Task 10; round end and
                                                     # barrier wait up to settle_timeout_s while > 0, then close)
    async def close(self) -> TrancheOutcome: ...     # returns the unused part with a committed settle event

PriceSource = Callable[[str], ModelSpec.Prices | None]   # LiteLLM model name → USD per 1M tokens, or None
def register_price_source(source: PriceSource) -> None: ...   # swarm_arena.llm registers LiteLLM's pinned price map at import (M3)
def judge_amount(judge: Judge, n: int, config: RunConfig, price_of: Callable[[str], ModelSpec.Prices | None],
                 *, llm: bool = True) -> Amount: ...
    # [plan] _budget/estimate.py; M1b, +M3 for LLM judges. The first rule that matches: ProgramJudge -> Amount();
    # Lexicographic -> sum of members (same llm); a judge with non-empty models() (an LLM judge) -> llm=False: Amount()
    # (its calls reserve through the scope's tranche); llm=True: the componentwise max of one tranche
    # (llm_tranche_tokens, llm_tranche_usd) and the sum over judge.models() (one alias per panel member, repeats kept)
    # of 2 x n x spec.max_tokens tokens priced at max(2 x input, output) per 1M (2 x n: the two-stage fallback, and one
    # call per candidate bounds any batching; 2 x input: a 1h cache write costs 2 x input) — [plan] M3 structured_call
    # keeps the formatting call within its member's terms: max_tokens = the member call's cap, and a separate formatter
    # alias only when its input and output prices are each <= the member's (else the same alias formats) —; a judge with
    # non-empty providers() -> n x default_run_reservation (settled estimated at the barrier); any other judge -> Amount() (no
    # paid call, R5.4). The barrier's up-front batch reservation passes llm=False; estimates pass llm=True.
def final_slice_estimate(plan: Plan, config: RunConfig, tools: Mapping[str, Tool | Sequence[tuple[Tool, ToolRequest]]],
                         price_of: Callable[[str], ModelSpec.Prices | None]) -> Amount: ...
    # [plan] _budget/estimate.py; M1b, +M3 for LLM terms. (1 + max_join_attempts) x sum over converge-team agents of
    # [max_tool_runs_per_round x (tool estimate or default_run_reservation) + sum over manager.models() of
    # max_turns_per_round x spec.max_tokens priced at max(input, output)] (0 in select mode)
    # + judge_amount(ranking, n_final, llm=True), where n_final = top_n in select mode and
    # converge-team agents x max_tool_runs_per_round + top_n in team mode (the final-judge batch).
    # tools keys are agent addresses. A variant sequence takes the componentwise maximum over every effective
    # ordinary round config/request for that address; converge addresses use their own effective converge config.
    # Requests retain round and round_overrides but use empty inputs because future manager inputs are unknown.
    # A single Tool remains supported with ToolRequest(). All hooks run in an abandonable daemon in the engine.
def worst_case_estimate(plans: Sequence[Plan], config: RunConfig, tools: Mapping[str, Tool | Sequence[tuple[Tool, ToolRequest]]],
                        price_of: Callable[[str], ModelSpec.Prices | None]) -> Amount: ...   # §5 step 2
    # The engine's price_of(alias): the catalog entry's prices first, then the registered source, else None.
```

#### R4.13.7 LLM scope — M1b (used from M3)

```python
@dataclass(frozen=True, kw_only=True)
class LLMScope:
    role: CallRole
    run_id: str; goal_id: str; team: str; agent: str; round: int
    candidate_id: str | None
    tranche: Tranche
    journal: JournalWriter
    artifacts: ArtifactStore
    models: ModelCatalog
    tracing: TracingConfig
    cache: CachePolicy | None
    force_no_cache_read: bool = False                # anchors and the final judge
    call_ids: list[str] = field(default_factory=list)   # [plan] record_call appends the id of every LLMCall made in this scope
def current_llm_scope() -> LLMScope | None: ...
@contextmanager
def llm_scope(scope: LLMScope) -> Iterator[None]: ...
# The core enters a scope around every manager round, judge batch, planner call, and curator call. LLM code
# reads it to build metadata and to reserve; record_call settles by metadata.reservation_id, not by context.
# After a planner call, the core journals scope.call_ids in the plan_revision payload; after a judge batch, it sets
# Score.llm_call_ids to scope.call_ids unless the judge set them.

# Call-id seam [plan], defined in M1b so that M3 changes no M1b signature:
@contextmanager
def requesting_call(call_id: str | None) -> Iterator[None]: ...   # sets a private ContextVar[str | None]
def current_requesting_call() -> str | None: ...
# The M3 action wrapper enters requesting_call(<the LLMCall.id whose response holds the tool call>) inside its own task,
# around ctx.run_tool and ctx.post_learning, so parallel tool calls never mix. execute_tool_run reads it into
# Candidate.requested_by_call, and post_learning reads it into Learning.posted_by_call. Outside it, both stay None.
```

#### R4.13.8 Process supervision — M1a

```python
class RLimits(FrozenModel):
    memory_bytes: int | None = None                  # from Resources.memory_gib
    cpu_seconds: int | None = None
    nofile: int | None = 4096                        # [plan]

class ProcessSupervisor:
    def __init__(self, *, host_id: str) -> None: ...
    async def run(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], stdout: Path, stderr: Path,
                  timeout_s: float | None, grace_s: float, rlimits: RLimits | None,
                  on_started: Callable[[Job.Started], Awaitable[None]] | None,
                  before_kill: Callable[[CompletedRun], Awaitable[None]] | None) -> CompletedRun: ...
    # [plan] before_kill gets a provisional CompletedRun(returncode=None, status=timeout|cancelled), so collect() runs
    # before SIGKILL
    # 1. spawn `sys.executable -I -m swarm_arena._launch` with start_new_session=True and a pass_fds "go" pipe
    #    [plan] -I: neither the workspace cwd nor an allowlisted PYTHONPATH can put a staged swarm_arena/ ahead of the
    #    installed one before the start event commits
    # 2. await on_started(Started(kind="process", host_id, pid, pgid, start_time)); on error: kill the launcher, re-raise
    # 3. write "go"; the launcher execs argv. A BrokenPipeError on the go write is ignored; the reap gives status error
    # 4. on timeout or cancel, in a shielded finally: SIGTERM group → wait until grace_s passes or the group is gone →
    #    before_kill(provisional), bounded by BEFORE_KILL_TIMEOUT_S = 30 s → in a nested finally that always runs:
    #    SIGKILL group, reap, proc.aclose(); only then note an exception or timeout of before_kill in the stderr log
    # status: ok/error by exit code; timeout; cancelled; oom when killed by SIGKILL or SIGSEGV that we did not send
    #         while memory_bytes was set [plan]
    # [plan] workspace files read after a run (swarm/result.json, outputs, logs) go through read_regular_file
    #        (O_NOFOLLOW | O_NONBLOCK, regular files only); log notes go through note_in_log (O_APPEND | O_NOFOLLOW |
    #        O_NONBLOCK, regular files only, otherwise dropped), so no diagnostic can block termination

def host_id() -> str: ...                            # [plan] f"{hostname}:{boot id}" (Linux boot_id, macOS kern.boottime)
def process_start_time(pid: int) -> str | None: ...  # [plan] `ps -o lstart= -p <pid>`
def group_has_live_member(pgid: int) -> bool: ...
def kill_orphan(started: Job.Started) -> bool: ...   # only if host_id and start_time match; else False and nothing is killed
```

#### R4.13.9 Files, paths, staging — M1a

```python
# _tooling/tool.py
def effective_tool_ref(tool: Tool | Ref, overrides: Mapping[str, Any]) -> Ref: ...
    # One core shallow-merge helper for admission, round execution, and both kit paths. Call only after raw
    # non-human authorization. Reject declared human-only fields and their validation aliases; validate through
    # Ref and the tool's Config before snapshots; retain raw types, omitted defaults, and frozen manifests.
    # The effective ref supplies execution settings. The snapshotted base ref supplies tool identity.
# The only tool timeout and grace computation; M1b passes round_spec.tool_timeout_s and limits.tool_timeout_s.
def effective_tool_timeout(tool: Tool, *, round_timeout_s: float | None, limit_s: float) -> float: ...
def tool_grace(tool: Tool) -> float: ...
def cap_timeout(configured: float | None, limit_s: float) -> float: ...  # shared cap, also used by CheckContext.execute

# _tooling/result_file.py
NOTE_PREFIX: Final = "swarm-arena: "
MAX_DECLARED_FILES: Final = 4096
_OUTPUT_BYTE_LIMIT: Final[ContextVar[int | None]]  # set/reset inside daemon collect from executor config
# The read_regular_file of R4.13.8: O_NOFOLLOW | O_NONBLOCK, regular files only, no symlinked component.
# Refuse an fstat size above max_bytes before reading (including sparse files); never read beyond the cap if it grows.
def read_regular_file(workspace: Path, rel: str, *, max_bytes: int) -> bytes: ...
def symlinked_component(base: Path, rel: str) -> str | None: ...  # shared component check for snapshots and collection
# Never blocks or follows: O_APPEND | O_NOFOLLOW | O_NONBLOCK, regular files only,
# otherwise the note is dropped; M4 job steps reuse it.
def note_in_log(path: Path, message: str) -> None: ...

RESERVED: Final = frozenset({"swarm", "parents", "candidate"})
def check_relpath(path: str) -> str: ...             # ValueError: absolute, "..", reserved top level, empty
def check_key(path: str) -> str: ...                 # [plan] normal form for dict-key paths: check_relpath, and the input must already be normalized
def matches(path: str, globs: Sequence[str]) -> bool: ...
def snapshot_fileset(fs: FileSet, base_dir: Path, artifacts: ArtifactStore) -> FileSet: ...   # rejects symlinks
def snapshot_plan(plan: Plan, artifacts: ArtifactStore) -> Plan: ...   # goal, parts, and every FileSet in component configs
def snapshot_ref(ref: Ref, base_dir: Path, artifacts: ArtifactStore) -> Ref: ...   # every FileSet and nested Ref in the
    # validated config; writes back explicit fields and freezes non-empty FileSet defaults under their canonical
    # field names. Only empty FileSet defaults and other defaults stay omitted; unrelated values keep their raw form.
# _tooling/staging.py
def parent_ref(candidate: Candidate) -> ParentRef: ...
def effective_files(goal: Goal, candidate: Candidate) -> dict[str, str]: ...
def validate_edits(edits: Mapping[str, str], *, editable: Sequence[str], protected: AbstractSet[str],
                   max_edit_bytes: int, root: AbstractSet[str] = frozenset()) -> None: ...  # raises SoftError
def edit_hashes(edits: Mapping[str, str]) -> dict[str, str]: ...  # refuse the reserved `sha256:<64 hex>` reference form
def put_edits(edits: Mapping[str, str], artifacts: ArtifactStore) -> None: ...  # refuse the reserved `sha256:<64 hex>` reference form
# callers run `validate_edits` with `protected = tool._protected(root) | private_paths`,
# and existing root paths as `root` (soft-refuse folded edit/root collisions),
# then `plan_tool_stage` with the same protection set; staging rechecks folded edit collisions
def fixed_config(ref: Ref, entry: CatalogEntry | None) -> dict[str, Any]: ...   # config minus settable fields (no overrides)
def tool_identity(ref: Ref, entry: CatalogEntry | None) -> str: ...            # sha256(canonical({path, fixed_config}))
def plan_tool_stage(*, goal_files: Mapping[str, str], tool_files: Mapping[str, str], editable: Sequence[str],
                    protected: AbstractSet[str], private_paths: AbstractSet[str], parents: Sequence[ParentRef],
                    receiving_identity: str, edits: Mapping[str, str], request_json: dict[str, Any] | None
                    ) -> tuple[Job.Stage, dict[str, str], list[str]]: ...
    # returns (stage, effective overlay, soft warnings). Order: goal files, base overlay (overlay rule), edits.
    # private_paths: every check and judge file path of the plan revision (§4.4, §4.12). A root path in it raises
    # RuntimeError, an internal error: validate_plan rejects such plans first (M1b deliverable 6).
def plan_script_stage(*, own_files: Mapping[str, str], candidate: Candidate,
                      effective_files: Mapping[str, str]) -> Job.Stage: ...
def materialize(stage: Job.Stage, workspace: Path, artifacts: ArtifactStore) -> None: ...
# [F4] The original stage is mandatory: reject output conflicts in the producing run.
def collect_output_files(workspace: Path, declared: Sequence[str], artifacts: ArtifactStore,
                         *, stage: Job.Stage, max_bytes: int) -> list[FileRef]: ...
    # max_bytes is the total output byte budget; check each file's size against the remaining budget before reading
    # or uploading it, including sparse files, and bound the read if it grows. Validate all declarations before reads.
```

#### R4.13.10 Agent runtime and the `run_tool` pipeline — M1b

```python
@dataclass
class AgentCounters:                                 # check-and-increment with no await in between (parallel-safe)
    turns: int = 0; tool_runs: int = 0; board_reads: int = 0; learnings: int = 0
    refused_repeats: int = 0; consecutive_failures: int = 0

@dataclass(kw_only=True)
class AgentRuntime:
    address: str
    holder: str                                      # [plan] the ledger holder that the agent's tool runs and tranche charge:
                                                     # the agent address, or "<goal>/:final" for a converge-team agent (§8.1)
    goal_id: str                                     # main goal id for events, candidates, learnings, logs, env and ledger
    goal: Goal; goal_key: GoalKey; part: str | None    # part sub-goal for inputs/files/checks; part GoalKey for repeats
    team: TeamSpec; spec: AgentSpec; round: int; round_spec: RoundSpec; fidelity: str
    tool: Tool; tool_ref: Ref; execution_ref: Ref; tool_identity: str; catalog_entry: CatalogEntry | None
        # tool_ref is the snapshotted base for identity/repeat keys; required private execution_ref is the
        # merged effective ref submitted to the executor, paired with the effective built tool
    checks: Sequence[tuple[str, Check]]
    view: BoardView; own: list[Candidate | Learning]; digest: str; notes: str
    ledger: Ledger; tranche: Tranche; limits: Limits; counters: AgentCounters
    repeats: RepeatIndex; watchdog: Watchdog
    journal: JournalWriter; board: Board; executor: Executor; artifacts: ArtifactStore; ids: IdGen
    config: RunConfig; resumed: tuple[Candidate, ...]
    started_runs: int = 0                            # increment only after durable tool_start returns; distinct from tool_runs
        # Resume restores this goal/team/agent/current-round count from committed tool_start events,
        # including starts of completed and lost candidates (not just currently pending jobs).
    seed_parents: tuple[str, ...] = ()                # ordered assigned ids; manager sees a read-only tuple
        # Runner supplies permitted visibility for assigned candidates; never insert foreign results into own.
    rankings: Mapping[str, str] = field(default_factory=dict)
        # Group key (empty for main, part name otherwise) to ranking judge; the round runner
        # fills the own group from Plan.ranking_name() or the part ranking.
    private_paths: frozenset[str] = frozenset()      # [plan] every check and judge file path of the revision (R4.13.9)
    stops: ProviderStops                             # [plan] M3: the run's one ProviderStops, shared by every goal round
                                                     # (EngineDeps.provider_stops, R4.13.13)

class ProviderStops:                                 # _engine/stops.py — M3 [plan]: one per run, shared by every goal round
    def register(self, address: str, providers: frozenset[str], cancel: Callable[[], None]) -> None: ...
        # an active agent and the providers it uses (manager models' providers | tool.providers()); cancel stops its round
    def unregister(self, address: str) -> None: ...
    async def fail(self, provider: str, *, source: str) -> list[str]: ...   # stops and cancels every registered agent with
                                                                             # that provider, in every goal; journals
                                                                             # provider_failed; returns addresses
    def stopped(self) -> dict[str, frozenset[str]]: ...                    # address → failed providers

async def execute_tool_run(rt: AgentRuntime, inputs: Mapping[str, Any], *, parents: Sequence[str],
                           replicate: int | None, edits: Mapping[str, str] | None,
                           read_ids: Sequence[str] = ()) -> Candidate: ...
    # Snapshot read_ids at action entry and stamp new candidates before journaling/indexing.
    # Repeat-linked and resumed results retain their original journaled read_ids.
    # Prepend seed_parents to explicit parents and stably deduplicate (first occurrence wins).
    # validate inputs, edits (validate_edits), and visible parents (including root/protection)
    # → repeat rules and replicate allocation → plan the stage → count the tool run → reserve (durable) → put_edits
    # → Job(on_started = durable start event) → executor.submit (refused when the round's admission is closed:
    # at Inflight.add and again in on_started; a refused run settles Amount() and raises LimitReached('stop'))
    # → logs → checks → durable completion event → settle (or keep charged) → board.put_candidate
    # The final request (including the allocated replicate) supplies key, stage.request_json, Job, and Candidate.
    # Pure stage planning completes before count/reserve; artifact writes follow reserve.
    # Logs use goals/<goal-id>/logs/<candidate-id>.{out,err}; board publication uses the completion event's position.
    # Cancelled preparation with no committed tool_start settles zero and launches no check commands.
    # Usage above USD_LIMIT retains an ERROR candidate and reported tokens, then settles via settle_overflow
    # after durable tool_end; live and replayed admission become terminal in all limits modes.
    # Replicate allocation hooks have type-only diagnostics; release uses retained allocation data without hooks.
    # AgentContext.run_tool applies the gate and counts the turn, then calls execute_tool_run, which owns every other check.

class Inflight:
    closed: bool
    def add(self, scope: anyio.CancelScope, *, grace_s: float = 0.0) -> None: ...
    def discard(self, scope: anyio.CancelScope) -> None: ...
    def max_grace_s(self) -> float: ...               # maximum active Job.grace_s, or 0.0 when idle
    def close(self) -> None: ...                     # close admission without dropping entries
    def cancel_all(self) -> None: ...                # retain entries until executor.submit exits
    # Drain waits must also preserve the executor-owned collection allowance after grace (P47).

class RepeatOutcome(FrozenModel):
    action: Literal["run", "refuse", "return_own", "link"]
    candidate: Candidate | None = None             # for return_own and link; a past-run hint goes in hint
    hint: Candidate | None = None
class RepeatIndex:
    def key(self, rt: AgentRuntime, request: ToolRequest) -> str: ...      # §6.2 repeat key
    def lookup(self, key: str, *, agent: str, deterministic: bool,
               resumed: AbstractSet[str] = frozenset()) -> RepeatOutcome: ...
    def next_replicate(self, rt: AgentRuntime, request: ToolRequest, *, requested: int | None = None) -> int: ...
        # [plan] requested = run_tool's replicate argument (request.replicate ignored); index i is used when an own
        # counted, non-partial entry has key(rt, request with replicate i) or i is pending
class Watchdog:
    def record_run(self, status: RunStatus, *, passed: bool) -> None: ...
    def record_refusal(self) -> None: ...
    def end_round(self, *, ran: bool) -> str | None: ...   # reason when the agent must stop
```

#### R4.13.11 Scoring runner — M1b (+M3)

```python
async def run_checks(candidate: Candidate, checks: Sequence[tuple[str, Check]], ctx: CheckContext) -> list[CheckResult]: ...
async def judge_round(goal_id: str, judges: Sequence[tuple[Ref, Judge]], candidates: Sequence[Candidate],
                      leaders: Sequence[Candidate], ctx_for: Callable[[Judge], JudgeContext], *,
                      prepared: tuple[list[Candidate], frozenset[str]] | None = None) -> JudgeOutcome: ...
    # stamps ids and hashes; non-finite or failing → judge_error events; anchors = leaders[:judge.anchor_count()]
    # [plan] M3: a judge that raises ProviderFailed gets JudgeOutcome.provider_failed[judge name] = provider, with no
    # judge_error events and no scores; the other judges still run (§6.4: ranking-judge pause, panel drop)
def prepare_judge_batch(judge: Judge, candidates: Sequence[Candidate], leaders: Sequence[Candidate]) -> tuple[list[Candidate], frozenset[str]]: ...
    # Eligible candidates plus unique anchors, prepared once before provider funding. The caller contains
    # hook failures; judge_round(prepared=...) consumes the funded batch without calling anchor_count again.
async def final_judge(goal_id: str, ranking: tuple[Ref, Judge], finalists: Sequence[Candidate],
                      ctx: JudgeContext) -> tuple[list[Candidate], JudgeOutcome]: ...     # one batch; read_cache=False; returns (order, outcome); scores stamped with ctx.ids
    # [plan] M3: re-raises ProviderFailed, so the goal writes NO_RESULT with reason provider_failed (§6.4)
# [plan] Fenced (from a command job's on_started) propagates out of run_checks, judge_round and final_judge; it is never
# a failed CheckResult or a judge_error.
# [plan] M3: judge_round and final_judge re-raise LimitReached (an LLM judge's BudgetExceeded included); it is never a
# judge_error, and the engine handles it.
def ranked(candidates: Iterable[Candidate], scores: Iterable[Score], *, identity: tuple[str, str, str],
           fidelity: str | None, allow_partial: bool) -> list[tuple[Candidate, Score]]: ...   # _scoring/ranking.py
    # one identity per call: each caller ranks each group under that group's own identity
    # (RoundState.ranking_identity[group]; M2 group_rows the same)
```

#### R4.13.12 Planning internals — M1b

```python
class RefSite(FrozenModel):
    pointer: str                                     # JSON pointer in the plan, e.g. "/teams/0/agents/1/tool"
    kind: CatalogKind
    ref: Ref
    carried_human: bool                              # unchanged from a human-origin ancestor
def ref_sites(plan: Plan) -> list[RefSite]: ...      # every Ref at any depth, incl. nested refs and overrides
def diff_revision(parent: Plan | None, child: Plan, *, parent_human: Mapping[str, bool]) -> list[RefSite]: ...
    # agents by stable address; checks, judges, parts by name
def check_catalog(site: RefSite, catalog: PlannerCatalog, *, overrides: Mapping[str, Any],
                  deferred: list[tuple[RefSite, CatalogEntry, tuple[str, ...]]] | None = None) -> list[str]: ...
    # deferred=None is strict: extra fixed fields are errors. A supplied collector records catalog-matched extra
    # fields for the typed default check after all import-free raw errors are rejected, before resolution/snapshotting.
    # Compares catalog_form(site config minus settable keys) with catalog_form(entry.config) using shared json_equal.
    # _planning/catalog_check.find_entry(ref: Ref, kind: CatalogKind, catalog: PlannerCatalog, *, raw: bool = False,
    # defer_defaults: bool = False) -> CatalogEntry | None returns the first same-kind/path canonical fixed-config match.
    # raw=True is import-free and used only by raw authorization (defer_defaults collects extra fixed fields for the
    # typed default check before snapshotting). The default mode also accepts an extra FileSet field equal to its
    # omitted Config default, loading only an already matched human catalog path; the round runner, manifest and
    # staging use it for every admitted base tool ref, including human refs. check_catalog retains same-kind/path
    # candidates and diagnoses fixed-field differences against the first candidate when no fixed config matches.
    # Settable/override values use _validate_config(entry.settable_model(), ..., where=site.pointer); unknown planner
    # keys render as <field>, and every diagnostic starts with the site's pointer followed by ": ".
def catalog_form(config: Mapping[str, Any]) -> dict[str, Any]: ...
    # [plan] _planning/catalog_check.py. No import. Two rules apply at any depth and on both sides. (1) A dumped FileSet
    # {"globs": g, "manifest": m} becomes g. (2) A nested-ref dict becomes {"path": p, "config": catalog_form(config or
    # {}), "name": name or None}. A nested-ref dict is a dict with a `path` in "module:qualname" form and no keys other
    # than path, config, and name (the M1b Task 14 nested-ref rule). So a site that the core snapshotted, a journaled
    # revision, and a later unchanged revision still equal the raw catalog entry. A changed nested path, config value,
    # or name still differs. Shared json_equal preserves bool/int/float distinctions recursively after normalization.
def authorize_revision(plan: Plan, config: RunConfig, *, history: Sequence[Plan] = (),
                       deferred: list[tuple[RefSite, CatalogEntry, tuple[str, ...]]] | None = None
                       ) -> tuple[Plan, list[str]]: ...
    # [plan] _planning/catalog_check.py. No import, no I/O. Human origin -> (plan, []). Non-human: check_catalog on every
    # site with carried_human False. In each such site, a dumped FileSet keeps its manifest only when a revision in
    # `history` (the admitted revisions, as journaled) holds a FileSet with equal globs and an equal manifest, at any
    # site and any depth. Every other dumped FileSet loses its manifest and becomes its globs list. So the core computes
    # the manifest of every new or changed planner-written component, and an unchanged or copied component keeps its
    # frozen files with no disk read, also after a resume. Returns (that plan, errors). The shared
    # _planning/catalog_check.parent_human(history: Sequence[Plan]) -> dict[str, bool] walker grants all sites of human
    # revisions authority and folds later revisions through diff_revision; validation reuses it. Effective-site authority
    # also requires unchanged raw overrides, even when an override edit leaves the merged Ref unchanged. Diagnostic
    # pointers retain configured nested keys and list indices while rendering planner-invented mapping keys as <field>.
    # deferred=None is strict; extra fields are errors. With a collector, reject all import-free raw errors first,
    # then check typed defaults only at matched catalog paths: accept extra fields only for FileSet declarations
    # equal to the Config default. Never trust a planner manifest merely because its declaration equals a default:
    # only admitted history grants manifest trust; otherwise snapshot from the human source, failing closed if absent.
def admit_revision(plan: Plan, config: RunConfig, artifacts: ArtifactStore, *, history: Sequence[Plan] = (),
                   other_active_agents: int = 0) -> tuple[Plan, list[str]]: ...
    # [plan] _planning/validate.py. The one entry point for r0 (RunRunner), replanner and join-failure revisions
    # (GoalRunner), and `swarm-arena plan` / `run` of a plan file (M3). Human: snapshot_plan -> validate_plan.
    # Non-human: authorize_revision with deferred collector (raw errors -> PlanError before any Ref.load) -> typed
    # catalog-default check (before any resolution/snapshotting) -> the revision takes r0's base_dir
    # (history[0]; its own for r0) -> snapshot_plan (part-goal FileSets resolve against the main goal's manifest, never
    # from disk) -> validate_plan. Returns (snapshotted revision, warnings).
def stamp(plan: Plan, *, origin: Origin, parent: Plan | None) -> Plan: ...   # sets origin, revision, parent_revision
ModelCheck = Callable[[Mapping[LLMRole, Sequence[str]], RunConfig], tuple[list[str], list[str]]]   # (errors, warnings)
    # [plan] a role → aliases map, not a Plan: the replanner, the curator, and a planning session have no plan. M1b
    # ships Callable[[Plan, RunConfig], tuple[list[str], list[str]]]; M3 Task 3 changes it to this form.
def register_model_check(check: ModelCheck) -> None: ...                 # swarm_arena.llm registers at import (M3)
def check_models(uses: Mapping[LLMRole, Sequence[str]], config: RunConfig) -> tuple[list[str], list[str]]: ...
    # M3, _planning/validate.py: alias, max_tokens, and price errors, then every registered ModelCheck. The one entry
    # point for validate_plan, _engine/run.py (replanner, curator), and planning_session (R4.13.13).
def agent_fingerprint(spec: AgentSpec, manager: Manager) -> str: ...      # tool ref + tool config + manager models
```

#### R4.13.13 Engine — M1b

```python
@dataclass(kw_only=True)
class EngineDeps:                                    # _engine/round.py — M1b (+M3); fields only
    run_id: str; config: RunConfig; store: Store; journal: JournalWriter; ledger: Ledger; board: Board
    executor: Executor; artifacts: ArtifactStore; ids: IdGen; views: ViewWriter
    stops: StopState; clock: RunClock                # M1b plan types (stop control, run clock)
    admission: RunAdmission | None = None            # one shared service per run; standalone GoalRunner initializes it
    provider_stops: ProviderStops                    # [plan] +M3: the run's one ProviderStops; every AgentRuntime.stops (R4.13.10)
class RunRunner:                                     # _engine/run.py
    async def run(self) -> dict[str, list[Candidate]]: ...
class GoalRunner:                                    # _engine/goal.py
    def __init__(self, deps: EngineDeps, *, goal_id: str, plan: Plan, replanner: Planner | None,
                 history: Sequence[Plan] = (), state: GoalState | None = None,
                 journal_r0: bool = True) -> None: ...
    # Reads the committed RunState on initialization for history, fingerprints and event positions.
    # Recovered repeat pairs after the unfinished round's frontier remain private to their owner;
    # they join repeat_entries at the barrier or an early break, deduplicated by candidate id.
    def active_agents(self) -> int: ...              # max(active main agents, future converge team); select mode counts zero converge agents
    repeat_entries: list[tuple[str, Candidate]]
    async def run(self) -> GoalOutcome: ...          # rounds → converge → final judge → solution view
@dataclass
class GoalOutcome:
    goal_id: str; status: Literal["finished", "stopped", "no_result"]
    finalists: list[Candidate]; reason: str | None
def r0_revision_draft(plan: Plan, base_dir: Path) -> EventDraft: ...
class RunAdmission:                                 # _engine/admission.py
    def __init__(self, max_agents_total: int) -> None: ...
    async def update(self, goal_id: str, peak: int, *,
                     publish: Callable[[], Awaitable[None]] | None = None) -> None: ...
    # One lock and per-goal reserved peaks. Snapshot/admit outside the lock; recheck the
    # sum under the lock, then durably publish and update capacity atomically. No planner
    # or snapshot hook runs under this lock. The scheduler releases finished goals.
    async def release(self, goal_id: str) -> None: ...
class RoundRunner:                                   # _engine/round.py
    async def run(self, round: int) -> RoundOutcome: ...
async def converge(goal: GoalRunner, *, select_only: bool) -> list[Candidate]: ...   # _engine/converge.py
    # Converge outputs receive ordinary judges before the uncached final ranking. An unfunded barrier
    # records reserve_exhausted and selects previously judged survivors or unjoined parts, not its new outputs.
def solution_view(state: RunState, goal_id: str, source: Head) -> bytes: ...         # _engine/solution.py, pure
async def run_preflight(plans: Sequence[Plan], config: RunConfig, executor: Executor) -> None: ...
    # _engine/preflight.py: each distinct (path, normalized built config) once per call. Main/part checks and judges,
    # managers, executor, and effective tool variants across ordinary and converge rounds only; no Policy/Planner hook.
    # Admitted later revisions use this same preflight before dispatch. User construction/hooks use run_sync_daemon;
    # failures aggregate reference paths and exception types only.
def build_manifest(*, run_id: str, config: RunConfig, plans: Sequence[Plan],
                   capacity: Resources | None) -> dict[str, Any]: ...  # _engine/manifest.py; synchronous callers only
async def build_manifest_async(*, run_id: str, config: RunConfig, plans: Sequence[Plan],
                              capacity: Resources | None) -> dict[str, Any]: ...  # supported async entry point
    # Tool identity config validation/serialization uses run_sync_daemon, then bounded host/git work uses to_thread.
    # Async callers must use build_manifest_async; never offload the whole sync builder to an ordinary AnyIO worker.
    # Validated JSON-mode RunManifest dump, with nulls preserved. Hash deterministic UTF-8 config/plan TOML dumps;
    # record GoalKey.files_sha256, base-ref tool identities (catalog settable fields excluded), and bounded git probes.
class GoalStatus(FrozenModel):
    status: str; round: int; rounds_total: int; candidates: int; best: float | None
    # best is the current main group's best ranking Score.value, or None when absent/non-scalar or unjoined.
    # Rank within one newest stamped identity/fidelity, using explicit plan judge metadata or stamped main scores.
    # Never import components to resolve names, combine parts, or substitute a tuple key coordinate for value.
class StatusReport(FrozenModel):
    run_id: str; epoch: int; owner_live: bool; head_seq: int
    stop: Literal["none", "stop", "stop-now"]; goals: dict[str, GoalStatus]; spent: Amount; lost: int
    # spent uses reservation amounts while open/estimated/keep_charged, actual for settled, preserving overflow actuals.
    # stop is committed RunState.stop; lost is len(RunState.lost). One owner read selects the committed chain.
def format_status(report: StatusReport) -> str: ...  # _engine/control.py
async def read_status(run_root: str, run_id: str) -> StatusReport: ...               # _engine/control.py
def stop_key(epoch: int, *, now: bool) -> str: ...   # _engine/control.py: "control/stop-e<epoch>" or "control/stop-now-e<epoch>"
async def request_stop(run_root: str, run_id: str, *, now: bool) -> int: ...         # _engine/control.py; returns the epoch
    # [plan] Stop files are per epoch, because Store has no delete. request_stop reads owner.json, fails with a clear
    # error when no live owner holds the lease, creates stop_key(epoch) (AlreadyExists counts as success), then reads
    # owner.json again and fails if the epoch changed meanwhile. The driver polls only its own epoch's two keys. So after
    # stop and resume, the new owner (epoch + 1) never obeys the old file, and a later stop creates a new key.
async def resume_run(config: RunConfig, run_id: str, *, force: bool, replanner: Planner | None) -> RunRunner: ...
    # +M2: check_storage(config) before the takeover loop
def check_storage(config: RunConfig) -> None: ...    # _engine/preflight.py — M2 [plan]: ConfigError for a local run_root
    # with an s3:// arena_root; then probe_conditional_writes once per distinct s3:// root among run_root, arena_root,
    # artifact_root. Called first in run preflight, resume_run (+M2), HumanReview.import_csv (after its CSV pre-check),
    # and M4 take_run
# _engine/run.py — M3
@asynccontextmanager
async def planning_session(config: RunConfig, goal: Goal, planner: Planner) -> AsyncIterator[LLMScope]: ...
    # [plan] first check_models({"planner": planner.models()}, config); errors → ValueError before anything is written.
    # Then a journaled, budgeted run of its own (lease, journal, bind_run, a tranche for "<goal.id>/:planner"), so an
    # LLMPlanner called outside a Swarm still logs and limits every call (§7.3, §8.1). Yields the planner LLMScope.
def mark_planning_session(plan: Plan, scope: LLMScope) -> Plan: ...
    # a copy with planning_session = scope.run_id and planner_call_ids = tuple(scope.call_ids) (R4.1)
```

#### R4.13.14 Board internals — M1b (`RunBoard`, frontier, digest), M2 (rest)

```python
def build_frontier(board_reads: Sequence[tuple[str, str, Head]], reviews: Sequence[str],
                   curation: Sequence[str]) -> Frontier: ...
def build_digest(view: BoardView, *, goal: GoalKey, round: int, ranking: str, cap_tokens: int, k: int = 5) -> str: ...
    # [M2] multi-run views (ArenaBoard) add a keyword-only run_id: the "new since the last round" count is per owning run;
    # M1b reads the run from RunBoard's single frontier mark
    # [plan] estimate_tokens(the whole returned string: markers, headers, rows, titles, and the counts line) <= cap_tokens.
    # When even the markers and the counts line do not fit, it returns the counts line only, cut to at most cap_tokens
    # UTF-8 bytes at a character boundary.
def estimate_tokens(text: str) -> int: ...
    # [plan] len(text.encode("utf-8")): an upper bound on the token count under every byte-level BPE or byte-fallback
    # tokenizer (each token covers at least one byte). So digest_tokens is a hard cap for every manager model and needs
    # no tokenizer dependency (§6.3). English text uses about a quarter of the cap; to show more titles, raise
    # digest_tokens.
class ArenaCache:                                    # M2; SQLite (WAL) file per namespace
    def __init__(self, path: Path) -> None: ...      # [plan] one connection (check_same_thread=False) guarded by one
        # threading.RLock held across each whole transaction and each query through fetch; downloads happen outside the
        # lock; holds a shared flock on <path>.lock until close, and reindex refuses while any is held
    def ingest(self, run_id: str, run_root: str, segments: Sequence[tuple[str, list[Event]]]) -> None: ...
        # idempotent by (run, epoch, seq); an id already held from another run -> item skipped, warning (R4.0)
    def newest_segment(self, run_id: str) -> str | None: ...
    def apply_human_objects(self, names: Sequence[str], objects: Sequence[HumanObject]) -> None: ...
    def search(self, namespace: str, query: str, *, frontier: Frontier, goal: GoalKey | None,
               tags: Sequence[str], limit: int) -> list[str]: ...
        # FTS5 bm25 over a per-frontier index of the frontier's learnings only, then §6.3 relevance
class RunPointer(StoredModel):  run_id: str; run_root: str
class HumanObject(StoredModel):                      # reviews/ and curation/ objects (one JSONL line each)
    kind: Literal["pin", "hide", "add", "review"]
    learning_id: str | None = None
    learning: Learning | None = None
    score: Score | None = None
    author: str
def human_object_order(name: str) -> tuple[str, str]: ...   # (timestamp in the name, full name)
def human_object_name(now: datetime) -> str: ...            # "<YYYYMMDDTHHMMSS.mmmZ>-<random 8 hex>.jsonl" [plan]
```

#### R4.13.15 Gateway — M4

```python
class ProfileReport(FrozenModel):
    verified: bool; failures: list[str]; batch_write_interval_s: float | None; request_timeout_s: float | None
class VirtualKey(FrozenModel):
    key_id: str; alias: str                          # the secret key value is never stored or journaled
class SpendRow(FrozenModel):
    litellm_call_id: str; model: str; usage: Usage; cost_usd: float; ts: datetime; metadata: dict[str, Any]
class GatewayAdapter(Protocol):
    terminal_accounting: bool                        # LiteLLM adapter: False
    async def preflight(self) -> ProfileReport: ...
    async def create_key(self, *, alias: str, max_budget_usd: float, duration_s: float,
                         metadata: Mapping[str, Any], default_estimated_output_tokens: int) -> tuple[VirtualKey, str]: ...  # (key, secret value)
    async def block_key(self, key: VirtualKey) -> None: ...
    async def closed_at(self, key: VirtualKey) -> datetime | None: ...   # closure time T, confirmed by the API
    async def spend_rows(self, key: VirtualKey) -> list[SpendRow]: ...
    async def delete_key(self, key: VirtualKey) -> None: ...
async def reconcile_run(config: RunConfig, run_id: str, adapter: GatewayAdapter, executor: Executor) -> ReconcileReport: ...
```

#### R4.13.16 Ray internals — M4

```python
def job_payload(job: Job) -> bytes: ...              # Job without on_started; Ref, not instances; holds secret_values,
                                                     # so the payload is never logged, journaled, or put in runtime_env
@ray.remote(max_retries=0, max_calls=1)
def run_job(payload: bytes) -> bytes: ...            # materialize, command(), Popen in the worker's group, poll, collect; → ToolOutput JSON
def find_task(task_name: str) -> TaskInfo | None: ...    # ray.util.state
def worker_dead(task_name: str) -> bool | None: ...
def submit_driver(address: str | None, entrypoint: str, *, runtime_env: Mapping[str, Any] | None) -> str: ...
    # address None → Ray reads RAY_ADDRESS, as `swarm-arena submit` does without --address
```

#### R4.13.17 LLM internals — M3

```python
def build_model(spec: ModelSpec) -> "agents.models.interface.Model": ...   # LitellmModel, or a registered FakeModel for "fake/…"
def model_settings(spec: ModelSpec, *, scope: LLMScope, reservation_id: str, read: bool, write: bool,
                   max_tokens: int | None = None) -> "agents.ModelSettings": ...
    # include_usage=True; metadata={run_id, goal_id, team, agent, round, role, reservation_id}, every value a str
    # ("round": str(scope.round)), because ModelSettings.metadata is dict[str, str];
    # extra_args: Claude injection points (anthropic/bedrock/vertex only) and the cache dict; [plan] when temperature is
    # set, extra_args also holds drop_params=True
class CallLogger("litellm.integrations.custom_logger.CustomLogger"):        # only maps kwargs/response_obj → record_call(settle=True) (R4.13.19)
    async def async_log_success_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None: ...
    async def async_log_failure_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None: ...
def install(config: RunConfig) -> None: ...          # idempotent: logger, cost map env, response cache
```

#### R4.13.18 CLI — M1a (root) and later

```python
def main(argv: Sequence[str] | None = None) -> int: ...   # _cli/main.py; argparse [plan]; exit codes 0 ok, 1 error, 2 usage
# Subcommand modules expose: def register(sub: argparse._SubParsersAction) -> None
# [plan] _cli imports swarm_arena.llm and swarm_arena.ray only inside a command handler, never at module level or in
# register(), so `swarm-arena --help` and the core commands work with no extras.
```

#### R4.13.19 LLM call recording — M3 (+M4)

```python
# _journal/llm_record.py. A core module: it imports no optional dependency, so the stdlib gateway client (M4),
# _tooling/llm_calls.py (M4), FakeModel, and llm/_logger.py all record through it.
def bind_run(run_id: str, *, journal: JournalWriter, ids: IdGen, artifacts: ArtifactStore, tracing: TracingConfig,
             committed: Sequence[LLMCall], request_sha256: Mapping[str, str]) -> None: ...
    # the engine calls it at run start (both empty) and on resume. committed: the LLMCalls of committed llm_call events
    # (the litellm_call_id dedupe set); request_sha256: call id → request sha256 (RunState.llm_request_sha256), so resume
    # can point a cache hit to its original call when payloads are not stored. Settlement goes through the scope's
    # tranche (pending_scope), so bind_run takes no Ledger.
def unbind_run(run_id: str) -> None: ...             # [plan] M3: at run end (RunRunner finally) and planning-session end
def open_call(scope: LLMScope, *, reservation_id: str, estimate: Amount, model_alias: str, model: str,
              provider: str) -> str: ...             # [plan] M3: registers a pending call; returns the pre-allocated LLMCall id
def pending_scope(metadata: Mapping[str, Any]) -> LLMScope | None: ...
    # the scope of the pending call named by metadata["run_id"] and metadata["reservation_id"]; None when there is none
async def record_call(*, metadata: Mapping[str, Any], model: str, provider: str, usage: Usage, cost_usd: float,
                      cache_hit: bool, latency_ms: float | None, status: Literal["ok", "error"], error: str | None,
                      request: JsonValue, response: JsonValue, litellm_call_id: str | None,
                      tool_call_ids: Sequence[tuple[str, str]], settle: bool,
                      charge_reservation: bool = False) -> LLMCall | None: ...
    # finds the run by metadata["run_id"] (not by context) and reads int(metadata["round"]); journals one llm_call event
    # with payload {"call", "request_sha256"} (the sha256 also when payloads are not stored); stores payloads when
    # tracing.store_payloads; appends the id to the current LLMScope.call_ids, if any. settle=True (the LiteLLM logger,
    # FakeModel) settles metadata["reservation_id"]. settle=False for gateway spend rows and for swarm/llm_calls.jsonl
    # rows of unkeyed tool runs: they never settle, so the full reservation stays charged (§8.1). The reported rows of a
    # keyed run are never recorded; they stay in ToolOutput.llm_calls_file for information (spec 8.1). Idempotent by
    # litellm_call_id: a known id writes nothing and returns None.
    # [plan] M3 claim: after payload storage and before journaling, record_call claims the call id with no await between
    # the check and the set. Only the claimer (record_call or drain) journals, appends to scope.call_ids, and settles; a
    # call id already claimed writes nothing and returns None.
    # [plan] M3: after the claim, the journal append and scope.call_ids append are synchronous, and the settle runs
    # shielded in a try/finally that always marks the pending call done; an exception after the claim is stored on the
    # pending call and re-raised.
    # [plan] M3 charge_reservation=True: the call is charged its pending reservation estimate and gets an "estimated:"
    # error. CallLogger provenance rule: a failed call without authoritative usage costs 0 only when swarm-arena's own
    # code refused it before reserve_call returned (nothing reserved, no open_call, no request leaves the process, no
    # LLMCall). Every failure LiteLLM reports, whatever its class, settles the usage its exception carries
    # (failure_usage); else CallLogger sets charge_reservation, so the full token and USD reservation stays charged,
    # because a LiteLLM exception class does not prove that no request was sent or billed (the SageMaker mapper raises
    # ContextWindowExceededError for an endpoint capacity failure; LiteLLM maps output blocks to
    # ContentPolicyViolationError with a synthetic 400). A failed call is over-counted by at most its reservation;
    # unknown-spend failures never release their reservation.
async def drain(scope: LLMScope, *, wait: bool) -> list[str]: ...   # [plan] M3: ids of the placeholder calls written
    # for this scope's pending calls: wait (when wait) until each is claimed or settle_timeout_s passes; each one
    # still open is claimed by drain and gets an "estimated" placeholder LLMCall charged its full estimate
    # a claimed call is awaited until done (always set); drain then settles a journaled but unsettled one at its recorded
    # amount, and after all placeholders re-raises the first recorder error unchanged
def abandon_call(scope: LLMScope, reservation_id: str) -> None: ...  # [plan] M3: marks the pending call of that reservation
    # abandoned; drain does not wait for an abandoned call
```

### R4.14 Journal event types — M1b (additions noted)

At startup, `run_config` precedes `run_started` so ledger replay initializes from the original limits. Both, all final slices, and every goal's r0/`goal_queued` are committed before any goal starts.

`EventType` [plan]: `run_started` (payload: the worst-case estimate and its warnings, §5 step 2, `manifest`, and `final_slices`, goal → `Amount`), `run_config`, `limits_changed` (`Ledger.replay` applies it at its position), `takeover`, `stop_requested`, `goal_queued`, `goal_started` (payload `pool`: an `Amount` dump, only when resume re-opens a continued goal), `goal_finished` (payload `reason`; `stop`, `stop --now`, `run_timeout_s`, `run budget`, `goal budget`, and `reserve_exhausted` mean stopped, R4.13.4), `plan_revision` (payload: plan, origin, planner call ids from `LLMScope.call_ids`, and `base_dir` for r0; from M3, r0 also holds `planning_session` and that session's `planner_call_ids`, R4.1; `RunRunner` journals every goal's r0 at run start, queued goals included), `plan_rejected` (a revision admission rejection or failed planner proposal; failed calls carry only `planner error: <Type>` and create no revision; payload `{"errors": [...], "origin": ...}`), `frontier`, `round_started`, `agent_status`, `agent_counters` (ordinary; payload `{turns, board_reads}`), `notes_saved`, `reserve`, `settle`, `tool_dispatch` (Ray intent, M4), `tool_start`, `tool_end` (payload: candidate, repeat_key, a2a, fingerprint: str | null; the producing effective tool computes the fingerprint, hook failures record only their type in candidate warnings and null; unusable error/lost outputs use null; null is always distinct), `command_start` (payload `{job_id, started, candidate_id, component}`), `command_end` (payload `{job_id, status}`), `run_lost`, `score`, `judge_error`, `learning`, `decision`, `no_result`, `part_kept`, `join_failed`, `reserve_exhausted`, `provider_failed` (M3), `watchdog`, `round_closed`, `converge_started` (payload `select_only`, optional `reason`: the pending lifecycle termination reason, retained through takeover; absent/null means a natural end), `final_order` (payload `candidate_ids`, `parts`, optional `reason`: the termination reason at committed selection, used by finish-only recovery; absent/null means a natural end), `solution_written`, `llm_call` (M3), `gateway_key` (M4), `reconcile` (M4), `curated` (M2), `review_imported` (M2). Write-through types: `reserve`, `settle`, `tool_dispatch`, `tool_start`, `tool_end`, `command_start`, `frontier`, `round_closed`.

### R4.15 Stored formats and `schema_version`

All start at `schema_version = 1`: journal `Event` and the segment link line, `OwnerDoc`, `RunConfig` (`run.toml`), `Plan` (`plan.rN.toml`), `manifest.json`, `solution.json`, `Candidate`, `Learning`, `LLMCall`, `Frontier`, `Job.Started`, `RunPointer`, `HumanObject`, `swarm/request.json` [plan], `candidate/request.json` and `candidate/output.json` [plan]. User-written `swarm/result.json` and `swarm/llm_calls.jsonl` may omit it (default 1). Older versions load through `_model/migrate.py`; newer versions fail with a clear `ValueError`.

`solution.json` is the canonical JSON serialization of the private
`_engine.solution.SolutionDocument(StoredModel)`, kind `solution`, with a validated
`source_head`. Its JSON-only candidate, score, part, replicate-statistic,
best-of-N, plan-revision, and run-metadata subtrees use `FiniteJsonObject` and
retain nulls. Readers use `SolutionDocument.model_validate_json` for the registered
upgrade path and newer-version refusal. Finalist `plan_revision` is the zero-based
final evaluation revision; `lineage` lists direct parent ids. In solution v1,
each `replicate_stats` row carries `part` (null for the main goal), `judge`,
`rubric_hash`, `panel_hash`, `fidelity`, `tool_identity`, `inputs_sha256`, `n`,
`mean`, `min` and `max`. Statistics use each ranking group's newest exact
`Score.identity` and fidelity, then aggregate by tool identity and canonical
input hash only within that same group, identity and fidelity. A cohort is
included only when at least one member has `replicate > 0`; main-goal and part
cohorts are never combined. `manifest.json` similarly uses a private
`RunManifest` StoredModel, kind `manifest`, with validated finite JSON subtrees
and nullable capacity/commit fields. Readers use its migration-aware loading path.
These artifact models are distinct from
`_model.data.Manifest`, the workspace-path-to-SHA-256 file-map alias.

## R5. Global constraints

### R5.1 From the spec (verbatim values)

- **Python:** 3.11 or newer, **POSIX** only (no Windows). `requires-python = ">=3.11"`.
- **Async:** the anyio API on the **asyncio backend only** (§4.8). Tests pin `anyio_backend = "asyncio"`. `Swarm` raises on another backend. The CLI uses `anyio.run(..., backend="asyncio")`.
- **Core dependencies:** `anyio>=4.15,<5`, `pydantic>=2`, `typing-extensions`. [plan] lower bounds: `pydantic>=2.11,<3`, `typing-extensions>=4.12`. `fsspec` (§7, §11) is not a direct core dependency; it comes with the `s3` extra through `s3fs` (deviation, R5.4 item 13).
- **Extras:** `llm`: `openai-agents[litellm]>=0.24,<0.25` and `litellm>=1.105.1,<1.106` (with `!=` pins for bad releases; an upper bound is raised only after the weekly upstream job passes on the newer release). `cache`: `litellm[caching,proxy]`. `ray`: `ray[default]>=2.59,<3`. `s3`: `s3fs>=<floor>`, `aiobotocore>=<floor>` (floors from M2 task 1). `langfuse`: `langfuse>=5,<6`, `openinference-instrumentation-openai-agents`. `a2a`: v2 (not declared in v1). The gateway client uses only the stdlib (`urllib`).
- **Docs:** sphinx, sphinx-rtd-theme, numpydoc, myst-parser. [plan] `sphinx>=9.0,<10` (the 3.11 floor locks 9.0.x; RTD builds on 3.13), `sphinx-rtd-theme>=3.1`, `numpydoc>=1.11`, `myst-parser>=4`.
- **Tooling:** uv (`uv init --lib`, src layout, `py.typed`, build backend `uv_build>=0.13.1,<0.14`), ruff with numpy docstring rules, ty pinned exactly, pytest 9 with anyio's plugin and `anyio_mode = "auto"`, `-W error::DeprecationWarning`, an `__all__` snapshot test per public module.
- **The core never imports** `agents`, `litellm`, `ray`, `s3fs`, `langfuse`, or `a2a`. A tool program is never imported by the core or the driver, except a `FunctionTool` function (R5.4 item 12). Managers, judges, planners, and the board run in the driver; only tool runs and `ctx.execute` go to the executor.
- **Never use `uv run`** to start a tool (it cannot forward SIGKILL). Call the venv's Python directly.
- **Secrets** are never written anywhere by the core (journals, arena objects, logs, LLM payloads, Ray `runtime_env`). `run.toml` holds secret names only.
- **Storage:** every root is a local path or `s3://…`. No conditional writes → refuse to run. Only the chain reached back from `owner.json`'s head counts. Segments, pointers, and review and curation objects are write-once. Views hold only committed content.
- **`schema_version`** on every stored format. Older versions load through migrations; newer versions are refused with a clear error (§9 rule 2).
- **Versioning:** SemVer; in 0.x a breaking change bumps the minor and is listed in `CHANGELOG.md`; one minor release of `DeprecationWarning` (`typing_extensions.deprecated`) before a removal; 1.0 after both connectors ran real campaigns.
- **ABCs grow only by optional methods with defaults.** `__init__` of a component does no I/O; side effects go in `preflight()`.
- **Clocks and constants:** `sync_interval_s` 60, `lease_ttl_s` 60 (renew every `lease_ttl_s / 3`), `settle_timeout_s` 60, group commit window about 200 ms, `reserve_fraction` 0.15, Ray poll `p.wait(timeout=0.2)`, Ray object-store file cap 256 MiB per run, response-cache `ttl_s` 2592000.
- **Ray:** `max_retries=0`, `max_calls=1`, no `setsid`, no `runtime_env.py_executable`, no Ray Client for long runs, `ray.init` through `anyio.to_thread.run_sync`.
- **LiteLLM:** `LITELLM_LOCAL_MODEL_COST_MAP=True`; `Cache(mode="default_off")`; judges use temperature 0 and a dict JSON schema; anchors and the final judge always send `no-cache: true`.

### R5.2 Limit defaults (§8.1)

| Limit | Default |
|---|---|
| `max_teams` (per goal; parts count) | 4 |
| `max_agents_per_team` | 4 |
| `max_agents_per_goal` | 12 |
| `max_concurrent_goals` (run) | 4 |
| `max_agents_total` (run, at the same moment) | 32 |
| `max_agents_per_fingerprint` | 2 |
| `max_rounds` | 3 |
| `max_turns_per_round` (per agent) | 20 |
| `max_tool_runs_per_round` (per agent; replicates count) | 3 |
| `max_tokens` (run) | required when any LLM component is used |
| `max_usd` (run) | optional; every used model needs a known price |
| `default_run_reservation` | [plan] 100,000 tokens / 1.00 USD |
| `llm_tranche_usd` / `llm_tranche_tokens` | 0.25 USD / 25,000 tokens |
| `tool_timeout_s` | 3600 |
| `round_timeout_s` | 4 × `tool_timeout_s` |
| `run_timeout_s` | 24 h |
| `max_learnings_per_agent_round` | 3 |
| `max_board_reads_per_round` (per agent) | 10 |
| `digest_tokens` | 1,500 |
| `max_edit_bytes` | 256 KiB |
| `max_consecutive_failures` | 3 |
| `max_refused_repeats` | 3 |
| `watchdog_idle_rounds` | 2 |
| `max_notes_bytes` | [plan] 16 KiB |

Other fixed values: `Learning.title` ≤ 120 characters, `Learning.text` ≤ 1,500 characters, `ConvergeSpec.top_n` 2, `parents_per_part` 1, `max_join_attempts` 1, review export `--top 20`, leaderboard on round pages top 10.

### R5.3 Project-wide rules (owner, binding)

- **uv** for everything: `uv add`, `uv add --dev`, `uv run`. `pyproject.toml` is the single source of truth. `uv.lock` is committed.
- **ruff** for lint and format, configured in `pyproject.toml` [plan]:
  ```toml
  [tool.ruff]
  line-length = 100
  target-version = "py311"
  extend-exclude = ["docs/superpowers"] # design docs: ruff 0.17 formats code blocks in Markdown
  [tool.ruff.lint]
  select = ["E", "W", "F", "I", "UP", "B", "ASYNC", "D", "RUF", "ANN", "SIM", "PT", "TID"]
  ignore = ["ANN401", "RUF012"]         # ANN401: Any is needed for config dicts and **fields;
                                        # RUF012: ruff cannot see that FrozenModel subclasses are pydantic models
  pydocstyle.convention = "numpy"
  [tool.ruff.lint.per-file-ignores]
  "tests/**" = ["D"]
  "examples/**" = ["D"]
  [tool.ruff.lint.flake8-tidy-imports]
  ban-relative-imports = "all"
  [tool.ruff.lint.flake8-tidy-imports.banned-api]
  "agents".msg = "only swarm_arena.llm and swarm_arena.testing._fake_model may import agents"
  "litellm".msg = "only swarm_arena.llm may import litellm"
  "ray".msg = "only swarm_arena.ray may import ray"
  "s3fs".msg = "only swarm_arena._store.s3 may import s3fs"
  "langfuse".msg = "only swarm_arena.llm may import langfuse"
  ```
  with per-file `TID251` ignores for exactly those modules (and their tests).
- **ty** for type checking, pinned exactly (`ty==0.0.89` [plan], the version in the research notes), `[tool.ty.environment] python-version = "3.11"`, `[tool.ty.terminal] error-on-warning = true`. `ty check` covers `src/`, `tests/`, and `examples/`, so it runs only where every extra is installed: the common gate (R2.0) and the `extras` CI job, never the `core` job. [verify] in M1a Task 1 that `ty check` resolves the `_fakes.*` imports of `tests/` (ty's default first-party roots include a `./tests` with no `__init__.py`). If not, add `extra-paths = ["tests"]` under `[tool.ty.environment]`.
- **Every function and method is fully type-annotated**, tests included (`ANN` rules on).
- **pytest** [plan]:
  ```toml
  [tool.pytest.ini_options]
  anyio_mode = "auto"
  testpaths = ["tests", "src"]
  pythonpath = ["tests"]                # tests import tests/_fakes as `_fakes.<module>` (refs use that path)
  addopts = ["-ra", "--strict-markers", "--import-mode=importlib", "--doctest-modules",
             "-m", "not ray and not s3 and not llm_live and not gateway_live"]
  doctest_optionflags = ["NORMALIZE_WHITESPACE", "ELLIPSIS"]
  filterwarnings = ["error::DeprecationWarning"]
  markers = ["ray", "s3", "llm_live", "gateway_live", "slow"]
  ```
  TDD: a failing test first, then the code. Every public function and method has at least one test. Tests mirror `src/` (R3).
- **Docstrings:** numpydoc style **with an `Examples` section** on every public object (every name in a public `__all__` and its public methods). `tests/test_docstrings.py` enforces it with `numpydoc.validate` over the public surface [plan]: all numpydoc checks except `SA01` (See Also) and `ES01` (extended summary); `EX01` (Examples) stays on. ruff `D` rules cover private modules. Examples run as doctests; an example that needs I/O or a long run uses `# doctest: +SKIP` and is covered by a test instead.
- **Docs:** Sphinx with `sphinx-rtd-theme` in `docs/`, pages in Markdown (myst) [plan]. `docs/superpowers/` holds specs and plans and is excluded from the build (`exclude_patterns`). README and docs are updated in the same PR as each feature.
- **Simplicity:** no new dependency beyond R5.1 without editing this roadmap. No speculative abstractions. Files small and focused.
- **Commits:** one commit per completed TDD task, imperative subject line, body says what and why. **No AI or Claude co-author line and no "generated with" text** in commits or PR descriptions.

### R5.4 Decisions this roadmap makes (not in the spec)

1. Public modules are pure facades; all code is in private modules (R0). Errors visible to users are `ValueError` or `RuntimeError` subclasses, so §9's lists stay exact (R4.0).
2. `Store` is a sync ABC with conditional `create` and `replace` only and no unconditional put; async callers use `anyio.to_thread` (R4.13.2). The `Board` ABC is sync too.
3. The core journals every item, then calls `Board.put_*(item, *, epoch, seq)` with the event's position; `put_*` only indexes, and `ArenaBoard` needs no journal handle (R4.6). **Deviation from §4.9 and §7.2** ("`ArenaBoard.put_*` appends to the run's own journal"): the same events go to the same journal, but the core appends them, not the board. Snapshot membership follows R4.6: `RunBoard` filters by position and is rebuilt from `RunState` on resume; `ArenaBoard` indexes its own run from committed chain segments only.
4. New fields: `Candidate.part`, `Candidate.warnings`, `Candidate.tool_identity`, `Score.value`, `ToolRequest.round`, `ToolOutput.failed_provider`, `Job.secret_values`, `LLMCall.litellm_call_id`, `Limits.max_notes_bytes`, `OwnerDoc.version`, and in M4 `ToolOutput.llm_calls_file` and `Job.drop_env` (R4.2, R4.7). New read-only properties (M3): `Plan.planning_session`, `Plan.planner_call_ids` (R4.1). New parameters: `Swarm.run(..., approve=...)`, `make_candidate(..., tool=...)`, `CheckContext(..., on_started=...)` and `JudgeContext(..., on_started=...)` (R4.3), and in M3 `check_judge(..., config=...)` (R4.12). Changed type: `RunConfig.cache` is `CachePolicy | None` (§4.2: `CachePolicy`), `None` meaning no response cache. New optional methods: `Tool.get_input_model`, `Tool.get_description`, `Judge.rubric_hash`/`panel_hash`/`anchor_count`/`providers`/`models`, `Manager.models`, `Planner.models`, `Planner.origin`, `Board.open`/`close`/`scores`, `Executor.confirm_dead`/`kill_orphan`.
5. Built-in LLM accounting reaches LLM code through an `LLMScope` context variable (R4.13.7), so `Planner.plan` and `Curator.curate` keep the spec's signatures.
6. `FakeModel` is an Agents SDK `Model` that feeds the same core `record_call` (`_journal/llm_record.py`, R4.13.19) as the LiteLLM logger, the gateway read-back, and `llm_calls.jsonl` collection (R4.12).
7. JSON-schema subset compiler instead of a `jsonschema` dependency; a deterministic TOML writer instead of `tomli-w` (R4.13.0).
8. `validate_plan(plan, config, *, history, other_active_agents) -> list[str]` (warnings). The core admits every revision through `admit_revision` (R4.13.12): a non-human revision passes the non-importing catalog check before any import or snapshot, and snapshots happen before `validate_plan`.
9. `Swarm.run` returns `dict[goal_id, list[Candidate]]` in final-judge order (R4.1).
10. CI splits `core` (no extras) and `extras` jobs; ruff `TID251` bans optional imports outside their modules (R2.0, R5.3).
11. **Deviation from the §7.1 layout:** stop files are per epoch, `control/stop-e<epoch>` and `control/stop-now-e<epoch>`, not `control/stop` and `control/stop-now`, because `Store` has no delete and a resumed owner must not obey an old stop (R4.13.13). The run folder also gets `review/`, the end-of-run review export (§5 step 8, M2).
12. **Exception to §4.3** ("the tool program is never imported by the core or the driver"): the driver imports a `FunctionTool` function to build its input model from the signature (R4.2). `FunctionTool` is for trusted, importable human code; `ScriptTool` scripts and connector tool programs are never imported.
13. **Deviation from §7 and §11** ("the core reaches storage through `fsspec`"): the core reaches storage through the `Store` ABC. `LocalStore` uses `os` calls (`O_EXCL`, `flock`, `os.replace`), and `S3Store` uses `s3fs`, an fsspec implementation, so `fsspec` is not a direct core dependency (R5.1). Another backend (for example GCS) is a new `Store` subclass with conditional writes.
14. **Stops on resume [plan]:** a goal stopped by `stop` or `stop --now` with rounds left continues on resume; a goal stopped by a limit (`run_timeout_s`, run or goal budget, `reserve_exhausted`) continues only when the resume changes the limits (`limits_changed`). Every goal end returns its unused pool (spec 4.10), live and in `Ledger.replay`. Resume re-opens each continued goal with a journaled `goal_started` `{pool}`: its `goal_weights` share of the free work pool, before that resume's `limits_changed`. It runs a new lifecycle, and its `solution.json` is replaced from the later head.
15. **Part-goal files of non-human revisions [plan]:** they resolve against the main goal's snapshotted manifest, never from disk (R4.1, R4.13.12 `admit_revision`).
16. **Judge reservations [plan]:** `Amount()` for `ProgramJudge`, for LLM judges (non-empty `models()`; their calls reserve through the `<goal>/:judges` or `<goal>/:final` tranche), and for judges without `providers()`; `n × default_run_reservation`, settled `estimated`, for judges with `providers()`. `judge_amount` (R4.13.6) holds the rule, and its `llm=True` form prices LLM judges for estimates and final slices.
17. **Deviation from §4.9** ("the run's current `rubric_hash` and `panel_hash`"): with no score of its own yet, a run's `best`/`leaderboard` judge identity is the newest identity for that judge name in the namespace, because `ArenaBoard` sees identities only through scores (M2).
18. **Deviation from §7.3 item 2a step 1 (gateway keying) [plan, M4]:** keying uses only the allowlist that the driver can read before dispatch (a `ScriptTool` `Config.env`). A `Command.env` allowlist that `Tool.command()` builds on the executor does not count, and with a gateway such a tool never gets `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. A tool that calls an LLM declares `providers()`. Reason: `command()` runs on the executor with the workspace.

19. **Deviation from §4.2 (A2A parent links):** `Task` has no `reference_task_ids` field in A2A v1.0.1. A tool-run task records its parent candidate ids on one request `Message.reference_task_ids` in `Task.history`. Tool-run and agent-round records take an explicit aware `at` timestamp from the output or event; their JSON uses proto camelCase field names and UTC millisecond timestamps.
