# swarm-arena M1b Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the run engine on top of M1a: the run journal and lease, the budget ledger, agents and their actions, policies, `validate_plan`, an in-memory board, `Swarm.run`/`resume`/`stop`, the `run`/`status`/`stop`/`resume` CLI, and the two offline examples.

**Architecture:** All durable state is one run journal: write-once segments on a `Store`, chained back from the head in `control/owner.json`, which is also the run lease. A `JournalWriter` task commits one segment at a time; the `Ledger` commits every reservation before the money is used; `RunState` is a pure fold of committed events and drives resume, status, and every view file. `Swarm` runs goals concurrently (`RunRunner` → `GoalRunner` → `RoundRunner`); managers act only through `AgentContext`, whose `run_tool` pipeline reserves, stages, submits to the M1a `Executor`, checks, and journals each candidate.

**Tech Stack:** Python 3.11+, anyio 4.15 on asyncio, pydantic 2, stdlib (`argparse`, `fcntl`, `contextvars`), pytest 9 with the anyio plugin, uv, ruff, ty 0.0.89, Sphinx + sphinx-rtd-theme + numpydoc + myst-parser.

**Spec:** `docs/superpowers/specs/2026-10-16-swarm-arena-design.md` (APPROVED r8). Binding roadmap: `docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md` (module map R3, interface registry R4, constraints R5). `§n` is a spec section, `Rn` a roadmap section. Read both before Task 1.

## Global Constraints

- Branch: `git switch main && git pull && git switch -c m1b-engine` after M1a is merged. Merge with a PR at the end (Task 34).
- Python `>=3.11`, POSIX only. anyio API on the asyncio backend only; tests use the `anyio_backend` fixture fixed to `"asyncio"` (M1a `tests/conftest.py`).
- No new dependency. Core deps stay `anyio>=4.15,<5`, `pydantic>=2.11,<3`, `typing-extensions>=4.12`. The core never imports `agents`, `litellm`, `ray`, `s3fs`, `langfuse`, `a2a`.
- M1b Must NOT add: S3, SQLite or FTS5, arena pointers, review or curation objects, `ArenaBoard`, `Curator`, any LLM code, Ray, the gateway (R2 M1b).
- Module paths, names, and signatures are exactly those of roadmap R3 and R4. A needed change edits the roadmap in the same commit and says why in the commit body (R0). This plan makes these edits (a task skips an edit that the roadmap already has): Task 3 adds `expect_epoch` to the R4.13.3 `Lease.take_over` and the consecutive-seq, link, and segment-name rules to the `read_chain` comment; Task 7 adds event types `plan_rejected` and `agent_counters` and the `goal_started.pool` payload to R4.14, and to R4.13.4 `RunState.ledger_events` (with `run_config` and `limits_changed`), the `config.limits` fold of `limits_changed`, and the new lifecycle of a `goal_started` after a `goal_finished`; Task 9 writes the R4.13.6 `Ledger.replay(state, *, journal, ids)` (history-only re-execution), `open_goal(..., *, pool=None)`, `free()`, `change_limits()`, the `end_goal` free-pool rule, and the holder-pool and exhaustion comments; Task 10 writes the R4.13.6 `Tranche` comment and the `final_slice_estimate` `n_final` judge term; Task 18 types `RepeatIndex.lookup(resumed=)` as a set of candidate ids and adds `next_replicate(..., *, requested=None)` to R4.13.10; Task 17 changes the R4.13.11 `final_judge` return to `tuple[list[Candidate], JudgeOutcome]` (the order, plus the stamped scores and judge errors the goal journals with round 0); Task 18 adds the field `AgentRuntime.holder: str` to R4.13.10 (the ledger holder that the agent's tool runs and LLM calls charge: its address, or `"<goal>/:final"` for a converge-team agent); Task 19 rewrites the R4.13.10 `execute_tool_run` comment to its check order (validate inputs, edits, and parents → repeat rules and replicate allocation → count the tool run → reserve → put_edits → ... → durable completion event → settle → `board.put_candidate`; the submit is refused at `Inflight.add` and again in `on_started`; `AgentContext.run_tool` applies only the gate and the turn count before it), adds `AgentRuntime.private_paths` to R4.13.10, and adds R3.2 rows for `tests/_fakes/engine.py` and `tests/_agent/conftest.py`; Task 20 adds the value `"stop"` to the R4.4 `LimitReached.limit` comment; Task 22 adds the R3.2 row for `tests/_engine/conftest.py`; Task 24 records in R5.4 the part-file resolution of non-human revisions and the judge reservation rule; Task 26 adds `run_started.final_slices` to R4.14 and that every goal's r0 is journaled at run start; Task 27 replaces R5.4 item 14 with the stopped-goal continuation rule ([plan]).
- M1a already provides the partial-output seam (`Job.on_partial`, R4.7 as edited by M1a) and `effective_tool_timeout(tool, *, round_timeout_s, limit_s)` and `tool_grace(tool)` in `_tooling/tool.py`. This plan uses them and changes no `_exec` module.
- Public facades hold only a docstring, imports, `__all__`, and one `_publish(__name__, __all__)` call (M1a `_facade.publish`; R0). No facade sets `__module__` by hand. Every name added to a public `__all__` is added to `tests/test_public_api.py` in the same task.
- Every public object and public method has a numpydoc docstring with an `Examples` section (`tests/test_docstrings.py`, M1a). Examples that need I/O use `# doctest: +SKIP` and are covered by a test.
- Every function and method is fully type-annotated, tests included. Tests mirror `src/` (R3 rule: `src/swarm_arena/_a/b.py` → `tests/_a/test_b.py`), no `__init__.py` in test dirs.
- Values pinned by spec and roadmap (R5.1, R5.2): `lease_ttl_s` 60, renew every `lease_ttl_s / 3`; `sync_interval_s` 60; `settle_timeout_s` 60; group-commit window 0.2 s; `reserve_fraction` 0.15; `llm_tranche_usd` 0.25, `llm_tranche_tokens` 25,000; `default_run_reservation` 100,000 tokens / 1.00 USD; `max_tool_runs_per_round` 3; `max_turns_per_round` 20; `max_rounds` 3; `max_teams` 4; `max_agents_per_team` 4; `max_agents_per_goal` 12; `max_concurrent_goals` 4; `max_agents_total` 32; `max_agents_per_fingerprint` 2; `tool_timeout_s` 3600; `round_timeout_s` 4 × `tool_timeout_s`; `run_timeout_s` 86,400; `max_learnings_per_agent_round` 3; `max_board_reads_per_round` 10; `digest_tokens` 1,500; `max_edit_bytes` 262,144; `max_notes_bytes` 16,384; `max_consecutive_failures` 3; `max_refused_repeats` 3; `watchdog_idle_rounds` 2; `ConvergeSpec.top_n` 2, `parents_per_part` 1, `max_join_attempts` 1; `Learning.title` ≤ 120 chars, `text` ≤ 1,500 chars.
- Ids: `<prefix>-<run tag>-e<epoch>-<n:04d>` from `IdGen` (R4.0). Epochs start at 1; `seq` starts at 1 and continues across epochs. Run id `YYYY-MM-DDTHH-MM-SSZ_<6 hex>`.
- Addresses: agent `"<goal>/<team>/<agent>"`, team `"<goal>/<team>"`; `Decision.prune` keys `"<team>"` or `"<team>/<agent>"`. Ledger holders: agent address, `"<goal>/:judges"`, `"<goal>/:planner"`, `"<goal>/:curator"`, `"<goal>/:final"`, `":run"` (R4.0).
- Run folder keys (relative to `open_store(run_root).child(run_id)`): `run.toml`, `manifest.json`, `control/owner.json`, `control/stop-e<epoch>`, `control/stop-now-e<epoch>`, `journal/e<epoch>-<last-seq>.jsonl`, `goals/<goal-id>/plan.r<N>.toml`, `goals/<goal-id>/logs/<id>.out` and `.err`, `goals/<goal-id>/solution.json` (§7.1, R5.4 item 11).
- Write-through event types: `reserve`, `settle`, `tool_dispatch`, `tool_start`, `tool_end`, `command_start`, `frontier`, `round_closed` (R4.14). Every event has `context_id = run_id` and a `task_id` (Task 7).
- User-visible errors: plan and config errors `ValueError` (`PlanError`, `ConfigError`); soft action errors `ValueError` (`SoftError`); storage errors `RuntimeError` (`StorageError` family); `Fenced`, `LeaseHeld` are `RuntimeError`; `LimitReached` is public (R4.0, R4.13.1).
- Verification after every implementation step: `uv run pytest <task test files> -v`, then `uv run ruff check . && uv run ruff format --check . && uv run ty check` (expected: all pass, ruff prints `All checks passed!`, ty prints `All checks passed!`).
- Commits: one per task, imperative subject, body says what and why. No AI or Claude co-author line and no "generated with" text.
- `CHANGELOG.md` gets one `Unreleased` → `Added` line per task that adds a user-facing name or command.
- Docs pages are the M1a stubs (R2 M1a deliverable 2): `docs/getting-started.md`, `docs/concepts.md`, `docs/own-files.md` ("Use your own files and script"), `docs/planning-modes.md`, `docs/limits-and-safety.md`, `docs/scoring-and-review.md`.

## Review Focus

1. **Parallel actions race a limit.** Two `ctx.run_tool` calls started together when one tool run is left must give exactly one candidate and one `LimitReached("max_tool_runs_per_round")`; the same for the last learning slot and the last budget share. Test: Task 20 `test_parallel_run_tool_respects_last_slot`, `test_parallel_post_learning_respects_rate_limit`; Task 9 `test_parallel_reserve_never_overdraws`.
2. **A connector raises.** A `Tool.command()`/`collect()` bug or an executor exception must become an `error` candidate with its reservation settled and the round going on, never a crashed swarm (§6.5 "A tool failure ... never crashes the swarm"). Test: Task 19 `test_executor_exception_becomes_error_candidate`.
3. **A manager raises.** An exception other than `LimitReached` from `Manager.run_round` must end only that agent's round, journal the error, and leave the other agents and the goal running. Test: Task 22 `test_manager_exception_ends_only_that_agent`.
4. **Nothing passes.** When every candidate of a goal is gated out (or the ranking judge errors on all), the goal must write `NO_RESULT` with a reason and a `solution.json` with `"status": "no_result"`, and `Swarm.run` must return `[]` for it, not raise. Whatever ended its rounds (all rounds run, stop, budget, run timeout), a goal whose converge and final judge give no finalist also writes `NO_RESULT`, never an `ok` solution with no finalists. Test: Task 24 `test_all_gated_out_writes_no_result`, `test_all_judge_errors_writes_no_result`; Task 26 `test_run_returns_empty_list_for_no_result_goal`.
5. **Names that break addresses or paths.** A team, agent, or part name with `/`, `:`, upper case, or spaces, or two teams (main or converge), two agents in one team, or two parts with the same name, would collide in addresses, ledger holders, and `goals/<goal-id>/...` keys; `validate_plan` must reject them. Test: Task 16 `test_bad_or_duplicate_names_rejected`.

---

## File Structure (this milestone)

Created (R3.1; the `tests/_fakes/` and `conftest.py` files are R3.2 rows): `_journal/{__init__,event,segment,lease,writer,reader,views,replay,a2a}.py`; `_budget/{__init__,amount,split,ledger,tranche,estimate,llm_scope}.py`; `_model/{planning_types,board_types}.py`; `_scoring/{ranking,runner}.py`; `_board/{__init__,base,run_board,frontier,digest}.py`; `_planning/{__init__,planner,provenance,catalog_check,validate,policy,policies}.py`; `_agent/{__init__,manager,runtime,context,tool_run,repeats,watchdog,scripted}.py`; `_engine/{__init__,swarm,run,goal,round,stops,parts,converge,solution,preflight,manifest,control,resume}.py`; `_cli/run_cmds.py`; `testing/_harness.py`; `tests/_fakes/planners.py`; `tests/_fakes/engine.py`; `tests/_fakes/import_canary.py`; `tests/_agent/conftest.py`; `tests/_engine/conftest.py`; `examples/rosenbrock/*`; `examples/own_script/*`; six `tests/e2e/` files.
Modified: `_errors.py`, `_store/base.py`, `_store/local.py`, `_cli/main.py`, `testing/_conformance.py`, facades `__init__.py`, `manager.py`, `planning.py`, `board.py`, `testing/__init__.py`, `tests/conftest.py`, `tests/_fakes/tools.py`, `tests/test_public_api.py`, `README.md`, `CHANGELOG.md`, `.gitignore`, five docs pages, the roadmap (the edits above).

Test files follow R3 (`tests/<path>/test_<name>.py`). Shared fixtures added to `tests/conftest.py` in Task 1: `store(tmp_path: Path) -> LocalStore` (root `tmp_path / "run"`) and `run_config(tmp_path: Path, artifact_root: str) -> RunConfig` (`run_root` and `arena_root` under `tmp_path`, `artifact_root` from the M1a fixture, `lease_ttl_s=60`). The M1a fixtures `artifact_root`, `artifacts`, and `local_executor` are reused, never redefined. Every test uses the name `run_config` for the run config fixture. Test-side helper types (`RtEnv`, `CtxEnv`, `EngineEnv`) and config helpers live in `tests/_fakes/engine.py` (importable as `_fakes.engine`, never collected), so fixtures and tests can annotate with them. Each fake tool, check, or function is added to `tests/_fakes/tools.py` by the first task that uses it, and that task lists the file in its Files block and its `git add`.

---

### Task 1: Conditional replace on `LocalStore` and the storage error set

**Files:**
- Modify: `src/swarm_arena/_errors.py`, `src/swarm_arena/_store/local.py`, `src/swarm_arena/_store/base.py`, `tests/conftest.py` (fixtures `store` and `run_config`, File Structure)
- Test: `tests/_store/test_local.py`, `tests/_store/test_base.py`, `tests/test_errors.py`

**Interfaces:**
- Consumes: M1a `Store`, `Blob`, `LocalStore.create/read/exists/list/child`, `StorageError`, `NotFound`, `AlreadyExists`, `PreconditionFailed`, `WriteOutcomeUnknown`, `ConditionalWritesUnsupported` (R4.13.1, R4.13.2).
- Produces: `LocalStore.replace(key: str, data: bytes, *, if_match: str) -> str` (new etag = sha256 hex of `data`); `open_store(url: str) -> Store`; `ChainError(StorageError)`, `Fenced(RuntimeError)`, `LeaseHeld(RuntimeError)` in `_errors.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_store/test_local.py
def test_replace_with_matching_etag(store: LocalStore) -> None:
    etag = store.create("control/owner.json", b"v1")
    new = store.replace("control/owner.json", b"v2", if_match=etag)
    assert new == sha256_hex(b"v2")
    assert store.read("control/owner.json") == Blob(b"v2", new)

def test_replace_with_stale_etag_fails(store: LocalStore) -> None:
    etag = store.create("k", b"v1")
    store.replace("k", b"v2", if_match=etag)
    with pytest.raises(PreconditionFailed):
        store.replace("k", b"v3", if_match=etag)
    assert store.read("k").data == b"v2"

def test_replace_missing_key_is_not_found(store: LocalStore) -> None:
    with pytest.raises(NotFound):
        store.replace("nope", b"x", if_match="0" * 64)

def test_concurrent_replace_has_one_winner(store: LocalStore) -> None:
    etag = store.create("k", b"v0")
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda i: _try_replace(store, etag, i), range(8)))
    assert results.count("ok") == 1 and results.count("lost") == 7

def test_replace_leaves_no_temp_files(store: LocalStore, tmp_path: Path) -> None:
    etag = store.create("k", b"v0")
    store.replace("k", b"v1", if_match=etag)
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["k", "k.lock"]

def test_create_takes_no_lock_and_list_skips_lock_files(store: LocalStore, tmp_path: Path) -> None:
    store.create("runs/a.json", b"{}")
    assert sorted(p.name for p in (tmp_path / "run" / "runs").iterdir()) == ["a.json"]
    etag = store.create("runs/b.json", b"{}")
    store.replace("runs/b.json", b"{ }", if_match=etag)
    assert store.list("runs/") == ["runs/a.json", "runs/b.json"]          # runs/b.json.lock exists but is not listed

def test_probe_conditional_writes_passes(store: LocalStore) -> None:
    store.probe_conditional_writes()
```

```python
# tests/_store/test_base.py
def test_open_store_local_path(tmp_path: Path) -> None:
    assert isinstance(open_store(str(tmp_path / "r")), LocalStore)

@pytest.mark.parametrize("url", ["gs://b/x", "http://h/x", "ftp://h/x"])
def test_open_store_refuses_scheme_without_conditional_writes(url: str) -> None:
    with pytest.raises(ConditionalWritesUnsupported, match="conditional writes"):
        open_store(url)

def test_open_store_s3_needs_extra() -> None:
    with pytest.raises(ConditionalWritesUnsupported, match=r"swarm-arena\[s3\]"):
        open_store("s3://bucket/runs")
```

```python
# tests/test_errors.py
def test_m1b_errors_are_runtime_errors() -> None:
    assert issubclass(ChainError, StorageError) and issubclass(StorageError, RuntimeError)
    assert issubclass(Fenced, RuntimeError) and issubclass(LeaseHeld, RuntimeError)
```

`_try_replace` (module-level helper in the test) returns `"ok"` or `"lost"` on `PreconditionFailed`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_store/test_local.py tests/_store/test_base.py tests/test_errors.py -v`
Expected: FAIL with `NotImplementedError` (M1a `replace` stub) and `ImportError` for `ChainError`.

- [ ] **Step 3: Implement**

`LocalStore.replace`: `fcntl.flock(LOCK_EX)` on `<root>/<key>.lock` (created on first use, kept), read the current bytes (`NotFound` if missing), compare `sha256_hex` with `if_match` (`PreconditionFailed` on mismatch), write `<key>.tmp-<token_hex(6)>`, `os.fsync`, `os.replace`, unlock. Only `replace` takes the lock; `create` stays the M1a `O_EXCL` temp plus `os.link`, which is atomic without a lock (a create of an existing key fails, a replace of a missing key fails), so write-once objects (journal segments, artifacts, M2 `runs/` pointers) never get a `.lock` file. M1a `LocalStore.list` already skips `.tmp-` and `.lock` names; the new test pins it. `open_store`: a path with no scheme or `file://` → `LocalStore`; `s3://` → `ConditionalWritesUnsupported("s3:// needs the swarm-arena[s3] extra (M2)")` until M2; any other scheme → `ConditionalWritesUnsupported("<scheme>:// has no conditional writes; swarm-arena refuses to run on it")`. Add the three error classes with numpy docstrings.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_store tests/test_errors.py -v`, then `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_errors.py src/swarm_arena/_store tests/_store tests/test_errors.py tests/conftest.py
git commit -m "feat(store): add conditional replace to LocalStore and refuse stores without conditional writes"
```

### Task 2: Journal events and segments

**Files:**
- Create: `src/swarm_arena/_journal/__init__.py` (empty), `src/swarm_arena/_journal/event.py`, `src/swarm_arena/_journal/segment.py`
- Test: `tests/_journal/test_event.py`, `tests/_journal/test_segment.py`

**Interfaces:**
- Consumes: `StoredModel`, `FrozenModel`, `canonical_json`, `utc_now`, `upgrade` (M1a).
- Produces: `EventType(StrEnum)` with exactly the R4.14 values, `COMMAND_START` (write-through) and `COMMAND_END` included (plus `plan_rejected`, added in Task 7); `WRITE_THROUGH: frozenset[EventType]`; `EventDraft`, `Event` (R4.13.3 fields); `segment_name(epoch: int, last_seq: int) -> str`; `encode_segment(link: str | None, events: Sequence[Event]) -> bytes`; `decode_segment(data: bytes) -> tuple[str | None, list[Event]]`; `parse_segment_name(name: str) -> tuple[int, int]` (private helper).

- [ ] **Step 1: Write the failing tests**

```python
# tests/_journal/test_event.py
def test_write_through_set_is_exact() -> None:
    assert {t.value for t in WRITE_THROUGH} == {
        "reserve", "settle", "tool_dispatch", "tool_start", "tool_end", "command_start", "frontier", "round_closed"}
    assert EventType.COMMAND_END not in WRITE_THROUGH

def test_event_round_trips_with_schema_version() -> None:
    ev = Event(epoch=1, seq=7, ts=datetime(2026, 10, 18, 13, 0, tzinfo=UTC), run_id="r",
               context_id="r", task_id="c-1", goal_id="g", type=EventType.TOOL_END, payload={"a": 1})
    raw = json.loads(canonical_json(ev.model_dump(mode="json")))
    assert raw["schema_version"] == 1 and raw["ts"] == "2026-10-18T13:00:00.000Z"
    assert Event.model_validate(raw) == ev

def test_newer_event_schema_refused() -> None:
    with pytest.raises(ValueError, match="newer"):
        Event.model_validate({**_raw_event(), "schema_version": 2})
```

```python
# tests/_journal/test_segment.py
def test_segment_name() -> None:
    assert segment_name(1, 100) == "journal/e1-100.jsonl"
    assert parse_segment_name("journal/e2-150.jsonl") == (2, 150)

def test_encode_first_line_is_link() -> None:
    data = encode_segment("journal/e1-100.jsonl", [_ev(101), _ev(102)])
    lines = data.decode().splitlines()
    assert json.loads(lines[0]) == {"schema_version": 1, "link": "journal/e1-100.jsonl"}
    assert [json.loads(x)["seq"] for x in lines[1:]] == [101, 102]
    assert data.endswith(b"\n")

def test_decode_round_trip_and_first_segment_link_none() -> None:
    assert decode_segment(encode_segment(None, [_ev(1)])) == (None, [_ev(1)])

def test_encoding_is_deterministic() -> None:
    assert encode_segment(None, [_ev(1)]) == encode_segment(None, [_ev(1)])

def test_decode_rejects_unsorted_or_empty() -> None:
    with pytest.raises(ValueError):
        decode_segment(encode_segment(None, []))
    with pytest.raises(ValueError):
        decode_segment(encode_segment(None, [_ev(2), _ev(1)]))

def test_decode_rejects_gap_inside_segment() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        decode_segment(encode_segment(None, [_ev(1), _ev(3)]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_journal/test_event.py tests/_journal/test_segment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._journal'`.

- [ ] **Step 3: Implement** `EventType`, `WRITE_THROUGH`, `EventDraft`, `Event` in `_journal/event.py` (register kind `"event"` in `_model/migrate.CURRENT`), and the segment functions in `_journal/segment.py`. Each line is `canonical_json(...)` plus `b"\n"`; the link line is `{"schema_version": 1, "link": ...}`. A segment holds at least one event, with consecutive `seq` values (each is the previous + 1); `decode_segment` raises `ValueError("seqs not consecutive")` otherwise.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_journal -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_journal tests/_journal src/swarm_arena/_model/migrate.py
git commit -m "feat(journal): add event model, event types, and segment encoding with the link line"
```

### Task 3: Lease and owner document

**Files:**
- Create: `src/swarm_arena/_journal/lease.py`
- Modify: roadmap R4.13.3 (`take_over(..., expect_epoch)`; the `read_chain` consecutive-seq, link, and segment-name rules that Task 4 implements; skipped when present)
- Test: `tests/_journal/test_lease.py`

**Interfaces:**
- Consumes: `Store`, `PreconditionFailed`, `AlreadyExists`, `Fenced`, `LeaseHeld`, `utc_now` (looked up as a module attribute so tests can monkeypatch `swarm_arena._journal.lease.utc_now`).
- Produces (R4.13.3): `Head`, `OwnerDoc`, `Lease.acquire_new`, `Lease.take_over(cls, store, *, owner_id, host, ttl_s, force, expect_epoch: int | None = None)`, `Lease.advance`, `Lease.renew`, `Lease.release`, `read_owner(store) -> tuple[OwnerDoc, str] | None`; `Lease.ttl_s: float`; `OWNER_KEY = "control/owner.json"`; `Lease.expired(doc: OwnerDoc) -> bool` (static helper).

- [ ] **Step 1: Write the failing tests**

```python
def test_acquire_new_creates_epoch_1(store: LocalStore) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    doc, etag = read_owner(store)
    assert (doc.epoch, doc.version, doc.head, doc.owner_id) == (1, 1, None, "o1")
    assert etag == lease.etag

def test_second_acquire_fails(store: LocalStore) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    with pytest.raises(AlreadyExists):
        Lease.acquire_new(store, owner_id="o2", host="h", ttl_s=60)

def test_take_over_live_lease_needs_force(store: LocalStore) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    with pytest.raises(LeaseHeld):
        Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=False)

def test_force_take_over_bumps_epoch_keeps_head_and_fences(store: LocalStore) -> None:
    old = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    old.advance(Head(epoch=1, seq=100, segment="journal/e1-100.jsonl"))
    new = Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
    assert new.doc.epoch == 2 and new.doc.head == Head(epoch=1, seq=100, segment="journal/e1-100.jsonl")
    with pytest.raises(Fenced):
        old.advance(Head(epoch=1, seq=105, segment="journal/e1-105.jsonl"))
    with pytest.raises(Fenced):
        old.renew()

def test_take_over_after_expiry_without_force(store: LocalStore, monkeypatch: pytest.MonkeyPatch) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    monkeypatch.setattr(lease_mod, "utc_now", lambda: _T0 + timedelta(seconds=61))
    assert Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=False).doc.epoch == 2

def test_release_keeps_epoch_and_head(store: LocalStore) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    lease.advance(Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))
    lease.release()
    doc, _ = read_owner(store)
    assert doc.epoch == 1 and doc.head.seq == 3 and doc.expires_at < utc_now()
    assert Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=False).doc.epoch == 2

def test_renew_and_advance_bump_version_and_expiry(store: LocalStore) -> None:
    lease = Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    first = lease.doc.expires_at
    lease.renew()
    assert lease.doc.version == 2 and lease.doc.expires_at >= first

def test_concurrent_take_over_has_one_winner(store: LocalStore) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60).release()
    # two threads call take_over(force=False); one returns, the other raises Fenced or LeaseHeld

def test_take_over_with_stale_expected_epoch_is_fenced(store: LocalStore) -> None:
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)          # epoch 2
    with pytest.raises(Fenced):
        Lease.take_over(store, owner_id="o3", host="h", ttl_s=60, force=True, expect_epoch=1)
    doc, _ = read_owner(store)
    assert (doc.owner_id, doc.epoch) == ("o2", 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_journal/test_lease.py -v`
Expected: FAIL with `ModuleNotFoundError: ... _journal.lease`.

- [ ] **Step 3: Implement** in `_journal/lease.py`. `OwnerDoc` is a `StoredModel` (kind `"owner"`). Every write is `store.replace(OWNER_KEY, ..., if_match=self.etag)` with `version + 1` and `expires_at = utc_now() + ttl_s`; `PreconditionFailed` → `Fenced`. `take_over` reads once; when `expect_epoch` is set and the read document's epoch differs, it raises `Fenced("another owner took the run (epoch <n>)")` before any write; it raises `LeaseHeld` when `expires_at > utc_now()` and not `force`, then one replace with `epoch + 1`, the new owner, the kept head; `PreconditionFailed` on that replace → `Fenced("another owner changed the lease")` (the caller retries from the read). `release` sets `expires_at = utc_now() - 1 s`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_journal/test_lease.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_journal/lease.py tests/_journal/test_lease.py src/swarm_arena/_model/migrate.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(journal): add the run lease with acquire, takeover, renewal, release, and fencing" \
  -m "take_over takes expect_epoch so that a resume fails cleanly when another takeover lands between its authority read and its own takeover. The read_chain comment gains the segment-name rule: each segment's last seq must equal the seq in its name, so every chain step goes strictly back in seq (R4.13.3 take_over and read_chain comments; skipped when the roadmap has them)."
```

### Task 4: Chain reader

**Files:**
- Create: `src/swarm_arena/_journal/reader.py`
- Test: `tests/_journal/test_reader.py`

**Interfaces:**
- Consumes: `Head`, `decode_segment`, `parse_segment_name`, `ChainError`, `Store.read`.
- Produces: `read_chain(store: Store, head: Head, *, stop_after: str | None = None) -> list[tuple[str, list[Event]]]` (oldest first); `read_committed(store: Store) -> tuple[OwnerDoc, list[Event]] | None` (private helper: one authority read, then the whole chain).

- [ ] **Step 1: Write the failing tests**

```python
def test_reads_chain_oldest_first(store: LocalStore) -> None:
    _put(store, None, [1, 2]); _put(store, "journal/e1-2.jsonl", [3])
    chain = read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))
    assert [name for name, _ in chain] == ["journal/e1-2.jsonl", "journal/e1-3.jsonl"]
    assert [e.seq for _, evs in chain for e in evs] == [1, 2, 3]

def test_stop_after_is_exclusive(store: LocalStore) -> None:
    _put(store, None, [1]); _put(store, "journal/e1-1.jsonl", [2])
    chain = read_chain(store, Head(epoch=1, seq=2, segment="journal/e1-2.jsonl"), stop_after="journal/e1-1.jsonl")
    assert [n for n, _ in chain] == ["journal/e1-2.jsonl"]

def test_unlinked_segment_is_never_read(store: LocalStore) -> None:
    _put(store, None, [1, 2])                           # e1-2
    _put(store, "journal/e1-2.jsonl", [3, 4, 5])        # e1-5: fenced owner's garbage
    _put(store, "journal/e1-2.jsonl", [3], epoch=2)     # e2-3: the real successor
    chain = read_chain(store, Head(epoch=2, seq=3, segment="journal/e2-3.jsonl"))
    assert [n for n, _ in chain] == ["journal/e1-2.jsonl", "journal/e2-3.jsonl"]

def test_seq_gap_rejected(store: LocalStore) -> None:
    _put(store, None, [1]); _put(store, "journal/e1-1.jsonl", [3])
    with pytest.raises(ChainError, match="gap"):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))

def test_head_seq_mismatch_rejected(store: LocalStore) -> None:
    _put(store, None, [1, 2])
    with pytest.raises(ChainError, match="head"):
        read_chain(store, Head(epoch=1, seq=5, segment="journal/e1-2.jsonl"))

def test_first_segment_must_start_at_seq_1(store: LocalStore) -> None:
    _put(store, None, [4])
    with pytest.raises(ChainError):
        read_chain(store, Head(epoch=1, seq=4, segment="journal/e1-4.jsonl"))

def test_gap_inside_segment_rejected(store: LocalStore) -> None:
    _put(store, None, [1, 3])
    with pytest.raises(ChainError):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))

def test_link_must_name_previous_seq(store: LocalStore) -> None:
    _put(store, None, [1, 2]); _put(store, "journal/e1-5.jsonl", [3])
    with pytest.raises(ChainError, match="link"):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))

def test_link_cycle_rejected(store: LocalStore) -> None:
    _put(store, "journal/e1-4.jsonl", [1, 2]); _put(store, "journal/e1-2.jsonl", [3, 4])
    with pytest.raises(ChainError):                             # the link check ends the walk: no hang
        read_chain(store, Head(epoch=1, seq=4, segment="journal/e1-4.jsonl"))

def test_stop_after_must_be_previous_seq(store: LocalStore) -> None:
    _put(store, None, [1]); _put(store, "journal/e1-1.jsonl", [3])
    with pytest.raises(ChainError):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"), stop_after="journal/e1-1.jsonl")

def test_segment_body_must_match_its_name(store: LocalStore) -> None:
    _put(store, None, [1], name="journal/e1-2.jsonl"); _put(store, "journal/e1-2.jsonl", [3])
    with pytest.raises(ChainError, match="name"):
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-3.jsonl"))

def test_self_link_rejected(store: LocalStore) -> None:
    _put(store, "journal/e1-2.jsonl", [3], name="journal/e1-2.jsonl")
    with pytest.raises(ChainError):                             # no hang
        read_chain(store, Head(epoch=1, seq=3, segment="journal/e1-2.jsonl"))
```

`_put(store, link, seqs, epoch=1, name=None)` is a test helper that creates `name` (default `segment_name(epoch, seqs[-1])`) with `encode_segment`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_journal/test_reader.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `read_chain`: start at `head.segment`, check `head.seq` equals that segment's last seq, and follow links until `stop_after` (exclusive) or a `None` link. The checks run while walking, before each linked segment is read: the link name's last seq (`parse_segment_name`) must equal the current segment's first seq − 1 (also when the link equals `stop_after`), else `ChainError("seq gap: link '<link>' ends at seq <n>, expected <first - 1>")` (so `test_seq_gap_rejected` and `test_link_must_name_previous_seq` both match); a `None` link needs first seq 1 when `stop_after` is `None`; a `decode_segment` `ValueError` (consecutive seqs inside a segment) becomes `ChainError`. After each segment is decoded (the head included), its last event seq must equal `parse_segment_name(name)[1]`, else `ChainError("segment '<name>' ends at seq <n>, its name says <m>")`; this check comes before the segment's link is followed. With this name check and the link rule, each step goes strictly back in seq, so no link cycle can loop and no visited set is needed. Then reverse. Never list `journal/`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_journal/test_reader.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_journal/reader.py tests/_journal/test_reader.py
git commit -m "feat(journal): read the committed chain back from the head with the chain check" \
  -m "Links and seqs are checked while walking, and each segment's body must end at the seq its name says, so a bad link or a link cycle ends the read with ChainError instead of a hang."
```

### Task 5: `JournalWriter`

**Files:**
- Create: `src/swarm_arena/_journal/writer.py`
- Test: `tests/_journal/test_writer.py` (defines a `FlakyStore(Store)` wrapper in the test module: it can delay `create`, count concurrent `create` calls, and raise `WriteOutcomeUnknown` before or after the inner write; `after_unknown: Callable[[Store], None] | None` is called once with the inner store right after it raises `WriteOutcomeUnknown`; `create_raises: BaseException | None` makes every `create` of a key under `journal/` raise it (`_writer`'s `Lease.acquire_new` still creates `control/owner.json`, so `test_storage_error_until_expiry_fails_waiters` reaches the writer); `prewrite: Mapping[str, bytes]` objects are created in the inner store before the test)

**Interfaces:**
- Consumes: `Lease`, `Head`, `read_owner`, `encode_segment`, `segment_name`, `read_chain`, `Event`, `EventDraft`, `WRITE_THROUGH`, `Fenced`, `AlreadyExists`, `WriteOutcomeUnknown`, `StorageError`, `ChainError`.
- Produces (R4.13.3): `JournalWriter(store, lease, *, run_id, next_seq, local_path, sync_interval_s, group_window_s=0.2)`, `__aenter__`, `__aexit__`, `append`, `append_durable`, `commit`, `head`, `committed`, `fenced: anyio.Event`; `JournalWriter.epoch: int` (from the lease).

- [ ] **Step 1: Write the failing tests**

```python
async def test_append_durable_returns_after_head_covers_it(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        ev = await w.append_durable(EventDraft(type=EventType.RESERVE, payload={"id": "r1"}))
        doc, _ = read_owner(store)
        assert doc.head is not None and doc.head.seq >= ev.seq == 1

async def test_group_commit_shares_one_segment(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        async with anyio.create_task_group() as tg:
            for i in range(5):
                tg.start_soon(w.append_durable, EventDraft(type=EventType.RESERVE, payload={"i": i}))
    assert len(w.committed) == 1         # one group commit; the exit commit is skipped because nothing is queued

async def test_commits_run_one_at_a_time(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_delay_s=0.05)
    async with _writer(flaky, tmp_path, group_window_s=0.0) as w:
        async with anyio.create_task_group() as tg:
            for _ in range(10):
                tg.start_soon(w.append_durable, EventDraft(type=EventType.SETTLE))
    assert flaky.max_concurrent_creates == 1

async def test_every_committed_event_once_in_seq_order(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        for i in range(20):
            w.append(EventDraft(type=EventType.LEARNING, payload={"i": i}))
            if i % 3 == 0:
                await w.commit()
    doc, _ = read_owner(store)
    assert [e.seq for _, evs in read_chain(store, doc.head) for e in evs] == list(range(1, 21))

async def test_unknown_create_reuses_same_segment_by_sha(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_after_create=1)
    async with _writer(flaky, tmp_path) as w:
        await w.append_durable(EventDraft(type=EventType.RESERVE))
    assert w.head.seq == 1

async def test_unknown_advance_that_succeeded_does_not_duplicate(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_after_replace=1)
    async with _writer(flaky, tmp_path) as w:
        await w.append_durable(EventDraft(type=EventType.RESERVE))
        await w.append_durable(EventDraft(type=EventType.SETTLE))
    doc, _ = read_owner(flaky)
    assert [e.seq for _, evs in read_chain(flaky, doc.head) for e in evs] == [1, 2]

async def test_unknown_advance_that_failed_recommits_same_events(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_before_replace=1)
    async with _writer(flaky, tmp_path) as w:
        await w.append_durable(EventDraft(type=EventType.RESERVE))
    doc, _ = read_owner(flaky)
    assert [e.seq for _, evs in read_chain(flaky, doc.head) for e in evs] == [1]

def _take(s: Store) -> None:
    Lease.take_over(s, owner_id="o2", host="h", ttl_s=60, force=True)

async def test_unknown_advance_succeeded_then_takeover_is_fenced(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_after_replace=1, after_unknown=_take)
    async with _writer(flaky, tmp_path) as w:
        with pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert w.fenced.is_set()
        version_after_takeover = read_owner(flaky)[0].version
    doc, _ = read_owner(flaky)
    assert (doc.owner_id, doc.epoch, doc.version) == ("o2", 2, version_after_takeover)   # the old writer wrote nothing more

async def test_unknown_advance_failed_then_takeover_is_fenced(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), unknown_before_replace=1, after_unknown=_take)
    async with _writer(flaky, tmp_path) as w:
        with pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.RESERVE))
    doc, _ = read_owner(flaky)
    assert doc.owner_id == "o2" and doc.head is None

async def test_storage_error_until_expiry_fails_waiters(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), create_raises=StorageError("disk full"))
    async with _writer(flaky, tmp_path, ttl_s=0.3) as w:
        with anyio.fail_after(2):
            with pytest.raises(Fenced):
                await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert w.fenced.is_set()

async def test_chain_error_fails_waiters_at_once(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), prewrite={"journal/e1-1.jsonl": b"junk"})
    async with _writer(flaky, tmp_path) as w:
        with pytest.raises(Fenced) as err:
            await w.append_durable(EventDraft(type=EventType.RESERVE))
        assert isinstance(err.value.__cause__, ChainError)
        with anyio.fail_after(0.1):
            with pytest.raises(Fenced):
                await w.append_durable(EventDraft(type=EventType.SETTLE))
    assert read_owner(flaky)[0].version == 1                                  # no release was written

async def test_takeover_fences_writer(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        await w.append_durable(EventDraft(type=EventType.RESERVE))
        Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
        with pytest.raises(Fenced):
            await w.append_durable(EventDraft(type=EventType.SETTLE))
        assert w.fenced.is_set()
    doc, _ = read_owner(store)
    assert doc.epoch == 2 and doc.head.seq == 1

async def test_lease_renewed_every_third_of_ttl(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path, ttl_s=0.6) as w:
        await anyio.sleep(0.5)
        assert read_owner(store)[0].version >= 2

async def test_fenced_when_renewal_fails_until_expiry(tmp_path: Path) -> None:
    flaky = FlakyStore(LocalStore(tmp_path / "run"), replace_raises=StorageError("down"))
    async with _writer(flaky, tmp_path, ttl_s=0.3) as w:
        with anyio.fail_after(1.0):
            await w.fenced.wait()

async def test_sync_interval_commits_ordinary_events(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path, sync_interval_s=0.1) as w:
        w.append(EventDraft(type=EventType.LEARNING))
        await anyio.sleep(0.4)
        assert read_owner(store)[0].head.seq == 1

async def test_exit_commits_then_releases(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        w.append(EventDraft(type=EventType.LEARNING))
    doc, _ = read_owner(store)
    assert doc.head.seq == 1 and doc.expires_at < utc_now()

async def test_events_also_appended_to_local_file(store: LocalStore, tmp_path: Path) -> None:
    async with _writer(store, tmp_path) as w:
        w.append(EventDraft(type=EventType.LEARNING))
    assert len((tmp_path / "journal.local.jsonl").read_text().splitlines()) == 1
```

`_writer(store, tmp_path, *, ttl_s=60, sync_interval_s=60, group_window_s=0.2)` acquires a lease and builds the writer with `next_seq=1`, `local_path=tmp_path / "journal.local.jsonl"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_journal/test_writer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `JournalWriter` in `_journal/writer.py`. One writer task owns every conditional write (commits and renewals) and runs one commit at a time. `append` stamps `epoch`, `seq`, `ts`, `run_id`, `context_id=run_id`, writes the line to `local_path` (append + flush), and queues it. `append_durable` appends, wakes the writer, and awaits a per-seq `anyio.Event` set when a head advance covers it; it raises `Fenced` if `fenced` is set first. The writer waits `group_window_s` after the first durable wake-up, or `sync_interval_s` for ordinary events. A commit:

```text
events = all queued events with seq > head.seq
name = segment_name(epoch, events[-1].seq); data = encode_segment(head.segment or None, events)
create(name, data): AlreadyExists or WriteOutcomeUnknown → read(name); sha256 equal → continue, else ChainError
lease.advance(Head(epoch, events[-1].seq, name)):
  WriteOutcomeUnknown → read_owner(): doc.epoch != lease.doc.epoch or doc.owner_id != lease.doc.owner_id
                           → Fenced (never adopt that etag);
                         else head == new → adopt etag, done; head == old → keep the re-read etag, retry next loop
  Fenced → set fenced, fail every waiter, stop the task
StorageError other than ChainError (create, read, replace) → retry every 1 s until the lease's own expires_at,
  then set fenced (as for renewals)
ChainError or any non-StorageError exception → keep it as the writer failure; set fenced at once; every waiter and
  every later append_durable/commit raise Fenced("journal writer failed: <Type>: <msg>") with __cause__ = the error;
  stop the task
```

The renewer path runs inside the same task: renew when `lease_ttl_s / 3` passed since the last successful write; if no write succeeded before the lease's own `expires_at`, set `fenced`. An unknown renewal follows the same identity-first rule: re-read, `Fenced` when epoch or owner changed, else adopt the re-read etag. `__aexit__` makes a final commit (skipped when nothing is queued), then `lease.release()`; when fenced or after a writer failure, it skips both. `committed` lists segment names oldest first (chain of this writer, including segments inherited from `lease.doc.head` via `read_chain` at enter).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_journal/test_writer.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_journal/writer.py tests/_journal/test_writer.py
git commit -m "feat(journal): add the single-task journal writer with group commit and unknown-result recovery" \
  -m "An unknown advance checks lease identity before it adopts an etag, so a takeover that lands in between fences the writer; a writer failure fails every waiter instead of leaving them blocked."
```

### Task 6: View files

**Files:**
- Create: `src/swarm_arena/_journal/views.py`
- Test: `tests/_journal/test_views.py`

**Interfaces:**
- Consumes: `JournalWriter.committed`, `Head`, `Store.create/read/replace`, `AlreadyExists`.
- Produces (R4.13.4): `ViewWriter(store, writer)`, `ViewWriter.write(key, content, *, source) -> None`, `embed_source_head(fmt, content, source) -> bytes`, `read_source_head(fmt, content) -> Head | None`.

- [ ] **Step 1: Write the failing tests**

```python
H1 = Head(epoch=1, seq=10, segment="journal/e1-10.jsonl")

def test_embed_formats() -> None:
    assert json.loads(embed_source_head("json", b'{"a": 1}', H1))["source_head"] == {
        "epoch": 1, "seq": 10, "segment": "journal/e1-10.jsonl"}
    assert embed_source_head("toml", b"a = 1\n", H1).splitlines()[0] == \
        b"# swarm-arena source_head: e1/10/journal/e1-10.jsonl"
    assert embed_source_head("md", b"# x\n", H1).splitlines()[0] == \
        b"<!-- swarm-arena source_head: e1/10/journal/e1-10.jsonl -->"
    for fmt, body in [("json", b"{}"), ("toml", b""), ("md", b"")]:
        assert read_source_head(fmt, embed_source_head(fmt, body, H1)) == H1

async def test_same_bytes_twice_is_noop(view_env: ViewEnv) -> None:
    await view_env.views.write("goals/g/solution.json", b"{}", source=view_env.h1)
    await view_env.views.write("goals/g/solution.json", b"{}", source=view_env.h1)

async def test_later_head_replaces_earlier(view_env: ViewEnv) -> None:
    await view_env.views.write("goals/g/solution.json", b'{"v": 1}', source=view_env.h1)
    await view_env.views.write("goals/g/solution.json", b'{"v": 2}', source=view_env.h2)
    assert json.loads(view_env.store.read("goals/g/solution.json").data)["v"] == 2

async def test_earlier_head_never_replaces_later(view_env: ViewEnv) -> None:
    await view_env.views.write("goals/g/solution.json", b'{"v": 2}', source=view_env.h2)
    await view_env.views.write("goals/g/solution.json", b'{"v": 1}', source=view_env.h1)
    assert json.loads(view_env.store.read("goals/g/solution.json").data)["v"] == 2

async def test_uncommitted_source_head_refused(view_env: ViewEnv) -> None:
    with pytest.raises(RuntimeError, match="not committed"):
        await view_env.views.write("goals/g/solution.json", b"{}",
                                   source=Head(epoch=1, seq=99, segment="journal/e1-99.jsonl"))
```

`view_env` fixture: a writer with two commits (`h1`, `h2`, both on `writer.committed`) and a `ViewWriter`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_journal/test_views.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** in `_journal/views.py`. `write` infers the format from the key suffix (`.json`, `.toml`, `.md`), embeds the source head, refuses a `source.segment` not in `writer.committed`, creates; on `AlreadyExists` it reads the object: same sha256 → return; existing head earlier on `writer.committed` → `replace(if_match=etag)`; otherwise return without writing. Store calls go through `anyio.to_thread.run_sync`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_journal/test_views.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_journal/views.py tests/_journal/test_views.py
git commit -m "feat(journal): write view files only from committed heads, never older over newer"
```

### Task 7: A2A records and `RunState` replay

**Files:**
- Create: `src/swarm_arena/_journal/a2a.py`, `src/swarm_arena/_journal/replay.py`
- Modify: `src/swarm_arena/_journal/event.py` (add `PLAN_REJECTED = "plan_rejected"`, and `AGENT_COUNTERS = "agent_counters"` if Task 2 did not get it from R4.14), roadmap R4.14 (add `plan_rejected`: a replanner revision that `validate_plan` rejected; payload `{"errors": [...], "origin": ...}`; add the ordinary type `agent_counters`, payload `{turns, board_reads}`; add the `goal_started` payload `pool`), roadmap R4.13.4 (`RunState.ledger_events` with `run_config` and `limits_changed`, the `config.limits` fold of `limits_changed`, and the continuation lifecycle rule) — each skipped when the roadmap already has it
- Test: `tests/_journal/test_a2a.py`, `tests/_journal/test_replay.py`

**Interfaces:**
- Consumes: `swarm_arena.protocol` types and `task_state_for` (M1a), `Candidate`, `Learning`, `Score`, `Plan`, `RunConfig`, `Limits`, `Job.Started`, `Event`.
- Produces:
  - `tool_run_task(candidate: Candidate, *, context_id: str) -> Task` (task id = candidate id, `reference_task_ids` = `parent_ids`, state from `task_state_for(output.status)`); `agent_round_task(*, context_id: str, address: str, round: int, learnings: Sequence[Learning], status: AgentStatus, summary: str) -> Task` (task id `agent_task_id(address, round)` = `"<address>#r<round>"`, one `Artifact` per learning: a `data` part with the learning JSON plus a text part with its title); `task_to_json(task: Task) -> dict[str, Any]`.
  - `RunState` (R4.13.4) plus `GoalState(FrozenModel)`: `status: Literal["queued", "running", "finished", "stopped", "no_result"]`, `round: int`, `rounds_closed: int`, `agents: dict[str, AgentStatus]` (`"<team>/<agent>"`), `decisions: list[tuple[int, Decision]]`, `frontiers: dict[int, Frontier]`, `end_reason: str | None`, `notes: dict[str, str]`, and the converge state that resume needs (defaults here; later tasks fold their events into them): `repeat_keys: dict[str, str] = {}` (candidate id → repeat key, from `tool_end.repeat_key`, Task 19), `final_order: list[str] = []` and `final_parts: dict[str, str | None] | None = None` (from `final_order`, Task 23), `kept_parts: dict[str, str] = {}` (part → candidate id, from `part_kept`), `join_attempts: int = 0` (from `join_failed`), `select_only: bool = False` (from `reserve_exhausted` and `converge_started.select_only`), `converge_started: bool = False` (Task 24), `counters: dict[str, dict[str, int]] = {}` (`"<team>/<agent>"` → the latest `agent_counters` payload `{turns, board_reads}` of the goal's current round, Task 20). Status rule: `goal_finished.reason` in `STOP_REASONS = {"stop", "stop --now", "run_timeout_s", "run budget", "goal budget", "reserve_exhausted"}` → `"stopped"`; any other reason (`None` included) → `"finished"`; a `no_result` event wins (`"no_result"`). The last `converge_started` and the last `final_order` win; a `round_started` event resets `converge_started`, `select_only`, `final_order`, `final_parts`, and `counters` (a new lifecycle or round). A `goal_started` for a goal that already has a `goal_finished` (a continuation, Task 27) sets `status = "running"` and `end_reason = None`, and resets `converge_started`, `select_only`, `final_order`, `final_parts`, and `counters` (a new lifecycle). So once that `goal_started` is committed, a continued goal is `"running"`: a crash inside its new lifecycle resumes by the running rules, and status shows `running round 2/3`. The `GoalState` docstring says that `end_reason`, `final_order`, and the converge fields describe the latest lifecycle only; an earlier lifecycle's `final_order` and `goal_finished` reason stay in the journal. `RunState.ledger_events: list[Event]` (the committed `run_config`, `run_started`, `goal_started`, `round_started`, `reserve`, `settle`, `goal_finished`, `limits_changed` events in seq order, read by `Ledger.replay`, Task 9); each `limits_changed.new` replaces `RunState.config.limits`, so the state holds the latest journaled limits; `RunState.started_commands: dict[str, Job.Started]` (`command_start.{job_id, started}` minus `command_end.job_id`); `ReservationState(FrozenModel)`: `reservation: dict[str, Any]`, `state: Literal["open", "settled", "estimated"]`, `actual: dict[str, Any] | None`; `RunState.run_elapsed_s: float`; `RunState.epoch: int`; `RunState.lost: set[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_journal/test_a2a.py
def test_tool_run_task_maps_status_and_parents() -> None:
    cand = make_candidate(ToolOutput(status=RunStatus.TIMEOUT, partial=True), parent_ids=["c-p"], id="c-x")
    task = tool_run_task(cand, context_id="run-1")
    assert task.id == "c-x" and task.context_id == "run-1"
    assert task.status.state == TaskState.FAILED and task.metadata["swarm_arena.reason"] == "timeout"
    assert task_to_json(task)["status"]["state"] == "TASK_STATE_FAILED"

def test_agent_round_task_has_one_artifact_per_learning() -> None:
    task = agent_round_task(context_id="r", address="g/t/a", round=2, learnings=[_learning("L-1")],
                            status=AgentStatus.FINISHED, summary="done")
    assert task.id == "g/t/a#r2" and len(task.artifacts) == 1
    assert {p.text is not None for p in task.artifacts[0].parts} == {True, False}
```

```python
# tests/_journal/test_replay.py
def test_origin_comes_from_journal_not_plan_body() -> None:
    plan = _plan().model_copy(update={"origin": "human"})
    state = RunState.from_events([_ev(1, "run_config", {"config": RunConfig().model_dump(mode="json")}),
                                  _ev(2, "plan_revision", {"plan": plan.model_dump(mode="json"), "origin": "llm"}, goal_id="g")])
    assert state.plans["g"][0][1] == "llm"

def test_reservation_without_settle_is_open_and_settled_is_settled() -> None:
    state = RunState.from_events([..., _reserve(3, "r-1"), _reserve(4, "r-2"), _settle(5, "r-2")])
    assert state.reservations["r-1"].state == "open" and state.reservations["r-2"].state == "settled"

def test_started_without_end_is_in_flight() -> None:
    state = RunState.from_events([..., _start(3, "c-1"), _start(4, "c-2"), _end(5, "c-2")])
    assert set(state.started) == {"c-1"} and "c-2" in state.candidates
    assert state.positions["c-2"] == (1, 5)

def test_id_counters_are_max_per_prefix() -> None:
    state = RunState.from_events([..., _end(3, "c-3f9a1c2b5d6e7f80-e1-0007"), _reserve(4, "r-3f9a1c2b5d6e7f80-e1-0002")])
    assert state.id_counters == {"c": 7, "r": 2}

def test_goal_status_from_goal_finished_reason() -> None:
    def status(*evs: Event) -> str:
        return RunState.from_events([*_PRE, _ev(3, "goal_started", {}, goal_id="g"), *evs]).goals["g"].status
    assert status(_ev(4, "goal_finished", {"reason": "stop"}, goal_id="g")) == "stopped"
    assert status(_ev(4, "goal_finished", {"reason": None}, goal_id="g")) == "finished"
    assert status(_ev(4, "no_result", {"reason": "x"}, goal_id="g"),
                  _ev(5, "goal_finished", {"reason": None}, goal_id="g")) == "no_result"

def test_round_started_after_final_order_resets_converge_state() -> None:
    state = RunState.from_events([..., _ev(4, "converge_started", {"select_only": True}, goal_id="g"),
                                  _ev(5, "final_order", {"candidate_ids": ["c-1"], "parts": None}, goal_id="g"),
                                  _ev(6, "round_started", {"round": 2, "run_elapsed_s": 1.0}, goal_id="g")])
    g = state.goals["g"]
    assert (g.converge_started, g.select_only, g.final_order, g.final_parts) == (False, False, [], None)

def test_goal_started_after_goal_finished_reopens() -> None:
    state = RunState.from_events([*_PRE, _ev(3, "goal_started", {}, goal_id="g"),
                                  _ev(4, "converge_started", {"select_only": False}, goal_id="g"),
                                  _ev(5, "final_order", {"candidate_ids": ["c-1"], "parts": None}, goal_id="g"),
                                  _ev(6, "goal_finished", {"reason": "stop"}, goal_id="g"),
                                  _ev(7, "goal_started", {"pool": {"tokens": 0, "usd": 15.5}}, goal_id="g", epoch=2)])
    g = state.goals["g"]
    assert (g.status, g.end_reason, g.converge_started, g.final_order) == ("running", None, False, [])

def test_limits_changed_folds_into_config_and_ledger_events() -> None:
    old, new = Limits(max_usd=100.0).model_dump(mode="json"), Limits(max_usd=200.0).model_dump(mode="json")
    state = RunState.from_events([*_PRE, _ev(3, "limits_changed", {"old": old, "new": new})])
    assert state.config.limits.max_usd == 200.0
    assert [e.type for e in state.ledger_events] == ["run_config", "limits_changed"]

def test_agent_counters_folded() -> None:
    state = RunState.from_events([..., _ev(3, "round_started", {"round": 1, "run_elapsed_s": 0.0}, goal_id="g"),
                                  _ev(4, "agent_counters", {"turns": 1, "board_reads": 0}, goal_id="g", team="t", agent="a"),
                                  _ev(5, "agent_counters", {"turns": 3, "board_reads": 2}, goal_id="g", team="t", agent="a")])
    assert state.goals["g"].counters == {"t/a": {"turns": 3, "board_reads": 2}}

def test_open_command_starts_folded() -> None:
    state = RunState.from_events([..., _cmd_start(3, "x-1"), _cmd_start(4, "x-2"),
                                  _ev(5, "command_end", {"job_id": "x-2", "status": "ok"})])
    assert set(state.started_commands) == {"x-1"}
    assert isinstance(state.started_commands["x-1"], Job.Started)

def test_ledger_events_keep_budget_events_in_seq_order() -> None:
    state = RunState.from_events([..., _reserve(3, "r-1"), _ev(4, "learning", {...}), _settle(5, "r-1")])
    assert [e.seq for e in state.ledger_events if e.type in {"reserve", "settle"}] == [3, 5]
    assert all(e.type != "learning" for e in state.ledger_events)

def test_takeover_resets_stop_and_sets_epoch() -> None:
    state = RunState.from_events([..., _ev(3, "stop_requested", {"now": False}), _ev(4, "takeover", {"force": True}, epoch=2)])
    assert state.stop == "none" and state.epoch == 2

def test_run_elapsed_is_max_recorded() -> None:
    state = RunState.from_events([..., _ev(3, "round_closed", {"run_elapsed_s": 12.5}, goal_id="g")])
    assert state.run_elapsed_s == 12.5

def test_from_events_is_pure() -> None:
    events = _full_run_events()
    assert RunState.from_events(events) == RunState.from_events(list(events))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_journal/test_a2a.py tests/_journal/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `_journal/a2a.py` and `_journal/replay.py`. `RunState.from_events` folds in `seq` order and reads only payload keys written by Tasks 9 to 27 (each later task that writes a payload key that `RunState` folds adds it here first, with a replay test, and lists `src/swarm_arena/_journal/replay.py` and `tests/_journal/test_replay.py` in its Files block and its `git add`: Tasks 19, 23, 24, 26). Payload keys pinned now: `run_config.config`, `plan_revision.{plan, origin, base_dir, planner_call_ids}`, `reserve.reservation`, `settle.{id, actual, estimated}`, `tool_start.{candidate_id, started, reservation_id}`, `tool_end.{candidate, a2a}`, `run_lost.{candidate_id, reservation_id}`, `score.score`, `learning.learning`, `decision.{round, decision}`, `agent_status.{status, reason}`, `notes_saved.text`, `frontier.{round, frontier}`, `round_started.{round, run_elapsed_s}`, `round_closed.{round, run_elapsed_s}`, `goal_*.{reason}`, `goal_started.pool` (present only on a continuation, Task 27), `stop_requested.now`, `takeover.force`, `limits_changed.{old, new}`, `no_result.reason`, `command_start.{job_id, started, candidate_id, component}`, `command_end.{job_id, status}`, `agent_counters.{turns, board_reads}` (keyed by the event's `team`/`agent`). `_cmd_start(seq, job_id)` is a test helper like `_start`; `_PRE` is the `run_config` + `plan_revision` prefix that `...` stands for in these tests. `ledger_events` keeps the whole events, so later payload keys they carry (`run_started.final_slices`, Task 26; `round_started.shares`, Task 22) need no fold here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_journal -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_journal tests/_journal docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(journal): add A2A task records and the RunState replay fold" -m "Roadmap R4.14 gains plan_rejected: a rejected replanner revision must be journaled, and no existing type fits. It also gains agent_counters, so turns and board_reads survive a resume (§6.5), and R4.13.4 gains RunState.ledger_events (run_config and limits_changed included), so Ledger.replay re-executes the committed budget events instead of recomputing them. Each limits_changed.new replaces RunState.config.limits, so the state holds the latest journaled limits, and a goal_started after a goal_finished starts a new lifecycle, so a continued goal is running once that event is committed; R4.14 gains the goal_started payload pool (edits skipped when the roadmap already has them)."
```

### Task 8: Budget amounts and splits

**Files:**
- Create: `src/swarm_arena/_budget/__init__.py` (empty), `src/swarm_arena/_budget/amount.py`, `src/swarm_arena/_budget/split.py`
- Test: `tests/_budget/test_amount.py`, `tests/_budget/test_split.py`

**Interfaces:**
- Produces: `Amount` (R4.13.6) with `__add__`, `__sub__`, `fits_in`, `Amount.from_estimate(e: Estimate) -> Amount`, `Amount.to_estimate() -> Estimate`; `MICRO_USD = 1e-6`; in `split.py`: `weighted_shares(total: Amount, weights: Mapping[str, float]) -> dict[str, Amount]`, `equal_shares(total: Amount, holders: Sequence[str]) -> dict[str, Amount]`, `round_pool(goal_left: Amount, rounds_left: int) -> Amount`. Rounding rule [plan]: tokens floor, USD floor to whole micro-dollars, so the sum of shares never exceeds the total; the remainder stays in the parent pool.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_budget/test_amount.py
def test_fits_in_respects_unset_caps() -> None:
    assert Amount(tokens=10**9, usd=5).fits_in(Amount(tokens=1, usd=5), cap_tokens=False, cap_usd=True)
    assert not Amount(usd=5.000001).fits_in(Amount(usd=5), cap_tokens=False, cap_usd=True)

def test_arithmetic() -> None:
    assert Amount(tokens=3, usd=1.5) - Amount(tokens=1, usd=0.5) == Amount(tokens=2, usd=1.0)
```

```python
# tests/_budget/test_split.py
def test_equal_shares_never_exceed_total() -> None:
    shares = equal_shares(Amount(tokens=100, usd=1.0), ["a", "b", "c"])
    assert all(s == Amount(tokens=33, usd=0.333333) for s in shares.values())
    assert sum(s.usd for s in shares.values()) <= 1.0

def test_weighted_shares() -> None:
    assert weighted_shares(Amount(usd=8.0), {"g1": 3.0, "g2": 1.0}) == {"g1": Amount(usd=6.0), "g2": Amount(usd=2.0)}

def test_round_pool_divides_by_rounds_left() -> None:
    assert round_pool(Amount(tokens=90, usd=0.9), 3) == Amount(tokens=30, usd=0.3)

def test_zero_weight_and_empty_holders() -> None:
    assert equal_shares(Amount(usd=1.0), []) == {}
    with pytest.raises(ValueError):
        weighted_shares(Amount(usd=1.0), {"g": 0.0})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_budget/test_amount.py tests/_budget/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `Amount` (`FrozenModel`) and the split functions with the floor rule above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_budget -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_budget tests/_budget
git commit -m "feat(budget): add amounts and budget splits that never exceed their pool"
```

### Task 9: `Ledger`, `LimitReached`, and the `Manager` ABC

**Files:**
- Create: `src/swarm_arena/_budget/ledger.py`, `src/swarm_arena/_agent/__init__.py` (empty), `src/swarm_arena/_agent/manager.py`, `src/swarm_arena/_agent/context.py` (the `AgentContext` class with a docstring only; Task 20 writes its body), `src/swarm_arena/_model/planning_types.py` (`RoundReport` only; Task 11 adds the other types)
- Modify: `src/swarm_arena/manager.py`, `tests/test_public_api.py`, `CHANGELOG.md` ("`Manager` ABC, `LimitReached`, and `RoundReport` (`swarm_arena.manager`)."), `tests/_fakes/tools.py` (add `class NoopManager(Manager)`: `Config(models: list[str] = [])`, `models(self) -> tuple[str, ...]` returns `tuple(config.models)`, `async def run_round(self, ctx: AgentContext) -> RoundReport` returns `RoundReport()`; Tasks 10 and 16 use it)
- Modify (also): `src/swarm_arena/_journal/replay.py` (add `continues_on_resume`; only resume, Task 27, uses it), roadmap R4.13.6 (`Ledger.replay(state, *, journal, ids)` as history-only re-execution, `open_goal(..., *, pool=None)`, `free()`, `change_limits()`, the `end_goal` free-pool rule, and the holder-pool, `run_exhausted`, and `goal_exhausted` comments; skipped when present)
- Test: `tests/_budget/test_ledger.py`, `tests/_agent/test_manager.py`

**Interfaces:**
- Consumes: `Amount`, `Limits`, `weighted_shares`, `equal_shares`, `round_pool`, `JournalWriter.append_durable`, `IdGen.next("r")`, `RunState.reservations`, `RunState.ledger_events`, `GoalState.status`/`rounds_closed`, `Plan.rounds_total`, `Configurable`, `FrozenModel`.
- Produces (R4.4): `LimitReached`, `Manager` (`preflight`, `models() -> tuple[str, ...]` default `()`, abstract `run_round(self, ctx: AgentContext) -> RoundReport`; `AgentContext` imported under `TYPE_CHECKING` from `_agent/context.py`), `RoundReport`; `swarm_arena.manager` exports `Manager`, `LimitReached`, `RoundReport` here, so Tasks 10 and 16 can use `Manager.models()` and `Ref.build(Manager)`.
- Produces (R4.13.6): `ReservationKind`, `Reservation`, `Ledger` with every listed method. Pool model [plan]: run cap = `limits.max_usd` / `limits.max_tokens` (unset = unlimited); reserve = `reserve_fraction` × cap; work pool = cap − reserve, split over all goals by `goal_weights` (equal by default) in `hold_final_slices`, which receives every goal id, queued goals included. Holder pools: agent → its round share; `<goal>/:judges`, `:planner`, `:curator` → the reserve left after final slices, then that goal's pool; `<goal>/:final` → that goal's final slice, then the reserve left after final slices, then that goal's pool (a second converge lifecycle and the final judge can run past the slice, R4.13.6); `":run"` → the reserve. A reserve that spans two sources debits them in that order. Exhaustion [plan]: `goal_exhausted(g)` is True when g's pool (its work-pool share plus returns, minus every charge to it and to its agents' shares: settled actual, estimated, and open reservations) is `<= 0` in a capped unit; `run_exhausted()` is the same over the whole work pool (cap − reserve). `Ledger.replay(state, *, journal, ids)` [plan] re-executes history only. It builds the `Ledger` from the journaled `run_config` event (limits, `reserve_fraction`, `goal_weights`), then re-applies `state.ledger_events` in seq order exactly as the live run did, with journaling off: `run_started.final_slices` → `hold_final_slices`; `goal_started` → `open_goal(g, pool=payload.get("pool"))`; `round_started.shares` → restore exactly those shares for that goal and round (no recompute); `reserve` → debit by the live funding rule above; `settle` → credit; `goal_finished` → `end_goal`, always (live calls it after every goal end, §4.10); `limits_changed` → `change_limits(new)`. The result never depends on a later event or on the resuming config. Every reservation with no committed settle stays fully spent. A pool that replay drives below zero counts as exhausted (no error). Live methods [plan]: `open_goal(goal_id, *, pool: Amount | None = None)` — `None` is a first start; an `Amount` re-opens an ended goal so that exactly `pool` is left to it, taken from the free pool, and raises `ValueError` when it does not fit in `free()`. `free() -> Amount` — the work pool that no goal holds; `end_goal` and `change_limits` feed it when no goal is left that is not ended, and re-opens reduce it. `change_limits(limits: Limits) -> None` (sync) — a raised cap adds new − old per capped unit; its work-pool part is split by `goal_weights` over the goals not ended (with none, it goes to `free()`), and its reserve part joins the reserve. `end_goal` — the unused pool (max 0) goes by `goal_weights` to the goals not ended (running or queued); with none, it joins `free()`. `continues_on_resume(state: RunState, goal_id: str, limits: Limits) -> bool` (plan-private, in `_journal/replay.py`; only resume, Task 27, uses it): the goal is `"stopped"`, `rounds_closed < plan.rounds_total(limits)` for its current revision, and its reason is `"stop"` or `"stop --now"`, or `limits != state.config.limits` (this resume journals `limits_changed`).

- [ ] **Step 1: Write the failing tests**

```python
# fixture `ledger`: Limits(max_usd=100.0), reserve_fraction=0.15, goal_weights=None, a real JournalWriter
async def test_final_slices_must_fit_reserve(ledger: Ledger) -> None:
    with pytest.raises(ValueError, match="reserve"):
        await ledger.hold_final_slices({"g1": Amount(usd=10), "g2": Amount(usd=10)})   # reserve is 15

async def test_round_and_agent_shares(ledger: Ledger) -> None:
    await ledger.hold_final_slices({"g1": Amount(usd=5), "g2": Amount(usd=5)})        # work pool 85 → 42.5 each
    ledger.open_goal("g1")
    ledger.open_round("g1", 1, ["g1/t/a", "g1/t/b"], rounds_left=2)                     # 21.25 → 10.625 each
    assert ledger.remaining("g1/t/a") == Amount(usd=10.625)

async def test_reserve_is_durable_and_refuses_overdraw(ledger: Ledger, store: LocalStore) -> None:
    ...  # setup as above
    res = await ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    assert read_owner(store)[0].head.seq >= _seq_of(store, res.id)
    with pytest.raises(LimitReached) as err:
        await ledger.reserve("g1/t/a", Amount(usd=1), kind="tool", goal_id="g1")
    assert err.value.limit == "budget"

async def test_settle_is_idempotent_and_returns_unused_to_holder(ledger: Ledger, store: LocalStore) -> None:
    res = await ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    await ledger.settle(res.id, Amount(usd=4))
    await ledger.settle(res.id, Amount(usd=4))
    assert ledger.remaining("g1/t/a") == Amount(usd=6.625) and _count(store, "settle") == 1

async def test_estimated_settle_keeps_full_charge(ledger: Ledger) -> None:
    res = await ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1")
    await ledger.settle(res.id, Amount(usd=0), estimated=True)
    assert ledger.remaining("g1/t/a") == Amount(usd=0.625)

async def test_unused_agent_share_stays_in_goal_pool(ledger: Ledger) -> None:
    res = await ledger.reserve("g1/t/a", Amount(usd=10), kind="tool", goal_id="g1"); await ledger.settle(res.id, Amount(usd=4))
    ledger.open_round("g1", 2, ["g1/t/a"], rounds_left=1)
    assert ledger.remaining("g1/t/a") == Amount(usd=38.5)                               # 42.5 − 4

async def test_end_goal_returns_unused_to_running_and_queued_goals(ledger: Ledger) -> None:
    await ledger.end_goal("g1")                                                        # g1 spent 4
    ledger.open_goal("g2"); ledger.open_round("g2", 1, ["g2/t/a"], rounds_left=1)
    assert ledger.remaining("g2/t/a") == Amount(usd=42.5 + 38.5)

async def test_final_uses_slice_then_reserve_then_goal_pool(ledger: Ledger) -> None:
    ...  # slices g1 5, g2 5 (reserve left 5); open_goal g1 (pool 42.5)
    await ledger.reserve("g1/:judges", Amount(usd=5), kind="judge", goal_id="g1")      # reserve left after slices: 5 → 0
    await ledger.reserve("g1/:judges", Amount(usd=1), kind="judge", goal_id="g1")      # from g1's pool → 41.5
    assert ledger.remaining("g1/:judges") == Amount(usd=41.5)                          # never the slice
    assert ledger.remaining("g1/:final") == Amount(usd=46.5)                           # slice 5 + pool 41.5
    await ledger.reserve("g1/:final", Amount(usd=5), kind="final", goal_id="g1")       # the slice
    await ledger.reserve("g1/:final", Amount(usd=1), kind="final", goal_id="g1")       # from g1's pool
    assert ledger.remaining("g1/:judges") == Amount(usd=40.5)
    await ledger.reserve("g1/:final", Amount(usd=40.5), kind="final", goal_id="g1")
    with pytest.raises(LimitReached):                                                   # all three are empty
        await ledger.reserve("g1/:final", Amount(usd=0.01), kind="final", goal_id="g1")

async def test_overspend_is_recorded(ledger: Ledger, store: LocalStore) -> None:
    res = await ledger.reserve("g1/t/a", Amount(usd=1), kind="tool", goal_id="g1")
    await ledger.settle(res.id, Amount(usd=1.5))
    assert _last(store, "settle").payload["overspend"] == {"tokens": 0, "usd": 0.5}

async def test_parallel_reserve_never_overdraws(ledger: Ledger) -> None:
    await ledger.hold_final_slices({"g1": Amount(usd=5), "g2": Amount(usd=5)})
    ledger.open_goal("g1")
    ledger.open_round("g1", 1, ["g1/t/a", "g1/t/b"], rounds_left=2)                     # share 10.625
    results: list[str] = []
    async def one() -> None:
        try:
            await ledger.reserve("g1/t/a", Amount(usd=1.0), kind="tool", goal_id="g1")
            results.append("ok")
        except LimitReached as err:
            results.append(err.limit)
    async with anyio.create_task_group() as tg:
        for _ in range(20):
            tg.start_soon(one)
    assert results.count("ok") == 10 and results.count("budget") == 10
    assert ledger.remaining("g1/t/a") == Amount(usd=0.625)

async def test_unlimited_when_caps_unset(tmp_path: Path) -> None:
    ...  # Limits() with no max_usd/max_tokens: reserve of Amount(usd=1e9) succeeds

async def test_replay_charges_unsettled_in_full(tmp_path: Path) -> None:
    ...  # journal: reserve r1 (10), reserve r2 (10), settle r2 (4);
    ...  # Ledger.replay(RunState.from_events(committed)) (the same re-execution path) → g1 spent == 14

HOLDERS = ["g1/t/a", "g1/t/b", "g1/:judges", "g1/:final", "g2/:final", ":run"]

async def test_replay_matches_live_ledger(ledger: Ledger, env: LedgerEnv) -> None:
    # live; the test journals run_started {final_slices}, goal_started, round_started {round: 1, shares: {h: remaining(h)}},
    # and goal_finished g2 {reason: None} where the engine would:
    await ledger.hold_final_slices({"g1": Amount(usd=5), "g2": Amount(usd=5)})
    ledger.open_goal("g1"); ledger.open_goal("g2")
    ledger.open_round("g1", 1, ["g1/t/a", "g1/t/b"], rounds_left=2)        # round_started.shares = remaining(h)
    res = await ledger.reserve("g1/t/a", Amount(usd=10.625), kind="tool", goal_id="g1")
    await ledger.settle(res.id, Amount(usd=10.625))
    await ledger.reserve("g1/:judges", Amount(usd=6), kind="judge", goal_id="g1")   # 5 from the reserve, 1 from g1's pool
    await ledger.end_goal("g2")
    live = {h: ledger.remaining(h) for h in HOLDERS}
    replayed = Ledger.replay(RunState.from_events(await env.committed()), journal=env.journal, ids=env.ids)
    assert {h: replayed.remaining(h) for h in HOLDERS} == live
    assert replayed.remaining("g1/t/a") == Amount() and replayed.remaining("g1/t/b") == Amount(usd=10.625)

async def test_replay_restores_final_slices(env: LedgerEnv) -> None:
    replayed = ...  # replay of a journal whose run_started.final_slices == {"g1": 5, "g2": 5}
    before = replayed.remaining("g1/:judges")
    await replayed.reserve("g1/:final", Amount(usd=5), kind="final", goal_id="g1")
    assert replayed.remaining("g1/:judges") == before                      # paid by the restored slice
    await replayed.reserve("g1/:final", Amount(usd=0.01), kind="final", goal_id="g1")
    assert replayed.remaining("g1/:judges") == before - Amount(usd=0.01)   # the slice is used up

async def test_limits_changed_replayed_at_its_position(env: LedgerEnv) -> None:
    ...  # live: Limits(max_usd=100), slices 5/5, g2 ended unspent; journal it
    env.ledger.change_limits(Limits(max_usd=200.0))                       # the test journals limits_changed {old, new}
    state = RunState.from_events(await env.committed())
    for _ in range(2):                                                    # replaying again gives the same value
        replayed = Ledger.replay(state, journal=env.journal, ids=env.ids)
        replayed.open_round("g1", 1, ["g1/t/a"], rounds_left=1)
        assert replayed.remaining("g1/t/a") == Amount(usd=170.0)          # 42.5 + g2's 42.5 + the added 85, applied once

async def test_continued_goal_reopens_from_free_pool(env: LedgerEnv) -> None:
    ...  # slices 5/5; open_goal g1, g2; g1 round 1 spends 4; goal_finished g1 {"reason": "stop"};
    ...  # end_goal("g1") sends g1's 38.5 to g2; g2 spends 50 (it uses g1's return); goal_finished g2 {"reason": "stop"};
    ...  # end_goal("g2") leaves no goal open. The test journals goal_started {pool} for each re-open below.
    assert env.ledger.free() == Amount(usd=31.0)
    env.ledger.open_goal("g1", pool=Amount(usd=15.5))
    with pytest.raises(ValueError):
        env.ledger.open_goal("g2", pool=Amount(usd=16.0))                 # only 15.5 is free
    env.ledger.open_goal("g2", pool=Amount(usd=15.5))
    assert env.ledger.free() == Amount()
    env.ledger.open_round("g1", 2, ["g1/t/a"], rounds_left=2)
    assert env.ledger.remaining("g1/t/a") == Amount(usd=7.75)

async def test_replay_is_history_only_across_lifecycles(env: LedgerEnv) -> None:
    ...  # the journal of test_continued_goal_reopens_from_free_pool up to the two re-opens
    a, prefix = _balances(env.ledger), await env.committed()
    ...  # then g1 round 2 spends 5; goal_finished g1 {"reason": None}; end_goal("g1")
    b, full = _balances(env.ledger), await env.committed()
    assert _balances(Ledger.replay(RunState.from_events(prefix), journal=env.journal, ids=env.ids)) == a
    assert _balances(Ledger.replay(RunState.from_events(full), journal=env.journal, ids=env.ids)) == b
    assert (a["charged"], b["charged"]) == (Amount(usd=54.0), Amount(usd=59.0))   # 4 + 50, then + 5: live and replay alike

async def test_exhaustion_flags(ledger: Ledger) -> None:
    ...  # slices 5/5, open_goal g1, g2; charge g1's whole 42.5 pool (agent and :judges charges)
    assert ledger.goal_exhausted("g1") and not ledger.run_exhausted()
    ...  # one more settle on g2 with an overspend so that charges on goal pools reach the 85.0 work pool
    assert ledger.run_exhausted()
```

```python
# tests/_agent/test_manager.py
class _M(Manager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        return RoundReport(summary="ok")

def test_manager_defaults() -> None:
    m = _M()
    assert m.models() == () and m.preflight() is None
    noop = Ref(path="_fakes.tools:NoopManager", config={"models": ["opus"]}).build(Manager)
    assert isinstance(noop, Manager) and noop.models() == ("opus",)

def test_limit_reached() -> None:
    err = LimitReached("budget", "goal budget used up")
    assert isinstance(err, RuntimeError) and err.limit == "budget" and str(err) == "goal budget used up"
```

`tests/test_public_api.py`: add `Manager`, `LimitReached`, `RoundReport` to the `swarm_arena.manager` list. `env: LedgerEnv` is a fixture of the test module: a dataclass with `ledger`, `journal`, `ids`, `limits`, `committed() -> list[Event]` (read back through `read_committed`), `count(type)`, `charged(holder)`, `last_reservation()`; `tests/_budget/test_tranche.py` (Task 10) defines the same fixture. `_balances(ledger: Ledger) -> dict[str, Amount]` is a test helper: `remaining(h)` for every `h` in `HOLDERS`, plus `"free"` → `free()` and `"charged"` → the sum of the ledger's charges (settled actual, estimated, and open reservations).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_budget/test_ledger.py tests/_agent/test_manager.py tests/test_public_api.py -v`
Expected: FAIL with `ModuleNotFoundError` and a snapshot mismatch.

- [ ] **Step 3: Implement** `Ledger` in `_budget/ledger.py`. All bookkeeping is synchronous and done before the first `await` of `reserve` (check, then debit), so parallel calls cannot overdraw; if the durable `reserve` commit raises, the debit stays (conservative). `reserve` journals `reserve` with payload `{"reservation": Reservation.model_dump()}`; `settle` journals `settle` with `{"id", "actual", "estimated", "overspend"}` (`overspend` = the part of `actual` above the reservation, or `null`; §8.1 "the journal records every overspend") and is a no-op for a settled id. `end_goal` journals nothing new beyond `goal_finished` (the engine writes it); returns (max 0) go by `goal_weights` among goals not yet ended, and with none to `free()`. `open_goal(pool=)` and `change_limits` journal nothing either (the engine writes `goal_started {pool}` and `limits_changed`). `Ledger.replay`, `free`, `change_limits`, `continues_on_resume`, the `:final` fallback, and the exhaustion flags follow the rules in Produces; replay builds a `Ledger` from the journaled `run_config` with a no-op journal, applies the events, then attaches `journal`. `LimitReached(limit, message)` calls `RuntimeError.__init__(message)` and sets `limit`. `Manager(Configurable, ABC)`; numpydoc docstrings with Examples on `Manager` (a small subclass and its `models()`), `LimitReached`, and `RoundReport`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_budget tests/_agent tests/test_public_api.py tests/test_docstrings.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_budget/ledger.py src/swarm_arena/_agent src/swarm_arena/_model/planning_types.py src/swarm_arena/manager.py src/swarm_arena/_journal/replay.py tests/_budget/test_ledger.py tests/_agent tests/_fakes/tools.py tests/test_public_api.py CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(budget): add the reservation ledger, LimitReached, and the Manager ABC" -m "The ledger raises LimitReached, and the estimate and validate_plan (Tasks 10, 16) need Manager.models(), so the Manager ABC comes first. Ledger.replay re-executes history only (final slices, journaled round shares, the live funding rule): every goal end returns its unused pool (spec 4.10), and a continued goal is re-opened by a journaled goal_started {pool} from the free pool, so a later resume reproduces every balance. R4.13.6 gets replay(state, *, journal, ids), open_goal(..., *, pool=None), free(), change_limits(), the end_goal free-pool rule, and the holder-pool and exhaustion comments (skipped when the roadmap has them)."
```

### Task 10: Tranches, worst-case estimate, LLM scope

**Files:**
- Create: `src/swarm_arena/_budget/tranche.py`, `src/swarm_arena/_budget/estimate.py`, `src/swarm_arena/_budget/llm_scope.py`
- Modify: `tests/_fakes/tools.py` (add `class ProviderJudge(Judge)`: `name = "prov"`, empty `Config`, `providers()` returns `frozenset({"openai"})`, `judge()` returns `Score.minimize(c.output.data["y"])` per candidate; `_provider_judge()` in the tests is `Ref(path="_fakes.tools:ProviderJudge")`), roadmap R4.13.6 (the `Tranche` comment; the `final_slice_estimate` comment's judge term becomes `judge_amount(ranking, n_final, llm=True)` with `n_final` as below; each skipped when present)
- Test: `tests/_budget/test_tranche.py`, `tests/_budget/test_estimate.py`, `tests/_budget/test_llm_scope.py`

**Interfaces:**
- Consumes: `Ledger`, `Amount`, `Plan`, `RunConfig`, `Tool.estimate`, `Manager.models` (Task 9; tests use `_fakes.tools:NoopManager` with `models=["opus"]`), `Judge` (with `models()` and `providers()`), `ProgramJudge`, `Lexicographic` (M1a), `ModelCatalog`, `ModelSpec.Prices`, `Limits.llm_tranche_tokens`/`llm_tranche_usd`.
- Produces (R4.13.6, R4.13.7): `Tranche` (`reserve_call`, `settle_call`, `close`, plus `pending() -> int`); `PriceSource`, `register_price_source`, `worst_case_estimate`; `final_slice_estimate(plan: Plan, config: RunConfig, tools: Mapping[str, Tool], price_of: Callable[[str], ModelSpec.Prices | None]) -> Amount`; `judge_amount(judge: Judge, n: int, config: RunConfig, price_of: Callable[[str], ModelSpec.Prices | None], *, llm: bool = True) -> Amount` (the judge reservation rule of R4.13.6 [plan], first match wins: `ProgramJudge` → `Amount()` (runs in the driver, spends nothing); `Lexicographic` → the sum over its members (same `llm`); a judge with non-empty `models()` → `Amount()` when `llm=False` (its calls are paid only through the scope's tranche), else the componentwise max of one tranche (`llm_tranche_tokens`, `llm_tranche_usd`) and Σ over `models()` (repeats kept) of `2 × n × spec.max_tokens` tokens priced at `max(2 × input, output)` per 1M; a judge with non-empty `providers()` → `n × default_run_reservation` (settled `estimated=True`: judges have no usage channel); any other judge (`ScriptJudge`, connector judges without providers) → `Amount()`, because the ledger counts only paid calls (§8.1). M3 adds the LLM judges that reach the `models()` branch: no barrier hold, and an LLM-priced final-judge amount (M3 Task 12)); `final_slice_estimate` = `(1 + max_join_attempts) × Σ converge-team agents [max_tool_runs_per_round × (tool estimate or default_run_reservation) + Σ aliases in manager.models() max_turns_per_round × spec.max_tokens priced at max(input, output)] + judge_amount(ranking, n_final, llm=True)`, with `n_final = top_n` in select mode and `converge-team agents × max_tool_runs_per_round + top_n` in team mode (the final-judge batch, Task 24); `make_price_of(models: ModelCatalog) -> Callable[[str], ModelSpec.Prices | None]`; `LLMScope`, `current_llm_scope`, `llm_scope`, `requesting_call`, `current_requesting_call`. `tools` keys are agent addresses.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_budget/test_tranche.py
async def test_calls_spend_locally_and_new_tranche_when_used_up(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(tokens=25_000, usd=0.25))
    ids = [await t.reserve_call(Amount(usd=0.1)) for _ in range(3)]       # 0.1 + 0.1, then a new tranche
    assert env.count("reserve") == 2 and len(set(ids)) == 3

async def test_call_bigger_than_tranche_gets_own_reservation(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    await t.reserve_call(Amount(usd=1.0))
    assert env.last_reservation().kind == "call"

async def test_close_returns_unused_and_charges_unsettled_calls(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    a = await t.reserve_call(Amount(usd=0.05)); await t.reserve_call(Amount(usd=0.05))
    await t.settle_call(a, Amount(usd=0.01))
    await t.close()
    assert env.charged("g1/t/a") == Amount(usd=0.06) and t.pending() == 0     # 0.01 + 0.05 unsettled in full

async def test_parallel_calls_open_one_new_tranche(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    ids = [await t.reserve_call(Amount(usd=0.2))]                          # 0.05 left: the next two do not fit
    async def one() -> None:
        ids.append(await t.reserve_call(Amount(usd=0.1)))
    async with anyio.create_task_group() as tg:
        tg.start_soon(one); tg.start_soon(one)
    assert env.count("reserve") == 2 and len(set(ids)) == 3                 # one new tranche, not two

async def test_settle_after_close_is_noop(env: LedgerEnv) -> None:
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(usd=0.25))
    a = await t.reserve_call(Amount(usd=0.05))
    await t.close()
    settles = env.count("settle")
    await t.settle_call(a, Amount(usd=0.01))
    assert env.charged("g1/t/a") == Amount(usd=0.05) and env.count("settle") == settles

async def test_new_tranche_shrinks_to_what_is_left(env: LedgerEnv) -> None:
    ...  # Limits(max_tokens=...) so that g1/t/a has 10_000 tokens left
    t = Tranche(env.ledger, "g1/t/a", "g1", size=Amount(tokens=25_000))
    await t.reserve_call(Amount(tokens=5_000))
    assert env.last_reservation().amount == Amount(tokens=10_000)
    with pytest.raises(LimitReached):
        await t.reserve_call(Amount(tokens=12_000))
```

```python
# tests/_budget/test_estimate.py
def test_worst_case_tool_runs_plus_reserve() -> None:
    plan = _plan(teams=2, agents_per_team=2, tool=_FixedEstimateTool(usd=0.5), rounds=3)
    total = worst_case_estimate([plan], RunConfig(), _tools(plan), lambda a: None)
    assert total.usd == pytest.approx(3 * 4 * 3 * 0.5 / 0.85)        # Σ rounds Σ agents × max_tool_runs × estimate, / (1 − reserve)

def test_no_estimate_uses_default_run_reservation() -> None:
    total = worst_case_estimate([_plan(teams=1, agents_per_team=1, tool=_NoEstimateTool(), rounds=1)], RunConfig(), ..., lambda a: None)
    assert total.tokens == round(3 * 100_000 / 0.85) and total.usd == pytest.approx(3 * 1.0 / 0.85)

def test_manager_tokens_priced_at_max_of_input_output() -> None:
    # catalog alias "opus": max_tokens 8000, prices input 5, output 25 (USD / 1M); fake manager models() == ("opus",)
    total = worst_case_estimate([...one agent, one round, tool estimate 0...], cfg, tools, make_price_of(cfg.models))
    assert total.tokens == round(20 * 8000 / 0.85) and total.usd == pytest.approx(20 * 8000 * 25 / 1e6 / 0.85)

def test_final_slice_covers_join_attempts() -> None:
    plan = _plan_with_converge_team(agents=2, tool=_FixedEstimateTool(usd=0.5), max_join_attempts=1)   # ranking: a ProgramJudge
    assert final_slice_estimate(plan, RunConfig(), ..., lambda a: None).usd == pytest.approx(2 * 2 * 3 * 0.5)

def test_final_slice_covers_converge_managers() -> None:
    # converge team of 2 NoopManager(models=["opus"]) agents; "opus": max_tokens 8000, prices 5 / 25;
    # tool estimate 0; ProgramJudge ranking; max_join_attempts 1
    total = final_slice_estimate(plan, cfg, tools, make_price_of(cfg.models))
    assert total.tokens == 2 * 2 * 20 * 8000                              # (1 + joins) × agents × turns × max_tokens
    assert total.usd == pytest.approx(2 * 2 * 20 * 8000 * 25 / 1e6)

def test_final_slice_final_judge_batch_in_team_mode() -> None:
    plan = _plan_with_converge_team(agents=2, tool=_FreeTool(), ranking_judge=_provider_judge(), top_n=2)
    assert final_slice_estimate(plan, RunConfig(), ..., lambda a: None).usd == pytest.approx((2 * 3 + 2) * 1.0)

def test_final_slice_adds_the_final_judge() -> None:
    plan = _plan(teams=1, agents_per_team=1, tool=_FixedEstimateTool(usd=0.5), rounds=1, ranking_judge=_provider_judge(), top_n=2)
    assert final_slice_estimate(plan, RunConfig(), ..., lambda a: None) == Amount(tokens=200_000, usd=2.0)   # select mode

def test_judge_amount() -> None:
    cfg, price_of = RunConfig(), lambda a: None
    assert judge_amount(ProgramJudge("_fakes.tools:rosen_score"), 3, cfg, price_of) == Amount()
    assert judge_amount(_script_judge(), 3, cfg, price_of) == Amount()                       # no paid call
    assert judge_amount(_provider_judge(), 3, cfg, price_of) == Amount(tokens=300_000, usd=3.0)
    assert judge_amount(_FakeLLMJudge(models=("opus",)), 3, cfg, price_of, llm=False) == Amount()
    assert judge_amount(Lexicographic([ProgramJudge("_fakes.tools:rosen_score"), _provider_judge()]), 3, cfg, price_of) \
        == Amount(tokens=300_000, usd=3.0)

def test_judge_amount_prices_llm_judge_models() -> None:
    # alias "j": max_tokens 1000, prices input 1 / output 2; small tranche so the model sum wins
    cfg = _cfg(models={"j": ...}, limits=Limits(llm_tranche_tokens=1_000, llm_tranche_usd=0.001))
    judge = _FakeLLMJudge(models=("j", "j"))
    amount = judge_amount(judge, 3, cfg, make_price_of(cfg.models))
    assert amount.tokens == 12_000 and amount.usd == pytest.approx(12_000 * 2 / 1e6)   # 2 aliases × 2 × 3 × 1000 at max(2 × 1, 2)
    assert judge_amount(judge, 3, cfg, make_price_of(cfg.models), llm=False) == Amount()
    big = _cfg(models={"j": ...})                                         # default tranche 25,000 / 0.25 wins
    assert judge_amount(judge, 3, big, make_price_of(big.models)) == Amount(tokens=25_000, usd=0.25)

def test_registered_source_used_after_catalog_prices() -> None:
    register_price_source(lambda model: ModelSpec.Prices(input=1, output=2) if model == "x/y" else None)
    ...  # alias with prices=None and model "x/y" resolves to the registered price; catalog prices win when set
```

Test-module helpers: `_FakeLLMJudge(models: tuple[str, ...])` is a `Judge` whose `models()` returns that tuple; `_FreeTool` is a tool whose `estimate()` is `Estimate()` (zero).

```python
# tests/_budget/test_llm_scope.py
def test_scope_nests_and_restores() -> None:
    assert current_llm_scope() is None
    with llm_scope(_scope("manager")):
        with llm_scope(_scope("judge")):
            assert current_llm_scope().role == "judge"
        assert current_llm_scope().role == "manager"
    assert current_llm_scope() is None

async def test_requesting_call_is_per_task() -> None:
    seen: dict[str, str | None] = {}
    async def work(cid: str) -> None:
        with requesting_call(cid):
            await anyio.sleep(0.01); seen[cid] = current_requesting_call()
    async with anyio.create_task_group() as tg:
        tg.start_soon(work, "llm-1"); tg.start_soon(work, "llm-2")
    assert seen == {"llm-1": "llm-1", "llm-2": "llm-2"} and current_requesting_call() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_budget/test_tranche.py tests/_budget/test_estimate.py tests/_budget/test_llm_scope.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** the three modules. `worst_case_estimate`: for each plan, rounds = `plan.rounds_total(limits)`; per agent per round `max_tool_runs_per_round × (tool.estimate(ToolRequest()) or default_run_reservation)` plus, for each alias in `manager.models()`, `max_turns_per_round × spec.max_tokens` tokens priced at `max(input, output)` per 1M; add `final_slice_estimate` (the formula in Produces: converge-team tool runs and manager turns × (1 + `max_join_attempts`), 0 for select mode, plus `judge_amount(ranking, n_final, llm=True)`); divide the sum by `1 − reserve_fraction` [plan]. `Tranche`: an `anyio.Lock` is held around "find room or open a new tranche", so parallel calls open at most one new tranche; a new tranche reserves `min(size, ledger.remaining(holder))` and raises `LimitReached("budget")` only when that is below the call estimate; `settle_call` after `close` is a no-op (the call stays charged in full). Context variables via `contextvars.ContextVar`; tasks started by anyio copy the context, so each task sees its own value.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_budget -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_budget tests/_budget tests/_fakes/tools.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(budget): add LLM tranches, the worst-case estimate, and the LLM scope seams" \
  -m "judge_amount reserves only for judges that can pay (providers, LLM models), because the ledger counts only paid calls and a reservation for a free judge would make the estimate refuse runs. The final slice covers converge-manager turns and the team-mode final-judge batch. R4.13.6 gets the Tranche comment (serialized reserve_call, no-op settle after close) and the n_final judge term (each skipped when present)."
```

### Task 11: Planning and board types, ranking

**Files:**
- Create: `src/swarm_arena/_model/board_types.py`, `src/swarm_arena/_scoring/ranking.py`
- Modify: `src/swarm_arena/_model/planning_types.py` (Task 9 created it with `RoundReport`), `src/swarm_arena/planning.py`, `src/swarm_arena/board.py` (facades), `tests/test_public_api.py`, `CHANGELOG.md`
- Test: `tests/_model/test_planning_types.py`, `tests/_model/test_board_types.py`, `tests/_scoring/test_ranking.py`

**Interfaces:**
- Consumes: `Candidate`, `Score`, `Plan`, `Learning`, `Estimate`, `AgentStatus`, `RoundReport` (Task 9).
- Produces: `Decision`, `RoundState` (with `ranking_identity: dict[str | None, tuple[str, str, str]]`, one identity per group), `History` (R4.5); `Frontier`, `Row`, `LineageGraph` (R4.6); in `ranking.py`: `ranked(...)` (R4.13.11), `groups(plan: Plan) -> dict[str | None, list[str]]` (main-goal group `None` first, then parts in plan order; converge team excluded), `group_of(plan: Plan, team: str) -> str | None`, `newest_fidelity(candidates: Iterable[Candidate]) -> str | None` (fidelity of the highest round). Facade names: `swarm_arena.planning`: `History`, `RoundState`, `Decision`; `swarm_arena.board`: `Frontier`, `Row`, `LineageGraph`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_scoring/test_ranking.py
ID = ("program", "rh", "")
def test_ranked_filters_and_orders() -> None:
    a = _cand("c-1", round=1, fid="f1"); b = _cand("c-2", round=2, fid="f2"); c = _cand("c-3", round=2, fid="f2")
    g = _cand("c-4", round=2, fid="f2", failed_check=True); p = _cand("c-5", round=2, fid="f2", partial=True)
    s = [_score(x, 1.0) for x in (a, g, p)] + [_score(b, 2.0), _score(c, 2.0), _score(c, 9.0, judge="other")]
    out = ranked([a, b, c, g, p], s, identity=ID, fidelity=None, allow_partial=False)
    assert [x.id for x, _ in out] == ["c-2", "c-3"]        # newest fidelity only; tie → earliest id; gated out and partial dropped

def test_allow_partial_keeps_partial() -> None:
    ...  # same input, allow_partial=True → "c-5" included after the ties

def test_groups_main_first_then_parts() -> None:
    assert groups(_plan_with_parts()) == {None: ["main-t"], "A": ["a1", "a2"], "B": ["b1"]}
```

```python
# tests/_model/test_planning_types.py
def test_decision_merge_unions_and_keeps_first_reason() -> None:
    d = Decision(prune={"t1": "r1"}, policy="A").merge(Decision(prune={"t1": "r2", "t2/a": "r3"}, no_result="none", policy="B"))
    assert d == Decision(prune={"t1": "r1", "t2/a": "r3"}, no_result="none", policy="A+B")

def test_round_state_team_best_by_round() -> None:
    st = _state(rounds=[{"t1": 3.0}, {"t1": None}, {"t1": 5.0}])
    assert [s.value if s else None for s in st.team_best_by_round("t1")] == [3.0, 3.0, 5.0]
    assert st.team_best("t1").value == 5.0

def test_round_state_ranks_each_group_by_its_own_identity() -> None:
    st = _parts_state(identities={"A": ("a", "ha", ""), "B": ("b", "hb", "")}, scores_from=["a", "b"])  # every candidate scored by both
    # _parts_state also gives B's candidates scores under ("b", "hb", "pb2")
    assert {s.judge for _, s in st.ranked("A")} == {"a"} and {s.judge for _, s in st.ranked("B")} == {"b"}
    before = st.ranked("A")
    st2 = _with_identity(st, "B", ("b", "hb", "pb2"))                     # B's panel_hash changes (identity: judge, rubric, panel)
    assert st2.ranked("B") != []
    assert all(s.panel_hash == "pb2" for _, s in st2.ranked("B"))
    assert st2.ranked("A") == before
    assert st2.ranked("C") == []                                         # a group with no identity
```

```python
# tests/_model/test_board_types.py
def test_frontier_digest_is_stable() -> None:
    m = Frontier.Mark(run_id="r", run_root="/x", epoch=1, seq=4, segment="journal/e1-4.jsonl")
    assert Frontier(marks=[m]).digest() == Frontier(marks=[m]).digest() != Frontier().digest()

def test_lineage_parents_and_ancestors() -> None:
    g = LineageGraph(root="c", nodes={...}, edges=[("c", "b"), ("b", "a"), ("c", "a")])
    assert g.parents("c") == ["b", "a"] and g.ancestors("c") == ["b", "a"]
```

`tests/test_public_api.py`: add the six names to the two snapshot lists.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_model/test_planning_types.py tests/_model/test_board_types.py tests/_scoring/test_ranking.py tests/test_public_api.py -v`
Expected: FAIL with `ModuleNotFoundError` and snapshot mismatches.

- [ ] **Step 3: Implement** the types and `ranking.py`. Order in `ranked`: score key descending, then candidate `round` ascending, then id ascending (ids increase within a run). `team_best_by_round(team)[i]` is the best score of the team's candidates up to round `i + 1`. `RoundState.ranking_identity` is `dict[str | None, tuple[str, str, str]]` (R4.5): `ranked(group)` ranks under `ranking_identity[group]` (absent → `[]`), one identity per call; `team_best` and `team_best_by_round` use the identity of the team's group (`group_of`). `RoundState.groups()` and `ranked()` call `ranking.py`. numpydoc docstrings with Examples on every public name and method (doctests build small objects in memory).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_model tests/_scoring/test_ranking.py tests/test_public_api.py tests/test_docstrings.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_model src/swarm_arena/_scoring/ranking.py src/swarm_arena/planning.py src/swarm_arena/board.py tests CHANGELOG.md
git commit -m "feat(model): add planning and board types and the single ranking owner"
```

### Task 12: Board ABCs, `RunBoard`, frontier, digest v1

**Files:**
- Create: `src/swarm_arena/_board/__init__.py` (empty), `_board/base.py`, `_board/run_board.py`, `_board/frontier.py`, `_board/digest.py`
- Modify: `src/swarm_arena/board.py`, `tests/test_public_api.py`, `CHANGELOG.md`
- Test: `tests/_board/test_base.py`, `tests/_board/test_run_board.py`, `tests/_board/test_frontier.py`, `tests/_board/test_digest.py`

**Interfaces:**
- Consumes: `Frontier`, `Row`, `LineageGraph`, `Decision`, `GoalKey`, `Candidate`, `Score`, `Learning`, `Head`, `ranked`, `RunState`.
- Produces: `Board`, `BoardView` (R4.6; `swarm_arena.board` exports both); `RunBoard(*, run_id: str, run_root: str, head: Callable[[], Head | None])` with `register_goal(key: GoalKey, goal_id: str, part: str | None) -> None` and `RunBoard.from_state(state: RunState, *, run_root: str, head: Callable[[], Head | None]) -> RunBoard`; `build_frontier`, `build_digest`, `estimate_tokens` (R4.13.14). `BoardView` read methods raise `KeyError` for an id outside the snapshot and `ValueError` for another goal or namespace.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_board/test_run_board.py
def test_snapshot_hides_items_after_mark(board: RunBoard) -> None:
    board.put_candidate(_c("c-1"), epoch=1, seq=4); board.put_candidate(_c("c-2"), epoch=1, seq=6)
    view = board.snapshot(KEY, _frontier(seq=5))
    assert view.get_candidate("c-1").id == "c-1"
    with pytest.raises(KeyError):
        view.get_candidate("c-2")

def test_view_rejects_other_goal(board: RunBoard) -> None:
    with pytest.raises(ValueError):
        board.snapshot(KEY, _frontier(seq=5)).best(OTHER_KEY, 1, judge="program")

def test_best_ranks_current_identity_only(board: RunBoard) -> None:
    ...  # scores for "program" with rubric_hash "old" then "new": best() uses "new" (last put); old-identity scores ignored

def test_leaderboard_rows_record_identity(board: RunBoard) -> None:
    rows = board.leaderboard(KEY, judge="program")
    assert [r.rank for r in rows] == [1, 2] and rows[0].rubric_hash == "new" and rows[0].built_on == ["c-0"]

def test_search_is_case_insensitive_substring_newest_first(board: RunBoard) -> None:
    ...  # titles "Fit log(nu)", "LOG scale"; search "log" → both, newest first; tags filter applies

def test_learnings_relevance_order(board: RunBoard) -> None:
    ...  # pinned first, then this goal, then verified, then matching tags, then newest

def test_from_state_equals_fed_board(tmp_path: Path) -> None:
    ...  # a board fed by put_* and RunBoard.from_state(RunState of the same events) give equal snapshots

def test_view_reads_other_goals_learnings_in_namespace(board: RunBoard) -> None:
    ...  # a learning of goal "g2" (same namespace, committed before the mark) is returned by the "g" view's learnings()

def test_frontier_marks_this_run_at_current_head(board: RunBoard) -> None:
    assert board.frontier().marks == [Frontier.Mark(run_id="run-1", run_root="/runs", epoch=1, seq=9, segment="journal/e1-9.jsonl")]
```

```python
# tests/_board/test_digest.py
def test_digest_sections_and_count_line(view: BoardView) -> None:
    d = build_digest(view, goal=KEY, ranking="program", cap_tokens=1500, k=5)
    lines = d.splitlines()
    assert lines[0] == "Leaderboard (program, top 5):"
    assert lines[1].startswith("1. c-") and "fd-solver/tuner-1" in lines[1] and "built on" in lines[1]
    assert "Pinned:" in d and d.index("Pinned:") < d.index("Newest:")
    assert lines[-1] == "412 more learnings, 37 from this round. Use search_learnings."

def test_digest_respects_token_cap_for_any_board_size(big_view: BoardView) -> None:   # 5,000 learnings
    assert estimate_tokens(build_digest(big_view, goal=KEY, ranking="program", cap_tokens=1500)) <= 1500

def test_digest_cap_holds_for_non_ascii_titles(non_ascii_view: BoardView) -> None:
    # 5,000 learnings with CJK, emoji, and combining-mark titles (each <= 120 chars)
    d = build_digest(non_ascii_view, goal=KEY, ranking="program", cap_tokens=1500)
    assert len(d.encode("utf-8")) <= 1500                                 # the test's own count, not estimate_tokens
    assert d.splitlines()[0].startswith("Leaderboard")

def test_empty_board_digest() -> None:
    assert build_digest(_empty_view(), goal=KEY, ranking="program", cap_tokens=1500) == "No results yet. 0 learnings."

def test_estimate_tokens() -> None:
    assert [estimate_tokens(t) for t in ("", "abcde", "é", "日本", "🙂")] == [0, 5, 2, 6, 4]
```

The `view` fixture of `test_digest_sections_and_count_line` (and of every test that checks listed content) fits within its `cap_tokens`, counted in UTF-8 bytes.

```python
# tests/_board/test_frontier.py
def test_build_frontier_sorts_marks() -> None:
    f = build_frontier([("r2", "/b", Head(epoch=1, seq=2, segment="journal/e1-2.jsonl")),
                        ("r1", "/a", Head(epoch=2, seq=7, segment="journal/e2-7.jsonl"))], [], [])
    assert [m.run_id for m in f.marks] == ["r1", "r2"] and f.marks[0].epoch == 2
```

`tests/_board/test_base.py`: a minimal concrete subclass proves `open` and `close` default to no-ops and that abstract methods are required.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_board -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** Digest v1 layout [plan]: `Leaderboard (<ranking>, top <k>):`, then `"<rank>. <id> <team>/<agent> score=<value or key> built on <ids or —>"`; `Pinned:` and `Newest:` sections of `"- <id> <title>"`; the count line `"<n> more learnings, <m> from this round. Use search_learnings."`. To meet the cap, drop `Newest:` lines from the end, then `Pinned:` lines, then leaderboard rows beyond 1; the count line always stays. The whole output is at most `cap_tokens`; when even the markers and the count line do not fit, it returns the count line only, cut at a character boundary (R4.13.14). `estimate_tokens(text)` is `len(text.encode("utf-8"))`; its numpydoc docstring says this is an upper bound on tokens for byte-level tokenizers, not an estimate. `RunBoard` membership: an item is in a snapshot when its `(epoch, seq) <= (mark.epoch, mark.seq)` in chain order (`seq <= mark.seq` is exact for one run, R4.6).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_board tests/test_public_api.py tests/test_docstrings.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_board src/swarm_arena/board.py tests CHANGELOG.md
git commit -m "feat(board): add Board and BoardView ABCs, the in-memory RunBoard, frontiers, and the capped digest"
```

### Task 13: Policies

**Files:**
- Create: `src/swarm_arena/_planning/__init__.py` (empty), `_planning/policy.py`, `_planning/policies.py`
- Modify: `src/swarm_arena/__init__.py`, `src/swarm_arena/planning.py`, `tests/test_public_api.py`, `CHANGELOG.md`
- Test: `tests/_planning/test_policy.py`, `tests/_planning/test_policies.py`

**Interfaces:**
- Consumes: `RoundState`, `Decision`, `ranking.groups`, `ranking.ranked`, `AgentStatus`, `Ref`.
- Produces: `Policy` (R4.5); `KeepTopTeams`, `SuccessiveHalving`, `Patience`, `Chain` (R4.5 constructors); `guard(decision: Decision, state: RoundState) -> Decision` in `policy.py` (the core applies it after every policy: drops a prune that would remove a group's last team or prune a `starved` agent; sets `no_result="no passing candidate"` when `state.round == state.rounds_total` and no group has a ranked candidate). `swarm_arena.planning` adds only `Policy`; the four policies are top-level exports only (§9 lists `swarm_arena.planning.__all__` as `Planner`, `History`, `Policy`, `RoundState`, `Decision`, `FlowSpec`).

- [ ] **Step 1: Write the failing tests**

```python
# Scores are Score.maximize(value); one candidate per agent; all agents active unless stated.
def test_successive_halving_keeps_top_half_per_group() -> None:
    st = _state({"A": {"a1": 4, "a2": 1}, "B": {"b1": 3, "b2": 2}})
    d = SuccessiveHalving(keep=0.5, min_per_team=1).after_round(st)
    assert d.prune == {"A/a2": "SuccessiveHalving: rank 4 of 4 (best 1.0)", "B/b2": "SuccessiveHalving: rank 3 of 4 (best 2.0)"}

def test_successive_halving_keeps_min_per_team() -> None:
    st = _state({"A": {"a1": 4, "a2": 3}, "B": {"b1": 1, "b2": 0.5}})
    assert set(SuccessiveHalving(keep=0.5, min_per_team=1).after_round(st).prune) == {"B/b2"}

def test_starved_agent_never_pruned_for_lack_of_results() -> None:
    st = _state({"A": {"a1": 4}, "B": {"b1": 3, "b2": None}}, status={"B/b2": AgentStatus.STARVED})
    assert "B/b2" not in SuccessiveHalving().after_round(st).prune

def test_keep_top_teams_per_group_never_last_team() -> None:
    st = _state({"t1": {"a": 5}, "t2": {"a": 3}}, parts={"P": {"p1": {"a": 1}}})
    assert KeepTopTeams(1).after_round(st).prune == {"t2": "KeepTopTeams: rank 2 of 2 (best 3.0)"}

def test_patience_prunes_team_without_improvement() -> None:
    st = _state_by_round({"t1": [3, 3, 3], "t2": [1, 2, 4]})
    assert Patience(rounds=2).after_round(st).prune == {"t1": "Patience: no improvement for 2 rounds (best 3.0)"}

def test_chain_merges_in_order() -> None:
    d = Chain(KeepTopTeams(1), Patience(rounds=2)).after_round(_state_by_round({"t1": [3, 3, 3], "t2": [1, 2, 4], "t3": [0, 0, 0]}))
    assert d.policy == "KeepTopTeams+Patience" and set(d.prune) == {"t1", "t3"}

def test_spec_and_ref_forms_build() -> None:
    assert Ref(path="swarm_arena:SuccessiveHalving", config={"keep": 0.5}).build(Policy).config.keep == 0.5
    assert Chain(Ref(path="swarm_arena:Patience", config={"rounds": 2})).config.policies[0].path == "swarm_arena:Patience"
```

```python
# tests/_planning/test_policy.py
def test_guard_keeps_a_parts_last_team_and_starved_agents() -> None:
    st = _state({}, parts={"P": {"p1": {"a": 1}, "p2": {"a": 2}}}, status={"p2/b": AgentStatus.STARVED})
    d = guard(Decision(prune={"p1": "x", "p2": "y", "p2/b": "z"}), st)
    assert d.prune == {"p1": "x"}

def test_guard_sets_no_result_on_last_round_with_nothing_ranked() -> None:
    assert guard(Decision(), _state({"t": {"a": None}}, round=3, rounds_total=3)).no_result == "no passing candidate"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_planning/test_policy.py tests/_planning/test_policies.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** `SuccessiveHalving` ranks agents in each group by their best ranked score; keeps `ceil(keep × n)` agents, plus each team's best agent while that team has fewer than `min_per_team` kept agents and has a passing candidate; agents with no candidate and status `starved` are skipped. When every agent of a team is pruned, the prune key is the team (`"<team>"`), else `"<team>/<agent>"`. `KeepTopTeams` ranks teams by `team_best`; teams with no score rank last. `Patience` prunes a team whose `team_best_by_round` did not increase over the last `rounds` rounds. Reason strings are exactly as in the tests (`best` uses `Score.value`, else `key[0]`). Each policy's `after_round` ends with `guard`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_planning tests/test_public_api.py tests/test_docstrings.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_planning src/swarm_arena/__init__.py src/swarm_arena/planning.py tests CHANGELOG.md
git commit -m "feat(planning): add the Policy ABC and the four built-in pruning policies"
```

### Task 14: Provenance, stamping, and the `Planner` ABC

**Files:**
- Create: `src/swarm_arena/_planning/provenance.py`, `src/swarm_arena/_planning/planner.py`, `tests/_fakes/planners.py`
- Modify: `src/swarm_arena/planning.py`, `tests/test_public_api.py`, `CHANGELOG.md` ("`Planner` ABC (`swarm_arena.planning`).")
- Test: `tests/_planning/test_provenance.py`, `tests/_planning/test_planner.py`

**Interfaces:**
- Consumes: `Plan`, `Ref`, `RunConfig`, `History`, `Origin`, `CatalogKind`.
- Produces (R4.13.12): `RefSite`, `ref_sites`, `diff_revision`, `stamp`; `site_key(site: RefSite) -> str` (stable key: `agent:<team>/<agent>:tool|manager`, `converge:<team>/<agent>:tool|manager`, `check:<name>`, `judge:<name>`, `part:<name>:check:<n>` / `:judge:<n>`, `policy`, nested `<parent key>/<config path>`, override `round:<i>:<team>/<agent>`, `converge-round:<team>/<agent>`); `Planner` (R4.5). Override sites carry the agent's effective tool ref (tool config with `tool_overrides` merged). `tests/_fakes/planners.py`: `SequencePlanner(plans: Sequence[Plan])` (returns them in order, then the last), `AddAgentPlanner(entry_name: str)`, `RaisingPlanner()`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ref_sites_cover_every_depth() -> None:
    ptrs = {s.pointer: s.kind for s in ref_sites(_plan_with_everything())}
    assert ptrs["/teams/0/agents/1/tool"] == "tool" and ptrs["/teams/0/agents/1/manager"] == "manager"
    assert ptrs["/judges/0/config/judges/1"] == "judge"            # Lexicographic member
    assert ptrs["/policy/config/policies/0"] == "policy"           # Chain member
    assert ptrs["/parts/0/checks/0"] == "check" and ptrs["/converge/team/agents/0/tool"] == "tool"
    assert "/rounds/0/tool_overrides@/teams/0/agents/1/tool" in ptrs

def test_diff_marks_unchanged_human_sites_carried() -> None:
    r0 = stamp(_plan(), origin="human", parent=None)
    r1 = stamp(_add_agent(r0, "t1", "new"), origin="autonomous", parent=r0)
    human = {site_key(s): True for s in ref_sites(r0)}
    carried = {s.pointer: s.carried_human for s in diff_revision(r0, r1, parent_human=human)}
    assert carried["/judges/0"] is True and carried["/teams/0/agents/1/tool"] is False

def test_changed_override_is_not_carried() -> None:
    ...  # r1 changes rounds[0].tool_overrides → that override site has carried_human False

def test_stamp_sets_revision_and_parent() -> None:
    r0 = stamp(_plan().model_copy(update={"origin": "human", "revision": 9}), origin="llm", parent=None)
    r1 = stamp(r0, origin="autonomous", parent=r0)
    assert (r0.origin, r0.revision, r0.parent_revision) == ("llm", 0, None)
    assert (r1.origin, r1.revision, r1.parent_revision) == ("autonomous", 1, 0)

def test_planner_subclass_cannot_claim_human() -> None:
    with pytest.raises(TypeError, match="human"):
        class Bad(Planner):
            origin = "human"
            async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan: ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_planning/test_provenance.py tests/_planning/test_planner.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** Nested refs: any config value that is a dict with a `path` key in `"module:qualname"` form, or a list of such, is a nested site of the parent's kind. `diff_revision` returns every site of `child`; `carried_human` is true only when the parent has a site with the same `site_key` and an equal `Ref` (path, config, name) and `parent_human[key]` is true. `Planner.__init_subclass__` raises `TypeError` for `origin == "human"`. Add `Planner` to `swarm_arena.planning`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_planning tests/test_public_api.py tests/test_docstrings.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_planning src/swarm_arena/planning.py tests CHANGELOG.md
git commit -m "feat(planning): add revision diffing, core-set provenance, and the Planner ABC"
```

### Task 15: Planner-catalog check

**Files:**
- Create: `src/swarm_arena/_planning/catalog_check.py`, `tests/_fakes/import_canary.py` (R3.2 row: on import it writes the file named by env `SWARM_ARENA_TEST_CANARY` when set, and defines `class Tool`)
- Test: `tests/_planning/test_catalog_check.py`

**Interfaces:**
- Consumes: `RefSite`, `ref_sites`, `diff_revision`, `PlannerCatalog.match` (compares `Ref.path` to `entry.ref` as strings; never imports), `CatalogEntry.settable_model`.
- Produces (R4.13.12): `check_catalog(site: RefSite, catalog: PlannerCatalog, *, overrides: Mapping[str, Any]) -> list[str]` (compares `catalog_form(site config minus settable keys) == catalog_form(entry.config)`); `catalog_form(config: Mapping[str, Any]) -> dict[str, Any]` (two rules at any depth, on both sides, with no import: (1) a dumped `FileSet` `{"globs": g, "manifest": m}` becomes `g`; (2) a nested-ref dict — a dict with a `module:qualname` `path` and keys only from `{path, config, name}` — becomes `{"path": p, "config": catalog_form(config or {}), "name": name or None}`); `authorize_revision(plan: Plan, config: RunConfig, *, history: Sequence[Plan] = ()) -> tuple[Plan, list[str]]` (no import, no I/O; human origin → `(plan, [])`; non-human: `check_catalog` on every site with `carried_human` False; the trusted pairs are `(tuple(fs.globs), tuple(sorted(fs.manifest.items())))` of every dumped `FileSet` of every revision in `history` (all Ref sites, any depth); in each non-carried site, a `FileSet` whose pair is trusted keeps its manifest and every other `FileSet` becomes its globs list, so the core computes the manifest of every new or changed planner-written component and an unchanged one keeps its frozen files with no disk read). Every message starts with `site.pointer` and `": "`.

- [ ] **Step 1: Write the failing tests**

```python
CAT = PlannerCatalog(entries=[CatalogEntry(name="phys", kind="tool", ref="swarm_arena:ScriptTool", description="d",
    config={"command": ["python", "problem/run.py"], "editable": ["problem/*.phyk"]},
    settable={"timeout_s": {"type": "number", "maximum": 1200}})])

def test_matching_ref_passes() -> None:
    assert check_catalog(_site({"command": ["python", "problem/run.py"], "editable": ["problem/*.phyk"], "timeout_s": 600}), CAT, overrides={}) == []

def test_unknown_ref_rejected() -> None:
    assert check_catalog(_site({}, path="evil:Tool"), CAT, overrides={}) == [
        "/teams/0/agents/0/tool: evil:Tool is not in the planner catalog for kind 'tool'"]

def test_fixed_field_change_rejected() -> None:
    assert any("fixed field 'command'" in e for e in check_catalog(_site({"command": ["sh", "x"], "editable": ["problem/*.phyk"]}), CAT, overrides={}))

def test_settable_schema_enforced() -> None:
    assert any("timeout_s" in e for e in check_catalog(_site({**FIXED, "timeout_s": 5000}), CAT, overrides={}))

def test_override_of_fixed_field_rejected() -> None:
    assert any("override sets fixed field 'editable'" in e for e in check_catalog(_site(FIXED), CAT, overrides={"editable": ["**"]}))

def test_extra_unknown_field_rejected() -> None:
    assert any("field 'python'" in e for e in check_catalog(_site({**FIXED, "python": "/bin/x"}), CAT, overrides={}))

# CAT_FILES: one ScriptTool entry "runner" with config {"command": ["python", "problem/run.py"], "files": ["problem/run.py"]}
def test_snapshotted_site_matches_raw_entry() -> None:
    site = _site({"command": ["python", "problem/run.py"],
                  "files": {"globs": ["problem/run.py"], "manifest": {"problem/run.py": "ab" * 32}}})
    assert check_catalog(site, CAT_FILES, overrides={}) == []

def test_noncatalog_module_never_imported() -> None:
    r0 = _human_plan()
    r1 = stamp(_add_agent(r0, tool=Ref(path="_fakes.import_canary:Tool")), origin="autonomous", parent=r0)
    _, errors = authorize_revision(r1, _cfg(CAT_FILES), history=[r0])
    assert len(errors) == 1 and "not in the planner catalog" in errors[0]
    assert "_fakes.import_canary" not in sys.modules

def test_planner_written_manifest_dropped() -> None:
    r0 = _human_plan()
    r1 = stamp(_add_agent(r0, tool=_runner_ref(files={"globs": ["problem/run.py"], "manifest": {"problem/run.py": "ab" * 32}})),
               origin="autonomous", parent=r0)
    plan, errors = authorize_revision(r1, _cfg(CAT_FILES), history=[r0])
    assert errors == [] and plan.teams[-1].agents[-1].tool.config["files"] == ["problem/run.py"]

def test_committed_manifest_kept_for_unchanged_site() -> None:
    r0 = _human_plan()
    r1 = ...  # the admitted, autonomous r1 with _runner_ref(files={"globs": ["problem/run.py"], "manifest": {"problem/run.py": "ab" * 32}})
    r2 = stamp(r1, origin="autonomous", parent=r1)
    plan, errors = authorize_revision(r2, _cfg(CAT_FILES), history=[r0, r1])
    assert errors == [] and plan.teams[-1].agents[-1].tool.config["files"]["manifest"] == {"problem/run.py": "ab" * 32}
    changed = ...  # the same r2 with manifest {"problem/run.py": "cd" * 32}
    plan, _ = authorize_revision(changed, _cfg(CAT_FILES), history=[r0, r1])
    assert plan.teams[-1].agents[-1].tool.config["files"] == ["problem/run.py"]

# CAT_NESTED: policies "patience" (swarm_arena:Patience, {"rounds": 2}) and "chain" (swarm_arena:Chain,
# {"policies": [{"path": "swarm_arena:Patience", "config": {"rounds": 2}}]}); judges: two FixedJudge refs and "lex"
# (Lexicographic, {"judges": [the two FixedJudge dicts with config, no name]})
def test_nested_ref_defaults_match_after_snapshot() -> None:
    member = {"path": "swarm_arena:Patience", "config": {"rounds": 2}, "name": None}
    assert check_catalog(_site({"policies": [member]}, path="swarm_arena:Chain", kind="policy"), CAT_NESTED, overrides={}) == []
    assert check_catalog(_lex_site(name=None), CAT_NESTED, overrides={}) == []           # "name": None on each member
    assert catalog_form({"p": [{"path": "a.b:C"}]}) == catalog_form({"p": [{"path": "a.b:C", "config": {}, "name": None}]})
    bad = {**member, "config": {"rounds": 3}}
    assert any("fixed field 'policies'" in e
               for e in check_catalog(_site({"policies": [bad]}, path="swarm_arena:Chain", kind="policy"), CAT_NESTED, overrides={}))
    renamed = {**member, "name": "other"}
    assert check_catalog(_site({"policies": [renamed]}, path="swarm_arena:Chain", kind="policy"), CAT_NESTED, overrides={}) != []
```

`_lex_site(name)` is a test helper: the `Lexicographic` judge site of CAT_NESTED's `"lex"` entry with `"name": name` on each member dict.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_planning/test_catalog_check.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `check_catalog`: entry = `catalog.match(site.ref, site.kind)` (same `ref` path as strings; no import); `catalog_form` of the site config minus settable keys must equal `catalog_form(entry.config)` exactly; settable keys validate through `entry.settable_model()`; each override key must be a settable key and validate the same way. `catalog_form` applies its two rules (FileSet → globs; nested-ref dict → `{path, config: catalog_form(config or {}), name: name or None}`) recursively, on the site and the entry alike, with no import. `authorize_revision` builds `parent_human` from `history` as `validate_plan` does (Task 16), runs `check_catalog` on each non-carried site of a non-human revision, collects the trusted pairs `(tuple(fs.globs), tuple(sorted(fs.manifest.items())))` from every dumped `FileSet` of every `history` revision (all Ref sites, any depth), and returns a copy whose non-carried sites keep each `FileSet` with a trusted pair and have every other dumped `FileSet` replaced by its globs list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_planning/test_catalog_check.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_planning/catalog_check.py tests/_planning/test_catalog_check.py tests/_fakes/import_canary.py
git commit -m "feat(planning): check LLM-written refs and overrides against the planner catalog" \
  -m "authorize_revision runs on unbuilt refs, so a ref outside the catalog is never imported; catalog_form (FileSets and nested-ref defaults) lets a snapshotted or journaled site still equal its raw catalog entry, and a FileSet that a history revision already holds keeps its frozen manifest with no disk read."
```

### Task 16: `validate_plan`

**Files:**
- Create: `src/swarm_arena/_planning/validate.py`
- Modify: `src/swarm_arena/__init__.py`, `tests/test_public_api.py`, `CHANGELOG.md`
- Test: `tests/_planning/test_validate.py`

**Interfaces:**
- Consumes: `ref_sites`, `diff_revision`, `site_key`, `check_catalog`, `authorize_revision` (Task 15), `snapshot_plan`, `matches`, `ArtifactStore`, `Ref.build`, `PlanError`, `GoalKey.of`, `matches`, `Manager` and `Manager.models` (Task 9: manager refs validate their `Config` through `Ref.build(Manager)`, and `agent_fingerprint(spec, manager: Manager)` reads `manager.models()`, R4.13.12), `Tool.is_deterministic`.
- Produces: `validate_plan(plan, config, *, history=(), other_active_agents=0) -> list[str]` (R4.1, top-level export); `admit_revision(plan, config, artifacts: ArtifactStore, *, history=(), other_active_agents=0) -> tuple[Plan, list[str]]` (R4.13.12: the one core entry point for r0, replanner, and join-failure revisions; human: `snapshot_plan` → `validate_plan`; non-human: `authorize_revision` (errors → `PlanError` before any `Ref.load`) → `_resolve_revision` → `snapshot_plan` → `validate_plan`; returns the snapshotted revision and the warnings); `_resolve_revision(child: Plan, r0: Plan, artifacts: ArtifactStore) -> Plan` (private; `r0 = history[0]`, or `child` itself for r0: `base_dir` is set from r0; snapshotted `FileSet`s are kept; an unsnapshotted part-goal `FileSet` resolves only against the main goal's frozen manifest, `{p: sha for p, sha in main.files.manifest.items() if matches(p, globs)}`, never from disk, and a glob that matches no main-goal file → `PlanError(["part '<name>': glob '<g>' matches no file of the main goal"])`; any other unsnapshotted `FileSet` is snapshotted from r0's `base_dir`); `ModelCheck`, `register_model_check`, `agent_fingerprint` (R4.13.12); `NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")` (team, agent, part, and `Ref.name` slugs [plan]).

- [ ] **Step 1: Write the failing tests** (each builds a small plan with `_fakes.tools` refs; `errs(plan, cfg, **kw)` returns `PlanError.errors`)

```python
def test_valid_plan_passes() -> None:
    assert validate_plan(_plan(), RunConfig()) == []

@pytest.mark.parametrize("plan,needle", [
    (_plan(teams=5), "max_teams"), (_plan(agents_per_team=5), "max_agents_per_team"),
    (_plan(teams=4, agents_per_team=4), "max_agents_per_goal"), (_plan(rounds=4), "max_rounds")])
def test_limits(plan: Plan, needle: str) -> None:
    assert any(needle in e for e in errs(plan, RunConfig()))

def test_max_agents_total_counts_other_goals() -> None:
    assert any("max_agents_total" in e for e in errs(_plan(agents_per_team=3), RunConfig(), other_active_agents=30))
    peak3 = _plan(converge_agents=3)                       # a converge team of 3 agents: the plan's peak is 3
    assert "max_agents_total: peak 3 agents + 30 active in other goals > 32" in errs(peak3, RunConfig(), other_active_agents=30)

def test_bad_or_duplicate_names_rejected() -> None:
    # _renamed() edits names with model_copy(update=...), which skips the M1a model validators, as a replanner's copy can
    for bad in ["a/b", "A", "x y", "t:1", ""]:
        for where, pointer in [("team", "/teams/0/name"), ("agent", "/teams/0/agents/0/name"), ("part", "/parts/0/name")]:
            assert any(e.startswith(pointer) and "name" in e for e in errs(_renamed(_parts_plan(), **{where: bad}), RunConfig())), (where, bad)
    assert any("duplicate team" in e for e in errs(_renamed(_plan(teams=2), team_names=["t", "t"]), RunConfig()))
    assert any("duplicate agent" in e for e in errs(_renamed(_plan(agents_per_team=2), agent_names=["a", "a"]), RunConfig()))
    assert any("duplicate part" in e for e in errs(_renamed(_parts_plan(parts=["p", "q"]), part_names=["p", "p"]), RunConfig()))
    assert any("duplicate team" in e for e in errs(_plan(converge_team_name="t0"), RunConfig()))

def test_invalid_component_config_names_pointer() -> None:
    assert any(e.startswith("/teams/0/agents/0/tool") for e in errs(_plan(tool_config={"timeout_s": "soon"}), RunConfig()))

def test_effective_config_after_overrides_is_validated() -> None:
    assert any("/rounds/0" in e for e in errs(_plan(rounds=[RoundSpec(tool_overrides={"timeout_s": "soon"})]), RunConfig()))

def test_env_must_be_allowed() -> None:
    assert any("ANTHROPIC_API_KEY" in e for e in errs(_script_plan(env=["ANTHROPIC_API_KEY"]), RunConfig()))
    assert validate_plan(_script_plan(env=["ANTHROPIC_API_KEY"]), RunConfig(secrets=SecretsPolicy(allow=["ANTHROPIC_API_KEY"]))) == []

def test_rationale_whitespace_rejected() -> None:
    assert any(e.startswith("/teams/0/rationale") for e in errs(_renamed(_plan(), team_rationale="  "), RunConfig()))
    assert any(e.startswith("/teams/0/agents/0/rationale") for e in errs(_renamed(_plan(), agent_rationale="\t\n"), RunConfig()))

def test_part_without_team_and_unknown_part_rejected() -> None:
    assert any(e.startswith("/parts/1") and "part 'b' has no team" in e
               for e in errs(_parts_plan(parts=["a", "b"], team_parts=["a"]), RunConfig()))
    assert any(e.startswith("/teams/1/part") and "unknown part 'c'" in e
               for e in errs(_renamed(_parts_plan(parts=["a"], team_parts=["a", "a"]), team_part={1: "c"}), RunConfig()))

def test_select_converge_with_parts_rejected() -> None:
    assert any(e.startswith("/converge/mode") and "parts need converge mode 'team'" in e
               for e in errs(_parts_plan(converge=ConvergeSpec(mode="select")), RunConfig()))

def test_ranking_must_name_a_judge_and_names_unique() -> None:
    assert any(e.startswith("/ranking") and "'nope'" in e for e in errs(_plan(judges=[J_A, J_B], ranking="nope"), RunConfig()))
    assert any(e.startswith("/judges") and "duplicate" in e for e in errs(_plan(judges=[J_A, J_A], ranking="a"), RunConfig()))

def test_replicates_need_nondeterministic_tool() -> None:
    assert any(e.startswith("/teams/0/agents/0/replicates") and "deterministic" in e
               for e in errs(_plan(replicates=2, tool=FunctionTool("_fakes.tools:add")), RunConfig()))
    assert validate_plan(_plan(replicates=2, tool=FunctionTool("_fakes.tools:add", deterministic=False)), RunConfig()) == []

def test_frozen_main_goal_key() -> None:
    r0 = _plan(); r1 = stamp(_plan(goal_inputs={"x": 2}), origin="autonomous", parent=r0)
    assert any("main goal" in e for e in errs(r1, RunConfig(), history=[r0]))

def test_non_human_noncatalog_ref_rejected_and_carried_human_accepted() -> None:
    r0 = _script_judge_plan()                                            # human, ScriptJudge not in catalog
    ok = stamp(_add_catalog_agent(r0), origin="autonomous", parent=r0)
    bad = stamp(_add_agent(r0, path="_fakes.tools:EchoTool"), origin="autonomous", parent=r0)
    assert validate_plan(ok, CFG_WITH_CATALOG, history=[r0]) == []
    assert any("not in the planner catalog" in e for e in errs(bad, CFG_WITH_CATALOG, history=[r0]))

def test_noncatalog_ref_rejected_before_import() -> None:
    r0 = _script_judge_plan()
    bad = stamp(_add_agent(r0, path="_fakes.import_canary:Tool"), origin="autonomous", parent=r0)
    with pytest.raises(PlanError):
        validate_plan(bad, CFG_WITH_CATALOG, history=[r0])
    assert "_fakes.import_canary" not in sys.modules

@pytest.mark.parametrize("mutate", ["delete", "modify"])
def test_catalog_script_tool_valid_after_snapshot_and_in_later_revision(tmp_path: Path, artifacts: ArtifactStore,
                                                                         mutate: str) -> None:
    r0 = _script_judge_plan(base_dir=tmp_path)                           # tmp_path/problem/run.py exists
    r1, warnings = admit_revision(stamp(_add_catalog_agent(r0, files=["problem/run.py"]), origin="autonomous", parent=r0),
                                  CFG_WITH_CATALOG, artifacts, history=[r0])
    h = sha256_hex((tmp_path / "problem/run.py").read_bytes())
    assert r1.teams[-1].agents[-1].tool.config["files"]["manifest"] == {"problem/run.py": h}
    ...  # mutate: delete tmp_path/problem/run.py, or overwrite it with other bytes
    history = [Plan.model_validate_json(p.model_dump_json()) for p in (r0, r1)]   # as resume rebuilds it
    ...  # history[0] gets its journaled base_dir (plan_revision.base_dir)
    r2, _ = admit_revision(stamp(history[1], origin="autonomous", parent=history[1]), CFG_WITH_CATALOG, artifacts,
                           history=history)                              # no PlanError: delete proves no disk read
    assert r2.teams[-1].agents[-1].tool.config["files"]["manifest"] == {"problem/run.py": h}   # modify: frozen hash kept

def test_catalog_chain_and_lexicographic_admitted_and_roundtrip(tmp_path: Path, artifacts: ArtifactStore) -> None:
    # CFG_NESTED: the catalog Chain policy and Lexicographic judge (Task 15 CAT_NESTED); r0 writes each with nested defaults omitted
    r0 = stamp(_nested_plan(base_dir=tmp_path), origin="llm", parent=None)          # LLM origin: every site is checked
    a0, _ = admit_revision(r0, CFG_NESTED, artifacts)                    # no PlanError
    assert a0.policy.config["policies"][0]["name"] is None              # the snapshot dump happened
    j0 = Plan.model_validate_json(a0.model_dump_json())                 # a journal round trip
    a1, _ = admit_revision(stamp(j0, origin="autonomous", parent=j0), CFG_NESTED, artifacts, history=[j0])   # no PlanError
    assert a1.policy.config == a0.policy.config

def test_ever_present_checks_kept() -> None:
    ...  # r0 check c1; r1 (autonomous) adds catalog check c2; r2 drops c2 → "check 'c2'" error; r2 changing c1 → error

def test_new_part_must_leave_scoring_unset() -> None:
    r0 = _parts_plan(parts=["a"])
    bad = stamp(_with_part(r0, "b", judges=[CATALOG_JUDGE]), origin="autonomous", parent=r0)    # adds part b with a catalog team
    assert any(e.startswith("/parts/1/judges") and "new part" in e for e in errs(bad, CFG_WITH_CATALOG, history=[r0]))
    ok = stamp(_with_part(r0, "b"), origin="autonomous", parent=r0)
    assert validate_plan(ok, CFG_WITH_CATALOG, history=[r0]) == []

def test_duplicate_fingerprint_capped() -> None:
    plan = _plan(agents_per_team=3, same_tool=True, instructions=["a", "b", "c"])
    assert any("max_agents_per_fingerprint" in e for e in errs(plan, RunConfig()))

def test_address_not_reused() -> None:
    r0 = _plan(agent_names=["a", "b"]); r1 = stamp(_drop(r0, "t0/b"), origin="autonomous", parent=r0)
    r2 = stamp(r0, origin="autonomous", parent=r1)
    assert any("address t0/b" in e for e in errs(r2, RunConfig(), history=[r0, r1]))

def test_judge_file_in_goal_rejected() -> None:
    plan = _script_judge_plan(goal_manifest={"judge/score.py": "ab" * 32})       # also a ScriptJudge file
    assert any("judge/score.py" in e for e in errs(plan, RunConfig()))

def test_non_human_part_files_outside_main_manifest_rejected() -> None:
    r0 = _parts_plan(main_manifest={"problem/a.txt": "aa" * 32})
    r1 = stamp(_with_part_files(r0, {"judge/hidden/x.npz": "bb" * 32}), origin="autonomous", parent=r0)
    assert any("judge/hidden/x.npz" in e for e in errs(r1, RunConfig(), history=[r0]))

def test_all_errors_listed_and_model_check_hook() -> None:
    register_model_check(lambda plan, cfg: (["model error"], ["model warning"]))
    with pytest.raises(PlanError) as err:
        validate_plan(_plan(teams=5, rationale=" "), RunConfig())
    assert len(err.value.errors) == 3 and "model error" in err.value.errors
```

The hook test unregisters in a fixture (`_planning.validate._MODEL_CHECKS.clear()`). Module-level helpers in the test file: `_plan(**kw: Any) -> Plan` and `_parts_plan(**kw: Any) -> Plan` build valid plans (`FunctionTool("_fakes.tools:add")` tools, `_fakes.tools:NoopManager` managers, `J_A`/`J_B` = `FixedJudge` refs named `a`/`b`); `_renamed(plan: Plan, **changes: Any) -> Plan` applies name, rationale, and part changes with `model_copy(update=...)`, so the M1a model validators do not run first and `validate_plan` is what rejects them; `_with_part(plan, name, **part_fields) -> Plan` adds a part and one catalog team for it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_planning/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `validate_plan` as a list of private rule functions `_rule_<name>(plan, config, history, ...) -> list[str]`, one per bullet of §4.6 plus the R2 M1b deliverable 6 additions (stable addresses, judge-private files) and the name rule; it collects every error, raises `PlanError(errors)` when any exist, and returns the warnings of the model-check hooks. Rule order: for a non-human revision, `authorize_revision` (the catalog and provenance rule) runs first, on unbuilt `Ref`s, and its errors raise `PlanError` before any `Ref.build` (the other rules build refs to check `Config`s); so a rejected non-human site is never imported or constructed. `admit_revision` follows Produces. Provenance: `parent_human` is built by walking `history` (human-origin revisions mark all their sites human; later revisions carry flags through `diff_revision`). "Ever present": the union of plan-level check names over `history` and their first refs. The duplicate-fingerprint rule counts agents of this revision and of the converge team. The `max_agents_total` rule counts the revision's peak, `max(main-team agents, converge-team agents)` (the `GoalRunner.active_agents()` count, Task 24), plus `other_active_agents`, so a goal that passes it always fits an empty goal queue. It is `_rule_max_agents_total(plan, config, history, *, other_active_agents) -> list[str]` (resume calls it too, Task 27), and its one error is `"max_agents_total: peak <peak> agents + <other_active_agents> active in other goals > <limit>"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_planning tests/test_public_api.py tests/test_docstrings.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_planning/validate.py src/swarm_arena/__init__.py tests CHANGELOG.md
git commit -m "feat(planning): add validate_plan with every non-LLM rule of the spec" \
  -m "The catalog rule runs before any Ref.build, so a non-catalog ref is never imported. admit_revision is the one entry point that snapshots and validates every revision; part-goal files of a planner revision resolve only against the main goal's manifest. max_agents_total counts a plan's peak (its main teams or its converge team), the count the goal queue uses, so a valid goal always fits an empty queue."
```

### Task 17: Scoring runner

**Files:**
- Create: `src/swarm_arena/_scoring/runner.py`
- Modify: roadmap R4.13.11 (`final_judge` returns `tuple[list[Candidate], JudgeOutcome]`; its comment becomes `# one batch; read_cache=False; returns (order, outcome); scores stamped with ctx.ids`)
- Test: `tests/_scoring/test_runner.py`

**Interfaces:**
- Consumes: `Check`, `CheckContext`, `Judge`, `JudgeContext` (with `ids`, `errors`, and `on_started`, M1a R4.3 as edited), `stamp_scores` and `is_finite_score(score: Score) -> bool` (`_scoring/judge.py`, M1a R4.3), `ranked`, `Fenced`.
- Produces (R4.13.11 as edited here): `run_checks`, `judge_round`, `final_judge`; `JudgeError(FrozenModel)`: `candidate_id`, `judge`, `reason`; `JudgeOutcome(FrozenModel)`: `scores: list[Score]`, `errors: list[JudgeError]`; `for_manager(candidate: Candidate, feedback: Mapping[str, str]) -> Candidate` (drops `reasons` of checks whose feedback is `"pass_fail"`).
- Score ids [plan]: `judge_round` calls `ctx = ctx_for(judge)` once per judge (`ctx_for` returns a new context per call, so `ctx.errors` belongs to that judge) and stamps that judge's scores with `stamp_scores(scores, judge=judge, judge_name=ref.component_name(), ids=ctx.ids, batch_id=batch_id)`; `final_judge` does the same with its `ctx`. `ctx.ids` is the only id source for score stamping and command jobs (M1a R4.3), so a judge's score ids and its `ctx.execute` job ids come from one `IdGen`. The runner has no other id seam.
- Judge errors [plan]: for each judge call, `JudgeOutcome.errors` gets at most one `JudgeError(candidate_id, judge=ref.component_name(), reason)` per batch member, in batch order. The first rule that applies gives the reason: (1) the member has a `ctx.errors` entry: that reason (M1a `ProgramJudge`, `ScriptJudge`, and `Lexicographic` return no score for a failed candidate and put the reason only in `ctx.errors`); (2) the judge raised: `"<Type>: <message>"`, and the call gives no scores; (3) `is_finite_score` is False for its returned score: `"non-finite score"`, and the score is dropped; (4) the judge returned no score for it: `"no score"` (the omission case of M3 Task 12). The caller journals one `score` event per `JudgeOutcome.scores` entry and one `judge_error` event (payload `JudgeError.model_dump()`) per `JudgeOutcome.errors` entry (Task 21 `OfflineHarness`, Task 24 `GoalRunner`).
- `final_judge(goal_id, ranking, finalists, ctx) -> tuple[list[Candidate], JudgeOutcome]` (roadmap edit above): one call with every finalist; the list holds the finalists with a score, by key descending, ties by id; the outcome holds the stamped scores and the judge errors of that call (rules 1–4), which the goal journals with round 0. Reason for the edit: a `list[Candidate]` return cannot carry the final judge's scores or judge errors, and the goal must journal both.

- [ ] **Step 1: Write the failing tests**

```python
async def test_check_exception_is_a_failed_check() -> None:
    res = await run_checks(_cand(), [("ok", _PassCheck()), ("boom", _RaisingCheck())], _check_ctx())
    assert [r.passed for r in res] == [True, False] and res[1].reasons == ["check error: RuntimeError: boom"]

async def test_judge_round_skips_gated_out_and_adds_anchors() -> None:
    judge = _RecordingJudge(anchor_count=1)
    out = await judge_round("g", [(Ref.of(judge), judge)], [_cand("c-1"), _cand("c-2", failed=True)],
                            leaders=[_cand("c-0"), _cand("c-9")], ctx_for=lambda j: _judge_ctx())
    assert judge.seen == [["c-1", "c-0"]]
    assert {s.candidate_id for s in out.scores} == {"c-1", "c-0"} and out.errors == []

async def test_scores_are_stamped_with_ctx_ids() -> None:
    judge = _RecordingJudge()
    out = await judge_round("g", [(Ref.of(judge), judge)], [_cand("c-1")], [],
                            ctx_for=lambda j: _judge_ctx(ids=IdGen("run-1", 1)))
    assert out.scores[0].id == IdGen("run-1", 1).next("s")      # not the default IdGen(run_id, 0) of the context

async def test_ctx_errors_become_judge_errors() -> None:
    judge = ProgramJudge("_fakes.tools:raising_score")          # M1a: returns no score, reasons only in ctx.errors
    out = await judge_round("g", [(Ref.of(judge), judge)], [_cand("c-1"), _cand("c-2")], [], ctx_for=lambda j: _judge_ctx())
    assert out.scores == [] and [e.candidate_id for e in out.errors] == ["c-1", "c-2"]
    assert all("RuntimeError: no" in e.reason for e in out.errors)

async def test_missing_score_is_a_judge_error() -> None:      # M3 Task 12 adds test_omitted_candidate_is_a_judge_error
    judge = _RecordingJudge(values={"c-1": 1.0, "c-2": None})                  # None → no score
    out = await judge_round("g", [(Ref.of(judge), judge)], [_cand("c-1"), _cand("c-2")], [], ctx_for=lambda j: _judge_ctx())
    assert [s.candidate_id for s in out.scores] == ["c-1"] and [(e.candidate_id, e.reason) for e in out.errors] == [("c-2", "no score")]

async def test_non_finite_and_raising_judges_become_errors() -> None:
    out = await judge_round("g", [(Ref.of(_NanJudge()), _NanJudge()), (Ref.of(_BoomJudge()), _BoomJudge())], [_cand("c-1")], [],
                            ctx_for=lambda j: _judge_ctx())
    assert out.scores == [] and {(e.candidate_id, e.judge) for e in out.errors} == {("c-1", "nan"), ("c-1", "boom")}

async def test_final_judge_one_batch_no_cache_read_orders_by_key() -> None:
    judge = _RecordingJudge(values={"c-1": 1.0, "c-2": 3.0, "c-3": None})     # None → no score
    ctx = _judge_ctx(final=True, ids=IdGen("run-1", 1))
    order, out = await final_judge("g", (Ref.of(judge), judge), [_cand("c-1"), _cand("c-2"), _cand("c-3")], ctx)
    assert [c.id for c in order] == ["c-2", "c-1"] and len(judge.seen) == 1
    assert judge.last_ctx.final is True and judge.last_ctx.read_cache is False
    assert {s.candidate_id for s in out.scores} == {"c-1", "c-2"} and all(s.id.startswith("s-") for s in out.scores)
    assert [(e.candidate_id, e.reason) for e in out.errors] == [("c-3", "no score")]

async def test_final_judge_raising_judge_gives_no_order() -> None:
    order, out = await final_judge("g", (Ref.of(_BoomJudge()), _BoomJudge()), [_cand("c-1"), _cand("c-2")], _judge_ctx(final=True))
    assert order == [] and out.scores == [] and [e.candidate_id for e in out.errors] == ["c-1", "c-2"]

def test_for_manager_hides_reasons_of_pass_fail_checks() -> None:
    c = for_manager(_cand(checks=[CheckResult(check="hidden", passed=False, reasons=["x"])]), {"hidden": "pass_fail"})
    assert c.checks[0].reasons == [] and c.checks[0].passed is False

async def test_fenced_from_command_start_propagates(local_executor: LocalExecutor) -> None:
    async def fenced(_: Job.Started) -> None:
        raise Fenced("another owner")
    with pytest.raises(Fenced):                                   # not a failed CheckResult
        await run_checks(_cand(), [("s", _script_check())], _check_ctx(executor=local_executor, on_started=fenced))
    with pytest.raises(Fenced):                                   # not a judge_error
        await judge_round("g", [(Ref.of(sj := _script_judge()), sj)], [_cand("c-1")], [],
                          ctx_for=lambda j: _judge_ctx(executor=local_executor, on_started=fenced))
```

Test helpers (in `test_runner.py`): `_judge_ctx(**fields: Any) -> JudgeContext` = `JudgeContext(goal=GOAL, executor=RecordingExecutor(), artifact_root="/a", **fields)` (fields override the defaults), a new context per call; `_check_ctx(**fields)` likewise; `_script_check()` and `_script_judge()` are `ScriptCheck`/`ScriptJudge` instances on a one-line `python -c` command; `_RecordingJudge(*, values: Mapping[str, float | None] | None = None, anchor_count: int = 0)` records each batch's candidate ids in `seen` and its context in `last_ctx`, and returns `Score.maximize(v)` per candidate (`v = values.get(id, 1.0)`; `None` → no score).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_scoring/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `runner.py`. Checks run in plan order; each check's exception becomes `CheckResult(check=name, passed=False, reasons=["check error: <Type>: <msg>"])`, except `Fenced` (from a command job's `on_started`), which `run_checks`, `judge_round`, and `final_judge` re-raise, never a failed `CheckResult` or a `judge_error` (R4.13.11). `judge_round` runs each judge once over gated-in candidates plus `leaders[:judge.anchor_count()]` that are not already in the batch, with `batch_id = "<goal_id>-<judge name>-<first candidate id>"` and `ctx = ctx_for(judge)`; it catches any exception of `judge.judge`, stamps with `ids=ctx.ids`, and builds `JudgeOutcome.errors` by the judge-error rules 1–4 above. `final_judge` calls the ranking judge once with every finalist on `ctx`, with the same catch, stamping, and rules, and returns `(order, outcome)`. Then edit roadmap R4.13.11 as listed under Files.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_scoring -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_scoring/runner.py tests/_scoring/test_runner.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(scoring): run checks after tool runs, barrier judging with anchors, and the final judge" -m "Score ids come only from ctx.ids, and ctx.errors, a raising judge, and a missing score each become a JudgeError, so the goal journals judge_error for every unscored candidate. Roadmap R4.13.11: final_judge returns (order, JudgeOutcome), because the goal must journal the final judge's scores and errors and a list of candidates carries neither."
```

### Task 18: Agent runtime, repeat rules, watchdog

**Files:**
- Create: `src/swarm_arena/_agent/runtime.py`, `_agent/repeats.py`, `_agent/watchdog.py`
- Modify: roadmap R4.13.10 (`AgentRuntime` gains `holder: str` after `address`: the ledger holder that `execute_tool_run` reserves for and the agent's tranche uses; the agent address, or `"<goal>/:final"` for a converge-team agent, §8.1 final slice; `RepeatIndex.lookup` takes `resumed: AbstractSet[str]`; `RepeatIndex.next_replicate(rt, request, *, requested: int | None = None)`; each skipped when the roadmap has it)
- Test: `tests/_agent/test_runtime.py`, `tests/_agent/test_repeats.py`, `tests/_agent/test_watchdog.py`

**Interfaces:**
- Consumes: `RunStatus`, `GoalKey`, `ToolRequest`, `Ref`, `canonical_json`, `sha256_hex`, `Limits`, `Manager` and `LimitReached` (Task 9).
- Produces: `AgentCounters`, `AgentRuntime` (R4.13.10, plus `holder`); `RepeatOutcome`; `repeat_key(*, goal_key: GoalKey, tool_ref: Ref, request: ToolRequest, deterministic: bool) -> str`; `RepeatIndex(run_id: str, entries: Iterable[tuple[str, Candidate]] = ())` with `add(key: str, candidate: Candidate) -> None` (also appends to `added: list[tuple[str, Candidate]]`, the entries made this round), `key` (calls `repeat_key` with `rt.goal_key`, `rt.tool_ref`, and `rt.tool.is_deterministic(request)`), `lookup(key, *, agent, deterministic, resumed: AbstractSet[str] = frozenset())` (an own match returns `return_own` only when its candidate id is in `resumed`, else `refuse`), `next_replicate(rt: AgentRuntime, request: ToolRequest, *, requested: int | None = None) -> int` (non-deterministic tools; `requested` is the `replicate` argument of `run_tool`/`execute_tool_run`, and `request.replicate` is ignored; rule in Step 3), `release(rt: AgentRuntime, request: ToolRequest, index: int) -> None` (plan-private: frees a pending index); `COUNTED = frozenset({RunStatus.OK, RunStatus.ERROR})`; `Watchdog(limits: Limits)` with `record_run`, `record_refusal`, `end_round`, `reason: str | None`, `limit: str | None` (the `Limits` field that tripped: `"max_refused_repeats"`, `"max_consecutive_failures"`, or `"watchdog_idle_rounds"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/_agent/test_repeats.py
def test_key_is_canonical_and_covers_spec_fields(rt: AgentRuntime) -> None:
    idx = RepeatIndex("this-run")
    k = idx.key(rt, ToolRequest(inputs={"a": 1, "b": 2}))
    assert k == idx.key(rt, ToolRequest(inputs={"b": 2, "a": 1}))
    assert k != idx.key(rt, ToolRequest(inputs={"a": 1, "b": 2}, edits={"p.txt": "x"}))
    assert k != idx.key(rt, ToolRequest(inputs={"a": 1, "b": 2}, round_overrides={"epochs": 100}))
    assert k != idx.key(_rt_for_part(rt, "A"), ToolRequest(inputs={"a": 1, "b": 2}))

def test_nondeterministic_key_includes_replicate(rt_nd: AgentRuntime) -> None:
    idx = RepeatIndex("this-run")
    assert idx.key(rt_nd, ToolRequest(replicate=0)) != idx.key(rt_nd, ToolRequest(replicate=1))

# _c(...) builds a Candidate with run_id="this-run" unless run_id is given; agent="t/a" sets team "t", agent "a"
@pytest.mark.parametrize("earlier,agent,resumed,action", [
    (_c(RunStatus.OK, agent="t/a"), "t/a", frozenset(), "refuse"),
    (_c(RunStatus.OK, agent="t/a", id="c-own"), "t/a", frozenset({"c-own"}), "return_own"),
    (_c(RunStatus.OK, agent="t/a", id="c-old"), "t/a", frozenset({"c-other"}), "refuse"),
    (_c(RunStatus.ERROR, agent="t/b"), "t/a", frozenset(), "link"),
    (_c(RunStatus.TIMEOUT, agent="t/a"), "t/a", frozenset(), "run"),
    (_c(RunStatus.OK, agent="t/a", partial=True), "t/a", frozenset(), "run"),
    (_c(RunStatus.OK, agent="t/b", run_id="past-run"), "t/a", frozenset(), "run")])
def test_lookup_rules(earlier: Candidate, agent: str, resumed: frozenset[str], action: str) -> None:
    out = RepeatIndex("this-run", [("k", earlier)]).lookup("k", agent=agent, deterministic=True, resumed=resumed)
    assert out.action == action
    if earlier.run_id == "past-run":
        assert out.hint == earlier

def test_nondeterministic_always_runs() -> None:
    idx = RepeatIndex("this-run", [("k", _c(RunStatus.OK, agent="t/a"))])
    assert idx.lookup("k", agent="t/a", deterministic=False).action == "run"

def test_add_records_own_entries() -> None:
    idx = RepeatIndex("this-run")
    c = _c(RunStatus.OK, agent="t/a")
    idx.add("k2", c)
    assert idx.added == [("k2", c)] and idx.lookup("k2", agent="t/a", deterministic=True).action == "refuse"

def test_next_replicate_in_order_and_exhaustion(rt_nd2: AgentRuntime) -> None:   # spec.replicates == 2
    idx, req = RepeatIndex("this-run"), ToolRequest(inputs={"x": 1.0})
    assert idx.next_replicate(rt_nd2, req) == 0
    idx.add(*_entry(idx, rt_nd2, req, replicate=0))
    assert idx.next_replicate(rt_nd2, req) == 1
    idx.add(*_entry(idx, rt_nd2, req, replicate=1))
    with pytest.raises(SoftError, match="replicates used up"):
        idx.next_replicate(rt_nd2, req)

def test_explicit_replicate_out_of_range_or_used_rejected(rt_nd2: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest(inputs={"x": 1.0})
    with pytest.raises(SoftError):
        idx.next_replicate(rt_nd2, req, requested=2)
    idx.add(*_entry(idx, rt_nd2, req, replicate=0))
    with pytest.raises(SoftError):
        idx.next_replicate(rt_nd2, req, requested=0)
    assert idx.next_replicate(rt_nd2, req, requested=1) == 1

def test_next_replicate_counts_journal_loaded_and_earlier_round_entries(rt_nd2: AgentRuntime) -> None:
    idx = RepeatIndex("this-run")
    live = ToolRequest(inputs={"x": 1.0}, edits={"p.txt": "x"}, round=2)
    loaded = ToolRequest.model_validate({**live.model_dump(mode="json"), "round": 1})   # edits as "sha256:<hex>"
    idx.add(*_entry(idx, rt_nd2, loaded, replicate=0))
    assert idx.next_replicate(rt_nd2, live) == 1

def test_next_replicate_counts_loaded_request_with_edits(rt_nd2: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest(inputs={"x": 1.0}, edits={"p/a.txt": "new"})
    loaded = ToolRequest.model_validate(req.model_copy(update={"replicate": 0}).model_dump(mode="json"))
    idx.add(*_entry(idx, rt_nd2, loaded, replicate=0))                   # an own entry whose candidate.request is loaded
    assert idx.next_replicate(rt_nd2, req) == 1
```

`_entry(idx, rt, request, *, replicate)` returns `(key, candidate)` for an own OK candidate of `rt` whose `request` is `request` with that replicate; `rt_nd2` is `rt_nd` with `spec.replicates = 2`. `_entry` calls `make_candidate(..., request=..., overlay={p: v.removeprefix("sha256:") for p, v in request.edits.items()})`: with `overlay` given, `make_candidate` skips `edit_hashes` and `put_edits`, which refuse the stored-reference form `sha256:<64 hex>` of a journal-loaded request (M1a Task 23 `test_make_candidate_refuses_stored_reference_edit`); without it, the two loaded-request tests raise `ValueError` before they can test anything.

```python
# tests/_agent/test_watchdog.py
def test_refused_repeats_over_limit_stop() -> None:
    w = Watchdog(Limits())
    for _ in range(3): w.record_refusal()
    assert w.reason is None
    w.record_refusal(); assert (w.reason, w.limit) == ("refused a repeated request more than 3 times", "max_refused_repeats")

def test_consecutive_failures_count_error_oom_timeout_only() -> None:
    w = Watchdog(Limits())
    for s in [RunStatus.ERROR, RunStatus.PROVIDER_FAILED, RunStatus.LOST, RunStatus.CANCELLED, RunStatus.BUDGET_EXCEEDED, RunStatus.OOM]:
        w.record_run(s, passed=False)
    assert w.reason is None
    w.record_run(RunStatus.TIMEOUT, passed=False); assert (w.reason, w.limit) == ("3 consecutive failures", "max_consecutive_failures")

def test_ok_resets_failures() -> None:
    w = Watchdog(Limits())
    w.record_run(RunStatus.ERROR, passed=False); w.record_run(RunStatus.TIMEOUT, passed=False)
    w.record_run(RunStatus.OK, passed=False)                                  # reset to 0
    w.record_run(RunStatus.OOM, passed=False); w.record_run(RunStatus.CANCELLED, passed=False); w.record_run(RunStatus.ERROR, passed=False)
    assert w.reason is None                                                   # 2 failures since the ok; cancelled neither counts nor resets
    w.record_run(RunStatus.ERROR, passed=False); assert w.reason == "3 consecutive failures"

def test_idle_rounds_skip_starved_rounds() -> None:
    w = Watchdog(Limits())
    w.record_run(RunStatus.OK, passed=False); assert w.end_round(ran=True) is None
    assert w.end_round(ran=False) is None
    w.record_run(RunStatus.OK, passed=False); assert w.end_round(ran=True) == "no new passing candidate in 2 rounds"
```

`tests/_agent/test_runtime.py`: `AgentCounters()` starts at zero; `AgentRuntime` builds with keyword fields only, and `holder` is a required field.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_agent/test_runtime.py tests/_agent/test_repeats.py tests/_agent/test_watchdog.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** `repeat_key`: `sha256_hex(canonical_json({goal_key, tool_ref.path, tool_ref.config, request.round_overrides, request.inputs, request.edits as path → sha256, parent ids, request.replicate if not deterministic}))`. The edit hashes come from `request.model_dump(mode="json")["edits"]` with the `"sha256:"` prefix removed, never from `edit_hashes` (which refuses the reserved reference form, so it cannot key a journal-loaded request); a live request and a journal-loaded one give the same key. `lookup`: among entries with this key and `status in COUNTED and not partial`, an entry with `run_id == self.run_id` by `agent` (`"<team>/<agent>"`) → `refuse` (`return_own` when its candidate id is in `resumed`); with `run_id == self.run_id` by another agent → `link`; with another `run_id` → `run` with `hint`. Which candidates are entries is the caller's choice (Task 22: earlier rounds of this run plus the agent's own items), so a same-round result of another agent is never linked. `next_replicate(rt, request, *, requested=None)` (`request.replicate` is ignored): index `i` is used when an own entry (this run, this agent, status in `COUNTED`, not partial) has key `== self.key(rt, request.model_copy(update={"replicate": i}))`, or when that key is pending. No `candidate.request` equality: the repeat key leaves out `round` and normalizes edits to hashes, so earlier-round and journal-loaded own candidates count. On resume the entries already hold the agent's committed candidates (earlier rounds through `repeat_entries`, the current round through its own resumed entries), so those count before any new index is given. `requested is None` → the smallest unused index `< rt.spec.replicates`; an explicit `requested` `>= replicates` or already used → `SoftError`; none left → the refusal path (`rt.watchdog.record_refusal()`, an ordinary `watchdog {"refused": ...}` event, then `LimitReached("max_refused_repeats", ...)` if the watchdog tripped, else `SoftError("repeat refused: replicates used up (<n> of <n>)")`). The returned index stays pending until `add()` of that candidate or `release()` (when `execute_tool_run` raises). `Watchdog` uses the three limits; statuses outside error, oom, timeout neither count nor reset; ok resets.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_agent -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_agent tests/_agent docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(agent): add the agent runtime, repeat rules, and the watchdog" -m "Roadmap R4.13.10: AgentRuntime gains holder, because a converge-team agent must charge the goal's final slice (spec 8.1), not its own address. RepeatIndex.lookup takes the set of resumed candidate ids, so only a run that was really resumed returns the agent's own result, and next_replicate(rt, request, *, requested=None) allocates replicate indices by repeat key (requested is the caller's replicate argument), so replicates bounds the identical runs of a non-deterministic tool, earlier-round and journal-loaded candidates included (each edit skipped when the roadmap has it)."
```

### Task 19: The `run_tool` pipeline

**Files:**
- Create: `src/swarm_arena/_agent/tool_run.py`, `tests/_fakes/engine.py` (`RtEnv`, `take_over_retrying`), `tests/_agent/conftest.py` (fixture `rt_env`)
- Modify: `src/swarm_arena/_journal/replay.py` (fold `tool_end.repeat_key` into `GoalState.repeat_keys`), `tests/_journal/test_replay.py`, `tests/_fakes/tools.py` (add `square(x: float) -> dict[str, float]` = `{"x": x, "y": x * x}`; `touch(marker: str) -> dict[str, str]` (creates the file `marker`, returns `{"marker": marker}`); `class CommandRaisesTool(PyTool)` whose `command(self, request: ToolRequest, workspace: Path) -> Command` raises `RuntimeError("boom")`; `WRITE_THEN_SLEEP: str` = `WRITE.format("{'data': {'step': 1}}") + "; import time; time.sleep(600)"`, `PyTool` code that writes a partial `swarm/result.json` and never ends on its own during a test; `write_then_sleep(marker: Path, seconds: float = 600) -> str`, `PyTool` code that writes the partial `swarm/result.json`, then creates `marker`, then sleeps `seconds`, so a test cancels only after the partial output exists), roadmap R4.13.10 (the `execute_tool_run` comment becomes: "validate inputs, edits (validate_edits), and parents → repeat rules and replicate allocation → count the tool run → reserve (durable) → put_edits → stage → Job(on_started = durable start event) → executor.submit (refused when the round's admission is closed: at Inflight.add and again in on_started; a refused run settles Amount() and raises LimitReached('stop')) → logs → checks → durable completion event → settle (or keep charged) → board.put_candidate"; plus the line "`AgentContext.run_tool` applies the gate and counts the turn, then calls `execute_tool_run`, which owns every other check"; `AgentRuntime` gains `private_paths: frozenset[str] = frozenset()`; each skipped when the roadmap has it), roadmap R3.2 (two new rows, because these test files mirror no module: `tests/_fakes/engine.py` | Test-side env types and helpers for agent and engine tests (`RtEnv`, `CtxEnv`, `EngineEnv`, the config helpers `with_limits` and `with_executor`, `take_over_retrying`); importable as `_fakes.engine`, never collected | M1b (+M2); and `tests/_agent/conftest.py` | Agent fixtures `rt_env` and `ctx_env` | M1b (+M2). M2 Task 9 changes their board to `ArenaBoard`)
- Test: `tests/_agent/test_tool_run.py`

**Interfaces:**
- Consumes: `AgentRuntime` (with `holder`), `RepeatIndex`, `Watchdog`, `Ledger.reserve/settle`, `JournalWriter.append/append_durable`, `plan_tool_stage`, `validate_edits`, `edit_hashes`, `put_edits`, `Tool._protected` (M1a), `CheckContext` with `on_started` (M1a R4.3), `Job` with `on_started` and `on_partial` (M1a), `Executor.submit`, `effective_tool_timeout` and `tool_grace` (M1a `_tooling/tool.py`), `run_checks`, `tool_run_task`, `current_requesting_call`, `ArtifactStore.get_bytes`, `Store.create`, `SoftError`, `LimitReached`, `Fenced`.
- Produces: `execute_tool_run` (R4.13.10), the single owner of every per-request check. Before its first `await` it does, with no `await` in between (Review Focus 1): (1) validate `inputs` with `tool.get_input_model()`, `edits` with `validate_edits(edits, editable=tool.editable(), protected=tool._protected(root) | rt.private_paths, max_edit_bytes=limits.max_edit_bytes)` (M1a), and `parents` (each id resolves through `rt.view` or `rt.own`; a past-run parent is allowed, §4.9) → `SoftError`; (2) for a non-deterministic tool, `index = rt.repeats.next_replicate(rt, request, requested=replicate)` and the request is rebuilt with that replicate (its refusal path as in Task 18); then the repeat rules (`rt.repeats.lookup(key, ..., resumed={c.id for c in rt.resumed})`): `refuse` → `rt.watchdog.record_refusal()` and an ordinary `watchdog {"refused": <own candidate id>}` event, then `LimitReached("max_refused_repeats", ...)` if the watchdog tripped, else `SoftError("repeat refused: ...")`; `return_own` and `link` → return that candidate, no run counted, nothing reserved; `run` (with or without `hint`) → go on; (3) count the tool run: `rt.counters.tool_runs >= limits.max_tool_runs_per_round` → `LimitReached("max_tool_runs_per_round", ...)`, else increment. Then it reserves on `rt.holder` and does the rest of R4.13.10. If it raises after (2), the pending replicate index is released. `AgentRuntime.private_paths` (R4.13.10 edit): the revision's check and judge file paths; `RoundRunner` sets it once per round. Also `Inflight` (plan-private registry in `tool_run.py`: `add(scope: anyio.CancelScope) -> None`, `discard(scope) -> None`, `close() -> None`, `cancel_all() -> None`, `closed: bool`; `add` on a closed registry raises `LimitReached("stop", "the round is stopping")`) and `track_inflight(registry: Inflight) -> AbstractContextManager[None]` (a private context variable: the round runner enters it around each manager task; `execute_tool_run` adds its `run_scope` right before `submit`, with no `await` between the `closed` check and the `add`, and discards it after) [plan]. A refused `add` settles the reservation with `Amount()` and dispatches nothing. Admission is checked again at the executor's start boundary: the job's `on_started` callback first checks `inflight.closed`, and when it is closed raises `LimitReached("stop", "the round is stopping")` before it appends `tool_start`, with no `await` between the check and the append. The executor kills the launcher before the tool's argv runs (M1a supervisor step 2; M4 `RayExecutor` submits nothing) and re-raises, and `execute_tool_run` settles the reservation with `Amount()` and re-raises: never an error candidate, and the pending replicate index is released. So a job queued in `submit` (for example behind `LocalExecutor`'s slot) when stop or the run budget closed admission never starts. The job gets `timeout_s = effective_tool_timeout(rt.tool, round_timeout_s=rt.round_spec.tool_timeout_s, limit_s=rt.limits.tool_timeout_s)` and `grace_s = tool_grace(rt.tool)` (M1a).
- Test helpers in `tests/_fakes/engine.py`: `take_over_retrying(store: Store, *, owner_id: str = "intruder") -> Lease` (`Lease.take_over(force=True)`, retried on `Fenced` while the epoch is unchanged, §7.1 step 1); `RtEnv` (dataclass: `rt: AgentRuntime`, `run_store: LocalStore`, `board: RunBoard`, `marker: Path`, `spy: RecordingExecutor`, `result: Candidate | None`; methods `with_tool(tool: str | Tool) -> AgentRuntime` (a `str` is a `Tool` class path), `rt_with_spy_executor(**rt_fields: Any) -> AgentRuntime`, `rt_with_checks(checks: Sequence[tuple[str, Check]]) -> AgentRuntime`, `rt_for_agent(agent: str, *, entries: Sequence[tuple[str, Candidate]]) -> AgentRuntime`, `events(type: str) -> list[Event]`, `types_after_start() -> list[str]`, `committed_types() -> list[str]`, `reservation_state(c: Candidate | str) -> str`, `charged(c: Candidate | str) -> Amount` (a `str` is a reservation id), `with_script_tool(*, editable: Sequence[str], **rt_fields: Any) -> AgentRuntime` (an agent with `ScriptTool(command=["python", "problem/run.py"], editable=...)` whose goal files `problem/a.phyk` = `"a = 1\n"` and `problem/run.py` (writes `swarm/result.json` `{"data": {}}`) are put in `artifacts`), `fence_on_append(type: EventType) -> None` (wraps `journal.append_durable` so that a draft of that type first calls `take_over_retrying(run_store)`), `after_append(type: EventType, fn: Callable[[], None]) -> None` (calls `fn` right after a draft of that type commits), `inflight: Inflight` (the registry the runtime's calls run under), `async wait_for(path: Path) -> None` (polls until the file exists), `async wait_started() -> None`, `cancel_inflight() -> None`). Fixture `rt_env(tmp_path: Path, run_config: RunConfig, artifacts: ArtifactStore) -> RtEnv` in `tests/_agent/conftest.py`: a real `JournalWriter`, `Ledger` (max_usd 100), `RunBoard`, `LocalExecutor`, and `FunctionTool("_fakes.tools:square")`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_happy_path_event_order_and_logs(rt_env: RtEnv) -> None:
    cand = await execute_tool_run(rt_env.rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert rt_env.types_after_start() == ["reserve", "tool_start", "tool_end", "settle"]
    assert cand.output.status == RunStatus.OK and cand.tool_identity and cand.fidelity == rt_env.rt.fidelity
    assert cand.output.logs_ref == f"goals/g/logs/{cand.id}"
    assert rt_env.run_store.exists(f"goals/g/logs/{cand.id}.out") and rt_env.run_store.exists(f"goals/g/logs/{cand.id}.err")
    assert rt_env.board.get_candidate(cand.id) == cand

async def test_checks_run_before_any_reserve(rt_env: RtEnv) -> None:
    with pytest.raises(SoftError):
        await execute_tool_run(rt_env.rt, {"x": "nope"}, parents=(), replicate=None, edits=None)
    with pytest.raises(SoftError, match="c-unknown"):
        await execute_tool_run(rt_env.rt, {"x": 1.0}, parents=["c-unknown"], replicate=None, edits=None)
    assert "reserve" not in rt_env.committed_types() and rt_env.rt.counters.tool_runs == 0

async def test_link_counts_no_run_and_reserves_nothing(rt_env: RtEnv) -> None:
    first = await execute_tool_run(rt_env.rt, {"x": 2.0}, parents=(), replicate=None, edits=None)
    rt_b = rt_env.rt_for_agent("b", entries=rt_env.rt.repeats.added)        # agent a's entry, as from an earlier round
    assert await execute_tool_run(rt_b, {"x": 2.0}, parents=(), replicate=None, edits=None) == first
    assert rt_b.counters.tool_runs == 0 and rt_env.committed_types().count("reserve") == 1

async def test_tool_end_journals_repeat_key(rt_env: RtEnv) -> None:
    cand = await execute_tool_run(rt_env.rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    key = rt_env.rt.repeats.key(rt_env.rt, cand.request)
    assert rt_env.events("tool_end")[-1].payload["repeat_key"] == key and rt_env.rt.repeats.added == [(key, cand)]

async def test_dispatch_after_reserve_commit(rt_env: RtEnv) -> None:
    ...  # spy executor records read_owner(store).head.seq at submit; it is >= the reserve event's seq

async def test_tool_never_runs_when_start_commit_fails(rt_env: RtEnv) -> None:
    rt_env.fence_on_append(EventType.TOOL_START)          # the reserve commit succeeds, then the lease is taken over
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:touch"))
    with pytest.raises(Fenced):
        await execute_tool_run(rt, {"marker": str(rt_env.marker)}, parents=(), replicate=None, edits=None)
    committed = rt_env.committed_types()
    assert "reserve" in committed and "tool_start" not in committed
    assert not rt_env.marker.exists()

async def test_fenced_at_tool_end_leaves_reservation_unsettled(rt_env: RtEnv) -> None:
    rt_env.fence_on_append(EventType.TOOL_END)
    with pytest.raises(Fenced):
        await execute_tool_run(rt_env.rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    committed = rt_env.committed_types()
    assert "reserve" in committed and "tool_start" in committed
    assert "tool_end" not in committed and "settle" not in committed      # replay charges it in full (§8.1)

async def test_closed_admission_refuses_dispatch(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:touch"))
    rt_env.after_append(EventType.RESERVE, rt_env.inflight.close)          # closed right after the reserve commit
    with pytest.raises(LimitReached) as err:
        await execute_tool_run(rt, {"marker": str(rt_env.marker)}, parents=(), replicate=None, edits=None)
    assert err.value.limit == "stop" and not rt_env.marker.exists() and "tool_start" not in rt_env.committed_types()
    res = rt_env.events("reserve")[-1].payload["reservation"]["id"]
    assert rt_env.reservation_state(res) == "settled" and rt_env.charged(res) == Amount()

async def test_admission_closed_while_queued_refuses_start(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:touch"))
    queued = _QueuedExecutor(rt.executor)           # test-local: submit sets `entered`, awaits `release`, then delegates
    rt = dataclasses.replace(rt, executor=queued)
    errors: list[LimitReached] = []                 # _run_capturing appends the LimitReached that execute_tool_run raises
    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_capturing, rt, {"marker": str(rt_env.marker)}, errors)
        await queued.entered.wait(); rt_env.inflight.close(); queued.release.set()
    err = errors[0]
    assert err.limit == "stop" and not rt_env.marker.exists() and "tool_start" not in rt_env.committed_types()
    res = rt_env.events("reserve")[-1].payload["reservation"]["id"]
    assert rt_env.reservation_state(res) == "settled" and rt_env.charged(res) == Amount()

async def test_replicates_one_refuses_second_identical_request(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", deterministic=False))   # spec.replicates == 1
    await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    with pytest.raises(SoftError, match="replicates used up"):
        await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert rt.counters.tool_runs == 1 and rt_env.committed_types().count("tool_start") == 1

async def test_edits_stored_before_dispatch(rt_env: RtEnv) -> None:
    rt = rt_env.with_script_tool(editable=["problem/*.phyk"])
    cand = await execute_tool_run(rt, {}, parents=(), replicate=None, edits={"problem/a.phyk": "a = 2\n"})
    sha = sha256_hex(b"a = 2\n")
    assert cand.output.status == RunStatus.OK and cand.overlay == {"problem/a.phyk": sha}
    assert rt.artifacts.get_bytes(sha) == b"a = 2\n"

async def test_judge_file_edit_refused(rt_env: RtEnv) -> None:
    rt = rt_env.with_script_tool(editable=["**"], private_paths=frozenset({"judge/score.py"}))
    with pytest.raises(SoftError):
        await execute_tool_run(rt, {}, parents=(), replicate=None, edits={"judge/score.py": "x"})
    assert rt.counters.tool_runs == 0

async def test_executor_exception_becomes_error_candidate(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(CommandRaisesTool({"code": ""}))         # command() raises RuntimeError("boom") inside submit
    cand = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    assert cand.output.status == RunStatus.ERROR and cand.warnings == ["tool error: RuntimeError: boom"]
    assert rt_env.reservation_state(cand) == "settled"
    assert cand.output.logs_ref is None and not rt_env.run_store.exists(f"goals/g/logs/{cand.id}.out")

async def test_round_cancel_keeps_partial_output(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(PyTool({"code": write_then_sleep(rt_env.marker)}))   # result.json, then marker, then 600 s
    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_and_store, rt, rt_env)                  # stores the returned candidate in rt_env.result
            await rt_env.wait_for(rt_env.marker); rt_env.cancel_inflight()
    cand = rt_env.result
    assert cand.output.status == RunStatus.CANCELLED and cand.output.partial is True and cand.output.data == {"step": 1}
    assert rt_env.reservation_state(cand) == "settled" and "tool_end" in rt_env.committed_types()

async def test_check_commands_journal_command_start_and_end(rt_env: RtEnv) -> None:
    rt = rt_env.rt_with_checks([("sc", ScriptCheck(command=["python", "-c", "print(1)"]))])
    cand = await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    start, end = rt_env.events("command_start")[-1], rt_env.events("command_end")[-1]
    assert (start.payload["candidate_id"], start.payload["component"]) == (cand.id, "sc")
    assert end.payload["job_id"] == start.payload["job_id"] and start.seq < end.seq

async def test_providers_without_usage_settle_estimated(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(FunctionTool("_fakes.tools:square", providers=["openai"]))
    cand = await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert rt_env.reservation_state(cand) == "estimated"

async def test_reported_usage_settles_actual(rt_env: RtEnv) -> None:
    rt = rt_env.with_tool(PyTool({"code": WRITE.format("{'data': {}, 'usage': {'usd': 0.2}}")}))
    cand = await execute_tool_run(rt, {}, parents=(), replicate=None, edits=None)
    assert rt_env.charged(cand) == Amount(usd=0.2)

async def test_requested_by_call_and_swarm_env(rt_env: RtEnv) -> None:
    with requesting_call("llm-x"):
        cand = await execute_tool_run(rt_env.rt_with_spy_executor(), {"x": 1.0}, parents=(), replicate=None, edits=None)
    assert cand.requested_by_call == "llm-x"
    assert {"SWARM_ARENA_RUN_ID", "SWARM_ARENA_GOAL_ID", "SWARM_ARENA_CANDIDATE_ID", "SWARM_ARENA_TEAM",
            "SWARM_ARENA_AGENT", "SWARM_ARENA_ROUND"} <= set(rt_env.spy.jobs[-1].env)

async def test_job_uses_m1a_timeout_and_partial_hook(rt_env: RtEnv) -> None:
    rt = rt_env.rt_with_spy_executor(round_spec=RoundSpec(tool_timeout_s=30))
    await execute_tool_run(rt, {"x": 1.0}, parents=(), replicate=None, edits=None)
    job = rt_env.spy.jobs[-1]
    assert job.timeout_s == effective_tool_timeout(rt.tool, round_timeout_s=30, limit_s=3600.0) == 30
    assert job.grace_s == tool_grace(rt.tool) and job.on_started is not None and job.on_partial is not None
```

```python
# tests/_journal/test_replay.py (one new test)
def test_repeat_keys_folded() -> None:
    state = RunState.from_events([..., _end(3, "c-1", goal_id="g", repeat_key="k1")])
    assert state.goals["g"].repeat_keys == {"c-1": "k1"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_agent/test_tool_run.py tests/_journal/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._agent.tool_run'` and a `test_repeat_keys_folded` assertion error.

- [ ] **Step 3: Implement** `execute_tool_run` in the order of R4.13.10 as edited above. Cancellation design [plan]:

```text
checks (1)-(3), no await                                   # SoftError / LimitReached / return_own / link; replicate index
res = await rt.ledger.reserve(rt.holder, estimate or default_run_reservation, kind="tool", goal_id=...)   # durable
await anyio.to_thread.run_sync(put_edits, edits, rt.artifacts)
stage, overlay, warnings = plan_tool_stage(..., protected=tool._protected(root) | rt.private_paths,
                                           private_paths=rt.private_paths)
partial: list[ToolOutput] = []
job = Job(..., on_started=<inflight.closed → LimitReached("stop"); else append_durable tool_start>,   # no await between
          on_partial=<async def: partial.append(output)>)
with anyio.CancelScope() as run_scope:
    inflight.add(run_scope)            # no await since the closed check; closed → settle(res, Amount()), LimitReached("stop")
    try: output = await rt.executor.submit(job)
    except Fenced: raise                                   # from on_started: the run never started; never an error candidate
    except LimitReached: await rt.ledger.settle(res.id, Amount()); raise   # from on_started: closed while queued
    except Exception as exc: output = ToolOutput(status=ERROR); warnings.append(f"tool error: {type(exc).__name__}: {exc}")
    finally: inflight.discard(run_scope)
if run_scope.cancelled_caught: output = partial[0] if partial else ToolOutput(status=CANCELLED)
with anyio.CancelScope(shield=True):   # logs → checks → watchdog.record_run → durable tool_end → settle (or keep charged)
    ...                                # → board.put_candidate → rt.own.append → rt.repeats.add(key, candidate)
```

The order in the shielded block is the §8.1 replay rule's other half: a run with no committed `tool_end` never has a settle, so the one replay rule (no committed settle → fully spent) keeps every lost run fully charged; a crash after `tool_end` and before the settle keeps the candidate and charges the reservation in full. Check commands: every `CheckContext` the pipeline builds gets `on_started` = a callback that `append_durable`s `command_start {job_id, started, candidate_id, component}` (`component` = the check name), and the core appends an ordinary `command_end {job_id, status}` after that job's execute returns; both come from the plan-private helper `command_journal(journal: JournalWriter, executor: Executor, *, candidate_id: str, component: str) -> tuple[Executor, Callable[[Job.Started], Awaitable[None]]]` in `tool_run.py` (an executor wrapper that journals `command_end`, and the `on_started` callback), which Tasks 21 and 24 reuse for judge contexts. The round runner cancels `run_scope` (round timeout, `stop --now`), so the candidate is kept. A cancel of an outer scope (the fence watcher of Task 26) propagates after the executor's shielded stop sequence (grace, partial collect, `on_partial`); nothing more is committed then. `Candidate.warnings` = the soft warnings of `plan_tool_stage` (a base parent not overlaid, §4.12) followed by the tool error text, if any. `Fenced` from `on_started` propagates (the supervisor kills the launcher first). Settle rule [plan]: `tool.providers()` non-empty and the output reports no usage (`usd is None` and `total_tokens == 0`) → `settle(..., estimated=True)`; else settle `Amount(tokens=usage.total_tokens, usd=usage.usd or 0.0)`. `tool_end` payload: `{"candidate": ..., "a2a": task_to_json(tool_run_task(...)), "repeat_key": key}`; `task_id` is the candidate id. Logs: read both blobs from `artifact_root` by the `logs_ref` hashes and `create` them under `goals/<goal-id>/logs/`; an output with `logs_ref` `None` (the synthetic error or cancelled output) writes no log files and keeps `logs_ref` `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_agent tests/_journal -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_agent/tool_run.py src/swarm_arena/_journal/replay.py tests/_agent tests/_journal/test_replay.py tests/_fakes docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(agent): add the run_tool pipeline with durable reservations, partial output on cancel, and error candidates" -m "execute_tool_run owns every request check, so the run count comes after the repeat lookup and before the reserve; roadmap R4.13.10 states this order. The settle comes after the durable tool_end, so a run without a committed completion never has a settle and replay charges it in full (spec 8.1). Edits are checked with M1a validate_edits against the tool's protected files and AgentRuntime.private_paths (the revision's check and judge files) and stored before dispatch. A closed round admission refuses dispatch at Inflight.add and again in on_started, so a job queued in submit when admission closed never starts; a refused run settles Amount() and raises LimitReached('stop') (roadmap R4.13.10 execute_tool_run comment). It uses M1a's Job.on_partial and effective_tool_timeout, so no _exec module changes. Roadmap R3.2 lists tests/_fakes/engine.py and tests/_agent/conftest.py, because they mirror no module."
```

### Task 20: `AgentContext`

**Files:**
- Modify: `src/swarm_arena/_agent/context.py` (Task 9 created the class shell), `src/swarm_arena/manager.py`, `tests/test_public_api.py`, `CHANGELOG.md` ("`AgentContext`, the manager action set (`swarm_arena.manager`)."), `tests/_fakes/engine.py` (add `CtxEnv`: `counters: AgentCounters`, `artifacts: ArtifactStore` (the store of the runtimes it builds), `rt: AgentRuntime` and `address: str` (of the last context built), `ctx(**rt_fields: Any) -> AgentContext` (shortcuts besides runtime fields: `tool_runs_used`, `learnings_used`, `agent`, `replicates` for `spec.replicates`), `script_ctx() -> AgentContext` (an agent with `ScriptTool(command=["python", "problem/run.py"])` whose goal files are `problem/a.phyk` = `"a = 1\n"` and `problem/run.py` = `"print(1)\n"`, both put in `artifacts`), `events(type: str) -> list[Event]`), `tests/_agent/conftest.py` (fixture `ctx_env`), `tests/_fakes/tools.py` (add `class AlwaysFailCheck(Check)`: `name = "always-fail"`, `Config(reason: str = "always fails", feedback: Literal["pass_fail", "reasons"] = "reasons")`, `feedback` read from the config, `check()` returns `CheckResult(check=name, passed=False, reasons=[config.reason])`), roadmap R4.4 (`LimitReached.limit` comment: "the Limits field name, `"budget"`, or `"stop"` [plan]: `"stop"` when a stop request blocks new actions; `"budget"` covers the agent share, the goal pool, and the run cap, and the message says which")
- Test: `tests/_agent/test_context.py`

**Interfaces:**
- Consumes: `AgentRuntime`, `execute_tool_run`, `RepeatIndex`, `Watchdog`, `BoardView`, `for_manager`, `SoftError`, `LimitReached`, `current_requesting_call`, `Learning`, `Action`, `ScriptTool` (M1a Task 19), `effective_files` (M1a Task 14), `ArtifactStore.get_bytes`.
- Produces: `AgentContext(rt: AgentRuntime, *, gate: Callable[[], LimitReached | None] = lambda: None)` with every attribute and method of R4.4, including the private `_read_artifact(self, sha256: str) -> bytes` that M1a Task 19 adds to R4.4 (the M1a `ScriptTool` `list_files`/`read_file` handlers call it through `FileViewContext`): it reads bytes only for files in the agent's view, that is, a value of `goal.files.manifest` or of `effective_files(goal, c)` for a candidate `c` that `get_candidate` returns (the round-start view plus the agent's own candidates), through `rt.artifacts.get_bytes(sha256)`; any other `sha256` → `SoftError("file not in your view")`. It counts no turn and no board read (the action that calls it already counted them). `gate()` returns the `LimitReached` to raise when new actions are blocked (Task 22 `StopState.gate_for`); every action raises it. Each action call counts one turn [plan]; reads of `goal`, `spec`, `round`, `round_spec`, `digest`, `notes`, `budget`, and `my_candidates()` do not. Every action that changes `turns` or `board_reads` appends one ordinary `agent_counters {turns, board_reads}` event (`task_id` = `agent_task_id(address, round)`), so a resume restores both (Task 27). `budget = rt.ledger.remaining(rt.holder).to_estimate()` (R4.4 types it `Estimate`). When `rt.watchdog.limit` is set, every action raises `LimitReached(rt.watchdog.limit, rt.watchdog.reason)`. `LimitReached.limit` values in this milestone: a `Limits` field name (`"max_turns_per_round"`, `"max_tool_runs_per_round"`, `"max_refused_repeats"`, `"max_consecutive_failures"`, `"run_timeout_s"`), `"budget"`, or `"stop"` (R4.4 as edited here).

- [ ] **Step 1: Write the failing tests** (fixture `ctx_env` from the Task 19 `rt_env`, `Limits(max_tool_runs_per_round=3)`)

```python
async def test_parallel_run_tool_respects_last_slot(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(tool_runs_used=2)
    results = await _gather(ctx.run_tool({"x": 1.0}), ctx.run_tool({"x": 2.0}))
    assert sum(isinstance(r, Candidate) for r in results) == 1
    assert [r.limit for r in results if isinstance(r, LimitReached)] == ["max_tool_runs_per_round"]

async def test_parallel_post_learning_respects_rate_limit(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(learnings_used=2); c = await ctx.run_tool({"x": 1.0})
    results = await _gather(*(ctx.post_learning(f"t{i}", "x", [c.id]) for i in range(2)))
    assert sum(isinstance(r, Learning) for r in results) == 1 and sum(isinstance(r, SoftError) for r in results) == 1

async def test_soft_errors_do_not_use_a_run(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    for bad in [ctx.run_tool({"x": "nope"}), ctx.run_tool({"x": 1.0}, parents=["c-unknown"]),
                ctx.run_tool({"x": 1.0}, edits={"secret.py": "x"}), ctx.run_tool({"x": 1.0}, edits={"problem/a.phyk": "x" * 262_145})]:
        with pytest.raises(SoftError):
            await bad
    assert ctx_env.counters.tool_runs == 0

async def test_refused_repeats_trip_watchdog(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(); await ctx.run_tool({"x": 1.0})
    for _ in range(3):
        with pytest.raises(SoftError, match="repeat refused"):
            await ctx.run_tool({"x": 1.0})
    with pytest.raises(LimitReached) as err:
        await ctx.run_tool({"x": 1.0})
    assert err.value.limit == "max_refused_repeats" and ctx_env.counters.tool_runs == 1
    assert len([e for e in ctx_env.events("watchdog") if "refused" in e.payload]) == 4

async def test_turn_limit(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(limits=Limits(max_turns_per_round=2, max_board_reads_per_round=10))
    ctx.search_learnings("a"); ctx.search_learnings("b")
    with pytest.raises(LimitReached) as err:
        ctx.search_learnings("c")
    assert err.value.limit == "max_turns_per_round"

def test_board_reads_capped(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    for i in range(10): ctx.search_learnings(str(i))
    with pytest.raises(SoftError, match="max_board_reads_per_round"):
        ctx.read_learning("L-x")

async def test_learning_rules(ctx_env: CtxEnv) -> None:
    ...  # empty title, 121-char title, 1,501-char text, no evidence, unknown evidence id → SoftError each

async def test_notes_cap_and_carry(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    with pytest.raises(SoftError):
        await ctx.save_notes("x" * 16_385)
    await ctx.save_notes("lr 0.01 diverged")
    assert ctx_env.events("notes_saved")[-1].payload == {"text": "lr 0.01 diverged"}

async def test_other_agents_round_items_hidden(ctx_env: CtxEnv) -> None:
    a, b = ctx_env.ctx(agent="a"), ctx_env.ctx(agent="b")
    c = await a.run_tool({"x": 1.0})
    with pytest.raises(SoftError, match="unknown candidate"):
        b.get_candidate(c.id)
    assert a.get_candidate(c.id).id == c.id

async def test_read_ids_recorded(ctx_env: CtxEnv) -> None:
    ...  # read_learning("L-1") then run_tool → candidate.read_ids == ["L-1"]

async def test_extra_action_counts_as_board_read(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.script_ctx()
    assert await ctx.call_action("list_files", {}) == {"files": ["problem/a.phyk", "problem/run.py"]}   # M1a Task 19 shape
    assert ctx_env.counters.board_reads == 1
    with pytest.raises(SoftError):
        await ctx.call_action("nope", {})

async def test_read_file_action_reads_only_the_agent_view(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.script_ctx()
    assert await ctx.call_action("read_file", {"path": "problem/a.phyk"}) == {"path": "problem/a.phyk", "content": "a = 1\n"}
    assert ctx_env.counters.board_reads == 1
    with pytest.raises(SoftError, match="no such file: secret.py"):
        await ctx.call_action("read_file", {"path": "secret.py"})
    outside = ctx_env.artifacts.put_bytes(b"not a file of this agent's view")
    with pytest.raises(SoftError, match="file not in your view"):
        ctx._read_artifact(outside)

async def test_pass_fail_feedback_hides_reasons(ctx_env: CtxEnv) -> None:
    hidden = AlwaysFailCheck({"reason": "secret threshold 0.3", "feedback": "pass_fail"})
    ctx = ctx_env.ctx(checks=[("hidden", hidden)])
    c = await ctx.run_tool({"x": 1.0})
    assert (c.checks[0].passed, c.checks[0].reasons) == (False, [])
    assert ctx.get_candidate(c.id).checks[0].reasons == [] and ctx.my_candidates()[0].checks[0].reasons == []
    assert ctx_env.events("tool_end")[-1].payload["candidate"]["checks"][0]["reasons"] == ["secret threshold 0.3"]

async def test_gate_blocks_new_actions(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(gate=lambda: LimitReached("stop", "the run is stopping"))
    with pytest.raises(LimitReached) as err:
        await ctx.run_tool({"x": 1.0})
    assert err.value.limit == "stop"

def test_actions_journal_counters(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    ctx.search_learnings("a"); ctx.search_learnings("b")
    last = ctx_env.events("agent_counters")[-1]
    assert last.payload == {"turns": 2, "board_reads": 2} and last.task_id == agent_task_id(ctx_env.address, 1)

async def test_parallel_identical_nondeterministic_requests_get_distinct_replicates(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx(tool=FunctionTool("_fakes.tools:square", deterministic=False), replicates=2)
    results = await _gather(ctx.run_tool({"x": 1.0}), ctx.run_tool({"x": 1.0}))
    assert {c.request.replicate for c in results} == {0, 1}

def test_budget_is_an_estimate(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    assert isinstance(ctx.budget, Estimate)
    assert ctx.budget.usd == ctx_env.rt.ledger.remaining(ctx_env.rt.holder).usd
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_agent/test_context.py -v`
Expected: FAIL with `ImportError: cannot import name 'AlwaysFailCheck'`; once the fake exists, with `TypeError` (the Task 9 `AgentContext` shell takes no arguments).

- [ ] **Step 3: Implement** `AgentContext`. `run_tool`: gate → watchdog limit → turn count → `execute_tool_run`, which validates inputs, edits, and parents, applies the repeat rules, and counts the tool run before it reserves (Task 19). Every check-and-increment happens with no `await` in between (Review Focus 1), so no check runs twice. `get_candidate`, `my_candidates`, and `best` return `for_manager(...)` copies with the plan's check feedback modes. Learnings: `Learning` built with an id from `rt.ids.next("L")`, `author="<team>/<agent>"`, `posted_by_call=current_requesting_call()`, journaled as `learning` (ordinary), then `board.put_learning`; `duplicate_reason` is accepted and ignored until M2. Add `AgentContext` to `swarm_arena.manager`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_agent tests/test_public_api.py tests/test_docstrings.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_agent/context.py src/swarm_arena/manager.py tests CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(agent): add AgentContext with limit-checked, parallel-safe actions and soft errors" -m "Roadmap R4.4: LimitReached.limit may also be \"stop\", because a stop request blocks new actions and is not a Limits field; run, goal, and agent budgets all use \"budget\". Actions journal agent_counters, so turns and board reads survive a resume (spec 6.5)."
```

### Task 21: `ScriptedManager` and `testing.check_manager`

**Files:**
- Create: `src/swarm_arena/_agent/scripted.py`, `src/swarm_arena/testing/_harness.py`
- Modify: `src/swarm_arena/testing/_conformance.py`, `src/swarm_arena/testing/__init__.py`, `src/swarm_arena/__init__.py`, `tests/test_public_api.py`, `CHANGELOG.md`, `tests/_fakes/tools.py` (add `score_min_y(goal: Goal, candidate: Candidate) -> Score` = `Score.minimize(candidate.output.data["y"])`, for `square`)
- Test: `tests/_agent/test_scripted.py`, `tests/testing/test_harness.py`, `tests/testing/test_conformance.py`

**Interfaces:**
- Consumes: `AgentContext`, `RoundReport`, `SoftError`, `LimitReached`, `judge_round` (Task 17), the Task 19 and 20 runtime pieces.
- Produces: `ScriptedManager` (R4.4; top-level export); `check_manager` (R4.12 signature unchanged; it passes no judges); `OfflineHarness` in `_harness.py`: `async def run_rounds(self, manager: Manager, *, goal: Goal, tool: Tool, rounds: int, limits: Limits, config: RunConfig, judges: Sequence[Ref] = ()) -> list[RoundReport]` (temp `LocalStore` run folder, a real `JournalWriter`, an unlimited `Ledger` unless `limits` caps it, a `RunBoard`, `LocalExecutor`, `IdGen("local", 1)`; frontier and snapshot per round, as the engine does; after each round, `judge_round` over that round's gated-in candidates with `judges` and `ctx_for` building a `JudgeContext` with `ids` = the harness `IdGen` and `on_started`/executor from `command_journal` (Task 19: `command_start` before, `command_end` after each judge command), scores journaled and put on the board and one `judge_error` event journaled per `JudgeOutcome.errors` entry, before the next round's frontier and snapshot); after `run_rounds`, `OfflineHarness.candidates: list[Candidate]` and `OfflineHarness.learnings: list[Learning]` hold what the manager made, and `OfflineHarness.store: LocalStore` is its run store (tests read the journal from it). `judges` is private to the harness: tests of `ScriptedManager` use it, `check_manager` does not.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_agent/test_scripted.py
async def test_runs_requests_in_order_and_reports() -> None:
    reports = await check_manager(ScriptedManager(requests=[{"inputs": {"x": 1.0}}, {"inputs": {"x": 2.0}}]),
                                  goal=GOAL, tool=FunctionTool("_fakes.tools:square"))
    assert reports == [RoundReport(summary="ran 2 of 2 requests")]

# _run(m, rounds) -> OfflineHarness: run_rounds(m, goal=GOAL, tool=SQUARE, rounds=rounds, limits=Limits(), config=RunConfig(...),
#     judges=[Ref.of(ProgramJudge("_fakes.tools:score_min_y"))]) on a fresh harness, which it returns
async def test_round_filter_and_best_parent() -> None:
    m = ScriptedManager([{"inputs": {"x": 3.0}, "round": 1}, {"inputs": {"x": 1.0}, "parents": "best", "round": 2}])
    h = await _run(m, rounds=2)
    assert [c.round for c in h.candidates] == [1, 2] and h.candidates[1].parent_ids == [h.candidates[0].id]

async def test_best_parent_with_empty_board_runs_without_parents() -> None:
    h = await _run(ScriptedManager([{"inputs": {"x": 1.0}, "parents": "best"}]), rounds=1)
    assert h.candidates[0].parent_ids == []

async def test_soft_error_skips_request() -> None:
    reports = await check_manager(ScriptedManager([{"inputs": {"x": "bad"}}, {"inputs": {"x": 1.0}}]), goal=GOAL, tool=SQUARE)
    assert reports[0].summary == "ran 1 of 2 requests (1 soft error)"

async def test_post_learnings_cites_best_own_candidate() -> None:
    m = ScriptedManager(ScriptedManager.Config(requests=[ScriptedManager.Request(inputs={"x": 3.0}, round=1),
                                                         ScriptedManager.Request(inputs={"x": 1.0}, round=2)], post_learnings=True))
    h = await _run(m, rounds=2)                               # y = x * x: round 1 makes y = 9, judged at the round-1 barrier
    first = h.candidates[0]
    assert [(l.round, l.title, l.text, l.evidence) for l in h.learnings] == [(2, f"Best so far: {first.id}", "score 9", [first.id])]
```

```python
# tests/testing/test_harness.py
async def test_harness_judges_each_round_before_the_next_snapshot() -> None:
    h = OfflineHarness()
    m = ScriptedManager([{"inputs": {"x": 2.0}, "round": 1}, {"inputs": {"x": 3.0}, "round": 2}])
    await h.run_rounds(m, goal=GOAL, tool=SQUARE, rounds=2, limits=Limits(),
                       config=RunConfig(), judges=[Ref.of(ProgramJudge("_fakes.tools:score_min_y"))])
    scores = _events(h, "score")
    assert [s.payload["score"]["candidate_id"] for s in scores] == [c.id for c in h.candidates]
    assert scores[0].seq < _events(h, "frontier")[1].seq                       # round-1 score precedes round-2 frontier

async def test_harness_judge_commands_journal_command_events() -> None:
    h = OfflineHarness()
    await h.run_rounds(ScriptedManager([{"inputs": {"x": 2.0}}]), goal=GOAL, tool=SQUARE, rounds=1, limits=Limits(),
                       config=RunConfig(), judges=[Ref.of(_script_judge())])          # ScriptJudge: one command job
    start, end = _events(h, "command_start")[-1], _events(h, "command_end")[-1]
    assert start.payload["job_id"] == end.payload["job_id"] and start.seq < end.seq
```

```python
# tests/testing/test_conformance.py (new)
async def test_check_manager_surfaces_manager_exception() -> None:
    with pytest.raises(AssertionError, match="raised RuntimeError"):
        await check_manager(_RaisingManager(), goal=GOAL, tool=SQUARE)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_agent/test_scripted.py tests/testing -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'check_manager'`.

- [ ] **Step 3: Implement.** `ScriptedManager.run_round` runs the requests whose `round` is `None` or `ctx.round`, in order; `"best"` → `[ctx.best(1)[0].candidate_id]` when a row exists, else `[]`; a `SoftError` is counted and skipped; `LimitReached` propagates. Summary format: `"ran <n> of <m> requests"` plus `" (<k> soft error)"` / `" (<k> soft errors)"`. With `post_learnings`, at the end of each round it posts `title="Best so far: <id>"`, `text=f"score {value:.6g}"`, `evidence=[id]` for its best own scored candidate: over `ctx.my_candidates()`, the one whose `ctx.scores_of(id)` has the greatest key (`value` is that score's `Score.value`, else `key[0]`). Candidates of the current round are not judged until the barrier, so in round 1 it has no scored candidate and posts nothing. `check_manager` wraps `OfflineHarness.run_rounds` and turns any exception other than `LimitReached` into `AssertionError(f"manager raised {type(exc).__name__}: {exc}")`. Docstring Examples: `ScriptedManager(requests=[{"inputs": {"x": 1.0}}])` (runs) and `check_manager` (`# doctest: +SKIP`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_agent tests/testing tests/test_public_api.py tests/test_docstrings.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_agent/scripted.py src/swarm_arena/testing src/swarm_arena/__init__.py tests CHANGELOG.md
git commit -m "feat(agent): add ScriptedManager and the check_manager conformance helper"
```

### Task 22: Round runner and mid-round stops

**Files:**
- Create: `src/swarm_arena/_engine/__init__.py` (empty), `src/swarm_arena/_engine/round.py`, `src/swarm_arena/_engine/stops.py`, `src/swarm_arena/_engine/control.py` (`stop_key` only), `tests/_engine/conftest.py` (fixture `env`)
- Modify: `tests/_fakes/engine.py` (add `EngineEnv` and the config helpers below), roadmap R3.2 (new row, because this test file mirrors no module: `tests/_engine/conftest.py` | Engine fixture `env` (an `EngineEnv` on a real lease, journal, ledger, board, and executor) | M1b (+M2). M2 Task 9 changes its board to `ArenaBoard`)
- Test: `tests/_engine/test_round.py`, `tests/_engine/test_stops.py`, `tests/_engine/test_control.py` (`test_stop_key`)

**Interfaces:**
- Consumes: `AgentContext`, `AgentRuntime` (with `private_paths`), `RepeatIndex`, `Inflight`, `track_inflight`, `Board.frontier/snapshot`, `build_digest`, `Ledger.open_round`, `Tranche`, `Watchdog`, `JournalWriter`, `Ledger.run_exhausted/goal_exhausted`, `Tranche.pending`, `Fenced`, `LLMScope`, `llm_scope` (Task 10, R4.13.7).
- Produces:
  - LLM scope per manager round [plan] (R4.13.7: the core enters a scope around every manager round): each slot's `manager.run_round(ctx)` runs inside `llm_scope(LLMScope(role="manager", run_id=deps.run_id, goal_id=goal_id, team=<team name>, agent=<agent name>, round=round, candidate_id=None, tranche=<the agent's tranche, the one in its AgentRuntime>, journal=deps.journal, artifacts=deps.artifacts, models=deps.config.models, tracing=deps.config.tracing, cache=deps.config.cache))`, entered inside that slot's own task, so parallel managers never share a scope. A converge-team agent's scope is the same, with its `"<goal>/:final"` tranche. M3 `MeteredModel` and `AgentsSDKManager` read this scope (M3 Tasks 8, 10).
  - `control.py`: `stop_key` only (R4.13.13); Task 25 adds the rest of the module.
  - `stops.py`: `StopState` (dataclass): `stop: bool`, `stop_now: bool`, `run_timeout: bool`, `run_budget: bool`, `goal_budget: set[str]`; `gate_for(goal_id: str) -> Callable[[], LimitReached | None]` (`stop` or `stop_now` → `LimitReached("stop", "the run is stopping")`; `run_timeout` → `LimitReached("run_timeout_s", ...)`; `run_budget` → `LimitReached("budget", "run budget used up")`; `goal_id in goal_budget` → `LimitReached("budget", "goal budget used up")`); `RunClock(elapsed_before: float)` with `elapsed() -> float`, `left(run_timeout_s: float) -> float`; `round_deadline(start: float, round_timeout_s: float, run_left_s: float) -> float` = `start + min(round_timeout_s, run_left_s)` (pure; one time basis, `anyio.current_time()`, computed at round start from `limits.effective_round_timeout_s()` and `clock.left(run_timeout_s)`); `async def watch_stop_files(store: Store, epoch: int, state: StopState, *, interval_s: float = 2.0) -> None` (polls `control/stop-e<epoch>` and `control/stop-now-e<epoch>` only).
  - `round.py`: `EngineDeps` (dataclass, kw-only, fields only): `run_id`, `config`, `store` (run store), `journal`, `ledger`, `board`, `executor`, `artifacts`, `ids`, `views`, `stops`, `clock`; `AgentSlot` (dataclass): `address`, `team`, `spec`, `manager`, `tool_ref`, `goal`, `goal_key`, `part`, `notes`, `watchdog`, `status`; `RoundOutcome` (dataclass): `round`, `candidates: list[Candidate]`, `learnings: list[Learning]`, `statuses: dict[str, AgentStatus]`, `reasons: dict[str, str]`, `repeat_entries: list[tuple[str, Candidate]]` (every `resumed` pair of a restarted round, then every agent's `RepeatIndex.added` of this round in slot order, deduplicated by candidate id, first kept); `RoundRunner(deps: EngineDeps, *, goal_id: str, plan: Plan, agents: Sequence[AgentSlot], round_spec: RoundSpec, rounds_left: int, resumed: Mapping[str, Sequence[tuple[str, Candidate]]] = {}, repeat_entries: Sequence[tuple[str, Candidate]] = (), frontier: Frontier | None = None, converge: bool = False)` with `run(round) -> RoundOutcome` (R4.13.13). `resumed` (restarted round, resume): `"<team>/<agent>"` → that agent's own committed `(repeat key, candidate)` pairs of this round, keys from `GoalState.repeat_keys` (the journaled `tool_end.repeat_key`); the agent's `AgentRuntime.resumed` is the candidates of its pairs. `repeat_entries`: the `(repeat key, candidate)` pairs of this goal's candidates of earlier rounds of this run (the `GoalRunner` keeps them from each `RoundOutcome`; resume rebuilds them as in Task 24, frontier-covered candidates only). `frontier`: the recorded frontier of a restarted round (resume); then the runner does not call `ledger.open_round` (the replayed ledger already holds that round's journaled shares, Task 9) and journals no new `round_started` or `frontier`; `None` builds and commits a new one. `converge=True`: the converge-team round (no `ledger.open_round`; every agent's `holder` is `"<goal>/:final"`; the agents' gate blocks only on `stop_now`, because after `stop` and the budget stops a converge team still runs, §6.4, §6.5).
  - Repeat index per agent [plan]: `RepeatIndex(deps.run_id, [*repeat_entries, *own resumed entries])`. These are the candidates of the round-start snapshot (frontier commits come before any tool start, so the snapshot holds earlier rounds only; past-run candidates join with the M2 board) plus the agent's own items. Another agent's candidate of the current round is never an entry, so identical requests in the same round both run (§6.2). The own resumed entries are the agent's `resumed` pairs: during the restarted round they are in the owner's index only; when the round ends they join `RoundOutcome.repeat_entries` with the new additions, so from the next round on they are earlier-round entries like any other (the owner's identical request is refused, another agent's is linked, and their replicate indices stay used). `return_own` adds nothing to `added`, so without this the recovered candidates would drop out of repeat tracking after the restarted round.
  - Payload keys added here: `round_started.shares` (address → `Amount` dump of each agent's share at round start) is read by `Ledger.replay` through `RunState.ledger_events` (Task 7 keeps the whole event); `agent_status.a2a` and `watchdog.reason` are not folded by `RunState` (the `agent_status` status covers a watchdog stop), so `replay.py` does not change.
  - Test helpers in `tests/_fakes/engine.py`: `with_limits(cfg: RunConfig, **limits: Any) -> RunConfig` and `with_executor(cfg: RunConfig, **fields: Any) -> RunConfig` (both `model_copy(update=...)`: `limits` updated, or `executor` with its `Ref.config` updated); `EngineEnv` (dataclass: `deps: EngineDeps`, `store: LocalStore` (the run store); methods `events(type: str) -> list[Event]` and `event_types() -> list[str]` (every appended event, from the writer's local file), `is_durable(type: str) -> bool` (the first event of that type is covered by the committed head), `candidate(id: str) -> Candidate`, `candidates() -> list[Candidate]`, `segment_index(seq: int) -> int` (index of the committed segment holding `seq`), `on_commit(type: str, fn: Callable[[Event], None]) -> None` (calls `fn` after an `append_durable` of that type returns), `async wait_for(path: Path) -> None` (polls until the file exists), `with_limits(**limits: Any) -> EngineEnv` and `with_executor(**fields: Any) -> EngineEnv` (a new `EngineEnv` on the same run folder whose `deps` use the updated config, a new `Ledger` and, for `with_executor`, a new `LocalExecutor`)). Fixture `env(tmp_path: Path, run_config: RunConfig, artifacts: ArtifactStore) -> AsyncIterator[EngineEnv]` in `tests/_engine/conftest.py`: a real lease, `JournalWriter` (entered for the test), `Ledger` with `hold_final_slices({"g": Amount()})` and `open_goal("g")`, `RunBoard`, `LocalExecutor`, `IdGen`, `ViewWriter`, `StopState()`, `RunClock(0.0)`.

- [ ] **Step 1: Write the failing tests** (fixture `env: EngineEnv`; agents use `ScriptedManager` and `_fakes.tools`; `_scripted(name, requests)`, `_runner(name)` (one `square` run), `_sleeper(name, spans)` (sleeps 0.5 s and appends its `(start, end)` `anyio.current_time()` pair to `spans`), `_raiser`, `_slow_script(name, marker)` (one `PyTool` run of `write_then_sleep(marker)`, Task 19), `_failing(name, runs)` (`boom` runs), and `_with_manager(name, manager)` (that `Manager` instance) build `AgentSlot`s; `_ScopeProbe(seen)` is a test-local `Manager` whose `run_round` appends `current_llm_scope()` to `seen`, reserves and settles one `Amount(usd=0.01)` call through `scope.tranche`, and returns `RoundReport(summary="")`; `_GateExecutor(inner)` is a test-local `Executor` wrapper like Task 19's `_QueuedExecutor`: `submit` sets `entered: anyio.Event`, awaits `release: anyio.Event`, then delegates to `inner`, and every other method delegates; `Inflight` comes from `swarm_arena._agent.tool_run`)

```python
def _round(env: EngineEnv, agents: Sequence[AgentSlot], **kw: Any) -> RoundRunner:
    return RoundRunner(env.deps, goal_id="g", plan=PLAN, agents=agents, round_spec=RoundSpec(), rounds_left=1, **kw)

async def test_frontier_committed_before_any_tool_start(env: EngineEnv) -> None:
    await _round(env, [_runner("a"), _runner("b")]).run(1)
    types = env.event_types()
    assert types.index("frontier") < types.index("tool_start") and env.is_durable("frontier")

async def test_round_started_commits_with_its_frontier(env: EngineEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    real = env.deps.board.frontier
    def committing_frontier() -> Frontier:                      # forces a commit while the frontier is built
        anyio.from_thread.run(env.deps.journal.commit)
        return real()
    monkeypatch.setattr(env.deps.board, "frontier", committing_frontier)
    await _round(env, [_runner("a")]).run(1)
    rs, fr = env.events("round_started")[0], env.events("frontier")[0]
    assert fr.seq == rs.seq + 1 and env.segment_index(rs.seq) == env.segment_index(fr.seq)

async def test_managers_run_in_parallel(env: EngineEnv) -> None:
    spans: list[tuple[float, float]] = []
    with anyio.fail_after(5):                                   # default round_timeout_s 14,400: the watchers must end
        await _round(env, [_sleeper("a", spans), _sleeper("b", spans)]).run(1)
    assert max(s for s, _ in spans) < min(e for _, e in spans)  # the two sleeps overlap

async def test_manager_exception_ends_only_that_agent(env: EngineEnv) -> None:
    out = await _round(env, [_raiser("a"), _runner("b")]).run(1)
    assert out.statuses == {"t/a": AgentStatus.FINISHED, "t/b": AgentStatus.FINISHED}
    assert out.reasons["t/a"] == "manager error: RuntimeError: boom" and len(out.candidates) == 1

async def test_fenced_propagates_from_manager(env: EngineEnv) -> None:
    env.on_commit("frontier", lambda ev: take_over_retrying(env.store))       # fenced before any reserve commits
    with pytest.raises(Fenced):
        await _round(env, [_runner("a"), _runner("b")]).run(1)
    assert not any("manager error" in (e.payload.get("reason") or "") for e in env.events("agent_status"))

async def test_same_round_identical_requests_both_run(env: EngineEnv) -> None:
    same = [{"inputs": {"x": 1.0}}]
    out = await _round(env, [_scripted("a", same), _scripted("b", same)]).run(1)
    assert len({c.id for c in out.candidates}) == 2 and env.event_types().count("tool_start") == 2
    assert {c.agent for c in out.candidates} == {"a", "b"}                     # neither result is linked to the other

async def test_earlier_round_identical_request_is_linked(env: EngineEnv) -> None:
    first = await _round(env, [_scripted("a", [{"inputs": {"x": 1.0}}])]).run(1)
    out = await _round(env, [_scripted("b", [{"inputs": {"x": 1.0}}])], repeat_entries=first.repeat_entries).run(2)
    assert out.candidates == [] and env.event_types().count("tool_start") == 1  # b got a's round-1 candidate, no new run

async def test_restarted_round_hands_on_its_resumed_entries_at_its_end(env: EngineEnv) -> None:
    x = [{"inputs": {"x": 1.0}}]
    before = await _round(env, [_scripted("a", x), _scripted("b", [])]).run(1)   # a's committed x=1.0, then the "crash"
    ((key, cx),) = before.repeat_entries
    frontier = Frontier.model_validate(env.events("frontier")[0].payload["frontier"])
    out = await _round(env, [_scripted("a", x), _scripted("b", x)], frontier=frontier, resumed={"t/a": [(key, cx)]}).run(1)
    assert [e.agent for e in env.events("tool_start")] == ["a", "b"]           # restart: a gets return_own, b runs its own
    cb = next(c for c in out.candidates if c.agent == "b")
    assert out.repeat_entries == [(key, cx), (key, cb)]                         # a's private pair joins at the round's end

async def test_round_timeout_cancels_with_partial(env: EngineEnv, tmp_path: Path) -> None:
    e = env.with_limits(round_timeout_s=5.0)                   # frontier, reserve, and start commits take about 0.6 s
    with anyio.fail_after(30):                                  # the tool sleeps 600 s; only the timeout ends it
        out = await _round(e, [_slow_script("a", tmp_path / "m")]).run(1)
    assert out.candidates[0].output.status == RunStatus.CANCELLED and out.candidates[0].output.partial
    assert out.candidates[0].round == 1

async def test_run_budget_drains_then_cancels_inflight(env: EngineEnv, tmp_path: Path) -> None:
    marker = tmp_path / "m"
    async def budget_when_marker() -> None:
        await env.wait_for(marker); env.deps.stops.run_budget = True
    with anyio.fail_after(40):
        async with anyio.create_task_group() as tg:
            tg.start_soon(budget_when_marker)
            out = await _round(env, [_slow_script("a", marker)]).run(1)
    c = out.candidates[0]
    assert c.output.status == RunStatus.CANCELLED and c.output.partial is True   # drained for its grace_s, then cancelled
    assert env.event_types().count("tool_start") == 1

async def test_run_budget_flag_is_never_cleared(env: EngineEnv) -> None:
    env.deps.stops.run_budget = True                            # the ledger is not exhausted
    spans: list[tuple[float, float]] = []
    out = await _round(env, [_runner("a"), _sleeper("b", spans)]).run(1)   # the sleeper makes the watch tick
    assert out.candidates == [] and env.deps.stops.run_budget is True

@pytest.mark.parametrize("flag", ["stop", "run_budget"])
async def test_queued_job_never_starts_after_admission_closes(env: EngineEnv, monkeypatch: pytest.MonkeyPatch,
                                                              flag: str) -> None:
    gate = _GateExecutor(env.deps.executor)                     # queue-entry barrier: `entered`; release barrier: `release`
    e = dataclasses.replace(env, deps=dataclasses.replace(env.deps, executor=gate))
    closed, real_close = anyio.Event(), Inflight.close
    def signalling_close(self: Inflight) -> None:
        real_close(self); closed.set()
    monkeypatch.setattr(Inflight, "close", signalling_close)
    async def close_while_queued() -> None:
        await gate.entered.wait()                               # b is reserved and waits inside submit
        setattr(e.deps.stops, flag, True)
        await closed.wait()                                     # the stop watcher has closed admission
        gate.release.set()                                      # only now can b reach on_started
    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(close_while_queued)
            out = await _round(e, [_runner("b")]).run(1)
    assert "tool_start" not in e.event_types() and out.candidates == []
    (res_b,) = [ev.payload["reservation"]["id"] for ev in e.events("reserve") if ev.payload["reservation"]["kind"] == "tool"]
    settle_b = next(ev for ev in e.events("settle") if ev.payload["id"] == res_b)
    assert Amount.model_validate(settle_b.payload["actual"]) == Amount()

async def test_starved_agent(env: EngineEnv, tmp_path: Path) -> None:
    e = env.with_executor(max_concurrent=1).with_limits(round_timeout_s=1.0)
    out = await _round(e, [_slow_script("a", tmp_path / "ma"), _slow_script("b", tmp_path / "mb")]).run(1)
    assert sorted(out.statuses.values()) == [AgentStatus.FINISHED, AgentStatus.STARVED]

async def test_watchdog_stops_agent(env: EngineEnv) -> None:
    out = await _round(env, [_failing("a", runs=3)]).run(1)
    assert out.statuses["t/a"] == AgentStatus.WATCHDOG_STOPPED and "watchdog" in env.event_types()

async def test_stop_blocks_new_runs(env: EngineEnv) -> None:
    env.deps.stops.stop = True
    out = await _round(env, [_runner("a")]).run(1)
    assert out.candidates == [] and "tool_start" not in env.event_types()
    assert out.statuses == {"t/a": AgentStatus.FINISHED}

async def test_stop_now_cancels_inflight_with_partial(env: EngineEnv, tmp_path: Path) -> None:
    marker = tmp_path / "m"
    async def stop_when_marker() -> None:                       # not on the tool_start commit, which comes before "go"
        await env.wait_for(marker); env.deps.stops.stop_now = True
    with anyio.fail_after(20):                                  # grace_s 10; the tool would sleep 600 s
        async with anyio.create_task_group() as tg:
            tg.start_soon(stop_when_marker)
            out = await _round(env, [_slow_script("a", marker)]).run(1)
    c = out.candidates[0]
    assert (c.output.status, c.output.partial, c.output.data) == (RunStatus.CANCELLED, True, {"step": 1})
    assert env.is_durable("tool_end")

async def test_goal_budget_exhausted_blocks_new_runs_and_drains(env: EngineEnv) -> None:
    ...  # max_usd so the goal pool holds 2 runs: the third run_tool raises LimitReached("budget"); in-flight runs finish

async def test_agent_round_task_journaled(env: EngineEnv) -> None:
    await _round(env, [_runner("a"), _runner("b")]).run(1)
    assert sorted(e.payload["a2a"]["id"] for e in env.events("agent_status")) == ["g/t/a#r1", "g/t/b#r1"]

async def test_manager_round_runs_inside_its_llm_scope(env: EngineEnv) -> None:
    seen: list[LLMScope] = []
    await _round(env, [_with_manager("a", _ScopeProbe(seen)), _with_manager("b", _ScopeProbe(seen))]).run(1)
    assert sorted((s.role, s.run_id, s.goal_id, s.team, s.agent, s.round, s.candidate_id) for s in seen) == [
        ("manager", env.deps.run_id, "g", "t", "a", 1, None), ("manager", env.deps.run_id, "g", "t", "b", 1, None)]
    assert all(s.journal is env.deps.journal and s.models is env.deps.config.models and s.cache is env.deps.config.cache
               for s in seen)
    tranche_holders = sorted(e.payload["reservation"]["holder"] for e in env.events("reserve")
                             if e.payload["reservation"]["kind"] == "tranche")
    assert tranche_holders == ["g/t/a", "g/t/b"]                 # scope.tranche is each agent's own tranche
    assert current_llm_scope() is None                           # no scope leaks out of the round
```

```python
# tests/_engine/test_stops.py
async def test_poll_reads_only_own_epoch(store: LocalStore) -> None:
    store.create("control/stop-e1", b""); st = StopState()
    with anyio.move_on_after(0.3):
        await watch_stop_files(store, 2, st, interval_s=0.05)
    assert not st.stop
    store.create("control/stop-now-e2", b"")
    with anyio.move_on_after(0.3):
        await watch_stop_files(store, 2, st, interval_s=0.05)
    assert st.stop_now and st.stop

def test_stop_key() -> None:                      # tests/_engine/test_control.py
    assert (stop_key(2, now=False), stop_key(2, now=True)) == ("control/stop-e2", "control/stop-now-e2")

def test_gate_reasons() -> None:
    st = StopState(goal_budget={"g"})
    err = st.gate_for("g")()
    assert err is not None and (err.limit, str(err)) == ("budget", "goal budget used up") and st.gate_for("h")() is None
    st.stop_now = True
    assert (err := st.gate_for("h")()) is not None and err.limit == "stop"

def test_run_clock_counts_only_running_time() -> None:
    assert RunClock(elapsed_before=10.0).elapsed() >= 10.0 and RunClock(elapsed_before=86_399.0).left(86_400.0) <= 1.0

def test_round_deadline_single_time_basis() -> None:
    assert round_deadline(1000.0, 5.0, 2.0) == 1002.0
    assert round_deadline(1000.0, 5.0, 86_400.0) == 1005.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_engine/test_round.py tests/_engine/test_stops.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** `RoundRunner.run`:

```text
frontier given (restarted round, resume) → use it; no open_round, no round_started, no frontier event (replayed shares)
else: frontier = await to_thread(board.frontier)           # first; no await from here to the append_durable below
      unless converge: ledger.open_round(goal_id, round, holders=active addresses, rounds_left)
      journal.append(round_started {round, run_elapsed_s, shares})
      await journal.append_durable(frontier {round, frontier})   # one commit takes both: same segment
deadline = round_deadline(anyio.current_time(), limits.effective_round_timeout_s(), clock.left(run_timeout_s))
inflight = Inflight(); private_paths = the revision's check and judge file paths (once per round)
per active slot: view = to_thread(board.snapshot, slot.goal_key, frontier); digest; holder = "<goal>/:final" if converge else address;
    tranche on holder; pairs = resumed.get("<team>/<agent>", ()); RepeatIndex(deps.run_id, [*repeat_entries, *pairs]);
    AgentRuntime(holder=..., private_paths=..., resumed=tuple(c for _, c in pairs));   # pairs: in this index only
    AgentContext(rt, gate=stops.gate_for(goal_id), or a stop_now-only gate when converge)
task group: the watchers below, plus one task per slot, each inside track_inflight(inflight) and its own CancelScope;
    the watchers are cancelled once every manager task has returned:
    with llm_scope(LLMScope(role="manager", run_id, goal_id, team, agent, round, candidate_id=None,
                            tranche=<the agent's tranche>, journal, artifacts, models, tracing, cache)):
        await slot.manager.run_round(ctx)
    LimitReached → FINISHED (WATCHDOG_STOPPED when rt.watchdog.reason is set)
    Fenced, or any exception while journal.fenced is set → re-raise (never a manager error)
    other Exception → FINISHED, reason "manager error: <Type>: <msg>"
after the task group: an exception group that holds a Fenced, or journal.fenced set → raise Fenced (bare, not wrapped)
stop watcher, polls StopState every 0.5 s:
    stop → inflight.close()                                     # no new dispatch; in-flight runs go on
    stop_now or anyio.current_time() >= deadline → inflight.close(); inflight.cancel_all();
        wait up to max grace_s + 5 s; then cancel manager scopes
    run_budget → inflight.close(); let in-flight runs finish for up to the max grace_s of the in-flight jobs;
        then inflight.cancel_all() (§6.4)
    a converge round closes only on stop_now and the deadline
budget watch: every 0.5 s and after each tool run, ledger.run_exhausted() → stops.run_budget = True (never cleared during
    a run; never stops.run_budget = ledger.run_exhausted()); goal_exhausted → stops.goal_budget.add(goal_id)
end: wait up to settle_timeout_s while any tranche.pending() > 0, then close tranches; watchdog.end_round(ran=...); STARVED when the agent asked for runs and none started;
     WATCHDOG_STOPPED (and journal watchdog {reason}) when rt.watchdog.reason is set;
     journal agent_status {status, reason, a2a: agent_round_task(...)} per agent;
     RoundOutcome.repeat_entries = [*every resumed pair, *Σ rt.repeats.added in slot order], deduplicated by candidate id
         (first kept): the recovered pairs leave their owners' indexes only now, at the round boundary
```

A writer commit takes the queued events as one list when it starts, and nothing awaits between the `round_started` append and the `frontier` `append_durable`. So `round_started` and its frontier always share one segment, a committed `round_started` always has its frontier, and resume's `frontiers[round]` always exists.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_engine -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_engine tests/_engine tests/_fakes/engine.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(engine): run one round with parallel managers, round timeout, and mid-round stop rules" -m "Each manager round runs inside its own LLMScope (R4.13.7) with the agent's tranche, so M3 LLM code can reserve and log its calls. The deadline uses one time basis, a closed admission refuses dispatch after a stop, and the run budget drains in-flight runs before it cancels them (spec 6.4); the run-budget flag is never cleared during a run, and a job queued when admission closed never starts. The frontier is built before round_started is appended, so both always commit in one segment and a restarted round reuses its journaled frontier and shares. A restarted round keeps each agent's recovered (resumed) repeat entries in that agent's index only and returns them in RoundOutcome.repeat_entries at its end, so they never link within the round (spec 6.2) and count as earlier-round entries after it. Roadmap R3.2 lists tests/_engine/conftest.py, because it mirrors no module."
```

### Task 23: Parts, converge selection, and `solution.json`

**Files:**
- Create: `src/swarm_arena/_engine/parts.py`, `src/swarm_arena/_engine/converge.py` (`select_finalists` only; Task 24 adds `converge`), `src/swarm_arena/_engine/solution.py`
- Modify: `src/swarm_arena/_journal/replay.py` (fold `final_order.{candidate_ids, parts}` into `GoalState.final_order` and `GoalState.final_parts`, and `run_started.manifest` into `RunState.manifest: dict[str, Any]`, a plan field like `run_elapsed_s`), `tests/_journal/test_replay.py`
- Test: `tests/_engine/test_parts.py`, `tests/_engine/test_converge.py`, `tests/_engine/test_solution.py`, `tests/_journal/test_replay.py`

**Interfaces:**
- Consumes: `ranking.ranked`, `ranking.groups`, `RunState`, `Head`, `embed_source_head`, `Tool.fingerprint`.
- Produces:
  - `parts.py`: `team_goal(plan: Plan, team: TeamSpec) -> Goal`; `part_scoring(plan: Plan, part: str | None) -> tuple[list[Ref], list[Ref], str]` (checks, judges, ranking; `None` fields fall back to the plan level); `kept_best(state: RoundState, part: str) -> Candidate | None`; `join_parents(state: RoundState, *, parents_per_part: int) -> list[Candidate]` (top candidates of each part, parts in plan order); both rank each part under its own identity (`state.ranked(part)`, `RoundState.ranking_identity[part]`), never the main goal's; `join_failure_reasons(candidates: Sequence[Candidate]) -> list[str]`.
  - `converge.py`: `select_finalists(ranked: Sequence[tuple[Candidate, Score]], *, top_n: int, distinct: bool, fingerprint: Callable[[Candidate], str]) -> list[Candidate]`. `converge()` is created in Task 24, with `GoalRunner`, which its signature names.
  - `solution.py`: `solution_view(state, goal_id, source) -> bytes` (R4.13.13). Format [plan]: `{"schema_version": 1, "run_id", "goal_id", "status": "ok" | "no_result" | "unjoined", "reason", "finalists": [{"rank", "candidate", "scores", "lineage", "plan_revision"}], "parts": [{"part", "candidate", "scores", "lineage"}], "replicate_stats": [{"tool_identity", "inputs_sha256", "n", "mean", "min", "max"}], "best_of_n": {"candidate_id", "value"} | null, "plan_revisions": [..], "manifest": {...}, "source_head": {...}}`. Reads `GoalState.final_order` and `GoalState.final_parts` (the `final_order` event: `{"candidate_ids": [...], "parts": null}` for a joined or main-goal result, `{"candidate_ids": [], "parts": {"<part>": "<candidate id>" | null, ...}}` for an unjoined one), `no_result` (`reason`), `goal_finished` (`reason`), and `run_started.manifest`. Status: `no_result` when a `no_result` event exists; else `unjoined` when `final_parts` is set; else `ok`. For `unjoined`, `finalists` is `[]` and `parts` lists every part in plan order, with `candidate: null`, `scores: []`, `lineage: []` for a part with no passing candidate; for the other statuses `parts` is `[]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_engine/test_converge.py
def test_select_top_n_distinct_by_fingerprint() -> None:
    rk = [(_c("c-1", data={"x": 1}), _s(9)), (_c("c-2", data={"x": 1}), _s(8)), (_c("c-3", data={"x": 2}), _s(7))]
    assert [c.id for c in select_finalists(rk, top_n=2, distinct=True, fingerprint=_fp)] == ["c-1", "c-3"]
    assert [c.id for c in select_finalists(rk, top_n=2, distinct=False, fingerprint=_fp)] == ["c-1", "c-2"]
```

```python
# tests/_engine/test_parts.py
def test_part_scoring_falls_back_to_plan_level() -> None:
    checks, judges, ranking = part_scoring(_parts_plan(part_a_judges=None), "A")
    assert judges == _parts_plan().judges and ranking == "program"

def test_join_parents_one_per_part_in_plan_order() -> None:
    assert [c.part for c in join_parents(_state_two_parts(), parents_per_part=1)] == ["A", "B"]

def test_kept_best_returns_earlier_candidate() -> None:
    assert kept_best(_state_two_parts(b_agents_stopped=True), "B").id == "c-b-best"

def test_join_parents_rank_each_part_by_its_own_judge() -> None:
    st = _state_two_parts(identities={"A": ("a", "", ""), "B": ("b", "", "")})   # judge "b" orders B opposite to judge "a"
    assert [c.id for c in join_parents(st, parents_per_part=1)] == ["c-a-top-by-a", "c-b-top-by-b"]
```

```python
# tests/_engine/test_solution.py
def test_ok_solution_in_final_order_with_lineage_and_stats() -> None:
    data = json.loads(solution_view(_state_finished(), "g", H))
    assert data["status"] == "ok" and [f["candidate"]["id"] for f in data["finalists"]] == ["c-2", "c-1"]
    assert data["finalists"][0]["lineage"] == ["c-0"] and data["source_head"]["seq"] == H.seq
    assert data["replicate_stats"] == [{"tool_identity": "ti", "inputs_sha256": ANY, "n": 3, "mean": 2.0, "min": 1.0, "max": 3.0}]

def test_no_result_solution() -> None:
    data = json.loads(solution_view(_state_no_result("no passing candidate"), "g", H))
    assert (data["status"], data["reason"], data["finalists"]) == ("no_result", "no passing candidate", [])

def test_unjoined_parts_solution_lists_best_of_each_part() -> None:
    # parts "a", "b" in plan order; final_order {"candidate_ids": [], "parts": {"a": "c-a2", "b": null}}; goal_finished reason "stop --now"
    data = json.loads(solution_view(_state_unjoined(), "g", H))
    assert (data["status"], data["reason"], data["finalists"]) == ("unjoined", "stop --now", [])
    assert [(p["part"], p["candidate"]["id"] if p["candidate"] else None) for p in data["parts"]] == [("a", "c-a2"), ("b", None)]
    assert data["parts"][0]["scores"][0]["candidate_id"] == "c-a2" and data["parts"][1] == {"part": "b", "candidate": None, "scores": [], "lineage": []}

def test_view_is_pure() -> None:
    assert solution_view(_state_finished(), "g", H) == solution_view(_state_finished(), "g", H)
```

```python
# tests/_journal/test_replay.py (one new test)
def test_final_order_and_manifest_folded() -> None:
    state = RunState.from_events([..., _ev(2, "run_started", {"estimate": {}, "warnings": [], "manifest": {"run_id": "r"}}),
                                  _ev(9, "final_order", {"candidate_ids": [], "parts": {"a": "c-1", "b": None}}, goal_id="g")])
    assert (state.goals["g"].final_order, state.goals["g"].final_parts) == ([], {"a": "c-1", "b": None})
    assert state.manifest == {"run_id": "r"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_engine/test_parts.py tests/_engine/test_converge.py tests/_engine/test_solution.py tests/_journal/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError` and a `test_final_order_and_manifest_folded` assertion error.

- [ ] **Step 3: Implement** the pure functions above (no `converge` stub: Task 24 creates it). Replicate statistics group non-deterministic candidates by `(tool_identity, sha256(canonical(inputs)))` over the ranking judge's `Score.value`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_engine tests/_journal -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_engine tests/_engine src/swarm_arena/_journal/replay.py tests/_journal/test_replay.py
git commit -m "feat(engine): add part helpers, converge selection, and the solution.json view"
```

### Task 24: `GoalRunner`

**Files:**
- Create: `src/swarm_arena/_engine/goal.py`
- Modify: `src/swarm_arena/_engine/converge.py` (add `converge`; `GoalRunner` imported under `TYPE_CHECKING`), `src/swarm_arena/_journal/replay.py` (fold `part_kept.{part, candidate_id}` into `GoalState.kept_parts`, `join_failed.{attempt, reasons}` into `join_attempts`, `reserve_exhausted.{round, needed}` and `converge_started.{select_only}` into `select_only` and `converge_started`; `plan_rejected.{errors, origin}` is not folded, because a rejected plan never becomes a revision), `tests/_journal/test_replay.py`, `tests/_fakes/tools.py` (add `join_sum() -> dict[str, float]`: reads every `parents/*/data.json` in its cwd and returns `{"value": <sum of their "value" fields>}`; a `FunctionTool` with no inputs; and `SCOPE_PROBE_SEEN: list[LLMScope]` with `class ScopeProbeJudge(Judge)`: `name = "probe"`, an empty `Config`; `judge()` appends `current_llm_scope()` to `SCOPE_PROBE_SEEN`, reserves and settles one `Amount(usd=0.01)` call through `scope.tranche`, appends `f"llm-{len(SCOPE_PROBE_SEEN)}"` to `scope.call_ids`, and returns `Score.minimize(c.output.data["y"])` per candidate with `llm_call_ids` left empty), roadmap R5.4 (part-goal files of non-human revisions resolve against the main goal's manifest; the judge reservation rule; each skipped when present)
- Test: `tests/_engine/test_goal.py`, `tests/_engine/test_converge.py`, `tests/_journal/test_replay.py`

**Interfaces:**
- Consumes: `RoundRunner` (with `repeat_entries`, `converge`), `EngineDeps`, `AgentSlot`, `judge_round`, `final_judge`, `judge_amount` (Task 10), `guard`, `Policy`, `Planner`, `stamp`, `admit_revision` (Task 16), `History`, `RoundState`, `ViewWriter`, `solution_view`, part helpers, `newest_fidelity`, `Ledger.reserve/settle/end_goal`, `Tranche`, `LLMScope`, `llm_scope` (Task 10, R4.13.7), `command_journal` (Task 19).
- Produces: `GoalRunner(deps: EngineDeps, *, goal_id: str, plan: Plan, replanner: Planner | None, history: Sequence[Plan] = (), state: GoalState | None = None, journal_r0: bool = True)` with `run() -> GoalOutcome` (R4.13.13; `journal_r0` is a private keyword: `RunRunner` passes `False` because it journals every r0 at run start, Task 26; standalone tests keep the default); `r0_revision_draft(plan: Plan, base_dir: Path) -> EventDraft` (the r0 `plan_revision` payload `{plan, origin, base_dir, planner_call_ids: []}`, shared with `RunRunner`); `GoalOutcome` (dataclass): `goal_id`, `status: Literal["finished", "stopped", "no_result"]`, `finalists: list[Candidate]`, `reason: str | None`; `GoalRunner.active_agents() -> int` (the most agents the goal can run at the same moment from now on: `max(<active main-team agents>, <converge-team agents>)`, the converge team counting 0 in select mode; so the goal queue, Tasks 26 and 27, counts a converge team before it starts, §4.10); `GoalRunner.repeat_entries: list[tuple[str, Candidate]]` (from each `RoundOutcome`; on resume, from `state.repeat_keys` only for the goal's candidates that the restarted round's recorded frontier covers — `RunState.positions[c.id]` at or before that frontier's mark for this run: the round-start snapshot, as in a live round; with no unfinished round, every committed candidate of the goal. Current-round candidates reach a restarted agent only through its own resumed entries, so another agent's same-round result is never linked, §6.2. The restarted round gets `resumed` = `"<team>/<agent>"` → that agent's committed `(state.repeat_keys[c.id], c)` pairs of the round (positions after the frontier mark). When it ends, its `RoundOutcome.repeat_entries` holds those recovered pairs with the new additions, and every merge into `GoalRunner.repeat_entries` is deduplicated by candidate id (first kept), so from the next round on a recovered candidate is an ordinary earlier-round entry: the owner's identical request is refused, another agent's is linked, and its replicate index stays used. When the loop breaks before the restarted round runs (an early end), its recovered pairs are merged at that break, so the converge round sees them too: `test_early_break_merges_recovered_pairs_into_converge`); `converge(goal: GoalRunner, *, select_only: bool) -> list[Candidate]` (R4.13.13).
- Judge reservation at the barrier [plan] (R5.4 item 16): before each judge batch, `amount = judge_amount(judge, len(batch), deps.config, price_of, llm=False)`; a zero amount (`ProgramJudge`, `ScriptJudge`, LLM judges) makes no `reserve` or `settle` event; a non-zero one is `ledger.reserve("<goal>/:judges", amount, kind="judge", goal_id=...)` (the ledger takes it from the reserve left after the final slices, then from the goal pool, §8.1) and after the batch is settled `estimated=True` (judges have no usage channel). `LimitReached` from this reserve → `reserve_exhausted {round, needed}`, `select_only = True`, and no more rounds. Final judge: the same rule with `n = len(finalists)` and `llm=False`, reserved on `"<goal>/:final"` with `kind="final"` inside the final-judge scope before `final_judge`, settled the same way; a reservation that does not fit → `no_result {reason: "final slice used up"}`. M3 Task 12 adds the LLM-priced final-judge amount.
- Ranking per group [plan]: `RoundState.ranking_identity` is built per group (part; `None` = main goal) from the newest score of that group's ranking judge (`part_scoring`); anchors per group come from that group's ranking; the digest and `ctx.best(part=p)` use the ranking judge of the agent's group.
- Command journaling: every `JudgeContext` (barrier and final judge) gets `on_started` and its executor from `command_journal(..., candidate_id=<first batch member>, component=<judge name>)` (Task 19).
- Revisions [plan]: r0 is journaled by `r0_revision_draft` (only when `journal_r0`); every replanner revision (after a round, and on a join failure) goes `stamp("autonomous", parent)` → `admit_revision(rev, config, deps.artifacts, history=..., other_active_agents=...)` (Task 16: catalog check before any import, r0's `base_dir`, part-goal `FileSet`s from the main goal's manifest, snapshot, `validate_plan`); `PlanError` → `plan_rejected {errors, origin}`; an accepted revision → `plan_revision` + `plan.r<N>.toml` view.
- Lifecycle [plan]: the round loop is `round = state.rounds_closed if state else 0; while round < current_revision.rounds_total(limits): round += 1; ...` (a replanner revision can add `RoundSpec`s). Ending order: `final_order` (or `no_result`) → `goal_finished {reason}` → commit → `solution.json` view → `ledger.end_goal`, so the view's source head covers the reason. A second lifecycle (a resumed stopped goal, Task 27) converges from what is left of `<goal>/:final`: a converge-team reserve that does not fit ends that agent (`LimitReached("budget")`), and a final-judge reservation that does not fit gives `no_result {reason: "final slice used up"}`.
- LLM scopes [plan] (R4.13.7: the core enters a scope around every judge batch and planner call; M3 `LLMJudge` and `LLMPlanner` read it, M3 Tasks 12, 13). Every scope here has `run_id=deps.run_id`, `goal_id`, `team=""`, `agent=""` (the `LLMCall` defaults for a call of no agent), `candidate_id=None`, and `journal`, `artifacts`, `models`, `tracing`, `cache` from `deps` and `deps.config`. Each tranche is `Tranche(deps.ledger, holder, goal_id, size=Amount(tokens=limits.llm_tranche_tokens, usd=limits.llm_tranche_usd))` and is closed right after its step (after waiting up to `settle_timeout_s` while `pending() > 0`, as in Task 22), so no step holds budget across rounds.
  - Barrier: one tranche on `"<goal>/:judges"` per barrier; each `judge_round` call (one judge, one batch) runs inside a new `llm_scope(LLMScope(role="judge", round=round, tranche=<that tranche>, ...))`, so `scope.call_ids` holds that batch's calls only.
  - Final judge: `final_judge` runs inside `llm_scope(LLMScope(role="judge", round=0, tranche=<a tranche on "<goal>/:final">, force_no_cache_read=True, ...))`. The holder is `"<goal>/:final"`, not `"<goal>/:judges"`, because the spec pays the final judge from the goal's final slice (§4.4 "Final judge", §8.1), and per-round judges cannot use that slice.
  - After each judge call, every score of its outcome whose `llm_call_ids` is empty becomes `s.model_copy(update={"llm_call_ids": list(scope.call_ids)})` before its `score` event and `board.put_score`; a judge's own non-empty `llm_call_ids` are kept.
  - Replanner: every `replanner.plan(...)` call (after a round, and the join-failure replan of §4.11) runs inside `llm_scope(LLMScope(role="planner", round=<the round just closed>, tranche=<a tranche on "<goal>/:planner">, ...))`. Because a scope is current, M3 `LLMPlanner` opens no planning-session run for a replan. Every `plan_revision` payload holds `planner_call_ids`: `list(scope.call_ids)` for a replanner revision, and `[]` for r0 (no planner call of this run made r0; M3 Task 13 writes the plan's own `planner_call_ids` there).
- Early ends [plan] (§6.4, §6.5): before each round the loop breaks on `stops.stop` (reason `"stop"`), `stops.stop_now` (`"stop --now"`), `stops.run_timeout` (`"run_timeout_s"`), `stops.run_budget` (`"run budget"`), or `goal_id in stops.goal_budget` (`"goal budget"`); the goal then has status `"stopped"` with that reason, and still converges. After `reserve_exhausted` at the barrier, the loop breaks with reason `"reserve_exhausted"`, the goal has status `"stopped"`, and `goal_finished {reason: "reserve_exhausted"}` is written. `select_only` is set for `stop --now`, `run_timeout_s`, and `reserve_exhausted`; after `stop`, `run budget`, or `goal budget` a converge team still runs, paid from the final slice. A goal whose converge and final judge give no finalist writes `no_result {reason: "no passing candidate"}` and has status `"no_result"`, whatever ended its rounds (§4.7: no silent empty result).

- [ ] **Step 1: Write the failing tests** (fixture `env: EngineEnv` from Task 22; in-process, `LocalExecutor`, `ScriptedManager`, `_fakes.tools`; `_plan(...)` gives each agent one request per round with distinct inputs, `FunctionTool("_fakes.tools:square")`, and the ranking judge `Y_JUDGE = Ref.of(ProgramJudge("_fakes.tools:score_min_y"))` named `"y"`; `EXTRA_SCRIPT_JUDGE` is a `ScriptJudge` ref named `"extra"`; `EXTRA_PROVIDER_JUDGE` is a `_fakes.tools:ProviderJudge` ref (Task 10) named `"extra"` and `PROVIDER_RANKING` the same judge named `"prov"`; `PROBE_JUDGE` is a `_fakes.tools:ScopeProbeJudge` ref named `"probe"`; `_ScopeProbePlanner(inner)` is a test-local `Planner` whose `plan` appends `current_llm_scope()` to `self.seen`, reserves and settles one `Amount(usd=0.01)` call through `scope.tranche`, appends `f"llm-p{len(self.seen)}"` to `scope.call_ids`, and returns `await inner.plan(goal, config, history)`)

```python
async def test_rounds_select_converge_and_final_order(env: EngineEnv) -> None:
    out = await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=2, rounds=2), replanner=None).run()
    assert out.status == "finished" and len(out.finalists) == 2
    assert env.events("final_order")[-1].payload["candidate_ids"] == [c.id for c in out.finalists]
    assert json.loads(env.store.read("goals/g/solution.json").data)["status"] == "ok"
    assert env.store.exists("goals/g/plan.r0.toml")

async def test_barrier_judges_only_new_passing_candidates(env: EngineEnv) -> None:
    await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=1, agents_per_team=1, rounds=2), replanner=None).run()
    c1, c2 = sorted(env.candidates(), key=lambda c: c.round)
    by_round = {r: [e.payload["score"]["candidate_id"] for e in env.events("score") if e.round == r] for r in (1, 2)}
    assert by_round == {1: [c1.id], 2: [c2.id]}            # c1 is not judged again in round 2; final-judge scores have round 0

async def test_policy_prunes_with_reasons_and_pruned_agents_stop(env: EngineEnv) -> None:
    ...  # SuccessiveHalving: decision event has reasons; pruned agents have no round-2 candidates
async def test_survivor_shares_grow(env: EngineEnv) -> None:
    ...  # with max_usd set: a survivor's round-2 share (round_started.shares) > its round-1 share

async def test_replanner_revision_is_autonomous_and_validated(env: EngineEnv) -> None:
    out = await GoalRunner(env.deps, goal_id="g", plan=_plan(rounds=2), replanner=AddAgentPlanner("catalog-agent")).run()
    rev = env.events("plan_revision")[-1].payload
    assert (rev["origin"], rev["plan"]["revision"], rev["plan"]["parent_revision"]) == ("autonomous", 1, 0)
    assert env.store.exists("goals/g/plan.r1.toml")

async def test_replanner_runs_in_planner_scope_and_revisions_journal_call_ids(env: EngineEnv) -> None:
    planner = _ScopeProbePlanner(AddAgentPlanner("catalog-agent"))
    await GoalRunner(env.deps, goal_id="g", plan=_plan(rounds=2), replanner=planner).run()
    s = planner.seen[0]
    assert (s.role, s.goal_id, s.team, s.agent, s.round, s.candidate_id) == ("planner", "g", "", "", 1, None)
    revs = [e.payload for e in env.events("plan_revision")]
    assert (revs[0]["planner_call_ids"], revs[1]["planner_call_ids"]) == ([], ["llm-p1"])   # r0, then the replan
    assert "g/:planner" in {e.payload["reservation"]["holder"] for e in env.events("reserve")
                            if e.payload["reservation"]["kind"] == "tranche"}

async def test_judges_run_in_llm_scopes_and_scores_get_their_call_ids(env: EngineEnv) -> None:
    SCOPE_PROBE_SEEN.clear()
    plan = _plan(teams=1, agents_per_team=1, rounds=1, judges=[PROBE_JUDGE], ranking="probe")
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert [(s.role, s.goal_id, s.round, s.force_no_cache_read) for s in SCOPE_PROBE_SEEN] == [
        ("judge", "g", 1, False), ("judge", "g", 0, True)]                # the barrier batch, then the final judge
    holders = [e.payload["reservation"]["holder"] for e in env.events("reserve") if e.payload["reservation"]["kind"] == "tranche"]
    assert holders == ["g/:judges", "g/:final"]                            # ScriptedManager agents make no LLM calls
    assert {e.round: e.payload["score"]["llm_call_ids"] for e in env.events("score")} == {1: ["llm-1"], 0: ["llm-2"]}

async def test_rejected_revision_is_journaled_and_old_plan_kept(env: EngineEnv) -> None:
    ...  # SequencePlanner returning a non-catalog ref → plan_rejected event with errors; no plan.r1.toml

async def test_replanned_part_files_resolve_from_main_manifest(env: EngineEnv) -> None:
    # main goal files include problem/a.txt; SequencePlanner adds part "c" with FileSet(globs=["problem/a.*"]) and a catalog team
    await GoalRunner(env.deps, goal_id="g", plan=_parts_plan(rounds=2), replanner=SequencePlanner([_with_part_c(["problem/a.*"])])).run()
    r1 = env.events("plan_revision")[1].payload["plan"]
    part_c = next(p for p in r1["parts"] if p["name"] == "c")
    assert part_c["goal"]["files"]["manifest"] == {"problem/a.txt": MAIN_SHA["problem/a.txt"]}
    assert env.store.exists("goals/g/plan.r1.toml")

async def test_replanned_part_glob_outside_main_manifest_rejected(env: EngineEnv) -> None:
    await GoalRunner(env.deps, goal_id="g", plan=_parts_plan(rounds=2),
                     replanner=SequencePlanner([_with_part_c(["judge/hidden/*.npz"])])).run()
    rejected = env.events("plan_rejected")
    assert rejected and any("judge/hidden/*.npz" in e for e in rejected[-1].payload["errors"])
    assert not env.store.exists("goals/g/plan.r1.toml")

async def test_replanned_catalog_script_tool_is_snapshotted_and_runs(env: EngineEnv) -> None:
    # AddAgentPlanner adds the catalog ScriptTool entry with files ["problem/run.py"]; the plan's base_dir holds
    # problem/run.py = RUN_PY, which writes the sha256 of its own bytes into swarm/result.json as data["sha"]
    await GoalRunner(env.deps, goal_id="g", plan=_plan(rounds=2), replanner=AddAgentPlanner("catalog-script")).run()
    cand = next(c for c in env.candidates() if c.agent == "catalog-script")      # round 2: the new agent's run
    assert cand.output.data["sha"] == sha256_hex(RUN_PY)                         # its workspace held the snapshotted file

async def test_parts_with_own_ranking_judges_prune_and_join(env: EngineEnv) -> None:
    # parts A and B with ranking judges "a" and "b" (opposite orders), two teams each, policy KeepTopTeams(1), 2 rounds
    out = await GoalRunner(env.deps, goal_id="g", plan=_parts_plan_own_judges(), replanner=None).run()
    pruned = env.events("decision")[0].payload["decision"]["prune"]
    assert set(pruned) == {"a2", "b1"}                                          # each part pruned by its own judge
    assert sorted(env.candidate(p).team for p in out.finalists[0].parent_ids) == ["a1", "b2"]   # top of each part by its judge

async def test_parts_join_and_main_goal_scoring(env: EngineEnv) -> None:
    out = await GoalRunner(env.deps, goal_id="g", plan=_parts_plan(converge_join="_fakes.tools:join_sum"), replanner=None).run()
    join = out.finalists[0]
    assert {env.candidate(p).part for p in join.parent_ids} == {"A", "B"} and join.part is None

async def test_part_with_no_active_agent_keeps_best(env: EngineEnv) -> None:
    ...  # all part-B agents watchdog-stopped in round 2 → part_kept event; the join still gets B's round-1 best

async def test_join_failure_replans_once_then_no_result(env: EngineEnv) -> None:
    ...  # join check always fails; SequencePlanner returns a new partition once; after max_join_attempts=1 → no_result with join reasons

async def test_all_gated_out_writes_no_result(env: EngineEnv) -> None:
    out = await GoalRunner(env.deps, goal_id="g", plan=_plan(check="_fakes.tools:AlwaysFailCheck"), replanner=None).run()
    assert (out.status, out.finalists, out.reason) == ("no_result", [], "no passing candidate")
    assert env.events("score") == []
    assert json.loads(env.store.read("goals/g/solution.json").data)["status"] == "no_result"

async def test_all_judge_errors_writes_no_result(env: EngineEnv) -> None:
    raising = Ref(path="swarm_arena:ProgramJudge", config={"function": "_fakes.tools:raising_score"})
    out = await GoalRunner(env.deps, goal_id="g", plan=_plan(rounds=1, judges=[raising]), replanner=None).run()
    assert (out.status, out.finalists, out.reason) == ("no_result", [], "no passing candidate")
    assert env.events("judge_error") and env.events("score") == []
    assert json.loads(env.store.read("goals/g/solution.json").data)["status"] == "no_result"

@pytest.mark.parametrize("flag,reason,select_only", [
    ("stop", "stop", False), ("stop_now", "stop --now", True), ("run_timeout", "run_timeout_s", True),
    ("run_budget", "run budget", False), ("goal_budget", "goal budget", False)])
async def test_early_end_stops_rounds_then_converges(env: EngineEnv, flag: str, reason: str, select_only: bool) -> None:
    env.on_commit("round_closed", lambda ev: _raise_flag(env.deps.stops, flag, goal_id="g"))   # set after round 1 closes
    out = await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=2, rounds=3, converge_team=True), replanner=None).run()
    assert [e.payload["round"] for e in env.events("round_closed")] == [1]
    assert env.events("converge_started")[0].payload["select_only"] is select_only
    assert any(e.team == "conv" for e in env.events("tool_start")) is not select_only    # the converge team runs unless select-only
    assert (out.status, out.reason) == ("stopped", reason) and out.finalists

async def test_reserve_exhausted_skips_to_select_only(env: EngineEnv) -> None:
    # max_usd 10: reserve 1.5, final slice 0 (select converge, ProgramJudge ranking), goal pool 8.5, round 1 of 2 gets 4.25.
    # The one agent reserves 2.5 (default_run_reservation) and reports 8.0 USD, so 0.5 is left in the goal pool.
    # Barrier: Y_JUDGE costs 0; EXTRA_PROVIDER_JUDGE needs 1 × 2.5 > 1.5 (reserve) + 0.5 (goal pool) → reserve_exhausted.
    e = env.with_limits(max_usd=10.0, default_run_reservation=Estimate(tokens=0, usd=2.5))
    plan = _plan(rounds=2, tool=PyTool({"code": WRITE.format("{'data': {'y': 1.0}, 'usage': {'usd': 8.0}}")}),
                 judges=[Y_JUDGE, EXTRA_PROVIDER_JUDGE], ranking="y")
    out = await GoalRunner(e.deps, goal_id="g", plan=plan, replanner=None).run()
    assert [ev.payload["round"] for ev in e.events("reserve_exhausted")] == [1]
    assert [ev.payload["round"] for ev in e.events("round_closed")] == [1]                    # round 2 never starts
    assert e.events("converge_started")[0].payload["select_only"] is True and len(out.finalists) == 1
    assert (out.status, out.reason) == ("stopped", "reserve_exhausted")

async def test_free_judges_make_no_judge_reservations(env: EngineEnv) -> None:
    plan = _plan(rounds=1, judges=[Y_JUDGE, EXTRA_SCRIPT_JUDGE], ranking="y")
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert not [e for e in env.events("reserve") if e.payload["reservation"]["kind"] in {"judge", "final"}]
    starts = [e for e in env.events("command_start") if e.payload["component"] == "extra"]   # the ScriptJudge's commands
    assert starts and {e.payload["job_id"] for e in starts} <= {e.payload["job_id"] for e in env.events("command_end")}

async def test_final_judge_reservation_on_final_slice(env: EngineEnv) -> None:
    plan = _plan(rounds=1, judges=[PROVIDER_RANKING], ranking="prov", converge=ConvergeSpec(mode="select", top_n=2))
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    finals = [e.payload["reservation"] for e in env.events("reserve") if e.payload["reservation"]["kind"] == "final"]
    assert len(finals) == 1 and finals[0]["holder"] == "g/:final" and finals[0]["amount"]["usd"] == 2.0
    assert next(e for e in env.events("settle") if e.payload["id"] == finals[0]["id"]).payload["estimated"] is True

async def test_stop_reason_is_in_solution_view(env: EngineEnv) -> None:
    env.on_commit("round_closed", lambda ev: _raise_flag(env.deps.stops, "stop", goal_id="g"))
    await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=2, rounds=3), replanner=None).run()
    view = json.loads(env.store.read("goals/g/solution.json").data)
    assert view["reason"] == "stop" and view["source_head"]["seq"] >= env.events("goal_finished")[-1].seq

async def test_team_converge_final_order_holds_top_n(env: EngineEnv) -> None:
    plan = _plan(teams=2, rounds=1, converge_team=True, converge_outputs=4, top_n=2)   # the converge team makes 4 passing outputs
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert len(env.events("final_order")[-1].payload["candidate_ids"]) == 2
    assert len(json.loads(env.store.read("goals/g/solution.json").data)["finalists"]) == 2

async def test_team_converge_keeps_newest_fidelity_only(env: EngineEnv) -> None:
    plan = _plan(teams=2, rounds=1, converge_team=True, converge_round=RoundSpec(fidelity="full"))
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()
    assert all(env.candidate(i).fidelity == "full" for i in env.events("final_order")[-1].payload["candidate_ids"])

async def test_stop_now_with_parts_writes_unjoined_without_final_judge(env: EngineEnv) -> None:
    env.on_commit("round_closed", lambda ev: _raise_flag(env.deps.stops, "stop_now", goal_id="g"))
    out = await GoalRunner(env.deps, goal_id="g", plan=_parts_plan(rounds=2), replanner=None).run()
    assert set(env.events("final_order")[-1].payload["parts"]) == {"A", "B"}
    assert not [e for e in env.events("score") if e.round == 0]                 # no final judge
    view = json.loads(env.store.read("goals/g/solution.json").data)
    assert (view["status"], view["reason"], out.status) == ("unjoined", "stop --now", "stopped")

async def test_converge_team_runs_one_round_paid_from_final_slice(env: EngineEnv) -> None:
    out = await GoalRunner(env.deps, goal_id="g", plan=_plan(teams=2, rounds=1, converge_team=True), replanner=None).run()
    holders = {e.payload["reservation"]["holder"] for e in env.events("reserve") if e.payload["reservation"]["kind"] == "tool"}
    assert "g/:final" in holders and not any(h.startswith("g/conv/") for h in holders)          # converge team "conv"
    survivors = {c.id for c in env.candidates() if c.team != "conv"}
    assert all(c.parent_ids and set(c.parent_ids) <= survivors for c in env.candidates() if c.team == "conv")

async def test_active_agents_counts_converge_team_before_it_starts(env: EngineEnv) -> None:
    plan = _plan(teams=1, agents_per_team=1, converge_team=True, converge_agents=2)
    assert GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).active_agents() == 2   # max(1 main, 2 converge)

async def test_early_break_merges_recovered_pairs_into_converge(env: EngineEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(teams=1, agents_per_team=1, rounds=2, converge_team=True)
    state = await _crash_after_tool_end(env, plan, round=2)        # round 1 closed (c1); round 2 unfinished: a's committed cx
    c1, cx = sorted(env.candidates(), key=lambda c: c.round)
    seen: list[tuple[bool, list[str]]] = []
    real = RoundRunner.run
    async def spy(self: RoundRunner, round: int) -> RoundOutcome:
        seen.append((self.converge, [c.id for _, c in self.repeat_entries]))   # as passed to the constructor
        return await real(self, round)
    monkeypatch.setattr(RoundRunner, "run", spy)
    env.deps.stops.stop = True                                      # the early end comes before the restarted round 2 runs
    await GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None, state=state, journal_r0=False).run()
    assert seen == [(True, [c1.id, cx.id])]                         # round 2 never runs; the converge round has cx's pair too

async def test_round_closed_is_durable_before_next_frontier(env: EngineEnv) -> None:
    await GoalRunner(env.deps, goal_id="g", plan=_plan(rounds=2), replanner=None).run()
    rc1 = next(e for e in env.events("round_closed") if e.payload["round"] == 1)
    f2 = next(e for e in env.events("frontier") if e.payload["round"] == 2)
    assert rc1.seq < f2.seq and env.segment_index(rc1.seq) < env.segment_index(f2.seq)   # a commit covers rc1 before f2 exists

async def test_candidate_fidelity_follows_round_spec(env: EngineEnv) -> None:
    ...  # rounds=[RoundSpec(fidelity="quick"), RoundSpec(tool_overrides={"epochs": 100})] → "quick", then the hash
```

`_raise_flag(stops: StopState, flag: str, *, goal_id: str) -> None` (test helper): `"goal_budget"` → `stops.goal_budget.add(goal_id)`; `"stop_now"` → sets `stop_now` and `stop`, as `watch_stop_files` does; else sets that flag. Other module helpers: `_plan(...)` also takes `converge_outputs: int` (requests per converge-team agent), `converge_agents: int` (agents of the converge team, default 1), and `converge_round: RoundSpec` (`ConvergeSpec.round`); `_crash_after_tool_end(env, plan, *, round) -> GoalState` runs `GoalRunner(env.deps, goal_id="g", plan=plan, replanner=None).run()` inside an `anyio.CancelScope` that an `env.on_commit("tool_end", ...)` callback cancels at the first `tool_end` of `round` (the "crash"), then folds every committed event (`read_chain(env.store, env.deps.journal.head)`) with `RunState.from_events` and returns its `goals["g"]` (that round has its frontier and the committed candidate, and no `round_closed`); `MAIN_SHA` is the main goal's manifest of `_parts_plan`; `_with_part_c(globs) -> Plan` adds an unsnapshotted part `"c"` with `FileSet(globs=globs)` and one catalog team; `RUN_PY` is the bytes of the base-dir `problem/run.py`; `_parts_plan_own_judges()` gives parts A (teams a1, a2; ranking judge `"a"`) and B (teams b1, b2; ranking judge `"b"`), where `"b"` ranks the opposite way to `"a"`.

```python
# tests/_journal/test_replay.py (one new test)
def test_converge_state_folded() -> None:
    state = RunState.from_events([..., _ev(7, "part_kept", {"part": "b", "candidate_id": "c-b1"}, goal_id="g"),
                                  _ev(8, "join_failed", {"attempt": 1, "reasons": ["x"]}, goal_id="g"),
                                  _ev(9, "converge_started", {"select_only": True}, goal_id="g")])
    g = state.goals["g"]
    assert (g.kept_parts, g.join_attempts, g.select_only, g.converge_started) == ({"b": "c-b1"}, 1, True, True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_engine/test_goal.py tests/_engine/test_converge.py tests/_journal/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._engine.goal'`, `ImportError: cannot import name 'converge'`, and a `test_converge_state_folded` assertion error.

- [ ] **Step 3: Implement** `GoalRunner.run`:

```text
if journal_r0: append(r0_revision_draft(plan, base_dir)) → commit → plan.r0.toml view
round = state.rounds_closed if state else 0
while round < current_revision.rounds_total(limits):          # a replanner revision can add RoundSpecs
    round += 1
    stop, stop_now, run_timeout, run_budget, or this goal's goal_budget → status "stopped", reason (above);
        merge the recovered pairs of a restarted round that has not run; break
    out = RoundRunner(..., resumed=<recovered pairs, restarted round only>, repeat_entries=self.repeat_entries).run(round)
    self.repeat_entries += [(k, c) for k, c in out.repeat_entries if c.id not in <ids already kept>]   # recovered pairs too
    barrier: judges_tranche = Tranche(ledger, "<goal>/:judges", goal_id, size=<llm tranche size>)
             per group (part checks/judges/ranking), per judge in plan order:
             amount = judge_amount(judge, len(batch), config, price_of, llm=False); non-zero → reserve on "<goal>/:judges";
             with llm_scope(scope := LLMScope(role="judge", round=round, tranche=judges_tranche, ...)):
                 outcome = judge_round(goal_id, [(ref, judge)], batch, leaders, ctx_for=...)
                 (ctx_for → a new JudgeContext with ids=deps.ids, on_started and executor from command_journal)
             scores with empty llm_call_ids get list(scope.call_ids); journal a score event per outcome.scores
             entry and a judge_error event per outcome.errors entry; settle estimated (non-zero only);
             leaders = top ranked of that group, under that group's ranking_identity
             LimitReached from that reserve → reserve_exhausted {round, needed}; select_only = True;
             status "stopped", reason "reserve_exhausted" (goal_finished {reason: "reserve_exhausted"});
             close judges_tranche; append_durable(round_closed {round, run_elapsed_s}); break
             end of barrier: wait while judges_tranche.pending() > 0 (up to settle_timeout_s), then close it
    decision = guard(policy.after_round(state), state); journal decision; board.put_decision; pruned → agent_status PRUNED
    decision.no_result → no_result {reason}; status "no_result"; skip converge and the final judge
    parts: a part with no active agent → part_kept {part, candidate_id}
    replanner (autonomous): History →
             with llm_scope(scope := LLMScope(role="planner", round=round, tranche=Tranche(ledger, "<goal>/:planner", ...), ...)):
                 plan = replanner.plan(goal, config, history)
             (then close that tranche) → stamp("autonomous", parent)
             → admit_revision(rev, config, deps.artifacts, history, other_active_agents)
             ok → plan_revision {plan, origin, planner_call_ids: list(scope.call_ids)} + view;
             PlanError → plan_rejected {errors, origin}; planner exception → event, skipped
    append_durable(round_closed {round, run_elapsed_s})
select_only |= stops.stop_now or stops.run_timeout
journal converge_started {select_only}
finalists = converge(self, select_only=select_only) → join retry loop (§4.11; its replanner call uses the same planner scope,
            its revision goes through admit_revision too, and its plan_revision payload holds planner_call_ids)
            → unjoined result (branch a below): no final judge
            → else with llm_scope(scope := LLMScope(role="judge", round=0, tranche=Tranche(ledger, "<goal>/:final", ...),
                                                    force_no_cache_read=True, ...)):
                  amount = judge_amount(ranking, len(finalists), config, price_of, llm=False); non-zero → reserve on
                  "<goal>/:final" (kind "final"); LimitReached → no_result {reason: "final slice used up"}
                  (order, outcome) = final_judge(...) (ctx with ids=deps.ids, on_started and executor from command_journal)
                  settle estimated (non-zero only)
            → close that tranche; scores with empty llm_call_ids get list(scope.call_ids); outcome.scores and outcome.errors
              journaled as score and judge_error events with round 0; finalists = order (first top_n in team mode)
no finalist → no_result {reason: "no passing candidate"}, status "no_result"
else final_order {candidate_ids, parts: null}; an unjoined parts result → final_order {candidate_ids: [], parts: {part: id | null}}
→ goal_finished {reason} → commit → solution.json view (source head covers the reason) → ledger.end_goal
```

`converge` has three branches [plan]. (a) Unjoined (a plan with parts and `select_only`): no final judge; `final_order {candidate_ids: [], parts: {part: id | null}}` with the best (or `part_kept`) candidate of each part, ranked under that part's identity (§6.5); `no_result` only when every part has none. (b) Team mode: one `RoundRunner(..., converge=True)` round with the converge team (no `ledger.open_round`; each agent's `holder` is `<goal>/:final`; its gate blocks only on `stop_now`, because after `stop` and the budget stops a converge team still runs, §6.4, §6.5; parents = survivors or `join_parents`, round settings from `ConvergeSpec.round` or the last round); survivors = `select_finalists` over the main ranking group (`top_n`, `distinct`); the finalist pool = the gated-in converge outputs, plus the survivors when the plan has no parts; only candidates of `newest_fidelity(pool)` are kept; the final judge runs on that batch; `final_order` = the first `top_n` of its order (distinct by fingerprint when `ConvergeSpec.distinct`). (c) Select mode (or `select_only` without parts): `select_finalists(top_n, distinct)` over the newest fidelity, the final judge, `final_order` = its order.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_engine tests/_journal -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_engine src/swarm_arena/_journal/replay.py tests/_engine tests/_journal/test_replay.py tests/_fakes/tools.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(engine): run one goal through rounds, barrier, policy, replanner, converge, and final judge" -m "Every judge batch, the final judge, and every replanner call run inside an LLMScope (R4.13.7) with a tranche on <goal>/:judges, <goal>/:final (the final judge is paid from the final slice, spec §4.4, §8.1), and <goal>/:planner; scores take scope.call_ids when the judge set none, and every plan_revision journals planner_call_ids. Replanner revisions go through admit_revision, so part-goal files resolve only against the main goal's manifest; only judges that can pay hold a reservation. Each part ranks under its own judge identity. goal_finished is committed before the solution view, so the view carries the stop reason. A restarted round's recovered repeat entries join the goal's entries when that round ends, or at the early break before it runs, deduplicated by candidate id, so later rounds and the converge round refuse, link, and count replicates for them. active_agents() counts the converge team before it starts (spec 4.10), so the goal queue never starts a converge team over max_agents_total. Roadmap R5.4 records the part-file resolution and the judge reservation rule (items skipped when present)."
```

### Task 25: Preflight, manifest, stop control, status

**Files:**
- Create: `src/swarm_arena/_engine/preflight.py`, `src/swarm_arena/_engine/manifest.py`
- Modify: `src/swarm_arena/_engine/control.py` (Task 22 created it with `stop_key`), `tests/_fakes/tools.py` (add `class CountingTool(PyTool)`: `calls: ClassVar[int] = 0`, `Config(code: str = "", k: int = 1)`, `preflight(self) -> None` adds 1 to `CountingTool.calls`; `class FailingPreflightTool(PyTool)`: `Config(code: str = "")`, `preflight(self) -> None` raises `RuntimeError("no license")`)
- Test: `tests/_engine/test_preflight.py`, `tests/_engine/test_manifest.py`, `tests/_engine/test_control.py`

**Interfaces:**
- Consumes: `Ref.build`, `canonical_json`, `host_id`, `read_owner`, `read_chain`, `RunState`, `open_store`, `Lease.expired`, `stops.watch_stop_files` key format.
- Produces: `run_preflight(plans: Sequence[Plan], config: RunConfig, executor: Executor) -> None` (each component's `preflight()` once per distinct `(path, canonical config)`; every failure collected into one `RuntimeError("preflight failed:\n- ...")`); `build_manifest(*, run_id: str, config: RunConfig, plans: Sequence[Plan], capacity: Resources | None) -> dict[str, Any]` with keys `schema_version`, `run_id`, `created_at`, `versions` (`swarm_arena`, `python`, `platform`), `host` (`host_id`, `cpu_count`), `executor_capacity`, `run_config_sha256`, `plans` (`goal_id`, `revision`, `sha256`, `files_sha256`), `tools` (`ref`, `identity`, `git_commit` from `git -C <plan.base_dir> rev-parse HEAD`, `None` outside a repo); `request_stop`, `read_status` (R4.13.13; `stop_key` exists since Task 22); `StatusReport(FrozenModel)`: `run_id`, `epoch`, `owner_live: bool`, `head_seq: int`, `stop: Literal["none", "stop", "stop-now"]`, `goals: dict[str, GoalStatus]`, `spent: Amount`, `lost: int`; `GoalStatus(FrozenModel)`: `status`, `round`, `rounds_total`, `candidates`, `best: float | None`; `format_status(report: StatusReport) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_engine/test_preflight.py
async def test_preflight_once_per_distinct_config() -> None:
    CountingTool.calls = 0
    await run_preflight([_plan(agents=3, same_config=True), _plan(agents=1, config={"k": 2})], RunConfig(), _executor())
    assert CountingTool.calls == 2

async def test_preflight_failures_collected() -> None:
    with pytest.raises(RuntimeError, match=r"preflight failed:\n- .*FailingPreflightTool.*no license"):
        await run_preflight([_plan(tool="_fakes.tools:FailingPreflightTool")], RunConfig(), _executor())
```

```python
# tests/_engine/test_manifest.py
def test_manifest_keys_and_no_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-123")
    m = build_manifest(run_id="r", config=RunConfig(secrets=SecretsPolicy(allow=["ANTHROPIC_API_KEY"])), plans=[_plan()], capacity=None)
    assert set(m) == {"schema_version", "run_id", "created_at", "versions", "host", "executor_capacity",
                      "run_config_sha256", "plans", "tools"}
    assert b"sk-secret-123" not in canonical_json(m)
```

```python
# tests/_engine/test_control.py (test_stop_key is from Task 22)
async def test_request_stop_needs_live_owner(run_root: Path) -> None:
    _released_run(run_root, "r1")
    with pytest.raises(RuntimeError, match="no live owner"):
        await request_stop(str(run_root), "r1", now=False)

async def test_request_stop_twice_is_ok_and_returns_epoch(run_root: Path) -> None:
    _live_run(run_root, "r1", epoch=1)
    assert await request_stop(str(run_root), "r1", now=False) == 1
    assert await request_stop(str(run_root), "r1", now=False) == 1

async def test_request_stop_fails_when_epoch_changes_meanwhile(run_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _live_run(run_root, "r1", epoch=1)                       # returns the run store
    real = control_mod.read_owner                                    # swarm_arena._engine.control
    calls: list[int] = []
    def read_then_take_over(s: Store) -> tuple[OwnerDoc, str] | None:
        calls.append(1)
        if len(calls) == 2:                                          # after the create, before the second authority read
            Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
        return real(s)
    monkeypatch.setattr(control_mod, "read_owner", read_then_take_over)
    with pytest.raises(RuntimeError, match="epoch changed from 1 to 2"):
        await request_stop(str(run_root), "r1", now=False)
    assert store.exists("control/stop-e1") and not store.exists("control/stop-e2")   # epoch 2 never polls stop-e1

async def test_status_reads_committed_journal_only(run_root: Path) -> None:
    rep = await read_status(str(run_root), _run_with_uncommitted_tail(run_root))
    assert rep.head_seq == 5 and rep.goals["g"].candidates == 1
    assert format_status(rep).splitlines()[0] == f"run {rep.run_id}  epoch 1  owner live  head 5  stop none"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_engine/test_preflight.py tests/_engine/test_manifest.py tests/_engine/test_control.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** the three modules. Preflight calls run through `anyio.to_thread.run_sync`; each failure line is `- <ref path>: <ExceptionType>: <message>`. `request_stop` looks up `read_owner` as a module attribute (so the test can patch it) and, when the second authority read shows another epoch, raises `RuntimeError(f"run {run_id}: the epoch changed from {a} to {b} during stop; run stop again")`. `format_status` lines [plan]:

```text
run <run-id>  epoch <n>  owner <live|expired>  head <seq>  stop <none|stop|stop-now>
goal <goal-id>  <status>  round <r>/<total>  candidates <n>  best <value or ->
spent <tokens> tokens, $<usd:.2f>  lost <n>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_engine -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_engine tests/_engine tests/_fakes/tools.py
git commit -m "feat(engine): add preflight once per tool config, the run manifest, per-epoch stop files, and status"
```

### Task 26: `RunRunner` and `Swarm`

**Files:**
- Create: `src/swarm_arena/_engine/run.py`, `src/swarm_arena/_engine/swarm.py`
- Modify: `src/swarm_arena/__init__.py`, `tests/test_public_api.py`, `CHANGELOG.md`, `src/swarm_arena/_journal/replay.py` and `tests/_journal/test_replay.py` (a replay test for the goal-queue events this task writes: `goal_queued` → status `queued`, `goal_started` → `running`, a queued goal's `no_result {reason: "stopped"}` → `no_result`), roadmap R4.14 (`run_started` gains `final_slices`; every goal's r0 `plan_revision` is journaled at run start; each skipped when present)
- Test: `tests/_engine/test_run.py` (`RunRunner`), `tests/_engine/test_swarm.py` (the `Swarm` facade), `tests/_journal/test_replay.py`

**Interfaces:**
- Consumes: `stamp`, `admit_revision` (Task 16), `worst_case_estimate`, `make_price_of`, `final_slice_estimate`, `run_preflight`, `build_manifest`, `Lease.acquire_new`, `JournalWriter` (and its `fenced` event), `ViewWriter`, `Ledger`, `RunBoard`, `GoalRunner` (with `journal_r0=False`), `r0_revision_draft` (Task 24), `watch_stop_files`, `RunClock`, `new_run_id`, `IdGen`, `Fenced`, `PlanError`.
- Produces: `RunRunner(config: RunConfig, plans: Sequence[Plan], *, run_id: str | None, approve: Callable[[Estimate, Sequence[str]], Awaitable[bool]] | None, replanner: Planner | None)` with `run() -> dict[str, list[Candidate]]` (R4.13.13); `_require_asyncio() -> None` (private: calls `asyncio.get_running_loop()` in a `try`; on `RuntimeError` raises `RuntimeError("swarm-arena runs on the asyncio backend only")`; no `sniffio`, which anyio 4.15 does not install); `Swarm` (R4.1; top-level export, numpydoc Examples using `# doctest: +SKIP` for `run`). A fenced owner (§7.1) stops dispatch at once, cancels its in-flight work, and exits: `RunRunner.run` and so `Swarm.run` and `Swarm.resume` raise a bare `Fenced`.
- Test helpers: config variants use `with_limits(run_config, ...)` and `with_executor(run_config, ...)` from `tests/_fakes/engine.py`; `take_over_retrying` from the same module.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_engine/test_run.py
async def test_run_folder_layout(run_config: RunConfig) -> None:
    with anyio.fail_after(30):                                 # default limits: run_timeout_s 86,400, stop poll 2 s
        out = await Swarm(run_config).run(_plan(goal_id="g"))
    root = _run_dir(run_config)
    for key in ["run.toml", "manifest.json", "control/owner.json", "goals/g/plan.r0.toml", "goals/g/solution.json"]:
        assert (root / key).exists()
    assert list(out) == ["g"] and len(out["g"]) == 2
    assert read_owner(LocalStore(root))[0].expires_at < utc_now()           # lease released

async def test_duplicate_goal_ids_rejected(run_config: RunConfig) -> None:
    with pytest.raises(PlanError, match="duplicate goal id"):
        await Swarm(run_config).run([_plan(goal_id="g"), _plan(goal_id="g")])
    assert not Path(run_config.run_root).exists()

async def test_every_r0_committed_before_any_goal_starts(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_concurrent_goals=1)
    await Swarm(cfg).run([_plan(goal_id="a"), _plan(goal_id="b")])
    r0 = [e for e in _events(cfg, "plan_revision") if e.payload["plan"]["revision"] == 0]
    assert {e.goal_id for e in r0} == {"a", "b"}
    assert max(e.seq for e in r0) < min(e.seq for e in _events(cfg, "goal_started"))

async def test_llm_plan_with_noncatalog_ref_imports_nothing(run_config: RunConfig) -> None:
    plan = _plan(tool_ref=Ref(path="_fakes.import_canary:Tool")).model_copy(update={"origin": "llm"})
    with pytest.raises(ValueError):
        await Swarm(run_config).run(plan)
    assert "_fakes.import_canary" not in sys.modules and not Path(run_config.run_root).exists()

async def test_estimate_over_limit_refuses(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_usd=0.5)                                # worst case > 0.5 USD
    with pytest.raises(RuntimeError, match="worst-case estimate"):
        await Swarm(cfg).run(_plan())
    assert not Path(cfg.run_root).exists()

async def test_final_slices_over_reserve_refuses(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_usd=100.0)                              # reserve 15
    plan = _plan(converge_team=True, converge_agents=3)                       # slice 3 agents × 3 runs × 1.0 × (1 + 1) = 18
    with pytest.raises(RuntimeError, match="reserve cannot hold the final slices"):
        await Swarm(cfg).run(plan)
    assert not Path(cfg.run_root).exists()

async def test_capacity_warning_reaches_approve(run_config: RunConfig) -> None:
    seen = await _approve_spy(Swarm(with_executor(run_config, max_concurrent=1)), _plan(teams=2))
    assert any("Executor.capacity" in w for w in seen.warnings)

def test_require_asyncio_outside_a_loop_raises() -> None:
    with pytest.raises(RuntimeError, match="asyncio backend only"):
        _require_asyncio()

async def test_goal_queue_slots_and_agents_total(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_concurrent_goals=2, max_agents_total=4)  # three goals of 2 agents each
    await Swarm(cfg).run([_plan(goal_id=g, agents=2) for g in ("a", "b", "c")])
    assert _max_running_goals(cfg) == 2

async def test_unused_goal_budget_returns_to_pool(run_config: RunConfig) -> None:
    # max_usd 10 → work pool 8.5 → 4.25 per goal; each goal: 1 round, 1 agent, one square run (reserves 1.0, settles 0)
    cfg = with_limits(run_config, max_usd=10.0, max_concurrent_goals=1)
    await Swarm(cfg).run([_plan(goal_id="a", rounds=1, agents=1), _plan(goal_id="b", rounds=1, agents=1)])
    assert _share(cfg, "a", round=1)["usd"] == 4.25                        # from round_started.shares
    assert _share(cfg, "b", round=1)["usd"] == 8.5                         # a's unused 4.25 came back to b, the only goal left

async def test_run_returns_empty_list_for_no_result_goal(run_config: RunConfig) -> None:
    assert (await Swarm(run_config).run(_plan(check="_fakes.tools:AlwaysFailCheck")))["g"] == []

async def test_run_timeout_stops_everything(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, run_timeout_s=1.0)
    with anyio.fail_after(25):                                             # 1 s + grace 10 s + margin; the tool sleeps 600 s
        out = await Swarm(cfg).run(_plan(goal_id="g", tool=PyTool({"code": WRITE_THEN_SLEEP})))
    sol = json.loads((_run_dir(cfg) / "goals/g/solution.json").read_text())
    assert (sol["status"], sol["reason"], out["g"]) == ("no_result", "no passing candidate", [])   # only a partial candidate
    assert _events(cfg, "converge_started")[0].payload["select_only"] is True

async def test_queued_goal_no_result_on_run_timeout(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_concurrent_goals=1, run_timeout_s=2.0)
    out = await Swarm(cfg).run([_plan(goal_id="a", tool=PyTool({"code": WRITE_THEN_SLEEP})), _plan(goal_id="b")])
    assert {e.goal_id: e.payload["reason"] for e in _events(cfg, "no_result") if e.goal_id == "b"} == {"b": "run_timeout_s"}
    assert out["b"] == [] and not [e for e in _events(cfg, "goal_started") if e.goal_id == "b"]

async def test_queued_goal_no_result_on_run_budget(run_config: RunConfig) -> None:
    # max_usd 10; goal a's one run reserves 1.0 and reports 9.0 USD, more than the 8.5 work pool → the run budget is used up
    cfg = with_limits(run_config, max_concurrent_goals=1, max_usd=10.0)
    spend = PyTool({"code": WRITE.format("{'data': {'y': 1.0}, 'usage': {'usd': 9.0}}")})
    out = await Swarm(cfg).run([_plan(goal_id="a", rounds=1, agents=1, tool=spend), _plan(goal_id="b", rounds=1, agents=1)])
    assert {e.goal_id: e.payload["reason"] for e in _events(cfg, "no_result") if e.goal_id == "b"} == {"b": "run budget"}
    assert out["b"] == []

async def test_fenced_cancels_inflight_and_raises(run_config: RunConfig) -> None:
    runner = RunRunner(run_config, [_plan(tool=PyTool({"code": WRITE_THEN_SLEEP}))], run_id=None, approve=None, replanner=None)
    raised: list[BaseException] = []
    with anyio.fail_after(30):                                             # the tool sleeps 600 s; only the fence ends the run
        async with anyio.create_task_group() as tg:
            tg.start_soon(_capture, runner.run, raised)                    # appends what run() raises
            store = await _wait_tool_started(run_config)                   # the run store, once a tool_start is committed
            take_over_retrying(store)
    assert [type(e) for e in raised] == [Fenced]                           # a bare Fenced, not an exception group
    assert not [e for e in _committed_events(store) if e.type == "tool_end"]

async def test_base_dir_import_rule(tmp_path: Path) -> None:
    ...  # plan.toml next to mymod.py with ref "mymod:fn" runs; r0 plan_revision payload has base_dir

async def test_every_event_has_context_and_task_ids(run_config: RunConfig) -> None:
    await Swarm(run_config).run(_plan(goal_id="g"))
    events = _all_events(run_config)
    assert all(e.context_id == e.run_id for e in events)
    assert all(e.task_id for e in events if e.type in {"tool_start", "tool_end", "agent_status", "learning"})
```

```python
# tests/_engine/test_swarm.py
async def test_approve_false_writes_nothing(run_config: RunConfig) -> None:
    with pytest.raises(RuntimeError, match="not approved"):
        await Swarm(run_config).run(_plan(), approve=_answer(False))
    assert not Path(run_config.run_root).exists()

async def test_non_asyncio_backend_raises(monkeypatch: pytest.MonkeyPatch, run_config: RunConfig) -> None:
    def refuse() -> None:
        raise RuntimeError("swarm-arena runs on the asyncio backend only")
    monkeypatch.setattr("swarm_arena._engine.run._require_asyncio", refuse)
    with pytest.raises(RuntimeError, match="asyncio"):
        await Swarm(run_config).run(_plan())
    assert not Path(run_config.run_root).exists()

def test_trio_backend_raises(run_config: RunConfig) -> None:
    pytest.importorskip("trio")
    with pytest.raises(RuntimeError, match="asyncio backend only"):
        anyio.run(Swarm(run_config).run, _plan(), backend="trio")

async def test_stop_creates_epoch_key_and_queued_goals_no_result(run_config: RunConfig) -> None:
    cfg = with_limits(run_config, max_concurrent_goals=1)
    swarm = Swarm(cfg)
    slow = PyTool({"code": "import time; time.sleep(2); " + WRITE.format("{'data': {'y': 1.0}}")})
    async with anyio.create_task_group() as tg:
        tg.start_soon(swarm.run, [_plan(goal_id="a", tool=slow), _plan(goal_id="b")])
        await _wait_tool_started(cfg)
        await swarm.stop()
    assert (_run_dir(cfg) / "control/stop-e1").exists()
    assert {e.goal_id: e.payload["reason"] for e in _events(cfg, "no_result") if e.goal_id == "b"} == {"b": "stopped"}

async def test_non_human_origin_is_kept_and_checked(run_config: RunConfig) -> None:
    with pytest.raises(PlanError, match="planner catalog"):
        await Swarm(run_config).run(_plan().model_copy(update={"origin": "llm"}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_engine/test_run.py tests/_engine/test_swarm.py tests/_journal/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `RunRunner.run`:

```text
_require_asyncio()
duplicate goal.id → PlanError(["duplicate goal id '<id>'"])          # before any file or lease exists
for each plan: sys.path.append(plan.base_dir) once; stamp(own origin, parent=None); admit_revision(plan, config, artifacts)
    (a non-human r0 passes the non-importing catalog check before any import or snapshot)
estimate = worst_case_estimate(...); over a set cap → RuntimeError("worst-case estimate ... exceeds ...")
slices = final_slice_estimate per goal; Σ slices > reserve_fraction × cap → RuntimeError("reserve cannot hold the final slices")
warnings += validate warnings + capacity warning when peak demand (Σ agent Resources) > executor.capacity()
approve(estimate, warnings) is False → RuntimeError("run not approved"); run_preflight
run_id = run_id or new_run_id(utc_now()); store = open_store(run_root).child(run_id)
Lease.acquire_new; create run.toml (RunConfig.dumps) and manifest.json; JournalWriter(next_seq=1)
journal run_started {estimate, warnings, manifest, final_slices: {goal: Amount dump}}, run_config {config};
hold_final_slices(the same slices)
per goal in plan order: append(r0_revision_draft(plan, base_dir)); append(goal_queued)
await journal.commit(); write every goals/<goal-id>/plan.r0.toml view; only then start the goal queue
with CancelScope() as run_scope:                       # the fence watcher cancels this outer run scope
  task group tg (watchers only): watch_stop_files; run-clock watcher (run_timeout_s → stops.run_timeout);
    fence watcher: await journal.fenced.wait(), then cancel run_scope (in-flight executor jobs get their grace and
        partial collect; nothing more is committed)
  the host task runs the goal queue inside tg (GoalRunner(..., journal_r0=False)):
    a goal starts when running goals < max_concurrent_goals and running agents + its agents <= max_agents_total, each
        count being GoalRunner.active_agents() (Task 24: a converge team counts before it starts);
        goal_started / goal_finished events; a queued goal that has not started when the run ends early gets
        no_result {reason}: "stopped" (stop or stop --now), "run_timeout_s", or "run budget" (§6.5), then goal_finished
    once every goal has an outcome (each GoalRunner wrote its view, goal_finished, and end_goal): tg.cancel_scope.cancel()
after the task group: journal.fenced set, or an exception group that holds a Fenced → raise a bare Fenced from it
return {goal_id: outcome.finalists}; journal writer exit commits and releases the lease (skipped when fenced, Task 5)
```

`Swarm.run` builds `RunRunner` and sets `self.run_id`; `Swarm.stop` calls `request_stop` for this run; `Swarm.resume` is wired in Task 27.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_engine tests/_journal tests/test_public_api.py tests/test_docstrings.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_engine src/swarm_arena/_journal/replay.py src/swarm_arena/__init__.py tests CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(engine): add Swarm.run with estimate approval, goal queue, shared budget, run clocks, and fencing" -m "A fence watcher cancels the run scope when the journal writer is fenced (spec 7.1), and Swarm.run raises Fenced itself, not an exception group. The watchers are cancelled once every goal has an outcome, so a run ends without waiting for run_timeout_s. Every r0 and the final slices (run_started.final_slices) are committed before any goal starts, so a resume can start a queued goal and replay the slices; R4.14 records both (skipped when present). Every plan goes through admit_revision, so a non-catalog ref of an LLM plan is never imported."
```

### Task 27: Resume

**Files:**
- Create: `src/swarm_arena/_engine/resume.py`
- Modify: `src/swarm_arena/_engine/swarm.py` (`resume`), `src/swarm_arena/_engine/run.py` (accept a resumed state), roadmap R5.4 (replaces item 14 with the [plan] decision: "**Stops on resume:** a goal stopped by `stop` or `stop --now` with rounds left continues on resume; a goal stopped by a limit (`run_timeout_s`, run or goal budget, `reserve_exhausted`) continues only when the resume changes the limits (`limits_changed`). Every goal end returns its unused pool (spec 4.10), live and in `Ledger.replay`. Resume re-opens each continued goal with a journaled `goal_started` `{pool}`: its `goal_weights` share of the free work pool, before that resume's `limits_changed`. It runs a new lifecycle, and its `solution.json` is replaced from the later head"; skipped when the roadmap already has this text)
- Test: `tests/_engine/test_resume.py`

**Interfaces:**
- Consumes: `read_owner`, `Lease.take_over` (with `expect_epoch`), `read_chain`, `RunState` (with `started_commands`), `GoalState` (with the converge fields and `counters` of Task 7), `continues_on_resume` (Task 9), `Ledger.replay`, `Ledger.free/open_goal(pool=)/change_limits` (Task 9), `weighted_shares` (Task 8), `RunBoard.from_state`, `IdGen(counters=state.id_counters)`, `Executor.kill_orphan`, `GoalRunner(state=..., journal_r0=False)`, `RoundRunner(frontier=..., resumed=..., repeat_entries=...)`, `RepeatIndex` with `resumed`, `Watchdog`, `ViewWriter`, `solution_view`, `_rule_max_agents_total` and `PlanError` (Task 16).
- Produces: `resume_run(config, run_id, *, force, replanner) -> RunRunner` (R4.13.13; raises `PlanError`, with the lease released and no event journaled, when `config.limits.max_agents_total` is below the peak of a goal it would start); `Swarm.resume(run_id, *, force=False)` (R4.1). `RunRunner` gains a private keyword `resumed: ResumeState | None = None`, where `ResumeState` (dataclass in `resume.py`) holds `state`, `lease`, `epoch`, `ledger` (the replayed `Ledger`), `orphans_killed: list[str]`. Docstring rule [plan]: the `Swarm.resume` numpydoc `Notes` state the R5.4 item 14 continuation rule and these two sentences: "A resumed stopped goal gets only its share of the free work pool (`Ledger.free()`), which can be 0 when other goals were still running and already got its unused budget; unless the resume raises the limits, such a goal stops again with `goal budget` at its first run." and "A continued goal's earlier lifecycle stays in the run journal: its `final_order` and its `goal_finished` stop reason are kept there, even after the later lifecycle replaces `solution.json` and the replay clears the goal's `end_reason`. `status` and `solution.json` show only the latest lifecycle." This task adds no event payload key that `RunState` folds (Task 7 already pins `goal_started.pool` and the continuation fold), so `replay.py` does not change.

- [ ] **Step 1: Write the failing tests** (each builds a crashed run in-process: a `Swarm.run` task cancelled with its journal writer abandoned without release, or a hand-written journal)

```python
async def test_live_lease_needs_force(crashed_live: Crashed) -> None:
    with pytest.raises(LeaseHeld):
        await Swarm(crashed_live.config).resume(crashed_live.run_id)

async def test_seq_continues_and_takeover_logged(crashed: Crashed) -> None:
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    ev = _events(crashed)
    assert [e.seq for e in ev] == list(range(1, len(ev) + 1))
    take = next(e for e in ev if e.type == "takeover")
    assert take.epoch == 2 and take.seq == crashed.head_seq + 1 and take.payload == {"force": True}

async def test_in_flight_runs_lost_and_charged(crashed: Crashed) -> None:
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    lost = _events(crashed, "run_lost")
    assert [e.payload["candidate_id"] for e in lost] == crashed.in_flight
    assert _settle_for(crashed, lost[0].payload["reservation_id"]) is None        # stays fully charged

async def test_orphan_killed_only_on_host_and_start_time_match(crashed: Crashed) -> None:
    ...  # spy executor: started with matching host_id/start_time → kill_orphan called; mismatch → not killed, run still lost

async def test_completed_runs_not_repeated(crashed: Crashed) -> None:
    ...  # the committed candidate's request makes no second tool_start; the restarted agent gets return_own

async def test_unfinished_round_uses_recorded_frontier(crashed: Crashed, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Frontier | None] = []
    real = RoundRunner.run
    async def spy(self: RoundRunner, round: int) -> RoundOutcome:
        seen.append(self.frontier)                                       # the frontier passed to the constructor
        return await real(self, round)
    monkeypatch.setattr(RoundRunner, "run", spy)
    await Swarm(crashed.config).resume(crashed.run_id, force=True)
    assert seen[0] == crashed.frontier                                   # the epoch-1 frontier of the unfinished round
    assert [e.epoch for e in _events(crashed, "frontier") if e.payload["round"] == crashed.round] == [1]   # none committed again

async def test_takeover_retries_when_old_owner_commits_meanwhile(crashed_live: Crashed, monkeypatch: pytest.MonkeyPatch) -> None:
    real = Lease.take_over.__func__
    attempts: list[int] = []
    def racing(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        attempts.append(1)
        if len(attempts) == 1:
            crashed_live.old_lease.renew()                               # the old owner writes owner.json between read and replace
            raise Fenced("another owner changed the lease")
        return real(cls, store, **kw)
    monkeypatch.setattr(Lease, "take_over", classmethod(racing))
    await Swarm(crashed_live.config).resume(crashed_live.run_id, force=True)
    assert len(attempts) == 2 and read_owner(crashed_live.store)[0].epoch == 2

async def test_concurrent_resume_fails_cleanly(crashed_live: Crashed, monkeypatch: pytest.MonkeyPatch) -> None:
    real = Lease.take_over.__func__
    def other_resumer_first(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        real(cls, store, **{**kw, "owner_id": "other"})                  # a second resume wins: epoch 2
        raise Fenced("another owner changed the lease")
    monkeypatch.setattr(Lease, "take_over", classmethod(other_resumer_first))
    with pytest.raises(Fenced, match="concurrent resume"):
        await Swarm(crashed_live.config).resume(crashed_live.run_id, force=True)
    assert read_owner(crashed_live.store)[0].owner_id == "other"

async def test_resume_after_a_concurrent_takeover_fails_cleanly(crashed_live: Crashed, monkeypatch: pytest.MonkeyPatch) -> None:
    real = Lease.take_over.__func__
    calls: list[int] = []
    def takeover_lands_first(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        calls.append(1)
        if len(calls) == 1:                                              # lands between the authority read and take_over
            real(cls, store, owner_id="other", host="h", ttl_s=60, force=True)
        return real(cls, store, **kw)                                    # the resume's own call, with expect_epoch=1
    monkeypatch.setattr(Lease, "take_over", classmethod(takeover_lands_first))
    with pytest.raises(Fenced, match="concurrent resume"):
        await Swarm(crashed_live.config).resume(crashed_live.run_id, force=True)
    doc = read_owner(crashed_live.store)[0]
    assert (doc.owner_id, doc.epoch) == ("other", 2)                     # not 3: the resume wrote nothing

async def test_completed_run_without_settle_stays_charged_in_full(crashed_tool_end_no_settle: Crashed) -> None:
    # hand-written journal: reserve r1, tool_start c1, tool_end c1, no settle
    c = crashed_tool_end_no_settle
    runner = await resume_run(c.config, c.run_id, force=True, replanner=None)
    assert not [e for e in _events(c, "run_lost") if e.payload["candidate_id"] == "c1"]
    assert "c1" in runner.resumed.state.candidates
    assert runner.resumed.ledger.remaining("g/t/a") == c.share - c.r1_amount     # r1 charged in full
    await runner.run()

async def test_resumed_round_keeps_agent_shares(crashed_share_used: Crashed) -> None:
    # crashed in round 1 after agent a used its whole share; b made committed charges
    c = crashed_share_used
    runner = await resume_run(c.config, c.run_id, force=True, replanner=None)
    assert runner.resumed.ledger.remaining("g/t/a") == Amount()
    assert runner.resumed.ledger.remaining("g/t/b") == c.share - c.b_charged
    await runner.run()
    assert c.recorders["a"].limits_hit[:1] == ["budget"]                # a's restarted run_tool raised LimitReached("budget")
    assert len([e for e in _events(c, "round_started") if e.payload["round"] == 1]) == 1

async def test_queued_goal_starts_from_journaled_r0(crashed_with_queued: Crashed) -> None:
    # max_concurrent_goals=1: goal a crashed in round 1, goal b still queued
    await Swarm(crashed_with_queued.config).resume(crashed_with_queued.run_id, force=True)
    assert [e.epoch for e in _events(crashed_with_queued, "goal_started") if e.goal_id == "b"] == [2]
    assert len([e for e in _events(crashed_with_queued, "plan_revision") if e.goal_id == "b"]) == 1
    assert crashed_with_queued.store.exists("goals/b/solution.json")

async def test_stopped_goal_with_rounds_left_continues(crashed_stopped_rounds_left: Crashed) -> None:
    # stop after round 1 of 3; the lease was released; resume without force
    await Swarm(crashed_stopped_rounds_left.config).resume(crashed_stopped_rounds_left.run_id)
    ev = _events(crashed_stopped_rounds_left)
    assert [e.payload["round"] for e in ev if e.type == "round_closed"] == [1, 2, 3]
    assert len([e for e in ev if e.type == "converge_started"]) == 2 and len([e for e in ev if e.type == "final_order"]) == 2
    assert _goal_status(crashed_stopped_rounds_left, "g") == "finished"
    view = json.loads(crashed_stopped_rounds_left.store.read("goals/g/solution.json").data)
    assert view["source_head"]["epoch"] == 2 and view["reason"] is None
    gs = [e for e in ev if e.type == "goal_started"]
    assert [e.epoch for e in gs] == [1, 2] and "pool" in gs[-1].payload             # re-opened from the free pool

async def test_two_continued_goals_split_the_free_pool_once(crashed_two_stopped: Crashed) -> None:
    # goals g1, g2 (goal_weights {"g1": 3.0, "g2": 1.0}) both stopped by "stop" after round 1 of 3; lease released;
    # the replayed free() is Amount(usd=40.0)
    c = crashed_two_stopped
    runner = await resume_run(c.config, c.run_id, force=False, replanner=None)
    assert runner.resumed.ledger.free() == Amount()                         # 30 + 10: one split of the same free()
    await runner.run()
    pools = {e.goal_id: Amount.model_validate(e.payload["pool"]) for e in _events(c, "goal_started") if e.epoch == 2}
    assert pools == {"g1": Amount(usd=30.0), "g2": Amount(usd=10.0)}        # not g2 = 2.5, a share of what g1 left

async def test_crash_in_continued_round_resumes_as_running(crashed_in_continued_round: Crashed) -> None:
    # stop after round 1 of 3, resume at epoch 2, then a crash in round 2 after agent a's committed candidate for
    # request X; after this resume a's recorder requests only X
    c = crashed_in_continued_round
    runner = await resume_run(c.config, c.run_id, force=True, replanner=None)
    assert {h: runner.resumed.ledger.remaining(h) for h in c.live_remaining} == c.live_remaining
    await runner.run()
    assert len([e for e in _events(c, "round_started") if e.payload["round"] == 2]) == 1
    assert [e.epoch for e in _events(c, "frontier") if e.payload["round"] == 2] == [2]
    assert not [e for e in _events(c, "tool_start") if e.epoch == 3 and e.agent == "a"]    # return_own
    assert [e.epoch for e in _events(c, "goal_started")] == [1, 2]                 # no goal_started again at epoch 3
    assert c.recorders["a"].limits_hit == [] and _goal_status(c, "g") == "finished"

async def test_resume_never_links_other_agents_same_round_result(crashed_same_round_pair: Crashed) -> None:
    # hand-written journal: unfinished round 1, a's committed ok candidate for {"x": 1.0}, b has not run it;
    # after resume both recorders request {"x": 1.0}
    c = crashed_same_round_pair
    await Swarm(c.config).resume(c.run_id, force=True)
    assert [e.agent for e in _events(c, "tool_start") if e.epoch == 2] == ["b"]     # a: return_own; b runs its own
    assert c.recorders["b"].limits_hit == [] and c.recorders["b"].soft_errors == []

async def test_recovered_candidates_are_earlier_round_entries_next_round(crashed_two_rounds_recovered: Crashed) -> None:
    # hand-written journal, 2 rounds, one team t with KeepTopTeams(1) (nothing pruned), unfinished round 1: committed ok
    # candidates c.cx (a, {"x": 1.0}), c.cy (a, {"x": 2.0}), c.cz (n, {"x": 3.0}, replicate 0); a and b use square, n uses
    # FunctionTool("_fakes.tools:square", deterministic=False) with replicates=2; b has run nothing. After resume the
    # recorders request, round 1: a x=1.0 then x=2.0, b x=1.0, n nothing; round 2: a x=1.0, b x=2.0, n x=3.0
    c = crashed_two_rounds_recovered
    await Swarm(c.config).resume(c.run_id, force=True)
    assert [(e.round, e.agent) for e in _events(c, "tool_start") if e.epoch == 2] == [(1, "b"), (2, "n")]
    assert [r.id for r in c.recorders["a"].results] == [c.cx, c.cy]                  # round 1: return_own, twice
    assert [(r.agent, r.round) for r in c.recorders["b"].results] == [("b", 1), ("a", 1)]   # own run (§6.2), then a link
    assert c.recorders["b"].results[1].id == c.cy and c.recorders["b"].soft_errors == []   # round 2: linked to a's c.cy
    assert [m.split(":")[0] for m in c.recorders["a"].soft_errors] == ["repeat refused"]    # round 2: a's own x=1.0
    assert [e.payload["refused"] for e in _events(c, "watchdog") if "refused" in e.payload] == [c.cx]
    assert [r.request.replicate for r in c.recorders["n"].results] == [1]          # replicate 0 of c.cz stays used

@pytest.mark.parametrize("raise_cap,rounds_after", [(False, []), (True, [2])])
async def test_limit_stopped_goal_continues_only_with_raised_limits(crashed_budget_stopped: Crashed, raise_cap: bool,
                                                                    rounds_after: list[int]) -> None:
    # goal g stopped with reason "run budget" after round 1 of 2
    cfg = crashed_budget_stopped.config
    if raise_cap:
        cfg = cfg.model_copy(update={"limits": cfg.limits.model_copy(update={"max_usd": cfg.limits.max_usd * 2})})
    await Swarm(cfg).resume(crashed_budget_stopped.run_id)
    started = [e.payload["round"] for e in _events(crashed_budget_stopped, "round_started") if e.epoch == 2]
    assert started == rounds_after
    if raise_cap:                                                                    # re-open first, then the raise
        reopen = next(e for e in _events(crashed_budget_stopped, "goal_started") if e.epoch == 2)
        assert reopen.seq < _events(crashed_budget_stopped, "limits_changed")[0].seq

async def test_continued_goal_waits_for_a_goal_slot(crashed_continued_behind_running: Crashed) -> None:
    # max_concurrent_goals=1, 2 rounds per goal: goal a stopped with "goal budget" after round 1, then the queued goal b
    # took the slot and crashed in its round 1; this resume doubles max_usd, so a continues
    c = crashed_continued_behind_running
    cfg = c.config.model_copy(update={"limits": c.config.limits.model_copy(update={"max_usd": c.config.limits.max_usd * 2})})
    await Swarm(cfg).resume(c.run_id, force=True)
    ev = _events(c)
    b_done = next(e.seq for e in ev if e.type == "goal_finished" and e.goal_id == "b")
    a_rounds = [e.seq for e in ev if e.type == "round_started" and e.goal_id == "a" and e.epoch == 2]
    assert a_rounds and b_done < min(a_rounds)                                     # a starts only once b frees the slot
    assert [e.epoch for e in ev if e.type == "goal_started" and e.goal_id == "a"] == [1, 2]   # the re-open only; admission adds none
    assert (_goal_status(c, "a"), _goal_status(c, "b")) == ("finished", "finished")

@pytest.mark.parametrize("max_goals,max_agents", [(1, 1), (2, 1)])
async def test_waiting_continued_goal_converges_on_early_end(crashed_continued_behind_running: Crashed,
                                                             monkeypatch: pytest.MonkeyPatch, max_goals: int,
                                                             max_agents: int) -> None:
    # the resume above plus max_concurrent_goals=max_goals, max_agents_total=max_agents; the barrier holds b's in-flight
    # round-1 run until the driver has seen the stop, so a still waits for admission when the run ends early. (1, 1): both
    # caps hold a back; (2, 1): a goal slot is free, so the agent cap alone holds a back
    c = crashed_continued_behind_running
    hold = _HoldExecutor(LocalExecutor())
    cfg = with_limits(c.config_with(hold), max_usd=c.config.limits.max_usd * 2, max_concurrent_goals=max_goals,
                      max_agents_total=max_agents)
    goals, agents = _Gauge(), _Gauge()
    real_goal, real_round, real_close = GoalRunner.run, _Recorder.run_round, Inflight.close
    async def counted_goal(self: GoalRunner) -> GoalOutcome:
        with goals.held(self.goal_id):
            return await real_goal(self)
    async def counted_round(self: _Recorder, ctx: AgentContext) -> RoundReport:
        with agents.held(ctx.spec.name):
            return await real_round(self, ctx)
    closed = anyio.Event()
    def signalling_close(self: Inflight) -> None:
        real_close(self); closed.set()
    monkeypatch.setattr(GoalRunner, "run", counted_goal)
    monkeypatch.setattr(_Recorder, "run_round", counted_round)
    monkeypatch.setattr(Inflight, "close", signalling_close)
    with anyio.fail_after(60):
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(Swarm(cfg).resume, c.run_id, force=True))
            await hold.entered.wait()                                  # b's run is in flight; a waits for admission
            assert await request_stop(cfg.run_root, c.run_id, now=False) == 2
            await closed.wait()                                        # the driver latched the stop; b's admission closed
            hold.release.set()
    assert agents.peak == 1                                            # max_agents_total=1 in both cases
    if max_goals == 1:
        assert goals.peak == 1                                         # (2, 1): the goal spans may overlap, not checked
    assert goals.entered == ["b", "a"] and agents.entered == ["b", "cb", "ca"]   # a: no main round, its converge team only
    a2 = [e for e in _events(c) if e.goal_id == "a" and e.epoch == 2]
    assert [e.type for e in a2 if e.type in {"round_closed", "converge_started", "final_order", "goal_finished"}] == [
        "converge_started", "final_order", "goal_finished"]
    assert next(e for e in a2 if e.type == "final_order").payload["candidate_ids"]
    assert next(e for e in a2 if e.type == "goal_finished").payload["reason"] == "stop"
    assert not [e for e in _events(c, "no_result") if e.goal_id == "a"]

async def test_finished_goal_only_repairs_view(crashed_before_view: Crashed) -> None:
    # crashed after the goal_finished commit, before the solution.json view
    head = crashed_before_view.head_seq
    await Swarm(crashed_before_view.config).resume(crashed_before_view.run_id, force=True)
    assert crashed_before_view.store.exists("goals/g/solution.json")
    assert not [e for e in _events(crashed_before_view) if e.seq > head and e.goal_id == "g" and e.type.startswith("goal_")]

async def test_round_counters_survive_resume(crashed_counters: Crashed) -> None:
    # hand-written journal, unfinished round 1: a's committed agent_counters {turns: 19, board_reads: 10}, b's
    # {turns: 0, board_reads: 10}; after resume a's recorder makes two requests (distinct inputs), b's one search_learnings
    c = crashed_counters
    await Swarm(c.config).resume(c.run_id, force=True)
    assert len([e for e in _events(c, "tool_start") if e.agent == "a" and e.epoch == 2]) == 1   # turn 20 runs
    assert c.recorders["a"].limits_hit == ["max_turns_per_round"]                              # turn 21 raises
    assert any("max_board_reads_per_round" in m for m in c.recorders["b"].soft_errors)

@pytest.mark.parametrize("same_host", [True, False])
async def test_orphan_command_job_killed_on_resume(crashed_in_command: Crashed, same_host: bool) -> None:
    # crashed during a ScriptJudge command with a committed command_start and no command_end
    c = crashed_in_command
    spy = c.spy_executor(same_host=same_host)            # records the Started values whose process kill_orphan killed
    await Swarm(c.config_with(spy)).resume(c.run_id, force=True)
    assert spy.killed == ([c.command_started] if same_host else [])
    assert _events(c, "run_lost") == []                  # a command job is not a tool run

async def test_resume_during_converge(crashed_in_converge: Crashed) -> None:
    # crashed after round_closed of round 2 (of 2) and converge_started {select_only: false}, before final_order
    await Swarm(crashed_in_converge.config).resume(crashed_in_converge.run_id, force=True)
    ev = _events(crashed_in_converge)
    assert [e.payload["round"] for e in ev if e.type == "round_closed"] == [1, 2]       # no main round runs again
    assert [e.payload["select_only"] for e in ev if e.type == "converge_started"] == [False, False]
    assert _goal_status(crashed_in_converge, "g") == "finished" and len(_final_order(crashed_in_converge)) == 2

async def test_watchdog_counters_rebuilt(crashed_failing: Crashed) -> None:
    # before the crash, agent a made 2 committed error runs in round 1; its third failure after resume stops it
    await Swarm(crashed_failing.config).resume(crashed_failing.run_id, force=True)
    st = [e for e in _events(crashed_failing, "agent_status") if e.agent == "a" and e.round == 1]
    assert (st[-1].payload["status"], st[-1].payload["reason"]) == ("watchdog_stopped", "3 consecutive failures")
    assert len([e for e in _events(crashed_failing, "tool_end") if e.agent == "a"]) == 3

async def test_limits_changed_journaled(crashed: Crashed) -> None:
    cfg = crashed.config.model_copy(update={"limits": crashed.config.limits.model_copy(update={"max_rounds": 3})})
    await Swarm(cfg).resume(crashed.run_id, force=True)
    assert _events(crashed, "limits_changed")[0].payload["new"]["max_rounds"] == 3

async def test_resume_refuses_max_agents_total_below_a_goal_peak(crashed: Crashed) -> None:
    with pytest.raises(PlanError) as err:                                 # goal g's peak is 2
        await Swarm(with_limits(crashed.config, max_agents_total=1)).resume(crashed.run_id, force=True)
    assert err.value.errors == ["goal 'g': max_agents_total: peak 2 agents + 0 active in other goals > 1"]
    assert [e.seq for e in _events(crashed)] == list(range(1, crashed.head_seq + 1))   # no event, not even takeover
    assert read_owner(crashed.store)[0].expires_at < utc_now()            # lease released

async def test_old_stop_file_ignored(crashed_stopped: Crashed) -> None:
    await Swarm(crashed_stopped.config).resume(crashed_stopped.run_id, force=True)
    assert _goal_status(crashed_stopped, "g") == "finished"             # control/stop-e1 exists; epoch 2 ignores it

async def test_base_dir_restored(crashed_from_file: Crashed, monkeypatch: pytest.MonkeyPatch) -> None:
    # the plan file sits in crashed_from_file.plan_dir next to mymod.py and uses "mymod:fn"
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(crashed_from_file.plan_dir)])
    monkeypatch.delitem(sys.modules, "mymod", raising=False)
    await Swarm(crashed_from_file.config).resume(crashed_from_file.run_id, force=True)
    assert str(crashed_from_file.plan_dir) in sys.path and _goal_status(crashed_from_file, "g") == "finished"
```

The `Crashed` fixtures (dataclass `Crashed` in the test module: `config`, `run_id`, `store`, `head_seq`, `round`, `frontier`, `in_flight`, `old_lease`, `plan_dir`, plus the per-fixture fields the tests name: `share`, `r1_amount`, `b_charged`, `recorders`, `command_started`, `spy_executor(same_host)`, `config_with(executor)`, `live_remaining: dict[str, Amount]` (the epoch-2 live ledger right before the crash)) hand-write or crash runs: `crashed` (expired lease; goal `g` with one main team of two agents and a converge team of at most two, so its peak is 2; one committed and one in-flight run in round 1), `crashed_live` (live lease; `old_lease` is the old owner's `Lease`), `crashed_stopped`, `crashed_in_converge`, `crashed_failing`, `crashed_from_file`, `crashed_tool_end_no_settle`, `crashed_share_used`, `crashed_with_queued` (`max_concurrent_goals=1`), `crashed_stopped_rounds_left` (stop after round 1 of 3, lease released), `crashed_two_stopped` (hand-written: goals g1 and g2, `goal_weights={"g1": 3.0, "g2": 1.0}`, both ended with `goal_finished {"reason": "stop"}` after round 1 of 3, lease released, balances such that the replayed `free()` is `Amount(usd=40.0)`), `crashed_budget_stopped`, `crashed_before_view`, `crashed_counters`, `crashed_in_command`, `crashed_in_continued_round` (stop after round 1 of 3, resume at epoch 2, crash in round 2 after agent a's committed candidate for request X), `crashed_same_round_pair` (hand-written: unfinished round 1, a's committed ok candidate for `{"x": 1.0}`, b has not run it), `crashed_two_rounds_recovered` (hand-written as its test says; fields `cx`, `cy`, `cz`: the candidate ids), `crashed_continued_behind_running` (`max_concurrent_goals=1`, expired lease: goal a ended with `goal_finished {"reason": "goal budget"}` after round 1 of 2, then goal b's `goal_started` and an unfinished round 1 with no committed candidate of b, so b's restarted round makes one new run; each goal has one main team `t` with one agent (`a`, `b`) and a converge team `conv` with one agent (`ca`, `cb`), every agent a `_Recorder` with one `square` request per round, all judges `ProgramJudge`s; `config_with(executor)`). Test-local helpers of the early-end test: `_HoldExecutor(inner)` is an `Executor` wrapper like Task 22's `_GateExecutor`, but it holds after the job: `submit` awaits `inner.submit(job)` (so the job's `tool_start` is committed), sets `entered: anyio.Event`, awaits `release: anyio.Event`, and returns the output, and every other method delegates; `_Gauge` (dataclass: `now: int`, `peak: int`, `entered: list[str]`) has the context manager `held(key: str)`, which appends `key` to `entered`, adds 1 to `now`, sets `peak = max(peak, now)`, and subtracts 1 on exit; `Inflight` comes from `swarm_arena._agent.tool_run`, `request_stop` from `swarm_arena._engine.control`, `with_limits` from `_fakes.engine`. `recorders` maps an agent name to its test-local `_Recorder(Manager)`, which runs its requests (one list, or a `{round: [...]}` map) through `ctx.run_tool` (or `ctx.search_learnings`), appends each returned `Candidate` to `results`, `err.limit` of a `LimitReached` to `limits_hit` (then returns), and `str(err)` of a `SoftError` to `soft_errors`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_engine/test_resume.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `resume_run`:

```text
loop:                                                      # §7.1 "the new owner starts again at step 1"
    doc0 = read_owner(store)                               # authority read
    try: lease = Lease.take_over(store, ..., force=force, expect_epoch=doc0.epoch); break
    except Fenced: doc1 = read_owner(store)
        doc1.epoch == doc0.epoch → retry (the old owner advanced its head or renewed)
        else → raise Fenced(f"run {run_id}: a concurrent resume took the run (epoch {doc1.epoch})")
        (expect_epoch makes a takeover that lands between the authority read and take_over take this branch)
read_chain → RunState.from_events
continued = [g for g in state.goals if continues_on_resume(state, g, config.limits)]
for each queued or running goal and each continued goal g: errors += [f"goal '{g}': {e}" for e in
    _rule_max_agents_total(state.plans[g][-1], config, (), other_active_agents=0)]   # Task 16, this resume's limits
errors → lease.release(); raise PlanError(errors)          # before any event: every goal that starts fits an empty queue
JournalWriter(next_seq=head.seq + 1) → takeover {force}
ledger = Ledger.replay(state, journal=..., ids=...)         # history only (Task 9)
→ RunBoard.from_state → for each state.started: await executor.kill_orphan(started), then run_lost {candidate_id, reservation_id}
    → for each state.started_commands entry: await executor.kill_orphan(started) (no run_lost: a command job is no tool run)
shares = weighted_shares(ledger.free(), {c: goal_weights weight of c (equal by default) for c in continued})
                                                           # once, before any re-open: each open_goal(pool=) makes free() smaller
for g in continued: ledger.open_goal(g, pool=shares[g]); append goal_started {pool: shares[g]}
config.limits != state.config.limits (the latest journaled limits) → ledger.change_limits(config.limits);
    append limits_changed {old, new}                       # after the re-opens, so the re-opened goals share the raise
base dirs from r0 payloads onto sys.path
then the goals start through the Task 26 goal queue: first the goals running at the crash, then the continued goals, then
    the queued goals, each group in plan order; each starts only when the Task 26 admission check passes (running goals
    < max_concurrent_goals and running agents + its agents <= max_agents_total, each count GoalRunner.active_agents(),
    so a converge team counts before it starts), so a continued goal waits for a slot like a queued goal (the check above
    makes every goal fit an empty queue, so each one starts once the goals before it end); the queue writes no second
    goal_started for a running or continued goal (the continued one's is above).
    An early end (stop, stop --now, run_timeout_s, run budget) changes no admission rule. Only a queued goal (no
    goal_started) gets the Task 26 no_result; a waiting running or continued goal keeps its candidates and its place,
    and starts through the same check when a running goal ends. No StopState flag is cleared during a run, so once it
    starts, its round loop breaks before its next round, and it converges and ends like a goal stopped between rounds
    (§6.5): final_order (or no_result "no passing candidate") and goal_finished {reason}. A no_result {reason:
    "stopped"} would discard its candidates (and the continuation fold reset a continued goal's final_order)
goal lifecycle rules (GoalState; every continued goal gets GoalRunner(state=..., journal_r0=False)):
    finished or no_result → view repair only: write solution.json from the current head when it is missing or older; no goal event
    queued (no goal_started) → starts from RunState.plans[g][0] and its r0 base_dir (the r0 is already journaled, Task 26)
    running, round loop not done → GoalRunner continues from rounds_closed; an unfinished round (no round_closed for
        GoalState.round) restarts with RoundRunner(frontier=frontiers[round], resumed=each agent's committed
        (repeat_keys[c.id], c) pairs of that round, repeat_entries as in Task 24 (frontier-covered candidates only)); the
        recovered pairs stay in their owners' indexes during that round and join GoalRunner.repeat_entries when it ends,
        or at the early break before it runs (Task 24, deduplicated by candidate id); no open_round, round_started, or frontier
        again (replayed shares); frontiers[round] always exists, because round_started and its frontier always commit
        in one segment (Task 22)
    running, every round closed or converge_started, no final_order → converge again with the journaled select_only,
        join_attempts, and kept_parts candidates; a converge round with committed candidates restarts like an unfinished round
    running, final_order and no goal_finished → goal_finished, commit, solution view, ledger.end_goal
    stopped and continues_on_resume (Task 9: rounds left and reason "stop" / "stop --now", or any stop reason when this resume
        journaled limits_changed) → continues at rounds_closed + 1 with a new lifecycle (rounds, barrier, policy, converge,
        final judge, a new final_order and goal_finished; ViewWriter replaces solution.json from the later head);
        re-opened by the journaled goal_started {pool} above; its rounds are paid from that pool; once that event is
        committed the fold makes it running (Task 7), so a crash in the new lifecycle resumes by the running rules
    any other stopped goal → view repair only
```

Kept counters of the unfinished round [plan]: `tool_runs` (from `tool_start` and `run_lost`) and `learnings`, counted from the committed events; `turns` and `board_reads` from `GoalState.counters` (the last committed `agent_counters`; counts after the last commit are lost, like every ordinary event, §7.1); the watchdog state, replayed into a fresh `Watchdog` per agent from its committed events in seq order (`record_run` per `tool_end` status, `record_refusal` per `watchdog {"refused"}`, `end_round` per closed round).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_engine -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_engine tests/_engine docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(engine): resume a run by takeover, replay, lost runs, and verified orphan kill" -m "A takeover that loses to a commit of the old owner retries from the authority read (spec 7.1); expect_epoch turns a takeover that lands in between into a clean concurrent-resume error. Turns and board reads come back from agent_counters (spec 6.5), and orphan command jobs are killed too. Each continued goal is re-opened from the replayed free pool with a journaled goal_started {pool} before this resume's limits_changed, so a later resume replays the same balances and a crash in the new lifecycle resumes by the running rules. A restarted round links only frontier-covered candidates, so another agent's same-round result is never linked (spec 6.2); its recovered candidates join the goal's repeat entries when it ends, so the next round refuses the owner's repeat, links another agent's, and keeps their replicate indices used. Running and continued goals start only through the goal queue's admission check (max_concurrent_goals, max_agents_total, a converge team counted before it starts), also after an early end: a waiting goal keeps its candidates, and with the stop flags latched it goes straight to converge once admitted. A resume that lowers max_agents_total below the peak of a goal it would start raises PlanError before any event and releases the lease, so every goal still fits an empty queue (Task 16). Roadmap R5.4 item 14 is replaced with the stopped-goal continuation rule: a user stop with rounds left continues, a limit stop continues only after the limits change, and every goal end returns its unused pool (skipped when present)."
```

### Task 28: CLI `run`, `status`, `stop`, `resume`

**Files:**
- Create: `src/swarm_arena/_cli/run_cmds.py`
- Modify: `src/swarm_arena/_cli/main.py` (register the four subcommands), `CHANGELOG.md`
- Test: `tests/_cli/test_run_cmds.py`, `tests/_cli/test_main.py`

**Interfaces:**
- Consumes: `Swarm`, `RunConfig.load`, `Plan.load`, `read_status`, `format_status`, `request_stop`, `PlanError`.
- Produces: `register(sub: argparse._SubParsersAction) -> None` (R4.13.18). Commands [plan]: `swarm-arena run [--yes] RUN_TOML PLAN [PLAN ...]` (a `PLAN` that is a directory means every `*.toml` in it, sorted); `swarm-arena status RUN_DIR`; `swarm-arena stop [--now] RUN_DIR`; `swarm-arena resume RUN_DIR [--config RUN_TOML] [--force]`. `RUN_DIR` is `<run_root>/<run-id>`; with no `--config`, resume uses the run's own `run.toml`. Exit codes: 0 ok, 1 error (message on stderr; a `PlanError` prints one `- <error>` line each), 2 usage.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_yes_prints_estimate_and_run_dir(example: Example, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--yes", str(example.run_toml), str(example.plan_toml)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("worst-case estimate: ") and re.match(r"run \S+ at \S+", out[-2])

class _Stdin:                                                     # test-module stand-in for sys.stdin
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty
    def isatty(self) -> bool:
        return self._tty
    def readline(self) -> str:
        return ""

def test_run_without_tty_or_yes_exits_2_and_writes_nothing(example: Example, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=False))
    assert main(["run", str(example.run_toml), str(example.plan_toml)]) == 2
    assert not example.run_root.exists()

def test_run_prompt_declined_exits_1(example: Example, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=True)); monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main(["run", str(example.run_toml), str(example.plan_toml)]) == 1 and not example.run_root.exists()

def test_run_accepts_plans_dir(example_two_goals: Example, capsys: pytest.CaptureFixture[str]) -> None:
    # example_two_goals.plans_dir holds b.toml (goal "b"), a.toml (goal "a"), and notes.txt
    assert main(["run", "--yes", str(example_two_goals.run_toml), str(example_two_goals.plans_dir)]) == 0
    done = [line.split()[1] for line in capsys.readouterr().out.splitlines() if line.startswith("done: ")]
    assert done == ["a", "b"]                                            # every *.toml in the dir, sorted; notes.txt ignored
def test_plan_error_lists_every_error(example_bad: Example, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--yes", ...]) == 1 and capsys.readouterr().err.count("\n- ") == 2

def test_status_stop_resume_dispatch(finished_run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", str(finished_run)]) == 0 and capsys.readouterr().out.startswith("run ")
    assert main(["stop", str(finished_run)]) == 1                       # no live owner

def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert all(c in capsys.readouterr().out for c in ("run", "status", "stop", "resume"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_cli -v`
Expected: FAIL with `ModuleNotFoundError: ... run_cmds`.

- [ ] **Step 3: Implement** `run_cmds.py`. Every handler calls `anyio.run(..., backend="asyncio")`. Output [plan]: `worst-case estimate: <tokens> tokens, $<usd:.2f>` (`unlimited` for an unset unit), one `warning: <text>` line per warning, prompt `Start the run? [y/N] `, then `run <run-id> at <run-dir>` at start and `done: <goal-id> <n> finalists` (or `no result: <goal-id> <reason>`) per goal at the end, in plan order (for a plans directory, sorted file order). The approval callback is a terminal prompt (only `y` or `yes` approves); with no TTY and no `--yes`, exit 2 before any file is written.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_cli -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_cli tests/_cli CHANGELOG.md
git commit -m "feat(cli): add swarm-arena run, status, stop, and resume"
```

### Task 29: Rosenbrock example, Getting started, README quickstart

**Files:**
- Create: `examples/rosenbrock/rosen.py`, `examples/rosenbrock/plan.toml`, `examples/rosenbrock/run.toml`, `examples/rosenbrock/README.md`, `tests/e2e/test_rosenbrock.py`
- Modify: `docs/getting-started.md` (M1a stub), `README.md`, `.gitignore` (add `.swarm-arena/`)

**Interfaces:**
- Consumes: the public API only (`swarm_arena`, `swarm_arena.scoring`, `swarm_arena.board`).
- Produces (`rosen.py`, module level, stdlib only): `gradient_descent(lr: float, steps: int, x0: float, y0: float) -> dict[str, float]` (`{"x", "y", "f"}`); `random_search(n: int, radius: float, seed: int | None = None) -> dict[str, float]`; `class BoundsCheck(Check)` (`name = "bounds"`, `Config.bound: float = 10.0`; passes when `x`, `y`, `f` are finite and `|x|, |y| <= bound`); `score(goal: Goal, candidate: Candidate) -> Score` (`Score.minimize(data["f"])`). `plan.toml` (§12): goal `id = "rosenbrock"`, `namespace = "toy"`; team `descent` (agents `gd-1`, `gd-2`, `FunctionTool` `rosen:gradient_descent`, `ScriptedManager` with two requests each, the second with `parents = "best"`); team `random` (agents `rs-1`, `rs-2`, `FunctionTool` `rosen:random_search` with `deterministic = false`, `replicates = 2`); `checks = [{path = "rosen:BoundsCheck"}]`; `judges = [{name = "f", path = "swarm_arena:ProgramJudge", config = {function = "rosen:score"}}]`; `policy = {path = "swarm_arena:SuccessiveHalving", config = {keep = 0.5}}`; `converge = {mode = "select", top_n = 2}`. `run.toml`: `[limits] max_rounds = 2`, roots under `.swarm-arena/`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/e2e/test_rosenbrock.py (public imports only)
async def test_rosenbrock(tmp_path: Path) -> None:
    cfg = _with_roots(RunConfig.load(EXAMPLE / "run.toml"), tmp_path)
    out = await Swarm(cfg).run(Plan.load(EXAMPLE / "plan.toml"))
    sol = json.loads((Path(cfg.run_root) / _only_run(cfg) / "goals/rosenbrock/solution.json").read_text())
    assert sol["status"] == "ok" and len(sol["finalists"]) == 2 == len(out["rosenbrock"])
    vals = [f["scores"][0]["value"] for f in sol["finalists"]]
    assert vals == sorted(vals) and len({f["candidate"]["id"] for f in sol["finalists"]}) == 2
    assert {f["candidate"]["team"] for f in sol["finalists"]} <= {"descent", "random"}
    assert _decision_events(cfg)                                      # SuccessiveHalving ran with reasons

def test_rosenbrock_cli(tmp_path: Path) -> None:
    shutil.copytree(EXAMPLE, tmp_path / "rosenbrock")
    res = subprocess.run(["swarm-arena", "run", "--yes", "rosenbrock/run.toml", "rosenbrock/plan.toml"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert "done: rosenbrock 2 finalists" in res.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/e2e/test_rosenbrock.py -v`
Expected: FAIL with `FileNotFoundError` for `examples/rosenbrock/run.toml`.

- [ ] **Step 3: Write the example and docs.** `docs/getting-started.md`: install (`uv add swarm-arena`), the picture of §1, the example through `literalinclude` of `rosen.py`, `plan.toml`, and `run.toml` (so the page cannot drift), the `swarm-arena run --yes ...` command, reading `solution.json`, `status`/`stop`/`resume`, and the one-line switch to `AgentsSDKManager` marked "needs the `llm` extra (M3)". `README.md`: replace "under construction" with the pitch, the picture, and an offline quickstart (the two commands of the exit criteria). `examples/rosenbrock/README.md`: how to run it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/e2e/test_rosenbrock.py -v && uv run swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`, then the lint trio.
Expected: PASS; the CLI prints `done: rosenbrock 2 finalists`; Sphinx builds with no warnings.

- [ ] **Step 5: Commit**

```bash
git add examples/rosenbrock tests/e2e/test_rosenbrock.py docs/getting-started.md README.md .gitignore
git commit -m "docs: add the offline Rosenbrock example, Getting started, and the README quickstart"
```

### Task 30: Bring-your-own-script example

**Files:**
- Create: `examples/own_script/problem/model_a.txt` (`slope = 1.0`), `problem/model_b.txt` (`intercept = 0.0`), `problem/run.py`, `checks/check.py`, `judge/score.py`, `judge/hidden/points.csv`, `plan.toml`, `run.toml`, `README.md`; `tests/e2e/test_own_script.py`
- Modify: `docs/own-files.md` (the M1a page "Use your own files and script": add a "Run it" section that `literalinclude`s this example), `README.md` (one line linking the example)

**Interfaces:**
- Consumes: public API; the §4.12 workspace contract (M1a).
- Produces: `plan.toml` with goal `id = "line-fit"`, `namespace = "toy"`, `files = ["problem/*.txt", "problem/run.py"]`; one team `fit` with agents `tuner-1` (round 1: `edits = {"problem/model_a.txt" = "slope = 2.0\n"}`) and `tuner-2` (round 2: `parents = "best"`, `edits = {"problem/model_b.txt" = "intercept = 1.0\n"}`), both `ScriptTool` with `command = ["python", "problem/run.py"]`, `editable = ["problem/model_*.txt"]`, `params = {scale = {type = "number", minimum = 0.1, maximum = 10.0, default = 1.0}}` (the default keeps `scale` optional: the M1a schema compiler makes a param with no default required, and neither scripted request sets inputs); `ScriptCheck` (`checks/check.py`); `ScriptJudge` `name = "mse"` (`judge/score.py`, `files = ["judge/score.py", "judge/hidden/*.csv"]`, `direction = "min"`); `policy = {path = "swarm_arena:KeepTopTeams", config = {k = 1}}` (with one team it prunes nothing, so `tuner-2`, which has no round-1 request and so no candidate, still runs in round 2; the default `SuccessiveHalving` would prune it); converge select `top_n = 1`. `run.toml`: `[limits] max_rounds = 2`, roots under `.swarm-arena/`. Scripts are stdlib only: `run.py` reads both model files and `swarm/request.json`, writes `{"data": {"slope", "intercept"}}`; `score.py` reads `candidate/output.json` and `judge/hidden/points.csv`, writes `{"value": <mean squared error>}`.

- [ ] **Step 1: Write the failing test**

```python
async def test_own_script(tmp_path: Path) -> None:
    cfg = _with_roots(RunConfig.load(EXAMPLE / "run.toml"), tmp_path).model_copy(update={"executor": Ref(
        path="swarm_arena.executor:LocalExecutor", config={"workspace_root": str(tmp_path / "ws"), "keep_workspaces": True})})
    await Swarm(cfg).run(Plan.load(EXAMPLE / "plan.toml"))
    c2 = _candidate_of(cfg, agent="tuner-2", round=2)
    assert set(c2.overlay) == {"problem/model_a.txt", "problem/model_b.txt"}
    assert c2.overlay["problem/model_a.txt"] == sha256(b"slope = 2.0\n").hexdigest()      # built on tuner-1's edit
    hidden = (EXAMPLE / "judge/hidden/points.csv").read_bytes()
    for ws in _tool_workspaces(tmp_path / "ws"):                                            # tool runs only, not judge runs
        assert not (ws / "judge").exists()
        assert all(p.read_bytes() != hidden for p in ws.rglob("*") if p.is_file())
    assert json.loads(_solution(cfg, "line-fit"))["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/e2e/test_own_script.py -v`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Write the example files and docs** as specified above (`_tool_workspaces` selects workspaces holding `swarm/request.json` and no `candidate/` dir).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/e2e/test_own_script.py -v && uv run swarm-arena run --yes examples/own_script/run.toml examples/own_script/plan.toml && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`, then the lint trio.
Expected: PASS; the CLI prints `done: line-fit 1 finalists`.

- [ ] **Step 5: Commit**

```bash
git add examples/own_script tests/e2e/test_own_script.py docs/own-files.md README.md
git commit -m "docs: add the bring-your-own-script example with edits, overlays, and hidden judge data"
```

### Task 31: E2E many goals and parts

**Files:**
- Create: `tests/e2e/test_many_goals_parts.py`
- Modify: `tests/_fakes/tools.py` (add `line_value(a: float) -> dict[str, float]`, `part_value(part: str, v: float) -> dict[str, float]`, `score_value(goal: Goal, candidate: Candidate) -> Score`; `join_sum` exists since Task 24)

**Interfaces:**
- Consumes: public API and `_fakes.tools` refs.

- [ ] **Step 1: Write the failing test**

```python
async def test_two_goals_shared_budget_and_parts_join(run_config: RunConfig) -> None:
    cfg = run_config.model_copy(update={"limits": run_config.limits.model_copy(update={"max_usd": 10.0})})   # tool estimate 0.1 USD
    out = await Swarm(cfg).run([_line_plan("g-line", rounds=1), _parts_plan("g-parts", parts=["a", "b"], rounds=2)])
    assert out["g-line"] and out["g-parts"]
    join = out["g-parts"][0]
    assert sorted(_part_of(cfg, p) for p in join.parent_ids) == ["a", "b"] and join.team == "join"
    started = _events(cfg, "goal_started")
    assert {e.goal_id for e in started} == {"g-line", "g-parts"}
    first = _first_agent_share(cfg, "g-parts", round=1); later = _first_agent_share(cfg, "g-parts", round=2)   # round_started.shares
    assert later > first   # the round-2 share is g-parts' pool left over one round left, at least the round-1 share because the tools settle 0; the return of g-line's unused pool itself is pinned by Task 26 test_unused_goal_budget_returns_to_pool
    assert _total_reserved(cfg) <= 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/e2e/test_many_goals_parts.py -v`
Expected: FAIL with `AttributeError: module '_fakes.tools' has no attribute 'line_value'`.

- [ ] **Step 3: Add the fake tools.** Any engine bug the test exposes is fixed in the owning module with a unit test there first.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/e2e/test_many_goals_parts.py -v`, then the lint trio.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_many_goals_parts.py tests/_fakes/tools.py
git commit -m "test(e2e): two goals on one budget and a goal split into parts and joined"
```

### Task 32: E2E lifecycle, crash resume, fencing

**Files:**
- Create: `tests/e2e/test_cli_lifecycle.py`, `tests/e2e/test_crash_resume.py`, `tests/e2e/test_fencing.py`
- Modify: `tests/_fakes/tools.py` (add `sleepy(seconds: float, marker: str) -> dict[str, float]`: writes `{"pid": os.getpid(), "pgid": os.getpgid(0)}` to `marker`, then sleeps)

**Interfaces:**
- Consumes: the CLI (subprocess, `env={**os.environ, "PYTHONPATH": "tests"}`), `swarm_arena` public API, and `_journal` internals for assertions only (`read_owner`, `read_chain`, `Lease`, `Fenced`). Each file defines its own small helpers (`_start_driver`, `_wait_for(pred, timeout_s)`; `test_fencing.py`: `_run_capture`, `_wait_tool_started`, `_take_over_retrying`, the same rule as `tests/_fakes/engine.py:take_over_retrying`, written out so the e2e file needs no engine fakes).

- [ ] **Step 1: Write the failing tests**

```python
# tests/e2e/test_cli_lifecycle.py — plan: 3 rounds, sleepy(1.0) tools; run.toml lease_ttl_s = 3
def test_run_stop_resume_status(tmp_path: Path) -> None:
    drv = _start_driver(["run", "--yes", RUN, PLAN]); run_dir = _wait_run_dir(drv)
    assert "epoch 1  owner live" in _cli(["status", run_dir])
    assert _cli_rc(["stop", run_dir]) == 0 and drv.wait(120) == 0
    assert "stop stop" in _cli(["status", run_dir]) and (Path(run_dir) / "control/stop-e1").exists()
    drv2 = _start_driver(["resume", run_dir]); _wait_epoch(run_dir, 2)        # lease released: no --force needed
    assert "round 2/3" in _wait_status(run_dir, "round 2/3")                  # the epoch-1 stop file is ignored
    assert _cli_rc(["stop", run_dir]) == 0 and drv2.wait(120) == 0             # a second stop works on epoch 2
    assert (Path(run_dir) / "control/stop-e2").exists()
    assert _cli_rc(["resume", run_dir]) == 0                                   # epoch 3 runs to completion
    assert "goal toy-goal  finished  round 3/3" in _cli(["status", run_dir])
```

```python
# tests/e2e/test_crash_resume.py — agent fast: two quick runs; agent slow: sleepy(60, marker) in round 1
def test_sigkill_mid_round_then_resume(tmp_path: Path) -> None:
    drv = _start_driver(["run", "--yes", RUN, PLAN]); run_dir = _wait_run_dir(drv)
    _wait_for(lambda: marker.exists() and _committed_tool_ends(run_dir) >= 2, 60)
    pgid = json.loads(marker.read_text())["pgid"]
    os.kill(drv.pid, signal.SIGKILL); drv.wait(10)
    assert _group_alive(pgid)                                                  # orphan survives the driver
    assert _cli_rc(["resume", run_dir, "--force"]) == 0
    ev = _committed_events(run_dir)
    assert _tool_starts_for(ev, request=FAST_1) == 1                           # committed runs are not repeated
    lost = [e for e in ev if e.type == "run_lost"]
    assert len(lost) == 1 and not _settled(ev, lost[0].payload["reservation_id"])   # charged in full
    assert not _group_alive(pgid)                                              # host and start time matched → killed
    assert json.loads((Path(run_dir) / "goals/toy-goal/solution.json").read_text())["status"] in {"ok", "no_result"}
```

```python
# tests/e2e/test_fencing.py — in-process Swarm.run with a sleepy(600, marker) tool, so the tool never ends on its own
async def test_forced_takeover_fences_old_owner(run_config: RunConfig, tmp_path: Path) -> None:
    swarm = Swarm(run_config)
    marker = tmp_path / "pid.json"
    raised: list[BaseException] = []
    with anyio.fail_after(60):                                                 # only the fence can end this run in time
        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_capture, swarm, _plan(marker=marker), raised)   # appends what Swarm.run raises
            await _wait_tool_started(run_config)
            store = open_store(run_config.run_root).child(swarm.run_id)
            new = _take_over_retrying(store)                                   # retries on Fenced while the epoch is unchanged (§7.1)
    assert [type(e) for e in raised] == [Fenced]                               # Swarm.run raised Fenced itself
    doc, _ = read_owner(store)
    assert doc.epoch == 2 and doc.head == new.doc.head                         # the old owner committed nothing after the takeover
    assert all(name.startswith("journal/e1-") for name, _ in read_chain(store, doc.head))
    assert not _group_alive(json.loads(marker.read_text())["pgid"])            # old owner cancelled its in-flight work
    new.release()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/e2e/test_cli_lifecycle.py tests/e2e/test_crash_resume.py tests/e2e/test_fencing.py -v`
Expected: FAIL with `AttributeError: module '_fakes.tools' has no attribute 'sleepy'`.

- [ ] **Step 3: Add `sleepy`.** Fix any engine bug found in its owning module, unit test first.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/e2e -v`, then the lint trio.
Expected: PASS (all six M1b e2e files plus `test_single_run.py`).

- [ ] **Step 5: Commit**

```bash
git add tests/e2e tests/_fakes/tools.py
git commit -m "test(e2e): stop and resume through the CLI, crash recovery after SIGKILL, and fencing"
```

### Task 33: Docs pages for this milestone

**Files:**
- Modify: `docs/planning-modes.md`, `docs/limits-and-safety.md`, `docs/scoring-and-review.md`, `docs/concepts.md` (add Manager, Policy, Board, Swarm), `CHANGELOG.md`

**Interfaces:**
- Consumes: the public names of this milestone; the examples.

- [ ] **Step 1: Write the failing check**

Run: `grep -l "Written in M1b" docs/*.md`
Expected: lists `planning-modes.md`, `limits-and-safety.md`, `scoring-and-review.md` (the M1a stubs say "Written in M1b."; `getting-started.md` no longer does after Task 29).

- [ ] **Step 2: Write the pages.** "Planning modes": manual mode only (TOML and Python plans, `Plan.load`, provenance and why `origin` is set by the core, the planner catalog and what a non-human revision may change; LLM, together, and autonomous marked "M3"). "Limits and safety": the §8.1 table with the defaults of Global Constraints (`digest_tokens` counts UTF-8 bytes, a hard bound on tokens for every model; English uses about a quarter of it), the budget split (reserve, final slices, goal weights, round and agent shares), committed reservations and the crash bound, the watchdog rules, the three clocks, mid-round stops, the stop modes and resume (with the two Task 27 notes: "A resumed stopped goal gets only its share of the free work pool (`Ledger.free()`), which can be 0 when other goals were still running and already got its unused budget; unless the resume raises the limits, such a goal stops again with `goal budget` at its first run." and "A continued goal's earlier lifecycle stays in the run journal: its `final_order` and its `goal_finished` stop reason are kept there, even after the later lifecycle replaces `solution.json` and the replay clears the goal's `end_reason`. `status` and `solution.json` show only the latest lifecycle."), and the §8.4 statement that v1 sandboxing is not strong isolation. "Scoring and review": checks and `feedback`, judges and judge identity, anchors, the final judge, the four policies with their exact pruning rules, `NO_RESULT`, and `solution.json` fields (human review marked "M2"). Each page links the API reference.

- [ ] **Step 3: Run the check and the build**

Run: `! grep -l "Written in M1b" docs/*.md && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: grep finds nothing; Sphinx builds with no warnings.

- [ ] **Step 4: Run the docstring and API tests**

Run: `uv run pytest tests/test_public_api.py tests/test_docstrings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs CHANGELOG.md
git commit -m "docs: write Planning modes, Limits and safety, and Scoring and review for the engine"
```

### Task 34: Milestone exit

**Files:** none new.

- [ ] **Step 1: Run the common gate (R2.0)**

```bash
uv sync --all-extras --all-groups
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Expected: every command exits 0; pytest reports no failures and no errors.

- [ ] **Step 2: Run the M1b exit criteria**

```bash
uv run pytest tests/e2e -v
uv run swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml
uv run swarm-arena run --yes examples/own_script/run.toml examples/own_script/plan.toml
```

Expected: `test_single_run.py`, `test_rosenbrock.py`, `test_own_script.py`, `test_many_goals_parts.py`, `test_cli_lifecycle.py`, `test_crash_resume.py`, and `test_fencing.py` pass; both commands exit 0 and print `done: rosenbrock 2 finalists` and `done: line-fit 1 finalists`.

- [ ] **Step 3: Check the required named tests and the Review Focus tests**

Run: `uv run pytest "tests/_planning/test_validate.py::test_address_not_reused" "tests/_planning/test_validate.py::test_judge_file_in_goal_rejected" "tests/_planning/test_validate.py::test_non_human_part_files_outside_main_manifest_rejected" "tests/_planning/test_validate.py::test_bad_or_duplicate_names_rejected" "tests/_agent/test_context.py::test_parallel_run_tool_respects_last_slot" "tests/_agent/test_context.py::test_parallel_post_learning_respects_rate_limit" "tests/_budget/test_ledger.py::test_parallel_reserve_never_overdraws" "tests/_agent/test_tool_run.py::test_executor_exception_becomes_error_candidate" "tests/_engine/test_round.py::test_manager_exception_ends_only_that_agent" "tests/_engine/test_goal.py::test_all_gated_out_writes_no_result" "tests/_engine/test_goal.py::test_all_judge_errors_writes_no_result" "tests/_engine/test_run.py::test_run_returns_empty_list_for_no_result_goal" -v`
Expected: 12 passed.

- [ ] **Step 4: Check the Must-NOT list and import isolation**

Run: `uv run pytest tests/test_import_isolation.py -v && ! grep -rnE '^[[:space:]]*(import[[:space:]]+(sqlite3|s3fs|ray|litellm|agents|langfuse|a2a)([.,[:space:]]|$)|from[[:space:]]+(sqlite3|s3fs|ray|litellm|agents|langfuse|a2a)(\.[A-Za-z0-9_.]*)?[[:space:]]+import[[:space:]])' src/swarm_arena --include='*.py'`
Expected: PASS; grep prints nothing (it matches import statements only, so M1a's `LLMCall.litellm_call_id` field and docstring text such as "from agents ..." do not match).

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin m1b-engine
gh pr create --base main --head m1b-engine --title "M1b Engine" --body "Implements roadmap milestone M1b (journal and lease, ledger, agents, policies, validate_plan, RunBoard, Swarm run/resume/stop, CLI, examples). Exit criteria: the common gate, tests/e2e, and both example runs pass. Roadmap edits (each skipped when the roadmap already had it): R4.13.3 Lease.take_over expect_epoch and the read_chain link and segment-name rules; R4.14 plan_rejected, agent_counters, goal_started.pool, run_started.final_slices, and r0 journaled at run start; R4.13.4 RunState.ledger_events (with run_config and limits_changed), the config.limits fold, and the continuation lifecycle; R4.13.6 the history-only Ledger.replay, open_goal(pool=), free(), change_limits(), the end_goal free-pool rule, holder pools, exhaustion, the Tranche comment, and the final_slice_estimate n_final term; R4.13.10 AgentRuntime.holder and private_paths, RepeatIndex.lookup(resumed) and next_replicate(requested=), and the execute_tool_run order with the settle after the completion event and the admission check at Inflight.add and on_started; R4.13.11 final_judge returns (order, JudgeOutcome); R4.4 LimitReached.limit \"stop\"; R5.4 part-file resolution, the judge reservation rule, and item 14 replaced by the stopped-goal continuation rule; R3.2 rows for tests/_fakes/engine.py, tests/_agent/conftest.py, and tests/_engine/conftest.py."
```

Merge after the owner's review; M2 starts from the merged `main`.
