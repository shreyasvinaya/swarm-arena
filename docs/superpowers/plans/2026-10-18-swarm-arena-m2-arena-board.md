# swarm-arena M2 Arena & board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the arena real: S3 storage, a multi-run `ArenaBoard` that reads every run journal of a namespace by recorded frontier into a local SQLite/FTS5 cache, the posting rules, digest v2, the `Curator` hook, round pages, human review, leaderboards, `trace`, `reindex`, the arena CLI, and the board viewer.

**Architecture:** Each run's committed journal stays its only arena record. `arena_root` holds only write-once run pointers and review and curation objects. `ArenaBoard.frontier()` makes one authority read of `owner.json` for each run with a pointer, downloads only the new chain segments into a per-namespace SQLite cache, and returns a `Frontier`. `snapshot(goal, frontier)` filters every read by that frontier, so the same frontier gives the same view on any machine. Pages, the review export, and leaderboards are pure functions of committed events and recorded frontiers.

**Tech Stack:** Python 3.11+, pydantic v2, anyio (asyncio backend), stdlib `sqlite3` (FTS5), `csv`, `html`, `http.server`; extra `s3` = `s3fs` + `aiobotocore` (floors from Task 1); tests with pytest 9, anyio plugin, moto (or MinIO, per Task 1); ruff, ty, Sphinx + sphinx-rtd-theme + numpydoc + myst-parser.

**Spec:** `docs/superpowers/specs/2026-10-16-swarm-arena-design.md` (APPROVED r8). Binding roadmap: `docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md` (module map R3, interface registry R4, constraints R5). M2 scope: §4.4 `HumanReview`, §4.9, §6.3, §6.3.1, §6.6, §7 (S3), §7.1 "What readers read", §7.2, §7.3 item 4 (non-LLM steps).

## Global Constraints

- Start from merged M1b `main`: `git switch -c m2-arena-board`. Every task ends with one commit. Commit messages: imperative subject, body says what and why, **no AI or Claude co-author line and no "generated with" text**.
- Module paths, names, and signatures come from roadmap R3 and R4. This plan adds only private helpers. It edits the roadmap in exactly these places, each in the commit of the task that needs it (R0): R4.6 `BoardView.count_learnings` (Task 8), R4.6 `ArenaBoard.Config.cache_dir` default honors `XDG_CACHE_HOME` (Task 8), R3.2 rows for `tests/_fakes/journals.py` (Task 4) and `tests/_fakes/curators.py` (Task 12), R5.4 item 11 and R2 "M2 Arena & board" deliverable 9, both with the review export path `<run_root>/<run-id>/review/e<epoch>-<seq>/` (Task 14), the R4.6 `Candidate.warnings` comment that adds the past-run hint entry (Task 9), the R3.1 `_journal/event.py` row, which drops "payload models" and its `+M2` (Task 4: M1b writes plain dict payloads, and M2 does not change this file), and a `+M2` marker on each R3.1 row that M2 changes but that has none yet, in the commit of the task that first changes the file: `_model/migrate.py` (Task 3), `_scoring/ranking.py` (Task 7), `_board/base.py` (Task 8), `_engine/round.py`, `_agent/tool_run.py`, and `testing/_harness.py` (Task 9), `_agent/context.py` and `_agent/scripted.py` (Task 10), `_journal/replay.py` and `_engine/goal.py` (Task 12), `_cli/main.py` (Task 17), the floors in R2 "M2" item 2 and in the R5.1 `s3` extra (Task 1: each `<floor>` placeholder becomes its Task 1 value), and R4.13.14 `build_digest`, whose `ranking` becomes `str | Mapping[str | None, str]` (Task 11: a group's own ranking judge, R2 "M2" item 8). These M2 roadmap entries are already in the roadmap, so their tasks confirm them and edit nothing: R2 "M2" items 2 (the `check_storage` probe at preflight and again in `resume`, `review import`, and `reconcile`), 3, and 8, the R3.1 `_engine/resume.py` row `M1b +M2 +M4` and R4.13.13 `check_storage` with the `resume_run` `+M2` comment (Task 2), R4.13.14 `estimate_tokens` (UTF-8 bytes) and the `build_digest` whole-output cap (Task 11), the R4.13.14 `ArenaCache` comment (lock scope, `.lock` file, per-frontier search index; Task 4), and R5.4 item 17 (the §4.9 identity fallback; Task 8).
- Journal events: M1b Task 2 defines every R4.14 `EventType` value, `curated` and `review_imported` included, so M2 adds no `EventType` member. There are no payload models. Every payload is a plain dict with the M1b keys (pinned in M1b Task 7, with `tool_end.repeat_key` from M1b Task 19): `plan_revision.{plan, origin, base_dir, planner_call_ids}`, `tool_end.{candidate, a2a, repeat_key}`, `score.score`, `learning.learning`, `decision.{round, decision}`, `frontier.{round, frontier}`, `round_closed.{round, run_elapsed_s}`. A model value goes in as `model_dump(mode="json")` and comes out with `model_validate`. M2 adds only the `curated` and `review_imported` payloads and their no-op replay folds (Tasks 12 and 14).
- Verification commands: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, and for docs `uv run sphinx-build -W --keep-going -b html docs docs/_build/html`.
- Python 3.11+, POSIX only. anyio on the asyncio backend only. `Store` and `Board` are sync; async callers use `anyio.to_thread.run_sync`. Board read methods and `put_*` run on the event-loop thread; `open`, `close`, `frontier`, and `snapshot` run in a worker thread (R4.6).
- The core never imports `s3fs`. Only `src/swarm_arena/_store/s3.py` imports it (and `botocore`); `open_store` imports that module lazily. Extra: `s3 = ["s3fs>=<S3FS_FLOOR>", "aiobotocore>=<AIOBOTOCORE_FLOOR>"]`. Task 1 sets both floors: the oldest releases whose botocore `PutObject` model has `IfNoneMatch` and `IfMatch` (s3fs filters `put_object` kwargs through that model, so an older botocore drops them silently).
- There is no unconditional put and no delete in `Store`. Every key is created once or replaced with `if_match`.
- Only the chain reached back from a run's `owner.json` head counts. Readers make one authority read per run per frontier, follow links back, and never list `journal/` (§7.1).
- `arena_root` holds only `<ns>/runs/<run-id>.json`, `<ns>/reviews/<reviewer>/<YYYYMMDDTHHMMSS.mmmZ>-<8 hex>.jsonl`, `<ns>/curation/<YYYYMMDDTHHMMSS.mmmZ>-<8 hex>.jsonl`. All are written once and never changed (§7.2). S3 `probe` keys are the one exception: `.swarm-arena-probe/` at the root of an `s3://` root.
- Human objects apply in one order: the timestamp in the name, then the full name. A later pin or hide of the same learning wins (§7.2).
- Arena human scores come from `reviews/` only. `review_imported` journal events are never indexed as scores (§7.2).
- `Learning.title` is one line of 1 to 120 characters. `Learning.text` is at most 1,500 characters. A learning needs at least one evidence candidate unless its author starts with `human:` (§4.2, D20).
- Duplicate check: the bm25 top hit for the title, plus title-token Jaccard ≥ 0.6, requires `refines` that names a duplicate or a non-blank `duplicate_reason` (roadmap M2 item 5).
- Digest cap: `limits.digest_tokens` (default 1,500), counted with `estimate_tokens(text) = len(text.encode("utf-8"))`, an upper bound on tokens (R4.13.14). Round pages show the top 10. Review export shows the top 20 of each goal. Human learning ids are `L-h-<16 hex>`.
- Secrets are never written. S3 credentials and endpoint come only from the standard AWS environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL_S3`), never from `run.toml`.
- Every function and method is fully type-annotated, tests included. No `# type: ignore`. Wrap untyped s3fs and botocore calls in small typed helpers.
- Every new public name (`ArenaBoard`, `Curator`, `HumanReview`, `BoardView.count_learnings`) and each of its public methods has a numpydoc docstring with an `Examples` section (a real doctest, or `# doctest: +SKIP` when it needs a run). Facades set `obj.__module__` to the public path. `tests/test_public_api.py` and `tests/test_docstrings.py` cover them.
- Tests mirror `src/` (R3): `src/swarm_arena/<path>/<name>.py` → `tests/<path>/test_<name>.py`. M1a Task 1 owns the `collect_ignore` entries: it already ignores every R3.2 extra test path (`s3` group: `_store/test_s3.py`) in `tests/conftest.py` and `src/swarm_arena/_store/s3.py` in the root `conftest.py`. M2 only confirms them. M1a Task 1 Step 2 also owns the `TID251` per-file-ignore keys in `pyproject.toml`, including `"src/swarm_arena/_store/s3.py"` and `"tests/_store/test_s3.py"`; M2 only confirms them too (a repeated key in one TOML table is a parse error). M2 adds no other test module that needs `s3fs` at import time.
- Docs pages: M1a created one Markdown page per §11 page. This plan edits two of them: `docs/arena.md` ("The arena") and `docs/scoring-and-review.md` ("Scoring and review"; M1b and M3 edit the same file). Each task that adds user-visible behavior adds one line under `Unreleased` → `Added` in `CHANGELOG.md`.

## Review Focus

1. **Search text that is FTS5 syntax.** An agent or a human searches for `log(nu)`, `fd-solver`, `"nu`, `-lr:(`, `NEAR(a b)`, or only punctuation. Expected: ranked results or `[]`, never `sqlite3.OperationalError`. Test: Task 6 `test_search_never_raises_on_fts_syntax`.
2. **A review CSV saved by a spreadsheet program, or a CSV with bad rows.** A UTF-8 BOM, CRLF line ends, and `Accept` in mixed case must import. Unknown ids, unknown verdicts, and non-numeric or non-finite scores must fail with every bad row number, before the lease is taken and before anything is written. Tests: Task 14 `test_import_accepts_spreadsheet_csv`, `test_import_rejects_bad_rows_and_writes_nothing`.
3. **A broken neighbor in the arena.** A pointer to an unreachable `run_root`, an `s3://` pointer on a reader without the `s3` extra, a run with no `owner.json` or no head yet, a corrupt pointer, or a pointer with a newer `schema_version`. Expected: `frontier()` skips that run with a logged warning. Another swarm's round never crashes. Test: Task 8 `test_frontier_skips_broken_runs`.
4. **Stale S3 listings.** s3fs caches listings and instances by default, so a new `runs/` pointer or `curation/` object would stay unseen and two swarms on S3 would never see each other. Expected: a key created after a `list` shows up in the next `list`. Test: Task 2 `test_list_sees_new_key_immediately`.
5. **Two drivers on one host share the default cache file.** Both open a new `~/.cache/swarm-arena/<hash>/<ns>.sqlite` (both see `user_version == 0`) and ingest at the same time. Expected: the schema is created once, no `database is locked`, and both runs are fully indexed. Test: Task 4 `test_two_caches_share_one_file_concurrently`.

---

### Task 1: Verify S3 conditional writes

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, roadmap R2 "M2" item 2 and R5.1 (`s3` extra floors)
- Create: `tasks/research/api-s3-conditional.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the locked versions of `s3fs`, `aiobotocore`, `botocore`, `fsspec`, `moto`; the floors `S3FS_FLOOR` and `AIOBOTOCORE_FLOOR`; the exact write calls and the error mapping that Task 2 implements; the S3 test backend (moto, or MinIO through `SWARM_ARENA_TEST_S3_ENDPOINT`).

- [ ] **Step 1: Add the extra and the test dependency**

```bash
git switch -c m2-arena-board
uv add --optional s3 s3fs                      # no floor yet: Step 4b sets the floors
uv add --dev "moto[server]>=5"
uv export --all-extras --all-groups --no-hashes | grep -E '^(s3fs|aiobotocore|botocore|fsspec|moto)=='
```
Expected: five pinned lines.

- [ ] **Step 2: Probe the locked versions against moto**

Write a scratch script outside the repo (`/tmp/s3probe/probe.py`; never committed). Start `moto.server.ThreadedMotoServer(ip_address="127.0.0.1", port=<free port>)`, set `AWS_ACCESS_KEY_ID=testing`, `AWS_SECRET_ACCESS_KEY=testing`, `AWS_DEFAULT_REGION=us-east-1`, `AWS_ENDPOINT_URL_S3=http://127.0.0.1:<port>`, and use `s3fs.S3FileSystem(skip_instance_cache=True, use_listings_cache=False)`. Record for each case: the HTTP status, the botocore error code, the exception type and message seen through s3fs, and whether s3fs retried it.
  1. `fs.call_s3("put_object", Bucket=b, Key=k, Body=..., IfNoneMatch="*")` on a new key, then again on the same key (expect 412 `PreconditionFailed`).
  2. `put_object(..., IfMatch=<stale etag>)` (expect 412), then `IfMatch=<current etag>` (expect success).
  3. `put_object(..., IfMatch=...)` on a missing key (expect 404 `NoSuchKey`).
  4. 409 `ConditionalRequestConflict`: read `s3fs.core._error_wrapper` and `s3fs.errors.translate_boto_error` and note whether 409 is retried and how it surfaces.
  5. The ETag format from `put_object`, `get_object`, and `head_object` (quoted or not; equal across the three).
  6. Whether the locked botocore honors `AWS_ENDPOINT_URL_S3`.
  7. Whether a key created after `fs.ls(prefix)` appears in the next `fs.ls(prefix)` with the two cache options above.
  8. Print `{"IfNoneMatch", "IfMatch"} & set(client.meta.service_model.operation_model("PutObject").input_shape.members)` for the locked botocore (expect both keys).

Run: `uv run python -I /tmp/s3probe/probe.py`
Expected: a printed table for cases 1 to 8.

Then find the floors in a throwaway venv under `/tmp/s3floor/` (never committed): the oldest `aiobotocore` release whose pinned botocore range gives both keys in case 8, and the oldest `s3fs` release whose requirements allow that `aiobotocore`.
Expected: two versions, `S3FS_FLOOR` and `AIOBOTOCORE_FLOOR`.

- [ ] **Step 3: If moto fails case 1, 2, or 3, repeat against MinIO**

```bash
docker run -d --name sa-minio -p 9000:9000 -e MINIO_ROOT_USER=minio -e MINIO_ROOT_PASSWORD=minio123 minio/minio server /data
```
Rerun the script with `AWS_ENDPOINT_URL_S3=http://127.0.0.1:9000` and those credentials.

- [ ] **Step 4: Write `tasks/research/api-s3-conditional.md`**

Sections: **Versions** (the five pins, the two floors, and the case 8 output for the locked botocore and at each floor); **Calls** (the exact code for create, replace, read, exists, list); **Error mapping** (a table: condition → HTTP status → botocore code → what s3fs raises → what `S3Store` raises: `AlreadyExists`, `PreconditionFailed`, `NotFound`, retry, `WriteOutcomeUnknown`); **Retries** (what s3fs retries on its own); **ETag**; **Endpoint**; **Listing cache**; **Test backend** (`moto`, or `MinIO` with the CI service and `SWARM_ARENA_TEST_S3_ENDPOINT`); **Fallback** (if s3fs hides the error code: call the aiobotocore client from `await fs.set_session()` inside `fsspec.asyn.sync(fs.loop, ...)`; record which path Task 2 uses).

- [ ] **Step 4b: Set the floors**

```bash
uv add --optional s3 "s3fs>=<S3FS_FLOOR>" "aiobotocore>=<AIOBOTOCORE_FLOOR>"
grep -n '^s3 = ' pyproject.toml
```
Expected: `s3 = ["s3fs>=<S3FS_FLOOR>", "aiobotocore>=<AIOBOTOCORE_FLOOR>"]` with the two values. Edit roadmap R2 "M2" item 2 and R5.1 `s3`: each `<floor>` placeholder becomes its value.

- [ ] **Step 5: Check that nothing else changed**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check .`
Expected: PASS (no code changed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tasks/research/api-s3-conditional.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Verify S3 conditional writes for the s3 extra" -m "Records the s3fs/botocore calls, error codes, ETag format, and test backend that S3Store relies on (roadmap M2 task 1). The s3 extra gets s3fs and aiobotocore floors whose botocore PutObject model has IfNoneMatch and IfMatch, because s3fs drops parameters the model lacks; roadmap R2 M2 item 2 and R5.1 record the floors."
```

---

### Task 2: S3Store, conditional-write probe, storage preflight

**Files:**
- Create: `src/swarm_arena/_store/s3.py`, `tests/_store/test_s3.py` (marker `s3`)
- Modify: `src/swarm_arena/_store/base.py`, `tests/_store/test_base.py` (new tests, and the `s3://` assertions of the M1a and M1b `open_store` tests)
- Modify: `src/swarm_arena/_engine/preflight.py`, `tests/_engine/test_preflight.py`
- Modify: `src/swarm_arena/_engine/resume.py` (`check_storage` first in `resume_run`), `tests/_engine/test_resume.py`. Confirm (no edit): the roadmap R3.1 row already reads `M1b +M2 +M4`, and R4.13.13 already has the `resume_run` `+M2` comment.
- Modify: `tests/conftest.py` (fixture `s3_url`; confirm that `_store/test_s3.py` is in the `s3` group of `collect_ignore`: M1a Task 1 already ignores every R3.2 extra test path, so do not add it again)
- Confirm (no edit): root `conftest.py` already ignores `src/swarm_arena/_store/s3.py` in doctest `collect_ignore` when `s3fs` is missing (M1a Task 1); keep it.
- Confirm (no edit): `pyproject.toml` already has the `TID251` keys `"src/swarm_arena/_store/s3.py"` and `"tests/_store/test_s3.py"` in `[tool.ruff.lint.per-file-ignores]` (M1a Task 1 Step 2). Do not add them again: a repeated key in one TOML table is a parse error.
- Modify: `.github/workflows/ci.yml` (on both paths: an `extras`-job step `uv run pytest -m s3 -v` after the default pytest step, because `addopts` deselects `s3` (R2.0); only if Task 1 chose MinIO, also a `minio` service in the `extras` job and `SWARM_ARENA_TEST_S3_ENDPOINT`, `AWS_*` env)
- Modify: `docs/arena.md` (section "Storage on S3"), `CHANGELOG.md`

**Interfaces:**
- Consumes: `Store`, `Blob`, `LocalStore`, `open_store` (R4.13.2); `NotFound`, `AlreadyExists`, `PreconditionFailed`, `WriteOutcomeUnknown`, `ConditionalWritesUnsupported`, `ConfigError` (R4.13.1); `Lease`, `Head`, `encode_segment`, `segment_name`, `read_chain`, `read_owner` (R4.13.3); `RunConfig`; `resume_run` (R4.13.13).
- Produces:
  - `class S3Store(Store)`: `__init__(self, url: str, **storage_options: Any) -> None`, and `read`, `create`, `replace`, `exists`, `list`, `child` with the R4.13.2 signatures.
  - `S3Store._put_object(self, key: str, data: bytes, **condition: str) -> str` (the one write seam; returns the ETag; raises `_S3Error`).
  - `S3Store._conditional_params(self) -> frozenset[str]` (the `PutObject` input members of the live client, computed once).
  - `S3Store._call(self, method: str, **kwargs: Any) -> dict[str, Any]` (the one read seam for `get_object`, `head_object`, `list_objects_v2`: the raw client call, with no error mapping).
  - `class _S3Error(Exception)`: `status: int | None`, `code: str`.
  - `Store.probe_conditional_writes(self) -> None` (generic body in `base.py`).
  - `open_store("s3://…")` → `S3Store`.
  - `def check_storage(config: RunConfig) -> None` in `_engine/preflight.py`.
  - Test fixture `s3_url` (yields `"s3://swarm-arena-test/<uuid4 hex>"` with the AWS env set).

- [ ] **Step 1: Write the failing tests**

`tests/_store/test_s3.py`:
```python
pytestmark = pytest.mark.s3

def test_create_then_read(s3_url: str) -> None:
    store = open_store(s3_url)
    etag = store.create("a/b.json", b"x")
    assert store.read("a/b.json") == Blob(b"x", etag)

def test_create_existing_raises_already_exists(s3_url: str) -> None:
    store = open_store(s3_url)
    store.create("k", b"1")
    with pytest.raises(AlreadyExists):
        store.create("k", b"2")
    assert store.read("k").data == b"1"

def test_replace_with_stale_etag_raises_precondition_failed(s3_url: str) -> None:
    store = open_store(s3_url)
    old = store.create("k", b"1")
    store.replace("k", b"2", if_match=old)
    with pytest.raises(PreconditionFailed):
        store.replace("k", b"3", if_match=old)

def test_replace_missing_raises_not_found(s3_url: str) -> None: ...      # replace("nope", b"", if_match='"x"') → NotFound
def test_read_missing_raises_not_found(s3_url: str) -> None: ...         # read("nope") → NotFound; exists("nope") is False

def test_list_sees_new_key_immediately(s3_url: str) -> None:            # Review Focus 4
    store = open_store(s3_url)
    store.create("ns/runs/r1.json", b"{}")
    assert store.list("ns/runs/") == ["ns/runs/r1.json"]
    store.create("ns/runs/r2.json", b"{}")
    assert store.list("ns/runs/") == ["ns/runs/r1.json", "ns/runs/r2.json"]

def test_child_prefixes_keys(s3_url: str) -> None: ...   # open_store(s3_url).child("run-1").create("x", b"1"); parent.read("run-1/x").data == b"1"
def test_probe_passes(s3_url: str) -> None: ...          # open_store(s3_url).probe_conditional_writes() returns None

def test_lease_and_chain_on_s3(s3_url: str) -> None:
    store = open_store(s3_url).child("run-1")
    lease = Lease.acquire_new(store, owner_id="o", host="h", ttl_s=60.0)
    # write two segments with encode_segment + store.create, advance the head after each
    # assert [name for name, _ in read_chain(store, lease.doc.head)] == [first, second]

def test_conflict_409_retries_then_unknown(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    store = open_store(s3_url)
    calls: list[str] = []
    def conflict(key: str, data: bytes, **condition: str) -> str:
        calls.append(key)
        raise _S3Error(409, "ConditionalRequestConflict")
    monkeypatch.setattr(store, "_put_object", conflict)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(WriteOutcomeUnknown):
        store.create("k", b"1")
    assert len(calls) == 4                                  # 1 try + 3 retries

def test_old_botocore_sends_no_put(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    store = open_store(s3_url)
    monkeypatch.setattr(store, "_conditional_params", lambda: frozenset({"Bucket", "Key", "Body"}))
    # _put_object replaced by a spy that records each call in `sent`
    with pytest.raises(ConditionalWritesUnsupported, match="IfNoneMatch"):
        store.create("k", b"1")
    with pytest.raises(ConditionalWritesUnsupported, match="IfMatch"):
        store.replace("k", b"2", if_match='"x"')
    assert sent == []

def test_locked_botocore_has_conditional_params(s3_url: str) -> None:
    store = open_store(s3_url)
    assert isinstance(store, S3Store)
    assert {"IfNoneMatch", "IfMatch"} <= store._conditional_params()

def test_probe_detects_endpoint_that_ignores_conditions(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # the client has the parameters, but the endpoint ignores the conditions
    store = open_store(s3_url)
    real = store._put_object
    monkeypatch.setattr(store, "_put_object", lambda key, data, **condition: real(key, data))
    assert {"IfNoneMatch", "IfMatch"} <= store._conditional_params()
    with pytest.raises(ConditionalWritesUnsupported, match="create"):
        store.probe_conditional_writes()

def test_read_path_errors_become_storage_error(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None: ...
    # _call patched to raise botocore.exceptions.EndpointConnectionError(endpoint_url="http://x")
    # → read("k"), exists("k"), and list("p/") each raise StorageError
```

`tests/_store/test_base.py`: `open_store("s3://…")` now returns an `S3Store`, or raises `ConditionalWritesUnsupported` (a `StorageError`, R5.4 item 1) when the extra is missing. Two older tests change in this task:
- M1a `test_open_store`: delete its `s3://` line (`pytest.raises(ConditionalWritesUnsupported, match="s3")`): with the extra installed, `open_store("s3://…")` now returns an `S3Store`. Keep its `gs://` line.
- M1b `test_open_store_s3_needs_extra`: keep it, and add `monkeypatch.setitem(sys.modules, "s3fs", None)` so that it also holds when the extra is installed. It still uses `pytest.raises(ConditionalWritesUnsupported, match=r"swarm-arena\[s3\]")`.
- M1b `test_open_store_refuses_scheme_without_conditional_writes` (`gs://`, `http://`, `ftp://`) stays as it is. It is the `gs://` refusal check, so this task adds no other one.

Additions:
```python
def test_probe_passes_on_local_store(tmp_path: Path) -> None: ...
def test_probe_rejects_store_that_ignores_conditions(tmp_path: Path) -> None:
    # NaiveStore(LocalStore) whose create overwrites an existing key
    with pytest.raises(ConditionalWritesUnsupported, match="create"):
        NaiveStore(tmp_path).probe_conditional_writes()
```

`tests/_engine/test_preflight.py` (additions):
```python
def test_local_run_root_with_s3_arena_refused(tmp_path: Path) -> None:
    config = RunConfig(run_root=str(tmp_path / "runs"), arena_root="s3://b/arena")
    with pytest.raises(ValueError, match="run_root"):
        check_storage(config)                               # raises before any network call
class RecordingStore:                                    # test-local fake for open_store; never builds an S3Store, so no s3fs import
    def __init__(self, url: str, probed: list[str], *, fail: bool = False) -> None: ...
    def probe_conditional_writes(self) -> None: ...      # appends self.url to probed; raises ConditionalWritesUnsupported when fail
def test_probe_runs_once_per_distinct_s3_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probed: list[str] = []
    monkeypatch.setattr(preflight, "open_store", lambda url: RecordingStore(url, probed))
    check_storage(RunConfig(run_root="s3://b/x", arena_root="s3://b/x", artifact_root="s3://b/x"))
    assert probed == ["s3://b/x"]
    # local roots under tmp_path → probed stays []
def test_probe_failure_stops_preflight(monkeypatch: pytest.MonkeyPatch) -> None: ...   # open_store patched to RecordingStore(..., fail=True) → check_storage re-raises ConditionalWritesUnsupported
```
These two tests patch `swarm_arena._engine.preflight.open_store`, so they run in the CI `core` job with no `s3fs`.

`tests/_engine/test_resume.py` (addition; M1b fixture `crashed`):
```python
async def test_resume_checks_storage_before_takeover(crashed: Crashed, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[RunConfig] = []
    def spy(config: RunConfig) -> None:
        calls.append(config)
        raise ConditionalWritesUnsupported("s3://b/runs: create did not behave conditionally")
    monkeypatch.setattr(resume, "check_storage", spy)                 # swarm_arena._engine.resume
    before = crashed.store.read("control/owner.json").data
    with pytest.raises(ConditionalWritesUnsupported):
        await Swarm(crashed.config).resume(crashed.run_id, force=True)
    assert len(calls) == 1
    assert crashed.store.read("control/owner.json").data == before
    assert read_owner(crashed.store)[0].epoch == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_store/test_s3.py -m s3 -v; uv run pytest tests/_store/test_base.py tests/_engine/test_preflight.py tests/_engine/test_resume.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._store.s3'`, `ImportError: cannot import name 'check_storage'`, and `AttributeError` (`swarm_arena._engine.resume` has no attribute `check_storage`).

- [ ] **Step 3: Implement `S3Store` in `src/swarm_arena/_store/s3.py`**

- `s3.py` has no module-level import of `s3fs` or `botocore`. So the module always imports, and a test that sets `sys.modules["s3fs"] = None` still sees the error.
- `__init__(url, **storage_options)`: run `import s3fs` inside a `try`, and turn a failure into `ConditionalWritesUnsupported("s3:// storage needs the s3 extra: pip install 'swarm-arena[s3]'")` (a `StorageError`, so it is a `RuntimeError` subclass, R5.4 item 1). Parse `s3://<bucket>/<prefix>`. Do no I/O. The `s3fs.S3FileSystem(skip_instance_cache=True, use_listings_cache=False, **storage_options)` is created on first use.
- Keys follow the `LocalStore` rules (relative, no `..`, non-empty) and map to `Key=<prefix>/<key>`.
- `_put_object` uses the call recorded in Task 1 and turns errors into `_S3Error(status, code)`. Timeouts and connection errors become `WriteOutcomeUnknown`.
- `create`: `IfNoneMatch="*"`. 412 → `AlreadyExists`. `replace`: `IfMatch=if_match`. 412 → `PreconditionFailed`, 404 → `NotFound`. For both, 409 → retry after 0.1, 0.2, 0.4 s, then `WriteOutcomeUnknown`.
- Before any `_put_object` call, `create` checks that `"IfNoneMatch"` is in `_conditional_params()`, and `replace` checks `"IfMatch"`. A missing one → `ConditionalWritesUnsupported(f"{self.url}: botocore {botocore.__version__} would drop {param} from PutObject; upgrade swarm-arena[s3]")`. So no caller (driver, review import, curation CLI, viewer, M4 reconcile) can send an unconditional write.
- `read`: `_call("get_object", ...)` → `Blob(body, ETag)`. `exists`: `_call("head_object", ...)`. `list(prefix)`: `_call("list_objects_v2", ...)`, paginated, keys relative to the store root, sorted. These three map `_call` errors in one place: a 404 → `NotFound` (`exists` → `False`); any other botocore, s3fs, or `OSError` exception → `StorageError(f"{self.url}/{key}: {type(exc).__name__}: {exc}")` (`key` is the prefix for `list`). So `frontier()` skips an unreachable S3 neighbor (Review Focus 3). `child(prefix)` → `S3Store` on `url + "/" + prefix` with the same options.
- The ETag returned by `create` and `replace` equals `read(key).etag` (the format recorded in Task 1).

- [ ] **Step 4: Implement the probe and the `s3://` branch in `src/swarm_arena/_store/base.py`**

`Store.probe_conditional_writes`: key `.swarm-arena-probe/<YYYYMMDDTHHMMSS.mmmZ>-<8 hex>`. `etag = create(key, b"1")`; a second `create` must raise `AlreadyExists`; `replace(key, b"3", if_match="probe-mismatch")` must raise `PreconditionFailed`; `replace(key, b"4", if_match=etag)` must succeed, and `read(key).data == b"4"`. Any other result → `ConditionalWritesUnsupported(f"{self.url}: <step> did not behave conditionally")`. `open_store("s3://…")` imports `swarm_arena._store.s3` inside the function and returns `S3Store(url)`, whose `__init__` raises `ConditionalWritesUnsupported("s3:// storage needs the s3 extra: pip install 'swarm-arena[s3]'")` when the extra is missing. `open_store` itself does no I/O.

- [ ] **Step 5: Implement `check_storage(config: RunConfig) -> None` in `src/swarm_arena/_engine/preflight.py`, and call it first in `resume_run`**

1. If `arena_root` starts with `s3://` and `run_root` does not → `ConfigError("run_root must be reachable by other readers: a local run_root cannot be used with an s3:// arena_root (§7.2)")`.
2. Then call `open_store(root).probe_conditional_writes()` once for each distinct `s3://` root among `run_root`, `arena_root`, `artifact_root`. Local roots are not probed: they use `O_EXCL`, `flock`, and `os.replace`.
3. Standalone `owner.json` writers. `check_storage(config)` also runs in every command that replaces `control/owner.json` outside a driver's preflight, through `anyio.to_thread.run_sync`, before its first takeover and before any write: `resume_run` (first line of `_engine/resume.py` `resume_run`; this task), `HumanReview.import_csv` (Task 14 import step 1b), and M4 `take_run` (M4 Task 16). These commands can run on another host or with another `AWS_ENDPOINT_URL_S3` than the driver, so the driver's probe does not cover them. Each command calls it once, so no cache is needed. Curation, `learnings add`, and viewer writes only create unique random names, so they do not probe. `resume.py` imports `check_storage` at module level, so tests patch `resume.check_storage`.

`preflight.py` imports `open_store` at module level (`from swarm_arena._store.base import open_store`), so tests patch `preflight.open_store`. Call `check_storage` first in the M1b preflight entry point, through `anyio.to_thread.run_sync`.

- [ ] **Step 6: Add the fixture and confirm the collection rules**

In `tests/conftest.py`, fixture `s3_url`: call `pytest.importorskip("s3fs")`. If `SWARM_ARENA_TEST_S3_ENDPOINT` is set, use it. Otherwise call `pytest.importorskip("moto.server")` and start `ThreadedMotoServer` on a free port (session scope). In CI (`CI` is set in the environment), a missing `s3fs`, or a missing `moto.server` with `SWARM_ARENA_TEST_S3_ENDPOINT` unset, calls `pytest.fail(...)` in place of the skip, so the `extras` job cannot pass with no S3 test run. Set the `AWS_*` env with `monkeypatch`, create the bucket `swarm-arena-test` with the call from Task 1, and yield `f"s3://swarm-arena-test/{uuid4().hex}"`.

Confirm, with no edit, that M1a Task 1 already put `_store/test_s3.py` in the `s3` group of `collect_ignore` in `tests/conftest.py` and `src/swarm_arena/_store/s3.py` in the root `conftest.py` `collect_ignore`, both applied when the `s3fs` distribution is missing (`importlib.metadata.version`, roadmap R3.2).

Confirm, with no edit, that M1a Task 1 Step 2 already put `"src/swarm_arena/_store/s3.py" = ["TID251"]` and `"tests/_store/test_s3.py" = ["TID251"]` in `[tool.ruff.lint.per-file-ignores]` of `pyproject.toml`. Run: `grep -c -E '^"(src/swarm_arena/_store/s3\.py|tests/_store/test_s3\.py)" = \["TID251"\]' pyproject.toml`. Expected: `2`. Do not add the keys again.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/_store tests/_engine/test_preflight.py tests/_engine/test_resume.py -v && uv run pytest -m s3 -v && uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: PASS. `ci.yml` has the `extras`-job step `uv run pytest -m s3 -v`.

- [ ] **Step 8: Docs**

`docs/arena.md`, section "Storage on S3": the three roots can each be `s3://…`. Install `swarm-arena[s3]`. Credentials and endpoint come from the AWS environment variables. S3 must support conditional writes (`If-None-Match`, `If-Match`): preflight probes it, and `resume` and `review import` probe it again before they take over a run (they can run with another host or endpoint than the driver). `S3Store` refuses to write when the installed botocore would drop either parameter. Each probe (preflight, `resume`, `review import`) writes one small object under `.swarm-arena-probe/` of every `s3://` root and never deletes it, because `Store` has no delete; a bucket lifecycle rule on that prefix removes them. A local `run_root` with an `s3://` `arena_root` is refused. Add a CHANGELOG line.

- [ ] **Step 9: Commit**

```bash
git add src/swarm_arena/_store tests/_store src/swarm_arena/_engine/preflight.py tests/_engine/test_preflight.py src/swarm_arena/_engine/resume.py tests/_engine/test_resume.py tests/conftest.py .github/workflows/ci.yml docs/arena.md CHANGELOG.md
git commit -m "Add S3Store with conditional writes and storage preflight" -m "S3 roots use If-None-Match and If-Match writes. Preflight probes them and refuses a local run_root with an s3 arena_root (spec 7.1, 7.2). resume probes again before its takeover, because it can run with another host or endpoint than the driver."
```

---

### Task 3: Run pointers and human objects

**Files:**
- Create: `src/swarm_arena/_board/pointers.py`, `tests/_board/test_pointers.py`
- Create: `src/swarm_arena/_board/human_objects.py`, `tests/_board/test_human_objects.py`
- Modify: `src/swarm_arena/_model/migrate.py` (`CURRENT["run_pointer"] = 1`, `CURRENT["human_object"] = 1`), `tests/_model/test_migrate.py`, roadmap R3.1 (`_model/migrate.py` gains `+M2`)

**Interfaces:**
- Consumes: `Store`, `StoredModel`, `canonical_json`, `utc_now`, `Learning`, `Score`, `CURRENT`, `upgrade` (`_model/migrate.py`).
- Produces:
```python
class RunPointer(StoredModel):                                     # schema_kind = "run_pointer"
    run_id: str
    run_root: str                                                  # absolute local path or "s3://…"
def pointer_key(namespace: str, run_id: str) -> str                # "<namespace>/runs/<run-id>.json"
def write_pointer(arena: Store, namespace: str, pointer: RunPointer) -> None
def list_pointers(arena: Store, namespace: str) -> tuple[list[RunPointer], list[str]]   # (sorted by run_id, warnings)
class HumanObject(StoredModel):                                    # schema_kind = "human_object"
    kind: Literal["pin", "hide", "add", "review"]
    learning_id: str | None = None
    learning: Learning | None = None
    score: Score | None = None
    author: str
def human_object_name(now: datetime) -> str                        # "<YYYYMMDDTHHMMSS.mmmZ>-<8 hex>.jsonl"
def human_object_order(name: str) -> tuple[str, str]               # (timestamp of the basename, full name)
def human_object_key(namespace: str, folder: Literal["reviews", "curation"], basename: str, *,
                     reviewer: str | None = None) -> str           # "<ns>/reviews/<reviewer>/<basename>" or "<ns>/curation/<basename>"
def write_human_objects(arena: Store, namespace: str, folder: Literal["reviews", "curation"],
                        objects: Sequence[HumanObject], *, now: datetime, reviewer: str | None = None,
                        name: str | None = None) -> str            # name: the basename to use (default human_object_name(now));
                                                                   # returns human_object_key(...)
def list_human_objects(arena: Store, namespaces: Sequence[str]) -> tuple[list[str], list[str]]   # (reviews, curation), each in order
def read_human_object(arena: Store, name: str) -> list[HumanObject]
def fold_curation(ops: Iterable[tuple[str, int, HumanObject]]) -> dict[str, tuple[bool, bool]]  # id → (pinned, hidden)
```
Names are arena-relative keys, for example `"toy/reviews/alice/20261018T130000.000Z-0a1b2c3d.jsonl"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pointer_written_once_and_idempotent(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    p = RunPointer(run_id="r1", run_root=str(tmp_path / "runs"))
    write_pointer(arena, "toy", p)
    write_pointer(arena, "toy", p)                                          # same bytes → success
    assert json.loads(arena.read("toy/runs/r1.json").data) == {"schema_version": 1, "run_id": "r1", "run_root": str(tmp_path / "runs")}
def test_pointer_conflict_raises(tmp_path: Path) -> None: ...             # same run id, other run_root → StorageError
def test_pointer_rejects_relative_run_root() -> None:
    with pytest.raises(ValueError, match="absolute"):
        RunPointer(run_id="r1", run_root="runs")
def test_list_pointers_skips_corrupt_and_newer(tmp_path: Path) -> None:   # Review Focus 3 (pointer half)
    # good pointer r1; raw b"{not json" at toy/runs/r2.json; schema_version 99 at toy/runs/r3.json
    pointers, warnings = list_pointers(arena, "toy")
    assert [p.run_id for p in pointers] == ["r1"]
    assert len(warnings) == 2
def test_write_uses_given_name(tmp_path: Path) -> None: ...               # name="20261018T130000.000Z-0a1b2c3d.jsonl", reviewer "alice" → returns "toy/reviews/alice/20261018T130000.000Z-0a1b2c3d.jsonl", that key exists
def test_human_object_name_format() -> None:
    name = human_object_name(datetime(2026, 10, 18, 13, 0, 0, tzinfo=UTC))
    assert re.fullmatch(r"20261018T130000\.000Z-[0-9a-f]{8}\.jsonl", name)
def test_order_is_timestamp_then_name() -> None:
    a = "toy/curation/20261018T130001.000Z-aaaa0000.jsonl"
    b = "toy/reviews/bob/20261018T130000.000Z-ffff0000.jsonl"
    c = "toy/curation/20261018T130000.000Z-0000ffff.jsonl"
    assert sorted([a, b, c], key=human_object_order) == [c, b, a]
def test_later_pin_or_hide_wins() -> None:
    hide = HumanObject(kind="hide", learning_id="L-1", author="human:a")
    pin = hide.model_copy(update={"kind": "pin"})
    assert fold_curation([("n1", 0, hide), ("n2", 0, pin)]) == {"L-1": (True, False)}
    assert fold_curation([("n1", 0, pin), ("n2", 0, hide)]) == {"L-1": (False, True)}
def test_write_and_read_roundtrip(tmp_path: Path) -> None: ...            # reviews → "toy/reviews/alice/<name>", curation → "toy/curation/<name>"; read returns the same objects in order
def test_reviewer_rules(tmp_path: Path) -> None: ...                      # reviews need a slug reviewer ("Alice Smith" → ValueError, match="reviewer"); curation with a reviewer → ValueError, match="curation"
```
`tests/_model/test_migrate.py`: `test_m2_kinds_registered` (`CURRENT["run_pointer"] == CURRENT["human_object"] == 1`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_pointers.py tests/_board/test_human_objects.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._board.pointers'`.

- [ ] **Step 3: Implement both modules**

- Bytes are `canonical_json(model.model_dump(mode="json"))`. A pointer holds exactly `schema_version`, `run_id`, `run_root` (no secrets). A relative `run_root` fails validation with "run_root must be absolute or s3://".
- `write_pointer` uses `create`. On `AlreadyExists` it reads the key back: the same bytes count as success, other bytes raise `StorageError`.
- Register the kinds in `_model/migrate.py`: `CURRENT["run_pointer"] = 1` and `CURRENT["human_object"] = 1`; set `RunPointer.schema_kind = "run_pointer"` and `HumanObject.schema_kind = "human_object"`.
- `list_pointers` lists `"<ns>/runs/"`. It loads each pointer through `upgrade("run_pointer", raw)`. A key that fails to parse, or has a newer version, goes into `warnings` as `"<key>: <reason>"` and is skipped.
- Human object files are JSONL with one `HumanObject` per line. `read_human_object` loads each line through `upgrade("human_object", raw)`.
- `write_human_objects` creates `human_object_key(namespace, folder, name or human_object_name(now), reviewer=reviewer)` and returns that key. A caller that must know the key before the write (Task 14) makes the basename first and passes it as `name`.
- `fold_curation` sorts by `(human_object_order(name), line)`. `pin` sets `(True, False)`. `hide` sets `(False, True)`.
- The reviewer slug is `[a-z0-9][a-z0-9._-]*`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board/test_pointers.py tests/_board/test_human_objects.py tests/_model/test_migrate.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

Edit roadmap R3.1: the `_model/migrate.py` row gets `+M2`.
```bash
git add src/swarm_arena/_board/pointers.py src/swarm_arena/_board/human_objects.py src/swarm_arena/_model/migrate.py tests/_board/test_pointers.py tests/_board/test_human_objects.py tests/_model/test_migrate.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Add write-once run pointers and human objects" -m "Pointers, review and curation objects are created once and applied in one fixed order (spec 7.2). The migration registry gains the run_pointer and human_object kinds; roadmap R3.1 marks migrate.py +M2."
```

---

### Task 4: Arena cache schema and chain ingest

**Files:**
- Create: `src/swarm_arena/_board/cache.py` (schema, `ingest`, `newest_segment`), `tests/_board/test_cache.py`
- Create: `src/swarm_arena/_board/ingest.py`, `tests/_board/test_ingest.py`
- Create: `tests/_fakes/journals.py`
- Modify: `docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md` (R3.2 row: `tests/_fakes/journals.py`, M2: a committed run journal written without the engine; R3.1 row `_journal/event.py`: contents `Event`, `EventDraft`, `EventType`, `WRITE_THROUGH`, milestone `M1b`, because M1b makes no payload models and M2 does not change the file)
- Confirm (no edit): the roadmap R4.13.14 `ArenaCache` comment already gives the lock scope, the `.lock` file, and the per-frontier search index of this task and Task 6.

**Interfaces:**
- Consumes: `Event`, `EventDraft`, `EventType` (`_journal/event.py`), with plain dict payloads that use the M1b keys (Global Constraints: `plan_revision.{plan, origin}`, `tool_end.candidate`, `score.score`, `learning.learning`); `Head`, `Lease`, `encode_segment`, `segment_name`, `read_chain` (R4.13.3); `ChainError`; `Frontier.Mark`; `Store`, `open_store`; `canonical_json`, `sha256_hex`; `make_candidate`, `RunStatus`; `GoalKey.of`, `PartSpec`; the M1b `tool_end.repeat_key` payload key.
- Produces:
```python
class ArenaCache:                                       # R4.13.14; the file stem is the namespace ("toy.sqlite" → "toy")
    SCHEMA_VERSION: ClassVar[int] = 1                   # PRAGMA user_version
    namespace: str
    warnings: list[str]                                 # id clashes skipped by ingest (below), in ingest order
    def __init__(self, path: Path) -> None: ...
    def ingest(self, run_id: str, run_root: str, segments: Sequence[tuple[str, list[Event]]]) -> None: ...
    def newest_segment(self, run_id: str) -> str | None: ...
    def has_segment(self, run_id: str, segment: str) -> bool: ...
    def run_root(self, run_id: str) -> str | None: ...
    def close(self) -> None: ...
def goal_key_digest(key: GoalKey) -> str                # sha256_hex(canonical_json(key.model_dump(mode="json")))
def _has_fts5(conn: sqlite3.Connection) -> bool         # private; the FTS5 presence check
# _board/ingest.py
def run_store(run_root: str, run_id: str) -> Store      # open_store(run_root).child(run_id)
def ingest_run(cache: ArenaCache, store: Store, run_id: str, run_root: str, head: Head) -> int   # new events stored
def ensure_covered(cache: ArenaCache, mark: Frontier.Mark) -> None
# tests/_fakes/journals.py
START: Final = datetime(2026, 10, 18, tzinfo=UTC)
class FakeRun:                                          # writes a committed journal the way JournalWriter commits
    def __init__(self, run_root: str, run_id: str, *, namespace: str = "toy", goal_id: str = "g1", start: datetime = START,
                 goal: Goal | None = None) -> None: ...    # a given goal replaces the one built from namespace and goal_id
    run_id: str
    run_root: str
    store: Store
    goal: Goal
    heads: list[Head]
    def commit(self, *drafts: EventDraft) -> Head: ...     # one segment (link = previous head) + Lease.advance; event ts = start + seq seconds
    def take_over(self) -> None: ...                       # Lease.take_over(force=True): epoch + 1, head kept
    def orphan(self, *drafts: EventDraft) -> str: ...      # a segment of the old epoch linked to the head, never named by a head
    def plan(self, *, revision: int = 0, ranking: str = "j", min_improvement: float | None = None, parts: Sequence[str] = ()) -> EventDraft: ...
        # each name p becomes PartSpec(name=p, goal=<the main goal with id f"{goal_id}-{p}">)
    def candidate(self, cid: str, *, team: str = "t1", agent: str = "a1", round: int = 1, part: str | None = None,
                  fidelity: str = "f1", partial: bool = False, passed: bool = True, parents: Sequence[str] = (),
                  status: RunStatus = RunStatus.OK, repeat_key: str | None = None) -> EventDraft: ...   # a tool_end draft
    def score(self, sid: str, cid: str, x: float, *, judge: str = "j", rubric_hash: str = "r1", panel_hash: str = "", minimize: bool = True) -> EventDraft: ...
    def learning(self, lid: str, title: str, *, text: str = "t", evidence: Sequence[str] = (), tags: Sequence[str] = (),
                 verified: bool = False, refines: Sequence[str] = (), supersedes: Sequence[str] = (), round: int = 1, author: str = "t1/a1") -> EventDraft: ...
def add_pointer(arena_root: str, run: FakeRun, namespace: str | None = None) -> None: ...
```

- [ ] **Step 1: Write the failing tests**

`tests/_board/test_cache.py`:
```python
def test_ingest_is_idempotent(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(), run.candidate("c-1"))
    cache = ArenaCache(tmp_path / "cache" / "toy.sqlite")
    segs = read_chain(run.store, head)
    cache.ingest("r1", run.run_root, segs)
    cache.ingest("r1", run.run_root, segs)
    assert candidate_ids(cache) == ["c-1"]                  # test helper: SELECT id FROM candidates ORDER BY id
def test_ingest_keeps_repeat_key(tmp_path: Path) -> None: ...   # tool_end of c-1 with repeat_key "k1", c-2 with none → candidates.repeat_key "k1" and NULL
def test_keeps_only_this_namespace(tmp_path: Path) -> None: ...   # goal in "deepretro" → no rows in toy.sqlite; goal in "toy" → rows
def test_rejects_gap_at_junction(tmp_path: Path) -> None:
    # three commits; ingest segments [0], then [2] → ChainError
def test_rejects_segment_with_other_seqs(tmp_path: Path) -> None: ...   # a cached name re-ingested with different first/last seq → ChainError
def test_indexed_event_before_plan_raises(tmp_path: Path) -> None: ...  # tool_end of a goal with no plan_revision → ChainError
def test_failed_segment_rolls_back(tmp_path: Path) -> None: ...
    # one segment: plan_revision of goal g0, then a tool_end of goal g1 (no plan) → ChainError; no goals, plans, candidates,
    # or segments row of that segment exists; cache._conn.in_transaction is False; a later valid ingest succeeds
def test_part_candidates_get_part_key(tmp_path: Path) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    head = run.commit(run.plan(parts=["A"]), run.candidate("c-1", part="A"), run.candidate("c-2"))
    # ingest; then SELECT id, goal_key, part_key FROM candidates:
    # c-1: goal_key == goal_key_digest(GoalKey.of(run.goal)), part_key == goal_key_digest(GoalKey.of(<part A goal>))
    # c-2: part_key == goal_key
def test_id_from_other_run_is_skipped_with_warning(tmp_path: Path) -> None: ...
    # r1 commits c-1 (team t1); r2 commits a candidate with the same id c-1 (team t2); ingest r1, then r2
    # → one candidates row "c-1", with run_id "r1"; cache.warnings == ["candidate c-1 of run r2 skipped: the cache holds it from run r1"];
    # a second ingest of r2 adds no warning
def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "toy.sqlite"
    ArenaCache(path).close()
    sqlite3.connect(path).execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="reindex"):
        ArenaCache(path)
def test_missing_fts5_error_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(swarm_arena._board.cache, "_has_fts5", lambda conn: False)
    with pytest.raises(RuntimeError, match="FTS5") as err:
        ArenaCache(tmp_path / "toy.sqlite")
    assert "reindex" not in str(err.value)
def test_two_caches_share_one_file_concurrently(tmp_path: Path) -> None:      # Review Focus 5
    # runs r1 and r2, each 20 commits of 10 tool_end events; path = tmp_path / "fresh" / "toy.sqlite" does not exist yet
    # two threads; each waits on one threading.Barrier(2), then constructs its own ArenaCache(path) (both see a new file)
    # and calls ingest_run for r1 and r2 (thread 1 in that order, thread 2 in reverse)
    # both threads finish without an exception; a third ArenaCache(path) has user_version == SCHEMA_VERSION and 400 candidates
```

`tests/_board/test_ingest.py`:
```python
def test_incremental_download_reads_only_new_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 3 commits, ingest_run; 2 more commits; count store.read calls on "journal/" keys during the second ingest_run == 2
def test_never_lists_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None: ...   # store.list raises AssertionError for "journal" prefixes; ingest_run works
def test_takeover_chain_ignores_fenced_segment(tmp_path: Path) -> None:
    run = FakeRun(...)
    run.commit(run.plan(), run.candidate("c-1"))
    run.take_over()
    run.orphan(run.candidate("c-ghost"))
    head = run.commit(run.candidate("c-2"))
    ingest_run(cache, run.store, "r1", run.run_root, head)
    assert candidate_ids(cache) == ["c-1", "c-2"]
def test_older_mark_is_covered_by_newer_cache(tmp_path: Path) -> None: ...   # cache at head 3; ensure_covered(mark of head 2) does no read and no error
def test_mark_not_on_cached_chain_raises(tmp_path: Path) -> None: ...        # mark naming the orphan segment → ChainError
def test_ensure_covered_ingests_from_recorded_run_root(tmp_path: Path) -> None: ...   # empty cache; ensure_covered(mark of head 2) indexes events up to head 2 only
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_cache.py tests/_board/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_fakes.journals'`.

- [ ] **Step 3: Write `tests/_fakes/journals.py`**

`FakeRun` uses `Lease.acquire_new` and `Lease.advance`, `encode_segment`, and `segment_name`. It never uses `JournalWriter`. Payloads are plain dicts with the M1b keys, and each model value is `model_dump(mode="json")`: `plan()` → `plan_revision` `{"plan": ..., "origin": "human"}`, `score()` → `score` `{"score": ...}`, `learning()` → `learning` `{"learning": ...}`. `candidate()` builds the `Candidate` with `make_candidate` and sets `run_id`, `goal_id`, `part`, `team`, `agent`, `round`, `fidelity`, `parent_ids`, `status`, and one `CheckResult(check="c", passed=passed)`. Its `tool_end` payload is `{"candidate": ..., "repeat_key": repeat_key}` (the M1b payload keys; `repeat_key` only when given).

- [ ] **Step 4: Implement `ArenaCache` in `src/swarm_arena/_board/cache.py`**

Open order. The parent directory is created here. First open `<path>.lock` (created if missing, never deleted) and hold `fcntl.flock(LOCK_SH)` on it until `close()`, so `reindex` can tell that a process uses the file (Task 15). Then `sqlite3.connect(path, check_same_thread=False, isolation_level=None)` (the code opens every transaction itself), `PRAGMA busy_timeout=30000` before any other statement, then `PRAGMA journal_mode=WAL`. If `__init__` raises, it closes the connection and the lock file first.

Lock rule (roadmap R4.13.14). One `threading.RLock` is held for the whole body of every public method: from `BEGIN IMMEDIATE` to `COMMIT` or `ROLLBACK` of each transaction, and for each query through its `fetchall()`. A failed transaction is rolled back before the lock is released. Board snapshots run in worker threads and the viewer is a `ThreadingHTTPServer`, so a lock per statement would let two threads interleave one transaction.

Checks. `_has_fts5(conn)` is `False` → `RuntimeError("this Python's sqlite3 has no FTS5; ArenaBoard needs SQLite built with FTS5")`; the persistent schema has no FTS table, but the per-frontier search index of Task 6 needs FTS5. A `user_version` that is neither 0 nor `SCHEMA_VERSION` → `RuntimeError` whose message says "run `swarm-arena reindex`". Only this second message names reindex. Schema creation, so that two processes that open a new file at the same time create it once:
1. `BEGIN IMMEDIATE` (it takes the write lock at once and waits up to `busy_timeout` for it).
2. Read `user_version` again inside this transaction. If it is `SCHEMA_VERSION`, `COMMIT` and stop; another opener made the schema. If it is neither 0 nor `SCHEMA_VERSION`, `ROLLBACK` and raise the `RuntimeError`.
3. Otherwise run every `CREATE TABLE IF NOT EXISTS`, set `PRAGMA user_version = SCHEMA_VERSION`, and `COMMIT`.

Every write transaction (each segment in `ingest`, each `apply_human_objects` call) also starts with `BEGIN IMMEDIATE`. A deferred transaction that reads and then writes can get `SQLITE_BUSY` at once in WAL mode, with no wait. Schema [plan]:

```sql
runs(run_id PRIMARY KEY, run_root)
segments(run_id, name, epoch, first_seq, last_seq, PRIMARY KEY(run_id, name))
goals(run_id, goal_id, namespace, goal_key, PRIMARY KEY(run_id, goal_id))             -- the main key, frozen at r0, every namespace
plans(run_id, goal_id, revision, origin, seq, ts, plan_json, PRIMARY KEY(run_id, goal_id, revision))
part_goals(run_id, goal_id, revision, part, part_key, PRIMARY KEY(run_id, goal_id, revision, part))   -- every revision's parts
candidates(id PRIMARY KEY, run_id, epoch, seq, ts, goal_id, goal_key, part_key, part, team, agent, round, fidelity, partial, gated_out, repeat_key, json)   -- repeat_key: tool_end.repeat_key or NULL
scores(id PRIMARY KEY, source, run_id, epoch, seq, ts, obj_name, line, candidate_id, judge, rubric_hash, panel_hash, json)
learnings(rowid INTEGER PRIMARY KEY, id UNIQUE, source, run_id, epoch, seq, ts, obj_name, line, goal_id, goal_key, author, verified, from_human, json)
learning_tags(learning_id, tag); learning_links(learning_id, kind, target_id); learning_evidence(learning_id, candidate_id)
curation_ops(obj_name, line, kind, learning_id); human_objects(name PRIMARY KEY)
```

`ingest` writes one transaction per segment, and the `segments` row goes last. Rules:
- A segment that is already cached is skipped only if its `(first_seq, last_seq)` match. Otherwise → `ChainError`.
- A new segment needs `first_seq == max(last_seq) + 1` when the run has cached segments. Otherwise → `ChainError`.
- `plan_revision` fills `goals` (at revision 0), `plans`, and `part_goals`, for every namespace. `part_goals` gets one row per `plan.parts[*]` of every revision, not only r0, because a join-failure replan can repartition: `part_key = goal_key_digest(GoalKey.of(part.goal))`. M1b gives part agents the part's `GoalKey` (spec §6.2) and snapshots by it (Task 8 "Part views").
- `tool_end`, `score`, and `learning` are indexed only when the goal's namespace is `self.namespace`. A goal with no known plan → `ChainError`.
- `candidates.part_key` is the main `goal_key` when `candidate.part is None`. Otherwise it is the `part_goals` row for `candidate.part` in the revision in effect: the goal's highest `plans.revision` with `plans.seq <` the `tool_end` seq. No such row → `ChainError`. `learnings.goal_key` is always the main key of `(run_id, goal_id)`.
- `tool_end` stores `payload["repeat_key"]` in `candidates.repeat_key` (NULL when the payload has none). A reader cannot compute the key of a past-run candidate again: a `Candidate` holds no tool ref or tool config. The past-run hint uses this column (Task 9).
- Every `ts` column holds `ts.astimezone(UTC).isoformat(timespec="microseconds")` (Task 5 stores human objects the same way).
- Run-level events and every other type are not indexed.
- Rows use `INSERT OR IGNORE` by id. A candidate, score, or learning whose id the cache already holds from another run is skipped (roadmap R4.0, R2 "M2" item 3): `f"{kind} {id} of run {run_id} skipped: the cache holds it from run {other}"` goes to `self.warnings` and to a `swarm_arena.board` log warning. `reindex` and the `board` commands print `warnings` (Tasks 15, 17).

- [ ] **Step 5: Implement `src/swarm_arena/_board/ingest.py`**

- `ingest_run`: if `head.segment == cache.newest_segment(run_id)`, return 0. Otherwise `read_chain(store, head, stop_after=cache.newest_segment(run_id))`, then `cache.ingest`. `ingest_run` and `ensure_covered` call `read_chain` (the network read) before they take the cache lock, never inside it.
- `ensure_covered(cache, mark)`: if `cache.has_segment(mark.run_id, mark.segment)`, return. If the cache's newest `last_seq` for the run is at or after `mark.seq`, raise `ChainError` (the mark is not on the cached chain). Otherwise `ingest_run(cache, run_store(mark.run_root, mark.run_id), …, Head(epoch=mark.epoch, seq=mark.seq, segment=mark.segment))`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/_board/test_cache.py tests/_board/test_ingest.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/swarm_arena/_board/cache.py src/swarm_arena/_board/ingest.py tests/_board/test_cache.py tests/_board/test_ingest.py tests/_fakes/journals.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Add the arena SQLite cache and incremental chain ingest" -m "Readers download only committed chain segments after one authority read, keyed by (run, epoch, seq), and keep only their namespace (spec 7.1, 7.2). One RLock covers each whole transaction, a shared flock on <ns>.sqlite.lock marks the file in use, part candidates get their part's goal key, and an id held from another run is skipped with a warning (roadmap R4.0, R4.13.14). Roadmap R3.2 lists the new test fake. Roadmap R3.1 drops the event.py payload models: no milestone makes them, and M1b writes plain dict payloads."
```

---

### Task 5: Cache reads filtered by frontier, and human objects

**Files:**
- Modify: `src/swarm_arena/_board/cache.py`, `tests/_board/test_cache.py`

**Interfaces:**
- Consumes: Task 3 `HumanObject`, `fold_curation`, `human_object_order`; Task 4 `ArenaCache`, `FakeRun`; `build_frontier` (M1b); `Frontier`.
- Produces (methods on `ArenaCache`; `Position = tuple[str, str, int]`, meaning (ISO ts, run id or object name, seq or line)):
```python
def apply_human_objects(self, names: Sequence[str], objects: Sequence[HumanObject]) -> None: ...   # names[i] is the object of objects[i]; line = index within its name
def has_object(self, name: str) -> bool: ...
def candidates(self, frontier: Frontier, *, ids: Sequence[str] | None = None, goal_key: str | None = None,
               part_key: str | None = None, run_id: str | None = None) -> list[tuple[Candidate, Position]]: ...
def scores(self, frontier: Frontier, *, candidate_ids: Sequence[str] | None = None,
           judge: str | None = None) -> list[tuple[Score, Position]]: ...
def flags(self, frontier: Frontier) -> dict[str, tuple[bool, bool]]: ...      # learning id → (pinned, hidden)
def learnings_by_id(self, frontier: Frontier, ids: Sequence[str]) -> list[Learning]: ...   # flags applied, hidden included
def plans(self, frontier: Frontier, run_id: str, goal_id: str) -> list[tuple[Plan, str, int]]: ...   # (plan, origin, seq), by revision
def membership(self, frontier: Frontier) -> bytes: ...
```
Membership rule [plan]: a journal item of run R is in the frontier only if the frontier has a mark for R and `item.seq <= mark.seq`. A review score or human-added learning is in only if its object name is in `frontier.reviews` or `frontier.curation`. Pin and hide apply only from objects named in `frontier.curation`. `membership` is `canonical_json({"candidates": [ids], "scores": [ids], "learnings": [[id, pinned, hidden]]})`, each list sorted.

- [ ] **Step 1: Write the failing tests**

```python
def test_items_after_mark_are_hidden(tmp_path: Path) -> None:
    h1 = run.commit(run.plan(), run.candidate("c-1"))
    h2 = run.commit(run.candidate("c-2"))
    ingest_run(cache, run.store, "r1", run.run_root, h2)
    f1 = build_frontier([("r1", run.run_root, h1)], [], [])
    f2 = build_frontier([("r1", run.run_root, h2)], [], [])
    assert [c.id for c, _ in cache.candidates(f1)] == ["c-1"]
    assert [c.id for c, _ in cache.candidates(f2)] == ["c-1", "c-2"]
def test_run_without_mark_is_hidden(tmp_path: Path) -> None: ...          # r2 ingested, frontier marks only r1 → no r2 items
def test_curation_applies_only_when_named(tmp_path: Path) -> None:
    cache.apply_human_objects([pin_name], [HumanObject(kind="pin", learning_id="L-1", author="human:a")])
    assert cache.flags(build_frontier(marks, [], [])) == {}
    assert cache.flags(build_frontier(marks, [], [pin_name])) == {"L-1": (True, False)}
def test_review_scores_come_from_named_objects(tmp_path: Path) -> None: ...   # review object with Score(judge="human:alice", candidate_id="c-1") → scores(judge="human:alice") has 1 entry only when named
def test_human_added_learning_needs_its_object(tmp_path: Path) -> None: ...
def test_membership_is_order_independent(tmp_path: Path) -> None: ...     # two caches, runs ingested in opposite order → equal bytes
def test_positions_are_commit_order(tmp_path: Path) -> None: ...          # the Position of c-1 sorts before the Position of c-2; both are (ISO ts, "r1", seq)
def test_human_ts_orders_with_journal_ts(tmp_path: Path) -> None: ...     # a journal score at START+1s and a review score whose object is
    # named START+2s, both named in the frontier → the review score's Position sorts after the journal one (Task 6 has the learning half)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_cache.py -v`
Expected: FAIL with `AttributeError: 'ArenaCache' object has no attribute 'candidates'`.

- [ ] **Step 3: Implement the methods**

Pass the frontier to SQL as CTEs: `marks(run_id, seq)` and `objs(name)`. SQLite rejects an empty `VALUES` list, so an empty list uses the sentinel row `('', -1)` or `('')`. `flags` reads `curation_ops` for the named objects and folds them with `fold_curation`. `apply_human_objects` is idempotent by `(name, line)`: `pin` and `hide` go to `curation_ops`; `add` adds a `learnings` row with `source='human'`, `obj_name`, `line`, `ts` = the name's timestamp, and `goal_key` NULL; `review` adds a `scores` row with `source='review'`. The name's `YYYYMMDDTHHMMSS.mmmZ` is parsed as UTC and stored as `ts.astimezone(UTC).isoformat(timespec="microseconds")`, the form of every journal ts (Task 4), so string order equals time order across journal and human items.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board/test_cache.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_board/cache.py tests/_board/test_cache.py
git commit -m "Filter arena cache reads exactly by frontier" -m "A snapshot holds only items on each marked chain and only the named review and curation objects, whatever order the cache got them in (spec 7.2)."
```

---

### Task 6: FTS5 search and relevance order

**Files:**
- Modify: `src/swarm_arena/_board/cache.py`, `tests/_board/test_cache.py`

**Interfaces:**
- Consumes: Task 5 reads.
- Produces:
```python
def fts_query(text: str) -> str | None: ...       # tokens re.findall(r"[^\W_]+", text.lower()) → '"t1" OR "t2"'; None when no token
    # [^\W_], not \w: unicode61 splits on "_", so \w+ could make an empty phrase such as '"___"'
class LearningFilter(FrozenModel):
    goal_key: str | None = None                   # goal_key and goal_id: the goal rule below
    goal_id: str | None = None
    run: str | None = None
    team: str | None = None
    agent: str | None = None
    round: int | None = None
    part: str | None = None
    tags: list[str] = []                          # any-of
    verified: bool | None = None
    human: bool | None = None
    pinned: bool | None = None
    hidden: bool | None = False                   # False: exclude hidden (agents); True: only hidden; None: both
# ArenaCache methods:
def find(self, frontier: Frontier, *, query: str | None, filters: LearningFilter,
         relevance_goal: tuple[str, str] | None, offset: int = 0, limit: int = 10) -> list[str]: ...
def count(self, frontier: Frontier, *, filters: LearningFilter) -> int: ...
def search(self, namespace: str, query: str, *, frontier: Frontier, goal: GoalKey | None,
           tags: Sequence[str], limit: int) -> list[str]: ...   # R4.13.14; ValueError if namespace != self.namespace
```
Goal rule [plan]: with both `goal_key = X` and `goal_id = Y` set, a learning matches when `goal_key = X OR (goal_key IS NULL AND goal_id = Y)`, so a human learning added with `--goal Y` (its `goal_key` is NULL) stays in a goal-scoped search. With only `goal_key`: `goal_key = X`. With only `goal_id`: `goal_id = Y`. With neither: no goal condition. The relevance goal is a separate argument of `find`, so a read with no goal filter can still put this goal first.
Relevance key (§6.3) [plan], ascending: `(not pinned, superseded, not goal_match, not verified, -matching_tags, -ts, id)`. `goal_match` means `goal_key` equals `relevance_goal[0]`, or (`goal_key` is NULL and `goal_id` equals `relevance_goal[1]`). `superseded` means the id is in the `supersedes` of a learning that is inside the frontier and not hidden under that frontier's curation fold (`flags(frontier)`), so hiding a summary undoes its merge (§6.3).
Search index [plan]. With a query, `find` matches and ranks on a per-frontier index, so corpus statistics (N, df, avgdl of bm25) come only from the frontier, never from post-frontier or unnamed rows in the cache. It is a TEMP FTS5 table `fts_<frontier.digest()[:16]>(title, text, tags)`, rowid = `learnings.rowid`, holding exactly the frontier's member learnings (journal and human-added, hidden included). It is built on first use under the cache lock, with rows inserted in id order; a connection keeps at most 8 (the least recently used is dropped). ORDER BY its `bm25()` first, then the relevance key. `find` runs after the frontier is covered (`frontier()` and `snapshot` ingest first, Task 8). `team` and `agent` come from `author` (`"team/agent"`). `part` matches a learning with an evidence candidate in that part.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("query", ['nu" OR', "-lr:(", "log(nu)", "fd-solver", "NEAR(a b)", "*", '"', "AND OR NOT", "()", "___", "a_b"])
def test_search_never_raises_on_fts_syntax(cache_with_learnings: Fixture, query: str) -> None:       # Review Focus 1
    ids = cache.search("toy", query, frontier=f, goal=None, tags=(), limit=10)
    assert isinstance(ids, list)
def test_search_finds_parenthesised_term(cache_with_learnings: Fixture) -> None:
    assert cache.search("toy", "log(nu)", frontier=f, goal=None, tags=(), limit=1) == [log_nu_id]   # title "Fitting log(nu) instead of nu converges about 3x faster."
def test_fts_query_quotes_tokens() -> None:
    assert fts_query('log(nu) "x"') == '"log" OR "nu" OR "x"'
    assert fts_query("()") is None
    assert fts_query("___") is None
    assert fts_query("fd_solver") == '"fd" OR "solver"'
def test_search_order_ignores_rows_outside_frontier(tmp_path: Path) -> None:
    # head h1 holds L-short (title "alpha", text "t") and L-long (text of about 60 tokens with "alpha" 3 times); f1 = the frontier at h1
    # cache A ingests only h1; cache B ingests h1 plus a later commit of 200 learnings of about 250 tokens without "alpha"
    args = dict(frontier=f1, goal=None, tags=())
    assert A.search("toy", "alpha", limit=10, **args) == B.search("toy", "alpha", limit=10, **args) == ["L-short", "L-long"]
    assert A.search("toy", "alpha", limit=1, **args) == B.search("toy", "alpha", limit=1, **args)   # the Task 10 duplicate-check input
    # with whole-table bm25, B would put L-long first
def test_human_added_learning_is_searchable(...) -> None: ...   # an `add` object whose learning text has "zebra": search("toy", "zebra", ...)
    # returns it when its object is in frontier.curation, and [] when it is not
def test_hidden_summary_restores_superseded(...) -> None:
    # L-sum supersedes L-a; L-a and L-b are otherwise equal, L-a is newer; f2 = f1 plus a curation object that hides L-sum
    assert order(f1, "L-a", "L-b") == ["L-b", "L-a"]           # order(f, ...): those ids in find(f, query=None, ...) order
    assert order(f2, "L-a", "L-b") == ["L-a", "L-b"]
    assert "L-sum" not in cache.find(f2, query=None, filters=LearningFilter(), relevance_goal=None, limit=10)
    assert order(f1, "L-a", "L-b") == ["L-b", "L-a"]           # f1 is unchanged
def test_human_learning_ts_orders_with_journal_ts(...) -> None: ...   # a human-added learning named at START and a journal learning at
    # START+1s, otherwise equal → find(query=None) lists the journal learning first (-ts; Task 5 ts form)
def test_one_cache_shared_by_threads(tmp_path: Path) -> None:      # Task 4 lock rule; here because it reads with candidates and find
    # one ArenaCache; two threads run ingest_run for r1 and r2 (20 commits × 10 tool_end each); at the same time two threads loop
    # cache.candidates(f) and cache.find(f, query="t", filters=LearningFilter(), relevance_goal=None) until both ingests end
    # → no exception in any thread, len(cache.candidates(f)) == 400, and not cache._conn.in_transaction
def test_goal_filter_keeps_human_learning_with_goal_id(...) -> None: ...   # "add" learning with goal_id "g1" and goal_key NULL is found by LearningFilter(goal_key=gk, goal_id="g1"); one with goal_id "g2" is not
def test_relevance_goal_without_goal_filter(...) -> None: ...             # find(query=None, filters=LearningFilter(), relevance_goal=(gk, "g1")): the g1 learning comes before an equal other-goal learning
def test_search_orders_by_bm25_then_relevance(...) -> None: ...   # two equal-text learnings: the verified one first
def test_relevance_order(...) -> None:
    # every learning has tag "a" (the filter is any-of); L-tag also has "b"
    # L-pin (pinned), L-goal (this goal), L-ns (other goal, oldest), L-ver (other goal, verified), L-tag (other goal, 2 matching tags),
    # L-new (other goal, newest), L-old-sup (this goal, superseded by L-goal)
    assert cache.find(f, query=None, filters=LearningFilter(tags=["a", "b"]), relevance_goal=(gk, "g1"), limit=10) == [
        "L-pin", "L-goal", "L-ver", "L-tag", "L-new", "L-ns", "L-old-sup"]
def test_hidden_excluded_by_default_and_filterable(...) -> None: ...   # hidden=False excludes; hidden=True only; None both
def test_tags_filter_is_any_of(...) -> None: ...
def test_count_matches_find(...) -> None: ...
def test_filters_run_team_agent_round_part_human_verified(...) -> None: ...
def test_search_rejects_other_namespace(...) -> None: ...                 # cache.search("other", ...) → ValueError
```
Make the expected list match the key above: pinned first; then not superseded before superseded; this goal before others; verified first; more matching tags first; newer first. Build the fixture so that each learning differs from the next in only the key element being tested.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_cache.py -k "search or relevance or fts or filter or count or hidden or tags or ts_orders or threads" -v`
Expected: FAIL with `ImportError: cannot import name 'fts_query'`.

- [ ] **Step 3: Implement `fts_query`, `LearningFilter`, `find`, `count`, `search`**

`fts_query` doubles any `"` inside a token (a `[^\W_]+` token has none, but keep the rule). Build the per-frontier index as in "Search index" above. `search` keeps the R4.13.14 signature. Its relevance goal is its `goal` filter, so a caller that needs this goal first with no goal filter (every `ArenaView` read, Task 8) calls `find` directly. `search` delegates to `find(frontier, query=query, filters=LearningFilter(goal_key=goal_key_digest(goal) if goal else None, goal_id=goal.id if goal else None, tags=list(tags)), relevance_goal=(goal_key_digest(goal), goal.id) if goal else None, limit=limit)`. When `fts_query` returns `None`, `find` with a query returns `[]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board/test_cache.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_board/cache.py tests/_board/test_cache.py
git commit -m "Add FTS5 learning search with the spec relevance order" -m "Search ranks by bm25 over a per-frontier index, so the order depends only on the frontier, then pinned, superseded, goal, verified, tags, and newest (spec 6.3). A hidden summary no longer demotes what it superseded. Queries are tokenized and quoted, so FTS5 syntax in user text never raises."
```

---

### Task 7: Leaderboard ranking

**Files:**
- Create: `src/swarm_arena/_board/leaderboard.py`, `tests/_board/test_leaderboard.py`
- Modify: `src/swarm_arena/_scoring/ranking.py` (private helper `ranked_groups`; `newest_fidelity` round tie), `tests/_scoring/test_ranking.py`, roadmap R3.1 (`_scoring/ranking.py` gains `+M2`)

**Interfaces:**
- Consumes: `Candidate`, `Score`, `Row`, `Plan` (R4.6, R4.1); `ranked`, `newest_fidelity` (`_scoring/ranking.py`, R4.13.11); `part_scoring` (M1b `_engine/parts.py`); `Position` from Task 5.
- Produces:
```python
# _scoring/ranking.py (+M2), private helper
def ranked_groups(candidates: Sequence[Candidate], scores: Iterable[Score], *, identity: Mapping[str | None, tuple[str, str, str]],
                  fidelity: str | None, every_fidelity: bool = False) -> dict[str | None, list[tuple[Candidate, Score]]]: ...
    # a group with no identity entry is left out
# _board/leaderboard.py
def pick_identity(scores: Sequence[tuple[Score, Position]], judge: str, *, run_id: str | None) -> tuple[str, str, str] | None: ...
    # newest identity (by Position) of scores named `judge`; restricted to run_id's scores when run_id is given and that run has any
def group_rows(candidates: Sequence[tuple[Candidate, Position]], scores: Sequence[tuple[Score, Position]], *,
               judge: str | Mapping[str | None, str], run_id: str | None = None, identity_run: str | None = None,
               fidelity: str | None = None, min_improvement: float | None = None) -> dict[str | None, list[Row]]: ...
def flatten(groups: Mapping[str | None, list[Row]]) -> list[Row]: ...   # the main group (None) first, then parts by name
def group_judges(plan: Plan) -> dict[str | None, str]: ...     # {None: plan.ranking_name()} plus, per part, part_scoring(plan, part)[2]
def format_value(value: float | None) -> str: ...              # f"{value:.4g}"; "-" for None (Score.value and Row.value may be None)
```
Every score value that Tasks 9, 11, 13, 14, 16, 17, and 18 print goes through `format_value`, never `.4g` directly: Task 14 makes verdict-only human scores with `value=None`.
Rules [plan]:
- `_scoring/ranking.py` stays the only owner of ranking groups and of ranking (R3.1). `ranked_groups` groups by `Candidate.part` (`None` = the main goal). Per group it calls `ranked(group, scores, identity=identity[group], fidelity=fidelity, allow_partial=False)`: one identity per call (R4.13.11). `fidelity=None` means `newest_fidelity(group)`, the M1b rule (the fidelity of the highest round). M2 adds one rule to `newest_fidelity`: a tie in the highest round, which can occur only across runs, goes to the last candidate in input order. `every_fidelity=True` calls `ranked` once per fidelity of the group and merges the results. Order inside a group: key descending, then input order (ties go to the earlier candidate; for one run this equals the `ranked` order of round, then id). A group with no ranked candidate is left out.
- `group_rows` judges each group (R2 "M2" item 8). A `str` judge applies to every group (`--by J`, `Board.leaderboard(goal, judge=J)`). A mapping gives each group its own ranking judge (`group_judges(<the revision in effect>)`); a group missing from it is left out.
- `group_rows` uses two separate run arguments. `run_id` filters candidates: only that run's candidates when it is given, else every candidate passed in. `identity_run` selects the judge identity per group: `pick_identity(<the scores of that group's candidates>, <that group's judge>, run_id=identity_run)`. A group with no identity is left out, so no identity at all → `{}`.
- `group_rows` keeps the latest score (by Position) per candidate for its group's identity. It calls `ranked_groups` with the candidates sorted by Position (commit order), the identity mapping, `fidelity`, and `every_fidelity=isinstance(judge, str) and judge.startswith("human:")` (human judges rank every fidelity together; plan ranking judges are never `human:`).
- `_board/leaderboard.py` does only three things: it selects the identity, applies `min_improvement` to first place (§4.9), and builds `Row`s. Ranks start at 1 in each group. `Row.built_on = candidate.parent_ids` and `Row.partial = candidate.output.partial`.

First place with `min_improvement` (the only algorithm here that the signature does not fix), per group:
```python
leader = None
for e in sorted(eligible, key=lambda e: position[e[0].id]):                    # commit order
    if leader is None or (e[1].key > leader[1].key and e[1].key[0] - leader[1].key[0] >= min_improvement):
        leader = e
rows = [] if leader is None else [leader] + [e for e in eligible if e[0].id != leader[0].id]   # eligible: the ranked_groups order
```

- [ ] **Step 1: Write the failing tests**

`tests/_board/test_leaderboard.py`:
```python
def test_ties_go_to_earliest() -> None: ...                 # equal keys → rank 1 is the earlier Position
def test_min_improvement_keeps_old_leader() -> None:
    # minimize values 0.0100 (earliest), then 0.0095; min_improvement=0.001
    assert [r.candidate_id for r in flatten(group_rows(cs, ss, judge="j", min_improvement=0.001))] == ["c-early", "c-late"]
    # add 0.0080 later → it leads (0.0100 - 0.0080 = 0.002 ≥ 0.001)
def test_min_improvement_rows_unique() -> None: ...         # retained leader (0.0100, then 0.0095) and replaced leader (then 0.0080):
    # in both cases len(rows) == len(eligible) and the candidate ids are unique
def test_parts_ranked_by_their_own_judges() -> None:
    # part A candidates scored by judges "a" and "b", part B candidates scored by "a" and "b", main candidates by "y"
    groups = group_rows(cs, ss, judge={None: "y", "A": "a", "B": "b"})
    assert {r.judge for r in groups["A"]} == {"a"} and {r.judge for r in groups["B"]} == {"b"}
    # each A row's value is that candidate's "a" score, each B row's value its "b" score
def test_group_judges_from_part_scoring() -> None: ...     # plan ranking "y", part A ranking "a", part B no ranking
    # → group_judges(plan) == {None: "y", "A": "a", "B": "y"}
def test_format_value() -> None:
    assert format_value(0.0041) == "0.0041"
    assert format_value(None) == "-"
def test_without_margin_best_key_leads() -> None: ...
def test_one_judge_identity_only() -> None: ...             # rubric r1 (older) and r2 (newer) → rows only r2; Row.rubric_hash == "r2"
def test_run_identity_preferred() -> None: ...              # identity_run="r1" uses r1's identity although r2's is newer
def test_run_filter_and_identity_run_are_separate() -> None:
    # c-1 (run r1) and c-2 (run r2) both scored under rubric "new"; r1 also scored c-1 under rubric "r1-latest" (newest Position)
    assert sorted(r.candidate_id for r in flatten(group_rows(cs, ss, judge="j", identity_run="r2"))) == ["c-1", "c-2"]
    assert [r.candidate_id for r in flatten(group_rows(cs, ss, judge="j", run_id="r1", identity_run="r2"))] == ["c-1"]
    assert {r.rubric_hash for r in flatten(group_rows(cs, ss, judge="j", identity_run="r2"))} == {"new"}
def test_newest_fidelity_only() -> None: ...                # "e10" in round 1, "e100" in round 2 → only e100 rows (ranking.newest_fidelity)
def test_human_ranks_all_fidelities() -> None: ...          # judge "human:alice" → both fidelities
def test_partial_and_gated_out_excluded() -> None: ...
def test_groups_ranked_separately() -> None:
    groups = group_rows(...)
    assert [r.rank for r in groups[None]] == [1, 2]
    assert [r.rank for r in groups["A"]] == [1]
    assert [r.part for r in flatten(groups)] == [None, None, "A"]
def test_latest_score_per_candidate_wins() -> None: ...     # an anchor re-score replaces the older score
def test_row_fields() -> None: ...                          # built_on, value, key, judge, rubric_hash, panel_hash
```
`tests/_scoring/test_ranking.py`: `test_ranked_groups_by_part` (each group ranked under its own `identity` entry; a group with no entry is absent), `test_ranked_groups_ties_follow_input_order`, `test_ranked_groups_every_fidelity`, `test_newest_fidelity_round_tie_goes_to_last` (two runs, both at round 2 with fidelities "a" then "b" in input order → "b").

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_leaderboard.py tests/_scoring/test_ranking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._board.leaderboard'` and `ImportError: cannot import name 'ranked_groups'`.

- [ ] **Step 3: Implement `ranked_groups` and the `newest_fidelity` tie rule in `src/swarm_arena/_scoring/ranking.py`, then `pick_identity`, `group_rows`, `flatten`, `group_judges`, `format_value` in `src/swarm_arena/_board/leaderboard.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board/test_leaderboard.py tests/_scoring tests/_engine -v && uv run ruff check . && uv run ty check`
Expected: PASS (the M1b ranking, policy, and parts tests are unchanged).

- [ ] **Step 5: Commit**

Edit roadmap R3.1: the `_scoring/ranking.py` row gets `+M2`.
```bash
git add src/swarm_arena/_board/leaderboard.py src/swarm_arena/_scoring/ranking.py tests/_board/test_leaderboard.py tests/_scoring/test_ranking.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Add leaderboard rows with one judge identity per ranking" -m "Rows rank one identity and the newest fidelity per group, each group under its own ranking judge unless one judge is given, ties go to the earliest, and min_improvement covers first place only (spec 4.4, 4.9). Groups and fidelity come from _scoring/ranking.py, so policies and leaderboards use one rule; roadmap R3.1 marks ranking.py +M2."
```

---

### Task 8: ArenaBoard and ArenaView

**Files:**
- Create: `src/swarm_arena/_board/arena.py`, `tests/_board/test_arena.py`
- Modify: `src/swarm_arena/_board/frontier.py`, `tests/_board/test_frontier.py`
- Modify: `src/swarm_arena/_board/base.py` (`BoardView.count_learnings`), `tests/_board/test_base.py`
- Modify: `src/swarm_arena/board.py` (export `ArenaBoard`), `tests/test_public_api.py`
- Modify: `docs/arena.md`, `CHANGELOG.md`, roadmap R4.6 (`count_learnings`; `cache_dir` default honors `XDG_CACHE_HOME`), roadmap R3.1 (`_board/base.py` gains `+M2`)

**Interfaces:**
- Consumes: Tasks 3 to 7; `Board`, `BoardView`, `Frontier`, `LineageGraph`, `Row`, `GoalKey` (R4.6); `read_owner` (R4.13.3); `build_frontier` (R4.13.14).
- Produces:
```python
class ArenaBoard(Board):                                    # R4.6
    class Config(BaseModel):
        arena_root: str
        cache_dir: str | None = None
    def __init__(self, config: "ArenaBoard.Config | None" = None, /, **fields: Any) -> None: ...   # no I/O
    # + every Board method (R4.6)
    def _cache_dir(self) -> Path: ...                       # private: the resolved cache dir (below)
    def _cache(self, namespace: str) -> ArenaCache: ...     # private: "<cache dir>/<namespace>.sqlite", opened on first use
class ArenaView(BoardView):                                 # private; built only by ArenaBoard.snapshot
    def __init__(self, board: ArenaBoard, goal: GoalKey, frontier: Frontier) -> None: ...
    part: str | None                                        # None: a main view; the part name: a part view ("Part views" below)
    learning_goal: tuple[str, str]                          # (main goal key digest, goal id) of the resolving row; a main or
        # no-candidate view: (goal_key_digest(goal), goal.id). The key of goal-scoped learning reads ("Part views" below)
# _board/base.py, new optional method on the public ABC (roadmap R4.6 edit):
def count_learnings(self, namespace: str, *, goal: GoalKey | None = None) -> int: ...
    # default len(self.learnings(namespace, goal=goal, limit=sys.maxsize)); never counts hidden learnings
```
Behavior [plan]:
- **Cache dir:** `config.cache_dir`, else `$XDG_CACHE_HOME` (default `~/.cache`) + `/swarm-arena/<sha256_hex(resolved arena_root)[:12]>`. A local `arena_root` is resolved with `Path.resolve()` before hashing. Each namespace's cache is `<cache_dir>/<namespace>.sqlite`, opened lazily.
- **`open(run_id, run_root, namespaces)`:** remember the namespaces. With a non-empty `run_id`, write a pointer in each namespace with `run_root` resolved to an absolute path (unchanged for `s3://`). When `write_pointer` raises `StorageError`, read the existing pointer: if it parses and has the own `run_id` with another `run_root` (a resume from another mount or another cwd), log one `swarm_arena.board` warning that names both paths and keep the old pointer. Any other case re-raises. `write_pointer` is unchanged. The own run (the `run_id` given to `open`) is always read through the `run_root` given to `open`, resolved, and never through its pointer. `open("", "", namespaces)` attaches readers only and writes nothing (CLI, viewer).
- **`close()`:** closes every opened `ArenaCache`, which releases its `.lock` flock (Task 4).
- **`frontier()`:**
  1. `list_pointers` for each namespace; take the union of runs. For the own run, use the open `run_root` in place of its pointer's.
  2. For each run: `read_owner(run_store(...))` (the authority read), then `ingest_run` into every opened namespace cache. The own run's authority read and `ingest_run` go through the open `run_root`, and the own mark's `run_root` is that path. Other runs keep using their pointers. So a resumed run that moved does not crash on its first frontier (old path gone) or mark a stale head (old path still holds a copy).
  3. Skip a run, with a `logging.getLogger("swarm_arena.board")` warning, if its pointer is bad, its root is unreachable (`StorageError` or `OSError`), its root needs an extra this reader lacks (`ConditionalWritesUnsupported`, a `StorageError`, from `open_store`; for example an `s3://` pointer in a local arena read without `swarm-arena[s3]`), it has no `owner.json`, or its head is `None`. For the own run (the `run_id` given to `open`), errors are raised, never skipped. The one exception: an own head of `None` (no commit yet) gives no mark and no error. M1b commits `plan_revision` r0 before the first round's frontier (M1b `RunRunner` commits every goal's r0 at run start, before any goal starts), so every recorded `frontier` event of a run has its own mark.
  4. `list_human_objects`, then read and apply the objects not yet cached.
  5. Return `build_frontier(...)`. Remember it as the latest frontier.
- **`snapshot(goal, frontier)`:** `ensure_covered` for every mark (through the mark's `run_root`, never a pointer), read every named object under `<goal.namespace>/` not yet cached, and return `ArenaView`, with `goal` resolved as in "Part views". A named object that is missing → `StorageError`.
- **Part views:** M1b gives part agents the part sub-goal key and calls `board.snapshot(slot.goal_key, ...)` with it (spec §6.2). `snapshot(goal, f)` resolves `goal_key_digest(goal)` against rows whose `plan_revision` is in `f` (the run has a mark and `plans.seq <= mark.seq`). A `goals` row with that `goal_key` → a main view (`part = None`): candidates filtered on `goal_key` and grouped by part, as above. Otherwise, `part_goals` rows with that `part_key` → a part view (`part` = the row's part name): candidates filtered on `part_key` and ranked as one group; `best` and `leaderboard` accept `part=None` or that part name, and any other part raises `ValueError`. `snapshot` sets `learning_goal` to the main key and goal id of the resolving row; with several rows, the smallest `(run_id, goal_id, revision)`. Goal-scoped learning reads and the relevance goal use `learning_goal`. Neither → a main view with no candidates. A main view and a no-candidate view have `learning_goal = (goal_key_digest(goal), goal.id)`.
- **Board-level reads** use the latest frontier, and call `frontier()` once if there is none. `put_*` keep the item in an in-memory hint map. Only `ArenaBoard.get_candidate`, `get_learning`, and `scores` fall back to hints. `ArenaView` never reads hints.
- **Reads of `Board` and `ArenaView`:** hidden learnings are never returned. An unknown id raises `KeyError` (use the same type `RunBoard` raised in M1b, so `AgentContext`'s mapping to `SoftError` stays). A `goal` or `namespace` argument that differs from the view's raises `ValueError`.
- **`best`, `leaderboard`:** candidates come from the whole namespace (§4.9 scope table), and the identity is this run's current one (§4.9 "One judge identity per ranking"). Both call `group_rows(..., identity_run=<the board's own run_id>, run_id=<the caller's run_id>, fidelity=<the caller's fidelity>)`. The own run id is the `run_id` given to `open`; `open("", "", ...)` gives `identity_run=None`, so the newest identity is used (§4.9, a reader with no current run). A run with no score of its own yet also gets the newest identity for that judge name in the namespace (roadmap R5.4 item 17, a deviation from §4.9: the board sees identities only through scores). `best(..., part=p)` returns the top k Candidates of group `p` (`None` = the main goal). `leaderboard(goal, *, judge)` returns `flatten(group_rows(..., identity_run=<own run_id>))` with no `run_id` filter and no `min_improvement`.
- **Learning reads of `ArenaView`:** `learnings` and `search_learnings` call `ArenaCache.find` directly, never `ArenaCache.search`. Each passes `relevance_goal=view.learning_goal`, also when the `goal` argument is `None`. So the digest's whole-namespace list and an agent's `scope="namespace"` search still put this goal first (§6.3). The `goal` argument is `None` or `view.goal`, else `ValueError`; `view.goal` sets the filter `LearningFilter(goal_key=view.learning_goal[0], goal_id=view.learning_goal[1])` (the Task 6 goal rule). `count_learnings` calls `ArenaCache.count` with the same filter. `goal_key_digest(view.goal)` (the part key in a part view) stays the key for candidate filtering and repeat lookup.
- **`lineage`:** breadth-first over `parent_ids` inside the view, with a visited set; edges only between resolvable nodes.
- **`count_learnings`:** `ArenaView` overrides it with `ArenaCache.count`.
- **`frontier.py` +M2:** `build_frontier` sorts marks by `run_id`, sorts the object names with `human_object_order`, and removes duplicates, so `Frontier.digest()` does not depend on read order.

- [ ] **Step 1: Write the failing tests**

`tests/_board/test_arena.py`:
```python
def test_init_does_no_io(tmp_path: Path) -> None:
    ArenaBoard(arena_root=str(tmp_path / "arena"), cache_dir=str(tmp_path / "cache"))
    assert not (tmp_path / "arena").exists()
    assert not (tmp_path / "cache").exists()
def test_default_cache_dir_uses_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert ArenaBoard(arena_root="/x/arena")._cache_dir() == tmp_path / "swarm-arena" / sha256_hex(b"/x/arena")[:12]
def test_open_writes_pointer_per_namespace(tmp_path: Path) -> None: ...      # open("r1", "runs", ["toy", "phys"]) → two pointers with absolute run_root
def test_reader_open_writes_nothing(tmp_path: Path) -> None: ...             # open("", "", ["toy"]) → arena dir unchanged
def test_open_resume_with_other_run_root_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None: ...
    # pointer r1 → /a exists; open("r1", "/b", ["toy"]) returns, logs one swarm_arena.board warning, and the pointer bytes are unchanged
def test_own_run_read_through_open_root_after_move(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # FakeRun r1 commits a plan and candidate c-1 under a = tmp_path / "a"; a board opens ("r1", str(a), ["toy"]) and closes
    shutil.move(a, b)                                                  # b = tmp_path / "b"; the pointer still names a
    board.open("r1", str(b), ["toy"])                                  # a new board; logs one swarm_arena.board warning
    f = board.frontier()                                               # does not raise, though a is gone
    (mark,) = [m for m in f.marks if m.run_id == "r1"]
    assert mark.run_root == str(b.resolve())
    assert (mark.epoch, mark.seq) == <the head of b's owner.json>
    assert board.snapshot(goal, f).get_candidate("c-1") == <c-1>
    # a reader board open("", "", ["toy"]): its frontier has no r1 mark and logs one swarm_arena.board warning
    # (other readers keep the write-once pointer)
def test_frontier_marks_every_run_with_a_head(tmp_path: Path) -> None: ...   # marks sorted by run id; each equals that run's owner.json head
def test_frontier_skips_broken_runs(tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:   # Review Focus 3
    # good run r1; pointers: r2 → run_root = a regular file tmp_path / "not-a-dir" (NotADirectoryError, an OSError);
    # r3 with no owner.json; r4 with a lease and head None; raw corrupt bytes at toy/runs/r5.json;
    # schema_version 99 at toy/runs/r6.json; r7 → "s3://b/runs" with monkeypatch.setitem(sys.modules, "s3fs", None) (ConditionalWritesUnsupported)
    assert [m.run_id for m in board.frontier().marks] == ["r1"]
    assert len([r for r in caplog.records if r.name == "swarm_arena.board"]) == 6
def test_frontier_raises_for_own_run(tmp_path: Path) -> None: ...            # opened with run_id "r9" whose run_root is a regular file → the OSError or StorageError is raised
def test_own_run_without_head_gets_no_mark(tmp_path: Path) -> None: ...      # opened as "r9" with a lease and head None → frontier() returns, no r9 mark, no warning
def test_snapshot_is_exactly_the_frontier(tmp_path: Path) -> None:
    f = board.frontier()
    run2.commit(run2.learning("L-late", "late"))
    # then write a pin object
    view = board.snapshot(goal, f)
    assert "L-late" not in [x.id for x in view.learnings("toy")]
    assert view.frontier == f
    assert "L-late" in [x.id for x in board.snapshot(goal, board.frontier()).learnings("toy")]
def test_snapshot_same_on_fresh_cache(tmp_path: Path) -> None: ...           # second board, empty cache_dir → same membership bytes
def test_view_rejects_other_goal_or_namespace(tmp_path: Path) -> None: ...   # ValueError
def test_hints_never_in_snapshot(tmp_path: Path) -> None:
    board.put_candidate(c, epoch=1, seq=99)
    assert board.get_candidate(c.id) == c
    with pytest.raises(KeyError):
        board.snapshot(goal, board.frontier()).get_candidate(c.id)
def test_gated_out_kept_but_not_ranked(tmp_path: Path) -> None: ...        # view.get_candidate works; not in view.best(...)
def test_hidden_never_returned(tmp_path: Path) -> None: ...                 # hide object named → absent from learnings, search, get_learning (KeyError)
def test_best_ranks_namespace_under_own_identity(tmp_path: Path) -> None:
    # r1: c-1 scored under rubric "old" (older), then "new", then "r1-latest" (the newest score in the namespace)
    # r2: c-2 scored under rubric "new"; board opened as "r2"
    view = board.snapshot(goal, board.frontier())
    assert sorted(c.id for c in view.best(goal, 5, judge="j")) == ["c-1", "c-2"]          # whole namespace, r2's identity
    assert {r.rubric_hash for r in view.leaderboard(goal, judge="j")} == {"new"}
    assert [c.id for c in view.best(goal, 5, judge="j", run_id="r1")] == ["c-1"]          # the caller's run_id filters candidates only
def test_view_for_part_key(tmp_path: Path) -> None:
    # r1 and r2 both commit plan(parts=["A"]) (the same part sub-goal), part-A candidates, and main-goal learnings
    view = board.snapshot(part_key, board.frontier())                  # part_key: GoalKey of the part-A goal; main_key: of the main goal
    assert sorted(c.id for c in view.best(part_key, 5, judge="j")) == <both runs' part-A candidate ids>
    with pytest.raises(ValueError):
        view.best(part_key, 5, judge="j", part="B")
    assert view.learning_goal == (goal_key_digest(main_key), main_key.id)
    assert {x.id for x in view.learnings("toy", goal=part_key)} == <the main goal's learning ids>
    assert view.count_learnings("toy", goal=part_key) == len(<the main goal's learning ids>)
    # a same-text learning of another goal g2 in "toy" ranks after the main goal's learning
    # in view.search_learnings("toy", <a word of that text>, goal=None)
def test_namespace_search_puts_own_goal_first(tmp_path: Path) -> None:
    # L-mine (goal g1) and L-other (goal g2 in the same namespace), equal title, text, tags, flags; L-other is newer
    view = board.snapshot(goal_g1, board.frontier())
    assert [x.id for x in view.search_learnings("toy", "viscosity", goal=None)] == ["L-mine", "L-other"]
    assert [x.id for x in view.learnings("toy")][:2] == ["L-mine", "L-other"]
def test_lineage_follows_parents_across_runs(tmp_path: Path) -> None: ...   # c-2 (run r2) built on c-1 (run r1): edges == [("c-2", "c-1")]
def test_count_learnings_matches_learnings(tmp_path: Path) -> None: ...
@pytest.mark.s3
def test_frontier_and_snapshot_on_s3(s3_url: str) -> None: ...              # FakeRun on s3_url + "/runs", arena on s3_url + "/arena"; snapshot finds the candidate
```
`tests/_board/test_frontier.py`: `test_build_frontier_is_order_independent`: the same reads in two orders give equal `digest()`.
`tests/_board/test_base.py`: `test_count_learnings_default_counts_learnings`, using a minimal `BoardView` subclass.
`tests/test_public_api.py`: `swarm_arena.board.__all__` now holds `ArenaBoard`, and `swarm_arena.board.ArenaBoard.__module__ == "swarm_arena.board"`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_arena.py tests/_board/test_frontier.py tests/_board/test_base.py tests/test_public_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._board.arena'`.

- [ ] **Step 3: Implement `count_learnings`, the `build_frontier` ordering, `ArenaBoard`, `ArenaView`, and the facade export**

Write numpydoc docstrings with `Examples` for `ArenaBoard` and each of its public methods. The class example is a real doctest:
```python
>>> import tempfile
>>> from swarm_arena.board import ArenaBoard
>>> root = tempfile.mkdtemp()
>>> board = ArenaBoard(arena_root=root + "/arena", cache_dir=root + "/cache")
>>> board.open("", "", ["physika"])
>>> board.frontier().marks
[]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board tests/test_public_api.py tests/test_docstrings.py -v && uv run pytest -m s3 tests/_board -v && uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Docs**

`docs/arena.md`, sections "What the arena stores" (the §7.2 tree: run journals are the record; pointers, reviews, curation; write-once), "Snapshots and frontiers" (one authority read per run; the same frontier gives the same view; deleting a pointer removes the run from later snapshots only; and this sentence: "A run resumed from another local path is read through its new path by its own driver. Pointers are write-once, so other readers keep its first pointer and see the run only while that path is reachable. Keep a shared local run_root at one stable path."), "The local cache" (location, `XDG_CACHE_HOME`; safe to delete only when no swarm-arena process uses it, and `swarm-arena reindex` refuses while one does). Edit roadmap R4.6, and R3.1 (the `_board/base.py` row gets `+M2`). Confirm, with no edit, that roadmap R5.4 item 17 gives the identity fallback above (Task 9 `test_past_run_candidate_as_parent` relies on it). Add a CHANGELOG line.

- [ ] **Step 6: Commit**

```bash
git add src/swarm_arena/_board src/swarm_arena/board.py tests/_board tests/test_public_api.py docs/arena.md CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Add ArenaBoard: multi-run frontiers and exact snapshots" -m "Frontiers mark every run in the namespace by its committed head, and snapshots filter by frontier, so views are the same on any machine (spec 4.9, 7.2). A part agent's snapshot by its part key gives a part view (spec 6.2). Roadmap R4.6 gains BoardView.count_learnings and the XDG cache default; R3.1 marks _board/base.py +M2."
```

---

### Task 9: ArenaBoard as the default board

**Files:**
- Delete: `src/swarm_arena/_board/run_board.py`, `tests/_board/test_run_board.py`
- Modify: the board builder in `src/swarm_arena/_engine/run.py` and `src/swarm_arena/_engine/resume.py` (find it with `grep -rn RunBoard src/`)
- Modify: `src/swarm_arena/_agent/tool_run.py` (the past-run hint reaches the manager), `tests/_agent/test_tool_run.py`
- Modify: `src/swarm_arena/_engine/round.py` (+M2: past-run repeat entries in each agent's `RepeatIndex`), `tests/_engine/test_round.py`
- Modify: `src/swarm_arena/_board/cache.py` (`ArenaCache.repeat_entries`), `src/swarm_arena/_board/arena.py` (`past_run_repeat_entries`), `tests/_board/test_cache.py`, `tests/_board/test_arena.py`
- Modify: `tests/conftest.py` (autouse `XDG_CACHE_HOME`; fixture `finished_run`), `tests/_engine/test_run.py`
- Modify: `src/swarm_arena/testing/_harness.py` (`OfflineHarness`, behind the public `check_manager`; M1b Task 21), `tests/testing/test_harness.py`, `tests/testing/test_conformance.py`
- Modify: the M1b test fixtures that build a `RunBoard`: `tests/_fakes/engine.py` (`RtEnv.board` and the `EngineEnv` board become `ArenaBoard`), `tests/_agent/conftest.py` (`rt_env`, M1b Task 19), `tests/_engine/conftest.py` (`env`, M1b Task 22)
- Modify: roadmap R3.1 (`_engine/round.py`, `_agent/tool_run.py`, and `testing/_harness.py` gain `+M2`; the `_engine/resume.py` row already has it, Task 2), roadmap R4.6 (`Candidate.warnings` comment: soft staging warnings and the past-run hint)

**Interfaces:**
- Consumes: Task 8 `ArenaBoard`, `ArenaView`; Task 7 `format_value`, `group_judges`; Task 4 `candidates.repeat_key`, `candidates.part_key`, `goal_key_digest`, `FakeRun`, `add_pointer`; `Swarm`, `Plan.load`, `RunConfig`, `Ref`; `RepeatIndex`, `RepeatOutcome.hint` (R4.13.10), `repeat_key`, `COUNTED` (M1b `_agent/repeats.py`); `RoundRunner` (R4.13.13).
- Produces:
```python
@dataclass(frozen=True, kw_only=True)
class FinishedRun:                         # tests/conftest.py
    config: RunConfig                      # roots under tmp_path, board=None
    config_path: Path                      # the same config as run.toml (RunConfig.dumps)
    plan: Plan                             # examples/rosenbrock/plan.toml, 2 RoundSpecs, every ScriptedManager with post_learnings=True
    run_id: str
    namespace: str
@pytest.fixture
async def finished_run(tmp_path: Path, anyio_backend: str) -> FinishedRun: ...
async def run_again(run: FinishedRun, *, run_id: str | None = None, plan: Plan | None = None) -> str: ...
    # same config; plan defaults to run.plan; returns the new run id
# _board/cache.py, private ArenaCache method
def repeat_entries(self, frontier: Frontier, *, goal_key: str, exclude_run: str) -> list[tuple[str, Candidate]]: ...
    # (repeat_key, candidate) of the candidates in the frontier whose candidates.part_key equals goal_key (the main key for
    # main-goal candidates, the part key for part candidates, Task 4), run_id != exclude_run, and repeat_key not NULL;
    # in Position (commit) order
# _board/arena.py, private helper
def past_run_repeat_entries(view: BoardView, run_id: str) -> list[tuple[str, Candidate]]: ...
    # an ArenaView → its cache's repeat_entries(view.frontier, goal_key=goal_key_digest(view.goal), exclude_run=run_id);
    # any other BoardView (a custom RunConfig.board) → [], so it gives no past-run hint
# _engine/round.py (+M2), private helper
def _past_run_entries(view: BoardView, run_id: str) -> list[tuple[str, Candidate]]: ...
    # past_run_repeat_entries(view, run_id), then per repeat key only the newest entry with status in COUNTED and
    # not partial (the entries RepeatIndex.lookup counts), so the hint is the same whatever the lookup order
```
`finished_run` requests `anyio_backend`. The anyio pytest plugin runs an async fixture only when `anyio_backend` is in the test's fixture closure, and auto mode adds it only to coroutine tests. With the parameter, the sync CLI tests of Tasks 17 and 18 (sync because `main()` calls `anyio.run`) also get a finished run, not an unhandled async fixture.

The past-run hint (§6.2, §4.9: "the manager gets its id and score"). M1b's `RepeatIndex.lookup` returns `RepeatOutcome.hint` for a match of another run, but M1b gives the index no past-run entries (M1b Task 22: "past-run candidates join with the M2 board") and never shows a hint to the manager. M2 does both:
- **Entries** [plan]. The cache keeps each candidate's `tool_end.repeat_key` (Task 4), because a reader cannot compute the key of a past-run candidate again. In `RoundRunner.run`, right after each slot's `view = to_thread(board.snapshot, ...)`, the agent's index becomes `RepeatIndex(deps.run_id, [*repeat_entries, *own resumed entries, *_past_run_entries(view, deps.run_id)])`. The past-run entries come last, so a match of this run (`refuse`, `return_own`, `link`) still wins. They are the round-start snapshot's other-run candidates of this goal key, so they are fixed by the recorded frontier: a restarted round (resume) gets the same entries. They never go into `RepeatIndex.added`, so `RoundOutcome.repeat_entries`, `GoalRunner.repeat_entries`, and `GoalState.repeat_keys` stay this run only.
- **Message.** In `execute_tool_run`, a `run` outcome with a `hint` adds one entry to the returned candidate's `warnings`, after the staging warnings and before any tool error text: `past-run match <hint id>: <ranking judge>=<format_value(value)>` (Task 7), from the hint's latest score by the ranking judge of the agent's group in the agent's view (`group_judges(plan)[<the agent's part>]`), or `past-run match <hint id>: unscored`. The request still runs.

The board of the harness and of the M1b fixtures [plan]. Each one builds `ArenaBoard(arena_root=<arena>, cache_dir=<cache>)` with an explicit `cache_dir`, then calls `board.open(run_id, <run_root>, [goal.namespace])`. Its run store must be `open_store(<run_root>).child(run_id)`, so that the own-run authority read finds `control/owner.json` (Task 8).
- `OfflineHarness`: `<tmp>` is its own temp dir (the one that holds its run folder). The run store is `<tmp>/runs/local` (`IdGen("local", 1)` gives the run id `local`), `<arena>` is `<tmp>/arena`, and `<cache>` is `<tmp>/cache`. The harness never uses `config.board`, `config.arena_root`, or the default cache dir. So `check_manager` writes no pointer into a real arena and writes nothing under `~/.cache`.
- `rt_env` and `env`: `<arena>` is `run_config.arena_root`, `<run_root>` is `run_config.run_root`, and `<cache>` is `tmp_path / "cache"`.
- `ArenaBoard` learns a goal's key and namespace only from a committed `plan_revision` r0 (Task 4 `goals` table), and a part's key only from a committed revision's parts (`part_goals`). Its snapshots hold only committed events, because `put_*` rows are hints (Task 8). So each of them commits a `plan_revision` r0 before its first `frontier()` if it does not do this already. A fixture test that needs an item in a snapshot commits the event of that item first.
- Delete each call to `RunBoard.register_goal` and `RunBoard.from_state`. `ArenaBoard` needs neither.

- [ ] **Step 1: Write the failing tests**

`tests/_engine/test_run.py`:
```python
async def test_default_board_is_arena(finished_run: FinishedRun) -> None:
    arena = Path(finished_run.config.arena_root)
    assert (arena / finished_run.namespace / "runs" / f"{finished_run.run_id}.json").exists()
    # every `frontier` event in the run journal has a mark for finished_run.run_id (M1b commits r0 before round 1's frontier, Task 8)
async def test_second_run_sees_first_but_never_selects_it(finished_run: FinishedRun) -> None:
    run2 = await run_again(finished_run)
    # run2's round-1 recorded frontier has a mark for finished_run.run_id, and its snapshot holds run 1's candidates
    # every finalist that Swarm.run returned for run2 has run_id == run2
async def test_past_run_match_runs_again(finished_run: FinishedRun) -> None:
    run2 = await run_again(finished_run)
    # gd-1's round-1 request equals its run-1 request (deterministic tool): run 2 makes a new candidate c2 with c2.run_id == run2
    # the manager got the hint: f"past-run match {c1.id}: f={format_value(s1.value)}" in c2.warnings, where c1 is run 1's candidate
    # for that request and s1 its latest score by the ranking judge "f"
async def test_past_run_candidate_as_parent(finished_run: FinishedRun) -> None:
    # plan2 = finished_run.plan with gd-1's round-1 request set to parents="best"; run 2 has no candidates yet, so
    # ctx.best(1) is run 1's top candidate c1 (no run-2 scores yet, so the newest identity is used: Task 7, roadmap R5.4 item 17)
    run2 = await run_again(finished_run, plan=plan2)
    # gd-1's round-1 candidate c2 of run2: c2.parent_ids == [c1.id]; c2.request.parents[0].run_id == finished_run.run_id;
    # c2.request.parents[0].data == c1.output.data; every c2.request.parents[0].files entry reads back from artifact_root
    # (staged from the shared artifact_root); run 2's lineage(c2.id) has the edge (c2.id, c1.id); no run2 finalist is c1
@pytest.mark.s3
async def test_run_with_all_roots_on_s3(s3_url: str, tmp_path: Path) -> None: ...  # run_root, arena_root, artifact_root under s3_url; the run finishes; solution.json exists
```
`tests/_agent/test_tool_run.py`: `test_past_run_hint_in_warnings` (a view with a past-run candidate for the same repeat key → the request runs, and `warnings` holds the `past-run match` entry with its score; an unscored hint → `unscored`).

`tests/_engine/test_round.py` (the `env` fixture now has an `ArenaBoard` on `run_config.arena_root`; `RANKING` is `PLAN`'s ranking judge name):
```python
async def test_round_gives_past_run_hint(env: EngineEnv, run_config: RunConfig, tmp_path: Path) -> None:
    slot = _runner("a")
    past = FakeRun(str(tmp_path / "past-runs"), "r-past", goal=slot.goal)
    add_pointer(run_config.arena_root, past)
    key = repeat_key(goal_key=slot.goal_key, tool_ref=slot.tool_ref, request=<the one square request of _runner>, deterministic=True)
    past.commit(past.plan(ranking=RANKING), past.candidate("c-past", repeat_key=key), past.score("s-past", "c-past", 0.25, judge=RANKING))
    out = await _round(env, [slot]).run(1)
    (c,) = out.candidates
    assert c.run_id == env.deps.run_id                                        # the request ran
    assert f"past-run match c-past: {RANKING}={format_value(0.25)}" in c.warnings   # the manager got the hint
    assert out.repeat_entries == [(key, c)]                                   # past-run entries stay out of RoundOutcome
async def test_round_gives_past_run_hint_for_part_agent(env: EngineEnv, run_config: RunConfig, tmp_path: Path) -> None: ...
    # a part slot (team with part "A"; slot.goal_key is the part-A key); the past run commits plan(parts=["A"]) with the slot's
    # part-A goal and a part-A candidate c-past whose repeat key uses goal_key=slot.goal_key
    # → a "past-run match c-past: " entry in c.warnings
async def test_round_hint_is_newest_counted_past_match(env: EngineEnv, run_config: RunConfig, tmp_path: Path) -> None: ...
    # same setup; the past run commits c-old (ok), then c-new (ok), then c-late (status=TIMEOUT), all with the same key
    # → "past-run match c-new: unscored" in c.warnings (no scores committed)
async def test_own_match_wins_over_past_run(env: EngineEnv, run_config: RunConfig, tmp_path: Path) -> None: ...
    # same key in the past run and in repeat_entries=[(key, an ok candidate of this run by slot.address)] → the request is refused (SoftError
    # to the manager, a watchdog {"refused": ...} event), no new candidate
```
`tests/_board/test_cache.py`: `test_repeat_entries_filter` (runs r1 and r2 in one goal key, r3 in another goal key; a keyed and an unkeyed candidate each; one r2 commit after the frontier → `repeat_entries(f, goal_key=<digest>, exclude_run="r1")` is only r2's keyed candidate inside `f`, in commit order).
`tests/_board/test_arena.py`: `test_past_run_repeat_entries` (an `ArenaView` opened as `r1` → the other runs' keyed candidates of its goal key; a minimal non-arena `BoardView` → `[]`).

`tests/testing/test_harness.py`:
```python
async def test_harness_uses_a_temp_arena_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    h = OfflineHarness()
    await h.run_rounds(ScriptedManager([{"inputs": {"x": 1.0}}]), goal=GOAL, tool=SQUARE, rounds=1, limits=Limits(),
                       config=RunConfig(arena_root=str(tmp_path / "real-arena")))
    tmp = Path(h.store.url).parent.parent                  # h.store is <tmp>/runs/local
    assert (tmp / "arena" / GOAL.namespace / "runs" / "local.json").exists()
    assert (tmp / "cache" / f"{GOAL.namespace}.sqlite").exists()
    assert not (tmp_path / "real-arena").exists() and not (tmp_path / "xdg").exists()
```
`tests/testing/test_conformance.py`:
```python
async def test_check_manager_after_run_board_removal() -> None:
    m = ScriptedManager([{"inputs": {"x": 1.0}, "round": 1}, {"inputs": {"x": 2.0}, "round": 2}])
    assert len(await check_manager(m, goal=GOAL, tool=SQUARE, rounds=2)) == 2
```
The M1b tests in `tests/_agent/test_scripted.py` and `tests/testing/` stay unchanged and must pass on `ArenaBoard`. `test_round_filter_and_best_parent` and `test_post_learnings_cites_best_own_candidate` show that the round-2 snapshot holds the committed round-1 candidate and score.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_engine/test_run.py tests/_engine/test_round.py tests/_agent/test_tool_run.py tests/_board/test_cache.py tests/_board/test_arena.py tests/testing -v`
Expected: FAIL: no `runs/<run-id>.json` pointer (the default is still `RunBoard`), no harness pointer or cache file, `AttributeError: 'ArenaCache' object has no attribute 'repeat_entries'`, and no `past-run match` warning.

- [ ] **Step 3: Switch the default and delete `RunBoard`**

- `RunConfig.board is None` → `ArenaBoard(arena_root=config.arena_root)`.
- The engine calls `board.open(run_id, config.run_root, sorted(goal namespaces))` after it takes the lease and before the first frontier, on `run` and on `resume`, and `board.close()` at run end. Add the calls if M1b did not.
- Add `ArenaCache.repeat_entries`, `past_run_repeat_entries`, and `_past_run_entries`, and give each agent's `RepeatIndex` the past-run entries in `_engine/round.py` (Interfaces above).
- Add the past-run hint to `warnings` in `_agent/tool_run.py` (Interfaces above).
- Change `OfflineHarness`, `tests/_fakes/engine.py`, `rt_env`, and `env` to `ArenaBoard` (Interfaces above).
- Delete `run_board.py` and its test.
- In `tests/conftest.py`, an autouse fixture sets `XDG_CACHE_HOME` to a per-test tmp dir, so no test writes `~/.cache`.

- [ ] **Step 4: Check that the engine reads no live state from the Board**

Run: `grep -rnE "RunBoard|register_goal" src tests; grep -rnE "\.board\.(best|learnings|search_learnings|get_learning|get_candidate|scores|lineage|leaderboard)\(" src/swarm_arena/_engine src/swarm_arena/_agent src/swarm_arena/testing`
Expected: no output. A `RunBoard` or `register_goal` hit is in a file of the Files list: change it as in Step 3. A board read hit: change the call to read `RunState` (this run's state) or the agent's `BoardView`. Board-level reads of `ArenaBoard` see only committed, ingested data.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v && uv run pytest -m s3 tests/_engine/test_run.py -v && uv run ruff check . && uv run ty check`
Expected: PASS, including every M1b e2e test (`tests/e2e/test_many_goals_parts.py` among them: its part agents snapshot by their part key, Task 8 "Part views") and every `tests/testing` test (`check_manager` with no `RunBoard`).

- [ ] **Step 6: Commit**

Edit roadmap R3.1: the `_engine/round.py`, `_agent/tool_run.py`, and `testing/_harness.py` rows get `+M2`. Edit roadmap R4.6: the `Candidate.warnings` comment names the past-run hint entry.
```bash
git add -A src/swarm_arena/_board src/swarm_arena/_engine src/swarm_arena/_agent src/swarm_arena/testing tests docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Make ArenaBoard the default board and remove RunBoard" -m "Every run now writes its pointer and reads the namespace arena by frontier (roadmap M2 item 3). Decisions still use this run only. A past-run match reaches the manager as a hint with its id and score (spec 6.2): the cache keeps each tool_end repeat key, and the round runner gives each agent's repeat index the other-run candidates of the round-start snapshot. The check_manager harness and the M1b test fixtures use an ArenaBoard on a temporary arena_root with an explicit cache_dir; roadmap R3.1 marks round.py, tool_run.py, and testing/_harness.py +M2."
```

---

### Task 10: Learning posting rules

**Files:**
- Create: `src/swarm_arena/_board/learnings.py`, `tests/_board/test_learnings.py`
- Modify: `src/swarm_arena/_agent/context.py` (`post_learning`), `tests/_agent/test_context.py`
- Modify: `src/swarm_arena/_agent/scripted.py`, `tests/_agent/test_scripted.py`
- Modify: `docs/arena.md` (section "Posting learnings"), `CHANGELOG.md`, roadmap R3.1 (`_agent/context.py` and `_agent/scripted.py` gain `+M2`)

**Interfaces:**
- Consumes: `BoardView`, `Learning`, `Candidate`, `RunStatus`, `SoftError`.
- Produces:
```python
DUPLICATE_JACCARD: Final = 0.6
LEARNING_GUIDE: Final = "Write for a colleague: what you tried, what happened, and the number."   # §6.3.1; M3 prompts reuse it
def title_tokens(title: str) -> frozenset[str]: ...          # re.findall(r"\w+", title.lower())
def jaccard(a: frozenset[str], b: frozenset[str]) -> float: ...   # len(a & b) / len(a | b); 0.0 when both are empty
def find_duplicates(view: BoardView, own: Sequence[Learning], *, namespace: str, title: str) -> list[Learning]: ...
    # the bm25 top hit of view.search_learnings(namespace, title, limit=1) if jaccard ≥ 0.6, plus own learnings with jaccard ≥ 0.6
def is_verified(evidence: Sequence[Candidate]) -> bool: ...  # any complete (not partial), status ok, not gated_out
def vet_learning(view: BoardView, own: Sequence[Candidate | Learning], *, namespace: str, author: str, title: str,
                 text: str, evidence: Sequence[str], refines: Sequence[str], duplicate_reason: str | None) -> bool: ...
    # returns verified; raises SoftError: empty, multi-line, or > 120-char title; text > 1,500 chars (both messages end with
    # LEARNING_GUIDE); no evidence unless author starts with "human:"; an evidence id that is no candidate in the view or own;
    # a refines id that is no learning in the view or own; duplicates with no duplicate id in refines and a blank duplicate_reason
```
The duplicate message: `near-duplicate of L-…: "<title>". Cite it with refines=["L-…"] or pass duplicate_reason.`

- [ ] **Step 1: Write the failing tests**

```python
def test_title_limit_120(view: BoardView) -> None: ...          # "x" * 120 passes; "x" * 121 → SoftError
def test_text_limit_1500(view: BoardView) -> None: ...
def test_empty_or_multiline_title_rejected(view: BoardView) -> None: ...
def test_evidence_must_be_in_view(...) -> None: ...             # a candidate committed after the frontier → SoftError "not in your view"
def test_own_candidate_counts_as_evidence(...) -> None: ...
def test_duplicate_requires_refines_or_reason(...) -> None:
    # the view has "Fitting log(nu) instead of nu converges faster" (L-old) and "Grid spacing h=0.01 is stable" (L-other,
    # a learning in the view that is not a duplicate of the title below)
    title = "Fitting log(nu) instead of nu converges 3x faster"
    with pytest.raises(SoftError, match="L-old"):
        vet_learning(view, [], namespace="toy", author="t1/a1", title=title, text="t", evidence=["c-1"], refines=[], duplicate_reason=None)
    assert vet_learning(..., refines=["L-old"], duplicate_reason=None) in (True, False)
    assert vet_learning(..., refines=[], duplicate_reason="new lr") in (True, False)
    with pytest.raises(SoftError, match="near-duplicate of L-old"):          # L-other is known, but it is not a duplicate
        vet_learning(..., refines=["L-other"], duplicate_reason=None)
def test_jaccard_threshold_is_inclusive() -> None:
    assert jaccard(frozenset("abc"), frozenset("abcde")) == 0.6
def test_hidden_learning_is_not_a_duplicate(...) -> None: ...
def test_own_learning_duplicate_detected(...) -> None: ...
def test_verified_needs_complete_passing_evidence() -> None: ...   # partial → False; gated_out → False; error status → False; ok complete passing → True
```
`tests/_agent/test_context.py`: `test_post_learning_sets_verified_from_evidence`, `test_post_learning_duplicate_is_soft_error`.
`tests/_agent/test_scripted.py`: `test_scripted_manager_posts_every_round` (2 rounds with `post_learnings=True` → 2 learnings).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_learnings.py tests/_agent/test_context.py tests/_agent/test_scripted.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._board.learnings'`.

- [ ] **Step 3: Implement `_board/learnings.py`, then wire it in**

`AgentContext.post_learning` calls `vet_learning(view, own items, namespace=goal.namespace, author="<team>/<agent>", …)` in place of M1b's inline evidence and refines checks, and stores `verified` from its result. `ScriptedManager` posts with `duplicate_reason="scripted manager: one learning per round"`, so the duplicate rule never refuses a baseline.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board tests/_agent -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Docs**

`docs/arena.md`, "Posting learnings": the limits (120 and 1,500), evidence in the agent's view, the duplicate rule (Jaccard ≥ 0.6, `refines` or `duplicate_reason`), `verified`, and `LEARNING_GUIDE`. Add a CHANGELOG line. Edit roadmap R3.1: the `_agent/context.py` and `_agent/scripted.py` rows get `+M2`.

- [ ] **Step 6: Commit**

```bash
git add src/swarm_arena/_board/learnings.py src/swarm_arena/_agent tests/_board/test_learnings.py tests/_agent docs/arena.md CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Enforce learning posting rules on the arena" -m "Evidence must be in the agent's view, near-duplicates need refines or a reason, and verified needs a complete passing evidence candidate (spec 6.3, 6.3.1). Roadmap R3.1 marks context.py and scripted.py +M2."
```

---

### Task 11: Digest v2

**Files:**
- Modify: `src/swarm_arena/_board/digest.py`, `tests/_board/test_digest.py` (new tests; delete `test_digest_sections_and_count_line` and rewrite `test_empty_board_digest` and the line-0 check of `test_digest_cap_holds_for_non_ascii_titles` from M1b Task 12, which pin the v1 text)
- Modify: `src/swarm_arena/_engine/round.py` (the M1b digest call site passes `ranking=group_judges(<the revision in effect>)`), `tests/_engine/test_round.py`
- Modify: `docs/arena.md` (section "The digest"), roadmap R4.13.14 (`build_digest`: `ranking: str | Mapping[str | None, str]`)

**Interfaces:**
- Consumes: `BoardView` (incl. `count_learnings`), `estimate_tokens`, Task 7 `group_judges` and `format_value`, Task 8 `ArenaView.part`, the M1b call site.
- Produces:
```python
UNTRUSTED_OPEN: Final = "<<from other agents, unverified>>"
UNTRUSTED_CLOSE: Final = "<<end of content from other agents>>"
def build_digest(view: BoardView, *, goal: GoalKey, ranking: str | Mapping[str | None, str], cap_tokens: int, k: int = 5) -> str: ...
    # R4.13.14 (roadmap edit): a str ranks every group under that judge (the M1b and M3 tests keep passing a str);
    # a mapping gives each group its own ranking judge, and a group missing from it is not shown
```
Group rows [plan]: the rows of group `g` under judge `J` are `[r for r in view.leaderboard(goal, judge=J) if r.part == g][:k]`. A view whose `part` attribute is set (an `ArenaView` part view, Task 8; any other `BoardView` is a main view) has one block: its part's rows under `ranking[part]` (or the str), headed `Leaderboard <goal-id> (…)`, and no `Part` block.
Layout [plan]. Every line between the markers is board content:
```
<<from other agents, unverified>>
Leaderboard <goal-id> (<judge>, rubric <rubric_hash[:8]>, panel <panel_hash[:8] or ->):
1. <candidate-id> <team>/<agent> <format_value(value)> built on <ids or —>[ · human: accept|reject]
Part <name> (<judge>, rubric <8>, panel <8 or ->):   # main view only: one block per part with rows, top k each, under its own judge
Pinned learnings:
- <id> <title>
Top learnings:
- <id> <title>[ · verified]                     # up to 10, by the view's relevance order, pinned ones excluded
<<end of content from other agents>>
<total - listed> more learnings in namespace <ns> (<goal total> for this goal). Use search_learnings.
```
"The view's relevance order" is the order of `view.learnings(namespace)` with no `goal` argument: Task 8 makes every `ArenaView` learning read pass the view's goal as the relevance goal, so this goal's learnings come first in the whole-namespace list (§6.3). `human:` marks a candidate's latest human verdict in the view (§6.6: verdicts feed later runs). The cap bounds the whole returned string: markers, headers, rows, titles, and the counts line (`estimate_tokens`, UTF-8 bytes, R4.13.14). To fit `cap_tokens`, drop lines in this order: top learnings from the end, then pinned from the end, then part rows, then main rows. The counts line counts every learning not listed. If even the markers plus the counts line do not fit, return only the counts line, cut to at most `cap_tokens` UTF-8 bytes at a character boundary (`text.encode()[:cap].decode("utf-8", "ignore")`).

- [ ] **Step 1: Write the failing tests**

`tests/_board/test_digest.py`: digest v2 changes the layout and the counts line, so the M1b Task 12 tests that pin the v1 text change in this task:
- M1b `test_digest_sections_and_count_line`: delete it. It asserts `lines[0] == "Leaderboard (program, top 5):"`, the `Pinned:`/`Newest:` sections, and `lines[-1] == "412 more learnings, 37 from this round. Use search_learnings."`. `test_board_content_inside_markers`, `test_leaderboard_names_judge_identity`, `test_pinned_first_then_relevance`, and the counts-line check in `test_digest_size_does_not_grow` below replace it.
- M1b `test_empty_board_digest`: rewrite it for v2. The text `"No results yet. 0 learnings."` is gone; an empty board gives the v2 layout with no rows:
  ```python
  def test_empty_board_digest() -> None:
      d = build_digest(_empty_view(), goal=KEY, ranking="program", cap_tokens=1500)
      lines = d.splitlines()
      assert lines[0] == UNTRUSTED_OPEN and UNTRUSTED_CLOSE in lines
      assert re.fullmatch(r"0 more learnings in namespace \S+ \(0 for this goal\)\. Use search_learnings\.", lines[-1])
  ```
- M1b `test_digest_cap_holds_for_non_ascii_titles`: keep the byte-count assertion; replace its line-0 check `d.splitlines()[0].startswith("Leaderboard")` (in v2, line 0 is the open marker):
  ```python
      lines = d.splitlines()
      assert lines[0] == UNTRUSTED_OPEN
      assert any(line.startswith("Leaderboard") for line in lines)
  ```
- M1b `test_digest_respects_token_cap_for_any_board_size` and `test_estimate_tokens` stay as they are.

Additions. The content tests (every test below except the two cap tests) pass a `cap_tokens` that fits their whole fixture, so a test that expects up to 10 top learnings drops no line:
```python
@pytest.mark.parametrize("n", [10, 2000])
def test_digest_size_does_not_grow(n: int) -> None:
    text = build_digest(view_with(n), goal=gk, ranking="j", cap_tokens=1500)
    assert estimate_tokens(text) <= 1500
    assert re.search(r"^\d+ more learnings in namespace toy", text, re.M)
@pytest.mark.parametrize("cap", [1, 20, 1500])
def test_never_exceeds_cap(cap: int) -> None: ...
    # a view whose titles hold CJK, emoji, combining marks, and long hex runs → len(text.encode("utf-8")) <= cap
def test_board_content_inside_markers() -> None: ...       # every leaderboard and title line sits between UNTRUSTED_OPEN and UNTRUSTED_CLOSE
def test_leaderboard_names_judge_identity() -> None:
    assert "Leaderboard g1 (j, rubric r1aaaaaa, panel -):" in build_digest(...)
def test_pinned_first_then_relevance() -> None: ...
def test_titles_only_never_text() -> None: ...             # the text "SECRET-TEXT" never appears
def test_covers_whole_namespace() -> None: ...             # another run's learning title is listed
def test_this_goal_first_in_top_learnings() -> None: ...   # equal learnings of this goal and of another goal (newer) → this goal's line comes first
def test_hidden_not_listed() -> None: ...
def test_human_verdict_marked() -> None: ...               # a review "accept" → " · human: accept" on that row
def test_part_blocks_use_their_own_judges() -> None: ...   # a main view; part-A candidates scored only by judge "a"
    # → ranking={None: "j", "A": "a"}: a line that starts with "Part A (a, rubric " and then A's row; ranking="j": no "Part A" line
def test_part_view_digest_has_no_part_blocks() -> None: ... # a view with part="A" → A's rows under "Leaderboard <goal-id> (a, …):"; no line starts with "Part "
```
`tests/_engine/test_round.py`: `test_digest_ranks_parts_by_their_judges` (a plan with part A ranked by `"a"` and a main-goal slot: after round 1 scores a part-A candidate, the round-2 `ctx.digest` of the main slot has a line that starts with `Part A (a, `).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_digest.py tests/_engine/test_round.py -k "digest" -v`
Expected: FAIL (the v1 digest has no markers, identity line, or part judges).

- [ ] **Step 3: Implement digest v2 in `src/swarm_arena/_board/digest.py`, and pass `group_judges(<the revision in effect>)` at the call site in `src/swarm_arena/_engine/round.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board tests/_agent tests/_engine -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Docs and commit**

`docs/arena.md`, "The digest": its contents, the cap (counted in UTF-8 bytes, an upper bound on tokens; more titles need a higher `digest_tokens`), the markers, and that agents search for more. Edit roadmap R4.13.14: `build_digest`'s `ranking` is `str | Mapping[str | None, str]`. Then:
```bash
git add src/swarm_arena/_board/digest.py src/swarm_arena/_engine/round.py tests/_board/test_digest.py tests/_engine/test_round.py docs/arena.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Build digest v2 over the whole namespace" -m "The digest lists the leaderboard with its judge identity and learning titles by relevance, inside untrusted markers, under a hard token cap (spec 6.3). Each part block ranks under its part's own ranking judge (roadmap R2 M2 item 8), so build_digest takes a judge per group; roadmap R4.13.14 records the wider ranking type."
```

---

### Task 12: Curator ABC and round-end hook

**Files:**
- Create: `src/swarm_arena/_board/curator.py`, `tests/_board/test_curator.py`, `tests/_fakes/curators.py`
- Modify: `src/swarm_arena/_journal/replay.py` (folds `curated` as a no-op), `tests/_journal/test_replay.py`. `EventType.CURATED` already exists (M1b Task 2 defines every R4.14 value), and the payload is a plain dict, so `_journal/event.py` does not change. This task adds only the `curated` payload and its replay fold.
- Modify: `src/swarm_arena/_engine/goal.py`, `tests/_engine/test_goal.py`
- Modify: `src/swarm_arena/board.py` (export `Curator`), `tests/test_public_api.py`, `docs/arena.md`, `CHANGELOG.md`, roadmap R3.2 (row: `tests/_fakes/curators.py`, M2), roadmap R3.1 (`_journal/replay.py`, `_engine/goal.py` gain `+M2`)

**Interfaces:**
- Consumes: `IdGen`, `BoardView`, `Learning`, `llm_scope`, `LLMScope`, `Tranche` (R4.13.6, R4.13.7), `JournalWriter`, `Board.put_learning`.
- Produces:
```python
class Curator(ABC):                                          # R4.6
    Config: ClassVar[type[BaseModel]]                       # default: an empty model
    def models(self) -> tuple[str, ...]: ...                 # default ()
    @abstractmethod
    async def curate(self, namespace: str, new: Sequence[Learning], view: BoardView) -> list[Learning]: ...
async def run_curator(curator: Curator, *, namespace: str, run_id: str, goal_id: str, round: int,
                      new: Sequence[Learning], view: BoardView, ids: IdGen) -> tuple[list[Learning], list[str]]: ...
    # (summaries to journal, errors). For each returned learning: supersedes must be non-empty and each id in new or the view,
    # else an error and the learning is dropped. The core sets id = ids.next("L"), author = "curator", namespace,
    # run_id, goal_id, round; evidence = the superseded learnings' evidence in order, without duplicates;
    # verified = any superseded verified; pinned = hidden = False. A summary whose evidence is empty (every superseded
    # learning is a human learning with no evidence) is dropped with the error "<title>: summary has no evidence", because
    # Learning(author="curator", evidence=[]) fails validation. If curate raises → ([], [repr(exc)]).
class MergeAll(Curator): ...                                 # tests/_fakes/curators.py: merges all `new` into one "Summary of N learnings"
```
Engine [plan]: in `_engine/goal.py`, after the policy and before the replanner. Skip when `config.curator is None` or the goal has no learnings this round. Build `config.curator.build(Curator)`. Run inside `llm_scope` with role `"curator"` and a `Tranche` for holder `"<goal>/:curator"` (paid from the reserve, §6.3). Journal each summary as a `learning` event (payload `{"learning": summary.model_dump(mode="json")}`, the M1b key), then one `curated` event with the plain dict payload `{"summaries": [ids], "supersedes": {id: [ids]}, "errors": [...]}`, then `board.put_learning`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_summary_supersedes_and_unions_evidence() -> None:
    # new: L1 (evidence c-1), L2 (evidence c-2, c-1); MergeAll
    summaries, errors = await run_curator(MergeAll(), namespace="toy", run_id="r1", goal_id="g1", round=1, new=[l1, l2], view=view, ids=IdGen("r1", 1))
    assert errors == []
    assert summaries[0].supersedes == ["L1", "L2"]
    assert summaries[0].evidence == ["c-1", "c-2"]
    assert summaries[0].author == "curator"
    assert summaries[0].id.startswith("L-")
async def test_unknown_supersedes_dropped_with_error() -> None: ...      # also: a summary of only evidence-free human learnings → dropped, error "summary has no evidence"
async def test_curator_exception_is_skipped() -> None: ...       # ([], ["RuntimeError('boom')"])
def test_curator_default_models_empty() -> None: ...
```
`tests/_engine/test_goal.py`: `test_engine_runs_curator_at_round_end`. A plan with `post_learnings=True` and `RunConfig(curator=Ref(path="_fakes.curators:MergeAll"))`. The journal has one `curated` event per round. In the next round's view, the index of the summary in `view.learnings(namespace)` is lower than the index of each learning it supersedes (relevance key: superseded learnings rank lower). Do not assert this through `search_learnings`: bm25 comes before the relevance key there, and it can put the "Summary of N learnings" title lower.
`tests/_journal/test_replay.py`: `test_replay_accepts_curated`.
`tests/test_public_api.py`: `Curator` is in `swarm_arena.board.__all__`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_curator.py tests/_engine/test_goal.py tests/_journal/test_replay.py tests/test_public_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._board.curator'`.

- [ ] **Step 3: Implement the ABC, the hook, the `curated` payload and its no-op replay fold (the `EventType` member exists from M1b), the engine call, and the facade export**

`Curator` and `Curator.curate` get numpydoc docstrings with `Examples` (a subclass sketch with `# doctest: +SKIP`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board tests/_engine tests/_journal tests/test_public_api.py tests/test_docstrings.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Docs and commit**

`docs/arena.md`, "Curation": what a curator does, `RunConfig.curator`, the reserve, and undo with `learnings hide`. Add a CHANGELOG line. Edit roadmap R3.2, and R3.1 (the `_journal/replay.py` and `_engine/goal.py` rows get `+M2`).
```bash
git add src/swarm_arena/_board/curator.py src/swarm_arena/_journal src/swarm_arena/_engine/goal.py src/swarm_arena/board.py tests docs/arena.md CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Add the Curator ABC and its round-end hook" -m "At round end an optional curator merges near-duplicate learnings into summary learnings with supersedes and the union of their evidence (spec 6.3). LLMCurator comes in M3. Roadmap R3.1 marks replay.py and goal.py +M2."
```

---

### Task 13: Round pages

**Files:**
- Create: `src/swarm_arena/_board/pages.py`, `tests/_board/test_pages.py`
- Modify: `src/swarm_arena/_engine/goal.py` (write the page after the round-close commit), `tests/_engine/test_goal.py`
- Modify: `src/swarm_arena/_journal/replay.py` (`GoalState.closed_seq`), `tests/_journal/test_replay.py`

**Interfaces:**
- Consumes: `RunState` (R4.13.4) and its `positions`, `GoalState` (M1b), `ViewWriter`, `embed_source_head`, `Head`, Task 7 `group_rows`/`flatten`/`group_judges`/`format_value`, `Plan.rounds_total`, `Plan.ranking_name`.
- Produces:
```python
def page_key(goal_id: str, round: int) -> str: ...      # "goals/<goal-id>/rounds/round-NN.md" (two digits)
def round_page(state: RunState, goal_id: str, round: int, *, namespace_learnings: int, source: Head) -> bytes: ...
def state_positions(state: RunState, items: Iterable[T], *, until_seq: int | None = None) -> list[tuple[T, Position]]: ...
    # T is Candidate or Score; Position = ("", state.run_id, state.positions[item.id][1]); items with seq > until_seq left out
# _journal/replay.py (+M2): GoalState gains
    closed_seq: dict[int, int] = {}                     # round → seq of that round's round_closed event
```
Positions from `RunState` [plan]. `RunState.positions` holds `(epoch, seq)` and no event ts, so a page or review Position is `("", state.run_id, seq)`. All items are from one run, so the order is seq order. The round-N cut-off is `state.goals[goal_id].closed_seq[N]`, which `RunState.from_events` sets when it folds the goal's `round_closed` event for round N. A page holds the candidates and scores with `seq <= closed_seq[N]`.
Content [plan]. It is a pure function of the events up to that round's `round_closed` event; only the embedded source head can differ between two writes:
```markdown
# <namespace> / <goal-id> · run <run-id> · round <n> of <rounds_total>
Leaderboard (<ranking judge>, lower is better|higher is better)
| # | Candidate | Team / agent | Score | Built on |
|---|-----------|--------------|-------|----------|
| 1 | c-… | <team> / <agent> | <format_value(value)> | <parent ids joined by ", " or —> |

New learnings (<count>)
- <id> **<title>** <author> · verified|unverified|from a human
  <text, each line indented two spaces> Evidence: <cid> (<format_value(value)> or <status>) · …

Pruned: team `<t>` (<reason>); agent `<t>/<a>` (<reason>).        # or "Pruned: nothing."
Full board: `swarm-arena board serve <run-id>` · <namespace_learnings> learnings in namespace `<namespace>`.
```
- Leaderboard: `flatten(group_rows(state_positions(state, <the goal's candidates>, until_seq=closed_seq[N]), state_positions(state, <their scores>, until_seq=closed_seq[N]), judge=group_judges(plan), run_id=state.run_id, identity_run=state.run_id, min_improvement=plan.converge.min_improvement))[:10]`, where `plan` is the revision in effect at round N. Each part's rows rank under that part's own ranking judge; the header names `plan.ranking_name()`.
- "lower is better" when the top row has `value is not None and key[0] == -value`; else "higher is better".
- Table cells escape `|` as `\|` and turn newlines into spaces.
- Engine: right after the `round_closed` commit, `ViewWriter.write(page_key(goal, n), round_page(...), source=writer.head)`. `namespace_learnings = round view.count_learnings(namespace) + this goal's learnings of round n`.

- [ ] **Step 1: Write the failing tests**

```python
def test_page_layout(state: RunState) -> None:
    lines = round_page(state, "ns-viscosity", 2, namespace_learnings=412, source=head).decode().splitlines()
    assert lines[1] == f"# physika / ns-viscosity · run {state.run_id} · round 2 of 3"     # line 0 is the source-head comment
    assert "Leaderboard (rel-error, lower is better)" in lines
    assert "| 1 | c-0193 | fd-solver / tuner-1 | 0.0041 | c-0112 |" in lines
    assert "| 2 | c-0171 | spectral / author-b | 0.009 | — |" in lines
    assert lines[-1] == f"Full board: `swarm-arena board serve {state.run_id}` · 412 learnings in namespace `physika`."
def test_top_10_only(state: RunState) -> None: ...                 # 15 candidates → 10 table rows
def test_ignores_later_rounds(state: RunState) -> None: ...        # round-3 items (seq > closed_seq[2]) are absent from the round-2 page
def test_new_learnings_and_pruned(state: RunState) -> None: ...    # "New learnings (1)", "- L-0088 **…** fd-solver/tuner-2 · verified", "Pruned: team `surrogate` (…)."
def test_table_cells_escape_pipes(state: RunState) -> None: ...    # team "a|b" → "a\|b"
def test_part_rows_use_part_judge(...) -> None: ...                # part A ranked by "a", scored only by "a" → A's row shows its "a" value
def test_source_head_embedded(state: RunState) -> None:
    assert read_source_head("md", round_page(state, "ns-viscosity", 2, namespace_learnings=0, source=head)) == head
```
`tests/_journal/test_replay.py`: `test_closed_seq_per_round` (round_closed events at seq 7 and 12 → `closed_seq == {1: 7, 2: 12}`).
`tests/_engine/test_goal.py`: `test_engine_writes_one_page_per_round`. After a 2-round run, `goals/<goal>/rounds/round-01.md` and `round-02.md` exist in the run store, and `round-01.md` does not change after round 2.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_pages.py tests/_engine/test_goal.py tests/_journal/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._board.pages'`.

- [ ] **Step 3: Implement `pages.py` and the engine call**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board tests/_engine tests/_journal -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Docs and commit**

`docs/arena.md`, "Round pages": an example page and the rule that it never grows. Add a CHANGELOG line.
```bash
git add src/swarm_arena/_board/pages.py src/swarm_arena/_engine/goal.py src/swarm_arena/_journal/replay.py tests/_board/test_pages.py tests/_engine/test_goal.py tests/_journal/test_replay.py docs/arena.md CHANGELOG.md
git commit -m "Write one human-readable page per round" -m "round-NN.md is a view of the committed journal written after the round-close commit, with the top 10, new learnings, and prune reasons (spec 6.3.1)."
```

---

### Task 14: Human review export and import

**Files:**
- Create: `src/swarm_arena/_scoring/human_review.py`, `tests/_scoring/test_human_review.py`
- Modify: `src/swarm_arena/_journal/replay.py` (no-op fold of `review_imported`), `tests/_journal/test_replay.py`, `tests/_board/test_cache.py`. `EventType.REVIEW_IMPORTED` already exists (M1b Task 2 defines every R4.14 value), so `_journal/event.py` does not change. This task adds only the payload, the plain dict `{"reviewer": str, "scores": [Score.model_dump(mode="json"), ...], "objects": list[str]}`, and its replay fold.
- Modify: `src/swarm_arena/_engine/run.py` (end-of-run export), `tests/_engine/test_run.py`
- Modify: `src/swarm_arena/scoring.py` (export `HumanReview`), `tests/test_public_api.py`
- Modify: `docs/scoring-and-review.md` (section "Human review"), `CHANGELOG.md`, roadmap R5.4 item 11 and roadmap R2 "M2 Arena & board" deliverable 9 (both get the export path `<run_root>/<run-id>/review/e<epoch>-<seq>/`; deliverable 9 now says `<run_root>/<run-id>/review/` [plan])

**Interfaces:**
- Consumes: `RunConfig`, `RunState`, `read_owner`, `read_chain`, `Lease.take_over`, `Lease.release`, `JournalWriter`, `IdGen`, `LeaseHeld`, `Fenced` (the takeover retry of §7.1 step 1, the same rule as M1b `resume_run`, Task 27), Task 3 `human_object_name`/`human_object_key`/`write_human_objects`, Task 7 `group_rows`/`flatten`/`group_judges`/`format_value`, Task 13 `state_positions`, `run_store`, Task 2 `check_storage`, `host_id` (R4.13.8).
- Produces:
```python
class HumanReview:                                         # R4.3
    def __init__(self, reviewer: str) -> None: ...         # slug [a-z0-9][a-z0-9._-]* else ValueError
    judge_name: str                                        # "human:<reviewer>"
    async def export(self, run_id: str, *, config: RunConfig, out_dir: Path, top: int = 20, goal: str | None = None) -> Path: ...
    async def import_csv(self, run_id: str, csv_path: Path, *, config: RunConfig) -> list[Score]: ...
CSV_HEADER: Final = ("id", "goal", "team", "agent", "scores", "verdict", "score", "notes")
def render_review(state: RunState, *, top: int = 20, goal: str | None = None) -> dict[str, bytes]: ...
    # "review.csv", "index.html", "candidates/<id>.json"; deterministic bytes
def parse_review_csv(data: bytes, state: RunState, *, judge: str) -> list[Score]: ...
async def write_review_export(state: RunState, store: Store, head: Head) -> None: ...   # _engine/run.py
```
Rules [plan]:
- **Rows.** Per goal: `flatten(group_rows(state_positions(state, <the goal's candidates>), state_positions(state, <their scores>), judge=group_judges(<the latest revision>), run_id=run_id, identity_run=run_id))[:top]` (each part under its own ranking judge). Positions are `("", run_id, seq)` (Task 13), with no cut-off.
- **`review.csv`.** UTF-8 with header `CSV_HEADER`. `scores` is `"<judge>=<format_value(value)>; …"` (read-only). `verdict`, `score`, and `notes` start empty.
- **`candidates/<id>.json`.** `canonical_json({"candidate": …, "scores": [...]})`.
- **`index.html`.** One static table, with every value passed through `html.escape`, no script, and links to `candidates/<id>.json`.
- **Parse.** Decode with `utf-8-sig`, read with `csv.DictReader` and `newline=""`. The verdict is `accept`, `reject`, or empty, in any letter case. `score` is a finite float or empty. Skip a row only when both verdict and score are empty. Key = `(1.0 | 0.0 | -1.0 for accept | empty | reject, score or 0.0)`; `value = score`; `verdict`, `notes`; `judge="human:<reviewer>"`; `rubric_hash = panel_hash = ""`. Collect every error as `"row <n>: <reason>"` (n counts the header as row 1), then raise one `ValueError`. Errors are: an unknown candidate id, a duplicate id, a bad verdict, a bad score, or a missing column.
- **Import order.** The takeover follows §7.1 step 1, the same rule as M1b `resume_run` (Task 27). The scores and the writer always use the head that the takeover fixed, and no event is written before the CSV parses against that head:
  1. Pre-check. Authority read (`read_owner`; `None` → `FileNotFoundError(f"run {run_id} has no control/owner.json")`; `doc.head is None` → `ValueError(f"run {run_id} has no committed events")`), `read_chain(store, doc.head)`, `RunState.from_events`, then `parse_review_csv` against that state. On any error, raise; nothing is written (`owner.json` is not touched).
  1b. Storage check. `await anyio.to_thread.run_sync(check_storage, config)` after the step-1 pre-check (so a bad CSV writes no probe object either) and before the takeover (Task 2 Step 5 item 3). `ConditionalWritesUnsupported` and `ConfigError` propagate. Nothing in the run or the arena is written. `human_review.py` imports `check_storage` at module level, so tests patch `human_review.check_storage`.
  2. Take over, in a loop: `doc0 = read_owner(store)` (the authority read), then `lease = Lease.take_over(store, owner_id=<new owner id>, host=host_id(), ttl_s=config.lease_ttl_s, force=False, expect_epoch=doc0.epoch)`. As in M1b Task 27 and M4 Task 16, `expect_epoch` sends a takeover that lands between the authority read and `take_over` into the `Fenced` branch. On `Fenced`, `read_owner` again: the same epoch as `doc0` (the old owner advanced its head or renewed) → retry from the authority read; another epoch → raise `Fenced(f"run {run_id}: another owner took the run (epoch {epoch})")`. `LeaseHeld` → `RuntimeError("run <id> has a live owner; run review import after the run ends")`. In both error cases nothing is written.
  3. Read the final state. `read_chain(store, lease.doc.head)` (the run is fenced now, so this chain is final), `state = RunState.from_events(...)`, and `parse_review_csv` again against this state. Every later step uses this `state` and `lease.doc.head`, never the step-1 head. If this parse fails, call `lease.release()` and raise; no event and no object is written.
  4. Name the objects first. For each namespace of the run's goals: `basename = human_object_name(now)` and `key = human_object_key(ns, "reviews", basename, reviewer=reviewer)`. Set each Score's `batch_id` to the key of its goal's namespace.
  5. Open `JournalWriter(next_seq=lease.doc.head.seq + 1)`. Score ids come from `IdGen(run_id, epoch=lease.doc.epoch, counters=state.id_counters)`. Append `takeover` with the payload `{"force": False}` (the M1b `resume_run` shape that replay folds; M4 reconcile copies it), then `review_imported` (the plain dict payload above) with `objects` = the keys of step 4 and the scores of step 4, and commit.
  6. Write each namespace's object with `write_human_objects(arena, ns, "reviews", <its review objects>, now=now, reviewer=reviewer, name=basename)`. The returned key equals the step-4 key, so the journal and `Score.batch_id` name exactly the written objects.
  7. Leave the writer context (it releases the lease).
- **End of run.** In `_engine/run.py`, after the last `writer.commit()` and before the writer context exits: `write_review_export(state, run store, head)` creates `<run_root>/<run-id>/review/e<epoch>-<seq>/<file>` (the key `review/e<epoch>-<seq>/<file>` in the run store; roadmap R2 M2 deliverable 9 and R5.4 item 11) for each file of `render_review(state, top=20)`. `AlreadyExists` with the same sha256 counts as success. Other bytes raise `StorageError`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_export_writes_csv_json_and_html(finished_run: FinishedRun, tmp_path: Path) -> None:
    out = await HumanReview("alice").export(finished_run.run_id, config=finished_run.config, out_dir=tmp_path / "rev")
    rows = list(csv.reader((out / "review.csv").read_text().splitlines()))
    assert tuple(rows[0]) == CSV_HEADER
    assert 1 < len(rows) <= 21
    assert all((out / "candidates" / f"{r[0]}.json").exists() for r in rows[1:])
    assert rows[1][0] in (out / "index.html").read_text()
async def test_export_top_and_goal_filter(...) -> None: ...               # top=1 → one data row per goal; goal="nope" → ValueError, match="nope"
def test_export_escapes_html(state: RunState) -> None: ...                # judge notes "<script>alert(1)</script>" → "&lt;script&gt;" and never "<script>alert"
def test_render_is_deterministic(state: RunState) -> None: ...
async def test_import_roundtrip(finished_run: FinishedRun, tmp_path: Path) -> None:
    # set row 1 verdict "accept" score "7", row 2 verdict "reject"
    scores = await HumanReview("alice").import_csv(run_id, csv_path, config=config)
    assert [s.key for s in scores] == [(1.0, 7.0), (-1.0, 0.0)]
    assert {s.judge for s in scores} == {"human:alice"}
    # the journal has a review_imported event; arena has exactly one <ns>/reviews/alice/*.jsonl object, its key equals
    # the event's objects == [key] and every score's batch_id == key; the import's takeover event payload == {"force": False}
async def test_import_run_without_commit_raises(...) -> None: ...        # owner.json with head None → ValueError, match="no committed events";
    # the owner.json bytes are unchanged
async def test_import_accepts_spreadsheet_csv(...) -> None:               # Review Focus 2: BOM + CRLF + "Accept" → imported
async def test_import_rejects_bad_rows_and_writes_nothing(...) -> None:   # Review Focus 2
    # rows 2..5: unknown id "c-nope", verdict "maybe", score "abc", score "nan"
    with pytest.raises(ValueError, match="row 2:") as err:
        await HumanReview("alice").import_csv(...)
    assert all(f"row {n}:" in str(err.value) for n in (2, 3, 4, 5))
    # owner.json bytes unchanged; no reviews/ object; a check_storage spy patched in as below is never called (the bad rows fail first)
async def test_import_checks_storage_before_takeover(finished_run: FinishedRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a valid CSV; human_review.check_storage patched with a spy that records its call and raises ConditionalWritesUnsupported
    with pytest.raises(ConditionalWritesUnsupported):
        await HumanReview("alice").import_csv(finished_run.run_id, csv_path, config=finished_run.config)
    assert len(calls) == 1
    # owner.json bytes unchanged; no review_imported event (the head is unchanged); no <ns>/reviews/ object
async def test_import_fails_while_owner_live(...) -> None: ...            # a live Lease held → RuntimeError "live owner"; nothing written
async def test_import_retries_takeover_and_uses_final_head(...) -> None: ...   # the first Lease.take_over call lets the old owner
    # commit one more segment and release, then raises Fenced; the import retries, its takeover event has seq == the new
    # head seq + 1, and the journal seqs are 1..n with no repeat
async def test_import_loses_to_another_owner(finished_run: FinishedRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # the M1b Task 27 pattern: the patched Lease.take_over, on its first call, runs a real takeover as owner "other"
    # (force=True), then the real call with the import's own kwargs, which it records in `seen`
    with pytest.raises(Fenced, match="another owner took the run"):
        await HumanReview("alice").import_csv(finished_run.run_id, csv_path, config=finished_run.config)
    assert seen[0]["expect_epoch"] == 1
    doc = read_owner(store)[0]
    assert (doc.owner_id, doc.epoch) == ("other", 2)                    # not 3: the import wrote nothing
    # no review_imported event, no reviews/ object
async def test_import_ids_use_new_epoch(...) -> None:
    assert re.fullmatch(rf"s-{run_tag(run_id)}-e{old_epoch + 1}-\d{{4}}", scores[0].id)
def test_reviewer_slug() -> None: ...                                     # HumanReview("Alice Smith") → ValueError, match="reviewer"; HumanReview("alice").judge_name == "human:alice"
```
`tests/_board/test_cache.py`: `test_review_imported_events_not_indexed`.
`tests/_journal/test_replay.py`: `test_replay_accepts_review_imported`.
`tests/_engine/test_run.py`: `test_run_end_writes_review_export`, with `review/e<epoch>-<seq>/review.csv` and `index.html` in the run store.
`tests/test_public_api.py`: `HumanReview` is in `swarm_arena.scoring.__all__`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_scoring/test_human_review.py tests/_board/test_cache.py tests/_journal/test_replay.py tests/_engine/test_run.py tests/test_public_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._scoring.human_review'`.

- [ ] **Step 3: Implement `human_review.py`, the `review_imported` payload and its no-op replay fold (the `EventType` member exists from M1b), the engine hook, and the facade export**

Docstrings with `Examples` for `HumanReview`, `export`, and `import_csv` (`# doctest: +SKIP`; both need a run).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_scoring tests/_board tests/_journal tests/_engine tests/test_public_api.py tests/test_docstrings.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Docs and commit**

`docs/scoring-and-review.md`, "Human review": export, the CSV columns and verdict rules, import, the lease rule (run it after the run ends; on `s3://` roots it probes conditional writes first, Task 2), and re-rank with `leaderboard --by human:<name>`. Add a CHANGELOG line. Edit roadmap R5.4 item 11 and roadmap R2 "M2 Arena & board" deliverable 9: both give the export path `<run_root>/<run-id>/review/e<epoch>-<seq>/`.
```bash
git add src/swarm_arena/_scoring/human_review.py src/swarm_arena/_journal src/swarm_arena/_engine/run.py src/swarm_arena/scoring.py tests docs/scoring-and-review.md CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Add human review export and import" -m "Export writes review.csv, one JSON per candidate, and a static index.html; import takes the run lease, stores human scores in the run and as one reviews/ object (spec 6.6). The takeover retries from the authority read when the old owner writes first (spec 7.1), and the scores and the journal use the head that the takeover fixed."
```

---

### Task 15: Reindex

**Files:**
- Create: `src/swarm_arena/_board/reindex.py`, `tests/_board/test_reindex.py`, `tests/e2e/test_reindex.py`

**Interfaces:**
- Consumes: `ArenaBoard`, `ArenaCache.membership`, `RunConfig`, `Event`, `Frontier`, the plain dict payload of the M1b `frontier` event (keys `round` and `frontier`), Task 3 `write_human_objects`, Task 14 `HumanReview`, fixture `finished_run`.
- Produces:
```python
def reindex(config: RunConfig, namespaces: Sequence[str] = (), *, cache_dir: Path | None = None) -> Frontier: ...
    # namespaces () → every "<ns>.sqlite" in the cache dir. Under an exclusive flock on "<ns>.sqlite.lock" (below), deletes
    # each file with its -wal and -shm files, then ArenaBoard(arena_root=config.arena_root, cache_dir=...).open("", "", namespaces)
    # and returns frontier().
def recorded_frontiers(events: Iterable[Event]) -> list[tuple[str, int, Frontier]]: ...   # from `frontier` events, in journal order:
    # (event.goal_id, payload["round"], Frontier.model_validate(payload["frontier"]))
def snapshot_membership(board: ArenaBoard, goal: GoalKey, frontier: Frontier) -> bytes: ...   # the snapshot's ArenaCache.membership
```
`reindex` first takes `fcntl.flock(LOCK_EX | LOCK_NB)` on each `<ns>.sqlite.lock` (Task 4). If it cannot, it raises `RuntimeError(f"{path} is in use by another swarm-arena process; stop it first")` before it touches any file. With the exclusive lock held, it deletes the db and its `-wal` and `-shm` files, releases the lock, and then builds the new `ArenaBoard`. The `.lock` file is never deleted. It prints nothing itself; the CLI prints the rebuilt caches' `warnings` (Task 17).

- [ ] **Step 1: Write the failing tests**

`tests/_board/test_reindex.py`:
```python
def test_reindex_without_namespaces_uses_cache_files(tmp_path: Path) -> None: ...   # toy.sqlite and phys.sqlite present → both rebuilt
def test_reindex_removes_wal_and_shm(tmp_path: Path) -> None: ...
def test_reindex_refuses_while_cache_open(tmp_path: Path) -> None:
    held = ArenaCache(path)          # same process: flock locks belong to open file descriptions, so this conflicts
    with pytest.raises(RuntimeError, match="in use"):
        reindex(config, ["toy"], cache_dir=path.parent)
    # the db and -wal bytes are unchanged
    held.close()
    reindex(config, ["toy"], cache_dir=path.parent)                 # now succeeds
def test_recorded_frontiers_in_order(tmp_path: Path) -> None: ...
```
`tests/e2e/test_reindex.py`:
```python
async def test_reindex_rebuilds_identical_membership(finished_run: FinishedRun, tmp_path: Path) -> None:
    # then: a pin, a hide, and an add curation object; a review import; run2 = await run_again(finished_run)
    frontiers = recorded_frontiers(<run 1 events>) + recorded_frontiers(<run 2 events>)
    before = [snapshot_membership(board, goal_key, f) for _, _, f in frontiers]
    reindex(config, [ns], cache_dir=tmp_path / "c2")
    assert [snapshot_membership(ArenaBoard(arena_root=..., cache_dir=str(tmp_path / "c2")), goal_key, f) for _, _, f in frontiers] == before
    os.remove(Path(config.arena_root) / ns / "runs" / f"{finished_run.run_id}.json")
    fresh = ArenaBoard(arena_root=config.arena_root, cache_dir=str(tmp_path / "c3"))
    assert [snapshot_membership(fresh, goal_key, f) for _, _, f in frontiers] == before    # rebuilt through recorded run_root
    fresh.open("", "", [ns])
    assert finished_run.run_id not in [m.run_id for m in fresh.frontier().marks]       # removed from later snapshots only
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_reindex.py tests/e2e/test_reindex.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._board.reindex'`.

- [ ] **Step 3: Implement `reindex.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board/test_reindex.py tests/e2e/test_reindex.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_board/reindex.py tests/_board/test_reindex.py tests/e2e/test_reindex.py
git commit -m "Add arena cache reindex" -m "reindex rebuilds the cache from run journals and human objects; recorded frontiers rebuild byte-identical snapshot membership, also after a pointer is deleted (spec 7.2). It refuses while another process holds the cache open, so it never unlinks a live WAL database."
```

---

### Task 16: Trace

**Files:**
- Create: `src/swarm_arena/_board/trace.py`, `tests/_board/test_trace.py`

**Interfaces:**
- Consumes: `ArenaBoard` (latest frontier, cache), `ArenaCache.candidates`, `scores`, `plans`, `find`; Task 7 `format_value`.
- Produces:
```python
def trace(board: ArenaBoard, candidate_id: str) -> list[str]: ...   # KeyError for an unknown id
```
Lines in this order [plan] (§7.3 item 4, non-LLM steps; M3 adds LLM calls):
```
plan     <run-id> <goal-id> r<revision> origin=<origin> team=<team> agent=<agent> rationale=<agent rationale>
run      <candidate-id> status=<status> partial=<bool> round=<n> fidelity=<f> parents=<ids or -> usage=in:<n> out:<n> usd:<x or ->
check    <name> passed|failed <reasons joined by "; ">
score    <judge> <format_value(value)> key=<key> rubric=<rubric_hash[:8]> panel=<panel_hash[:8] or ->     # journal and review scores
learning <id> <title>                                                                         # learnings that cite it as evidence
```
The plan revision in effect is the goal's latest `plan_revision` with a seq lower than the candidate's.

- [ ] **Step 1: Write the failing tests**

```python
def test_trace_chain_order(board_with_run: ArenaBoard) -> None:
    kinds = [line.split()[0] for line in trace(board_with_run, "c-1")]
    assert kinds == sorted(kinds, key=["plan", "run", "check", "score", "learning"].index)
    assert kinds[0] == "plan"
def test_trace_uses_revision_in_effect(...) -> None: ...      # r1 committed before c-2 → "r1"; c-1 made before r1 → "r0"
def test_trace_includes_human_scores(...) -> None: ...        # "score    human:alice …"
def test_trace_verdict_only_score(...) -> None: ...           # a review score with key (1.0, 0.0) and value None → a line that
    # starts with "score    human:alice - key=(1.0, 0.0)"
def test_trace_unknown_candidate(...) -> None: ...            # KeyError
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_trace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._board.trace'`.

- [ ] **Step 3: Implement `trace.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board/test_trace.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_board/trace.py tests/_board/test_trace.py
git commit -m "Add the trace chain for a candidate" -m "trace prints the plan revision, tool run, checks, scores, and citing learnings behind a result (spec 7.3 item 4, non-LLM steps)."
```

---

### Task 17: Arena CLI commands

**Files:**
- Create: `src/swarm_arena/_cli/arena_cmds.py`, `tests/_cli/test_arena_cmds.py`, `tests/e2e/test_review_roundtrip.py`
- Modify: `src/swarm_arena/_cli/main.py` (call `arena_cmds.register`), `README.md`, `docs/arena.md` (section "Command line"), `CHANGELOG.md`, roadmap R3.1 (`_cli/main.py` gains `+M2`)

**Interfaces:**
- Consumes: Tasks 3, 6, 7 (incl. `group_judges`, `format_value`), 8, 10 (`is_verified`), 14, 15, 16; `RunConfig.load`; `main(argv)` (R4.13.18).
- Produces: `def register(sub: argparse._SubParsersAction) -> None`. Commands [plan]:

| Command | Behavior |
|---|---|
| `review export <run> [--goal ID] [--top 20] [--out DIR]` | `HumanReview(...).export`; default out `./review-<run-id>/`; prints the path |
| `review import <run> <csv> --reviewer NAME` | `import_csv`; prints `imported <n> scores as human:<name>` |
| `leaderboard <run-or-namespace> [--goal ID] [--by JUDGE]` | per goal and group: `== <ns>/<goal>[/<part>] (<judge>, rubric <8>, panel <8 or ->) ==`, then lines `<rank>\t<candidate>\t<run>\t<team>/<agent>\t<format_value(value)>\t<built_on or ->`. A run: that run's rows with its `min_improvement` (`group_rows(..., run_id=<run>, identity_run=<run>)`). A namespace: all runs (`identity_run=None`: the newest identity, §4.9). Without `--by`: `judge=group_judges(<the newest plan revision of that goal>)`, so each group ranks under its own ranking judge; `--by J` applies J to every group. No scores → `no scores for <judge>` |
| `board search "<query>" (--namespace NS \| --run ID) [--goal] [--tag ...] [--verified] [--human] [--pinned] [--hidden] [--limit 10]` | lines `<id>\t<title>\t<author>\t<markers>` (verified, human, pinned, hidden, superseded) |
| `board show <id> (--namespace NS \| --run ID)` | `c-…`: request, status, checks, scores, parents, changed file names; `L-…`: full text and evidence with each candidate's score or error |
| `learnings pin\|hide <id> --namespace NS [--author NAME]` | one curation object; an unknown id in that namespace → exit 1, writes nothing |
| `learnings add --namespace NS --title T --text X [--goal ID] [--tag ...] [--evidence ID ...] [--author NAME]` | one curation object `add` holding `Learning(id=human_learning_id(), namespace=NS, run_id="", goal_id=<--goal or "">, round=0, author="human:<name>", title, text, evidence, tags, verified=is_verified(<the evidence candidates in the latest frontier>))` (id `L-h-<16 hex>`); prints the id |
| `trace <candidate-id> (--namespace NS \| --run ID)` | `trace(...)` lines |
| `reindex [--namespace NS ...]` | `reindex(...)`; prints `<ns>: <n> runs`, then each rebuilt cache's `warnings` (Task 4) on stderr |

Every command takes `--config PATH` (default `run.toml`). The board is `config.board` when it is set (it must build an `ArenaBoard`, else exit 1 with `arena commands need ArenaBoard`). Otherwise it is `ArenaBoard(arena_root=config.arena_root)`. A run id matches `^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z_[0-9a-f]{6}$`. Its namespaces come from its committed journal (authority read, `read_chain`, `RunState.plans`). `--author` defaults to `getpass.getuser()`. The `board` commands print the caches' id-clash `warnings` (Task 4) on stderr. Exit codes: 0 ok, 1 error (message on stderr), 2 usage. Command handlers import nothing optional at module level (R4.13.18).

- [ ] **Step 1: Write the failing tests**

`tests/_cli/test_arena_cmds.py` (call `main([...])` and read `capsys`). These tests are sync, because `main()` calls `anyio.run`. They can use the async `finished_run` fixture because it requests `anyio_backend` (Task 9).
```python
def test_leaderboard_namespace(finished_run: FinishedRun, capsys: pytest.CaptureFixture[str]) -> None: ...   # exit 0; "== toy/" header; rank 1 line
def test_leaderboard_run_by_judge(...) -> None: ...
def test_leaderboard_verdict_only_import(...) -> None: ...       # import a CSV with row 1 "accept" and row 2 "reject", both scores
    # empty → `leaderboard <run> --by human:alice` exits 0, row 1's candidate has rank 1, and its value column is "-"
def test_board_search_and_show(...) -> None: ...
def test_learnings_pin_hide_add_write_one_object_each(...) -> None:
    # three commands → three files in <arena>/<ns>/curation/; the add output matches r"L-h-[0-9a-f]{16}"
    # the add object's learning has run_id == "", round == 0, and goal_id == "" (no --goal)
def test_learnings_pin_unknown_id_writes_nothing(...) -> None: ...      # exit 1; curation dir empty
def test_trace_prints_chain(...) -> None: ...
def test_reindex_cmd(...) -> None: ...
def test_arena_cmds_need_arena_board(...) -> None: ...                  # config.board = another Board → exit 1, message
```
`tests/e2e/test_review_roundtrip.py`:
```python
async def test_review_roundtrip(finished_run: FinishedRun, tmp_path: Path) -> None:
    sa = ["uv", "run", "swarm-arena"]
    cfg = ["--config", str(finished_run.config_path)]
    # 1. `leaderboard <run> --by human:alice` → "no scores for human:alice"
    # 2. `review export <run> --out <tmp>/rev`
    # 3. edit review.csv: verdict "accept" for every row, score = 1, 2, …, n in reverse leaderboard order
    # 4. `review import <run> <tmp>/rev/review.csv --reviewer alice` → exit 0
    # 5. `leaderboard <run> --by human:alice` → candidate order equals the reversed export order
    # 6. `leaderboard <run>` (ranking judge) → order unchanged from the export
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_cli/test_arena_cmds.py tests/e2e/test_review_roundtrip.py -v`
Expected: FAIL with `invalid choice: 'leaderboard'` (exit code 2).

- [ ] **Step 3: Implement `arena_cmds.py` and register it in `main.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_cli tests/e2e/test_review_roundtrip.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Docs**

`README.md`: section "The arena", with `leaderboard`, `board search`, `learnings pin`, and `review export/import` examples. `docs/arena.md`, "Command line": the table above. Add a CHANGELOG line. Edit roadmap R3.1: the `_cli/main.py` row gets `+M2`.

- [ ] **Step 6: Commit**

```bash
git add src/swarm_arena/_cli tests/_cli/test_arena_cmds.py tests/e2e/test_review_roundtrip.py README.md docs/arena.md CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "Add arena CLI: review, leaderboard, board, learnings, trace, reindex" -m "Every arena read and human action is available from a laptop with no driver connection (spec 6.3, 6.3.1, 6.6, 7.2, 7.3). Roadmap R3.1 marks _cli/main.py +M2."
```

---

### Task 18: Board viewer

**Files:**
- Create: `src/swarm_arena/_board/serve.py`, `tests/_board/test_serve.py`
- Modify: `src/swarm_arena/_cli/arena_cmds.py` (`board serve`), `tests/_cli/test_arena_cmds.py`, `docs/arena.md` (section "Board viewer"), `CHANGELOG.md`

**Interfaces:**
- Consumes: `ArenaBoard`, `ArenaCache.find`/`count`/`learnings_by_id`/`candidates`/`scores`, `LearningFilter`, Task 16 `trace`, Task 3 `write_human_objects`, Task 7 `format_value`, Task 10 `is_verified`, `human_learning_id`.
- Produces:
```python
def make_server(board: ArenaBoard, namespaces: Sequence[str], *, author: str, host: str = "127.0.0.1", port: int = 8765,
                run_id: str | None = None, refresh_s: float = 5.0) -> ThreadingHTTPServer: ...
```
Routes [plan]. Every interpolated value goes through `html.escape`, and every score value through `format_value` (Task 7). There is no JavaScript and no external resource.
- Before routing, a request whose `Host` header is not `127.0.0.1:<port>`, `localhost:<port>`, or `<--host>:<port>` gets `403`. Without this check, DNS rebinding would let a web page read the per-server token and POST.
- `GET /?q=&ns=&goal=&part=&team=&agent=&round=&tag=&verified=1&human=1&pinned=1&hidden=1|all&page=N`: 50 learnings per page, with prev and next links. `hidden` is absent → excluded; `1` → only hidden; `all` → both. With `run_id`, the `run` filter defaults to it.
- `GET /learning/<id>`: title, text, author, markers, evidence with each candidate's score or error, and pin, hide forms.
- `GET /candidate/<id>`: summary, lineage (parents and ancestors as links), and the `trace` lines.
- `POST /learning/<id>/pin`, `POST /learning/<id>/hide`, `POST /learnings/add` (form fields `ns`, `title`, `text`, `goal`, `tags`): each writes one curation object, then `303` to the page. An add builds the same `Learning` as `learnings add` (Task 17): `run_id=""`, `goal_id=<goal or "">`, `round=0`, `author="human:<author>"`, `verified=is_verified(...)`. A missing or wrong form `token` → `403`. The token is `secrets.token_urlsafe(16)` per server. An unknown id → `404`.
- The latest frontier is refreshed at most every `refresh_s` seconds, on request.
- CLI `board serve <run-or-namespace> [--host 127.0.0.1] [--port 8765] [--author NAME]` prints `Serving http://<host>:<port>/` (the real port when `--port 0`), flushes stdout, and serves until Ctrl-C (exit 0).

- [ ] **Step 1: Write the failing tests**

`tests/_board/test_serve.py` (server on port 0 in a thread; `urllib.request`):
```python
def test_index_paginates(server: Server) -> None: ...                  # 120 learnings → page 1 has 50 items, page 3 has 20
def test_search_and_filters(server: Server) -> None: ...               # ?q=viscosity&verified=1 → only the verified match
def test_learning_page_shows_evidence_scores(server: Server) -> None: ...
def test_candidate_page_shows_lineage_and_trace(server: Server) -> None: ...
def test_html_is_escaped(server: Server) -> None:
    body = get(server, f"/learning/{xss_id}")                          # title "<script>alert(1)</script>"
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "<script>alert(1)" not in body
def test_pin_requires_token(server: Server) -> None: ...               # POST without token → 403, no curation object; with token → 303, one object
def test_add_writes_human_learning(server: Server) -> None: ...        # id matches L-h-<16 hex>, author "human:<author>"
def test_hidden_shown_only_with_filter(server: Server) -> None: ...
def test_unknown_id_404(server: Server) -> None: ...
def test_rejects_foreign_host(server: Server) -> None: ...             # GET / with Host "evil.example" → 403 and no learning title in the
    # body; POST /learning/<id>/pin with a valid token and that Host → 403 and no curation object
```
`tests/_cli/test_arena_cmds.py`:
```python
def test_board_serve_smoke(finished_run: FinishedRun) -> None:
    proc = subprocess.Popen(["uv", "run", "swarm-arena", "board", "serve", finished_run.namespace, "--config",
                             str(finished_run.config_path), "--port", "0"], stdout=subprocess.PIPE, text=True)
    url = proc.stdout.readline().split()[-1]                           # "Serving http://127.0.0.1:<port>/"
    for path in ("", f"learning/{learning_id}", f"candidate/{candidate_id}"):
        assert urllib.request.urlopen(url + path, timeout=10).status == 200
    proc.terminate()
    proc.wait(timeout=10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/_board/test_serve.py tests/_cli/test_arena_cmds.py::test_board_serve_smoke -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._board.serve'`.

- [ ] **Step 3: Implement `serve.py` and the `board serve` command**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/_board/test_serve.py tests/_cli -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Docs and commit**

`docs/arena.md`, "Board viewer": the command, the filters, local-only binding and the `Host` check, and the pin, hide, and add actions writing curation objects. Add a CHANGELOG line.
```bash
git add src/swarm_arena/_board/serve.py src/swarm_arena/_cli/arena_cmds.py tests/_board/test_serve.py tests/_cli/test_arena_cmds.py docs/arena.md CHANGELOG.md
git commit -m "Add the local board viewer" -m "board serve runs a stdlib HTTP viewer on the SQLite cache with search, filters, pages, learning and candidate pages, and pin, hide, and add as curation objects (spec 6.3.1)."
```

---

### Task 19: Two concurrent swarms end to end

**Files:**
- Create: `tests/e2e/test_arena_two_runs.py`
- Modify: `tests/_fakes/tools.py`

**Interfaces:**
- Consumes: `Swarm.run(plans, run_id=...)`, `FunctionTool`, `ProgramJudge`, `ScriptedManager`, `KeepTopTeams`, `ArenaBoard`, `recorded_frontiers`, `LocalStore`.
- Produces (in `tests/_fakes/tools.py`):
```python
def wait_for_file(path: str, timeout_s: float = 60.0) -> dict[str, Any]: ...   # polls every 0.1 s; returns {"waited_s": s}; raises TimeoutError
def score_one(goal: Goal, candidate: Candidate) -> Score: ...                  # Score.maximize(1.0)
```

- [ ] **Step 1: Write the test**

```python
# Test-local helpers: events_of(run_id) reads that run's committed chain (read_owner + read_chain); goal_key(run_id) is its main
# GoalKey; round1_learning_id(run_id) is the id in its round-1 `learning` event. board = ArenaBoard on the shared arena_root.
WRITER: ContextVar[str] = ContextVar("WRITER", default="")      # the run whose task (or worker thread) makes a store call
async def test_two_swarms_share_learnings_within_one_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[tuple[str, str, str, bool]] = []
    # (WRITER.get(), op, f"{store.url}/{key}", ok), recorded by wrapping LocalStore.create and .replace; ok is False when
    # the call raised AlreadyExists or PreconditionFailed, so a failed second create of a key is recorded too.
    # anyio.to_thread.run_sync copies the context, so the worker threads of a run see its WRITER.
    # shared arena_root and artifact_root; run_root A = tmp/runs-a, B = tmp/runs-b; namespace "toy"; goals "a" and "b"
    # run ids fixed: A = "2026-10-18T13-00-00Z_aaaaaa", B = "2026-10-18T13-00-00Z_bbbbbb"
    # each plan: 3 RoundSpecs, policy KeepTopTeams(1), one team, one agent:
    #   tool FunctionTool("_fakes.tools:wait_for_file") (deterministic), judge ProgramJudge("_fakes.tools:score_one"),
    #   ScriptedManager(ScriptedManager.Config(requests=[
    #       {"inputs": {"path": <its own plan file>}, "round": 1},
    #       {"inputs": {"path": <other run>/goals/<other goal>/rounds/round-01.md}, "round": 2},
    #       {"inputs": {"path": <its own plan file>, "timeout_s": 30.0}, "round": 3}],   # other inputs than round 1: no §6.2 refusal
    #       post_learnings=True))
    async def run_as(name: str, swarm: Swarm, plan: Plan) -> None:
        WRITER.set(name)
        await swarm.run(plan, run_id=name)
    async with anyio.create_task_group() as tg:
        tg.start_soon(run_as, A, Swarm(config_a), plan_a)
        tg.start_soon(run_as, B, Swarm(config_b), plan_b)
    # 1. delay of at most one round: A's round-3 recorded frontier snapshot holds B's round-1 learning, and B's holds A's
    for me, other in ((A, B), (B, A)):
        _, _, f3 = next(x for x in recorded_frontiers(events_of(me)) if x[1] == 3)
        assert round1_learning_id(other) in [x.id for x in board.snapshot(goal_key(me), f3).learnings("toy", limit=100)]
    # 2. no file has two writers
    tried_by: dict[str, set[str]] = defaultdict(set)
    for run, _, path, _ in writes:
        tried_by[path].add(run)
    shared = sorted(path for path, runs in tried_by.items() if len(runs) > 1 and not path.startswith(artifact_root))
    assert shared == []                       # outside the content-addressed artifact_root, no key is tried by both runs
    assert sorted(path for _, _, path, ok in writes if ok and path.startswith(arena_root)) == [
        f"{arena_root}/toy/runs/{A}.json", f"{arena_root}/toy/runs/{B}.json"]
    replaced = {path for _, op, path, ok in writes if ok and op == "replace"}
    assert replaced == {f"{run_root_a}/{A}/control/owner.json", f"{run_root_b}/{B}/control/owner.json"}
```
Round 2 of each run waits until the other run's round-1 page exists. The page is written only after that run's round-1 close commit. So the other run's round-1 items are committed before this run's round-3 frontier.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/e2e/test_arena_two_runs.py -v`
Expected: FAIL with `AttributeError: module '_fakes.tools' has no attribute 'wait_for_file'`.

- [ ] **Step 3: Add `wait_for_file` and `score_one` to `tests/_fakes/tools.py`**

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/e2e/test_arena_two_runs.py -v && uv run ruff check . && uv run ty check`
Expected: PASS. If it fails, fix the cause in the owning module (Tasks 8 to 13), not in the test.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_arena_two_runs.py tests/_fakes/tools.py
git commit -m "Test two concurrent swarms sharing one arena" -m "Proves learnings cross runs within one round and that no arena or run file has two writers (spec 7.2, D18)."
```

---

### Task 20: Milestone exit

**Files:**
- Modify: `CHANGELOG.md` (only if a line is missing)

**Interfaces:**
- Consumes: everything above.
- Produces: M2 exit criteria green on branch `m2-arena-board`.

- [ ] **Step 1: Run the common gate (roadmap R2.0)**

```bash
uv sync --all-extras --all-groups
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```
Expected: every command exits 0. `docs/superpowers/` is not built.

- [ ] **Step 2: Run the M2 exit criteria**

If Task 1 chose MinIO, start it first and export `SWARM_ARENA_TEST_S3_ENDPOINT=http://127.0.0.1:9000`, `AWS_ACCESS_KEY_ID=minio`, and `AWS_SECRET_ACCESS_KEY=minio123`.
```bash
uv run pytest -m s3 -v
uv run pytest tests/e2e/test_arena_two_runs.py tests/e2e/test_reindex.py tests/e2e/test_review_roundtrip.py -v
uv run pytest tests/_cli/test_arena_cmds.py::test_board_serve_smoke -v
```
Expected: PASS.

- [ ] **Step 3: Check the core with no extras (CI `core` job)**

```bash
uv sync --all-groups
uv run pytest
uv run python -c "import sys, swarm_arena, swarm_arena.board, swarm_arena.scoring; assert 's3fs' not in sys.modules"
uv sync --all-extras --all-groups
```
Expected: PASS. `_store/test_s3.py` is not collected, and no `s3fs` is imported.

- [ ] **Step 4: Check the surface and the leftovers**

Run: `grep -rn "RunBoard" src tests docs --include=*.py --include=*.md | grep -v superpowers; git status --short`
Expected: no `RunBoard` hits and a clean tree. `swarm_arena.board.__all__` and `swarm_arena.scoring.__all__` equal spec §9.

- [ ] **Step 5: Commit any CHANGELOG fix, then hand off**

```bash
git add CHANGELOG.md && git commit -m "Complete the M2 changelog" -m "Lists every M2 user-visible change under Unreleased."
```
Then use superpowers:finishing-a-development-branch to merge `m2-arena-board` with a PR. Ask the owner before you push.
